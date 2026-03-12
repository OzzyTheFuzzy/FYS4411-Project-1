# import copy
import sys
import warnings
import torch
sys.path.insert(0, "../src/")

#from qs.utils import errors
from qs.utils import generate_seed_sequence
from qs.utils import setup_logger
from qs.utils import State
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np
import pandas as pd

from qs.models import VMC

from numpy.random import default_rng
from tqdm.auto import tqdm

from physics.hamiltonians import HarmonicOscillator as HO

from samplers.metropolis import Metropolis as Metro

from samplers.langevin_metropolis import LangevinMetropolis as Langevin
import optimizers as opt

warnings.filterwarnings("ignore", message="divide by zero encountered")


class QS:
    def __init__(
        self,
        backend="torch",
        log=True,
        logger_level="INFO",
        rng=None,
        seed=None,
    ):
        """Quantum State
        It is conceptually important to understand that this is the system.
        The system is composed of a wave function, a hamiltonian, a sampler and an optimizer.
        This is the high level class that ties all the other classes together.
        """

        #self._check_logger(log, logger_level)
        
        self._log = log
        self.hamiltonian = None
        self._backend = backend
        self.mcmc_alg = None
        self._optimizer = None
        self.wf = None
        self._seed = seed
        self.logger = setup_logger(self.__class__.__name__, level=logger_level) if self._log else None
        self.sampler = None
        self.burn_in = None
        self.a = 0.0        # Jastrow factor strength, set to 0 for no Jastrow factor
        if rng is None:
            self.rng = default_rng(self._seed)
        else:
            self.rng = rng

        # Suggestion of checking flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._sampling_performed = False

    def set_wf(self, wf_type, nparticles, dim, a=0.0):
        self._N = nparticles
        self._dim = dim
        self._wf_type = wf_type
        self.a = a
        # Create the wavefunction object
        self.wf = VMC(nparticles=nparticles, dim=dim, backend=self._backend, a=self.a)

        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, omega_ho, omega_z=None):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.
        """
        self.hamiltonian = HO(nparticles=self._N, dim=self._dim, int_type=int_type, backend=self._backend, omega_ho=omega_ho, omega_z=omega_z)

        if self.sampler is not None: #such that if we set the hamiltonian after setting the sampler, we update the sampler with the new hamiltonian
            self.sampler.set_hamiltonian(self.hamiltonian)
            
    def _make_initial_state(self):
        # Start positions near origin (Gaussian). Shape: (N, dim)
        
        positions = 0.1 * torch.randn(self._N, self._dim, dtype=torch.float64)


        # Wavefunction log-probability at positions. self.wf is a VMC object
        logp = self.wf.wf.log_prob(positions)  

        return State(positions=positions, logp=logp, n_accepted=0, delta=0)

    def set_sampler(self, mcmc_alg, scale=0.1):

        self.mcmc_alg = mcmc_alg
        self._scale = scale

        # Create sampler instance (only Metropolis supported for now)
        if mcmc_alg == "metropolis":
            self.sampler = Metro(rng=self.rng, scale=scale, logger=self.logger)
        elif mcmc_alg == "langevin":
            self.sampler = Langevin(rng=self.rng, scale=scale, logger=self.logger)
            

        # If Hamiltonian already set, propagate it
        if self.hamiltonian is not None:
            self.sampler.set_hamiltonian(self.hamiltonian)

        
    def set_optimizer(self, optimizer, eta):
        """
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
        
        # check Gd script
        if optimizer == "gd":
            self._optimizer = opt.Gd(eta=eta)
            
        elif optimizer == "adam":
            self._optimizer = opt.Gd(eta=eta, optimizer_name="adam")


    def train(self, MC_training_cycles, alpha_array, burn_in=0, num=False):
        """
        Train the wave function parameters using grid search over alpha_array.
        """

        self._is_initialized()
        self._training_cycles = MC_training_cycles

        self.alpha_array = alpha_array
        self.mean_num_energies = []
        self.mean_ana_energies = []

        for a in tqdm(alpha_array, desc="[Training progress]", colour="green") if self._log else alpha_array:

            print("N, D:", self.hamiltonian._N, self.hamiltonian._dim)
            print("omega_ho:", self.hamiltonian.omega_ho, "omega_z:", self.hamiltonian.omega_z)

            a_tensor = torch.tensor(a, dtype=torch.float64)

            # update wavefunction alpha
            self.wf.alpha = a_tensor
            self.wf.wf.alpha = a_tensor

            # update sampler scale
            if self.mcmc_alg == "metropolis" or self.mcmc_alg == "langevin":
                self.sampler.scale = self._scale * np.sqrt(1.0 / a)

            state= self._make_initial_state() # make initial state for sampling
            # call sample function from sampler class
            E_ana, E_num, _, accept_rate = self.sampler._sample_energy_and_optional_O(
                self.wf, state,
                MC_training_cycles, seed=self._seed,
                burn_in=burn_in,
                need_O=False,
                num=num,
            )

            # compute mean energies (same as before)
            mean_ana_energy = E_ana.mean().item()
            mean_num_energy = E_num.mean().item() if num else mean_ana_energy

            # store results
            self.mean_num_energies.append(mean_num_energy)
            self.mean_ana_energies.append(mean_ana_energy)

            print(
                f"alpha={a:.3f} accept_rate={accept_rate:.3f} "
                f"mean_E_ana={mean_ana_energy:.6f}, scale={self.sampler.scale:.4f}")

        # pick best alpha
        best_idx = np.argmin(self.mean_ana_energies)
        a_tensor = torch.tensor(self.alpha_array[best_idx], dtype=torch.float64)

        self.wf.alpha = a_tensor
        self.wf.wf.alpha = a_tensor

        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")


    def train_steepest_descent(self, MC_training_cycles, num_iterations, burn_in=0, alpha_0=1.0):
        """
        Train using steepest descent for alpha.
        """
        self._is_initialized()
        self._training_cycles = MC_training_cycles
        
        self.mean_num_energies = []
        self.mean_ana_energies = []
        self.alpha_array = []

        alpha = torch.tensor(alpha_0, dtype=torch.float64)

        tol = 1e-4      #tolerance for early stopping based on energy improvement
        patience = 5    #number of iterations to wait for improvement before stopping
        no_improve_count = 0 
        best_energy = float("inf")
        for alpha_j in range(num_iterations):

            self.wf.alpha = alpha
            self.wf.wf.alpha = alpha
            
            a_val = alpha.detach().item()
        
            state = self._make_initial_state()

            E_ana, E_num, O, accept_rate = self.sampler._sample_energy_and_optional_O(
                self.wf, state,
                MC_training_cycles, seed=self._seed,
                burn_in=burn_in,
                need_O=True,
                )
            
            # Compute the gradient dE/d alpha and the mean analytical energy
            dE_dalpha, mean_ana_energy, = self.hamiltonian.compute_gradient(O, E_ana)
            
            # Update alpha using the optimizer either with gradient descent or Adam 
            alpha = self._optimizer.step(alpha, dE_dalpha, self._optimizer)

            # Store global stats for plotting
            self.alpha_array.append(a_val)
            self.mean_ana_energies.append(mean_ana_energy.item())

            print(f"iteration={alpha_j} accept_rate={accept_rate:.4f} mean_E_ana={mean_ana_energy.item():.6f}, alpha={a_val:.7f}")

            current_energy = mean_ana_energy.item()

            if best_energy - current_energy > tol:
                best_energy = current_energy
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Stopping early at iteration {alpha_j} (energy plateau)")
                break
            
        # finding the best alpha after training and updating the wave function with it
        best_idx = np.argmin(self.mean_ana_energies)
        a_tensor = torch.tensor(self.alpha_array[best_idx], dtype=torch.float64) # convert alpha to tensor for use in wave function

        self.wf.alpha = a_tensor    # update to the best variational parameter in the wave function 
        self.wf.wf.alpha = a_tensor # update WaveFunction.alpha to the best variational parameter
    
        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")


    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""
            # DEBUG

        self._is_initialized() # check if the system is initialized
        self._is_trained() # check if the system is trained

        # Pass burn_in to sampler if set
        if hasattr(self, 'burn_in'):
            self.sampler.burn_in = self.burn_in
        
        self._results = self.sampler.sample(self.wf, self._make_initial_state(), nsamples, nchains, seed) # call the sample method from the sampler class, which will perform the sampling and return the results
        return self._results
    

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise ValueError(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise ValueError(msg)
"""
    def _sampling_performed(self):
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise errors.SamplingNotPerformed(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str") """