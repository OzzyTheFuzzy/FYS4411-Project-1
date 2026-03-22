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
        self.obd = False      # whether to compute one-body density during final sampling
        self.n_bins = 80        # number of bins for one-body density histogram
        self.r_max = None       # maximum radius for one-body density histogram

        if rng is None:
            self.rng = default_rng(self._seed)
        else:
            self.rng = rng

        # Suggestion of checking flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._sampling_performed = False

    def set_wf(self, wf_type, nparticles, dim, a=0.0, beta=None):
        self._N = nparticles
        self._dim = dim
        self._wf_type = wf_type
        self.a = a
        # Create the wavefunction object
        self.wf = VMC(nparticles=nparticles, dim=dim, backend=self._backend, a=self.a, beta=beta)

        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, omega_ho=1.0, omega_z=None):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.
        """
        self.hamiltonian = HO(nparticles=self._N, dim=self._dim, int_type=int_type, backend=self._backend, omega_ho=omega_ho, omega_z=omega_z)

        if self.sampler is not None: #such that if we set the hamiltonian after setting the sampler, we update the sampler with the new hamiltonian
            self.sampler.set_hamiltonian(self.hamiltonian)
            
    def _make_initial_state(self):

        # Start positions near origin (Gaussian). Shape: (N, dim)
        while True:
            positions = 0.1 * torch.randn(self._N, self._dim, dtype=torch.float64)

            # safeguard for the hard core condition
            if self.a != 0.0:
                r_ij_abs, _ = self.wf.wf.distance_and_distance_vec(positions)
                iu = torch.triu_indices(self._N, self._N, offset=1)
                rij = r_ij_abs[iu[0], iu[1]]
                if torch.any(rij <= self.a):
                    continue

            # Wavefunction log-probability at positions. self.wf is a VMC object
            logp = self.wf.wf.log_prob(positions)  

            return State(positions=positions, logp=logp, n_accepted=0, delta=0, obd=self.obd,        
            n_bins=self.n_bins,  r_max=self.r_max)

    def set_sampler(self, mcmc_alg, scale=0.1, obd=False, n_bins=80, r_max=None):

        self.mcmc_alg = mcmc_alg
        self._scale = scale
        self.obd = obd
        self.n_bins = n_bins
        self.r_max = r_max

        # Create sampler instance 
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


    def train(self, MC_training_cycles, alpha_array, burn_in=0, num_iterations=20, alpha_0=None, num=False):
        """
        Train the wave function parameters using grid search over alpha_array.
        """

        self._is_initialized()
        self._training_cycles = MC_training_cycles
        self.burn_in = burn_in
        self.alpha_array = alpha_array
        self.mean_num_energies = []
        self.mean_ana_energies = []
        self.alpha_array_tested =[]
        self.accept_rate_array = []

        if alpha_0 is not None:
            alpha = torch.tensor(alpha_0, dtype=torch.float64)
            alpha_array_tested = []
            alpha_array_tested.append(alpha_0)
            alpha_array =np.zeros(num_iterations) # placeholder for alpha values during training
            tol = 1e-4      #tolerance for early stopping based on energy improvement
            patience = 5    #number of iterations to wait for improvement before stopping
            no_improve_count = 0 
            best_energy = float("inf")
            iteration = 0
            need_O = True
            idx = 0

        else:
            need_O = False
            alpha_array_tested=alpha_array
        
        for alpha_i in tqdm(alpha_array, desc="[Training progress]", colour="green") if self._log else alpha_array:
            
            print("N, D:", self.hamiltonian._N, self.hamiltonian._dim)
            print("omega_ho:", self.hamiltonian.omega_ho, "omega_z:", self.hamiltonian.omega_z)
            
            # create tensor for current alpha and retrieve alpha value
            if alpha_0 is None:
                alpha = alpha_i
                idx = np.where(alpha_array == alpha_i)[0].item()
                a_tensor = torch.tensor(float(alpha), dtype=torch.float64)
                a_val = float(alpha)
            else:
                idx += 10
                a_tensor = alpha.detach().clone()
                a_val = alpha.detach().item()

            # use different seed for each alpha to get different samples
            self.sampler.rng = np.random.default_rng(self._seed + idx * 17) 
            
            # update wavefunction alpha
            self.wf.alpha = a_tensor
            self.wf.wf.alpha = a_tensor

            # update sampler scale
            if self.mcmc_alg == "metropolis" or self.mcmc_alg == "langevin":
                self.sampler.scale = self._scale * np.sqrt(1.0 / a_val)

            # make initial state for sampling
            state= self._make_initial_state() 
            
            # call sample function from sampler class
            if num:
                E_ana, E_num, O, accept_rate, t_ana_tot, t_num_tot = self.sampler._sample_energy_and_optional_O(
                self.wf, state,
                MC_training_cycles, seed=self._seed, 
                burn_in=burn_in,
                need_O=need_O,
                num=num, )

            else:
                E_ana, E_num, O, accept_rate = self.sampler._sample_energy_and_optional_O(
                    self.wf, state,
                    MC_training_cycles, seed=self._seed, 
                    burn_in=burn_in,
                    need_O=need_O,
                    num=num,
                )
            # compute mean energies
            mean_ana_energy = E_ana.mean().item()
            mean_num_energy = E_num.mean().item() if num else mean_ana_energy

            # store results
            self.mean_num_energies.append(mean_num_energy)
            self.mean_ana_energies.append(mean_ana_energy)
            self.alpha_array_tested.append(a_val)
            self.accept_rate_array.append(accept_rate)

            print(
                f"alpha={a_val:.3f} accept_rate={accept_rate:.3f} "
                f"mean_E_ana={mean_ana_energy:.6f}, scale={self.sampler.scale:.4f}")
            
            # if gd is activated compute next alpha
            if alpha_0 is not None:
                # Compute the gradient dE/d alpha and the mean analytical energy
                dE_dalpha, _ = self.hamiltonian.compute_gradient(O, E_ana)
                
                # Update alpha using the optimizer either with gradient descent or Adam 
                alpha = self._optimizer.step(alpha, dE_dalpha)

                # Store global stats for plotting
                iteration+=1
                
                if best_energy - mean_ana_energy > tol:
                    best_energy = mean_ana_energy
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"Stopping early at iteration {iteration} (energy plateau)")
                    break

        # pick best alpha
        best_idx = np.argmin(self.mean_ana_energies)
        a_tensor = torch.tensor(self.alpha_array_tested[best_idx], dtype=torch.float64)

        self.wf.alpha = a_tensor
        self.wf.wf.alpha = a_tensor

        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")

    def sample(self, nsamples, final_burn_in, nchains=1, seed=None, num=False, write_to_file=False, name_of_file="energy", obd=False):
    
        """helper for the sample method from the Sampler class"""
        
        self._is_initialized() # check if the system is initialized
        self._is_trained() # check if the system is trained

        # Pass burn_in to sampler if set
        if hasattr(self, 'burn_in'):
            self.sampler.burn_in = self.burn_in

        self.sampler.scale = self._scale * np.sqrt(1.0 / self.wf.alpha.item())
        # call the sample method from the sampler class
        self._results = self.sampler._sample(wf=self.wf, nsamples=nsamples, state=self._make_initial_state(),
        scale=self.sampler.scale, seed=seed, chain_id=0, burn_in=final_burn_in, num=num,
        write_to_file=write_to_file, name_of_file=name_of_file, obd=obd
    )
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