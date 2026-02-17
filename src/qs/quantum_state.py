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
        self.sampler=None

        if rng is None:
            self.rng = default_rng(self._seed)
        else:
            self.rng = rng

        # Suggestion of checking flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._sampling_performed = False

    def set_wf(self, wf_type, nparticles, dim, ):
        self._N = nparticles
        self._dim = dim
        self._wf_type = wf_type

        # Create the wavefunction object
        self.wf = VMC(nparticles=nparticles, dim=dim, backend=self._backend)

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
        
        positions = torch.randn(self._N, self._dim, dtype=torch.float64)

        # Wavefunction log-probability at positions. self.wf is a VMC object
        logp = self.wf.wf.log_prob(positions)  

        return State(positions=positions, logp=logp, n_accepted=0, delta=0)

    def set_sampler(self, mcmc_alg, scale=0.5):

        self.mcmc_alg = mcmc_alg
        self._scale = scale

        # Create sampler instance (only Metropolis supported for now)
        self.sampler = Metro(rng=self.rng, scale=scale, logger=self.logger)

        # If Hamiltonian already set, propagate it
        if self.hamiltonian is not None:
            self.sampler.set_hamiltonian(self.hamiltonian)


    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
        
        # check Gd script
        self._optimizer = opt.Gd(eta=eta)


    def train(self, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters.
        Here you should calculate sampler statistics and update the wave function parameters based on the derivative of the (statistical) local energy.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size

        if self._log:
            t_range = tqdm(
                range(max_iter),
                desc="[Training progress]",
                position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(max_iter)

        steps_before_optimize = batch_size

        epoch = 0
        for _ in t_range:
            # Here you collect batch_size samples and calculate the local energy
            # After you have collected batch_size samples, you update the parameters of the wave function
            
            
            steps_before_optimize -= 1
            if steps_before_optimize == 0:
                epoch += 1
            
            
                # Make Descent step with optimizer

            
                steps_before_optimize = batch_size

        
        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")


    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

        self._is_initialized() # check if the system is initialized
        self._is_trained() # check if the system is trained

        self._results = self.sampler.sample(self.wf, self._make_initial_state(), nsamples, nchains, seed) # call the sample method from the sampler class and store the results in self._results
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