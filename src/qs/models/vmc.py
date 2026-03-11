import jax
import jax.numpy as jnp
import numpy as np
import torch
from qs.utils import Parameter # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State



class WaveFunction:
    """Simple wavefunction object exposing `alpha` and `beta` and callable as `wf(r)`.

    This is a thin, backend-agnostic container for the analytic Gaussian used
    by the template. It stores references to `alpha`/`beta` so updates to the
    arrays are visible to the instance.
    """
    def __init__(self, backend, alpha, beta=None, a=0.0):
        self.backend = backend
        self.alpha = alpha
        self.beta = beta
        self.a = a
        
    def __call__(self, r, alpha=None, beta=None):
        a = self.alpha if alpha is None else alpha #if alpha is not provided, use self.alpha
        b = self.beta if beta is None else beta #if beta is not provided, use self.beta

        #if jastrow factor is not zero, calculate it and add it to the gaussian log psi
        if a != 0:
            j = self.jastrow(r) # calculate jastrow factor
        else:
            j = 0
        return self.gaussian_log_psi(r, a, b) + j

    def gaussian_log_psi(self, r, alpha=None, beta=None):
        """Instance Gaussian log-psi using this WaveFunction's backend.

        Uses `self.backend` so no backend argument is needed and torch tensors
        will be used when `self.backend` is `torch`.
        """
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        bk = self.backend
        
        if b is None:
            return -a * bk.sum(r**2)
        else:
            return -a * (bk.sum(r[:, :-1]**2) + b * bk.sum(r[:, -1]**2))
        
    def jastrow(self, r):
        """
        calculates the jastrow factor for a given configuration of particles r
        r: shape (N, dim)
        
        """
        a = self.a             # hard-core radius characterizing the interaction strength
        N = r.shape[0] #
        bk = self.backend

        if a is None or N == 1: #no interactions if a is zero or only one particle
            return 0.0

        diff = r[:, None, :] - r[None, :, :]         # make a  (N, N, dim) array with differences
        r_ij = bk.linalg.norm(diff, axis=-1)         # (N, N) distance matrix with r_ij[i, j] = |r_i - r_j| 

        # take only upper triangle to acccount for double counting and avoid self-interaction (i=j)
        iu = bk.triu_indices(N, N, offset=1) #for torch
        rij = r_ij[iu] 

        # hard-core condition 
        if bk.any(rij <= a):
            return -bk.inf
        
        # return the total jastrow factor in log space

        return bk.sum(bk.log(1.0 - a / rij))
 

    def log_prob(self, r):
        # function for calculating log|psi|^2 from log|psi|

        logpsi = self(r)      # log|psi|
        return 2.0 * logpsi   # log|psi|^2
    
    def log_prob_single(self, r_single_wf, alpha=None, beta=None):
        
        """Calculate log|psi|^2 for a single particle"""

        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        bk = self.backend
        
        if b is None:
            return -2 * a * bk.sum(r_single_wf**2)
        else:
            return -2 * a * (bk.sum(r_single_wf[:-1]**2) + b * r_single_wf[-1]**2) #returns logprob for a single particle with 3 dimensions
        
    def quantum_force_single(self, r_single_wf, alpha=None, beta=None):
        """Calculate the quantum force for a single particle"""

        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        bk = self.backend

        #calculate quantum force
        q_force = -4 * a * r_single_wf
        
        if b is not None:
            q_force = q_force.clone()
            q_force[..., -1] *= b
            
        return q_force
    
class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="torch",
    ):
        self._configure_backend(backend)
        self._initialize_vars(nparticles, dim, rng, log, logger, logger_level)

        r = 0 # initialize the positions randomly

        self._initialize_variational_params()
        self.state = 0 # take a look at the qs.utils State class

        """
        msg = f"VMC initialized with {self._N} particles in {self._dim} dimensions with {self.params.get('alpha').size} parameters"
            self._logger.info(msg) 
        """

    def _configure_backend(self, backend):
        """
        Here we configure the backend, for example, numpy or jax
        You can use this to change the linear alebra methods or do just in time compilation.
        Note however that depending on how you build your code, you might not need this.
        """
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg

            # Overwrite the closures with JAX variants and jit them.
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self._jit_functions()
        elif backend == "torch":
            self.backend = torch
            self.la = torch.linalg
        else:
            raise ValueError("Invalid backend:", backend)

        return self
    
    def _jit_functions(self):
        """JIT commonly used functions when using the JAX backend.

        Note: we avoid jitting `wf` here because `self.wf` is an instance
        of `WaveFunction`. For JAX use, prefer calling the pure
        `gaussian_log_psi` with explicit params and jitting that instead.
        """
        functions_to_jit = [
            "prob_closure",
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
        ]
        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self


    def _initialize_vars(self, nparticles, dim, rng, log, logger, logger_level):
        self._N = nparticles
        self._dim = dim
        self._log = log if log else False

        if logger:
            self._logger = logger
        else:
            import logging
            self._logger = logging.getLogger(__name__)

        self._logger_level = logger_level
        self._rng = rng if rng else np.random.default_rng()

    def _initialize_variational_params(self, rng=None, init_alpha=1, init_beta=None):
        # Initialize variational parameters in the correct range with the correct shape
        # Create `alpha` (per-particle, per-dimension) and optional `beta`.
        rng = rng if rng is not None else self._rng


        if self.backend is torch:
            # use float64 for numerical parity with jax's x64 mode
            alpha = torch.tensor(init_alpha, dtype=torch.float64)
            beta = torch.tensor(init_beta, dtype=torch.float64) if init_beta is not None else None
        elif self.backend is jnp:
            alpha = jnp.array(init_alpha)
            beta = jnp.array(init_beta) if init_beta is not None else None
        else:
            alpha = init_alpha  # Just a Python float for numpy
            beta = init_beta

        self.alpha = alpha
        self.beta = beta

        # pack into params dict for compatibility with other code
        try:
            self.params
        except AttributeError:
            self.params = {}
        self.params["alpha"] = self.alpha
        self.params["beta"] = self.beta

        # Attach a WaveFunction instance using the selected backend
        self.wf = WaveFunction(self.backend, self.alpha, self.beta)

        return self
