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
    def __init__(self, backend, alpha, beta=None):
        self.backend = backend
        self.alpha = alpha
        self.beta = beta

    def __call__(self, r, alpha=None, beta=None):
        a = self.alpha if alpha is None else alpha #if alpha is not provided, use self.alpha
        b = self.beta if beta is None else beta #if beta is not provided, use self.beta
        return self.gaussian_log_psi(r, a, b)

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

    def log_prob(self, r):
        # function for calculating log|psi|^2 from log|psi|

        logpsi = self(r)      # log|psi|
        return 2.0 * logpsi   # log|psi|^2
    
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

    def _initialize_variational_params(self, rng=None):
        # Initialize variational parameters in the correct range with the correct shape
        # Create `alpha` (per-particle, per-dimension) and optional `beta`.
        rng = rng if rng is not None else self._rng

        init_alpha = 0.5
        beta = None

        if self.backend is torch:
            # use float64 for numerical parity with jax's x64 mode
            alpha = torch.full((self._N, self._dim), float(init_alpha), dtype=torch.float64)
        elif self.backend is jnp:
            alpha = jnp.full((self._N, self._dim), init_alpha)
        else:
            alpha = np.full((self._N, self._dim), init_alpha, dtype=float)

        self.alpha = alpha
        self.beta = beta

        # pack into params dict for compatibility with other code
        try:
            self.params
        except AttributeError:
            self.params = {}
        self.params["alpha"] = self.alpha
        self.params["beta"] = self.beta

        # Attach a proper WaveFunction instance so `wf.alpha` and `wf(r)` work.
        # Attach a WaveFunction instance using the selected backend
        self.wf = WaveFunction(self.backend, self.alpha, self.beta)

        return self