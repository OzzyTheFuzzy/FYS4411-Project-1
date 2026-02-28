from tracemalloc import start
import jax
import jax.numpy as jnp
import numpy as np
import torch 
import time
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Hamiltonian:
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        """
        Note that this assumes that the wavefunction form is in the log domain
        """
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case "jax":
                self.backend = jnp
                self.la = jnp.linalg
            case "torch":
                self.backend = torch
                self.la = torch.linalg
            case _: # noqa
                raise ValueError("Invalid backend:", backend)

    def local_energy(self, wf, r):
        raise NotImplementedError("Subclasses must implement local_energy")




class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
        omega_ho,
        omega_z, 
    ):
        super().__init__(nparticles, dim, int_type, backend)
        self.omega_ho = omega_ho
        self.omega_z = omega_z


    def local_energy(self, wf, r, num=False):
        """Local energy of the system
        Returns either numerical or analytical local energy
        """

        V = self.potential_energy(r)

        if num == True:
            # branch for numerical and analytical energy
            start = time.perf_counter()
            K_num = self.kinetic_energy_numerical(wf, r)
            end = time.perf_counter()
            E_L_num = K_num + V

            start = time.perf_counter()
            K_ana = self.kinetic_energy_analytical(wf, r)
            end = time.perf_counter()

            E_L_ana = K_ana + V
            return E_L_num, E_L_ana
        
        else:
            K_ana = self.kinetic_energy_analytical(wf, r)
            E_L_ana = K_ana + V
            return E_L_ana
    
    def potential_energy(self, r):
        if self._dim <= 2:
            return 0.5 * self.omega_ho**2 * self.backend.sum(r**2)

        omega_z = self.omega_ho if (self.omega_z is None) else self.omega_z
        return 0.5 * (
            self.omega_ho**2 * self.backend.sum(r[:, :-1]**2)
            + omega_z**2 * self.backend.sum(r[:, -1]**2)
        )
    
    def kinetic_energy_numerical(self, wf, r):
        
        """Kinetic energy of the system using logspace wf and numerical derivation from torch
        Returns the numerical kinetic energy of the system 
        """
  
        laplacian_logwf, grad_2_logwf= self.compute_gradients(wf, r)
        K_num = - 0.5 * (laplacian_logwf + grad_2_logwf)
        return K_num
    
    def kinetic_energy_analytical(self, wf, r):
        
        """Analytical kinetic energy of the system
        Returns the analytical kinetic energy of the system nabla^2 psi in logspace
        """
        alpha = wf.alpha

        if wf.beta == None:  
             
            K_ana = -0.5 * (-2.0 * alpha * self._N * self._dim + 4.0 * alpha**2 * self.backend.sum(r**2))


        else:
            beta = wf.beta
            K_ana = wf.alpha * self._N *(2+wf.beta) - 2*alpha**2 * self.backend.sum(self.backend.sum(r[:,:-1]**2)) - 2 * alpha**2 * beta**2 * self.backend.sum(self.backend.sum(r[:, -1]**2))
     
        return K_ana
    
    def compute_gradients(self, wf, r):
        r = r.clone().detach().requires_grad_(True)

        def logpsi_flat(r_flat):
            r_t = r_flat.view(r.shape)  # reshape back to (N, dim)
            return wf.wf(r_t)

        r_flat = r.view(-1)  # flatten r vector for torch's functions
        logpsi = logpsi_flat(r_flat)
        grad = torch.autograd.grad(logpsi, r_flat, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(logpsi_flat, r_flat)

        lap = torch.trace(hessian)
        grad_sq = torch.dot(grad, grad)

        return lap, grad_sq  # (tr(H) + grad^2) is the kinetic energy in log space