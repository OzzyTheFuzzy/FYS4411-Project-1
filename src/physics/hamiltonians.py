from turtle import rt
import jax
import jax.numpy as jnp
import numpy as np
import torch 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import time
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
            case  "jax":
                self.backend = jnp
                self.la = jnp.linalg
                # You might also be able to jit some functions here
            case _: # noqa
                raise ValueError("Invalid backend:", backend)

    def local_energy(self, wf, r):
        """Local energy of the system"""
        raise NotImplementedError



class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        
        
        super().__init__(nparticles, dim, int_type, backend)

    def local_energy(self, wf, r):
        """Local energy of the system
        Returns both numerical and analytical local energy for comparison
        """

        V = self.potential_energy(r)
        K_num = self.kinetic_energy_numerical(wf, r)
        K_ana = self.kinetic_energy_analytical(wf, r)
       
        E_L_num = K_num + V
        E_L_ana = K_ana + V

        return E_L_num, E_L_ana
    
    def potential_energy(self, r, omega_ho=1.0, omega_z=None):
        
        """Potential energy of the system
        
        r is the position of the particles, shape (nparticles, dim)
        Assuming mass = m = 1
        """
        if self._dim <=2:  #if the particles has less then dimensions of movement
            # Calculating the potential energy
            V = 0.5 * omega_ho * self.backend.sum(r**2)
        else:
            if omega_z==None:
                omega_z = omega_ho 
    
            V = 0.5 * (omega_ho * self.backend.sum(r[:,:-1]**2) + omega_z * self.backend.sum(r[:, -1]**2))

        return V
    
    def kinetic_energy_numerical(self, wf, r):
        
        """Kinetic energy of the system
        Returns the numerical kinetic energy of the system nabla^2 psi
        """
        wf_value = wf.value(r)
        laplacian_wf= wf.laplacian(r)# Need to check out the laplacian function in the wf class and how to use it, also check if there are some other methods that might be useful for this
        K_num = -0.5 * laplacian_wf / wf_value
        return K_num
    
    def kinetic_energy_analytical(self, wf, r):
        """Analytical kinetic energy of the system
        Returns the analytical kinetic energy of the system nabla^2 psi in logspace
        """
        alpha = wf.alpha

        if wf.beta == None:  
             
            K_ana = -0.5 * self.backend.sum((-2.0 * alpha) + (4.0 * alpha**2) * r**2)

        else:
            beta = wf.beta
            K_ana = wf.alpha * self._N *(2+wf.beta) - 2*alpha**2 * self.backend.sum(self.backend.sum(r[:,:-1]**2)) - 2 * alpha**2 * beta**2 * self.backend.sum(self.backend.sum(r[:, -1]**2))
     
        return K_ana
    
    def compute_Laplacian(self, wf, r):
        r = r.clone().detach().requires_grad_(True)

        def logpsi_flat_r(r_flat):
            return self.wf(rt, wf.alpha, wf.beta)
        
        r_flat = r.reshape(-1)

        grad = torch.autograd.grad(logpsi_flat_r(r_flat), r_flat, create_graph=True)[0]

        hessian = torch.functional.hessian(logpsi_flat_r, r_flat)
        
        return wf * (torch.trace(hessian) + torch.dot(grad, grad)) # wf * (tr(H) + grad^2) is the kinetic energy in log space  