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


    def local_energy(self, vmc, r, num=False):
        """Local energy of the system
        Returns either numerical or analytical local energy
        """

        V = self.potential_energy(r)

        if num: # for timing the numerical and analytical kinetic energy calculations
            start = time.perf_counter()
            K_num = self.kinetic_energy_numerical(vmc, r)
            end = time.perf_counter()
            t_num = end - start
            E_L_num = K_num + V

            start = time.perf_counter()
            K_ana = self.kinetic_energy_analytical(vmc, r)
            if vmc.a > 0.0:
                K_ana += self.kinetic_energy_jastrow(vmc, r)
            end = time.perf_counter()
            t_ana = end - start
            E_L_ana = K_ana + V

            return E_L_num, E_L_ana, t_num, t_ana

        K_ana = self.kinetic_energy_analytical(vmc, r)
        if vmc.a != 0.0:
            K_ana += self.kinetic_energy_jastrow(vmc, r)
    
        return K_ana + V
        
    def compute_gradient(self, O, E_L):
        """
        Finding gradient with respect to alpha using the formula: dE/dalpha = 2 * ( <E_L O> - <E_L><O> )
        and returns that and mean energy

        E_L: shape (MC_cycles,)
        O: shape (MC_cycles,)
        """

        E_mean = E_L.mean()
        O_mean = O.mean()

        OE_mean = (E_L * O).mean()

        dE_dalpha = 2 * (OE_mean - E_mean * O_mean)

        return dE_dalpha, E_mean

    def potential_energy(self, r):
        """
        Function for gaussian potential energy 
        """
        if self._dim <= 2:
            return 0.5 * self.omega_ho**2 * self.backend.sum(r**2)

        omega_z = self.omega_ho if (self.omega_z is None) else self.omega_z
        return 0.5 * (
            self.omega_ho**2 * self.backend.sum(r[:, :-1]**2)
            + omega_z**2 * self.backend.sum(r[:, -1]**2)
        )

    
    def kinetic_energy_numerical(self, vmc, r):
        
        """Kinetic energy of the system using logspace wf and numerical derivation from torch
        Returns the numerical kinetic energy of the system 
        """
  
        laplacian_logwf, grad_2_logwf= self.compute_gradients(vmc, r)
        K_num = - 0.5 * (laplacian_logwf + grad_2_logwf)
        return K_num
    
    def kinetic_energy_analytical(self, vmc, r):
        
        """Analytical kinetic energy of the system
        Returns the analytical kinetic energy of the system nabla^2 psi in logspace
        """
        alpha = vmc.alpha

        if vmc.beta == None:  
             
            K_ana = -0.5 * (-2.0 * alpha * self._N * self._dim + 4.0 * alpha**2 * self.backend.sum(r**2))

        else:
            beta =vmc.beta
            K_ana = vmc.alpha * self._N *(2+vmc.beta) - 2*alpha**2 * self.backend.sum(self.backend.sum(r[:,:-1]**2)) - 2 * alpha**2 * beta**2 * self.backend.sum(self.backend.sum(r[:, -1]**2))
     
        return K_ana
    
    def compute_gradients(self, vmc, r):

        """
        function for numerical gradients using torch's autograd
        """
        r = r.clone().detach().requires_grad_(True)

        def logpsi_flat(r_flat):
            r_t = r_flat.view(r.shape)  # reshape back to (N, dim)
            return vmc.wf(r_t)

        r_flat = r.view(-1)  # flatten r vector for torch's functions
        logpsi = logpsi_flat(r_flat)
        grad = torch.autograd.grad(logpsi, r_flat, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(logpsi_flat, r_flat)

        lap = torch.trace(hessian) #laplaccien is the trace of the hessian
        grad_sq = torch.dot(grad, grad) # grad^2 is the dot product of the gradient with itself

        return lap, grad_sq  # (tr(H) + grad^2) is the kinetic energy in log space. 
    
    def O_alpha_analytic(self, vmc, r):
        # r: shape (N, dim)
        bk=self.backend

        if vmc.beta is None:
            return -bk.sum(r**2)
        
        beta = vmc.beta
        r_perp2 = bk.sum(r[:, :-1]**2)
        z2 = bk.sum(r[:, -1]**2)
        return -(r_perp2 + beta * z2)   
    

    def kinetic_energy_jastrow(self, vmc, r):
        """
        Finding the analytical kinetic energy for a given wavefunction and positions.

        r: shape (N, dim)

        """
        a = vmc.a
        N = r.shape[0]
        if a == 0 or N==1:
            return 0.0
        
        N  = r.shape[0] #
        bk = self.backend
       

        # retrieve raltive positions between all the particles ij. (rij=rji)
        r_ij_abs, r_ij= vmc.wf.distance_and_distance_vec(r)
        
        # take only upper triangle to acccount for double counting and avoid self-interaction (i=j)
        iu = bk.triu_indices(N, N, offset=1) #for torch
        rij = r_ij_abs[iu[0], iu[1]] 

        if bk.any(rij <= a):
            return -bk.inf
        #retrieve laplacien, graident and cross_term from functions and calculate kinetic energy
        laplacien_log_jastrow = self.laplacien_log_jastrow(rij, a)
        gradient = self.grad_log_jastrow(rij, r_ij, a, iu, r)
        gradient_log_jastrow = bk.sum(gradient**2)
        cross_term = self.cross_term_jastrow(vmc, r, rij, r_ij, a, iu)

        jastrow_kin_energy = -0.5 * (laplacien_log_jastrow + gradient_log_jastrow + cross_term)
        
        return jastrow_kin_energy
        
    def cross_term_jastrow(self, vmc, r, rij, r_ij, a, iu):
        bk = self.backend

        grad_log_gaussian = self.grad_log_gaussian(vmc, r)             # shape (N, dim)
        grad_log_jastrow = self.grad_log_jastrow(rij, r_ij, a, iu, r)  # shape (N, dim)

        cross = 2.0 * bk.sum(grad_log_gaussian * grad_log_jastrow)
        return cross
        
    def laplacien_log_jastrow(self, rij, a):
        """
        calculates the jastrow factor for a given configuration of particles r
        r: shape (N, dim)
        
        """
        bk = self.backend
        # hard-core condition 
        if bk.any(rij <= a):
            return -bk.inf
        
        #calculate laplacien in two steps
        term_1 = 2.0 * bk.sum(2.0 * a / (rij * (rij - a)))
        term_2 = 2.0 * bk.sum(a**2 / (rij**2 * (rij - a)**2))

        laplacien_log_jastrow = term_1 - term_2

   
        return laplacien_log_jastrow


    def grad_log_jastrow(self, rij, r_ij, a, iu, r):

        bk = self.backend
        if bk.any(rij <= a):
            return -bk.inf
        # constants
        A = a / (rij**2 * (rij - a)) 
        
        # the 
        drij = r_ij[iu[0], iu[1], :]
        gij = A[:, None] * drij

        # creating array for gradients
        gradient = bk.zeros_like(r)
        
        # insert 
        gradient.index_add_(0, iu[0], gij)
        gradient.index_add_(0, iu[1], -gij)

        return gradient
    
    def grad_log_gaussian(self, vmc, r):
        """
        function for finding the gradient of the gaussian part of the 
        wavefunction for a given configuration of particles r

        """

        alpha = vmc.alpha

        if vmc.beta is None:
            return -2.0 * alpha * r
        
        # 3D anisotropic case: exp[-alpha(x^2 + y^2 + beta z^2)]
        grad_gaussian = -2.0 * alpha * r.clone()
        grad_gaussian[:, -1] = -2.0 * alpha * vmc.beta * r[:, -1]

        return grad_gaussian

