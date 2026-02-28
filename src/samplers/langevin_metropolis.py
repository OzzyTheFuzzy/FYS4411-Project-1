import torch
import numpy as np
from qs.utils import State

from .sampler import Sampler


class LangevinMetropolis(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)
        self.rng = rng
        
    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis hastings algorithm with Langevin drift

        Arguments:
        wf: the wave function
        state: the current state of the Markov chain
        seed: the random seed for reproducibility
        """

        D = 0.5          # diffusion coefficient for the Langevin proposal
        dt = self.scale  # time step for the Langevin proposal
        

        # Choose one random particle to move and find its index
        n_particles = state.positions.shape[0]
        particle_idx = self.rng.integers(0, n_particles)

        # Create proposed configuration with all particles and define old positions of the particle we want to move
        r_all = state.positions.clone()
        r_old = r_all[particle_idx]
        eta = torch.randn_like(r_old)  # N(0,1) torch tensor for the Langevin proposal

        # Log probability and quantum force at the old position
        log_prop_old = wf.wf.log_prob_single(r_old)
        F_old = wf.wf.quantum_force_single(r_old)

        # Propose new single-particle position using Langevin drift + Gaussian diffusion
        r_prop_single = r_old + D * F_old * dt + (2*D*dt)**0.5 * eta

        # Quantum force at the proposed position
        F_new = wf.wf.quantum_force_single(r_prop_single)

        # Green's function displacement terms
        diff_fwd = r_prop_single - r_old - D * F_old * dt
        diff_bwd = r_old - r_prop_single - D * F_new * dt

        #  find log alpha with the Green's function terms and the log probabilities at the old and proposed positions
        log_prop_new = wf.wf.log_prob_single(r_prop_single)
        logG_fwd = -torch.sum(diff_fwd**2) / (4 * D * dt)   # log G(new | old)
        logG_bwd = -torch.sum(diff_bwd**2) / (4 * D * dt)   # log G(old | new)
        log_alpha = log_prop_new - log_prop_old + logG_bwd - logG_fwd  # log acceptance probability

        # pick out random number and accept/reject the move and update the state accordingly
        u = self.rng.uniform()
        accept = np.log(u) < min(0.0, float(log_alpha))

        if accept:
            new_positions = r_all.clone()
            new_positions[particle_idx] = r_prop_single
            new_logp = state.logp - log_prop_old + log_prop_new
            new_n_accepted = state.n_accepted + 1
        else:
            new_positions = state.positions
            new_logp = state.logp
            new_n_accepted = state.n_accepted
        
        new_state = State(
            positions=new_positions,
            logp=new_logp,
            n_accepted=new_n_accepted,
            delta=state.delta + 1
        )
        return new_state


    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
