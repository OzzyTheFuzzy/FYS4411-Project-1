# import jax
import torch
import numpy as np
from qs.utils import advance_PRNG_state
from qs.utils import State

from .sampler import Sampler


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)
        self.rng = rng
        
    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis algorithm

        Arguments:
        wf: the wave function
        state: the current state of the Markov chain
        seed: the random seed for reproducibility
        """

        # Choose one random particle to move
        n_particles = state.positions.shape[0]
        particle_idx = self.rng.integers(0, n_particles)

        # Create proposed configuration with all particles
        r_prop = state.positions.clone()
        log_prop_old= wf.wf.log_prob_single(r_prop[particle_idx]) # find the log probability of the current positions of the particle we want to move
        
        r_prop[particle_idx] = state.positions[particle_idx] + self.scale * self.rng.normal(size=state.positions[particle_idx].shape)
        
        logp_prop_new = wf.wf.log_prob_single(r_prop[particle_idx])    # find the log probability of the proposed positions
        
        log_alpha = logp_prop_new - log_prop_old        # calculate the log acceptance probability

        u = self.rng.uniform()                          # pick a random number between 0 and 1
        accept = np.log(u) < min(0.0, float(log_alpha)) # accept the move if log(u) < log_alpha, otherwise reject
        new_positions = torch.where(torch.tensor(accept), r_prop, state.positions) # update positions if accept=true

        log_prop_new_total = state.logp - log_prop_old + logp_prop_new #calculate the new total log probability if we accept the move
        new_logp = torch.where(torch.tensor(accept), log_prop_new_total, state.logp) # update log probability if accepted, otherwise keep the old one

        new_n_accepted = state.n_accepted + accept # update the number of accepted moves

        new_state = State(positions=new_positions, logp=new_logp, n_accepted=new_n_accepted, delta=state.delta + 1)
        if state.delta < 5:
            print(f"log_prop_old: {log_prop_old:.4f}, logp_prop_new: {logp_prop_new:.4f}, log_alpha: {log_alpha:.4f}")

        return new_state


    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
