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
        r = state.positions
        # Create array for proposed configuration with all particles
        r_prop = r.clone()

        # find the log probability of the current positions of the particle we want to move
        log_prop_old= wf.wf.log_prob_single(r_prop[particle_idx], r, particle_idx) 
        
        # change the position of the chosen particle by adding a random displacement
        r_prop[particle_idx] = state.positions[particle_idx] + self.scale * self.rng.normal(size=state.positions[particle_idx].shape)
        
        # find the log probability of the proposed positions of the particle we want to move
        logp_prop_new = wf.wf.log_prob_single(r_prop[particle_idx], r_prop, particle_idx)    # find the log probability of the proposed positions
        
        # calculate the log acceptance probability
        log_alpha = logp_prop_new - log_prop_old       

        # draw random number and accept/reject the move and update the state accordingly
        u = self.rng.uniform()
        accept = np.log(u) < min(0.0, log_alpha.detach().item()) # detach gradients for alpha updates and convert to scalar for comparison

        if accept:
            new_positions = r_prop
            new_logp = state.logp - log_prop_old + logp_prop_new
            new_n_accepted = state.n_accepted + 1
        else:
            new_positions = state.positions
            new_logp = state.logp
            new_n_accepted = state.n_accepted


        new_delta = state.delta + 1


        """
        # periodic resync every 100 steps to prevent logp drift 
        if new_delta % 100 == 0:
            new_logp = wf.wf.log_prob(new_positions)
        """

        new_state = State(
            positions=new_positions,
            logp=new_logp,
            n_accepted=new_n_accepted,
            delta=state.delta + 1,
            obd=state.obd,
            n_bins=state.n_bins,
            r_max=state.r_max,
        )
        return new_state


    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
