# import jax
import numpy as np
from qs.utils import advance_PRNG_state
from qs.utils import State

from .sampler import Sampler


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)

    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis algorithm

        Arguments:
        wf: the wave function
        state: the current state of the Markov chain
        seed: the random seed for reproducibility
        """
        
        r_prop = state.positions + self.scale * self.rng.normal(state.positions.shape) # propose new positions
        logp_prop = wf.log_prob(r_prop) #find the log probability of the proposed positions
        log_alpha = logp_prop - state.logp # calculate the log acceptance probability

        u = self.rng.uniform() #pick a random number between 0 and 1
        accept = np.log(u) < min(0.0, float(log_alpha)) #accept is either true or false
        new_positions = np.where(accept, r_prop, state.positions) # update positions if accept=true
        new_logp = np.where(accept, logp_prop, state.logp) # update log probability if accepted, otherwise keep the old one
        new_n_accepted = state.n_accepted + accept # update the number of accepted moves

        new_state = State(positions=new_positions, logp=new_logp, n_accepted=new_n_accepted, delta=state.delta + 1)
        return new_state


    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
