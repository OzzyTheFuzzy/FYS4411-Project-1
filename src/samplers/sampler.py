import copy

import jax
import numpy as np
from qs.utils import check_and_set_nchains # we suggest you use this function to check and set the number of chains when you parallelize
from qs.utils import generate_seed_sequence
from qs.utils import State
from tqdm.auto import tqdm  # progress bar


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Sampler:
    def __init__(self, rng, scale, hamiltonian, logger=None):

        self._logger = logger
        self.results = None
        self.scale = scale
        self.set_hamiltonian(hamiltonian)

    def sample(self, wf, state, nsamples, nchains=1, seed=None):
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        
        if nchains == 1:
            chain_id = 0
            self.results = self._sample(wf, nsamples, state, self.scale, seeds[0], chain_id)  # ADD THIS LINE
        else:
            # Parallelize
            pass

        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")

        return self.results

    def _sample(self, wf, nsamples, state, scale, seed, chain_id, burn_in=0, num=False):
        """To be called by process. Here the actual sampling is performed."""

        if self._logger is not None:
            t_range = tqdm(
                range(nsamples),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)

        # Config
        state = State(state.positions, state.logp, 0, state.delta)
        analytical_energies = np.zeros(nsamples-burn_in)
        numerical_energies = np.zeros(nsamples-burn_in) 

        for i in t_range:
            metropolis_state = self.step(wf, state, seed) #
            r_new = metropolis_state.positions

            if burn_in is not None and i >= burn_in:
        
                if num == True:
                    num_energy, ana_energy = self.hamiltonian.local_energy(wf, r_new, num) #calculate num, ana energies
                    numerical_energies[i-burn_in] = num_energy.detach()
                else:
                    ana_energy= self.hamiltonian.local_energy(wf, r_new) #calculate num, ana energies

                analytical_energies[i-burn_in] = ana_energy.detach()
            state = metropolis_state # update the state for next iteration


        if self._logger is not None:
            t_range.clear()


        sample_results = {
            "chain_id": chain_id,
            "energy_numerical":  np.mean(numerical_energies), # calculate mean numerical energy
            "energy_analytical": np.mean(analytical_energies), # calculate mean analytical energy
            "std_error": np.std(analytical_energies) / np.sqrt(nsamples), # calculate standard error of the mean for analytical energies
            "variance": np.var(analytical_energies), # calculate variance of the analytical energies
            "accept_rate": state.n_accepted / nsamples, # calculate acceptance rate
            "scale": self.scale, 
            "nsamples": nsamples,
            "alpha": wf.alpha.item(), # get the alpha value from the wave function
        }

        return sample_results

    def set_hamiltonian(self, hamiltonian):

        self.hamiltonian = hamiltonian
