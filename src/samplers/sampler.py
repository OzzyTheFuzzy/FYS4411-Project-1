import copy
from pathlib import Path
import sys
import os
import time

data_dir = Path(__file__).resolve().parents[2] / "data"
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder

import jax
import numpy as np
import torch
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
        self.burn_in = None
        self.set_hamiltonian(hamiltonian)



    def _sample_energy_and_optional_O(self, wf, state, MC_training_cycles,seed, burn_in=0, need_O = False, num= False):
        """
        Run an MCMC batch at fixed alpha and return:
        - E_ana: tensor of analytic local energies (shape [n_samples])
        - E_num: tensor of numeric local energies (shape [n_samples]) or None if num=False
        - O:     tensor of O_alpha values (shape [n_samples]) or None if need_O=False
        - accept_rate: float
        """

        E_ana_list = []
        E_num_list = [] if num else None
        t_ana_tot = 0
        t_num_tot = 0
        O_list = [] if need_O else None

        for i in range(MC_training_cycles):
            state = self.step(wf, state, seed)
            
            if i < burn_in:
                continue

            if num:
                E_ana, t_ana, V = self.hamiltonian.local_energy(wf, state.positions, num=True)
                t_ana_tot+=t_ana

                E_num, t_num = self.hamiltonian.numerical_energy(wf, state.positions, V)
                t_num_tot+=t_num

                E_num_list.append(E_num.detach())
                E_ana_list.append(E_ana.detach())
            else:
                E_ana = self.hamiltonian.local_energy(wf, state.positions, num=False)
                E_ana_list.append(E_ana.detach())

            if need_O:
                O_val = self.hamiltonian.O_alpha_analytic(wf, state.positions)
                O_list.append(O_val.detach())

        E_ana = torch.stack(E_ana_list)
        E_num = torch.stack(E_num_list) if num else None
        O = torch.stack(O_list) if need_O else None

        accept_rate = state.n_accepted / MC_training_cycles
        
        if num:
            return E_ana, E_num, O, accept_rate, t_ana_tot, t_num_tot 
        
        return E_ana, E_num, O, accept_rate

    def _sample(self, wf, nsamples, state, scale, seed, chain_id, burn_in=0, num=False, write_to_file=False, name_of_file="energy"):
        """
        Function for final sampling 
        """
        
        # Use different seed for final alpha

        if num:
            E_ana, E_num, _, accept_rate, t_ana_tot, t_num_tot = self._sample_energy_and_optional_O(
            wf=wf, state=state,
            MC_training_cycles=nsamples, 
            seed=seed, 
            burn_in=burn_in,
            num=num, )

        else:
            E_ana, E_num, _, accept_rate= self._sample_energy_and_optional_O(
            wf=wf, state=state,
            MC_training_cycles=nsamples, 
            seed=seed, 
            burn_in=burn_in,
            num=num, )

        n_effective = nsamples - burn_in
        
        # compute mean energies 
        mean_ana_energy = E_ana.mean().item()
        mean_num_energy = E_num.mean().item() if num else mean_ana_energy

        sample_results = {
            "chain_id": chain_id,
            "energy_numerical":  mean_num_energy,  # calculate mean numerical energy 
            "energy_analytical": mean_ana_energy,  # calculate mean analytical energy
            "std_error": ( E_ana.std(unbiased=True) 
                        / torch.sqrt(torch.tensor(n_effective, dtype=E_ana.dtype))).item(),
            "variance": E_ana.var(unbiased=True).item(),
            "scale": self.scale, 
            "effective samples": n_effective,
            "MC cycles": nsamples,
            "alpha": wf.alpha.item(), # get the alpha value from the wave function
            "accept rate": accept_rate,
            }
        
        if write_to_file:
            out = np.column_stack( ( E_ana.detach().cpu().numpy().flatten(),
            E_num.detach().cpu().numpy().flatten() if num else E_ana.detach().cpu().numpy().flatten()) )

            filepath = data_dir / name_of_file
            os.makedirs(filepath.parent, exist_ok=True)
            np.savetxt(filepath, out, header="E_ana E_num", fmt="%.12f  %.12f")
            
            
        if num:
            return sample_results, t_ana_tot, t_num_tot
        
        return sample_results

    def set_hamiltonian(self, hamiltonian):

        self.hamiltonian = hamiltonian

