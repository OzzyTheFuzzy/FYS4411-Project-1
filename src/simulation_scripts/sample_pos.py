#get particle positions 
# (uses config.py as configuration)

import sys
from pathlib import Path
import jax
import sys
import numpy as np
# Project paths
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
data_dir = project_root / "positions_energy_data"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "simulation_scripts"))

from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy #for running simulation

import config_sample_pos as config
from qs.functions.write_to_file import write_to_file

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def train_and_sample_pos():
    # set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
    system = vmc_and_exact_energy.find_energy_vmc(config.dim, config.nparticles, config, config.scale)
    

    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)

    # make initial state for final sampling and run final sampling
    system._make_initial_state()

    # run final sampling and retrieve all the positions and energy
    
    r_all, E_ana = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, write_pos_to_file=True)
    
    # Flatten positions
    positions_flat = r_all.reshape(r_all.shape[0], -1)

    # Convert energies into numpy array and reshape to be a column vector for each MC cycle
    energies = E_ana.detach().cpu().numpy().reshape(-1, 1)

    # Combine
    out = np.column_stack((positions_flat, energies))

    # Coordinate labels for header
    coords = ["x", "y", "z"]

    # Build header dynamically to support 1, 2 and 3 dimensions
    header_list = []
    for i in range(config.nparticles):
        for d in range(config.dim):
            header_list.append(f"p{i}_{coords[d]}")

    header_list.append("E_ana")
    header = " ".join(header_list)

    # Save file
    filepath = data_dir / f"{config.name_of_file_positions}.txt"

    np.savetxt(filepath, out, header=header, fmt="%.12f")

    print("Saved to:", filepath)
    return 0

train_and_sample_pos()