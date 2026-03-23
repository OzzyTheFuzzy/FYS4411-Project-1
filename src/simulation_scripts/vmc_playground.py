#for testing stuff and finding the best alpha with plot (uses config.py as configuration)
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import jax
import numpy as np
import sys
import torch

#sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
# Project paths
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
data_dir = project_root / "data"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "simulation_scripts"))
from qs.functions.write_to_file import write_to_file
from qs.functions.onebody_density import plot_column_density
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy

from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def train_and_sample_obd():
    # set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
    system = vmc_and_exact_energy.find_energy_vmc(config.dim, config.nparticles, config, config.scale)
    
    # define r_max and reset the sampler with the onebody density settings for final sampling
    r_max  =2 * torch.sqrt(1 / (2 * system.wf.alpha)) if config.r_max is None else config.r_max
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale, obd=config.obd, n_bins=config.n_bins, r_max=r_max)

    # make initial state for final sampling and run final sampling
    system._make_initial_state()

    results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                            num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file, obd=config.obd)
    print(results)
    r_centers, rho =results.get("r_centers"), results.get("rho")

    if config.write_to_file:
        
        write_to_file(
            [r_centers, rho],
            ["r_centers", "rho"],
            config.filename,
            data_dir=data_dir
        )
    
    return system, results

system, results = train_and_sample_obd()


def plot_density(name_of_file=data_dir / config.filename):
    
    data      = np.loadtxt(name_of_file, skiprows=1) # load the data from the file, skip the header
    r_centers = data[:, 0]
    rho       = data[:, 1]


    plot_column_density(r_centers, rho, config)

plot_density(data_dir/config.filename)

