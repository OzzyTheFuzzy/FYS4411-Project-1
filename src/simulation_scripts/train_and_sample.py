# For training and writing to file. Also for making plots for E(alpha) vs alpha
import sys
import matplotlib.pyplot as plt
from pathlib import Path
#sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
# Project paths
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
data_dir = project_root / "data"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "simulation_scripts"))
from qs.functions.write_to_file import write_to_file

import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
import config_train_and_sample as config   # uncomment for alpha vs E with metropolis and hastings


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def train_and_sample():

    system = vmc_and_exact_energy.find_energy_vmc(config.dim, config.nparticles, config, config.scale)

    # make initial state for final sampling and run final sampling
    system._make_initial_state()
    if config.num:
        results, t_ana_tot_final, t_num_tot_final = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                            num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)

    else:
        results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                            num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)
    return system, results

system, results = train_and_sample()

# Extract training results
energies = np.array(system.mean_ana_energies)
alpha_array = np.array(system.alpha_array_tested)
accept_rate = np.array(system.accept_rate_array)

# Save data if requested from config
if config.write_to_file_training:
    filename=config.filename
    write_to_file(
        [alpha_array, energies, accept_rate],
        ["alpha", "mean_ana_energy", "acceptance_rate"],
        config.filename,
        data_dir=data_dir
    )

def plot(name_of_file=data_dir / config.filename):
    print(name_of_file)
    data = np.loadtxt(name_of_file, skiprows=1) # load the data from the file, skip the header
    alpha_array = data[:, 0]
    energies= data[:, 1]
    accept_rate = data[:, 2]
    print("acceptance_rate:", accept_rate)
    plt.figure()
    plt.plot(alpha_array, energies, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("Energy(alpha)")
    plt.title(f"Energy vs alpha for {config.mcmc_alg} algorithm, N={config.nparticles}, dim={config.dim}")
    plt.grid(True)

    # Mark best alpha
    best_idx = np.argmin(energies)
    plt.scatter(alpha_array[best_idx], energies[best_idx])
    plt.show()

plot(data_dir / filename)

