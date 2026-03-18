#for testing stuff and finding the best alpha with plot (uses config.py as configuration)
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

from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")



# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
system = quantum_state.QS(
    backend=config.backend,
    log=True,
    logger_level="INFO",
    seed=config.seed,
)

# set up the wave function with some of its properties 
system.set_wf(
    config.wf_type,
    config.nparticles,
    config.dim,
    config.a,
)

# choose the sampler algorithm and scale
system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)

# choose the hamiltonian
system.set_hamiltonian(type_="ho", int_type="Coulomb", omega_ho=1.0, omega_z=config.omega_z)

# choose the optimizer, learning rate, and other properties depending on the optimizer
system.set_optimizer(
    optimizer=config.optimizer,
    eta=config.eta,
)

# train the system, meaning we find the optimal variational parameters for the wave function
system.train(MC_training_cycles=config.training_cycles, 
            alpha_array=config.alpha_array, 
            burn_in=config.burn_in, 
            num_iterations=config.iterations, 
            alpha_0=config.alpha_0, 
            num=config.num)


# make initial state for final sampling and run final sampling
system._make_initial_state()
if config.num:
    results, t_ana_tot_final, t_num_tot_final = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)

else:
    results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)

# Extract training results
energies = np.array(system.mean_ana_energies)
alpha_array = np.array(system.alpha_array_tested)
accept_rate = np.array(system.accept_rate_array)

# Save data if requested
filename=config.filename
if config.write_to_file:
    write_to_file(
        [alpha_array, energies, accept_rate],
        ["alpha", "mean_ana_energy", "acceptance_rate"],
        "energy_vs_alpha.txt",
        data_dir=data_dir
    )

def plot(name_of_file):
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
    plt.title(f"Energy vs alpha for standard metropolis sampling")
    plt.grid(True)

    # Mark best alpha
    best_idx = np.argmin(energies)
    plt.scatter(alpha_array[best_idx], energies[best_idx])
    plt.show()

plot(data_dir / filename)
