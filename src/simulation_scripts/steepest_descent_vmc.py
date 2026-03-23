#steepest_descent
import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src/ (the parent directory) to Python path to import functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
from simulation_scripts import config_steepest_descent_vmc as config
from qs.functions.write_to_file import write_to_file
project_root = Path(__file__).resolve().parents[2]

dim = config.dim; nparticles = config.nparticles; alpha_array = config.alpha_array

system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, config, config.scale)
alphas=system.alpha_array_tested
energies=system.mean_ana_energies

if config.write_to_file_training:
    filename=config.filename
    write_to_file(
        [alphas, system.mean_ana_energies, system.accept_rate_array],
        ["alpha", "mean_ana_energy", "acceptance_rate"],
        config.filename,
        data_dir=project_root / "data"
    )

if config.write_to_file:
    results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                            num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)

print(alphas, energies)
iterations = np.linspace(0,len(alphas)-1, len(alphas))

plt.scatter(iterations, alphas, label="alpha values")
plt.xlabel("Iteration")
plt.ylabel("Alpha")
plt.title(f"Updated alpha using steepest descent for {nparticles} particles in {dim}D")
plt.grid("True")

plt.savefig(project_root / "figures" / f"iteration_alpha_{config.alpha_0}_N{nparticles}.pdf", dpi=300)
plt.show()