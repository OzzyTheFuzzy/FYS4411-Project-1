#steepest_descent
import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src/ (the parent directory) to Python path to import functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
from simulation_scripts import config_steepest_descent_vmc as config_steep

dim=config_steep.dim; nparticles=config_steep.nparticles; alpha_array = config_steep.alpha_array

results, system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, alpha_array, config_steep)

plt.scatter(system.alpha_array, system.mean_ana_energies)
plt.xlabel("alpha")
plt.ylabel("Energy")
plt.title(f"E(alpha) for {config_steep.num_iterations} alphas")
#plt.grid("True")
plt.show()
