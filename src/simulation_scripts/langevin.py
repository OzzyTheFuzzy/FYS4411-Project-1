#testing langevin for different timesteps and plotting the results

import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src/ (the parent directory) to Python path to import functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
from simulation_scripts import config_langevin as config_langevin

dim=config_langevin.dim; nparticles=config_langevin.nparticles; alpha_array = config_langevin.alpha_array
mean_energies = []
accept_array = []

#loop over dt values
for dt in config_langevin.dt_array:

    results, system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, config_langevin, scale=dt)
    mean_energies.append(results["energy_analytical"])
    accept_array.append(results["accept rate"])

for dt, acc in zip(config_langevin.dt_array, accept_array):
    print(f"dt = {dt:.3f} → acceptance = {acc:.4f}")

plt.scatter(config_langevin.dt_array, mean_energies, label="Energy")
plt.xlabel("dt")
plt.ylabel("Energy")
plt.title(f"E(dt)")
#plt.grid("True")
plt.show()
