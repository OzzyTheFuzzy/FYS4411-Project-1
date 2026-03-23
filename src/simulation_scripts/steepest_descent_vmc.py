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

project_root = Path(__file__).resolve().parents[2]

dim = config_steep.dim; nparticles = config_steep.nparticles; alpha_array = config_steep.alpha_array

system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, config_steep, config_steep.scale)

print(system.alpha_array_tested, system.mean_ana_energies)
iterations = np.linspace(0,len(system.alpha_array_tested)-1, len(system.alpha_array_tested))

plt.scatter(iterations, system.alpha_array_tested, label="alpha values")
plt.xlabel("Iteration")
plt.ylabel("Alpha")
plt.title(f"Updated alpha using steepest descent for {nparticles} particles in {dim}D")
plt.grid("True")

plt.savefig(project_root / "figures" / f"iteration_alpha_{config_steep.alpha_0}.pdf", dpi=300)
plt.show()
