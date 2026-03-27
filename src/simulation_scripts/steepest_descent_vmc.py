#Program for finding alpha with steepest_descent and plotting alpha vs iteration

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src/ (the parent directory) to Python path to import functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
from simulation_scripts import config_steepest_descent_vmc as config
from qs.functions.write_to_file import write_to_file
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"

def find_alpha_opt():
    dim = config.dim; nparticles = config.nparticles

    system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, config, config.scale)

    # retrieve the alphas and energies from the system object after training
    alphas=system.alpha_array_tested
    energies=system.mean_ana_energies

    # write energies vs alpha during training to file
    if config.write_to_file_training:
        write_to_file(
            [alphas, system.mean_ana_energies, system.accept_rate_array],
            ["alpha", "mean_ana_energy", "acceptance_rate"],
            config.filename,
            data_dir=project_root / "data"
        )

    print(alphas, energies)
    iterations = np.linspace(0,len(alphas)-1, len(alphas))
    print(f'Lowest energy alpha: {system.wf.alpha }')

    # write alpha vs iteration during training to file
    write_to_file([iterations, alphas], ["Iteration", "Alpha"], f"{config.name_of_file}", data_dir=project_root / "data")

find_alpha_opt()

def plot(name_of_file=data_dir / config.name_of_file):
    """ Plot alpha vs iteration from the file generated during training """

    data        = np.loadtxt(name_of_file, skiprows=1) # load the data from the file, skip the header
    iterations  = data[:, 0]
    alphas      = data[:, 1]

    # for making figure of alpha vs iteration
    plt.scatter(iterations, alphas, label="alpha values")
    plt.xlabel("Iteration")
    plt.ylabel(f"$\\alpha$")
    plt.title(f"$\\alpha$ vs iteration for $N={config.nparticles}$ and $a={config.a}$ ")
    plt.grid("True")

    # save the figure to figures folder
    plt.savefig(project_root / "figures" / f"iteration_alpha_a{config.a}_{config.alpha_0}_N{config.nparticles}.pdf", dpi=300)
    plt.show()

plot(data_dir/ config.name_of_file)