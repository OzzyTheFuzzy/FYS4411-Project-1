
# This script performs the statistical analysis of the VMC energy data 
# this program uses config.py for configuration, and the data file generated during the final sampling in sample_column_density.py 
import sys
import matplotlib.pyplot as plt
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
import jax
import numpy as np
#steepest_descent
import sys
import time
import config as config
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qs.functions import blocking as blocking
from qs.functions.write_to_file import write_to_file as write_to_file


def error(name_of_file):
    """
    Calls function for std and variance with correct filename
    """
    data = np.loadtxt(name_of_file)
    E_ana_array = data[:, 0]  # second column = E_num
    print(f'VMC energy for {config.nparticles} particles and a={config.a}: {np.mean(E_ana_array):.9f}')

    variance_array, error_array, B_list, n_list = blocking.blocking_error(E_ana_array)

    return variance_array, error_array, B_list, n_list

name_of_file= Path(__file__).resolve().parents[2] / "data" / f"{config.name_of_file}"

variance_array, error_array, B_list, n_list = error(name_of_file)


def plot(x, y):
    """ Plots the standard error of the mean as a function of block size B """

    name_of_plot=f"error_plot_a{config.a}_N{config.nparticles}.pdf"

    if config.beta is None:
        name_of_plot=f"error_plot_a{config.a}_N{config.nparticles}_spherical.pdf"

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Block size B")
    plt.ylabel("Standard error of the mean")
    plt.title(f"Standard error of the mean vs block size for N={config.nparticles}")
    plt.grid(True)
    plt.savefig(Path(__file__).resolve().parents[2] / "figures" / name_of_plot)
    plt.show()

plot(B_list, error_array)

# write statistical data to file for spherical an non-spherical case
if config.beta is None:
    name_of_file=f"error_analysis_a{config.a}_N{config.nparticles}_spherical.txt"
else:
    name_of_file=f"error_analysis_a{config.a}_N{config.nparticles}.txt"

write_to_file(arrays=[variance_array, error_array, B_list, n_list], 
              names=["Variance (Varr)", "Standard Error (SEM)", "Block Size (B)", "Number of Blocks (n)"],
                name_of_file=name_of_file)
