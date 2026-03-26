#statistical_analysis.py
# Add src/ (the parent directory) to Python path to import functions
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


def error(filename):
    """
    Calls function for std and variance with correct filename
    """
    data = np.loadtxt(filename)
    E_ana_array = data[:, 0]  # second column = E_num

    variance_array, error_array, B_list, n_list = blocking.blocking_error(E_ana_array)

    return variance_array, error_array, B_list, n_list

filename = Path(__file__).resolve().parents[2] / "data" / f"{config.name_of_file}"
print(filename)
variance_array, error_array, B_list, n_list = error(filename)


def plot(x, y):

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Block size B")
    plt.ylabel("Standard error of the mean")
    plt.title(f"Standard error of the mean vs block size for N={config.nparticles}")
    plt.grid(True)
    plt.savefig(Path(__file__).resolve().parents[2] / "figures" / f"error_plot_a{config.a}_N{config.nparticles}.pdf")
    plt.show()

plot(B_list, error_array)

print(f"The SEM for N={config.nparticles} particles and a={config.a} is {error_array[-1]:.5f}", )