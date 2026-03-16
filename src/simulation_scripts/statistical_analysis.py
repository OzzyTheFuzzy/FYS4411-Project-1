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
import config
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qs.functions import blocking as blocking


def error(filename):
    """
    Calls function for std and variance with correct filename
    """
    E_ana_array = np.loadtxt(filename)
    variance_array, error_array, B_list, n_list = blocking.blocking_error(E_ana_array)

    return variance_array, error_array, B_list, n_list

filename = Path(__file__).resolve().parents[1] / "data" / f"{config.name_of_file}.txt"

variance_array, error_array, B_list, n_list = error(filename)


def plot(x, y):

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Block size B")
    plt.ylabel("Estimated standard error")
    plt.title("Blocking estimate of standard error")
    plt.grid(True)
    plt.show()

plot(B_list, error_array)
plot(B_list, variance_array, )
print(B_list, error_array, variance_array, n_list )