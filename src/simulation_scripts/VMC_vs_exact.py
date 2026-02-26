# VMC_vs_exact.py
import sys
from os import system
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder

import numpy as np
import matplotlib.pyplot as plt

from physics.exact_energy import exact_energy
from qs import quantum_state
import config

def find_energy_vmc(dim, nparticles, alpha_array):
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
        nparticles,
        dim,
    )

    # choose the sampler algorithm and scale
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=0.2)

    # choose the hamiltonian
    system.set_hamiltonian(type_="ho", int_type="Coulomb", omega_ho=1.0, omega_z=1.0)

    # choose the optimizer, learning rate, and other properties depending on the optimizer
    system.set_optimizer(
        optimizer=config.optimizer,
        eta=config.eta,
    )
    system.train(
    max_iter=config.training_cycles,
    batch_size=config.batch_size,
    alpha_array=alpha_array,
)
    # now we get the results or do whatever we want with them
    sample_results = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)
    alpha = sample_results["alpha"]
    energy_numerical = sample_results["energy_numerical"]
    energy_analytical = sample_results["energy_analytical"]
    
    return energy_analytical, energy_numerical, alpha

def VMC_vs_exact():
    """Compare the VMC results with the exact results for different dimensions and number of particles. 
    """

    alpha_array = np.linspace(0.1, 0.9, 11) # array of alpha values to train on
    dimensions = np.array([1, 2, 3]) # dimensions
    nparticles_array = np.array([1, 10, 100, 500]) # number of particles
    omega = 1.0 # frequency of the harmonic oscillator

    energies_vmc = np.zeros((len(dimensions), len(nparticles_array))) # array to store the VMC energies
    energies_exact = np.zeros((len(dimensions), len(nparticles_array))) # array to store the exact energies

    # Loop over dimensions and number of particles, calculate the energies and store them in the arrays
    for d in range(len(dimensions)):
        for n in range(len(nparticles_array)):
            print(f"Calculating energy for d={dimensions[d]}, n={nparticles_array[n]}")
            energy_analytical, energy_numerical, alpha = find_energy_vmc(dimensions[d], nparticles_array[n], alpha_array) 
            
            print(f'alpha = {alpha}') # print the alpha that gave the lowest energy

            energies_vmc[d, n] = energy_analytical #store the vmc energy
            energies_exact[d, n] = exact_energy(nparticles_array[n], omega, dimensions[d]) #store the exact energy
    # Create matching grids
    dim_grid, n_grid = np.meshgrid(dimensions, nparticles_array, indexing="ij")

    # Flatten everything
    dimensions_flat = dim_grid.flatten()
    n_particles_flat = n_grid.flatten()
    energies_vmc_flat = energies_vmc.flatten()
    energies_exact_flat = energies_exact.flatten()
    np.savetxt("../../data/vmc_results_test.1d.1.10.100.500.txt", np.column_stack((dimensions_flat, n_particles_flat, energies_vmc_flat, energies_exact_flat)),
    header="dimension n_particles energy_vmc energy_exact"     )
    
    return 0




def plot_VMC_vs_exact():
    """
    Plot the VMC vs Exact energies for all dimensions and number of particles
    
    """
    data = np.loadtxt("../../data/vmc_results_test.1d.1.10.100.500.txt", skiprows=1) # load the data from the file, skip the header
    dimensions = data[:, 0]
    n_particles = data[:, 1]
    energies_vmc = data[:, 2]
    energies_exact = data[:, 3]
    plt.figure()

    for d in np.unique(dimensions):
        mask = (dimensions == d)

        N_d = n_particles[mask]
        delta_E_d = energies_vmc[mask] - energies_exact[mask]

        plt.scatter(N_d, delta_E_d, s=100, label=f"d={int(d)}")

        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Number of particles")
        plt.ylabel("Energy difference (VMC - Exact)")
        plt.legend()
        plt.show()

