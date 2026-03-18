# VMC_vs_exact.py
import sys
import os
import time
from os import system
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder

import numpy as np
import matplotlib.pyplot as plt

from physics.exact_energy import exact_energy
from qs import quantum_state
from simulation_scripts import config_vmc_vs_exact as config



def find_energy_vmc(dim, nparticles, alpha_array, config=config):
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
        config.a
    )

    # choose the sampler algorithm and scale
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)

    # choose the hamiltonian
    system.set_hamiltonian(type_="ho", int_type="Coulomb", omega_ho=config.omega_ho, omega_z=config.omega_z)

    # scale learningrate with # of particles
    eta=config.eta/(np.sqrt(nparticles))

    # choose the optimizer, learning rate, and other properties depending on the optimizer
    system.set_optimizer(
        optimizer=config.optimizer,
        eta=eta,
    )

    # scale the training cycles with the number of particles for better convergence
    training_cycles = config.training_cycles  
    
    # train the system and find the best alpha (scale adapts per alpha internally)
    
  
    system.train(MC_training_cycles=config.training_cycles, 
            alpha_array=config.alpha_array, 
            burn_in=config.burn_in, 
            num_iterations=config.iterations, 
            alpha_0=config.alpha_0, 
            num=config.num)


    # retrieve results for the best alpha with burn-in
    sample_results = system.sample(config.nsamples, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num)


    return sample_results, system

def vmc_vs_exact(name_of_file = "../../data/vmc_results_test.txt"):
    """Compare the VMC results with the exact results for different dimensions and number of particles. 
    """

    alpha_array = config.alpha_array
    dimensions = config.dimensions
    nparticles_array = config.nparticles_array
    omega = config.omega_ho

    energies_vmc = np.zeros((len(dimensions), len(nparticles_array))) # array to store the VMC energies
    energies_exact = np.zeros((len(dimensions), len(nparticles_array))) # array to store the exact energies
    alphas_best = np.zeros_like(energies_vmc)
    std_errors  = np.zeros_like(energies_vmc)
    variances   = np.zeros_like(energies_vmc)
    scales      = np.zeros_like(energies_vmc)
    # Loop over dimensions and number of particles, calculate the energies and store them in the arrays
    
    for d in range(len(dimensions)):
        for n in range(len(nparticles_array)):

            print(f"Calculating energy and std for d={dimensions[d]}, n={nparticles_array[n]}")
            sample_results, system = find_energy_vmc(dimensions[d], nparticles_array[n], alpha_array) 

            # extract the relevant results from the sample_results dictionary
            alpha = sample_results["alpha"]
            energy_analytical = sample_results["energy_analytical"]
            std_error = sample_results["std_error"]
            variance = sample_results["variance"]
            scale = sample_results["scale"]
            
            # store the best alpha, std error, variance, and scale for the current dimension and number of particles   
            alphas_best[d, n] = alpha
            std_errors[d, n] = std_error
            variances[d, n] = variance
            scales[d, n] = float(scale)
            print(f'alpha = {alpha}') # print the alpha that gave the lowest energy

            energies_vmc[d, n] = energy_analytical #store the vmc energy
            energies_exact[d, n] = exact_energy(nparticles_array[n], omega, dimensions[d]) #store the exact energy

    # Create matching grids
    dim_grid, n_grid = np.meshgrid(dimensions, nparticles_array, indexing="ij")

    # Flatten everything
    dimensions_flat      = dim_grid.flatten()
    n_particles_flat     = n_grid.flatten()

    energies_vmc_flat    = energies_vmc.flatten()
    std_errors_flat      = std_errors.flatten()
    variances_flat       = variances.flatten()
    alphas_best_flat     = alphas_best.flatten()
    scales_flat          = scales.flatten()

    energies_exact_flat  = energies_exact.flatten()
    diff_flat            = (energies_vmc - energies_exact).flatten()

    out = np.column_stack((
        dimensions_flat,
        n_particles_flat,
        energies_vmc_flat,
        std_errors_flat,
        variances_flat,
        alphas_best_flat,
        scales_flat,
        energies_exact_flat,
        diff_flat,
    ))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(name_of_file) or '.', exist_ok=True)
    
    np.savetxt(
        name_of_file,
        out,
        header="dimension n_particles E_vmc std_error variance alpha scale E_exact E_diff",
        fmt="%d %d %.12f %.12e %.12e %.12f %.12f %.12f %.12e",
    )
    return 0


def plot_vmc_vs_exact(name_of_file):
    """
    Plot the VMC vs Exact energies for all dimensions and number of particles
    
    """
    data = np.loadtxt(name_of_file, skiprows=1) # load the data from the file, skip the header
    dimensions = data[:, 0]
    n_particles = data[:, 1]
    energies_vmc = data[:, 2]
    energies_exact = data[:, 7]
    plt.figure()

    for d in np.unique(dimensions):
        mask = (dimensions == d)
        print(dimensions, n_particles, energies_vmc, energies_exact)
        N_d = n_particles[mask]
        delta_E_d = energies_vmc[mask] - energies_exact[mask]
        
        plt.scatter(N_d, delta_E_d, s=100, label=f"d={int(d)}")

        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Number of particles")
        plt.ylabel("Energy difference (VMC - Exact)")
        plt.legend()
        plt.show()
