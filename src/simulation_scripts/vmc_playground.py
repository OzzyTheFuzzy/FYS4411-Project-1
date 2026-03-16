import sys
import matplotlib.pyplot as plt
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
import jax
import numpy as np

from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")



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
    config.nparticles,
    config.dim,
    config.a,
    config.beta
)

# choose the sampler algorithm and scale
system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)

# choose the hamiltonian
system.set_hamiltonian(type_="ho", int_type="Coulomb", omega_ho=1.0, omega_z=config.omega_z)

# choose the optimizer, learning rate, and other properties depending on the optimizer
system.set_optimizer(
    optimizer=config.optimizer,
    eta=config.eta,
)

# train the system, meaning we find the optimal variational parameters for the wave function
system.train(MC_training_cycles=config.training_cycles, 
            alpha_array=config.alpha_array, 
            burn_in=config.burn_in, 
            num_iterations=config.iterations, 
            alpha_0=config.alpha_0, 
            num=config.num)

# make initial state for final sampling and run final sampling
system._make_initial_state() 
results = system.sample(config.nsamples, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num, write_to_file=True)

print(system._scale)
print(results)


# display the results
energies = np.array(system.mean_ana_energies)  # the mean ana energies for each alpha tested
alpha_array = system.alpha_array_tested  # the alpha values that were actually tested during training
plt.figure()
plt.plot(alpha_array, energies, marker="o")
plt.xlabel("Alpha")
plt.ylabel("Energy")
plt.title("Energy vs Alpha")
plt.grid(True)

# Mark best alpha
best_idx = np.argmin(energies)
plt.scatter(alpha_array[best_idx], energies[best_idx])
plt.show()
