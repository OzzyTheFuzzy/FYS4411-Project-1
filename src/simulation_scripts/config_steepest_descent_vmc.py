# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../../data/vmc_vs_exact_test.txt"

""" System parameters """
dim               = 3
nparticles        = 10
omega             = 1.0   # harmonic oscillator frequency
wf_type           = "vmc"
beta              = 2.82843     #beta = 2.82843 for elliptical trap, beta = 1.0 for spherical trap
omega_z           = beta     # omexga_z=beta for elliptical trap, omega_z=0.0 for spherical trap

""" Monte Carlo parameters """
nsamples          = int(5000)   # samples for the final calculation of the energy after training
final_burn_in     = int(nsamples // 10)
nchains           = 1
training_cycles   = 100000 # number of training cycles for each alpha
num_iterations    = 30     # number of optimization iterations
mcmc_alg          = "langevin"  # either metropolis or langevin 
backend           = "torch"
batch_size        = 200
detailed          = True
final_sampling_seed = 999
seed                = 42
burn_in             = int(training_cycles // 10) * 2  # number of burn-in samples to discard
scale               = 0.4   # scale for the MCMC proposal distribution
num                 = False
alpha_array         = np.linspace(0.1, 0.9, 11)  # array of alphas (not used in GD)

""" Optimizer """
optimizer         = "gd"   # either "gd" or "adam"
eta               = 0.005   # learning rate for steepest descent optimizer 0.01 for 100 particles, 0.0075 for 500 particles. 0.005 for 10 particles with interactions
alpha_0           = 0.4    # put None for using jumps and not gd

""" Interaction parameters """
a    = 0.0043  # Jastrow factor strength (0.0 = no interactions) 0.0043 for standard

"""write to file"""
write_to_file          = False   # True if you want to write energies to file for final sampling (used for blocking)
write_to_file_training = True   # True if you want to write energies vs alpha during training to file
name_of_file           = f"energy_N{nparticles}_d{dim}_.txt"  #  name of txt file for energies of last sample
filename               = f'energy_vs_alpha_a{a}__N{nparticles}_.txt' # name of txt file for energies vs alpha during training







