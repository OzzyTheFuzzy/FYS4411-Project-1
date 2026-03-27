# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../../data/vmc_vs_exact_test.txt"

""" System parameters """
dim               = 3
nparticles        = 100
omega             = 1.0   # harmonic oscillator frequency
wf_type           = "vmc"
beta              = 2.82843     #beta = 2.82843 for elliptical trap, beta = None for spherical trap
omega_z           = beta     # omexga_z=beta for elliptical trap, omega_z=1.0 for spherical trap

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
eta               = 0.001   # learning rate for steepest descent optimizer.  0.006 for 10 particles with interactions and 0.001 50 and 100 particles with interactions
alpha_0           = 0.4    # put None for using jumps and not gd

""" Interaction parameters """
a    = 0.0043  # Jastrow factor strength (0.0 = no interactions) 0.0043 for standard

"""write to file"""

write_to_file_training = True   # True if you want to write energies vs alpha during training to file
name_of_file           = f"iterations_alpha_a{a}_N{nparticles}.txt"  #  name of txt file for alpha vs iteration during training
filename               = f'energy_vs_alpha_a{a}__N{nparticles}.txt' # name of txt file for energies vs alpha during training







