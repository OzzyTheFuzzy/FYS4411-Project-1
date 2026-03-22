# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../../data/vmc_vs_exact_test.txt"

""" System parameters """
dim               = 3
nparticles        = 10
omega             = 1.0   # harmonic oscillator 
wf_type           = "vmc"
beta              = 1.0     #beta = 2.82843 for elliptical trap, beta = 1.0 for spherical trap
omega_z           = 0.0     # omexga_z=beta for elliptical trap, omega_z=0.0 for spherical trap

""" Monte Carlo parameters """
nsamples          = int(2**15)   # samples for the final calculation of the energy after training
final_burn_in     = int(nsamples // 10)
nchains           = 1
eta               = 0.04   # learning rate for steepest descent optimizer
training_cycles   = 500000 # number of training cycles for each alpha
num_iterations    = 30     # number of optimization iterations
mcmc_alg          = "langevin"  # either metropolis or langevin 
backend           = "torch"
batch_size        = 200
detailed          = True

final_sampling_seed = 999
seed                = 42
burn_in             = training_cycles // 10  # number of burn-in samples to discard
scale               = 0.4   # scale for the MCMC proposal distribution
num                 = False
alpha_array         = np.linspace(0.4, 0.6, 11)  # array of alphas (not used in GD)
alpha_0             = 0.4

""" Optimizer """
optimizer         = "gd"   # either "gd" or "adam"

""" Interaction parameters """
a         = 0.0    # Jastrow factor strength (0 = no interactions) 0.0043  




