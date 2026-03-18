#config_langevin_results

# Config
import numpy as np

""" System parameters"""
nparticles = 100 
dim        = 1
wf_type    = "vmc" 
beta       = 1  #2.82843    # for wavefunction with eliptical trap set beta not 0, for spherical trap set beta = 1
omega_z    = 0.0    # for elliptical trap set omega_z = beta, for spherical trap set omega_z = 

""" Monte Carlo parameters"""
training_cycles = 10000  # this is cycles for the ansatz
mcmc_alg        = "langevin" # "metropolis" or "langevin"
dt_array        = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 2.0])  # scale goes as dt. Try these steps [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
backend         = "torch"
batch_size      = 200
detailed        = True
num             = False      # Set num=True to calculate the second derivatives with numerical derivation
nsamples        = int(100000)  # 2**18 = 262144
seed            = 142
final_burn_in   = int(nsamples//10 * 2)

final_sampling_seed = 999
burn_in             = int(training_cycles//10 * 2) # number of initial samples to discard as burn-in when training 10-20%
alpha_array         = np.array([0.5]) # array of alpha values to train on
nchains             = 1 # number of Markov chains
write_to_file       = False    # True if you want to write energies to file
name_of_file        = f"langevin_energies{nparticles}_d{dim}"  #  name of txt file for energies
name_of_time_file   = f"langevin_times{nparticles}_d{dim}"     # name of time file

""" Gradient descent parameters"""
optimizer   = "gd"      # "gd" for gradient descent, "adam" for Adam optimizer
iterations  = 30      # different alphas tested
alpha_0     = None    # starting alpha ( set to None for no optimizer) ()
eta         = 0.001   # learning rate for alpha updates (0.01 is good choice)
need_O      = False
""" Inteaction parameter"""
a = 0.0    # Jastrow factor strength, set to 0 for no interactions and to 0.0043 with beta=2.82843 to get the same energy as in project 1 for 10 particles in 3D with elliptical trap. For spherical trap set a=0.
