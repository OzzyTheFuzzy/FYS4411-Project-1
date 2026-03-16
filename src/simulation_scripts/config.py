# Config
import numpy as np

output_filename = "../data/vmc_playground.csv"

""" System parameters"""
nparticles = 1
dim = 1
wf_type = "vmc" 
beta = 1  #2.82843    # for wavefunction with eliptical trap set beta not 0, for spherical trap set beta = 1
omega_z =1.0    # for elliptical trap set omega_z = beta, for spherical trap set omega_z = 1.0

""" Monte Carlo parameters"""
training_cycles = 5_000  # this is cycles for the ansatz
mcmc_alg = "metropolis" # "metropolis" or "langevin"
scale = 0.2         # scale for the new proposed position in metropolis algorithm (metropolis 0.2, langevin 0.5)
backend = "torch"
batch_size = 200
detailed = True
num = False
nsamples = int(1_000_000)  # 2**18 = 262144
seed = 142
final_sampling_seed = 999
burn_in = training_cycles * 0.2 # number of initial samples to discard as burn-in when training 10-20%
alpha_array = np.linspace(0.1, 0.9, 11) # array of alpha values to train on
nchains = 1 # number of Markov chains
write_to_file = True    # True if you want to write energies to file
name_of_file = f"new_laplacien_E_a0.0043_N{nparticles}_d{dim}"  #  name of txt file

""" Gradient descent parameters"""
optimizer = "gd"      # "gd" for gradient descent, "adam" for Adam optimizer
iterations  = 30      # different alphas tested
alpha_0     = None    # starting alpha ( set to None for no optimizer) ()
eta         = 0.001   # learning rate for alpha updates (0.01 is good choice)

""" Inteaction parameter"""
a = 0.0043    # Jastrow factor strength, set to 0 for no interactions and to 0.0043 with beta=2.82843 to get the same energy as in project 1 for 10 particles in 3D with elliptical trap. For spherical trap set a=0.
