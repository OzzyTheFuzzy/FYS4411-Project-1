# Config
import numpy as np

""" System parameters"""
nparticles = 100
dim        = 1
wf_type    = "vmc" 
beta       = 1  #2.82843    # for wavefunction with eliptical trap set beta not 0, for spherical trap set beta = 1
omega_z    = 0.0    # for elliptical trap set omega_z = beta, for spherical trap set omega_z = 1.0

""" Monte Carlo parameters"""
training_cycles = 500000       # this is cycles for training
mcmc_alg        = "langevin" # "metropolis" or "langevin"
scale           = 0.5        # scale for the new proposed position in metropolis algorithm (metropolis 0.2, langevin 0.5)
backend         = "torch"
batch_size      = 200
detailed        = True
num             = False      # Set num=True to calculate the second derivatives with numerical derivation
nsamples        = int(1000)  # 2**18 = 262144
seed            = 142
final_burn_in   = int(nsamples//10)*2

final_sampling_seed = 999
burn_in             = int(training_cycles//10 * 2) # number of initial samples to discard as burn-in when training 10-20%
alpha_array         = np.linspace(0.1, 0.9, 21) # array of alpha values to train on
nchains             = 1 # number of Markov chains
write_to_file       = True    # True if you want to write energies to file
write_to_file_training= True    # True if you want to write energies vs alpha during training to file
name_of_file        = f"testing{nparticles}_d{dim}"  #  name of txt file for energies of last sample
filename            = 'energy_vs_alpha_l.txt' # name of txt file for energies vs alpha during training

""" Gradient descent parameters"""
optimizer = "gd"      # "gd" for gradient descent, "adam" for Adam optimizer
iterations  = 30      # different alphas tested
alpha_0     = None    # starting alpha ( set to None for no optimizer) ()
eta         = 0.001   # learning rate for alpha updates (0.01 is good choice)
need_O = False

""" Inteaction parameter"""
a = 0.0    # Jastrow factor strength, set to 0 for no interactions and to 0.0043 with beta=2.82843 to get the same energy as in project 1 for 10 particles in 3D with elliptical trap. For spherical trap set a=0.
