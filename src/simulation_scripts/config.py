# Config
import numpy as np

""" System parameters"""
nparticles = 50
dim        = 3
wf_type    = "vmc" 
beta       = 2.82843    # for wavefunction with eliptical trap set beta 2.82843, for spherical trap set beta = 1
omega_z    = beta    # for elliptical trap set omega_z = beta, for spherical trap set omega_z = 1.0

""" Monte Carlo parameters"""
training_cycles = 50       # this is cycles for training
mcmc_alg        = "langevin" # "metropolis" or "langevin"
scale           = 0.4      # scale for the new proposed position in metropolis algorithm (metropolis 0.2, langevin 0.5)
backend         = "torch"
batch_size      = 200
detailed        = True
num             = False      # Set num=True to calculate the second derivatives with numerical derivation
nsamples        = int(1000000)  # 2**18 = 262144
seed            = 142
final_burn_in   = int(nsamples//10)*2
omega           = 1.0
final_sampling_seed = 999
burn_in             = int(training_cycles//10 * 2) # number of initial samples to discard as burn-in when training 10-20%
alpha_array         = np.array([0.5]) # array of alpha values to train on
nchains             = 1 # number of Markov chains

"""For one body density"""
obd    = True # set True to compute onebody density, set False to not compute it
n_bins = 25 # number of bins for onebody density
r_max   = 3# maximum radius for onebody density

""" Gradient descent parameters"""
optimizer   = "gd"      # "gd" for gradient descent, "adam" for Adam optimizer
iterations  = 30      # different alphas tested
alpha_0     = 0.0    # starting alpha ( set to None for no optimizer) ()
eta         = 0.001   # learning rate for alpha updates (0.01 is good choice)
need_O      = False
num_iterations= 100 # number of iterations for training with steepest descent

""" Inteaction parameter"""
a = 0.0    # Jastrow factor strength, set to 0 for no interactions and to 0.0043 with beta=2.82843 to get the same energy as in project 1 for 10 particles in 3D with elliptical trap. For spherical trap set a=0.

""" Write to file """
write_obd_to_file   = True    # True if you want to write column density data to file
write_to_file       = True   # True if you want to write energies to file for last sampling
name_of_file        = f"energy_a{a}_N{nparticles}_d{dim}"  #  name of txt file for energies of last sample
filename            = f'column_density_a{a}_N{nparticles}.txt' # name of txt file for column density data

