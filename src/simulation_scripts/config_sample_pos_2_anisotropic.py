# Config for anisotropic trap sampling
import numpy as np

""" System parameters"""
nparticles = 1 # does not matter here 
dim = 3         # does not matter here 
wf_type = "vmc"
beta = 2.82843  # anisotropic trap
omega_z = beta
omega = 1.0

""" Monte Carlo parameters"""
training_cycles = 12000 # small number since we have the correct alpha
mcmc_alg = "langevin" 
scale = 0.25
backend = "torch"
batch_size = 200
detailed = True
num = False
nsamples = int(1000000) # huge number for making configurations for PINN in PJ2
seed = 24
final_burn_in = int(nsamples // 10) * 2

final_sampling_seed = 999
burn_in = int(training_cycles // 10 * 2)
alpha_array = np.array([0.5])
nchains = 1
write_to_file = False
write_to_file_training = False
name_of_file = f"testing{nparticles}_d{dim}" # does not matter here 
filename = f"energy_vs_alpha_{mcmc_alg}.txt" # does not matter here 

""" Gradient descent parameters"""
optimizer = "gd"
num_iterations = 30
alpha_0 = None
eta = 0.001
need_O = False

""" Interaction parameter"""
a = 0.0