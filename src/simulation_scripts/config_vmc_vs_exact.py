# Configuration for VMC_vs_exact.py
import numpy as np


name_of_file = "../../data/vmc_vs_exact_all_configs.txt"

""" System parameters """
nparticles_array = np.array([1, 10, 100, 500])
dimensions = np.array([1, 2, 3])
wf_type = "vmc"
beta = 1               # for elliptical trap set beta != 1, for spherical set beta = 1
omega = 1.0
omega_z = None          # for elliptical trap set omega_z = beta, for spherical set omega_z = 1.0

""" Monte Carlo parameters """
nsamples = int(2**20)  # samples for the final calculation of the energy after training
nchains = 1
training_cycles = 100000
mcmc_alg = "metropolis" # metropolis or langevin
backend = "torch"
batch_size = 200
detailed = True
seed = 42
final_sampling_seed = 999
burn_in = 2 * training_cycles // 10  # 10-20% is usually good
scale = 0.4            # scale for the MCMC proposal distribution
num = False

""" Gradient descent parameters """
optimizer = "gd"
eta = 0.0              # keep this 0 because we do not use optimizer
alpha_array = np.linspace(0.1, 0.9, 11)
alpha_0 = None         # starting alpha (set to None for no optimizer)
iterations = 30

""" Interaction parameter """
a = 0.0                # Jastrow factor strength, set to 0 for no interactions

""" Output """
write_to_file = False
