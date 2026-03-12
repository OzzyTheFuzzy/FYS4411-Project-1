# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../../data/vmc_vs_exact_test.txt"

nsamples = int(2**15)   # samples for the final calculation of the energy after training
nchains = 1
eta = 0.0               #keep this 0 because we do not use optimizer
training_cycles = 20_000 # number of training cycles for each alpha 
mcmc_alg = "metropolis" # metropolis or langevin 
backend = "torch"
optimizer = "gd"
batch_size = 200
detailed = True
wf_type = "vmc"
seed = 42
burn_in = 2 * training_cycles // 10 # number of burn-in samples to discard 10-20% is usually good
scale = 0.4 # scale for the MCMC proposal distribution (will be tuned with alpha during training)

# Arrays for VMC_vs_exact.py
alpha_array = np.linspace(0.1, 0.9, 11)
dimensions = np.array([1, 2])
nparticles_array = np.array([1])
omega = 1.0
a=0.0    # Jastrow factor strength, set to 0 for no interactions