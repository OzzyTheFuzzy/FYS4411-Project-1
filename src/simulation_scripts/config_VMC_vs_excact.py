# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../data/vmc_playground.csv"

nsamples = int(2**15)   # samples for the final calculation of the energy after training
nchains = 1
eta = 0.01
training_cycles = 4000 # number of training cycles for each alpha (Will be scaled with the number of particles in VMC_vs_exact.py)
mcmc_alg = "m"
backend = "torch"
optimizer = "gd"
batch_size = 200
detailed = True
wf_type = "vmc"
seed = 42
burn_in = training_cycles // 10 # number of burn-in samples to discard
scale = 0.3 # scale for the MCMC proposal distribution (will be tuned with alpha during training)

# Arrays for VMC_vs_exact.py
alpha_array = np.linspace(0.1, 0.9, 11)
dimensions = np.array([1, 2, 3])
nparticles_array = np.array([1, 10, 100, 500])
omega = 1.0
