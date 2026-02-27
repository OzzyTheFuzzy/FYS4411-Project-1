# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../data/vmc_playground.csv"

nparticles = 500
dim = 3
nsamples = int(2**13)  # 4096
nchains = 1
eta = 0.01
training_cycles = 1000
mcmc_alg = "m"
backend = "torch"
optimizer = "gd"
batch_size = 200
detailed = True
wf_type = "vmc"
seed = 42
burn_in = 1000
scale = 0.1

# Arrays for VMC_vs_exact.py
alpha_array = np.linspace(0.1, 0.5, 2)
dimensions = np.array([3])
nparticles_array = np.array([500])
omega = 1.0
