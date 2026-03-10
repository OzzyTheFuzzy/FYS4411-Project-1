# Configuration for VMC_vs_exact.py
import numpy as np

output_filename = "../../data/vmc_vs_exact_test.txt"

nsamples = int(2**15)   # samples for the final calculation of the energy after training
nchains = 1
eta = 0.12 # learning rate for steepest descent optimizer
training_cycles = 10_000 # number of training cycles for each alpha 
num_iterations = 50 # number of different alphas
mcmc_alg = "langevin" # either user metropolis or langevin 
backend = "torch"
optimizer = "adam" # either "gd" or "adam"
batch_size = 200
detailed = True
wf_type = "vmc"
seed = 42
burn_in = 1 * training_cycles // 10 # number of burn-in samples to discard
scale = 1.5 # scale for the MCMC proposal distribution (will be tuned with alpha during training)

# Arrays for VMC_vs_exact.py
alpha_array = np.array([0.9])
dim = 1
nparticles = 100
omega = 1.0 # harmonic oscillator 