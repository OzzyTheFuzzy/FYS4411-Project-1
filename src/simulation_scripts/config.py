# Config
import numpy as np

output_filename = "../data/vmc_playground.csv"

nparticles = 100
dim = 3
nsamples = int(10_000)  # 2**18 = 262144
nchains = 1 # number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.01
training_cycles = 10_000  # this is cycles for the ansatz
mcmc_alg = "metropolis" # "metropolis" or "langevin"
scale = 0.9
backend = "torch"
optimizer = "gd"
batch_size = 200
detailed = True
wf_type = "vmc" 
seed = 142
burn_in = training_cycles * 0.2 # number of initial samples to discard as burn-in when training
alpha_array = np.linspace(0.1, 0.9, 11) # array of alpha values to train on
a = 0.01     # Jastrow factor strength, set to 0 for no interactions
