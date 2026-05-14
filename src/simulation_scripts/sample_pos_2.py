#sample_pos_2.py

# For training and writing to file. Also for making plots for E(alpha) vs alpha
import sys
import matplotlib.pyplot as plt
from pathlib import Path
#sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
# Project paths
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
data_dir = project_root / "positions_energy_data"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "simulation_scripts"))
from qs.functions.write_to_file import write_to_file

import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
import config_sample_pos_2 as config  


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def train_and_sample():

    system = vmc_and_exact_energy.find_energy_vmc(config.dim, config.nparticles, config, config.scale)

    # make initial state for final sampling and run final sampling
    system._make_initial_state()
    # set seed =24 for writing positions to file
    results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=24, 
                           num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)
    return system, results

system, results = train_and_sample()

#data = np.load(data_dir / "r_all_E_N1_d3.npz")
#r_all = data["r_all"]
#E_ana = data["E"]

#rint(r_all.shape)