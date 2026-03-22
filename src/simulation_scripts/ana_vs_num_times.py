#analytical vs numerical computational times (uses config.py as configuration
import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append("/Users/oskarfausko/Desktop/compfys 2/Project1/project1/FYS4411-Template/src/") # append yout path to the src folder
data_dir = Path(__file__).resolve().parents[2] / "data"
import os
import jax
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


import config_ana_vs_num_times as config
from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

system = vmc_and_exact_energy.find_energy_vmc(config.dim, config.nparticles, config, config.scale)

# make initial state for final sampling and run final sampling
system._make_initial_state()

if config.num:
    t_ana_tot_final=0
    t_num_tot_final=0
    results, t_ana_tot_final, t_num_tot_final = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)
    
    # store computation times to file
    times = np.column_stack( (t_ana_tot_final/(config.nsamples-config.final_burn_in), t_num_tot_final/(config.nsamples-config.final_burn_in)) )
    filepath = data_dir / config.name_of_time_file
    os.makedirs(filepath.parent, exist_ok=True)
    np.savetxt(filepath, times, header=f"analytical time/MC_cycle, numerical time/MC_cycle for N={config.nparticles} and d={config.dim}", fmt="%.15f  %.15f")
    # prints total time/sample
    print(f"Analytical time per MC cycle: {t_ana_tot_final/(config.nsamples-config.final_burn_in):.10f} seconds")
    print(f"Numerical time per MC cycle: {t_num_tot_final/(config.nsamples-config.final_burn_in):.10f} seconds")

else:
    results = system.sample(config.nsamples, config.final_burn_in, nchains=config.nchains, seed=config.final_sampling_seed, 
                        num=config.num, write_to_file=config.write_to_file, name_of_file=config.name_of_file)



# display the results
energies = np.array(system.mean_ana_energies)  # the mean ana energies for each alpha tested
alpha_array = system.alpha_array_tested  # the alpha values that were actually tested during training
plt.figure()
plt.plot(alpha_array, energies, marker="o")
plt.xlabel("Alpha")
plt.ylabel("Energy")
plt.title("Energy vs Alpha")
plt.grid(True)

# Mark best alpha
best_idx = np.argmin(energies)
plt.scatter(alpha_array[best_idx], energies[best_idx])
plt.show()
