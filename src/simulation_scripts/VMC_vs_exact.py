from vmc_and_exact_energy import vmc_vs_exact, plot_vmc_vs_exact
import time


name_of_file = "../data/vmc_results_run.txt"

start = time.perf_counter()
vmc_vs_exact(name_of_file)
end = time.perf_counter()

print(f"Total time taken: {end - start:.2f} seconds") 
plot_vmc_vs_exact(name_of_file)