import sys
import time
from pathlib import Path
import config_vmc_vs_exact as config
# Add src/ (the parent directory) to Python path to import functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy


name_of_file = config.name_of_file # correct path to the output file + changing 

start = time.perf_counter() 
vmc_and_exact_energy.vmc_vs_exact(name_of_file)
end = time.perf_counter()

print(f"Total time taken: {end - start:.2f} seconds") 


