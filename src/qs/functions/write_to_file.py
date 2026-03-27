
import numpy as np
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

def write_to_file(arrays, names, name_of_file, data_dir=DEFAULT_DATA_DIR):
    """
    arrays: list of numpy arrays
    names: list of column names (strings)
    name_of_file: output filename (string)
    """
    
    # Stack arrays column-wise
    out = np.column_stack(arrays)

    # Create full path
    filepath = data_dir / name_of_file
    print(f"Saving to: {filepath}")
    
    # Ensure directory exists
    os.makedirs(filepath.parent, exist_ok=True)

    # Create header
    header = " ".join(names)

    # Save file
    np.savetxt(filepath, out, header=header, fmt="%.12f")