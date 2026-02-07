import os
import argparse
import subprocess
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict


def extract_ca_chemical_shifts_from_file(filename):
    """
    Extract CA chemical shifts from the SS_SHIFT column from a file
    """
    ca_shifts = []
    residue_numbers = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Find where the data starts (after the VARS line)
        data_start = False
        for line in lines:
            if line.startswith('VARS'):
                data_start = True
                continue
            if line.startswith('FORMAT'):
                continue
            if data_start and line.strip():
                # Parse the data line
                parts = line.split()
                if len(parts) >= 6 and parts[2] == 'CA':  # Check if it's a CA atom
                    resid = int(parts[0])
                    ss_shift = float(parts[4])  # SHIFT is the 5th column
                    ca_shifts.append(ss_shift)
                    residue_numbers.append(resid)
        
        return residue_numbers, ca_shifts
    
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], []

def create_cs_difference_matrix(ca_shifts):
    """
    Create a square matrix of CA chemical shift differences
    """
    n = len(ca_shifts)
    diff_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            diff_matrix[i, j] = abs(ca_shifts[i] - ca_shifts[j])
    
    return diff_matrix
    
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file

    # Derive pdb_name and create output directory
    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)

    # Input and output for SPARTA+ inside out_dir
    input_file = os.path.join(out_dir, f"{pdb_name}_idr_pro_merged.pdb")
    output_file = os.path.join(out_dir, f"chemical_shift_{pdb_name}.tab")

    # Copy original pdb into out_dir (optional — remove if not needed)
    if not os.path.exists(input_file):
        os.system(f"cp {pdb_file} {input_file}")

    # Run SPARTA+
    if os.path.isfile(input_file):
        print(f"Processing: {input_file}")
        try:
            subprocess.run(
                ["sparta+", "-in", input_file, "-out", output_file],
                check=True
            )
            print(f"Saved chemical shifts: {output_file}")
        except subprocess.CalledProcessError:
            print("Error running SPARTA+")
    else:
        print(f"Warning: File {input_file} not found, skipping...")
        
    residue_numbers, ca_shifts = extract_ca_chemical_shifts_from_file(output_file)
    diff_matrix = create_cs_difference_matrix(ca_shifts) 
    
    # Save inside the directory
    out_path = os.path.join(out_dir, f"{pdb_name}_pairwise_chemical_shifts.pkl")
    pkl.dump(diff_matrix, open(out_path, "wb"))

    print(f"Saved Chemical Shifts to: {out_path}")
        
        
        


