import os
import argparse
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

def compute_sasa_difference_map(traj, mode='residue'):
    """
    Compute SASA difference map for a trajectory
    """
    # Compute SASA for each residue
    sasa = md.shrake_rupley(traj, probe_radius=0.14, n_sphere_points=960, mode=mode)
    
    n_residues = sasa.shape[1]
    diff_map = np.zeros((n_residues, n_residues))
    
    # Create difference matrix: |SASA_i - SASA_j|
    for i in range(n_residues):
        for j in range(n_residues):
            diff_map[i, j] = abs(sasa[0, i] - sasa[0, j]) 
    
    return sasa, diff_map
 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file

    # Use pdb_file to derive pdb_name
    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)

    traj = md.load_pdb(pdb_file)

    sasa_values, sasa_diff_map = compute_sasa_difference_map(traj, mode='residue')
    
    # Save inside the directory
    out_path = os.path.join(out_dir, f"{pdb_name}_pairwise_sasa.pkl")
    pkl.dump(sasa_diff_map, open(out_path, "wb"))

    print(f"Saved SASA features to: {out_path}")
    
        
    
    
    
