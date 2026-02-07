import os
import argparse
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

   
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

    # Get atom indices for CA atoms in each chain
    chain_atoms = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']


    # Build pairs of all residues between chain A and chain B
    pairs = np.array([[i, j] for i in chain_atoms for j in chain_atoms])

    # Compute distances
    distances = md.compute_distances(traj, pairs)[0]  # shape: (n_pairs,)
    distance_map = distances.reshape((len(chain_atoms), len(chain_atoms)))
    
    # Save inside the directory
    out_path = os.path.join(out_dir, f"{pdb_name}_pairwise_distance.pkl")
    pkl.dump(distance_map*10, open(out_path, "wb"))

    print(f"Saved distance features to: {out_path}")
    
        
    
    
    
