import os
import argparse
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict


def gram_schmidt_rotation_matrix(N, CA, C):
    v1 = C - CA
    v2 = N - CA
    e1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(e1, v2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    R = np.column_stack((e1, e2, e3))
    return R

def get_backbone_atoms(traj, chain_id):
    residues_dict = {}
    chain = traj.topology.chain(chain_id)
    for residue in chain.residues:
        residue_atoms = {}
        for atom in residue.atoms:
            if atom.name in ['N', 'CA', 'C']:
                atom_idx = atom.index
                coords = traj.xyz[0, atom_idx]
                residue_atoms[atom.name] = coords
        if len(residue_atoms) == 3:
            residues_dict[residue.index] = residue_atoms
    return residues_dict

def calculate_rotation_matrices(residues_dict):
    rotation_matrices = []
    for residue_index, atoms in residues_dict.items():
        try:
            R = gram_schmidt_rotation_matrix(atoms['N'], atoms['CA'], atoms['C'])
            rotation_matrices.append(R)
        except:
            continue
    return rotation_matrices
    
def calculate_pairwise_orientation_matrices(rotation_matrices_list):
    """
    Calculate pairwise relative orientation matrices for each PDB file.
    
    Args:
        rotation_matrices_list: List where each element is a concatenated array of 
                               rotation matrices for one PDB file
    
    Returns:
        pairwise_orientations: List of 3D arrays where each array has shape 
                             (n_residues, n_residues, 3, 3) containing 
                             relative rotation matrices R_ij = R_j @ R_i.T
    """
    pairwise_orientations = []
    
    for rotation_matrices in tqdm(rotation_matrices_list, desc="Calculating pairwise orientations"):
        n_residues = len(rotation_matrices)
        
        # Initialize array for pairwise relative rotations
        # Shape: (n_residues, n_residues, 3, 3)
        pairwise_orientation = np.zeros((n_residues, n_residues, 3, 3))
        
        # Calculate relative rotation for each pair (i, j)
        for i in range(n_residues):
            for j in range(n_residues):
                R_i = rotation_matrices[i]  # Rotation matrix for residue i
                R_j = rotation_matrices[j]  # Rotation matrix for residue j
                
                # Calculate relative rotation: R_ij = R_j @ R_i.T
                R_ij = R_j @ R_i.T
                pairwise_orientation[i, j] = R_ij
        
        pairwise_orientations.append(pairwise_orientation)
    
    return pairwise_orientations
    
def rotation_matrix_to_angle(R):
    """
    Extract rotation angle from 3x3 rotation matrix.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        angle: rotation angle in radians (0 to π)
    """
    # The trace formula: trace(R) = 1 + 2*cos(θ)
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    
    # Clamp to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate angle
    angle = np.arccos(cos_theta)
    
    return angle    
 
 
def create_angle(pairwise_orientation):
    """Convert rotation matrices to angle-based edge features for graph k"""
    n_residues = pairwise_orientation.shape[0]
    
    # Initialize angle matrix (symmetric)
    angle_matrix = np.zeros((n_residues, n_residues))
    
    for i in range(n_residues):
        for j in range(n_residues):
            R_ij = pairwise_orientation[i, j]
            angle_matrix[i, j] = rotation_matrix_to_angle(R_ij)
    
    return angle_matrix
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file

    # Use pdb_file to derive pdb_name
    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(pdb_file):
        try:
            # Load PDB file
            traj = md.load_pdb(pdb_file)

            # Get rotation matrices for both chains
            idr_residues = get_backbone_atoms(traj, 0)
            protein_residues = get_backbone_atoms(traj, 1)

            idr_matrices = calculate_rotation_matrices(idr_residues)
            protein_matrices = calculate_rotation_matrices(protein_residues)

            # Store in a single list (NOT list of lists)
            all_rotation_matrices = np.concatenate([idr_matrices, protein_matrices])

        except Exception as e:
            print(f"Error processing file {pdb_file}: {e}")

    else:
        print(f"File not found: {pdb_file}")
        exit()

    # Calculate pairwise orientations
    pairwise_orientations = calculate_pairwise_orientation_matrices([all_rotation_matrices])

    angle_features = []
    for idx in tqdm(range(len(pairwise_orientations))):
        angle_features.append(create_angle(pairwise_orientations[idx]))

    # Save inside the directory
    out_path = os.path.join(out_dir, f"{pdb_name}_pairwise_angles.pkl")
    pkl.dump(angle_features[0], open(out_path, "wb"))

    print(f"Saved angle features to: {out_path}")
    
        
    
    
    
