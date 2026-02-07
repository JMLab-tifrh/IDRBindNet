import os
import torch
import argparse
import subprocess
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data 
from sklearn.preprocessing import StandardScaler

def create_graph_for_complex(idr_cutoff_distance, protein_cutoff_distance, protein_idr_cutoff_distance, idr_length, protein_length, idr_embeddings, protein_embeddings, distance_map, rotation_map, chemical_shift_map, sasa_map):
    """Create a graph for a single IDR-protein complex"""
      
    total_length = idr_length + protein_length
     
    assert distance_map.shape == (total_length, total_length), f"Distance map shape mismatch"
    assert rotation_map.shape == (total_length, total_length), f"Rotation map shape mismatch"
    assert chemical_shift_map.shape == (total_length, total_length), f"Chemical shift map shape mismatch"
    assert sasa_map.shape == (total_length, total_length), f"SASA map shape mismatch"
    
    
    assert idr_embeddings.shape[0] == idr_length, f"IDR embedding length mismatch for index {idx}"
    assert protein_embeddings.shape[0] == protein_length, f"Protein embedding length mismatch for index {idx}"
    
    # Combine node features (IDR + Protein)
    node_features = np.vstack([idr_embeddings, protein_embeddings])
    
    # Create edge indices and edge features
    edge_indices = []
    edge_features = []
    residue_mapping = {}  # For the dictionary output
    
    # Build residue mapping dictionary
    for i in range(total_length):
        if i < idr_length:
            residue_mapping[i] = ('idr', i)  # (type, 0-indexed residue index)
        else:
            residue_mapping[i] = ('protein', i - idr_length)  # (type, 0-indexed residue index)
    
    # Create edges based on cutoff distances
    for i in range(total_length):
        for j in range(i + 1, total_length):  # Avoid self-loops and duplicates
            distance = distance_map[i, j]
            
            # Determine residue types
            i_type = 'idr' if i < idr_length else 'protein'
            j_type = 'idr' if j < idr_length else 'protein'
            
            # Apply appropriate cutoff based on residue types
            if i_type == 'protein' and j_type == 'idr' or i_type == 'idr' and j_type == 'protein':
                if distance <= protein_idr_cutoff_distance:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Undirected graph
                    edge_feature = [
                        distance,
                        rotation_map[i, j],
                        chemical_shift_map[i, j],
                        sasa_map[i, j]
                    ]
                    edge_features.append(edge_feature)
                    edge_features.append(edge_feature)  # Same feature for reverse edge
                    
            elif i_type == 'protein' and j_type == 'protein':
                if distance <= protein_cutoff_distance:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    edge_feature = [
                        distance,
                        rotation_map[i, j],
                        chemical_shift_map[i, j],
                        sasa_map[i, j]
                    ]
                    edge_features.append(edge_feature)
                    edge_features.append(edge_feature)
                    
            elif i_type == 'idr' and j_type == 'idr':
                if distance <= idr_cutoff_distance:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    edge_feature = [
                        distance,
                        rotation_map[i, j],
                        chemical_shift_map[i, j],
                        sasa_map[i, j]
                    ]
                    edge_features.append(edge_feature)
                    edge_features.append(edge_feature)
    
    # Convert to tensors
    if edge_indices:  # Check if there are any edges
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Create empty tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create graph data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.tensor(StandardScaler().fit_transform(edge_attr), dtype=torch.float),
        num_nodes=total_length)
    
    return graph_data, residue_mapping
 


  
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file
    
    # Cutoff distances
    protein_idr_cutoff_distance = 8
    protein_cutoff_distance = 6
    idr_cutoff_distance = 6

    # Derive pdb_name and create output directory
    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)
    
    embed_idr_file = os.path.join(out_dir, f"{pdb_name}_idr_embedding.npy")
    embed_pro_file = os.path.join(out_dir, f"{pdb_name}_protein_embedding.npy")
    distance_file = os.path.join(out_dir, f"{pdb_name}_pairwise_distance.pkl")
    rotation_file = os.path.join(out_dir, f"{pdb_name}_pairwise_angles.pkl")
    cs_file = os.path.join(out_dir, f"{pdb_name}_pairwise_chemical_shifts.pkl")
    sasa_file = os.path.join(out_dir, f"{pdb_name}_pairwise_sasa.pkl") 
    
    traj = md.load_pdb(pdb_file)
    chain_lengths = [len(list(chain.residues)) for chain in traj.topology.chains]
    
    idr_length=chain_lengths[0]
    protein_length=chain_lengths[1]    
    
    idr_embeddings=np.load(f'{embed_idr_file}', allow_pickle=True)
    idr_embeddings=StandardScaler().fit_transform(idr_embeddings)
    protein_embeddings=np.load(f'{embed_pro_file}', allow_pickle=True)
    protein_embeddings=StandardScaler().fit_transform(protein_embeddings)
    distance_map=pkl.load(open(f'{distance_file}', 'rb'))
    rotation_map=pkl.load(open(f'{rotation_file}', 'rb'))
    chemical_shift_map=pkl.load(open(f'{cs_file}', 'rb'))
    sasa_map=pkl.load(open(f'{sasa_file}', 'rb'))
    
    # Build graph
    graph_data, residue_mapping = create_graph_for_complex(idr_cutoff_distance, protein_cutoff_distance, protein_idr_cutoff_distance, idr_length, protein_length, idr_embeddings, protein_embeddings, distance_map, rotation_map, chemical_shift_map, sasa_map)
    print(f"Graph succesfully created for {pdb_name}.pdb")
          
    # Save graph inside the directory
    out_path = os.path.join(out_dir, f"{pdb_name}_graph.pkl")
    pkl.dump(graph_data, open(out_path, "wb"))

    print(f"Saved graph to: {out_path}")
        
        
        


