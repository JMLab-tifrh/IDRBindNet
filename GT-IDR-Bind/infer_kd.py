import os
import torch
import argparse
import subprocess
import numpy as np
import mdtraj as md
import pickle as pkl
from tqdm import tqdm
from GTR import GTR
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file
    gpu_id = 6

    # Derive pdb_name and create output directory
    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)

    # Load structure
    traj = md.load_pdb(pdb_file)

    chain_lengths = [len(list(chain.residues)) for chain in traj.topology.chains]
    total_chain_length = chain_lengths[0] + chain_lengths[1]

    # Extract sequences
    chain0 = traj.topology.chain(0)
    chain_A_seq = ''.join([res.code for res in chain0.residues])

    chain1 = traj.topology.chain(1)
    chain_B_seq = ''.join([res.code for res in chain1.residues])

    # Load graph
    graph_path = os.path.join(out_dir, f"{pdb_name}_graph.pkl")
    graph = pkl.load(open(graph_path, "rb"))

    input_dim = graph.x.shape[1]
    edge_dimension = int(len(graph.edge_attr[0]))

    # Device setup
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graph = graph.to(device)  # Move once

    path_to_model='../Prost_T5_BFD/'

    Kd_uM = []

    # Loop over 5 splits
    for split in tqdm(range(5)):
        print(f"Running model {split + 1}")

        model_path = (
            path_to_model
            + f"kd_regression_model_prot_t5_bfd_split_{split}_model.pth"
        )

        model = GTR(input_dim, edge_dimension).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Forward pass
        with torch.no_grad():
            y_pred = model(graph).cpu().numpy()[0]

            # Kd in molar
            Kd_M = 10 ** (-y_pred * total_chain_length)

            # Convert to uM
            kd_um = Kd_M * 1e6
            Kd_uM.append(kd_um)

        # Save individual model result
        df_split = pd.DataFrame(
            [[pdb_name, chain_A_seq, chain_B_seq, kd_um]],
            columns=["ID", "Chain_A", "Chain_B", "Kd_uM"],
        )

        df_output = os.path.join(out_dir, f"{pdb_name}_split_{split}_kd.csv")
        df_split.to_csv(df_output, index=False)

        print(f"Results saved to {df_output}")

    # Save summary statistics
    mean_kd = np.mean(Kd_uM)
    std_kd = np.std(Kd_uM)

    df_summary = pd.DataFrame(
        [[pdb_name, chain_A_seq, chain_B_seq, mean_kd, std_kd]],
        columns=["ID", "Chain_A", "Chain_B", "Mean_Kd_uM", "Std_Kd_uM"],
    )

    summary_output = os.path.join(out_dir, f"{pdb_name}_kd.csv")
    df_summary.to_csv(summary_output, index=False)

    print(f"Summary saved to {summary_output}")
