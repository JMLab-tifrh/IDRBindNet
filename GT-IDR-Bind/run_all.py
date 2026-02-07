import os
import subprocess
import pandas as pd
import argparse

# -----------------------------
# ARGUMENT PARSING
# -----------------------------
parser = argparse.ArgumentParser(description="Run Kd prediction pipeline on all PDB files in a directory")
parser.add_argument(
    "--pdb_dir",
    required=True,
    help="Path to directory containing PDB files"
)
args = parser.parse_args()

PDB_DIR = os.path.abspath(args.pdb_dir)

SCRIPTS = [
    "get_embedding.py",
    "get_distance.py",
    "get_rotation.py",
    "split_combine.py",
    "get_chemical_shift.py",
    "get_sasa.py",
    "make_graph.py",
    "infer_kd.py",
]

# -----------------------------
# 1. RUN ALL SCRIPTS ON EACH PDB
# -----------------------------
for pdb in sorted(os.listdir(PDB_DIR)):
    if pdb.endswith(".pdb"):
        pdb_path = os.path.join(PDB_DIR, pdb)

        print(f"\n===== Running for {pdb} =====")
        
        for script in SCRIPTS:
            print(f"Running {script} on {pdb}...")
            subprocess.run(
                ["python", script, "--pdb_file", pdb_path],
                check=True
            )

# -----------------------------
# 2. CONCATENATE ALL *_kd.csv FILES
# -----------------------------
print("\n===== Concatenating KD CSV Files =====")

all_rows = []

for pdb in sorted(os.listdir(PDB_DIR)):
    if pdb.endswith(".pdb"):
        pdb_name = pdb.replace(".pdb", "")
        csv_path = os.path.join(PDB_DIR, pdb_name, f"{pdb_name}_kd.csv")

        if os.path.exists(csv_path):
            print(f"Reading: {csv_path}")
            df = pd.read_csv(csv_path)
            all_rows.append(df)
        else:
            print(f"Missing KD CSV for {pdb_name}")

if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    output_file = os.path.join(PDB_DIR, "All_Kd.csv")
    final_df.to_csv(output_file, index=False)
    print(f"\nALL KD results saved to: {output_file}")
else:
    print("\n❌ No KD CSV files found to concatenate")

