import os
import subprocess
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Run Kd prediction pipeline"
)

parser.add_argument(
    "--pdb_dir",
    required=True,
    help="Path to directory containing PDB files"
)

parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU ID to use (default=0)"
)

args = parser.parse_args()

PDB_DIR = os.path.abspath(args.pdb_dir)
GPU_ID = args.gpu_id

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# -------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------
for pdb in sorted(os.listdir(PDB_DIR)):
    if pdb.endswith(".pdb"):
        pdb_path = os.path.join(PDB_DIR, pdb)

        print(f"\n===== Running for {pdb} =====")

        for script in SCRIPTS:
            script_path = os.path.join(SCRIPT_DIR, script)

            cmd = [
                "python3",
                script_path,
                "--pdb_file",
                pdb_path,
            ]

            # Only pass gpu_id to infer_kd
            if script == "infer_kd.py":
                cmd.extend(["--gpu_id", str(GPU_ID)])

            print("Running:", " ".join(cmd))

            subprocess.run(
                cmd,
                check=True,
                cwd=PDB_DIR
            )

# -------------------------------------------------
# CONCATENATE RESULTS
# -------------------------------------------------
print("\n===== Concatenating KD CSV Files =====")

all_rows = []

for pdb in sorted(os.listdir(PDB_DIR)):
    if pdb.endswith(".pdb"):
        pdb_name = pdb.replace(".pdb", "")
        csv_path = os.path.join(PDB_DIR, pdb_name, f"{pdb_name}_kd.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_rows.append(df)

if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    output_file = os.path.join(PDB_DIR, "All_Kd.csv")
    final_df.to_csv(output_file, index=False)
    print(f"\nALL KD results saved to: {output_file}")
else:
    print("\nNo KD CSV files found.")
