import os
import os
import argparse
import numpy as np
import mdtraj as md

   
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
    
    # Chain A: IDR
    idr_chain = traj.atom_slice(traj.topology.select("chainid 0"))
    
    # Chain B: Protein  
    pro_chain = traj.atom_slice(traj.topology.select("chainid 1"))
    
    # Save separated chains
    idr_output = os.path.join(out_dir, f"{pdb_name}_idr.pdb")
    idr_chain.save(idr_output)
    
    pro_output = os.path.join(out_dir, f"{pdb_name}_pro.pdb")
    pro_chain.save(pro_output)
    
    traj1 = md.load_pdb(idr_output)
    traj2 = md.load_pdb(pro_output)
    
    # Create new topology with continuous residue numbering
    new_topology = md.Topology()
    new_chain = new_topology.add_chain()
    
    # Keep track of residue count for continuous numbering
    residue_count = 1
    
    # Add residues and atoms from first structure
    for residue in traj1.topology.residues:
        new_res = new_topology.add_residue(residue.name, new_chain)
        residue_count += 1
        
        for atom in residue.atoms:
            new_topology.add_atom(atom.name, atom.element, new_res)
    
    # Add residues and atoms from second structure  
    for residue in traj2.topology.residues:
        new_res = new_topology.add_residue(residue.name, new_chain)
        residue_count += 1
        
        for atom in residue.atoms:
            new_topology.add_atom(atom.name, atom.element, new_res)
    
    # Combine coordinates - since these are single-frame structures
    new_xyz = np.concatenate([traj1.xyz, traj2.xyz], axis=1)
    
    # Create new trajectory with the modified topology
    new_traj = md.Trajectory(new_xyz, new_topology)
    
    # Save the merged structure
    merged_output = os.path.join(out_dir, f"{pdb_name}_idr_pro_merged.pdb")
    new_traj.save(merged_output)
    
    
    
    
    
    
