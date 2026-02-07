import os
import re
import torch
import pickle
import argparse
import mdtraj as md
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from transformers import T5Tokenizer, T5EncoderModel

# ----------------------------
# --- Helper: ProtT5 embed ---
# ----------------------------

def embedding_chunk(model, input_ids, attention_mask):
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][: seq_len - 1]
        features.append(seq_emd)
    return features[0]

def get_embedding(fasta, model, tokenizer, ncpus=16, gpu_id=0):
    fasta = re.sub(r"[UZOJB]", "X", fasta)  
    sequence = [' '.join(fasta)]

    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)

    torch.set_num_threads(ncpus)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = model.to(device)

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    features = embedding_chunk(model, input_ids, attention_mask)

    return features


# ----------------------------------------
# --- Extract FASTA sequence from chain ---
# ----------------------------------------

def get_chain_sequence(traj, chain_index):
    topology = traj.topology
    chain = topology.chain(chain_index)

    aa_map = {res.index: res.code for res in chain.residues if res.is_protein}

    seq = "".join([aa_map[i] for i in aa_map])
    return seq


# ---------------------------
# --- Main embedding util ---
# ---------------------------

def embed_pdb(pdb_path, gpu_id=0):

    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    out_dir = pdb_name
    os.makedirs(out_dir, exist_ok=True)

    # Load PDB using MDTraj
    traj = md.load_pdb(pdb_path)

    # Extract sequences
    idr_seq = get_chain_sequence(traj, chain_index=0)     # chain A
    prot_seq = get_chain_sequence(traj, chain_index=1)    # chain B

    print(f"Chain A (IDR) length = {len(idr_seq)}")
    print(f"Chain B (Protein) length = {len(prot_seq)}")

    # Load ProtT5 model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")

    print("Extracting embedding using concatenated sequence...")
    # Concatenate sequences
    concat_seq = idr_seq + prot_seq
    idr_len = len(idr_seq)
    prot_len = len(prot_seq)

    # Get embedding for concatenated sequence
    concat_emb = get_embedding(concat_seq, model, tokenizer, gpu_id=gpu_id)
    
    # Split embeddings
    idr_emb = concat_emb[:idr_len]
    prot_emb = concat_emb[idr_len:idr_len + prot_len]
    
    print("Extracting embedding of IDR...")
    pickle.dump(idr_emb,  open(f"{out_dir}/{pdb_name}_idr_embedding.npy", "wb"))
    
    print("Extracting embedding of Protein...")
    pickle.dump(prot_emb, open(f"{out_dir}/{pdb_name}_protein_embedding.npy", "wb"))

    print(f"Saved embeddings in: {out_dir}/")


# ---------------------------
# --- Run on a single pdb ---
# ---------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    args = parser.parse_args()

    pdb_file = args.pdb_file  
    
    gpu_id=1
    embed_pdb(pdb_file, gpu_id=gpu_id)

