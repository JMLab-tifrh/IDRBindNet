# IDRBindNet

This repository provides a graph-transformer–based pipeline to predict binding affinities for IDR–Protein complexes from PDB structures. The workflow includes feature extraction, graph construction, and inference using a pretrained model.

# Environment Setup 

```
conda env create -f kd_predict.yml -y
conda activate kd_predict
```

# SPARTA+ setup
#### For Kd prediction, the Chemical shift needs to be calculated using SPARTA+.

```
1. Update the sparta+ sparta+Init.com file using the file provided in the repository. Change username as per your name
2. Run install.com using csh shell
3. export PATH="/home/username/Softwares/sparta+/SPARTA+/:$PATH"  <----- put this in bashrc
4. Ensure that sparta+ runs from terminal
```


# Running Kd Prediction 
### The model has been trained for IDR-Protein complex where the first chain is the IDR and second chain is the Protein
### Go to directory where the pdb files are present
### Run the command below. Change the path as per where "run_all.py" is present

```
python3 /home/username/IDRBindNet-main/GT-IDR-Bind/run_all.py --pdb_dir . --gpu_id 0
```


# Output
The pipeline creates output directory where the pdb file is located. If the pdb file is complex_1.pdb, then a directory "complex_1" will be created which will contain all the results.

Inside this directory, the final result is saved as : complex_1_kd.csv

The complex_1_kd.csv contains ID, Chain_A, Chain_B, Mean_Kd_uM, Std_Kd_uM

The Kd value from each model will also be saved as a csv file as complex_1_split_0_kd.csv, complex_1_split_1_kd.csv and so on for the five models.
```
ID – PDB identifier

Chain_A – Amino acid sequence of IDR

Chain_B – Amino acid sequence of Protein

Mean_Kd_uM – Mean predicted Kd

Std_Kd_uM – Standard deviation across model splits
```


## Kd values are reported in micromolar (µM).

### If you have single or multiple pdb(s), then a All_Kd.csv file will be created, which will be the concatenated csv files for all the complexes.
