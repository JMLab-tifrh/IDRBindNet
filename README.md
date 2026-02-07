# GT-IDR-Bind

This repository provides a graph-transformer–based pipeline to predict binding affinities for IDR–Protein complexes from PDB structures. The workflow includes feature extraction, graph construction, and inference using a pretrained model.

# Environment Setup 

```
conda env create -f kd_predict.yml -y
conda activate kd_predict.yml
```

# SPARTA+ setup
#### For Kd prediction, the Chemical shift needs to be calculated using SPARTA+.

```
1. Update the sparta+ sparta+Init.com file using the file provided in the repository. Change username as per your name
2. Run install.com using csh shell
3. export PATH="/home/username/Softwares/sparta+/SPARTA+/:$PATH"  <----- put this in bashrc
```


# Running Kd Prediction 
### The model has been trained for IDR-Protein complex where the first chain is the IDR and second chain is the Protein

```
python3 run_all.py --pdb_dir path/to/complex.pdb
```

#### path/to/complex.pdb : This should be the location of the pdb file of the complex. No need to include the name of the pdb file




# Output
The pipeline creates an output directory where the pdb file (complex.pdb) is located : complex/

Inside this directory, the final result is saved as : complex_kd.csv

The complex_kd.csv contains ID, Chain_A, Chain_B, Mean_Kd_uM, Std_Kd_uM
```
ID – PDB identifier

Chain_A – Amino acid sequence of IDR

Chain_B – Amino acid sequence of Protein

Mean_Kd_uM – Mean predicted dissociation constant

Std_Kd_uM – Standard deviation across model splits
```


## Kd values are reported in micromolar (µM).

### If you have single or multiple pdb(s), then a All_Kd.csv file will be created, which will be the concatenated csv files for all the complexes for which the Kd was computed in /path/to/complex.pdb.
