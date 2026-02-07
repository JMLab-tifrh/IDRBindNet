This repository provides a graph-transformer–based pipeline to predict binding affinities for IDR–protein complexes from PDB structures. The workflow includes feature extraction, graph construction, and inference using a pretrained model.

# Environment Setup
 
```
conda env create -f kd_predict.yml -y
conda activate kd_predict.yml
```

# Running Kd Prediction
[ path/to/complex.pdb : This should be the location of the pdb file of the complex. No need to include the name of the pdb file]

```
python3 run_all.py --pdb_dir path/to/complex.pdb
```
