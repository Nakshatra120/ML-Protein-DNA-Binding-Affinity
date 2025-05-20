# Combining Physics-Based Protein–DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor–DNA Binding

This repository contains data, code, and analysis scripts for integrating Molecular Mechanics-Generalized Born Surface Area (MMGBSA) calculations with machine learning (ML) models to predict protein–DNA binding affinities. We specifically focus on the Myc/Max transcription factor (TF) system, emphasizing interpretability through physically meaningful energy features derived from all-atom molecular dynamics (MD) simulations.

## Project Structure

```
MMGBSA_ML_Project/
├── gcPBM/
├── MD_simulations/
├── DNA_library/
├── ML_models/
│   ├── ML_SVM/
│   ├── ML_RF/
│   ├── ML_NN/
│   └── ML_REG/
└── scripts/
    ├── process_gcPBM.ipynb
    ├── Mutate_PDB_w3dna.ipynb
    ├── AMBER_MMGBSA.ipynb
    └── process_results.ipynb
```

## Directory Descriptions

### gcPBM

Experimental datasets, including genomic-context protein-binding microarray (gcPBM) data and universal PBM (uPBM) 8-mer binding affinity scores.

### MD\_simulations

All-atom MD simulation input files, trajectories, and submission scripts. See [`MD_simulations/readme.md`](MD_simulations/readme.md).

### DNA\_library

Protein–DNA complex structures (`MycMax_[seq].pdb`) generated via [`Mutate_PDB_w3dna.ipynb`](scripts/Mutate_PDB_w3dna.ipynb). Reference structure: `MycMax_PDB.pdb` (PDB ID: 1NKP).

### ML\_models

Stores trained ML models by type:

* `ML_SVM`: Support Vector Machines
* `ML_RF`: Random Forests
* `ML_NN`: Neural Networks
* `ML_REG`: Linear Regression

See [`ML_models/readme.md`](ML_models/readme.md).

### scripts

Data processing and analysis scripts:

* **[`process_gcPBM.ipynb`](scripts/process_gcPBM.ipynb)**: Prepares balanced datasets and labels for ML modeling.
* **[`Mutate_PDB_w3dna.ipynb`](scripts/Mutate_PDB_w3dna.ipynb)**: Generates protein–DNA complex structures.
* **[`AMBER_MMGBSA.ipynb`](scripts/AMBER_MMGBSA.ipynb)**: Computes MMGBSA energy features from MD simulations. Outputs include:

  * `energy_corrections_df.csv`, `entropy_df.csv`, `energy_MMGBSA_df.csv`, `rawdat.csv`
* **[`process_results.ipynb`](scripts/process_results.ipynb)**: Evaluates ML results and generates analytical visualizations.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/calmasri7/ML-Protein-DNA-Binding-Affinity.git
cd ML-Protein-DNA-Binding-Affinity
```

### 2. Setup environment

**Prerequisites**

* Python 3.9
* Conda (recommended) or pip

**Conda installation**

```bash
conda env create -f environment.yml
conda activate mmgbsa_ml
```

**Alternative pip installation**

```bash
pip install numpy==1.20.0 pandas==1.3.5 scipy==1.7.3 scikit-learn==1.1.3 \
matplotlib==3.7.3 seaborn==0.13.2 plotly==5.17.0 shap==0.44.1 \
joblib==1.4.2 selenium==4.27.1 webdriver-manager==4.0.2 \
MDAnalysis==2.4.3 hyperopt==0.2.7 pyspark==3.5.4
```

### 3. Run analyses and ML models

Execute notebooks in the `scripts/` directory in this order:

1. `process_gcPBM.ipynb`
2. `Mutate_PDB_w3dna.ipynb`
3. MD simulations + MMGBSA (`MD_simulations/1_run_MD.sh`,`MD_simulations/2_strip_water.sh`,`MD_simulations/3_run_mmgbsa.sh`,`MD_simulations/4_run_naccess.sh`)
4. `AMBER_MMGBSA.ipynb`
5. ML model hyperparameter optimization + training/testing (`ML_models/ML_[model]/run_hyperopt.sh`,`ML_models/ML_[model]/run_ML.sh`)
6. `process_results.ipynb`

For further details, refer to individual directory READMEs.

## Software Versions

| Package           | Version |
| ----------------- | ------- |
| numpy             | 1.20.0  |
| pandas            | 1.3.5   |
| scipy             | 1.7.3   |
| scikit-learn      | 1.1.3   |
| matplotlib        | 3.7.3   |
| seaborn           | 0.13.2  |
| plotly            | 5.17.0  |
| shap              | 0.44.1  |
| joblib            | 1.4.2   |
| selenium          | 4.27.1  |
| webdriver-manager | 4.0.2   |
| MDAnalysis        | 2.4.3   |
| hyperopt          | 0.2.7   |
| pyspark           | 3.5.4   |
| AmberTools        | 21.12   |
| GROMACS           | 2022.1  |
| jupyterlab        | 4.2.1   |

## environment.yml

```yaml
name: mmgbsa_ml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9          
  - numpy>=1.24,<1.27  
  - pandas=1.3.5
  - scipy=1.10         
  - scikit-learn=1.3   
  - shap=0.47.2        
  - matplotlib=3.7.3
  - seaborn=0.13.2
  - plotly=5.17.0
  - joblib=1.4.2
  - selenium=4.27.1
  - webdriver-manager=4.0.2
  - mdanalysis=2.4.3
  - hyperopt=0.2.7
  - pyspark=3.5.4
  - ambertools=21.12
  - gromacs=2022.1
  - jupyterlab=4.2
```

NACCESS also needs to be installed, available on http://www.bioinf.manchester.ac.uk/naccess/

## Contact

For questions, contact Carmen Al Masri ([calmasri@uci.edu](mailto:calmasri@uci.edu)).

## Citation

If you use this repository, please cite:

```
Al Masri C, Yu J. Combining Physics-Based Protein–DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor-DNA Binding. ChemRxiv. 2025; [doi:10.26434/chemrxiv-2025-mc5q4](https://chemrxiv.org/engage/chemrxiv/article-details/6816805350018ac7c54c4d62). This content is a preprint and has not been peer-reviewed.
```
