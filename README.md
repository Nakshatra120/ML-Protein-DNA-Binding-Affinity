# Tutorials for Physics-Based ML of Protein–DNA Binding (Myc/Max)

This repository provides **tutorials and documentation** for the project  
**Combining Physics-Based Protein–DNA Energetics with Machine Learning**, which integrates MMGBSA-derived, physically interpretable features with machine-learning models to predict protein–DNA binding affinities in the Myc/Max system.

The purpose of this repository is to make the original pipeline more accessible and reproducible for new users through guided notebooks and organized examples.

---

## Credits

The original dataset preparation, pipeline scripts, and overall project design were developed by **Carmen Masri and Jin Yu**,  
*Journal of Chemical Information and Modeling* **2025**, 65 (21), 11804–11817.  
https://doi.org/10.1021/acs.jcim.5c01143

This work originates from the **Jin Yu Lab**, University of California, Irvine:  
https://sites.uci.edu/jinyulab/

This tutorial section was written, structured, and designed by **Nakshatra Bansal** to provide clearer documentation and to guide new users through the modeling process, in collaboration with the Jin Yu Lab.

---

## What this tutorial repository provides

* **Beginner-friendly tutorial notebooks** that walk through data loading, feature merging, exploratory analysis, and baseline model training.
* A **lightweight, reproducible entry point** that mirrors the full pipeline at a smaller scale.
* Organized examples that help users understand how computed MMGBSA features connect to experimental binding data.

This repository is intended for **learning, onboarding, and reproducibility**, rather than for introducing new scientific methodology.

---

## Project Structure (high-level)

```
ML-Protein-DNA-Binding-Affinity/
├── gcPBM/
├── MD_simulations/
├── DNA_library/
├── ML_models/
│   ├── ML_SVM/
│   ├── ML_RF/
│   ├── ML_NN/
│   └── ML_REG/
├── scripts/
└── tutorial notebooks/            <-- New: guided, beginner-friendly entry point
```


---

## Tutorials (recommended starting point)

The **`tutorial notebooks/`** directory contains step-by-step Jupyter notebooks designed to run quickly and demonstrate the core ideas of the pipeline.

Key files include:
* `walkthrough_notebook.ipynb` – load and merge datasets, perform basic EDA, train a baseline model
* `plot.ipynb` – visualization and correlation analysis
* `rawdat.csv`, `exp_data_all.csv`, `merged.csv` – minimal datasets for tutorial use

These tutorials reproduce key components of the full workflow without requiring HPC resources.

---

## Full pipeline (advanced users)

Users who wish to run the complete pipeline may follow the original workflow, depending on their compute environment:

1. `scripts/process_gcPBM.ipynb`
2. `scripts/Mutate_PDB_w3dna.ipynb`
3. Molecular dynamics and MMGBSA calculations:
   * `MD_simulations/1_run_MD.sh`
   * `MD_simulations/2_strip_water.sh`
   * `MD_simulations/3_run_mmgbsa.sh`
   * `MD_simulations/4_run_naccess.sh` (requires NACCESS)
4. `scripts/AMBER_MMGBSA.ipynb` → generates feature tables (e.g., `rawdat.csv`)
5. Model training and evaluation:
   * `ML_models/ML_[model]/run_hyperopt.sh`
   * `ML_models/ML_[model]/run_ML.sh`
6. `scripts/process_results.ipynb`

---

## Additional documentation

Additional step-by-step tutorial documents (Word/PDF) expanding on parts of the original pipeline are available here:

**Protein–DNA Binding ML Tutorials (Google Drive):**  
https://drive.google.com/drive/folders/1-VZr-uEI5vbdHhD7oE_DEY3p4iczt1As

These documents cover:
- gcPBM processing
- Linear Regression, Random Forest, SVM, and Neural Network workflows
- Sampling and stratification strategies
- Model run scripts and preprocessing logic

---

## Citation

If you use this tutorial repository, please cite the original work:

```
Al Masri C, Yu J. Combining Physics-Based Protein–DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor-DNA Binding. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-mc5q4.
```


---

## Getting Started

### 1) Clone

```bash
git clone https://github.com/Nakshatra120/ML-Protein-DNA-Binding-Affinity.git
cd ML-Protein-DNA-Binding-Affinity
```

### 2) Environment

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate mmgbsa_ml
```

Or via pip (see versions in the table below):

```bash
pip install -r requirements.txt   # if you maintain one; otherwise see versions table below
```

### 3) Quickstart (tutorial path)

Open **`tutorial notebooks/walkthrough_notebook.ipynb`** and run cells top-to-bottom to:

* Load `rawdat.csv` (computed features) + `exp_data_all.csv` (experimental ΔΔG)
* Merge, clean, and explore features
* Train/evaluate a baseline model (e.g., Random Forest)

### 4) Full pipeline (advanced)

Follow the original script order (as applicable to your compute environment):

1. `scripts/process_gcPBM.ipynb`
2. `scripts/Mutate_PDB_w3dna.ipynb`
3. MD + MMGBSA:

   * `MD_simulations/1_run_MD.sh`
   * `MD_simulations/2_strip_water.sh`
   * `MD_simulations/3_run_mmgbsa.sh`
   * `MD_simulations/4_run_naccess.sh` (requires NACCESS)
4. `scripts/AMBER_MMGBSA.ipynb` → generates `rawdat.csv` and energy tables
5. `ML_models/ML_[model]/run_hyperopt.sh` and `run_ML.sh`
6. `scripts/process_results.ipynb`

---

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

> **NACCESS** must be installed separately: [http://www.bioinf.manchester.ac.uk/naccess/](http://www.bioinf.manchester.ac.uk/naccess/)

---

## Additional Documentation & Tutorials

Alongside the notebooks in this repository, we maintain a collection of **step-by-step tutorial documents** (Word/PDF) that expand on various parts of the original Carmen Al Masri pipeline.  

**Full documentation folder on Google Drive:**  
[Protein–DNA Binding ML Tutorials (Google Drive)](https://drive.google.com/drive/folders/1-VZr-uEI5vbdHhD7oE_DEY3p4iczt1As?usp=sharing)

These documents include:
- gcPBM processing (end-to-end)
- Linear Regression, Random Forest, SVM, and Neural Network pipelines
- Stratification and sampling tutorials
- Model run scripts (RF, NN, etc.)
- Expanded notes on logic and preprocessing steps

---

## Contact

* Questions about this repository and the extensions: **Nakshatra Bansal** — [nabansal@ucsd.edu](mailto:nabansal@ucsd.edu)
* Original project contact: **Carmen Al Masri** — [calmasri@uci.edu](mailto:calmasri@uci.edu)
