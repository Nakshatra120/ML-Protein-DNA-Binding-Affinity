# Physics-Informed ML for Protein–DNA Binding (Myc/Max) — Extended with Time-Dependent Models & Tutorials

This repository extends the original **Combining Physics-Based Protein–DNA Energetics with Machine Learning** project by integrating the same MMGBSA-derived, physically interpretable features with additional ML pipelines, including **time-dependent / sequence-context models**, and by providing **beginner-friendly tutorial notebooks** for reproducibility.

> **Provenance & Attribution**
>
> This codebase began as a fork of Carmen Al Masri’s repository:
>
> * Original repo: [https://github.com/calmasri7/ML-Protein-DNA-Binding-Affinity](https://github.com/calmasri7/ML-Protein-DNA-Binding-Affinity)
> * Preprint: *Al Masri C, Yu J. Combining Physics-Based Protein–DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor–DNA Binding. ChemRxiv (2025).*
>
> As my development diverged (new model families, data handling, and a tutorials track), I **unforked** to avoid confusion about scope and maintenance responsibility while preserving full attribution to the original work.

---

## What’s new in this repository

* **Time-dependent / context-aware models:** additional pipelines beyond the original SVM/RF/NN/REG baselines.
* **`tutorial notebooks/`**: step-by-step Jupyter notebooks to load data, merge computed and experimental features, run EDA, and train baseline models without digging into internal scripts.
* **Reorganized ML directories** and helper scripts to simplify k-fold splits, feature prep, and reproducible training.

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

### `tutorial notebooks/` (start here if you’re new)

* `walkthrough_notebook.ipynb` – load/merge datasets, quick EDA, baseline model
* `plot.ipynb` – visualizations & correlations
* `rawdat.csv`, `exp_data_all.csv`, `merged.csv` – minimal data artifacts for tutorials

> The tutorials mirror the full pipeline at a smaller scale so new users can reproduce key results quickly.

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

## Citation

If your work builds on this repository, please **cite both** the original preprint and this extension:

**Original work:**

```
Al Masri C, Yu J. Combining Physics-Based Protein–DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor-DNA Binding. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-mc5q4.
```

**This extension (time-dependent models & tutorials):**

```
Bansal N, Yu J (Group). Physics-Informed ML for Protein–DNA Binding: Time-Dependent Models and Reproducible Tutorials. GitHub repository, 2025. https://github.com/Nakshatra120/ML-Protein-DNA-Binding-Affinity
```

*(Replace with DOI/Zenodo badge if you mint one.)*

---

## License & Attribution

This repository preserves attribution to the original authors and scientific work. If the original project’s license applies to derivative works, this repository adheres to it and includes clear credit above. Please review the `LICENSE` file; if you use or redistribute substantial portions of either codebase, **credit both projects**.

---

## Contact

* Questions about this repository and the extensions: **Nakshatra Bansal** — [nabansal@ucsd.edu](mailto:nabansal@ucsd.edu)
* Original project contact: **Carmen Al Masri** — [calmasri@uci.edu](mailto:calmasri@uci.edu)
