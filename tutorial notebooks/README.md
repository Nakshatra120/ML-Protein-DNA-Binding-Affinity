# Tutorial Notebooks

This folder provides step-by-step tutorials that demonstrate how to:
1. Access and explore the datasets (`rawdat.csv`, `exp_data_all.csv`, etc.).
2. Merge computed and experimental data.
3. Perform exploratory plots and statistics.
4. Train and evaluate machine learning models (e.g., Random Forest).
5. Reproduce the main results from the project without digging into the full codebase.

---

## Files

- **exp_data_all.csv**  
  Experimental protein–DNA binding affinity measurements.

- **rawdat.csv**  
  Computed/processed features from simulations and energy terms.

- **merged.csv**  
  Data after joining experimental and computed features.

- **plot.ipynb**  
  Demonstrates data visualization and correlation plots.

- **walkthrough_notebook.ipynb**  
  Main step-by-step tutorial notebook that integrates the workflow from data loading to model training.

---

## How to Use

1. Start with [`walkthrough_notebook.ipynb`](./walkthrough_notebook.ipynb).  
   This notebook introduces the dataset, explains preprocessing, and guides you through a Random Forest regression example.

2. Explore [`plot.ipynb`](./plot.ipynb) for visualizations of data distributions and correlations.

3. If you’d like to use the raw CSVs directly:
   - `exp_data_all.csv` → experimental binding data
   - `rawdat.csv` → computed features
   - `merged.csv` → merged dataset for ML models

---

## Environment

All notebooks can be run with the dependencies in the root `environment.yml`.  
Activate the conda environment and launch Jupyter:

```bash
conda env create -f ../environment.yml
conda activate protein-dna-ml
jupyter notebook
