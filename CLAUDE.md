# CLAUDE.md — ML-Protein-DNA-Binding-Affinity

Persistent project notes for Claude Code. This is a **tutorial / reproduction** repo that wraps and documents the Carmen Al Masri / Jin Yu pipeline (J. Chem. Inf. Model. 2025, doi:10.1021/acs.jcim.5c01143): combine physics-based MMGBSA energies from MD simulations of Myc/Max–DNA complexes with ML models that predict gcPBM-derived binding affinity.

> The directory `tutorial notebooks/` (note the **space** in the name, not an underscore) is **deliberately not analyzed here** — it contains known mistakes that the user will fix one notebook at a time.

---

## 1. Repo layout

```
ML-Protein-DNA-Binding-Affinity/
├── README.md, environment.yml, .gitignore
├── DNA_library/             # Reference + example mutant PDBs of Myc/Max–DNA (from 1NKP)
├── gcPBM/                   # Raw genomic-context PBM data (probe seqs, intensities, 8-mer E-scores)
├── MD_simulations/          # SLURM/GROMACS/AmberTools pipeline (run on HPC); sample replica included
├── scripts/                 # All Python/notebook glue between gcPBM, MD output, and ML inputs
├── ML_models/               # ML_RF, ML_NN, ML_SVM, ML_REG (linear) + shared README
└── tutorial notebooks/      # IGNORED — user will repair these one-by-one
```

### Pipeline at 30 000 ft

1. `scripts/process_gcPBM.ipynb` → from gcPBM raw → `dataset.csv` (sequences to simulate) + `exp_data_all.csv` (ML labels).
2. `scripts/Mutate_PDB_w3dna.ipynb` → for each sequence in `dataset.csv`, drives the w3DNA web mutation tool (Selenium, headless Chrome, local) and writes `DNA_library/MycMax_PDB_<seq>.pdb`.
3. `MD_simulations/` → HPC: 20 replicates × ~14 ns/replicate per sequence (equil → NVT → restrained NPT0 → production NPT), water stripped, then MMGBSA + NACCESS.
4. `scripts/AMBER_MMGBSA.ipynb` → reads every `MycMax_<seq>/<run>/FINAL_RESULTS_strain.csv` + naccess + trajectories → `rawdat.csv` (per-frame ML feature table).
5. `ML_models/ML_<RF|NN|SVM|REG>/` → hyperopt → repeated 5×5 K-fold training → predictions + metrics.
6. `scripts/process_results.ipynb` → aggregates fold-level predictions, makes plots, RF feature importances, KW tests, CACGTG/flank analyses.

### Handoff from MD work to this repo

The MD outputs are **read directly from `../MD_simulations/<MycMax_seq>/<run>/`** by `scripts/AMBER_MMGBSA.ipynb`. Required files per replicate:
- `FINAL_RESULTS_strain.csv` (gmx_MMPBSA output → VDWAALS, EEL, EGB, ESURF)
- `npt_nowat.tpr`, `npt_whole_nowat.xtc` (water-stripped topology + trajectory, used by MDAnalysis)
- `naccess/prot*.rsa`, `dna*.rsa`, `complex*.rsa` (SASA per frame, used for the knowledge-based entropy term)

The notebook caches intermediates (`energy_corrections_df.csv`, `entropy_df.csv`, `energy_MMGBSA_df.csv`) so reruns only process new sequence/run pairs.

---

## 2. Data layer

### Top-level / gcPBM raw inputs

| File | Rows | Columns / schema | Origin | Consumed by |
|---|---|---|---|---|
| `gcPBM/GSM2746661_MycMax_8mers_11111111.txt` | ~32k | `8-mer, 8-mer.1, E-score, Median, Z-score` (TSV, `#` comments) | Universal PBM E-scores from GEO GSM2746661 | `process_gcPBM.ipynb` (cell 4) |
| `gcPBM/gcPBM_probe_sequence.txt` | ~36k | `ID, Name, SEQUENCE` (36-bp probes) | gcPBM array design | `process_gcPBM.ipynb` |
| `gcPBM/gcPBM_myc_normalized_intensity.txt` | ~36k | `ID_REF, VALUE` (already log-normalized intensity) | gcPBM Myc/Max experiment | `process_gcPBM.ipynb` |
| `DNA_library/MycMax_PDB.pdb` | — | 1NKP-derived processed PDB, waters removed, His protonation pre-assigned, DNA chains A/B (resids 1–36 / 72–37), protein chains E/F | Manual prep | `Mutate_PDB_w3dna.ipynb` (template) |
| `DNA_library/MycMax_<SEQ>.pdb` | — | Mutant PDB for a specific 36-bp DNA | Output of `Mutate_PDB_w3dna.ipynb` | `MD_simulations/1_run_MD.sh` |

### Scripts working files (`scripts/`)

| File | Shape | Schema | Produced by | Consumed by |
|---|---|---|---|---|
| `gcPBM_myc_final.csv` | 7160 | `Name,SEQUENCE,Mean Intensity,Probe,Log Intensity,12mer,10mer,8mer,6mer` | `process_gcPBM.ipynb` (cell 25) | `stratification.ipynb` |
| `dataset_old.csv` | 99 | same as above (no `new_label`) | older `process_gcPBM` run / `stratification.ipynb` (commented save) | reloaded in `process_gcPBM.ipynb` cell 32 to augment |
| `dataset.csv` | 168 | `…,new_label` ∈ {unbound, weak, strong} | `process_gcPBM.ipynb` (cell 32, balanced 78 unbound + 45 weak + 45 strong with threshold at 7.7) | `Mutate_PDB_w3dna.ipynb` (sequences to mutate) |
| `exp_data_all.csv` | 168 | `sequence, bind_avg, binding_type, improving` | `process_gcPBM.ipynb` (cell 34–35) | All ML models (label source); copied to `ML_models/ML_RF/Inputs/` |
| `rawdat.csv` | 272 160 | `sequence, run, VDWAALS, EEL, EGB, ESURF, HB Energy, Hydrophobic Energy, Pi-Pi Energy, Delta_Entropy` | `AMBER_MMGBSA.ipynb` (cell 35) | All ML models (feature source); manually copied to `ML_models/ML_RF/Inputs/` |
| `vdw_mapping.csv` | 150 | `Atom Name, VDW Radius` | Curated atomistic VDW radii for AMBER atom names (incl. DNA + amino acids) | `AMBER_MMGBSA.ipynb` (hydrophobic-energy term) |
| `pca_loadings.csv` | 8 × 8 | rows=features, cols=PC1..PC8 | exploratory PCA (likely from a not-checked-in or tutorial cell) | nothing in the canonical pipeline; tutorial reference |
| `pca_scaler_stats.csv` | 8 | `feature, mean, scale` (StandardScaler fit) | same exploratory PCA | tutorial reference; **note**: numbers differ from per-fold `gbsa_reg_train_stats.csv` produced in ML_RF |
| `pca_scores.csv` | ~272 160 | `PC1..PC8, sequence, run` | same exploratory PCA over `rawdat.csv` | not consumed by the canonical RF/NN/SVM/REG pipeline |
| `pca_variance.csv` | 8 | explained variance per PC; PC1+PC2 ≈ 68%, top 4 ≈ 93% | same | reference / plots |
| `gc_pbm_processing_pseudocode.md` | — | Implementation notes / blueprint for `process_gcPBM.ipynb` | documentation | reading |

### `exp_data_all.csv` label schema

`bind_avg` is `Log Intensity - 7.7` (centered around the binding threshold).
`binding_type` ∈ {0=weak/unbound, 1=medium, 2=strong}, using `bind_avg` ≤ 0 → 0, < (9-7.7)=1.3 → 1, else 2 (computed from un-centered Log Intensity ≤ 7.7 / < 9 / ≥ 9).
`improving` is binary 0/1: 0 if `bind_avg ≥ 0` (i.e. binder), 1 otherwise (note: `bind` is the **negative** of "improving" — see open Q3).

### ML\_models internal files (RF as the canonical example)

| File | Schema | Notes |
|---|---|---|
| `Inputs/rawdat.csv`, `Inputs/exp_data_all.csv` | as above | These are the user-staged inputs; **a copy** of the scripts/ files. |
| `Data/gbsa_reg_scr0p00_trn_final.csv` | `sequence, <8 energy terms>, bind_avg` (un-averaged, un-standardized) | Output of `torch_prep_kfold.py --initial_split` (post 80/20 sequence-level split + label scramble). |
| `Data/gbsa_reg_scr0p00_tst_preprocess.csv` | same | Test rows before averaging+standardization. |
| `Data/gbsa_reg_train_stats.csv` | `colname, mean, std` (8 rows, one per feature) | Means/stds computed on the averaged training set; used to standardize train and test. **This is the per-run persisted scaler** (not `scripts/pca_scaler_stats.csv`). |
| `Data/gbsa_reg_scr0p00_trn_{rep}_{fold}.csv`, `..._val_{rep}_{fold}.csv` | `sequence, <8 standardized features>, bind_avg` | Repeated K-fold splits used by `run_model.py`. |
| `Data/gbsa_reg_scr0p00_tst_final.csv` | same | Final standardized test set used at evaluation time. |
| `Model/rf_fold_{rep}_{fold}_reg_scr0p00.pkl` | joblib-pickled `RandomForestRegressor` | One per (rep × fold). |
| `predictions_reg_scr0p00_rep{r}_fold{f}.csv` | `Label, Predicted, True` (rows = validation samples; Label=sequence) | Per-fold validation predictions. |
| `predictions_reg_final_avg_scr0p00.csv` | `Label, AvgPredicted, AvgTrue` | Aggregated by sequence across all repeats × folds (mean for reg, majority for cls). |
| `predictions_test_reg_scr0p00.csv` | `Label, Predicted, True` | Concatenated per-fold predictions on the held-out test set. |
| `final_metrics_reg_trn_scr0p00.csv` / `..._tst_scr0p00.csv` | `MSE, R2, Pear, MCC, Accuracy` | Mean over all (rep × fold) entries. |
| `hyperopt_results_reg.csv`, `best_hyperparams_reg.json` | Hyperopt trials + best config | Best for RF reg currently: `keep_last_percent=90, navg=280, n_estimators=350, max_depth=10, max_features=sqrt, min_samples_split=50, min_samples_leaf=20`. |
| `Random_Forest_metrics.csv`, `summary_metrics.csv` | Aggregated metrics (used by `process_results.ipynb`) | — |

`ML_NN/` mirrors this layout but stores `nn_fold_{rep}_{fold}_reg_scr0p00.pth` checkpoints and `predictions_nn_reg_...csv` outputs. The NN folder currently has **25 trained checkpoints** (5 rep × 5 fold) but the `Data/` folder is **empty** — the on-disk fold CSVs were regenerated elsewhere or cleaned up.

`ML_SVM/` and `ML_REG/` contain only the scripts (`run_model.py`, `hyperopt.py`); no trained models or data are checked in.

---

## 3. Scripts inventory & DAG

### Execution order (full pipeline)

```
process_gcPBM.ipynb
  ├── inputs:  gcPBM/GSM2746661_MycMax_8mers_11111111.txt
  │            gcPBM/gcPBM_probe_sequence.txt
  │            gcPBM/gcPBM_myc_normalized_intensity.txt
  │            scripts/dataset_old.csv   (used to augment to 168 sequences)
  └── outputs: scripts/gcPBM_myc_final.csv
               scripts/dataset.csv
               scripts/exp_data_all.csv  (also shutil-copied → ML_models/ML_RF/Inputs/)

       │
       ▼
Mutate_PDB_w3dna.ipynb  (runs LOCALLY, not HPC; Selenium / web3DNA)
  ├── inputs:  DNA_library/MycMax_PDB.pdb  (1NKP-derived reference)
  │            scripts/dataset.csv
  └── outputs: DNA_library/MycMax_PDB_<SEQUENCE>.pdb  (one per row)

       │
       ▼
MD_simulations/1_run_MD.sh  (SLURM)
  ├── reads from ../DNA_library_new/*.pdb  (NB: path mismatch with this repo's "DNA_library/")
  ├── creates: MycMax_<SEQ>/<1..20>/ replicate directories
  └── chains: initial_setup → submit_nvt → submit_npt0 → submit_npt

MD_simulations/2_strip_water.sh   → npt_whole_nowat.xtc, topol_nowat.top, npt0_nowat.gro, npt_nowat.tpr
MD_simulations/3_run_mmgbsa.sh    → FINAL_RESULTS_strain.csv (via gmx_MMPBSA)
MD_simulations/4_run_naccess.sh   → naccess/{dna,protein,complex}*.rsa per frame

       │
       ▼
AMBER_MMGBSA.ipynb
  ├── inputs:  MD_simulations/<MycMax_SEQ>/<run>/FINAL_RESULTS_strain.csv
  │            MD_simulations/<MycMax_SEQ>/<run>/npt_nowat.tpr + npt_whole_nowat.xtc
  │            MD_simulations/<MycMax_SEQ>/<run>/naccess/*.rsa
  │            scripts/vdw_mapping.csv
  ├── caches:  energy_corrections_df.csv, entropy_df.csv, energy_MMGBSA_df.csv
  └── output:  scripts/rawdat.csv         (per-frame, 8 features + sequence + run)

       │
       ▼
ML_models/ML_<RF|NN|SVM|REG>/run_hyperopt.sh  → best_hyperparams_<type>.json
ML_models/ML_<RF|NN|SVM|REG>/run_ML.sh        → trained models + predictions + metrics

       │
       ▼
scripts/process_results.ipynb
  ├── inputs:  ML_models/ML_<X>/predictions_*.csv + Model/*.pkl/.pth
  │            scripts/rawdat.csv, scripts/exp_data_all.csv
  └── outputs: plots (figures shown inline), <ModelName>_metrics.csv

stratification.ipynb  — standalone walkthrough of the stratified-sampling logic used inside process_gcPBM.ipynb. Reads gcPBM_myc_final.csv, optionally writes dataset_old.csv.
```

### Per-script summary

- **`process_gcPBM.ipynb`** — load 8-mer E-scores + gcPBM intensities → flank filter (no flanking 8-mer with E-score ≥ 0.3) → central-motif dominance filter (center 8-mer must beat both adjacent 8-mers and any non-overlapping 8-mer must not exceed adjacent scores) → log-transform (`np.log` of `Mean Intensity` which is already log-normalized — see open Q1) → extract 6/8/10/12-mers around positions 14–22 → stratified sample 33 per class (unbound ≤ 8, weak (8,9), strong ≥ 9) avoiding motif duplication using a 6→8→10→12-mer rotation → augment to 168 with new threshold 7.7 → relabel and export `exp_data_all.csv` with `bind_avg = LogIntensity − 7.7`.
- **`Mutate_PDB_w3dna.ipynb`** — Selenium-driven local script. For each sequence, uploads reference PDB to `web.x3dna.org/index.php/mutation`, clicks base-substitution radios for residues 1–36 (with paired strand 72→37), downloads the mutant PDB. Run locally with Chrome (headless).
- **`AMBER_MMGBSA.ipynb`** — heavy MDAnalysis notebook. Reads gmx_MMPBSA output (`VDWAALS, EEL, EGB, ESURF` × `conversion=1.7` to convert kcal/mol → kT), computes four custom corrections per frame: hydrogen-bond energy (HBA from MDAnalysis), pi-pi stacking energy (centroid distances/angles for aromatic side-chains × DNA bases), hydrophobic energy (heavy-atom LJ-style sum with `vdw_mapping.csv` radii, 8 Å cutoff), and knowledge-based entropy (rotamer counts × relative SASA from naccess output, per-residue, `max_rots` lookup for 20 AAs). The four-term VSGB 2.0–style energy model is documented in MD cell 6.
- **`stratification.ipynb`** — pedagogical breakdown of the sampling routine in `process_gcPBM.ipynb` (different thresholds: 8.0 / 9.0; not used by the canonical pipeline).
- **`process_results.ipynb`** — load per-fold prediction CSVs, parse filenames for `(scr_frac, rep, fold, is_test)`, compute per-fold and aggregated metrics (PCC, MSE, R²; binary/multiclass ACC, F1, MCC), plot regression quadrants, confusion matrices, RF feature importances (`feature_importances_` averaged across folds), metrics-vs-scramble curves, CACGTG-flank effect boxplots, Kruskal–Wallis tests on energy terms vs. binding category.

---

## 4. ML pipeline (priority deep-dive)

### Target

Three task variants, switched via `--model_type`:

| Mode | Label col in `exp_data_all.csv` | Type | Notes |
|---|---|---|---|
| `reg` | `bind_avg` (= `Log Intensity − 7.7`, float, units: log of normalized intensity) | regression | This is the primary target. **It is NOT a ΔΔG** in physical units, despite `process_results.ipynb` labeling axes "Experimental ΔΔG (bind_avg)". |
| `bin` | `improving` (0/1) | binary classification | |
| `mclass` | `binding_type` (0/1/2 = weak/medium/strong) | 3-class | |

### Features (8 total)

Identical for every model and every task:
`VDWAALS, EEL, EGB, ESURF, HB Energy, Hydrophobic Energy, Pi-Pi Energy, Delta_Entropy`

- First four come from gmx_MMPBSA (`FINAL_RESULTS_strain.csv`), multiplied by `conversion=1.7` to go from kcal/mol to kT.
- HB / Hydrophobic / Pi-Pi computed per frame in `AMBER_MMGBSA.ipynb` (VSGB 2.0–style).
- `Delta_Entropy` is the knowledge-based rotamer×SASA entropy term (also per frame).

PCA-reduced features (`pca_scores.csv`, 8 PCs) exist in `scripts/` but are **not consumed** by the canonical RF/NN/SVM/REG pipelines.

### Models

| Dir | Library | Class | Default hparams in `run_model.py` | Loss / training |
|---|---|---|---|---|
| `ML_RF` | sklearn | `RandomForestRegressor` / `Classifier` | n_estimators=100, max_depth=None, max_features="auto" *(deprecated since sklearn 1.3 — see Q5)*, min_samples_split=2, min_samples_leaf=1 | Default sklearn fit. |
| `ML_NN` | PyTorch | MLP (`Net`): input → Linear(44) → ReLU → Dropout(0.1) → 2× Linear(44)+ReLU+Dropout(0.1) → Linear(out) | hidden_size=44, hidden_layers=3, lr=1e-4, wt_decay=1e-4, max_epochs=5000, dropout=0.1, Xavier init | Full-batch Adam, MSE/BCE/CE; optional early stopping with `--use_early_stopping --patience 50`. |
| `ML_SVM` | sklearn | `SVR` / `SVC` | kernel=rbf, C=1.0, gamma=0.1, epsilon=0.1 | — |
| `ML_REG` | PyTorch (same `Net` class as NN) | Linear regression iff `--hidden_layers 0`; otherwise an MLP duplicate of NN | hidden_layers=3 default — must pass `--hidden_layers 0` for a true linear baseline | — |

Output layer: `1` for `reg`/`bin`, `num_classes=3` for `mclass`.

### Split + preprocessing pipeline (`Data/torch_prep_kfold.py`)

Pure-Python sklearn-only data prep, used by all four model dirs. Three modes:

1. **`--initial_split`** — load `exp_data_all.csv` (reference: sequence + label) and `rawdat.csv` (features), inner-merge on `sequence`, shuffle, then:
   - `reg`: shuffle unique sequences, take `(1 − test_percentage)` for train, rest for test (sequence-level grouping → **no sequence appears in both splits**). `test_percentage` is `0.2` (run scripts) although default in script is `0.15`.
   - `bin`/`mclass`: `StratifiedGroupKFold(n_splits=7)`, take first split → ~1/7 ≈ 14% test.
   - **Optional label scrambling** on training set only: `scramble_sequences()` picks `frac × n_seq` sequences and overwrites each with the label of a different random sequence. Used for leakage / scramble-fraction sensitivity tests (`scr0p00, scr0p25, scr1p00`).
   - Outputs: `gbsa_<type>_scr<frac>_trn_final.csv`, `gbsa_<type>_scr<frac>_tst_preprocess.csv`.

2. **`--process train`** — load `trn_final.csv`:
   - `keep_last_percent`: keep only the last X% of frames per sequence (sorted by `run`); intended to drop early/unequilibrated frames.
   - `navg`: shuffle rows within each sequence, chunk into groups of `navg` rows, mean each chunk's features (label is preserved from the first row of the chunk). **This dramatically expands the effective sample size per sequence** (one chunk = one row).
   - Compute mean/std of features on the averaged training set → save to `gbsa_<type>_train_stats.csv` and standardize features.
   - Repeated K-fold (`num_repeats × kfold`) using `GroupKFold` (reg) or `StratifiedGroupKFold` (cls), with sequence as the group → no sequence in both train and val of the same fold.
   - Outputs: `..._trn_{rep}_{fold}.csv`, `..._val_{rep}_{fold}.csv`.

3. **`--process test`** — same filtering + averaging, but standardizes with the training stats CSV (no refit). Output: `..._tst_final.csv`.

### Hyperparameter search

- Each model dir has its own `hyperopt.py` using `hyperopt.SparkTrials` + TPE.
- Search space includes data-prep knobs (`keep_last_percent`, `navg`) **as well as** model hyperparameters — meaning the optimal `navg` and the optimal model arch are jointly tuned. Each trial re-runs `torch_prep_kfold.py --process train` (`SCR_FRACTION=0.0`, no scrambling during tuning) and then `run_model.py --mode 0`.
- Loss: `1 − Pearson` for reg, `0.5*(1−ACC) + 0.5*(1−MCC)` for classification.
- `run_hyperopt.sh` then `run_ML.sh` are the SLURM wrappers. `run_ML.sh` reads the saved `best_hyperparams_<type>.json` with `jq`, regenerates folds, trains 5 reps × 5 folds = 25 models per scramble fraction, then evaluates them all on the held-out test set.

### Metrics

`evaluate_model()` in each `run_model.py` aggregates row-level predictions to **per-sequence** predictions (mean for regression, majority vote for classification) before computing metrics. Reported metrics:
- regression: `MSE`, `R²`, `Pearson`, plus `MCC`/`Accuracy` derived by thresholding predictions at `regression_threshold_log = 0.0` (since data is centered).
- binary: `MCC`, `Accuracy` (others NaN).
- multiclass: same plus macro-F1 via `process_results.ipynb`'s `mclass_metrics`.

### Persistence

- RF: `joblib.dump(rf, "Model/rf_fold_{rep}_{fold}_{type}_scr{frac}.pkl")`.
- NN/REG: `T.save(net.state_dict(), "Model/nn_fold_{rep}_{fold}_{type}_scr{frac}.pth")`.
- SVM: `joblib.dump(svm, "Model/svm_fold_{rep}_{fold}_{type}_scr{frac}.joblib")`.
- Per-fold predictions and aggregated predictions live alongside the scripts (`predictions_*.csv`); not under `Model/`.

### What's actually trained on disk right now

- `ML_RF/`: full RF reg, scr=0.0, including the regenerated `Data/` fold CSVs, 1 of 25 `.pkl` files committed (`rf_fold_0_0_reg_scr0p00.pkl`), all 25 prediction CSVs missing (only `rep0_fold0` checked in).
- `ML_NN/`: 25 `.pth` checkpoints + all 25 prediction CSVs for reg/scr=0.0. **`ML_NN/Data/` is empty** — the fold CSVs were not committed; you can't re-evaluate without regenerating them.
- `ML_SVM/`, `ML_REG/`: scripts only.

---

## 5. Environment & reproducibility

`environment.yml` → `conda env create -f environment.yml && conda activate mmgbsa_ml`.

Key versions (README expands further):
- Python 3.9, numpy ≥1.24 <1.27, **pandas pinned 1.3.5** (this is old and frequently fights with newer numpy — watch for `np.float_` removals etc.)
- scikit-learn 1.3 (in `environment.yml`; README mentions 1.1.3 — discrepancy)
- MDAnalysis 2.4.3, hyperopt 0.2.7, pyspark 3.5.4, jupyterlab 4.2
- shap 0.47.2 (env) vs 0.44.1 (README) — minor mismatch
- selenium 4.27.1, webdriver-manager 4.0.2 — for `Mutate_PDB_w3dna.ipynb` only
- AmberTools 21.12, GROMACS 2022.1 — needed for MD/MMGBSA only (commented out in env)
- NACCESS — external install required for SASA / entropy term

`environment.yml` does NOT pin numpy to 1.20 (README does) — env will pull a newer numpy that may collide with pandas 1.3.5.

---

## 6. Open questions / risks (do not auto-fix — flag only)

These are spots where things look inconsistent or potentially buggy in the **non-tutorial** code. Surface them; let the user decide:

1. **`process_gcPBM.ipynb` cell 24 takes `np.log(Mean Intensity)`**, but `gcPBM_myc_normalized_intensity.txt` header reads "Normalized signal intensity" and the values are ~10³–10⁵ — i.e. raw intensity, not log. So `Log Intensity = ln(MeanIntensity)` is consistent. **However** the existing comment in cell 13 says "Treat Mean Intensity as log-intensity directly" — and uses thresholds `UNBOUND_MAX = 8.65`, `STRONG_MIN = 9.70` on the **raw** Mean Intensity values. The thresholds used downstream (7.7 / 9 / etc.) are on `ln(MeanIntensity)`. Worth confirming the two threshold systems aren't accidentally mixed.

2. **Threshold drift between `stratification.ipynb` (8.0 / 9.0), `process_gcPBM.ipynb` (initial 8 / 9, later 7.7 / 9), and `dataset_old.csv` vs `dataset.csv` (re-labeled with 7.7).** `dataset_old.csv` has 99 rows (33 × 3), `dataset.csv` has 168 (78 + 45 + 45 with the 7.7 threshold). If you (or anyone) re-runs `stratification.ipynb`, the saved CSV would clobber the canonical `dataset.csv` because both paths point at the same name (cell 7's save is commented out, but check before running).

3. **`improving` semantics seem inverted**: `improving = 0 if bind_avg ≥ threshold else 1`. So `improving=1` means *below threshold* = **non-binder/unbound**. The name and the meaning go in opposite directions and downstream `run_hyperopt.sh` uses it as the binary label — you'll want to double-check which class is "positive" in any binary metrics.

4. **`ML_REG/run_model.py` line 370 has `(np.array(preds) - np.array(tgts))2` — that is a syntax/logic typo for `**2`.** This file will fail at import-time on Python 3 because `juxtaposition` of a tuple and `2` raises `SyntaxError`. So the linear-regression script is currently broken. (Did not verify by executing.)

5. **`max_features="auto"` is deprecated/removed in sklearn ≥ 1.3.** `run_model.py` defaults to it, and the Hyperopt space includes `"auto"` as a choice. With `scikit-learn=1.3` from `environment.yml`, the RF trial that samples `"auto"` will raise.

6. **`run_ML.sh` uses `test_percentage 0.2` but `run_hyperopt.sh` uses `test_percentage 0.2` too** — both consistent (20% test). README claims 85/15 in some places — minor doc drift.

7. **PCA artifacts (`pca_*.csv`)** are not produced by any committed script. They might come from an old notebook, the deleted/old tutorial path, or a separate exploration. If `pca_scaler_stats.csv` is supposed to be the canonical scaler, then per-run `gbsa_*_train_stats.csv` (slightly different values) is overriding it. The numbers don't match (`Delta_Entropy std = 2.04` vs `0.79`), so the two scalers were fit on different data — clarify which is authoritative.

8. **`ML_models/ML_RF/Inputs/exp_data_all.csv` and `scripts/exp_data_all.csv` are both modified vs HEAD per `git status`**, and `scripts/process_gcPBM.ipynb` is modified — there's in-flight work in `process_gcPBM.ipynb` and these CSVs may not be in sync. Check before any pipeline rerun.

9. **MD scripts point at `../DNA_library_new/*.pdb`** (`1_run_MD.sh` line 6) but this repo only has `DNA_library/`. Either renamed during dev or a forgotten path — full MD reruns will fail until corrected (cosmetic for tutorial purposes).

10. **`Data/` empty in `ML_NN/`** but checkpoints exist. Anyone who tries to re-evaluate NN without first regenerating folds will get FileNotFoundError. The flow `run_ML.sh` does regenerate them, but if someone runs just `run_model.py --mode 1` it won't work.

11. **`hyperopt.py` Hyperopt result CSV building** (`ML_RF/hyperopt.py` line 302–306) does `rows.append({"loss": ..., "status": ..., trial_params})` — Python dict with a bare positional expression after named ones is a `SyntaxError`. So that file appears broken as-is. (Did not run.)

12. **Sample MD output (`MD_simulations/MycMax_TTTTTTTTTTTTTTGAGAAAATGAAGACAATTATCT/1/`)** contains only replicate 1, not the full 1–20 set. The notebook iterates `range(1, 21)` so 19 replicates will be silently skipped on a real rerun — fine for demo, just noting.

---

## 7. Conventions

- The model dirs use a naming scheme of `<prefix>_<model_type>_scr<frac>_<step>_<rep>_<fold>.csv` where `<frac>` is `0p00`, `0p25`, `1p00` (period replaced by `p`). `model_type ∈ {reg, bin, mclass}`. Prefix is always `gbsa` from `run_*.sh`.
- `Label`/`AvgTrue`/`AvgPredicted` are the column names in aggregated prediction CSVs; per-fold predictions use `Label, Predicted, True`. `Label` is the sequence string.
- Sequence is a 36-mer; all canonical motif positions are computed off `seq[14:22]` as the central 8-mer (CACGTG is the canonical E-box at `seq[15:21]`).

---

## 8. .claude/ setup decision

Skipped creating any project-specific `.claude/` config. A `.claude/settings.local.json` already exists (auto-generated local permissions cache, not in git). No project-wide hooks, agent definitions, or shared settings are warranted yet because:

- No bespoke build/test/lint loop the harness should auto-run.
- No subagents have clear value over the default toolchain for the upcoming "fix tutorial notebooks one at a time" workflow.
- Permissions for `python3` are already in `settings.local.json`.

We can revisit if a recurring pattern (e.g. "always run notebook X after editing Y") emerges.
