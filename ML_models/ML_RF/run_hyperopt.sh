#!/bin/bash
#SBATCH --job-name=hyperopt
#SBATCH --account=jyu20_lab
#SBATCH --partition=standard
#SBATCH --mem=30000
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# Activate your conda environment or module load python
source /opt/apps/anaconda/2020.07/bin/activate /data/homezvol1/calmasri/.conda/envs/mmgbsa_ml

###############################################################################
# 1) Parse Model Type
###############################################################################
# Usage: sbatch run_hyperopt.sh [model_type]
MODEL_TYPE=${1:-reg}

# Pick a reference label column based on the model_type
if [ "$MODEL_TYPE" = "bin" ]; then
  REF_LABEL_COL="improving"
elif [ "$MODEL_TYPE" = "mclass" ]; then
  REF_LABEL_COL="binding_type"
elif [ "$MODEL_TYPE" = "reg" ]; then
  REF_LABEL_COL="bind_avg"
else
  echo "Unknown model type: $MODEL_TYPE"
  exit 1
fi

echo "Model type: $MODEL_TYPE"
echo "Reference label col: $REF_LABEL_COL"

###############################################################################
# 2) Perform Initial 85/15 Split (if not done already)
###############################################################################
cd Data
python torch_prep_kfold.py \
  --reference_file ../Inputs/exp_data_all.csv \
  --ref_id_col sequence \
  --ref_label_col "$REF_LABEL_COL" \
  --filenames ../Inputs/rawdat.csv \
  --feature_id_col sequence \
  --model_type "$MODEL_TYPE" \
  --random_state 42 \
  --test_percentage 0.2 \
  --prefix "gbsa" \
  --initial_split
cd ..
echo "Initial train/test split completed."

###############################################################################
# 3) Run Hyperparameter Tuning with Hyperopt
###############################################################################
python hyperopt.py \
  --reference_file ../Inputs/exp_data_all.csv \
  --ref_id_col sequence \
  --ref_label_col "$REF_LABEL_COL" \
  --filenames ../Inputs/rawdat.csv \
  --feature_id_col sequence \
  --model_type "$MODEL_TYPE" \
  --data_scale log \
  --max_evals 1000 \
  --prefix "gbsa" \
  --data_dir "Data" \
  --model_dir "Model"

echo "Hyperparameter optimization completed!"
