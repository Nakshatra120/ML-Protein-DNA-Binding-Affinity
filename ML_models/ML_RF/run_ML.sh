#!/bin/bash
#
#SBATCH --job-name=rf_final
##SBATCH --account=some_account
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=4

# Activate your environment
source /opt/apps/anaconda/2020.07/bin/activate /data/homezvol1/calmasri/.conda/envs/mmgbsa_ml

# Abort if any command fails
set -e

###############################################################################
# 1) Parse Inputs
###############################################################################
# Usage: sbatch run_ML.sh [model_type] [scramble_fractions]
# Example: sbatch run_ML.sh reg "0.0 0.25 1.0"
MODEL_TYPE=${1:-reg}
SCRAMBLE_FRACTIONS="${2:-0.0}"  # e.g., "0.0 0.25 1.0"

# Convert the scramble fractions string into an array
read -ra FRAC_ARRAY <<< "$SCRAMBLE_FRACTIONS"

# Set the reference label column based on model type
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
echo "Reference label column: $REF_LABEL_COL"
echo "Scramble fractions: ${FRAC_ARRAY[@]}"

# Define repeated K-fold parameters
NUM_REPEATS=5    # Number of repeats for the K-fold split (e.g., 5 repeats)
KFOLD=5          # Number of folds per repeat (e.g., 5 folds => 25 splits total)

###############################################################################
# 2) For each scramble fraction, run the pipeline
###############################################################################
for FRACTION in "${FRAC_ARRAY[@]}"; do
  echo "============================================================="
  echo "=== Processing pipeline for scramble fraction=${FRACTION} ==="
  echo "============================================================="
  FRAC_STR=$(echo "${FRACTION}" | sed 's/\./p/')

  #############################################################################
  # 2A) Initial 85/15 Train/Test Split
  #############################################################################
  # This step merges your reference and feature data, and splits it into training and test sets.
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
    --scramble_fractions "$FRACTION" \
    --initial_split
  cd ..
  echo "[Fraction=${FRACTION}] Initial train/test split completed."

  #############################################################################
  # 2B) Load Best Hyperparameters (from JSON via jq)
  #############################################################################
  BEST_JSON="best_hyperparams_${MODEL_TYPE}.json"
  if [ ! -f "$BEST_JSON" ]; then
    echo "Error: $BEST_JSON not found. Did you run hyperopt?"
    exit 1
  fi
  KEEP_LAST_PERCENT=$(jq .keep_last_percent "$BEST_JSON")
  NAVG=$(jq .navg "$BEST_JSON")
  N_EST=$(jq .n_estimators "$BEST_JSON")
  MAX_DEPTH=$(jq .max_depth "$BEST_JSON" | sed 's/"//g')
  MAX_FEATURES=$(jq .max_features "$BEST_JSON" | sed 's/"//g')
  MIN_SPLIT=$(jq .min_samples_split "$BEST_JSON")
  MIN_LEAF=$(jq .min_samples_leaf "$BEST_JSON")

  echo "[Fraction=${FRACTION}] Loaded best hyperparameters from $BEST_JSON:"
  echo "  keep_last_percent = $KEEP_LAST_PERCENT"
  echo "  navg              = $NAVG"
  echo "  n_estimators      = $N_EST"
  echo "  max_depth         = $MAX_DEPTH"
  echo "  max_features      = $MAX_FEATURES"
  echo "  min_samples_split = $MIN_SPLIT"
  echo "  min_samples_leaf  = $MIN_LEAF"

  #############################################################################
  # 2C) Re-run Data Prep for Training (Repeated K-fold)
  #############################################################################
  cd Data
  python torch_prep_kfold.py \
    --reference_file ../Inputs/exp_data_all.csv \
    --ref_id_col sequence \
    --ref_label_col "$REF_LABEL_COL" \
    --filenames ../Inputs/rawdat.csv \
    --feature_id_col sequence \
    --model_type "$MODEL_TYPE" \
    --keep_last_percent "$KEEP_LAST_PERCENT" \
    --navg "$NAVG" \
    --random_state 42 \
    --test_percentage 0.2 \
    --prefix "gbsa" \
    --scramble_fractions "$FRACTION" \
    --kfold $KFOLD \
    --num_repeats $NUM_REPEATS \
    --process train
  cd ..
  echo "[Fraction=${FRACTION}] Data prep for training (repeated K-fold) completed."

  #############################################################################
  # 2D) Process the Final Test Set
  #############################################################################
  cd Data
  python torch_prep_kfold.py \
    --reference_file ../Inputs/exp_data_all.csv \
    --ref_id_col sequence \
    --ref_label_col "$REF_LABEL_COL" \
    --filenames ../Inputs/rawdat.csv \
    --feature_id_col sequence \
    --model_type "$MODEL_TYPE" \
    --keep_last_percent "$KEEP_LAST_PERCENT" \
    --navg "$NAVG" \
    --random_state 42 \
    --test_percentage 0.2 \
    --prefix "gbsa" \
    --scramble_fractions "$FRACTION" \
    --kfold $KFOLD \
    --num_repeats $NUM_REPEATS \
    --process test
  cd ..
  echo "[Fraction=${FRACTION}] Data prep for test set completed."

  #############################################################################
  # 2E) Run Random Forest Training (mode=0)
  #############################################################################
  echo "[Fraction=${FRACTION}] === Running Random Forest Training ==="
  python run_model.py \
    --mode 0 \
    --model_type "$MODEL_TYPE" \
    --data_scale "log" \
    --kfold $KFOLD \
    --num_repeats $NUM_REPEATS \
    --n_estimators "$N_EST" \
    --max_depth "$MAX_DEPTH" \
    --max_features "$MAX_FEATURES" \
    --min_samples_split "$MIN_SPLIT" \
    --min_samples_leaf "$MIN_LEAF" \
    --ref_id_col sequence \
    --ref_label_col "$REF_LABEL_COL" \
    --model_dir "Model" \
    --data_dir "Data" \
    --output_file "predictions" \
    --scramble_fractions "$FRACTION"
  echo "[Fraction=${FRACTION}] Random Forest training completed."

  #############################################################################
  # 2F) Evaluate on Final Test Set (mode=1)
  #############################################################################
  echo "[Fraction=${FRACTION}] === Evaluating on Final Test Set ==="
  python run_model.py \
    --mode 1 \
    --model_type "$MODEL_TYPE" \
    --data_scale "log" \
    --kfold $KFOLD \
    --num_repeats $NUM_REPEATS \
    --n_estimators "$N_EST" \
    --max_depth "$MAX_DEPTH" \
    --max_features "$MAX_FEATURES" \
    --min_samples_split "$MIN_SPLIT" \
    --min_samples_leaf "$MIN_LEAF" \
    --ref_id_col sequence \
    --ref_label_col "$REF_LABEL_COL" \
    --model_dir "Model" \
    --data_dir "Data" \
    --output_file "predictions" \
    --scramble_fractions "$FRACTION"
  echo "[Fraction=${FRACTION}] Final test set evaluation completed."

done

echo "All scramble fractions completed: ${FRAC_ARRAY[@]}"
