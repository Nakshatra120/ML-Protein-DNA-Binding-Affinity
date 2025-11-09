#!/usr/bin/env python

"""
run_model.py

This script trains or evaluates RandomForest models using K-fold cross-validation,
supporting multiple scramble fractions and repeated K-fold splits.

Modes:
------
1) mode=0 (training):
   - For each scramble fraction in --scramble_fractions,
     for each repeat (0 to num_repeats-1) and each fold (0 to kfold-1),
     it loads the corresponding training and validation CSV files,
     trains a RandomForest, saves the model checkpoint, and writes fold-level predictions.
   - Finally, it aggregates metrics across all repeats and folds.

2) mode=1 (evaluation):
   - For each scramble fraction, it loads the final test CSV file.
   - Then, for each repeat and fold, it loads the corresponding model checkpoint,
     computes predictions on the test set, and aggregates metrics and predictions.

"""

import argparse
import sys
import os
import csv
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from collections import defaultdict, Counter
from scipy.stats import pearsonr
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    r2_score,
    mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

###############################################################################
# Logging Configuration
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Utility Functions
###############################################################################
def fmt_float(x: float) -> str:
    """Safely format a float to 4 decimal places."""
    return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"

def majority_vote(values: List[int]) -> int:
    """Return the most frequent value in a list (ties: first mode encountered)."""
    return int(pd.Series(values).mode()[0])

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Random Forest training/evaluation with repeated K-fold splits.")
    parser.add_argument("--mode", type=int, default=0, help="Mode: 0 for training, 1 for evaluation.")
    parser.add_argument("--model_type", type=str, default="reg", choices=["reg", "bin", "mclass"],
                        help="Model type: 'reg', 'bin', or 'mclass'.")
    parser.add_argument("--model_obj", type=str, default="gbsa", help="Model/dataset descriptor (unused in filenames).")
    parser.add_argument("--data_scale", type=str, default="log", choices=["log", "nonlog"],
                        help="Data scale descriptor for thresholding metrics.")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--num_repeats", type=int, default=5, help="Number of times to repeat the K-fold split.")
    parser.add_argument("--model_dir", type=str, default="Model/", help="Directory for saving/loading models.")
    parser.add_argument("--data_dir", type=str, default="Data/", help="Directory where CSV data files are located.")
    parser.add_argument("--output_file", type=str, default="predictions",
                        help="Prefix for output CSV files for predictions and metrics.")
    parser.add_argument("--ref_id_col", type=str, default="sequence", help="ID column name in CSV.")
    parser.add_argument("--ref_label_col", type=str, default="label", help="Label column name in CSV.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=str, default="None", help="Maximum depth of the trees ('None' for unlimited).")
    parser.add_argument("--max_features", type=str, default="auto", help="Feature selection at each split.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum samples required at a leaf node.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--prefix", type=str, default="gbsa", help="Prefix for data file names.")
    parser.add_argument("--scramble_fractions", type=float, nargs="+", default=[0.0],
                        help="List of scramble fractions (e.g., 0.0 0.25 1.0).")
    return parser.parse_args()

###############################################################################
# Data Loading
###############################################################################
def load_csv_data(csv_file: str, args) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Load CSV file assuming it contains an ID column and a label column.
    Returns features (X), targets (y), and IDs.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Data file not found: {csv_file}")
    df = pd.read_csv(csv_file, header=0)
    id_col = args.ref_id_col
    label_col = args.ref_label_col
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV {csv_file} must contain '{id_col}' and '{label_col}'. Found: {df.columns.tolist()}")
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    id_list = df[id_col].tolist()
    return X, y, id_list

###############################################################################
# Build Random Forest
###############################################################################
def build_random_forest(args):
    """
    Create and return a RandomForest (regressor or classifier) using the provided hyperparameters.
    """
    md = None if args.max_depth.lower() == "none" else int(args.max_depth)
    try:
        mf = float(args.max_features)
    except ValueError:
        mf = args.max_features
    if args.model_type == "reg":
        return RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=md, max_features=mf,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state
        )
    else:
        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=md, max_features=mf,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state
        )

###############################################################################
# Evaluate a Model
###############################################################################
def evaluate_model(rf, X_test, y_test, ids, args) -> Tuple[float, float, float, float, float, List[Tuple[Any, float, float]]]:
    """
    Evaluate the given RandomForest model on test data.
    For regression: returns MSE, R2, Pearson, plus binary metrics (MCC, Accuracy) based on threshold.
    For classification: returns MCC and Accuracy (other metrics as NaN).
    Also returns row-level data (list of tuples: (ID, predicted, true)).
    """
    from collections import defaultdict
    if args.model_type == "mclass":
        probas = rf.predict_proba(X_test)
        y_int = y_test.astype(int)
        aggregator = defaultdict(lambda: {"probas": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["probas"].append(probas[i])
            aggregator[uid]["tgt"].append(y_int[i])
            row_data.append((uid, int(np.argmax(probas[i])), int(y_int[i])))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            sum_probs = np.sum(d["probas"], axis=0)
            pred_class = int(np.argmax(sum_probs))
            true_class = int(majority_vote(d["tgt"]))
            agg_preds.append(pred_class)
            agg_tgts.append(true_class)
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc_val = matthews_corrcoef(agg_tgts, agg_preds)
            acc_val = accuracy_score(agg_tgts, agg_preds)
        return (np.nan, np.nan, np.nan, mcc_val, acc_val, row_data)
    elif args.model_type == "bin":
        preds = rf.predict(X_test)
        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(int(preds[i]))
            aggregator[uid]["tgt"].append(int(y_test[i]))
            row_data.append((uid, int(preds[i]), int(y_test[i])))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            agg_preds.append(majority_vote(d["preds"]))
            agg_tgts.append(majority_vote(d["tgt"]))
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc_val = matthews_corrcoef(agg_tgts, agg_preds)
            acc_val = accuracy_score(agg_tgts, agg_preds)
        return (np.nan, np.nan, np.nan, mcc_val, acc_val, row_data)
    else:
        preds = rf.predict(X_test)
        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(preds[i])
            aggregator[uid]["tgt"].append(y_test[i])
            row_data.append((uid, preds[i], y_test[i]))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            agg_preds.append(np.mean(d["preds"]))
            agg_tgts.append(np.mean(d["tgt"]))
        mse_val = mean_squared_error(agg_tgts, agg_preds)
        r2_val, pear_val = float('nan'), float('nan')
        if len(agg_preds) > 1:
            r2_val = r2_score(agg_tgts, agg_preds)
            pear_val, _ = pearsonr(agg_tgts, agg_preds)
        thr = 0.0 if args.data_scale == "log" else 1.0
        pred_cls = (np.array(agg_preds) > thr).astype(int)
        tgt_cls = (np.array(agg_tgts) > thr).astype(int)
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(tgt_cls)) > 1:
            mcc_val = matthews_corrcoef(tgt_cls, pred_cls)
            acc_val = accuracy_score(tgt_cls, pred_cls)
        return (mse_val, r2_val, pear_val, mcc_val, acc_val, row_data)

###############################################################################
# Save Predictions to CSV
###############################################################################
def save_predictions(predictions, filename: str):
    """
    Save row-level predictions to a CSV file with columns: [Label, Predicted, True].
    """
    with open(filename, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Label", "Predicted", "True"])
        wr.writerows(predictions)

###############################################################################
# TRAINING MODE: Repeated K-Fold
###############################################################################
def run_training_mode(args):
    """
    For each scramble fraction in --scramble_fractions, and for each repeat and each fold,
    load the corresponding training and validation CSVs, train a model, save it,
    and collect metrics and predictions.
    """
    os.makedirs(args.model_dir, exist_ok=True)
    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        logging.info(f"=== Training: scramble fraction={fraction} => scr{frac_str} ===")
        all_metrics = []
        all_predictions = []
        for repeat_idx in range(args.num_repeats):
            for fold_idx in range(args.kfold):
                trn_file = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_{repeat_idx}_{fold_idx}.csv"
                )
                val_file = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_val_{repeat_idx}_{fold_idx}.csv"
                )
                if not (os.path.isfile(trn_file) and os.path.isfile(val_file)):
                    logging.warning(f"Missing CSV for fraction={fraction}, repeat={repeat_idx}, fold={fold_idx}")
                    continue
                X_train, y_train, ids_train = load_csv_data(trn_file, args)
                X_val, y_val, ids_val = load_csv_data(val_file, args)
                if args.model_type in ["bin", "mclass"]:
                    y_train = y_train.astype(int)
                    y_val = y_val.astype(int)
                rf = build_random_forest(args)
                logging.info(f"[Frac={fraction}] Repeat={repeat_idx}, Fold={fold_idx}: training model...")
                rf.fit(X_train, y_train)
                model_path = os.path.join(
                    args.model_dir,
                    f"rf_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}.pkl"
                )
                joblib.dump(rf, model_path)
                mse, r2, pear, mcc, acc, fold_data = evaluate_model(rf, X_val, y_val, ids_val, args)
                all_metrics.append({"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc})
                all_predictions.extend(fold_data)
                fold_pred_file = f"{args.output_file}_{args.model_type}_scr{frac_str}_rep{repeat_idx}_fold{fold_idx}.csv"
                save_predictions(fold_data, fold_pred_file)
                logging.info(
                    f"[Frac={fraction}, Rep={repeat_idx}, Fold={fold_idx}] "
                    f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, Pear={fmt_float(pear)}, "
                    f"MCC={fmt_float(mcc)}, Acc={fmt_float(acc)}"
                )
        if all_metrics:
            keys = all_metrics[0].keys()
            avg_metrics = {k: float(np.mean([m[k] for m in all_metrics if not np.isnan(m[k])])) for k in keys}
            logging.info(f"[Frac={fraction}] Final Average Training Metrics (all repeats/folds):")
            for k, v in avg_metrics.items():
                logging.info(f"  {k} = {fmt_float(v)}")
            final_csv = f"final_metrics_{args.model_type}_trn_scr{frac_str}.csv"
            with open(final_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE", "R2", "Pear", "MCC", "Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(avg_metrics[k]) for k in ["MSE", "R2", "Pear", "MCC", "Accuracy"]})
        if all_predictions:
            aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
            for uid, pred, tgt in all_predictions:
                aggregator[uid]["preds"].append(pred)
                aggregator[uid]["tgt"].append(tgt)
            final_labels, final_preds, final_tgts = [], [], []
            for lbl, d in aggregator.items():
                if args.model_type in ["bin", "mclass"]:
                    final_labels.append(lbl)
                    final_preds.append(majority_vote(d["preds"]))
                    final_tgts.append(majority_vote(d["tgt"]))
                else:
                    final_labels.append(lbl)
                    final_preds.append(np.mean(d["preds"]))
                    final_tgts.append(np.mean(d["tgt"]))
            agg_df = pd.DataFrame({
                "Label": final_labels,
                "AvgPredicted": final_preds,
                "AvgTrue": final_tgts
            })
            agg_csv = f"{args.output_file}_{args.model_type}_final_avg_scr{frac_str}.csv"
            agg_df.to_csv(agg_csv, index=False)
            logging.info(f"Final aggregated predictions saved: {agg_csv}")

###############################################################################
# EVALUATION MODE
###############################################################################
def run_evaluation_mode(args):
    """
    For each scramble fraction, load the test CSV and evaluate all models from each repeat and fold.
    """
    os.makedirs(args.model_dir, exist_ok=True)
    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        test_csv = os.path.join(args.data_dir, f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv")
        if not os.path.isfile(test_csv):
            logging.warning(f"[Frac={fraction}] Test CSV not found: {test_csv}")
            continue
        logging.info(f"=== Evaluating for scramble fraction={fraction}, test file={test_csv} ===")
        X_test, y_test, ids_test = load_csv_data(test_csv, args)
        if args.model_type in ["bin", "mclass"]:
            y_test = y_test.astype(int)
        predictions = []
        test_metrics = []
        for repeat_idx in range(args.num_repeats):
            for fold_idx in range(args.kfold):
                model_path = os.path.join(
                    args.model_dir,
                    f"rf_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}.pkl"
                )
                if not os.path.isfile(model_path):
                    logging.warning(f"Missing model for frac={fraction}, rep={repeat_idx}, fold={fold_idx}: {model_path}")
                    continue
                rf = joblib.load(model_path)
                mse, r2, pear, mcc, acc, fold_preds = evaluate_model(rf, X_test, y_test, ids_test, args)
                predictions.extend(fold_preds)
                test_metrics.append({"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc})
                logging.info(
                    f"[Frac={fraction}, Rep={repeat_idx}, Fold={fold_idx}] "
                    f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, Pear={fmt_float(pear)}, "
                    f"MCC={fmt_float(mcc)}, Acc={fmt_float(acc)}"
                )
        pred_csv = f"{args.output_file}_test_{args.model_type}_scr{frac_str}.csv"
        save_predictions(predictions, pred_csv)
        logging.info(f"Test predictions saved: {pred_csv}")
        if test_metrics:
            keys = test_metrics[0].keys()
            avg_metrics = {k: float(np.mean([m[k] for m in test_metrics if not np.isnan(m[k])])) for k in keys}
            logging.info(f"[Frac={fraction}] Final Average Test Metrics (all repeats/folds):")
            for k, v in avg_metrics.items():
                logging.info(f"  {k} = {fmt_float(v)}")
            final_test_csv = f"final_metrics_{args.model_type}_tst_scr{frac_str}.csv"
            with open(final_test_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE", "R2", "Pear", "MCC", "Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(avg_metrics[k]) for k in ["MSE", "R2", "Pear", "MCC", "Accuracy"]})
            logging.info(f"Final test metrics saved: {final_test_csv}")

###############################################################################
# Main 
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    logging.info(f"Scramble fractions: {args.scramble_fractions}")
    logging.info(f"K-fold={args.kfold}, num_repeats={args.num_repeats}")
    if args.mode == 0:
        run_training_mode(args)
    elif args.mode == 1:
        run_evaluation_mode(args)
    else:
        logging.error("Mode must be 0 or 1.")
        sys.exit(1)

if __name__ == "__main__":
    main()
