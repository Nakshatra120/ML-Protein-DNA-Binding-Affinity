#!/usr/bin/env python
"""
hyperopt.py

Bayesian hyper-parameter optimisation wrapper for a Random-Forest
model evaluated via repeated K-fold cross-validation.

Flow of each Hyperopt trial
---------------------------
1. Data preparation  
   Execute `torch_prep_kfold.py --process train`, which (re)builds the
   train/validation K-fold splits for the current hyper-parameters
   (fraction of rows kept, row-averaging window, etc.).
2. Training / CV  
   Call `run_model.py --mode 0`, which trains the Random-Forest on each
   fold (using the hyper-parameters sampled by Hyperopt) and writes a CSV
   summarising averaged validation metrics.
3. Objective  
   Parse the CSV and convert the metrics into a single scalar *loss*
   (lower is better) returned to Hyperopt.
4. SparkTrials  
   Many trials are executed in parallel using a Spark session.
"""

###############################################################################
# Imports
###############################################################################
import os, csv, json, logging, argparse, subprocess
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, SparkTrials, space_eval
from pyspark.sql import SparkSession

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# CLI
###############################################################################
def parse_args():
    """
    Build / parse command-line arguments. 
    """
    parser = argparse.ArgumentParser(
        description="Bayesian hyper-parameter optimisation for Random-Forest (K-fold)."
    )

    # -------- external scripts & directories -------------------------- #
    parser.add_argument("--prep_script",  default="torch_prep_kfold.py")
    parser.add_argument("--train_script", default="run_model.py")
    parser.add_argument("--data_dir",  default="Data")
    parser.add_argument("--model_dir", default="Model")

    # -------- reference data & feature CSVs --------------------------- #
    parser.add_argument("--reference_file", default="../Inputs/exp_data_all.csv")
    parser.add_argument("--ref_id_col",    default="sequence")
    parser.add_argument("--ref_label_col", default="label")
    parser.add_argument("--filenames", nargs="+", required=True)
    parser.add_argument("--feature_id_col", default="sequence")
    parser.add_argument("--usecols", nargs="+")

    # -------- data-prep settings ------------------------------------- #
    parser.add_argument("--keep_last_percent", type=float, default=0.0)
    parser.add_argument("--model_type", choices=["reg","bin","mclass"], required=True)
    parser.add_argument("--data_scale", choices=["log","nonlog"], default="log")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--spark_parallelism", type=int, default=4)
    parser.add_argument("--max_evals", type=int, default=50)

    # -------- naming / output ---------------------------------------- #
    parser.add_argument("--prefix",      default="gbsa")
    parser.add_argument("--output_file", default="predictions")
    return parser.parse_args()

###############################################################################
# Hyperopt search space
###############################################################################
search_space = {
    # ---------- data-prep -------------------------------------------- #
    'keep_last_percent': hp.quniform('keep_last_percent', 5, 100, 5),
    'navg':              hp.quniform('navg', 1, 400, 20),

    # ---------- Random-Forest hyper-parameters ----------------------- #
    'n_estimators':      hp.quniform('n_estimators', 50, 500, 50),
    'max_depth':         hp.choice('max_depth', [None, 10, 15, 20, 25, 30]),
    'max_features':      hp.choice('max_features',
                                   ['auto', 'sqrt', 'log2', None, 0.5]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 50, 2),
    'min_samples_leaf':  hp.quniform('min_samples_leaf', 1, 20, 1),
}

###############################################################################
# Constant (no label scrambling during tuning)
###############################################################################
SCR_FRACTION = 0.0
FRAC_STR = f"{SCR_FRACTION:.2f}".replace(".", "p")   # "0p00"

###############################################################################
# ---------------- helper: data-prep ---------------------------------------- #
###############################################################################
def run_data_prep(keep_last_percent: float, navg: int, args):
    """
    Execute *torch_prep_kfold.py* (in args.data_dir).  Any non-zero exit
    aborts the current trial.
    """
    original_cwd = os.getcwd()
    os.chdir(args.data_dir)
    try:
        cmd = [
            "python", args.prep_script,
            "--process", "train",
            "--reference_file", args.reference_file,
            "--ref_id_col",     args.ref_id_col,
            "--ref_label_col",  args.ref_label_col,
            "--feature_id_col", args.feature_id_col,
            "--keep_last_percent", str(keep_last_percent),
            "--model_type",     args.model_type,
            "--navg",           str(int(navg)),
            "--random_state",   str(args.random_state),
            "--prefix",         args.prefix,
            "--scramble_fractions", str(SCR_FRACTION)
        ]
        if args.filenames: cmd += ["--filenames"] + args.filenames
        if args.usecols:   cmd += ["--usecols"]   + args.usecols

        logging.info("[DataPrep] %s", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        if res.returncode != 0:
            logging.error("Data-prep failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
                          res.stdout, res.stderr)
            raise RuntimeError("Data-prep script error.")
    finally:
        os.chdir(original_cwd)

###############################################################################
# ---------------- helper: training / evaluation --------------------------- #
###############################################################################
def run_training_evaluation(
    *,
    keep_last_percent: float,
    navg: int,
    n_estimators: int,
    max_depth: Any,
    max_features: Any,
    min_samples_split: int,
    min_samples_leaf: int,
    args
) -> dict:
    """
    Execute *run_model.py* (`--mode 0`) with Random-Forest hyper-parameters.
    Reads the resulting metrics CSV and returns it as a dict.
    """
    md_str = "None" if max_depth is None else str(max_depth)
    mf_str = "None" if max_features is None else str(max_features)

    cmd = [
        "python", args.train_script,
        "--mode", "0",
        "--model_type", args.model_type,
        "--data_scale", args.data_scale,
        "--kfold", "5",

        "--n_estimators",      str(n_estimators),
        "--max_depth",         md_str,
        "--max_features",      mf_str,
        "--min_samples_split", str(min_samples_split),
        "--min_samples_leaf",  str(min_samples_leaf),

        "--ref_id_col",    args.ref_id_col,
        "--ref_label_col", args.ref_label_col,
        "--scramble_fractions", str(SCR_FRACTION)
    ]
    logging.info("[Training] %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        logging.error("Training failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
                      res.stdout, res.stderr)
        raise RuntimeError("run_model.py error.")

    metrics_csv = f"final_metrics_{args.model_type}_trn_scr{FRAC_STR}.csv"
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"{metrics_csv} not generated.")

    with open(metrics_csv) as f:
        row = next(csv.DictReader(f))

    return (
        {"MSE": float(row["MSE"]), "Pear": float(row["Pear"])}
        if args.model_type == "reg"
        else {"Accuracy": float(row["Accuracy"]), "MCC": float(row["MCC"])}
    )

###############################################################################
# ---------------- Hyperopt objective --------------------------------------- #
###############################################################################
def evaluate_hyperparams(params: dict, args) -> float:
    """
    Converts sampled hyper-parameters → scalar loss.
    Large penalty (1e6) on failure or NaN metrics.
    """
    # unpack numeric params
    keep_last = float(params['keep_last_percent'])
    navg      = int(params['navg'])
    n_estim   = int(params['n_estimators'])
    max_depth = params['max_depth']
    max_feat  = params['max_features']
    min_split = int(params['min_samples_split'])
    min_leaf  = int(params['min_samples_leaf'])

    # 1. Data prep
    try:
        run_data_prep(keep_last, navg, args)
        metrics = run_training_evaluation(
            keep_last_percent = keep_last,
            navg              = navg,
            n_estimators      = n_estim,
            max_depth         = max_depth,
            max_features      = max_feat,
            min_samples_split = min_split,
            min_samples_leaf  = min_leaf,
            args = args
        )
    except Exception as e:
        logging.error("[Objective] Trial failed: %s", e)
        return 1e6

    # 2. Metrics → loss
    if args.model_type == "reg":
        mse, pear = metrics["MSE"], metrics["Pear"]
        if np.isnan(mse) or np.isnan(pear): return 1e6
        loss = 1.0 - pear          # maximise Pearson
    else:
        acc, mcc = metrics["Accuracy"], metrics["MCC"]
        if np.isnan(acc) or np.isnan(mcc): return 1e6
        loss = 0.5 * (1.0 - acc) + 0.5 * (1.0 - mcc)

    logging.info(
        "[Objective] params=%s, metrics=%s, loss=%.4f",
        params, metrics, loss
    )
    return loss

###############################################################################
# Main entry-point
###############################################################################
def main():
    args = parse_args()

    # Spark backend for parallel trials
    spark = (SparkSession.builder
             .appName("HyperoptRF")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    trials = SparkTrials(parallelism=args.spark_parallelism)

    # run Hyperopt optimisation
    best_indices = fmin(
        fn=lambda p: evaluate_hyperparams(p, args),
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        verbose=False
    )

    # convert choice indices → actual values
    best_params = space_eval(search_space, best_indices)
    logging.info("Best hyper-params (actual): %s", best_params)

    # persist best configuration
    best_cfg = {
        "keep_last_percent": float(best_params['keep_last_percent']),
        "navg":              int(best_params['navg']),
        "n_estimators":      int(best_params['n_estimators']),
        "max_depth":         best_params['max_depth'],
        "max_features":      best_params['max_features'],
        "min_samples_split": int(best_params['min_samples_split']),
        "min_samples_leaf":  int(best_params['min_samples_leaf']),
        "model_type":        args.model_type,
        "data_scale":        args.data_scale
    }
    with open(f"best_hyperparams_{args.model_type}.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logging.info("Best hyper-parameters saved.")

    # collect each trial → CSV
    rows = []
    for t in trials.trials:
        res  = t['result']
        vals = t['misc']['vals']
        # convert indices to actual values for each trial
        trial_params = space_eval(search_space, {k: v[0] for k,v in vals.items()})
        rows.append({
            "loss":   res.get('loss'),
            "status": res.get('status'),
            trial_params
        })
    pd.DataFrame(rows).to_csv(f"hyperopt_results_{args.model_type}.csv", index=False)
    logging.info("All trial results saved.")

    spark.stop()


if __name__ == "__main__":
    main()
