import argparse

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

import os
import sys
import time
import gc
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import cocoex  # COCO benchmarking
import cocopp
from scipy.interpolate import interp1d

from optimization import Optimizer
from optimization_transfer import TransferOptimizer
from optimization_reinforcement import ReinforcementOptimizer
from model import Model
from scaler import Scalers
from ratio import RealEvaluationRatio

def compute_coco_ecdf_from_dat(dat_path: str, dim: int,
                               targets=None, budget_grid=None):
    """
    COCO-style ECDF from a single .dat file (bbob-new2 format):
      - uses the 'best noise-free fitness - fopt' column
      - 51 logarithmic targets (100 .. 1e-8) by default
      - returns x_log (log10 budget), y (fraction solved), and the flat runtimes array
    """
    if targets is None:
        targets = np.logspace(2, -8, 51)  # 100 .. 1e-8, 5 per decade
    if budget_grid is None:
        budget_grid = np.logspace(0, 7, 400)  # 1 .. 1e7 evals/dim

    # --- parse .dat into runs: each run = list of (fevals, best_noise_free_minus_fopt)
    runs, cur = [], []
    with open(dat_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%"):           # new run starts after a '%' header
                if cur:
                    runs.append(cur); cur = []
                continue
            parts = s.split()
            # columns: f-evals, g-evals, best_noise_free_minus_fopt, best_measured, x1, ...
            try:
                fevals = int(parts[0])
                best_nf = float(parts[2].replace("+", ""))  # noise-free − fopt
                cur.append((fevals, best_nf))
            except Exception:
                pass
    if cur:
        runs.append(cur)

    # --- first-hitting times for each (run, target)
    all_runtimes = []
    for run in runs:
        run.sort(key=lambda t: t[0])
        fe = np.array([t[0] for t in run], dtype=float)
        fbest = np.array([t[1] for t in run], dtype=float)

        # enforce monotone best-so-far (COCO’s fbest is nonincreasing)
        fbest = np.minimum.accumulate(fbest)

        for t in targets:
            idx = np.where(fbest <= t)[0]
            if idx.size == 0:
                all_runtimes.append(np.inf)           # unsolved target
            else:
                all_runtimes.append(fe[idx[0]] / dim) # normalize by D

    all_runtimes = np.asarray(all_runtimes, dtype=float)      # length = #runs * #targets
    n_total = all_runtimes.size

    # --- ECDF on a log-spaced budget grid
    y = np.array([np.mean(all_runtimes <= b) for b in budget_grid], dtype=float)
    x_log = np.log10(budget_grid)
    return x_log, y, all_runtimes


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# ----------------------------
# Per-instance Optimizer
# ----------------------------
def optimize_instance(args):
    solveds = []
    evals = []
    out_folders = []
        
    for instance in range(args.instances_start, args.instances_end+1):
        suite = cocoex.Suite("bbob", "", f"dimensions:{args.dimension} function_indices:{args.function_id} instance_indices:{instance}")
        observer = cocoex.Observer("bbob", f"result_folder: {args.experiment_name}-{instance}")

        for problem in suite:
            if problem.dimension == args.dimension:
                problem.observe_with(observer)
                setattr(args, "problem_id", problem.id)
                args.model = Model(args)
                
                optimizer = Optimizer()
                #optimizer = TransferOptimizer()
                #optimizer = ReinforcementOptimizer()
                solved, evls = optimizer.optimize(problem, args)
                solveds.append(solved)
                evals.append(evls)
                
        out_folders.append(observer.result_folder)

    # ⚠️ Clean up everything to finalize logging
    del suite
    del observer
    del problem
    gc.collect()
    time.sleep(0.5)
    return out_folders, solveds, evals

# ----------------------------
# Main: CLI + Dispatch
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare black-box optimization algorithms on COCO.")
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--max_evals_per_dim", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--instances_start", type=int, default=1)
    parser.add_argument("--instances_end", type=int, default=15)
    parser.add_argument("--max_best_difference", type=float, default=1e-8)

    parser.add_argument("--sigma", type=float, default=1.0)
    
    parser.add_argument("--experiment_name", type=str, required=True)
    
    parser.add_argument("--scaler", type=str, default="")
    parser.add_argument("--y_scaler", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--ensemble_type", type=str, choices=["mean", "best", "weighted", "actor-critic"], default = "mean")

    parser.add_argument("--init", type=str, choices=["zeros", "lhs", "slsqp", "sobel"], default = "zeros")
    parser.add_argument("--init_slsqp_iterations", type=int, default=300)
    parser.add_argument("--init_slsqp_accuracy", type=float, default=1e-11)
    parser.add_argument("--pop_increase", type=str, choices=["none", "constant", "ipop", "bipop", "nipop", "nbipop"], default = "none")
    parser.add_argument("--elitism", type=str2bool, nargs="?", const=True, default=False)
    
    parser.add_argument("--real_evaluation_ratio", type=str, default="static-0.5")
    parser.add_argument("--cma_only_real", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--train_only_real", type=str2bool, nargs="?", const=True, default=False)
    
    parser.add_argument("--check_flatness", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--flatness_alpha", type=float, default=0.05) # lower alpha = stricted flatness
    parser.add_argument("--check_train_kendalltau", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--fit_until", type=str, choices=["none", "unchanged_ranking", "unchanged_set"], default = "none")
    
    parser.add_argument("--tss", type=str, choices=["full", "knn", "nearest"], default = "full")
    
    parser.add_argument("--check_surrogate_life", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_pair_model", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--optimizer", type=str, choices=["cma", "slsqp", "fmincon", "fminunc", "csl", "hees", "shade"], default = "cma")
    parser.add_argument("--slsqp_ftol", type=float, default=1e-15)
    parser.add_argument("--hees_tol", type=float, default=1e-10)
    parser.add_argument("--mlsl_gamma", type=float, default=0.1)
    parser.add_argument("--slsqp_difference_gradient", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--cma_type", type=str, choices=["basic", "led", "lm", "mo"], default = "basic")
    parser.add_argument("--active_cma", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--ask_by", type=str, choices=["cma", "around_best"], default = "cma")
    parser.add_argument("--center_injection", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--chaotic_extend", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prediction_transform", type=str, choices=["none", "ucb", "ilcb"], default = "none")

    parser.add_argument("--step", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--step_tol", type=float, default=1e-10)
    parser.add_argument("--step_rho", type=float, default=0.5)
    parser.add_argument("--step_min_iters", type=float, default=10)

    parser.add_argument("--eval_problem", type=str, choices=["standard", "local_search"], default = "standard")
    parser.add_argument("--local_search_frac", type=float, default=0.1)
    parser.add_argument("--local_search_check", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--local_search_distance_tol", type=float, default=1e-3)
    parser.add_argument("--local_search_merit_tol", type=float, default=1e-6)
    parser.add_argument("--local_search_de_refinement", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--shade_lm", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--load_metadata", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_metadata", type=str2bool, nargs="?", const=True, default=False)

    args = parser.parse_args()
    args.real_evaluation_ratio = RealEvaluationRatio(args)

    args.scaler = Scalers.get(args.scaler, args)
    args.y_scaler = Scalers.get(args.y_scaler, args)

    res_folders, solveds, evals = optimize_instance(args = args)
            
    print(res_folders)
    print("\n📊 Running COCO post-processing (HTML will NOT auto-open)...")
            
    #subprocess.run([sys.executable, "-m", "cocopp", *res_folders])
    x_logs, ys, runtimes = [], [], []
    for res_folder in res_folders:
        # Load experiment data
        dsList = cocopp.load(res_folder)
        data_path = f"{res_folder}/data_f{args.function_id}/bbobexp_f{args.function_id}_DIM{args.dimension}.dat"
        x_log, y, rntms = compute_coco_ecdf_from_dat(data_path, dim=5)
        x_logs.append(x_log)
        ys.append(y)
        runtimes.append(rntms)

    y = np.mean(ys, axis = 0)
    y_std = np.std(ys, axis = 0)
    y_lo = np.clip(y - y_std, 0.0, 1.0)
    y_hi = np.clip(y + y_std, 0.0, 1.0)

    if np.sum(solveds) == len(solveds):
        print(f"Mean evals: {np.mean(evals)} +- {np.std(evals)}, Best FFTPS: {y[-1]} +- {y_std[-1]}")
        with open(f"results/{args.experiment_name}-{args.function_id}.txt", "a") as f:
            f.write(f"[{datetime.now()}] Mean evals: {np.mean(evals)} +- {np.std(evals)}, Best FFTPS: {y[-1]} +- {y_std[-1]}\n")
    else:
        print(f"Mean evals: np.inf (some instances was not solved), Best FFTPS: {y[-1]} +- {y_std[-1]}")
        with open(f"results/{args.experiment_name}-{args.function_id}.txt", "a") as f:
            f.write(f"Mean evals: np.inf (some instances was not solved), Best FFTPS: {y[-1]} +- {y_std[-1]}\n")

    with open(f"results/{args.experiment_name}-{args.function_id}.txt", "a") as f:
        for i in range(len(x_log)):
            f.write(f"{x_log[i]}:{y[i]}\n")

    if False:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.plot(x_log, y)
        plt.fill_between(x_log, y_lo, y_hi, alpha=0.25, label="±1 std")
        plt.ylim(0,1); plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("log10(# evals / dimension)")
        plt.ylabel("Fraction of function–target pairs solved")
        plt.title("COCO-style ECDF (manual from .dat)")
        plt.show()


if __name__ == "__main__":
    main()
