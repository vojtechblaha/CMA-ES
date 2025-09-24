import argparse

import os
import sys
import time
import gc
import subprocess

import cocoex  # COCO benchmarking

from optimization import Optimizer
from model import Model
from scaler import Scalers
from ratio import RealEvaluationRatio

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
    suite = cocoex.Suite("bbob", "", f"dimensions:{args.dimension} function_indices:{args.function_id} instance_indices:1-{args.instances}")
    observer = cocoex.Observer("bbob", f"result_folder: {args.experiment_name}")
    
    for problem in suite:
        if problem.dimension == args.dimension:
            problem.observe_with(observer)
            
            optimizer = Optimizer()
            optimizer.optimize(problem, args)

    out_folder = observer.result_folder

    # ⚠️ Clean up everything to finalize logging
    del suite
    del observer
    del problem
    gc.collect()
    time.sleep(0.5)
    return out_folder

# ----------------------------
# Main: CLI + Dispatch
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare black-box optimization algorithms on COCO.")
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--max_evals_per_dim", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--instances", type=int, default=15)

    parser.add_argument("--sigma", type=float, default=1.0)
    
    parser.add_argument("--experiment_name", type=str, required=True)
    
    parser.add_argument("--scaler", type=str, default="")
    parser.add_argument("--y_scaler", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--ensemble_type", type=str, choices=["mean", "best", "weighted", "actor-critic"], default = "mean")

    parser.add_argument("--init", type=str, choices=["zeros", "lhs"], default = "zeros")
    parser.add_argument("--pop_increase", type=str, choices=["none", "ipop", "bipop", "nipop", "nbipop"], default = "none")
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
    parser.add_argument("--cma_type", type=str, choices=["basic", "led", "lm", "mo"], default = "basic")
    parser.add_argument("--active_cma", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--ask_by", type=str, choices=["cma", "around_best"], default = "cma")
    parser.add_argument("--center_injection", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--chaotic_extend", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prediction_transform", type=str, choices=["none", "ucb", "ilcb"], default = "none")
    
    args = parser.parse_args()

    args.model = Model(args)
    args.scaler = Scalers.get(args.scaler, args)
    args.y_scaler = Scalers.get(args.y_scaler, args)
    args.real_evaluation_ratio = RealEvaluationRatio(args)

    res_folder = optimize_instance(args = args)
    
    print(res_folder)
    print("\n📊 Running COCO post-processing (HTML will NOT auto-open)...")
    
    subprocess.run([sys.executable, "-m", "cocopp", *[res_folder]])


if __name__ == "__main__":
    main()
