import argparse

import os
import sys
import time
import gc
import subprocess

import cocoex  # COCO benchmarking

from optimization import Optimization

# ----------------------------
# Per-instance Optimizer
# ----------------------------
def optimize_instance(algorithm, function_id, dimension, instances, iterations, pop_size, result_folder):
    suite = cocoex.Suite("bbob", "", f"dimensions:{dimension} function_indices:{function_id} instance_indices:1-{instances}")
    observer = cocoex.Observer("bbob", f"result_folder: {result_folder}")
    
    for problem in suite:
        if problem.dimension == dimension:
            problem.observe_with(observer)

            Optimization.get_method(algorithm)(problem, dimension, iterations, pop_size)

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
    parser.add_argument("--algorithm", type=str, choices=[
        "s-cma-es", "dts-cma-es", "mf-gp-ucb", "es-cma-es", "ga-galapagos", "cma-es", "gp-cma-es",
        "saacm-es", "acm-es", "lcc-cma-es", "meta-es", "cma-es-led", "llama-es", "rbf-cma-es",
        "svm-cma-es", "lq-cma-es", "lmm-cma-es", "ensemble-cma-es", "transformer-cma-es", "hnn-rl-cma-es"
    ], required=True)
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--instances", type=int, default=15)
    parser.add_argument("--experiment_name", type=str, required=True)
    
    args = parser.parse_args()


    Optimization.initialize()

    result_folders = []
    if True:#try:
        res_folder = optimize_instance(
            algorithm=args.algorithm,
            function_id=args.function_id,
            dimension=args.dimension,
            instances=args.instances,
            iterations=args.iterations,
            pop_size=args.pop_size,
            result_folder=args.experiment_name
        )
        result_folders.append(res_folder)
    #except Exception as e:
    #    print(f"❌ Error occured: {e}")

    print(result_folders)
    # ✅ Post-process results (HTML not opened)
    print("\n📊 Running COCO post-processing (HTML will NOT auto-open)...")
    subprocess.run([sys.executable, "-m", "cocopp", *result_folders])


if __name__ == "__main__":
    main()
