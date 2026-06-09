#!/bin/bash
#PBS -N Fixed_GP_DTS
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=20gb
#PBS -l walltime=06:00:00
#PBS -J 1-24

set -u

PROJECT_HOME="/storage/brno2/home/furdav/CMA-ES"
EXPERIMENT="fixed_gp_matern_dts"
FIXED_SURROGATE="gp_matern_dts"

cd "$SCRATCHDIR"

export TMPDIR="$SCRATCHDIR"
export PIP_CACHE_DIR="$SCRATCHDIR"

cleanup() {
  mkdir -p "$PROJECT_HOME/results/$EXPERIMENT"
  cp -r "results/$EXPERIMENT/f"* "$PROJECT_HOME/results/$EXPERIMENT/" 2>/dev/null || true

  mkdir -p "$PROJECT_HOME/results/$EXPERIMENT/cocopp"
  cp -r "results/$EXPERIMENT/cocopp/"* "$PROJECT_HOME/results/$EXPERIMENT/cocopp/" 2>/dev/null || true

  mkdir -p "$PROJECT_HOME/results/$EXPERIMENT/exdata"
  cp -r exdata/* "$PROJECT_HOME/results/$EXPERIMENT/exdata/" 2>/dev/null || true
}

trap cleanup EXIT TERM INT

cp -r "$PROJECT_HOME"/* .

singularity exec pytorch_env.sif python -m venv --system-site-packages venv_fixed_eval
singularity exec pytorch_env.sif venv_fixed_eval/bin/pip install --no-cache-dir -e .

singularity exec pytorch_env.sif venv_fixed_eval/bin/python examples/run_coco_experiment.py \
  --experiment_name "$EXPERIMENT" \
  --dimension 5 \
  --instances_start 1 \
  --instances_end 15 \
  --seed 42 \
  --max_generations 10000 \
  --max_true_evals 1000 \
  --pop_size 10 \
  --device cpu \
  --function_id "$PBS_ARRAY_INDEX" \
  --fixed_surrogate "$FIXED_SURROGATE"

cleanup
