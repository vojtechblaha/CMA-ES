#!/bin/bash
#PBS -N Eval_PFN_Array
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=20gb
#PBS -l walltime=06:00:00
#PBS -J 1-24

set -u

PROJECT_HOME="/storage/brno2/home/furdav/CMA-ES"
TRAIN_EXPERIMENT="full_run"
EVAL_EXPERIMENT="full_run_phase3"

cd "$SCRATCHDIR"

export TMPDIR="$SCRATCHDIR"
export PIP_CACHE_DIR="$SCRATCHDIR"

cleanup() {
  mkdir -p "$PROJECT_HOME/results/$EVAL_EXPERIMENT"
  cp -r "results/$EVAL_EXPERIMENT/f"* "$PROJECT_HOME/results/$EVAL_EXPERIMENT/" 2>/dev/null || true

  mkdir -p "$PROJECT_HOME/results/$EVAL_EXPERIMENT/cocopp"
  cp -r "results/$EVAL_EXPERIMENT/cocopp/"* "$PROJECT_HOME/results/$EVAL_EXPERIMENT/cocopp/" 2>/dev/null || true

  mkdir -p "$PROJECT_HOME/results/$EVAL_EXPERIMENT/exdata"
  cp -r exdata/* "$PROJECT_HOME/results/$EVAL_EXPERIMENT/exdata/" 2>/dev/null || true
}

trap cleanup EXIT TERM INT

cp -r "$PROJECT_HOME"/* .

mkdir -p "results/$EVAL_EXPERIMENT"
cp -r "results/$TRAIN_EXPERIMENT/models" "results/$EVAL_EXPERIMENT/models"

CHECKPOINT="results/$EVAL_EXPERIMENT/models/decision_model_dim5_heldout_f${PBS_ARRAY_INDEX}.pt"
if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT"
  exit 1
fi

singularity exec pytorch_env.sif python -m venv --system-site-packages venv_eval
singularity exec pytorch_env.sif venv_eval/bin/pip install --no-cache-dir -e .

singularity exec pytorch_env.sif venv_eval/bin/python examples/run_coco_experiment.py \
  --experiment_name "$EVAL_EXPERIMENT" \
  --dimension 5 \
  --instances_start 1 \
  --instances_end 15 \
  --seed 42 \
  --max_generations 10000 \
  --max_true_evals 1000 \
  --pop_size 10 \
  --device cpu \
  --function_id "$PBS_ARRAY_INDEX"

cleanup
