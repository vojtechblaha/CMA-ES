#!/bin/bash
#PBS -N PFN_Oracle_Agreement
#PBS -l select=1:ncpus=2:mem=16gb:scratch_local=20gb
#PBS -l walltime=04:00:00
#PBS -J 1-24

set -u

PROJECT_HOME="/storage/brno2/home/furdav/CMA-ES"
TRAIN_EXPERIMENT="full_run"
OUT_NAME="oracle_agreement"
SIF="$PROJECT_HOME/pytorch_env.sif"

cd "$SCRATCHDIR"

export TMPDIR="$SCRATCHDIR"
export PIP_CACHE_DIR="$SCRATCHDIR/pip_cache"

cleanup() {
  mkdir -p "$PROJECT_HOME/results/$TRAIN_EXPERIMENT/$OUT_NAME"
  cp -r "results/$TRAIN_EXPERIMENT/$OUT_NAME/f${PBS_ARRAY_INDEX}" \
    "$PROJECT_HOME/results/$TRAIN_EXPERIMENT/$OUT_NAME/" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

mkdir -p CMA-ES
cd CMA-ES

# Copy only code and the dataset/model experiment needed for this offline evaluation.
cp -r "$PROJECT_HOME/examples" .
cp -r "$PROJECT_HOME/src" .
cp -r "$PROJECT_HOME/scripts" .
cp "$PROJECT_HOME/pyproject.toml" .
cp "$PROJECT_HOME/README.md" . 2>/dev/null || true

mkdir -p "results"
mkdir -p "results/$TRAIN_EXPERIMENT"
cp -r "$PROJECT_HOME/results/$TRAIN_EXPERIMENT/models" "results/$TRAIN_EXPERIMENT/models"
cp -r "$PROJECT_HOME/results/$TRAIN_EXPERIMENT"/f${PBS_ARRAY_INDEX}_i*_d5_s42 "results/$TRAIN_EXPERIMENT/"

# Create venv. Use --no-deps to avoid downloading/reinstalling CUDA Torch wheels.
singularity exec "$SIF" python -m venv --system-site-packages venv_oracle_eval
singularity exec "$SIF" venv_oracle_eval/bin/pip install --no-cache-dir --no-deps -e .

singularity exec "$SIF" venv_oracle_eval/bin/python scripts/evaluate_pfn_oracle_agreement.py \
  --experiment-root "results/$TRAIN_EXPERIMENT" \
  --models-dir "results/$TRAIN_EXPERIMENT/models" \
  --out-dir "results/$TRAIN_EXPERIMENT/$OUT_NAME" \
  --dimension 5 \
  --device cpu \
  --function-id "$PBS_ARRAY_INDEX" \
  --generation-bin-width 50 \
  --write-mistakes 200

cleanup
