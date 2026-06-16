#!/bin/bash
#PBS -N Train_PFN_V2_Array
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb:scratch_local=80gb:gpu_cap=compute_80
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -J 1-24

set -euo pipefail

PROJECT_HOME="${PROJECT_HOME:-/storage/brno2/home/furdav/CMA-ES}"
DATA_EXPERIMENT="${DATA_EXPERIMENT:-full_run}"
TRAIN_EXPERIMENT="${TRAIN_EXPERIMENT:-full_run_v2}"
SIF="${SIF:-pytorch_env.sif}"

DIMENSION="${DIMENSION:-5}"
SEED="${SEED:-42}"
MAX_TRUE_EVALS="${MAX_TRUE_EVALS:-1000}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-35}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-128}"
TRAIN_LR="${TRAIN_LR:-0.001}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.00001}"
TRAIN_SHUFFLE_BUFFER="${TRAIN_SHUFFLE_BUFFER:-10000}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-10000}"
TRAIN_MAX_RECORDS="${TRAIN_MAX_RECORDS:-0}"

TRAIN_TARGET_MODE="${TRAIN_TARGET_MODE:-softmax_kl}"
TRAIN_SCORE_TRANSFORM="${TRAIN_SCORE_TRANSFORM:-log1p_minmax}"
TRAIN_SOFT_LABEL_TEMPERATURE="${TRAIN_SOFT_LABEL_TEMPERATURE:-0.20}"
TRAIN_CLASS_WEIGHTING="${TRAIN_CLASS_WEIGHTING:-inverse_oracle}"
TRAIN_CLASS_WEIGHT_POWER="${TRAIN_CLASS_WEIGHT_POWER:-0.5}"
TRAIN_CLASS_WEIGHT_CLIP="${TRAIN_CLASS_WEIGHT_CLIP:-10.0}"
TRAIN_ENTROPY_BONUS="${TRAIN_ENTROPY_BONUS:-0.01}"

cd "$SCRATCHDIR"

export TMPDIR="$SCRATCHDIR"
export PIP_CACHE_DIR="$SCRATCHDIR"

cleanup() {
  mkdir -p "$PROJECT_HOME/results/$TRAIN_EXPERIMENT/models"
  cp -r "results/$TRAIN_EXPERIMENT/models/"* "$PROJECT_HOME/results/$TRAIN_EXPERIMENT/models/" 2>/dev/null || true
}

trap cleanup EXIT TERM INT

cp -r "$PROJECT_HOME"/* .

mkdir -p "results/$TRAIN_EXPERIMENT"
for run_dir in "results/$DATA_EXPERIMENT"/f*_i*_d*_s*; do
  if [ -d "$run_dir" ]; then
    ln -s "../$DATA_EXPERIMENT/$(basename "$run_dir")" "results/$TRAIN_EXPERIMENT/$(basename "$run_dir")" 2>/dev/null || true
  fi
done

if ! compgen -G "results/$TRAIN_EXPERIMENT/f*_i*_d*_s*/dataset.jsonl" >/dev/null; then
  echo "No dataset.jsonl files found through results/$TRAIN_EXPERIMENT; falling back to copy."
  cp -r "results/$DATA_EXPERIMENT"/f*_i*_d*_s* "results/$TRAIN_EXPERIMENT/" 2>/dev/null || true
fi

if ! compgen -G "results/$TRAIN_EXPERIMENT/f*_i*_d*_s*/dataset.jsonl" >/dev/null; then
  echo "Missing training dataset. Expected results/$DATA_EXPERIMENT/f*_i*_d*_s*/dataset.jsonl"
  exit 1
fi

singularity exec "$SIF" python -m venv --system-site-packages venv_pfn_v2
singularity exec "$SIF" venv_pfn_v2/bin/pip install --no-cache-dir -e .

singularity exec --nv "$SIF" venv_pfn_v2/bin/python examples/run_coco_experiment.py \
  --experiment_name "$TRAIN_EXPERIMENT" \
  --dimension "$DIMENSION" \
  --seed "$SEED" \
  --max_true_evals "$MAX_TRUE_EVALS" \
  --train_epochs "$TRAIN_EPOCHS" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --train_hidden_dim "$TRAIN_HIDDEN_DIM" \
  --train_lr "$TRAIN_LR" \
  --train_weight_decay "$TRAIN_WEIGHT_DECAY" \
  --train_max_records "$TRAIN_MAX_RECORDS" \
  --train_shuffle_buffer "$TRAIN_SHUFFLE_BUFFER" \
  --train_steps_per_epoch "$TRAIN_STEPS_PER_EPOCH" \
  --train_target_mode "$TRAIN_TARGET_MODE" \
  --train_score_transform "$TRAIN_SCORE_TRANSFORM" \
  --train_soft_label_temperature "$TRAIN_SOFT_LABEL_TEMPERATURE" \
  --train_class_weighting "$TRAIN_CLASS_WEIGHTING" \
  --train_class_weight_power "$TRAIN_CLASS_WEIGHT_POWER" \
  --train_class_weight_clip "$TRAIN_CLASS_WEIGHT_CLIP" \
  --train_entropy_bonus "$TRAIN_ENTROPY_BONUS" \
  --train_max_true_evals "$MAX_TRUE_EVALS" \
  --pfn_v2_features \
  --device cuda \
  --train_decision_model \
  --function_id "$PBS_ARRAY_INDEX"

cleanup
