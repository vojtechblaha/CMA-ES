#!/bin/bash
#PBS -N Train_PFN_Array
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb:scratch_local=60gb:gpu_cap=compute_80
#PBS -q gpu
#PBS -l walltime=24:00:00
# Run parallel training for all 24 functions
#PBS -J 1-24

cd $SCRATCHDIR

export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR

cp -r /storage/brno2/home/furdav/CMA-ES/* .

# Create venv and install
singularity exec pytorch_env.sif python -m venv --system-site-packages venv_gpu
singularity exec pytorch_env.sif venv_gpu/bin/pip install --no-cache-dir -e .

# Run the full 25-epoch training loop!
singularity exec --nv pytorch_env.sif venv_gpu/bin/python examples/run_coco_experiment.py \
  --experiment_name full_run \
  --dimension 5 \
  --seed 42 \
  --train_epochs 25 \
  --train_hidden_dim 128 \
  --train_max_records 0 \
  --train_shuffle_buffer 10000 \
  --train_steps_per_epoch 10000 \
  --device cuda \
  --train_decision_model \
  --function_id $PBS_ARRAY_INDEX

# Save the models
mkdir -p /storage/brno2/home/furdav/CMA-ES/results/full_run/models/
cp -r results/full_run/models/* /storage/brno2/home/furdav/CMA-ES/results/full_run/models/
