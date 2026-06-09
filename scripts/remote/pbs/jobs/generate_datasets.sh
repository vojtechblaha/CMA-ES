#!/bin/bash
#PBS -N CMAES_Dataset_Gen
#PBS -l select=1:ncpus=2:mem=8gb:scratch_local=20gb
#PBS -l walltime=24:00:00
# Spawns 24 parallel jobs for the 24 functions
#PBS -J 1-24

# 1. Load standard Python module
module add python310-modules-gcc

# 2. Move to the fast SSD and copy your code from brno2
cd $SCRATCHDIR
cp -r /storage/brno2/home/furdav/CMA-ES/* .

# 3. Install uv for lightning-fast setup
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 4. Install the project using your Makefile
make install

# 5. Activate the newly created environment
source .venv/bin/activate

# 6. Install CPU PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 7. Run the full pipeline!
# $PBS_ARRAY_INDEX automatically injects 1 through 24
python examples/run_coco_experiment.py \
  --experiment_name full_run \
  --dimension 5 \
  --instances_start 1 \
  --instances_end 20 \
  --seed 42 \
  --max_generations 1000 \
  --max_true_evals 10000 \
  --pop_size 10 \
  --generate_dataset \
  --function_id $PBS_ARRAY_INDEX

# 8. Copy everything back to your brno2 home directory safely
cp -r results /storage/brno2/home/furdav/CMA-ES/
