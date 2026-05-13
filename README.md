# PFN-driven surrogate selection for CMA-ES

This repository contains a research-grade experimental skeleton for **surrogate-assisted CMA-ES** where a **decision model** (later e.g. PFN-based) selects one surrogate + evolution-control pair at each generation.

## Prerequisites

### Python

You need to have python installed on your system.
It is recommended to use [uv](https://docs.astral.sh/uv/guides/install-python/) or [pyenv](https://github.com/pyenv/pyenv).

### UV

Although the structure allows using whatever build-tool you want, it's better to use [uv](https://github.com/astral-sh/uv).

### Setting up environments

```shell
make install
```

### pre-commit

Consider installing [pre-commit](https://pre-commit.com/) to run checks automatically.

Install pre-commit tool:
```shell
uv tool install pre-commit
```

Install git hooks:
```shell
uv tool run pre-commit install
```

Trigger check manually on all files:
```shell
uv tool run pre-commit run --all-files
```


## Main ideas

There are two modes:

1. **Dataset generation mode** (`generate_dataset=True`)
   - all surrogate bundles are run in parallel on the same CMA-ES generation,
   - each bundle is evaluated by a **one-step counterfactual lookahead**,
   - the real optimizer still advances using only **purely true evaluations**,
   - dataset records are saved continuously for later PFN training.

2. **Training mode** (`train_decision_model=True`)
   - training PFN decision model

3. **Decision mode**
   - the decision model scores all available surrogate bundles,
   - the best bundle is selected,
   - its merged ranking is passed into CMA-ES,
   - generation-level logs are stored.


## Package structure

- `config.py` — experiment configuration dataclasses
- `interfaces.py` — abstract interfaces for surrogates, evolution control, decision model, optimizer backend
- `experiment.py` — main orchestration loop
- `controllers/dataset_generator.py` — dataset generation logic with counterfactual lookahead
- `controllers/decision_controller.py` — surrogate selection logic
- `optimizers/cmaes.py` — cloneable pycma backend
- `storage/` — JSONL logging utilities
- `coco/runner.py` — COCO execution wrapper
- `stubs/` — simple dummy components for smoke tests

## Expected surrogate API

Each surrogate is a class implementing:

```python
predict(history_x, history_y, query_x) -> SurrogatePopulation
```

## Expected evolution-control API

Each evolution control strategy implements:

```python
select_and_evaluate(surrogate_population, objective) -> EvolutionControlResult
```

## Expected decision model API

The PFN decision model later implements:

```python
score(state, surrogate_names) -> SurrogateDecision
```

## CUDA support

```bash
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


## Example

```bash
python examples/run_coco_experiment.py \
  --experiment_name demo \
  --function_id 1 \
  --dimension 5 \
  --instances_start 1 \
  --instances_end 3 \
  --max_generations 20 \
  --max_true_evals 200 \
  --pop_size 10 \
  --generate_dataset
```

### Generating dataset:
```bash
python examples/run_coco_experiment.py --experiment_name demo --generate_dataset --function_id 1
```
- Generated datasets are in files "results/{experiment_name}/f{function_id}_i{instance_id}_d{dimension}_s{seed}/dataset.jsonl"

### Training PFN decision model:
```bash
python examples/run_coco_experiment.py --experiment_name demo --train_decision_model --function_id 1
```
- Trained models are in files "results/{experiment_name}/models/decision_model_dim{dimension}_heldout_f{function_id}.pt"
- Training streams dataset records from JSONL files instead of loading the full dataset into RAM.
- For large datasets, use `--train_steps_per_epoch` to cap epoch length and `--train_shuffle_buffer` to control bounded streaming shuffle memory.

### Testing PFN decision strategy:
```bash
python examples/run_coco_experiment.py --experiment_name demo --function_id 1
```
- Testing logs are in files "results/{experiment_name}/f{function_id}_i{instance_id}_d{dimension}_s{seed}/generation_logs.jsonl"
- COCO standartized results are available in files "results/{experiment_name}/cocopp/{experiment_name}_bbob_dim{dimension}_f{function_id}-{experiment_order}_{timestamp}/index1.html"

## TODO
- check again overall correctness of algorithm
- choose appropriate surrogate models and evolution strategies
- run testing on dimension 5
- maybe (if neccesary) - adding out memory loading datasets during PFN training
- maybe (after the first tests) - adding fine-tuning training of PFN during optimization run

