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
or
```shell
pip install -e .
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

Download TabPFN:
```shell
$env:TABPFN_TOKEN="tvuj_api_key"
python -c "from tabpfn import TabPFNRegressor; TabPFNRegressor().fit([[0.0],[1.0]], [0.0, 1.0])"
```
Api key here: https://ux.priorlabs.ai/account/licenses

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

### Generating evaluation graphs:
```bash
python coco_eval_graph.py exdata 5 1 24 demo --ref-years 2020 2021 --cache-dir coco_cache 
```
- Generate evalution graphs for exdata folder for dimension 5 for functions 1-24, as reference algoritms it uses all algorithm from years 2020 and 2021, as cache folder it uses coco_cache.
- Warning: Everytime it uses for evaluation the last runs from exdata.
- Comparing with all algorithms:
```bash
python coco_eval_graph.py exdata-meta 5 1 24 --ref-tags 2009 2010 2012 2013 2014-others 2015-CEC 2015-GECCO 2016 2017 2017-others 2018 2018-others 2019 2020 2021 2022 2023 --cache-dir coco_cache --coco-alg "CMA-ES-2019_Hansen,CMA-ES-2019 (Hansen),#4C72B0" --coco-alg "BIPOP-saACM-k_loshchilov_noiseless,BIPOP-saACM-k (Loshchilov),#55A868" --coco-alg "lq-CMA-ES_Hansen,lq-CMA-ES (Hansen),#C44E52" --coco-alg "DTS-CMA-ES_005-2pop_v26_1model_Bajer,DTS-CMA-ES (Bajer),#D81B60" --coco-alg "RF5-CMAES_Bajer_2013instances,RF5-CMA-ES (Bajer),#8172B2" --out coco_plots_test/D5 --max_evals_per_dim 500 --local-exp "pfns_sig_10_dim_5_test,TabPFN-CMA-ES-D5,#FF8C00"
```

### Generating evaluation tables:
```bash
python coco_eval_table.py exdata 5 1 24 demo --ref-years 2020 2021 --cache-dir coco_cache --evals 50 100 200 500 1000
```
- Generate evalution tables for exdata folder for dimension 5 for functions 1-24, as reference algoritms it uses all algorithm from years 2020 and 2021, as cache folder it uses coco_cache, it computes metrics for budgets 50, ..., 1000.
- Warning: Everytime it uses for evaluation the last runs from exdata.
- Comparing with all algorithms:
```bash
python coco_eval_table.py exdata-meta 5 1 24 --evals_per_dim 100 200 300 400 500 --ref-tags 2009 2010 2012 2013 2014-others 2015-CEC 2015-GECCO 2016 2017 2017-others 2018 2018-others 2019 2020 2021 2022 2023 --cache-dir coco_cache --coco-alg "CMA-ES-2019_Hansen,CMA-ES-2019 (Hansen),#4C72B0" --coco-alg "BIPOP-saACM-k_loshchilov_noiseless,BIPOP-saACM-k (Loshchilov),#55A868" --coco-alg "lq-CMA-ES_Hansen,lq-CMA-ES (Hansen),#C44E52" --coco-alg "DTS-CMA-ES_005-2pop_v26_1model_Bajer,DTS-CMA-ES (Bajer),#D81B60" --coco-alg "RF5-CMAES_Bajer_2013instances,RF5-CMA-ES (Bajer),#8172B2" --out coco_tables_test/D5 --local-exp "pfns_sig_10_dim_5_test,Our approach,#F9A825"
```

## TODO
- check again overall correctness of algorithm
- choose appropriate surrogate models and evolution strategies
- run testing on dimension 5
- maybe (if neccesary) - adding out memory loading datasets during PFN training
- maybe (after the first tests) - adding fine-tuning training of PFN during optimization run

## Other commands
module load python/3.11
pip install -e . --no-build-isolation
pip install -e .

mkdir -p /storage/brno2/home/blahavo/hf_cache
mkdir -p /storage/brno2/home/blahavo/tmp
mkdir -p /storage/brno2/home/blahavo/torch_cache

export HF_HOME=/storage/brno2/home/blahavo/hf_cache
export HUGGINGFACE_HUB_CACHE=/storage/brno2/home/blahavo/hf_cache/hub
export HF_HUB_CACHE=/storage/brno2/home/blahavo/hf_cache/hub
export XDG_CACHE_HOME=/storage/brno2/home/blahavo/.cache
export TORCH_HOME=/storage/brno2/home/blahavo/torch_cache
export TMPDIR=/storage/brno2/home/blahavo/tmp

python -c "from tabpfn import TabPFNRegressor; TabPFNRegressor().fit([[0.0],[1.0]],[0.0,1.0])"

qsub -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=32gb -I

scp -r blahavo@zenith.metacentrum.cz:/storage/brno2/home/blahavo/CMA-ES/exdata C:\CMA-ES\exdata-meta