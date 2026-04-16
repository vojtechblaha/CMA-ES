:: Dataset generation
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 1000 --max_true_evals 10000? --pop_size 16 --generate_dataset --function_id 1
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 1000 --max_true_evals 10000? --pop_size 16 --generate_dataset --function_id ...
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 1000 --max_true_evals 10000? --pop_size 16 --generate_dataset --function_id 24

:: PFN training
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --seed 42 --train_epochs 25? --train_hidden_dim 128? --train_max_records 0? --device cuda --train_decision_model --function_id 1
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --seed 42 --train_epochs 25? --train_hidden_dim 128? --train_max_records 0? --device cuda --train_decision_model --function_id ...
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --seed 42 --train_epochs 25? --train_hidden_dim 128? --train_max_records 0? --device cuda --train_decision_model --function_id 24

:: PFN strategy testing
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 10000 --max_true_evals 100000? --pop_size 16 --function_id 1
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 10000 --max_true_evals 100000? --pop_size 16 --function_id ...
python examples/run_coco_experiment.py --experiment_name demo --dimension 5 --instances_start 1 --instances_end 20? --seed 42 --max_generations 10000 --max_true_evals 100000? --pop_size 16 --function_id 24

:: Info: argument values with ? has to be tested to choose the most right one