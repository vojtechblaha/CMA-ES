1) CMA-ES (Plain CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name CMA-ES

2) CMA-ES-LED (CMA-ES with Latent Encoding Decoding) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name CMA-ES-LED --led_cma True

3) S-CMA-ES (Surrogate-assisted CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name S-CMA-ES --models GaussianProcessRegressorMAT --scaler StandardScaler --real_evaluation_ratio static-1.0,0.0 --train_only_real True

4) DTS-CMA-ES (Double-Trust-Region CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name DTS-CMA-ES --models GaussianProcessRegressorSE --scaler DTSScaler --y_scaler StandardScaler --real_evaluation_ratio static-0.5 --tss knn --check_flatness True

5) aCMA-ES (Active CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name aCMA-ES --active_cma True

6) aCMA-ES-LED (Active CMA-ES-LED Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name aCMA-ES-LED --active_cma True --led_cma True

7) ACM-ES (Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name ACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio static-1.0,0.0  --train_only_real True

8) SAACM-ES (Surrogate-Assisted Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name SAACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio adaptive-saacm  --train_only_real True

9) SAACM-ES-K (SAACM-ES with models ensemble) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name SAACM-ES-K --models GaussianProcessRegressorSAACM,SVR,RBFModel --ensemble_type best --real_evaluation_ratio adaptive-saacm  --train_only_real True

10) CMA-SAO (RBF surrogate CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name CMA-SAO --model RBFModel --scaler SAOScaler --sigma 0.5 --train_only_real True --real_evaluation_ratio adaptive

11) LQ-CMA-ES (Quadratic regression CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name LQ-CMA-ES --models LQModel --sigma 0.5 --real_evaluation_ratio static-0.1 --train_only_real True --check_train_kendalltau True

12) LMM-CMA-ES (Local meta-model CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name LMM-CMA-ES --models LocalQuadraticModel --sigma 0.3 --real_evaluation_ratio static-0.0 --train_only_real True --fit_until unchanged_ranking

13) NLMM-CMA-ES (New Local meta-model CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name NLMM-CMA-ES --models LocalQuadraticModel --sigma 0.3 --real_evaluation_ratio static-0.0 --train_only_real True --fit_until unchanged_set

14) ES-CMA-ES (Elite-driven Surrogate-assisted CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name ES-CMA-ES --models GaussianProcessRegressorRBF --real_evaluation_ratio static-0.5 --prediction_transform ilcb --cma_only_real True --train_only_real True

15) IPOP-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-CMA-ES --pop_increase ipop

16) BIPOP-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-CMA-ES --pop_increase bipop

17) IPOP-aCMA-ES (IPOP Active CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-aCMA-ES --active_cma True --pop_increase ipop

18) BIPOP-aCMA-ES (IPOP Active CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-aCMA-ES --active_cma True --pop_increase bipop

19) NIPOP-aCMA-ES (IPOP Active CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name NIPOP-aCMA-ES --active_cma True --pop_increase nipop

20) NBIPOP-aCMA-ES (IPOP Active CMA-ES Optimization) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name NBIPOP-aCMA-ES --active_cma True --pop_increase nbipop

21) IPOP-SAACM-ES (Surrogate-Assisted Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-SAACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase ipop

22) BIPOP-SAACM-ES (Surrogate-Assisted Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-SAACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase bipop

23) IPOP-SAACM-ES-K (SAACM-ES with models ensemble) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-SAACM-ES-K --models GaussianProcessRegressorSAACM,SVR,RBFModel --ensemble_type best --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase ipop

24) BIPOP-SAACM-ES-K (SAACM-ES with models ensemble) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-SAACM-ES-K --models GaussianProcessRegressorSAACM,SVR,RBFModel --ensemble_type best --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase bipop

25) IPOP-elitism-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-elitism-CMA-ES --pop_increase ipop --elitism True

26) BIPOP-elitism-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-elitism-CMA-ES --pop_increase bipop --elitism True

27) IPOP!-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP!-CMA-ES --pop_increase ipop --center_injection True

28) IPOP!-elitism-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP!-elitism-CMA-ES --pop_increase ipop --center_injection True --elitism True

29) IPOP-elitism-SAACM-ES (Surrogate-Assisted Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-elitism-SAACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase ipop --elitism True

30) BIPOP-elitism-SAACM-ES (Surrogate-Assisted Adaptive CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name BIPOP-elitism-SAACM-ES --models GaussianProcessRegressorSAACM --real_evaluation_ratio adaptive-saacm  --train_only_real True --pop_increase bipop --elitism True

31) SVM-CMA-ES (SVM surrogate CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name SVM-CMA-ES --model SVC --use_pair_model True --real_evaluation_ratio static-1.0,0.5 --train_only_real True

32) LM-CMA-ES (Limited-Memory CMA-ES) - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name LM-CMA-ES --cma_type lm

33) MO-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name MO-CMA-ES --cma_type mo

34) IPOP-MO-CMA-ES - verified:
python main.py --function_id 1 --dimension 5 --max_evals_per_dim 40 --pop_size 10 --instances 1 --experiment_name IPOP-MO-CMA-ES --cma_type mo --pop_increase ipop



3) MF-GP-UCB (Multi-Fidelity GP):
python main.py --function_id 1 --dimension 5 --iterations 50 --pop_size 10 --instances 1 --experiment_name MF-GP-UCB --model GaussianProcessRegressor --scaler StandardScaler --real_evaluation_ratio static-0.1 --train_only_real True --prediction_transform ucb --ask_by around_best

9) LCC-CMA-ES (Limited Covariance CMA-ES):
python main.py --function_id 1 --dimension 5 --iterations 50 --pop_size 10 --instances 1 --experiment_name LCC-CMA-ES --sigma 0.5



13) P-SEP-LMM-CMA-ES - na problémy se separabilní vyhodnocovací funkcí, takze nepouzitelne: