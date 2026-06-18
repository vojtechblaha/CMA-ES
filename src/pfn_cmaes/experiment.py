from __future__ import annotations

from typing import Callable

import numpy as np

from .config import ExperimentConfig
from .controllers.dataset_generator import DatasetGenerationController
from .controllers.decision_controller import DecisionController
from .controllers.surrogate_bundle import SurrogateBundle
from .history import EvaluationHistory
from .metrics import CounterfactualImprovementMetric
from .storage.run_logger import RunLogger
from .types import GenerationLog, RunSummary


class PFNSurrogateExperiment:
    """Top-level orchestration for PFN-driven surrogate-assisted CMA-ES."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.bundles = [SurrogateBundle.from_spec(spec) for spec in config.surrogate_specs]
        self.decision_model = config.decision_model
        self.metric = CounterfactualImprovementMetric(
            eps=config.dataset.metric_eps,
            clip_value=config.dataset.metric_clip_value,
        )

    def run(self, objective: Callable[[np.ndarray], float]) -> RunSummary:
        run_cfg = self.config.run
        run_id = f"f{run_cfg.function_id}_i{run_cfg.instance_id}_d{run_cfg.dimension}_s{run_cfg.seed}"
        logger = RunLogger(self.config.logging, run_cfg.experiment_name, run_id)
        optimizer = self.config.optimizer_backend_cls(**self.config.optimizer_kwargs)
        history = EvaluationHistory()
        generation_logs: list[GenerationLog] = []

        dataset_controller = DatasetGenerationController()
        decision_controller = DecisionController(decision_model=self.decision_model)

        generation_index = 0
        while generation_index < run_cfg.max_generations and history.num_evaluations < run_cfg.max_true_evals:
            if optimizer.should_stop():
                break

            candidate_x = optimizer.ask()
            incumbent_before = history.incumbent_y
            run_meta = {
                "run_id": run_id,
                "function_id": run_cfg.function_id,
                "instance_id": run_cfg.instance_id,
                "dimension": run_cfg.dimension,
            }

            if self.config.dataset.generate_dataset:
                artifacts, dataset_record, real_population = dataset_controller.evaluate_all_bundles(
                    bundles=self.bundles,
                    optimizer=optimizer,
                    history=history,
                    candidate_x=candidate_x,
                    objective=objective,
                    run_metadata=run_meta,
                    generation_index=generation_index,
                )
                logger.log_dataset_record(dataset_record)
                optimizer.tell(real_population.x, real_population.y)
                history.append(real_population)
                log = GenerationLog(
                    generation_index=generation_index,
                    mode="generate_dataset",
                    surrogate_name=None,
                    incumbent_y_before=incumbent_before,
                    incumbent_y_after=history.incumbent_y,
                    num_true_evals=len(real_population.y),
                    num_surrogate_evals=len(candidate_x) * len(self.bundles),
                    dataset_scores={name: art.score for name, art in artifacts.items()},
                    metadata={
                        "optimizer_state_after": optimizer.get_state(),
                        "optimizer_state_before": optimizer.get_state(),
                        "counterfactual_lookahead_best_y": {
                            name: art.lookahead_best_y for name, art in artifacts.items()
                        },
                    },
                )
            else:
                chosen_bundle, decision = decision_controller.choose_bundle(
                    bundles=self.bundles,
                    history=history,
                    candidate_x=candidate_x,
                    optimizer_state=optimizer.get_state(),
                    generation_index=generation_index,
                )
                optimizer_state_before = optimizer.get_state()
                if hasattr(chosen_bundle.surrogate, "set_optimizer_state"):
                    chosen_bundle.surrogate.set_optimizer_state(optimizer_state_before)
                surrogate_population = chosen_bundle.surrogate.predict(history.x, history.y, candidate_x)
                ec_result = chosen_bundle.evolution_control.select_and_evaluate(surrogate_population, objective)
                optimizer.tell(ec_result.merged_ranking.x, ec_result.merged_ranking.y)
                history.append(ec_result.true_evaluated)
                true_ratio = len(ec_result.true_evaluated.y) / max(len(candidate_x), 1)
                improvement = self.metric(
                    previous_best=incumbent_before,
                    next_best=history.incumbent_y,
                )
                decision_controller.update_statistics(
                    chosen_bundle_name=chosen_bundle.name,
                    decision=decision,
                    true_ratio=true_ratio,
                    improvement=improvement,
                )
                log = GenerationLog(
                    generation_index=generation_index,
                    mode="decision",
                    surrogate_name=chosen_bundle.name,
                    incumbent_y_before=incumbent_before,
                    incumbent_y_after=history.incumbent_y,
                    num_true_evals=len(ec_result.true_evaluated.y),
                    num_surrogate_evals=len(ec_result.surrogate_evaluated.y_pred),
                    decision_goodness=decision.goodness,
                    metadata={
                        "decision_metadata": decision.metadata,
                        "ec_metadata": ec_result.metadata,
                        "optimizer_state_after": optimizer.get_state(),
                        "optimizer_state_before": optimizer_state_before,
                    },
                )

            generation_logs.append(log)
            logger.log_generation(log)

            if run_cfg.target_f is not None and history.incumbent_y <= run_cfg.target_f:
                break

            generation_index += 1

        summary = RunSummary(
            run_id=run_id,
            function_id=run_cfg.function_id,
            instance_id=run_cfg.instance_id,
            dimension=run_cfg.dimension,
            generations=len(generation_logs),
            true_evaluations=history.num_evaluations,
            best_y=history.incumbent_y,
            solved=run_cfg.target_f is not None and history.incumbent_y <= run_cfg.target_f,
            logs=generation_logs,
            metadata={
                "experiment_name": run_cfg.experiment_name,
                "optimizer_state_final": optimizer.get_state(),
            },
        )
        logger.log_summary(summary)
        return summary
