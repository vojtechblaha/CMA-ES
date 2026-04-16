from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..history import EvaluationHistory
from ..interfaces import ObjectiveFunction, OptimizerBackend
from ..types import DatasetRecord, EvolutionControlResult, EvaluatedPopulation
from .surrogate_bundle import SurrogateBundle


@dataclass(slots=True)
class BundleGenerationArtifacts:
    result: EvolutionControlResult
    score: float
    lookahead_best_y: float
    lookahead_population: EvaluatedPopulation


class DatasetGenerationController:
    """Runs all surrogate bundles in parallel on the same generation.

    For each surrogate bundle, we:
    1. predict on the current population,
    2. apply evolution control,
    3. update a clone of the optimizer with the merged ranking,
    4. ask the next generation from the clone,
    5. truly evaluate that lookahead population,
    6. score the bundle by one-step normalized log-improvement.

    The real optimizer trajectory remains purely true-evaluated.
    """

    def __init__(self):
        pass

    def evaluate_all_bundles(
        self,
        bundles: list[SurrogateBundle],
        optimizer: OptimizerBackend,
        history: EvaluationHistory,
        candidate_x: np.ndarray,
        objective: ObjectiveFunction,
        run_metadata: Dict[str, int | str],
        generation_index: int,
    ) -> tuple[dict[str, BundleGenerationArtifacts], DatasetRecord, EvaluatedPopulation]:
        history_x, history_y = history.x, history.y
        incumbent_y = history.incumbent_y
        incumbent_x = history.incumbent_x.tolist()

        artifacts: dict[str, BundleGenerationArtifacts] = {}
        scores: dict[str, float] = {}

        for bundle in bundles:
            surrogate_population = bundle.surrogate.predict(history_x, history_y, candidate_x)
            ec_result = bundle.evolution_control.select_and_evaluate(surrogate_population, objective)

            optimizer_clone = optimizer.clone()
            optimizer_clone.tell(ec_result.merged_ranking.x, ec_result.merged_ranking.y)
            lookahead_x = optimizer_clone.ask()
            lookahead_y = np.asarray([objective(x) for x in lookahead_x], dtype=float)
            lookahead_population = EvaluatedPopulation(x=lookahead_x, y=lookahead_y).sorted()
            
            lookahead_best_y = float(lookahead_population.best_y)

            # Number of expensive objective evaluations consumed by the bundle
            # in the current generation.
            num_true_evals = int(len(ec_result.true_evaluated.x))
            if num_true_evals <= 0:
                num_true_evals = 1

            improvement = max(incumbent_y - lookahead_best_y, 0.0)
            score = improvement / float(num_true_evals)


            artifacts[bundle.name] = BundleGenerationArtifacts(
                result=ec_result,
                score=score,
                lookahead_best_y=lookahead_population.best_y,
                lookahead_population=lookahead_population,
            )
            scores[bundle.name] = score

        true_y = np.asarray([objective(x) for x in candidate_x], dtype=float)
        real_population = EvaluatedPopulation(x=candidate_x, y=true_y).sorted()

        max_history_for_record = 512  # nebo z configu

        if len(history_y) > max_history_for_record:
            record_history_x = history_x[-max_history_for_record :].copy()
            record_history_y = history_y[-max_history_for_record :].copy()
        else:
            record_history_x = history_x.copy()
            record_history_y = history_y.copy()

        record = DatasetRecord(
            run_id=str(run_metadata["run_id"]),
            function_id=int(run_metadata["function_id"]),
            instance_id=int(run_metadata["instance_id"]),
            dimension=int(run_metadata["dimension"]),
            generation_index=generation_index,
            history_x=record_history_x,
            history_y=record_history_y,
            candidate_x=candidate_x.copy(),
            incumbent_x=incumbent_x,
            incumbent_y=float(incumbent_y),
            surrogate_scores=scores,
            optimizer_state=optimizer.get_state(),
            metadata={
                "lookahead_best_y": {name: art.lookahead_best_y for name, art in artifacts.items()},
                "real_generation_best_y": float(real_population.best_y),
            },
        )
        return artifacts, record, real_population
