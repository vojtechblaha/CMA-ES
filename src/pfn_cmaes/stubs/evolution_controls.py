from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import EvolutionControl, ObjectiveFunction
from ..types import EvaluatedPopulation, EvolutionControlResult, SurrogatePopulation


def _empty_surrogate_population(dimension: int) -> SurrogatePopulation:
    return SurrogatePopulation(
        x=np.empty((0, dimension), dtype=float),
        y_pred=np.empty((0,), dtype=float),
    )


def _stack_merge_true_and_surrogate(
    true_x: np.ndarray,
    true_y: np.ndarray,
    surrogate_x: np.ndarray,
    surrogate_y: np.ndarray,
) -> EvaluatedPopulation:
    if len(surrogate_x) == 0:
        return EvaluatedPopulation(x=true_x.copy(), y=true_y.copy()).sorted()
    if len(true_x) == 0:
        return EvaluatedPopulation(x=surrogate_x.copy(), y=surrogate_y.copy()).sorted()

    merged_x = np.vstack([true_x, surrogate_x])
    merged_y = np.concatenate([true_y, surrogate_y])
    return EvaluatedPopulation(x=merged_x, y=merged_y).sorted()


class EvaluateAll(EvolutionControl):
    """Pure true-evaluated baseline."""

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        y_true = np.asarray([objective(x) for x in surrogate_population.x], dtype=float)
        true_pop = EvaluatedPopulation(x=surrogate_population.x, y=y_true).sorted()
        empty_surrogate = _empty_surrogate_population(surrogate_population.x.shape[1])

        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=empty_surrogate,
            merged_ranking=true_pop,
            metadata={
                "ec_type": "evaluate_all",
                "fraction": 1.0,
                "num_true": len(y_true),
                "num_surrogate": 0,
            },
        )


class EvaluateTopFraction(EvolutionControl):
    """
    Evaluate the top predicted fraction truly and keep surrogate values for the rest.
    """

    def __init__(self, fraction: float = 0.5):
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must satisfy 0 < fraction <= 1")
        self.fraction = float(fraction)

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        ordered = surrogate_population.sorted()
        n = len(ordered.y_pred)
        k = max(1, int(np.ceil(self.fraction * n)))

        true_x = ordered.x[:k]
        true_y = np.asarray([objective(x) for x in true_x], dtype=float)
        true_pop = EvaluatedPopulation(x=true_x, y=true_y).sorted()

        surrogate_x = ordered.x[k:]
        surrogate_y = ordered.y_pred[k:]
        surrogate_pop = SurrogatePopulation(x=surrogate_x, y_pred=surrogate_y)

        merged = _stack_merge_true_and_surrogate(
            true_x=true_x,
            true_y=true_y,
            surrogate_x=surrogate_x,
            surrogate_y=surrogate_y,
        )

        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=surrogate_pop,
            merged_ranking=merged,
            metadata={
                "ec_type": "evaluate_top_fraction",
                "fraction": self.fraction,
                "num_true": int(k),
                "num_surrogate": int(n - k),
            },
        )


class TopFractionPlusUncertaintyControl(EvolutionControl):
    """
    Evaluate:
    - top predicted fraction
    - plus a fraction of most uncertain points, if uncertainty is provided

    If uncertainty is unavailable, this falls back to EvaluateTopFraction.
    """

    def __init__(
        self,
        top_fraction: float = 0.3,
        uncertainty_fraction: float = 0.2,
    ) -> None:
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError("top_fraction must satisfy 0 < top_fraction <= 1")
        if not (0.0 <= uncertainty_fraction <= 1.0):
            raise ValueError("uncertainty_fraction must satisfy 0 <= uncertainty_fraction <= 1")

        self.top_fraction = float(top_fraction)
        self.uncertainty_fraction = float(uncertainty_fraction)
        self.iter = 0

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        n = len(surrogate_population.y_pred)
        ordered = surrogate_population.sorted()

        k_top = max(1, int(np.ceil(self.top_fraction * n)))
        top_indices = list(range(k_top))

        uncertainty = getattr(ordered, "uncertainty", None)
        if uncertainty is None:
            chosen_indices = sorted(set(top_indices))
        else:
            uncertainty = np.asarray(uncertainty, dtype=float)
            k_unc = int(np.ceil(self.uncertainty_fraction * n))
            unc_order = np.argsort(-uncertainty)
            unc_indices = unc_order[:k_unc].tolist()
            chosen_indices = sorted(set(top_indices + unc_indices))

        chosen_mask = np.zeros(n, dtype=bool)
        chosen_mask[chosen_indices] = True

        true_x = ordered.x[chosen_mask]
        true_y = np.asarray([objective(x) for x in true_x], dtype=float)
        true_pop = EvaluatedPopulation(x=true_x, y=true_y).sorted()

        surrogate_x = ordered.x[~chosen_mask]
        surrogate_y = ordered.y_pred[~chosen_mask]
        surrogate_pop = SurrogatePopulation(x=surrogate_x, y_pred=surrogate_y)

        merged = _stack_merge_true_and_surrogate(
            true_x=true_x,
            true_y=true_y,
            surrogate_x=surrogate_x,
            surrogate_y=surrogate_y,
        )

        self.iter += 1

        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=surrogate_pop,
            merged_ranking=merged,
            metadata={
                "ec_type": "top_fraction_plus_uncertainty",
                "top_fraction": self.top_fraction,
                "uncertainty_fraction": self.uncertainty_fraction,
                "num_true": int(chosen_mask.sum()),
                "num_surrogate": int((~chosen_mask).sum()),
            },
        )


@dataclass(slots=True)
class AdaptiveModelLifelengthControl(EvolutionControl):
    """
    Simplified s*ACM-ES-style control:
    - maintain a surrogate lifetime in generations
    - when surrogate agreement is good, lifetime increases
    - when surrogate agreement is poor, lifetime decreases
    - on refresh generations, evaluate everything truly

    This class is stateful across generations.
    """

    initial_lifelength: int = 1
    max_lifelength: int = 5
    min_lifelength: int = 1
    top_fraction: float = 0.5
    good_agreement_threshold: float = 0.7
    bad_agreement_threshold: float = 0.3

    current_lifelength: int = field(init=False)
    generations_since_refresh: int = field(init=False, default=0)
    last_agreement: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.current_lifelength = int(self.initial_lifelength)

    @staticmethod
    def _rank_agreement(
        predicted: np.ndarray,
        realized: np.ndarray,
    ) -> float:
        """
        Cheap monotone proxy for ranking agreement.
        """
        if len(predicted) <= 1:
            return 1.0
        pred_rank = np.argsort(np.argsort(predicted))
        real_rank = np.argsort(np.argsort(realized))
        corr = np.corrcoef(pred_rank, real_rank)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float((corr + 1.0) / 2.0)

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        n = len(surrogate_population.y_pred)

        refresh_now = self.generations_since_refresh >= self.current_lifelength - 1
        if refresh_now:
            y_true = np.asarray([objective(x) for x in surrogate_population.x], dtype=float)
            agreement = self._rank_agreement(surrogate_population.y_pred, y_true)
            self.last_agreement = agreement

            if agreement >= self.good_agreement_threshold:
                self.current_lifelength = min(self.current_lifelength + 1, self.max_lifelength)
            elif agreement <= self.bad_agreement_threshold:
                self.current_lifelength = max(self.current_lifelength - 1, self.min_lifelength)

            self.generations_since_refresh = 0

            true_pop = EvaluatedPopulation(x=surrogate_population.x, y=y_true).sorted()
            empty_surrogate = _empty_surrogate_population(surrogate_population.x.shape[1])

            return EvolutionControlResult(
                true_evaluated=true_pop,
                surrogate_evaluated=empty_surrogate,
                merged_ranking=true_pop,
                metadata={
                    "ec_type": "adaptive_model_lifelength",
                    "refresh": True,
                    "agreement": agreement,
                    "lifelength": self.current_lifelength,
                    "num_true": n,
                    "num_surrogate": 0,
                },
            )

        ordered = surrogate_population.sorted()
        k = max(1, int(np.ceil(self.top_fraction * n)))

        true_x = ordered.x[:k]
        true_y = np.asarray([objective(x) for x in true_x], dtype=float)
        true_pop = EvaluatedPopulation(x=true_x, y=true_y).sorted()

        surrogate_x = ordered.x[k:]
        surrogate_y = ordered.y_pred[k:]
        surrogate_pop = SurrogatePopulation(x=surrogate_x, y_pred=surrogate_y)

        merged = _stack_merge_true_and_surrogate(
            true_x=true_x,
            true_y=true_y,
            surrogate_x=surrogate_x,
            surrogate_y=surrogate_y,
        )

        self.generations_since_refresh += 1

        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=surrogate_pop,
            merged_ranking=merged,
            metadata={
                "ec_type": "adaptive_model_lifelength",
                "refresh": False,
                "agreement": self.last_agreement,
                "lifelength": self.current_lifelength,
                "num_true": int(k),
                "num_surrogate": int(n - k),
            },
        )


@dataclass(slots=True)
class DoublyTrainedControl(EvolutionControl):
    """
    Simplified DTS-CMA-ES-style control:
    - periodically force full true re-evaluation
    - otherwise use surrogate ranking with partial true validation
    - adapt exploitation depth using observed ranking agreement
    """

    refresh_interval: int = 5
    partial_fraction: float = 0.3
    exploit_multiplier_if_good: float = 1.2
    exploit_multiplier_if_bad: float = 0.8
    agreement_threshold: float = 0.65

    generation_counter: int = field(init=False, default=0)
    current_fraction: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_fraction = float(self.partial_fraction)

    @staticmethod
    def _rank_agreement(predicted: np.ndarray, realized: np.ndarray) -> float:
        if len(predicted) <= 1:
            return 1.0
        pred_rank = np.argsort(np.argsort(predicted))
        real_rank = np.argsort(np.argsort(realized))
        corr = np.corrcoef(pred_rank, real_rank)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float((corr + 1.0) / 2.0)

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        self.generation_counter += 1
        n = len(surrogate_population.y_pred)

        if (self.generation_counter - 1) % self.refresh_interval == 0:
            y_true = np.asarray([objective(x) for x in surrogate_population.x], dtype=float)
            true_pop = EvaluatedPopulation(x=surrogate_population.x, y=y_true).sorted()
            empty_surrogate = _empty_surrogate_population(surrogate_population.x.shape[1])

            return EvolutionControlResult(
                true_evaluated=true_pop,
                surrogate_evaluated=empty_surrogate,
                merged_ranking=true_pop,
                metadata={
                    "ec_type": "doubly_trained",
                    "refresh": True,
                    "fraction": 1.0,
                    "num_true": n,
                    "num_surrogate": 0,
                },
            )

        ordered = surrogate_population.sorted()
        k = max(1, int(np.ceil(self.current_fraction * n)))

        true_x = ordered.x[:k]
        true_y = np.asarray([objective(x) for x in true_x], dtype=float)

        agreement = self._rank_agreement(ordered.y_pred[:k], true_y)
        if agreement >= self.agreement_threshold:
            self.current_fraction = max(
                1.0 / n,
                min(1.0, self.current_fraction * self.exploit_multiplier_if_good),
            )
        else:
            self.current_fraction = max(
                1.0 / n,
                min(1.0, self.current_fraction * self.exploit_multiplier_if_bad),
            )

        true_pop = EvaluatedPopulation(x=true_x, y=true_y).sorted()

        surrogate_x = ordered.x[k:]
        surrogate_y = ordered.y_pred[k:]
        surrogate_pop = SurrogatePopulation(x=surrogate_x, y_pred=surrogate_y)

        merged = _stack_merge_true_and_surrogate(
            true_x=true_x,
            true_y=true_y,
            surrogate_x=surrogate_x,
            surrogate_y=surrogate_y,
        )

        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=surrogate_pop,
            merged_ranking=merged,
            metadata={
                "ec_type": "doubly_trained",
                "refresh": False,
                "agreement": agreement,
                "fraction": self.current_fraction,
                "num_true": int(k),
                "num_surrogate": int(n - k),
            },
        )
