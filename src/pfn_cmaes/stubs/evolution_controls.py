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
        regime = "static", # static / adaptive / manual
    ) -> None:
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError("top_fraction must satisfy 0 < top_fraction <= 1")
        if not (0.0 <= uncertainty_fraction <= 1.0):
            raise ValueError("uncertainty_fraction must satisfy 0 <= uncertainty_fraction <= 1")

        self.top_fraction = float(top_fraction)
        self.uncertainty_fraction = float(uncertainty_fraction)
        self.true_evals = 0
        self.regime = regime

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        n = len(surrogate_population.y_pred)
        ordered = surrogate_population.sorted()

        top_frac = self.top_fraction
        if self.regime == "adaptive":
            # Example adaptive scheme: increase top fraction as we do more true evals
            top_frac = min(1.0, max(self.top_fraction, 0.5 * np.log10(self.true_evals / max(1, n))))
            print("Adaptive top fraction:", top_frac)
        elif self.regime == "manual":
            if self.true_evals > 4000:
                top_frac = 0.9
            elif self.true_evals > 2000:
                top_frac = 0.8
            elif self.true_evals > 1000:
                top_frac = 0.7
            elif self.true_evals > 400:
                top_frac = 0.6
            elif self.true_evals > 300:
                top_frac = 0.5
            elif self.true_evals > 200:
                top_frac = 0.4
            else:
                top_frac = 0.3
            top_frac = min(1.0, max(self.top_fraction, top_frac))       
            print("Manual top fraction:", top_frac)

        k_top = max(1, int(np.ceil(top_frac * n)))
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
        self.true_evals += len(true_y)

        surrogate_x = ordered.x[~chosen_mask]
        surrogate_y = ordered.y_pred[~chosen_mask]
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


class TrustRegionAdaptiveRankControl(EvolutionControl):
    """
    Covariance-aware evolution control for the TabPFN surrogate.

    It combines three safeguards:
    1. evaluate the predicted top fraction truly,
    2. force true evaluation outside the surrogate trust region, using distances
       exported by TabPFNSurrogate metadata (CMA-whitened Mahalanobis distance),
    3. adapt the future true-evaluation fraction based on rank agreement on the
       truly evaluated subset.

    Lower objective values are assumed to be better.
    """

    def __init__(
        self,
        top_fraction: float = 0.5,
        min_top_fraction: float = 0.2,
        max_top_fraction: float = 1.0,
        adaptation_step: float = 0.1,
        good_agreement_threshold: float = 0.75,
        bad_agreement_threshold: float = 0.55,
        trust_region_radius: float | None = None,
        min_validation_points: int = 2,
        uncertainty_fraction: float = 0.0,
        random_state: int = 0,
    ) -> None:
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError("top_fraction must satisfy 0 < top_fraction <= 1")
        if not (0.0 < min_top_fraction <= 1.0):
            raise ValueError("min_top_fraction must satisfy 0 < min_top_fraction <= 1")
        if not (0.0 < max_top_fraction <= 1.0):
            raise ValueError("max_top_fraction must satisfy 0 < max_top_fraction <= 1")
        if min_top_fraction > max_top_fraction:
            raise ValueError("min_top_fraction must be <= max_top_fraction")
        if not (0.0 <= uncertainty_fraction <= 1.0):
            raise ValueError("uncertainty_fraction must satisfy 0 <= uncertainty_fraction <= 1")

        self.initial_top_fraction = float(top_fraction)
        self.current_top_fraction = float(np.clip(top_fraction, min_top_fraction, max_top_fraction))
        self.min_top_fraction = float(min_top_fraction)
        self.max_top_fraction = float(max_top_fraction)
        self.adaptation_step = float(adaptation_step)
        self.good_agreement_threshold = float(good_agreement_threshold)
        self.bad_agreement_threshold = float(bad_agreement_threshold)
        self.trust_region_radius = None if trust_region_radius is None else float(trust_region_radius)
        self.min_validation_points = int(max(0, min_validation_points))
        self.uncertainty_fraction = float(uncertainty_fraction)
        self.rng = np.random.default_rng(random_state)
        self.last_rank_agreement: float | None = None
        self.generation_index = 0

    @staticmethod
    def _rank_agreement(predicted: np.ndarray, realized: np.ndarray) -> float:
        if len(predicted) <= 1:
            return 1.0
        pred_rank = np.argsort(np.argsort(np.asarray(predicted, dtype=float), kind="stable"), kind="stable")
        real_rank = np.argsort(np.argsort(np.asarray(realized, dtype=float), kind="stable"), kind="stable")
        corr = np.corrcoef(pred_rank, real_rank)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(corr)

    def _adapt(self, agreement: float | None) -> None:
        if agreement is None or not np.isfinite(agreement):
            return
        if agreement >= self.good_agreement_threshold:
            self.current_top_fraction = max(
                self.min_top_fraction,
                self.current_top_fraction - self.adaptation_step,
            )
        elif agreement <= self.bad_agreement_threshold:
            self.current_top_fraction = min(
                self.max_top_fraction,
                self.current_top_fraction + self.adaptation_step,
            )

    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        n = len(surrogate_population.y_pred)
        if n == 0:
            dim = int(surrogate_population.x.shape[1]) if surrogate_population.x.ndim == 2 else 0
            empty_true = EvaluatedPopulation(
                x=np.empty((0, dim), dtype=float),
                y=np.empty((0,), dtype=float),
            )
            empty_sur = _empty_surrogate_population(dim)
            return EvolutionControlResult(empty_true, empty_sur, empty_true, metadata={"ec_type": "trust_region_adaptive_rank", "empty": True})

        x = np.asarray(surrogate_population.x, dtype=float)
        y_pred = np.asarray(surrogate_population.y_pred, dtype=float).reshape(-1)
        order = np.argsort(y_pred, kind="stable")
        x_ord = x[order]
        y_pred_ord = y_pred[order]

        chosen_mask = np.zeros(n, dtype=bool)
        k_top = max(1, int(np.ceil(self.current_top_fraction * n)))
        chosen_mask[:k_top] = True

        uncertainty = getattr(surrogate_population, "uncertainty", None)
        if uncertainty is not None and self.uncertainty_fraction > 0.0:
            uncertainty_ord = np.asarray(uncertainty, dtype=float).reshape(-1)[order]
            k_unc = int(np.ceil(self.uncertainty_fraction * n))
            if k_unc > 0:
                chosen_mask[np.argsort(-uncertainty_ord, kind="stable")[:k_unc]] = True

        trust_distances = None
        raw_distances = surrogate_population.metadata.get("query_trust_distances") if surrogate_population.metadata else None
        if raw_distances is not None:
            trust_distances = np.asarray(raw_distances, dtype=float).reshape(-1)[order]

        num_outside_trust = 0
        if self.trust_region_radius is not None and trust_distances is not None and len(trust_distances) == n:
            outside = trust_distances > self.trust_region_radius
            num_outside_trust = int(np.sum(outside))
            chosen_mask |= outside

        if self.min_validation_points > 0:
            available = np.flatnonzero(~chosen_mask)
            need = max(0, self.min_validation_points - int(np.sum(chosen_mask)))
            if need > 0 and len(available) > 0:
                val = self.rng.choice(available, size=min(need, len(available)), replace=False)
                chosen_mask[val] = True

        true_x = x_ord[chosen_mask]
        true_y = np.asarray([objective(candidate) for candidate in true_x], dtype=float)
        true_pop = EvaluatedPopulation(x=true_x, y=true_y).sorted()

        surrogate_x = x_ord[~chosen_mask]
        surrogate_y = y_pred_ord[~chosen_mask]
        surrogate_pop = SurrogatePopulation(x=surrogate_x, y_pred=surrogate_y)

        # Compute agreement in the ordered coordinate frame, before sorting true_pop.
        agreement = None
        if len(true_y) >= 2:
            agreement = self._rank_agreement(y_pred_ord[chosen_mask], true_y)
            self.last_rank_agreement = agreement
            self._adapt(agreement)

        merged = _stack_merge_true_and_surrogate(
            true_x=true_x,
            true_y=true_y,
            surrogate_x=surrogate_x,
            surrogate_y=surrogate_y,
        )

        metadata = {
            "ec_type": "trust_region_adaptive_rank",
            "top_fraction_used": float(k_top / n),
            "current_top_fraction_next": float(self.current_top_fraction),
            "min_top_fraction": float(self.min_top_fraction),
            "max_top_fraction": float(self.max_top_fraction),
            "rank_agreement": None if agreement is None else float(agreement),
            "trust_region_radius": self.trust_region_radius,
            "num_outside_trust": int(num_outside_trust),
            "num_true": int(chosen_mask.sum()),
            "num_surrogate": int((~chosen_mask).sum()),
            "coordinate_mode": surrogate_population.metadata.get("coordinate_mode") if surrogate_population.metadata else None,
        }
        if trust_distances is not None and len(trust_distances) == n:
            metadata.update({
                "trust_distance_min": float(np.min(trust_distances)),
                "trust_distance_max": float(np.max(trust_distances)),
                "trust_distance_mean": float(np.mean(trust_distances)),
            })

        self.generation_index += 1
        return EvolutionControlResult(
            true_evaluated=true_pop,
            surrogate_evaluated=surrogate_pop,
            merged_ranking=merged,
            metadata=metadata,
        )
