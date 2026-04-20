from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

Array = np.ndarray


@dataclass(slots=True)
class EvaluatedPopulation:
    """Population together with true objective values."""

    x: Array
    y: Array

    def sorted(self) -> "EvaluatedPopulation":
        order = np.argsort(self.y)
        return EvaluatedPopulation(x=self.x[order], y=self.y[order])

    @property
    def best_x(self) -> Array:
        return self.sorted().x[0]

    @property
    def best_y(self) -> float:
        return float(self.sorted().y[0])

    def as_training_data(self) -> tuple[Array, Array]:
        return self.x, self.y


@dataclass(slots=True)
class SurrogatePopulation:
    """Population together with surrogate predictions and optional uncertainty."""

    x: Array
    y_pred: Array
    uncertainty: Array | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y_pred):
            raise ValueError(f"x and y_pred must have the same length, got {len(self.x)} and {len(self.y_pred)}")
        if self.uncertainty is not None and len(self.uncertainty) != len(self.y_pred):
            raise ValueError(
                f"uncertainty and y_pred must have the same length, got {len(self.uncertainty)} and {len(self.y_pred)}"
            )

    def sorted(self) -> "SurrogatePopulation":
        order = np.argsort(self.y_pred)

        sorted_uncertainty = None
        if self.uncertainty is not None:
            sorted_uncertainty = self.uncertainty[order]

        return SurrogatePopulation(
            x=self.x[order],
            y_pred=self.y_pred[order],
            uncertainty=sorted_uncertainty,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class EvolutionControlResult:
    """Result returned by an evolution control strategy."""

    true_evaluated: EvaluatedPopulation
    surrogate_evaluated: SurrogatePopulation
    merged_ranking: EvaluatedPopulation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationState:
    """State visible to the decision model before choosing a surrogate."""

    generation_index: int
    evaluated_history_x: Array
    evaluated_history_y: Array
    candidate_x: Array
    incumbent_x: Array
    incumbent_y: float
    optimizer_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SurrogateDecision:
    """Decision model output over all available surrogate specs."""

    goodness: Dict[str, float]
    chosen_surrogate_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetRecord:
    """Single record stored while generating the dataset for the decision model."""

    run_id: str
    function_id: int
    instance_id: int
    dimension: int
    generation_index: int
    history_x: Array
    history_y: Array
    candidate_x: Array
    incumbent_x: Array
    incumbent_y: float
    surrogate_scores: Dict[str, float]
    optimizer_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationLog:
    generation_index: int
    mode: str
    surrogate_name: Optional[str]
    incumbent_y_before: float
    incumbent_y_after: float
    num_true_evals: int
    num_surrogate_evals: int
    decision_goodness: Dict[str, float] = field(default_factory=dict)
    dataset_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    function_id: int
    instance_id: int
    dimension: int
    generations: int
    true_evaluations: int
    best_y: float
    solved: bool
    logs: List[GenerationLog]
    metadata: Dict[str, Any] = field(default_factory=dict)
