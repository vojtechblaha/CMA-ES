from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict

from .types import (
    Array,
    EvolutionControlResult,
    GenerationState,
    SurrogateDecision,
    SurrogatePopulation,
)

ObjectiveFunction = Callable[[Array], float]


class SurrogateModel(ABC):
    """Abstract interface for a surrogate model.

    The model receives all truly evaluated points from the current run and predicts
    objective values for the current candidate population.
    """

    @abstractmethod
    def predict(
        self,
        history_x: Array,
        history_y: Array,
        query_x: Array,
    ) -> SurrogatePopulation:
        raise NotImplementedError


class EvolutionControl(ABC):
    """Abstract evolution control interface.

    The strategy decides which candidate points should be evaluated truly, performs
    the expensive evaluations, and returns the merged ranking used by the optimizer.
    """

    @abstractmethod
    def select_and_evaluate(
        self,
        surrogate_population: SurrogatePopulation,
        objective: ObjectiveFunction,
    ) -> EvolutionControlResult:
        raise NotImplementedError


class DecisionModel(ABC):
    """Abstract decision model used to choose the surrogate at each generation."""

    @abstractmethod
    def score(self, state: GenerationState, surrogate_names: list[str]) -> SurrogateDecision:
        raise NotImplementedError


class OptimizerBackend(ABC):
    """Ask/tell optimizer backend that supports cloning for counterfactual rollouts."""

    @abstractmethod
    def ask(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def tell(self, x: Array, y: Array) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_stop(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> "OptimizerBackend":
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        raise NotImplementedError

    @property
    @abstractmethod
    def population_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
