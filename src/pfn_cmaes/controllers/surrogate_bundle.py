from __future__ import annotations

from dataclasses import dataclass

from ..config import SurrogateSpec
from ..interfaces import EvolutionControl, SurrogateModel


@dataclass(slots=True)
class SurrogateBundle:
    name: str
    surrogate: SurrogateModel
    evolution_control: EvolutionControl

    @classmethod
    def from_spec(cls, spec: SurrogateSpec) -> "SurrogateBundle":
        return cls(
            name=spec.name,
            surrogate=spec.surrogate_cls(**spec.surrogate_kwargs),
            evolution_control=spec.evolution_control_cls(**spec.evolution_control_kwargs),
        )
