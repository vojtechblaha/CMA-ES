from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Array, EvaluatedPopulation


@dataclass
class EvaluationHistory:
    """Container for all truly evaluated points within one optimization run."""

    x_batches: list[Array] = field(default_factory=list)
    y_batches: list[Array] = field(default_factory=list)

    def append(self, population: EvaluatedPopulation) -> None:
        if population.x.size == 0:
            return
        self.x_batches.append(np.asarray(population.x, dtype=float))
        self.y_batches.append(np.asarray(population.y, dtype=float))

    @property
    def x(self) -> Array:
        if not self.x_batches:
            return np.empty((0, 0), dtype=float)
        return np.vstack(self.x_batches)

    @property
    def y(self) -> Array:
        if not self.y_batches:
            return np.empty((0,), dtype=float)
        return np.concatenate(self.y_batches)

    @property
    def num_evaluations(self) -> int:
        return int(sum(batch.shape[0] for batch in self.x_batches))

    @property
    def incumbent_y(self) -> float:
        if not self.y_batches:
            return float("inf")
        return float(np.min(self.y))

    @property
    def incumbent_x(self) -> Array:
        if not self.y_batches:
            return np.empty((0,), dtype=float)
        idx = int(np.argmin(self.y))
        return self.x[idx]

    def export_last_n_generations(self, n_points: int | None = None) -> tuple[Array, Array]:
        x, y = self.x, self.y
        if x.size == 0:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        if n_points is None or n_points >= len(y):
            return x, y
        return x[-n_points:], y[-n_points:]
