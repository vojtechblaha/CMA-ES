from __future__ import annotations

from dataclasses import dataclass

import cocoex
import numpy as np


@dataclass(slots=True)
class CocoProblemWrapper:
    problem: cocoex.Problem

    def __call__(self, x: np.ndarray) -> float:
        return float(self.problem(x))

    @property
    def dimension(self) -> int:
        return int(self.problem.dimension)

    @property
    def final_target(self) -> float:
        # COCO exposes best final target via final_target_hit, but for a generic optimizer
        # we only pass the objective value. The actual "solved" decision can stay external.
        return float(getattr(self.problem, "best_observed_fvalue1", np.inf))

    def observe_with(self, observer: cocoex.Observer) -> "CocoProblemWrapper":
        self.problem.observe_with(observer)
        return self
