from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class CounterfactualImprovementMetric:
    """Stable target value for dataset generation.

    The score is based on one-step lookahead true improvement after applying a
    surrogate-specific ask/tell update to a clone of the optimizer.

    We use normalized log-improvement:
        score = clip(log(prev_best + eps) - log(next_best + eps), -clip, clip)

    This is numerically stable across functions with different scales, focuses on
    multiplicative progress, and is common in black-box optimization analyses where
    log-target reductions are more meaningful than raw differences.
    """

    eps: float = 1e-12
    clip_value: float = 5.0

    def __call__(self, previous_best: float, next_best: float) -> float:
        prev = max(float(previous_best), self.eps)
        nxt = max(float(next_best), self.eps)
        score = math.log(prev) - math.log(nxt)
        return max(-self.clip_value, min(self.clip_value, score))
