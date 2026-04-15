from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from ..interfaces import OptimizerBackend
from ..types import Array

try:
    import cma
except ImportError as exc:  # pragma: no cover - dependency expected in real use
    raise ImportError(
        "The 'cma' package is required for PyCMAOptimizerBackend. Install it via 'pip install cma'."
    ) from exc


class PyCMAOptimizerBackend(OptimizerBackend):
    """Minimal cloneable wrapper around pycma's ask/tell interface."""

    def __init__(
        self,
        x0: Array,
        sigma0: float,
        population_size: int,
        seed: int,
        inopts: Dict[str, object] | None = None,
    ):
        options = dict(inopts or {})
        options.setdefault("popsize", population_size)
        options.setdefault("seed", seed)
        self._es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float), sigma0, options)

    def ask(self) -> Array:
        return np.asarray(self._es.ask(), dtype=float)

    def tell(self, x: Array, y: Array) -> None:
        self._es.tell(np.asarray(x, dtype=float), np.asarray(y, dtype=float).tolist())

    def should_stop(self) -> bool:
        return bool(self._es.stop())

    def clone(self) -> "PyCMAOptimizerBackend":
        cloned = object.__new__(PyCMAOptimizerBackend)
        cloned._es = copy.deepcopy(self._es)
        return cloned

    def get_state(self) -> Dict[str, object]:
        covariance = np.asarray(self._es.sm.C, dtype=float)
        return {
            "mean": np.asarray(self._es.mean, dtype=float).copy(),
            "sigma": float(self._es.sigma),
            "covariance": covariance.copy(),
            "population_size": int(self._es.popsize),
            "generation": int(self._es.countiter),
            "evaluations": int(self._es.countevals),
            "dimension": int(self._es.N),
        }

    @property
    def population_size(self) -> int:
        return int(self._es.popsize)

    @property
    def dimension(self) -> int:
        return int(self._es.N)
