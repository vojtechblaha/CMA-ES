from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any

import numpy as np

from ..interfaces import SurrogateModel
from ..types import SurrogatePopulation


SelectionMode = Literal["recent", "nearest", "hybrid", "all"]
PredictionMode = Literal["mean", "median"]
CoordinateMode = Literal["raw", "standardized", "cma_whitened"]


def _as_2d_float32(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def _as_1d_float32(y: np.ndarray, name: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)


def _fallback_population(
    query_x: np.ndarray,
    value: float,
    reason: str,
) -> SurrogatePopulation:
    return SurrogatePopulation(
        x=np.asarray(query_x, dtype=float),
        y_pred=np.full(len(query_x), float(value), dtype=float),
        uncertainty=np.zeros(len(query_x), dtype=float),
        metadata={
            "surrogate_type": "tabpfn_regressor",
            "fallback": True,
            "reason": reason,
        },
    )


def _nearest_indices(train_x: np.ndarray, query_x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty(0, dtype=int)

    if len(train_x) <= k:
        return np.arange(len(train_x), dtype=int)

    center = np.mean(query_x, axis=0)
    distances = np.sum((train_x - center[None, :]) ** 2, axis=1)

    idx = np.argpartition(distances, k - 1)[:k]
    idx = idx[np.argsort(distances[idx], kind="stable")]
    return idx.astype(int)


def _select_subset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    max_train_size: int,
    max_train_size_ratio: float | None,
    selection_mode: SelectionMode,
    recent_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(train_x)

    if max_train_size_ratio is not None and 0.0 < max_train_size_ratio < 1.0:
        max_train_size = max(1, int(round(n * max_train_size_ratio)))

    if max_train_size <= 0:
        raise ValueError("max_train_size must be positive.")

    if n <= max_train_size or selection_mode == "all":
        return train_x, train_y

    mode = selection_mode.lower()

    if mode == "recent":
        idx = np.arange(n - max_train_size, n, dtype=int)

    elif mode == "nearest":
        idx = _nearest_indices(train_x, query_x, max_train_size)

    elif mode == "hybrid":
        n_recent = int(round(max_train_size * recent_fraction))
        n_recent = max(0, min(n_recent, max_train_size))

        recent_idx = (
            np.arange(n - n_recent, n, dtype=int)
            if n_recent > 0
            else np.empty(0, dtype=int)
        )

        remaining_mask = np.ones(n, dtype=bool)
        remaining_mask[recent_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)

        n_nearest = max_train_size - len(recent_idx)

        if n_nearest > 0 and len(remaining_idx) > 0:
            local_nearest = _nearest_indices(
                train_x[remaining_idx],
                query_x,
                min(n_nearest, len(remaining_idx)),
            )
            nearest_idx = remaining_idx[local_nearest]
        else:
            nearest_idx = np.empty(0, dtype=int)

        idx = np.concatenate([nearest_idx, recent_idx])
        idx = np.unique(idx)

        if len(idx) > max_train_size:
            idx = idx[-max_train_size:]

    else:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. "
            "Use one of: 'all', 'recent', 'nearest', 'hybrid'."
        )

    return train_x[idx], train_y[idx]


def _safe_eigendecomposition(covariance: np.ndarray, min_eigenvalue: float) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, float(min_eigenvalue))
    return eigenvalues, eigenvectors


@dataclass(slots=True)
class TabPFNSurrogate(SurrogateModel):
    """
    Pretrained TabPFN regression surrogate for CMA-ES.

    New options added for ill-conditioned functions:
    - coordinate_mode="cma_whitened" maps x to z=C^(-1/2)(x-m)/sigma using the
      current CMA-ES state before selecting the local training set and calling TabPFN.
    - trust-region distances in the same z-space are exported in metadata for
      evolution controls that want to avoid surrogate extrapolation.
    """

    min_train_size: int = 5
    max_train_size: int = 1000
    max_train_size_ratio: float | None = None
    selection_mode: SelectionMode = "hybrid"
    recent_fraction: float = 0.35

    target_mode: str = "reg"

    device: str = "auto"
    random_state: int = 0
    n_estimators: int = 8
    prediction_mode: PredictionMode = "mean"

    normalize_y: bool = True
    min_y_std: float = 1e-12

    return_uncertainty: bool = True
    quantiles: tuple[float, float] = (0.16, 0.84)

    fallback_to_incumbent: bool = True
    raise_on_error: bool = True

    # Coordinate handling for covariance-aware TabPFN-CMA-ES.
    coordinate_mode: CoordinateMode = "raw"
    covariance_jitter: float = 1e-12
    min_sigma: float = 1e-12
    standardize_x_eps: float = 1e-12

    tabpfn_kwargs: dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, repr=False, default=None)
    _optimizer_state: dict[str, Any] | None = field(init=False, repr=False, default=None)
    _last_transform_metadata: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_train_size < 1:
            raise ValueError("min_train_size must be >= 1.")

        if self.max_train_size < self.min_train_size:
            raise ValueError("max_train_size must be >= min_train_size.")

        if not 0.0 <= self.recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1].")

        if len(self.quantiles) != 2 or not self.quantiles[0] < self.quantiles[1]:
            raise ValueError("quantiles must be a tuple like (0.16, 0.84).")

        if self.coordinate_mode not in ("raw", "standardized", "cma_whitened"):
            raise ValueError("coordinate_mode must be one of: raw, standardized, cma_whitened")

        self._model = self._create_model()

    def set_optimizer_state(self, optimizer_state: dict[str, Any] | None) -> None:
        """Called by the experiment loop before predict()."""
        self._optimizer_state = dict(optimizer_state or {})

    def _create_model(self):
        from tabpfn import TabPFNRegressor

        kwargs: dict[str, Any] = dict(self.tabpfn_kwargs)

        kwargs.setdefault("random_state", self.random_state)
        kwargs.setdefault("n_estimators", self.n_estimators)

        if self.device != "auto":
            kwargs.setdefault("device", self.device)

        try:
            return TabPFNRegressor(**kwargs)
        except TypeError:
            kwargs.pop("n_estimators", None)
            try:
                return TabPFNRegressor(**kwargs)
            except TypeError:
                kwargs.pop("device", None)
                kwargs.pop("random_state", None)
                return TabPFNRegressor(**kwargs)

    def _fallback_value(self, train_y: np.ndarray) -> float:
        if len(train_y) == 0:
            return 0.0

        if self.fallback_to_incumbent:
            return float(np.min(train_y))

        return float(np.mean(train_y))

    def _transform_inputs(
        self,
        train_x: np.ndarray,
        query_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return transformed train/query inputs and query trust-region distances."""
        self._last_transform_metadata = {"coordinate_mode": self.coordinate_mode}
        dim = int(query_x.shape[1])

        if self.coordinate_mode == "raw":
            query_dist = np.linalg.norm(query_x - np.mean(query_x, axis=0, keepdims=True), axis=1)
            return train_x, query_x, query_dist.astype(float)

        if self.coordinate_mode == "standardized":
            if len(train_x) > 0:
                center = np.mean(train_x, axis=0)
                scale = np.std(train_x, axis=0)
            else:
                center = np.mean(query_x, axis=0)
                scale = np.std(query_x, axis=0)
            scale = np.maximum(scale, self.standardize_x_eps)
            train_z = (train_x - center[None, :]) / scale[None, :]
            query_z = (query_x - center[None, :]) / scale[None, :]
            query_dist = np.linalg.norm(query_z, axis=1)
            self._last_transform_metadata.update({
                "x_center_source": "history" if len(train_x) > 0 else "query",
                "x_scale_min": float(np.min(scale)),
                "x_scale_max": float(np.max(scale)),
            })
            return train_z.astype(np.float32), query_z.astype(np.float32), query_dist.astype(float)

        # CMA-whitened coordinates: z = C^{-1/2} (x - m) / sigma.
        state = self._optimizer_state or {}
        mean = np.asarray(state.get("mean", np.mean(query_x, axis=0)), dtype=float).reshape(-1)
        covariance = np.asarray(state.get("covariance", np.eye(dim)), dtype=float)
        sigma = max(float(state.get("sigma", 1.0)), self.min_sigma)

        if mean.size != dim:
            mean = np.mean(query_x, axis=0)
        if covariance.shape != (dim, dim):
            covariance = np.eye(dim, dtype=float)

        eigenvalues, eigenvectors = _safe_eigendecomposition(covariance, self.covariance_jitter)
        inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        train_z = ((train_x - mean[None, :]) / sigma) @ inv_sqrt.T
        query_z = ((query_x - mean[None, :]) / sigma) @ inv_sqrt.T
        query_dist = np.linalg.norm(query_z, axis=1)

        self._last_transform_metadata.update({
            "sigma": float(sigma),
            "cov_eig_min": float(np.min(eigenvalues)),
            "cov_eig_max": float(np.max(eigenvalues)),
            "cov_condition": float(np.max(eigenvalues) / max(np.min(eigenvalues), self.covariance_jitter)),
            "query_mahalanobis_min": float(np.min(query_dist)) if len(query_dist) else 0.0,
            "query_mahalanobis_max": float(np.max(query_dist)) if len(query_dist) else 0.0,
        })
        return train_z.astype(np.float32), query_z.astype(np.float32), query_dist.astype(float)

    def _prepare_training_data(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_x = _as_2d_float32(history_x, "history_x")
        train_y = _as_1d_float32(history_y, "history_y")
        query_x = _as_2d_float32(query_x, "query_x")

        if len(train_x) != len(train_y):
            raise ValueError(
                f"history_x and history_y must have same length, "
                f"got {len(train_x)} and {len(train_y)}."
            )

        if train_x.shape[1] != query_x.shape[1]:
            raise ValueError(
                f"history_x and query_x must have same dimension, "
                f"got {train_x.shape[1]} and {query_x.shape[1]}."
            )

        train_x_transformed, query_x_transformed, query_distances = self._transform_inputs(train_x, query_x)

        train_x_selected, train_y_selected = _select_subset(
            train_x=train_x_transformed,
            train_y=train_y,
            query_x=query_x_transformed,
            max_train_size=self.max_train_size,
            max_train_size_ratio=self.max_train_size_ratio,
            selection_mode=self.selection_mode,
            recent_fraction=self.recent_fraction,
        )

        return train_x_selected, train_y_selected, query_x_transformed, query_distances

    def _normalize_targets(self, train_y: np.ndarray) -> tuple[np.ndarray, float, float]:
        if self.target_mode == "rank":
            order = np.argsort(train_y, kind="stable")
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(len(train_y), dtype=np.float32)

            if len(train_y) > 1:
                fit_y = ranks / float(len(train_y) - 1)
            else:
                fit_y = np.zeros_like(ranks, dtype=np.float32)

            return fit_y.astype(np.float32), 0.0, 1.0

        if not self.normalize_y:
            return train_y.astype(np.float32), 0.0, 1.0

        y_mean = float(np.mean(train_y))
        y_std = max(float(np.std(train_y)), self.min_y_std)
        fit_y = ((train_y - y_mean) / y_std).astype(np.float32)
        return fit_y, y_mean, y_std

    def _predict_mean(
        self,
        train_x: np.ndarray,
        fit_y: np.ndarray,
        query_x: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> np.ndarray:
        self._model.fit(train_x, fit_y)

        try:
            pred_scaled = self._model.predict(
                query_x,
                output_type=self.prediction_mode,
            )
        except TypeError:
            pred_scaled = self._model.predict(query_x)

        pred_scaled = np.asarray(pred_scaled, dtype=float).reshape(-1)

        if len(pred_scaled) != len(query_x):
            raise RuntimeError(
                f"TabPFN returned {len(pred_scaled)} predictions "
                f"for {len(query_x)} query points."
            )

        if self.target_mode == "rank":
            return pred_scaled

        return pred_scaled * y_std + y_mean

    def _predict_uncertainty(
        self,
        query_x: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> np.ndarray:
        if self.target_mode == "rank":
            y_mean = 0.0
            y_std = 1.0

        if not self.return_uncertainty:
            return np.zeros(len(query_x), dtype=float)

        try:
            q = self._model.predict(
                query_x,
                output_type="quantiles",
                quantiles=list(self.quantiles),
            )

            q_low, q_high = q
            q_low = np.asarray(q_low, dtype=float).reshape(-1) * y_std + y_mean
            q_high = np.asarray(q_high, dtype=float).reshape(-1) * y_std + y_mean

            return np.maximum(0.0, 0.5 * (q_high - q_low))

        except Exception:
            return np.zeros(len(query_x), dtype=float)

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        query_x_original = np.asarray(query_x, dtype=np.float32)

        if query_x_original.ndim != 2:
            raise ValueError(f"query_x must have shape [Q, D], got {query_x_original.shape}")

        if len(query_x_original) == 0:
            return SurrogatePopulation(
                x=query_x_original.astype(float),
                y_pred=np.zeros(0, dtype=float),
                uncertainty=np.zeros(0, dtype=float),
                metadata={
                    "surrogate_type": "tabpfn_regressor",
                    "empty_query": True,
                    "coordinate_mode": self.coordinate_mode,
                },
            )

        history_y_arr = np.asarray(history_y, dtype=np.float32).reshape(-1)

        if len(history_y_arr) < self.min_train_size:
            return _fallback_population(
                query_x=query_x_original,
                value=self._fallback_value(history_y_arr),
                reason=f"not_enough_training_points:{len(history_y_arr)}",
            )

        try:
            train_x, train_y, query_x_transformed, query_distances = self._prepare_training_data(
                history_x,
                history_y,
                query_x_original,
            )

            if len(train_x) < self.min_train_size:
                return _fallback_population(
                    query_x=query_x_original,
                    value=self._fallback_value(train_y),
                    reason=f"not_enough_training_points:{len(train_x)}",
                )

            fit_y, y_mean, y_std = self._normalize_targets(train_y)

            pred = self._predict_mean(
                train_x=train_x,
                fit_y=fit_y,
                query_x=query_x_transformed,
                y_mean=y_mean,
                y_std=y_std,
            )

            uncertainty = self._predict_uncertainty(
                query_x=query_x_transformed,
                y_mean=y_mean,
                y_std=y_std,
            )

            metadata = {
                "surrogate_type": "tabpfn_regressor",
                "fallback": False,
                "train_size": int(len(train_x)),
                "dimension": int(train_x.shape[1]),
                "selection_mode": self.selection_mode,
                "normalized_y": bool(self.normalize_y),
                "prediction_mode": self.prediction_mode,
                "return_uncertainty": bool(self.return_uncertainty),
                "coordinate_mode": self.coordinate_mode,
                "query_trust_distances": np.asarray(query_distances, dtype=float).tolist(),
            }
            metadata.update(self._last_transform_metadata)

            return SurrogatePopulation(
                x=query_x_original.astype(float, copy=False),
                y_pred=np.asarray(pred, dtype=float),
                uncertainty=np.asarray(uncertainty, dtype=float),
                metadata=metadata,
            )

        except Exception as exc:
            if self.raise_on_error:
                raise

            train_y = np.asarray(history_y, dtype=float).reshape(-1)
            return _fallback_population(
                query_x=query_x_original,
                value=self._fallback_value(train_y),
                reason=f"exception:{type(exc).__name__}:{exc}",
            )
