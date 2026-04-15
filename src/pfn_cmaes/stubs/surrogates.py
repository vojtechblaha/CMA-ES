from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..interfaces import SurrogateModel
from ..types import SurrogatePopulation


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    return np.maximum(aa + bb - 2.0 * ab, 0.0)


def _rbf_local_weights(
    train_x: np.ndarray,
    query_x: np.ndarray,
    *,
    lengthscale: float,
) -> np.ndarray:
    d2 = _pairwise_sq_dists(query_x, train_x)
    return np.exp(-0.5 * d2 / max(lengthscale**2, 1e-12))


def _normalize_xy(
    train_x: np.ndarray,
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)

    train_xn = (train_x - mean) / std
    query_xn = (query_x - mean) / std
    return train_xn, query_xn, mean, std


def _fallback_predictions(query_x: np.ndarray) -> SurrogatePopulation:
    """Fallback used when no true-evaluated history is available yet."""
    query_x = np.asarray(query_x, dtype=float)
    return SurrogatePopulation(
        x=query_x,
        y_pred=np.zeros(len(query_x), dtype=float),
    )


@dataclass(slots=True)
class LocalLinearSurrogate(SurrogateModel):
    ridge: float = 1e-6
    lengthscale: float = 1.0

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        train_x = np.asarray(history_x, dtype=float)
        train_y = np.asarray(history_y, dtype=float)
        query_x = np.asarray(query_x, dtype=float)

        if len(train_x) == 0:
            return _fallback_predictions(query_x)

        train_xn, query_xn, _, _ = _normalize_xy(train_x, query_x)
        weights = _rbf_local_weights(train_xn, query_xn, lengthscale=self.lengthscale)

        phi = np.column_stack([np.ones(len(train_xn)), train_xn])

        preds = []
        for i, xq in enumerate(query_xn):
            w = weights[i]
            W = np.diag(w + 1e-12)

            A = phi.T @ W @ phi + self.ridge * np.eye(phi.shape[1])
            b = phi.T @ W @ train_y
            beta = np.linalg.solve(A, b)

            phi_q = np.concatenate([[1.0], xq])
            preds.append(float(phi_q @ beta))

        return SurrogatePopulation(
            x=query_x,
            y_pred=np.asarray(preds, dtype=float),
        )


@dataclass(slots=True)
class LocalQuadraticSurrogate(SurrogateModel):
    ridge: float = 1e-5
    lengthscale: float = 1.0

    @staticmethod
    def _quadratic_features(x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        features = [np.ones((n, 1)), x]

        quad_terms = []
        for i in range(d):
            for j in range(i, d):
                quad_terms.append((x[:, i] * x[:, j])[:, None])

        if quad_terms:
            features.append(np.hstack(quad_terms))

        return np.hstack(features)

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        train_x = np.asarray(history_x, dtype=float)
        train_y = np.asarray(history_y, dtype=float)
        query_x = np.asarray(query_x, dtype=float)

        if len(train_x) == 0:
            return _fallback_predictions(query_x)

        train_xn, query_xn, _, _ = _normalize_xy(train_x, query_x)
        weights = _rbf_local_weights(train_xn, query_xn, lengthscale=self.lengthscale)

        Phi = self._quadratic_features(train_xn)
        Phi_q = self._quadratic_features(query_xn)

        preds = []
        for i in range(len(query_xn)):
            w = weights[i]
            W = np.diag(w + 1e-12)

            A = Phi.T @ W @ Phi + self.ridge * np.eye(Phi.shape[1])
            b = Phi.T @ W @ train_y
            beta = np.linalg.solve(A, b)

            preds.append(float(Phi_q[i] @ beta))

        return SurrogatePopulation(
            x=query_x,
            y_pred=np.asarray(preds, dtype=float),
        )


@dataclass(slots=True)
class GaussianProcessMaternSurrogate(SurrogateModel):
    nu: float = 2.5
    alpha: float = 1e-6
    normalize_y: bool = False
    return_std: bool = True
    n_restarts_optimizer: int = 0
    random_state: int = 0
    min_train_size: int = 2

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        import warnings

        train_x = np.asarray(history_x, dtype=float)
        train_y = np.asarray(history_y, dtype=float).reshape(-1)
        query_x = np.asarray(query_x, dtype=float)

        if query_x.ndim != 2:
            raise ValueError(f"query_x must have shape [N, D], got {query_x.shape}")

        if train_x.ndim != 2:
            raise ValueError(f"history_x must have shape [N, D], got {train_x.shape}")

        if len(train_x) != len(train_y):
            raise ValueError(
                "history_x and history_y must have the same number of samples."
            )

        if len(query_x) == 0:
            return SurrogatePopulation(
                x=query_x,
                y_pred=np.zeros((0,), dtype=float),
                uncertainty=np.zeros((0,), dtype=float) if self.return_std else None,
            )

        if len(train_x) < self.min_train_size:
            return _fallback_predictions(query_x)

        if train_x.shape[1] != query_x.shape[1]:
            raise ValueError(
                f"history_x and query_x must have the same feature dimension, "
                f"got {train_x.shape[1]} and {query_x.shape[1]}"
            )

        train_x = np.nan_to_num(train_x, nan=0.0, posinf=1e6, neginf=-1e6)
        train_y = np.nan_to_num(train_y, nan=0.0, posinf=1e6, neginf=-1e6)
        query_x = np.nan_to_num(query_x, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            x_scaler = StandardScaler()
            train_x_scaled = x_scaler.fit_transform(train_x)
            query_x_scaled = x_scaler.transform(query_x)

            if self.normalize_y:
                y_scaler = StandardScaler()
                train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
            else:
                y_scaler = None
                train_y_scaled = train_y

            kernel = (
                ConstantKernel(1.0, (1e-2, 1e4))
                * Matern(
                    length_scale=np.ones(train_x.shape[1], dtype=float),
                    length_scale_bounds=(1e-2, 1e3),
                    nu=self.nu,
                )
                + WhiteKernel(
                    noise_level=max(self.alpha, 1e-8),
                    noise_level_bounds=(1e-8, 1e1),
                )
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=max(self.alpha, 1e-8),
                normalize_y=False,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(train_x_scaled, train_y_scaled)

            if self.return_std:
                mean_scaled, std_scaled = gp.predict(query_x_scaled, return_std=True)

                mean_scaled = np.asarray(mean_scaled, dtype=float).reshape(-1)
                std_scaled = np.asarray(std_scaled, dtype=float).reshape(-1)

                if y_scaler is not None:
                    mean = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
                    y_scale = float(y_scaler.scale_[0]) if np.ndim(y_scaler.scale_) > 0 else float(y_scaler.scale_)
                    std = std_scaled * abs(y_scale)
                else:
                    mean = mean_scaled
                    std = std_scaled

                std = np.maximum(std, 0.0)

                return SurrogatePopulation(
                    x=query_x,
                    y_pred=np.asarray(mean, dtype=float),
                    uncertainty=np.asarray(std, dtype=float),
                )

            mean_scaled = gp.predict(query_x_scaled, return_std=False)
            mean_scaled = np.asarray(mean_scaled, dtype=float).reshape(-1)

            if y_scaler is not None:
                mean = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            else:
                mean = mean_scaled

            return SurrogatePopulation(
                x=query_x,
                y_pred=np.asarray(mean, dtype=float),
            )

        except Exception:
            return _fallback_predictions(query_x)


@dataclass(slots=True)
class RandomForestSurrogate(SurrogateModel):
    n_estimators: int = 200
    min_samples_leaf: int = 2
    random_state: int = 0
    return_std: bool = True

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        from sklearn.ensemble import RandomForestRegressor

        train_x = np.asarray(history_x, dtype=float)
        train_y = np.asarray(history_y, dtype=float)
        query_x = np.asarray(query_x, dtype=float)

        if len(train_x) == 0:
            return _fallback_predictions(query_x)

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(train_x, train_y)

        tree_preds = np.stack([tree.predict(query_x) for tree in rf.estimators_], axis=0)
        mean = tree_preds.mean(axis=0)

        if self.return_std:
            std = tree_preds.std(axis=0)
            return SurrogatePopulation(
                x=query_x,
                y_pred=np.asarray(mean, dtype=float),
                uncertainty=np.asarray(std, dtype=float),
            )

        return SurrogatePopulation(
            x=query_x,
            y_pred=np.asarray(mean, dtype=float),
        )


@dataclass(slots=True)
class RankSVMSurrogate(SurrogateModel):
    """
    Practical ranking-oriented surrogate.

    This uses SVR as a smooth score estimator. It is not a strict pairwise RankSVM
    implementation, but it works well as a ranking surrogate because CMA-ES mainly
    relies on the induced ordering of candidate solutions.
    """

    C: float = 10.0
    epsilon: float = 0.01
    gamma: str = "scale"

    def predict(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        query_x: np.ndarray,
    ) -> SurrogatePopulation:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        train_x = np.asarray(history_x, dtype=float)
        train_y = np.asarray(history_y, dtype=float)
        query_x = np.asarray(query_x, dtype=float)

        if len(train_x) == 0:
            return _fallback_predictions(query_x)

        model = make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma,
            ),
        )
        model.fit(train_x, train_y)
        pred = model.predict(query_x)

        return SurrogatePopulation(
            x=query_x,
            y_pred=np.asarray(pred, dtype=float),
        )