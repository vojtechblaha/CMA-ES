from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import copy
import json
import math
import time

import numpy as np

from ..surrogates.tabpfn_surrogate import TabPFNSurrogate
from ..types import GenerationLog, RunSummary

ObjectiveFunction = Callable[[np.ndarray], float]
TargetMode = Literal["metric", "rank", "normal_rank"]
Criterion = Literal["sd2", "expectedrank", "fvalues", "top"]


def _rankdata_stable(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(y, kind="stable")
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.arange(len(y), dtype=float)
    return ranks


def _normal_score_ranks(y: np.ndarray) -> np.ndarray:
    ranks = _rankdata_stable(y)
    n = len(ranks)
    if n <= 1:
        return np.zeros(n, dtype=float)
    # Acklam-like fallback not needed: numpy has no erfinv, so use logistic-normal-ish scores.
    # This is monotone and bounded enough for TabPFN.
    p = (ranks + 0.5) / n
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _target_transform(y: np.ndarray, mode: TargetMode) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if mode == "metric":
        return y
    if mode == "rank":
        ranks = _rankdata_stable(y)
        if len(y) <= 1:
            return np.zeros_like(ranks)
        return ranks / float(len(y) - 1)
    if mode == "normal_rank":
        return _normal_score_ranks(y)
    raise ValueError(f"unknown target mode: {mode}")


def _kendall_tau_approx(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        da = a[i] - a[i + 1:]
        db = b[i] - b[i + 1:]
        s = da * db
        concordant += int(np.sum(s > 0))
        discordant += int(np.sum(s < 0))
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _err_rank_mu(a: np.ndarray, b: np.ndarray, mu: int) -> float:
    """Rank difference error proxy used for DTS diagnostics/adaptation.

    Returns a value in [0, 1], where 0 means that the top-mu sets agree.
    This is not a byte-for-byte copy of Matlab errRankMu, but follows the same
    purpose: rank-difference error relevant for CMA-ES selection.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = len(a)
    if n == 0:
        return float("nan")
    mu = max(1, min(int(mu), n))
    top_a = set(np.argsort(a, kind="stable")[:mu].tolist())
    top_b = set(np.argsort(b, kind="stable")[:mu].tolist())
    return 1.0 - (len(top_a & top_b) / float(mu))


def _safe_get_cma_state(es: Any) -> dict[str, Any]:
    cov = np.asarray(es.sm.C, dtype=float)
    return {
        "mean": np.asarray(es.mean, dtype=float).copy(),
        "sigma": float(es.sigma),
        "covariance": cov.copy(),
        "population_size": int(es.popsize),
        "lambda": int(es.popsize),
        "mu": int(getattr(es, "sp", object()).mu if hasattr(getattr(es, "sp", object()), "mu") else max(1, es.popsize // 2)),
        "generation": int(es.countiter),
        "evaluations": int(es.countevals),
        "dimension": int(es.N),
    }


def _mahalanobis_distances(x: np.ndarray, mean: np.ndarray, sigma: float, covariance: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.zeros(0)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    vals, vecs = np.linalg.eigh(covariance)
    vals = np.maximum(vals, 1e-14)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    z = ((x - mean[None, :]) / max(float(sigma), 1e-14)) @ inv_sqrt.T
    return np.linalg.norm(z, axis=1)


@dataclass
class DTSArchive:
    x: list[np.ndarray] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    generations: list[int] = field(default_factory=list)

    def save(self, x: np.ndarray, y: np.ndarray, generation: int) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x[None, :]
        for xi, yi in zip(x, y):
            self.x.append(np.asarray(xi, dtype=float).copy())
            self.y.append(float(yi))
            self.generations.append(int(generation))

    @property
    def n(self) -> int:
        return len(self.y)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.x:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.x), np.asarray(self.y, dtype=float)

    def get_near_point(
        self,
        n_archive_points: int,
        mean: np.ndarray,
        train_range: float,
        sigma: float,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        x, y = self.arrays()
        if len(y) == 0:
            return x, y, 0
        d = _mahalanobis_distances(x, mean, sigma, covariance)
        in_range = np.flatnonzero(d <= float(train_range))
        n_in_range = int(len(in_range))
        if n_in_range == 0:
            # Original Archive.getDataNearPoint can fail if no points are in range.
            # For robustness, choose nearest points, but report 0 in range.
            order = np.argsort(d, kind="stable")
        else:
            order = in_range[np.argsort(d[in_range], kind="stable")]
        k = min(max(0, int(n_archive_points)), len(order))
        idx = order[:k]
        return x[idx], y[idx], n_in_range

    def get_from_generations(self, generations: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not self.x:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        gset = set(int(g) for g in generations)
        idx = [i for i, g in enumerate(self.generations) if g in gset]
        if not idx:
            return np.empty((0, len(self.x[0])), dtype=float), np.empty((0,), dtype=float)
        return np.vstack([self.x[i] for i in idx]), np.asarray([self.y[i] for i in idx], dtype=float)

    @property
    def best_y(self) -> float:
        return float(np.min(self.y)) if self.y else float("inf")

    @property
    def best_x(self) -> np.ndarray | None:
        if not self.y:
            return None
        return self.x[int(np.argmin(self.y))].copy()


@dataclass
class TabPFNDTSModel:
    train_x: np.ndarray
    train_y_true: np.ndarray
    target_mode: TargetMode
    surrogate_kwargs: dict[str, Any]
    optimizer_state: dict[str, Any]
    train_generation: int
    model_id: int = 0
    is_trained: bool = True
    last_uncertainty: np.ndarray | None = None

    def _make_surrogate(self) -> TabPFNSurrogate:
        kwargs = dict(self.surrogate_kwargs)
        if self.target_mode == "rank":
            kwargs["target_mode"] = "rank"
            fit_y = self.train_y_true
        elif self.target_mode == "normal_rank":
            # Feed transformed target as metric values; no additional y normalization.
            kwargs["target_mode"] = "reg"
            kwargs["normalize_y"] = False
            fit_y = _normal_score_ranks(self.train_y_true)
        else:
            kwargs["target_mode"] = "reg"
            fit_y = self.train_y_true
        s = TabPFNSurrogate(**kwargs)
        if hasattr(s, "set_optimizer_state"):
            s.set_optimizer_state(self.optimizer_state)
        # Store transformed target for this single prediction call.
        s._dts_fit_y_override = np.asarray(fit_y, dtype=float)  # type: ignore[attr-defined]
        return s

    def predict(self, query_x: np.ndarray) -> np.ndarray:
        s = self._make_surrogate()
        train_y = getattr(s, "_dts_fit_y_override")
        pop = s.predict(self.train_x, train_y, query_x)
        self.last_uncertainty = None if pop.uncertainty is None else np.asarray(pop.uncertainty, dtype=float)
        return np.asarray(pop.y_pred, dtype=float).reshape(-1)

    def output(self, query_x: np.ndarray, criterion: Criterion = "sd2") -> np.ndarray:
        y = self.predict(query_x)
        unc = self.last_uncertainty
        if criterion in ("sd2", "expectedrank"):
            if unc is None or not np.any(np.isfinite(unc)) or float(np.nanmax(unc)) <= 0:
                # If TabPFN cannot provide quantiles, use a deterministic proxy that
                # encourages re-evaluating points whose predicted ranks are central.
                ranks = _rankdata_stable(y)
                if len(y) <= 1:
                    return np.ones_like(y)
                center = (len(y) - 1) / 2.0
                return 1.0 - np.abs(ranks - center) / max(center, 1.0)
            return np.nan_to_num(unc, nan=0.0, posinf=1e9, neginf=0.0)
        if criterion in ("fvalues", "top"):
            return y
        return y


@dataclass
class DTSPopulation:
    x: np.ndarray
    y: np.ndarray
    orig_evaled: np.ndarray
    phase: np.ndarray
    first_model_y: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "DTSPopulation":
        return cls(
            x=np.empty((0, dim), dtype=float),
            y=np.empty((0,), dtype=float),
            orig_evaled=np.empty((0,), dtype=bool),
            phase=np.empty((0,), dtype=int),
            first_model_y=np.empty((0,), dtype=float),
        )

    @property
    def n(self) -> int:
        return len(self.y)

    def add(self, x: np.ndarray, y: np.ndarray | float | None, orig_evaled: bool, phase: int) -> None:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        n = len(x)
        if y is None:
            yy = np.full(n, np.nan, dtype=float)
        else:
            yy = np.asarray(y, dtype=float).reshape(-1)
            if len(yy) == 1 and n > 1:
                yy = np.full(n, float(yy[0]), dtype=float)
        if len(yy) != n:
            raise ValueError("x/y length mismatch")
        self.x = np.vstack([self.x, x]) if self.n else x.copy()
        self.y = np.concatenate([self.y, yy])
        self.orig_evaled = np.concatenate([self.orig_evaled, np.full(n, bool(orig_evaled), dtype=bool)])
        self.phase = np.concatenate([self.phase, np.full(n, int(phase), dtype=int)])
        self.first_model_y = np.concatenate([self.first_model_y, np.full(n, np.nan, dtype=float)])

    def update_modeled(self, model: TabPFNDTSModel, phase: int, save_first: bool = False) -> None:
        idx = np.flatnonzero(~self.orig_evaled)
        if len(idx) == 0:
            return
        y_pred = model.predict(self.x[idx])
        self.y[idx] = y_pred
        self.phase[idx] = int(phase)
        if save_first:
            self.first_model_y[idx] = y_pred

    def remove_indices(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx, dtype=int)
        removed_x = self.x[idx].copy()
        removed_y = self.y[idx].copy()
        mask = np.ones(self.n, dtype=bool)
        mask[idx] = False
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.orig_evaled = self.orig_evaled[mask]
        self.phase = self.phase[mask]
        self.first_model_y = self.first_model_y[mask]
        return removed_x, removed_y

    def sort(self) -> None:
        order = np.argsort(self.y, kind="stable")
        self.x = self.x[order]
        self.y = self.y[order]
        self.orig_evaled = self.orig_evaled[order]
        self.phase = self.phase[order]
        self.first_model_y = self.first_model_y[order]

    def min_modeled(self) -> float | None:
        idx = np.flatnonzero(~self.orig_evaled)
        if len(idx) == 0:
            return None
        return float(np.min(self.y[idx]))

    def shift_y(self, diff: float) -> None:
        self.y = self.y + float(diff)


class ExactLikeDTSTabPFNExperiment:
    """A close Python port of the DoubleTrainedEC control flow with TabPFN models.

    This runner mirrors the public Matlab DTS-CMA-ES structure as closely as the
    pycma ask/tell interface allows: archive-near-point training-set selection,
    presampling for minimum training size, model archive with accepted age,
    restricted true-evaluation ratio, optional preselection, double-training
    reevaluation cycle, model-value shifting, and DTS statistics.
    """

    def __init__(
        self,
        *,
        x0: np.ndarray,
        sigma0: float,
        population_size: int | None,
        seed: int,
        max_true_evals: int,
        max_generations: int,
        target_f: float | None,
        experiment_name: str,
        function_id: int,
        instance_id: int,
        dimension: int,
        output_dir: Path,
        surrogate_kwargs: dict[str, Any],
        target_mode: TargetMode = "rank",
        evo_control_restricted_param: float = 0.05,
        evo_control_train_range: float = 10.0,
        evo_control_train_n_archive_points: int | str = "15*dim",
        evo_control_accepted_model_age: int = 2,
        evo_control_model_archive_length: int = 5,
        evo_control_use_double_training: bool = True,
        evo_control_max_double_train_iterations: int = 1,
        evo_control_min_points_for_expected_rank: int = 4,
        evo_control_orig_points_round_fcn: str = "ceil",
        evo_control_n_best_points: tuple[float, float] = (0.0, 0.0),
        evo_control_preselection_pop_ratio: int = 50,
        evo_control_validation_generation_period: int = 1,
        evo_control_validation_pop_size: int = 0,
        reevaluation_criterion: Criterion = "sd2",
        shift_model_values: bool = True,
        inopts: dict[str, Any] | None = None,
    ) -> None:
        import cma

        opts = dict(inopts or {})
        # Keep CMA-ES hyperparameters identical to the reference implementation by default.
        # In the original s_cmaes.m, PopSize is not set unless the experiment passes it,
        # so CMA-ES uses the default lambda = 4 + floor(3*log(N)).
        if population_size is not None:
            opts.setdefault("popsize", int(population_size))
        opts.setdefault("seed", int(seed))
        self.es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float), float(sigma0), opts)
        self.max_true_evals = int(max_true_evals)
        self.max_generations = int(max_generations)
        self.target_f = target_f
        self.experiment_name = experiment_name
        self.function_id = int(function_id)
        self.instance_id = int(instance_id)
        self.dimension = int(dimension)
        self.output_dir = Path(output_dir)
        self.surrogate_kwargs = dict(surrogate_kwargs)
        self.target_mode = target_mode
        self.restricted_param = float(evo_control_restricted_param)
        self.train_range = float(evo_control_train_range)
        self.train_n_archive_points_expr = evo_control_train_n_archive_points
        self.accepted_model_age = int(evo_control_accepted_model_age)
        self.model_archive_length = int(evo_control_model_archive_length)
        self.use_double_training = bool(evo_control_use_double_training)
        self.max_double_train_iterations = max(1, int(evo_control_max_double_train_iterations))
        self.min_points_for_expected_rank = int(evo_control_min_points_for_expected_rank)
        self.orig_points_round_fcn = evo_control_orig_points_round_fcn
        self.n_best_points = tuple(evo_control_n_best_points)
        self.preselection_pop_ratio = int(evo_control_preselection_pop_ratio)
        self.validation_generation_period = int(evo_control_validation_generation_period)
        self.validation_pop_size = int(evo_control_validation_pop_size)
        self.reevaluation_criterion = reevaluation_criterion
        self.shift_model_values = bool(shift_model_values)
        self.archive = DTSArchive()
        self.model_archive: list[TabPFNDTSModel | None] = [None] * self.model_archive_length
        self.model_archive_generations = np.full(self.model_archive_length, np.nan)
        self.model_counter = 0
        self.true_evals = 0
        self.logs: list[GenerationLog] = []

    @property
    def lambda_(self) -> int:
        return int(self.es.popsize)

    @property
    def mu(self) -> int:
        return max(1, int(getattr(self.es.sp, "mu", self.lambda_ // 2)))

    def _eval_train_n_archive_points(self) -> int:
        if isinstance(self.train_n_archive_points_expr, str):
            dim = self.dimension
            return int(round(eval(self.train_n_archive_points_expr, {"__builtins__": {}}, {"dim": dim, "lambda_": self.lambda_})))
        return int(self.train_n_archive_points_expr)

    def _round_orig_points(self, value: float) -> int:
        if self.orig_points_round_fcn == "ceil":
            return int(math.ceil(value))
        if self.orig_points_round_fcn == "floor":
            return int(math.floor(value))
        if self.orig_points_round_fcn == "round":
            return int(round(value))
        # Matlab DTS also allows a probabilistic helper for non-integers.
        if self.orig_points_round_fcn == "prob":
            base = int(math.floor(value))
            return base + int(np.random.random() < (value - base))
        raise ValueError(f"unknown orig point rounding function: {self.orig_points_round_fcn}")

    def _ask(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dimension), dtype=float)
        return np.asarray(self.es.ask(number=int(n)), dtype=float)

    def _true_evaluate(self, objective: ObjectiveFunction, x: np.ndarray, generation: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        remaining = self.max_true_evals - self.true_evals
        if len(x) > remaining:
            x = x[:remaining]
        y = np.asarray([objective(xi) for xi in x], dtype=float)
        self.true_evals += len(y)
        if len(y):
            self.archive.save(x, y, generation)
        return y

    def _make_model(self, train_x: np.ndarray, train_y: np.ndarray, optimizer_state: dict[str, Any], generation: int) -> TabPFNDTSModel | None:
        if len(train_y) < int(self.surrogate_kwargs.get("min_train_size", 5)):
            return None
        self.model_counter += 1
        return TabPFNDTSModel(
            train_x=np.asarray(train_x, dtype=float).copy(),
            train_y_true=np.asarray(train_y, dtype=float).copy(),
            target_mode=self.target_mode,
            surrogate_kwargs=self.surrogate_kwargs,
            optimizer_state=copy.deepcopy(optimizer_state),
            train_generation=int(generation),
            model_id=self.model_counter,
        )

    def _update_model_archive(self, model: TabPFNDTSModel, generation: int) -> None:
        # Matlab shifts previous models if slot 1 is older than current generation.
        if np.isfinite(self.model_archive_generations[0]) and self.model_archive_generations[0] < generation:
            self.model_archive[1:] = self.model_archive[:-1]
            self.model_archive_generations[1:] = self.model_archive_generations[:-1]
            self.model_archive[0] = None
            self.model_archive_generations[0] = np.nan
        self.model_archive[0] = model
        self.model_archive_generations[0] = generation

    def _get_old_model(self, generation: int, generation_diffs: range | list[int]) -> tuple[TabPFNDTSModel | None, int | None]:
        targets = {generation - int(d) for d in generation_diffs if generation - int(d) >= 0}
        if not targets:
            return None, None
        for m, g in zip(self.model_archive, self.model_archive_generations):
            if m is not None and np.isfinite(g) and int(g) in targets:
                return m, generation - int(g)
        return None, None

    def _choose_for_reevaluation(self, pop: DTSPopulation, model: TabPFNDTSModel, n_points: int) -> np.ndarray:
        not_orig = np.flatnonzero(~pop.orig_evaled)
        if len(not_orig) == 0:
            return np.empty((0,), dtype=int)
        n_points = min(int(n_points), len(not_orig))
        criterion = self.reevaluation_criterion
        if criterion == "expectedrank" and len(not_orig) < self.min_points_for_expected_rank:
            criterion = "sd2"
        scores = model.output(pop.x[not_orig], criterion)
        if criterion in ("sd2", "expectedrank"):
            order_local = np.argsort(-scores, kind="stable")
        else:
            order_local = np.argsort(scores, kind="stable")
        return not_orig[order_local[:n_points]]

    def _preselection(self, objective: ObjectiveFunction, model: TabPFNDTSModel, n_points_available: int, generation: int) -> tuple[np.ndarray, np.ndarray]:
        if n_points_available <= 0:
            return np.empty((0, self.dimension)), np.empty((0,))
        n_best_exact = self.n_best_points[0] if n_points_available <= 1 else self.n_best_points[1]
        # getProbNumber behavior.
        base = int(math.floor(n_best_exact))
        n_best = base + int(np.random.random() < (n_best_exact - base))
        n_best = min(max(0, n_best), n_points_available)
        if n_best <= 0:
            return np.empty((0, self.dimension)), np.empty((0,))
        n_pre = max(n_best, int(self.preselection_pop_ratio * self.lambda_))
        x_pre = self._ask(n_pre)
        y_pre = model.predict(x_pre)
        best_idx = np.argsort(y_pre, kind="stable")[:n_best]
        x_best = x_pre[best_idx]
        y_best = self._true_evaluate(objective, x_best, generation)
        return x_best[: len(y_best)], y_best

    def _finalize_generation(self, pop: DTSPopulation) -> dict[str, Any]:
        pop.sort()
        if self.shift_model_values and self.archive.n > 0:
            fmin_model = pop.min_modeled()
            if fmin_model is not None:
                diff = max(self.archive.best_y - fmin_model, 0.0)
                if diff > 0:
                    pop.shift_y(1.000001 * diff)
        # Statistics analogous to DTS output columns.
        modeled_first = pop.first_model_y.copy()
        is_original_after_presample = pop.orig_evaled & (pop.phase != 0)
        rmse_reeval = float("nan")
        kendall_reeval = float("nan")
        rank_err_reeval = float("nan")
        if np.any(is_original_after_presample) and np.any(np.isfinite(modeled_first[is_original_after_presample])):
            pred = modeled_first[is_original_after_presample]
            true = pop.y[is_original_after_presample]
            rmse_reeval = float(np.sqrt(np.mean((pred - true) ** 2)))
            kendall_reeval = _kendall_tau_approx(pred, true)
            mixed = modeled_first.copy()
            mixed[is_original_after_presample] = pop.y[is_original_after_presample]
            valid = np.isfinite(modeled_first) & (pop.phase != 0)
            if np.any(valid):
                rank_err_reeval = _err_rank_mu(modeled_first[valid], mixed[valid], self.mu)
        return {
            "rmse_reeval": rmse_reeval,
            "kendall_reeval": kendall_reeval,
            "rank_err_reeval": rank_err_reeval,
            "num_modeled": int(np.sum(~pop.orig_evaled)),
            "num_original": int(np.sum(pop.orig_evaled)),
            "archive_best_y": self.archive.best_y,
        }

    def run(self, objective: ObjectiveFunction) -> RunSummary:
        run_id = f"f{self.function_id}_i{self.instance_id}_d{self.dimension}_s{int(self.es.opts['seed'])}"
        out_root = self.output_dir / self.experiment_name / run_id
        out_root.mkdir(parents=True, exist_ok=True)
        logs_path = out_root / "generation_logs.jsonl"
        summary_path = out_root / "summary.json"

        generation = 0
        while generation < self.max_generations and self.true_evals < self.max_true_evals:
            if self.es.stop():
                break
            incumbent_before = self.archive.best_y
            cma_state = _safe_get_cma_state(self.es)
            lambda_ = self.lambda_
            dim = self.dimension
            pop = DTSPopulation.empty(dim)
            n_presampled = 0
            model_age = 0
            used_old_model = False
            train_model_ok = False
            n_data_in_range = 0
            retrain_rank_err = float("nan")
            used_best_points = 0

            n_archive_points = self._eval_train_n_archive_points()
            x_train, y_train, n_data_in_range = self.archive.get_near_point(
                n_archive_points,
                np.asarray(cma_state["mean"], dtype=float),
                self.train_range,
                float(cma_state["sigma"]),
                np.asarray(cma_state["covariance"], dtype=float),
            )
            min_train_size = int(self.surrogate_kwargs.get("min_train_size", 5))

            # Presample true points if archive does not provide enough data.
            if len(y_train) < min_train_size and self.true_evals < self.max_true_evals:
                need = min(min_train_size - len(y_train), lambda_)
                x_pre = self._ask(need)
                y_pre = self._true_evaluate(objective, x_pre, generation)
                n_presampled = len(y_pre)
                if n_presampled:
                    pop.add(x_pre[:n_presampled], y_pre, orig_evaled=True, phase=0)
                    x_train = np.vstack([x_train, x_pre[:n_presampled]]) if len(x_train) else x_pre[:n_presampled]
                    y_train = np.concatenate([y_train, y_pre])

            model = self._make_model(x_train, y_train, cma_state, generation)
            if model is None:
                old_model, age = self._get_old_model(generation, range(0, self.accepted_model_age + 1))
                if old_model is not None:
                    model = old_model
                    model_age = int(age or 0)
                    used_old_model = True

            n_lambda_rest = max(0, lambda_ - n_presampled)
            # Existing pycma state cannot know the original DTS cmaesState.thisGenerationMaxevals;
            # cap by remaining true-eval budget and by the rest of this generation.
            maxevals = min(n_lambda_rest, self.max_true_evals - self.true_evals)

            if model is None or n_lambda_rest <= 0:
                # Standard CMA-ES fallback: fill remaining population by true evals.
                x_rest = self._ask(n_lambda_rest)
                y_rest = self._true_evaluate(objective, x_rest, generation)
                if len(y_rest):
                    pop.add(x_rest[:len(y_rest)], y_rest, orig_evaled=True, phase=3)
                # If budget clipped the generation, break without tell if not enough points.
                if pop.n < lambda_:
                    break
                pop.sort()
                self.es.tell(pop.x, pop.y.tolist())
                ec_meta = {"ec_type": "dts_exact_like_fallback", "reason": "no_model", "num_true": int(np.sum(pop.orig_evaled)), "num_surrogate": 0}
            else:
                # Sample unevaluated rest of the population.
                x_rest = self._ask(n_lambda_rest)
                pop.add(x_rest, None, orig_evaled=False, phase=4)

                if not used_old_model:
                    # Train first model on archive-near-point data + original points in current pop.
                    orig_idx = np.flatnonzero(pop.orig_evaled)
                    if len(orig_idx):
                        x_model_train = np.vstack([x_train, pop.x[orig_idx]])
                        y_model_train = np.concatenate([y_train, pop.y[orig_idx]])
                    else:
                        x_model_train, y_model_train = x_train, y_train
                    maybe_model = self._make_model(x_model_train, y_model_train, cma_state, generation)
                    if maybe_model is not None:
                        model = maybe_model
                        self._update_model_archive(model, generation)
                        train_model_ok = True

                # Validation generation can increase the true evaluation ratio.
                restricted_param = self.restricted_param
                if self.validation_pop_size > 0 and self.validation_generation_period > 0 and (generation % self.validation_generation_period == 0):
                    restricted_param = max(min(1.0, self.validation_pop_size / max(n_lambda_rest, 1)), restricted_param)
                n_points = self._round_orig_points(n_lambda_rest * restricted_param)
                n_points = min(n_points, maxevals)

                # Optional DTS preselection: evaluate best predicted out of a larger presampled population.
                if n_points > 0:
                    x_best, y_best = self._preselection(objective, model, n_points, generation)
                    used_best_points = len(y_best)
                    if used_best_points > 0:
                        pop.add(x_best, y_best, orig_evaled=True, phase=1)
                        # Remove the same number of arbitrary not-original points, following Matlab TODO behavior.
                        not_orig = np.flatnonzero(~pop.orig_evaled)
                        if len(not_orig) >= used_best_points:
                            pop.remove_indices(not_orig[:used_best_points])
                    n_points = max(0, n_points - used_best_points)

                # First model predictions for all not-original points.
                pop.update_modeled(model, phase=2, save_first=True)
                last_model = model
                double_train_iteration = 0
                n_to_reeval_per_iteration = n_points / float(self.max_double_train_iterations)
                remaining_orig_evals = n_points

                while True:
                    double_train_iteration += 1
                    n_to_reeval = 0
                    if remaining_orig_evals > 0:
                        if (
                            double_train_iteration == self.max_double_train_iterations
                            or (n_to_reeval_per_iteration > 1 and (remaining_orig_evals - math.floor(n_to_reeval_per_iteration)) == 1)
                        ):
                            n_to_reeval_per_iteration = remaining_orig_evals
                        n_to_reeval = min(remaining_orig_evals, max(1, int(round(n_to_reeval_per_iteration))))
                        idx = self._choose_for_reevaluation(pop, last_model, n_to_reeval)
                        if len(idx) > 0:
                            x_eval, _ = pop.remove_indices(idx)
                            y_eval = self._true_evaluate(objective, x_eval, generation)
                            x_eval = x_eval[: len(y_eval)]
                            n_to_reeval = len(y_eval)
                            if n_to_reeval:
                                pop.add(x_eval, y_eval, orig_evaled=True, phase=1)
                        else:
                            n_to_reeval = 0

                    if n_to_reeval > 0 and self.use_double_training:
                        orig_idx = np.flatnonzero(pop.orig_evaled)
                        x_retrain = np.vstack([x_train, pop.x[orig_idx]]) if len(x_train) else pop.x[orig_idx]
                        y_retrain = np.concatenate([y_train, pop.y[orig_idx]]) if len(y_train) else pop.y[orig_idx]
                        retrained = self._make_model(x_retrain, y_retrain, cma_state, generation)
                        if retrained is not None:
                            y_model_1 = model.predict(pop.x)
                            y_model_2 = retrained.predict(pop.x)
                            ref = y_model_2.copy()
                            ref[pop.orig_evaled] = pop.y[pop.orig_evaled]
                            retrain_rank_err = _err_rank_mu(y_model_1, ref, self.mu)
                            last_model = retrained
                            self._update_model_archive(retrained, generation)
                            pop.update_modeled(retrained, phase=2, save_first=False)
                    remaining_orig_evals -= n_to_reeval
                    not_everything = (
                        double_train_iteration < self.max_double_train_iterations
                        and max(n_points, math.floor(lambda_ * restricted_param)) > int(np.sum(pop.orig_evaled))
                    )
                    if not not_everything:
                        break

                # Safety: any NaN/non-evaluated points get last model prediction.
                if np.any(~np.isfinite(pop.y)):
                    pop.update_modeled(last_model, phase=2, save_first=False)
                if pop.n != lambda_:
                    # Fill if presampling/preselection/budget edge case left a short population.
                    need = lambda_ - pop.n
                    if need > 0:
                        x_fill = self._ask(need)
                        y_fill = self._true_evaluate(objective, x_fill, generation)
                        if len(y_fill):
                            pop.add(x_fill[:len(y_fill)], y_fill, orig_evaled=True, phase=3)
                    if pop.n != lambda_:
                        break
                stats = self._finalize_generation(pop)
                self.es.tell(pop.x, pop.y.tolist())
                ec_meta = {
                    "ec_type": "doubletrained_exact_like_tabpfn",
                    "restricted_param": restricted_param,
                    "num_true": int(np.sum(pop.orig_evaled)),
                    "num_surrogate": int(np.sum(~pop.orig_evaled)),
                    "num_presampled": int(n_presampled),
                    "num_preselected": int(used_best_points),
                    "model_age": int(model_age),
                    "used_old_model": bool(used_old_model),
                    "model_trained_this_generation": bool(train_model_ok),
                    "n_data_in_range": int(n_data_in_range),
                    "n_archive_points_used": int(len(y_train)),
                    "rank_err_2_models": retrain_rank_err,
                    "target_mode": self.target_mode,
                    "reevaluation_criterion": self.reevaluation_criterion,
                    **stats,
                }

            best_after = self.archive.best_y
            log = GenerationLog(
                generation_index=generation,
                mode="dts_tabpfn_exact_like",
                surrogate_name=f"dts_tabpfn_{self.target_mode}",
                incumbent_y_before=incumbent_before,
                incumbent_y_after=best_after,
                num_true_evals=int(ec_meta.get("num_true", 0)),
                num_surrogate_evals=int(ec_meta.get("num_surrogate", 0)),
                metadata={"ec_metadata": ec_meta, "optimizer_state_after": _safe_get_cma_state(self.es), "optimizer_state_before": cma_state},
            )
            self.logs.append(log)
            with logs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_generation_log_to_json(log)) + "\n")
            if self.target_f is not None and best_after <= self.target_f:
                break
            generation += 1

        best_x = self.archive.best_x
        summary = RunSummary(
            run_id=run_id,
            function_id=self.function_id,
            instance_id=self.instance_id,
            dimension=self.dimension,
            generations=len(self.logs),
            true_evaluations=self.true_evals,
            best_y=self.archive.best_y,
            solved=self.target_f is not None and self.archive.best_y <= self.target_f,
            logs=self.logs,
            metadata={
                "experiment_name": self.experiment_name,
                "target_mode": self.target_mode,
                "restricted_param": self.restricted_param,
                "best_x": None if best_x is None else best_x.tolist(),
            },
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_run_summary_to_json(summary), f, indent=2)
        return summary


def _generation_log_to_json(log: GenerationLog) -> dict[str, Any]:
    return {
        "generation_index": log.generation_index,
        "mode": log.mode,
        "surrogate_name": log.surrogate_name,
        "incumbent_y_before": log.incumbent_y_before,
        "incumbent_y_after": log.incumbent_y_after,
        "num_true_evals": log.num_true_evals,
        "num_surrogate_evals": log.num_surrogate_evals,
        "decision_goodness": log.decision_goodness,
        "dataset_scores": log.dataset_scores,
        "metadata": _jsonify(log.metadata),
    }


def _run_summary_to_json(summary: RunSummary) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "function_id": summary.function_id,
        "instance_id": summary.instance_id,
        "dimension": summary.dimension,
        "generations": summary.generations,
        "true_evaluations": summary.true_evaluations,
        "best_y": summary.best_y,
        "solved": summary.solved,
        "metadata": _jsonify(summary.metadata),
    }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj
