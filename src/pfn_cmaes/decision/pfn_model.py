from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..interfaces import DecisionModel
from ..types import GenerationState, SurrogateDecision


@dataclass(slots=True)
class PFNDecisionConfig:
    checkpoint_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"

    max_history: int = 256
    normalize_targets: bool = True
    include_ranks: bool = True
    include_recency: bool = True
    include_optimizer_features_in_context: bool = True
    include_budget_features: bool = False
    include_history_trend_features: bool = False
    include_candidate_distribution_features: bool = False
    max_true_evals: int | None = None
    tie_margin: float = 1e-3
    temperature: float = 1.0


class PFNBackboneProtocol(Protocol):
    def __call__(
        self,
        context_x: "torch.Tensor",
        context_y: "torch.Tensor",
        candidate_x: "torch.Tensor",
        action_ids: "torch.Tensor",
        context_mask: "torch.Tensor | None" = None,
        candidate_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor": ...


class PFNStateFeaturizer:
    """
    Build set-based PFN inputs:

    - context_x/context_y from evaluated history
    - candidate_x from current candidate set
    - action_ids from surrogate_names ordering
    """

    def __init__(self, config: PFNDecisionConfig) -> None:
        self.config = config

    def build(
        self,
        state: GenerationState,
        surrogate_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        x_hist, y_hist = self._extract_history(state)
        cand_x = self._extract_candidates(state)
        optimizer_features = self._extract_optimizer_features(state)

        context_x = self._build_context_features(
            x_hist=x_hist,
            y_hist=y_hist,
            optimizer_features=optimizer_features,
        )
        context_y = self._build_context_targets(
            y_hist=y_hist,
            incumbent_y=float(state.incumbent_y),
        )
        candidate_x = self._build_candidate_features(
            candidate_x=cand_x,
            incumbent_x=np.asarray(state.incumbent_x, dtype=np.float32),
            optimizer_features=optimizer_features,
        )
        action_ids = np.arange(len(surrogate_names), dtype=np.int64)

        context_x = self._sanitize_array(context_x)
        context_y = self._sanitize_array(context_y)
        candidate_x = self._sanitize_array(candidate_x)

        debug = {
            "history_size": int(len(y_hist)),
            "candidate_count": int(len(cand_x)),
            "context_dim": int(context_x.shape[1]) if context_x.ndim == 2 else 0,
            "candidate_dim": int(candidate_x.shape[1]) if candidate_x.ndim == 2 else 0,
        }
        return context_x, context_y, candidate_x, action_ids, debug

    @staticmethod
    def _sanitize_array(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def _extract_history(self, state: GenerationState) -> tuple[np.ndarray, np.ndarray]:
        x_hist = np.asarray(state.evaluated_history_x, dtype=np.float32)
        y_hist = np.asarray(state.evaluated_history_y, dtype=np.float32).reshape(-1)

        if x_hist.ndim != 2:
            raise ValueError(f"Expected evaluated_history_x shape [N, D], got {x_hist.shape}")
        if y_hist.ndim != 1:
            raise ValueError(f"Expected evaluated_history_y shape [N], got {y_hist.shape}")
        if len(x_hist) != len(y_hist):
            raise ValueError("evaluated_history_x and evaluated_history_y must have same length.")

        if len(y_hist) > self.config.max_history:
            x_hist = x_hist[-self.config.max_history :]
            y_hist = y_hist[-self.config.max_history :]

        return self._sanitize_array(x_hist), self._sanitize_array(y_hist)

    def _extract_candidates(self, state: GenerationState) -> np.ndarray:
        cand_x = np.asarray(state.candidate_x, dtype=np.float32)
        if cand_x.ndim != 2:
            raise ValueError(f"Expected candidate_x shape [Q, D], got {cand_x.shape}")

        hist_x = np.asarray(state.evaluated_history_x, dtype=np.float32)
        if hist_x.ndim == 2 and hist_x.shape[0] > 0 and cand_x.shape[1] != hist_x.shape[1]:
            raise ValueError("candidate_x and evaluated_history_x must have same feature dimension.")
        return self._sanitize_array(cand_x)

    def _extract_optimizer_features(self, state: GenerationState) -> np.ndarray:
        optimizer_state = state.optimizer_state or {}
        metadata = state.metadata or {}

        hist_x = np.asarray(state.evaluated_history_x, dtype=np.float32)
        cand_x = np.asarray(state.candidate_x, dtype=np.float32)

        if hist_x.ndim == 2 and hist_x.shape[1] > 0:
            inferred_dim = int(hist_x.shape[1])
        elif cand_x.ndim == 2 and cand_x.shape[1] > 0:
            inferred_dim = int(cand_x.shape[1])
        else:
            inferred_dim = 0

        history_y = np.asarray(state.evaluated_history_y, dtype=np.float32).reshape(-1)
        history_y = self._sanitize_array(history_y)

        population_size = float(optimizer_state.get("population_size", metadata.get("population_size", len(cand_x))))
        sigma = float(optimizer_state.get("sigma", metadata.get("sigma", 0.0)))

        # Preserve the legacy feature value unless v2 budget features are explicitly enabled.
        legacy_true_evals = float(optimizer_state.get("num_true_evals", len(history_y)))
        evals_used = float(
            optimizer_state.get(
                "evaluations",
                optimizer_state.get("num_true_evals", metadata.get("evaluations", len(history_y))),
            )
        )

        feats: list[float] = [
            float(state.generation_index),
            float(optimizer_state.get("dimension", metadata.get("dimension", inferred_dim))),
            population_size,
            float(state.incumbent_y) if np.isfinite(float(state.incumbent_y)) else 0.0,
            sigma,
            legacy_true_evals,
        ]

        if self.config.include_budget_features:
            feats.extend(
                self._budget_features(
                    generation_index=int(state.generation_index),
                    evals_used=evals_used,
                    population_size=population_size,
                    history_size=float(len(history_y)),
                )
            )

        if self.config.include_history_trend_features:
            feats.extend(
                self._history_trend_features(
                    history_y=history_y,
                    incumbent_y=float(state.incumbent_y),
                    evals_used=evals_used,
                )
            )

        if self.config.include_candidate_distribution_features:
            feats.extend(
                self._candidate_distribution_features(
                    candidate_x=cand_x,
                    incumbent_x=np.asarray(state.incumbent_x, dtype=np.float32),
                )
            )

        return self._sanitize_array(np.asarray(feats, dtype=np.float32))

    def _budget_features(
        self,
        *,
        generation_index: int,
        evals_used: float,
        population_size: float,
        history_size: float,
    ) -> list[float]:
        budget = float(self.config.max_true_evals or 0)
        if budget <= 0:
            budget = max(evals_used, history_size, population_size, 1.0)

        remaining = max(budget - evals_used, 0.0)
        generations_budget = max(budget / max(population_size, 1.0), 1.0)

        return [
            evals_used,
            budget,
            remaining,
            evals_used / max(budget, 1.0),
            remaining / max(budget, 1.0),
            float(generation_index) / generations_budget,
            np.log1p(max(evals_used, 0.0)) / max(np.log1p(budget), 1e-12),
            history_size / max(budget, 1.0),
        ]

    @staticmethod
    def _safe_relative_improvement(previous_best: float, current_best: float) -> float:
        if not np.isfinite(previous_best) or not np.isfinite(current_best):
            return 0.0
        return max(previous_best - current_best, 0.0) / (abs(previous_best) + 1.0)

    def _history_trend_features(
        self,
        *,
        history_y: np.ndarray,
        incumbent_y: float,
        evals_used: float,
    ) -> list[float]:
        y = self._sanitize_array(history_y).reshape(-1)
        if len(y) == 0:
            return [0.0] * 12

        current_best = float(np.min(y))
        mean_y = float(np.mean(y))
        std_y = float(np.std(y))
        finite_incumbent = float(incumbent_y) if np.isfinite(float(incumbent_y)) else current_best

        def window_improvement(window: int) -> tuple[float, float, float]:
            if len(y) <= window:
                return 0.0, 0.0, 0.0
            previous_best = float(np.min(y[:-window]))
            recent_best = float(np.min(y[-window:]))
            rel = self._safe_relative_improvement(previous_best, min(previous_best, recent_best))
            recent_mean = float(np.mean(y[-window:]))
            recent_std = float(np.std(y[-window:]))
            return rel, recent_mean, recent_std

        imp10, recent10_mean, recent10_std = window_improvement(10)
        imp50, recent50_mean, recent50_std = window_improvement(50)

        best_idx = int(np.argmin(y))
        since_best = float(max(len(y) - 1 - best_idx, 0))
        budget = float(self.config.max_true_evals or max(evals_used, len(y), 1.0))

        return [
            float(len(y)),
            current_best,
            finite_incumbent - current_best,
            mean_y,
            std_y,
            recent10_mean,
            recent10_std,
            imp10,
            recent50_mean,
            recent50_std,
            imp50,
            since_best / max(budget, 1.0),
        ]

    def _candidate_distribution_features(
        self,
        *,
        candidate_x: np.ndarray,
        incumbent_x: np.ndarray,
    ) -> list[float]:
        cand = self._sanitize_array(np.asarray(candidate_x, dtype=np.float32))
        if cand.ndim != 2 or cand.shape[0] == 0:
            return [0.0] * 8

        incumbent = self._sanitize_array(np.asarray(incumbent_x, dtype=np.float32).reshape(-1))
        if incumbent.ndim != 1 or len(incumbent) != cand.shape[1]:
            incumbent = np.zeros((cand.shape[1],), dtype=np.float32)

        deltas = cand - incumbent[None, :]
        distances = np.linalg.norm(deltas, axis=1)
        coord_std = np.std(cand, axis=0)
        delta_abs = np.abs(deltas)

        return [
            float(np.mean(distances)),
            float(np.std(distances)),
            float(np.min(distances)),
            float(np.max(distances)),
            float(np.mean(coord_std)),
            float(np.max(coord_std)),
            float(np.mean(delta_abs)),
            float(np.max(delta_abs)),
        ]

    def _build_context_features(
        self,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
        optimizer_features: np.ndarray,
    ) -> np.ndarray:
        x_dim = x_hist.shape[1] if x_hist.ndim == 2 and x_hist.shape[1] > 0 else 0
        if x_dim == 0:
            x_dim = int(optimizer_features[1]) if len(optimizer_features) > 1 else 0

        extra_dim = 0
        if self.config.include_optimizer_features_in_context:
            extra_dim += len(optimizer_features)
        if self.config.include_ranks:
            extra_dim += 1
        if self.config.include_recency:
            extra_dim += 1

        total_dim = x_dim + extra_dim
        if len(x_hist) == 0:
            return np.zeros((0, total_dim), dtype=np.float32)

        parts: list[np.ndarray] = [x_hist]

        if self.config.include_optimizer_features_in_context:
            repeated_optimizer = np.repeat(optimizer_features[None, :], repeats=len(x_hist), axis=0)
            parts.append(repeated_optimizer)

        if self.config.include_ranks:
            parts.append(self._rank_normalize(y_hist)[:, None])

        if self.config.include_recency:
            parts.append(self._recency_feature(len(y_hist))[:, None])

        return np.concatenate(parts, axis=1).astype(np.float32)

    def _build_context_targets(
        self,
        y_hist: np.ndarray,
        incumbent_y: float,
    ) -> np.ndarray:
        y = y_hist.astype(np.float32).copy()

        if np.isfinite(incumbent_y):
            y = y - float(incumbent_y)

        if self.config.normalize_targets and len(y) > 1:
            mean = float(np.mean(y))
            std = float(np.std(y))
            if std > 1e-12:
                y = (y - mean) / std
            else:
                y = y - mean

        return y[:, None].astype(np.float32)

    def _build_candidate_features(
        self,
        candidate_x: np.ndarray,
        incumbent_x: np.ndarray,
        optimizer_features: np.ndarray,
    ) -> np.ndarray:
        if candidate_x.ndim != 2:
            raise ValueError(f"candidate_x must have shape [Q, D], got {candidate_x.shape}")

        q, d = candidate_x.shape
        extra_dim = d + len(optimizer_features)
        total_dim = d + extra_dim

        if q == 0:
            return np.zeros((0, total_dim), dtype=np.float32)

        if incumbent_x.ndim != 1 or len(incumbent_x) != d:
            incumbent_x = np.zeros(d, dtype=np.float32)

        deltas = candidate_x - incumbent_x[None, :]
        repeated_optimizer = np.repeat(optimizer_features[None, :], repeats=q, axis=0)

        return np.concatenate(
            [
                candidate_x,
                deltas.astype(np.float32),
                repeated_optimizer.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _rank_normalize(y: np.ndarray) -> np.ndarray:
        if len(y) <= 1:
            return np.zeros_like(y, dtype=np.float32)
        order = np.argsort(np.argsort(y))
        return (order / (len(y) - 1)).astype(np.float32)

    @staticmethod
    def _recency_feature(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones((n,), dtype=np.float32)
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)


class PFNDecisionModel(DecisionModel):
    """Set-based PFN surrogate bundle selector."""

    def __init__(
        self,
        backbone: PFNBackboneProtocol | None = None,
        config: PFNDecisionConfig | None = None,
    ) -> None:
        self.config = config or PFNDecisionConfig()
        self.featurizer = PFNStateFeaturizer(self.config)

        if torch is None:
            raise ImportError("PFNDecisionModel requires PyTorch.")

        self.device = torch.device(self.config.device)
        self.dtype = self._resolve_dtype(self.config.dtype)

        self.backbone = backbone
        if self.backbone is None:
            if self.config.checkpoint_path is None:
                raise ValueError("Either `backbone` or `checkpoint_path` must be provided.")
            self.backbone = self._load_backbone(self.config.checkpoint_path)

        if hasattr(self.backbone, "to"):
            self.backbone = self.backbone.to(self.device)
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()

    def score(
        self,
        state: GenerationState,
        surrogate_names: list[str],
    ) -> SurrogateDecision:
        if not surrogate_names:
            raise ValueError("surrogate_names must not be empty.")

        context_x_np, context_y_np, candidate_x_np, action_ids_np, debug = self.featurizer.build(
            state=state,
            surrogate_names=surrogate_names,
        )

        context_x = self._to_tensor(context_x_np)[None, :, :]
        context_y = self._to_tensor(context_y_np)[None, :, :]
        candidate_x = self._to_tensor(candidate_x_np)[None, :, :]
        action_ids = torch.as_tensor(action_ids_np[None, :], device=self.device, dtype=torch.long)

        context_mask = None
        if context_x.shape[1] > 0:
            context_mask = torch.ones((1, context_x.shape[1]), device=self.device, dtype=self.dtype)

        candidate_mask = None
        if candidate_x.shape[1] > 0:
            candidate_mask = torch.ones((1, candidate_x.shape[1]), device=self.device, dtype=self.dtype)

        with torch.no_grad():
            scores_t = self.backbone(
                context_x=context_x,
                context_y=context_y,
                candidate_x=candidate_x,
                action_ids=action_ids,
                context_mask=context_mask,
                candidate_mask=candidate_mask,
            )

        scores = scores_t.detach().cpu().numpy().reshape(-1)

        if len(scores) != len(surrogate_names):
            raise ValueError(f"Backbone returned {len(scores)} scores, expected {len(surrogate_names)}.")
        if not np.isfinite(scores).all():
            raise ValueError(f"PFN backbone returned non-finite scores: {scores}")

        scaled_scores = scores / max(float(self.config.temperature), 1e-8)
        goodness = {name: float(score) for name, score in zip(surrogate_names, scaled_scores, strict=True)}

        score_arr = np.asarray(scaled_scores, dtype=np.float32)
        best_idx = int(np.argmax(score_arr))
        best_score = float(score_arr[best_idx])
        near_best = np.flatnonzero(score_arr >= best_score - float(self.config.tie_margin))
        chosen_idx = int(near_best[0])
        chosen_name = surrogate_names[chosen_idx]

        ordered = sorted(goodness.items(), key=lambda kv: kv[1], reverse=True)
        top1_top2_margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0

        probs = 1.0 / (1.0 + np.exp(-scores))
        binary_accept = {name: int(prob >= 0.5) for name, prob in zip(surrogate_names, probs, strict=True)}

        return SurrogateDecision(
            goodness=goodness,
            chosen_surrogate_name=chosen_name,
            metadata={
                "model_type": "set_conditioned_pfn_decision_model",
                "ordered_goodness": ordered,
                "top1_top2_margin": float(top1_top2_margin),
                "binary_accept": binary_accept,
                "debug": debug,
            },
        )

    def _load_backbone(self, checkpoint_path: str) -> PFNBackboneProtocol:
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=True)

        if "model" in checkpoint:
            model = checkpoint["model"]
            if hasattr(model, "to"):
                model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            return model

        if "model_state_dict" in checkpoint:
            from ..stubs.decision_models import PFNBackboneConfig, SetConditionedPFNBackbone

            context_dim = int(checkpoint["context_dim"])
            candidate_dim = int(checkpoint.get("candidate_dim", checkpoint.get("query_dim")))

            raw_backbone_config = checkpoint.get("backbone_config", {})
            if raw_backbone_config is None:
                raw_backbone_config = {}
            if not isinstance(raw_backbone_config, dict):
                raise ValueError("`backbone_config` in checkpoint must be a dict if present.")

            backbone_config = PFNBackboneConfig(
                hidden_dim=int(raw_backbone_config.get("hidden_dim", 256)),
                num_heads=int(raw_backbone_config.get("num_heads", 8)),
                num_context_layers=int(raw_backbone_config.get("num_context_layers", 4)),
                num_candidate_layers=int(
                    raw_backbone_config.get(
                        "num_candidate_layers",
                        raw_backbone_config.get("num_query_layers", 2),
                    )
                ),
                num_action_layers=int(raw_backbone_config.get("num_action_layers", 2)),
                ff_multiplier=int(raw_backbone_config.get("ff_multiplier", 4)),
                dropout=float(raw_backbone_config.get("dropout", 0.1)),
                activation=str(raw_backbone_config.get("activation", "gelu")),
                use_type_embeddings=bool(raw_backbone_config.get("use_type_embeddings", True)),
                max_action_tokens=int(raw_backbone_config.get("max_action_tokens", 64)),
                use_action_features=bool(raw_backbone_config.get("use_action_features", False)),
            )

            model = SetConditionedPFNBackbone(
                context_dim=context_dim,
                candidate_dim=candidate_dim,
                config=backbone_config,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            return model

        raise ValueError(
            "Checkpoint must contain either `model` or `model_state_dict` + `context_dim` + `candidate_dim`."
        )

    def _to_tensor(self, array: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(array, device=self.device, dtype=self.dtype)

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> "torch.dtype":
        mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        return mapping[dtype_name]
