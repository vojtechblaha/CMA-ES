from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from ..decision.pfn_model import PFNDecisionConfig, PFNStateFeaturizer
from ..stubs.decision_models import PFNBackboneConfig, SetConditionedPFNBackbone
from ..types import GenerationState


def _is_finite_scalar(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _is_valid_record(
    record: dict[str, Any],
    surrogate_names: Sequence[str],
) -> bool:
    history_x = np.asarray(record.get("history_x", []), dtype=np.float32)
    history_y = np.asarray(record.get("history_y", []), dtype=np.float32)
    candidate_x = np.asarray(record.get("candidate_x", []), dtype=np.float32)
    incumbent_x = np.asarray(record.get("incumbent_x", []), dtype=np.float32)
    dimension = int(record.get("dimension", 0))
    surrogate_scores = record.get("surrogate_scores", {})

    if history_x.ndim != 2:
        return False
    if history_y.ndim != 1:
        return False
    if candidate_x.ndim != 2:
        return False
    if len(history_x) != len(history_y):
        return False
    if len(history_y) == 0:
        return False
    if history_x.shape[1] != dimension:
        return False
    if candidate_x.shape[1] != dimension:
        return False
    if incumbent_x.ndim != 1 or len(incumbent_x) != dimension:
        return False
    if not _is_finite_scalar(record.get("incumbent_y", np.nan)):
        return False
    if not isinstance(surrogate_scores, dict):
        return False

    for name in surrogate_names:
        if name not in surrogate_scores:
            return False
        if not _is_finite_scalar(surrogate_scores[name]):
            return False

    return True


def _build_multilabel_targets(
    surrogate_scores: dict[str, float],
    surrogate_names: Sequence[str],
    *,
    comparable_margin: float = 0.10,
    min_positive: int = 1,
) -> np.ndarray:
    scores = np.asarray(
        [_safe_float(surrogate_scores[name]) for name in surrogate_names],
        dtype=np.float32,
    )

    best = float(np.max(scores))
    worst = float(np.min(scores))
    spread = best - worst

    if spread < 1e-12:
        labels = np.ones_like(scores, dtype=np.float32)
    else:
        threshold = comparable_margin * spread
        labels = ((best - scores) <= threshold).astype(np.float32)

    if int(labels.sum()) < min_positive:
        labels[int(np.argmax(scores))] = 1.0

    return labels.astype(np.float32)


def _score_vector(
    surrogate_scores: dict[str, float],
    surrogate_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [_safe_float(surrogate_scores[name]) for name in surrogate_names],
        dtype=np.float32,
    )


def _normalize_scores_for_soft_targets(
    scores: np.ndarray,
    *,
    transform: str,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if transform == "identity":
        values = scores.copy()
    elif transform == "log1p":
        values = np.log1p(np.maximum(scores, 0.0)).astype(np.float32)
    elif transform == "rank":
        order = np.argsort(np.argsort(scores))
        if len(scores) <= 1:
            return np.zeros_like(scores, dtype=np.float32)
        return (order / float(len(scores) - 1)).astype(np.float32)
    elif transform == "minmax":
        values = scores.copy()
    elif transform == "log1p_minmax":
        values = np.log1p(np.maximum(scores, 0.0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown score transform: {transform}")

    if transform.endswith("minmax"):
        worst = float(np.min(values))
        best = float(np.max(values))
        spread = best - worst
        if spread <= 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        values = (values - worst) / spread

    return values.astype(np.float32)


def _build_softmax_targets(
    surrogate_scores: dict[str, float],
    surrogate_names: Sequence[str],
    *,
    temperature: float,
    transform: str,
) -> np.ndarray:
    scores = _score_vector(surrogate_scores, surrogate_names)
    values = _normalize_scores_for_soft_targets(scores, transform=transform)
    temp = max(float(temperature), 1e-8)
    logits = values / temp
    logits = logits - float(np.max(logits))
    probs = np.exp(logits).astype(np.float32)
    total = float(np.sum(probs))
    if total <= 1e-12:
        return np.full_like(probs, fill_value=1.0 / max(len(probs), 1), dtype=np.float32)
    return (probs / total).astype(np.float32)


def _build_training_targets(
    surrogate_scores: dict[str, float],
    surrogate_names: Sequence[str],
    *,
    training_config: "DecisionTrainingConfig",
) -> np.ndarray:
    if training_config.target_mode == "multilabel_bce":
        return _build_multilabel_targets(
            surrogate_scores=surrogate_scores,
            surrogate_names=surrogate_names,
            comparable_margin=training_config.comparable_margin,
            min_positive=training_config.min_positive_labels,
        )
    if training_config.target_mode == "softmax_kl":
        return _build_softmax_targets(
            surrogate_scores=surrogate_scores,
            surrogate_names=surrogate_names,
            temperature=training_config.soft_label_temperature,
            transform=training_config.score_transform,
        )
    raise ValueError(f"Unsupported target_mode: {training_config.target_mode}")


@dataclass(slots=True)
class DecisionTrainingConfig:
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    device: str = "cpu"
    checkpoint_dirname: str = "models"
    max_records: int | None = None
    shuffle_buffer_size: int = 4096
    steps_per_epoch: int | None = None
    seed: int = 0

    comparable_margin: float = 0.10
    min_positive_labels: int = 1
    positive_class_weight: float = 1.0
    target_mode: str = "multilabel_bce"
    soft_label_temperature: float = 0.20
    score_transform: str = "log1p_minmax"
    class_weighting: str = "none"
    class_weight_power: float = 0.5
    class_weight_clip: float = 10.0
    entropy_bonus: float = 0.0
    use_action_features: bool = False
    print_class_stats: bool = True


TrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _record_to_training_sample(
    record: dict[str, Any],
    *,
    surrogate_names: Sequence[str],
    featurizer: PFNStateFeaturizer,
    training_config: DecisionTrainingConfig,
) -> TrainingSample:
    state = GenerationState(
        generation_index=int(record["generation_index"]),
        evaluated_history_x=np.asarray(record["history_x"], dtype=np.float32),
        evaluated_history_y=np.asarray(record["history_y"], dtype=np.float32),
        candidate_x=np.asarray(record["candidate_x"], dtype=np.float32),
        incumbent_x=np.asarray(record["incumbent_x"], dtype=np.float32),
        incumbent_y=float(record["incumbent_y"]),
        optimizer_state=dict(record["optimizer_state"]),
        metadata=dict(record.get("metadata", {})),
    )

    context_x, context_y, candidate_x, action_ids, _ = featurizer.build(
        state=state,
        surrogate_names=list(surrogate_names),
    )

    target_labels = _build_training_targets(
        surrogate_scores=record["surrogate_scores"],
        surrogate_names=surrogate_names,
        training_config=training_config,
    )

    return context_x, context_y, candidate_x, action_ids, target_labels


def _iter_valid_training_records(
    *,
    experiment_root: Path,
    target_dimension: int,
    held_out_function_id: int,
    surrogate_names: Sequence[str],
    max_records: int | None = None,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    skipped_invalid = 0

    for dataset_file in find_dataset_files(experiment_root):
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)

                if int(rec["dimension"]) != target_dimension:
                    continue
                if int(rec["function_id"]) == held_out_function_id:
                    continue

                if not _is_valid_record(rec, surrogate_names):
                    skipped_invalid += 1
                    continue

                yield rec
                yielded += 1

                if max_records is not None and yielded >= max_records:
                    print(
                        f"[train_decision_model] streamed {yielded} valid records (skipped_invalid={skipped_invalid})"
                    )
                    return

    print(f"[train_decision_model] streamed {yielded} valid records (skipped_invalid={skipped_invalid})")


def _shuffle_buffered(
    samples: Iterator[TrainingSample],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[TrainingSample]:
    if buffer_size <= 1:
        yield from samples
        return

    rng = random.Random(seed)
    buffer: list[TrainingSample] = []

    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue

        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = sample

    rng.shuffle(buffer)
    yield from buffer


class StreamingDecisionTrainingDataset(IterableDataset):
    """
    Streams JSONL records and featurizes them on demand.

    Each yielded sample contains:
    - context_x
    - context_y
    - candidate_x
    - action_ids
    - multilabel targets
    """

    def __init__(
        self,
        *,
        experiment_root: Path,
        target_dimension: int,
        held_out_function_id: int,
        surrogate_names: Sequence[str],
        pfn_config: PFNDecisionConfig,
        training_config: DecisionTrainingConfig,
        epoch: int,
    ) -> None:
        self.experiment_root = experiment_root
        self.target_dimension = int(target_dimension)
        self.held_out_function_id = int(held_out_function_id)
        self.surrogate_names = list(surrogate_names)
        self.featurizer = PFNStateFeaturizer(pfn_config)
        self.training_config = training_config
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[TrainingSample]:
        records = _iter_valid_training_records(
            experiment_root=self.experiment_root,
            target_dimension=self.target_dimension,
            held_out_function_id=self.held_out_function_id,
            surrogate_names=self.surrogate_names,
            max_records=self.training_config.max_records,
        )
        samples = (
            _record_to_training_sample(
                record,
                surrogate_names=self.surrogate_names,
                featurizer=self.featurizer,
                training_config=self.training_config,
            )
            for record in records
        )
        yield from _shuffle_buffered(
            samples,
            buffer_size=self.training_config.shuffle_buffer_size,
            seed=self.training_config.seed + self.epoch,
        )


def collate_variable_set_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_x_list, context_y_list, candidate_x_list, action_ids_list, target_list = zip(*batch)

    batch_size = len(batch)
    max_ctx_len = max(cx.shape[0] for cx in context_x_list)
    max_cand_len = max(kx.shape[0] for kx in candidate_x_list)
    context_dim = context_x_list[0].shape[1]
    candidate_dim = candidate_x_list[0].shape[1]
    num_actions = action_ids_list[0].shape[0]

    context_x = np.zeros((batch_size, max_ctx_len, context_dim), dtype=np.float32)
    context_y = np.zeros((batch_size, max_ctx_len, 1), dtype=np.float32)
    context_mask = np.zeros((batch_size, max_ctx_len), dtype=np.float32)

    candidate_x = np.zeros((batch_size, max_cand_len, candidate_dim), dtype=np.float32)
    candidate_mask = np.zeros((batch_size, max_cand_len), dtype=np.float32)

    action_ids = np.zeros((batch_size, num_actions), dtype=np.int64)
    targets = np.zeros((batch_size, num_actions), dtype=np.float32)

    for i, (cx, cy, kx, aids, tgt) in enumerate(batch):
        n_ctx = cx.shape[0]
        n_cand = kx.shape[0]

        if n_ctx > 0:
            context_x[i, :n_ctx] = cx
            context_y[i, :n_ctx] = cy
            context_mask[i, :n_ctx] = 1.0

        if n_cand > 0:
            candidate_x[i, :n_cand] = kx
            candidate_mask[i, :n_cand] = 1.0

        action_ids[i] = aids
        targets[i] = tgt

    return (
        torch.from_numpy(context_x),
        torch.from_numpy(context_y),
        torch.from_numpy(context_mask),
        torch.from_numpy(candidate_x),
        torch.from_numpy(candidate_mask),
        torch.from_numpy(action_ids),
        torch.from_numpy(targets),
    )


def find_dataset_files(experiment_root: Path) -> list[Path]:
    return sorted(experiment_root.glob("f*_i*_d*_s*/dataset.jsonl"))


def load_training_records(
    experiment_root: Path,
    *,
    target_dimension: int,
    held_out_function_id: int,
    surrogate_names: Sequence[str],
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_invalid = 0

    for dataset_file in find_dataset_files(experiment_root):
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)

                if int(rec["dimension"]) != target_dimension:
                    continue
                if int(rec["function_id"]) == held_out_function_id:
                    continue

                if not _is_valid_record(rec, surrogate_names):
                    skipped_invalid += 1
                    continue

                records.append(rec)

                if max_records is not None and len(records) >= max_records:
                    print(
                        f"[train_decision_model] loaded {len(records)} valid records "
                        f"(skipped_invalid={skipped_invalid})"
                    )
                    return records

    print(f"[train_decision_model] loaded {len(records)} valid records (skipped_invalid={skipped_invalid})")
    return records


def infer_model_dims_from_records(
    records: list[dict[str, Any]],
    surrogate_names: Sequence[str],
    pfn_config: PFNDecisionConfig,
) -> tuple[int, int]:
    if not records:
        raise ValueError("Cannot infer PFN dimensions from an empty record list.")

    sample = records[0]
    state = GenerationState(
        generation_index=int(sample["generation_index"]),
        evaluated_history_x=np.asarray(sample["history_x"], dtype=np.float32),
        evaluated_history_y=np.asarray(sample["history_y"], dtype=np.float32),
        candidate_x=np.asarray(sample["candidate_x"], dtype=np.float32),
        incumbent_x=np.asarray(sample["incumbent_x"], dtype=np.float32),
        incumbent_y=float(sample["incumbent_y"]),
        optimizer_state=dict(sample["optimizer_state"]),
        metadata=dict(sample.get("metadata", {})),
    )

    featurizer = PFNStateFeaturizer(pfn_config)
    context_x, _, candidate_x, _, _ = featurizer.build(
        state=state,
        surrogate_names=list(surrogate_names),
    )
    return int(context_x.shape[1]), int(candidate_x.shape[1])


def build_checkpoint_path(
    experiment_root: Path,
    *,
    dimension: int,
    held_out_function_id: int,
    dirname: str,
) -> Path:
    model_dir = experiment_root / dirname
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"decision_model_dim{dimension}_heldout_f{held_out_function_id}.pt"


def _print_target_statistics(
    records: Sequence[dict[str, Any]],
    surrogate_names: Sequence[str],
    training_config: DecisionTrainingConfig,
) -> None:
    positive_counts = {name: 0 for name in surrogate_names}
    oracle_counts = {name: 0 for name in surrogate_names}

    for rec in records:
        scores = rec["surrogate_scores"]
        labels = _build_multilabel_targets(
            surrogate_scores=scores,
            surrogate_names=surrogate_names,
            comparable_margin=training_config.comparable_margin,
            min_positive=training_config.min_positive_labels,
        )

        best_idx = int(np.argmax(np.asarray([float(scores[name]) for name in surrogate_names], dtype=np.float32)))
        oracle_counts[surrogate_names[best_idx]] += 1

        for name, label in zip(surrogate_names, labels, strict=True):
            positive_counts[name] += int(label > 0.5)

    print("[train_decision_model] oracle best counts:")
    for name in surrogate_names:
        print(f"  - {name}: {oracle_counts[name]}")

    print("[train_decision_model] multi-label positive counts:")
    for name in surrogate_names:
        print(f"  - {name}: {positive_counts[name]}")


@dataclass(slots=True)
class TargetStatistics:
    total: int
    oracle_counts: dict[str, int]
    positive_counts: dict[str, int]


def _collect_streaming_target_statistics(
    *,
    experiment_root: Path,
    target_dimension: int,
    held_out_function_id: int,
    surrogate_names: Sequence[str],
    training_config: DecisionTrainingConfig,
) -> TargetStatistics:
    positive_counts = {name: 0 for name in surrogate_names}
    oracle_counts = {name: 0 for name in surrogate_names}
    total = 0

    for rec in _iter_valid_training_records(
        experiment_root=experiment_root,
        target_dimension=target_dimension,
        held_out_function_id=held_out_function_id,
        surrogate_names=surrogate_names,
        max_records=training_config.max_records,
    ):
        scores = rec["surrogate_scores"]
        labels = _build_multilabel_targets(
            surrogate_scores=scores,
            surrogate_names=surrogate_names,
            comparable_margin=training_config.comparable_margin,
            min_positive=training_config.min_positive_labels,
        )

        best_idx = int(np.argmax(np.asarray([float(scores[name]) for name in surrogate_names], dtype=np.float32)))
        oracle_counts[surrogate_names[best_idx]] += 1

        for name, label in zip(surrogate_names, labels, strict=True):
            positive_counts[name] += int(label > 0.5)

        total += 1

    return TargetStatistics(
        total=total,
        oracle_counts=oracle_counts,
        positive_counts=positive_counts,
    )


def _print_target_statistics_from_counts(
    stats: TargetStatistics,
    surrogate_names: Sequence[str],
) -> None:
    print(f"[train_decision_model] target statistics from {stats.total} streamed records:")
    print("[train_decision_model] oracle best counts:")
    for name in surrogate_names:
        print(f"  - {name}: {stats.oracle_counts[name]}")

    print("[train_decision_model] multi-label positive counts:")
    for name in surrogate_names:
        print(f"  - {name}: {stats.positive_counts[name]}")


def _compute_class_weight_vector(
    *,
    stats: TargetStatistics | None,
    surrogate_names: Sequence[str],
    training_config: DecisionTrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    mode = training_config.class_weighting
    if mode == "none" or stats is None:
        return torch.ones((len(surrogate_names),), dtype=torch.float32, device=device)

    if mode == "inverse_oracle":
        counts = np.asarray([max(stats.oracle_counts[name], 1) for name in surrogate_names], dtype=np.float64)
    elif mode == "inverse_positive":
        counts = np.asarray([max(stats.positive_counts[name], 1) for name in surrogate_names], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported class_weighting: {mode}")

    mean_count = float(np.mean(counts))
    weights = np.power(mean_count / counts, float(training_config.class_weight_power))
    clip = float(training_config.class_weight_clip)
    if clip > 0:
        weights = np.clip(weights, 1.0 / clip, clip)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _softmax_kl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    entropy_bonus: float,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    weighted_targets = targets * class_weights[None, :]
    denom = weighted_targets.sum(dim=-1).clamp_min(1e-12)
    loss_per_sample = -(weighted_targets * log_probs).sum(dim=-1) / denom
    loss = loss_per_sample.mean()

    if entropy_bonus > 0.0:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = loss - float(entropy_bonus) * entropy

    return loss


def _first_valid_training_record(
    *,
    experiment_root: Path,
    target_dimension: int,
    held_out_function_id: int,
    surrogate_names: Sequence[str],
) -> dict[str, Any]:
    for record in _iter_valid_training_records(
        experiment_root=experiment_root,
        target_dimension=target_dimension,
        held_out_function_id=held_out_function_id,
        surrogate_names=surrogate_names,
        max_records=1,
    ):
        return record

    raise ValueError(
        f"No training records found for dimension={target_dimension} "
        f"excluding function_id={held_out_function_id} in {experiment_root}."
    )


def train_decision_model(
    *,
    experiment_root: Path,
    target_dimension: int,
    held_out_function_id: int,
    surrogate_names: Sequence[str],
    pfn_config: PFNDecisionConfig,
    training_config: DecisionTrainingConfig,
) -> Path:
    first_record = _first_valid_training_record(
        experiment_root=experiment_root,
        target_dimension=target_dimension,
        held_out_function_id=held_out_function_id,
        surrogate_names=surrogate_names,
    )

    target_stats: TargetStatistics | None = None
    if training_config.print_class_stats or training_config.class_weighting != "none":
        target_stats = _collect_streaming_target_statistics(
            experiment_root=experiment_root,
            target_dimension=target_dimension,
            held_out_function_id=held_out_function_id,
            surrogate_names=surrogate_names,
            training_config=training_config,
        )
        if training_config.print_class_stats:
            _print_target_statistics_from_counts(target_stats, surrogate_names)

    context_dim, candidate_dim = infer_model_dims_from_records(
        records=[first_record],
        surrogate_names=surrogate_names,
        pfn_config=pfn_config,
    )

    device = torch.device(training_config.device)
    backbone = SetConditionedPFNBackbone(
        context_dim=context_dim,
        candidate_dim=candidate_dim,
        config=PFNBackboneConfig(
            hidden_dim=training_config.hidden_dim,
            num_heads=8,
            num_context_layers=4,
            num_candidate_layers=2,
            num_action_layers=2,
            ff_multiplier=4,
            dropout=0.1,
            activation="gelu",
            use_type_embeddings=True,
            max_action_tokens=max(64, len(surrogate_names)),
            use_action_features=training_config.use_action_features,
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    class_weights = _compute_class_weight_vector(
        stats=target_stats,
        surrogate_names=surrogate_names,
        training_config=training_config,
        device=device,
    )
    if training_config.class_weighting != "none":
        print("[train_decision_model] class weights:")
        for name, weight in zip(surrogate_names, class_weights.detach().cpu().numpy().tolist(), strict=True):
            print(f"  - {name}: {weight:.4f}")

    if training_config.target_mode == "multilabel_bce":
        if training_config.class_weighting in {"inverse_oracle", "inverse_positive"}:
            pos_weight = class_weights * float(training_config.positive_class_weight)
        else:
            pos_weight = torch.full(
                (len(surrogate_names),),
                fill_value=float(training_config.positive_class_weight),
                dtype=torch.float32,
                device=device,
            )
        criterion: nn.Module | None = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif training_config.target_mode == "softmax_kl":
        criterion = None
    else:
        raise ValueError(f"Unsupported target_mode: {training_config.target_mode}")

    backbone.train()
    for epoch in range(training_config.epochs):
        dataset = StreamingDecisionTrainingDataset(
            experiment_root=experiment_root,
            target_dimension=target_dimension,
            held_out_function_id=held_out_function_id,
            surrogate_names=surrogate_names,
            pfn_config=pfn_config,
            training_config=training_config,
            epoch=epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            collate_fn=collate_variable_set_batch,
        )

        epoch_loss = 0.0
        num_batches = 0

        for (
            context_x,
            context_y,
            context_mask,
            candidate_x,
            candidate_mask,
            action_ids,
            targets,
        ) in loader:
            context_x = context_x.to(device)
            context_y = context_y.to(device)
            context_mask = context_mask.to(device)
            candidate_x = candidate_x.to(device)
            candidate_mask = candidate_mask.to(device)
            action_ids = action_ids.to(device)
            targets = targets.to(device)

            logits = backbone(
                context_x=context_x,
                context_y=context_y,
                candidate_x=candidate_x,
                action_ids=action_ids,
                context_mask=context_mask,
                candidate_mask=candidate_mask,
            )

            if training_config.target_mode == "softmax_kl":
                loss = _softmax_kl_loss(
                    logits,
                    targets,
                    class_weights=class_weights,
                    entropy_bonus=training_config.entropy_bonus,
                )
            else:
                assert criterion is not None
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

            if training_config.steps_per_epoch is not None and num_batches >= training_config.steps_per_epoch:
                break

        if num_batches == 0:
            raise ValueError(
                f"No training batches produced for dimension={target_dimension} "
                f"excluding function_id={held_out_function_id} in {experiment_root}."
            )

        mean_loss = epoch_loss / max(num_batches, 1)
        print(
            f"[train_decision_model] epoch={epoch + 1}/{training_config.epochs} "
            f"batches={num_batches} loss={mean_loss:.6f}"
        )

    checkpoint_path = build_checkpoint_path(
        experiment_root=experiment_root,
        dimension=target_dimension,
        held_out_function_id=held_out_function_id,
        dirname=training_config.checkpoint_dirname,
    )

    torch.save(
        {
            "model_state_dict": backbone.state_dict(),
            "context_dim": context_dim,
            "candidate_dim": candidate_dim,
            "surrogate_names": list(surrogate_names),
            "dimension": target_dimension,
            "held_out_function_id": held_out_function_id,
            "training_mode": training_config.target_mode,
            "target_construction": {
                "comparable_margin": training_config.comparable_margin,
                "min_positive_labels": training_config.min_positive_labels,
                "soft_label_temperature": training_config.soft_label_temperature,
                "score_transform": training_config.score_transform,
                "class_weighting": training_config.class_weighting,
                "class_weight_power": training_config.class_weight_power,
                "class_weight_clip": training_config.class_weight_clip,
                "entropy_bonus": training_config.entropy_bonus,
                "use_action_features": training_config.use_action_features,
            },
            "pfn_config": asdict(pfn_config),
            "backbone_type": "set_conditioned_pfn",
            "backbone_config": {
                "hidden_dim": backbone.config.hidden_dim,
                "num_heads": backbone.config.num_heads,
                "num_context_layers": backbone.config.num_context_layers,
                "num_candidate_layers": backbone.config.num_candidate_layers,
                "num_action_layers": backbone.config.num_action_layers,
                "ff_multiplier": backbone.config.ff_multiplier,
                "dropout": backbone.config.dropout,
                "activation": backbone.config.activation,
                "use_type_embeddings": backbone.config.use_type_embeddings,
                "max_action_tokens": backbone.config.max_action_tokens,
                "use_action_features": backbone.config.use_action_features,
            },
        },
        checkpoint_path,
    )

    print(f"[train_decision_model] saved checkpoint to {checkpoint_path}")
    return checkpoint_path
