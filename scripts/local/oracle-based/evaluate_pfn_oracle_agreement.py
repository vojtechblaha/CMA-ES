from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pfn_cmaes.decision.pfn_model import PFNDecisionConfig, PFNDecisionModel
from pfn_cmaes.types import GenerationState

RUN_RE = re.compile(r"f(?P<function_id>\d+)_i(?P<instance_id>\d+)_d(?P<dimension>\d+)_s(?P<seed>\d+)")

DEFAULT_SURROGATE_NAMES = [
    "lq_linear_top50",
    "lq_quadratic_top30",
    "gp_matern_dts",
    "gp_matern_unc",
    "rf_lifelength",
    "svm_rank_top50",
    "real_only",
]


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_run_dir(path: Path) -> dict[str, int] | None:
    match = RUN_RE.fullmatch(path.name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def dataset_files_for_function(experiment_root: Path, function_id: int, dimension: int | None) -> list[Path]:
    files: list[Path] = []
    for dataset_file in sorted(experiment_root.glob(f"f{function_id}_i*_d*_s*/dataset.jsonl")):
        fields = parse_run_dir(dataset_file.parent)
        if fields is None:
            continue
        if dimension is not None and fields["dimension"] != dimension:
            continue
        files.append(dataset_file)
    return files


def checkpoint_path(models_dir: Path, dimension: int, function_id: int) -> Path:
    return models_dir / f"decision_model_dim{dimension}_heldout_f{function_id}.pt"


def load_model(checkpoint: Path, device: str) -> PFNDecisionModel:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to evaluate PFN checkpoints.") from exc

    meta = torch.load(checkpoint, map_location=device, weights_only=True)
    pfn_config_dict = dict(meta.get("pfn_config", {}))
    valid_pfn_keys = {field.name for field in fields(PFNDecisionConfig)}
    pfn_kwargs = {key: value for key, value in pfn_config_dict.items() if key in valid_pfn_keys}
    pfn_kwargs.update({"checkpoint_path": str(checkpoint), "device": device})
    pfn_config = PFNDecisionConfig(**pfn_kwargs)
    return PFNDecisionModel(config=pfn_config)


def valid_scores(record: dict[str, Any], surrogate_names: list[str]) -> dict[str, float] | None:
    scores = record.get("surrogate_scores") or record.get("dataset_scores") or {}
    if not isinstance(scores, dict):
        return None

    out: dict[str, float] = {}
    for name in surrogate_names:
        if name not in scores:
            return None
        value = finite_float(scores[name])
        if value is None:
            return None
        out[name] = value
    return out


def oracle_names(
    scores: dict[str, float],
    surrogate_names: list[str],
    *,
    comparable_margin: float,
) -> tuple[str, list[str], float]:
    values = np.asarray([scores[name] for name in surrogate_names], dtype=np.float64)
    best_idx = int(np.argmax(values))
    best_name = surrogate_names[best_idx]

    best = float(np.max(values))
    worst = float(np.min(values))
    spread = best - worst
    if spread < 1e-12:
        acceptable = list(surrogate_names)
    else:
        threshold = comparable_margin * spread
        acceptable = [
            name for name, score in zip(surrogate_names, values, strict=True) if (best - float(score)) <= threshold
        ]
    return best_name, acceptable, spread


def record_to_state(record: dict[str, Any]) -> GenerationState:
    return GenerationState(
        generation_index=int(record["generation_index"]),
        evaluated_history_x=np.asarray(record["history_x"], dtype=np.float32),
        evaluated_history_y=np.asarray(record["history_y"], dtype=np.float32),
        candidate_x=np.asarray(record["candidate_x"], dtype=np.float32),
        incumbent_x=np.asarray(record["incumbent_x"], dtype=np.float32),
        incumbent_y=float(record["incumbent_y"]),
        optimizer_state=dict(record.get("optimizer_state", {})),
        metadata=dict(record.get("metadata", {})),
    )


def rank_of_name(ordered_names: list[str], target: str) -> int:
    try:
        return ordered_names.index(target) + 1
    except ValueError:
        return len(ordered_names) + 1


def generation_bin(generation_index: int, width: int) -> str:
    start = (generation_index // width) * width
    end = start + width - 1
    return f"{start}-{end}"


def update_counter_table(
    table: dict[str, Counter[str]],
    row_key: str,
    col_key: str,
    amount: int = 1,
) -> None:
    table[row_key][col_key] += amount


def write_confusion_csv(path: Path, table: dict[str, Counter[str]], surrogate_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["oracle"] + surrogate_names)
        writer.writeheader()
        for oracle in surrogate_names:
            row = {"oracle": oracle}
            for predicted in surrogate_names:
                row[predicted] = table[oracle][predicted]
            writer.writerow(row)


def write_distribution_csv(
    path: Path,
    oracle_counts: Counter[str],
    predicted_counts: Counter[str],
    surrogate_names: list[str],
) -> None:
    total = sum(oracle_counts.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "surrogate",
                "oracle_count",
                "oracle_pct",
                "pfn_count",
                "pfn_pct",
                "pfn_minus_oracle_pct",
            ],
        )
        writer.writeheader()
        for name in surrogate_names:
            oracle_pct = 100.0 * oracle_counts[name] / total if total else 0.0
            pfn_pct = 100.0 * predicted_counts[name] / total if total else 0.0
            writer.writerow(
                {
                    "surrogate": name,
                    "oracle_count": oracle_counts[name],
                    "oracle_pct": round(oracle_pct, 4),
                    "pfn_count": predicted_counts[name],
                    "pfn_pct": round(pfn_pct, 4),
                    "pfn_minus_oracle_pct": round(pfn_pct - oracle_pct, 4),
                }
            )


def write_group_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(group: str, stats: Counter[str]) -> dict[str, Any]:
    total = stats["total"]
    return {
        "group": group,
        "records": total,
        "top1_accuracy": round(stats["top1"] / total, 6) if total else 0.0,
        "acceptable_accuracy": round(stats["acceptable"] / total, 6) if total else 0.0,
        "top2_accuracy": round(stats["top2"] / total, 6) if total else 0.0,
        "top3_accuracy": round(stats["top3"] / total, 6) if total else 0.0,
        "mean_reciprocal_rank": round(stats["mrr_sum"] / total, 6) if total else 0.0,
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def evaluate_function(
    *,
    experiment_root: Path,
    models_dir: Path,
    out_dir: Path,
    function_id: int,
    dimension: int,
    device: str,
    surrogate_names: list[str],
    comparable_margin: float,
    generation_bin_width: int,
    max_records: int | None,
    write_mistakes: int,
) -> dict[str, Any]:
    datasets = dataset_files_for_function(experiment_root, function_id, dimension)
    checkpoint = checkpoint_path(models_dir, dimension, function_id)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not datasets:
        raise FileNotFoundError(f"No dataset.jsonl files found for f{function_id}, dimension {dimension}")

    print(f"[f{function_id}] loading {checkpoint}")
    model = load_model(checkpoint, device=device)

    stats: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    by_instance: dict[str, Counter[str]] = defaultdict(Counter)
    by_generation_bin: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    oracle_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    mistakes: list[dict[str, Any]] = []

    for dataset_file in datasets:
        fields = parse_run_dir(dataset_file.parent) or {}
        instance_key = str(fields.get("instance_id", "unknown"))

        for record in iter_jsonl(dataset_file):
            if max_records is not None and stats["total"] >= max_records:
                break

            scores = valid_scores(record, surrogate_names)
            if scores is None:
                skipped["invalid_scores"] += 1
                continue

            try:
                state = record_to_state(record)
            except Exception:
                skipped["invalid_state"] += 1
                continue

            try:
                decision = model.score(state=state, surrogate_names=surrogate_names)
            except Exception as exc:
                skipped[f"model_error:{type(exc).__name__}"] += 1
                continue

            oracle, acceptable, spread = oracle_names(
                scores,
                surrogate_names,
                comparable_margin=comparable_margin,
            )
            predicted = decision.chosen_surrogate_name
            ordered_names = [name for name, _ in decision.metadata.get("ordered_goodness", [])]
            if not ordered_names:
                ordered_names = sorted(decision.goodness, key=decision.goodness.get, reverse=True)

            oracle_rank = rank_of_name(ordered_names, oracle)
            top1 = predicted == oracle
            acceptable_hit = predicted in set(acceptable)

            stats["total"] += 1
            stats["top1"] += int(top1)
            stats["acceptable"] += int(acceptable_hit)
            stats["top2"] += int(oracle_rank <= 2)
            stats["top3"] += int(oracle_rank <= 3)
            stats["mrr_sum"] += 1.0 / oracle_rank

            by_instance[instance_key]["total"] += 1
            by_instance[instance_key]["top1"] += int(top1)
            by_instance[instance_key]["acceptable"] += int(acceptable_hit)
            by_instance[instance_key]["top2"] += int(oracle_rank <= 2)
            by_instance[instance_key]["top3"] += int(oracle_rank <= 3)
            by_instance[instance_key]["mrr_sum"] += 1.0 / oracle_rank

            bin_key = generation_bin(int(record["generation_index"]), generation_bin_width)
            by_generation_bin[bin_key]["total"] += 1
            by_generation_bin[bin_key]["top1"] += int(top1)
            by_generation_bin[bin_key]["acceptable"] += int(acceptable_hit)
            by_generation_bin[bin_key]["top2"] += int(oracle_rank <= 2)
            by_generation_bin[bin_key]["top3"] += int(oracle_rank <= 3)
            by_generation_bin[bin_key]["mrr_sum"] += 1.0 / oracle_rank

            update_counter_table(confusion, oracle, predicted)
            oracle_counts[oracle] += 1
            predicted_counts[predicted] += 1

            if not top1 and len(mistakes) < write_mistakes:
                mistakes.append(
                    {
                        "run_id": dataset_file.parent.name,
                        "generation_index": int(record["generation_index"]),
                        "oracle": oracle,
                        "acceptable_oracles": acceptable,
                        "predicted": predicted,
                        "oracle_rank_in_pfn_scores": oracle_rank,
                        "oracle_score_spread": spread,
                        "surrogate_scores": scores,
                        "pfn_goodness": decision.goodness,
                    }
                )

        if max_records is not None and stats["total"] >= max_records:
            break

    function_dir = out_dir / f"f{function_id}"
    function_dir.mkdir(parents=True, exist_ok=True)
    write_confusion_csv(function_dir / "confusion_oracle_vs_pfn.csv", confusion, surrogate_names)
    write_distribution_csv(
        function_dir / "oracle_vs_pfn_distribution.csv",
        oracle_counts,
        predicted_counts,
        surrogate_names,
    )
    instance_keys = sorted(
        by_instance,
        key=lambda x: int(x) if x.isdigit() else x,
    )
    write_group_metrics_csv(
        function_dir / "metrics_by_instance.csv",
        [metric_row(instance, by_instance[instance]) for instance in instance_keys],
    )
    generation_bin_keys = sorted(
        by_generation_bin,
        key=lambda x: int(x.split("-", 1)[0]),
    )
    write_group_metrics_csv(
        function_dir / "metrics_by_generation_bin.csv",
        [metric_row(bin_key, by_generation_bin[bin_key]) for bin_key in generation_bin_keys],
    )
    if mistakes:
        with (function_dir / "sample_mistakes.json").open("w", encoding="utf-8") as handle:
            json.dump(mistakes, handle, indent=2)

    summary = metric_row(f"f{function_id}", stats)
    summary.update(
        {
            "function_id": function_id,
            "dimension": dimension,
            "dataset_files": len(datasets),
            "checkpoint": str(checkpoint),
            "skipped": dict(skipped),
            "oracle_distribution": dict(oracle_counts),
            "pfn_distribution": dict(predicted_counts),
        }
    )
    with (function_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[f{function_id}] records={summary['records']} "
        f"top1={summary['top1_accuracy']:.4f} "
        f"acceptable={summary['acceptable_accuracy']:.4f} "
        f"top3={summary['top3_accuracy']:.4f} "
        f"mrr={summary['mean_reciprocal_rank']:.4f} "
        f"skipped={dict(skipped)}"
    )
    return summary


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "function_id",
        "dimension",
        "records",
        "top1_accuracy",
        "acceptable_accuracy",
        "top2_accuracy",
        "top3_accuracy",
        "mean_reciprocal_rank",
        "dataset_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda row: int(row["function_id"])):
            writer.writerow({key: summary.get(key, "") for key in fieldnames})


def collect_existing_summaries(out_dir: Path) -> None:
    summaries: list[dict[str, Any]] = []
    for summary_file in sorted(out_dir.glob("f*/summary.json")):
        with summary_file.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    if not summaries:
        raise FileNotFoundError(f"No f*/summary.json files found under {out_dir}")
    write_summary_csv(out_dir / "summary_by_function.csv", summaries)
    with (out_dir / "summary_by_function.json").open("w", encoding="utf-8") as handle:
        json.dump(sorted(summaries, key=lambda row: int(row["function_id"])), handle, indent=2)
    total_records = sum(int(summary["records"]) for summary in summaries)
    weighted_top1 = (
        sum(float(summary["top1_accuracy"]) * int(summary["records"]) for summary in summaries) / total_records
        if total_records
        else 0.0
    )
    weighted_acceptable = (
        sum(float(summary["acceptable_accuracy"]) * int(summary["records"]) for summary in summaries) / total_records
        if total_records
        else 0.0
    )
    print(
        f"[collect] functions={len(summaries)} records={total_records} "
        f"weighted_top1={weighted_top1:.4f} "
        f"weighted_acceptable={weighted_acceptable:.4f}"
    )


def parse_function_ids(args: argparse.Namespace) -> list[int]:
    if args.function_id:
        return sorted(set(args.function_id))
    return list(range(args.first_function, args.last_function + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out PFN decisions against oracle surrogate labels from dataset.jsonl records."
    )
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--function-id", type=int, action="append", default=None)
    parser.add_argument("--first-function", type=int, default=1)
    parser.add_argument("--last-function", type=int, default=24)
    parser.add_argument("--surrogate-names", nargs="+", default=DEFAULT_SURROGATE_NAMES)
    parser.add_argument(
        "--comparable-margin",
        type=float,
        default=0.10,
        help="Oracle scores within this fraction of the score spread count as acceptable, matching training labels.",
    )
    parser.add_argument("--generation-bin-width", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--write-mistakes", type=int, default=200)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing f*/summary.json files under --out-dir into summary_by_function.{csv,json}.",
    )
    args = parser.parse_args()

    if args.collect_only:
        collect_existing_summaries(args.out_dir)
        return

    if args.experiment_root is None:
        raise ValueError("--experiment-root is required unless --collect-only is used")

    models_dir = args.models_dir or (args.experiment_root / "models")
    max_records = None if args.max_records <= 0 else args.max_records
    summaries: list[dict[str, Any]] = []

    for function_id in parse_function_ids(args):
        summary = evaluate_function(
            experiment_root=args.experiment_root,
            models_dir=models_dir,
            out_dir=args.out_dir,
            function_id=function_id,
            dimension=args.dimension,
            device=args.device,
            surrogate_names=list(args.surrogate_names),
            comparable_margin=args.comparable_margin,
            generation_bin_width=args.generation_bin_width,
            max_records=max_records,
            write_mistakes=args.write_mistakes,
        )
        summaries.append(summary)

    write_summary_csv(args.out_dir / "summary_by_function.csv", summaries)
    with (args.out_dir / "summary_by_function.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    total_records = sum(int(summary["records"]) for summary in summaries)
    weighted_top1 = (
        sum(float(summary["top1_accuracy"]) * int(summary["records"]) for summary in summaries) / total_records
        if total_records
        else 0.0
    )
    weighted_acceptable = (
        sum(float(summary["acceptable_accuracy"]) * int(summary["records"]) for summary in summaries) / total_records
        if total_records
        else 0.0
    )
    print(
        f"[overall] functions={len(summaries)} records={total_records} "
        f"weighted_top1={weighted_top1:.4f} "
        f"weighted_acceptable={weighted_acceptable:.4f}"
    )


if __name__ == "__main__":
    main()
