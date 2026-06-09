#!/usr/bin/env python3
"""Summarize surrogate choices recorded in CMA-ES generation logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

RUN_RE = re.compile(r"f(?P<function>\d+)_i\d+_d\d+_s\d+")
LOG_NAME = "generation_logs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize surrogate usage by BBOB function and generation.")
    parser.add_argument(
        "logs_root",
        nargs="?",
        type=Path,
        default=Path("eval_results"),
        help="Directory containing **/generation_logs.jsonl (default: eval_results)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("eval_results/analysis/surrogate_usage"),
        help="Output directory (default: eval_results/analysis/surrogate_usage)",
    )
    parser.add_argument(
        "--source",
        choices=("actual", "oracle", "auto"),
        default="actual",
        help="Use logged choices, best dataset score, or logged choices with score fallback.",
    )
    parser.add_argument(
        "--exclude-real-only",
        action="store_true",
        help="Exclude real_only from oracle score comparisons.",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def oracle_choices(record: dict[str, Any], include_real_only: bool) -> list[str]:
    raw_scores = record.get("dataset_scores") or record.get("surrogate_scores")
    if not isinstance(raw_scores, dict):
        return []

    scores = {
        str(name): score
        for name, raw in raw_scores.items()
        if (include_real_only or name != "real_only") and (score := finite_float(raw)) is not None
    }
    if not scores:
        return []

    best = max(scores.values())
    return [name for name, score in scores.items() if math.isclose(score, best, abs_tol=1e-12)]


def choices(record: dict[str, Any], source: str, include_real_only: bool) -> tuple[str, list[str]]:
    actual = record.get("surrogate_name")
    if source in {"actual", "auto"} and actual:
        return "actual", [str(actual)]
    if source in {"oracle", "auto"}:
        selected = oracle_choices(record, include_real_only)
        if selected:
            return "oracle", selected
    return "missing", []


def records(log_file: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with log_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, {}


def add_fractional(counter: Counter[str], selected: list[str]) -> None:
    weight = 1.0 / len(selected)
    for name in selected:
        counter[name] += weight


def collect(
    root: Path, source: str, include_real_only: bool
) -> tuple[
    Counter[str],
    dict[int, Counter[str]],
    dict[int, Counter[str]],
    Counter[str],
]:
    overall: Counter[str] = Counter()
    by_function: dict[int, Counter[str]] = defaultdict(Counter)
    by_generation: dict[int, Counter[str]] = defaultdict(Counter)
    status: Counter[str] = Counter()

    log_files = sorted(root.rglob(LOG_NAME))
    status["files"] = len(log_files)

    for log_file in log_files:
        match = RUN_RE.fullmatch(log_file.parent.name)
        function_id = int(match["function"]) if match else None

        for _, record in records(log_file):
            if not record:
                status["invalid_json"] += 1
                continue

            status["records"] += 1
            source_used, selected = choices(record, source, include_real_only)
            if not selected:
                status["missing"] += 1
                continue

            try:
                generation = int(record["generation_index"])
            except (KeyError, TypeError, ValueError):
                status["invalid_generation"] += 1
                continue

            status[source_used] += 1
            add_fractional(overall, selected)
            add_fractional(by_generation[generation], selected)
            if function_id is not None:
                add_fractional(by_function[function_id], selected)

    return overall, by_function, by_generation, status


def distribution_rows(groups: dict[int, Counter[str]], models: list[str], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, counts in sorted(groups.items()):
        total = sum(counts.values())
        row: dict[str, Any] = {key: group, "total": round(total, 6)}
        for model in models:
            row[f"{model}_count"] = round(counts[model], 6)
            row[f"{model}_pct"] = round(100 * counts[model] / total, 4) if total else 0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_by_function(path: Path, rows: list[dict[str, Any]], models: list[str]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; plot skipped")
        return

    labels = [str(row["function"]) for row in rows]
    bottom = [0.0] * len(rows)
    plt.figure(figsize=(13, 6))
    for model in models:
        values = [float(row[f"{model}_pct"]) for row in rows]
        plt.bar(labels, values, bottom=bottom, label=model)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    plt.ylim(0, 100)
    plt.xlabel("BBOB function")
    plt.ylabel("selection share (%)")
    plt.title("Surrogate Usage by BBOB Function")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    overall, by_function, by_generation, status = collect(
        args.logs_root,
        args.source,
        include_real_only=not args.exclude_real_only,
    )
    if not status["files"]:
        raise SystemExit(f"No {LOG_NAME} files found under {args.logs_root}")
    if not overall:
        raise SystemExit(f"No usable {args.source} choices found in {status['records']} records")

    models = sorted(overall)
    total = sum(overall.values())
    overall_rows = [
        {
            "surrogate": model,
            "count": round(overall[model], 6),
            "pct": round(100 * overall[model] / total, 4),
        }
        for model in sorted(models, key=overall.get, reverse=True)
    ]
    function_rows = distribution_rows(by_function, models, "function")
    generation_rows = distribution_rows(by_generation, models, "generation")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "overall.csv", overall_rows)
    write_csv(args.out / "by_function.csv", function_rows)
    write_csv(args.out / "by_generation.csv", generation_rows)
    plot_by_function(args.out / "by_function.png", function_rows, models)

    print(
        f"Read {status['records']} records from {status['files']} files; "
        f"counted {status['actual'] + status['oracle']} choices."
    )
    for row in overall_rows:
        print(f"{row['surrogate']:24} {row['pct']:6.2f}%  ({row['count']:g})")
    if status["missing"] or status["invalid_json"] or status["invalid_generation"]:
        print(
            "Skipped: "
            f"{status['missing']} missing choices, "
            f"{status['invalid_json']} invalid JSON, "
            f"{status['invalid_generation']} invalid generations."
        )
    print(f"Wrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
