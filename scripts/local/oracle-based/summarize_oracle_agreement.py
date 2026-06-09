from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SURROGATE_NAMES = [
    "lq_linear_top50",
    "lq_quadratic_top30",
    "gp_matern_dts",
    "gp_matern_unc",
    "rf_lifelength",
    "svm_rank_top50",
    "real_only",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def function_id_from_dir(path: Path) -> int:
    return int(path.name.removeprefix("f"))


def collect_summaries(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_file in sorted(root.glob("f*/summary.json"), key=lambda p: function_id_from_dir(p.parent)):
        with summary_file.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    return summaries


def aggregate_distributions(root: Path, surrogate_names: list[str]) -> tuple[Counter[str], Counter[str]]:
    oracle_counts: Counter[str] = Counter()
    pfn_counts: Counter[str] = Counter()
    for path in sorted(root.glob("f*/oracle_vs_pfn_distribution.csv")):
        for row in read_csv_rows(path):
            name = row["surrogate"]
            oracle_counts[name] += int(row["oracle_count"])
            pfn_counts[name] += int(row["pfn_count"])
    for name in surrogate_names:
        oracle_counts[name] += 0
        pfn_counts[name] += 0
    return oracle_counts, pfn_counts


def aggregate_confusion(root: Path, surrogate_names: list[str]) -> dict[str, Counter[str]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(root.glob("f*/confusion_oracle_vs_pfn.csv")):
        for row in read_csv_rows(path):
            oracle = row["oracle"]
            for predicted in surrogate_names:
                table[oracle][predicted] += int(row[predicted])
    return table


def write_aggregate_distribution(
    path: Path,
    oracle_counts: Counter[str],
    pfn_counts: Counter[str],
    surrogate_names: list[str],
) -> None:
    total = sum(oracle_counts.values())
    rows: list[dict[str, Any]] = []
    for name in surrogate_names:
        oracle_pct = 100.0 * oracle_counts[name] / total if total else 0.0
        pfn_pct = 100.0 * pfn_counts[name] / total if total else 0.0
        rows.append(
            {
                "surrogate": name,
                "oracle_count": oracle_counts[name],
                "oracle_pct": round(oracle_pct, 4),
                "pfn_count": pfn_counts[name],
                "pfn_pct": round(pfn_pct, 4),
                "pfn_minus_oracle_pct": round(pfn_pct - oracle_pct, 4),
            }
        )
    write_csv(
        path,
        ["surrogate", "oracle_count", "oracle_pct", "pfn_count", "pfn_pct", "pfn_minus_oracle_pct"],
        rows,
    )


def write_aggregate_confusion(
    path: Path,
    table: dict[str, Counter[str]],
    surrogate_names: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for oracle in surrogate_names:
        row: dict[str, Any] = {"oracle": oracle}
        row.update({predicted: table[oracle][predicted] for predicted in surrogate_names})
        rows.append(row)
    write_csv(path, ["oracle"] + surrogate_names, rows)


def write_ranked_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = sorted(summaries, key=lambda row: float(row["top1_accuracy"]))
    fieldnames = [
        "function_id",
        "records",
        "top1_accuracy",
        "acceptable_accuracy",
        "top2_accuracy",
        "top3_accuracy",
        "mean_reciprocal_rank",
    ]
    write_csv(path, fieldnames, [{key: row[key] for key in fieldnames} for row in rows])


def plot_metrics(root: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    summaries = sorted(summaries, key=lambda row: int(row["function_id"]))
    funcs = [str(row["function_id"]) for row in summaries]
    top1 = [float(row["top1_accuracy"]) for row in summaries]
    acceptable = [float(row["acceptable_accuracy"]) for row in summaries]
    top3 = [float(row["top3_accuracy"]) for row in summaries]

    plt.figure(figsize=(12, 5))
    x = range(len(funcs))
    plt.plot(x, top1, marker="o", label="top1 exact")
    plt.plot(x, acceptable, marker="o", label="acceptable")
    plt.plot(x, top3, marker="o", label="top3")
    plt.xticks(list(x), funcs)
    plt.ylim(0, 1)
    plt.xlabel("BBOB function")
    plt.ylabel("agreement")
    plt.title("PFN Agreement With Held-Out Oracle Labels")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "agreement_by_function.png", dpi=180)
    plt.close()


def plot_distribution(
    root: Path,
    oracle_counts: Counter[str],
    pfn_counts: Counter[str],
    surrogate_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    total = sum(oracle_counts.values())
    oracle_pct = [100.0 * oracle_counts[name] / total if total else 0.0 for name in surrogate_names]
    pfn_pct = [100.0 * pfn_counts[name] / total if total else 0.0 for name in surrogate_names]

    x = list(range(len(surrogate_names)))
    width = 0.38
    plt.figure(figsize=(13, 5))
    plt.bar([i - width / 2 for i in x], oracle_pct, width=width, label="oracle")
    plt.bar([i + width / 2 for i in x], pfn_pct, width=width, label="PFN")
    plt.xticks(x, surrogate_names, rotation=35, ha="right")
    plt.ylabel("share (%)")
    plt.title("Oracle Label Distribution vs PFN Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "oracle_vs_pfn_distribution.png", dpi=180)
    plt.close()


def plot_confusion(
    root: Path,
    table: dict[str, Counter[str]],
    surrogate_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    matrix = np.asarray(
        [[table[oracle][predicted] for predicted in surrogate_names] for oracle in surrogate_names],
        dtype=float,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    plt.figure(figsize=(9, 7))
    plt.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(label="PFN prediction share within oracle class")
    plt.xticks(range(len(surrogate_names)), surrogate_names, rotation=45, ha="right")
    plt.yticks(range(len(surrogate_names)), surrogate_names)
    plt.xlabel("PFN predicted")
    plt.ylabel("oracle best")
    plt.title("Aggregate Oracle vs PFN Confusion Matrix")
    for i in range(len(surrogate_names)):
        for j in range(len(surrogate_names)):
            plt.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "confusion_oracle_vs_pfn.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate oracle-agreement result directories.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--surrogate-names", nargs="+", default=DEFAULT_SURROGATE_NAMES)
    args = parser.parse_args()

    summaries = collect_summaries(args.root)
    if not summaries:
        raise FileNotFoundError(f"No f*/summary.json files found under {args.root}")

    surrogate_names = list(args.surrogate_names)
    oracle_counts, pfn_counts = aggregate_distributions(args.root, surrogate_names)
    confusion = aggregate_confusion(args.root, surrogate_names)

    write_ranked_summary(args.root / "summary_by_function_ranked_by_top1.csv", summaries)
    write_aggregate_distribution(
        args.root / "aggregate_oracle_vs_pfn_distribution.csv", oracle_counts, pfn_counts, surrogate_names
    )
    write_aggregate_confusion(args.root / "aggregate_confusion_oracle_vs_pfn.csv", confusion, surrogate_names)
    plot_metrics(args.root, summaries)
    plot_distribution(args.root, oracle_counts, pfn_counts, surrogate_names)
    plot_confusion(args.root, confusion, surrogate_names)

    total_records = sum(int(row["records"]) for row in summaries)
    weighted_top1 = sum(float(row["top1_accuracy"]) * int(row["records"]) for row in summaries) / total_records
    weighted_acceptable = (
        sum(float(row["acceptable_accuracy"]) * int(row["records"]) for row in summaries) / total_records
    )
    print(f"functions: {len(summaries)}")
    print(f"records: {total_records}")
    print(f"weighted_top1_accuracy: {weighted_top1:.4f}")
    print(f"weighted_acceptable_accuracy: {weighted_acceptable:.4f}")
    print(f"wrote aggregate outputs under {args.root}")


if __name__ == "__main__":
    main()
