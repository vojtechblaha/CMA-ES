#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from coco_eval_table import DEFAULT_TARGETS, cocopp_hit_data, ratio_at_evals

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BUDGETS = [50, 100, 200, 500, 1000]
FUNCTION_GROUPS = [
    ("separable", "Separable Functions", range(1, 6)),
    (
        "low_moderate_conditioning",
        "Functions with Low or Moderate Conditioning",
        range(6, 10),
    ),
    (
        "high_conditioning_unimodal",
        "High Conditioning and Unimodal Functions",
        range(10, 15),
    ),
    (
        "multimodal_adequate_structure",
        "Multi-modal Functions with Adequate Global Structure",
        range(15, 20),
    ),
    (
        "multimodal_weak_structure",
        "Multi-modal Functions with Weak Global Structure",
        range(20, 25),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-surrogate COCO target-hit ratios with PFN results.")
    parser.add_argument("--fixed-root", type=Path, default=Path("results/fixed"))
    parser.add_argument("--pfn-tables", type=Path, default=Path("eval_results/eval_tables"))
    parser.add_argument("--out", type=Path, default=Path("eval_results/fixed_vs_pfn"))
    return parser.parse_args()


def group_for_function(function_id: int) -> tuple[str, str]:
    for key, label, functions in FUNCTION_GROUPS:
        if function_id in functions:
            return key, label
    raise ValueError(f"Unknown BBOB function: f{function_id}")


def read_pfn_rows(table_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for function_id in range(1, 25):
        path = table_dir / f"bbob_f{function_id}_dim5_target_ratios.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            candidates = list(csv.DictReader(handle, delimiter=";"))
        local = next(row for row in candidates if row["source"] == "local")
        group_key, group_label = group_for_function(function_id)
        result = {
            "method": "PFN",
            "method_type": "trained",
            "function": function_id,
            "group": group_key,
            "group_label": group_label,
            "runs": int(local["runs"]),
            "pairs": int(local["pairs"]),
        }
        result.update({f"evals_{b}": float(local[f"evals_{b}"]) for b in BUDGETS})
        rows.append(result)
    return rows


def compute_fixed_rows(fixed_root: Path) -> list[dict]:
    rows: list[dict] = []
    targets = np.sort(DEFAULT_TARGETS)[::-1]
    for method_dir in sorted(path for path in fixed_root.iterdir() if path.is_dir()):
        exdata = method_dir / "exdata"
        if not exdata.is_dir():
            continue
        method = method_dir.name.removeprefix("fixed_")
        for function_id in range(1, 25):
            function_dir = exdata / (f"{method_dir.name}_bbob_dim5_f{function_id}")
            hit_data = cocopp_hit_data(function_dir, 5, function_id, targets)
            if hit_data is None:
                raise RuntimeError(f"Could not load COCO data from {function_dir}")
            ratios = ratio_at_evals(hit_data, BUDGETS)
            group_key, group_label = group_for_function(function_id)
            result = {
                "method": method,
                "method_type": "fixed",
                "function": function_id,
                "group": group_key,
                "group_label": group_label,
                "runs": hit_data.n_runs,
                "pairs": hit_data.n_pairs,
            }
            result.update({f"evals_{b}": ratios[b] for b in BUDGETS})
            rows.append(result)
    return rows


def add_budget_mean(rows: list[dict]) -> None:
    for row in rows:
        row["mean_budgets"] = float(np.mean([row[f"evals_{budget}"] for budget in BUDGETS]))


def aggregate_groups(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    methods = sorted({row["method"] for row in rows})
    for group_key, group_label, functions in FUNCTION_GROUPS:
        function_set = set(functions)
        for method in methods:
            selected = [row for row in rows if row["method"] == method and row["function"] in function_set]
            if not selected:
                continue
            result = {
                "group": group_key,
                "group_label": group_label,
                "method": method,
                "method_type": selected[0]["method_type"],
                "functions": len(selected),
            }
            for budget in BUDGETS:
                result[f"evals_{budget}"] = float(np.mean([row[f"evals_{budget}"] for row in selected]))
            result["mean_budgets"] = float(np.mean([result[f"evals_{budget}"] for budget in BUDGETS]))
            output.append(result)
    return output


def build_comparison(per_function: list[dict], group_rows: list[dict]) -> list[dict]:
    comparison: list[dict] = []
    fixed_methods = sorted({row["method"] for row in per_function if row["method_type"] == "fixed"})
    for group_key, group_label, functions in FUNCTION_GROUPS:
        pfn_group = next(row for row in group_rows if row["group"] == group_key and row["method"] == "PFN")
        fixed_group = [row for row in group_rows if row["group"] == group_key and row["method_type"] == "fixed"]
        function_set = set(functions)
        for metric in [f"evals_{b}" for b in BUDGETS] + ["mean_budgets"]:
            best_fixed = max(fixed_group, key=lambda row: row[metric])
            oracle_values = []
            for function_id in function_set:
                candidates = [
                    row[metric]
                    for row in per_function
                    if row["function"] == function_id and row["method"] in fixed_methods
                ]
                oracle_values.append(max(candidates))
            oracle = float(np.mean(oracle_values))
            comparison.append(
                {
                    "group": group_key,
                    "group_label": group_label,
                    "metric": metric,
                    "pfn": pfn_group[metric],
                    "best_fixed_method": best_fixed["method"],
                    "best_fixed": best_fixed[metric],
                    "pfn_minus_best_fixed": pfn_group[metric] - best_fixed[metric],
                    "per_function_fixed_oracle": oracle,
                    "pfn_minus_fixed_oracle": pfn_group[metric] - oracle,
                }
            )
    return comparison


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_pfn_rows(args.pfn_tables) + compute_fixed_rows(args.fixed_root)
    add_budget_mean(rows)
    group_rows = aggregate_groups(rows)
    comparison = build_comparison(rows, group_rows)

    write_csv(args.out / "per_function_target_ratios.csv", rows)
    write_csv(args.out / "group_target_ratios.csv", group_rows)
    write_csv(args.out / "pfn_vs_best_fixed.csv", comparison)

    print(f"Wrote analysis to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
