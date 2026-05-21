#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_TARGETS = np.logspace(2, -8, 51)


@dataclass(frozen=True)
class RunCurve:
    evals: np.ndarray
    best_delta: np.ndarray
    source_file: Path


@dataclass(frozen=True)
class HitData:
    hits: np.ndarray
    n_pairs: int
    n_runs: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("exdata", type=Path)
    p.add_argument("dim", type=int)
    p.add_argument("first_func", type=int)
    p.add_argument("last_func", type=int)
    p.add_argument("exp_name", type=str, help="Last BBOB function id, inclusive")

    p.add_argument("--evals", type=int, nargs="+", required=True,
                   help="Raw evaluation budgets, e.g. --evals 50 100 200 500 1000")
    p.add_argument("--targets", type=float, nargs="*", default=None)

    p.add_argument("--ref-year", type=int, action="append", default=None)
    p.add_argument("--ref-years", type=int, nargs="+", default=None)
    p.add_argument("--refs", nargs="*", default=None)
    p.add_argument("--no-refs", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=None)

    p.add_argument("--out", type=Path, default=Path("coco_eval_tables"))
    p.add_argument("--local-label", default="local exdata")
    p.add_argument("--debug-refs", action="store_true")
    return p.parse_args()


def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def resolve_ref_years(args: argparse.Namespace) -> list[int]:
    years: list[int] = []
    if args.ref_years:
        years.extend(int(y) for y in args.ref_years)
    if args.ref_year:
        years.extend(int(y) for y in args.ref_year)
    if not years:
        years = [2020]

    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            out.append(y)
            seen.add(y)
    return out


def safe_cache_name(text: str, max_prefix: int = 40) -> str:
    norm = str(text).replace("\\", "/")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", norm).strip("._-") or "entry"
    h = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{base[:max_prefix]}__{h}"


def cache_root_from_args(args: argparse.Namespace) -> Path:
    root = args.cache_dir if args.cache_dir is not None else Path(tempfile.gettempdir()) / "coco_bbob_archive_cache"
    root.mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(parents=True, exist_ok=True)
    (root / "extracted").mkdir(parents=True, exist_ok=True)
    return root


def cached_archive_file(cache_root: Path, item: str | Path, source_file: Path) -> Path | None:
    try:
        if not source_file.exists() or not source_file.is_file():
            return None
        suffixes = "".join(source_file.suffixes) or source_file.suffix or ".dat"
        dest = cache_root / "archives" / (safe_cache_name(str(item)) + suffixes)
        if dest.exists() and dest.stat().st_size == source_file.stat().st_size:
            return dest
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(source_file, tmp)
        tmp.replace(dest)
        return dest
    except Exception as e:
        print(f"[WARN] cannot cache archive file {source_file}: {e}", file=sys.stderr)
        return None


def archive_entries_for_year(archive, year: int) -> list[str]:
    candidates: list[str] = []
    for q in (f"{year}/", str(year)):
        try:
            vals = [str(v) for v in as_list(archive.find(q))]
            print(f"[INFO] archive.find({q!r}) -> {len(vals)} entries")
            candidates.extend(vals)
        except Exception as e:
            print(f"[WARN] archive.find({q!r}) failed: {e}")

    out: list[str] = []
    for e in candidates:
        norm = e.replace("\\", "/")
        if norm.startswith(f"{year}/") or f"/{year}/" in norm or f"bbob/{year}/" in norm:
            out.append(e)

    uniq: list[str] = []
    seen: set[str] = set()
    for e in out:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq


def download_archive_entry(archive, item: str | Path, cache_root: Path) -> Path | None:
    p = Path(str(item))
    if p.exists():
        if p.is_file():
            return cached_archive_file(cache_root, item, p) or p
        return p

    prefix = safe_cache_name(str(item))
    for m in sorted((cache_root / "archives").glob(prefix + "*")):
        if m.is_file() and m.stat().st_size > 0:
            return m

    try:
        local = archive.get(str(item))
        lp = Path(str(local))
        if lp.exists():
            if lp.is_file():
                return cached_archive_file(cache_root, item, lp) or lp
            return lp
    except Exception as e:
        print(f"[WARN] cannot download archive entry {item!r}: {e}", file=sys.stderr)

    return None


def get_reference_roots(args: argparse.Namespace) -> dict[str, Path]:
    if args.no_refs:
        return {}

    try:
        import cocopp  # type: ignore
    except Exception as e:
        print(f"[WARN] cocopp is not installed; references disabled: {e}")
        return {}

    archive = cocopp.archives.bbob
    raw: list[str | Path] = []

    if args.refs:
        for r in args.refs:
            rp = Path(r)
            if rp.exists():
                raw.append(rp)
            else:
                try:
                    matches = [str(v) for v in as_list(archive.find(r))]
                except Exception:
                    matches = []
                raw.extend(matches or [r])
    else:
        years = resolve_ref_years(args)
        print(f"[INFO] reference years: {years}")
        for y in years:
            raw.extend(archive_entries_for_year(archive, y))

    entries: list[str | Path] = []
    seen: set[str] = set()
    for e in raw:
        k = str(e)
        if k not in seen:
            entries.append(e)
            seen.add(k)

    print(f"[INFO] selected {len(entries)} reference entries")
    cache = cache_root_from_args(args)

    roots: dict[str, Path] = {}
    for item in entries:
        local = download_archive_entry(archive, item, cache)
        if local is None:
            continue

        label = str(item).replace("\\", "/").split("/")[-1]
        label = label.replace(".tar.gz", "").replace(".tgz", "").replace(".zip", "")
        base = label or local.stem

        label_unique = base
        n = 2
        while label_unique in roots:
            label_unique = f"{base}#{n}"
            n += 1

        # Important: keep archive path for cocopp; do not flat-extract first.
        roots[label_unique] = local

    print(f"[INFO] reference roots ready: {len(roots)}")
    return roots


def newest_local_function_dir(exdata: Path, dim: int, func: int, exp_name) -> Path | None:
    pat = re.compile(rf"^{exp_name}_bbob_dim{dim}_f{func}(?:-(\d+))?$")
    candidates: list[tuple[int, Path]] = []
    if not exdata.exists():
        return None

    for child in exdata.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if m:
            candidates.append((int(m.group(1)) if m.group(1) else -1, child))

    return max(candidates, key=lambda t: t[0])[1] if candidates else None


def cocopp_hit_data(root: Path, dim: int, func: int, targets: np.ndarray, debug: bool = False) -> HitData | None:
    try:
        import cocopp  # type: ignore
        ds_list = cocopp.load(str(root))
    except Exception as e:
        if debug:
            print(f"[DEBUG] cocopp.load failed for {root}: {e}")
        return None

    hits: list[float] = []
    n_pairs = 0
    n_runs = 0

    for ds in ds_list:
        if getattr(ds, "funcId", None) != func or getattr(ds, "dim", None) != dim:
            continue

        try:
            evals = np.asarray(ds.detEvals(targets), dtype=float)
        except Exception as e:
            if debug:
                print(f"[DEBUG] detEvals failed for {root}, f{func}, dim{dim}: {e}")
            continue

        if evals.ndim == 1:
            evals = evals.reshape(len(targets), -1)

        n_runs += evals.shape[1]
        n_pairs += evals.size

        finite = evals[np.isfinite(evals)]
        hits.extend(float(x) for x in finite)

    if n_pairs == 0:
        return None

    return HitData(
        hits=np.asarray(sorted(hits), dtype=float),
        n_pairs=n_pairs,
        n_runs=n_runs,
    )


def ratio_at_evals(hit_data: HitData | None, eval_budgets: list[int]) -> dict[int, float]:
    if hit_data is None or hit_data.n_pairs == 0:
        return {b: float("nan") for b in eval_budgets}

    hits = np.asarray(hit_data.hits, dtype=float)
    return {
        b: float(np.count_nonzero(hits <= b) / hit_data.n_pairs)
        for b in eval_budgets
    }


def write_csv_for_function(
    out_dir: Path,
    func: int,
    dim: int,
    rows: list[dict],
    eval_budgets: list[int],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bbob_f{func}_dim{dim}_target_ratios.csv"

    fieldnames = ["algorithm", "source", "runs", "pairs"] + [f"evals_{b}" for b in eval_budgets]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            formatted = {}

            for k, v in row.items():
                if isinstance(v, float):
                    if np.isnan(v):
                        formatted[k] = ""
                    else:
                        formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v

            writer.writerow(formatted)

    return out_path


def main() -> int:
    args = parse_args()

    targets = np.asarray(args.targets if args.targets else DEFAULT_TARGETS, dtype=float)
    targets = np.sort(targets)[::-1]

    eval_budgets = sorted(set(int(x) for x in args.evals))
    print(f"[INFO] targets: {len(targets)} from {targets[0]:.1e} to {targets[-1]:.1e}")
    print(f"[INFO] eval budgets: {eval_budgets}")

    ref_roots = get_reference_roots(args)

    for func in range(args.first_func, args.last_func + 1):
        rows: list[dict] = []

        local_folder = newest_local_function_dir(args.exdata, args.dim, func, args.exp_name)
        local_hits = None
        if local_folder is not None:
            local_hits = cocopp_hit_data(local_folder, args.dim, func, targets, debug=args.debug_refs)

        local_ratios = ratio_at_evals(local_hits, eval_budgets)
        local_row = {
            "algorithm": args.local_label,
            "source": "local",
            "runs": local_hits.n_runs if local_hits is not None else 0,
            "pairs": local_hits.n_pairs if local_hits is not None else 0,
        }
        for b in eval_budgets:
            local_row[f"evals_{b}"] = local_ratios[b]
        rows.append(local_row)

        for label, root in ref_roots.items():
            hd = cocopp_hit_data(root, args.dim, func, targets, debug=args.debug_refs)
            if hd is None:
                print(f"[WARN] f{func}: skipping reference {label}, cocopp failed")
                continue

            ratios = ratio_at_evals(hd, eval_budgets)
            row = {
                "algorithm": label,
                "source": "reference",
                "runs": hd.n_runs,
                "pairs": hd.n_pairs,
            }
            for b in eval_budgets:
                row[f"evals_{b}"] = ratios[b]
            rows.append(row)

        out_path = write_csv_for_function(args.out, func, args.dim, rows, eval_budgets)
        print(f"[OK] saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())