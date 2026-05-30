#!/usr/bin/env python3
# VERSION: 2026-05-17-coco-ecdf-log10-fevals-over-dim-51-targets
"""
Plot COCO/BBOB-style per-function ECDF curves from local exdata and official
COCO/BBOB archive reference algorithms.

Important: this implements the COCO-style ECDF used in performance comparison
plots:
  y = fraction of (instance, target) pairs reached
  x = log10(number of function evaluations / DIM)

For a single BBOB function this is the ECDF over all available instances/runs
and all target precisions. By default 51 COCO-like targets are used:
  10^2, 10^1.8, ..., 10^-8

Example:
  python plot_coco_ecdf_corrected_v4.py exdata 5 1 24 --ref-years 2009 2020

Debug reference loading:
  python plot_coco_ecdf_corrected_v4.py exdata 5 1 1 --ref-years 2020 --debug-refs --max-refs 5

List archive entries only:
  python plot_coco_ecdf_corrected_v4.py exdata 5 1 1 --ref-years 2009 2020 --list-refs
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
import tempfile
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np
from matplotlib.ticker import MultipleLocator

VERSION = "2026-05-17-coco-ecdf-cache-short-flat-names-windows-fix"
DEFAULT_TARGETS = np.logspace(2, -8, 51)  # COCO-like 51 targets: 1e2 ... 1e-8


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
    p = argparse.ArgumentParser(
        description=(
            "Plot COCO/BBOB-style per-function ECDF curves. "
            f"VERSION={VERSION}"
        )
    )
    p.add_argument("exdata", type=Path, help="Root directory with local COCO observer output, e.g. exdata")
    p.add_argument("dim", type=int, help="BBOB dimension, e.g. 5")
    p.add_argument("first_func", type=int, help="First BBOB function id, inclusive")
    p.add_argument("last_func", type=int, help="Last BBOB function id, inclusive")
    p.add_argument("exp_name", type=str, help="Last BBOB function id, inclusive")

    p.add_argument("--budget", type=int, default=1000,
                   help="Maximum raw number of function evaluations; default 1000")
    p.add_argument("--targets", type=float, nargs="*", default=None,
                   help="Target precisions f-fopt. Default: 51 log-spaced COCO-like targets 1e2 ... 1e-8")
    p.add_argument("--ref-year", type=int, action="append", default=None,
                   help="BBOB archive publication year for reference curves. Can be repeated. Deprecated alias for --ref-years.")
    p.add_argument("--ref-years", type=int, nargs="+", default=None,
                   help="One or more BBOB archive publication years for reference curves; default: 2020. Example: --ref-years 2009 2020")
    p.add_argument(
        "--ref-tags",
        type=str,
        nargs="+",
        default=["2020"],
        help=(
            "Reference archive tags/substrings. "
            "Examples: 2020 GECCO2018 CMA-ES"
        ),
    )
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory for downloaded/extracted COCO reference archives. Default: system temp/coco_bbob_archive_cache")
    p.add_argument("--refs", nargs="*", default=None,
                   help=("Specific reference archive entries, substrings, or local paths. "
                         "If omitted, uses all cocopp.archives.bbob entries from --ref-year."))
    p.add_argument("--max-refs", type=int, default=None,
                   help="Optional cap on number of reference datasets, useful for debugging")
    p.add_argument("--no-refs", action="store_true", help="Disable official COCO/BBOB reference curves")
    p.add_argument("--list-refs", action="store_true", help="List selected reference archive entries and exit")
    p.add_argument("--debug-refs", action="store_true", help="Print detailed reference-loading diagnostics")
    p.add_argument("--label-refs", action="store_true", help="Show individual labels for reference algorithms")

    p.add_argument("--out", type=Path, default=Path("coco_plots"), help="Output directory for PNG plots")
    p.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving PNG files")
    p.add_argument("--local-label", default="local exdata", help="Legend label for your local data")
    p.add_argument("--linear-x", action="store_true",
                   help="Use raw evaluations on x-axis instead of COCO log10(evals / DIM). Not recommended.")
    return p.parse_args()


def load_cocopp_datasets(root: Path):
    import cocopp
    return cocopp.load(str(root))

def cocopp_hit_data(root: Path, dim: int, func: int, targets: np.ndarray, budget: int, debug: bool = False) -> HitData | None:
    """Use cocopp's own DataSet.detEvals, closer to official COCO ECDF."""
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

        evals = np.asarray(ds.detEvals(targets), dtype=float)

        if evals.ndim == 1:
            evals = evals.reshape(len(targets), -1)

        n_runs += evals.shape[1]
        n_pairs += evals.size

        finite = evals[np.isfinite(evals)]
        finite = finite[finite <= budget]
        hits.extend(float(x) for x in finite)

    if n_pairs == 0:
        return None

    return HitData(
        hits=np.asarray(sorted(hits), dtype=float),
        n_pairs=n_pairs,
        n_runs=n_runs,
    )

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
            # no suffix is older than any numbered suffix
            candidates.append((int(m.group(1)) if m.group(1) else -1, child))
    return max(candidates, key=lambda t: t[0])[1] if candidates else None


def read_dat_file(path: Path) -> list[RunCurve]:
    """Read COCO .dat/.tdat file.

    The first column is function evaluations. The third column is
    best noise-free fitness - Fopt, according to the COCO header:
      % f evaluations | g evaluations | best noise-free fitness - Fopt ...

    A single file can contain multiple runs/instances separated by '%' headers.
    """
    runs: list[RunCurve] = []
    cur_evals: list[float] = []
    cur_delta: list[float] = []

    def flush() -> None:
        nonlocal cur_evals, cur_delta
        if not cur_evals:
            return
        ev = np.asarray(cur_evals, dtype=float)
        de = np.asarray(cur_delta, dtype=float)
        ok = np.isfinite(ev) & np.isfinite(de) & (ev > 0)
        ev, de = ev[ok], de[ok]
        if ev.size:
            order = np.argsort(ev)
            ev, de = ev[order], de[order]
            # When duplicate evaluations appear, keep the best value up to that evaluation.
            uniq_ev, last_idx = np.unique(ev, return_index=True)
            # np.unique returns first index; aggregate manually for duplicated evals.
            best_per_eval = []
            for u in uniq_ev:
                best_per_eval.append(np.min(de[ev == u]))
            best = np.minimum.accumulate(np.asarray(best_per_eval, dtype=float))
            runs.append(RunCurve(uniq_ev.astype(float), best, path))
        cur_evals, cur_delta = [], []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("%"):
                    flush()
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    cur_evals.append(float(parts[0]))
                    cur_delta.append(float(parts[2]))
                except ValueError:
                    continue
        flush()
    except OSError as e:
        print(f"[WARN] cannot read {path}: {e}", file=sys.stderr)
    return runs


def find_dat_files(root: Path, dim: int, func: int) -> list[Path]:
    """Find COCO data files for a specific function and dimension."""
    if not root.exists():
        return []
    found: set[Path] = set()
    dim_token = f"DIM{dim}".upper()
    f_patterns = [
        re.compile(rf"(^|[^0-9])f{func}([^0-9]|$)", re.IGNORECASE),
        re.compile(rf"data_f{func}([^0-9]|$)", re.IGNORECASE),
    ]
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".dat", ".tdat"}:
            continue
        full = str(p).replace("\\", "/")
        upper_name = p.name.upper()
        if dim_token not in upper_name:
            continue
        if any(rx.search(full) for rx in f_patterns):
            found.add(p)
    return sorted(found)


def load_local_runs(exdata: Path, dim: int, func: int, exp_name, debug: bool = False) -> list[RunCurve]:
    folder = newest_local_function_dir(exdata, dim, func, exp_name=exp_name)
    if folder is None:
        print(f"[WARN] f{func}: no local folder {exp_name}_bbob_dim{dim}_f{func}[-NNNN] found")
        return []
    dat_files = find_dat_files(folder, dim, func)
    if debug:
        print(f"[DEBUG] f{func}: local folder={folder}, dat_files={len(dat_files)}")
        for p in dat_files[:5]:
            print(f"         local file: {p}")
    runs: list[RunCurve] = []
    for dat in dat_files:
        runs.extend(read_dat_file(dat))
    print(f"[INFO] f{func}: local files={len(dat_files)}, local runs={len(runs)}")
    return runs


def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]

def archive_entries_for_tags(archive, tags: list[str]) -> list[str]:
    candidates: list[str] = []

    for tag in tags:
        try:
            vals = [str(v) for v in as_list(archive.find(tag))]
            print(f"[INFO] archive.find({tag!r}) -> {len(vals)} entries")
            candidates.extend(vals)
        except Exception as e:
            print(f"[WARN] archive.find({tag!r}) failed: {e}")

    # deduplicate
    out: list[str] = []
    seen: set[str] = set()

    for e in candidates:
        if e not in seen:
            out.append(e)
            seen.add(e)

    return out


def resolve_ref_years(args: argparse.Namespace) -> list[int]:
    """Resolve reference years from --ref-years and repeated --ref-year.

    --ref-years 2020 2021 is the preferred syntax.
    --ref-year 2020 --ref-year 2021 is kept as a backward-compatible alias.
    If neither is provided, use 2020.
    """
    years: list[int] = []
    if getattr(args, "ref_years", None):
        years.extend(int(y) for y in args.ref_years)
    if getattr(args, "ref_year", None):
        years.extend(int(y) for y in args.ref_year)
    if not years:
        years = [2020]

    # Deduplicate preserving order.
    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            out.append(y)
            seen.add(y)
    return out


def safe_cache_name(text: str, max_prefix: int = 80) -> str:
    """Create a deterministic filesystem-safe cache name."""
    norm = str(text).replace('\\', '/')
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', norm).strip('._-')
    if not base:
        base = 'entry'
    h = hashlib.sha1(norm.encode('utf-8', errors='ignore')).hexdigest()[:12]
    return f"{base[:max_prefix]}__{h}"


def cache_root_from_args(args: argparse.Namespace) -> Path:
    """Return and create cache root for COCO archives/extractions."""
    if getattr(args, 'cache_dir', None) is not None:
        root = Path(args.cache_dir)
    else:
        root = Path(tempfile.gettempdir()) / 'coco_bbob_archive_cache'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'archives').mkdir(parents=True, exist_ok=True)
    (root / 'extracted').mkdir(parents=True, exist_ok=True)
    return root


def cached_archive_file(cache_root: Path, item: str | Path, source_file: Path) -> Path | None:
    """Copy an archive file returned by cocopp into our stable cache directory.

    Returns the cached file path. If copying fails, returns None and the caller
    can fall back to source_file.
    """
    try:
        if not source_file.exists() or not source_file.is_file():
            return None
        suffixes = ''.join(source_file.suffixes) or source_file.suffix
        if not suffixes:
            suffixes = '.dat'
        name = safe_cache_name(str(item), max_prefix=40) + suffixes
        dest = cache_root / 'archives' / name
        if dest.exists() and dest.stat().st_size == source_file.stat().st_size:
            return dest
        tmp = dest.with_suffix(dest.suffix + '.tmp')
        shutil.copy2(source_file, tmp)
        tmp.replace(dest)
        return dest
    except Exception as e:
        print(f"[WARN] cannot cache archive file {source_file}: {e}", file=sys.stderr)
        return None


def archive_entries_for_year(archive, year: int) -> list[str]:
    """List archive entries for the requested publication year."""
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

    # Deduplicate preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for e in out:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq

def normalize_bbob_archive_item(item: str) -> str:
    item = str(item)
    print(item)
    # cocopp index sometimes points to broken 2018-others paths
    mappings = [
        ["2014-others/","2014/"],
        ["2015-CEC/","2015/"],
        ["2015-GECCO/","2015/"],
        ["2017-others/","2017/"],
        ["2018-others/","2018/"],
    ]

    for mapping in mappings:
        if mapping[0] in item:
            candidate = item.replace(mapping[0], mapping[1], 1)
            #print(f"[INFO] remapping archive item: {item} -> {candidate}")
            return candidate

    return item

def try_direct_bbob_download(item: str, cache_root: Path) -> Path | None:
    import urllib.request

    item = item.replace("\\", "/")
    urls = []
    alg_name = item.split("/")[-1].split(".")[0]

    if item.startswith("2014/") or alg_name in {"GNN-CMA-ES_Faury", "IPOP-CMA-ES-2019_Faury"}:
        urls.append(f"https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/incomplete/{item}")

    else:
        urls.append(f"https://numbbo.github.io/data-archive/data-archive/bbob/{item}")


    for url in urls:
        out = cache_root / "archives" / (safe_cache_name(url, max_prefix=60) + ".tgz")

        if out.exists() and out.stat().st_size > 0:
            return out

        try:
            print(f"[INFO] direct download: {url}")
            urllib.request.urlretrieve(url, out)
            if out.exists() and out.stat().st_size > 0:
                return out
        except Exception as e:
            print(f"[WARN] direct download failed: {url}: {e}")

    return None


def download_archive_entry(archive, item: str | Path, cache_root: Path) -> Path | None:
    """Return a cached local archive/path for a COCO archive entry.

    The first run may call cocopp.archive.get(...). The resulting archive file is
    then copied into --cache-dir (or system temp), so repeated runs reuse this
    stable cache location and do not depend on re-downloading.
    """
    p = Path(str(item))
    if p.exists():
        if p.is_file():
            cached = cached_archive_file(cache_root, item, p)
            return cached or p
        return p

    # Already cached from a previous run? We do not know the suffix, so search by
    # the deterministic prefix.
    prefix = safe_cache_name(str(item))
    archives_dir = cache_root / "archives"
    if archives_dir.exists():
        matches = sorted(archives_dir.glob(prefix + "*"))
        for m in matches:
            if m.is_file() and m.stat().st_size > 0:
                return m

    try:
        #local = archive.get(str(item)
        norm_item = normalize_bbob_archive_item(item)
        direct = try_direct_bbob_download(str(norm_item), cache_root)
        if direct is not None:
            return direct
        print(f"archive get for {item}...")
        local = archive.get(str(item))
        #local = f"https://numbbo.github.io/data-archive/data-archive/bbob/{str(item)}"
        #local = f"https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/{str(item)}"
        print(f"archive.get({item!r}) -> {local}")
        #local = normalize_bbob_archive_item(local)
        #print(f"normalized archive item: {local}")
        lp = Path(str(local))
        if lp.exists():
            if lp.is_file():
                cached = cached_archive_file(cache_root, item, lp)
                return cached or lp
            return lp
        print(f"[WARN] archive.get({item!r}) returned non-existing path: {lp}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] cannot download archive entry {item!r}: {e}", file=sys.stderr)
    return None


def _wanted_coco_data_member(name: str) -> bool:
    """Return True for COCO data files we need for ECDF parsing."""
    low = name.replace("\\", "/").lower()
    return low.endswith((".dat", ".tdat")) and ("/data_f" in low or "data_f" in low)


def _flat_member_name(member_name: str) -> str:
    """Very short deterministic filename for a COCO archive member.

    Windows often fails even with flat extraction when the filename still
    contains the entire original COCO path. Keep only the tokens needed by
    find_dat_files/read_dat_file: function id, dimension, optional instance id,
    and a short hash for uniqueness. Example:
        data_f10_bbobexp_f10_DIM5_i1.dat -> f10_DIM5_i1__abc123.dat
    """
    norm = member_name.replace("\\", "/")
    low = norm.lower()
    h = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = Path(norm).suffix.lower()
    if suffix not in {".dat", ".tdat"}:
        suffix = ".dat"

    # Prefer explicit tokens from standard COCO filenames.
    f_match = re.search(r"(?:^|[/_])(?:bbobexp_)?f(\d+)(?:[_/.]|$)", low)
    if not f_match:
        f_match = re.search(r"data_f(\d+)(?:[/_]|$)", low)
    d_match = re.search(r"dim(\d+)", low, re.IGNORECASE)
    i_match = re.search(r"[_/-]i(\d+)(?:[_/.]|$)", low)

    if f_match and d_match:
        f_token = f"f{int(f_match.group(1))}"
        d_token = f"DIM{int(d_match.group(1))}"
        i_token = f"_i{int(i_match.group(1))}" if i_match else ""
        return f"{f_token}_{d_token}{i_token}__{h}{suffix}"

    # Fallback: still keep it short.
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(norm).name).strip("._-")
    stem = Path(base).stem[:40] or "coco_data"
    return f"{stem}__{h}{suffix}"


def ensure_extracted(path: Path, cache_root: Path) -> Path | None:
    if path.is_dir():
        return path
    if not path.exists():
        return None

    extracted_root = cache_root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    # SHORT deterministic directory name, not based on full absolute path
    h = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")[:24] or "archive"
    dest = extracted_root / f"{stem}__{h}"

    marker = dest / ".flat_extracted_short_v3"
    if marker.exists():
        return dest

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    try:
        if path.suffix.lower() == ".tgz" or path.name.lower().endswith(".tar.gz"):
            with tarfile.open(path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile() or not _wanted_coco_data_member(m.name):
                        continue

                    src = tf.extractfile(m)
                    if src is None:
                        continue

                    out = dest / _flat_member_name(m.name)
                    with src, out.open("wb") as f:
                        shutil.copyfileobj(src, f)
                    extracted_count += 1

            marker.write_text(f"flat_extracted_files={extracted_count}\n", encoding="utf-8")
            return dest

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not _wanted_coco_data_member(info.filename):
                        continue

                    out = dest / _flat_member_name(info.filename)
                    with zf.open(info, "r") as src, out.open("wb") as f:
                        shutil.copyfileobj(src, f)
                    extracted_count += 1

            marker.write_text(f"flat_extracted_files={extracted_count}\n", encoding="utf-8")
            return dest

    except Exception as e:
        print(f"[WARN] cannot flat-extract {path}: {e}", file=sys.stderr)
        return None

    return path.parent


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
                continue
            matches = []
            try:
                matches = [str(v) for v in as_list(archive.find(r))]
            except Exception:
                pass
            raw.extend(matches or [r])
    else:
        tags = args.ref_tags

        print(f"[INFO] reference tags: {tags}")

        raw.extend(archive_entries_for_tags(archive, tags))
        #years = resolve_ref_years(args)
        #print(f"[INFO] reference years: {years}")
        #for y in years:
        #    raw.extend(archive_entries_for_year(archive, y))

    seen: set[str] = set()
    entries: list[str | Path] = []
    for e in raw:
        k = str(e)
        if k not in seen:
            entries.append(e)
            seen.add(k)

    if args.max_refs is not None:
        entries = entries[:args.max_refs]

    print(f"[INFO] selected {len(entries)} reference entries")
    for e in entries[:30]:
        print(f"       - {e}")
    if len(entries) > 30:
        print(f"       ... {len(entries) - 30} more")

    if args.list_refs:
        return {}

    cache = cache_root_from_args(args)
    print(f"[INFO] COCO reference cache: {cache}")
    roots: dict[str, Path] = {}
    for item in entries:
        local = download_archive_entry(archive, item, cache)
        if local is None:
            continue
        #root = ensure_extracted(local, cache)
        #if root is None:
        #    continue
        root = local
        label = str(item).replace("\\", "/").split("/")[-1]
        label = label.replace(".tar.gz", "").replace(".tgz", "").replace(".zip", "")
        base = label or local.stem
        label_unique = base
        n = 2
        while label_unique in roots:
            label_unique = f"{base}#{n}"
            n += 1
        roots[label_unique] = root
        if args.debug_refs:
            print(f"[DEBUG] ref {item!r}: local={local}, root={root}")

    print(f"[INFO] reference roots ready: {len(roots)}")
    return roots


def load_reference_runs(ref_roots: dict[str, Path], dim: int, func: int, debug: bool = False) -> dict[str, list[RunCurve]]:
    datasets: dict[str, list[RunCurve]] = {}
    total_files = 0
    total_runs = 0
    for label, root in ref_roots.items():
        files = find_dat_files(root, dim, func)
        total_files += len(files)
        if debug:
            print(f"[DEBUG] f{func}: ref={label}, files={len(files)}, root={root}")
            for p in files[:5]:
                print(f"         ref file: {p}")
        runs: list[RunCurve] = []
        for f in files:
            runs.extend(read_dat_file(f))
        total_runs += len(runs)
        if runs:
            datasets[label] = runs
    print(f"[INFO] f{func}: reference files={total_files}, reference algorithms with runs={len(datasets)}, reference runs={total_runs}")
    return datasets

def hitting_times(runs: list[RunCurve], targets: np.ndarray, budget: int) -> tuple[np.ndarray, int]:
    """Fallback/manual first hitting raw evaluations and total instance-target pairs."""
    n_pairs = len(runs) * len(targets)
    hits: list[float] = []

    for run in runs:
        ev = run.evals
        de = run.best_delta

        if ev.size == 0:
            continue

        for t in targets:
            idx = np.flatnonzero(de <= t)
            if idx.size:
                hit = float(ev[idx[0]])
                if hit <= budget:
                    hits.append(hit)

    return np.asarray(sorted(hits), dtype=float), n_pairs

def ecdf_curve_from_hits(hit_data: HitData, budget: int, dim: int, linear_x: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if hit_data.n_pairs == 0:
        return np.array([]), np.array([])

    hits = hit_data.hits

    if linear_x:
        x_hits = hits
        x_min = 0.0
        x_max = float(budget)
    else:
        x_hits = np.log10(np.maximum(hits / float(dim), 1.0))
        x_min = 0.0
        x_max = np.log10(max(float(budget) / float(dim), 1.0))

    x_hits = np.sort(x_hits[x_hits <= x_max])

    xs = [x_min]
    ys = [0.0]

    for i, h in enumerate(x_hits, start=1):
        xs.extend([float(h), float(h)])
        ys.extend([ys[-1], i / hit_data.n_pairs])

    xs.append(x_max)
    ys.append(ys[-1])

    return np.asarray(xs), np.asarray(ys)


def format_coco_log_x_axis(ax, xmin: float, xmax: float) -> None:
    """Format x-axis like COCO: integer major ticks and logarithmic minor ticks.

    The plotted coordinate is already log10(evaluations / DIM). Therefore major
    ticks are integers, and minor ticks are n + log10(2), ..., n + log10(9).
    """
    lo = int(np.floor(xmin))
    hi = int(np.ceil(xmax))
    major = [v for v in range(lo, hi + 1) if xmin <= v <= xmax]
    minor: list[float] = []
    for n in range(lo - 1, hi + 1):
        for k in range(2, 10):
            pos = n + np.log10(k)
            if xmin < pos < xmax:
                minor.append(float(pos))
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    ax.xaxis.set_minor_formatter(NullFormatter())

def merge_hit_data(items: list[HitData]) -> HitData | None:
    items = [x for x in items if x is not None and x.n_pairs > 0]

    if not items:
        return None

    hit_arrays = [x.hits for x in items if len(x.hits)]

    if len(hit_arrays) == 0:
        hits = np.empty((0,), dtype=float)
    else:
        hits = np.concatenate(hit_arrays)
        hits = np.asarray(sorted(hits), dtype=float)

    return HitData(
        hits=hits,
        n_pairs=sum(x.n_pairs for x in items),
        n_runs=sum(x.n_runs for x in items),
    )

def plot_aggregate_ecdf(
    dim: int,
    budget: int,
    funcs: range,
    local_label: str,
    local_hits_all: list[HitData],
    ref_hits_all: dict[str, list[HitData]],
    out_dir: Path,
    show: bool,
    label_refs: bool,
    linear_x: bool,
) -> None:
    plt.figure(figsize=(8, 5))

    ref_plotted = 0
    for label, hit_list in ref_hits_all.items():
        merged = merge_hit_data(hit_list)
        if merged is None:
            continue

        x, y = ecdf_curve_from_hits(merged, budget, dim, linear_x)
        if not x.size:
            continue

        plt.step(
            x, y,
            where="post",
            color="0.72",
            alpha=0.45,
            linewidth=0.4,
            label=label if label_refs else ("COCO/BBOB reference algorithms" if ref_plotted == 0 else None),
            zorder=1,
        )
        ref_plotted += 1

    local_merged = merge_hit_data(local_hits_all)

    if local_merged is not None:
        x, y = ecdf_curve_from_hits(local_merged, budget, dim, linear_x)
        plt.step(
            x, y,
            where="post",
            color="black",
            linewidth=0.4,
            label=f"{local_label} (runs={local_merged.n_runs})",
            zorder=5,
        )

    if linear_x:
        xlabel = "number of function evaluations"
        xmin, xmax = 0.0, float(budget)
    else:
        xlabel = "log10(number of function evaluations / dimension)"
        xmin, xmax = 0.0, np.log10(max(float(budget) / float(dim), 1.0))

    plt.title(f"BBOB f{funcs.start}–f{funcs.stop - 1}, dim={dim}: ECDF over all functions")
    plt.xlabel(xlabel)
    plt.ylabel("fraction of function-instance-target pairs")

    ax = plt.gca()
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1.02)

    if not linear_x:
        format_coco_log_x_axis(ax, xmin, xmax)

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(True, which="major", alpha=0.45, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)

    plt.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bbob_f{funcs.start}_to_f{funcs.stop - 1}_dim{dim}_ecdf_aggregate.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"[OK] saved {out_path}  (refs plotted={ref_plotted})")

    if show:
        plt.show()

    plt.close()

def plot_one(
    func: int,
    dim: int,
    budget: int,
    targets: np.ndarray,
    local_label: str,
    local_hits: HitData | None,
    ref_hits: dict[str, HitData],
    out_dir: Path,
    show: bool,
    label_refs: bool,
    linear_x: bool,
) -> None:
    plt.figure(figsize=(8, 5))

    ref_plotted = 0
    for label, hit_data in ref_hits.items():
        x, y = ecdf_curve_from_hits(hit_data, budget, dim, linear_x)
        if not x.size:
            continue
        plt.step(
            x, y, where="post",
            color="0.72", alpha=0.45, linewidth=0.4,
            label=label if label_refs else (f"COCO/BBOB {ref_plotted + 1}+ reference algorithms" if ref_plotted == 0 else None),
            zorder=1,
        )
        ref_plotted += 1

    if local_hits is not None:
        x, y = ecdf_curve_from_hits(local_hits, budget, dim, linear_x)
    else:
        x, y = np.array([]), np.array([])

    if x.size and local_hits is not None:
        plt.step(x, y, where="post", color="black", linewidth=0.4,
                label=f"{local_label} (runs={local_hits.n_runs})", zorder=5)
    else:
        print(f"[WARN] f{func}: no local data to plot")

    if linear_x:
        xlabel = "number of function evaluations"
        xmin, xmax = 0.0, float(budget)
    else:
        xlabel = "log10(number of function evaluations / dimension)"
        xmin, xmax = 0.0, np.log10(max(float(budget) / float(dim), 1.0))

    plt.title(f"BBOB f{func}, dim={dim}: ECDF of reached instance-target pairs")
    plt.xlabel(xlabel)
    plt.ylabel("fraction of instance-target pairs")
    ax = plt.gca()
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1.02)
    if not linear_x:
        format_coco_log_x_axis(ax, xmin, xmax)
        ax.grid(True, which="major", alpha=0.45, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    else:
        ax.grid(True, which="both", alpha=0.3)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bbob_f{func}_dim{dim}_ecdf.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    local_n = local_hits.n_runs if local_hits is not None else 0
    print(f"[OK] saved {out_path}  (refs plotted={ref_plotted}, local runs={local_n})")
    if show:
        plt.show()
    plt.close()


def main() -> int:
    args = parse_args()
    targets = np.asarray(args.targets if args.targets else DEFAULT_TARGETS, dtype=float)
    targets = np.sort(targets)[::-1]
    print(f"[INFO] script version: {VERSION}")
    print(f"[INFO] targets: {len(targets)} from {targets[0]:.1e} to {targets[-1]:.1e}")
    print(f"[INFO] x-axis: {'raw evaluations' if args.linear_x else 'log10(evaluations / DIM)'}; raw budget={args.budget}")

    ref_roots = get_reference_roots(args)
    if args.list_refs:
        return 0
    if not args.no_refs and not ref_roots:
        print("[WARN] no reference roots loaded; plots will contain only local exdata")

    local_hits_all: list[HitData] = []
    ref_hits_all: dict[str, list[HitData]] = {}

    out_dir = Path(str(args.out) + "/" + args.exp_name)

    for func in range(args.first_func, args.last_func + 1):
        local_folder = newest_local_function_dir(args.exdata, args.dim, func, args.exp_name)
        local_hits = None
        if local_folder is not None:
            local_hits = cocopp_hit_data(local_folder, args.dim, func, targets, args.budget, debug=args.debug_refs)

        if local_hits is None:
            print(f"[WARN] f{func}: cocopp failed for local data, falling back to manual parser")
            local_runs = load_local_runs(args.exdata, args.dim, func, exp_name=args.exp_name, debug=args.debug_refs)
            hits, n_pairs = hitting_times(local_runs, targets, args.budget)
            local_hits = HitData(hits=hits, n_pairs=n_pairs, n_runs=len(local_runs)) if n_pairs else None

        if local_hits is not None:
            local_hits_all.append(local_hits)

        ref_hits: dict[str, HitData] = {}

        for label, root in ref_roots.items():
            hd = cocopp_hit_data(
                root, args.dim, func, targets, args.budget,
                debug=args.debug_refs
            )

            if hd is not None:
                ref_hits[label] = hd
                ref_hits_all.setdefault(label, []).append(hd)
                continue

            print(f"[WARN] f{func}: cocopp failed for reference {label}, falling back to manual parser")

            fallback_root = ensure_extracted(root, cache_root_from_args(args))
            if fallback_root is None:
                continue

            fallback_runs = load_reference_runs({label: fallback_root}, args.dim, func, debug=args.debug_refs).get(label, [])
            hits, n_pairs = hitting_times(fallback_runs, targets, args.budget)

            if n_pairs:
                hd = HitData(
                    hits=hits,
                    n_pairs=n_pairs,
                    n_runs=len(fallback_runs),
                )
                ref_hits[label] = hd
                ref_hits_all.setdefault(label, []).append(hd)

        plot_one(
            func=func,
            dim=args.dim,
            budget=args.budget,
            targets=targets,
            local_label=args.local_label,
            local_hits=local_hits,
            ref_hits=ref_hits,
            out_dir=out_dir,
            show=args.show,
            label_refs=args.label_refs,
            linear_x=args.linear_x,
        )

    plot_aggregate_ecdf(
        dim=args.dim,
        budget=args.budget,
        funcs=range(args.first_func, args.last_func + 1),
        local_label=args.local_label,
        local_hits_all=local_hits_all,
        ref_hits_all=ref_hits_all,
        out_dir=out_dir,
        show=args.show,
        label_refs=args.label_refs,
        linear_x=args.linear_x,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
