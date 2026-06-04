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


VERSION = "2026-06-04-coco-ecdf-table-compatible-with-plot-generator"
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


@dataclass(frozen=True)
class PlotSpec:
    key: str
    label: str
    color: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"COCO/BBOB ECDF table generator. VERSION={VERSION}")

    p.add_argument("exdata", type=Path)
    p.add_argument("dim", type=int)
    p.add_argument("first_func", type=int)
    p.add_argument("last_func", type=int)

    p.add_argument(
        "--local-exp",
        action="append",
        default=[],
        help="Local experiment spec: exp_name,label,color. Example: pfn_sur,Our Approach,#AA00FF",
    )

    p.add_argument(
        "--coco-alg",
        action="append",
        default=[],
        help="COCO algorithm spec: archive_label,label,color. Example: CMA-ES-2019,CMA-ES-2019,#5B84B1",
    )

    p.add_argument(
        "--ref-tags",
        type=str,
        nargs="+",
        default=["2020"],
        help="Reference archive tags/substrings. Examples: 2020 2018 DTS",
    )

    p.add_argument(
        "--refs",
        nargs="*",
        default=None,
        help="Specific reference archive entries, substrings, or local paths.",
    )

    p.add_argument("--no-refs", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=None)

    p.add_argument(
        "--evals_per_dim",
        type=int,
        nargs="+",
        required=True,
        help="Raw evaluation budgets, e.g. --evals_per_dim 50 100 200 500 1000",
    )

    p.add_argument("--targets", type=float, nargs="*", default=None)
    p.add_argument("--out", type=Path, default=Path("coco_eval_tables"))
    p.add_argument("--debug-refs", action="store_true")

    return p.parse_args()


def parse_plot_specs(values: list[str]) -> list[PlotSpec]:
    specs: list[PlotSpec] = []

    for v in values:
        parts = [x.strip() for x in v.split(",")]

        if len(parts) != 3:
            raise ValueError(f"Invalid spec {v!r}. Expected format: key,label,color")

        specs.append(
            PlotSpec(
                key=parts[0],
                label=parts[1],
                color=parts[2],
            )
        )

    return specs


def match_spec(label: str, specs: dict[str, PlotSpec]) -> PlotSpec | None:
    label_low = label.lower()

    for spec in specs.values():
        key_low = spec.key.lower()

        if key_low == label_low or key_low in label_low or label_low in key_low:
            return spec

    return None


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

    out: list[str] = []
    seen: set[str] = set()

    for e in candidates:
        if e not in seen:
            out.append(e)
            seen.add(e)

    return out


def safe_cache_name(text: str, max_prefix: int = 80) -> str:
    norm = str(text).replace("\\", "/")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", norm).strip("._-")
    if not base:
        base = "entry"

    h = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{base[:max_prefix]}__{h}"


def cache_root_from_args(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        root = Path(args.cache_dir)
    else:
        root = Path(tempfile.gettempdir()) / "coco_bbob_archive_cache"

    root.mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(parents=True, exist_ok=True)
    (root / "extracted").mkdir(parents=True, exist_ok=True)
    return root


def cached_archive_file(cache_root: Path, item: str | Path, source_file: Path) -> Path | None:
    try:
        if not source_file.exists() or not source_file.is_file():
            return None

        suffixes = "".join(source_file.suffixes) or source_file.suffix
        if not suffixes:
            suffixes = ".dat"

        name = safe_cache_name(str(item), max_prefix=40) + suffixes
        dest = cache_root / "archives" / name

        if dest.exists() and dest.stat().st_size == source_file.stat().st_size:
            return dest

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(source_file, tmp)
        tmp.replace(dest)

        return dest

    except Exception as e:
        print(f"[WARN] cannot cache archive file {source_file}: {e}", file=sys.stderr)
        return None


def normalize_bbob_archive_item(item: str | Path) -> str:
    item = str(item).replace("\\", "/")

    mappings = [
        ("2014-others/", "2014/"),
        ("2015-CEC/", "2015/"),
        ("2015-GECCO/", "2015/"),
        ("2017-others/", "2017/"),
        ("2018-others/", "2018/"),
    ]

    for old, new in mappings:
        if old in item:
            return item.replace(old, new, 1)

    return item


def try_direct_bbob_download(item: str, cache_root: Path) -> Path | None:
    import urllib.request

    item = item.replace("\\", "/")
    urls: list[str] = []

    alg_name = item.split("/")[-1].split(".")[0]

    if item.startswith("2014/") or alg_name in {"GNN-CMA-ES_Faury", "IPOP-CMA-ES-2019_Faury"}:
        urls.append(
            f"https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/incomplete/{item}"
        )
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
    p = Path(str(item))

    if p.exists():
        if p.is_file():
            cached = cached_archive_file(cache_root, item, p)
            return cached or p
        return p

    prefix = safe_cache_name(str(item))
    archives_dir = cache_root / "archives"

    if archives_dir.exists():
        matches = sorted(archives_dir.glob(prefix + "*"))
        for m in matches:
            if m.is_file() and m.stat().st_size > 0:
                return m

    try:
        norm_item = normalize_bbob_archive_item(item)

        direct = try_direct_bbob_download(str(norm_item), cache_root)
        if direct is not None:
            return direct

        print(f"[INFO] archive.get for {item} ...")
        local = archive.get(str(item))
        print(f"[INFO] archive.get({item!r}) -> {local}")

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
    low = name.replace("\\", "/").lower()
    return low.endswith((".dat", ".tdat")) and ("/data_f" in low or "data_f" in low)


def _flat_member_name(member_name: str) -> str:
    norm = member_name.replace("\\", "/")
    low = norm.lower()

    h = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = Path(norm).suffix.lower()

    if suffix not in {".dat", ".tdat"}:
        suffix = ".dat"

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

            try:
                matches = [str(v) for v in as_list(archive.find(r))]
            except Exception:
                matches = []

            raw.extend(matches or [r])
    else:
        tags = args.ref_tags
        print(f"[INFO] reference tags: {tags}")
        raw.extend(archive_entries_for_tags(archive, tags))

    entries: list[str | Path] = []
    seen: set[str] = set()

    for e in raw:
        k = str(e)
        if k not in seen:
            entries.append(e)
            seen.add(k)

    print(f"[INFO] selected {len(entries)} reference entries")
    for e in entries[:30]:
        print(f"       - {e}")
    if len(entries) > 30:
        print(f"       ... {len(entries) - 30} more")

    cache = cache_root_from_args(args)
    print(f"[INFO] COCO reference cache: {cache}")

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

        roots[label_unique] = local

        if args.debug_refs:
            print(f"[DEBUG] ref {item!r}: local={local}")

    print(f"[INFO] reference roots ready: {len(roots)}")
    return roots


def newest_local_function_dir(exdata: Path, dim: int, func: int, exp_name: str) -> Path | None:
    pat = re.compile(rf"^{re.escape(exp_name)}_bbob_dim{dim}_f{func}(?:-(\d+))?$")

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


def cocopp_hit_data(
    root: Path,
    dim: int,
    func: int,
    targets: np.ndarray,
    debug: bool = False,
) -> HitData | None:
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


def read_dat_file(path: Path) -> list[RunCurve]:
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

            uniq_ev = np.unique(ev)

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


def load_local_runs(
    exdata: Path,
    dim: int,
    func: int,
    exp_name: str,
    debug: bool = False,
) -> list[RunCurve]:
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

    print(f"[INFO] f{func}: local {exp_name}, files={len(dat_files)}, runs={len(runs)}")
    return runs


def load_reference_runs(
    ref_roots: dict[str, Path],
    dim: int,
    func: int,
    debug: bool = False,
) -> dict[str, list[RunCurve]]:
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

    print(
        f"[INFO] f{func}: reference files={total_files}, "
        f"reference algorithms with runs={len(datasets)}, reference runs={total_runs}"
    )

    return datasets


def hitting_times(runs: list[RunCurve], targets: np.ndarray) -> tuple[np.ndarray, int]:
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
                hits.append(hit)

    return np.asarray(sorted(hits), dtype=float), n_pairs


def ratio_at_evals(hit_data: HitData | None, eval_budgets: list[int]) -> dict[int, float]:
    if hit_data is None or hit_data.n_pairs == 0:
        return {b: float("nan") for b in eval_budgets}

    hits = np.asarray(hit_data.hits, dtype=float)

    return {
        b: float(np.count_nonzero(hits <= b) / hit_data.n_pairs)
        for b in eval_budgets
    }


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


def get_local_hit_data(
    args: argparse.Namespace,
    spec: PlotSpec,
    func: int,
    targets: np.ndarray,
) -> HitData | None:
    local_folder = newest_local_function_dir(args.exdata, args.dim, func, spec.key)

    local_hits = None

    if local_folder is not None:
        local_hits = cocopp_hit_data(
            local_folder,
            args.dim,
            func,
            targets,
            debug=args.debug_refs,
        )

    if local_hits is not None:
        return local_hits

    print(f"[WARN] f{func}: cocopp failed for local {spec.key}, falling back to manual parser")

    local_runs = load_local_runs(
        args.exdata,
        args.dim,
        func,
        exp_name=spec.key,
        debug=args.debug_refs,
    )

    hits, n_pairs = hitting_times(local_runs, targets)

    if n_pairs == 0:
        return None

    return HitData(
        hits=hits,
        n_pairs=n_pairs,
        n_runs=len(local_runs),
    )


def get_reference_hit_data(
    args: argparse.Namespace,
    label: str,
    root: Path,
    func: int,
    targets: np.ndarray,
) -> HitData | None:
    hd = cocopp_hit_data(
        root,
        args.dim,
        func,
        targets,
        debug=args.debug_refs,
    )

    if hd is not None:
        return hd

    print(f"[WARN] f{func}: cocopp failed for reference {label}, falling back to manual parser")

    fallback_root = ensure_extracted(root, cache_root_from_args(args))

    if fallback_root is None:
        return None

    fallback_runs = load_reference_runs(
        {label: fallback_root},
        args.dim,
        func,
        debug=args.debug_refs,
    ).get(label, [])

    hits, n_pairs = hitting_times(fallback_runs, targets)

    if n_pairs == 0:
        return None

    return HitData(
        hits=hits,
        n_pairs=n_pairs,
        n_runs=len(fallback_runs),
    )


def format_float(v: float) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def write_csv(
    out_path: Path,
    rows: list[dict],
    eval_budgets: list[int],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "key",
        "source",
        "func",
        "dim",
        "runs",
        "pairs",
    ] + [f"evals_{b}" for b in eval_budgets]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for row in rows:
            formatted = {}

            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    formatted[k] = format_float(v)
                else:
                    formatted[k] = v

            writer.writerow(formatted)

    return out_path


def make_row(
    algorithm: str,
    key: str,
    source: str,
    func: str | int,
    dim: int,
    hit_data: HitData | None,
    eval_budgets: list[int],
) -> dict:
    ratios = ratio_at_evals(hit_data, eval_budgets)

    row = {
        "algorithm": algorithm,
        "key": key,
        "source": source,
        "func": func,
        "dim": dim,
        "runs": hit_data.n_runs if hit_data is not None else 0,
        "pairs": hit_data.n_pairs if hit_data is not None else 0,
    }

    for b in eval_budgets:
        row[f"evals_{b}"] = ratios[b]

    return row


def main() -> int:
    args = parse_args()

    local_specs_list = parse_plot_specs(args.local_exp)
    coco_specs_list = parse_plot_specs(args.coco_alg)

    if not local_specs_list:
        raise ValueError("Musíš zadat aspoň jeden --local-exp exp_name,label,color")

    coco_specs = {s.key: s for s in coco_specs_list}

    targets = np.asarray(args.targets if args.targets else DEFAULT_TARGETS, dtype=float)
    targets = np.sort(targets)[::-1]

    eval_budgets = sorted(set((int(x) * int(args.dim)) for x in args.evals_per_dim))

    print(f"[INFO] script version: {VERSION}")
    print(f"[INFO] targets: {len(targets)} from {targets[0]:.1e} to {targets[-1]:.1e}")
    print(f"[INFO] eval budgets: {eval_budgets}")

    ref_roots = get_reference_roots(args)

    selected_ref_roots: dict[str, tuple[Path, PlotSpec]] = {}

    for label, root in ref_roots.items():
        spec = match_spec(label, coco_specs)

        if spec is None:
            continue

        if spec.key in selected_ref_roots:
            print(f"[WARN] duplicate match for {spec.key}: already have one, skipping {label}")
            continue

        selected_ref_roots[label] = (root, spec)

    print(f"[INFO] selected highlighted COCO refs for table: {len(selected_ref_roots)}")
    for label, (_, spec) in selected_ref_roots.items():
        print(f"       - {label} -> {spec.label}")

    all_rows: list[dict] = []

    local_hits_all: dict[str, list[HitData]] = {s.key: [] for s in local_specs_list}
    ref_hits_all: dict[str, list[HitData]] = {label: [] for label in selected_ref_roots}

    for func in range(args.first_func, args.last_func + 1):
        rows: list[dict] = []

        for spec in local_specs_list:
            hd = get_local_hit_data(args, spec, func, targets)

            if hd is not None:
                local_hits_all[spec.key].append(hd)

            row = make_row(
                algorithm=spec.label,
                key=spec.key,
                source="local",
                func=func,
                dim=args.dim,
                hit_data=hd,
                eval_budgets=eval_budgets,
            )

            rows.append(row)
            all_rows.append(row)

        for label, (root, spec) in selected_ref_roots.items():
            hd = get_reference_hit_data(args, label, root, func, targets)

            if hd is not None:
                ref_hits_all[label].append(hd)

            row = make_row(
                algorithm=spec.label,
                key=label,
                source="reference",
                func=func,
                dim=args.dim,
                hit_data=hd,
                eval_budgets=eval_budgets,
            )

            rows.append(row)
            all_rows.append(row)

        out_path = args.out / f"bbob_f{func}_dim{args.dim}_target_ratios.csv"
        write_csv(out_path, rows, eval_budgets)
        print(f"[OK] saved {out_path}")

    aggregate_rows: list[dict] = []

    func_label = f"f{args.first_func}-f{args.last_func}"

    for spec in local_specs_list:
        merged = merge_hit_data(local_hits_all[spec.key])

        aggregate_rows.append(
            make_row(
                algorithm=spec.label,
                key=spec.key,
                source="local",
                func=func_label,
                dim=args.dim,
                hit_data=merged,
                eval_budgets=eval_budgets,
            )
        )

    for label, (_, spec) in selected_ref_roots.items():
        merged = merge_hit_data(ref_hits_all[label])

        aggregate_rows.append(
            make_row(
                algorithm=spec.label,
                key=label,
                source="reference",
                func=func_label,
                dim=args.dim,
                hit_data=merged,
                eval_budgets=eval_budgets,
            )
        )

    aggregate_path = args.out / f"bbob_f{args.first_func}_to_f{args.last_func}_dim{args.dim}_target_ratios_aggregate.csv"
    write_csv(aggregate_path, aggregate_rows, eval_budgets)
    print(f"[OK] saved {aggregate_path}")

    all_path = args.out / f"bbob_f{args.first_func}_to_f{args.last_func}_dim{args.dim}_target_ratios_all.csv"
    write_csv(all_path, all_rows + aggregate_rows, eval_budgets)
    print(f"[OK] saved {all_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())