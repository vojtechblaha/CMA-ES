from __future__ import annotations

import os
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_POSTHOG_ANALYTICS"] = "1"
os.environ["ANALYTICS_DISABLED"] = "1"

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from pfn_cmaes.dts import ExactLikeDTSTabPFNExperiment
from pfn_cmaes.coco.objectives import CocoProblemWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DTS-CMA-ES DoubleTrainedEC reference-style control with the GP model "
            "replaced by a TabPFN surrogate on COCO/BBOB."
        )
    )
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--function_id", type=int, default=10)
    p.add_argument("--dimension", type=int, default=5)

    # The public surrogate-cmaes README recommends [1:5 41:50] for final BBOB results.
    p.add_argument(
        "--instances",
        type=str,
        default="1-5,41-50",
        help="COCO instance string, e.g. '1-5,41-50' or '1-15'. Overrides instances_start/end.",
    )
    p.add_argument("--instances_start", type=int, default=None)
    p.add_argument("--instances_end", type=int, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_generations", type=int, default=100000)
    p.add_argument("--max_true_evals", type=int, default=None)
    p.add_argument(
        "--budget_multiplier",
        type=int,
        default=250,
        help="Used when --max_true_evals is not given. surrogate-cmaes README recommends 250*dim.",
    )
    p.add_argument("--output_dir", type=Path, default=Path("results"))
    p.add_argument("--device", type=str, default="cuda")

    # CMA-ES input parameters. Defaults are chosen to avoid overriding the reference CMA-ES defaults.
    p.add_argument(
        "--pop_size",
        type=int,
        default=None,
        help="If omitted, do not set PopSize and use CMA-ES default lambda=4+floor(3*log(N)).",
    )
    p.add_argument(
        "--sigma0",
        type=float,
        default=2.0,
        help="Initial sigma. BBOB CMA-ES experiments commonly use x0 in [-4,4]^D and sigma0=2.",
    )
    p.add_argument(
        "--init_mode",
        choices=["bbob_uniform", "zeros"],
        default="bbob_uniform",
        help="Default uses x0 = -4 + 8*rand(D), matching common BBOB CMA-ES initialization.",
    )
    p.add_argument(
        "--x0_seed_offset",
        type=int,
        default=1000003,
        help="Offset used to derive deterministic x0 per function/instance/dimension/seed.",
    )
    p.add_argument(
        "--cma_active",
        action="store_true",
        help="Enable active CMA. Omitted by default because the Matlab s_cmaes.m default has CMA.active=0.",
    )
    p.add_argument("--cma_restart_count", type=int, default=0)
    p.add_argument("--cma_inc_pop_size", type=float, default=2.0)
    p.add_argument("--cma_verb_disp", type=int, default=0)

    # TabPFN model replacing the GP.
    p.add_argument("--target_mode", choices=["metric", "rank", "normal_rank"], default="rank")
    p.add_argument("--coordinate_mode", choices=["raw", "standardized", "cma_whitened"], default="raw")
    p.add_argument(
        "--selection_mode",
        choices=["all", "recent", "nearest", "hybrid"],
        default="all",
        help="Keep 'all' for the closest DTS replacement; DTS already selects archive-near-point training data.",
    )
    p.add_argument("--recent_fraction", type=float, default=0.1)
    p.add_argument("--min_train_size", type=int, default=5)
    p.add_argument("--max_train_size", type=int, default=1000)
    p.add_argument("--n_estimators", type=int, default=8)
    p.add_argument("--return_uncertainty", action="store_true")
    p.add_argument("--no_return_uncertainty", dest="return_uncertainty", action="store_false")
    p.set_defaults(return_uncertainty=True)

    # DTS parameters using public DoubleTrainedEC names and defaults.
    p.add_argument(
        "--dts_profile",
        choices=["dts_005", "dts_readme_02", "custom"],
        default="dts_005",
        help="dts_005 follows the public DoubleTrainedEC default/referenced 005 configuration; readme_02 uses 0.2.",
    )
    p.add_argument("--evoControlRestrictedParam", type=float, default=None)
    p.add_argument("--evoControlTrainRange", type=float, default=10.0)
    p.add_argument("--evoControlTrainNArchivePoints", type=str, default="15*dim")
    p.add_argument("--evoControlAcceptedModelAge", type=int, default=2)
    p.add_argument("--evoControlModelArchiveLength", type=int, default=5)
    p.add_argument("--evoControlUseDoubleTraining", action="store_true")
    p.add_argument("--no_evoControlUseDoubleTraining", dest="evoControlUseDoubleTraining", action="store_false")
    p.set_defaults(evoControlUseDoubleTraining=True)
    p.add_argument("--evoControlMaxDoubleTrainIterations", type=int, default=1)
    p.add_argument("--evoControlMinPointsForExpectedRank", type=int, default=4)
    p.add_argument("--evoControlOrigPointsRoundFcn", choices=["ceil", "floor", "round", "prob"], default="ceil")
    p.add_argument("--evoControlNBestPoints", type=str, default="0")
    p.add_argument("--evoControlPreselectionPopRatio", type=int, default=50)
    p.add_argument("--evoControlValidationGenerationPeriod", type=int, default=1)
    p.add_argument("--evoControlValidationPopSize", type=int, default=0)
    p.add_argument("--reevaluation_criterion", choices=["sd2", "expectedrank", "fvalues", "top"], default="sd2")
    p.add_argument("--no_shift_model_values", action="store_true")

    p.add_argument("--enable_coco_observer", action="store_true")
    p.add_argument("--disable_coco_observer", dest="enable_coco_observer", action="store_false")
    p.set_defaults(enable_coco_observer=True)
    p.add_argument("--enable_coco_postprocessing", action="store_true")
    p.add_argument("--disable_coco_postprocessing", dest="enable_coco_postprocessing", action="store_false")
    p.set_defaults(enable_coco_postprocessing=False)
    p.add_argument("--coco_result_folder", type=str, default=None)
    p.add_argument("--coco_algorithm_name", type=str, default=None)
    return p.parse_args()


def resolve_instances(args: argparse.Namespace) -> str:
    if args.instances:
        return args.instances
    if args.instances_start is None and args.instances_end is None:
        return "1-5,41-50"
    start = 1 if args.instances_start is None else args.instances_start
    end = start if args.instances_end is None else args.instances_end
    return str(start) if start == end else f"{start}-{end}"


def resolve_restricted_param(args: argparse.Namespace) -> float:
    if args.evoControlRestrictedParam is not None:
        return float(args.evoControlRestrictedParam)
    if args.dts_profile == "dts_readme_02":
        return 0.2
    # DoubleTrainedEC constructor default, and the COCO reference name DTS-CMA-ES_005 suggests this profile.
    return 0.05


def parse_n_best_points(s: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) == 0:
        return (0.0, 0.0)
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError("--evoControlNBestPoints must contain one or two comma-separated values")


def make_x0(args: argparse.Namespace, instance_id: int) -> np.ndarray:
    if args.init_mode == "zeros":
        return np.zeros(args.dimension, dtype=float)
    seed = (
        int(args.seed)
        + int(args.x0_seed_offset)
        + 1009 * int(args.function_id)
        + 9176 * int(instance_id)
        + 104729 * int(args.dimension)
    ) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return -4.0 + 8.0 * rng.random(args.dimension)


def build_cma_inopts(args: argparse.Namespace) -> dict[str, Any]:
    # These are the settings that are explicitly controlled to match the old Matlab s_cmaes setup.
    # PopSize is intentionally not set here unless --pop_size is provided.
    opts: dict[str, Any] = {
        "CMA_active": bool(args.cma_active),
        "verb_disp": int(args.cma_verb_disp),
        "verbose": -9 if int(args.cma_verb_disp) == 0 else 0,
    }
    if args.cma_restart_count:
        opts["restarts"] = int(args.cma_restart_count)
        opts["incpopsize"] = float(args.cma_inc_pop_size)
    return opts


def _coco_safe_token(value: str, max_len: int = 60) -> str:
    """Keep COCO observer option values short and whitespace-free.

    The COCO C observer has strict internal string-size limits; long
    algorithm_info strings can fail with `coco_vstrdupf(): string is too long`.
    """
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe[:max_len]


def build_observer(args: argparse.Namespace):
    if not args.enable_coco_observer:
        return None
    import cocoex

    folder = _coco_safe_token(
        args.coco_result_folder or f"{args.experiment_name}_d{args.dimension}_f{args.function_id}",
        max_len=80,
    )
    alg_name = _coco_safe_token(args.coco_algorithm_name or args.experiment_name, max_len=40)
    rp = resolve_restricted_param(args)
    ps = "def" if args.pop_size is None else str(args.pop_size)
    info = _coco_safe_token(
        f"DTSPFN_tm-{args.target_mode}_rp-{rp:g}_crit-{args.reevaluation_criterion}_init-{args.init_mode}_sig-{args.sigma0:g}_ps-{ps}",
        max_len=120,
    )
    options = f"result_folder: {folder} algorithm_name: {alg_name} algorithm_info: {info}"
    return cocoex.Observer("bbob", options)


def run_instances(args: argparse.Namespace):
    import cocoex

    suite = cocoex.Suite(
        "bbob",
        f"instances: {resolve_instances(args)}",
        f"dimensions: {args.dimension} function_indices: {args.function_id}",
    )
    observer = build_observer(args)
    summaries = []
    max_true_evals = args.max_true_evals if args.max_true_evals is not None else int(args.budget_multiplier * args.dimension)
    surrogate_kwargs = {
        "min_train_size": args.min_train_size,
        "max_train_size": args.max_train_size,
        "selection_mode": args.selection_mode,
        "recent_fraction": args.recent_fraction,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "return_uncertainty": args.return_uncertainty,
        "raise_on_error": True,
        "coordinate_mode": args.coordinate_mode,
    }

    for problem in suite:
        wrapped = CocoProblemWrapper(problem)
        if observer is not None:
            wrapped = wrapped.observe_with(observer)
        instance_id = int(problem.id_instance)
        exp = ExactLikeDTSTabPFNExperiment(
            x0=make_x0(args, instance_id),
            sigma0=float(args.sigma0),
            population_size=args.pop_size,
            seed=int(args.seed),
            max_true_evals=max_true_evals,
            max_generations=args.max_generations,
            target_f=None,
            experiment_name=args.experiment_name,
            function_id=args.function_id,
            instance_id=instance_id,
            dimension=args.dimension,
            output_dir=args.output_dir,
            surrogate_kwargs=surrogate_kwargs,
            target_mode=args.target_mode,
            evo_control_restricted_param=resolve_restricted_param(args),
            evo_control_train_range=args.evoControlTrainRange,
            evo_control_train_n_archive_points=args.evoControlTrainNArchivePoints,
            evo_control_accepted_model_age=args.evoControlAcceptedModelAge,
            evo_control_model_archive_length=args.evoControlModelArchiveLength,
            evo_control_use_double_training=args.evoControlUseDoubleTraining,
            evo_control_max_double_train_iterations=args.evoControlMaxDoubleTrainIterations,
            evo_control_min_points_for_expected_rank=args.evoControlMinPointsForExpectedRank,
            evo_control_orig_points_round_fcn=args.evoControlOrigPointsRoundFcn,
            evo_control_n_best_points=parse_n_best_points(args.evoControlNBestPoints),
            evo_control_preselection_pop_ratio=args.evoControlPreselectionPopRatio,
            evo_control_validation_generation_period=args.evoControlValidationGenerationPeriod,
            evo_control_validation_pop_size=args.evoControlValidationPopSize,
            reevaluation_criterion=args.reevaluation_criterion,
            shift_model_values=not args.no_shift_model_values,
            inopts=build_cma_inopts(args),
        )
        summary = exp.run(wrapped)
        summaries.append(summary)
        print(
            f"run={summary.run_id} generations={summary.generations} "
            f"true_evals={summary.true_evaluations} best_y={summary.best_y:.6e}"
        )
    return summaries


def maybe_postprocess(args: argparse.Namespace) -> None:
    if not args.enable_coco_observer or not args.enable_coco_postprocessing:
        return
    try:
        import cocopp
    except ImportError:
        print("[cocopp] not installed; skipping postprocessing")
        return
    folder = args.coco_result_folder or f"{args.experiment_name}_bbob_dim{args.dimension}_f{args.function_id}"
    exdata = Path("exdata")
    candidates = sorted([p for p in exdata.glob(f"{folder}*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"[cocopp] no exdata folder matching {folder}*")
        return
    result_folder = candidates[-1]
    out = args.output_dir / args.experiment_name / "cocopp"
    out.mkdir(parents=True, exist_ok=True)
    original_open = webbrowser.open
    try:
        webbrowser.open = lambda *a, **k: False
        cocopp.main(["-o", str(out), str(result_folder)])
    finally:
        webbrowser.open = original_open
    print(f"[cocopp] HTML report written to: {out}")


def main() -> None:
    args = parse_args()
    # Save the exact command configuration for reproducibility.
    cfg_dir = args.output_dir / args.experiment_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with (cfg_dir / "runner_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    run_instances(args)
    maybe_postprocess(args)


if __name__ == "__main__":
    main()
