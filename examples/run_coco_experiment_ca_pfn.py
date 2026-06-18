from __future__ import annotations

import os

os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_POSTHOG_ANALYTICS"] = "1"
os.environ["ANALYTICS_DISABLED"] = "1"

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pfn_cmaes.coco.runner import CocoExperimentRunner
from pfn_cmaes.config import (
    DatasetConfig,
    ExperimentConfig,
    LoggingConfig,
    RunConfig,
    SurrogateSpec,
)
from pfn_cmaes.decision.pfn_model import (
    PFNDecisionConfig,
    PFNDecisionModel,
    PFNStateFeaturizer,
)
from pfn_cmaes.optimizers.cmaes import PyCMAOptimizerBackend
from pfn_cmaes.stubs.decision_models import (
    PFNBackboneConfig,
    SetConditionedPFNBackbone,
    UniformDecisionModel,
)
from pfn_cmaes.stubs.evolution_controls import (
    AdaptiveModelLifelengthControl,
    DoublyTrainedControl,
    EvaluateAll,
    EvaluateTopFraction,
    TopFractionPlusUncertaintyControl,
    TrustRegionAdaptiveRankControl,
)
from pfn_cmaes.stubs.surrogates import (
    GaussianProcessMaternSurrogate,
    LocalLinearSurrogate,
    LocalQuadraticSurrogate,
    RandomForestSurrogate,
    RankSVMSurrogate,
)
from pfn_cmaes.training.decision_model_trainer import (
    DecisionTrainingConfig,
    train_decision_model,
)
from pfn_cmaes.types import GenerationState

from pfn_cmaes.surrogates.tabpfn_surrogate import TabPFNSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFN-driven surrogate-assisted CMA-ES on COCO.")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--instances_start", type=int, default=1)
    parser.add_argument("--instances_end", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_generations", type=int, default=50000)
    parser.add_argument("--max_true_evals", type=int, default=500)
    parser.add_argument("--sigma0", type=float, default=2.0)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--generate_dataset", action="store_true")
    parser.add_argument("--train_decision_model", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))

    parser.add_argument("--train_epochs", type=int, default=25)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_hidden_dim", type=int, default=128)
    parser.add_argument("--train_max_records", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--max_train_size", type=int, default=100)
    parser.add_argument("--max_train_size_ratio", type=float, default=None)
    parser.add_argument("--recent_fraction", type=float, default=0.35)
    parser.add_argument("--top_fraction", type=float, default=0.4)
    parser.add_argument("--uncertainty_fraction", type=float, default=0.0)
    parser.add_argument("--evolution_control_regime", type=str, default="static")

    # Covariance-aware TabPFN options.
    parser.add_argument(
        "--coordinate_mode",
        type=str,
        default="raw",
        choices=["raw", "standardized", "cma_whitened"],
        help="Coordinates used as TabPFN inputs. Use cma_whitened for ill-conditioned experiments.",
    )
    parser.add_argument(
        "--evolution_control",
        type=str,
        default="top_uncertainty",
        choices=["top_fraction", "top_uncertainty", "trust_adaptive", "evaluate_all"],
        help="Evolution-control strategy for the selected surrogate bundle.",
    )
    parser.add_argument("--trust_region_radius", type=float, default=None)
    parser.add_argument("--min_top_fraction", type=float, default=0.2)
    parser.add_argument("--max_top_fraction", type=float, default=1.0)
    parser.add_argument("--adaptation_step", type=float, default=0.1)
    parser.add_argument("--good_rank_agreement", type=float, default=0.75)
    parser.add_argument("--bad_rank_agreement", type=float, default=0.55)
    parser.add_argument("--min_validation_points", type=int, default=2)
    parser.add_argument("--target_mode", type=str, default="reg", choices=["reg", "rank"])
    parser.add_argument("--surrogate_name", type=str, default=None)

    return parser.parse_args()


def build_surrogate_specs(args) -> list[SurrogateSpec]:
    """
    Build the experimental surrogate + evolution-control bundle.

    The important new combination for the next paper is:
      --coordinate_mode cma_whitened --evolution_control trust_adaptive
    """
    surrogate_kwargs = {
        "min_train_size": 5,
        "max_train_size": args.max_train_size,
        "max_train_size_ratio": args.max_train_size_ratio,
        "selection_mode": "hybrid",
        "recent_fraction": args.recent_fraction,
        "device": args.device,
        "n_estimators": 8,
        "return_uncertainty": args.uncertainty_fraction > 0.0,
        "raise_on_error": True,
        "coordinate_mode": args.coordinate_mode,
        "target_mode": args.target_mode,
    }

    if args.evolution_control == "evaluate_all":
        evolution_control_cls = EvaluateAll
        evolution_control_kwargs = {}
    elif args.evolution_control == "top_fraction":
        evolution_control_cls = EvaluateTopFraction
        evolution_control_kwargs = {"fraction": args.top_fraction}
    elif args.evolution_control == "top_uncertainty":
        evolution_control_cls = TopFractionPlusUncertaintyControl
        evolution_control_kwargs = {
            "top_fraction": args.top_fraction,
            "uncertainty_fraction": args.uncertainty_fraction,
            "regime": args.evolution_control_regime,
        }
    elif args.evolution_control == "trust_adaptive":
        evolution_control_cls = TrustRegionAdaptiveRankControl
        evolution_control_kwargs = {
            "top_fraction": args.top_fraction,
            "min_top_fraction": args.min_top_fraction,
            "max_top_fraction": args.max_top_fraction,
            "adaptation_step": args.adaptation_step,
            "good_agreement_threshold": args.good_rank_agreement,
            "bad_agreement_threshold": args.bad_rank_agreement,
            "trust_region_radius": args.trust_region_radius,
            "min_validation_points": args.min_validation_points,
            "uncertainty_fraction": args.uncertainty_fraction,
            "random_state": args.seed,
        }
    else:  # pragma: no cover; argparse choices prevent this.
        raise ValueError(f"Unknown evolution_control={args.evolution_control!r}")

    name = args.surrogate_name
    if name is None:
        name = f"tabpfn_{args.coordinate_mode}_{args.evolution_control}"

    return [
        SurrogateSpec(
            name=name,
            surrogate_cls=TabPFNSurrogate,
            evolution_control_cls=evolution_control_cls,
            surrogate_kwargs=surrogate_kwargs,
            evolution_control_kwargs=evolution_control_kwargs,
        )
    ]


def build_bootstrap_state(
    dimension: int,
    population_size: int,
    seed: int,
) -> GenerationState:
    rng = np.random.default_rng(seed)
    history_size = max(8, 2 * population_size)

    evaluated_history_x = rng.normal(size=(history_size, dimension)).astype(np.float32)
    evaluated_history_y = rng.normal(size=(history_size,)).astype(np.float32)
    candidate_x = rng.normal(size=(population_size, dimension)).astype(np.float32)

    incumbent_idx = int(np.argmin(evaluated_history_y))

    return GenerationState(
        generation_index=0,
        evaluated_history_x=evaluated_history_x,
        evaluated_history_y=evaluated_history_y,
        candidate_x=candidate_x,
        incumbent_x=evaluated_history_x[incumbent_idx],
        incumbent_y=float(evaluated_history_y[incumbent_idx]),
        optimizer_state={
            "dimension": dimension,
            "population_size": population_size,
            "sigma": 1.0,
            "num_true_evals": history_size,
            "target_mode": "rank",
        },
        metadata={},
    )


def build_dummy_pfn_decision_model(
    *,
    dimension: int,
    population_size: int,
    surrogate_names: Sequence[str],
    seed: int,
    device: str = "cpu",
) -> PFNDecisionModel:
    pfn_config = PFNDecisionConfig(
        checkpoint_path=None,
        device=device,
        max_history=128,
        temperature=1.0,
        tie_margin=1e-3,
    )

    bootstrap_state = build_bootstrap_state(
        dimension=dimension,
        population_size=population_size,
        seed=seed,
    )

    featurizer = PFNStateFeaturizer(pfn_config)
    context_x, _, candidate_x, action_ids, _ = featurizer.build(
        state=bootstrap_state,
        surrogate_names=list(surrogate_names),
    )

    backbone = SetConditionedPFNBackbone(
        context_dim=int(context_x.shape[1]),
        candidate_dim=int(candidate_x.shape[1]),
        config=PFNBackboneConfig(
            hidden_dim=256,
            num_heads=8,
            num_context_layers=4,
            num_candidate_layers=2,
            num_action_layers=2,
            ff_multiplier=4,
            dropout=0.1,
            activation="gelu",
            use_type_embeddings=True,
            max_action_tokens=max(64, len(action_ids)),
        ),
    )

    return PFNDecisionModel(
        backbone=backbone,
        config=pfn_config,
    )


def build_checkpoint_path(
    *,
    output_dir: Path,
    experiment_name: str,
    dimension: int,
    function_id: int,
) -> Path:
    return output_dir / experiment_name / "models" / f"decision_model_dim{dimension}_heldout_f{function_id}.pt"


def build_trained_pfn_decision_model(
    *,
    checkpoint_path: Path,
    surrogate_names: Sequence[str],
    device: str = "cpu",
) -> PFNDecisionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    trained_names = list(checkpoint.get("surrogate_names", []))
    if trained_names and list(surrogate_names) != trained_names:
        raise ValueError(
            "Runtime surrogate_names do not match checkpoint surrogate_names.\n"
            f"runtime={list(surrogate_names)!r}\n"
            f"checkpoint={trained_names!r}"
        )

    cfg = checkpoint["backbone_config"]

    backbone = SetConditionedPFNBackbone(
        context_dim=int(checkpoint["context_dim"]),
        candidate_dim=int(checkpoint["candidate_dim"]),
        config=PFNBackboneConfig(**cfg),
    )

    if len(surrogate_names) > backbone.config.max_action_tokens:
        raise ValueError(
            f"Checkpoint supports at most {backbone.config.max_action_tokens} action tokens, "
            f"but runtime has {len(surrogate_names)} surrogate bundles."
        )

    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.eval()

    pfn_config_dict = checkpoint.get("pfn_config", {})
    pfn_config = PFNDecisionConfig(
        checkpoint_path=str(checkpoint_path),
        device=device,
        dtype=str(pfn_config_dict.get("dtype", "float32")),
        max_history=int(pfn_config_dict.get("max_history", 128)),
        normalize_targets=bool(pfn_config_dict.get("normalize_targets", True)),
        include_ranks=bool(pfn_config_dict.get("include_ranks", True)),
        include_recency=bool(pfn_config_dict.get("include_recency", True)),
        include_optimizer_features_in_context=bool(pfn_config_dict.get("include_optimizer_features_in_context", True)),
        temperature=float(pfn_config_dict.get("temperature", 1.0)),
        tie_margin=float(pfn_config_dict.get("tie_margin", 1e-3)),
    )

    return PFNDecisionModel(backbone=backbone, config=pfn_config)


def build_or_load_decision_model(
    *,
    args: argparse.Namespace,
    surrogate_names: Sequence[str],
) -> PFNDecisionModel:
    checkpoint_path = build_checkpoint_path(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        dimension=args.dimension,
        function_id=args.function_id,
    )

    if checkpoint_path.exists():
        print(f"[decision_model] loading trained checkpoint: {checkpoint_path}")
        return build_trained_pfn_decision_model(
            checkpoint_path=checkpoint_path,
            surrogate_names=surrogate_names,
            device=args.device,
        )

    print("[decision_model] checkpoint not found, falling back to dummy backbone")
    return build_dummy_pfn_decision_model(
        dimension=args.dimension,
        population_size=args.pop_size,
        surrogate_names=surrogate_names,
        seed=args.seed,
        device=args.device,
    )


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    surrogate_specs = build_surrogate_specs(args)
    surrogate_names = [spec.name for spec in surrogate_specs]

    decision_model = UniformDecisionModel()
    #decision_model = build_or_load_decision_model(
    #    args=args,
    #    surrogate_names=surrogate_names,
    #)

    run_cfg = RunConfig(
        experiment_name=args.experiment_name,
        seed=args.seed,
        dimension=args.dimension,
        function_id=args.function_id,
        instance_id=args.instances_start,
        max_generations=args.max_generations,
        max_true_evals=args.max_true_evals,
        target_f=None,
    )

    x0 = np.zeros(args.dimension, dtype=float)

    return ExperimentConfig(
        run=run_cfg,
        optimizer_backend_cls=PyCMAOptimizerBackend,
        optimizer_kwargs={
            "x0": x0,
            "sigma0": args.sigma0,
            "population_size": args.pop_size,
            "seed": args.seed,
            "inopts": {},
        },
        surrogate_specs=surrogate_specs,
        decision_model=decision_model,
        dataset=DatasetConfig(generate_dataset=args.generate_dataset),
        logging=LoggingConfig(output_dir=args.output_dir),
    )


def maybe_train_decision_model(args: argparse.Namespace) -> None:
    surrogate_specs = build_surrogate_specs(args)
    surrogate_names = [spec.name for spec in surrogate_specs]

    experiment_root = args.output_dir / args.experiment_name
    pfn_config = PFNDecisionConfig(
        checkpoint_path=None,
        device=args.device,
        max_history=128,
        normalize_targets=True,
        include_ranks=True,
        include_recency=True,
        include_optimizer_features_in_context=True,
        temperature=1.0,
        tie_margin=1e-3,
    )

    training_config = DecisionTrainingConfig(
        batch_size=args.train_batch_size,
        epochs=args.train_epochs,
        learning_rate=args.train_lr,
        weight_decay=args.train_weight_decay,
        hidden_dim=args.train_hidden_dim,
        device=args.device,
        max_records=None if args.train_max_records <= 0 else args.train_max_records,
    )

    train_decision_model(
        experiment_root=experiment_root,
        target_dimension=args.dimension,
        held_out_function_id=args.function_id,
        surrogate_names=surrogate_names,
        pfn_config=pfn_config,
        training_config=training_config,
    )


def maybe_run_coco_postprocessing(
    *,
    config: ExperimentConfig,
    runner: CocoExperimentRunner,
) -> None:
    if config.dataset.generate_dataset:
        return
    if not config.logging.enable_coco_observer:
        return
    if not config.logging.enable_coco_postprocessing:
        return

    result_folder = runner.get_coco_result_folder_path()
    if result_folder is None:
        return

    try:
        import cocopp
    except ImportError as exc:
        raise ImportError(
            "cocopp is required for COCO HTML postprocessing. Install it with: python -m pip install cocopp"
        ) from exc

    output_dir = (
        Path(config.logging.output_dir) / config.run.experiment_name / config.logging.coco_postprocess_output_dirname
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # cocopp.main supports command-line style invocation and generates HTML output
    import webbrowser

    original_open = webbrowser.open
    original_open_new = getattr(webbrowser, "open_new", None)
    original_open_new_tab = getattr(webbrowser, "open_new_tab", None)

    try:
        webbrowser.open = lambda *args, **kwargs: False
        if original_open_new is not None:
            webbrowser.open_new = lambda *args, **kwargs: False
        if original_open_new_tab is not None:
            webbrowser.open_new_tab = lambda *args, **kwargs: False

        cocopp.main(
            [
                "-o",
                str(output_dir),
                str(result_folder),
            ]
        )
    finally:
        webbrowser.open = original_open
        if original_open_new is not None:
            webbrowser.open_new = original_open_new
        if original_open_new_tab is not None:
            webbrowser.open_new_tab = original_open_new_tab

    print(f"[cocopp] HTML report written to: {output_dir}")


def main() -> None:
    args = parse_args()

    if args.train_decision_model:
        maybe_train_decision_model(args)
        return

    config = build_experiment_config(args)
    runner = CocoExperimentRunner(config)
    summaries = runner.run_instances(range(args.instances_start, args.instances_end + 1))

    maybe_run_coco_postprocessing(
        config=config,
        runner=runner,
    )

    for summary in summaries:
        print(
            f"run={summary.run_id} "
            f"generations={summary.generations} "
            f"true_evals={summary.true_evaluations} "
            f"best_y={summary.best_y:.6e}"
        )


if __name__ == "__main__":
    main()
