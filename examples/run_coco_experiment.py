from __future__ import annotations

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
from pfn_cmaes.interfaces import DecisionModel
from pfn_cmaes.optimizers.cmaes import PyCMAOptimizerBackend
from pfn_cmaes.stubs.decision_models import (
    PFNBackboneConfig,
    SetConditionedPFNBackbone,
)
from pfn_cmaes.stubs.evolution_controls import (
    AdaptiveModelLifelengthControl,
    DoublyTrainedControl,
    EvaluateAll,
    EvaluateTopFraction,
    TopFractionPlusUncertaintyControl,
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
from pfn_cmaes.types import GenerationState, SurrogateDecision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFN-driven surrogate-assisted CMA-ES on COCO.")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--instances_start", type=int, default=1)
    parser.add_argument("--instances_end", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_generations", type=int, default=50)
    parser.add_argument("--max_true_evals", type=int, default=500)
    parser.add_argument("--sigma0", type=float, default=2.0)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--generate_dataset", action="store_true")
    parser.add_argument("--train_decision_model", action="store_true")
    parser.add_argument(
        "--fixed_surrogate",
        type=str,
        default=None,
        help="Bypass PFN and always choose this surrogate bundle in decision mode.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results"))

    parser.add_argument("--train_epochs", type=int, default=25)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_hidden_dim", type=int, default=128)
    parser.add_argument("--train_max_records", type=int, default=0)
    parser.add_argument("--train_shuffle_buffer", type=int, default=4096)
    parser.add_argument("--train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


class FixedSurrogateDecisionModel(DecisionModel):
    def __init__(self, surrogate_name: str):
        self.surrogate_name = surrogate_name

    def score(self, state: GenerationState, surrogate_names: list[str]) -> SurrogateDecision:
        if self.surrogate_name not in surrogate_names:
            raise ValueError(
                f"Unknown fixed surrogate {self.surrogate_name!r}. Available surrogate bundles: {surrogate_names!r}"
            )

        return SurrogateDecision(
            goodness={name: 1.0 if name == self.surrogate_name else 0.0 for name in surrogate_names},
            chosen_surrogate_name=self.surrogate_name,
            metadata={
                "model_type": "fixed_surrogate_decision_model",
                "fixed_surrogate": self.surrogate_name,
            },
        )


def build_surrogate_specs() -> list[SurrogateSpec]:
    """
    Build the experimental set of surrogate + evolution-control bundles.
    """
    return [
        SurrogateSpec(
            name="lq_linear_top50",
            surrogate_cls=LocalLinearSurrogate,
            surrogate_kwargs={
                "ridge": 1e-6,
                "lengthscale": 1.0,
                "max_train_size": 200,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=EvaluateTopFraction,
            evolution_control_kwargs={
                "fraction": 0.5,
            },
        ),
        SurrogateSpec(
            name="lq_quadratic_top30",
            surrogate_cls=LocalQuadraticSurrogate,
            surrogate_kwargs={
                "ridge": 1e-5,
                "lengthscale": 1.0,
                "max_train_size": 200,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=EvaluateTopFraction,
            evolution_control_kwargs={
                "fraction": 0.3,
            },
        ),
        SurrogateSpec(
            name="gp_matern_dts",
            surrogate_cls=GaussianProcessMaternSurrogate,
            surrogate_kwargs={
                "nu": 2.5,
                "alpha": 1e-6,
                "normalize_y": True,
                "return_std": True,
                "max_train_size": 200,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=DoublyTrainedControl,
            evolution_control_kwargs={
                "refresh_interval": 5,
                "partial_fraction": 0.3,
                "exploit_multiplier_if_good": 1.2,
                "exploit_multiplier_if_bad": 0.8,
                "agreement_threshold": 0.65,
            },
        ),
        SurrogateSpec(
            name="gp_matern_unc",
            surrogate_cls=GaussianProcessMaternSurrogate,
            surrogate_kwargs={
                "nu": 2.5,
                "alpha": 1e-6,
                "normalize_y": True,
                "return_std": True,
                "max_train_size": 200,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=TopFractionPlusUncertaintyControl,
            evolution_control_kwargs={
                "top_fraction": 0.3,
                "uncertainty_fraction": 0.2,
            },
        ),
        SurrogateSpec(
            name="rf_lifelength",
            surrogate_cls=RandomForestSurrogate,
            surrogate_kwargs={
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "random_state": 0,
                "return_std": True,
                "max_train_size": 400,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=AdaptiveModelLifelengthControl,
            evolution_control_kwargs={
                "initial_lifelength": 1,
                "max_lifelength": 5,
                "min_lifelength": 1,
                "top_fraction": 0.5,
                "good_agreement_threshold": 0.7,
                "bad_agreement_threshold": 0.3,
            },
        ),
        SurrogateSpec(
            name="svm_rank_top50",
            surrogate_cls=RankSVMSurrogate,
            surrogate_kwargs={
                "C": 10.0,
                "epsilon": 0.01,
                "gamma": "scale",
                "max_train_size": 400,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=EvaluateTopFraction,
            evolution_control_kwargs={
                "fraction": 0.5,
            },
        ),
        SurrogateSpec(
            name="real_only",
            surrogate_cls=LocalLinearSurrogate,
            surrogate_kwargs={
                "ridge": 1e-6,
                "lengthscale": 1.0,
                "max_train_size": 200,
                "selection_mode": "hybrid",
                "recent_fraction": 0.25,
            },
            evolution_control_cls=EvaluateAll,
            evolution_control_kwargs={},
        ),
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
) -> DecisionModel:
    if args.fixed_surrogate is not None:
        if args.fixed_surrogate not in surrogate_names:
            raise ValueError(
                f"Unknown --fixed_surrogate {args.fixed_surrogate!r}. "
                f"Available surrogate bundles: {list(surrogate_names)!r}"
            )
        print(f"[decision_model] using fixed surrogate: {args.fixed_surrogate}")
        return FixedSurrogateDecisionModel(args.fixed_surrogate)

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
    surrogate_specs = build_surrogate_specs()
    surrogate_names = [spec.name for spec in surrogate_specs]

    decision_model = build_or_load_decision_model(
        args=args,
        surrogate_names=surrogate_names,
    )

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
    surrogate_specs = build_surrogate_specs()
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
        shuffle_buffer_size=args.train_shuffle_buffer,
        steps_per_epoch=None if args.train_steps_per_epoch <= 0 else args.train_steps_per_epoch,
        seed=args.seed,
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
