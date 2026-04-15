from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .interfaces import DecisionModel, EvolutionControl, OptimizerBackend, SurrogateModel


@dataclass(slots=True)
class SurrogateSpec:
    """Factory configuration for one surrogate + evolution control pair."""

    name: str
    surrogate_cls: Type[SurrogateModel]
    evolution_control_cls: Type[EvolutionControl]
    surrogate_kwargs: Dict[str, Any] = field(default_factory=dict)
    evolution_control_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    generate_dataset: bool = False
    flush_every: int = 10
    metric_eps: float = 1e-12
    metric_clip_value: float = 5.0
    keep_full_history_in_records: bool = True


@dataclass(slots=True)
class LoggingConfig:
    output_dir: Path = Path("results")
    save_jsonl_logs: bool = True
    save_dataset_jsonl: bool = True
    save_numpy_snapshots: bool = False


@dataclass(slots=True)
class RunConfig:
    experiment_name: str
    seed: int
    dimension: int
    function_id: int
    instance_id: int
    max_generations: int
    max_true_evals: int
    target_f: Optional[float] = None


@dataclass(slots=True)
class ExperimentConfig:
    run: RunConfig
    optimizer_backend_cls: Type[OptimizerBackend]
    optimizer_kwargs: Dict[str, Any]
    surrogate_specs: list[SurrogateSpec]
    decision_model: DecisionModel
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
