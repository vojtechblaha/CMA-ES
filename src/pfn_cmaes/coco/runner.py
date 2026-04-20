from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import cocoex

from ..config import ExperimentConfig
from ..experiment import PFNSurrogateExperiment
from ..types import RunSummary
from .objectives import CocoProblemWrapper


class CocoExperimentRunner:
    """Runs the PFN-surrogate experiment over selected COCO instances."""

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config

    def _build_observer(self) -> cocoex.Observer | None:
        logging_cfg = self.base_config.logging
        run_cfg = self.base_config.run
        dataset_mode = self.base_config.dataset.generate_dataset

        # Only benchmark with COCO observer in testing mode
        if dataset_mode:
            return None
        if not logging_cfg.enable_coco_observer:
            return None

        result_folder = (
            logging_cfg.coco_result_folder
            or f"{run_cfg.experiment_name}_bbob_dim{run_cfg.dimension}_f{run_cfg.function_id}"
        )
        algorithm_name = logging_cfg.coco_algorithm_name or f"pfn_cmaes/{run_cfg.experiment_name}"
        algorithm_info = logging_cfg.coco_algorithm_info or (
            "PFN-driven surrogate-assisted CMA-ES; "
            f"function={run_cfg.function_id}, dim={run_cfg.dimension}, seed={run_cfg.seed}"
        )

        observer_options = (
            f"result_folder: {result_folder} algorithm_name: {algorithm_name} algorithm_info: {algorithm_info}"
        )

        return cocoex.Observer("bbob", observer_options)

    def run_instances(self, instance_ids: Iterable[int]) -> list[RunSummary]:
        run_cfg = self.base_config.run
        summaries: list[RunSummary] = []
        observer = self._build_observer()

        for instance_id in instance_ids:
            suite = cocoex.Suite(
                "bbob",
                "",
                (
                    f"dimensions:{run_cfg.dimension} "
                    f"function_indices:{run_cfg.function_id} "
                    f"instance_indices:{instance_id}"
                ),
            )
            for problem in suite:
                wrapped_problem = CocoProblemWrapper(problem)
                if observer is not None:
                    wrapped_problem = wrapped_problem.observe_with(observer)

                cfg = replace(
                    self.base_config,
                    run=replace(self.base_config.run, instance_id=instance_id),
                )
                experiment = PFNSurrogateExperiment(cfg)
                summary = experiment.run(wrapped_problem)
                summaries.append(summary)

        return summaries

    def get_coco_result_folder_path(self) -> Path | None:
        logging_cfg = self.base_config.logging
        if self.base_config.dataset.generate_dataset:
            return None
        if not logging_cfg.enable_coco_observer:
            return None

        run = self.base_config.run
        result_folder = (
            logging_cfg.coco_result_folder or f"{run.experiment_name}_bbob_dim{run.dimension}_f{run.function_id}"
        )

        # cocoex writes results under exdata/
        exdata_dir = Path("exdata")
        if not exdata_dir.exists():
            return None

        # Observer may append suffixes like -0001, -0002 if folder already exists.
        candidates = sorted(exdata_dir.glob(f"{result_folder}*"))
        if not candidates:
            return exdata_dir / result_folder

        # Pick the most recently modified matching result directory.
        candidates = [p for p in candidates if p.is_dir()]
        if not candidates:
            return exdata_dir / result_folder

        return max(candidates, key=lambda p: p.stat().st_mtime)
