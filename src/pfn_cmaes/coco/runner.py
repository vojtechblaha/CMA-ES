from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

import cocoex
import numpy as np

from ..config import ExperimentConfig
from ..experiment import PFNSurrogateExperiment
from ..types import RunSummary
from .objectives import CocoProblemWrapper


class CocoExperimentRunner:
    """Runs the PFN-surrogate experiment over selected COCO instances."""

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config

    def run_instances(self, instance_ids: Iterable[int]) -> list[RunSummary]:
        run_cfg = self.base_config.run
        summaries: list[RunSummary] = []

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
                cfg = replace(
                    self.base_config,
                    run=replace(self.base_config.run, instance_id=instance_id),
                )
                experiment = PFNSurrogateExperiment(cfg)
                summary = experiment.run(CocoProblemWrapper(problem))
                summaries.append(summary)
        return summaries
