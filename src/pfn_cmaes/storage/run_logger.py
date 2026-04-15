from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import LoggingConfig
from ..types import DatasetRecord, GenerationLog, RunSummary
from .jsonl import JSONLWriter


class RunLogger:
    def __init__(self, config: LoggingConfig, experiment_name: str, run_id: str):
        self.config = config
        base_dir = Path(config.output_dir) / experiment_name / run_id
        self.generation_writer: Optional[JSONLWriter] = None
        self.dataset_writer: Optional[JSONLWriter] = None
        self.summary_writer: Optional[JSONLWriter] = None

        if config.save_jsonl_logs:
            self.generation_writer = JSONLWriter(base_dir / "generation_logs.jsonl")
            self.summary_writer = JSONLWriter(base_dir / "run_summary.jsonl")
        if config.save_dataset_jsonl:
            self.dataset_writer = JSONLWriter(base_dir / "dataset.jsonl")

    def log_generation(self, log: GenerationLog) -> None:
        if self.generation_writer is not None:
            self.generation_writer.append(asdict(log))

    def log_dataset_record(self, record: DatasetRecord) -> None:
        if self.dataset_writer is not None:
            self.dataset_writer.append(asdict(record))

    def log_summary(self, summary: RunSummary) -> None:
        if self.summary_writer is not None:
            self.summary_writer.append(asdict(summary))
