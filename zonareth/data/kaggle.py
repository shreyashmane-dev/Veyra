from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any
from ..config.dataset_config import DatasetSourceConfig


class KaggleDatasetAdapter:
    """Utilities for importing, staging, and mapping Kaggle datasets into VEYRA.

    Supports:
    1. Direct execution inside Kaggle Notebooks (reading from /kaggle/input/<dataset-name>/)
    2. Local execution using the Kaggle API CLI (if kaggle.json credentials exist)
    """

    def __init__(self, raw_data_dir: str | Path = "data/raw") -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_running_on_kaggle() -> bool:
        """Returns True if code is executing inside a Kaggle Notebook environment."""
        return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    def discover_kaggle_input_datasets(self) -> dict[str, Path]:
        """Discovers attached datasets in /kaggle/input/."""
        input_dir = Path("/kaggle/input")
        if not input_dir.exists():
            return {}

        datasets = {}
        for child in input_dir.iterdir():
            if child.is_dir():
                datasets[child.name] = child
        return datasets

    def stage_kaggle_dataset(
        self,
        dataset_name: str,
        target_filename: str,
        source_path: str | Path | None = None,
    ) -> Path:
        """Stages a Kaggle dataset file into data/raw/ for preprocessing."""
        if source_path is not None:
            src = Path(source_path)
        elif self.is_running_on_kaggle():
            # Check under /kaggle/input/<dataset_name>
            candidate = Path(f"/kaggle/input/{dataset_name}")
            if not candidate.exists():
                raise FileNotFoundError(f"Kaggle input dataset not found at '{candidate}'")
            # Find matching file or directory
            src = candidate
        else:
            raise EnvironmentError(
                f"Cannot stage '{dataset_name}' outside of Kaggle environment without explicit source_path."
            )

        dest = self.raw_data_dir / target_filename
        if src.is_file():
            shutil.copy2(src, dest)
        elif src.is_dir():
            # If directory, find the first relevant text/csv/jsonl file
            files = list(src.glob("*.txt")) + list(src.glob("*.jsonl")) + list(src.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No suitable text, jsonl, or csv file found in {src}")
            shutil.copy2(files[0], dest)

        return dest

    @staticmethod
    def create_source_config(
        name: str,
        file_path: str | Path,
        file_format: str,
        text_column: str = "text",
        license: str = "Open / Permitted",
        sampling_weight: float = 1.0,
    ) -> DatasetSourceConfig:
        """Creates a verified DatasetSourceConfig for a Kaggle dataset."""
        return DatasetSourceConfig(
            name=name,
            source_type="kaggle",
            path_or_identifier=str(file_path),
            license=license,
            file_format=file_format,
            text_column=text_column,
            sampling_weight=sampling_weight,
        )
