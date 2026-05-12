from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .config import InferenceConfig
from .datasets import DatasetExample
from .statistics import BranchStatistics


def make_run_dir(config: InferenceConfig, project_root: Path) -> Path:
    output_root = Path(config.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return (
        output_root
        / config.method
        / f"{sanitize_filename(config.model_name)}_{sanitize_filename(config.dataset_name)}@{config.branch_num}"
        / timestamp
    )


def make_question_dir(run_dir: Path, example: DatasetExample) -> Path:
    return run_dir / f"question_{example.index:04d}_{sanitize_filename(example.example_id)}"


def write_branch_statistics(question_dir: Path, stats: BranchStatistics) -> Path:
    question_dir.mkdir(parents=True, exist_ok=True)
    path = question_dir / f"branch_{stats.branch_index:04d}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_json_dict(), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    return safe.strip("_") or "unknown"

