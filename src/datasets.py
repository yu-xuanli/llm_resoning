from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QUESTION_FIELDS = ("question", "problem", "input", "query", "prompt")
DEFAULT_ANSWER_FIELDS = ("answer", "target", "label", "output", "solution", "gold")


@dataclass(frozen=True)
class DatasetExample:
    index: int
    example_id: str
    question: str
    answer: str | None
    answer_instruction: str | None
    raw_record: dict[str, Any]


class BaseDatasetReader(ABC):
    # Each dataset subclass owns its schema defaults. This keeps benchmark
    # knowledge close to the reader that normalizes raw records.
    default_question_field: str | None = None
    default_answer_field: str | None = None
    default_answer_instruction: str | None = None

    def __init__(
        self,
        path: str | Path,
        question_field: str | None = None,
        answer_field: str | None = None,
        answer_instruction: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.question_field = question_field or self.default_question_field
        self.answer_field = answer_field or self.default_answer_field
        # Per-reader answer instruction (e.g. "an integer", "A, B, C, or D")
        self.answer_instruction = answer_instruction or self.default_answer_instruction

    @abstractmethod
    def read_raw_records(self) -> list[dict[str, Any]]:
        """Return raw dataset records."""

    def read_examples(
        self,
        num_examples: int | None = None,
        example_ids: list[str] | None = None,
    ) -> list[DatasetExample]:
        records = self.read_raw_records()
        if example_ids is not None:
            records = _filter_records_by_id(records, example_ids)
        # Apply the run limit before normalization so every reader subclass
        # gets consistent "only run N problems" behavior.
        if num_examples is not None:
            records = records[:num_examples]
        return [
            self.normalize_record(record, index)
            for index, record in enumerate(records, start=1)
        ]

    def normalize_record(self, record: dict[str, Any], index: int) -> DatasetExample:
        question = _get_field(record, self.question_field, DEFAULT_QUESTION_FIELDS)
        answer = _get_field(record, self.answer_field, DEFAULT_ANSWER_FIELDS)
        if question is None:
            raise ValueError(
                f"example {index} does not contain a question field; "
                f"set question_field or use one of {DEFAULT_QUESTION_FIELDS}"
            )
        return DatasetExample(
            index=index,
            example_id=str(record.get("id", index)),
            question=str(question),
            answer=None if answer is None else str(answer),
            answer_instruction=self.answer_instruction,
            raw_record=record,
        )


class FileDatasetReader(BaseDatasetReader):
    def read_raw_records(self) -> list[dict[str, Any]]:
        # The configured dataset path may be a direct data file or a local
        # Hugging Face-style directory; resolve to the real data file first.
        dataset_file = _resolve_dataset_file(self.path)
        suffix = dataset_file.suffix.lower()
        if suffix == ".json":
            return _read_json(dataset_file)
        if suffix == ".jsonl":
            return _read_jsonl(dataset_file)
        if suffix == ".csv":
            return _read_csv(dataset_file)
        if suffix == ".parquet":
            return _read_parquet(dataset_file)
        raise ValueError("unsupported dataset format; use .json, .jsonl, .csv, or .parquet")


class Aime24DatasetReader(FileDatasetReader):
    default_question_field = "problem"
    default_answer_field = "solution"
    default_answer_instruction = "an integer"


class Aime25DatasetReader(FileDatasetReader):
    default_question_field = "problem"
    default_answer_field = "answer"
    default_answer_instruction = "an integer"


class Aime26DatasetReader(FileDatasetReader):
    default_question_field = "problem"
    default_answer_field = "answer"
    default_answer_instruction = "an integer"


class Amc23DatasetReader(FileDatasetReader):
    default_question_field = "question"
    default_answer_field = "answer"
    default_answer_instruction = "an integer"


class Hmmt2026DatasetReader(FileDatasetReader):
    default_question_field = "problem"
    default_answer_field = "answer"
    default_answer_instruction = "a simplified mathematical expression"


class GpqaDiamondDatasetReader(FileDatasetReader):
    default_question_field = "question"
    default_answer_field = "answer"
    default_answer_instruction = "A, B, C, or D"


DATASET_READERS: dict[str, type[BaseDatasetReader]] = {
    "aime24": Aime24DatasetReader,
    "aime25": Aime25DatasetReader,
    "aime26": Aime26DatasetReader,
    "amc23": Amc23DatasetReader,
    "hmmt2026": Hmmt2026DatasetReader,
    "gpqa_diamond": GpqaDiamondDatasetReader,
}


def build_dataset_reader(
    dataset_name: str,
    dataset_type: str,
    path: str | Path,
    question_field: str | None,
    answer_field: str | None,
) -> BaseDatasetReader:
    if dataset_type not in ("auto", "file", "aime", "math", "generic"):
        raise ValueError(f"unsupported dataset_type: {dataset_type}")
    # dataset_name selects a subclass with field defaults. Unknown datasets
    # still work through FileDatasetReader's fallback field detection.
    reader_cls = DATASET_READERS.get(_normalize_dataset_name(dataset_name), FileDatasetReader)
    return reader_cls(path, question_field, answer_field)


def _normalize_dataset_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _get_field(
    record: dict[str, Any],
    explicit_field: str | None,
    fallback_fields: tuple[str, ...],
) -> Any:
    if explicit_field:
        return record.get(explicit_field)
    for field in fallback_fields:
        if field in record and record[field] not in (None, ""):
            return record[field]
    return None


def _resolve_dataset_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"dataset path does not exist: {path}")

    patterns = (
        "test-*.parquet",
        "test*.parquet",
        "data/test-*.parquet",
        "data/test*.parquet",
        "data/train-*.parquet",
        "data/train*.parquet",
        "*.parquet",
        "data/*.parquet",
        "*.jsonl",
        "data/*.jsonl",
        "*.json",
        "data/*.json",
        "*.csv",
        "data/*.csv",
    )
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise ValueError(f"dataset directory contains no supported data file: {path}")


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return _ensure_records(data)
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "records"):
            if isinstance(data.get(key), list):
                return _ensure_records(data[key])
    raise ValueError("JSON dataset must be a list or contain data/examples/items/records")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet datasets require pyarrow or pandas.") from exc
    dataframe = pd.read_parquet(path)
    return dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")


def _ensure_records(items: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} is not a JSON object")
        records.append(item)
    return records


def _filter_records_by_id(records: list[dict[str, Any]], example_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {str(example_id) for example_id in example_ids}
    matched = [record for record in records if str(record.get("id", "")) in wanted]
    found = {str(record.get("id", "")) for record in matched}
    missing = [example_id for example_id in example_ids if str(example_id) not in found]
    if missing:
        raise ValueError(f"example_ids not found in dataset: {','.join(missing)}")
    return matched
