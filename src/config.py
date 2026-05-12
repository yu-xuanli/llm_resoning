from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime.
    yaml = None


@dataclass(frozen=True)
class InferenceConfig:
    method: str
    model_path: str
    model_name: str
    dataset_name: str
    dataset_path: str
    dataset_type: str
    question_field: str | None
    answer_field: str | None
    prompt_template_path: str
    output_root: str
    branch_num: int
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    seed: int
    num_examples: int | None = None
    example_ids: list[str] | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any], project_root: Path) -> "InferenceConfig":
        method = str(data.get("method") or project_root.name)
        return cls(
            method=method,
            model_path=str(data.get("model_path") or ""),
            model_name=str(data.get("model_name") or "model"),
            dataset_name=str(data.get("dataset_name") or "dataset"),
            dataset_path=str(data.get("dataset_path") or ""),
            dataset_type=str(data.get("dataset_type") or "auto"),
            question_field=_optional_str(data.get("question_field")),
            answer_field=_optional_str(data.get("answer_field")),
            prompt_template_path=str(data.get("prompt_template_path") or "templates/default_prompt.txt"),
            output_root=str(data.get("output_root") or "outputs"),
            branch_num=int(data.get("branch_num") or 1),
            temperature=float(data.get("temperature") if data.get("temperature") is not None else 0.7),
            top_p=float(data.get("top_p") if data.get("top_p") is not None else 0.95),
            top_k=int(data.get("top_k") if data.get("top_k") is not None else -1),
            max_new_tokens=int(data.get("max_new_tokens") or 2048),
            seed=int(data.get("seed") if data.get("seed") is not None else 42),
            num_examples=_optional_int(data.get("num_examples")),
            example_ids=_optional_str_list(data.get("example_ids")),
            tensor_parallel_size=int(data.get("tensor_parallel_size") or 1),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization") or 0.9),
            max_model_len=_optional_int(data.get("max_model_len")),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
        )

    def validate(self, require_model: bool = True) -> None:
        if require_model and not self.model_path:
            raise ValueError("请在 config 中填写本地模型路径: model_path")
        if not self.dataset_path:
            raise ValueError("请在 config 中填写数据集路径: dataset_path")
        if self.branch_num < 1:
            raise ValueError("branch_num must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.num_examples is not None and self.num_examples < 1:
            raise ValueError("num_examples must be >= 1")
        if self.example_ids is not None and not self.example_ids:
            raise ValueError("example_ids must not be empty")


def load_config(path: str | Path, project_root: Path | None = None) -> InferenceConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files. Install pyyaml first.")
    config_path = Path(path)
    if project_root is None:
        project_root = config_path.resolve().parents[1]
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping")
    return InferenceConfig.from_mapping(raw, project_root)


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item]
