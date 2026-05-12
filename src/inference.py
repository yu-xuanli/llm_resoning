from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import InferenceConfig
from .statistics import TokenProbability, probability_from_logprob


@dataclass
class GenerationResult:
    branch_index: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    truncated: bool
    finish_reason: str
    tokens: list[TokenProbability]
    top_logprobs: list[list[TokenProbability]]


class VLLMGenerator:
    def __init__(self, config: InferenceConfig) -> None:
        _patch_transformers_tokenizer_compat()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is required for inference. Install vllm first.") from exc

        llm_kwargs: dict[str, Any] = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.max_model_len

        self.config = config
        self._sampling_params_cls = SamplingParams
        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt: str) -> list[GenerationResult]:
        sampling_kwargs: dict[str, Any] = {
            "n": self.config.branch_num,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
            "logprobs": 5,
            "seed": self.config.seed,
        }
        if self.config.top_k > 0:
            sampling_kwargs["top_k"] = self.config.top_k

        outputs = self.llm.generate([prompt], self._sampling_params_cls(**sampling_kwargs))
        if not outputs:
            return []

        request_output = outputs[0]
        prompt_token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
        results: list[GenerationResult] = []
        for branch_index, output in enumerate(getattr(request_output, "outputs", []), start=1):
            token_ids = list(getattr(output, "token_ids", []) or [])
            text = str(getattr(output, "text", "") or "").strip()
            finish_reason = str(getattr(output, "finish_reason", "") or "")
            token_logprobs, top_logprobs = _collect_logprobs(output, token_ids)
            results.append(
                GenerationResult(
                    branch_index=branch_index,
                    text=text,
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=len(token_ids),
                    total_tokens=len(prompt_token_ids) + len(token_ids),
                    truncated=finish_reason == "length" or len(token_ids) >= self.config.max_new_tokens,
                    finish_reason=finish_reason,
                    tokens=token_logprobs,
                    top_logprobs=top_logprobs,
                )
            )
        return results


def _collect_logprobs(
    output: Any,
    token_ids: list[int],
) -> tuple[list[TokenProbability], list[list[TokenProbability]]]:
    raw_logprobs = list(getattr(output, "logprobs", []) or [])
    tokens: list[TokenProbability] = []
    top_steps: list[list[TokenProbability]] = []

    for index, token_id in enumerate(token_ids):
        step = raw_logprobs[index] if index < len(raw_logprobs) else None
        step_items = _step_top_items(step)
        token_item = _find_token_item(step_items, token_id)
        tokens.append(token_item)
        top_steps.append(step_items[:5])
    return tokens, top_steps


def _step_top_items(step: Any) -> list[TokenProbability]:
    if not step:
        return []
    if isinstance(step, dict):
        iterable = step.items()
    else:
        iterable = []

    items: list[TokenProbability] = []
    for raw_token_id, value in iterable:
        token_id = _safe_int(raw_token_id)
        logprob = _extract_attr(value, "logprob")
        text = _extract_attr(value, "decoded_token")
        if text is None:
            text = _extract_attr(value, "token")
        if text is None:
            text = ""
        items.append(
            TokenProbability(
                token_id=token_id,
                text=str(text),
                logprob=None if logprob is None else float(logprob),
                prob=probability_from_logprob(None if logprob is None else float(logprob)),
            )
        )
    items.sort(key=lambda item: float("-inf") if item.logprob is None else item.logprob, reverse=True)
    return items


def _find_token_item(items: list[TokenProbability], token_id: int) -> TokenProbability:
    for item in items:
        if item.token_id == token_id:
            return item
    return TokenProbability(token_id=token_id, text="", logprob=None, prob=None)


def _extract_attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _patch_transformers_tokenizer_compat() -> None:
    """Patch older tokenizer classes for vLLM versions expecting newer Transformers."""
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    def all_special_tokens_extended(self: Any) -> list[Any]:
        return list(getattr(self, "all_special_tokens", []) or [])

    PreTrainedTokenizerBase.all_special_tokens_extended = property(all_special_tokens_extended)  # type: ignore[attr-defined]
