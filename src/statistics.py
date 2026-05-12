from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TokenProbability:
    token_id: int | None
    text: str
    logprob: float | None
    prob: float | None


@dataclass
class BranchStatistics:
    example_id: str
    question: str
    prompt: str
    reasoning: str
    extracted_answer: str
    correct: bool | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    truncated: bool
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    model_path: str
    model_name: str
    branch_index: int
    seed: int
    # Raw vLLM finish reason: "stop" means normal stop, "length" means the
    # generation hit max_new_tokens, and other values are passed through.
    raw_finish_reason: str
    tokens: list[TokenProbability] = field(default_factory=list)
    top_logprobs: list[list[TokenProbability]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokens"] = [asdict(item) for item in self.tokens]
        data["top_logprobs"] = [
            [asdict(item) for item in step]
            for step in self.top_logprobs
        ]
        return data


def extract_final_answer(reasoning: str) -> str:
    for tag in ("answer", "final_answer"):
        match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", reasoning, flags=re.DOTALL | re.IGNORECASE)
        if match and not _is_placeholder_answer(match.group(1)):
            return normalize_answer(match.group(1))

    final_matches = list(re.finditer(r"final answer\s*:\s*(.+)", reasoning, flags=re.IGNORECASE))
    if final_matches:
        return normalize_answer(final_matches[-1].group(1))

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", reasoning)
    if boxed_matches:
        return normalize_answer(boxed_matches[-1])

    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    return normalize_answer(lines[-1]) if lines else ""


def is_correct_prediction(extracted_answer: str, gold_answer: str | None) -> bool | None:
    """Compare extracted model answer with dataset answer after light normalization."""
    if gold_answer is None:
        return None
    prediction = normalize_answer(extracted_answer)
    gold = normalize_answer(gold_answer)
    if not prediction or not gold:
        return False
    return prediction == gold


def extract_method_spans(reasoning: str, tokens: list[TokenProbability]) -> list[dict[str, Any]]:
    """Find explicit Method lines and map them to best-effort generated token spans.

    The model is asked to write short lines beginning with "Method:" before each
    reasoning move. The character offsets are exact in the saved reasoning text.
    Token spans are best effort because vLLM logprob entries may not always expose
    the exact decoded text for every sampled token.
    """
    search_area = reasoning.split("<method_trace>", 1)[0]
    token_offsets = _token_char_offsets(tokens)
    spans: list[dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"(?im)^method\s*:\s*(.+?)\s*$", search_area), start=1):
        token_start, token_end = _char_span_to_token_span(match.start(), match.end(), token_offsets)
        spans.append(
            {
                "id": index,
                "text": match.group(0).strip(),
                "statement": match.group(1).strip(),
                "start_char": match.start(),
                "end_char": match.end(),
                "start_token": token_start,
                "end_token": token_end,
                "source": "detected_method_line",
            }
        )
    return spans


def normalize_answer(value: str | int | float | None) -> str:
    r"""Normalize common math answer wrappers such as \boxed{204} before compare."""
    if value is None:
        return ""
    text = _clean_answer(str(value))
    text = re.sub(r"^final answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = _strip_wrapping_answer_tags(text)
    text = _strip_boxed(text)
    text = _strip_math_delimiters(text)
    text = _clean_answer(text)
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _is_placeholder_answer(value: str) -> bool:
    normalized = normalize_answer(value).lower()
    return normalized in {
        "final answer here",
        "your final answer",
        "the final answer",
        "final numeric answer",
    }


def probability_from_logprob(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    try:
        return math.exp(float(logprob))
    except OverflowError:
        return None


def _clean_answer(value: str) -> str:
    value = value.strip()
    value = value.strip("`")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(" .")


def _strip_wrapping_answer_tags(text: str) -> str:
    for tag in ("answer", "final_answer"):
        match = re.fullmatch(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
    return text


def _strip_boxed(text: str) -> str:
    changed = True
    while changed:
        changed = False
        match = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
            changed = True
    return text


def _strip_math_delimiters(text: str) -> str:
    text = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right):
            return text[len(left) : len(text) - len(right)].strip()
    return text


def _token_char_offsets(tokens: list[TokenProbability]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        text = token.text or ""
        start = cursor
        cursor += len(text)
        offsets.append((start, cursor))
    return offsets


def _char_span_to_token_span(
    start_char: int,
    end_char: int,
    token_offsets: list[tuple[int, int]],
) -> tuple[int | None, int | None]:
    if not token_offsets:
        return None, None

    start_token: int | None = None
    end_token: int | None = None
    for index, (token_start, token_end) in enumerate(token_offsets):
        if start_token is None and token_end > start_char:
            start_token = index
        if token_start < end_char:
            end_token = index + 1
        elif end_token is not None:
            break
    return start_token, end_token
