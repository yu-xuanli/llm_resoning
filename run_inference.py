#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from src.config import load_config, resolve_project_path
from src.datasets import build_dataset_reader
from src.inference import VLLMGenerator
from src.prompting import PromptBuilder
from src.recorder import make_question_dir, make_run_dir, write_branch_statistics
from src.statistics import (
    BranchStatistics,
    extract_final_answer,
    extract_method_spans,
    is_correct_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE: TextIO | None = None
PENDING_LOG_LINES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM inference for prompt templates.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset, and prompt rendering without loading vLLM.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, file=sys.stderr, flush=True)
    if LOG_FILE is not None:
        print(line, file=LOG_FILE, flush=True)
    else:
        PENDING_LOG_LINES.append(line)


def open_log_file(run_dir: Path) -> Path:
    global LOG_FILE
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    LOG_FILE = log_path.open("a", encoding="utf-8")
    for line in PENDING_LOG_LINES:
        print(line, file=LOG_FILE, flush=True)
    PENDING_LOG_LINES.clear()
    return log_path


def close_log_file() -> None:
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()
        LOG_FILE = None


def main() -> int:
    args = parse_args()
    log("stage=config_load start")
    config = load_config(args.config, PROJECT_ROOT)
    config.validate(require_model=not args.dry_run)
    log(
        "stage=config_load done "
        f"model_name={config.model_name} dataset_name={config.dataset_name} "
        f"branch_num={config.branch_num}"
    )

    log("stage=path_resolve start")
    dataset_path = resolve_project_path(PROJECT_ROOT, config.dataset_path)
    template_path = resolve_project_path(PROJECT_ROOT, config.prompt_template_path)
    log(f"stage=path_resolve done dataset_path={dataset_path} template_path={template_path}")

    log("stage=dataset_load start")
    reader = build_dataset_reader(
        config.dataset_name,
        config.dataset_type,
        dataset_path,
        config.question_field,
        config.answer_field,
    )
    examples = reader.read_examples(config.num_examples, config.example_ids)
    log(
        "stage=dataset_load done "
        f"examples={len(examples)} num_examples={config.num_examples} "
        f"example_ids={','.join(config.example_ids) if config.example_ids else 'None'}"
    )

    log("stage=template_load start")
    prompt_builder = PromptBuilder.from_file(template_path)
    log("stage=template_load done")

    if args.dry_run:
        if not examples:
            raise ValueError("dataset contains no examples")
        log("stage=prompt_render start example_index=1")
        prompt = prompt_builder.build(examples[0].question, examples[0].answer_instruction)
        log(f"stage=prompt_render done prompt_chars={len(prompt)}")
        print(prompt)
        return 0

    run_dir = make_run_dir(config, PROJECT_ROOT)
    log_path = open_log_file(run_dir)
    log(f"stage=run_dir done run_dir={run_dir} log_path={log_path}")

    log("stage=model_load start")
    generator = VLLMGenerator(config)
    log("stage=model_load done")

    for example in examples:
        log(f"stage=prompt_render start example_index={example.index} example_id={example.example_id}")
        prompt = prompt_builder.build(example.question, example.answer_instruction)
        response_prefix = prompt_builder.response_prefix(prompt)
        log(f"stage=prompt_render done example_index={example.index} prompt_chars={len(prompt)}")
        question_dir = make_question_dir(run_dir, example)
        log(f"stage=generation start example_index={example.index} branches={config.branch_num}")
        for generation in generator.generate(prompt):
            log(
                "stage=generation_branch done "
                f"example_index={example.index} branch={generation.branch_index} "
                f"tokens={generation.total_tokens} truncated={generation.truncated}"
            )
            reasoning = response_prefix + generation.text.lstrip()
            extracted_answer = extract_final_answer(reasoning)
            method_spans = extract_method_spans(reasoning, generation.tokens)
            stats = BranchStatistics(
                example_id=example.example_id,
                question=example.question,
                prompt=prompt,
                reasoning=reasoning,
                extracted_answer=extracted_answer,
                correct=is_correct_prediction(extracted_answer, example.answer),
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                truncated=generation.truncated,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                model_path=config.model_path,
                model_name=config.model_name,
                branch_index=generation.branch_index,
                seed=config.seed,
                raw_finish_reason=generation.finish_reason,
                tokens=generation.tokens,
                top_logprobs=generation.top_logprobs,
                extra={
                    "dataset_name": config.dataset_name,
                    "dataset_path": str(dataset_path),
                    "answer": example.answer,
                    "raw_record": example.raw_record,
                    "method_spans": method_spans,
                },
            )
            log(
                "stage=statistics done "
                f"example_index={example.index} branch={generation.branch_index} "
                f"answer={stats.extracted_answer or 'N/A'}"
            )
            path = write_branch_statistics(question_dir, stats)
            log(f"stage=record_write done path={path}")
        log(f"stage=example_done example_index={example.index} question_dir={question_dir}")
    close_log_file()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        log(f"error: {exc}")
        close_log_file()
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        log(f"error: {type(exc).__name__}: {exc}")
        print(f"error: {exc}", file=sys.stderr)
        close_log_file()
        raise SystemExit(1)
