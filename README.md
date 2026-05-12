# llm_resoning

Config-driven local vLLM inference for prompt-template experiments.

This project is a scaffold for running one prompt template against a dataset,
sampling multiple reasoning branches, and writing one JSON result per branch.
All semantic template tags are lowercase, such as `<question>`, `<plan>`,
`<solution>`, and `<answer>`.

## Files

| Path | Purpose |
| --- | --- |
| `configs/default.yaml` | Default run config. Fill `model_path` and `dataset_path` before real inference. |
| `templates/default_prompt.txt` | Default lowercase-tag prompt template. |
| `run_inference.py` | Main entry point. |
| `src/config.py` | YAML config loading and validation. |
| `src/datasets.py` | Dataset reader abstraction and generic file readers. |
| `src/prompting.py` | Reusable question replacement logic. |
| `src/inference.py` | Local vLLM generation wrapper. |
| `src/statistics.py` | Branch-level answer extraction and JSON statistics structures. |
| `src/recorder.py` | Output directory and JSON writing helpers. |

## Config

Edit `configs/default.yaml`:

```yaml
model_path: ""  # TODO: set local model path
model_name: qwen3_4b
dataset_name: aime24
dataset_path: ""  # TODO: set dataset path
question_field:  # Optional override. Leave empty to use dataset_name defaults.
answer_field:  # Optional override. Leave empty to use dataset_name defaults.
num_examples:  # Optional: how many problems to run this time.
example_ids:  # Optional: comma-separated dataset ids, for example 60,72.
prompt_template_path: templates/default_prompt.txt
output_root: outputs
branch_num: 2
temperature: 0.7
top_p: 0.9
top_k: -1
max_new_tokens: 2048
```

Supported dataset files: `.json`, `.jsonl`, `.csv`, `.parquet`.

Dataset-specific field defaults live on reader subclasses in `src/datasets.py`.
For example, `Aime24DatasetReader` uses `problem` as the question field and
`solution` as the answer field. Config values for `question_field` and
`answer_field` override those class defaults when they are set.

To run fixed dataset ids instead of the first `num_examples` problems, set
`example_ids` as a comma-separated value:

```yaml
example_ids: 60,72
num_examples:
```

If both `example_ids` and `num_examples` are set, the runner first filters by
`example_ids`, then keeps the first `num_examples` matched examples.

## Prompt Templates

The default template uses brace-style placeholders. Example:

```text
{question}
```

`PromptBuilder` performs `{question}` and `{answer_instruction}` substitutions at runtime. The dataset reader extracts the question and provides a dataset-specific `answer_instruction` string which will be injected into `{answer_instruction}` in the template. Prompt replacement stays centralized in `src/prompting.py`.

## Run

`CUDA_VISIBLE_DEVICES=7`

Validate config, dataset loading, and prompt rendering without loading vLLM:

```bash
python run_inference.py --config configs/default.yaml --dry-run
```

Run real inference after setting `model_path`:

```bash
python run_inference.py --config configs/default.yaml
```

For a smoke test, set `num_examples` in YAML:

```bash
python run_inference.py --config configs/default.yaml
```

To run selected problems, set `example_ids` in YAML, for example:

```yaml
example_ids: 60,72
```

## Output

Outputs are written to:

```text
{output_root}/{method}/{model_name}_{dataset_name}@{branch_num}/{timestamp}/question_{index}_{id}/branch_{branch}.json
```

Each branch JSON includes the final prompt, reasoning text, extracted answer,
correctness against the dataset answer, token counts, truncation flag, sampling
config, generated token probabilities, and top-5 token logprobs/probabilities
when vLLM provides them.

`raw_finish_reason` is the finish reason returned by vLLM for that branch.
Common values are `stop`, meaning generation ended normally, and `length`,
meaning it hit the configured `max_new_tokens` limit.

## TODO

- Add dataset-specific reader subclasses when field-based extraction is not enough.
- Add dataset-specific answer extraction and normalization modes.
- Add optional summary files once branch-level JSON schema stabilizes.
