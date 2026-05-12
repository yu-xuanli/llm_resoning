from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptBuilder:
    template: str

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptBuilder":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(f.read())

    def build(self, question: str, answer_instruction: str | None = None) -> str:
        """Perform brace-style substitutions for `{question}` and `{answer_instruction}`.

        Replacements remove the placeholder tokens entirely and insert the
        provided content. Raise on missing `{question}` placeholder so templates
        must opt-in to the brace-style syntax.
        """
        question = question.strip()
        tgt = self.template
        if "{question}" not in tgt:
            raise ValueError("prompt template must contain {question} placeholder")
        # Replace only the first occurrence so templates can include other
        # examples or commentary.
        tgt = tgt.replace("{question}", question, 1)
        if "{answer_instruction}" in tgt:
            replacement = "" if answer_instruction is None else str(answer_instruction)
            tgt = tgt.replace("{answer_instruction}", replacement, 1)
        return tgt

    @staticmethod
    def response_prefix(prompt: str) -> str:
        """Return prefilled assistant text after the final chat assistant marker."""
        marker = "<|im_start|>assistant\n"
        if marker not in prompt:
            return ""
        prefix = prompt.rsplit(marker, 1)[1]
        return prefix if prefix.strip() else ""
