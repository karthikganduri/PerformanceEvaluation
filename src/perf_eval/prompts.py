from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass(frozen=True)
class PromptSpec:
    name: str
    instruction: str


PROMPTS: List[PromptSpec] = [
    PromptSpec(
        name="factual_qa",
        instruction=(
            "Read the context and answer the question with concise, evidence-backed points. "
            "Call out uncertainty when evidence is weak."
        ),
    ),
    PromptSpec(
        name="reasoning_planning",
        instruction=(
            "Read the context and produce a step-by-step plan with tradeoffs, risks, and "
            "clear assumptions."
        ),
    ),
    PromptSpec(
        name="structured_extraction",
        instruction=(
            "Read the context and return a JSON object with key entities, dates, and a short "
            "summary. Keep output valid JSON."
        ),
    ),
]


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def make_context_block(target_tokens: int, model: str) -> str:
    enc = _encoding_for_model(model)
    seed = (
        "System performance logs indicate seasonal load spikes, mixed request distributions, "
        "queue contention, and varying response quality across cohorts. "
        "Teams need comparative analysis, latency root-cause hints, and mitigation options. "
    )
    seed_tokens = enc.encode(seed)
    repeat_count = max(1, target_tokens // max(1, len(seed_tokens)) + 3)
    text = (" ".join([seed] * repeat_count)).strip()
    token_ids = enc.encode(text)
    if len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
        text = enc.decode(token_ids)
    return text


def build_user_prompt(spec: PromptSpec, input_tokens: int, model: str) -> str:
    context = make_context_block(input_tokens, model)
    return (
        f"Task type: {spec.name}\n\n"
        f"Instruction:\n{spec.instruction}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Now produce the best possible answer."
    )
