from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .prompts import PROMPTS, build_user_prompt


@dataclass
class RunMetrics:
    prompt_name: str
    input_target_tokens: int
    run_index: int
    ttft_seconds: Optional[float]
    ttfrt_seconds: Optional[float]
    total_latency_seconds: float
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    output_tokens_per_second: Optional[float]
    output_tokens_per_second_e2e: Optional[float]
    error: Optional[str] = None


def _is_context_limit_error(message: str) -> bool:
    if not message:
        return False
    text = message.lower()
    patterns = [
        "maximum context length",
        "context length",
        "context window",
        "too many tokens",
        "prompt is too long",
        "exceeds",
        "requested",
        "tokens",
    ]
    if "token" not in text and "context" not in text:
        return False
    return any(p in text for p in patterns)


def _to_dict(chunk: Any) -> Dict[str, Any]:
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if isinstance(chunk, dict):
        return chunk
    try:
        return dict(chunk)
    except Exception:
        return {}


def _extract_choice_delta(chunk_dict: Dict[str, Any]) -> Dict[str, Any]:
    choices = chunk_dict.get("choices") or []
    if not choices:
        return {}
    delta = choices[0].get("delta") or {}
    return delta if isinstance(delta, dict) else {}


def _extract_reasoning_text(delta: Dict[str, Any]) -> str:
    candidates = [
        delta.get("reasoning"),
        delta.get("reasoning_content"),
        delta.get("reasoning_text"),
        delta.get("thoughts"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            joined = "".join(str(x) for x in value if x is not None).strip()
            if joined:
                return joined
    return ""


def _extract_visible_text(delta: Dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for entry in content:
            if isinstance(entry, dict):
                part = entry.get("text")
                if isinstance(part, str):
                    parts.append(part)
        return "".join(parts)
    return ""


def _safe_usage_value(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
        return value if isinstance(value, int) else None
    value = getattr(usage, key, None)
    return value if isinstance(value, int) else None


def _run_single(
    client: OpenAI,
    model: str,
    user_prompt: str,
    prompt_name: str,
    input_target_tokens: int,
    run_index: int,
    max_output_tokens: int,
    temperature: float,
) -> RunMetrics:
    start = time.perf_counter()
    first_visible_token_at: Optional[float] = None
    first_reasoning_token_at: Optional[float] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            now = time.perf_counter()
            chunk_dict = _to_dict(chunk)
            delta = _extract_choice_delta(chunk_dict)

            if first_visible_token_at is None:
                visible = _extract_visible_text(delta)
                if visible:
                    first_visible_token_at = now

            if first_reasoning_token_at is None:
                reasoning = _extract_reasoning_text(delta)
                if reasoning:
                    first_reasoning_token_at = now

            usage = chunk_dict.get("usage")
            if usage:
                completion_tokens = _safe_usage_value(usage, "completion_tokens")
                if completion_tokens is None and isinstance(usage, dict):
                    completion_tokens = _safe_usage_value(
                        usage.get("output_tokens"), "value"
                    )
                reasoning_tokens = _safe_usage_value(usage, "reasoning_tokens")
                if reasoning_tokens is None and isinstance(usage, dict):
                    details = usage.get("completion_tokens_details") or {}
                    if isinstance(details, dict):
                        reasoning_tokens = _safe_usage_value(details, "reasoning_tokens")

        end = time.perf_counter()

        ttft = (first_visible_token_at - start) if first_visible_token_at else None
        ttfrt = (first_reasoning_token_at - start) if first_reasoning_token_at else None
        total_latency = end - start

        generation_window = None
        if first_visible_token_at:
            generation_window = max(end - first_visible_token_at, 1e-9)

        otps = None
        if completion_tokens is not None and generation_window is not None:
            otps = completion_tokens / generation_window

        otps_e2e = None
        if completion_tokens is not None and total_latency > 0:
            otps_e2e = completion_tokens / total_latency

        return RunMetrics(
            prompt_name=prompt_name,
            input_target_tokens=input_target_tokens,
            run_index=run_index,
            ttft_seconds=ttft,
            ttfrt_seconds=ttfrt,
            total_latency_seconds=total_latency,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens_per_second=otps,
            output_tokens_per_second_e2e=otps_e2e,
        )
    except Exception as exc:
        end = time.perf_counter()
        return RunMetrics(
            prompt_name=prompt_name,
            input_target_tokens=input_target_tokens,
            run_index=run_index,
            ttft_seconds=None,
            ttfrt_seconds=None,
            total_latency_seconds=end - start,
            completion_tokens=None,
            reasoning_tokens=None,
            output_tokens_per_second=None,
            output_tokens_per_second_e2e=None,
            error=str(exc),
        )


def _probe_input_size_support(
    client: OpenAI,
    model: str,
    input_tokens: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    probe_prompt = build_user_prompt(
        spec=PROMPTS[0], input_tokens=input_tokens, model=model
    )
    try:
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": probe_prompt},
            ],
            max_tokens=min(max_output_tokens, 8),
            temperature=0.0,
            stream=False,
        )
        return {
            "input_target_tokens": input_tokens,
            "supported": True,
            "reason": None,
        }
    except Exception as exc:
        error_text = str(exc)
        return {
            "input_target_tokens": input_tokens,
            "supported": not _is_context_limit_error(error_text),
            "reason": error_text,
        }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [v for v in values if isinstance(v, (int, float))]
    if not present:
        return None
    return statistics.mean(present)


def _p95(values: Iterable[Optional[float]]) -> Optional[float]:
    present = sorted(v for v in values if isinstance(v, (int, float)))
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    idx = int(round(0.95 * (len(present) - 1)))
    return present[idx]


def aggregate(results: List[RunMetrics]) -> Dict[str, Any]:
    groups: Dict[str, List[RunMetrics]] = {}
    input_groups: Dict[int, List[RunMetrics]] = {}
    for row in results:
        key = f"{row.prompt_name}|{row.input_target_tokens}"
        groups.setdefault(key, []).append(row)
        input_groups.setdefault(row.input_target_tokens, []).append(row)

    by_group = []
    for key, rows in groups.items():
        prompt_name, input_tokens_raw = key.split("|")
        input_tokens = int(input_tokens_raw)
        by_group.append(
            {
                "prompt_name": prompt_name,
                "input_target_tokens": input_tokens,
                "runs": len(rows),
                "error_count": len([r for r in rows if r.error]),
                "ttft_mean_s": _mean(r.ttft_seconds for r in rows),
                "ttft_p95_s": _p95(r.ttft_seconds for r in rows),
                "ttfrt_mean_s": _mean(r.ttfrt_seconds for r in rows),
                "total_latency_mean_s": _mean(r.total_latency_seconds for r in rows),
                "total_latency_p95_s": _p95(r.total_latency_seconds for r in rows),
                "otps_mean": _mean(r.output_tokens_per_second for r in rows),
                "otps_e2e_mean": _mean(r.output_tokens_per_second_e2e for r in rows),
                "completion_tokens_mean": _mean(r.completion_tokens for r in rows),
                "reasoning_tokens_mean": _mean(r.reasoning_tokens for r in rows),
            }
        )

    overall = {
        "ttft_mean_s": _mean(r.ttft_seconds for r in results),
        "ttfrt_mean_s": _mean(r.ttfrt_seconds for r in results),
        "total_latency_mean_s": _mean(r.total_latency_seconds for r in results),
        "otps_mean": _mean(r.output_tokens_per_second for r in results),
        "otps_e2e_mean": _mean(r.output_tokens_per_second_e2e for r in results),
    }

    by_input_size = []
    for input_tokens in sorted(input_groups.keys()):
        rows = input_groups[input_tokens]
        by_input_size.append(
            {
                "input_target_tokens": input_tokens,
                "runs": len(rows),
                "error_count": len([r for r in rows if r.error]),
                "ttft_mean_s": _mean(r.ttft_seconds for r in rows),
                "ttfrt_mean_s": _mean(r.ttfrt_seconds for r in rows),
                "total_latency_mean_s": _mean(r.total_latency_seconds for r in rows),
                "otps_mean": _mean(r.output_tokens_per_second for r in rows),
                "otps_e2e_mean": _mean(r.output_tokens_per_second_e2e for r in rows),
            }
        )
    return {"overall": overall, "by_input_size": by_input_size, "by_group": by_group}


def write_markdown_report(
    report_path: Path,
    model: str,
    input_sizes: List[int],
    runs_per_case: int,
    max_output_tokens: int,
    summary: Dict[str, Any],
    input_size_support: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# LLM Performance Evaluation")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Input token targets: `{input_sizes}`")
    lines.append(f"- Prompt variants: `{len(PROMPTS)}`")
    lines.append(f"- Runs per case: `{runs_per_case}`")
    lines.append(f"- Max output tokens: `{max_output_tokens}`")
    lines.append("")
    lines.append("## Overall")
    ov = summary["overall"]
    lines.append(f"- Mean TTFT (s): `{ov['ttft_mean_s']}`")
    lines.append(f"- Mean TTFR (s): `{ov['ttfrt_mean_s']}`")
    lines.append(f"- Mean Total Latency (s): `{ov['total_latency_mean_s']}`")
    lines.append(f"- Mean OTPS: `{ov['otps_mean']}`")
    lines.append(f"- Mean OTPS (E2E): `{ov['otps_e2e_mean']}`")
    lines.append("")
    lines.append("## By Input Length")
    for row in summary.get("by_input_size", []):
        lines.append(
            f"- `{row['input_target_tokens']}` tokens: "
            f"TTFT mean `{row['ttft_mean_s']}`, "
            f"latency mean `{row['total_latency_mean_s']}`, "
            f"OTPS mean `{row['otps_mean']}`, "
            f"errors `{row['error_count']}/{row['runs']}`"
        )
    lines.append("")
    unsupported = [row for row in input_size_support if not row.get("supported", True)]
    lines.append("## Input Size Support")
    if unsupported:
        for row in unsupported:
            lines.append(
                f"- `{row['input_target_tokens']}` tokens: not supported by this model/endpoint. "
                f"Reason: `{row.get('reason')}`"
            )
    else:
        lines.append("- All configured input sizes were accepted by the model/endpoint.")
    lines.append("")
    lines.append("## By Prompt/Input")
    for row in summary["by_group"]:
        lines.append(
            f"- `{row['prompt_name']}` @ `{row['input_target_tokens']}` tokens: "
            f"TTFT mean `{row['ttft_mean_s']}`, TTFR mean `{row['ttfrt_mean_s']}`, "
            f"latency mean `{row['total_latency_mean_s']}`, OTPS mean `{row['otps_mean']}`, "
            f"errors `{row['error_count']}/{row['runs']}`"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    input_sizes: List[int],
    runs_per_case: int,
    max_output_tokens: int,
    temperature: float,
    out_dir: str,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=base_url, api_key=api_key)

    results: List[RunMetrics] = []
    input_size_support: List[Dict[str, Any]] = []
    for input_tokens in input_sizes:
        support = _probe_input_size_support(
            client=client,
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
        )
        input_size_support.append(support)
        if not support["supported"]:
            continue
        stop_input_size = False
        for spec in PROMPTS:
            if stop_input_size:
                break
            user_prompt = build_user_prompt(
                spec=spec, input_tokens=input_tokens, model=model
            )
            for run_index in range(1, runs_per_case + 1):
                row = _run_single(
                    client=client,
                    model=model,
                    user_prompt=user_prompt,
                    prompt_name=spec.name,
                    input_target_tokens=input_tokens,
                    run_index=run_index,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                if row.error and _is_context_limit_error(row.error):
                    input_size_support = [
                        {
                            **s,
                            "supported": False
                            if s["input_target_tokens"] == input_tokens
                            else s["supported"],
                            "reason": row.error
                            if s["input_target_tokens"] == input_tokens
                            else s["reason"],
                        }
                        for s in input_size_support
                    ]
                    stop_input_size = True
                results.append(row)
                if stop_input_size:
                    break

    summary = aggregate(results)
    payload = {
        "config": {
            "base_url": base_url,
            "model": model,
            "input_sizes": input_sizes,
            "runs_per_case": runs_per_case,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        },
        "input_size_support": input_size_support,
        "summary": summary,
        "runs": [asdict(r) for r in results],
    }

    json_path = out / "results.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(
        report_path=md_path,
        model=model,
        input_sizes=input_sizes,
        runs_per_case=runs_per_case,
        max_output_tokens=max_output_tokens,
        summary=summary,
        input_size_support=input_size_support,
    )
    return payload
