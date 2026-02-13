from __future__ import annotations

import argparse
import json
import os
from typing import List

from .benchmark import run_benchmark


def _parse_input_sizes(raw: str) -> List[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("input sizes cannot be empty")
    return vals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OpenAI-compatible model endpoints for TTFT, TTFR, OTPS, and latency."
        )
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-sizes", default="100,1000,10000")
    parser.add_argument("--runs-per-case", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", default="results")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.base_url:
        parser.error("Missing --base-url or OPENAI_BASE_URL")
    if not args.api_key:
        parser.error("Missing --api-key or OPENAI_API_KEY")

    input_sizes = _parse_input_sizes(args.input_sizes)
    payload = run_benchmark(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        input_sizes=input_sizes,
        runs_per_case=args.runs_per_case,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        out_dir=args.out_dir,
    )

    print(json.dumps(payload["summary"], indent=2))
    print(f"\nSaved full run data to {args.out_dir}/results.json")
    print(f"Saved markdown report to {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
