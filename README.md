# LLM Performance Evaluation

Benchmark any OpenAI-compatible endpoint for:
- TTFT (time to first visible output token)
- TTFR (time to first reasoning token, when available)
- OTPS (output tokens per second)
- Total latency

The tool runs **3 distinct prompt types** for each configured input size (default: **100**, **1k**, and **10k** tokens), then writes detailed per-run metrics and aggregated analysis.

## What It Measures

- `ttft_seconds`: time from request start until first assistant output token
- `ttfrt_seconds`: time from request start until first reasoning token (if the provider streams reasoning)
- `total_latency_seconds`: end-to-end request duration
- `output_tokens_per_second`: `completion_tokens / (end_time - first_output_token_time)`
- `output_tokens_per_second_e2e`: `completion_tokens / total_latency`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run benchmark:

```bash
llm-perf-eval \
  --base-url "https://your-openai-compatible-endpoint/v1" \
  --api-key "your_api_key" \
  --model "your_model_name"
```

Defaults:
- Input sizes: `100,1000,10000`
- Prompt variants: `3`
- Runs per case: `2`
- Max output tokens: `512`

## CLI Options

```bash
llm-perf-eval --help
```

Important options:
- `--input-sizes "100,1000,10000"` token targets for prompt context
- `--runs-per-case 3` repeat each prompt/size combo
- `--max-output-tokens 512`
- `--temperature 0.0`
- `--out-dir results`

You can also set:
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`

## Output

After each run:
- `results/results.json`: full per-run data + summary
- `results/report.md`: concise markdown analysis

Each run stores:
- prompt name and input size
- TTFT, TTFR, latency
- completion tokens and reasoning tokens (if exposed)
- OTPS metrics
- error string (if any request fails)

The report also includes an **Input Size Support** section.  
If the model/endpoint rejects an input size (for example 10k context too long), it is explicitly marked as unsupported with the API error reason.

## Notes on Reasoning Metrics

Different OpenAI-compatible providers expose reasoning fields differently.  
This tool attempts common streaming keys (such as `reasoning` / `reasoning_content`) and records `ttfrt_seconds` when detected.  
If a provider does not stream reasoning separately, `ttfrt_seconds` may be `null`.
