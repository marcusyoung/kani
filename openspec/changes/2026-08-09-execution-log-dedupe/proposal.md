---
change_type: implementation
priority: high
references:
  - src/kani/proxy.py
  - tests/test_api_keys_proxy.py
  - openspec/specs/proxy-api/spec.md
---

# Execution log deduplication: log usage once per request

**Change Type**: implementation (bug fix)

## Problem / Context

The `_log_usage` function in `proxy.py` was called for every streamed SSE chunk
that carried a `usage` field. Some providers (notably Synthetic) emit usage on
nearly every chunk, while others (Ollama Cloud) emit it once on the final chunk.

Evidence from 24h of execution logs (2026-08-08 to 2026-08-09):

| Provider / model | Unique requests | Log lines | Lines/request |
| --- | --- | --- | --- |
| Synthetic `syn:large:vision` (REASONING) | 27 | 24,420 | 904 |
| Synthetic `syn:large:text` (COMPLEX) | 76 | 35,252 | 464 |
| Ollama Cloud `deepseek-v4-flash` (MEDIUM) | 231 | 231 | 1 |
| Ollama Cloud `glm-5.2` (COMPLEX) | 168 | 168 | 1 |

Each per-chunk record contained cumulative (not delta) token counts and a
monotonically growing `elapsed_ms`. Consequences:

- Log volume inflated ~60K lines/day to ~500.
- Dashboard analytics corrupted by cumulative duplicates (each request counted
  hundreds of times).
- Synchronous JSONL append + SQLite connect/insert/commit ran inside the async
  stream generator on every usage-bearing chunk, blocking the event loop.

## Proposed Solution

Log usage once per request at stream end. In the `_stream()` generator in
`src/kani/proxy.py`:

- Accumulate the last seen `usage` dict into a `last_usage` variable while
  streaming (values are cumulative, so last-one-wins).
- Call `_log_usage` once in the `finally` block, after `upstream.aclose()`,
  with `elapsed` computed at that point (true wall-time).

The `finally` block runs on normal completion, client disconnect, and
cancellation, so the request is always logged when usage was seen. The
non-streaming path already logged once and is unchanged.

## Backward compatibility

- Execution log JSONL schema unchanged (`log_execution_event` signature
  unchanged); only the write frequency changes.
- Downstream consumers (dashboard ingest, log analysis) see fewer, correct rows
  instead of many cumulative duplicates.
- No change to the OpenAI-compatible proxy surface.

## Affected specs

- `openspec/specs/proxy-api/spec.md` — added scenario under 使用量ログ:
  streaming requests produce exactly one execution-log record per request with
  final cumulative token values and total wall-time.

## Test plan

- `uv run pytest tests/test_api_keys_proxy.py::TestApiKeyProxy::test_streaming_multi_chunk_usage_logs_once -q` — new test: 3 usage-bearing chunks, asserts `log_execution_event` called exactly once with `completion_tokens=15`.
- `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/` — pass.
- Manual live verification: streaming COMPLEX requests to Synthetic now produce exactly one `USAGE` line and one execution-log record per request_id (verified 2026-08-09, 5 requests, 1 record each).

## Tasks

- [x] Modify `_stream()` in `src/kani/proxy.py` to accumulate `last_usage` and log once in `finally` (verification: unit — `test_streaming_multi_chunk_usage_logs_once`).
- [x] Add `test_streaming_multi_chunk_usage_logs_once` to `tests/test_api_keys_proxy.py` (verification: unit — exactly one `log_execution_event` call with final cumulative values).
- [x] Run quality gates: ruff check, ruff format --check, pyright, pytest (verification: integration — all pass; one pre-existing unrelated sklearn failure in `test_agentic_training_script.py` verified on clean main).
- [x] Update docs: proxy-api spec scenario, REVIEW.md coverage note, README.md execution-logs paragraph (verification: manual).

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-09-execution-log-dedupe --archive-gate`
