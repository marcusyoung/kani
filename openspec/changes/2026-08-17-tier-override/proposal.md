---
change_type: feature
priority: medium
references:
  - src/kani/router.py
  - src/kani/proxy.py
  - src/kani/cli.py
  - tests/test_tier_override.py
  - openspec/specs/routing/spec.md
  - openspec/specs/proxy-api/spec.md
  - backlog/docs/decisions/doc-1
  - backlog/docs/decisions/doc-2
---

# Per-turn tier override via `/kani:<tier>` slash command

**Change Type**: feature

## Problem / Context

kani routes prompts through a learned distilled feature classifier that assigns one of four tiers (SIMPLE, MEDIUM, COMPLEX, REASONING) and selects a model per profile tier config. There is no mechanism for a client to override the tier for a specific turn. Users sometimes know the scorer will under-classify a prompt and want to force a stronger tier, or conversely force a cheaper tier for a prompt the scorer would over-classify.

A per-message command is the right mechanism: the override lives in the message itself, works from any client (chat UIs, CLI, API), and is naturally per-turn with no sticky state to reset.

## Proposed Solution

Introduce a `/kani:<tier>` token at position 0 of the latest user message content. A shared helper `parse_tier_override(messages)` extracts the tier override and strips the token from the message text before forwarding upstream. `Router.route()` gains a `tier_override` parameter that, when valid, skips the scorer and pins the tier. The proxy (`/v1/chat/completions`), the CLI (`kani route`), and the debug endpoint (`/v1/route`) all call the helper and pass the override to `route()`.

### Syntax

- Token: `/kani:<tier>` at the start of the latest user message content.
- Regex: `^/kani:(\w+)\s*` (case-insensitive tier matching).
- Valid tiers: `simple`, `medium`, `complex`, `reasoning` (matched against `_TIER_ORDER`, upper-cased).
- The token is stripped from the message before forwarding upstream so the model never sees it.
- Invalid tier values (e.g. `/kani:foo`): the token is still stripped but `tier_override` is set to `None` and the scorer runs normally. A warning is logged.
- Only the latest user message is scanned. Tokens in assistant, system, or earlier user messages are ignored.
- For list content (multimodal), only the first `{"type": "text", "text": ...}` part is checked.

### Routing behavior when override is active

When a valid override is set, `Router.route()` skips the scorer and pins the tier. Synthetic values are used for audit fields: `score=1.0`, `confidence=1.0`, `signals=["tier_override"]`, `agentic_score=0.0`. Capability filtering, input-limit checks, tier fallback, and primary round-robin selection all still apply normally.

## Backward compatibility

- No new configuration keys; the feature is always available.
- No change to the OpenAI-compatible proxy surface or request/response shapes.
- Requests without the `/kani:` prefix behave identically to before (no parsing overhead beyond a regex match on the latest user message).
- The token is stripped before compaction so it cannot leak into compaction summaries.

## Affected specs

- `openspec/specs/routing/spec.md` — add a "Per-turn tier override" requirement covering token parsing, stripping, valid/invalid tier handling, latest-user-message-only scanning, and the scorer-skip routing behavior.
- `openspec/specs/proxy-api/spec.md` — add a requirement that the proxy, CLI, and debug endpoint honor the override; add a scenario that the token is stripped from the upstream payload.

## Decision records

- `doc-1` — Invalid tier override warns and falls through to normal scoring (token still stripped).
- `doc-2` — Tier override token must be at position 0 of the latest user message content.

## Test plan

- `uv run pytest tests/test_tier_override.py -q` — 44 tests covering `parse_tier_override` helper (25 router), proxy integration (12), and CLI integration (7).
- `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/` — pass.
- `uv run pytest tests/ -q` — full suite passes (1 pre-existing unrelated failure in `test_agentic_training_script`).

## Tasks

- [x] Implement `parse_tier_override()` helper + `Router.route(tier_override=...)` in `src/kani/router.py` (TASK-1.1).
- [x] Integrate tier override into proxy `chat_completions` + `route_debug` in `src/kani/proxy.py` (TASK-1.2).
- [x] Integrate tier override into CLI `route_cmd` in `src/kani/cli.py` (TASK-1.3).
- [x] Document the feature in README.md and OpenSpec specs (TASK-1.4).

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-17-tier-override --archive-gate`