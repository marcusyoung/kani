## Implementation Tasks

- [x] Extend `ModelEntry` in `src/kani/config.py` with optional `context_window_tokens` and ensure existing string entries and `{model, provider}` entries still validate. (verification: unit - added assertions in `tests/test_context_window_routing.py`; `uv run pytest tests/test_context_window_routing.py -q` passed)
- [x] Preserve candidate metadata through router candidate resolution so routing can evaluate model, provider, and optional context window without losing existing provider precedence semantics. (verification: unit - added `tests/test_context_window_routing.py` cases for string entries, object entries, tier provider defaults, and entry provider overrides; `uv run pytest tests/test_context_window_routing.py -q` passed)
- [x] Add context-window eligibility filtering in `src/kani/router.py` using `_estimate_tokens(messages, model)` so candidates with `context_window_tokens < prompt_tokens` are excluded before cooldown and round-robin selection. (verification: unit - added `tests/test_context_window_routing.py::TestContextWindowRouting::test_long_request_skips_too_small_primary`; `uv run pytest tests/test_context_window_routing.py -q` passed)
- [x] Apply the same eligibility rule to fallback and tier-escalation paths so a too-small current tier can promote an eligible fallback or higher-tier candidate. (verification: unit - added `tests/test_context_window_routing.py` cases for fallback promotion and higher-tier escalation; `uv run pytest tests/test_context_window_routing.py -q` passed)
- [x] Preserve capability and cooldown semantics while adding context filtering. (verification: integration - added equivalent `tests/test_context_window_routing.py` cases; `uv run pytest tests/test_context_window_routing.py -q` passed)
- [x] Update `config.yaml` examples or comments to show `context_window_tokens` on object model entries used for mixed small/large model profiles. (verification: manual - updated `config.example.yaml`; inspected `context_window_tokens`; `KANI_CONFIG=config.example.yaml uv run kani config` passed)
- [x] Run focused and broad quality checks for the Python change. (verification: integration - `uv run pytest tests/test_router_logging.py tests/test_capability_routing.py tests/test_fallback_backoff.py tests/test_context_window_routing.py -q`, `uv run ruff check src/`, and `uv run pyright src/` passed)

## Future Work

- Add automatic model metadata discovery if an upstream provider exposes reliable context-window information.
- Add an explicit router error for cases where every configured candidate has a known insufficient context window, if operators prefer fail-fast behavior over compaction/upstream handling.

## Final Validation

Archive validation itself is the authoritative final OpenSpec validation gate.
Expected archive gate: `cflx openspec validate add-context-window-aware-routing --archive-gate`
