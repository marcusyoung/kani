# Tasks

## Implementation Tasks

- [x] Update routing model config metadata in `src/kani/config.py` from `context_window_tokens` to `max_input_tokens` on `ModelEntry` and `ResolvedModelCandidate`, preserving positive integer validation and provider override behavior. (verification: unit - update or add config assertions in `tests/test_context_window_routing.py` or its renamed equivalent)
- [x] Ensure legacy per-model `context_window_tokens` is not silently ignored by either rejecting it with a clear validation error or mapping it to `max_input_tokens` with a deprecation warning. (verification: unit - add a regression test that fails if legacy input is accepted as an unbounded candidate without warning/error)
- [x] Update `src/kani/router.py` routing eligibility helpers, variables, comments, and log fields to use input-limit terminology while preserving candidate filtering behavior: skip only when `prompt_tokens > max_input_tokens`. (verification: unit - routing test proves too-small primary is skipped using `max_input_tokens`)
- [x] Preserve routing ordering and fallback semantics with the renamed metadata: capability filtering remains mandatory, unknown limits stay eligible, fallback and higher-tier promotion can satisfy long input, and cooldown is applied after input-limit filtering. (verification: unit - focused routing tests cover unknown limit, fallback, higher-tier, capability, and cooldown scenarios)
- [x] Update routing metadata in `config.yaml`, `config.example.yaml`, README routing documentation if present, and any non-archived OpenSpec references from `context_window_tokens` to `max_input_tokens`. (verification: manual - search repository non-archive docs/specs for routing metadata references and confirm only smart-proxy compaction or archived history still uses `context_window_tokens`)
- [x] Keep smart-proxy compaction semantics unchanged: `smart_proxy.context_compaction.context_window_tokens` remains valid and continues to drive threshold math in `src/kani/proxy.py`. (verification: unit - existing `tests/test_compaction.py` coverage remains passing without renaming compaction config)
- [x] Rename or update `tests/test_context_window_routing.py` so test names and assertions describe input-limit routing rather than context-window routing. (verification: unit - `uv run pytest tests/test_context_window_routing.py -q` or the renamed test file passes)
- [x] Run focused and broad quality checks after implementation. (verification: integration - `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/`, `uv run pytest tests/ -q`, and `uv build` pass)

## Future Work

- Automatic provider/model metadata discovery remains out of scope for this rename.
- Long-term support for both field names is out of scope unless a future proposal defines a formal deprecation window.

## Final Validation

Expected archive gate: `cflx openspec validate rename-max-input-tokens --archive-gate`
