# Tasks

## Implementation Tasks

- [x] Update routing model config metadata in `src/kani/config.py` from `context_window_tokens` to `max_input_tokens` on `ModelEntry` and `ResolvedModelCandidate`, preserving positive integer validation and provider override behavior. (verification: unit - update or add config assertions in `tests/test_input_limit_routing.py` or its renamed equivalent)
- [x] Ensure legacy per-model `context_window_tokens` is not silently ignored by either rejecting it with a clear validation error or mapping it to `max_input_tokens` with a deprecation warning. (verification: unit - `tests/test_input_limit_routing.py::TestInputLimitConfig::test_legacy_context_window_tokens_is_rejected`; `uv run pytest tests/test_input_limit_routing.py -q` passed)
- [x] Update `src/kani/router.py` routing eligibility helpers, variables, comments, and log fields to use input-limit terminology while preserving candidate filtering behavior: skip only when `prompt_tokens > max_input_tokens`. (verification: unit - `src/kani/router.py` uses `max_input_tokens`; `tests/test_input_limit_routing.py::TestInputLimitRouting::test_long_request_skips_too_small_primary`; `uv run pytest tests/test_input_limit_routing.py -q` passed)
- [x] Preserve routing ordering and fallback semantics with the renamed metadata: capability filtering remains mandatory, unknown limits stay eligible, fallback and higher-tier promotion can satisfy long input, and cooldown is applied after input-limit filtering. (verification: unit - `tests/test_input_limit_routing.py::TestInputLimitRouting` covers unknown limit, fallback, higher-tier, capability, and cooldown scenarios; `uv run pytest tests/test_input_limit_routing.py -q` passed)
- [x] Update routing metadata in `config.yaml`, `config.example.yaml`, README routing documentation if present, and any non-archived OpenSpec references from `context_window_tokens` to `max_input_tokens`. (verification: manual - `config.yaml`, `config.example.yaml`, `README.md`, `openspec/specs/config/spec.md`, and `openspec/specs/routing/spec.md` inspected; repository search confirms remaining `context_window_tokens` references are smart-proxy compaction, legacy rejection tests/specs, proposal history, or archived history)
- [x] Keep smart-proxy compaction semantics unchanged: `smart_proxy.context_compaction.context_window_tokens` remains valid and continues to drive threshold math in `src/kani/proxy.py`. (verification: unit - existing `tests/test_compaction.py` coverage remains passing without renaming compaction config)
- [x] Rename or update `tests/test_input_limit_routing.py` so test names and assertions describe input-limit routing rather than context-window routing. (verification: unit - `uv run pytest tests/test_input_limit_routing.py -q` or the renamed test file passes)
- [x] Run focused and broad quality checks after implementation. (verification: integration - `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/`, `uv run pytest tests/ -q`, and `uv build` pass)

## Future Work

- Automatic provider/model metadata discovery remains out of scope for this rename.
- Long-term support for both field names is out of scope unless a future proposal defines a formal deprecation window.

## Final Validation

Expected archive gate: `cflx openspec validate rename-max-input-tokens --archive-gate`

## Acceptance #1 Failure Follow-up

- [x] Active task checklist is complete: no unchecked `- [ ]` items were found under `openspec/changes/rename-max-input-tokens/tasks.md`; implementation evidence and focused/broad checks otherwise support the behavior (`cflx openspec validate rename-max-input-tokens --strict`, focused routing tests, ruff, format check, pyright, full pytest, and `uv build` passed). However PASS is forbidden while the archive-gate commit path is blocked. (verification: manual - tasks checklist updated and rechecked in `openspec/changes/rename-max-input-tokens/tasks.md`)
- [x] Archive commitability is blocked by the real archive-gate validation step: `cflx openspec validate rename-max-input-tokens --archive-gate` exited 1. Evidence: `openspec/changes/rename-max-input-tokens/tasks.md:6`, `:7`, `:8`, and `:9` fail because each verification note does not cite repository-verifiable evidence such as source paths, tests, or runnable commands. Action: update those completed task verification notes to include concrete repository evidence/commands, then rerun the archive gate. (verification: integration - verification notes on tasks 2-5 now cite repository evidence and `cflx openspec validate rename-max-input-tokens --archive-gate` passed)

## Acceptance #2 Failure Follow-up

- [x] Rename the focused routing test file so the repository artifact no longer describes this feature as context-window routing. (verification: unit - `tests/test_input_limit_routing.py` exists with `TestInputLimitConfig` and `TestInputLimitRouting`; `uv run pytest tests/test_input_limit_routing.py -q` passed)
- [x] Remove the remaining non-compaction canonical spec wording that described routing metadata as context-window metadata. (verification: manual - `openspec/specs/config/spec.md` now names `max_input_tokens` for routing-time input-limit candidate filtering)
- [x] Replace self-referential OpenSpec-validation follow-up checkboxes with repository-verifiable implementation follow-ups and keep final OpenSpec validation only in the non-checkbox `## Final Validation` section. (verification: integration - this section no longer contains final validation as a checkbox task; `cflx openspec validate rename-max-input-tokens --strict` passed)
