## Implementation Tasks

- [x] Add `extra_body: dict[str, Any] | None` to `ModelRuleEntry` in `src/kani/config.py` (verification: unit - config loads a rule with `extra_body`; `ModelCapabilityEntry` inherits the field; completion condition: `ModelRuleEntry(prefix="x", extra_body={"service_tier": "flex"})` validates and `ModelCapabilityEntry` exposes the same field).
- [x] Add `_get_model_extra_body()` in `src/kani/proxy.py` using the same prefix/provider scoring precedence as `reasoning_style` (provider-match > prefix length), and merge the best-matching rule's `extra_body` into `_prepare_body_for_candidate` last so it wins over client fields (verification: unit - `TestModelExtraBody`; completion condition: primary and fallback candidates both receive the merged field).
- [x] Add `TestModelExtraBody` to `tests/test_proxy_reload.py` covering: merge into prepared body, client-field override, no-rule no-op, other-provider exclusion, and provider-specific rule precedence (verification: unit - `uv run pytest tests/test_proxy_reload.py::TestModelExtraBody -q` passes 5 tests).
- [x] Run quality gates for touched areas (verification: integration - `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/`, `uv run pytest tests/test_proxy_reload.py tests/test_config.py -q` all pass).
- [x] Update docs: config spec (`extra_body` on `ModelRuleEntry` + precedence), proxy-api spec (extra_body merge scenario), README.md (`model_rules` example) (verification: manual).

## Future Work

- Consider a provider-level `extra_body` default as a fallback if operators need a whole provider on async without per-model rules.
- Consider surfacing the applied `extra_body` in `/v1/route` diagnostics for auditability.

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-14-add-model-extra-body --archive-gate`
