# Tasks

## Implementation Tasks

- [ ] Remove the unsafe final selection fallback in `src/kani/router.py` that reuses `tier_cfg.resolve_primary_candidate_entries()` after input-limit filtering leaves no eligible candidates. (verification: unit - regression test fails if a known-over-limit primary is selected when no eligible candidate exists)
- [ ] Add or reuse a clear routing failure path for “no input-limit-eligible candidate” and ensure proxy/API error handling remains structured. (verification: unit/integration - focused router test checks the exception; proxy-boundary coverage is added or existing error response coverage is updated if the exception can reach FastAPI)
- [ ] Preserve valid fallback promotion when selected-tier primaries are over limit but selected-tier fallback candidates are eligible. (verification: unit - routing test with over-limit primary and eligible fallback selects fallback)
- [ ] Preserve higher-tier promotion when selected-tier primary/fallback candidates are over limit but a higher-tier candidate is eligible. (verification: unit - routing test with higher-tier eligible candidate selects that candidate)
- [ ] Preserve unknown-limit backward compatibility: candidates without `max_input_tokens` remain eligible and can be selected when known-limit candidates are over limit. (verification: unit - routing test proves unknown-limit candidate remains selectable)
- [ ] Keep cooldown ordering safe: if input-limit-eligible candidates are cooling down, cooldown fallback may ignore cooldown only among input-limit-eligible candidates and must not re-add known-over-limit candidates. (verification: unit - routing test with over-limit candidate plus cooled eligible candidate never selects over-limit candidate)
- [ ] Update routing logs/messages to avoid wording that implies kani will fall back to unsafe upstream handling after input-limit filtering. (verification: manual - inspect `src/kani/router.py` log strings and run focused tests)
- [ ] Run focused and broad quality checks. (verification: integration - `uv run pytest tests/test_context_window_routing.py -q` or renamed focused test, `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/`, and `uv run pytest tests/ -q` pass)

## Future Work

- Operator policy for strict “unknown limit is ineligible” routing is out of scope and should be a separate proposal if needed.

## Final Validation

Expected archive gate: `cflx openspec validate enforce-input-limit-fallback --archive-gate`
