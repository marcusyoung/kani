---
change_type: feature
priority: medium
references:
  - src/kani/config.py
  - src/kani/proxy.py
  - tests/test_proxy_reload.py
  - openspec/specs/config/spec.md
  - openspec/specs/proxy-api/spec.md
---

# Per-model extra request-body field injection (`model_rules[].extra_body`)

**Change Type**: feature

## Problem / Context

kani routes requests to upstream providers but has no way to inject
provider-specific request-body fields for a selected model. This blocks
Doubleword's async (flex) pricing, which requires `service_tier: flex` in the
request body. Without injection, Doubleword serves realtime pricing, defeating
the cost saving.

The field must be **per-model**, not per-provider, so operators can pick and
choose sync vs async per model on the same provider (e.g. K3 async, GLM-5.2
sync), and can alias a model ID to run the same model both ways.

## Proposed Solution

Add an optional `extra_body: dict[str, Any] | None` field to `ModelRuleEntry`
(the existing per-model metadata registry, keyed by prefix + optional provider
filter). In `_prepare_body_for_candidate` — the single chokepoint both primary
and fallback candidates pass through — merge the best-matching rule's
`extra_body` into the upstream payload, last, so these values win over
client-provided fields.

Matching uses the same prefix/provider scoring precedence as
`reasoning_style` and `supports_reasoning_content`: provider-specific rules
outrank provider-agnostic rules before prefix specificity is compared.

Example:

```yaml
model_rules:
  - prefix: "moonshotai/kimi-k3"
    provider: "doubleword"
    capabilities: [tools, json_mode]
    reasoning_style: "xai"
    extra_body:
      service_tier: flex        # async
  - prefix: "zai-org/GLM-5.2-FP8"
    provider: "doubleword"
    capabilities: [tools, json_mode]
    reasoning_style: "xai"
    # no extra_body → sync (immediate)
```

## Backward compatibility

- `extra_body` defaults to `None`; existing configs are unaffected.
- `ModelCapabilityEntry = ModelRuleEntry`, so the legacy `model_capabilities`
  alias also gains `extra_body` (harmless additive field).
- No change to the OpenAI-compatible proxy surface or routing semantics.
- `extra_body` values are merged last, so they intentionally override any
  client-provided field of the same name for matching candidates.

## Affected specs

- `openspec/specs/config/spec.md` — document `extra_body` on `ModelRuleEntry`
  and its precedence in the "Model metadata documentation" requirement; add
  `extra_body` to the `model_rules` primary-metadata scenario enumeration.
- `openspec/specs/proxy-api/spec.md` — add a scenario under the reasoning
  control injection requirement documenting that `model_rules[].extra_body`
  fields are merged into the upstream payload for matching candidates and win
  over client-provided fields.

## Test plan

- `uv run pytest tests/test_proxy_reload.py::TestModelExtraBody -q` — new tests:
  merge into prepared body, client-field override, no-rule no-op, other-provider
  exclusion, provider-specific rule precedence.
- `uv run ruff check src/`, `uv run ruff format --check src/ tests/`,
  `uv run pyright src/` — pass.
- `uv run pytest tests/test_proxy_reload.py tests/test_config.py -q` — pass
  (no regressions).

## Tasks

- [x] Add `extra_body: dict[str, Any] | None` to `ModelRuleEntry` in
  `src/kani/config.py` (verification: unit — config loads a rule with
  `extra_body`; `ModelCapabilityEntry` inherits the field).
- [x] Add `_get_model_extra_body()` in `src/kani/proxy.py` using the same
  prefix/provider scoring as `reasoning_style`, and merge the result into
  `_prepare_body_for_candidate` last (verification: unit —
  `TestModelExtraBody`).
- [x] Add `TestModelExtraBody` to `tests/test_proxy_reload.py` covering merge,
  client-field override, no-rule no-op, other-provider exclusion, and
  provider-specific precedence (verification: unit — 5 tests pass).
- [x] Run quality gates: ruff check, ruff format --check, pyright, pytest
  (verification: integration — all pass).
- [x] Update docs: config spec, proxy-api spec, README.md (verification:
  manual).

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-14-add-model-extra-body --archive-gate`
