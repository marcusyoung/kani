## Specification Tasks

- [ ] Promote config metadata documentation requirements to canonical `config` spec. Expected canonical result: `config` states `model_rules` is the primary model metadata list, `model_capabilities` is normalized as a legacy alias only when `model_rules` is unset, and capability-required routing fails closed when no configured candidate declares required capabilities. Verification: manual - compare `openspec/specs/config/spec.md` against `src/kani/config.py` and `src/kani/router.py` after archive.

- [ ] Promote routing provider precedence and literal model ID documentation requirements to canonical `routing` spec. Expected canonical result: `routing` documents model-entry provider, tier provider, and default provider precedence, and clarifies model IDs are sent literally to the selected provider. Verification: manual - compare canonical spec with `TierModelConfig` and router provider resolution behavior.

- [ ] Promote smart-proxy session documentation requirements to canonical `smart-proxy-context-compaction` spec. Expected canonical result: compaction session requirements describe explicit headers and no-header cases without stale derived-session claims; no-header requests have no session ID and therefore cannot use cache reuse, persistence, incremental summarization, or background precompaction, while inline compaction may still run. Verification: manual - compare canonical spec with `resolve_session_id` and `_resolve_compaction`.

- [ ] Update README/config examples as part of this docs/spec-only change application to match the canonical requirements. Expected canonical result: user-facing docs explain model metadata, provider overrides, literal model IDs, and compaction session behavior consistently. Verification: manual - README examples and config snippets do not contradict canonical specs.

- [ ] Track any code/spec contradiction discovered during the audit separately. Expected canonical result: this spec-only change does not silently redefine runtime behavior when code does not match. Verification: manual - contradictions are linked to separate issues/proposals rather than hidden in wording.

## Future Work

Implementation proposals may be needed if the audit discovers code behavior that should change rather than documentation that should change.

## Final Validation

Expected archive gate: `cflx openspec validate audit-routing-docs --archive-gate`.
