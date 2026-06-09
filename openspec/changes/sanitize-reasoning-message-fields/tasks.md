## Implementation Tasks

- [ ] Define the exact repo-local compatibility flag and where it lives, choosing provider-level, `model_rules`-level, or both, for preserving `messages[].reasoning_content`. Verification: unit - config/model-rule tests prove only that explicit flag controls preservation of `messages[].reasoning_content`. Completion condition: preservation behavior is controlled by repo-local metadata rather than an unreviewable external assumption.

- [ ] Implement routed proxy-layer message sanitization before primary upstream requests are sent. Verification: integration - proxy test captures the primary upstream JSON payload and asserts unsupported `messages[].reasoning_content` is absent. Completion condition: the upstream payload is sanitized while the client request object used for routing classification is not treated as a tier escalation signal.

- [ ] Apply sanitization independently for fallback attempts. Verification: integration - proxy fallback test captures fallback upstream JSON payload and asserts compatibility is evaluated for the fallback model/provider. Completion condition: fallback payload does not reuse stale primary-provider reasoning-message compatibility.

- [ ] Preserve explicitly supported `messages[].reasoning_content`. Verification: integration - proxy test configures model/provider support and asserts `messages[].reasoning_content` remains in the upstream payload. Completion condition: unsupported and supported cases are both covered by tests.

- [ ] Document pass-through behavior as unchanged by this feature. Verification: integration - proxy test sends a non-`kani/` model request with `messages[].reasoning_content` and asserts routed-request sanitization logic is not applied. Completion condition: pass-through behavior is explicitly covered and does not depend on routed compatibility metadata.

- [ ] Prove routing tier is not forced by reasoning metadata alone. Verification: unit/integration - route the same simple prompt with and without prior `reasoning_content` metadata and assert both requests resolve to the same tier. Completion condition: adding tier escalation based on `reasoning_content` causes a test failure.

- [ ] Run targeted and relevant project checks after implementation. Verification: manual - run targeted proxy/config/routing pytest files; run ruff/pyright if Python code changed. Completion condition: command output shows no failures.

## Future Work

Additional provider-specific message fields can be added as separate compatibility issues when concrete failures are verified.

## Final Validation

Expected archive gate: `cflx openspec validate sanitize-reasoning-message-fields --archive-gate`.
