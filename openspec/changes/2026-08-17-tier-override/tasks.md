## Implementation Tasks

- [x] Implement `parse_tier_override()` helper and `Router.route(tier_override=...)` in `src/kani/router.py` (TASK-1.1, verification: unit — `tests/test_tier_override.py` `TestParseTierOverride` + `TestRouterTierOverride`).
- [x] Integrate tier override into proxy `chat_completions` + `route_debug` in `src/kani/proxy.py` (TASK-1.2, verification: unit — `TestProxyTierOverride`).
- [x] Integrate tier override into CLI `route_cmd` in `src/kani/cli.py` (TASK-1.3, verification: unit — `TestCliTierOverride`).
- [x] Document the feature in `README.md` and OpenSpec spec deltas (TASK-1.4, verification: manual — README has "Per-turn tier override" section; OpenSpec change `2026-08-17-tier-override` has proposal + routing/proxy-api spec deltas).

## Future Work

- TASK-2: Per-turn async model preference via `/kani:<tier>:async` modifier.

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-17-tier-override --archive-gate`