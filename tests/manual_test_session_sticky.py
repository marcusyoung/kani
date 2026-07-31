"""Manual E2E test for session-sticky routing against real config.

Loads the user's actual config.yaml and exercises the MEDIUM tier's
session_sticky primary selection. No server or API keys needed.

    uv run python tests/manual_test_session_sticky.py
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from kani.config import load_config
from kani.router import Router

CONFIG_PATH = r"C:\Users\myoun\.config\kani\config.yaml"


def _classify_medium() -> dict:
    """Stub scorer output that routes to MEDIUM tier."""
    return {
        "tier": "MEDIUM",
        "score": 0.5,
        "confidence": 0.8,
        "signals": ["test"],
        "signal_details": {"test": {"raw": "manual-test"}},
        "agentic_score": 0.0,
    }


def main():
    print("Session-Sticky Routing — Real Config E2E Test")
    print("=" * 55)
    print(f"Config: {CONFIG_PATH}")
    print()

    # Load real config
    cfg = load_config(CONFIG_PATH, strict=True)

    # Verify MEDIUM tier is session_sticky
    medium_tier = cfg.profiles["auto"].tiers.get("MEDIUM")
    assert medium_tier is not None, "MEDIUM tier not found in auto profile"
    assert medium_tier.primary_selection == "session_sticky", (
        f"Expected session_sticky, got {medium_tier.primary_selection}"
    )
    primaries = medium_tier.primary_model_ids()
    print(f"MEDIUM tier: primary_selection={medium_tier.primary_selection}")
    print(f"  primaries: {primaries}")
    print(f"  fallbacks: {medium_tier.fallback_model_ids()}")
    print()

    router = Router(cfg)
    failures = 0

    # -----------------------------------------------------------------------
    # Test 1: Deterministic — same key always picks same model
    # -----------------------------------------------------------------------
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        results = [
            router.route(
                [{"role": "user", "content": "hello"}],
                profile="auto",
                session_key="conv-abc-123",
            ).model
            for _ in range(10)
        ]

    unique = set(results)
    if len(unique) == 1:
        print(f"  PASS: deterministic — 10 calls with 'conv-abc-123' all returned '{results[0]}'")
    else:
        print(f"  FAIL: deterministic — got {len(unique)} models: {unique}")
        failures += 1

    # -----------------------------------------------------------------------
    # Test 2: Different keys can select different models
    # -----------------------------------------------------------------------
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        selections = set()
        for i in range(50):
            r = router.route(
                [{"role": "user", "content": "hello"}],
                profile="auto",
                session_key=f"conv-{i:04d}",
            )
            selections.add(r.model)

    if len(selections) >= 2:
        print(f"  PASS: different keys — 50 keys → {len(selections)} distinct models: {sorted(selections)}")
    else:
        print(f"  FAIL: different keys — only {len(selections)} model: {selections}")
        failures += 1

    # -----------------------------------------------------------------------
    # Test 3: No session_key → round-robin fallback
    # -----------------------------------------------------------------------
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        r1 = router.route([{"role": "user", "content": "hello"}], profile="auto")
        r2 = router.route([{"role": "user", "content": "hello"}], profile="auto")
        r3 = router.route([{"role": "user", "content": "hello"}], profile="auto")

    # With 2 primaries, round-robin should alternate
    if r1.model != r2.model and r2.model != r3.model:
        print(f"  PASS: no session_key → round-robin [{r1.model}, {r2.model}, {r3.model}]")
    else:
        print(f"  FAIL: no session_key — expected alternating, got [{r1.model}, {r2.model}, {r3.model}]")
        failures += 1

    # -----------------------------------------------------------------------
    # Test 4: Cross-run determinism — same key, fresh Router
    # -----------------------------------------------------------------------
    router2 = Router(cfg)
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        r_a = router.route(
            [{"role": "user", "content": "hello"}],
            profile="auto",
            session_key="conv-abc-123",
        )
        r_b = router2.route(
            [{"role": "user", "content": "hello"}],
            profile="auto",
            session_key="conv-abc-123",
        )

    if r_a.model == r_b.model:
        print(f"  PASS: cross-run determinism — fresh Router, same key → '{r_a.model}'")
    else:
        print(f"  FAIL: cross-run determinism — '{r_a.model}' vs '{r_b.model}'")
        failures += 1

    # -----------------------------------------------------------------------
    # Test 5: Hash distribution across 200 keys
    # -----------------------------------------------------------------------
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        counts: dict[str, int] = {}
        for i in range(200):
            r = router.route(
                [{"role": "user", "content": "hello"}],
                profile="auto",
                session_key=f"key-{i:05d}",
            )
            counts[r.model] = counts.get(r.model, 0) + 1

    # With 2 candidates, expect roughly 100 each (allow 75-125)
    all_ok = True
    for model, count in counts.items():
        if not (75 <= count <= 125):
            all_ok = False
            print(f"  FAIL: hash distribution — '{model}' got {count} (expected ~100)")
    if all_ok:
        print(f"  PASS: hash distribution — {counts}")
    else:
        failures += 1

    # -----------------------------------------------------------------------
    # Test 6: Fallback entries are populated
    # -----------------------------------------------------------------------
    with patch.object(Router, "_classify", return_value=_classify_medium()):
        decision = router.route(
            [{"role": "user", "content": "hello"}],
            profile="auto",
            session_key="conv-abc-123",
        )

    if decision.fallbacks:
        fb_models = [f.model for f in decision.fallbacks]
        print(f"  PASS: fallbacks populated — {fb_models}")
    else:
        print(f"  FAIL: no fallbacks in decision")
        failures += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 55)
    if failures:
        print(f"RESULT: {failures} test(s) FAILED")
        sys.exit(1)
    else:
        print("RESULT: All 6 tests PASSED — session-sticky routing works end-to-end")


if __name__ == "__main__":
    main()
