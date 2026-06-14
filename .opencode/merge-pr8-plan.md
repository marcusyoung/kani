# Plan: Merge upstream PR #8 (learned feature classifier restore)

**Goal:** Merge upstream `tumf/kani` main (which includes PR #8) into our fork,
preserving our orthogonal local changes while adopting upstream's principled
classifier fix.

**Branch:** `merge-upstream-classifier-fix` (from current `main`)

---

## Step 1: Create branch and merge upstream/main

```bash
cd C:\Users\myoun\.config\kani\kani-repo
git checkout main
git checkout -b merge-upstream-classifier-fix
git merge upstream/main
```

This will produce conflicts in `src/kani/scorer.py` (both sides changed it
heavily). All other files should merge cleanly — upstream PR #8 touched
`cli.py`, `scorer.py`, `openspec/`, `tests/`; we touched `scorer.py`,
`router.py`, `config.py`, `proxy.py`, `feature_training.py`.

## Step 2: Resolve `scorer.py` — keep upstream logic, inject our additions

The merged file must reflect this policy:

| Section | Source | Notes |
|---------|--------|-------|
| Module constants (`RUNTIME_FEATURE_CLASSIFIER_SUPPORTED`, `FEATURE_EMBEDDING_TIMEOUT_SECONDS`) | **upstream** | New infrastructure |
| `ScoringConfig` | **ours** | Add `disable_axis_overrides: bool = False` alongside upstream's `fallback_tier`/`fallback_confidence` |
| `_DEFAULT_THRESHOLDS` | **ours** | 0.2 / 0.55 / 0.75 |
| `_DEFAULT_WEIGHTS` | **either** | Same values both sides |
| `_build_embedding_client()` | **upstream** | Has `embedding.enabled: false` → RuntimeError (better than env fallthrough) |
| `DistilledFeatureClassifier` | **upstream** | Uses `from_bundle()` with schema validation, `embedding_timeout_seconds`, `feature_schema_version`; no singleton — `load()` returns fresh instance |
| `FeatureClassifierStatus` | **upstream** | Doctor diagnostics support |
| `inspect_feature_classifier_runtime_status()` | **upstream** | Doctor integration |
| `Scorer.__init__` | **upstream** | `_load_feature_classifier` with load-once-cached-error pattern; add `self.config` with `disable_axis_overrides` support |
| `Scorer._load_feature_classifier()` | **upstream** | Load-once, cached error, no singleton |
| `Scorer._classify_with_features()` | **upstream** | Raises RuntimeError if no classifier (caught by `classify()` → conservative default). **No heuristic fallback.** |
| `Scorer.classify()` | **upstream** | Catches exception → default fallback (MEDIUM, 0.35, score=0.0) |
| `_tier_from_axes()` | **ours** | Keep `disable_axis_overrides` param and `agenticTask==high` gate on REASONING tier |
| `_tier_from_score()` | **ours** | Defaults from `_DEFAULT_THRESHOLDS` (0.2/0.55/0.75) |

Specific merge actions for `scorer.py`:

1. Take **all upstream code** as the base
2. Modify `_DEFAULT_THRESHOLDS` to `{"SIMPLE": 0.2, "MEDIUM": 0.55, "COMPLEX": 0.75}`
3. Add `disable_axis_overrides: bool = False` to `ScoringConfig`
4. Modify `_tier_from_axes` signature to accept `disable_axis_overrides: bool = False`
5. Add the `agenticTask==high` gate:
   ```python
   if semantic_labels.get("agenticTask") == "high" and (
       reasoning_score >= 0.75
       or semantic_labels.get("reasoningMarkers") == "high"
   ):
       axis_tier = Tier.REASONING
   ```
   (upstream's version has this block without the `agenticTask` guard —
   replace it with our guarded version)
6. Wire `disable_axis_overrides` through: `_classify_with_features` → `_tier_from_axes(self.config.disable_axis_overrides)` with early return if True
7. Ensure `_tier_from_score` falls back to `_DEFAULT_THRESHOLDS` when key missing (already in our version)

## Step 3: Verify other local changes survived

After merge resolution, diff against `upstream/main` and confirm these
local additions are present:

| File | Change | Expected status after merge |
|------|--------|-----------------------------|
| `src/kani/config.py` | `disable_axis_overrides: bool = False` in KaniConfig | Should survive (no upstream conflict) |
| `src/kani/router.py` | Passes `disable_axis_overrides` to `ScoringConfig` | Should survive — upstream didn't touch router.py |
| `src/kani/proxy.py` | Tool capability detection fix + reasoning content sanitization | Should survive — upstream didn't touch proxy.py |
| `src/kani/feature_training.py` | `DEFAULT_THRESHOLDS` to 0.2/0.55/0.75 | Should survive — upstream didn't touch this file |

## Step 4: Verify `feature_classifier.pkl` compatibility

```bash
uv run pytest tests/test_scorer.py -q -k "bundle_compat or bundle_schema or load"
```

Upstream added bundle schema validation (`from_bundle` checks
`embedding_dim`, `semantic_dimensions`, `feature_schema_version`). If our
bundled `feature_classifier.pkl` is incompatible (wrong schema), this test
will catch it and we'll need to retrain — but since our DistilledFeatureClassifier
was loading successfully from this bundle before, it should pass.

Also run the full test suite:
```bash
uv run pytest tests/
```

## Step 5: Run `kani doctor` to confirm diagnostics

```bash
kani doctor
```

Expected: classifies `feature_classifier.pkl` using the new runtime-support
marker, reports it as loadable (or warns if absent).

## Step 6: Manual verification of routing

Start kani and send a test prompt. Check the routing log for
`signals.method.raw == "distilled-features"` with 15 dimensions — the
same verification the upstream PR used.

## Files checklist

| File | Action |
|------|--------|
| `src/kani/scorer.py` | **Conflict** — merge as described in Step 2 |
| `src/kani/cli.py` | Accept upstream (doctor uses runtime marker) |
| `src/kani/router.py` | Keep ours (disable_axis_overrides wiring) |
| `src/kani/config.py` | Keep ours (disable_axis_overrides field) |
| `src/kani/proxy.py` | Keep ours (tool detection + reasoning sanitization) |
| `src/kani/feature_training.py` | Keep ours (0.2/0.55/0.75 thresholds) |
| `tests/test_scorer.py` | Accept upstream (has new bundle/doctor tests) |
| `tests/test_cli.py` | Accept upstream (updated doctor tests) |
| `tests/test_proxy_reload.py` | Accept upstream (minor adjustments) |
| `openspec/` | Accept upstream (spec updates for PR #8) |

## Rollback

If anything breaks, `git merge --abort` from the working branch, or
check out `main` directly. No destructive operations.
