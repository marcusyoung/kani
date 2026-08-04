# Proposal: Scoring calibration fixes — simpleIndicators inversion, calibration wiring, per-boundary ambiguity

## Motivation

Three independent scoring/annotation calibration issues identified during a review of the distilled feature pipeline:

1. **`simpleIndicators` inversion**: the annotator prompt and calibration text state "high LOWERS score", but `_build_dimensions` mapped `high` → 1.0 and added it positively to the composite score. With `disable_axis_overrides: true` (the current production config), this bug actively inflated scores for trivial prompts, pushing simple turns toward MEDIUM with no axis logic to compensate.

2. **Calibration text dead code**: `_semantic_dimension_calibration_text()` renders the full 3-level `SEMANTIC_DIMENSION_CALIBRATION` dict (authored in-repo via the `calibrate-feature-annotator` change, 2026-06-09), but was never wired into `_SYSTEM_PROMPT`. The annotator only saw compressed one-line definitions. The rich calibration text and its tests existed but were dead-ended.

3. **No per-boundary ambiguity handling**: the tier mapping used hard threshold cuts with no confidence calibration. Scores landing 0.01 above a boundary were treated as confidently in that tier. ClawRouter's sigmoid calibration (steepness=12, threshold=0.7) was evaluated but rejected — 60% of kani's prompts fall within 0.07 of a boundary, so the ClawRouter parameters would route 60% of traffic to MEDIUM and eliminate COMPLEX entirely. A kani-native per-boundary design is needed instead.

## Design

### simpleIndicators inversion

Invert the value mapping for `simpleIndicators` only in `_build_dimensions` (`src/kani/scorer.py`): `high` → 0.0, `medium` → 0.5, `low` → 1.0. All other dimensions keep the standard mapping. This is a runtime-only change; no retrain is needed (the bundle and weights are untouched).

### Calibration wiring

Replace the inline one-line definitions in `LLMFeatureAnnotator._SYSTEM_PROMPT` with a call to `_semantic_dimension_calibration_text()`, which renders `SEMANTIC_DIMENSION_CALIBRATION` as the full per-dimension, per-label calibration block. The function and dict already exist with tests; this completes the intended design by making them the single source of truth for dimension definitions in the annotator prompt.

### Per-boundary ambiguity handling

Add `ambiguous_bands` to `ScoringConfig` and `KaniConfig`. The config is keyed by boundary name (`SIMPLE_MEDIUM`, `MEDIUM_COMPLEX`, `COMPLEX_REASONING`) with `band` (float) and `prefer` (`"LOWER"` or `"UPPER"`). A score within `band` of a boundary is rerouted to the preferred tier. Each boundary's fail direction is an independent cost/quality decision, not a blanket default.

When unset or empty, behaviour is identical to the current plain threshold mapping.

## Scope

- `src/kani/scorer.py` — `simpleIndicators` inversion, `ambiguous_bands` field, `_ambiguous_bands_normalized()`, `_tier_from_score(..., ambiguous_bands=)`, `_tier_from_axes` forwarding, `_classify_with_features` wiring
- `src/kani/config.py` — `KaniConfig.ambiguous_bands` field
- `src/kani/router.py` — pass `ambiguous_bands` into `ScoringConfig`
- `src/kani/training_data.py` — wire `_semantic_dimension_calibration_text()` into `_SYSTEM_PROMPT`
- `tests/test_scorer.py` — tests for ambiguity bands (prefer UPPER, prefer LOWER, zero band, invalid rejection, axis-override interaction)
- `tests/test_agentic_training_data.py` — existing tests cover the calibration contract (unchanged)
- `README.md` — scoring approach section updated
- `config.example.yaml` — `disable_axis_overrides` and `ambiguous_bands` examples

## Out of scope

- Weight recalibration (deferred; requires outcome-labelled golden set)
- Annotator model upgrade (deferred)
- Force re-annotation of log-sourced labels (deferred)
- Axis override re-enablement (config choice, not a code change)

## Risks

- **`simpleIndicators` inversion changes routing for existing labels**: prompts with `simpleIndicators=high` (trivial) will now score lower. This is the intended fix; the old behaviour was a bug. No retrain needed.
- **Ambiguity bands affect ~18% of traffic**: at band 0.02, approximately 2.1% (SIMPLE_MEDIUM) + 6.2% (MEDIUM_COMPLEX) + 9.7% (COMPLEX_REASONING) of prompts fall within a band. These are coin-flip cases; the per-boundary preference decides the fail direction.
- **Calibration wiring increases annotator prompt size**: system message grows from ~0.9 KB to ~2.6 KB. Negligible cost impact on annotation batches.

## Tasks

- [x] Invert `simpleIndicators` in `_build_dimensions` (verification: unit — `tests/test_scorer.py` behavioural check confirms high→0.0, low→1.0, high lowers composite)
- [x] Wire `_semantic_dimension_calibration_text()` into `_SYSTEM_PROMPT` (verification: unit — existing `test_llm_feature_annotator_prompt_includes_calibration_and_json_contract` passes)
- [x] Implement `ambiguous_bands` in `ScoringConfig`, `_ambiguous_bands_normalized()`, `_tier_from_score`, `_tier_from_axes`, `_classify_with_features` (verification: unit — `tests/test_scorer.py::TestAmbiguousBands`)
- [x] Add `ambiguous_bands` to `KaniConfig` and wire through `router.py` (verification: unit — config load + end-to-end tier assertions)
- [x] Update `README.md` scoring approach section (verification: manual)
- [x] Update `config.example.yaml` with `disable_axis_overrides` and `ambiguous_bands` examples (verification: manual)