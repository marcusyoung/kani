---
change_type: implementation
priority: high
dependencies:
  - add-configurable-runtime-embedding
references:
  - src/kani/feature_training.py
  - scripts/compare_embeddings.py
  - openspec/specs/routing/spec.md
---

# MLP classifier head for the distilled feature classifier

**Change Type**: implementation

## Problem / Context

The distilled feature classifier used a multi-output logistic regression
(`MultiOutputClassifier(LogisticRegression)`) over 1024-dim embedding vectors.
Stratified 5-fold cross-validation (via `scripts/compare_embeddings.py`) showed
the linear head caps held-out accuracy at ~0.74 regardless of the embedding
model: bge-m3, voyage-4, and voyage-4-large all converged to the same plateau.
The linear decision boundary is too weak to exploit the separation available in
the embedding space, so the bottleneck was the classifier head, not the
embeddings.

## Proposed Solution

Replace the linear base estimator with a scaled MLP pipeline in
`src/kani/feature_training.py`:

- `StandardScaler` + `MLPClassifier(hidden_layer_sizes=(128, 32), relu, adam,
  alpha=1e-3, max_iter=120, early_stopping=True, n_iter_no_change=10,
  random_state=42)` wrapped in the existing `MultiOutputClassifier`.
- After fitting, strip the per-estimator Adam optimizer state
  (`mlp._optimizer = None`) before persisting: optimizer moments are
  training-only and account for ~2/3 of each MLP's pickle size. Predictions
  use only `coefs_`/`intercepts_`, so this is safe and shrinks the bundle from
  ~22.6 MB to ~7.8 MB (~3x).

Runtime (`src/kani/scorer.py`) is unchanged: `DistilledFeatureClassifier`
calls `classifier.predict(embedding.reshape(1, -1))`, and the bundle now
contains `Pipeline` estimators with the same `predict` contract and the same
1x1024 input shape. Bundle schema fields (`embedding_model`, `embedding_dim`,
`weights`, `tier_thresholds`, `feature_schema_version`) are unchanged.

## Evidence

`scripts/compare_embeddings.py` — stratified 5-fold CV, seed 42, anchored on
`agenticTask`, identical folds for both heads, voyage-4 embeddings:

| Head | Mean accuracy | Mean macro-F1 | Dims won |
| --- | --- | --- | --- |
| linear (previous) | 0.741 | 0.699 | 0/14 |
| MLP (new) | 0.799 | 0.742 | 14/14 |

MLP improves every dimension; largest gains: `negationComplexity` +0.086,
`technicalTerms` +0.072, `domainSpecificity` +0.068, `simpleIndicators`
+0.068. The starved-class dimensions (`creativeMarkers` min-class=6,
`negationComplexity` min-class=77) remain noisy and benefit most from more
annotated data rather than further model changes.

## Backward compatibility

- No change to the OpenAI-compatible proxy surface, routing tiers, or fallback
  safety (CONSTITUTION §2 preserved).
- Persisted bundle remains loadable by the existing runtime adapter; no
  migration required.
- `MLPClassifier` and `StandardScaler` come from `scikit-learn`, already a
  dependency (`scikit-learn>=1.8.0`); no new dependencies.

## Affected specs

- `openspec/specs/routing/spec.md` — no requirement text changes; the learned
  classifier's internal head is an implementation detail. A note that the
  classifier head is MLP-based may be added to the future `training` domain.

## Test plan

- `uv run pytest tests/test_scorer.py -q` — 15 passed (runtime adapter
  contract unchanged).
- `uv run pytest tests/test_feature_training.py -q -m heavy` — 1 passed
  (training embedding config/identity).
- Manual: `kani-train.ps1` retrains successfully; bundle load verified via
  `DistilledFeatureClassifier.load("models")` (loadable, `mismatch=False`,
  optimizer stripped, predictions shape (1, 14)).
