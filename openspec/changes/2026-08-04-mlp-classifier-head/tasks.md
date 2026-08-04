## Implementation Tasks

- [x] Swap the distilled classifier head from `LogisticRegression` to a scaled MLP pipeline in `src/kani/feature_training.py` (verification: integration - `uv run pytest tests/test_scorer.py -q` passes with the retrained bundle; completion condition: trained bundle reports `embedding_model=voyage-4`, loads via `DistilledFeatureClassifier.load`, predictions shape (1, 14)).
- [x] Strip per-estimator Adam optimizer state (`mlp._optimizer = None`) after fit before persisting the bundle (verification: manual - bundle size drops 22.6 MB -> 7.8 MB; `mlp._optimizer is None` and `mlp.coefs_` present after load; completion condition: predictions identical to pre-strip state).
- [x] Add `scripts/compare_embeddings.py` to evaluate embedding models and classifier heads via stratified k-fold CV on cached embeddings (verification: manual - run with `--models voyage-4 --heads linear,mlp --folds 5` reproduces the documented +0.058 accuracy / +0.043 macro-F1; completion condition: linear and mlp rows reported with matching means).
- [ ] Run quality gates for touched areas (verification: integration - `uv run ruff check src/`, `uv run ruff format --check src/ tests/`, `uv run pyright src/`, `uv run pytest tests/test_scorer.py tests/test_feature_training.py -q`).

## Future Work

- Re-validate the head choice when the dataset grows, especially for the starved dimensions (`creativeMarkers` min-class=6, `negationComplexity` min-class=77).
- Consider adding a unit test for the optimizer-strip behaviour so the bundle-size invariant is enforced by CI.
- Evaluate MLP hyperparameters (width, alpha, max_iter) via the compare script before any further tuning.

## Final Validation

Expected archive gate: `cflx openspec validate 2026-08-04-mlp-classifier-head --archive-gate`
