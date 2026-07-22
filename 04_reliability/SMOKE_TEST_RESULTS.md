# Smoke-Test Results

Generated on 2026-07-14. These are execution-integrity checks, not scientific experiments.

## Environment

- Host: Intel `x86_64` macOS, Intel UHD Graphics 617 with 1.5 GB dynamic graphics memory.
- PyTorch environment inspection: CUDA unavailable; MPS unavailable; CPU-only training.
- Free disk space at final verification, after removing only disposable dependency caches: approximately 110 MiB.
- A separate existing Med-MICN environment contains PyTorch 2.2.2 and torchvision 0.17.2, but the new local test environment intentionally did not duplicate those large packages.

Because real ResNet50 fold checkpoints are large and disk headroom is critically low, the smoke tests used the explicitly labeled `smoke` backend. That backend uses lightweight image summaries and logistic models solely to exercise data loading, isolation, alignment, checkpoint writing, and output contracts. Every output records `scientific_result: false`.

## OOF feature-generation smoke test

- Input: a deterministic, group-safe 60-row development subset.
- Folds: 2 stratified group folds.
- Grouping: augmentation lineage; patient grouping was not verifiable.
- Fold 0: 21 fit, 9 inner validation, 30 excluded-fold inference; group overlap 0.
- Fold 1: 23 fit, 7 inner validation, 30 excluded-fold inference; group overlap 0.
- Result: 60/60 rows received exactly one `out-of-fold` prediction.
- Checks passed: schema, exact 10-feature ordering, uniqueness, finiteness, probability ranges, metadata alignment, mode, and zero test-row access.
- Configuration ID: `d861113b5021`.

Artifacts are in `outputs/smoke/oof_med_micn/`.

## Encoder-comparison smoke test

- Input: 60 official training rows and 30 official validation rows.
- Completed contracts: current-workspace reproduction, frozen backbone, and gradual final-block configurations.
- Broader/full fine-tuning: correctly skipped because the config is disabled pending validation evidence.
- Test rows accessed: 0.
- All three smoke models produced the same surrogate metrics because the lightweight backend deliberately does not emulate backbone fine-tuning. The observed validation balanced accuracy `0.6333` and AUC `0.72` must not be used to compare encoder strategies.
- Exact configs, checkpoint metadata, provenance rows, validation predictions, and metrics were saved for each completed contract.

Artifacts are in `outputs/smoke/encoder_comparison/`.

## Fold-safe fusion smoke test

- Input: a deterministic, group-safe 60-row development subset with the smoke OOF Med features.
- Outer/inner folds: 2/2.
- Evaluated subset: top 3 for runtime; the code verified the saved hierarchy `[18, 15, 12, 9, 6, 3]`.
- Decision Tree completed both folds. Imputation, zero screening, mutual information, top-k selection, class-weight/hyperparameter search, calibration, and threshold selection were fitted without outer-validation access.
- Fold 0: outer balanced accuracy `0.6667`, AUC `0.5644`, threshold `0.56`.
- Fold 1: outer balanced accuracy `0.5000`, AUC `0.6000`, threshold `0.61`.
- RuleFit: correctly recorded as `SKIPPED_DEPENDENCY_MISSING`; no substitute was mislabeled as RuleFit.
- Test feature file loaded: false. Test labels used: false.
- Locked smoke configuration ID: `a4340f0c0ff2`.

These small-surrogate metrics are not scientific performance estimates. Final current-code artifacts are in `outputs/smoke/fold_safe_fusion_final/`; earlier smoke passes are retained separately and remain non-scientific.

## Automated tests

Command: `../.venv/bin/python -m pytest`

Result: **7 passed** in 5.75 seconds. Coverage includes exact/cross-split audit detection, OOF group isolation, feature-frame alignment, rejection of replacement-feature metadata misalignment, fold-local selector fitting, and the verified subset hierarchy. One harmless joblib warning reported that physical CPU-core count could not be queried.

## Full-run decision

No long or real encoder training was launched. On this CPU-only host, five OOF ResNet50 trainings plus the encoder comparison are expected to take many hours to potentially days depending on epoch stopping, and full checkpoints/intermediate artifacts require well over the available disk headroom. Run the full commands only after freeing several gigabytes; a CUDA-capable host is strongly recommended.
