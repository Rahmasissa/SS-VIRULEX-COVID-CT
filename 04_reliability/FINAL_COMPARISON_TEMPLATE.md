# Final Comparison Template

Populate validation/nested-CV columns while selecting configurations. Populate the test column exactly once only after the corresponding configuration ID is locked. Do not copy smoke-backend values into this table.

| Version | Upstream Med features | Encoder policy | Fold-safe selection | Calibration/threshold source | Selection evidence | Locked config ID | Test evaluated once? | Balanced accuracy | Accuracy | Precision | Sensitivity | Specificity | F1 | MCC | ROC-AUC | Scientific status |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Original baseline | Single-checkpoint in-sample train features | Original full fine-tune | No | Reused validation; historical test-informed selection | Historical artifacts only | N/A | No, repeatedly evaluated |  |  |  |  |  |  |  |  | Invalid for unbiased comparison |
| Leakage-safe baseline | Existing Med features | Original encoder | Yes | Development OOF only | Nested development CV |  |  |  |  |  |  |  |  |  |  | Pending |
| OOF-fusion version | OOF train/development; final-model test | Reproducible baseline | Yes | Development OOF only | Nested development CV |  |  |  |  |  |  |  |  |  |  | Pending |
| Fine-tuned encoder version | OOF from selected encoder | Frozen or gradual final block | Yes | Development OOF only | Same split/folds and validation protocol |  |  |  |  |  |  |  |  |  |  | Pending |
| Combined final version | OOF from locked encoder; one final-test inference | Locked winning encoder | Yes | Development OOF only | Locked before test |  |  |  |  |  |  |  |  |  |  | Pending |

Also report per-fold metrics and confidence intervals where appropriate, not just aggregate point estimates. The historical original baseline is not directly comparable to leakage-safe rows because its test set participated in selection.
