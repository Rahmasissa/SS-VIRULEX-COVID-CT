# Historical versus reliability result

## Evidence and comparability

The historical prediction file and the Colab reliability prediction file each contain 203 rows. Their `image_id` sets are identical, with zero filename or label mismatches, so row-level outcome comparison is valid. The historical headline is the repository's ranked SS-VIRULEX calibrated logistic-regression `selected_9` result. Historical figures are taken from repository artifacts or derived directly from those 203 predictions; no thesis figure is assumed.

The historical workflow is not an unbiased benchmark: the repository leakage audit shows that feature selection was outside inner CV and many configurations were evaluated/ranked using test balanced accuracy. The reliability workflow improves experimental separation, but its preserved final execution did not use the intended Med feature block.

## Metrics

| Field | Historical SS-VIRULEX headline | Colab reliability final | Change (reliability - historical) |
|---|---:|---:|---:|
| Accuracy | 0.733990 | 0.546798 | -0.187192 |
| Balanced accuracy | 0.731293 | 0.550340 | -0.180952 |
| Weighted F1 | 0.732072 | 0.542279 | -0.189793 |
| Macro F1 | 0.731113 | 0.543597 | -0.187515 |
| ROC AUC | 0.759086 | 0.575802 | -0.183285 |
| COVID sensitivity/recall | 0.653061 | 0.653061 | 0.000000 |
| Specificity | 0.809524 | 0.447619 | -0.361905 |
| Confusion matrix | `[[85, 20], [34, 64]]` | `[[47, 58], [34, 64]]` | 38 more false positives |
| Test samples | 203 | 203 | 0 |
| Calibration | Sigmoid | Sigmoid | Same family; different development procedure |
| Threshold | 0.63 | 0.5277318484 | -0.1022681516 |
| Grouping | Not verifiable from historical artifacts | Augmentation lineage; patient grouping not verifiable | Reliability improvement |
| Configuration ID | Not verifiable from available artifacts | `2a35557fbf41` (downstream), `112592618942` (encoder) | n/a |

Historical `selected_9` features:

1. `concept_score_bilateral_involvement`
2. `shannon_entropy`
3. `concept_score_crazy_paving_pattern`
4. `concept_score_absence_of_lobar_consolidation`
5. `concept_score_localized_or_diffuse_presentation`
6. `concept_score_ground_glass_appearance`
7. `med_task_prob_covid`
8. `entropy`
9. `med_neural_prob_covid`

The reliability lock intended top-15 selection from all 36 predictors. The actual preserved final ranking selected these 15 from only the 26 statistical predictors: `std_dev`, `variance`, `entropy`, `energy`, `contrast`, `mean_abs_dev`, `iqr`, `percentile_75`, `signal_to_noise`, `coef_of_var`, `shannon_entropy`, `root_mean_square`, `harmonic_mean`, `geometric_mean`, and `std_error_mean`.

For an additional like-family reference, the historical selected-15 decision tree achieved accuracy 0.669951, balanced accuracy 0.674150, weighted F1 0.665583, macro F1 0.666716, ROC AUC 0.734937, sensitivity 0.795918, and specificity 0.552381 on the same 203 rows. It also exceeds the preserved reliability final result, but it came from the same historically test-exposed selection workflow.

## Per-sample changes

Against the historical headline model:

| Outcome | Count |
|---|---:|
| Correct in both | 87 |
| Incorrect in both | 30 |
| Fixed by reliability model | 24 |
| Became incorrect in reliability model | 62 |
| Reliability false positives | 58 |
| Reliability false negatives | 34 |

Against the historical selected-15 decision tree, the corresponding counts are 86 correct in both, 42 incorrect in both, 25 fixed, and 50 became incorrect.

## Interpretation

- **Reliability/reproducibility:** improved through development-only locking, group-aware OOF Med features, nested fold-local selection, separated calibration/threshold tuning, immutable identifiers, and a one-time final boundary.
- **Explainability:** improved provenance and explicit ranking/rules were produced. The preserved final rules are interpretable, but they explain a statistical-only model rather than the intended 36-feature fusion candidate.
- **Prediction performance:** did not improve in the preserved result. Accuracy, balanced accuracy, F1, ROC AUC, and specificity are lower; sensitivity is unchanged versus the historical headline. No accuracy-improvement claim is supported.
