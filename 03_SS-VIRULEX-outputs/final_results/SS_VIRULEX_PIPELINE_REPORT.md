# SS-VIRULEX Pipeline Report

Generated on May 24, 2026.

## Short Summary

SS-VIRULEX combines three sources of information:

- SVIS-RULEX statistical image features.
- Med-MICN concept scores.
- Med-MICN diagnosis probabilities added in the improved pipeline.

The improved pipeline then trains stronger fused classifiers, calibrates their probabilities, and tunes the final COVID decision threshold on the validation set.

The best current SS-VIRULEX model is:

| Model | Accuracy | Balanced Accuracy | F1 Weighted | ROC AUC |
|---|---:|---:|---:|---:|
| Calibrated threshold logistic regression, selected_9 features | 0.7340 | 0.7313 | 0.7321 | 0.7591 |

This result is higher than the current Med-MICN and SVIS-RULEX comparison rows by balanced accuracy:

| Model Family | Selected Model | Balanced Accuracy |
|---|---|---:|
| SS-VIRULEX | ss_virulex_calibrated_threshold_logistic_regression_selected_9 | 0.7313 |
| Med-MICN | med_micn_neural_symbolic | 0.7112 |
| SVIS-RULEX | svis_rulex_decision_tree_selected_6 | 0.5544 |

Main comparison file:

`03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_three_model_comparison_covid.csv`

## Pipeline Sequence

### 1. Dataset and Split Preparation

Purpose:

Prepare the COVID-CT image data and make sure train, validation, and test rows are aligned.

Inputs:

- COVID-CT image folders.
- COVID and NonCOVID train, validation, and test split files.

Outputs:

- Official split table.
- Augmented training manifest.
- Augmented split table.

Important output files:

- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/covid_ct_official_splits.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/01_augmented_splits.csv`
- `03_SS-VIRULEX-outputs/dataset/covid_metadata.csv`

### 2. SVIS-RULEX Statistical Branch

Purpose:

Use the SVIS-RULEX visual model to extract image features and convert them into statistical descriptors.

Inputs:

- Augmented split table.
- Trained SVIS-RULEX MobileNetV2 model.
- COVID-CT images.

Process:

- Load or train the custom MobileNetV2 model.
- Extract deep image features.
- Convert deep features into statistical features such as entropy, variance, contrast, coefficient of variation, and signal-to-noise.
- Apply ZFMIS-style feature selection.
- Train SVIS-RULEX baseline rule models.

Outputs:

- Statistical features for train, validation, and test.
- SVIS-RULEX baseline metrics.
- SVIS-RULEX rule files and models.

Important output files:

- `03_SS-VIRULEX-outputs/statistical_branch/features/covid_statistical_features.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_svis_rulex_covid.csv`

### 3. Med-MICN Concept Branch

Purpose:

Use Med-MICN as the concept and neural-symbolic branch of SS-VIRULEX.

Inputs:

- Completed COVID-CT concept CSV.
- Trained Med-MICN checkpoint.
- COVID-CT images.

Process:

- Validate the Med-MICN concept embedding module.
- Validate the Med-MICN neural-symbolic layer.
- Generate or load per-image Med-MICN concept probabilities.
- Generate or load Med-MICN diagnosis probabilities.

Outputs:

- Concept labels.
- Per-image concept probabilities.
- Med-MICN task classifier probabilities.
- Med-MICN neural-symbolic probabilities.
- Med-MICN metrics.

Important output files:

- `03_SS-VIRULEX-outputs/concept_branch/labels/covid_concept_labels.csv`
- `03_SS-VIRULEX-outputs/concept_branch/probabilities/covid_concept_probabilities.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_med_micn_covid.csv`

### 4. New Improvement: Add Med-MICN Diagnosis Probabilities

Purpose:

Before the improvement, SS-VIRULEX used Med-MICN concept scores but did not use Med-MICN diagnosis confidence as fused model input.

The improved pipeline adds two new fused features:

- `med_task_prob_covid`
- `med_neural_prob_covid`

These come from Med-MICN's task classifier and neural-symbolic branch.

Why this helps:

Med-MICN was already strong. Adding its diagnosis probabilities lets SS-VIRULEX keep Med-MICN's strength while also using SVIS-RULEX statistical evidence and concept evidence.

Inputs:

- `task_probability_covid`
- `neural_probability_covid`
- Statistical features.
- Concept score features.

Outputs:

- Fused feature table with statistical features, concept scores, and Med-MICN diagnosis probabilities.

Important output file:

- `03_SS-VIRULEX-outputs/fusion/fused_features_covid.csv`

Current fused feature file shape:

- Rows: 1257
- Columns: 42

The final fused columns include:

- `concept_score_ground_glass_appearance`
- `med_task_prob_covid`
- `med_neural_prob_covid`

### 5. Combined Feature Selection

Purpose:

Select useful features from the combined SS-VIRULEX feature pool.

Inputs:

- Statistical features.
- Med-MICN concept scores.
- Med-MICN diagnosis probabilities.
- Train labels.

Process:

- Build the complete feature pool.
- Remove features with excessive zero values.
- Rank features using mutual information.
- Create selected feature sets such as `selected_18`, `selected_15`, `selected_12`, `selected_9`, `selected_6`, and `selected_3`.

Outputs:

- Combined train, validation, and test feature files.
- Combined ZFMIS feature sets.
- Selected feature folders.

Important output files:

- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/features/04_train_combined_stat_concept_features.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/features/04_val_combined_stat_concept_features.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/features/04_test_combined_stat_concept_features.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/features/04_combined_zfmis_feature_sets.json`
- `03_SS-VIRULEX-outputs/zfmis/zfmis_ranking_covid.csv`

### 6. Explainable Rule Models

Purpose:

Keep an explainable rule-based SS-VIRULEX baseline.

Inputs:

- Combined selected feature sets.
- Train, validation, and test labels.

Process:

- Train combined Decision Tree models.
- Train combined RuleFit models.
- Extract readable rules.

Outputs:

- Decision Tree metrics.
- RuleFit metrics.
- Rule text and CSV files.

Important output files:

- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/05c_combined_gridsearchfortree_results.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/05d_combined_rulefitgridsearchcode_results.csv`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/rules/06_combined_best_decision_tree_rules.txt`
- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/rules/06_combined_best_rulefit_rules.csv`

### 7. New Improvement: Stronger Fused Classifiers

Purpose:

The original SS-VIRULEX fused model relied mainly on Decision Tree and RuleFit. These are explainable, but they can be weaker than modern tabular classifiers.

The improved pipeline adds stronger classifiers:

- Random Forest.
- Extra Trees.
- Gradient Boosting.
- HistGradientBoosting.
- SVM with RBF kernel.
- Logistic Regression.

Inputs:

- Combined feature sets.
- Train labels.
- Validation labels.
- Test labels.

Process:

- Tune each classifier on training data using balanced accuracy.
- Evaluate on validation data.
- Retrain on train plus validation.
- Evaluate once on test data.

Outputs:

- Stronger classifier metrics.
- Saved stronger classifier models.

Important output file:

- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/05e_combined_stronger_classifiers_results.csv`

Best uncalibrated stronger model:

| Model | Feature Set | Balanced Accuracy | ROC AUC |
|---|---|---:|---:|
| SVM RBF | all | 0.6980 | 0.7485 |

This improved over the basic fused Decision Tree, but it still did not exceed Med-MICN.

### 8. New Improvement: Probability Calibration and Threshold Tuning

Purpose:

The stronger classifiers produced useful probabilities, but the default threshold of `0.5` was not optimal for balanced accuracy.

The improved pipeline adds calibration and validation-based threshold tuning.

Process:

1. Train the classifier on training data.
2. Calibrate probabilities on validation data using sigmoid calibration.
3. Search for the threshold that gives the best validation balanced accuracy.
4. Apply that fixed threshold to the test set.
5. Report the final test metrics.

This avoids choosing the threshold directly from the test set.

Inputs:

- Stronger classifier settings from `05e`.
- Combined train features.
- Combined validation features.
- Combined test features.

Outputs:

- Calibrated-threshold metrics.
- Saved calibrated models.
- Final SS-VIRULEX metrics included in the main comparison.

Important output file:

- `02_SVIS-RULEX/covid_ct_run/exact_sequence/outputs/05f_combined_calibrated_threshold_results.csv`

Best calibrated-threshold model:

| Model | Feature Set | Threshold | Accuracy | Balanced Accuracy | F1 Weighted | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | selected_9 | 0.63 | 0.7340 | 0.7313 | 0.7321 | 0.7591 |

Confusion matrix:

```text
[[85, 20],
 [34, 64]]
```

### 9. Final Metrics and Comparison

Purpose:

Collect metrics from SVIS-RULEX, Med-MICN, and SS-VIRULEX into one comparison.

Inputs:

- SVIS-RULEX metrics.
- Med-MICN metrics.
- SS-VIRULEX Decision Tree metrics.
- SS-VIRULEX RuleFit metrics.
- SS-VIRULEX stronger classifier metrics.
- SS-VIRULEX calibrated-threshold metrics.

Outputs:

- Full metrics table.
- Per-family metrics tables.
- Three-model comparison table.
- Plots.

Important output files:

- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_svis_rulex_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_med_micn_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_ss_virulex_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_three_model_comparison_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/evaluation_metrics/metrics_three_model_comparison_covid.png`

Final comparison:

| Rank | Model Family | Selected Model | Accuracy | Balanced Accuracy | F1 Weighted | ROC AUC |
|---:|---|---|---:|---:|---:|---:|
| 1 | SS-VIRULEX | ss_virulex_calibrated_threshold_logistic_regression_selected_9 | 0.7340 | 0.7313 | 0.7321 | 0.7591 |
| 2 | Med-MICN | med_micn_neural_symbolic | 0.7143 | 0.7112 | 0.7106 | 0.7459 |
| 3 | SVIS-RULEX | svis_rulex_decision_tree_selected_6 | 0.5616 | 0.5544 | 0.5415 | 0.5475 |

### 10. Predictions and Explanations

Purpose:

Save per-image predictions and keep visual explanation outputs organized.

Inputs:

- Test split.
- Best decision tree model.
- Best calibrated-threshold model.
- Concept probabilities.
- Existing heatmaps.

Outputs:

- Per-image predictions.
- Feature importance.
- Heatmap case list.
- Combined explanation images.

Important output files:

- `03_SS-VIRULEX-outputs/final_results/predictions/predictions_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/feature_importance/feature_importance_covid.csv`
- `03_SS-VIRULEX-outputs/visual_explanations/heatmaps/heatmap_cases_covid.csv`
- `03_SS-VIRULEX-outputs/final_results/combined_explanations/`

## What Was New in the Improved Pipeline

The main new additions were:

1. Med-MICN diagnosis probabilities were added to the fused feature table.
2. Stronger fused classifiers were trained in addition to Decision Tree and RuleFit.
3. Probability calibration was added using sigmoid calibration.
4. The final COVID decision threshold was tuned on validation data for balanced accuracy.
5. The reporting builder was updated so calibrated-threshold SS-VIRULEX results appear in the final metrics and three-model comparison.

## Simple End-to-End Flow

```text
COVID-CT images
  -> dataset split and augmentation
  -> SVIS-RULEX statistical features
  -> Med-MICN concept scores
  -> Med-MICN diagnosis probabilities
  -> fused SS-VIRULEX feature table
  -> feature selection
  -> Decision Tree and RuleFit explainable baselines
  -> stronger fused classifiers
  -> probability calibration
  -> validation threshold tuning
  -> final SS-VIRULEX test metrics
  -> comparison against Med-MICN and SVIS-RULEX
```

## Main Conclusion

The most important improvement was not just adding a stronger classifier. The best result came from the full combination:

- Fused statistical features.
- Fused Med-MICN concept scores.
- Fused Med-MICN diagnosis probabilities.
- Calibrated probabilities.
- Validation-tuned threshold.

That combination produced the current best SS-VIRULEX result:

```text
balanced_accuracy = 0.7313
```

This is higher than the current Med-MICN comparison row:

```text
balanced_accuracy = 0.7112
```

So, in the current run, SS-VIRULEX is the top model in the three-model comparison.
