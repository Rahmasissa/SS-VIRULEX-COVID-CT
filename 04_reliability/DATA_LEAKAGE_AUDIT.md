# Data Leakage Audit

Generated: `2026-07-14T01:42:38.094771+00:00`

This report is generated from the current on-disk manifests and image bytes. It does not infer patient identity from filenames.

## Overall result

**FAIL**

## Checks

| Check | Status | Count | Finding |
|---|---:|---:|---|
| `official_split_preserved` | **PASS** | 0 | official original-image assignments match |
| `missing_files` | **PASS** | 0 | all manifest image paths exist |
| `unreadable_images` | **PASS** | 0 | all existing images were readable and hashable |
| `duplicate_file_paths_cross_split` | **PASS** | 0 | no duplicate file path crosses splits |
| `duplicate_filenames_cross_split` | **PASS** | 0 | no duplicate filename crosses splits |
| `duplicate_image_ids_cross_split` | **PASS** | 0 | no duplicate explicit image ID crosses splits |
| `exact_duplicates_cross_split` | **PASS** | 0 | no SHA-256-identical image crosses splits |
| `near_duplicates_cross_split` | **WARNING** | 9 | perceptual-hash candidates require manual review |
| `high_similarity_near_duplicates_cross_split` | **FAIL** | 2 | cross-split candidates also meet normalized pixel-correlation and MAE thresholds |
| `conflicting_labels` | **PASS** | 0 | no conflicting labels found for paths, explicit image IDs, or exact hashes |
| `images_assigned_multiple_splits` | **PASS** | 0 | no explicit/content identity is assigned to multiple splits |
| `augmented_samples_train_only` | **PASS** | 0 | all augmented rows are assigned to train |
| `patient_overlap` | **WARNING** |  | not verifiable: no reliable explicit metadata field is available |
| `publication_overlap` | **WARNING** |  | not verifiable: no reliable explicit metadata field is available |
| `split_before_augmentation` | **PASS** |  | The official split table is loaded before augmentation and only train rows are augmented. |
| `statistical_preprocessing_fit_scope` | **PASS** |  | Statistical descriptors are deterministic transforms and zero filtering/MI selection use the train split only in the baseline pipeline. |
| `feature_selection_nested_in_cv` | **FAIL** |  | ZFMIS is fitted once on the complete train split before downstream GridSearchCV, so inner validation folds influence selected features. |
| `med_micn_training_features_out_of_fold` | **FAIL** |  | A single checkpoint trained on train images generates features for those same training images; saved rows have no fold/checkpoint provenance. |
| `calibration_threshold_fold_independence` | **FAIL** |  | Sigmoid calibration and threshold tuning both reuse the same validation labels and fitted validation predictions. |
| `test_labels_excluded_from_model_selection` | **FAIL** |  | Many candidate configurations are evaluated on test and result/final-rule selection is sorted by test balanced accuracy. |

## Image counts

| Split | Class | Source | Images |
|---|---|---|---:|
| test | COVID | original | 98 |
| test | NonCOVID | original | 105 |
| train | COVID | augmented | 277 |
| train | COVID | original | 191 |
| train | NonCOVID | augmented | 234 |
| train | NonCOVID | original | 234 |
| val | COVID | original | 60 |
| val | NonCOVID | original | 58 |

## Identity metadata

- Explicit image ID column: `image_id`
- Explicit patient ID column: `not available - patient isolation is not verifiable`
- Explicit publication/source column: `not available - publication isolation is not verifiable`

The manifest `source` field means original versus augmented; it is not publication provenance.

## Near-duplicate interpretation

Perceptual-hash candidates use Hamming distance <= 6. High-similarity candidates additionally require normalized grayscale correlation >= 0.9 and MAE <= 0.08.
These measurements are screening evidence; candidate images still require visual/source-metadata review.

### High-similarity cross-split candidates

| Left split/class/path | Right split/class/path | pHash distance | Correlation | MAE | Cross-label |
|---|---|---:|---:|---:|---:|
| test/1: `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Images-processed/CT_COVID/kjr-21-e24-p5-29.png` | train/0: `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Images-processed/CT_NonCOVID/71%0.jpg` | 2 | 0.961210 | 0.037378 | True |
| test/1: `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Images-processed/CT_COVID/kjr-21-e24-p5-31.png` | train/0: `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Images-processed/CT_NonCOVID/71%1.jpg` | 4 | 0.929987 | 0.057060 | True |

Both high-similarity pairs are cross-label candidates. Visual inspection indicates matching CT content at different resize/compression levels, but no automatic relabeling was performed.

## Machine-readable artifacts

- `audit_checks.csv`
- `image_counts.csv`
- `audited_manifest.csv`
- `duplicate_groups.csv`
- `exact_duplicate_pairs.csv`
- `near_duplicate_candidates.csv`
- `unreadable_or_missing_files.csv`
- `official_split_differences.csv`
- `audit_summary.json`

Suspicious files were not deleted, moved, or relabeled.
