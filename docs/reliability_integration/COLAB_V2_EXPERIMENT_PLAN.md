# Controlled Colab V2 experiment plan

## Boundary and selection rule

Use training and validation/development data only. The existing 203-row official test split has already been consumed and must not be reopened, rescored, or used to choose V2. V2 selection uses repeated grouped development evaluation only. A new final generalization estimate requires a genuinely untouched external cohort or a prospectively locked new test set.

Use a new root; never write into the completed `/content/drive/MyDrive/SS_VIRULEX_Reliability` run:

```python
from pathlib import Path
REPO = Path('/content/SS-VIRULEX-COVID-CT')
PACKAGE = REPO / 'colab_package'
DATASET = PACKAGE / 'assets/COVID-CT-Dataset-master'
WORK_ROOT = Path('/content/SS_VIRULEX_Reliability_V2')
PERSIST_ROOT = Path('/content/drive/MyDrive/SS_VIRULEX_Reliability_V2')
assert PERSIST_ROOT != Path('/content/drive/MyDrive/SS_VIRULEX_Reliability')
WORK_ROOT.mkdir(parents=True, exist_ok=False)
```

Primary selection criterion: mean outer-fold balanced accuracy across seeds 42, 142, and 242. Ties within one standard error are resolved by higher macro F1, lower between-fold variance, better Brier score, and then lower model complexity, in that order. Record sensitivity, specificity, ROC AUC, weighted F1, calibration error, and feature stability; do not optimize any choice against official-test outcomes.

## Source/config work before GPU experiments

| File | Planned controlled change |
|---|---|
| `04_reliability/ss_virulex_reliability/trainers.py` | Add explicit task class weights and focal-loss `gamma`/`alpha`; add per-concept loss weights; record all resolved loss settings in checkpoint metadata. |
| `04_reliability/ss_virulex_reliability/encoder_experiments.py` | Export per-concept precision/recall/F1/AUC, Brier score, calibration error, and repeated grouped seed summaries. |
| `04_reliability/ss_virulex_reliability/audit_concepts.py` (new) | Audit missingness, prevalence by class/split, lineage consistency, contradictory labels, and all eight concept columns without reading official-test labels. |
| `04_reliability/ss_virulex_reliability/fold_safe_fusion.py` | Add elastic-net logistic regression and gradient boosting; feature-group ablations; top-k values 3, 6, 9, 12, 15, 18, 21, 24, 30, and all; calibration choices none/sigmoid/isotonic; repeated grouped aggregation. |
| `04_reliability/configs/v2/*.yaml` (new) | One immutable file per experiment; include `num_workers`, seed, loss, augmentation, optimizer, fine-tuning, and preprocessing explicitly. |

Required V2 configs:

- `encoder_locked_baseline.yaml`
- `encoder_lr_1e5_wd_1e4.yaml`, `encoder_lr_5e6_wd_1e4.yaml`, `encoder_lr_1e6_wd_1e5.yaml`
- `encoder_frozen.yaml`, `encoder_layer4.yaml`, `encoder_layer3_layer4.yaml`
- `encoder_class_weighted.yaml`, `encoder_focal_g1.yaml`, `encoder_focal_g2.yaml`
- `encoder_concept_weight_025.yaml`, `encoder_concept_weight_050.yaml`
- `encoder_aug_none.yaml`, `encoder_aug_conservative.yaml`, `encoder_aug_ct_window_noise.yaml`

Do not use horizontal flips unless a radiologist-approved rationale is recorded. Conservative CT augmentation should be training-only and limited to small rotation/translation/scale, mild contrast/intensity changes, and carefully bounded noise; never augment validation.

## Colab cells and commands

### 1. Setup and development-only manifests

```bash
cd /content/SS-VIRULEX-COVID-CT
python colab_package/scripts/runtime_info.py
python colab_package/scripts/prepare_manifests.py \
  --package-root colab_package \
  --dataset-root colab_package/assets/COVID-CT-Dataset-master
PYTHONPATH=04_reliability python -m ss_virulex_reliability.audit_leakage \
  --manifest colab_package/work/manifests/augmented_splits.csv \
  --official-manifest colab_package/work/manifests/official_splits.csv \
  --metadata-csv colab_package/work/manifests/concept_labels_development.csv \
  --provenance-json 04_reliability/configs/current_pipeline_provenance.json \
  --output-dir /content/SS_VIRULEX_Reliability_V2/00_audit/artifacts \
  --markdown /content/SS_VIRULEX_Reliability_V2/00_audit/report.md
```

Immediately assert that the development manifest has zero test rows, every augmented row maps to one original lineage, train/validation group overlap is zero, and all eight concept columns exist.

### 2. Concept-label audit

After adding `audit_concepts.py`:

```bash
PYTHONPATH=04_reliability python -m ss_virulex_reliability.audit_concepts \
  --manifest colab_package/work/manifests/augmented_splits.csv \
  --concept-csv colab_package/work/manifests/concept_labels_development.csv \
  --output-dir /content/SS_VIRULEX_Reliability_V2/01_concept_audit
```

Review all eight concepts: peripheral ground-glass opacities, bilateral involvement, multilobar distribution, crazy paving, absence of lobar consolidation, localized/diffuse presentation, increased lung density, and ground-glass appearance. Freeze corrections before any encoder comparison; write a new versioned concept CSV and checksum. Do not audit official-test labels.

### 3. Locked baseline reproduction

```bash
PYTHONPATH=04_reliability python -m ss_virulex_reliability.encoder_experiments \
  --manifest colab_package/work/manifests/augmented_splits.csv \
  --concept-csv colab_package/work/manifests/concept_labels_development_v2.csv \
  --configs 04_reliability/configs/v2/encoder_locked_baseline.yaml \
  --med-source-root 01_Med-MICN \
  --output-dir /content/SS_VIRULEX_Reliability_V2/10_encoder/baseline_seed42 \
  --backend med_micn --seed 42
```

Repeat to `baseline_seed142` and `baseline_seed242`. Inputs are the development manifest, frozen concept-label checksum, and active Med-MICN source. Compare task balanced accuracy/macro F1/AUC, all per-concept metrics, calibration, best epoch, and variance across seeds. This is the trust anchor; stop if it cannot be reproduced within a predeclared tolerance.

### 4. Med-MICN tuning matrix

Run one immutable config per directory and seed:

```bash
for seed in 42 142 242; do
  PYTHONPATH=04_reliability python -m ss_virulex_reliability.encoder_experiments \
    --manifest colab_package/work/manifests/augmented_splits.csv \
    --concept-csv colab_package/work/manifests/concept_labels_development_v2.csv \
    --configs 04_reliability/configs/v2/encoder_lr_1e5_wd_1e4.yaml \
    --med-source-root 01_Med-MICN \
    --output-dir "/content/SS_VIRULEX_Reliability_V2/20_encoder_grid/lr_1e5_wd_1e4_seed${seed}" \
    --backend med_micn --seed "$seed"
done
```

Apply the same command pattern to fine-tuning depth, class-weight/focal-loss, concept-loss weight, and CT-augmentation configs. Change one factor family at a time. Use early stopping on grouped development validation only; compare learning rate, weight decay, best epoch, calibration, class recall, and eight concept metrics. Promote one encoder before OOF generation.

### 5. OOF generation for promoted encoders

```bash
for seed in 42 142 242; do
  PYTHONPATH=04_reliability python -m ss_virulex_reliability.oof_med_micn oof \
    --manifest colab_package/work/manifests/augmented_splits.csv \
    --concept-csv colab_package/work/manifests/concept_labels_development_v2.csv \
    --encoder-config 04_reliability/configs/v2/ENCODER_PROMOTED.yaml \
    --med-source-root 01_Med-MICN \
    --output-dir "/content/SS_VIRULEX_Reliability_V2/30_oof/promoted_seed${seed}" \
    --backend med_micn --folds 5 --seed "$seed" --resume
done
```

Require exactly one OOF prediction per development image, zero outer/inner group overlap, zero test rows, matching configuration IDs, and validated resume provenance.

### 6. Fusion, ablation, top-k, calibration, and threshold comparison

After the planned fusion CLI extensions, run each seed:

```bash
PYTHONPATH=04_reliability python -m ss_virulex_reliability.fold_safe_fusion evaluate \
  --manifest colab_package/work/manifests/augmented_splits.csv \
  --concept-csv colab_package/work/manifests/concept_labels_development_v2.csv \
  --train-features colab_package/assets/metadata/train_statistical_features_historical.csv \
  --val-features colab_package/assets/metadata/val_statistical_features_historical.csv \
  --feature-set-json colab_package/assets/metadata/statistical_feature_sets.json \
  --oof-med-features /content/SS_VIRULEX_Reliability_V2/30_oof/promoted_seed42/oof_med_micn_features.csv \
  --output-dir /content/SS_VIRULEX_Reliability_V2/40_fusion/seed42 \
  --outer-folds 5 --inner-folds 3 --seed 42 \
  --models decision_tree logistic_elasticnet gradient_boosting \
  --feature-groups statistical concept diagnosis all \
  --subset-sizes 3 6 9 12 15 18 21 24 30 36 \
  --calibration-methods none sigmoid isotonic
```

Repeat for seeds 142 and 242 with their matching OOF files. Thresholds are selected from group-disjoint development predictions only. Compare repeated mean/std balanced accuracy, macro/weighted F1, ROC AUC, sensitivity, specificity, Brier/ECE, feature stability, selected groups, calibration, and thresholds.

### 7. Lock and stop before any test

Write the winning source commit, concept checksum, encoder config, OOF IDs, fusion config, calibration method, threshold rule, selected feature order, and repeated-development summary into a lock file. Recompute an immutable configuration ID and verify it twice. Do **not** run the existing `final-test` commands.

If and only if a new untouched external cohort is acquired, create a separate one-time evaluation command with an explicit acknowledgement and new output directory. Otherwise report repeated grouped development performance as the V2 result and label external performance unavailable.

### 8. Persistent synchronization

After every completed experiment directory:

```bash
rsync -a --checksum /content/SS_VIRULEX_Reliability_V2/ \
  /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/
```

Then verify file counts and SHA-256 manifests on Drive. Sync back to this repository later by selecting only summaries, configs, predictions permitted by the evaluation policy, manifests, and small final models; never import checkpoint forests, caches, datasets, or the whole Colab run.
