# Controlled Colab V2 experiment plan

Validated starting checkpoint: `78a06737d67b46b3961dab2f8a26c16db7de2b24`. The user-reported validation result is that its complete lightweight test suite passed in a fresh Colab CPU runtime on 2026-07-23 (`10 passed`). Setup-only gate files may be added in a clean descendant, but do not change the validated model implementation, development labels, locked configuration bytes, or runtime assumptions before completing the locked-baseline reproduction below.

## Boundary and selection rule

Use training and validation/development data only. The existing 203-row official test split has already been consumed and must not be reopened, rescored, or used to choose V2. V2 selection uses repeated grouped development evaluation only. A new final generalization estimate requires a genuinely untouched external cohort or a prospectively locked new test set.

Use a new Drive root; never write into the completed `/content/drive/MyDrive/SS_VIRULEX_Reliability` run:

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
REPO = Path('/content/SS-VIRULEX-COVID-CT')
PACKAGE = Path('/content/drive/MyDrive/colab_package')
DATASET = PACKAGE / 'assets/COVID-CT-Dataset-master'
V2_ROOT = Path('/content/drive/MyDrive/SS_VIRULEX_Reliability_V2')
COMPLETED_ROOT = Path('/content/drive/MyDrive/SS_VIRULEX_Reliability')
assert V2_ROOT != COMPLETED_ROOT
assert REPO.is_dir() and PACKAGE.is_dir() and DATASET.is_dir()
V2_ROOT.mkdir(parents=True, exist_ok=True)
```

Primary selection criterion: mean outer-fold balanced accuracy across seeds 42, 142, and 242. Ties within one standard error are resolved by higher macro F1, lower between-fold variance, better Brier score, and then lower model complexity, in that order. Record sensitivity, specificity, ROC AUC, weighted F1, calibration error, and feature stability; do not optimize any choice against official-test outcomes.

## Setup gate implemented before locked-baseline reproduction

| File | Purpose |
|---|---|
| `04_reliability/notebooks/Controlled_Colab_V2_Setup.ipynb` | Setup-only Colab controller. It uses checked subprocesses and gate tokens, contains no training command, and stops on any validation failure. |
| `04_reliability/ss_virulex_reliability/validate_v2_setup.py` | Verifies the repository/checkpoint boundary, exact dependency pins and imports, CUDA, required paths, development-input hashes/schema/grouping, leakage-audit result, and locked encoder identity. |
| `04_reliability/configs/v2/encoder_locked_baseline.yaml` | Byte-identical V2 alias of the active and preserved locked encoder configuration; SHA-256 `ed6a48211eaedb964b0b2fffaded1959165134a2978dd75a913aae427eff43eb`, configuration ID `112592618942`. |
| `04_reliability/configs/v2/setup_tools.txt` | Exact setup-tool pin: `pytest==9.1.1`. Keeping the test runner separate prevents it from changing the locked model configuration ID. |

The authoritative input package is `/content/drive/MyDrive/colab_package`. It is read only: the validator reads only the three hash-pinned development manifests and development images, then confirms the manifest hashes are unchanged. It never opens the full concept-label source containing official-test labels. The only output root is `/content/drive/MyDrive/SS_VIRULEX_Reliability_V2`; the completed `/content/drive/MyDrive/SS_VIRULEX_Reliability` run is never reused.

## Source/config work after locked-baseline reproduction

The files in this section are future V2 work and do not yet exist at the validated checkpoint. Do not implement or run the tuning matrix until the unchanged locked baseline has reproduced successfully and its report is preserved.

| File | Planned controlled change |
|---|---|
| `04_reliability/ss_virulex_reliability/trainers.py` | Add explicit task class weights and focal-loss `gamma`/`alpha`; add per-concept loss weights; record all resolved loss settings in checkpoint metadata. |
| `04_reliability/ss_virulex_reliability/encoder_experiments.py` | Export per-concept precision/recall/F1/AUC, Brier score, calibration error, and repeated grouped seed summaries. |
| `04_reliability/ss_virulex_reliability/audit_concepts.py` (new) | Audit missingness, prevalence by class/split, lineage consistency, contradictory labels, and all eight concept columns without reading official-test labels. |
| `04_reliability/ss_virulex_reliability/fold_safe_fusion.py` | Add elastic-net logistic regression and gradient boosting; feature-group ablations; top-k values 3, 6, 9, 12, 15, 18, 21, 24, 30, and all; calibration choices none/sigmoid/isotonic; repeated grouped aggregation. |
| Additional `04_reliability/configs/v2/*.yaml` files | One immutable file per post-baseline experiment; include `num_workers`, seed, loss, augmentation, optimizer, fine-tuning, and preprocessing explicitly. |

Required V2 configs:

- `encoder_lr_1e5_wd_1e4.yaml`, `encoder_lr_5e6_wd_1e4.yaml`, `encoder_lr_1e6_wd_1e5.yaml`
- `encoder_frozen.yaml`, `encoder_layer4.yaml`, `encoder_layer3_layer4.yaml`
- `encoder_class_weighted.yaml`, `encoder_focal_g1.yaml`, `encoder_focal_g2.yaml`
- `encoder_concept_weight_025.yaml`, `encoder_concept_weight_050.yaml`
- `encoder_aug_none.yaml`, `encoder_aug_conservative.yaml`, `encoder_aug_ct_window_noise.yaml`

Do not use horizontal flips unless a radiologist-approved rationale is recorded. Conservative CT augmentation should be training-only and limited to small rotation/translation/scale, mild contrast/intensity changes, and carefully bounded noise; never augment validation.

## Colab setup and commands

### 1. Run the setup-only notebook

Select a fresh Colab GPU runtime at the outset, provision a clean checkout containing the setup implementation at `/content/SS-VIRULEX-COVID-CT`, and open `04_reliability/notebooks/Controlled_Colab_V2_Setup.ipynb`. Run its five code cells in order. Do not copy cells from the old `Untitled2.ipynb`.

The notebook hard-stops unless all of the following are true:

- branch `reliability-integration` is clean and checkpoint `78a06737d67b46b3961dab2f8a26c16db7de2b24` is an ancestor;
- the validated implementation files remain unchanged from that checkpoint;
- all exact package dependency pins plus `pytest==9.1.1` import with their intended versions, Med-MICN imports from `01_Med-MICN`, and CUDA is available;
- competing `opencv-python`, `opencv-contrib-python`, and `opencv-contrib-python-headless` distributions are absent, `opencv-python-headless==4.10.0.84` owns the namespace, and the imported `cv2` reports version `4.10.0`;
- all lightweight tests pass;
- the authoritative package files and development image directories exist;
- the three development manifests match their recorded SHA-256 values;
- every loaded split is train or validation, never test;
- the prepared frame has 1,054 rows (936 train and 118 validation), the exact ordered eight concepts, and zero train/validation group overlap;
- the development-only leakage audit contains no `FAIL` and every mandatory split/content/lineage check is `PASS`;
- active, V2, and preserved locked configs are byte-identical and resolve to `112592618942`;
- the package checksums are unchanged after validation.

Each run writes a new non-overwriting report to:

`/content/drive/MyDrive/SS_VIRULEX_Reliability_V2/00_setup_validation/<UTC_RUN_ID>/SETUP_VALIDATION_COMPLETE.json`

The historical `current_pipeline_provenance.json` is deliberately not appended to this setup audit: it documents known failures in the old thesis workflow rather than current V2 input leakage. It remains preserved as historical evidence. The V2 validator independently enforces official-test exclusion and current development leakage gates.

The earlier Colab `pip check` conflicts came from unrelated preinstalled packages whose declared requirements disagree with the locked NumPy/pandas/SciPy/scikit-learn stack. A globally clean Colab environment is therefore not the acceptance criterion. The setup records `python -m pip check` output and return code as diagnostic evidence, but gates on exact installed distribution versions, successful imports, matching imported-module versions, CUDA/Med-MICN imports, and the focused test suite. OpenCV is handled more strictly because several wheels write the same `cv2` namespace: the notebook removes every competing OpenCV distribution before installing the intended headless wheel, and the validator fails if a competitor remains or `cv2.__version__` differs.

### 2. Stop after setup

The notebook ends with `STOP: setup gates passed`. It contains no encoder/fusion execution and no official-test command. Preserve the completion JSON and leakage-audit directory, then wait for separate authorization before running the locked baseline.

### 3. Reproduce the unchanged locked baseline only after authorization

First verify that the active config is the preserved lock and resolves to encoder ID `112592618942`:

```bash
cmp 04_reliability/configs/encoder_gradual_final_block.yaml \
  04_reliability/configs/v2/encoder_locked_baseline.yaml
cmp 04_reliability/configs/v2/encoder_locked_baseline.yaml \
  results/reliability_colab/final_test/provenance/LOCKED_ENCODER_CONFIG.yaml
PYTHONPATH=04_reliability python -m ss_virulex_reliability.oof_med_micn config-id \
  --encoder-config 04_reliability/configs/v2/encoder_locked_baseline.yaml
```

Required ID: `112592618942`. Confirm that the session started in a GPU runtime, and only then run the first training job:

```python
import torch
assert torch.cuda.is_available(), 'Select a Colab GPU runtime before baseline training'
print(torch.cuda.get_device_name(0))
```

```bash
test ! -e /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/10_locked_baseline/seed_42
PYTHONPATH=04_reliability python -m ss_virulex_reliability.encoder_experiments \
  --manifest /content/drive/MyDrive/colab_package/work/manifests/augmented_splits.csv \
  --concept-csv /content/drive/MyDrive/colab_package/work/manifests/concept_labels_development.csv \
  --configs 04_reliability/configs/v2/encoder_locked_baseline.yaml \
  --med-source-root 01_Med-MICN \
  --output-dir /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/10_locked_baseline/seed_42 \
  --backend med_micn --seed 42
```

No `--max-train-rows`, `--max-val-rows`, final-test command, or test manifest is permitted.

### 4. Validate and freeze the baseline reproduction

```python
import json
from pathlib import Path

baseline = Path('/content/drive/MyDrive/SS_VIRULEX_Reliability_V2/10_locked_baseline/seed_42')
summary = json.loads((baseline / 'encoder_experiment_summary.json').read_text())
experiment = summary['experiments'][0]
metadata = experiment['checkpoint_metadata']
metrics = experiment['metrics']
assert summary['test_rows_accessed'] == 0
assert summary['train_rows'] == 936 and summary['validation_rows'] == 118
assert metadata['model_id'] == 'gradual_final_block-112592618942-validation'
assert metadata['feature_order'] == [
    'concept_score_peripheral_ground_glass_opacities',
    'concept_score_bilateral_involvement',
    'concept_score_multilobar_distribution',
    'concept_score_crazy_paving_pattern',
    'concept_score_absence_of_lobar_consolidation',
    'concept_score_localized_or_diffuse_presentation',
    'concept_score_increased_density_in_the_lung',
    'concept_score_ground_glass_appearance',
    'med_task_prob_covid',
    'med_neural_prob_covid',
]
reference = {
    'balanced_accuracy': 0.7054597701149425,
    'roc_auc': 0.8060344827586207,
    'concept_accuracy': 0.6991525423728814,
}
for name, expected in reference.items():
    observed = metrics[name]
    print(name, 'observed=', observed, 'reference=', expected, 'delta=', observed - expected)
    assert abs(observed - expected) <= 0.02, f'{name} exceeds predeclared tolerance'
print('LOCKED BASELINE REPRODUCTION ACCEPTED')
```

The invariant checks must be exact. The numerical tolerance is predeclared as absolute `0.02` for balanced accuracy, ROC AUC, and aggregate concept accuracy because Colab's supplied torch/CUDA versions are not pinned by the package. If any check fails, preserve the output and diagnose the runtime/configuration; do not tune a model and do not relax the tolerance after observing results.

Only after this seed-42 baseline is accepted should it be repeated at seeds 142 and 242, followed by the eight-concept audit and the planned V2 source/config work. No official-test evaluation is part of this plan.

### 5. Concept-label audit before tuning

After adding the planned `audit_concepts.py`, review all eight concepts: peripheral ground-glass opacities, bilateral involvement, multilobar distribution, crazy paving, absence of lobar consolidation, localized/diffuse presentation, increased lung density, and ground-glass appearance. Freeze any corrections in a new versioned development-only concept CSV and checksum. Do not inspect or edit official-test labels. If labels change, reproduce the locked baseline once against the corrected development labels before tuning.

### 6. Med-MICN tuning matrix

Run one immutable config per directory and seed:

```bash
for seed in 42 142 242; do
  PYTHONPATH=04_reliability python -m ss_virulex_reliability.encoder_experiments \
    --manifest /content/drive/MyDrive/colab_package/work/manifests/augmented_splits.csv \
    --concept-csv /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/01_concept_audit/concept_labels_development_v2.csv \
    --configs 04_reliability/configs/v2/encoder_lr_1e5_wd_1e4.yaml \
    --med-source-root 01_Med-MICN \
    --output-dir "/content/SS_VIRULEX_Reliability_V2/20_encoder_grid/lr_1e5_wd_1e4_seed${seed}" \
    --backend med_micn --seed "$seed"
done
```

Apply the same command pattern to fine-tuning depth, class-weight/focal-loss, concept-loss weight, and CT-augmentation configs. Change one factor family at a time. Use early stopping on grouped development validation only; compare learning rate, weight decay, best epoch, calibration, class recall, and eight concept metrics. Promote one encoder before OOF generation.

### 7. OOF generation for promoted encoders

```bash
for seed in 42 142 242; do
  PYTHONPATH=04_reliability python -m ss_virulex_reliability.oof_med_micn oof \
    --manifest /content/drive/MyDrive/colab_package/work/manifests/augmented_splits.csv \
    --concept-csv /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/01_concept_audit/concept_labels_development_v2.csv \
    --encoder-config 04_reliability/configs/v2/ENCODER_PROMOTED.yaml \
    --med-source-root 01_Med-MICN \
    --output-dir "/content/SS_VIRULEX_Reliability_V2/30_oof/promoted_seed${seed}" \
    --backend med_micn --folds 5 --seed "$seed" --resume
done
```

Require exactly one OOF prediction per development image, zero outer/inner group overlap, zero test rows, matching configuration IDs, and validated resume provenance.

### 8. Fusion, ablation, top-k, calibration, and threshold comparison

After the planned fusion CLI extensions, run each seed:

```bash
PYTHONPATH=04_reliability python -m ss_virulex_reliability.fold_safe_fusion evaluate \
  --manifest /content/drive/MyDrive/colab_package/work/manifests/augmented_splits.csv \
  --concept-csv /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/01_concept_audit/concept_labels_development_v2.csv \
  --train-features /content/drive/MyDrive/colab_package/assets/metadata/train_statistical_features_historical.csv \
  --val-features /content/drive/MyDrive/colab_package/assets/metadata/val_statistical_features_historical.csv \
  --feature-set-json /content/drive/MyDrive/colab_package/assets/metadata/statistical_feature_sets.json \
  --oof-med-features /content/SS_VIRULEX_Reliability_V2/30_oof/promoted_seed42/oof_med_micn_features.csv \
  --output-dir /content/SS_VIRULEX_Reliability_V2/40_fusion/seed42 \
  --outer-folds 5 --inner-folds 3 --seed 42 \
  --models decision_tree logistic_elasticnet gradient_boosting \
  --feature-groups statistical concept diagnosis all \
  --subset-sizes 3 6 9 12 15 18 21 24 30 36 \
  --calibration-methods none sigmoid isotonic
```

Repeat for seeds 142 and 242 with their matching OOF files. Thresholds are selected from group-disjoint development predictions only. Compare repeated mean/std balanced accuracy, macro/weighted F1, ROC AUC, sensitivity, specificity, Brier/ECE, feature stability, selected groups, calibration, and thresholds.

### 9. Lock and stop before any test

Write the winning source commit, concept checksum, encoder config, OOF IDs, fusion config, calibration method, threshold rule, selected feature order, and repeated-development summary into a lock file. Recompute an immutable configuration ID and verify it twice. Do **not** run the existing `final-test` commands.

If and only if a new untouched external cohort is acquired, create a separate one-time evaluation command with an explicit acknowledgement and new output directory. Otherwise report repeated grouped development performance as the V2 result and label external performance unavailable.

### 10. Persistent synchronization

The locked-baseline command above writes directly to `V2_ROOT` on Google Drive, so it does not require a second copy step. Verify that its summary, checkpoints, and logs are present there before ending the runtime.

If a later experiment deliberately writes first to the ephemeral runtime path `/content/SS_VIRULEX_Reliability_V2`, synchronize it with:

```bash
test -d /content/SS_VIRULEX_Reliability_V2
rsync -a --checksum /content/SS_VIRULEX_Reliability_V2/ \
  /content/drive/MyDrive/SS_VIRULEX_Reliability_V2/
```

Then verify file counts and SHA-256 manifests on Drive. Sync back to this repository later by selecting only summaries, configs, predictions permitted by the evaluation policy, manifests, and small final models; never import checkpoint forests, caches, datasets, or the whole Colab run.
