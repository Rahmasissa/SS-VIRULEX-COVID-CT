# Experiment Commands

Run from the reliability directory. Output directories are intentionally new; the CLIs refuse silent replacement.

```bash
ROOT=/Users/samehissa/SS-VIRULEX-COVID-CT
cd "$ROOT/04_reliability"
LOCAL_PY="$ROOT/.venv/bin/python"
MED_PY=/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main/.venv-py312/bin/python
MED_ROOT=/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main
DATA_ROOT=/Users/samehissa/Downloads/COVID-CT-Dataset-master
CONCEPTS=/Users/samehissa/Downloads/covid_ct_concepts_completed.csv
SVIS_ROOT="$ROOT/03_SS-VIRULEX-outputs/02_SVIS-RULEX/covid_ct_run/exact_sequence"
MANIFEST="$SVIS_ROOT/outputs/01_augmented_splits.csv"
OFFICIAL_MANIFEST="$SVIS_ROOT/outputs/covid_ct_official_splits.csv"
TRAIN_FEATURES="$SVIS_ROOT/outputs/features/04_train_combined_stat_concept_features.csv"
VAL_FEATURES="$SVIS_ROOT/outputs/features/04_val_combined_stat_concept_features.csv"
TEST_FEATURES="$SVIS_ROOT/outputs/features/04_test_combined_stat_concept_features.csv"
FEATURE_SETS="$SVIS_ROOT/outputs/features/04_combined_zfmis_feature_sets.json"
```

The existing `MED_PY` contains PyTorch 2.2.2 and torchvision 0.17.2. Free several gigabytes before training:

```bash
df -h .
"$MED_PY" -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

## Leakage audit

The present data returns exit code 2 because confirmed checks fail; its reports are still complete.

```bash
"$LOCAL_PY" -m ss_virulex_reliability.audit_leakage \
  --manifest "$MANIFEST" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --metadata-csv "$CONCEPTS" \
  --provenance-json configs/current_pipeline_provenance.json \
  --output-dir outputs/leakage_audit \
  --markdown DATA_LEAKAGE_AUDIT.md \
  --overwrite
```

## Archived baseline reproduction

This invokes the original cached runner. It reproduces a historical, test-informed workflow and is not valid for new selection. Omit `--force` to preserve artifacts.

```bash
"$LOCAL_PY" "$ROOT/03_SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py" \
  --med-root "$MED_ROOT" \
  --med-artifact-root /Users/samehissa/Downloads/SS-VIRULEX-outputs/01_Med-MICN \
  --xai-root /Users/samehissa/Downloads/SS-VIRULEX-outputs/02_SVIS-RULEX \
  --data-root "$DATA_ROOT" \
  --concept-csv "$CONCEPTS" \
  --med-python "$MED_PY" \
  --ss-output-root /Users/samehissa/Downloads/SS-VIRULEX-outputs \
  --skip-rulefit
```

To reproduce only the recorded Med-MICN encoder contract on train/validation with exact new provenance:

```bash
PYTHONPATH=. "$MED_PY" -m ss_virulex_reliability.encoder_experiments \
  --manifest "$OFFICIAL_MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --configs configs/encoder_current_reproduction.yaml \
  --med-source-root "$MED_ROOT" \
  --output-dir outputs/full/encoder_current_reproduction \
  --backend med_micn
```

## Encoder comparison without test access

```bash
PYTHONPATH=. "$MED_PY" -m ss_virulex_reliability.encoder_experiments \
  --manifest "$OFFICIAL_MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --configs \
    configs/encoder_current_reproduction.yaml \
    configs/encoder_frozen_baseline.yaml \
    configs/encoder_gradual_final_block.yaml \
    configs/encoder_broader_finetune.yaml \
  --med-source-root "$MED_ROOT" \
  --output-dir outputs/full/encoder_comparison \
  --backend med_micn
```

The broader/full config is disabled and will be skipped. Run it in a new directory with `--include-disabled` only if earlier validation results justify it.

## OOF Med-MICN features

Set `ENCODER_CONFIG` only after choosing the policy without test data. This example uses the frozen baseline.

```bash
ENCODER_CONFIG=configs/encoder_frozen_baseline.yaml
LOCKED_ENCODER_ID=$("$MED_PY" -m ss_virulex_reliability.oof_med_micn config-id --encoder-config "$ENCODER_CONFIG")

PYTHONPATH=. "$MED_PY" -m ss_virulex_reliability.oof_med_micn oof \
  --manifest "$MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --encoder-config "$ENCODER_CONFIG" \
  --med-source-root "$MED_ROOT" \
  --output-dir outputs/full/oof_med_micn \
  --backend med_micn \
  --folds 5 \
  --seed 42
```

The output is `outputs/full/oof_med_micn/oof_med_micn_features.csv`; historical features are not replaced.

## Fold-safe downstream evaluation

First quantify the downstream correction while retaining and documenting the historical Med-feature risk:

```bash
"$LOCAL_PY" -m ss_virulex_reliability.fold_safe_fusion evaluate \
  --manifest "$MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --train-features "$TRAIN_FEATURES" \
  --val-features "$VAL_FEATURES" \
  --feature-set-json "$FEATURE_SETS" \
  --output-dir outputs/full/fold_safe_current_med \
  --outer-folds 5 \
  --inner-folds 3 \
  --models decision_tree rulefit \
  --seed 42
```

Then evaluate OOF fusion. All verified sizes 18, 15, 12, 9, 6, and 3 run by default:

```bash
"$LOCAL_PY" -m ss_virulex_reliability.fold_safe_fusion evaluate \
  --manifest "$MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --train-features "$TRAIN_FEATURES" \
  --val-features "$VAL_FEATURES" \
  --feature-set-json "$FEATURE_SETS" \
  --oof-med-features outputs/full/oof_med_micn/oof_med_micn_features.csv \
  --output-dir outputs/full/fold_safe_oof_fusion \
  --outer-folds 5 \
  --inner-folds 3 \
  --models decision_tree rulefit \
  --seed 42
```

RuleFit is currently unavailable and is recorded as skipped. Install a compatible `rulefit` package first if those results are required; no substitute is mislabeled as RuleFit.

## One-time final test

Run only after the encoder, final epoch count, and `selection_summary.json` are locked from development evidence.

```bash
ENCODER_CONFIG=configs/encoder_frozen_baseline.yaml
LOCKED_ENCODER_ID=$("$MED_PY" -m ss_virulex_reliability.oof_med_micn config-id --encoder-config "$ENCODER_CONFIG")

PYTHONPATH=. "$MED_PY" -m ss_virulex_reliability.oof_med_micn final-test \
  --manifest "$MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --encoder-config "$ENCODER_CONFIG" \
  --med-source-root "$MED_ROOT" \
  --output-dir outputs/final/med_test_once \
  --backend med_micn \
  --locked-config-id "$LOCKED_ENCODER_ID" \
  --final-epochs 15 \
  --allow-final-test \
  --seed 42

FUSION_SELECTION=outputs/full/fold_safe_oof_fusion/selection_summary.json
LOCKED_FUSION_ID=$(sed -n 's/.*"locked_configuration_id": "\([^"]*\)".*/\1/p' "$FUSION_SELECTION")

"$LOCAL_PY" -m ss_virulex_reliability.fold_safe_fusion final-test \
  --selection-summary "$FUSION_SELECTION" \
  --locked-config-id "$LOCKED_FUSION_ID" \
  --allow-final-test \
  --manifest "$MANIFEST" \
  --concept-csv "$CONCEPTS" \
  --train-features "$TRAIN_FEATURES" \
  --val-features "$VAL_FEATURES" \
  --test-features "$TEST_FEATURES" \
  --oof-med-features outputs/full/oof_med_micn/oof_med_micn_features.csv \
  --final-test-med-features outputs/final/med_test_once/final_test_med_micn_features.csv \
  --output-dir outputs/final/fusion_test_once \
  --seed 42
```

These commands require explicit acknowledgements, validate locked IDs, fit calibration/thresholds from development OOF predictions, and evaluate test once.
