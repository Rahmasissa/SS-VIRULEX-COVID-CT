# Implementation Changes

All changes are additive under `04_reliability/`. No original source, data, manifest, checkpoint, feature, result, thesis, or article file was modified.

## Package and configurations

| File | Purpose / why changed | Inputs and use | Outputs |
|---|---|---|---|
| `pyproject.toml` | Defines the reliability package and pytest settings. | Install/import from this directory. | Package/test metadata. |
| `requirements.txt` | Records lightweight audit/test dependencies separately from deep learning. | Install into an audit environment. | Reproducible dependency list. |
| `ss_virulex_reliability/__init__.py` | Makes the code importable. | Imported by CLIs/tests. | Package metadata. |
| `ss_virulex_reliability/common.py` | Centralizes the verified 26+8+2 order, manifest normalization, seeds, JSON writing, and feature validation. | Manifests and feature frames. | Normalized/validated frames and JSON. |
| `configs/current_pipeline_provenance.json` | Makes source-level leakage findings machine-readable. | `audit_leakage --provenance-json`. | Appended PASS/FAIL checks with evidence locations. |
| `configs/encoder_current_reproduction.yaml` | Records the observed full-fine-tune/no-augmentation baseline. | Encoder/OOF runner. | Immutable config ID and checkpoint config. |
| `configs/encoder_frozen_baseline.yaml` | Defines a frozen baseline with conservative train-only augmentation. | Encoder comparison/OOF. | Checkpoints, provenance, predictions. |
| `configs/encoder_gradual_final_block.yaml` | Adds frozen warm-up then final-block tuning with a 10x lower backbone LR. | Encoder comparison/OOF if selected. | Checkpoints, provenance, predictions. |
| `configs/encoder_broader_finetune.yaml` | Defines broader tuning but disables it pending validation evidence. | Requires `--include-disabled`. | Separate experiment when justified. |
| `configs/fold_safe_fusion.yaml` | Records folds, hierarchy, fold-fitted operations, calibration/threshold sources, and test policy. | Protocol reference. | Reproducible configuration record. |

## Executable modules

| File | Purpose / why changed | Inputs | Outputs |
|---|---|---|---|
| `ss_virulex_reliability/audit_leakage.py` | Non-destructive split/duplication audit: counts, originals/augmentations, missing/unreadable files, duplicate paths/names/IDs, SHA-256, pHash, pixel similarity, labels, official assignments, explicit patient/publication overlap, and source-level provenance. It never infers patient identity from filenames. | Complete manifest; optional official manifest, metadata, provenance, thresholds. | Markdown plus CSV/JSON; nonzero exit on FAIL. |
| `ss_virulex_reliability/med_data.py` | Aligns concepts/images, inherits original labels for augmentations, validates paths/labels, and creates safe groups. Explicit patient groups are preferred; augmentation lineage is the fallback. | Manifest and concept CSV. | Validated frames, concept order, group IDs, safe subsets. |
| `ss_virulex_reliability/trainers.py` | Implements one trainer contract for a clearly non-scientific smoke backend and the real Med-MICN model. Adds config-driven transforms, differential LRs, frozen/final/full modes, gradual phases, seeds, early stopping/best restoration, parameter counts, and checkpoint provenance. | Config, fit/validation frames, concept order, Med source root. | Checkpoint, history, provenance CSV, metadata JSON, ordered 10-feature predictions. |
| `ss_virulex_reliability/oof_med_micn.py` | Creates exactly one excluded-fold prediction per development row using stratified group folds and group-disjoint inner validation. A separate locked command trains full development and infers test once. | Manifest, concepts, config, source root, folds/backend; locked ID for final test. | OOF/final-test features, fold assignments, checkpoints/provenance, summary. |
| `ss_virulex_reliability/encoder_experiments.py` | Compares current/frozen/gradual policies on one official train/validation protocol and never loads test. Disabled broader tuning is skipped unless explicitly enabled. | Official manifest, concepts, configs, source root, backend. | Exact configs, checkpoints, validation features/metrics, comparison CSV/JSON. |
| `ss_virulex_reliability/fold_safe_fusion.py` | Moves imputation, zero screening, MI, top-k, class weighting, and hyperparameter search inside nested group CV. Reports Decision Tree and genuine RuleFit when available, separates calibration and threshold tuning within development OOF predictions, records stability, and gates test behind a locked ID. | Manifest/concepts, train/val fused files, hierarchy, optional OOF Med file; test files only in final subcommand. | Fold metrics/ranks/stability/OOF predictions, locked summary, model/rules, one-time final metrics. |

## Tests

| File | Purpose | Inputs | Outputs |
|---|---|---|---|
| `tests/test_audit_leakage.py` | Tests exact cross-split duplication and augmentation-leakage statuses. | Temporary synthetic images/manifests. | Audit assertions. |
| `tests/test_oof_isolation.py` | Tests augmentation-lineage fold isolation and required feature alignment/order validation. | Synthetic frames/features. | Isolation/alignment assertions. |
| `tests/test_fold_safe_selection.py` | Tests the saved hierarchy and verifies selectors see only fold-training indices. | Synthetic folds and hierarchy JSON. | Fold-local fitting assertions. |

## Documentation and templates

| File | Purpose | Inputs | Outputs |
|---|---|---|---|
| `DATA_LEAKAGE_AUDIT.md` | Full human-readable audit. | Current manifest, 1,257 images, metadata, provenance. | PASS/WARNING/FAIL and suspicious-pair evidence. |
| `CURRENT_PIPELINE_FINDINGS.md` | Verified architecture, preprocessing, loss, provenance, source/paper conflicts, and leakage. | Source, artifacts, supplied papers, docs in evidence order. | Baseline description. |
| `IMPLEMENTATION_CHANGES.md` | This per-file handoff. | New files/artifacts. | Purpose/use/input/output map. |
| `EXPERIMENT_COMMANDS.md` | Exact audit, archived baseline, encoder, OOF, nested fusion, and locked test commands. | Current paths/environments. | Copyable runbook. |
| `SMOKE_TEST_RESULTS.md` | Records only executed results and labels surrogates non-scientific. | Smoke summaries and pytest. | Execution-integrity report. |
| `FINAL_COMPARISON_TEMPLATE.md` | Comparison form with leakage/test-status fields. | Future validated results. | Human-readable final comparison. |
| `FINAL_COMPARISON_TEMPLATE.csv` | Machine-readable comparison form. | Future validated results. | Analysis-ready CSV. |
| `README.md` | Entry map and safety contract. | Files above. | Quick navigation. |

## Full-audit artifacts

Files under `outputs/leakage_audit/`:

| File | Contents |
|---|---|
| `audit_checks.csv` | One PASS/WARNING/FAIL row per check. |
| `audit_summary.json` | Overall status, thresholds, metadata availability, and checks. |
| `image_counts.csv` | Original/augmented counts by split/class. |
| `audited_manifest.csv` | Normalized rows with SHA-256 and pHash. |
| `duplicate_groups.csv` | Duplicate path, filename, and explicit-ID groups. |
| `exact_duplicate_pairs.csv` | SHA-identical pairs with split/label relation. |
| `near_duplicate_candidates.csv` | Cross-split pHash pairs with correlation, normalized MAE, high-similarity, and cross-label flags. |
| `unreadable_or_missing_files.csv` | Missing/unreadable rows; empty in this run. |
| `official_split_differences.csv` | Official assignment differences; empty in this run. |

## Smoke artifacts

- `outputs/smoke/oof_med_micn/`: OOF features, fold assignments/summary, and each fold's checkpoint metadata, smoke model, and training provenance. Inputs are the 60-row safe subset; outputs validate one-row/one-fold alignment.
- `outputs/smoke/encoder_comparison/`: comparison CSV/JSON and each enabled config's exact YAML, validation features/metrics, model metadata/model/provenance. Inputs are 60 train/30 validation rows; outputs validate the experiment contract.
- `outputs/smoke/fold_safe_fusion_final/`: final current-code fold/model metrics, selected-feature ranks, feature/subset stability, development OOF predictions, selection summary, and skipped-model record. Inputs are 60 development rows with smoke OOF Med features; outputs validate nested isolation and reporting. Earlier smoke passes are retained separately and remain non-scientific.

Every smoke summary/checkpoint records `scientific_result: false`.
