# Colab reliability integration report

Date: 2026-07-22
Branch: `reliability-integration`

## Audit outcome

- Initial Git state had no staged changes, conflicts, or tracked deletions. It contained an in-scope `.gitignore` edit plus untracked `04_reliability/` and `reliability_diffs/`.
- `colab_package/` and `colab_final_test_update.zip` were present and are ignored. Neither was modified, moved, deleted, or staged.
- Free storage was critically low: approximately 725 MiB. The imported package is 2.0 GiB; its largest content is duplicate 102 MiB neural checkpoints. No checkpoint or large experiment directory was copied.
- `04_reliability/` is approximately 1.3 MiB, including about 872 KiB of generated local outputs. `/04_reliability/outputs/` is ignored and is not proposed for staging.
- No `AGENTS.md` was present. Root and reliability README guidance was reviewed.
- The requested branch already existed and was already checked out, so no branch creation or switch was needed.

## Integration decisions

| File | Important difference | Decision | Reason | Risk | Verification required |
|---|---|---|---|---|---|
| `fold_safe_fusion.py` | Colab allows OOF Med columns to be inserted when historical statistical CSVs have no placeholders and appends them to the predictor list. | Manually merge and strengthen | This is the specific `before_med_feature_fix` correction and is required for OOF fusion. The completed final-test path had a second schema bug, so exact locked-schema validation and duplicate/missing/order checks were added. | High: silently omitting Med features changes the fitted final model. | Schema/order unit tests; final locked-feature equality; syntax and focused tests. |
| `oof_med_micn.py` | Colab adds per-fold resume files. | Manually merge and strengthen | Resume is valuable for Colab, but the raw patch trusted any two CSVs and broke programmatic callers without `args.resume`. Resume now validates checkpoint presence, model/config metadata, feature schema, fold assignment, and training provenance. | Medium: stale caches could otherwise contaminate OOF features. | Fresh smoke OOF, validated resume, corrupted-cache rejection, no holdout IDs in training provenance. |
| `trainers.py` | Colab uses configurable DataLoader workers and CUDA pinned memory, but removes MPS auto-detection. | Manually merge | Keep local MPS/CPU/CUDA portability; adopt configurable workers and CUDA pinning. Default `2` preserves the completed Colab execution behavior without altering the locked YAML/configuration ID. | Low: worker behavior can affect portability/resource use. | Syntax/import check; real GPU behavior remains a Colab check. |
| `fold_safe_fusion.py.before_med_feature_fix` | Snapshot from immediately before the Med-column fix. | Do not copy | It is debugging evidence only and contains the known failure. | High if imported as source. | Confirm absent from active source and staging. |
| `med_micn_source/` | Vendored Med-MICN and `torch_explain` source. | Keep active repository version | Every non-cache file is byte-identical to `01_Med-MICN/`; copying it would duplicate source. | Low. | Per-file `cmp`/SHA comparison completed. |
| top-level Colab `configs/`, `scripts/`, notebook | Runtime/package helpers with `/content`/Drive assumptions; `prepare_final_test.py` is hard-locked to the completed run. | Do not copy as active implementation | Active reliability configs already match the package. The Colab helpers are package-specific, and the optional augmentation helper permits horizontal flips contrary to the conservative active configuration. A parameterized V2 plan is documented separately. | Medium if mistaken for reusable local tooling. | Path scan and manual review. |

## Applied source behavior

- Historical statistical features may receive the exact expected ordered Med-MICN feature block only after CSV header, metadata, alignment, finiteness, probability-range, duplicate, missing, and order checks pass.
- Development and final-test feature lists must match the development-locked ordered feature list exactly. Final evaluation now refuses a 26-versus-36 feature mismatch.
- OOF resume is opt-in and backward-compatible with programmatic `Namespace` callers. A fold is resumed only when its feature CSV, assignment CSV, checkpoint, checkpoint metadata, and training provenance all match the deterministic fold/configuration.
- Augmentation-lineage grouping, nested fold-local selection, development-only selection, deterministic seeds, and explicit final-test acknowledgement remain intact.
- DataLoader `num_workers` is configurable and must be nonnegative; CUDA pinned memory is enabled only on CUDA. MPS detection remains available.

## Deliberately not copied or staged

- The full `colab_package/`, `colab_final_test_update.zip`, `.before_med_feature_fix`, Python caches, `.pytest_cache`, local smoke outputs, datasets, augmented images, and all large/duplicate neural checkpoints.
- `reliability_diffs/` is an existing generated inspection artifact; it is preserved in place, ignored, and not part of the proposed integration.
- Colab `med_micn_source/` because it duplicates tracked `01_Med-MICN/` source byte-for-byte.
- Colab package scripts/configuration/notebook because they are package/run-specific and include `/content` or Drive layout assumptions.

## Final evidence

The only copy of all five requested final artifacts was under `colab_package/SS_VIRULEX_Reliability/17_fusion_test_once/`. It is tied to downstream lock `2a35557fbf41`; the OOF and final Med summaries use encoder configuration `112592618942`. The five files and their provenance JSON/YAML files were copied byte-for-byte to `results/reliability_colab/final_test/`. See `artifact_manifest.csv` there for source paths, sizes, SHA-256 hashes, identifiers, and Git policy.

Integrity results:

- All JSON files parse.
- Both CSVs have headers and nonzero rows.
- Predictions contain 203 rows; this agrees with `test_rows: 203` and the confusion-matrix total.
- The five preserved artifacts are byte-identical to their Colab-package sources.
- The 5,998-byte model is not covered by the repository's `*.pt` LFS rule and is suitable for ordinary Git.
- Model loading was not attempted because the local Python installation lacks `joblib`, NumPy, pandas, and scikit-learn. Loading it is a lightweight remaining check once the existing project dependencies are available.

### Important final-run limitation

The locked development candidate records 36 ordered predictors (26 statistical plus 10 Med-MICN). The preserved final ranking contains only 26 rows and selects 15 statistical features; the saved rules likewise contain only statistical features. This demonstrates that the completed final-test code inserted Med values into frames but did not append them to the final predictor list. Therefore:

- the final metrics are valid descriptions of the preserved artifact execution;
- they are **not** a faithful evaluation of the locked 36-feature fusion candidate;
- the historical artifacts must remain unchanged;
- the corrected active code must not be used to rerun the already-consumed official test split.

The provenance chain says development selection loaded no test features, OOF generation accessed zero test rows, the final Med fit did not use test labels for fitting/selection, and the final evaluation count was one. The schema defect is configuration fidelity, not evidence of test-label selection.

## Verification status

- PASS: Python syntax compilation of the three changed source modules.
- PASS: JSON/CSV parsing, row-count/confusion-matrix agreement, artifact hashes, and byte-for-byte copy comparison.
- PASS: static checks of ignored package/archive/output/cache paths and absence of hard-coded Colab/Mac paths in active source changes.
- BLOCKED LOCALLY: pytest, import smoke tests, model load, and runtime schema tests because no existing local interpreter has the lightweight project dependencies. A new environment was not installed because disk is critically low and the task forbids unnecessary environment expansion.
- COLAB-ONLY: actual Med-MICN training, GPU DataLoader behavior, and all expensive V2 experiments.

## Remaining limitations

- Patient IDs are unavailable/incomplete; grouping therefore protects augmentation lineage but cannot prove patient-level separation.
- The completed final-test result has the 26-versus-36 schema defect above.
- Historical headline results were selected in a workflow that evaluated many candidates on test, so they are descriptive rather than a clean untouched-test estimate.
- The existing official test has already been evaluated and must not be used again for V2 selection or tuning. A genuinely untouched external cohort is required for another final generalization estimate.
