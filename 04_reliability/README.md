# SS-VIRULEX Reliability Layer

This additive module preserves the original pipeline while providing leakage audits, OOF Med-MICN features, reproducible encoder comparisons, and fold-safe downstream selection.

Read in order:

1. `CURRENT_PIPELINE_FINDINGS.md`
2. `DATA_LEAKAGE_AUDIT.md`
3. `IMPLEMENTATION_CHANGES.md`
4. `SMOKE_TEST_RESULTS.md`
5. `EXPERIMENT_COMMANDS.md`
6. `FINAL_COMPARISON_TEMPLATE.md`

Colab integration evidence and the next controlled experiment plan are under
`../docs/reliability_integration/`. The preserved completed-run artifacts and
their SHA-256 manifest are under `../results/reliability_colab/final_test/`.

Safety contract:

- Original artifacts are never silently overwritten.
- Patient identity is not inferred from filenames.
- Augmentation lineage is the fallback grouping boundary.
- Development OOF and final-test inference are separate commands.
- Test actions require explicit acknowledgement and locked IDs.
- Smoke outputs are marked `scientific_result: false`.
- RuleFit is reported only when the actual dependency is available.
