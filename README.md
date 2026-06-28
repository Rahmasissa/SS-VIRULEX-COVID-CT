# SS-VIRULEX Workspace

Open this folder in VS Code to see the project as one workspace root.

## Folders

- `01_Med-MICN`: Med-MICN source repository.
- `02_SVIS-RULEX`: SVIS-RULEX / XAI source repository.
- `03_SS-VIRULEX-outputs`: current COVID SS-VIRULEX artifacts and organized outputs.

These entries are symlinks to the existing folders, so the original paths and pipeline scripts continue to work.

## Output Layout

`03_SS-VIRULEX-outputs` keeps the restored trained-artifact folders and the organized output files generated from them:

- `01_Med-MICN`: trained Med-MICN checkpoint and metrics consumed by SS-VIRULEX.
- `02_SVIS-RULEX`: trained SVIS-RULEX model, statistical features, ZFMIS, and rule artifacts consumed by SS-VIRULEX.
- `dataset`
- `statistical_branch`
- `concept_branch`
- `fusion`: statistical features fused with Med-MICN checkpoint-generated concept probabilities.
- `zfmis`
- `final_results`
- `visual_explanations`
