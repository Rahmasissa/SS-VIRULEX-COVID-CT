# Verified Current Pipeline Findings

## Evidence and scope

This record follows the requested hierarchy: executable workspace code, then on-disk artifacts, then the supplied Med-MICN and SVIS-RULEX papers, then documentation. The papers were rendered and visually checked, but they are not used to fill gaps in the implementation.

The current executable entry points are:

- `03_SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py`: reconstructed fusion, rule-model, calibration, threshold, and output runner.
- `03_SS-VIRULEX-outputs/02_SVIS-RULEX/covid_ct_run/exact_sequence/scripts/common.py`: official split loading, augmentation, MobileNetV2 training, 128-dimensional feature extraction, 26 statistical descriptors, and original ZFMIS.
- `03_SS-VIRULEX-outputs/01_Med-MICN/Med_MICN_COVID_CT_training.ipynb`: the only executable record of the saved Med-MICN training procedure.

No pre-existing automated tests were present. Before edits, the runner's `--help` command failed because `numpy` was unavailable in the active interpreter. It passed after a project-local test environment was created. No original source or artifact was modified.

## Verified representation and split

The saved fusion files contain 36 predictors plus `label`:

- 26 scalar statistics calculated over each 128-dimensional SVIS feature vector;
- 8 ordered Med-MICN concept probabilities;
- `med_task_prob_covid`;
- `med_neural_prob_covid`.

Therefore the implemented feature vector is exactly `[S_i || C_i || p_task,i || p_neural,i]` with 36 predictors. The saved train/validation/test matrices are respectively `936 x 37`, `118 x 37`, and `203 x 37` including the label column.

The code reads the official COVID-CT text files before augmentation. The original-image counts are:

| Split | COVID | NonCOVID | Total |
|---|---:|---:|---:|
| Train | 191 | 234 | 425 |
| Validation | 60 | 58 | 118 |
| Test | 98 | 105 | 203 |

Training-only augmentation adds 277 COVID and 234 NonCOVID rows, yielding 936 training rows. Validation and test remain original-only. The saved official and augmented manifests agree on every original-image split assignment.

Neither the official split files nor the supplied concept CSV contains a reliable patient identifier or publication/source identifier. Patient-level and publication-level isolation are therefore **not verifiable**. Filenames were not treated as patient identities.

## Statistical/SVIS branch

| Item | Executable behavior |
|---|---|
| Backbone | Keras MobileNetV2, ImageNet `no_top` weights, global average pooling, frozen |
| Image input | RGB, `224 x 224`, Keras MobileNetV2 `preprocess_input` |
| Trainable model | Dense/BatchNorm/Dropout head ending in a 128-unit `xai_feature_layer` and 2-way softmax |
| Selection criterion | Validation balanced accuracy across the dense-head grid |
| Training controls | Class weights; early stopping on validation loss with best-weight restoration; reduce-on-plateau |
| Statistical output | 26 deterministic descriptors over the 128 activations |
| Augmentation | Train only: horizontal mirror probability 0.55, rotation approximately +/-7 degrees, scale 0.94-1.06, brightness/contrast perturbation |

The horizontal mirror is implemented but lacks a recorded clinical justification. It remains part of the preserved baseline; the new conservative Med-MICN configurations disable flipping.

## Med-MICN branch

| Item | Executable behavior |
|---|---|
| Backbone | Torchvision ResNet50 with ImageNet weights and its 1000-class output used as the backbone feature vector |
| Heads | 8-unit concept embedding/probability head, 2-way task head, 2-way neural-symbolic probability branch |
| Input | PIL RGB; resize shorter side to 256; center crop 224; ImageNet normalization |
| Baseline augmentation | None; train and evaluation transforms are identical |
| Fine-tuning | `FREEZE_BACKBONE=False`; the entire backbone and heads are updated at one AdamW learning rate |
| Loss | task cross-entropy + 0.1 concept BCE + 0.1 neural BCE; concept-task loss weight is 0 |
| Schedule | 20 epochs; best validation task macro-F1 checkpoint saved, but training does not stop early |

The saved history runs all 20 epochs. Its best validation task macro-F1 is approximately `0.7541` at epoch 18 while training macro-F1 is approximately `0.9929`, indicating substantial overfitting; it does not prove that another fine-tuning policy is better.

The completed concept CSV supplies eight binary labels, but the workspace contains no executable GPT-4V/BioVIL labeling pipeline, annotator record, adjudication record, or source-level provenance for those labels. Their semantic provenance is therefore unverified. The labels are consumed as ground truth by the notebook.

The repository working-tree checkpoint path is absent and the Git object is an LFS pointer. A 320,122,217-byte checkpoint exists at `/Users/samehissa/Downloads/SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt`; its SHA-256, `9a71d31680f88560b91742c835171f4dd837f507fe8696dbd44bc843aaf8964b`, exactly matches the LFS object ID. Read-only loading verified top-level model and optimizer states, epoch 18, best validation F1 `0.7540783327`, and a config containing only RN50, the ordered eight concept columns/names, and embedding size 8. It does not record preprocessing, augmentation, freezing, optimizer/LRs, loss weights, or seed. The external file must be passed explicitly or restored with Git LFS for archived inference. The checked-in `trained_model_manifest.json` contains stale paths and incorrectly says the repository copy exists.

## Existing feature provenance

The runner loads one best checkpoint and infers all 1,257 manifest rows with it. The notebook trains that checkpoint on the official training images. Consequently the original training fusion rows are in-sample predictions from a model trained on those same original training images. Augmented training rows also descend from those originals. The probability CSV contains no fold identifier or per-row checkpoint identifier, so it is not OOF-protected and cannot establish excluded-fold provenance.

## Downstream selection, calibration, and test use

The verified subset hierarchy is `[18, 15, 12, 9, 6, 3]`.

The old runner fits zero-fraction screening and mutual-information selection once on all 936 training rows, saves the selected columns, and then gives those already-selected matrices to downstream grid searches. Inner validation folds therefore influenced the selected feature set.

The old calibration stage fits sigmoid calibration on the validation labels and then chooses a balanced-accuracy threshold using the same validation predictions/labels. The old runner also evaluates multiple feature sets and models on test and sorts several result tables by `test_balanced_accuracy`; final rule extraction takes the top test-ranked row. The archived test metrics are descriptive of that historical workflow, not a valid untouched final estimate after model selection.

## Code/paper/document conflicts

- The SVIS-RULEX paper describes subject-exclusive 70/15/15 partitioning. The code uses the official COVID-CT 425/118/203 image lists and has no subject IDs.
- The paper's augmentation includes broader transforms (including both-axis flips, fog, and RGB shifts); the code implements the narrower transform listed above.
- The paper describes an ROI/cropping workflow that is not present in the executable branch.
- The Med-MICN paper describes experiments including ResNet50/VGG19/DenseNet169, 256-sized inputs, Adam at `5e-5`, 100 epochs, and an 80/20 protocol. This workspace uses ResNet50, resize-256 then crop-224, AdamW, 20 epochs, and official train/validation/test lists.
- Workspace documentation describes the intended artifact cache but does not supersede these executable behaviors.

## Reproducibility risks outside the algorithms

The pre-existing Git index contains 1,142 staged deletions and the corresponding files appear untracked in the working tree. These changes predate this work. Nothing was reset, staged, deleted, or overwritten. All reliability additions are isolated under `04_reliability/`.
