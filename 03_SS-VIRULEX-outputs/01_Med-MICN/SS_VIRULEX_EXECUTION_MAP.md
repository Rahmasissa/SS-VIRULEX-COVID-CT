# SS-VIRULEX Execution Map

This map is based on the local folders:

- `/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main 2`
- `/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main`

The uploaded diagram combines the SVIS-RULEX statistical pipeline with a Med-MICN concept branch. In SS-VIRULEX, Med-MICN's **Concept Embedding Module** and **Neural Symbolic Layer** are required components, not optional references. The GPT-4V concept vocabulary generation and BioVIL concept alignment/scoring scripts shown in the diagram are not present as runnable files, so the current workspace uses the already completed concept table at `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv` as that branch's materialized semantic-alignment output.

The pipeline now also exports the consolidated SS-VIRULEX outputs to one folder outside both source repositories:

`/Users/samehissa/Downloads/SS-VIRULEX-outputs`

Within that consolidated output root, `02_SVIS-RULEX` is the trained SVIS-RULEX
artifact cache consumed by SS-VIRULEX. It contains the custom MobileNetV2 model,
statistical feature CSVs, ZFMIS feature sets, and stat-only rule baselines used
as the statistical branch.

## Summary Table

| Screenshot stage | Workspace implementation | Notes |
|---|---|---|
| 1. Input and Pre-processing | `XAI-Med-Images-Stat-Visual-Rules-main 2/covid_ct_run/exact_sequence/notebooks/01_image_augmentation_code_covid_ct.ipynb`; functions in `covid_ct_run/exact_sequence/scripts/common.py` | Reads COVID-CT official split text files, validates image paths, creates augmented training images and split manifests. |
| 2. Deep Feature Extraction, MobileNetV2 | `02_SVIS-RULEX/covid_ct_run/exact_sequence/notebooks/02_dl_model_hyperparameter_optimization_using_grid_search_covid_ct.ipynb`, `03_custom_mobilenetv2_complete_code_covid_ct.ipynb`; functions in `common.py` | Uses the trained SVIS-RULEX artifact root by default, saving Keras models and deep feature arrays there. |
| 3a. Statistical Feature Engineering, SVIS-RULEX | `02_SVIS-RULEX/covid_ct_run/exact_sequence/notebooks/04_statistical_features_extraction_and_feature_selection_using_zfmis_covid_ct.ipynb`; `calculate_statistical_features()` and `run_statistical_features_zfmis()` in `common.py` | Computes 26 metrics over the learned feature vector and writes train/val/test CSVs into the trained SVIS-RULEX artifact root. |
| 3b. Concept Semantic Alignment, Med-MICN | `NeurIPS24-Med_MICN-main/Med_MICN_COVID_CT_training.ipynb`; `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv` | Consumes completed concept labels/scores. No GPT-4V/BioVIL generation script is present. |
| 3c. Concept Embedding Module, Med-MICN | `NeurIPS24-Med_MICN-main/models.py`; `torch_explain/nn/concepts.py`; `03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt` | Required SS-VIRULEX component. The runner validates `ConceptEmbedding` from the source repo and consumes the trained checkpoint from the SS-VIRULEX output artifact root on every run. |
| 3d. Neural Symbolic Layer, Med-MICN | `torch_explain/nn/concepts.py::ConceptReasoningLayer`; `03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json`; `test_metrics.csv` | Required SS-VIRULEX component. The runner validates neural-symbolic metrics such as `neural_accuracy` and concept metrics from the trained Med-MICN artifact folder before fusion. |
| 4. ZFMIS Feature Selection, augmented | Existing stat-only ZFMIS is in `common.py`; integrated stat-plus-concept ZFMIS is implemented in `run_ss_virulex_pipeline.py` | The runner fuses `X_stat` with Med-MICN concept probabilities regenerated from `best_model.pt`, then runs zero filtering and mutual information selection. |
| 5. Rule-based Modeling, augmented | Existing stat-only notebooks: `05a_gridsearchfortree_covid_ct.ipynb`, `05b_rulefitgridsearchcode_covid_ct.ipynb`, `06_rule_extraction_code_covid_ct.ipynb`; integrated stat-plus-concept rule modeling is in `run_ss_virulex_pipeline.py` | Decision Tree and RuleFit are trained on selected combined features. |
| 6. SFMOV Visualization, augmented | Existing stat-only SFMOV: `07_sfmov_heatmaps_covid_ct.ipynb`; concept-aware heatmap metadata and overlays are in `run_ss_virulex_pipeline.py` | Uses Med-MICN concept scores as alpha/salience values after required Med-MICN component validation. |

## Execution Map

| Step | Pipeline stage from screenshot | Source file/script path | Input required | Output generated |
|---:|---|---|---|---|
| 1 | Input and Pre-processing | `XAI-Med-Images-Stat-Visual-Rules-main 2/covid_ct_run/exact_sequence/notebooks/01_image_augmentation_code_covid_ct.ipynb`; `common.py::build_official_split_table`, `common.py::create_augmented_manifest` | `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Data-split/**`; `/Users/samehissa/Downloads/COVID-CT-Dataset-master/Images-processed/**` | `outputs/covid_ct_official_splits.csv`, `outputs/01_augmented_train_manifest.csv`, `outputs/01_augmented_splits.csv`, `outputs/augmented_train_images/**` |
| 2 | Deep Feature Extraction and DL grid search | `02_dl_model_hyperparameter_optimization_using_grid_search_covid_ct.ipynb`; `common.py::run_dl_grid_search` | `outputs/01_augmented_splits.csv`; cached MobileNetV2 weights under `~/.keras/models` | `outputs/features/02_base_features_augmented.npz`, `outputs/02_dl_grid_search_results.csv`, `outputs/02_best_dl_hyperparameters.json` |
| 3 | Final custom MobileNetV2 training | `03_custom_mobilenetv2_complete_code_covid_ct.ipynb`; `common.py::train_final_custom_mobilenet` | `outputs/features/02_base_features_augmented.npz`; `outputs/02_best_dl_hyperparameters.json` | `outputs/models/03_custom_mobilenetv2_head.keras`, `outputs/models/03_custom_mobilenetv2_complete_covid_ct.keras`, `outputs/03_custom_mobilenetv2_training_history.csv`, `outputs/03_custom_mobilenetv2_test_metrics.json` |
| 4 | Statistical Feature Engineering, SVIS-RULEX | `04_statistical_features_extraction_and_feature_selection_using_zfmis_covid_ct.ipynb`; `common.py::run_statistical_features_zfmis` | `outputs/models/03_custom_mobilenetv2_complete_covid_ct.keras`; `outputs/01_augmented_splits.csv` | `outputs/features/04_xai_deep_features_custom_mobilenetv2.npz`, `outputs/features/04_train_statistical_features.csv`, `04_val_statistical_features.csv`, `04_test_statistical_features.csv`, `04_zfmis_feature_sets.json` |
| 5 | Concept Semantic Alignment, Med-MICN | `NeurIPS24-Med_MICN-main/Med_MICN_COVID_CT_training.ipynb`; completed concept CSV | `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv`; COVID-CT image paths | Concept labels/scores for COVID CT images |
| 6 | Concept Embedding Module, Med-MICN | `models.py`; `torch_explain/nn/concepts.py::ConceptEmbedding`; `03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt` | Concept CSV; COVID-CT images; trained Med-MICN checkpoint | Validated concept embedding model artifact and concept prediction metrics |
| 7 | Neural Symbolic Layer, Med-MICN | `torch_explain/nn/concepts.py::ConceptReasoningLayer`; `03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json` | Concept embeddings and predicted concept scores from Med-MICN | Validated neural-symbolic metrics/rules summary in `outputs/05_med_micn_required_components.json` |
| 8 | Per-image Med-MICN concept probabilities | `03_SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py::generate_med_micn_concept_probabilities`; `03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt` | Augmented split manifest image paths; trained Med-MICN checkpoint | `outputs/covid_ct_med_micn/per_image_concept_probabilities.csv`; `outputs/features/04_med_micn_concept_probabilities.csv`; `concept_branch/probabilities/covid_concept_probabilities.csv` |
| 9 | Combined Feature Pool and augmented ZFMIS | `03_SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py` | `02_SVIS-RULEX/.../outputs/features/04_*_statistical_features.csv`; `02_SVIS-RULEX/.../outputs/01_augmented_splits.csv`; Med-MICN per-image concept probabilities | `outputs/features/04_*_combined_stat_concept_features.csv`, `outputs/features/04_combined_zfmis_feature_sets.json`, selected combined feature folders |
| 10 | Rule-based Modeling, augmented | `03_SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py`; existing patterns from `05a` and `05b` notebooks | Combined feature CSVs and combined ZFMIS feature-set JSON | `outputs/05c_combined_gridsearchfortree_results.csv`, `outputs/05d_combined_rulefitgridsearchcode_results.csv`, combined model pickles and rule files |
| 11 | SFMOV Visualization, augmented | Existing SFMOV code in `common.py::run_sfmov_heatmaps`; concept-aware extension in `run_ss_virulex_pipeline.py` | `03_custom_mobilenetv2_complete_covid_ct.keras`; official test split; Med-MICN per-image concept probabilities | `outputs/heatmaps/**`, `outputs/heatmaps/concept_aware/**`, `outputs/07_concept_aware_sfmov_heatmap_files.json` |
| 12 | Final summary | `run_ss_virulex_pipeline.py` | Metrics/rules/heatmap outputs from prior stages | `outputs/ss_virulex_pipeline_summary.json`, `outputs/ss_virulex_final_metrics_summary.csv` |
| 13 | Consolidated SS-VIRULEX export | `run_ss_virulex_pipeline.py::export_consolidated_outputs` | SS-VIRULEX summaries, features, rules, models, heatmaps, and logs | `/Users/samehissa/Downloads/SS-VIRULEX-outputs/export_manifest.json` plus `summaries/`, `features/`, `rules/`, `models/`, `heatmaps/`, `logs/` |
