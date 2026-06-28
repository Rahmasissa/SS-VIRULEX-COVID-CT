# Chapter 2 Regeneration Guide

## Executive Summary

### Purpose of This Guide

This file is a blueprint for regenerating Chapter 2, not a draft of Chapter 2 itself. Claude should use it to write an implementation-aligned background and literature-review chapter without needing to inspect the workspace.

The primary source of truth is the exported COVID-CT SS-VIRULEX run under:

`/Users/samehissa/Downloads/SS-VIRULEX-outputs`

The current implementation is not an exact monolithic merge of the original Med-MICN and SVIS-RULEX pipelines. It is an evolved late-fusion system that validates and uses executable components from both lineages, then adds new fusion, predictive modelling, calibration, threshold-selection, visualization, and report-export stages.

### Current Architecture in One Paragraph

The implemented COVID-CT SS-VIRULEX runtime has two distinct evidence-producing branches. The SVIS-RULEX branch uses a MobileNetV2-based image pipeline, extracts a learned `xai_feature_layer` representation, and computes 26 statistical descriptors. The Med-MICN branch loads an `RN50` checkpoint, produces eight clinically named concept probabilities, and emits both a task probability and a neural-symbolic reasoning probability for COVID-19. These outputs are combined into a 36-feature late-fusion candidate space: 26 statistical descriptors, eight concept scores, and two Med-MICN diagnostic probabilities. Augmented ZFMIS performs sparse-value filtering and mutual-information ranking. Transparent Decision Tree and RuleFit models provide readable explanatory rules, while a broader classifier family and calibrated validation-threshold variants provide the final predictive comparison. The visual layer produces concept-aware SFMOV overlays by modulating statistical heatmaps with a scalar derived from concept scores, then combines prediction, rule-path, heatmap, and top-concept information into exported explanation artifacts.

### Major Architectural Additions

- Separate SVIS-RULEX and Med-MICN runtime branches with explicit provenance rather than a single shared representation.
- Eight Med-MICN concept probabilities inferred from an `RN50` checkpoint.
- Two additional Med-MICN diagnostic-probability features:
  - `med_task_prob_covid`
  - `med_neural_prob_covid`
- A fused 36-feature candidate evidence space.
- Combined ZFMIS over statistical, semantic, and model-derived diagnostic evidence.
- Stronger downstream classifier evaluation:
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Histogram Gradient Boosting
  - RBF SVM
  - Logistic Regression
- Probability calibration and validation-set threshold selection.
- A distinction between transparent explanatory models and the final best predictive classifier.
- Concept-aware SFMOV overlays and combined explanation panels.
- Export manifests, result summaries, and an implementation report for the reconstructed pipeline.

### Major Architectural Removals or Corrections

- Remove the old claim that one MobileNetV2 feature vector `F` is the shared representational substrate for both branches.
- Remove any claim that GPT-4V or BioViL is called during the current exported runtime.
- Remove any claim that the current COVID-CT run produces concept-specific Grad-CAM images.
- Remove any claim that a named concept is spatially localized by the current concept-aware SFMOV overlay.
- Remove any ROI-cropping stage unless Chapter 3 introduces and evidences a new implementation after this guide was generated.
- Remove the claim that the fused final predictor is itself a rule-based model.
- Remove any implication that statistical descriptors are injected into the internal Med-MICN neural-symbolic reasoning layer. The branches remain distinct until late fusion.

### Important Terminology Updates

| Old or ambiguous wording | Required current wording | Reason |
|---|---|---|
| Shared feature vector `F` | Late-fused evidence space | The implemented branches use different model artifacts and representations. |
| Concept scores only | Med-MICN concept probabilities plus Med-MICN diagnostic probabilities | The fused space includes eight semantic scores and two diagnostic outputs. |
| ZFMIS | Augmented or combined ZFMIS when referring to the fused pipeline | Current feature selection is applied after multi-source fusion. |
| Concept heatmap | Scalar concept-aware SFMOV overlay | Current overlays are statistical SFMOV maps modulated by a concept-derived scalar, not per-concept localization maps. |
| Concept Grad-CAM | Reserved output category, not produced in the exported COVID-CT run | The output tree contains a placeholder README rather than generated concept Grad-CAM PNGs. |
| Neural-symbolic rules enriched by statistics | Med-MICN neural-symbolic probability fused with statistical and semantic evidence | Statistics do not enter the internal Med-MICN fuzzy reasoning layer in the current runner. |
| Current GPT-4V/BioViL alignment stage | Materialized concept annotation input inherited from the Med-MICN workflow | The runner checks for runnable stages and reports that they are not present. |
| Rule-based final prediction | Parallel explanatory rule models and predictive classifier family | The best exported predictor is a calibrated thresholded logistic-regression configuration. |

### Architecture Verdict

The correct answer to the central comparison question is:

> The current implementation preserves the conceptual ancestry of Med-MICN and SVIS-RULEX, but it no longer follows the original proposed hybrid design exactly. It is an additive, execution-aware late-fusion architecture with separate branch provenance, augmented feature selection, model-derived diagnostic-probability fusion, calibrated prediction, threshold optimization, and layered explanation exports.

Chapter 2 must establish the background concepts needed to understand that architecture. It must not present the detailed execution sequence, hyperparameters, output metrics, or implementation procedure that belong in Chapters 3 and 4.

## Source-of-Truth Registry

Use these aliases consistently when reading this guide. Each alias expands to an exact workspace location.

| Alias | Exact path | Role |
|---|---|---|
| `PRIMARY_OUTPUT_ROOT` | `/Users/samehissa/Downloads/SS-VIRULEX-outputs` | Primary exported COVID-CT SS-VIRULEX run and the main source of truth for the thesis implementation. |
| `PRIMARY_RUNNER` | `/Users/samehissa/Downloads/SS-VIRULEX-outputs/01_Med-MICN/run_ss_virulex_pipeline.py` | Reconstructed master runtime showing the implemented stage sequence. |
| `SVIS_RUNTIME_SOURCE` | `/Users/samehissa/Downloads/SS-VIRULEX-outputs/02_SVIS-RULEX/covid_ct_run/exact_sequence/scripts/common.py` | Executable SVIS-RULEX functions used by the COVID-CT run. |
| `MED_SOURCE_ROOT` | `/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main` | Original Med-MICN code lineage and the current writable repository. |
| `UNIFIED_WORKSPACE` | `/Users/samehissa/Downloads/SS-VIRULEX-workspace` | Workspace index linking the Med-MICN source, SVIS-RULEX source, and exported outputs. |
| `WORKSPACE_INDEX` | `/Users/samehissa/Downloads/SS-VIRULEX-workspace/README.md` | Unified-workspace documentation and linked-root layout. |
| `EXECUTION_MAP` | `/Users/samehissa/Downloads/SS_VIRULEX_EXECUTION_MAP.md` | Pipeline execution map for the integrated COVID-CT workflow. |
| `EXPORTED_EXECUTION_MAP` | `/Users/samehissa/Downloads/SS-VIRULEX-outputs/01_Med-MICN/SS_VIRULEX_EXECUTION_MAP.md` | Exported copy of the execution map stored beside the primary runtime. |
| `OLD_CHAPTER_DRAFT` | `/Users/samehissa/Downloads/THESIS/thesis draft/SS-VIRULEX Chapter 1 & 2.docx` | Old Chapter 2 draft used for comparison. |
| `OLD_METHODOLOGY_OUTLINE` | `/Users/samehissa/Downloads/Chapter_3_Methodology_Outline.md` | Existing Chapter 3 boundary reference. |
| `PROPOSED_PIPELINE_ASSETS` | `/Users/samehissa/Downloads/THESIS/proposed pipeline` | Earlier pipeline figures. Treat them as reference material requiring redraw and validation. |
| `DDI_COMPANION_ROOT` | `/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main` | Separate DDI companion implementation. Do not silently merge its disease-specific behavior into the COVID-CT thesis run. |

### Primary COVID-CT Evidence Files

These files should be treated as the first-line implementation evidence:

- `PRIMARY_RUNNER`
- `SVIS_RUNTIME_SOURCE`
- `WORKSPACE_INDEX`
- `EXECUTION_MAP`
- `EXPORTED_EXECUTION_MAP`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_pipeline_summary.json`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_final_metrics_summary.csv`
- `PRIMARY_OUTPUT_ROOT/features/04_combined_zfmis_feature_sets.json`
- `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv`
- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_decision_tree_rules.txt`
- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_rulefit_rules.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/06_combined_rule_extraction_summary.json`
- `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json`
- `PRIMARY_OUTPUT_ROOT/current_output_layout_manifest.json`
- `PRIMARY_OUTPUT_ROOT/export_manifest.json`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md`

### Med-MICN Lineage Evidence Files

- `MED_SOURCE_ROOT/README.md`
- `MED_SOURCE_ROOT/models.py`
- `MED_SOURCE_ROOT/torch_explain/nn/concepts.py`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/Med_MICN_COVID_CT_training.ipynb`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/concept_metrics.csv`
- `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv`

### Separate DDI Companion Files

These files may help explain the wider research trajectory, but they are not evidence that a feature exists in the exported COVID-CT run:

- `DDI_COMPANION_ROOT/SS_VIRULEX_CURRENT_IMPLEMENTATION_MAP.md`
- `DDI_COMPANION_ROOT/SS_VIRULEX_METHODOLOGY_BRIEF_FOR_CLAUDE.md`
- `DDI_COMPANION_ROOT/SS_VIRULEX_RUN_AND_IMPLEMENTATION_SUMMARY.md`
- `DDI_COMPANION_ROOT/docs/SS_VIRULEX_Testing_and_Results_Report.md`
- `DDI_COMPANION_ROOT/SS_VIRULEX_OUTPUT_LAYOUT.md`
- `DDI_COMPANION_ROOT/ss_virulex/`

### Source-of-Truth Rule for Claude

When sources conflict, use this order:

1. The exported COVID-CT runner and its generated artifacts under `PRIMARY_OUTPUT_ROOT`.
2. The executable source modules invoked by that runner.
3. The Med-MICN lineage source and checkpoint artifacts.
4. The old thesis draft and earlier diagrams only as historical material.
5. The DDI companion implementation only as explicitly labelled companion context.

Do not import DDI-specific lesion-mask, salience, or visualization behavior into the COVID-CT Chapter 2 description unless the final thesis scope is deliberately broadened and Chapter 3 is updated to match.

## Implemented Pipeline Snapshot

Use this as the implementation-alignment checklist while writing Chapter 2. Most exact procedural details belong in Chapter 3; they are recorded here so that Chapter 2 does not contradict them.

### Branch A: SVIS-RULEX Statistical and Visual Branch

- Official COVID-CT split metadata is loaded.
- Training data are augmented; validation and test data are not augmented.
- A cached ImageNet MobileNetV2 base is used with a learned head.
- A 128-dimensional `xai_feature_layer` activation is extracted.
- Twenty-six statistical descriptors are computed from that learned representation.
- Base statistical ZFMIS filtering and ranking are available.
- Base Decision Tree, RuleFit, and SFMOV components are validated.

### Branch B: Med-MICN Semantic and Neural-Symbolic Branch

- A trained Med-MICN checkpoint is loaded.
- The checkpoint configuration identifies an `RN50` backbone and eight COVID-CT concepts.
- The eight concept names are:
  - peripheral ground glass opacities
  - bilateral involvement
  - multilobar distribution
  - crazy paving pattern
  - absence of lobar consolidation
  - localized or diffuse presentation
  - increased density in the lung
  - ground glass appearance
- Per-image outputs include:
  - eight concept probabilities
  - one task probability
  - one neural-symbolic reasoning probability
- The completed concept CSV is treated as a materialized semantic-alignment artifact.
- GPT-4V and BioViL are part of the Med-MICN architectural genealogy, but runnable GPT-4V and BioViL stages are not present in the reconstructed exported runtime.

### Late Fusion and Augmented ZFMIS

- The fused candidate space has 36 columns:
  - 26 statistical descriptors
  - eight Med-MICN concept probabilities
  - two Med-MICN diagnostic probabilities
- Sparse-value filtering removes features whose zero fraction exceeds the configured threshold.
- Mutual-information ranking is applied after filtering.
- Exported selected feature-set sizes are:
  - 18
  - 15
  - 12
  - 9
  - 6
  - 3
- The selected nine-feature configuration demonstrates the intended cross-source mixture:
  - semantic concept probabilities
  - entropy-related statistical descriptors
  - both Med-MICN diagnostic probabilities

### Explanatory and Predictive Layers

- Decision Tree and RuleFit models remain the readable rule-extraction layer.
- A broader classifier family is evaluated for predictive performance.
- Calibration is applied to trained estimators using validation data.
- A fixed threshold selected on validation balanced accuracy is applied to test predictions.
- The final predictive layer and the readable rule layer must be described as complementary, not identical.

### Visual Explanation Layer

- SFMOV constructs statistical maps using mean, skewness, and entropy-related weights.
- The concept-aware extension derives a scalar `concept_alpha` from concept scores.
- That scalar modulates the statistical SFMOV heatmap.
- Combined explanation exports pair:
  - final fused prediction
  - readable Decision Tree path
  - concept-aware heatmap
  - top Med-MICN concept probabilities
- The current COVID-CT export does not contain concept-specific Grad-CAM PNGs.

## Explicit Mismatch Register

Claude must resolve every mismatch below when regenerating Chapter 2.

| Old outline or draft claim | Current evidence | Required regeneration action |
|---|---|---|
| A fine-tuned MobileNetV2 feature vector `F` is the shared substrate for both branches. | `PRIMARY_RUNNER`, `SVIS_RUNTIME_SOURCE`, the Med-MICN checkpoint configuration, and `MED_SOURCE_ROOT/models.py` show separate MobileNetV2 and `RN50` branch paths. | Remove the shared-vector statement. Introduce separate branch provenance and late fusion. |
| Med-MICN concept embeddings and the statistical pipeline operate on one runtime feature source. | Med-MICN inference is loaded from its own trained checkpoint, while SVIS descriptors are extracted from the MobileNetV2 `xai_feature_layer`. | Replace with a two-branch architecture explanation. |
| Feature selection is statistical-only ZFMIS. | `PRIMARY_OUTPUT_ROOT/features/04_combined_zfmis_feature_sets.json` and `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv` show combined selection over 26 + 8 + 2 inputs. | Retain the original ZFMIS concept, then add combined or augmented ZFMIS. |
| The rule layer receives only statistical features or only concept-enriched scores. | The readable rules operate over selected fused evidence and may include statistical, semantic, and diagnostic-probability inputs. | Describe a mixed fused rule space. |
| Rule models are the final SS-VIRULEX predictor. | `PRIMARY_OUTPUT_ROOT/summaries/06_combined_rule_extraction_summary.json` and `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_final_metrics_summary.csv` show stronger and calibrated predictive variants. | Distinguish explanatory rules from the final predictive comparison layer. |
| GPT-4V is executed inside the current pipeline. | `PRIMARY_RUNNER` explicitly checks for runnable GPT-4V/BioViL stages and reports that the completed concept CSV is used as a materialized output. | Keep GPT-4V in literature and Med-MICN genealogy only. Do not describe it as a current runtime call. |
| BioViL directly aligns every image with concepts during the exported runtime. | No runnable BioViL stage is present in `PRIMARY_RUNNER`; semantic supervision is materialized in `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv`. | Keep BioViL as architectural ancestry and background context. |
| Current visual explanations are concept-specific saliency maps. | `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json` records scalar `concept_alpha` values; `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md` records that concept Grad-CAM PNGs were not saved. | Use the precise term `scalar concept-aware SFMOV overlay`. |
| A concept-aware overlay localizes each named concept. | The implemented modulation is a scalar applied to SFMOV maps rather than a per-concept spatial localization method. | Remove localization overclaims. State the visual scope conservatively. |
| Neural-symbolic rules are directly enriched with statistical descriptors. | The Med-MICN neural-symbolic output is generated within the Med-MICN branch; its diagnostic probability is subsequently fused with statistical descriptors. | Explain post-hoc probability fusion rather than internal rule enrichment. |
| ROI cropping is an implemented preprocessing stage. | The validated SVIS runtime source contains augmentation and split handling, but no evidenced ROI-crop stage. | Remove ROI-crop claims from Chapter 2 and any as-built figure. |
| The architecture consists only of a statistical branch and a semantic branch. | The current fusion layer also receives task and neural-symbolic diagnostic probabilities. | Add a model-derived diagnostic-evidence category. |
| Threshold choice is a fixed default or is not conceptually important. | The exported runtime performs validation-based threshold optimization after calibration. | Add calibration and operating-threshold background. Keep exact procedure in Chapter 3. |
| The visual explanation inventory includes generated concept Grad-CAM images. | The output category exists only as a placeholder README in the exported COVID-CT run. | Do not claim generated concept Grad-CAM results. |
| Earlier SVG diagrams are accurate as-built pipeline figures. | Earlier files in `PROPOSED_PIPELINE_ASSETS` retain assumptions such as shared `F`, direct GPT-4V/BioViL runtime stages, or ROI cropping. | Redraw before reuse. Label old assets as conceptual references only. |
| DDI companion visual modules can be described as part of the COVID-CT run. | `DDI_COMPANION_ROOT` is a separate implementation lineage with its own reports and output behavior. | Keep COVID-CT and DDI descriptions explicitly separated. |

## Old Outline Disposition

This table maps every visible old Chapter 2 outline item to its correct treatment.

| Old outline item | Disposition | Placement in regenerated chapter |
|---|---|---|
| Deep Learning and Feature Extraction for Medical Image Analysis | Keep and update | `2.1.1` |
| Statistical Feature Engineering | Keep and update | `2.1.2` |
| Feature Selection: Zero-based Filtering with Mutual Importance Selection (ZFMIS) | Keep and expand | `2.1.3` |
| Rule-based Interpretability: Decision Trees and RuleFit | Keep and update | `2.1.4` |
| Statistical Feature Map Overlay Visualisation (SFMOV) | Keep and narrow claims | `2.1.5` |
| Concept Bottleneck Models (CBMs) | Keep | `2.1.6` |
| Large Multimodal Models (LMMs) for Automated Concept Generation | Keep as literature context only | `2.1.7` |
| Vision-Language Models for Concept Alignment: BioViL | Keep as literature context only | `2.1.7` |
| Neural-Symbolic Reasoning and Fuzzy Logic Rules | Keep and clarify | `2.1.8` |
| Explainability Modalities: A Taxonomy | Keep and expand | `2.1.11` |
| Pillar I: Post-hoc Visualisation Methods | Keep | `2.2.1` |
| Early Gradient-based and Activation Methods | Keep inside Pillar I | `2.2.1` |
| SHAP, LIME, and Instance-level Attribution | Keep inside Pillar I | `2.2.1` |
| Concept-level Post-hoc Methods: TCAV | Keep inside Pillar I | `2.2.1` |
| Pillar II: Rule-based and Statistical Interpretability Frameworks | Keep and expand | `2.2.2` |
| Decision Trees and Surrogate Models | Keep inside Pillar II | `2.2.2` |
| Statistical Feature Engineering in Deep Learning Pipelines | Keep inside Pillar II | `2.2.2` |
| SVIS-RULEX: Architectural Genealogy and Contributions | Keep with inheritance boundaries | `2.2.2` |
| Pillar III: Concept-based Ante-hoc Models | Keep and broaden | `2.2.3` |
| Concept Bottleneck Models and Extensions | Keep inside Pillar III | `2.2.3` |
| Vision-Language Models in Medical Concept Alignment | Keep inside Pillar III | `2.2.3` |
| Med-MICN: Architectural Genealogy and Contributions | Keep with runtime caveats | `2.2.3` |
| Hybrid late fusion of multi-source evidence | Add | `2.1.9`, `2.2.4` |
| Diagnostic-probability fusion | Add | `2.1.9`, `2.2.4` |
| Calibration and validation-based threshold selection | Add | `2.1.10`, `2.2.4` |
| Layered explanation outputs and scope limits | Add | `2.1.11`, `2.3.2` |

## Section-by-Section Mapping

### 2.1 Conceptual Foundations for SS-VIRULEX

#### Purpose

Introduce only the concepts needed to understand the evolved SS-VIRULEX architecture before the literature review. The section should progress from learned image representations, to statistical evidence, to semantic and neural-symbolic evidence, to late fusion, to predictive and explanatory outputs.

#### Keep Unchanged

- Retain the old chapter's broad motivation for interpretable medical-image analysis.
- Retain the distinction between predictive image representations and human-understandable explanation layers.
- Retain the rationale for combining multiple complementary explanation modalities.

#### Update

- Replace any shared-feature-vector narrative with separate branch provenance.
- Explain that current SS-VIRULEX includes statistical, semantic, and model-derived diagnostic evidence.
- Make the predictive-versus-explanatory model distinction explicit.

#### Add

- Add a short orienting paragraph explaining that Chapter 2 supplies conceptual foundations while Chapter 3 presents the implemented sequence.
- Add a compact architecture preview using the terms defined in this guide.

#### Remove

- Remove step-by-step runtime narration.
- Remove metrics, best-model values, and threshold values from Chapter 2.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `EXECUTION_MAP`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_pipeline_summary.json`

#### Suggested Figures

- Figure 2.1: Conceptual overview of the implemented SS-VIRULEX evidence flow.

#### Suggested Tables

- Table 2.1: Terminology and evidence-family overview.

---

### 2.1.1 Deep Learning and Learned Feature Representations for Medical Images

#### Purpose

Explain why convolutional and residual deep networks provide useful learned image representations for downstream classification and explainability pipelines.

#### Keep Unchanged

- Keep the old background discussion of deep learning for medical-image analysis.
- Keep the general explanation that learned features can capture patterns not represented adequately by hand-crafted raw-pixel measurements.
- Keep MobileNetV2 as a relevant lightweight CNN example.

#### Update

- State that the implemented system uses more than one backbone lineage.
- Explain that MobileNetV2 supports the SVIS statistical and visual branch, while the Med-MICN checkpoint configuration identifies an `RN50` backbone.
- Avoid describing either representation as the unique shared source for all downstream evidence.

#### Add

- Add a paragraph introducing branch-specific representation learning.
- Explain why an interpretable hybrid may combine representations produced by separately trained components.

#### Remove

- Remove the claim that a single fine-tuned MobileNetV2 vector `F` is shared by both the statistical and Med-MICN branches.
- Remove ROI-crop claims unless later implementation evidence is added.

#### Evidence From Current Workspace

- `SVIS_RUNTIME_SOURCE`
- `PRIMARY_RUNNER`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt`
- `MED_SOURCE_ROOT/models.py`

#### Suggested Figures

- Figure 2.2: Separate backbone provenance diagram: MobileNetV2 statistical branch and `RN50` Med-MICN branch.

#### Suggested Tables

- Table 2.2: Representation sources and their downstream roles.

---

### 2.1.2 Statistical Feature Engineering Over Learned Representations

#### Purpose

Explain how summary statistics can transform a learned neural representation into a compact, inspectable quantitative profile suitable for feature selection, rule extraction, and statistical visualization.

#### Keep Unchanged

- Keep the old explanations of descriptive statistics, dispersion, entropy, energy, and distribution-shape measures.
- Keep the motivation that statistical summaries provide a bridge between opaque activations and interpretable evidence.

#### Update

- Clarify that the implemented statistics are computed from learned `xai_feature_layer` activations, not directly from raw pixels.
- Use cautious language when relating descriptors to pathology. Statistical values may reflect representation structure; they are not direct clinical measurements unless independently validated.

#### Add

- Mention that the current SVIS branch computes 26 descriptors.
- Group the descriptors conceptually rather than reproducing implementation code:
  - central tendency
  - spread and robust spread
  - extrema and percentiles
  - distribution shape
  - information and energy
  - signal and correlation summaries
  - error and deviation summaries

#### Remove

- Remove unqualified statements that a descriptor directly measures tissue density or a specific pathology.
- Move the exact 26-feature list and implementation formula details to Chapter 3 or an appendix.

#### Evidence From Current Workspace

- `SVIS_RUNTIME_SOURCE`, especially `calculate_statistical_features`
- `PRIMARY_OUTPUT_ROOT/statistical_branch/features/covid_statistical_features.csv`
- `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv`

#### Suggested Figures

- Figure 2.3: Learned activation vector transformed into grouped statistical descriptors.

#### Suggested Tables

- Table 2.3: Statistical descriptor groups with representative examples and conceptual roles.

---

### 2.1.3 Augmented ZFMIS for Sparse-Value Filtering and Mutual-Information Ranking

#### Purpose

Introduce the feature-selection logic used to reduce redundant or uninformative evidence before rule extraction and predictive modelling.

#### Keep Unchanged

- Keep the old explanation of zero-based filtering.
- Keep the old explanation of mutual-information ranking.
- Keep the original rationale: compact selected sets can improve interpretability and reduce irrelevant inputs.

#### Update

- Explain that the original statistical-only selection concept has evolved into combined or augmented ZFMIS after late fusion.
- State that the current feature candidates include statistical descriptors, concept probabilities, and diagnostic probabilities.

#### Add

- Introduce the idea that selection over a heterogeneous fused space reveals which evidence families complement each other.
- Explain that multiple selected-set sizes support comparison between compact interpretability and predictive coverage.

#### Remove

- Remove any statement that ZFMIS is applied only to statistical descriptors in the final SS-VIRULEX layer.
- Keep exact thresholds, selected sizes, selected columns, and ranking output analysis in Chapter 3 or Chapter 4.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `SVIS_RUNTIME_SOURCE`
- `PRIMARY_OUTPUT_ROOT/zfmis/zfmis_ranking_covid.csv`
- `PRIMARY_OUTPUT_ROOT/zfmis/selected_features_base_covid.csv`
- `PRIMARY_OUTPUT_ROOT/zfmis/selected_features_covid.csv`
- `PRIMARY_OUTPUT_ROOT/features/04_combined_zfmis_feature_sets.json`

#### Suggested Figures

- Figure 2.4: Augmented ZFMIS concept: heterogeneous evidence pool -> sparse-value filtering -> mutual-information ranking -> compact feature sets.

#### Suggested Tables

- Table 2.4: Original statistical ZFMIS versus augmented fused-space ZFMIS.

---

### 2.1.4 Rule-Based Interpretability: Decision Trees and RuleFit

#### Purpose

Explain why readable conditional rules remain useful when a system also evaluates stronger predictive models.

#### Keep Unchanged

- Keep the old Decision Tree explanation.
- Keep the old RuleFit explanation.
- Keep the motivation for human-readable conditions, thresholds, and feature interactions.

#### Update

- Describe Decision Tree and RuleFit as transparent explanatory models operating over selected fused evidence.
- Explain that their rules may combine statistical descriptors, concept probabilities, and Med-MICN diagnostic probabilities.
- Clarify that transparent rules and final predictive models have complementary roles.

#### Add

- Introduce the interpretability-performance tradeoff.
- Explain that an interpretable system may report a readable rule path alongside a calibrated prediction from a different estimator.

#### Remove

- Remove the implication that the strongest exported SS-VIRULEX predictor must be a Decision Tree or RuleFit model.
- Remove any claim that the current RuleFit outputs are persisted as complete per-image serialized models if not evidenced.

#### Evidence From Current Workspace

- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_decision_tree_rules.txt`
- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_rulefit_rules.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/06_combined_rule_extraction_summary.json`
- `PRIMARY_OUTPUT_ROOT/current_output_layout_manifest.json`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`

#### Suggested Figures

- Figure 2.5: Parallel outputs: readable rule explanation and calibrated predictive score.

#### Suggested Tables

- Table 2.5: Decision Tree, RuleFit, and predictive classifier roles.

---

### 2.1.5 Statistical Feature Map Overlay Visualisation and Its Concept-Aware Extension

#### Purpose

Explain the statistical visual explanation lineage and define precisely what the current concept-aware extension does.

#### Keep Unchanged

- Keep the old high-level motivation for SFMOV as a visual complement to global rules and tabular statistics.
- Keep the explanation that statistical summaries can influence visual overlays derived from convolutional activations.

#### Update

- Define the current visual output as a `scalar concept-aware SFMOV overlay`.
- State that the current extension modulates statistical SFMOV heatmaps with `concept_alpha`, a scalar derived from Med-MICN concept scores.
- Explain that exported combined panels pair the heatmap with top concept probabilities and a readable rule path.

#### Add

- Add a limitation paragraph: concept-aware modulation is not the same as concept-specific spatial localization.
- Add a boundary statement: the current exported COVID-CT run does not contain generated concept Grad-CAM PNGs.

#### Remove

- Remove claims that the current overlay identifies the image region instantiating each named clinical concept.
- Remove claims that BioViL directly creates the exported SFMOV maps.
- Remove references to generated concept Grad-CAM results for the current COVID-CT run.

#### Evidence From Current Workspace

- `SVIS_RUNTIME_SOURCE`
- `PRIMARY_RUNNER`, especially `make_concept_overlay`
- `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json`
- `PRIMARY_OUTPUT_ROOT/heatmaps/concept_aware/`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/sfmov/`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/ss_virulex_concept_aware/`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`

#### Suggested Figures

- Figure 2.6: Statistical SFMOV map construction and scalar concept-aware modulation.
- Figure 2.7: Layered combined explanation panel anatomy.

#### Suggested Tables

- Table 2.6: Visual explanation claims that are supported, limited, or unsupported in the exported run.

---

### 2.1.6 Concept Bottleneck Models and Concept-Based Interpretability

#### Purpose

Introduce concept-based interpretability as the theoretical basis for exposing semantically meaningful intermediate evidence.

#### Keep Unchanged

- Keep the old Concept Bottleneck Model definition.
- Keep the discussion of concepts as an intermediate vocabulary that can improve inspectability.
- Keep the discussion of limitations, including concept incompleteness and annotation dependence.

#### Update

- Distinguish general CBM literature from the current Med-MICN-derived implementation.
- Avoid claiming that SS-VIRULEX is a textbook CBM with a single strictly enforced concept bottleneck.

#### Add

- Explain that the current system consumes concept probabilities as one evidence family within a broader fused space.
- Explain that concept probabilities support semantic interpretation even when the final predictor also uses non-concept evidence.

#### Remove

- Remove any implication that all predictive information must pass exclusively through concepts in the current SS-VIRULEX runtime.

#### Evidence From Current Workspace

- `MED_SOURCE_ROOT/models.py`
- `MED_SOURCE_ROOT/torch_explain/nn/concepts.py`
- `PRIMARY_OUTPUT_ROOT/concept_branch/probabilities/covid_concept_probabilities.csv`
- `PRIMARY_OUTPUT_ROOT/concept_branch/labels/covid_concept_labels.csv`

#### Suggested Figures

- Figure 2.8: Classical CBM bottleneck versus SS-VIRULEX concept evidence inside a wider fused space.

#### Suggested Tables

- Table 2.7: Concept-based model patterns and how SS-VIRULEX relates to them.

---

### 2.1.7 Large Multimodal Models and Vision-Language Models for Medical Concept Semantics

#### Purpose

Provide literature background for automated concept generation and medical image-text alignment while separating architectural genealogy from current runtime behavior.

#### Keep Unchanged

- Keep the old high-level discussion of Large Multimodal Models for candidate concept generation.
- Keep BioViL as a relevant medical vision-language alignment model in the literature review.
- Keep the motivation for aligning visual evidence with clinically meaningful language.

#### Update

- Present GPT-4V and BioViL as Med-MICN genealogy and semantic-alignment context.
- State carefully that the current exported SS-VIRULEX runtime uses a completed concept CSV as a materialized input rather than executing runnable GPT-4V or BioViL stages.

#### Add

- Add the distinction between:
  - upstream semantic-alignment workflow
  - materialized semantic supervision artifact
  - runtime Med-MICN concept inference
- Explain that literature ancestry and executable runtime must not be conflated.

#### Remove

- Remove any sentence saying that GPT-4V or BioViL is called per image during the current pipeline run.
- Remove any claim that the present overlays are BioViL similarity maps.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/Med_MICN_COVID_CT_training.ipynb`
- `MED_SOURCE_ROOT/README.md`

#### Suggested Figures

- Figure 2.9: Semantic genealogy versus executable runtime boundary.

#### Suggested Tables

- Table 2.8: Semantic-stage status matrix: literature concept, inherited workflow, materialized artifact, or active runtime stage.

---

### 2.1.8 Med-MICN Concept Embeddings and Neural-Symbolic Fuzzy Reasoning

#### Purpose

Explain the Med-MICN lineage that supplies semantic concept probabilities and a neural-symbolic diagnostic output to current SS-VIRULEX.

#### Keep Unchanged

- Keep the old background explanation of neural-symbolic reasoning.
- Keep the motivation for fuzzy logic rules when concepts are probabilistic rather than binary certainties.
- Keep the discussion of concept embeddings and reasoning layers.

#### Update

- State that the current Med-MICN checkpoint emits eight concept probabilities and a neural-symbolic diagnostic probability.
- Describe the neural-symbolic probability as a fused downstream feature in SS-VIRULEX.
- Clarify that statistical descriptors do not modify the internal fuzzy reasoning layer in the current implementation.

#### Add

- Add a brief distinction between:
  - Med-MICN task output
  - Med-MICN neural-symbolic output
  - Med-MICN concept outputs
- Explain why retaining both diagnostic probabilities can provide complementary model-derived evidence.

#### Remove

- Remove the statement that statistical evidence is injected into Med-MICN's internal neural-symbolic rules.
- Remove any claim that the final extracted Decision Tree rules are the same artifact as Med-MICN fuzzy reasoning explanations.

#### Evidence From Current Workspace

- `MED_SOURCE_ROOT/models.py`
- `MED_SOURCE_ROOT/torch_explain/nn/concepts.py`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/best_model.pt`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/concept_metrics.csv`
- `PRIMARY_RUNNER`

#### Suggested Figures

- Figure 2.10: Med-MICN output taxonomy: concepts, task probability, and neural-symbolic probability.

#### Suggested Tables

- Table 2.9: Med-MICN-derived evidence items consumed by SS-VIRULEX.

---

### 2.1.9 Late Fusion of Statistical, Semantic, and Model-Derived Diagnostic Evidence

#### Purpose

Introduce the major conceptual addition missing from the old Chapter 2 outline: heterogeneous late fusion.

#### Keep Unchanged

- Keep the old motivation that statistical and semantic interpretability offer complementary perspectives.

#### Update

- Replace the old two-source or shared-vector explanation with a late-fusion evidence-space explanation.
- Describe three evidence families:
  - statistical descriptors
  - concept probabilities
  - diagnostic probabilities

#### Add

- Explain late fusion as the combination of evidence after branch-specific inference.
- Introduce model-derived diagnostic probabilities as stacked evidence features without overexplaining the implementation procedure.
- Explain why preserving provenance matters for interpretability.

#### Remove

- Remove any claim of end-to-end joint training across both branches in the exported reconstructed runtime.
- Remove any claim that all evidence families have the same representation origin.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv`
- `PRIMARY_OUTPUT_ROOT/features/04_combined_zfmis_feature_sets.json`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_pipeline_summary.json`

#### Suggested Figures

- Figure 2.11: Late-fusion evidence vector with 26 statistical, eight semantic, and two diagnostic-probability inputs.

#### Suggested Tables

- Table 2.10: Fused evidence families, origins, dimensions, and interpretive meanings.

---

### 2.1.10 Predictive Classifiers, Probability Calibration, and Validation-Based Threshold Selection

#### Purpose

Provide the missing background concepts needed to explain why the current system evaluates calibrated predictive models in addition to transparent rules.

#### Keep Unchanged

- Keep any old general discussion that interpretability should be evaluated alongside predictive reliability.

#### Update

- Expand the background beyond Decision Trees and RuleFit.
- Explain that classification probability quality and decision thresholds matter in medical decision-support contexts.

#### Add

- Introduce:
  - predictive classifier comparison
  - probability calibration
  - validation-based operating-threshold selection
  - separation of model fitting, calibration, threshold selection, and held-out testing
- Explain conceptually why a default threshold may not optimize balanced performance.
- Mention that exact estimators, calibration method, threshold-search logic, and chosen values belong in Chapter 3.

#### Remove

- Remove exact current best-model metrics and threshold values from Chapter 2.
- Remove any implication that a threshold may be tuned on the held-out test set.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `PRIMARY_OUTPUT_ROOT/summaries/05e_combined_stronger_classifiers_results.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/05f_combined_calibrated_threshold_results.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_final_metrics_summary.csv`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`

#### Suggested Figures

- Figure 2.12: Conceptual evaluation flow: fit -> calibrate -> choose validation threshold -> test once.

#### Suggested Tables

- Table 2.11: Transparent explanations versus predictive reliability components.

---

### 2.1.11 Explainability Modalities and Scope Boundaries

#### Purpose

Provide a taxonomy that prevents the final thesis from collapsing distinct explanation artifacts into one claim.

#### Keep Unchanged

- Keep the old explainability taxonomy motivation.
- Keep distinctions among visual, feature-level, concept-level, and rule-level explanations.

#### Update

- Align the taxonomy with the actual exported artifacts.
- Explicitly distinguish global, local, semantic, quantitative, visual, and predictive outputs.

#### Add

- Add the implemented output layers:
  - global statistical feature ranking
  - global Decision Tree rules
  - global RuleFit rules
  - per-image concept probabilities
  - per-image Med-MICN diagnostic probabilities
  - scalar concept-aware SFMOV overlay
  - combined explanation panels
  - calibrated predictive score and class decision
- Add unsupported or reserved categories:
  - concept-specific Grad-CAM is reserved but not exported
  - per-concept spatial localization is not established by scalar modulation

#### Remove

- Remove any tripartite claim that every rule, concept, and region has a one-to-one correspondence.
- Remove any visual-localization wording stronger than the generated artifacts support.

#### Evidence From Current Workspace

- `PRIMARY_OUTPUT_ROOT/current_output_layout_manifest.json`
- `PRIMARY_OUTPUT_ROOT/export_manifest.json`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md`
- `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json`

#### Suggested Figures

- Figure 2.13: SS-VIRULEX explanation taxonomy and claim boundaries.

#### Suggested Tables

- Table 2.12: Explanation modality, granularity, artifact, and supported interpretation.

---

### 2.2 Literature Review

#### Purpose

Position SS-VIRULEX across four literature pillars. The fourth pillar is new and is required because the implemented architecture now includes heterogeneous evidence fusion and calibrated decision support.

#### Keep Unchanged

- Keep the old three-pillar review structure as the foundation.
- Keep the progression from post-hoc explanations, to statistical and rule methods, to concept-based ante-hoc models.

#### Update

- Add a fourth pillar for hybrid evidence fusion, calibration, and layered interpretability.
- Make each genealogy discussion distinguish inherited ideas from currently executed modules.

#### Add

- Add an explicit synthesis paragraph at the end of each pillar explaining what that literature contributes and what limitation remains.

#### Remove

- Remove implementation procedure details.
- Remove universal novelty claims unless supported by a systematic review.

#### Evidence From Current Workspace

- `OLD_CHAPTER_DRAFT`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_RUNNER`

#### Suggested Figures

- Figure 2.14: Four-pillar literature map.

#### Suggested Tables

- Table 2.13: Literature pillars, representative method families, and unresolved limitations.

---

### 2.2.1 Pillar I: Post-Hoc Visual and Attribution Methods

#### Purpose

Review explanation methods that operate after model fitting and contextualize SFMOV as a distinct statistical visual approach.

#### Keep Unchanged

- Keep early gradient-based and activation-map methods.
- Keep SHAP, LIME, and instance-level attribution.
- Keep concept-level post-hoc methods such as TCAV.

#### Update

- Compare visual localization, feature attribution, and concept sensitivity without treating them as interchangeable.
- Use this pillar to explain why an overlay alone is not a full semantic explanation.

#### Add

- Add a transition to SFMOV and the current layered output strategy.

#### Remove

- Remove any implication that the implemented scalar concept-aware SFMOV overlay is equivalent to concept-specific Grad-CAM, TCAV, or a VLM similarity map.

#### Evidence From Current Workspace

- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/`
- `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md`

#### Suggested Figures

- Figure 2.15: Visual explanation method taxonomy.

#### Suggested Tables

- Table 2.14: Post-hoc method families and explanation granularity.

---

### 2.2.2 Pillar II: Statistical and Rule-Based Interpretability Frameworks

#### Purpose

Review interpretable statistical summaries, Decision Trees, surrogate models, RuleFit, and the SVIS-RULEX lineage.

#### Keep Unchanged

- Keep Decision Trees and surrogate models.
- Keep statistical feature engineering in deep-learning pipelines.
- Keep the SVIS-RULEX architectural genealogy.

#### Update

- Add RuleFit explicitly as a transparent sparse rule model.
- State that the current SS-VIRULEX implementation inherits executable SVIS-RULEX functions for statistical features, ZFMIS, rules, and SFMOV.
- Explain that the inherited branch has been extended by fusion rather than replaced.

#### Add

- Add a synthesis paragraph: statistical rules are inspectable but may lack clinically named semantics when used alone.

#### Remove

- Remove any statement that SVIS-RULEX alone describes the final architecture.

#### Evidence From Current Workspace

- `SVIS_RUNTIME_SOURCE`
- `PRIMARY_OUTPUT_ROOT/summaries/04_svis_rulex_required_components.json`
- `PRIMARY_OUTPUT_ROOT/statistical_branch/features/covid_statistical_features.csv`
- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_decision_tree_rules.txt`
- `PRIMARY_OUTPUT_ROOT/rules/06_combined_best_rulefit_rules.csv`

#### Suggested Figures

- Figure 2.16: SVIS-RULEX genealogy and inherited component boundary.

#### Suggested Tables

- Table 2.15: SVIS-RULEX inherited concepts and SS-VIRULEX extensions.

---

### 2.2.3 Pillar III: Concept-Based and Neural-Symbolic Ante-Hoc Models

#### Purpose

Review CBMs, concept embeddings, medical VLM alignment, fuzzy reasoning, and the Med-MICN lineage.

#### Keep Unchanged

- Keep CBMs and extensions.
- Keep vision-language models in medical concept alignment.
- Keep Med-MICN architectural genealogy and contributions.

#### Update

- Separate Med-MICN's conceptual ancestry from the exact executable COVID-CT integration.
- Explain that current SS-VIRULEX consumes Med-MICN outputs rather than jointly retraining one merged architecture.

#### Add

- Add the eight-concept COVID-CT semantic vocabulary as a concrete implementation-alignment note, without turning the literature review into methodology.
- Add a synthesis paragraph: concepts improve semantic interpretability but may not capture all quantitative evidence needed for reliable classification.

#### Remove

- Remove any current-runtime GPT-4V or BioViL claim.
- Remove any claim that every visual explanation is produced by Med-MICN.

#### Evidence From Current Workspace

- `MED_SOURCE_ROOT/README.md`
- `MED_SOURCE_ROOT/models.py`
- `MED_SOURCE_ROOT/torch_explain/nn/concepts.py`
- `PRIMARY_OUTPUT_ROOT/summaries/05_med_micn_required_components.json`
- `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json`
- `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv`

#### Suggested Figures

- Figure 2.17: Med-MICN genealogy and active-output boundary.

#### Suggested Tables

- Table 2.16: Med-MICN inherited concepts, active outputs, and inactive upstream runtime stages.

---

### 2.2.4 Pillar IV: Hybrid Evidence Fusion and Calibrated Clinical Decision Support

#### Purpose

Add the literature area required by the evolved implementation: combining heterogeneous evidence while maintaining reliable operating decisions and layered explanations.

#### Keep Unchanged

- Keep the old general argument that no single explanation modality is sufficient for all stakeholders.

#### Update

- Reframe SS-VIRULEX as a hybrid late-fusion framework.
- Explain that statistical, semantic, neural-symbolic, and predictive evidence can coexist without being generated by one shared representation.

#### Add

- Review literature concepts for:
  - late fusion
  - stacking or use of model-derived probability features
  - probability calibration
  - operating-threshold selection
  - interpretability-performance tradeoffs
  - layered clinical decision-support explanations
- Conclude with the unresolved need for explanation outputs whose provenance and scope are explicit.

#### Remove

- Remove procedural details such as the exact classifier grid, calibration API, or chosen operating threshold.
- Remove claims that calibration alone makes a system clinically deployable.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/05e_combined_stronger_classifiers_results.csv`
- `PRIMARY_OUTPUT_ROOT/summaries/05f_combined_calibrated_threshold_results.csv`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`

#### Suggested Figures

- Figure 2.18: Hybrid late-fusion and layered-decision-support literature position.

#### Suggested Tables

- Table 2.17: Hybrid-system design concepts and their role in SS-VIRULEX.

---

### 2.3 Implementation-Aligned Research Gap

#### Purpose

Define the problem addressed by the current architecture without overstating novelty or drifting into Chapter 3.

#### Keep Unchanged

- Keep the old concern that purely statistical explanations may be difficult to interpret clinically.
- Keep the old concern that concept-only methods may omit useful quantitative evidence.
- Keep the old motivation for complementary explanation layers.

#### Update

- State the gap as a need for provenance-aware integration of complementary evidence and explanation modalities.
- Include calibrated predictive decision support as part of the evolved gap.

#### Add

- Add the requirement that claims remain proportional to generated artifacts.
- Add the need to distinguish explanatory rule models from the strongest predictive model.

#### Remove

- Remove universal statements such as "no existing framework" unless backed by a systematic literature review.
- Remove claims that the implemented framework establishes one-to-one rule-concept-region alignment.

#### Evidence From Current Workspace

- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_OUTPUT_ROOT/current_output_layout_manifest.json`
- `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md`
- `PRIMARY_OUTPUT_ROOT/visual_explanations/heatmaps/concept_gradcam/README.md`

#### Suggested Figures

- Figure 2.19: Gap synthesis: semantic opacity, quantitative incompleteness, reliability needs, and explanation-scope discipline.

#### Suggested Tables

- Table 2.18: Research gaps and corresponding architectural responses.

---

### 2.3.1 Limitations of Isolated Explanation Modalities

#### Purpose

Explain why statistical, rule-based, concept-based, and visual explanations each solve only part of the interpretability problem.

#### Keep Unchanged

- Keep the old comparison among statistical, visual, concept-based, and rule-based explanations.

#### Update

- Include diagnostic-probability evidence and calibrated decisions in the comparison.
- Discuss provenance and scope, not only explanation availability.

#### Add

- Add the following balanced limitations:
  - statistical summaries can be inspectable but semantically indirect
  - concept scores can be clinically named but incomplete
  - visual overlays can be intuitive but easy to overinterpret
  - readable rules can be auditable but may trail stronger classifiers
  - calibrated scores support decision reliability but are not explanations by themselves

#### Remove

- Remove any implication that adding more modalities automatically resolves clinical interpretability.

#### Evidence From Current Workspace

- `PRIMARY_OUTPUT_ROOT/summaries/06_combined_rule_extraction_summary.json`
- `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_final_metrics_summary.csv`

#### Suggested Figures

- No new figure required if Figure 2.19 includes this synthesis.

#### Suggested Tables

- Table 2.19: Strengths and limitations of each explanation modality.

---

### 2.3.2 SS-VIRULEX Design Positioning

#### Purpose

State what the architecture contributes at a high level without duplicating the methodology.

#### Keep Unchanged

- Keep the old positioning of SS-VIRULEX as a semantic-statistical visual interpretable rule-based framework.

#### Update

- Define the architecture as a provenance-aware late-fusion framework.
- Describe layered explanation exports conservatively.
- Include model-derived diagnostic probabilities and calibrated prediction.

#### Add

- Add this positioning sequence:
  1. inherit statistical and visual evidence from SVIS-RULEX
  2. inherit concept and neural-symbolic evidence from Med-MICN
  3. fuse heterogeneous evidence after branch-specific inference
  4. select compact mixed evidence sets
  5. provide transparent rules and scalar concept-aware visual overlays
  6. evaluate predictive models with calibration and validation-based threshold selection

#### Remove

- Remove exact stage numbering, class counts, augmentation settings, hyperparameters, and result values.
- Remove any claim of end-to-end joint training.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `EXECUTION_MAP`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_OUTPUT_ROOT/export_manifest.json`

#### Suggested Figures

- Reuse Figure 2.1 or Figure 2.19 rather than adding a duplicate.

#### Suggested Tables

- Reuse Table 2.18.

---

### 2.3.3 Boundary Between Background, Methodology, and Results

#### Purpose

Prevent Chapter 2 from absorbing procedural or evaluative content that belongs in later chapters.

#### Keep Unchanged

- Keep concise transitions that foreshadow the methodology.

#### Update

- Ensure Chapter 2 defines concepts while Chapter 3 documents their implementation.
- Ensure Chapter 4 reports measurements and comparisons.

#### Add

- Explicitly defer these items to Chapter 3:
  - official split construction
  - augmentation procedure
  - exact backbone initialization
  - checkpoint loading
  - the full 26-descriptor list and formulas
  - fused-column construction
  - sparse-value threshold
  - selected-set sizes
  - classifier grid
  - calibration implementation
  - validation threshold search
  - artifact-export procedure
- Explicitly defer these items to Chapter 4:
  - metric values
  - best configuration
  - comparison tables
  - rule examples
  - feature rankings
  - example heatmaps

#### Remove

- Remove code-level narration from Chapter 2.
- Remove result commentary from Chapter 2.

#### Evidence From Current Workspace

- `OLD_METHODOLOGY_OUTLINE`
- `PRIMARY_RUNNER`
- `EXECUTION_MAP`
- `EXPORTED_EXECUTION_MAP`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`
- `PRIMARY_OUTPUT_ROOT/summaries/ss_virulex_final_metrics_summary.csv`

#### Suggested Figures

- No figure required.

#### Suggested Tables

- Table 2.20: Chapter ownership map: Background, Methodology, Results.

---

### 2.4 Chapter Summary

#### Purpose

Close Chapter 2 by summarizing the conceptual path and transitioning to the implemented methodology.

#### Keep Unchanged

- Keep a concise summary of the literature foundations and the identified research gap.

#### Update

- Mention the evolved late-fusion architecture.
- Mention the distinction between explanation layers and calibrated prediction.

#### Add

- End with a direct transition: Chapter 3 specifies how the independent branches, fused evidence space, augmented ZFMIS, rule models, calibrated predictors, and explanation exports are instantiated for COVID-CT.

#### Remove

- Remove metric values.
- Remove procedural detail.
- Remove unsupported spatial-localization claims.

#### Evidence From Current Workspace

- `PRIMARY_RUNNER`
- `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md`

#### Suggested Figures

- No new figure required.

#### Suggested Tables

- No new table required.

## Updated Chapter Flow

Use this exact order for the regenerated Chapter 2:

1. `2.1 Conceptual Foundations for SS-VIRULEX`
2. `2.1.1 Deep Learning and Learned Feature Representations for Medical Images`
3. `2.1.2 Statistical Feature Engineering Over Learned Representations`
4. `2.1.3 Augmented ZFMIS for Sparse-Value Filtering and Mutual-Information Ranking`
5. `2.1.4 Rule-Based Interpretability: Decision Trees and RuleFit`
6. `2.1.5 Statistical Feature Map Overlay Visualisation and Its Concept-Aware Extension`
7. `2.1.6 Concept Bottleneck Models and Concept-Based Interpretability`
8. `2.1.7 Large Multimodal Models and Vision-Language Models for Medical Concept Semantics`
9. `2.1.8 Med-MICN Concept Embeddings and Neural-Symbolic Fuzzy Reasoning`
10. `2.1.9 Late Fusion of Statistical, Semantic, and Model-Derived Diagnostic Evidence`
11. `2.1.10 Predictive Classifiers, Probability Calibration, and Validation-Based Threshold Selection`
12. `2.1.11 Explainability Modalities and Scope Boundaries`
13. `2.2 Literature Review`
14. `2.2.1 Pillar I: Post-Hoc Visual and Attribution Methods`
15. `2.2.2 Pillar II: Statistical and Rule-Based Interpretability Frameworks`
16. `2.2.3 Pillar III: Concept-Based and Neural-Symbolic Ante-Hoc Models`
17. `2.2.4 Pillar IV: Hybrid Evidence Fusion and Calibrated Clinical Decision Support`
18. `2.3 Implementation-Aligned Research Gap`
19. `2.3.1 Limitations of Isolated Explanation Modalities`
20. `2.3.2 SS-VIRULEX Design Positioning`
21. `2.3.3 Boundary Between Background, Methodology, and Results`
22. `2.4 Chapter Summary`

### Logical Flow

The narrative should move through these questions:

1. How are useful image representations learned?
2. How can learned representations be summarized quantitatively?
3. How can a compact evidence set be selected?
4. How can transparent rules be extracted?
5. How can statistical evidence be visualized?
6. How do clinically named concepts improve interpretability?
7. Where do multimodal and vision-language methods fit in the semantic lineage?
8. How does Med-MICN add concept embeddings and fuzzy neural-symbolic reasoning?
9. Why does the evolved architecture combine independent evidence sources by late fusion?
10. Why are calibration and validation-based threshold choice needed alongside readable rules?
11. What claims can each exported explanation modality support?
12. How do the four literature pillars motivate the implementation-aligned research gap?

## Figures and Diagrams Inventory

### Required Figures

| Figure | Title | Required content | Source support | Reuse guidance |
|---|---|---|---|---|
| Figure 2.1 | Conceptual Overview of the Implemented SS-VIRULEX Architecture | Separate SVIS and Med-MICN branches; 26 statistical descriptors; eight concept probabilities; two diagnostic probabilities; late fusion; augmented ZFMIS; explanatory rules; calibrated predictors; scalar concept-aware SFMOV; combined exports. | `PRIMARY_RUNNER`, `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md` | Create a new as-built diagram. |
| Figure 2.2 | Branch-Specific Representation Provenance | MobileNetV2 SVIS path versus `RN50` Med-MICN path; no shared feature tensor. | `SVIS_RUNTIME_SOURCE`, Med-MICN checkpoint config, `MED_SOURCE_ROOT/models.py` | Create new. |
| Figure 2.4 | Augmented ZFMIS | Multi-source input -> sparse-value filtering -> MI ranking -> compact sets. | `PRIMARY_OUTPUT_ROOT/features/04_combined_zfmis_feature_sets.json` | Create new. |
| Figure 2.6 | Scalar Concept-Aware SFMOV | SFMOV statistical maps plus scalar `concept_alpha` modulation. Include explicit note that this is not per-concept localization. | `PRIMARY_RUNNER`, `PRIMARY_OUTPUT_ROOT/summaries/07_concept_aware_sfmov_heatmap_files.json` | Create new. |
| Figure 2.11 | Late-Fusion Evidence Space | 26 + 8 + 2 feature-family diagram with provenance labels. | `PRIMARY_OUTPUT_ROOT/fusion/fused_features_covid.csv` | Create new. |
| Figure 2.13 | Explanation Modalities and Claim Boundaries | Global rules, concept scores, diagnostic probabilities, scalar SFMOV overlays, combined explanation panels, calibrated decisions, and reserved concept Grad-CAM category. | `PRIMARY_OUTPUT_ROOT/current_output_layout_manifest.json` | Create new. |
| Figure 2.14 | Four-Pillar Literature Map | Post-hoc visual; statistical and rule-based; concept and neural-symbolic; hybrid fusion and calibration. | This guide and literature sources | Create new. |
| Figure 2.19 | Implementation-Aligned Gap Synthesis | Limits of isolated modalities and the evolved SS-VIRULEX response. | This guide and `PRIMARY_OUTPUT_ROOT/final_results/SS_VIRULEX_PIPELINE_REPORT.md` | Create new. |

### Optional Figures

| Figure | Title | Use when | Source support |
|---|---|---|---|
| Figure 2.3 | Learned Representation to Statistical Descriptor Groups | The statistical-feature explanation needs a visual aid. | `SVIS_RUNTIME_SOURCE` |
| Figure 2.5 | Readable Rule and Calibrated Predictor as Complementary Outputs | The predictive-versus-explanatory distinction needs emphasis. | `PRIMARY_OUTPUT_ROOT/summaries/06_combined_rule_extraction_summary.json` |
| Figure 2.7 | Combined Explanation Panel Anatomy | An exported example panel is introduced conceptually. Use actual examples in Chapter 4. | `PRIMARY_OUTPUT_ROOT/final_results/combined_explanations/README.md` |
| Figure 2.8 | Classical CBM Versus SS-VIRULEX Fused Concept Evidence | The CBM distinction is difficult to explain textually. | `MED_SOURCE_ROOT/models.py`, `PRIMARY_RUNNER` |
| Figure 2.9 | Semantic Genealogy Versus Runtime Boundary | GPT-4V/BioViL lineage needs clear separation from the exported run. | `PRIMARY_RUNNER`, `/Users/samehissa/Downloads/covid_ct_concepts_completed.csv` |
| Figure 2.10 | Med-MICN Output Taxonomy | The three Med-MICN-derived output categories need emphasis. | `PRIMARY_OUTPUT_ROOT/01_Med-MICN/outputs/covid_ct_med_micn/run_summary.json` |
| Figure 2.12 | Calibration and Threshold Selection Workflow | Clinical operating-point background is introduced in depth. | `PRIMARY_RUNNER` |
| Figure 2.16 | SVIS-RULEX Genealogy | The thesis includes detailed inherited-component discussion. | `SVIS_RUNTIME_SOURCE` |
| Figure 2.17 | Med-MICN Genealogy | The thesis includes detailed inherited-component discussion. | `MED_SOURCE_ROOT/README.md`, `MED_SOURCE_ROOT/models.py` |

### Earlier Pipeline Assets That Require Redraw

Inspect but do not reuse unchanged:

- `PROPOSED_PIPELINE_ASSETS/SS_VIRULEX_pipeline.png`
- `PROPOSED_PIPELINE_ASSETS/new pipeline.png`
- `PROPOSED_PIPELINE_ASSETS/concept_semantics_branch_detail.svg`
- `PROPOSED_PIPELINE_ASSETS/integrated_pipeline_overview.svg`

Before any old asset is reused, verify that it no longer implies:

- one shared feature vector `F`
- direct GPT-4V execution during the current runtime
- direct BioViL execution during the current runtime
- ROI cropping
- per-concept spatial heatmaps
- omission of the two Med-MICN diagnostic probabilities
- omission of calibration and validation-based threshold selection
- a rule model as the only final predictor

## Tables Inventory

### Required Tables

| Table | Title | Required rows or columns | Chapter role |
|---|---|---|---|
| Table 2.1 | Current SS-VIRULEX Terminology | Old term; current term; reason for update | Prevent terminology drift. |
| Table 2.2 | Representation Sources and Roles | Branch; backbone lineage; representation; downstream use; runtime status | Replace the shared-vector assumption. |
| Table 2.4 | Original Versus Augmented ZFMIS | Input families; filtering; ranking; output sets; interpretive role | Explain selection evolution. |
| Table 2.6 | Visual Explanation Scope Matrix | Artifact; generated in current run; spatial meaning; semantic meaning; prohibited overclaim | Prevent visual-explanation overstatement. |
| Table 2.10 | Fused Evidence Families | Evidence family; count; source branch; examples; interpretation | Establish the 26 + 8 + 2 architecture. |
| Table 2.12 | Explanation Modality Taxonomy | Modality; local/global; artifact; supported interpretation; limitation | Align claims with outputs. |
| Table 2.13 | Four Literature Pillars | Pillar; method families; contribution; unresolved limitation | Organize the review. |
| Table 2.18 | Research Gap and Architectural Response | Gap; limitation; SS-VIRULEX response; caveat | Support the research-gap section. |
| Table 2.20 | Chapter Ownership Map | Topic; Chapter 2 depth; Chapter 3 depth; Chapter 4 depth | Prevent methodology duplication. |

### Optional Tables

| Table | Title | Use when |
|---|---|---|
| Table 2.3 | Statistical Descriptor Groups | Use grouped examples only; move full 26-feature list to Chapter 3 or appendix. |
| Table 2.5 | Decision Tree, RuleFit, and Predictive Classifier Roles | Use to clarify explanatory versus predictive responsibilities. |
| Table 2.7 | Concept-Based Model Patterns | Use if the CBM literature review is extensive. |
| Table 2.8 | Semantic Stage Status Matrix | Use to distinguish GPT-4V/BioViL ancestry from active runtime stages. |
| Table 2.9 | Med-MICN-Derived Evidence Items | Use if the three output categories need a compact summary. |
| Table 2.11 | Predictive Reliability Concepts | Use if calibration and threshold-selection literature receives a full subsection. |
| Table 2.14 | Post-Hoc Method Families | Use to compare saliency, attribution, and concept sensitivity methods. |
| Table 2.15 | SVIS-RULEX Inheritance Map | Use to show inherited versus extended components. |
| Table 2.16 | Med-MICN Inheritance Map | Use to show inherited, materialized, and inactive stages. |
| Table 2.17 | Hybrid-System Design Concepts | Use to summarize late fusion, stacking, calibration, and layered outputs. |
| Table 2.19 | Strengths and Limitations of Isolated Modalities | Use in the research-gap synthesis. |

### Tables That Do Not Belong in Chapter 2

Move these to Chapter 3, Chapter 4, or an appendix:

- exact split counts
- augmentation counts
- complete 26-feature implementation list
- selected-feature rankings
- exact feature subsets
- classifier hyperparameters
- calibration API details
- selected threshold values
- model performance metrics
- confusion matrices
- rule examples with learned thresholds
- per-image prediction outputs

## References and Citations Guidance

### Concepts That Require Literature Citations

Claude must cite external scholarly sources for:

- deep learning and CNNs in medical-image analysis
- MobileNetV2 and residual-network backbones
- descriptive statistical feature engineering over learned representations
- mutual information for feature selection
- Decision Trees
- RuleFit
- post-hoc saliency and activation-map methods
- SHAP
- LIME
- TCAV
- Concept Bottleneck Models and extensions
- Large Multimodal Models used for concept generation
- BioViL and medical vision-language alignment
- fuzzy logic and neural-symbolic reasoning
- late fusion and stacking
- probability calibration
- validation-based operating-threshold selection
- clinical decision-support interpretability and reliability

### Implementation Details That Must Not Be Presented as Literature Claims

The following are workspace facts. Describe them as implementation-alignment statements, supported by Chapter 3 or internal artifacts rather than by external citations:

- the current checkpoint uses `RN50`
- the current COVID-CT vocabulary contains eight concepts
- the SVIS activation layer is named `xai_feature_layer`
- the statistical branch computes 26 descriptors
- the fused space contains 36 candidates
- the fused space is partitioned into 26 statistical, eight semantic, and two diagnostic-probability features
- the current runner evaluates specific classifier families
- the current runner calibrates estimators and selects thresholds on validation data
- the current visual extension uses scalar `concept_alpha`
- generated concept Grad-CAM PNGs are absent from the exported COVID-CT run
- combined explanation panels contain prediction, rule path, heatmap, and top concept scores

### Citation Discipline

- Cite original papers for named methods whenever possible.
- Cite review papers only for broad synthesis.
- Do not use implementation artifacts as substitutes for scholarly citations.
- Do not cite GPT-4V or BioViL as active runtime dependencies of the exported pipeline.
- Do not convert internal filenames into academic references.
- Do not claim clinical validity from implementation artifacts alone.
- Verify the bibliography entries before drafting the final chapter.

## Methodology Concepts That Must Not Be Discussed Deeply in Chapter 2

Chapter 2 may define these ideas briefly, but their concrete implementation belongs in Chapter 3:

| Concept | Chapter 2 treatment | Chapter 3 treatment |
|---|---|---|
| Official COVID-CT split loading | Mention reproducible split discipline only. | Document source files, split construction, and counts. |
| Image augmentation | Define why augmentation is used. | Document probabilities, angle, scale, brightness, contrast, and generated manifest. |
| MobileNetV2 initialization | Introduce lightweight CNN representations. | Document cached ImageNet weights, frozen base, and trained head. |
| `xai_feature_layer` extraction | Explain learned-representation summaries. | Document the 128-dimensional activation extraction. |
| Twenty-six statistical descriptors | Explain descriptor families. | List every descriptor and formula. |
| Med-MICN checkpoint loading | Explain inherited semantic and neural-symbolic evidence. | Document checkpoint path, `RN50`, configuration, and inference procedure. |
| Materialized concept CSV | Explain upstream-versus-runtime distinction. | Document dataset columns, provenance, and validation. |
| Late fusion | Define the concept. | Document exact columns and merge keys. |
| Augmented ZFMIS | Define filtering and MI ranking. | Document zero-fraction criterion, ranking procedure, and selected sets. |
| Transparent rules | Explain Decision Tree and RuleFit conceptually. | Document grid search, training, extraction, and export. |
| Stronger classifiers | Explain the rationale for predictive comparison. | Document estimator family and hyperparameters. |
| Calibration | Explain why probability reliability matters. | Document calibration method and data partition discipline. |
| Threshold selection | Explain operating-point selection. | Document objective, validation search, selected threshold, and test application. |
| SFMOV | Explain statistical overlay motivation. | Document exact map computation. |
| Scalar concept-aware SFMOV | Explain scope and limitation. | Document `concept_alpha` computation and overlay generation. |
| Export layout | Mention layered explanation outputs. | Document output directories, manifests, and generation order. |

## Claude Writing Instructions

### Core Instruction

Write a completely regenerated Chapter 2 using the structure in this guide. Do not patch the old chapter paragraph by paragraph. Preserve valid literature background where useful, but rebuild the narrative around the current late-fusion implementation.

### Academic Tone

- Use formal thesis prose.
- Prefer precise, bounded claims.
- Avoid promotional language.
- Avoid claims of clinical deployment readiness.
- Avoid universal novelty claims unless supported by a documented systematic review.
- Distinguish motivation, literature finding, implementation fact, and inference.

### Thesis Style

- Begin each major section with its purpose and end with a transition.
- Use the updated section order exactly unless a supervisor requests a structural change.
- Define a term before using its acronym.
- Use one canonical term consistently after definition.
- Keep the chapter conceptual; defer procedural detail to Chapter 3 and results to Chapter 4.

### Implementation Alignment

- Describe two independent runtime branches:
  - MobileNetV2-based SVIS statistical and visual branch
  - `RN50` Med-MICN semantic and neural-symbolic branch
- Describe late fusion after branch-specific inference.
- Describe three fused evidence families:
  - 26 statistical descriptors
  - eight concept probabilities
  - two Med-MICN diagnostic probabilities
- Describe combined ZFMIS as an extension of the original statistical selection concept.
- Describe Decision Tree and RuleFit as readable explanatory models.
- Describe stronger and calibrated models as a complementary predictive layer.
- Describe concept-aware SFMOV as scalar concept modulation, not concept localization.
- State that generated concept Grad-CAM PNGs are absent from the current exported COVID-CT run.

### Terminology Rules

Use:

- `SS-VIRULEX`
- `Semantic-Statistical Visual Interpretable Rule-based Explainer`
- `late-fused evidence space`
- `statistical descriptors`
- `Med-MICN concept probabilities`
- `Med-MICN task probability`
- `Med-MICN neural-symbolic probability`
- `model-derived diagnostic probabilities`
- `augmented ZFMIS` or `combined ZFMIS`
- `transparent explanatory models`
- `calibrated predictive models`
- `validation-based threshold selection`
- `scalar concept-aware SFMOV overlay`
- `materialized semantic-alignment artifact`

Do not use:

- `shared feature vector F` as a current-runtime statement
- `current GPT-4V stage`
- `current BioViL stage`
- `concept-specific Grad-CAM output` for the exported COVID-CT run
- `per-concept spatial localization` for scalar SFMOV modulation
- `statistics-enriched Med-MICN fuzzy rules`
- `end-to-end jointly trained SS-VIRULEX` for the exported reconstructed pipeline

### Avoid Methodology Duplication

Do not include:

- code paths in the final prose
- Python symbol names in the final prose unless a methodology cross-reference requires them
- exact hyperparameter grids
- exact augmentation probabilities
- exact split counts
- exact selected-feature lists
- exact operating thresholds
- result metrics
- detailed output-directory trees

Those details are recorded in this guide only so that the conceptual chapter stays accurate.

### Maintain Consistency With Chapter 3

Before finalizing the regenerated Chapter 2:

1. Check that Chapter 3 uses the same branch names.
2. Check that Chapter 3 does not claim one shared feature tensor.
3. Check that Chapter 3 documents the 26 + 8 + 2 fused evidence composition.
4. Check that Chapter 3 separates Med-MICN task and neural-symbolic probabilities.
5. Check that Chapter 3 distinguishes base statistical ZFMIS from combined ZFMIS.
6. Check that Chapter 3 separates readable rule extraction from calibrated prediction.
7. Check that Chapter 3 calls the visual output scalar concept-aware SFMOV.
8. Check that Chapter 3 does not claim generated concept Grad-CAM PNGs.
9. Check that Chapter 3 treats GPT-4V and BioViL as upstream lineage or materialized-artifact context unless executable stages are later added and evidenced.
10. Check that Chapter 4, not Chapter 2, reports metrics and example outputs.

### Handling the DDI Companion Implementation

- Do not merge DDI-specific statements into the COVID-CT Chapter 2 architecture description.
- If the thesis discusses the DDI companion, label it as a separate extension, companion implementation, or future generalization.
- Do not use DDI-specific visualization behavior as evidence for COVID-CT outputs.

### Final Self-Check for Claude

Before delivering the regenerated Chapter 2, answer each question:

1. Did the chapter remove the shared-MobileNetV2-vector assumption?
2. Did it state that the current runtime has separate MobileNetV2 and `RN50` branches?
3. Did it introduce late fusion?
4. Did it mention all three evidence families?
5. Did it add augmented ZFMIS?
6. Did it distinguish transparent rules from predictive classifiers?
7. Did it introduce calibration and validation-based threshold selection conceptually?
8. Did it keep GPT-4V and BioViL in genealogy and literature context rather than current runtime?
9. Did it avoid concept-localization overclaims?
10. Did it state the limitation of scalar concept-aware SFMOV?
11. Did it keep implementation procedure in Chapter 3?
12. Did it keep metric values in Chapter 4?
13. Did it avoid silently mixing the DDI companion implementation with the COVID-CT run?

If any answer is no, revise the chapter before delivery.
