#!/usr/bin/env python3
"""Build the requested SS-VIRULEX output layout from the current run artifacts."""

from __future__ import annotations

import ast
import json
import pickle
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


WORKSPACE = Path("/Users/samehissa/Downloads/SS-VIRULEX-workspace")
OUT_ROOT = WORKSPACE / "03_SS-VIRULEX-outputs"
MED_RUN = OUT_ROOT / "01_Med-MICN" / "outputs" / "covid_ct_med_micn"
XAI_RUN = OUT_ROOT / "02_SVIS-RULEX" / "covid_ct_run" / "exact_sequence" / "outputs"
FEATURE_DIR = XAI_RUN / "features"
MODEL_DIR = XAI_RUN / "models"
RULE_DIR = XAI_RUN / "rules"
CONCEPT_CSV = Path("/Users/samehissa/Downloads/covid_ct_concepts_completed.csv")
CLASS_NAMES = {0: "NonCOVID", 1: "COVID"}

STAT_RENAME = {
    "mean": "stat_mean",
    "std_dev": "stat_std_dev",
    "variance": "stat_variance",
    "median": "stat_median",
    "range": "stat_range",
    "skewness": "stat_skewness",
    "kurtosis": "stat_kurtosis",
    "entropy": "stat_entropy",
    "energy": "stat_energy",
    "contrast": "stat_contrast",
    "mean_abs_dev": "stat_mean_abs_dev",
    "min_value": "stat_min_value",
    "max_value": "stat_max_value",
    "iqr": "stat_iqr",
    "percentile_25": "stat_percentile_25",
    "percentile_50": "stat_percentile_50",
    "percentile_75": "stat_percentile_75",
    "signal_to_noise": "stat_signal_to_noise",
    "coef_of_var": "stat_coef_of_var",
    "autocorrelation": "stat_autocorrelation",
    "shannon_entropy": "stat_shannon_entropy",
    "root_mean_square": "stat_root_mean_square",
    "harmonic_mean": "stat_harmonic_mean",
    "geometric_mean": "stat_geometric_mean",
    "std_error_mean": "stat_std_error_mean",
    "median_abs_dev": "stat_median_abs_dev",
}
STAT_COLUMNS = list(STAT_RENAME)
CONCEPT_SCORE_COLUMNS = [
    "concept_score_peripheral_ground_glass_opacities",
    "concept_score_bilateral_involvement",
    "concept_score_multilobar_distribution",
    "concept_score_crazy_paving_pattern",
    "concept_score_absence_of_lobar_consolidation",
    "concept_score_localized_or_diffuse_presentation",
    "concept_score_increased_density_in_the_lung",
    "concept_score_ground_glass_appearance",
]
MED_DIAGNOSIS_COLUMNS = ["med_task_prob_covid", "med_neural_prob_covid"]
ID_COLUMNS = ["image_id", "file_name", "class_name", "true_label", "split", "image_path"]


def safe_id(class_name: str, filename: str) -> str:
    stem = Path(str(filename)).stem
    prefix = "covid" if class_name == "COVID" else "noncovid"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return f"{prefix}_{cleaned}"


def ensure_dirs() -> None:
    for rel in [
        "dataset",
        "statistical_branch/features",
        "concept_branch/labels",
        "concept_branch/metrics",
        "concept_branch/probabilities",
        "fusion",
        "zfmis",
        "final_results/evaluation_metrics",
        "final_results/predictions",
        "final_results/feature_importance",
        "final_results/plots",
        "final_results/combined_explanations",
        "visual_explanations/heatmaps/original",
        "visual_explanations/heatmaps/sfmov",
        "visual_explanations/heatmaps/concept_gradcam",
        "visual_explanations/heatmaps/ss_virulex_concept_aware",
    ]:
        (OUT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"filepath": "image_path", "filename": "file_name", "label": "true_label"})
    df["image_id"] = [safe_id(cls, name) for cls, name in zip(df["class_name"], df["file_name"])]
    return df[ID_COLUMNS].copy()


def metadata_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    official = load_metadata(XAI_RUN / "covid_ct_official_splits.csv")
    augmented = load_metadata(XAI_RUN / "01_augmented_splits.csv")

    official.to_csv(OUT_ROOT / "dataset" / "covid_metadata.csv", index=False)
    write_json(OUT_ROOT / "dataset" / "class_counts.json", official["class_name"].value_counts().sort_index().to_dict())

    split_counts = (
        official.groupby(["split", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["split", "class_name"])
    )
    split_counts.to_csv(OUT_ROOT / "dataset" / "split_counts.csv", index=False)
    write_json(OUT_ROOT / "dataset" / "split_counts.json", split_counts.to_dict(orient="records"))
    return official, augmented


def attach_split_metadata(prefix: str, metadata: pd.DataFrame, feature_kind: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        if feature_kind == "stat":
            src = FEATURE_DIR / f"04_{split}_statistical_features.csv"
        else:
            src = FEATURE_DIR / f"04_{split}_combined_stat_concept_features.csv"
        feat = pd.read_csv(src)
        meta = metadata[metadata["split"] == split].reset_index(drop=True)
        if len(meta) != len(feat):
            raise ValueError(f"{src} has {len(feat)} rows but metadata split {split} has {len(meta)} rows")
        feat = feat.rename(columns={"label": "true_label", **STAT_RENAME})
        merged = pd.concat([meta[ID_COLUMNS].reset_index(drop=True), feat.drop(columns=["true_label"], errors="ignore")], axis=1)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def feature_outputs(augmented_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat = attach_split_metadata("covid", augmented_meta, "stat")
    stat_cols = [STAT_RENAME[col] for col in STAT_COLUMNS if STAT_RENAME[col] in stat.columns]
    stat = stat[ID_COLUMNS + stat_cols]
    stat.to_csv(OUT_ROOT / "statistical_branch" / "features" / "covid_statistical_features.csv", index=False)

    fused = attach_split_metadata("covid", augmented_meta, "combined")
    stat_cols = [STAT_RENAME[col] for col in STAT_COLUMNS if STAT_RENAME[col] in fused.columns]
    concept_cols = [col for col in CONCEPT_SCORE_COLUMNS if col in fused.columns]
    med_diag_cols = [col for col in MED_DIAGNOSIS_COLUMNS if col in fused.columns]
    fused = fused[ID_COLUMNS + stat_cols + concept_cols + med_diag_cols]
    fused.to_csv(OUT_ROOT / "fusion" / "fused_features_covid.csv", index=False)
    return stat, fused


def concept_outputs() -> None:
    concept = pd.read_csv(CONCEPT_CSV)
    concept = concept.rename(columns={"file_name": "file_name", "true_label": "true_label"})
    keep = [col for col in ["image_id", "file_name", "class_name", "true_label", "split", "image_path"] if col in concept.columns]
    label_cols = [col for col in concept.columns if col.startswith("concept_") and col.endswith("_label")]
    concept[keep + label_cols].to_csv(OUT_ROOT / "concept_branch" / "labels" / "covid_concept_labels.csv", index=False)

    shutil.copy2(MED_RUN / "training_history.csv", OUT_ROOT / "concept_branch" / "metrics" / "training_history.csv")
    shutil.copy2(
        MED_RUN / "test_concept_metrics.csv",
        OUT_ROOT / "concept_branch" / "metrics" / "covid_concept_encoder_metrics.csv",
    )
    shutil.copy2(
        MED_RUN / "test_metrics.csv",
        OUT_ROOT / "concept_branch" / "metrics" / "covid_concept_training_metrics.csv",
    )
    probabilities_path = MED_RUN / "per_image_concept_probabilities.csv"
    if probabilities_path.exists():
        shutil.copy2(
            probabilities_path,
            OUT_ROOT / "concept_branch" / "probabilities" / "covid_concept_probabilities.csv",
        )


def med_micn_artifact_outputs() -> None:
    artifact_root = OUT_ROOT / "01_Med-MICN"
    run_summary_path = MED_RUN / "run_summary.json"
    test_metrics = {}
    if run_summary_path.exists():
        test_metrics = json.loads(run_summary_path.read_text(encoding="utf-8")).get("test_metrics", {})

    artifacts = {
        "checkpoint": MED_RUN / "best_model.pt",
        "training_history": MED_RUN / "training_history.csv",
        "test_metrics": MED_RUN / "test_metrics.csv",
        "test_concept_metrics": MED_RUN / "test_concept_metrics.csv",
        "run_summary": run_summary_path,
    }
    manifest = {
        "role": "trained_med_micn_model_artifacts",
        "description": "This folder is the trained Med-MICN artifact cache consumed by the SS-VIRULEX pipeline.",
        "artifact_root": str(artifact_root),
        "trained_model_dir": str(MED_RUN),
        "ss_virulex_usage": [
            "validate the required Med-MICN concept-embedding checkpoint",
            "validate the neural-symbolic branch metrics",
            "export Med-MICN concept and task metrics into concept_branch/metrics",
            "fuse Med-MICN checkpoint-generated concept probabilities with SVIS-RULEX statistical features",
        ],
        "artifacts": {
            key: {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for key, path in artifacts.items()
        },
        "per_image_concept_probabilities": {
            "path": str(MED_RUN / "per_image_concept_probabilities.csv"),
            "exists": (MED_RUN / "per_image_concept_probabilities.csv").exists(),
            "bytes": (MED_RUN / "per_image_concept_probabilities.csv").stat().st_size
            if (MED_RUN / "per_image_concept_probabilities.csv").exists()
            else None,
        },
        "test_metrics": test_metrics,
    }
    write_json(artifact_root / "trained_model_manifest.json", manifest)

    readme = """# 01_Med-MICN

This folder is the trained Med-MICN model artifact cache used by SS-VIRULEX.

Primary checkpoint:

`outputs/covid_ct_med_micn/best_model.pt`

SS-VIRULEX consumes this folder to validate the Med-MICN concept embedding and
neural-symbolic branch before building the fused statistical-plus-concept
features. The exported Med-MICN metrics are also copied into
`../concept_branch/metrics`.

Per-image concept probabilities generated by `best_model.pt` are stored at:

`outputs/covid_ct_med_micn/per_image_concept_probabilities.csv`

The Med-MICN source repository remains separate at:

`/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main`

When the SS-VIRULEX runner is used with `--run-med-micn`, newly trained Med-MICN
artifacts should be synced back into this folder via `--med-artifact-root`.
"""
    (artifact_root / "README.md").write_text(readme, encoding="utf-8")


def svis_rulex_artifact_outputs() -> None:
    artifact_root = OUT_ROOT / "02_SVIS-RULEX"
    metrics_path = XAI_RUN / "03_custom_mobilenetv2_test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    artifacts = {
        "custom_mobilenet_complete": XAI_RUN / "models" / "03_custom_mobilenetv2_complete_covid_ct.keras",
        "custom_mobilenet_head": XAI_RUN / "models" / "03_custom_mobilenetv2_head.keras",
        "custom_mobilenet_metrics": metrics_path,
        "train_statistical_features": XAI_RUN / "features" / "04_train_statistical_features.csv",
        "val_statistical_features": XAI_RUN / "features" / "04_val_statistical_features.csv",
        "test_statistical_features": XAI_RUN / "features" / "04_test_statistical_features.csv",
        "statistical_zfmis_feature_sets": XAI_RUN / "features" / "04_zfmis_feature_sets.json",
        "svis_decision_tree_results": XAI_RUN / "05a_gridsearchfortree_results.csv",
        "svis_rulefit_results": XAI_RUN / "05b_rulefitgridsearchcode_results.csv",
    }
    manifest = {
        "role": "trained_svis_rulex_artifacts",
        "description": "This folder is the trained SVIS-RULEX artifact cache consumed by the SS-VIRULEX pipeline.",
        "artifact_root": str(artifact_root),
        "trained_output_dir": str(XAI_RUN),
        "ss_virulex_usage": [
            "load the trained custom MobileNetV2 model for visual explanation stages",
            "consume SVIS-RULEX statistical features as the statistical branch of SS-VIRULEX",
            "use SVIS-RULEX ZFMIS/rule outputs as the stat-only baseline in final metrics",
            "fuse SVIS-RULEX statistical features with Med-MICN concept probabilities",
        ],
        "artifacts": {
            key: {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for key, path in artifacts.items()
        },
        "custom_mobilenet_metrics": {
            key: metrics.get(key)
            for key in ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc", "epochs_run"]
        },
    }
    write_json(artifact_root / "trained_artifact_manifest.json", manifest)

    readme = """# 02_SVIS-RULEX

This folder is the trained SVIS-RULEX artifact cache used by SS-VIRULEX.

Primary trained model:

`covid_ct_run/exact_sequence/outputs/models/03_custom_mobilenetv2_complete_covid_ct.keras`

SS-VIRULEX consumes this folder for the statistical branch: trained MobileNetV2
outputs, statistical feature CSVs, ZFMIS feature sets, and stat-only rule
baselines. The fused SS-VIRULEX branch combines these SVIS-RULEX statistical
features with Med-MICN checkpoint-generated concept probabilities.

The original SVIS-RULEX source repository remains separate at:

`/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main 2`
"""
    (artifact_root / "README.md").write_text(readme, encoding="utf-8")


def build_zfmis(fused: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    with (FEATURE_DIR / "04_combined_zfmis_feature_sets.json").open(encoding="utf-8") as fh:
        feature_sets = json.load(fh)
    selected = [STAT_RENAME.get(name, name) for name in feature_sets["selected_18"]]

    feature_cols = [
        col
        for col in fused.columns
        if col.startswith("stat_") or col.startswith("concept_score_") or col in MED_DIAGNOSIS_COLUMNS
    ]
    train = fused[fused["split"] == "train"].copy()
    X = train[feature_cols].fillna(0)
    y = train["true_label"].astype(int)
    scores = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    zero_fraction = (X == 0).mean().to_dict()

    rows = []
    for feature, score in zip(feature_cols, scores):
        rows.append(
            {
                "feature": feature,
                "feature_type": "med_diagnosis_probability"
                if feature in MED_DIAGNOSIS_COLUMNS
                else "concept_score"
                if feature.startswith("concept_score_")
                else "statistical",
                "zero_fraction": zero_fraction.get(feature, np.nan),
                "mutual_info_score": score,
                "selected": feature in selected,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["mutual_info_score", "feature"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    ranking = ranking[["feature", "feature_type", "zero_fraction", "mutual_info_score", "rank", "selected"]]
    ranking.to_csv(OUT_ROOT / "zfmis" / "zfmis_ranking_covid.csv", index=False)

    base = fused[ID_COLUMNS + selected].copy()
    base.to_csv(OUT_ROOT / "zfmis" / "selected_features_base_covid.csv", index=False)

    final = base.copy()
    for col in selected:
        if col.startswith("concept_score_"):
            final[col.replace("concept_score_", "concept_") + "_label"] = (final[col] >= 0.5).astype(int)
    final.to_csv(OUT_ROOT / "zfmis" / "selected_features_covid.csv", index=False)
    return ranking, selected


def parse_confusion(value: object) -> list[list[int]] | None:
    if pd.isna(value):
        return None
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None


def normalize_metric_rows(path: Path, family: str, model_prefix: str, rulefit: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        model_type = "rulefit" if rulefit else "decision_tree"
        rows.append(
            {
                "model_family": family,
                "model": f"{model_prefix}_{model_type}_{row['feature_set']}",
                "feature_set": row.get("feature_set"),
                "n_features": row.get("n_features"),
                "accuracy": row.get("test_accuracy"),
                "balanced_accuracy": row.get("test_balanced_accuracy"),
                "f1_weighted": row.get("test_f1_weighted"),
                "roc_auc": row.get("test_roc_auc", np.nan),
                "confusion_matrix": row.get("confusion_matrix"),
            }
        )
    return pd.DataFrame(rows)


def normalize_stronger_classifier_rows(path: Path, model_prefix: str = "ss_virulex") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model_family": "SS-VIRULEX",
                "model": f"{model_prefix}_{row['classifier']}_{row['feature_set']}",
                "feature_set": row.get("feature_set"),
                "n_features": row.get("n_features"),
                "accuracy": row.get("test_accuracy"),
                "balanced_accuracy": row.get("test_balanced_accuracy"),
                "f1_weighted": row.get("test_f1_weighted"),
                "roc_auc": row.get("test_roc_auc", np.nan),
                "confusion_matrix": row.get("confusion_matrix"),
            }
        )
    return pd.DataFrame(rows)


def metrics_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    svis = pd.concat(
        [
            normalize_metric_rows(XAI_RUN / "05a_gridsearchfortree_results.csv", "SVIS-RULEX", "svis_rulex"),
            normalize_metric_rows(XAI_RUN / "05b_rulefitgridsearchcode_results.csv", "SVIS-RULEX", "svis_rulex", True),
        ],
        ignore_index=True,
    )
    ss = pd.concat(
        [
            normalize_metric_rows(XAI_RUN / "05c_combined_gridsearchfortree_results.csv", "SS-VIRULEX", "ss_virulex"),
            normalize_metric_rows(XAI_RUN / "05d_combined_rulefitgridsearchcode_results.csv", "SS-VIRULEX", "ss_virulex", True),
            normalize_stronger_classifier_rows(XAI_RUN / "05e_combined_stronger_classifiers_results.csv"),
            normalize_stronger_classifier_rows(
                XAI_RUN / "05f_combined_calibrated_threshold_results.csv",
                "ss_virulex_calibrated_threshold",
            ),
        ],
        ignore_index=True,
    )

    med_raw = pd.read_csv(MED_RUN / "test_metrics.csv").iloc[0].to_dict()
    med = pd.DataFrame(
        [
            {
                "model_family": "Med-MICN",
                "model": "med_micn_task_classifier",
                "feature_set": "concept_branch",
                "n_features": np.nan,
                "accuracy": med_raw.get("task_accuracy"),
                "balanced_accuracy": med_raw.get("task_recall_macro"),
                "f1_weighted": med_raw.get("task_f1_macro"),
                "roc_auc": med_raw.get("task_auc"),
                "confusion_matrix": "",
            },
            {
                "model_family": "Med-MICN",
                "model": "med_micn_neural_symbolic",
                "feature_set": "neural_symbolic",
                "n_features": np.nan,
                "accuracy": med_raw.get("neural_accuracy"),
                "balanced_accuracy": med_raw.get("neural_recall_macro"),
                "f1_weighted": med_raw.get("neural_f1_macro"),
                "roc_auc": med_raw.get("neural_auc"),
                "confusion_matrix": "",
            },
            {
                "model_family": "Med-MICN",
                "model": "med_micn_concept_encoder",
                "feature_set": "concept_encoder",
                "n_features": np.nan,
                "accuracy": med_raw.get("concept_accuracy"),
                "balanced_accuracy": np.nan,
                "f1_weighted": med_raw.get("concept_f1_macro"),
                "roc_auc": np.nan,
                "confusion_matrix": "",
            },
        ]
    )

    all_metrics = pd.concat([svis, med, ss], ignore_index=True)
    metrics_dir = OUT_ROOT / "final_results" / "evaluation_metrics"
    all_metrics.to_csv(metrics_dir / "metrics_covid.csv", index=False)
    svis.to_csv(metrics_dir / "metrics_svis_rulex_covid.csv", index=False)
    med.to_csv(metrics_dir / "metrics_med_micn_covid.csv", index=False)
    ss.to_csv(metrics_dir / "metrics_ss_virulex_covid.csv", index=False)
    return all_metrics, svis, med, ss


def three_model_comparison_outputs(all_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_dir = OUT_ROOT / "final_results" / "evaluation_metrics"
    plots_dir = OUT_ROOT / "final_results" / "plots"
    family_order = ["SVIS-RULEX", "Med-MICN", "SS-VIRULEX"]
    metric_cols = ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]

    comparable = all_metrics.dropna(subset=["balanced_accuracy"]).copy()
    comparable = comparable[comparable["model_family"].isin(family_order)]
    comparable["family_order"] = pd.Categorical(comparable["model_family"], categories=family_order, ordered=True)
    comparable = comparable.sort_values(
        ["family_order", "balanced_accuracy", "accuracy", "f1_weighted", "roc_auc"],
        ascending=[True, False, False, False, False],
    )

    comparison = comparable.groupby("model_family", sort=False, as_index=False).head(1).copy()
    comparison = comparison.sort_values("family_order").drop(columns=["family_order"])
    comparison = comparison.rename(columns={"model": "selected_model"})
    comparison.insert(2, "selection_basis", "best_diagnosis_row_by_balanced_accuracy")

    ranked = comparison["balanced_accuracy"].rank(method="dense", ascending=False).astype(int)
    comparison.insert(0, "rank_by_balanced_accuracy", ranked)

    baseline_rows = comparison[comparison["model_family"] == "SVIS-RULEX"]
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        for metric in metric_cols:
            comparison[f"delta_{metric}_vs_svis_rulex"] = comparison[metric] - baseline[metric]

    output_columns = [
        "rank_by_balanced_accuracy",
        "model_family",
        "selected_model",
        "selection_basis",
        "feature_set",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
        "roc_auc",
        "delta_accuracy_vs_svis_rulex",
        "delta_balanced_accuracy_vs_svis_rulex",
        "delta_f1_weighted_vs_svis_rulex",
        "delta_roc_auc_vs_svis_rulex",
        "confusion_matrix",
    ]
    available_columns = [col for col in output_columns if col in comparison.columns]
    comparison = comparison[available_columns]
    comparison.to_csv(metrics_dir / "metrics_three_model_comparison_covid.csv", index=False)

    plot_df = comparison.set_index("model_family")
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "f1_weighted": "F1 weighted",
        "roc_auc": "ROC AUC",
    }
    x = np.arange(len(metric_cols))
    width = 0.24
    colors = {"SVIS-RULEX": "#4c78a8", "Med-MICN": "#f58518", "SS-VIRULEX": "#54a24b"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, family in enumerate(family_order):
        if family not in plot_df.index:
            continue
        values = [plot_df.loc[family, metric] for metric in metric_cols]
        ax.bar(x + (idx - 1) * width, values, width, label=family, color=colors[family])
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[metric] for metric in metric_cols])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Three-Model COVID Diagnosis Metrics")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(metrics_dir / "metrics_three_model_comparison_covid.png", dpi=180)
    fig.savefig(plots_dir / "three_model_comparison_covid.png", dpi=180)
    plt.close(fig)

    return comparison


def plot_metrics(df: pd.DataFrame, path: Path, title: str) -> None:
    plot_df = df.dropna(subset=["accuracy"]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["model"].str.replace("_", "\n")
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.85), 5))
    ax.bar(range(len(plot_df)), plot_df["accuracy"], color="#2f6f9f")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], labels=["NonCOVID", "COVID"])
    ax.set_yticks([0, 1], labels=["NonCOVID", "COVID"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(arr):
        ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plotting_outputs(all_metrics: pd.DataFrame, svis: pd.DataFrame, med: pd.DataFrame, ss: pd.DataFrame, ranking: pd.DataFrame) -> None:
    metrics_dir = OUT_ROOT / "final_results" / "evaluation_metrics"
    plots_dir = OUT_ROOT / "final_results" / "plots"
    plot_metrics(all_metrics, metrics_dir / "metrics_covid.png", "COVID Model Metrics")
    plot_metrics(svis, metrics_dir / "metrics_svis_rulex_covid.png", "SVIS-RULEX Metrics")
    plot_metrics(med, metrics_dir / "metrics_med_micn_covid.png", "Med-MICN Metrics")
    plot_metrics(ss, metrics_dir / "metrics_ss_virulex_covid.png", "SS-VIRULEX Metrics")
    plot_metrics(all_metrics, plots_dir / "metric_comparison_bar_chart.png", "COVID Model Metrics")

    top = ranking.head(20).sort_values("mutual_info_score")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mutual_info_score"], color="#6f8f3f")
    ax.set_xlabel("Mutual information")
    ax.set_title("Top ZFMIS Features")
    fig.tight_layout()
    fig.savefig(plots_dir / "zfmis_ranking_top20.png", dpi=180)
    plt.close(fig)

    for _, row in all_metrics.iterrows():
        matrix = parse_confusion(row.get("confusion_matrix"))
        if matrix:
            plot_confusion(matrix, plots_dir / f"confusion_matrix_{row['model']}.png", str(row["model"]))

    concept_metrics = pd.read_csv(OUT_ROOT / "concept_branch" / "metrics" / "covid_concept_encoder_metrics.csv")
    if "concept" in concept_metrics.columns and "f1" in concept_metrics.columns:
        cm = concept_metrics.sort_values("f1")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(cm["concept"], cm["f1"], color="#9467bd")
        ax.set_xlabel("F1")
        ax.set_title("Concept Encoder Performance")
        fig.tight_layout()
        fig.savefig(plots_dir / "concept_encoder_performance.png", dpi=180)
        plt.close(fig)


def model_predictions_and_importance(fused: pd.DataFrame, selected: list[str]) -> None:
    pred_dir = OUT_ROOT / "final_results" / "predictions"
    imp_dir = OUT_ROOT / "final_results" / "feature_importance"
    plots_dir = OUT_ROOT / "final_results" / "plots"

    test = fused[fused["split"] == "test"].reset_index(drop=True)
    pred = test[["image_id", "file_name", "class_name", "true_label", "split", "image_path"]].copy()
    importance_rows = []

    model_specs = [
        ("svis_rulex_decision_tree", XAI_RUN / "05a_gridsearchfortree_results.csv", "05a_decision_tree", False),
        ("ss_virulex_decision_tree", XAI_RUN / "05c_combined_gridsearchfortree_results.csv", "05c_combined_decision_tree", True),
    ]
    for model_name, metrics_path, file_prefix, combined in model_specs:
        metrics = pd.read_csv(metrics_path).iloc[0]
        feature_set = str(metrics["feature_set"])
        model_path = MODEL_DIR / f"{file_prefix}_{feature_set}.pkl"
        with model_path.open("rb") as fh:
            model = pickle.load(fh)
        if combined:
            source = FEATURE_DIR / f"04_combined_{feature_set}_features" / f"test_combined_{feature_set}_features.csv"
        else:
            source = FEATURE_DIR / f"04_{feature_set}_features" / f"test_{feature_set}_features.csv"
        data = pd.read_csv(source).rename(columns={"label": "true_label", **STAT_RENAME})
        X = data.drop(columns=["true_label"], errors="ignore")
        X = X.rename(columns=STAT_RENAME)
        y_pred = model.predict(X)
        pred[f"{model_name}_prediction"] = y_pred
        if hasattr(model, "predict_proba"):
            pred[f"{model_name}_probability_positive"] = model.predict_proba(X)[:, 1]
        else:
            pred[f"{model_name}_probability_positive"] = np.nan
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            for feature, importance in zip(X.columns, importances):
                importance_rows.append({"model": model_name, "feature": feature, "importance": importance, "source": "decision_tree"})

    calibrated_path = XAI_RUN / "05f_combined_calibrated_threshold_results.csv"
    strong_path = calibrated_path if calibrated_path.exists() else XAI_RUN / "05e_combined_stronger_classifiers_results.csv"
    feature_sets_path = FEATURE_DIR / "04_combined_zfmis_feature_sets.json"
    if strong_path.exists() and feature_sets_path.exists():
        strong = pd.read_csv(strong_path).sort_values("test_balanced_accuracy", ascending=False).iloc[0]
        feature_sets = json.loads(feature_sets_path.read_text(encoding="utf-8"))
        columns = feature_sets[str(strong["feature_set"])]
        model_path = Path(strong["model_path"])
        if model_path.exists():
            with model_path.open("rb") as fh:
                model = pickle.load(fh)
            data = pd.read_csv(FEATURE_DIR / "04_test_combined_stat_concept_features.csv")
            X = data[columns]
            calibrated = strong_path.name.startswith("05f_")
            model_name = f"ss_virulex_{'calibrated_threshold_' if calibrated else ''}{strong['classifier']}"
            if hasattr(model, "predict_proba"):
                probability_positive = model.predict_proba(X.values)[:, 1]
            else:
                probability_positive = np.full(len(X), np.nan)
            if calibrated and not pd.isna(strong.get("threshold", np.nan)):
                pred[f"{model_name}_prediction"] = (probability_positive >= float(strong["threshold"])).astype(int)
            else:
                pred[f"{model_name}_prediction"] = model.predict(X.values)
            pred[f"{model_name}_probability_positive"] = probability_positive

            estimator = model
            if hasattr(model, "named_steps"):
                estimator = list(model.named_steps.values())[-1]
            importances = getattr(estimator, "feature_importances_", None)
            if importances is None and hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_).ravel()
            if importances is not None and len(importances) == len(columns):
                for feature, importance in zip(columns, importances):
                    importance_rows.append(
                        {
                            "model": model_name,
                            "feature": STAT_RENAME.get(feature, feature),
                            "importance": importance,
                            "source": "strong_classifier",
                        }
                    )

    pred["svis_rulex_rulefit_prediction"] = np.nan
    pred["svis_rulex_rulefit_probability_positive"] = np.nan
    pred["ss_virulex_rulefit_prediction"] = np.nan
    pred["ss_virulex_rulefit_probability_positive"] = np.nan
    pred.to_csv(pred_dir / "predictions_covid.csv", index=False)

    for rules_path, model_name in [
        (RULE_DIR / "05b_rulefit_selected_3_rules.csv", "svis_rulex_rulefit"),
        (RULE_DIR / "05d_combined_rulefit_selected_3_rules.csv", "ss_virulex_rulefit_selected_3"),
        (RULE_DIR / "05d_combined_rulefit_selected_6_rules.csv", "ss_virulex_rulefit_selected_6"),
    ]:
        if rules_path.exists():
            rules = pd.read_csv(rules_path)
            for _, row in rules.iterrows():
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": row.get("rule"),
                        "importance": row.get("importance"),
                        "source": "rulefit_rule",
                    }
                )

    importance = pd.DataFrame(importance_rows).sort_values("importance", ascending=False)
    importance.to_csv(imp_dir / "feature_importance_covid.csv", index=False)
    if not importance.empty:
        top = importance.head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"].astype(str), top["importance"], color="#b85c38")
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        fig.tight_layout()
        fig.savefig(plots_dir / "feature_importance_top15.png", dpi=180)
        plt.close(fig)


def resolve_current_file(path_text: str) -> Path | None:
    src = Path(str(path_text))
    candidates = [
        src,
        XAI_RUN / "heatmaps" / src.name,
        OUT_ROOT / "heatmaps" / "concept_aware" / src.name,
        OUT_ROOT / "visual_explanations" / "heatmaps" / "ss_virulex_concept_aware" / src.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(OUT_ROOT.rglob(src.name))
    return matches[0] if matches else None


def copy_if_found(src_text: str, dst: Path) -> str:
    src = resolve_current_file(src_text)
    if src is None or not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def visual_outputs() -> None:
    heatmap_dir = OUT_ROOT / "visual_explanations" / "heatmaps"
    cases = []
    explanation_context = load_combined_explanation_context()
    aware_path = XAI_RUN / "07_concept_aware_sfmov_heatmap_files.json"
    if aware_path.exists():
        aware = json.loads(aware_path.read_text(encoding="utf-8"))
    else:
        aware = []
    sfmov_files = json.loads((XAI_RUN / "07_sfmov_heatmap_files.json").read_text(encoding="utf-8"))

    for item in aware:
        image = Path(item["image"])
        class_name = item.get("class_name", image.parent.name)
        image_id = safe_id(class_name, image.name)
        original_dst = heatmap_dir / "original" / f"{image_id}_original{image.suffix}"
        original_path = copy_if_found(str(image), original_dst)
        aware_dst = heatmap_dir / "ss_virulex_concept_aware" / f"{image_id}_concept_aware.png"
        raw_aware = resolve_current_file(item["heatmap"])
        if raw_aware is not None and raw_aware.exists():
            build_annotated_concept_aware_heatmap(image_id, raw_aware, aware_dst, item, explanation_context)
            aware_out = str(aware_dst)
        else:
            aware_out = copy_if_found(item["heatmap"], aware_dst)

        sfmov_match = ""
        for sfmov in sfmov_files:
            sf = Path(sfmov)
            if image.stem in sf.stem and "combined" in sf.stem:
                sfmov_match = copy_if_found(sfmov, heatmap_dir / "sfmov" / f"{image_id}_sfmov.png")
                break

        combined_path = OUT_ROOT / "final_results" / "combined_explanations" / f"{image_id}_combined_explanation.png"
        build_combined_explanation(
            image_id=image_id,
            original_path=original_path,
            sfmov_path=sfmov_match,
            aware_path=aware_out,
            item=item,
            dst=combined_path,
            context=explanation_context,
        )

        cases.append(
            {
                "image_id": image_id,
                "image_path": str(image),
                "class_name": class_name,
                "true_label": 1 if class_name == "COVID" else 0,
                "original_image_path": original_path,
                "sfmov_heatmap_path": sfmov_match,
                "concept_gradcam_path": "",
                "ss_virulex_heatmap_path": aware_out,
                "concept_alpha": item.get("concept_alpha"),
                "weights": json.dumps(item.get("weights", {})),
                "combined_explanation_path": str(combined_path) if combined_path.exists() else "",
            }
        )

    pd.DataFrame(cases).to_csv(heatmap_dir / "heatmap_cases_covid.csv", index=False)
    readme = heatmap_dir / "concept_gradcam" / "README.md"
    readme.write_text(
        "The current reconstructed SS-VIRULEX run did not include concept Grad-CAM PNG artifacts. "
        "This folder is reserved for those files when the concept salience stage is rerun.\n",
        encoding="utf-8",
    )
    combined_readme = OUT_ROOT / "final_results" / "combined_explanations" / "README.md"
    combined_readme.write_text(
        "Each PNG is a single SS-VIRULEX explanation figure assembled from existing outputs: "
        "final fused prediction and readable decision path, the generated concept-aware heatmap, "
        "and top Med-MICN concept probabilities.\n",
        encoding="utf-8",
    )


def build_annotated_concept_aware_heatmap(
    image_id: str,
    raw_heatmap_path: Path,
    dst: Path,
    item: dict,
    context: dict,
) -> None:
    heatmap = mpimg.imread(raw_heatmap_path)
    top = top_concept_rows(context, image_id, top_n=3)
    fig = plt.figure(figsize=(5.2, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.82, 0.18], top=0.94, bottom=0.05, hspace=0.08)

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(heatmap)
    ax_img.set_title("SS-VIRULEX Concept-Aware Heatmap", fontsize=10, fontweight="bold", color="#0b2e6d")
    ax_img.axis("off")
    ax_img.text(
        0.02,
        0.96,
        f"concept alpha={float(item.get('concept_alpha', 0.0)):.3f}",
        transform=ax_img.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#0f172a", "alpha": 0.72, "edgecolor": "none"},
    )

    ax_txt = fig.add_subplot(grid[1, 0])
    ax_txt.axis("off")
    if top.empty:
        ax_txt.text(0.5, 0.5, "Top concept probabilities unavailable", ha="center", va="center", fontsize=8)
    else:
        x_positions = np.linspace(0.17, 0.83, len(top))
        colors = ["#4c78a8", "#72b7b2", "#f58518"]
        for x, (_, row), color in zip(x_positions, top.iterrows(), colors):
            label = "\n".join(wrapped_lines(str(row["concept"]), width=18, max_lines=2))
            ax_txt.text(
                x,
                0.50,
                f"{label}\n{float(row['score']):.3f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color="#111827",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "alpha": 0.18, "edgecolor": color},
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor="white")
    plt.close(fig)


def load_combined_explanation_context() -> dict:
    context: dict = {
        "predictions": pd.DataFrame(),
        "concept_probabilities": pd.DataFrame(),
        "fused_test": pd.DataFrame(),
        "model": None,
        "model_name": "",
        "feature_set": "",
        "feature_names": [],
        "test_features": pd.DataFrame(),
    }

    predictions_path = OUT_ROOT / "final_results" / "predictions" / "predictions_covid.csv"
    if predictions_path.exists():
        context["predictions"] = pd.read_csv(predictions_path)

    probabilities_path = OUT_ROOT / "concept_branch" / "probabilities" / "covid_concept_probabilities.csv"
    if probabilities_path.exists():
        context["concept_probabilities"] = pd.read_csv(probabilities_path)

    fused_path = OUT_ROOT / "fusion" / "fused_features_covid.csv"
    if fused_path.exists():
        fused = pd.read_csv(fused_path)
        context["fused_test"] = fused[fused["split"] == "test"].reset_index(drop=True)

    metrics_path = XAI_RUN / "05c_combined_gridsearchfortree_results.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).sort_values("test_balanced_accuracy", ascending=False).iloc[0]
        feature_set = str(metrics["feature_set"])
        model_path = MODEL_DIR / f"05c_combined_decision_tree_{feature_set}.pkl"
        feature_path = FEATURE_DIR / f"04_combined_{feature_set}_features" / f"test_combined_{feature_set}_features.csv"
        if model_path.exists() and feature_path.exists():
            with model_path.open("rb") as fh:
                context["model"] = pickle.load(fh)
            test_features = pd.read_csv(feature_path)
            context["test_features"] = test_features
            context["feature_names"] = [col for col in test_features.columns if col != "label"]
            context["feature_set"] = feature_set
            context["model_name"] = f"SS-VIRULEX Decision Tree ({feature_set})"
    return context


def get_row_by_image_id(frame: pd.DataFrame, image_id: str) -> pd.Series | None:
    if frame.empty or "image_id" not in frame.columns:
        return None
    matched = frame[frame["image_id"] == image_id]
    if matched.empty:
        return None
    return matched.iloc[0]


def label_name(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return CLASS_NAMES.get(int(value), str(value))


def display_feature_name(name: str) -> str:
    if name.startswith("concept_score_"):
        return "concept: " + name.removeprefix("concept_score_").replace("_", " ")
    stat_name = STAT_RENAME.get(name, name)
    if stat_name.startswith("stat_"):
        return "stat: " + stat_name.removeprefix("stat_").replace("_", " ")
    return name.replace("_", " ")


def wrapped_lines(text: str, width: int, max_lines: int | None = None) -> list[str]:
    lines: list[str] = []
    for part in str(text).splitlines():
        wrapped = textwrap.wrap(part, width=width, replace_whitespace=False) or [""]
        lines.extend(wrapped)
    if max_lines is not None and len(lines) > max_lines:
        return lines[: max_lines - 1] + ["..."]
    return lines


def decision_path_lines(context: dict, image_id: str, max_conditions: int = 8) -> list[str]:
    model = context.get("model")
    fused_test = context.get("fused_test", pd.DataFrame())
    test_features = context.get("test_features", pd.DataFrame())
    feature_names = context.get("feature_names", [])
    if model is None or fused_test.empty or test_features.empty or not feature_names:
        return ["Readable rule unavailable from existing artifacts."]

    matches = fused_test.index[fused_test["image_id"] == image_id].tolist()
    if not matches:
        return ["Readable rule unavailable: image was not found in the test feature table."]
    row_index = matches[0]
    if row_index >= len(test_features):
        return ["Readable rule unavailable: feature row index is out of range."]

    x_row = test_features.loc[row_index, feature_names].astype(float)
    tree = getattr(model, "tree_", None)
    if tree is None:
        return ["Readable rule unavailable: final model has no decision tree path."]

    node_indicator = model.decision_path(x_row.to_numpy().reshape(1, -1))
    leaf_id = model.apply(x_row.to_numpy().reshape(1, -1))[0]
    node_ids = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

    conditions = []
    for node_id in node_ids:
        if node_id == leaf_id:
            continue
        feature_idx = tree.feature[node_id]
        if feature_idx < 0:
            continue
        threshold = tree.threshold[node_id]
        feature = feature_names[feature_idx]
        value = float(x_row.iloc[feature_idx])
        op = "<=" if value <= threshold else ">"
        conditions.append(f"{display_feature_name(feature)} {op} {threshold:.4f} (value {value:.4f})")

    if not conditions:
        return ["Leaf reached without split conditions."]

    lines = ["IF " + conditions[0]]
    lines.extend("AND " + condition for condition in conditions[1:max_conditions])
    if len(conditions) > max_conditions:
        lines.append(f"... {len(conditions) - max_conditions} more conditions")
    return lines


def top_concept_rows(context: dict, image_id: str, top_n: int = 5) -> pd.DataFrame:
    probabilities = context.get("concept_probabilities", pd.DataFrame())
    row = get_row_by_image_id(probabilities, image_id)
    if row is None:
        return pd.DataFrame(columns=["concept", "score"])
    scores = []
    for col in probabilities.columns:
        if col.startswith("concept_score_"):
            scores.append(
                {
                    "concept": col.removeprefix("concept_score_").replace("_", " "),
                    "score": float(row[col]),
                }
            )
    return pd.DataFrame(scores).sort_values("score", ascending=False).head(top_n)


def prediction_lines(context: dict, image_id: str) -> list[str]:
    predictions = context.get("predictions", pd.DataFrame())
    row = get_row_by_image_id(predictions, image_id)
    if row is None:
        return ["Prediction row unavailable."]
    true_label = int(row["true_label"])
    pred_label = int(row["ss_virulex_decision_tree_prediction"])
    prob = float(row["ss_virulex_decision_tree_probability_positive"])
    correctness = "CORRECT" if true_label == pred_label else "INCORRECT"
    return [
        f"Image ID: {image_id}",
        f"True label: {label_name(true_label)}",
        f"Final fused prediction: {label_name(pred_label)} [{correctness}]",
        f"COVID probability: {prob:.3f}",
        f"Model: {context.get('model_name') or 'SS-VIRULEX fused model'}",
    ]


def build_combined_explanation(
    image_id: str,
    original_path: str,
    sfmov_path: str,
    aware_path: str,
    item: dict,
    dst: Path,
    context: dict,
) -> None:
    if not aware_path or not Path(aware_path).exists():
        return
    heatmap = mpimg.imread(aware_path)
    top_concepts = top_concept_rows(context, image_id, top_n=5)
    prediction = prediction_lines(context, image_id)
    rule = decision_path_lines(context, image_id)

    title_text = "\n".join(wrapped_lines("SS-VIRULEX Explanations: " + image_id, width=105, max_lines=2))
    title_lines = title_text.count("\n") + 1
    grid_top = 0.78 if title_lines > 1 else 0.82

    fig = plt.figure(figsize=(18, 6.5), dpi=180)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 1.0, 1.05],
        left=0.035,
        right=0.985,
        top=grid_top,
        bottom=0.10,
        wspace=0.28,
    )
    fig.suptitle(title_text, fontsize=16, fontweight="bold", color="#0b2e6d", x=0.035, ha="left", y=0.97)

    ax_text = fig.add_subplot(grid[0, 0])
    ax_text.set_title("Final Prediction + Readable Rule", fontsize=12, fontweight="bold", color="#0b2e6d", pad=10)
    ax_text.set_facecolor("#f8fafc")
    for spine in ax_text.spines.values():
        spine.set_edgecolor("#d5dce8")
        spine.set_linewidth(1.0)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    text_lines = prediction + ["", "Readable decision path:"] + rule
    wrapped = []
    for idx, line in enumerate(text_lines):
        width = 60 if idx < 6 else 68
        wrapped.extend(wrapped_lines(line, width=width, max_lines=None))
    ax_text.text(
        0.04,
        0.94,
        "\n".join(wrapped[:26]),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        color="#1f2937",
        linespacing=1.25,
        family="DejaVu Sans",
    )

    ax_heatmap = fig.add_subplot(grid[0, 1])
    pred_row = get_row_by_image_id(context.get("predictions", pd.DataFrame()), image_id)
    pred_text = ""
    if pred_row is not None:
        pred_text = f"Predicted diagnosis: {label_name(pred_row['ss_virulex_decision_tree_prediction'])}"
    ax_heatmap.set_title("Concept-Annotated Heatmap", fontsize=12, fontweight="bold", color="#0b2e6d", pad=10)
    ax_heatmap.imshow(heatmap)
    ax_heatmap.axis("off")
    if pred_text:
        ax_heatmap.text(
            0.5,
            -0.08,
            pred_text,
            transform=ax_heatmap.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#1f2937",
        )

    ax_bar = fig.add_subplot(grid[0, 2])
    ax_bar.set_title("Top Detected Concepts", fontsize=12, fontweight="bold", color="#0b2e6d", pad=10)
    if top_concepts.empty:
        ax_bar.text(0.5, 0.5, "Concept probabilities unavailable", ha="center", va="center", fontsize=10)
        ax_bar.set_axis_off()
    else:
        plot_df = top_concepts.sort_values("score", ascending=True)
        y_pos = np.arange(len(plot_df))
        colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"][: len(plot_df)]
        ax_bar.barh(y_pos, plot_df["score"], color=colors, height=0.68)
        labels = ["\n".join(wrapped_lines(concept, width=22, max_lines=2)) for concept in plot_df["concept"]]
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels, fontsize=8)
        ax_bar.set_xlim(0, 1)
        ax_bar.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax_bar.set_axisbelow(True)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_color("#cbd5e1")
        ax_bar.spines["bottom"].set_color("#cbd5e1")
        ax_bar.tick_params(axis="x", labelsize=8)
        for y, score in zip(y_pos, plot_df["score"]):
            ax_bar.text(min(score + 0.015, 0.985), y, f"{score:.3f}", va="center", fontsize=8, color="#1f2937")

    fig.text(
        0.035,
        0.035,
        f"Concept alpha={float(item.get('concept_alpha', 0.0)):.3f} | Built from existing SS-VIRULEX outputs",
        fontsize=8,
        color="#64748b",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor="white")
    plt.close(fig)


def write_manifest(notes: list[str]) -> None:
    manifest = {
        "source": "current reconstructed SS-VIRULEX artifacts",
        "generated_at": pd.Timestamp.now().isoformat(),
        "notes": notes,
        "paths": {
            "med_micn_trained_artifacts": str(OUT_ROOT / "01_Med-MICN"),
            "med_micn_checkpoint": str(MED_RUN / "best_model.pt"),
            "med_micn_concept_probabilities": str(MED_RUN / "per_image_concept_probabilities.csv"),
            "svis_rulex_trained_artifacts": str(OUT_ROOT / "02_SVIS-RULEX"),
            "svis_rulex_trained_model": str(XAI_RUN / "models" / "03_custom_mobilenetv2_complete_covid_ct.keras"),
        },
    }
    write_json(OUT_ROOT / "current_output_layout_manifest.json", manifest)


def main() -> None:
    ensure_dirs()
    _, augmented_meta = metadata_outputs()
    stat, fused = feature_outputs(augmented_meta)
    concept_outputs()
    med_micn_artifact_outputs()
    svis_rulex_artifact_outputs()
    ranking, selected = build_zfmis(fused)
    all_metrics, svis, med, ss = metrics_outputs()
    three_model_comparison_outputs(all_metrics)
    plotting_outputs(all_metrics, svis, med, ss, ranking)
    model_predictions_and_importance(fused, selected)
    visual_outputs()
    write_manifest(
        [
            "RuleFit per-image prediction models were not saved in the current reconstructed run; predictions_covid.csv contains decision-tree predictions and blank RuleFit prediction columns.",
            "Concept Grad-CAM PNGs were not saved in the current reconstructed run; visual_explanations/heatmaps/concept_gradcam contains a README placeholder rather than fabricated images.",
        ]
    )


if __name__ == "__main__":
    main()
