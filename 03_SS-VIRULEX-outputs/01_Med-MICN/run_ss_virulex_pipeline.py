#!/usr/bin/env python3
"""Master runner for the reconstructed SS-VIRULEX pipeline.

The runner uses the executable pieces already present in the two local
repositories, then adds the glue needed to fuse SVIS-RULEX statistical features
with the required Med-MICN concept, concept-embedding, and neural-symbolic
artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text


DEFAULT_MED_ROOT = Path("/Users/samehissa/Downloads/NeurIPS24-Med_MICN-main")
DEFAULT_DATA_ROOT = Path("/Users/samehissa/Downloads/COVID-CT-Dataset-master")
DEFAULT_CONCEPT_CSV = Path("/Users/samehissa/Downloads/covid_ct_concepts_completed.csv")
DEFAULT_SS_OUTPUT_ROOT = Path("/Users/samehissa/Downloads/SS-VIRULEX-outputs")
DEFAULT_MED_ARTIFACT_ROOT = DEFAULT_SS_OUTPUT_ROOT / "01_Med-MICN"
DEFAULT_XAI_ARTIFACT_ROOT = DEFAULT_SS_OUTPUT_ROOT / "02_SVIS-RULEX"
SELECTED_K = [18, 15, 12, 9, 6, 3]
MED_DIAGNOSIS_RENAME = {
    "task_probability_covid": "med_task_prob_covid",
    "neural_probability_covid": "med_neural_prob_covid",
}
MED_DIAGNOSIS_FEATURES = list(MED_DIAGNOSIS_RENAME.values())


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot be completed."""


@dataclass
class PipelineContext:
    med_root: Path
    med_artifact_root: Path
    xai_root: Path
    data_root: Path
    concept_csv: Path
    med_python: Path
    ss_output_root: Path
    force: bool
    run_med_micn: bool
    skip_rulefit: bool
    skip_sfmov: bool
    force_concept_probabilities: bool
    concept_batch_size: int
    samples_per_class: int
    common: object
    logger: logging.Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reconstructed SS-VIRULEX pipeline.")
    parser.add_argument("--med-root", type=Path, default=DEFAULT_MED_ROOT)
    parser.add_argument(
        "--med-artifact-root",
        type=Path,
        default=None,
        help=(
            "Trained Med-MICN artifact folder consumed by SS-VIRULEX. "
            "Defaults to <ss-output-root>/01_Med-MICN."
        ),
    )
    parser.add_argument(
        "--xai-root",
        type=Path,
        default=DEFAULT_XAI_ARTIFACT_ROOT,
        help=(
            "Trained SVIS-RULEX artifact folder consumed by SS-VIRULEX. "
            "Defaults to <ss-output-root>/02_SVIS-RULEX."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--concept-csv", type=Path, default=DEFAULT_CONCEPT_CSV)
    parser.add_argument("--med-python", type=Path, default=DEFAULT_MED_ROOT / ".venv-py312" / "bin" / "python")
    parser.add_argument(
        "--ss-output-root",
        type=Path,
        default=DEFAULT_SS_OUTPUT_ROOT,
        help="Consolidated SS-VIRULEX output folder outside the source repositories.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute stages even when expected outputs exist.")
    parser.add_argument(
        "--run-med-micn",
        action="store_true",
        help="Rebuild the required Med-MICN concept-embedding and neural-symbolic artifacts.",
    )
    parser.add_argument("--skip-rulefit", action="store_true", help="Skip RuleFit stages.")
    parser.add_argument("--skip-sfmov", action="store_true", help="Skip concept-aware SFMOV generation.")
    parser.add_argument(
        "--force-concept-probabilities",
        action="store_true",
        help="Regenerate Med-MICN per-image concept probabilities even when the CSV already exists.",
    )
    parser.add_argument(
        "--concept-batch-size",
        type=int,
        default=16,
        help="Batch size for Med-MICN concept-probability inference.",
    )
    parser.add_argument("--samples-per-class", type=int, default=4)
    return parser.parse_args()


def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ss_virulex")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "ss_virulex_pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise PipelineError(f"Missing {label}: {path}")


def is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_ss_output_root(ctx: PipelineContext) -> None:
    output_root = ctx.ss_output_root.resolve()
    for label, root in [("Med-MICN repository", ctx.med_root), ("XAI repository", ctx.xai_root)]:
        root_resolved = root.resolve()
        if output_root == root_resolved or is_relative_to_path(output_root, root_resolved):
            raise PipelineError(
                f"SS-VIRULEX output root must be outside the {label}: {ctx.ss_output_root}"
            )


def import_common(xai_root: Path):
    common_path = xai_root / "covid_ct_run" / "exact_sequence" / "scripts" / "common.py"
    require_path(common_path, "XAI exact-sequence common.py")
    spec = importlib.util.spec_from_file_location("ss_virulex_xai_common", common_path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Could not import common.py from {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_common_paths(common, xai_root: Path, data_root: Path) -> None:
    run_root = xai_root / "covid_ct_run" / "exact_sequence"
    common.PROJECT_ROOT = xai_root
    common.DATA_ROOT = data_root
    common.RUN_ROOT = run_root
    common.OUTPUT_DIR = run_root / "outputs"
    common.NOTEBOOK_DIR = run_root / "notebooks"
    common.MODEL_DIR = common.OUTPUT_DIR / "models"
    common.FEATURE_DIR = common.OUTPUT_DIR / "features"
    common.RULE_DIR = common.OUTPUT_DIR / "rules"
    common.HEATMAP_DIR = common.OUTPUT_DIR / "heatmaps"
    common.AUGMENTED_DIR = common.OUTPUT_DIR / "augmented_train_images"
    common.PLOT_DIR = common.OUTPUT_DIR / "plots"
    common.MPLCONFIG_DIR = common.OUTPUT_DIR / "_mplconfig"


def outputs_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def run_stage(ctx: PipelineContext, name: str, outputs: list[Path], func: Callable[[], object]) -> object:
    if outputs and not ctx.force and outputs_exist(outputs):
        ctx.logger.info("SKIP %s: expected outputs already exist", name)
        return None

    ctx.logger.info("START %s", name)
    started = time.time()
    try:
        result = func()
    except Exception as exc:
        ctx.logger.exception("FAILED %s", name)
        raise PipelineError(f"{name} failed: {exc}") from exc
    ctx.logger.info("DONE %s in %.1fs", name, time.time() - started)
    return result


def stage_preprocess(ctx: PipelineContext) -> None:
    ctx.common.setup_environment()
    official = ctx.common.build_official_split_table()
    augmented = ctx.common.create_augmented_manifest()
    ctx.logger.info("Official rows: %s", len(official))
    ctx.logger.info("Augmented rows: %s", len(augmented))


def stage_dl_grid_search(ctx: PipelineContext) -> None:
    results, best_params = ctx.common.run_dl_grid_search()
    ctx.logger.info("DL grid-search candidates: %s", len(results))
    ctx.logger.info("Best DL hyperparameters: %s", best_params)


def stage_final_mobilenet(ctx: PipelineContext) -> None:
    metrics = ctx.common.train_final_custom_mobilenet()
    ctx.logger.info(
        "Final MobileNetV2 test metrics: accuracy=%.4f balanced_accuracy=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_weighted"],
    )


def stage_statistical_features(ctx: PipelineContext) -> None:
    feature_sets = ctx.common.run_statistical_features_zfmis()
    ctx.logger.info("Statistical feature sets: %s", {key: len(value) for key, value in feature_sets.items()})


def validate_svis_rulex_required_components(ctx: PipelineContext) -> dict:
    """Validate the trained SVIS-RULEX artifacts consumed by SS-VIRULEX."""
    common = ctx.common
    artifacts = {
        "custom_mobilenet_complete": common.MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras",
        "custom_mobilenet_head": common.MODEL_DIR / "03_custom_mobilenetv2_head.keras",
        "custom_mobilenet_metrics": common.OUTPUT_DIR / "03_custom_mobilenetv2_test_metrics.json",
        "train_statistical_features": common.FEATURE_DIR / "04_train_statistical_features.csv",
        "val_statistical_features": common.FEATURE_DIR / "04_val_statistical_features.csv",
        "test_statistical_features": common.FEATURE_DIR / "04_test_statistical_features.csv",
        "statistical_zfmis_feature_sets": common.FEATURE_DIR / "04_zfmis_feature_sets.json",
        "svis_decision_tree_results": common.OUTPUT_DIR / "05a_gridsearchfortree_results.csv",
        "svis_rulefit_results": common.OUTPUT_DIR / "05b_rulefitgridsearchcode_results.csv",
    }
    missing = [f"{label}: {path}" for label, path in artifacts.items() if not path.exists()]
    if missing:
        raise PipelineError(
            "Required trained SVIS-RULEX artifacts are missing from the configured --xai-root. "
            "Missing: " + "; ".join(missing)
        )

    metrics = json.loads(artifacts["custom_mobilenet_metrics"].read_text(encoding="utf-8"))
    stat_shapes = {}
    for split in ["train", "val", "test"]:
        path = common.FEATURE_DIR / f"04_{split}_statistical_features.csv"
        stat_shapes[split] = {"rows": int(len(pd.read_csv(path)))}

    manifest = {
        "status": "required_components_validated",
        "svis_rulex_artifact_root": str(ctx.xai_root),
        "svis_rulex_outputs": str(common.OUTPUT_DIR),
        "svis_rulex_artifacts": {label: str(path) for label, path in artifacts.items()},
        "custom_mobilenet_metrics": {
            key: metrics.get(key)
            for key in ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc", "epochs_run"]
        },
        "statistical_feature_rows": stat_shapes,
    }
    out_path = common.OUTPUT_DIR / "04_svis_rulex_required_components.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.logger.info(
        "SVIS-RULEX required components validated: custom_mobilenet_accuracy=%.4f",
        float(metrics["accuracy"]),
    )
    return manifest


def warn_missing_concept_generation_scripts(ctx: PipelineContext) -> None:
    searchable = [ctx.med_root, ctx.xai_root]
    names = []
    for root in searchable:
        if root.exists():
            names.extend(path.name.lower() for path in root.rglob("*") if path.is_file())
    has_generation = any("gpt" in name or "biovil" in name or "concept_vocab" in name for name in names)
    if not has_generation:
        ctx.logger.warning(
            "No GPT-4V/BioVIL concept generation script was found in either repository. "
            "Using the completed concept CSV as the materialized output of that screenshot branch."
        )


def validate_concept_csv(ctx: PipelineContext) -> pd.DataFrame:
    require_path(ctx.concept_csv, "completed COVID-CT concept CSV")
    concept_df = pd.read_csv(ctx.concept_csv)
    required = {"image_id", "file_name", "class_name", "true_label", "split", "image_path"}
    missing = required.difference(concept_df.columns)
    if missing:
        raise PipelineError(f"Concept CSV is missing required columns: {sorted(missing)}")
    concept_cols = [col for col in concept_df.columns if col.startswith("concept_") and col.endswith("_label")]
    if not concept_cols:
        raise PipelineError("Concept CSV has no columns matching concept_*_label")
    duplicate_files = concept_df["file_name"].duplicated().sum()
    if duplicate_files:
        raise PipelineError(f"Concept CSV has duplicate file_name values: {duplicate_files}")
    ctx.logger.info("Concept rows: %s; concept columns: %s", len(concept_df), len(concept_cols))
    return concept_df


def med_micn_output_dir(ctx: PipelineContext) -> Path:
    return ctx.med_artifact_root / "outputs" / "covid_ct_med_micn"


def med_micn_source_output_dir(ctx: PipelineContext) -> Path:
    return ctx.med_root / "outputs" / "covid_ct_med_micn"


def med_micn_required_artifacts(ctx: PipelineContext) -> dict[str, Path]:
    output_dir = med_micn_output_dir(ctx)
    return {
        "checkpoint": output_dir / "best_model.pt",
        "training_history": output_dir / "training_history.csv",
        "test_metrics": output_dir / "test_metrics.csv",
        "test_concept_metrics": output_dir / "test_concept_metrics.csv",
        "run_summary": output_dir / "run_summary.json",
    }


def validate_med_micn_required_components(ctx: PipelineContext) -> dict:
    """Validate that Med-MICN's concept embedding and neural-symbolic parts are present."""
    module_paths = {
        "model_wrapper": ctx.med_root / "models.py",
        "concept_embedding_and_reasoning": ctx.med_root / "torch_explain" / "nn" / "concepts.py",
        "logic_metrics": ctx.med_root / "torch_explain" / "logic" / "metrics.py",
    }
    for label, path in module_paths.items():
        require_path(path, f"Med-MICN {label} module")

    concepts_source = module_paths["concept_embedding_and_reasoning"].read_text(encoding="utf-8")
    required_symbols = ["ConceptEmbedding", "ConceptReasoningLayer", "def explain"]
    missing_symbols = [symbol for symbol in required_symbols if symbol not in concepts_source]
    if missing_symbols:
        raise PipelineError(
            "Med-MICN concept/reasoning module is missing required symbols: "
            + ", ".join(missing_symbols)
        )

    artifacts = med_micn_required_artifacts(ctx)
    missing_artifacts = [f"{label}: {path}" for label, path in artifacts.items() if not path.exists()]
    if missing_artifacts:
        raise PipelineError(
            "Required Med-MICN concept-embedding/neural-symbolic artifacts are missing. "
            "Run with --run-med-micn to rebuild them. Missing: "
            + "; ".join(missing_artifacts)
        )

    run_summary = json.loads(artifacts["run_summary"].read_text(encoding="utf-8"))
    test_metrics = run_summary.get("test_metrics", {})
    required_metric_keys = [
        "task_accuracy",
        "neural_accuracy",
        "neural_f1_macro",
        "concept_accuracy",
        "concept_f1_macro",
    ]
    missing_metrics = [key for key in required_metric_keys if key not in test_metrics]
    if missing_metrics:
        raise PipelineError(
            "Med-MICN run_summary.json does not include required concept/neural metrics: "
            + ", ".join(missing_metrics)
        )

    concept_metrics = pd.read_csv(artifacts["test_concept_metrics"])
    expected_concept_metric_cols = {"concept", "accuracy", "f1", "precision", "recall"}
    missing_concept_metric_cols = expected_concept_metric_cols.difference(concept_metrics.columns)
    if missing_concept_metric_cols:
        raise PipelineError(
            "Med-MICN test_concept_metrics.csv is missing columns: "
            + ", ".join(sorted(missing_concept_metric_cols))
        )

    manifest = {
        "status": "required_components_validated",
        "med_micn_source_root": str(ctx.med_root),
        "med_micn_artifact_root": str(ctx.med_artifact_root),
        "med_micn_modules": {label: str(path) for label, path in module_paths.items()},
        "med_micn_artifacts": {label: str(path) for label, path in artifacts.items()},
        "required_components": {
            "concept_embedding_module": "torch_explain.nn.concepts.ConceptEmbedding",
            "neural_symbolic_layer": "torch_explain.nn.concepts.ConceptReasoningLayer",
        },
        "test_metrics": {key: test_metrics.get(key) for key in required_metric_keys},
    }
    out_path = ctx.common.OUTPUT_DIR / "05_med_micn_required_components.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.logger.info(
        "Med-MICN required components validated: concept_accuracy=%.4f neural_accuracy=%.4f",
        float(test_metrics["concept_accuracy"]),
        float(test_metrics["neural_accuracy"]),
    )
    return manifest


def run_med_micn_notebook(ctx: PipelineContext) -> None:
    notebook = ctx.med_root / "Med_MICN_COVID_CT_training.ipynb"
    require_path(notebook, "Med-MICN COVID notebook")
    require_path(ctx.med_python, "Med-MICN Python environment")
    output_dir = ctx.med_root / "outputs" / "executed_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ctx.med_python),
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output-dir",
        str(output_dir),
        "--output",
        "Med_MICN_COVID_CT_training.executed.ipynb",
        "--ExecutePreprocessor.timeout=-1",
    ]
    ctx.logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=ctx.med_root, check=True)


def sync_med_micn_artifacts(ctx: PipelineContext) -> None:
    """Copy freshly trained Med-MICN outputs into the SS-VIRULEX artifact root."""
    source_dir = med_micn_source_output_dir(ctx)
    artifact_dir = med_micn_output_dir(ctx)
    if source_dir.resolve() == artifact_dir.resolve():
        return
    require_path(source_dir, "trained Med-MICN output directory")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "best_model.pt",
        "training_history.csv",
        "test_metrics.csv",
        "test_concept_metrics.csv",
        "run_summary.json",
    ]:
        copy_file_if_exists(source_dir / name, artifact_dir / name, [])
    ctx.logger.info("Synced trained Med-MICN artifacts to %s", artifact_dir)


def stage_concept_branch(ctx: PipelineContext) -> None:
    warn_missing_concept_generation_scripts(ctx)
    validate_concept_csv(ctx)
    if ctx.run_med_micn:
        run_med_micn_notebook(ctx)
        sync_med_micn_artifacts(ctx)
    validate_med_micn_required_components(ctx)


def concept_key_from_manifest_row(row: pd.Series) -> str:
    source_path = row.get("source_filepath")
    if isinstance(source_path, str) and source_path.strip():
        return Path(source_path).name
    return str(row["filename"])


def concept_score_name(label_col: str) -> str:
    stem = label_col.removeprefix("concept_").removesuffix("_label")
    return f"concept_score_{stem}"


def safe_image_id(class_name: str, filename: str) -> str:
    stem = Path(str(filename)).stem
    prefix = "covid" if str(class_name).lower() == "covid" else "noncovid"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return f"{prefix}_{cleaned}"


def med_micn_concept_probabilities_path(ctx: PipelineContext) -> Path:
    return med_micn_output_dir(ctx) / "per_image_concept_probabilities.csv"


def xai_concept_probabilities_path(ctx: PipelineContext) -> Path:
    return ctx.common.FEATURE_DIR / "04_med_micn_concept_probabilities.csv"


def _load_med_micn_checkpoint(path: Path, torch_module):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def _load_torchvision_backbone_for_inference(backbone_name: str):
    from torchvision import models

    name = backbone_name.lower()
    if name == "rn50":
        return models.resnet50(weights=None)
    if name == "densenet":
        return models.densenet169(weights=None)
    if name == "vgg":
        return models.vgg16(weights=None)
    raise PipelineError(f"Unsupported Med-MICN checkpoint backbone: {backbone_name}")


def build_med_micn_from_checkpoint_config(ctx: PipelineContext, config: dict, n_concepts: int):
    import torch.nn as nn

    med_root_str = str(ctx.med_root)
    inserted_med_root = med_root_str not in sys.path
    if inserted_med_root:
        sys.path.insert(0, med_root_str)
    try:
        import torch_explain as te
        from models import Neural_Concat_Model
        from torch_explain.nn.concepts import ConceptReasoningLayer
    finally:
        if inserted_med_root:
            try:
                sys.path.remove(med_root_str)
            except ValueError:
                pass

    backbone_name = config.get("backbone", "RN50")
    embedding_size = int(config.get("embedding_size", 8))
    backbone = _load_torchvision_backbone_for_inference(backbone_name)

    feature_size = 1000
    if backbone_name.lower() == "densenet":
        concept_hidden = 128
        fused_hidden = 64
    elif backbone_name.lower() == "vgg":
        concept_hidden = 10
        fused_hidden = 1000
    else:
        concept_hidden = 10
        fused_hidden = 1000

    concept_encoder = nn.Sequential(
        nn.Linear(feature_size, concept_hidden),
        nn.LeakyReLU(),
        te.nn.ConceptEmbedding(concept_hidden, n_concepts, embedding_size),
    )
    task_neural_predictor = ConceptReasoningLayer(embedding_size, 2)
    task_predictor = nn.Sequential(
        nn.Linear(n_concepts * embedding_size + feature_size, fused_hidden),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(fused_hidden, 2),
    )
    task_concept_predictor = nn.Linear(n_concepts * embedding_size, 2)
    return Neural_Concat_Model(
        backbone,
        concept_encoder,
        task_neural_predictor,
        task_predictor,
        task_concept_predictor,
    )


def choose_torch_device(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def build_med_micn_inference_frame(ctx: PipelineContext, concept_df: pd.DataFrame) -> pd.DataFrame:
    manifest_path = ctx.common.OUTPUT_DIR / "01_augmented_splits.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        required = {"filepath", "filename", "split", "class_name", "label"}
        missing = required.difference(manifest.columns)
        if missing:
            raise PipelineError(f"Augmented split manifest is missing columns: {sorted(missing)}")
        frame = pd.DataFrame(
            {
                "file_name": manifest["filename"],
                "class_name": manifest["class_name"],
                "true_label": manifest["label"].astype(int),
                "split": manifest["split"],
                "image_path": manifest["filepath"],
                "source": manifest.get("source", "original"),
                "source_filepath": manifest.get("source_filepath", ""),
            }
        )
        frame["concept_key"] = manifest.apply(concept_key_from_manifest_row, axis=1)
    else:
        frame = concept_df[["file_name", "class_name", "true_label", "split", "image_path"]].copy()
        frame["source"] = "original"
        frame["source_filepath"] = ""
        frame["concept_key"] = frame["file_name"]

    concept_cols = [col for col in concept_df.columns if col.startswith("concept_") and col.endswith("_label")]
    concept_lookup = concept_df.set_index("file_name")
    missing_keys = sorted(set(frame["concept_key"]) - set(concept_lookup.index))
    if missing_keys:
        raise PipelineError(
            f"{len(missing_keys)} inference rows are missing concept metadata. "
            f"First missing key: {missing_keys[0]}"
        )

    image_ids = []
    for _, row in frame.iterrows():
        if row["file_name"] in concept_lookup.index and "image_id" in concept_lookup.columns:
            image_ids.append(concept_lookup.loc[row["file_name"], "image_id"])
        else:
            image_ids.append(safe_image_id(row["class_name"], row["file_name"]))
    frame.insert(0, "image_id", image_ids)
    for col in concept_cols:
        frame[col] = concept_lookup.loc[frame["concept_key"], col].to_numpy(dtype="int64")

    frame["resolved_image_path"] = [resolve_inference_image_path(ctx, path) for path in frame["image_path"]]
    missing_paths = [
        path
        for path, resolved in zip(frame["image_path"], frame["resolved_image_path"])
        if not Path(str(resolved)).exists()
    ]
    if missing_paths:
        raise PipelineError(f"{len(missing_paths)} Med-MICN inference image paths are missing. First: {missing_paths[0]}")
    return frame


def resolve_inference_image_path(ctx: PipelineContext, path: object) -> str:
    original = Path(str(path))
    if original.exists():
        return str(original)
    parts = original.parts
    if "augmented_train_images" in parts:
        index = parts.index("augmented_train_images")
        candidate = ctx.common.OUTPUT_DIR / Path(*parts[index:])
        if candidate.exists():
            return str(candidate)
    return str(original)


def generate_med_micn_concept_probabilities(ctx: PipelineContext) -> pd.DataFrame:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    artifacts = med_micn_required_artifacts(ctx)
    require_path(artifacts["checkpoint"], "trained Med-MICN checkpoint")
    concept_df = validate_concept_csv(ctx)
    concept_cols = [col for col in concept_df.columns if col.startswith("concept_") and col.endswith("_label")]

    checkpoint = _load_med_micn_checkpoint(artifacts["checkpoint"], torch)
    config = checkpoint.get("config", {})
    checkpoint_concepts = config.get("concept_cols") or concept_cols
    if list(checkpoint_concepts) != concept_cols:
        raise PipelineError(
            "Checkpoint concept columns do not match the concept CSV. "
            f"checkpoint={list(checkpoint_concepts)} csv={concept_cols}"
        )

    model = build_med_micn_from_checkpoint_config(ctx, config, n_concepts=len(concept_cols))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_torch_device(torch)
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor.float() + 1e-6 if tensor.std() == 0 else tensor),
            normalize,
        ]
    )

    frame = build_med_micn_inference_frame(ctx, concept_df)

    class InferenceDataset(Dataset):
        def __init__(self, rows: pd.DataFrame):
            self.rows = rows.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            row = self.rows.iloc[idx]
            image = Image.open(row["resolved_image_path"]).convert("RGB")
            return idx, transform(image)

    loader = DataLoader(
        InferenceDataset(frame),
        batch_size=max(1, ctx.concept_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    concept_probs = np.zeros((len(frame), len(concept_cols)), dtype="float32")
    task_probs = np.zeros((len(frame), 2), dtype="float32")
    neural_probs_out = np.zeros((len(frame), 2), dtype="float32")

    with torch.no_grad():
        total_batches = len(loader)
        for batch_index, (indices, images) in enumerate(loader, start=1):
            images = images.to(device)
            _, task_logits, neural_probs, _, batch_concept_probs, _ = model(images)
            batch_task_probs = torch.softmax(task_logits, dim=1)
            idx = indices.cpu().numpy()
            concept_probs[idx] = batch_concept_probs.detach().cpu().numpy()
            task_probs[idx] = batch_task_probs.detach().cpu().numpy()
            neural_probs_out[idx] = neural_probs.detach().cpu().numpy()
            if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
                ctx.logger.info(
                    "Med-MICN concept inference batch %s/%s rows=%s/%s",
                    batch_index,
                    total_batches,
                    min(batch_index * max(1, ctx.concept_batch_size), len(frame)),
                    len(frame),
                )

    result = frame.copy()
    result["task_prediction"] = np.argmax(task_probs, axis=1)
    result["task_probability_noncovid"] = task_probs[:, 0]
    result["task_probability_covid"] = task_probs[:, 1]
    result["neural_prediction"] = np.argmax(neural_probs_out, axis=1)
    result["neural_probability_noncovid"] = neural_probs_out[:, 0]
    result["neural_probability_covid"] = neural_probs_out[:, 1]
    for idx, label_col in enumerate(concept_cols):
        score_col = concept_score_name(label_col)
        result[score_col] = concept_probs[:, idx]
        result[score_col.replace("concept_score_", "concept_pred_") + "_label"] = (concept_probs[:, idx] >= 0.5).astype(int)

    id_cols = [
        "image_id",
        "file_name",
        "class_name",
        "true_label",
        "split",
        "image_path",
        "resolved_image_path",
        "source",
        "source_filepath",
        "concept_key",
    ]
    model_cols = [
        "task_prediction",
        "task_probability_noncovid",
        "task_probability_covid",
        "neural_prediction",
        "neural_probability_noncovid",
        "neural_probability_covid",
    ]
    score_cols = [concept_score_name(col) for col in concept_cols]
    pred_cols = [col.replace("concept_score_", "concept_pred_") + "_label" for col in score_cols]
    result = result[id_cols + concept_cols + model_cols + score_cols + pred_cols]

    med_path = med_micn_concept_probabilities_path(ctx)
    xai_path = xai_concept_probabilities_path(ctx)
    med_path.parent.mkdir(parents=True, exist_ok=True)
    xai_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(med_path, index=False)
    result.to_csv(xai_path, index=False)

    summary_path = artifacts["run_summary"]
    run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    run_summary["per_image_concept_probabilities_csv"] = str(med_path)
    run_summary["per_image_concept_probabilities_rows"] = int(len(result))
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    ctx.logger.info(
        "Saved Med-MICN per-image concept probabilities: rows=%s path=%s device=%s",
        len(result),
        med_path,
        device,
    )
    return result


def load_med_micn_concept_probabilities(ctx: PipelineContext) -> pd.DataFrame:
    path = xai_concept_probabilities_path(ctx)
    if not path.exists():
        path = med_micn_concept_probabilities_path(ctx)
    require_path(path, "Med-MICN per-image concept probabilities")
    probs = pd.read_csv(path)
    score_cols = [col for col in probs.columns if col.startswith("concept_score_")]
    if not score_cols:
        raise PipelineError(f"Med-MICN concept probability file has no concept_score_* columns: {path}")
    return probs


def build_combined_feature_pool(ctx: PipelineContext) -> dict[str, list[str]]:
    common = ctx.common
    concept_df = validate_concept_csv(ctx)
    concept_cols = [col for col in concept_df.columns if col.startswith("concept_") and col.endswith("_label")]
    probability_df = load_med_micn_concept_probabilities(ctx)
    concept_score_cols = [concept_score_name(col) for col in concept_cols]
    missing_score_cols = [col for col in concept_score_cols if col not in probability_df.columns]
    if missing_score_cols:
        raise PipelineError(
            "Med-MICN concept probability file is missing required columns: "
            + ", ".join(missing_score_cols)
        )
    missing_diagnosis_cols = [col for col in MED_DIAGNOSIS_RENAME if col not in probability_df.columns]
    if missing_diagnosis_cols:
        raise PipelineError(
            "Med-MICN probability file is missing required diagnosis columns: "
            + ", ".join(missing_diagnosis_cols)
            + ". Regenerate it with --force-concept-probabilities after installing torch in the runner environment."
        )

    manifest_path = common.OUTPUT_DIR / "01_augmented_splits.csv"
    require_path(manifest_path, "augmented split manifest")
    manifest = pd.read_csv(manifest_path)
    probability_lookup = (
        probability_df.set_index("image_path")[concept_score_cols + list(MED_DIAGNOSIS_RENAME)]
        .astype(float)
        .rename(columns=MED_DIAGNOSIS_RENAME)
    )
    missing_keys = sorted(set(manifest["filepath"].astype(str)) - set(probability_lookup.index))
    if missing_keys:
        raise PipelineError(
            f"{len(missing_keys)} images in the split manifest are missing Med-MICN concept probabilities. "
            f"First missing key: {missing_keys[0]}"
        )

    combined_frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        stats_path = common.FEATURE_DIR / f"04_{split}_statistical_features.csv"
        require_path(stats_path, f"{split} statistical features")
        stats = pd.read_csv(stats_path).reset_index(drop=True)
        split_manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(stats) != len(split_manifest):
            raise PipelineError(
                f"Row-count mismatch for {split}: stats={len(stats)} manifest={len(split_manifest)}"
            )

        concept_block = probability_lookup.loc[split_manifest["filepath"].astype(str)].reset_index(drop=True)
        combined = pd.concat(
            [
                stats.drop(columns=["label"]).reset_index(drop=True),
                concept_block.reset_index(drop=True),
                stats["label"].astype(int).reset_index(drop=True),
            ],
            axis=1,
        )
        out_path = common.FEATURE_DIR / f"04_{split}_combined_stat_concept_features.csv"
        combined.to_csv(out_path, index=False)
        combined_frames[split] = combined
        ctx.logger.info("Saved %s combined rows to %s", len(combined), out_path)

    feature_columns = [col for col in combined_frames["train"].columns if col != "label"]
    zero_fraction = (combined_frames["train"][feature_columns] == 0).mean().sort_values(ascending=False)
    zero_fraction_path = common.FEATURE_DIR / "04_combined_feature_zero_fraction.csv"
    zero_fraction.to_csv(zero_fraction_path, header=["zero_fraction"])
    filtered_columns = zero_fraction[zero_fraction <= 0.50].index.tolist() or feature_columns

    feature_sets: dict[str, list[str]] = {
        "all": feature_columns,
        "filtered": filtered_columns,
        "statistical_all": [
            col for col in feature_columns if not col.startswith("concept_score_") and not col.startswith("med_")
        ],
        "concept_all": [col for col in feature_columns if col.startswith("concept_score_")],
        "med_diagnosis_all": [col for col in MED_DIAGNOSIS_FEATURES if col in feature_columns],
    }
    X_train = combined_frames["train"][filtered_columns]
    y_train = combined_frames["train"]["label"].values
    for k in SELECTED_K:
        actual_k = min(k, len(filtered_columns))
        selector = SelectKBest(
            score_func=lambda x, y: mutual_info_classif(x, y, random_state=common.SEED),
            k=actual_k,
        )
        selector.fit(X_train, y_train)
        selected = list(np.array(filtered_columns)[selector.get_support()])
        feature_sets[f"selected_{k}"] = selected
        out_dir = common.FEATURE_DIR / f"04_combined_selected_{k}_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, frame in combined_frames.items():
            frame[selected + ["label"]].to_csv(out_dir / f"{split}_combined_selected_{k}_features.csv", index=False)

    feature_sets_path = common.FEATURE_DIR / "04_combined_zfmis_feature_sets.json"
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    ctx.logger.info("Combined feature sets: %s", {key: len(value) for key, value in feature_sets.items()})
    return feature_sets


def load_combined_frames(ctx: PipelineContext) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ["train", "val", "test"]:
        path = ctx.common.FEATURE_DIR / f"04_{split}_combined_stat_concept_features.csv"
        require_path(path, f"{split} combined stat-plus-concept features")
        frames[split] = pd.read_csv(path)
    return frames


def load_combined_feature_sets(ctx: PipelineContext) -> dict[str, list[str]]:
    path = ctx.common.FEATURE_DIR / "04_combined_zfmis_feature_sets.json"
    require_path(path, "combined ZFMIS feature-set JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def probability_metrics(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = np.argmax(probs, axis=1)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def threshold_metrics(name: str, y_true: np.ndarray, prob_positive: np.ndarray, threshold: float) -> dict:
    y_pred = (prob_positive >= threshold).astype("int32")
    probs = np.column_stack([1.0 - prob_positive, prob_positive])
    metrics = {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def choose_threshold_by_balanced_accuracy(y_true: np.ndarray, prob_positive: np.ndarray) -> tuple[float, dict]:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(prob_positive, np.linspace(0.02, 0.98, 49)),
                prob_positive,
            ]
        )
    )
    best_threshold = 0.5
    best_metrics: dict | None = None
    for threshold in candidates:
        metrics = threshold_metrics("threshold_candidate", y_true, prob_positive, float(threshold))
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        key = (metrics["balanced_accuracy"], metrics["f1_weighted"], metrics["accuracy"])
        best_key = (best_metrics["balanced_accuracy"], best_metrics["f1_weighted"], best_metrics["accuracy"])
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = threshold_metrics("threshold_candidate", y_true, prob_positive, 0.5)
    return best_threshold, best_metrics


def combined_strong_model_specs(seed: int):
    return [
        (
            "random_forest",
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [200],
                "max_depth": [None, 6],
                "min_samples_leaf": [1, 3],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [300],
                "max_depth": [None, 6],
                "min_samples_leaf": [1, 3],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2],
            },
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=seed),
            {
                "max_iter": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [7],
            },
        ),
        (
            "svm_rbf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(random_state=seed, probability=True)),
                ]
            ),
            {
                "svc__C": [0.3, 1.0, 3.0],
                "svc__gamma": ["scale"],
                "svc__class_weight": [None, "balanced"],
            },
        ),
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logistic", LogisticRegression(random_state=seed, max_iter=2000, solver="liblinear")),
                ]
            ),
            {
                "logistic__C": [0.1, 1.0, 10.0],
                "logistic__class_weight": [None, "balanced"],
            },
        ),
    ]


def combined_strong_estimator_by_name(seed: int) -> dict[str, object]:
    return {name: estimator for name, estimator, _ in combined_strong_model_specs(seed)}


def evaluate_combined_tree_grid(ctx: PipelineContext, feature_set_names: list[str] | None = None) -> pd.DataFrame:
    common = ctx.common
    frames = load_combined_frames(ctx)
    feature_sets = load_combined_feature_sets(ctx)
    if feature_set_names is None:
        feature_set_names = ["all", "filtered", "selected_18", "selected_15", "selected_12", "selected_9", "selected_6", "selected_3"]

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 8, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
    }
    rows = []
    for name in feature_set_names:
        columns = feature_sets[name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values

        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=common.SEED),
            param_grid=param_grid,
            cv=5,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        clf = DecisionTreeClassifier(random_state=common.SEED, **grid.best_params_)
        clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        probs = clf.predict_proba(X_test)
        metrics = probability_metrics(f"combined_decision_tree_{name}", y_test, probs)

        model_path = common.MODEL_DIR / f"05c_combined_decision_tree_{name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(clf, handle)
        rules_path = common.RULE_DIR / f"05c_combined_decision_tree_{name}_rules.txt"
        rules_path.write_text(export_text(clf, feature_names=columns), encoding="utf-8")

        row = {
            "feature_set": name,
            "n_features": len(columns),
            "best_params": json.dumps(grid.best_params_),
            "cv_balanced_accuracy": float(grid.best_score_),
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_f1_weighted": metrics["f1_weighted"],
            "test_roc_auc": metrics["roc_auc"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "model_path": str(model_path),
            "rules_path": str(rules_path),
        }
        rows.append(row)
        ctx.logger.info("Combined DT %s: balanced_accuracy=%.4f", name, row["test_balanced_accuracy"])

    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(common.OUTPUT_DIR / "05c_combined_gridsearchfortree_results.csv", index=False)
    return results


def evaluate_combined_rulefit_grid(ctx: PipelineContext, feature_set_names: list[str] | None = None) -> pd.DataFrame:
    try:
        from rulefit import RuleFit
    except Exception as exc:
        raise PipelineError("RuleFit is not importable. Install rulefit or rerun with --skip-rulefit.") from exc

    common = ctx.common
    frames = load_combined_frames(ctx)
    feature_sets = load_combined_feature_sets(ctx)
    if feature_set_names is None:
        feature_set_names = ["selected_3", "selected_6"]

    param_grid = [
        {"tree_size": tree_size, "sample_fract": sample_fract, "max_rules": max_rules}
        for tree_size in [3, 4]
        for sample_fract in [0.8]
        for max_rules in [50, 100]
    ]
    rows = []
    for name in feature_set_names:
        columns = feature_sets[name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values

        best = None
        for params in param_grid:
            started = time.time()
            rf = RuleFit(rfmode="classify", model_type="r", random_state=common.SEED, **params)
            rf.fit(X_train, y_train, feature_names=columns)
            val_pred = rf.predict(X_val).astype("int32")
            candidate = {
                **params,
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
                "seconds": time.time() - started,
            }
            if best is None or candidate["val_balanced_accuracy"] > best["val_balanced_accuracy"]:
                best = candidate

        final_rf = RuleFit(
            rfmode="classify",
            model_type="r",
            tree_size=int(best["tree_size"]),
            sample_fract=float(best["sample_fract"]),
            max_rules=int(best["max_rules"]),
            random_state=common.SEED,
        )
        final_rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]), feature_names=columns)
        test_pred = final_rf.predict(X_test).astype("int32")
        rules = final_rf.get_rules()
        rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
        rules_path = common.RULE_DIR / f"05d_combined_rulefit_{name}_rules.csv"
        rules.to_csv(rules_path, index=False)

        row = {
            "feature_set": name,
            "n_features": len(columns),
            "best_params": json.dumps({key: best[key] for key in ["tree_size", "sample_fract", "max_rules"]}),
            "val_balanced_accuracy": best["val_balanced_accuracy"],
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
            "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
            "confusion_matrix": json.dumps(confusion_matrix(y_test, test_pred).tolist()),
            "n_nonzero_rules": int(len(rules)),
            "rules_path": str(rules_path),
        }
        rows.append(row)
        ctx.logger.info("Combined RuleFit %s: balanced_accuracy=%.4f", name, row["test_balanced_accuracy"])

    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(common.OUTPUT_DIR / "05d_combined_rulefitgridsearchcode_results.csv", index=False)
    return results


def evaluate_combined_strong_classifier_grid(
    ctx: PipelineContext,
    feature_set_names: list[str] | None = None,
) -> pd.DataFrame:
    common = ctx.common
    frames = load_combined_frames(ctx)
    feature_sets = load_combined_feature_sets(ctx)
    if feature_set_names is None:
        feature_set_names = ["all", "filtered", "selected_18", "selected_9", "selected_6", "med_diagnosis_all"]
    model_specs = combined_strong_model_specs(common.SEED)

    rows = []
    for feature_set_name in feature_set_names:
        if feature_set_name not in feature_sets or not feature_sets[feature_set_name]:
            continue

        columns = feature_sets[feature_set_name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values

        for classifier_name, estimator, param_grid in model_specs:
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=3,
                scoring="balanced_accuracy",
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            val_probs = grid.best_estimator_.predict_proba(X_val)
            val_metrics = probability_metrics(f"combined_{classifier_name}_{feature_set_name}", y_val, val_probs)

            final_model = grid.best_estimator_
            final_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
            test_probs = final_model.predict_proba(X_test)
            test_metrics = probability_metrics(f"combined_{classifier_name}_{feature_set_name}", y_test, test_probs)

            model_path = common.MODEL_DIR / f"05e_combined_{classifier_name}_{feature_set_name}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(final_model, handle)

            row = {
                "classifier": classifier_name,
                "feature_set": feature_set_name,
                "n_features": len(columns),
                "best_params": json.dumps(grid.best_params_),
                "cv_balanced_accuracy": float(grid.best_score_),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_roc_auc": test_metrics["roc_auc"],
                "confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
                "model_path": str(model_path),
            }
            rows.append(row)
            ctx.logger.info(
                "Combined %s %s: val_balanced_accuracy=%.4f test_balanced_accuracy=%.4f",
                classifier_name,
                feature_set_name,
                row["val_balanced_accuracy"],
                row["test_balanced_accuracy"],
            )

    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(common.OUTPUT_DIR / "05e_combined_stronger_classifiers_results.csv", index=False)
    return results


def evaluate_combined_calibrated_threshold_grid(ctx: PipelineContext) -> pd.DataFrame:
    common = ctx.common
    frames = load_combined_frames(ctx)
    feature_sets = load_combined_feature_sets(ctx)
    strong_path = common.OUTPUT_DIR / "05e_combined_stronger_classifiers_results.csv"
    require_path(strong_path, "combined stronger-classifier results")
    strong_results = pd.read_csv(strong_path)
    estimator_templates = combined_strong_estimator_by_name(common.SEED)

    rows = []
    for _, candidate in strong_results.iterrows():
        classifier_name = str(candidate["classifier"])
        feature_set_name = str(candidate["feature_set"])
        if classifier_name not in estimator_templates or feature_set_name not in feature_sets:
            continue

        columns = feature_sets[feature_set_name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values

        estimator = clone(estimator_templates[classifier_name])
        estimator.set_params(**json.loads(candidate["best_params"]))
        estimator.fit(X_train, y_train)

        uncalibrated_val_probs = estimator.predict_proba(X_val)
        uncalibrated_val_metrics = probability_metrics(
            f"combined_uncalibrated_{classifier_name}_{feature_set_name}",
            y_val,
            uncalibrated_val_probs,
        )

        calibrated_model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        calibrated_model.fit(X_val, y_val)
        val_probs = calibrated_model.predict_proba(X_val)
        threshold, val_tuned_metrics = choose_threshold_by_balanced_accuracy(y_val, val_probs[:, 1])

        test_probs = calibrated_model.predict_proba(X_test)
        default_test_metrics = probability_metrics(
            f"combined_calibrated_{classifier_name}_{feature_set_name}",
            y_test,
            test_probs,
        )
        tuned_test_metrics = threshold_metrics(
            f"combined_calibrated_threshold_{classifier_name}_{feature_set_name}",
            y_test,
            test_probs[:, 1],
            threshold,
        )

        model_path = common.MODEL_DIR / f"05f_combined_calibrated_{classifier_name}_{feature_set_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(calibrated_model, handle)

        row = {
            "classifier": classifier_name,
            "feature_set": feature_set_name,
            "n_features": len(columns),
            "best_params": candidate["best_params"],
            "calibration_method": "sigmoid",
            "threshold_selection": "validation_balanced_accuracy",
            "threshold": threshold,
            "uncalibrated_val_balanced_accuracy": uncalibrated_val_metrics["balanced_accuracy"],
            "calibrated_val_accuracy": val_tuned_metrics["accuracy"],
            "calibrated_val_balanced_accuracy": val_tuned_metrics["balanced_accuracy"],
            "calibrated_val_f1_weighted": val_tuned_metrics["f1_weighted"],
            "calibrated_val_roc_auc": val_tuned_metrics["roc_auc"],
            "test_default_accuracy": default_test_metrics["accuracy"],
            "test_default_balanced_accuracy": default_test_metrics["balanced_accuracy"],
            "test_accuracy": tuned_test_metrics["accuracy"],
            "test_balanced_accuracy": tuned_test_metrics["balanced_accuracy"],
            "test_f1_weighted": tuned_test_metrics["f1_weighted"],
            "test_roc_auc": tuned_test_metrics["roc_auc"],
            "confusion_matrix": json.dumps(tuned_test_metrics["confusion_matrix"]),
            "model_path": str(model_path),
        }
        rows.append(row)
        ctx.logger.info(
            "Calibrated %s %s: threshold=%.4f val_balanced_accuracy=%.4f test_balanced_accuracy=%.4f",
            classifier_name,
            feature_set_name,
            threshold,
            row["calibrated_val_balanced_accuracy"],
            row["test_balanced_accuracy"],
        )

    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(common.OUTPUT_DIR / "05f_combined_calibrated_threshold_results.csv", index=False)
    return results


def extract_combined_final_rules(ctx: PipelineContext) -> dict:
    common = ctx.common
    tree_path = common.OUTPUT_DIR / "05c_combined_gridsearchfortree_results.csv"
    require_path(tree_path, "combined Decision Tree results")
    tree_results = pd.read_csv(tree_path)
    best_tree = tree_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
    tree_rules = Path(best_tree["rules_path"]).read_text(encoding="utf-8")
    tree_out = common.RULE_DIR / "06_combined_best_decision_tree_rules.txt"
    tree_out.write_text(tree_rules, encoding="utf-8")

    summary = {
        "best_combined_decision_tree": best_tree,
        "combined_decision_tree_rules": str(tree_out),
    }
    rulefit_path = common.OUTPUT_DIR / "05d_combined_rulefitgridsearchcode_results.csv"
    if rulefit_path.exists():
        rulefit_results = pd.read_csv(rulefit_path)
        best_rulefit = rulefit_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
        rulefit_rules = pd.read_csv(best_rulefit["rules_path"])
        rulefit_out = common.RULE_DIR / "06_combined_best_rulefit_rules.csv"
        rulefit_rules.to_csv(rulefit_out, index=False)
        summary["best_combined_rulefit"] = best_rulefit
        summary["combined_rulefit_rules"] = str(rulefit_out)

    strong_path = common.OUTPUT_DIR / "05e_combined_stronger_classifiers_results.csv"
    if strong_path.exists():
        strong_results = pd.read_csv(strong_path)
        best_strong = strong_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
        summary["best_combined_stronger_classifier"] = best_strong

    calibrated_path = common.OUTPUT_DIR / "05f_combined_calibrated_threshold_results.csv"
    if calibrated_path.exists():
        calibrated_results = pd.read_csv(calibrated_path)
        best_calibrated = calibrated_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
        summary["best_combined_calibrated_threshold_classifier"] = best_calibrated

    out_path = common.OUTPUT_DIR / "06_combined_rule_extraction_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def make_concept_overlay(raw_rgb: np.ndarray, heatmap: np.ndarray, alpha: float):
    import cv2
    from PIL import Image

    heatmap_u8 = (normalize_map(heatmap) * 255).astype("uint8")
    heatmap_resized = cv2.resize(heatmap_u8, tuple(raw_rgb.shape[:2][::-1]))
    color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    opacity = float(np.clip(0.25 + 0.40 * alpha, 0.25, 0.65))
    blended = cv2.addWeighted(raw_rgb.astype("uint8"), 1.0 - opacity, color, opacity, 0)
    return Image.fromarray(blended)


def run_concept_aware_sfmov(ctx: PipelineContext) -> list[dict]:
    from pathlib import Path as LocalPath

    from PIL import Image
    from scipy.stats import entropy, skew

    common = ctx.common
    concept_df = validate_concept_csv(ctx)
    concept_cols = [col for col in concept_df.columns if col.startswith("concept_") and col.endswith("_label")]
    concept_score_cols = [concept_score_name(col) for col in concept_cols]
    probability_df = load_med_micn_concept_probabilities(ctx)
    concept_lookup = probability_df.set_index("image_path")[concept_score_cols].astype(float)

    tf = common.import_tensorflow()
    model = tf.keras.models.load_model(common.MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras")
    backbone = model.get_layer("mobilenetv2_backbone")
    conv_model = tf.keras.Model(inputs=backbone.input, outputs=backbone.get_layer("Conv_1_bn").output)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("xai_feature_layer").output)

    splits = common.load_official_split_table()
    test_df = splits[splits["split"] == "test"].reset_index(drop=True)
    sample_df = test_df.groupby("class_name", group_keys=False).sample(
        n=ctx.samples_per_class,
        random_state=common.SEED,
    )

    out_dir = common.HEATMAP_DIR / "concept_aware"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for _, row in sample_df.iterrows():
        concept_scores = concept_lookup.loc[row["filepath"]].to_numpy(dtype="float64")
        concept_alpha = float(np.clip(np.mean(concept_scores), 0.0, 1.0))
        raw = np.asarray(Image.open(row["filepath"]).convert("RGB").resize(common.IMG_SIZE), dtype="float32")
        batch = np.expand_dims(raw, axis=0)
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch.copy())
        conv = conv_model.predict(preprocessed, verbose=0)[0]
        dense = feature_model.predict(batch, verbose=0)[0]

        mean_map = np.mean(conv, axis=-1)
        skewness_map = np.zeros(conv.shape[:2], dtype="float64")
        entropy_map = np.zeros(conv.shape[:2], dtype="float64")
        for channel in range(conv.shape[-1]):
            fmap = conv[:, :, channel]
            skewness_map += float(np.nan_to_num(skew(fmap.ravel()), nan=0.0)) * fmap
            entropy_map += float(entropy(np.abs(fmap.ravel()) + 1e-6)) * fmap

        dense_weights = np.array(
            [
                abs(float(np.mean(dense))),
                abs(float(np.nan_to_num(skew(dense), nan=0.0))),
                abs(float(entropy(np.abs(dense) + 1e-6))),
            ],
            dtype="float64",
        )
        dense_weights = dense_weights / (dense_weights.sum() + 1e-8)
        combined = concept_alpha * (
            dense_weights[0] * mean_map
            + dense_weights[1] * skewness_map
            + dense_weights[2] * entropy_map
        )

        safe_stem = LocalPath(row["filename"]).stem.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"{row['class_name']}_{safe_stem}_concept_aware_combined.png"
        make_concept_overlay(raw, combined, concept_alpha).save(out_path)
        saved.append(
            {
                "image": str(row["filepath"]),
                "heatmap": str(out_path),
                "class_name": row["class_name"],
                "concept_alpha": concept_alpha,
                "weights": {
                    "mean": float(dense_weights[0]),
                    "skewness": float(dense_weights[1]),
                    "entropy": float(dense_weights[2]),
                },
            }
        )

    out_json = common.OUTPUT_DIR / "07_concept_aware_sfmov_heatmap_files.json"
    out_json.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def write_final_summary(ctx: PipelineContext) -> None:
    common = ctx.common
    rows = []
    custom_metrics_path = common.OUTPUT_DIR / "03_custom_mobilenetv2_test_metrics.json"
    if custom_metrics_path.exists():
        custom_metrics = json.loads(custom_metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": "custom_mobilenetv2_complete",
                "accuracy": custom_metrics.get("accuracy"),
                "balanced_accuracy": custom_metrics.get("balanced_accuracy"),
                "f1_weighted": custom_metrics.get("f1_weighted"),
                "roc_auc": custom_metrics.get("roc_auc"),
            }
        )
    tree_path = common.OUTPUT_DIR / "05c_combined_gridsearchfortree_results.csv"
    if tree_path.exists():
        tree = pd.read_csv(tree_path).iloc[0]
        rows.append(
            {
                "stage": f"combined_decision_tree_{tree['feature_set']}",
                "accuracy": tree["test_accuracy"],
                "balanced_accuracy": tree["test_balanced_accuracy"],
                "f1_weighted": tree["test_f1_weighted"],
                "roc_auc": tree.get("test_roc_auc"),
            }
        )
    rulefit_path = common.OUTPUT_DIR / "05d_combined_rulefitgridsearchcode_results.csv"
    if rulefit_path.exists():
        rulefit = pd.read_csv(rulefit_path).iloc[0]
        rows.append(
            {
                "stage": f"combined_rulefit_{rulefit['feature_set']}",
                "accuracy": rulefit["test_accuracy"],
                "balanced_accuracy": rulefit["test_balanced_accuracy"],
                "f1_weighted": rulefit["test_f1_weighted"],
                "roc_auc": None,
            }
        )
    strong_path = common.OUTPUT_DIR / "05e_combined_stronger_classifiers_results.csv"
    if strong_path.exists():
        strong = pd.read_csv(strong_path).iloc[0]
        rows.append(
            {
                "stage": f"combined_{strong['classifier']}_{strong['feature_set']}",
                "accuracy": strong["test_accuracy"],
                "balanced_accuracy": strong["test_balanced_accuracy"],
                "f1_weighted": strong["test_f1_weighted"],
                "roc_auc": strong.get("test_roc_auc"),
            }
        )
    calibrated_path = common.OUTPUT_DIR / "05f_combined_calibrated_threshold_results.csv"
    if calibrated_path.exists():
        calibrated = pd.read_csv(calibrated_path).iloc[0]
        rows.append(
            {
                "stage": f"combined_calibrated_threshold_{calibrated['classifier']}_{calibrated['feature_set']}",
                "accuracy": calibrated["test_accuracy"],
                "balanced_accuracy": calibrated["test_balanced_accuracy"],
                "f1_weighted": calibrated["test_f1_weighted"],
                "roc_auc": calibrated.get("test_roc_auc"),
            }
        )

    summary_df = pd.DataFrame(rows)
    metrics_path = common.OUTPUT_DIR / "ss_virulex_final_metrics_summary.csv"
    summary_df.to_csv(metrics_path, index=False)
    summary = {
        "xai_root": str(ctx.xai_root),
        "med_root": str(ctx.med_root),
        "med_artifact_root": str(ctx.med_artifact_root),
        "svis_rulex_artifact_root": str(ctx.xai_root),
        "svis_rulex_trained_model": str(common.MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras"),
        "med_micn_trained_checkpoint": str(med_micn_required_artifacts(ctx)["checkpoint"]),
        "med_micn_concept_probabilities": str(med_micn_concept_probabilities_path(ctx)),
        "data_root": str(ctx.data_root),
        "concept_csv": str(ctx.concept_csv),
        "ss_output_root": str(ctx.ss_output_root),
        "metrics_summary": str(metrics_path),
        "med_micn_required_components": str(common.OUTPUT_DIR / "05_med_micn_required_components.json"),
        "combined_features": str(common.FEATURE_DIR / "04_combined_zfmis_feature_sets.json"),
        "combined_rules": str(common.OUTPUT_DIR / "06_combined_rule_extraction_summary.json"),
        "concept_aware_sfmov": str(common.OUTPUT_DIR / "07_concept_aware_sfmov_heatmap_files.json"),
    }
    (common.OUTPUT_DIR / "ss_virulex_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def copy_file_if_exists(src: Path, dst: Path, copied: list[dict]) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"type": "file", "source": str(src), "destination": str(dst)})


def copy_dir_if_exists(src: Path, dst: Path, copied: list[dict]) -> None:
    if not src.exists() or not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    copied.append({"type": "directory", "source": str(src), "destination": str(dst)})


def copy_globbed_files(src_dir: Path, pattern: str, dst_dir: Path, copied: list[dict]) -> None:
    if not src_dir.exists():
        return
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            copy_file_if_exists(src, dst_dir / src.name, copied)


def export_consolidated_outputs(ctx: PipelineContext) -> dict:
    validate_ss_output_root(ctx)
    common = ctx.common
    out_root = ctx.ss_output_root
    out_root.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    root_files = [
        "ss_virulex_pipeline_summary.json",
        "ss_virulex_final_metrics_summary.csv",
        "04_svis_rulex_required_components.json",
        "05_med_micn_required_components.json",
        "05c_combined_gridsearchfortree_results.csv",
        "05d_combined_rulefitgridsearchcode_results.csv",
        "05e_combined_stronger_classifiers_results.csv",
        "05f_combined_calibrated_threshold_results.csv",
        "06_combined_rule_extraction_summary.json",
        "07_concept_aware_sfmov_heatmap_files.json",
    ]
    for name in root_files:
        copy_file_if_exists(common.OUTPUT_DIR / name, out_root / "summaries" / name, copied)

    feature_files = [
        "04_med_micn_concept_probabilities.csv",
        "04_train_combined_stat_concept_features.csv",
        "04_val_combined_stat_concept_features.csv",
        "04_test_combined_stat_concept_features.csv",
        "04_combined_feature_zero_fraction.csv",
        "04_combined_zfmis_feature_sets.json",
    ]
    for name in feature_files:
        copy_file_if_exists(common.FEATURE_DIR / name, out_root / "features" / name, copied)
    copy_file_if_exists(
        med_micn_concept_probabilities_path(ctx),
        out_root / "concept_branch" / "probabilities" / "covid_concept_probabilities.csv",
        copied,
    )
    for selected_dir in sorted(common.FEATURE_DIR.glob("04_combined_selected_*_features")):
        copy_dir_if_exists(selected_dir, out_root / "features" / selected_dir.name, copied)

    copy_globbed_files(common.RULE_DIR, "05c_combined_*", out_root / "rules", copied)
    copy_globbed_files(common.RULE_DIR, "05d_combined_*", out_root / "rules", copied)
    copy_globbed_files(common.RULE_DIR, "06_combined_*", out_root / "rules", copied)
    copy_globbed_files(common.MODEL_DIR, "05c_combined_*", out_root / "models", copied)
    copy_globbed_files(common.MODEL_DIR, "05d_combined_*", out_root / "models", copied)
    copy_globbed_files(common.MODEL_DIR, "05e_combined_*", out_root / "models", copied)
    copy_globbed_files(common.MODEL_DIR, "05f_combined_*", out_root / "models", copied)

    copy_dir_if_exists(common.HEATMAP_DIR / "concept_aware", out_root / "heatmaps" / "concept_aware", copied)
    copy_file_if_exists(
        ctx.med_artifact_root / "outputs" / "ss_virulex_logs" / "ss_virulex_pipeline.log",
        out_root / "logs" / "ss_virulex_pipeline.log",
        copied,
    )

    manifest = {
        "ss_output_root": str(out_root),
        "source_cache_root": str(common.OUTPUT_DIR),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "copied_count": len(copied),
        "copied": copied,
    }
    manifest_path = out_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.logger.info("Consolidated SS-VIRULEX outputs exported to %s", out_root)
    return manifest


def main() -> int:
    args = parse_args()
    med_artifact_root = args.med_artifact_root or args.ss_output_root / "01_Med-MICN"
    logger = configure_logging(med_artifact_root / "outputs" / "ss_virulex_logs")

    try:
        require_path(args.med_root, "Med-MICN repository root")
        if args.run_med_micn:
            med_artifact_root.mkdir(parents=True, exist_ok=True)
        else:
            require_path(med_artifact_root, "trained Med-MICN artifact root")
        require_path(args.xai_root, "XAI repository root")
        require_path(args.data_root, "COVID-CT dataset root")
        common = import_common(args.xai_root)
        bind_common_paths(common, args.xai_root, args.data_root)

        ctx = PipelineContext(
            med_root=args.med_root,
            med_artifact_root=med_artifact_root,
            xai_root=args.xai_root,
            data_root=args.data_root,
            concept_csv=args.concept_csv,
            med_python=args.med_python,
            ss_output_root=args.ss_output_root,
            force=args.force,
            run_med_micn=args.run_med_micn,
            skip_rulefit=args.skip_rulefit,
            skip_sfmov=args.skip_sfmov,
            force_concept_probabilities=args.force_concept_probabilities,
            concept_batch_size=args.concept_batch_size,
            samples_per_class=args.samples_per_class,
            common=common,
            logger=logger,
        )
        validate_ss_output_root(ctx)

        run_stage(
            ctx,
            "1_input_preprocessing",
            [common.OUTPUT_DIR / "01_augmented_splits.csv"],
            lambda: stage_preprocess(ctx),
        )
        run_stage(
            ctx,
            "2_mobilenetv2_grid_search_and_base_features",
            [common.OUTPUT_DIR / "02_best_dl_hyperparameters.json", common.FEATURE_DIR / "02_base_features_augmented.npz"],
            lambda: stage_dl_grid_search(ctx),
        )
        run_stage(
            ctx,
            "3_final_custom_mobilenetv2",
            [common.MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras"],
            lambda: stage_final_mobilenet(ctx),
        )
        run_stage(
            ctx,
            "4_statistical_features_zfmis",
            [common.FEATURE_DIR / "04_zfmis_feature_sets.json"],
            lambda: stage_statistical_features(ctx),
        )
        run_stage(
            ctx,
            "4b_required_svis_rulex_trained_components",
            [],
            lambda: validate_svis_rulex_required_components(ctx),
        )
        run_stage(
            ctx,
            "5_required_med_micn_concept_embedding_and_neural_symbolic",
            [],
            lambda: stage_concept_branch(ctx),
        )
        run_stage(
            ctx,
            "5b_med_micn_per_image_concept_probabilities",
            [] if args.force_concept_probabilities else [med_micn_concept_probabilities_path(ctx), xai_concept_probabilities_path(ctx)],
            lambda: generate_med_micn_concept_probabilities(ctx),
        )
        run_stage(
            ctx,
            "6_combined_stat_concept_zfmis",
            [] if args.force_concept_probabilities else [common.FEATURE_DIR / "04_combined_zfmis_feature_sets.json"],
            lambda: build_combined_feature_pool(ctx),
        )
        run_stage(
            ctx,
            "7a_combined_decision_tree_rules",
            [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "05c_combined_gridsearchfortree_results.csv"],
            lambda: evaluate_combined_tree_grid(ctx),
        )
        if not args.skip_rulefit:
            run_stage(
                ctx,
                "7b_combined_rulefit_rules",
                [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "05d_combined_rulefitgridsearchcode_results.csv"],
                lambda: evaluate_combined_rulefit_grid(ctx),
            )
        run_stage(
            ctx,
            "7c_combined_stronger_classifiers",
            [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "05e_combined_stronger_classifiers_results.csv"],
            lambda: evaluate_combined_strong_classifier_grid(ctx),
        )
        run_stage(
            ctx,
            "7d_combined_calibrated_threshold_classifiers",
            [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "05f_combined_calibrated_threshold_results.csv"],
            lambda: evaluate_combined_calibrated_threshold_grid(ctx),
        )
        run_stage(
            ctx,
            "7e_extract_combined_final_rules",
            [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "06_combined_rule_extraction_summary.json"],
            lambda: extract_combined_final_rules(ctx),
        )
        if not args.skip_sfmov:
            run_stage(
                ctx,
                "8_concept_aware_sfmov",
                [] if args.force_concept_probabilities else [common.OUTPUT_DIR / "07_concept_aware_sfmov_heatmap_files.json"],
                lambda: run_concept_aware_sfmov(ctx),
            )
        run_stage(ctx, "9_final_summary", [], lambda: write_final_summary(ctx))
        run_stage(ctx, "10_export_consolidated_ss_virulex_outputs", [], lambda: export_consolidated_outputs(ctx))
    except PipelineError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("SS-VIRULEX pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
