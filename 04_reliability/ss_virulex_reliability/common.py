from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
METADATA_COLUMNS = [
    "image_id",
    "file_name",
    "image_path",
    "true_label",
    "original_split",
    "fold_id",
    "model_id",
    "feature_generation_mode",
]
STATISTICAL_FEATURES = [
    "mean",
    "std_dev",
    "variance",
    "median",
    "range",
    "skewness",
    "kurtosis",
    "entropy",
    "energy",
    "contrast",
    "mean_abs_dev",
    "min_value",
    "max_value",
    "iqr",
    "percentile_25",
    "percentile_50",
    "percentile_75",
    "signal_to_noise",
    "coef_of_var",
    "autocorrelation",
    "shannon_entropy",
    "root_mean_square",
    "harmonic_mean",
    "geometric_mean",
    "std_error_mean",
    "median_abs_dev",
]
CONCEPT_FEATURES = [
    "concept_score_peripheral_ground_glass_opacities",
    "concept_score_bilateral_involvement",
    "concept_score_multilobar_distribution",
    "concept_score_crazy_paving_pattern",
    "concept_score_absence_of_lobar_consolidation",
    "concept_score_localized_or_diffuse_presentation",
    "concept_score_increased_density_in_the_lung",
    "concept_score_ground_glass_appearance",
]
DIAGNOSIS_FEATURES = ["med_task_prob_covid", "med_neural_prob_covid"]
FEATURE_COLUMNS = STATISTICAL_FEATURES + CONCEPT_FEATURES + DIAGNOSIS_FEATURES


def set_deterministic_seed(seed: int = DEFAULT_SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_id(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "filepath": "image_path",
        "filename": "file_name",
        "label": "true_label",
        "split": "original_split",
    }
    result = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns}).copy()
    ensure_columns(result, ["image_path", "file_name", "true_label", "original_split"], "manifest")
    if "class_name" not in result:
        result["class_name"] = result["true_label"].map({0: "NonCOVID", 1: "COVID"}).fillna("unknown")
    if "source" not in result:
        result["source"] = "original"
    if "source_filepath" not in result:
        result["source_filepath"] = ""
    result["source"] = result["source"].fillna("original").astype(str).str.lower()
    result["original_split"] = result["original_split"].astype(str).str.lower().replace({"valid": "val", "validation": "val"})
    result["true_label"] = pd.to_numeric(result["true_label"], errors="raise").astype(int)
    result["image_path"] = result["image_path"].astype(str)
    result["file_name"] = result["file_name"].astype(str)
    result["row_id"] = [
        stable_row_id(path, name, split, source)
        for path, name, split, source in zip(
            result["image_path"], result["file_name"], result["original_split"], result["source"]
        )
    ]
    result["parent_file_name"] = [
        Path(str(source_path)).name if source == "augmented" and str(source_path).strip() else ""
        for source_path, source in zip(result["source_filepath"], result["source"])
    ]
    return result


def find_explicit_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def probability_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("concept_score_") or column in DIAGNOSIS_FEATURES
    ]


def validate_feature_frame(
    frame: pd.DataFrame,
    expected_features: Sequence[str],
    expected_rows: int | None = None,
    require_metadata: bool = True,
) -> list[str]:
    errors: list[str] = []
    if expected_rows is not None and len(frame) != expected_rows:
        errors.append(f"row_count: expected {expected_rows}, observed {len(frame)}")
    required = list(expected_features)
    if require_metadata:
        required += METADATA_COLUMNS
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        errors.append(f"missing_columns: {missing}")
        return errors
    if list(frame[expected_features].columns) != list(expected_features):
        errors.append("feature_order_mismatch")
    numeric = frame[list(expected_features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("non_finite_feature_values")
    for column in probability_columns(frame):
        if column not in expected_features:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if ((values < 0) | (values > 1)).any():
            errors.append(f"probability_range_violation:{column}")
    if require_metadata:
        if frame["image_id"].isna().any() or (frame["image_id"].astype(str).str.strip() == "").any():
            errors.append("missing_image_id")
        if frame["image_id"].duplicated().any():
            errors.append("duplicate_image_id")
        if frame["image_path"].duplicated().any():
            errors.append("duplicate_image_path")
    return errors
