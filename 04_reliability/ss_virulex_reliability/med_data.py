from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .common import ensure_columns, normalize_manifest, stable_row_id


PATIENT_ID_CANDIDATES = ("patient_id", "patientid", "subject_id", "subjectid")


def _explicit_patient_column(columns: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in PATIENT_ID_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def prepare_med_frame(manifest_path: Path, concept_csv_path: Path) -> tuple[pd.DataFrame, list[str], str]:
    manifest = normalize_manifest(pd.read_csv(manifest_path))
    concepts = pd.read_csv(concept_csv_path)
    ensure_columns(
        concepts,
        ["image_id", "file_name", "true_label", "split", "image_path"],
        "concept CSV",
    )
    concept_columns = [
        column for column in concepts.columns if column.startswith("concept_") and column.endswith("_label")
    ]
    if not concept_columns:
        raise ValueError("concept CSV has no concept_*_label columns")
    if concepts["file_name"].duplicated().any():
        duplicates = concepts.loc[concepts["file_name"].duplicated(keep=False), "file_name"].head().tolist()
        raise ValueError(f"concept CSV has duplicate file_name values: {duplicates}")

    patient_column = _explicit_patient_column(concepts.columns.tolist())
    lookup_columns = ["image_id", "true_label", "split", "image_path", *concept_columns]
    if patient_column:
        lookup_columns.append(patient_column)
    lookup = concepts.set_index("file_name")[lookup_columns]
    manifest["concept_key"] = manifest["file_name"].where(
        manifest["source"] != "augmented", manifest["parent_file_name"]
    )
    missing = sorted(set(manifest["concept_key"]) - set(lookup.index))
    if missing:
        raise ValueError(f"{len(missing)} manifest rows lack concept metadata; first={missing[0]}")

    mapped = lookup.loc[manifest["concept_key"]].reset_index(drop=True)
    manifest = manifest.reset_index(drop=True)
    label_mismatch = manifest["true_label"].to_numpy() != mapped["true_label"].astype(int).to_numpy()
    if label_mismatch.any():
        first = int(label_mismatch.nonzero()[0][0])
        raise ValueError(f"manifest/concept label mismatch at row {first}: {manifest.iloc[first]['file_name']}")

    manifest["parent_image_id"] = mapped["image_id"].astype(str).to_numpy()
    manifest["image_id"] = [
        parent_id
        if source != "augmented"
        else f"{parent_id}::aug::{stable_row_id(file_name, image_path)[:12]}"
        for parent_id, source, file_name, image_path in zip(
            manifest["parent_image_id"], manifest["source"], manifest["file_name"], manifest["image_path"]
        )
    ]
    for column in concept_columns:
        manifest[column] = mapped[column].astype(float).to_numpy()

    if patient_column:
        manifest["patient_id"] = mapped[patient_column].to_numpy()
        if manifest["patient_id"].notna().all() and (manifest["patient_id"].astype(str).str.strip() != "").all():
            manifest["group_id"] = "patient::" + manifest["patient_id"].astype(str)
            grouping_mode = f"explicit_patient_id:{patient_column}"
        else:
            manifest["group_id"] = "lineage::" + manifest["parent_image_id"].astype(str)
            grouping_mode = "augmentation_lineage; patient IDs incomplete"
    else:
        manifest["group_id"] = "lineage::" + manifest["parent_image_id"].astype(str)
        grouping_mode = "augmentation_lineage; patient-level grouping not verifiable"

    manifest["file_name"] = manifest["file_name"].astype(str)
    manifest["image_path"] = manifest["image_path"].map(lambda value: str(Path(value).resolve()))
    missing_paths = manifest.loc[~manifest["image_path"].map(lambda value: Path(value).is_file()), "image_path"]
    if len(missing_paths):
        raise FileNotFoundError(f"{len(missing_paths)} image paths are missing; first={missing_paths.iloc[0]}")
    return manifest, concept_columns, grouping_mode


def deterministic_group_subset(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    count_column = "row_id" if "row_id" in frame.columns else "image_id"
    group_table = frame.groupby("group_id", as_index=False).agg(
        true_label=("true_label", "first"), rows=(count_column, "size")
    )
    sampled_parts = []
    target_per_class = max(1, max_rows // max(1, group_table["true_label"].nunique()))
    for label, groups in group_table.groupby("true_label"):
        shuffled = groups.sample(frac=1.0, random_state=seed + int(label)).reset_index(drop=True)
        chosen = []
        total = 0
        for row in shuffled.itertuples(index=False):
            if chosen and total + row.rows > target_per_class:
                continue
            chosen.append(row.group_id)
            total += int(row.rows)
            if total >= target_per_class:
                break
        sampled_parts.append(frame[frame["group_id"].isin(chosen)])
    result = pd.concat(sampled_parts, ignore_index=True)
    return result.sort_values(["true_label", "group_id", "image_id"]).reset_index(drop=True)
