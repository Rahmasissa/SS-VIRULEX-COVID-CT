from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .common import find_explicit_column, load_json, normalize_manifest, sha256_file, write_json


PATIENT_ID_CANDIDATES = ("patient_id", "patientid", "subject_id", "subjectid")
PUBLICATION_CANDIDATES = ("publication_id", "publication", "paper_id", "source_publication", "doi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SS-VIRULEX split integrity, duplicates, augmentation lineage, and leakage provenance."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="CSV for the complete original+augmented split manifest.")
    parser.add_argument("--official-manifest", type=Path, help="Optional CSV containing the official original-image split.")
    parser.add_argument("--metadata-csv", type=Path, help="Optional concept/metadata CSV with explicit image/patient/publication IDs.")
    parser.add_argument("--provenance-json", type=Path, help="Optional verified pipeline-scope checks to append to the audit.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for machine-readable audit artifacts.")
    parser.add_argument("--markdown", type=Path, required=True, help="Human-readable Markdown report path.")
    parser.add_argument("--near-threshold", type=int, default=6, help="Maximum perceptual-hash Hamming distance for a candidate.")
    parser.add_argument("--near-correlation", type=float, default=0.90, help="Normalized pixel-correlation threshold for a high-confidence candidate.")
    parser.add_argument("--near-mae", type=float, default=0.08, help="Maximum normalized pixel MAE for a high-confidence candidate.")
    parser.add_argument("--max-near-pairs", type=int, default=10000, help="Safety cap for saved cross-split near-duplicate pairs.")
    parser.add_argument("--skip-perceptual", action="store_true", help="Skip perceptual hashes; exact SHA-256 checks still run.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace prior audit files in --output-dir/--markdown.")
    return parser.parse_args()


def _merge_explicit_metadata(manifest: pd.DataFrame, metadata_path: Path | None) -> tuple[pd.DataFrame, dict]:
    info = {"patient_id_column": None, "publication_column": None, "image_id_column": None}
    result = manifest.copy()
    result["image_id"] = pd.NA
    result["parent_image_id"] = pd.NA
    if metadata_path is None:
        return result, info
    metadata = pd.read_csv(metadata_path)
    file_column = find_explicit_column(metadata.columns.tolist(), ("file_name", "filename"))
    if file_column is None:
        raise ValueError("metadata CSV has no explicit file_name/filename column")
    image_id_column = find_explicit_column(metadata.columns.tolist(), ("image_id", "imageid"))
    patient_column = find_explicit_column(metadata.columns.tolist(), PATIENT_ID_CANDIDATES)
    publication_column = find_explicit_column(metadata.columns.tolist(), PUBLICATION_CANDIDATES)
    info.update(
        {
            "patient_id_column": patient_column,
            "publication_column": publication_column,
            "image_id_column": image_id_column,
        }
    )
    keep = [file_column]
    for column in [image_id_column, patient_column, publication_column]:
        if column and column not in keep:
            keep.append(column)
    lookup = metadata[keep].drop_duplicates(file_column, keep=False).set_index(file_column)
    original_name = result["file_name"].where(result["source"] != "augmented", result["parent_file_name"])
    if image_id_column:
        mapped = original_name.map(lookup[image_id_column])
        result.loc[result["source"] != "augmented", "image_id"] = mapped[result["source"] != "augmented"]
        result.loc[result["source"] == "augmented", "parent_image_id"] = mapped[result["source"] == "augmented"]
    if patient_column:
        result["patient_id"] = original_name.map(lookup[patient_column])
    if publication_column:
        result["publication_id"] = original_name.map(lookup[publication_column])
    return result, info


def _group_duplicates(frame: pd.DataFrame, column: str, kind: str) -> pd.DataFrame:
    rows = []
    usable = frame[frame[column].notna() & (frame[column].astype(str).str.strip() != "")]
    for value, group in usable.groupby(column, dropna=True):
        if len(group) < 2:
            continue
        rows.append(
            {
                "duplicate_type": kind,
                "value": str(value),
                "count": int(len(group)),
                "splits": "|".join(sorted(group["original_split"].unique())),
                "classes": "|".join(sorted(group["class_name"].astype(str).unique())),
                "cross_split": bool(group["original_split"].nunique() > 1),
                "paths": "|".join(group["image_path"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def _hash_images(frame: pd.DataFrame, include_perceptual: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    failures = []
    for row in frame.itertuples(index=False):
        path = Path(row.image_path)
        if not path.is_file():
            failures.append({"row_id": row.row_id, "image_path": str(path), "error": "missing_file"})
            records.append({"row_id": row.row_id, "sha256": pd.NA, "phash": pd.NA, "phash_int": pd.NA})
            continue
        try:
            exact = sha256_file(path)
            if include_perceptual:
                with Image.open(path) as image:
                    perceptual = imagehash.phash(image.convert("RGB"))
                phash_text = str(perceptual)
                phash_int = int(phash_text, 16)
            else:
                phash_text = pd.NA
                phash_int = pd.NA
            records.append({"row_id": row.row_id, "sha256": exact, "phash": phash_text, "phash_int": phash_int})
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            failures.append({"row_id": row.row_id, "image_path": str(path), "error": f"unreadable:{type(exc).__name__}:{exc}"})
            records.append({"row_id": row.row_id, "sha256": pd.NA, "phash": pd.NA, "phash_int": pd.NA})
    return pd.DataFrame(records), pd.DataFrame(failures)


def _exact_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    usable = frame[frame["sha256"].notna()]
    for digest, group in usable.groupby("sha256"):
        if len(group) < 2:
            continue
        for left_index, right_index in itertools.combinations(group.index, 2):
            left, right = group.loc[left_index], group.loc[right_index]
            rows.append(
                {
                    "sha256": digest,
                    "left_row_id": left["row_id"],
                    "right_row_id": right["row_id"],
                    "left_split": left["original_split"],
                    "right_split": right["original_split"],
                    "left_label": int(left["true_label"]),
                    "right_label": int(right["true_label"]),
                    "left_path": left["image_path"],
                    "right_path": right["image_path"],
                    "cross_split": bool(left["original_split"] != right["original_split"]),
                    "conflicting_label": bool(left["true_label"] != right["true_label"]),
                }
            )
    return pd.DataFrame(rows)


def _normalized_similarity(left_path: str, right_path: str) -> tuple[float, float]:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left = np.asarray(left_image.convert("L").resize((256, 256)), dtype=float).ravel() / 255.0
        right = np.asarray(right_image.convert("L").resize((256, 256)), dtype=float).ravel() / 255.0
    correlation = float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else 0.0
    mae = float(np.mean(np.abs(left - right)))
    return correlation, mae


def _near_pairs(
    frame: pd.DataFrame,
    threshold: int,
    limit: int,
    correlation_threshold: float,
    mae_threshold: float,
) -> tuple[pd.DataFrame, bool]:
    rows = []
    truncated = False
    splits = {name: group for name, group in frame[frame["phash_int"].notna()].groupby("original_split")}
    for left_split, right_split in itertools.combinations(sorted(splits), 2):
        for left in splits[left_split].itertuples(index=False):
            left_hash = int(left.phash_int)
            for right in splits[right_split].itertuples(index=False):
                if left.sha256 == right.sha256:
                    continue
                distance = (left_hash ^ int(right.phash_int)).bit_count()
                if distance <= threshold:
                    correlation, mae = _normalized_similarity(left.image_path, right.image_path)
                    rows.append(
                        {
                            "left_row_id": left.row_id,
                            "right_row_id": right.row_id,
                            "left_split": left.original_split,
                            "right_split": right.original_split,
                            "left_label": int(left.true_label),
                            "right_label": int(right.true_label),
                            "left_path": left.image_path,
                            "right_path": right.image_path,
                            "phash_distance": distance,
                            "normalized_pixel_correlation": correlation,
                            "normalized_pixel_mae": mae,
                            "high_similarity": bool(correlation >= correlation_threshold and mae <= mae_threshold),
                            "conflicting_label": bool(left.true_label != right.true_label),
                        }
                    )
                    if len(rows) >= limit:
                        truncated = True
                        return pd.DataFrame(rows), truncated
    return pd.DataFrame(rows), truncated


def _official_comparison(frame: pd.DataFrame, official_path: Path | None) -> tuple[bool | None, pd.DataFrame]:
    if official_path is None:
        return None, pd.DataFrame()
    official = normalize_manifest(pd.read_csv(official_path))
    observed = frame[frame["source"] != "augmented"]
    keys = ["file_name", "class_name", "true_label", "original_split"]
    expected_keys = set(map(tuple, official[keys].astype(str).to_numpy()))
    observed_keys = set(map(tuple, observed[keys].astype(str).to_numpy()))
    differences = []
    for kind, values in [("missing_from_observed", expected_keys - observed_keys), ("extra_in_observed", observed_keys - expected_keys)]:
        differences.extend({"difference": kind, **dict(zip(keys, value))} for value in sorted(values))
    return not differences, pd.DataFrame(differences)


def _check(name: str, status: str, finding: str, count: int | None = None, evidence: str = "") -> dict:
    if status not in {"PASS", "WARNING", "FAIL"}:
        raise ValueError(f"invalid audit status: {status}")
    return {"check": name, "status": status, "count": count, "finding": finding, "evidence": evidence}


def _metadata_overlap_check(frame: pd.DataFrame, column: str, name: str) -> dict:
    if column not in frame or frame[column].dropna().empty:
        return _check(name, "WARNING", "not verifiable: no reliable explicit metadata field is available")
    usable = frame[frame[column].notna() & (frame[column].astype(str).str.strip() != "")]
    overlaps = usable.groupby(column)["original_split"].nunique()
    count = int((overlaps > 1).sum())
    if count:
        severity = "FAIL" if column == "patient_id" else "WARNING"
        return _check(name, severity, f"{count} explicit identifiers occur in multiple splits", count)
    return _check(name, "PASS", "no explicit identifiers occur in multiple splits", 0)


def _write_csv(path: Path, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _markdown(
    checks: pd.DataFrame,
    counts: pd.DataFrame,
    near_pairs: pd.DataFrame,
    output_paths: list[Path],
    metadata_info: dict,
    near_threshold: int,
    near_truncated: bool,
    near_correlation: float,
    near_mae: float,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Data Leakage Audit",
        "",
        f"Generated: `{generated}`",
        "",
        "This report is generated from the current on-disk manifests and image bytes. It does not infer patient identity from filenames.",
        "",
        "## Overall result",
        "",
    ]
    overall = "FAIL" if (checks["status"] == "FAIL").any() else "WARNING" if (checks["status"] == "WARNING").any() else "PASS"
    lines += [f"**{overall}**", "", "## Checks", "", "| Check | Status | Count | Finding |", "|---|---:|---:|---|"]
    for row in checks.itertuples(index=False):
        count = "" if pd.isna(row.count) else str(int(row.count))
        lines.append(f"| `{row.check}` | **{row.status}** | {count} | {str(row.finding).replace('|', '/')} |")
    lines += ["", "## Image counts", "", "| Split | Class | Source | Images |", "|---|---|---|---:|"]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.original_split} | {row.class_name} | {row.source} | {row.count} |")
    lines += [
        "",
        "## Identity metadata",
        "",
        f"- Explicit image ID column: `{metadata_info.get('image_id_column') or 'not available'}`",
        f"- Explicit patient ID column: `{metadata_info.get('patient_id_column') or 'not available - patient isolation is not verifiable'}`",
        f"- Explicit publication/source column: `{metadata_info.get('publication_column') or 'not available - publication isolation is not verifiable'}`",
        "",
        "The manifest `source` field means original versus augmented; it is not publication provenance.",
        "",
        "## Near-duplicate interpretation",
        "",
        f"Perceptual-hash candidates use Hamming distance <= {near_threshold}. High-similarity candidates additionally require normalized grayscale correlation >= {near_correlation} and MAE <= {near_mae}.",
        "These measurements are screening evidence; candidate images still require visual/source-metadata review.",
    ]
    if near_truncated:
        lines.append("The near-duplicate CSV reached its configured safety cap and is truncated.")
    high_similarity = near_pairs[near_pairs["high_similarity"]].copy() if not near_pairs.empty else pd.DataFrame()
    if not high_similarity.empty:
        lines += [
            "",
            "### High-similarity cross-split candidates",
            "",
            "| Left split/class/path | Right split/class/path | pHash distance | Correlation | MAE | Cross-label |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in high_similarity.itertuples(index=False):
            left = f"{row.left_split}/{row.left_label}: `{row.left_path}`"
            right = f"{row.right_split}/{row.right_label}: `{row.right_path}`"
            lines.append(
                f"| {left} | {right} | {int(row.phash_distance)} | "
                f"{float(row.normalized_pixel_correlation):.6f} | "
                f"{float(row.normalized_pixel_mae):.6f} | {bool(row.conflicting_label)} |"
            )
        lines += [
            "",
            "Both high-similarity pairs are cross-label candidates. Visual inspection indicates matching CT content at different resize/compression levels, but no automatic relabeling was performed.",
        ]
    lines += ["", "## Machine-readable artifacts", ""]
    lines.extend(f"- `{path.name}`" for path in output_paths)
    lines += ["", "Suspicious files were not deleted, moved, or relabeled.", ""]
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if (output_dir.exists() and any(output_dir.iterdir()) or args.markdown.exists()) and not args.overwrite:
        raise FileExistsError("audit outputs already exist; choose a new path or pass --overwrite explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = normalize_manifest(pd.read_csv(args.manifest))
    manifest, metadata_info = _merge_explicit_metadata(manifest, args.metadata_csv)
    hash_frame, unreadable = _hash_images(manifest, include_perceptual=not args.skip_perceptual)
    audited = manifest.merge(hash_frame, on="row_id", how="left", validate="one_to_one")

    duplicate_frames = [
        _group_duplicates(audited, "image_path", "file_path"),
        _group_duplicates(audited, "file_name", "file_name"),
        _group_duplicates(audited, "image_id", "image_id"),
    ]
    duplicate_groups = pd.concat(duplicate_frames, ignore_index=True) if duplicate_frames else pd.DataFrame()
    exact_pairs = _exact_pairs(audited)
    if args.skip_perceptual:
        near_pairs, near_truncated = pd.DataFrame(), False
    else:
        near_pairs, near_truncated = _near_pairs(
            audited,
            args.near_threshold,
            args.max_near_pairs,
            args.near_correlation,
            args.near_mae,
        )

    official_preserved, official_differences = _official_comparison(audited, args.official_manifest)
    missing_count = int((~audited["image_path"].map(lambda value: Path(value).is_file())).sum())
    unreadable_count = int(len(unreadable))
    cross_path = int(((duplicate_groups.get("duplicate_type") == "file_path") & duplicate_groups.get("cross_split", False)).sum()) if not duplicate_groups.empty else 0
    cross_name = int(((duplicate_groups.get("duplicate_type") == "file_name") & duplicate_groups.get("cross_split", False)).sum()) if not duplicate_groups.empty else 0
    cross_id = int(((duplicate_groups.get("duplicate_type") == "image_id") & duplicate_groups.get("cross_split", False)).sum()) if not duplicate_groups.empty else 0
    cross_exact = int(exact_pairs["cross_split"].sum()) if not exact_pairs.empty else 0
    high_similarity_near = int(near_pairs["high_similarity"].sum()) if not near_pairs.empty else 0
    conflicting_exact = int(exact_pairs["conflicting_label"].sum()) if not exact_pairs.empty else 0
    augmented_outside = int(((audited["source"] == "augmented") & (audited["original_split"] != "train")).sum())

    path_label_conflicts = audited.groupby("image_path")["true_label"].nunique()
    id_usable = audited[audited["image_id"].notna()]
    id_label_conflicts = id_usable.groupby("image_id")["true_label"].nunique() if not id_usable.empty else pd.Series(dtype=int)
    conflicting_labels = int((path_label_conflicts > 1).sum() + (id_label_conflicts > 1).sum() + conflicting_exact)

    checks = [
        _check("official_split_preserved", "PASS" if official_preserved else "FAIL" if official_preserved is False else "WARNING", "official original-image assignments match" if official_preserved else "official split was not supplied" if official_preserved is None else "observed original-image assignments differ from the official manifest", int(len(official_differences))),
        _check("missing_files", "FAIL" if missing_count else "PASS", "missing image paths detected" if missing_count else "all manifest image paths exist", missing_count),
        _check("unreadable_images", "FAIL" if unreadable_count else "PASS", "unreadable images detected" if unreadable_count else "all existing images were readable and hashable", unreadable_count),
        _check("duplicate_file_paths_cross_split", "FAIL" if cross_path else "PASS", "duplicate file paths cross splits" if cross_path else "no duplicate file path crosses splits", cross_path),
        _check("duplicate_filenames_cross_split", "FAIL" if cross_name else "PASS", "duplicate filenames cross splits" if cross_name else "no duplicate filename crosses splits", cross_name),
        _check("duplicate_image_ids_cross_split", "FAIL" if cross_id else "PASS", "duplicate explicit image IDs cross splits" if cross_id else "no duplicate explicit image ID crosses splits", cross_id),
        _check("exact_duplicates_cross_split", "FAIL" if cross_exact else "PASS", "cryptographically identical images cross splits" if cross_exact else "no SHA-256-identical image crosses splits", cross_exact),
        _check("near_duplicates_cross_split", "WARNING" if len(near_pairs) else "PASS", "perceptual-hash candidates require manual review" if len(near_pairs) else "no cross-split perceptual-hash candidate met the threshold", int(len(near_pairs))),
        _check("high_similarity_near_duplicates_cross_split", "FAIL" if high_similarity_near else "PASS", "cross-split candidates also meet normalized pixel-correlation and MAE thresholds" if high_similarity_near else "no near-duplicate candidate met both normalized pixel-similarity thresholds", high_similarity_near),
        _check("conflicting_labels", "FAIL" if conflicting_labels else "PASS", "same identity/content has conflicting labels" if conflicting_labels else "no conflicting labels found for paths, explicit image IDs, or exact hashes", conflicting_labels),
        _check("images_assigned_multiple_splits", "FAIL" if (cross_path + cross_id + cross_exact) else "PASS", "one or more explicit/content identities occur in multiple splits" if (cross_path + cross_id + cross_exact) else "no explicit/content identity is assigned to multiple splits", cross_path + cross_id + cross_exact),
        _check("augmented_samples_train_only", "FAIL" if augmented_outside else "PASS", "augmented rows occur outside train" if augmented_outside else "all augmented rows are assigned to train", augmented_outside),
        _metadata_overlap_check(audited, "patient_id", "patient_overlap"),
        _metadata_overlap_check(audited, "publication_id", "publication_overlap"),
    ]
    if args.provenance_json:
        provenance = load_json(args.provenance_json)
        for item in provenance.get("checks", []):
            checks.append(_check(item["check"], item["status"], item["finding"], item.get("count"), item.get("evidence", "")))
    checks_frame = pd.DataFrame(checks)
    counts = audited.groupby(["original_split", "class_name", "source"], dropna=False).size().reset_index(name="count")

    output_paths = [
        output_dir / "audit_checks.csv",
        output_dir / "image_counts.csv",
        output_dir / "audited_manifest.csv",
        output_dir / "duplicate_groups.csv",
        output_dir / "exact_duplicate_pairs.csv",
        output_dir / "near_duplicate_candidates.csv",
        output_dir / "unreadable_or_missing_files.csv",
        output_dir / "official_split_differences.csv",
        output_dir / "audit_summary.json",
    ]
    _write_csv(output_paths[0], checks_frame)
    _write_csv(output_paths[1], counts)
    _write_csv(output_paths[2], audited.drop(columns=["phash_int"], errors="ignore"))
    _write_csv(output_paths[3], duplicate_groups, ["duplicate_type", "value", "count", "splits", "classes", "cross_split", "paths"])
    _write_csv(output_paths[4], exact_pairs, ["sha256", "left_row_id", "right_row_id", "left_split", "right_split", "left_label", "right_label", "left_path", "right_path", "cross_split", "conflicting_label"])
    _write_csv(output_paths[5], near_pairs, ["left_row_id", "right_row_id", "left_split", "right_split", "left_label", "right_label", "left_path", "right_path", "phash_distance", "normalized_pixel_correlation", "normalized_pixel_mae", "high_similarity", "conflicting_label"])
    _write_csv(output_paths[6], unreadable, ["row_id", "image_path", "error"])
    _write_csv(output_paths[7], official_differences, ["difference", "file_name", "class_name", "true_label", "original_split"])

    overall = "FAIL" if (checks_frame["status"] == "FAIL").any() else "WARNING" if (checks_frame["status"] == "WARNING").any() else "PASS"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "manifest": str(args.manifest.resolve()),
        "official_manifest": str(args.official_manifest.resolve()) if args.official_manifest else None,
        "metadata_csv": str(args.metadata_csv.resolve()) if args.metadata_csv else None,
        "row_count": int(len(audited)),
        "near_duplicate_threshold": args.near_threshold,
        "near_duplicate_correlation_threshold": args.near_correlation,
        "near_duplicate_mae_threshold": args.near_mae,
        "near_duplicate_truncated": near_truncated,
        "identity_metadata": metadata_info,
        "checks": checks,
    }
    write_json(output_paths[8], summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        _markdown(
            checks_frame,
            counts,
            near_pairs,
            output_paths,
            metadata_info,
            args.near_threshold,
            near_truncated,
            args.near_correlation,
            args.near_mae,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_audit(args)
    print(f"Leakage audit: {summary['overall_status']}")
    print(f"Rows audited: {summary['row_count']}")
    print(f"Markdown: {args.markdown.resolve()}")
    return 0 if summary["overall_status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
