from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .common import METADATA_COLUMNS, validate_feature_frame, write_json
from .med_data import deterministic_group_subset, prepare_med_frame
from .trainers import configuration_id, med_feature_names, trainer_for_backend


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--concept-csv", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--med-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new, non-existing output directory.")
    parser.add_argument("--backend", choices=["med_micn", "smoke"], default="med_micn")
    parser.add_argument("--seed", type=int, default=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leakage-safe Med-MICN OOF or final-test features.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_id = subparsers.add_parser("config-id", help="Print the immutable ID for an encoder configuration.")
    config_id.add_argument("--encoder-config", type=Path, required=True)

    oof = subparsers.add_parser("oof", help="Generate one excluded-fold prediction for every development row.")
    _add_common(oof)
    oof.add_argument("--folds", type=int, default=5)
    oof.add_argument("--development-splits", nargs="+", default=["train", "val"])
    oof.add_argument("--max-rows", type=int, default=0, help="Group-safe subset for smoke testing only; 0 uses all rows.")
    oof.add_argument("--resume", action="store_true", help="Reuse only complete, provenance-validated per-fold outputs.")

    final = subparsers.add_parser("final-test", help="Fit full development data and infer the untouched test once.")
    _add_common(final)
    final.add_argument("--development-splits", nargs="+", default=["train", "val"])
    final.add_argument("--test-split", default="test")
    final.add_argument("--locked-config-id", required=True)
    final.add_argument("--allow-final-test", action="store_true", help="Required acknowledgement that selection is locked.")
    final.add_argument("--final-epochs", type=int, required=True, help="Locked epoch count; test labels are never loaded by training.")
    return parser.parse_args()


def _outer_splits(frame: pd.DataFrame, folds: int, seed: int):
    group_labels = frame.groupby("group_id")["true_label"].nunique()
    if (group_labels > 1).any():
        raise ValueError("a grouping identifier has conflicting labels")
    groups_per_class = frame.groupby("true_label")["group_id"].nunique()
    max_folds = int(groups_per_class.min())
    if folds > max_folds:
        raise ValueError(f"requested folds={folds}, but the smallest class has only {max_folds} groups")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(frame, frame["true_label"], groups=frame["group_id"]))


def _inner_train_validation(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=seed)
    for train_indices, validation_indices in splitter.split(frame, groups=frame["group_id"]):
        train = frame.iloc[train_indices].copy()
        validation = frame.iloc[validation_indices].copy()
        if train["true_label"].nunique() == 2 and validation["true_label"].nunique() == 2:
            return train, validation
    raise ValueError("could not create a group-disjoint inner validation split containing both classes")


def _metadata_block(frame: pd.DataFrame, fold_id: str, model_id: str, mode: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": frame["image_id"].astype(str).to_numpy(),
            "file_name": frame["file_name"].astype(str).to_numpy(),
            "image_path": frame["image_path"].astype(str).to_numpy(),
            "true_label": frame["true_label"].astype(int).to_numpy(),
            "original_split": frame["original_split"].astype(str).to_numpy(),
            "fold_id": fold_id,
            "model_id": model_id,
            "feature_generation_mode": mode,
        }
    )


def _validate_alignment(features: pd.DataFrame, expected: pd.DataFrame, feature_names: list[str]) -> list[dict]:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    errors = validate_feature_frame(features, feature_names, expected_rows=len(expected), require_metadata=True)
    add("feature_frame_schema", not errors, "; ".join(errors) if errors else "schema, ordering, finiteness, ranges, and uniqueness valid")
    expected_key = expected[["image_id", "file_name", "image_path", "true_label", "original_split"]].copy()
    actual_key = features[["image_id", "file_name", "image_path", "true_label", "original_split"]].copy()
    expected_key = expected_key.sort_values("image_id").reset_index(drop=True).astype(str)
    actual_key = actual_key.sort_values("image_id").reset_index(drop=True).astype(str)
    aligned = expected_key.equals(actual_key)
    add("image_to_feature_alignment", aligned, "all metadata keys align" if aligned else "metadata mismatch")
    mode_ok = set(features["feature_generation_mode"]) <= {"out-of-fold", "validation", "final-test inference"}
    add("feature_generation_mode", mode_ok, f"observed={sorted(features['feature_generation_mode'].unique())}")
    return checks


def _fold_assignment(
    development: pd.DataFrame,
    outer_train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    fold_index: int,
) -> pd.DataFrame:
    assignment = development.iloc[np.concatenate([outer_train_indices, holdout_indices])][
        ["image_id", "group_id", "true_label", "original_split"]
    ].copy()
    assignment["fold_id"] = fold_index
    assignment["role"] = ["outer_train"] * len(outer_train_indices) + ["outer_holdout"] * len(holdout_indices)
    return assignment


def _assert_same_rows(observed: pd.DataFrame, expected: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in observed]
    if missing:
        raise ValueError(f"resumed {label} lacks columns: {missing}")
    left = observed[columns].astype(str).sort_values(columns).reset_index(drop=True)
    right = expected[columns].astype(str).sort_values(columns).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"resumed {label} does not match the deterministic fold definition")


def _load_resumed_fold(
    fold_dir: Path,
    holdout: pd.DataFrame,
    assignment: pd.DataFrame,
    fit_frame: pd.DataFrame,
    inner_validation: pd.DataFrame,
    feature_names: list[str],
    fold_index: int,
    model_id: str,
    backend: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "features": fold_dir / "fold_features.csv",
        "assignment": fold_dir / "fold_assignment.csv",
        "metadata": fold_dir / "checkpoint_metadata.json",
        "provenance": fold_dir / "training_provenance.csv",
        "checkpoint": fold_dir / ("best_model.pt" if backend == "med_micn" else "smoke_model.joblib"),
    }
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        raise FileNotFoundError(f"no completed cache exists for fold {fold_index}")
    if not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        raise RuntimeError(f"fold {fold_index} cache is incomplete; missing={missing}")

    features = pd.read_csv(paths["features"])
    cached_assignment = pd.read_csv(paths["assignment"])
    expected_feature_columns = METADATA_COLUMNS + feature_names
    if features.columns.tolist() != expected_feature_columns:
        raise ValueError(
            f"resumed fold {fold_index} feature order mismatch: "
            f"observed={features.columns.tolist()} expected={expected_feature_columns}"
        )
    checks = _validate_alignment(features, holdout, feature_names)
    if any(check["status"] == "FAIL" for check in checks):
        raise ValueError(f"resumed fold {fold_index} feature validation failed: {checks}")
    if set(features["feature_generation_mode"].astype(str)) != {"out-of-fold"}:
        raise ValueError(f"resumed fold {fold_index} has an invalid feature-generation mode")
    if set(features["fold_id"].astype(str)) != {str(fold_index)}:
        raise ValueError(f"resumed fold {fold_index} has inconsistent fold IDs")
    if set(features["model_id"].astype(str)) != {model_id}:
        raise ValueError(f"resumed fold {fold_index} was produced by a different model/configuration")
    _assert_same_rows(
        cached_assignment,
        assignment,
        ["image_id", "group_id", "true_label", "original_split", "fold_id", "role"],
        f"fold {fold_index} assignment",
    )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("backend") != backend or metadata.get("model_id") != model_id:
        raise ValueError(f"resumed fold {fold_index} checkpoint metadata does not match the requested run")
    if metadata.get("feature_order") != feature_names or metadata.get("config") != config:
        raise ValueError(f"resumed fold {fold_index} checkpoint schema/configuration mismatch")
    expected_provenance = fit_frame[["image_id", "image_path", "original_split", "group_id"]].copy()
    expected_provenance["role"] = "fit"
    validation_provenance = inner_validation[["image_id", "image_path", "original_split", "group_id"]].copy()
    validation_provenance["role"] = "inner_validation"
    expected_provenance = pd.concat([expected_provenance, validation_provenance], ignore_index=True)
    _assert_same_rows(
        pd.read_csv(paths["provenance"]),
        expected_provenance,
        ["image_id", "image_path", "original_split", "group_id", "role"],
        f"fold {fold_index} training provenance",
    )
    return features, cached_assignment


def generate_oof(args: argparse.Namespace) -> dict:
    resume = bool(getattr(args, "resume", False))
    if args.output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    config = _load_yaml(args.encoder_config)
    frame, concept_columns, grouping_mode = prepare_med_frame(args.manifest, args.concept_csv)
    development = frame[frame["original_split"].isin([value.lower() for value in args.development_splits])].copy()
    if args.max_rows:
        if args.backend != "smoke":
            raise ValueError("--max-rows is restricted to --backend smoke")
        development = deterministic_group_subset(development, args.max_rows, args.seed)
    if (development["original_split"] == "test").any():
        raise ValueError("development rows include the test split")
    args.output_dir.mkdir(parents=True, exist_ok=resume)
    config_id = configuration_id(config)
    feature_names = med_feature_names(concept_columns)
    predictions = []
    assignments = []
    fold_checks = []
    for fold_index, (outer_train_indices, holdout_indices) in enumerate(
        _outer_splits(development, args.folds, args.seed)
    ):
        outer_train = development.iloc[outer_train_indices].copy()
        holdout = development.iloc[holdout_indices].copy()
        overlap = set(outer_train["group_id"]) & set(holdout["group_id"])
        if overlap:
            raise RuntimeError(f"fold {fold_index} group leakage: {sorted(overlap)[:3]}")
        fit_frame, inner_validation = _inner_train_validation(outer_train, args.seed + fold_index)
        if set(fit_frame["group_id"]) & set(inner_validation["group_id"]):
            raise RuntimeError(f"fold {fold_index} inner group leakage")
        model_id = f"{config.get('experiment_id', 'encoder')}-{config_id}-fold-{fold_index}"
        fold_dir = args.output_dir / "checkpoints" / f"fold_{fold_index}"
        assignment = _fold_assignment(development, outer_train_indices, holdout_indices, fold_index)
        if resume and fold_dir.exists():
            cached_features, cached_assignment = _load_resumed_fold(
                fold_dir,
                holdout,
                assignment,
                fit_frame,
                inner_validation,
                feature_names,
                fold_index,
                model_id,
                args.backend,
                config,
            )
            predictions.append(cached_features)
            assignments.append(cached_assignment)
            fold_checks.append(
                {
                    "fold_id": fold_index,
                    "resumed": True,
                    "fit_rows": int(len(fit_frame)),
                    "inner_validation_rows": int(len(inner_validation)),
                    "outer_holdout_rows": int(len(holdout)),
                    "group_overlap": 0,
                    "checkpoint_dir": str(fold_dir),
                }
            )
            continue
        trainer = trainer_for_backend(args.backend, config, args.seed + fold_index, args.med_source_root)
        trainer.fit(fit_frame, inner_validation, concept_columns, fold_dir, model_id)
        predicted = trainer.predict(holdout).reset_index(drop=True)
        metadata = _metadata_block(holdout.reset_index(drop=True), str(fold_index), model_id, "out-of-fold")
        fold_features = pd.concat([metadata, predicted], axis=1)
        predictions.append(fold_features)
        assignments.append(assignment)
        fold_features.to_csv(fold_dir / "fold_features.csv", index=False)
        assignment.to_csv(fold_dir / "fold_assignment.csv", index=False)
        fold_checks.append(
            {
                "fold_id": fold_index,
                "resumed": False,
                "fit_rows": int(len(fit_frame)),
                "inner_validation_rows": int(len(inner_validation)),
                "outer_holdout_rows": int(len(holdout)),
                "group_overlap": 0,
                "checkpoint_dir": str(fold_dir),
            }
        )

    features = pd.concat(predictions, ignore_index=True)
    features = features[METADATA_COLUMNS + feature_names].sort_values("image_id").reset_index(drop=True)
    checks = _validate_alignment(features, development, feature_names)
    if any(check["status"] == "FAIL" for check in checks):
        raise RuntimeError(f"OOF validation failed: {checks}")
    features.to_csv(args.output_dir / "oof_med_micn_features.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(args.output_dir / "fold_assignments.csv", index=False)
    summary = {
        "backend": args.backend,
        "scientific_result": args.backend == "med_micn",
        "configuration_id": config_id,
        "encoder_config": str(args.encoder_config.resolve()),
        "development_splits": args.development_splits,
        "test_rows_accessed": 0,
        "rows": int(len(features)),
        "folds": args.folds,
        "grouping_mode": grouping_mode,
        "feature_order": feature_names,
        "checks": checks,
        "fold_details": fold_checks,
    }
    write_json(args.output_dir / "oof_summary.json", summary)
    return summary


def generate_final_test(args: argparse.Namespace) -> dict:
    if not args.allow_final_test:
        raise PermissionError("final-test inference requires --allow-final-test after configuration lock")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    config = _load_yaml(args.encoder_config)
    config_id = configuration_id(config)
    if args.locked_config_id != config_id:
        raise ValueError(f"locked config ID mismatch: supplied={args.locked_config_id} actual={config_id}")
    frame, concept_columns, grouping_mode = prepare_med_frame(args.manifest, args.concept_csv)
    development = frame[frame["original_split"].isin([value.lower() for value in args.development_splits])].copy()
    test = frame[frame["original_split"] == args.test_split.lower()].copy()
    if not len(test):
        raise ValueError("test split is empty")
    if set(development["group_id"]) & set(test["group_id"]):
        raise ValueError("development/test group overlap; final-test inference refused")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_id = f"{config.get('experiment_id', 'encoder')}-{config_id}-final"
    trainer = trainer_for_backend(args.backend, config, args.seed, args.med_source_root)
    trainer.fit(
        development,
        None,
        concept_columns,
        args.output_dir / "checkpoint",
        model_id,
        fixed_epochs=args.final_epochs,
    )
    predicted = trainer.predict(test).reset_index(drop=True)
    metadata = _metadata_block(test.reset_index(drop=True), "final", model_id, "final-test inference")
    features = pd.concat([metadata, predicted], axis=1)
    feature_names = med_feature_names(concept_columns)
    features = features[METADATA_COLUMNS + feature_names].sort_values("image_id").reset_index(drop=True)
    checks = _validate_alignment(features, test, feature_names)
    if any(check["status"] == "FAIL" for check in checks):
        raise RuntimeError(f"final-test feature validation failed: {checks}")
    features.to_csv(args.output_dir / "final_test_med_micn_features.csv", index=False)
    summary = {
        "backend": args.backend,
        "scientific_result": args.backend == "med_micn",
        "configuration_id": config_id,
        "locked_configuration": True,
        "test_inference_count": 1,
        "test_labels_used_for_fit_or_selection": False,
        "development_rows": int(len(development)),
        "test_rows": int(len(test)),
        "grouping_mode": grouping_mode,
        "feature_order": feature_names,
        "checks": checks,
    }
    write_json(args.output_dir / "final_test_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.command == "config-id":
        print(configuration_id(_load_yaml(args.encoder_config)))
        return 0
    summary = generate_oof(args) if args.command == "oof" else generate_final_test(args)
    print(f"Backend: {summary['backend']}")
    print(f"Scientific result: {summary['scientific_result']}")
    print(f"Configuration ID: {summary['configuration_id']}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
