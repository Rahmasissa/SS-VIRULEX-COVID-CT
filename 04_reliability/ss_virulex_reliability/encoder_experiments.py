from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .common import METADATA_COLUMNS, validate_feature_frame, write_json
from .med_data import deterministic_group_subset, prepare_med_frame
from .oof_med_micn import _metadata_block
from .trainers import configuration_id, med_feature_names, trainer_for_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current, frozen, gradual-final-block, and optionally broader Med-MICN encoder configurations."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Use the official original-image manifest for baseline comparability.")
    parser.add_argument("--concept-csv", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--med-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new, non-existing output directory.")
    parser.add_argument("--backend", choices=["med_micn", "smoke"], default="med_micn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Group-safe smoke-only limit.")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Group-safe smoke-only limit.")
    parser.add_argument("--include-disabled", action="store_true", help="Explicitly enable configs marked enabled: false.")
    return parser.parse_args()


def _read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid config: {path}")
    return config


def _metrics(frame: pd.DataFrame) -> dict:
    y = frame["true_label"].to_numpy(dtype=int)
    task_probability = frame["med_task_prob_covid"].to_numpy(dtype=float)
    neural_probability = frame["med_neural_prob_covid"].to_numpy(dtype=float)
    task_prediction = (task_probability >= 0.5).astype(int)
    neural_prediction = (neural_probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, task_prediction, labels=[0, 1]).ravel()
    concept_columns = [column for column in frame if column.startswith("concept_score_")]
    result = {
        "accuracy": float(accuracy_score(y, task_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, task_prediction)),
        "precision": float(precision_score(y, task_prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, task_prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y, task_prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, task_prediction)),
        "roc_auc": float(roc_auc_score(y, task_probability)) if len(np.unique(y)) == 2 else None,
        "neural_balanced_accuracy": float(balanced_accuracy_score(y, neural_prediction)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if concept_columns:
        label_columns = [
            "concept_" + column.removeprefix("concept_score_") + "_label" for column in concept_columns
        ]
        if set(label_columns) <= set(frame.columns):
            predicted = (frame[concept_columns].to_numpy(dtype=float) >= 0.5).astype(int)
            true = frame[label_columns].to_numpy(dtype=int)
            result["concept_accuracy"] = float((predicted == true).mean())
    return result


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    frame, concept_columns, grouping_mode = prepare_med_frame(args.manifest, args.concept_csv)
    train = frame[frame["original_split"] == "train"].copy()
    validation = frame[frame["original_split"] == "val"].copy()
    if not len(train) or not len(validation):
        raise ValueError("non-empty train and val rows are required")
    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("train/validation group overlap")
    if args.max_train_rows or args.max_val_rows:
        if args.backend != "smoke":
            raise ValueError("row limits are restricted to the smoke backend")
        if args.max_train_rows:
            train = deterministic_group_subset(train, args.max_train_rows, args.seed)
        if args.max_val_rows:
            validation = deterministic_group_subset(validation, args.max_val_rows, args.seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    experiment_details = []
    for config_path in args.configs:
        config = _read_config(config_path)
        enabled = bool(config.get("enabled", True))
        if not enabled and not args.include_disabled:
            rows.append(
                {
                    "experiment_id": config.get("experiment_id", config_path.stem),
                    "status": "SKIPPED_DISABLED",
                    "scientific_result": False,
                    "reason": "configuration is disabled; use --include-disabled only after earlier experiments justify it",
                }
            )
            continue
        config_id = configuration_id(config)
        experiment_id = str(config.get("experiment_id", config_path.stem))
        model_id = f"{experiment_id}-{config_id}-validation"
        experiment_dir = args.output_dir / experiment_id
        trainer = trainer_for_backend(args.backend, config, args.seed, args.med_source_root)
        metadata = trainer.fit(train, validation, concept_columns, experiment_dir / "checkpoint", model_id)
        predicted = trainer.predict(validation).reset_index(drop=True)
        meta = _metadata_block(validation.reset_index(drop=True), "validation", model_id, "validation")
        labels = validation[concept_columns].reset_index(drop=True)
        predictions = pd.concat([meta, predicted, labels], axis=1)
        feature_names = med_feature_names(concept_columns)
        validation_errors = validate_feature_frame(
            predictions[METADATA_COLUMNS + feature_names], feature_names, expected_rows=len(validation)
        )
        if validation_errors:
            raise RuntimeError(f"encoder prediction validation failed for {experiment_id}: {validation_errors}")
        predictions.to_csv(experiment_dir / "validation_features.csv", index=False)
        shutil.copy2(config_path, experiment_dir / "exact_config.yaml")
        metrics = _metrics(predictions)
        write_json(experiment_dir / "validation_metrics.json", metrics)
        parameter_counts = metadata.get("parameter_counts", {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "configuration_id": config_id,
                "status": "COMPLETED",
                "backend": args.backend,
                "scientific_result": args.backend == "med_micn",
                "fine_tune_mode": config["fine_tuning"]["mode"],
                "backbone_learning_rate": config["optimization"]["backbone_learning_rate"],
                "head_learning_rate": config["optimization"]["head_learning_rate"],
                "trainable_parameters": parameter_counts.get("trainable"),
                "backbone_trainable_parameters": parameter_counts.get("backbone_trainable"),
                **metrics,
            }
        )
        experiment_details.append(
            {
                "experiment_id": experiment_id,
                "config": str(config_path.resolve()),
                "checkpoint_metadata": metadata,
                "metrics": metrics,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "encoder_validation_comparison.csv", index=False)
    summary = {
        "backend": args.backend,
        "scientific_result": args.backend == "med_micn",
        "selection_data": "validation only",
        "test_rows_accessed": 0,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "grouping_mode": grouping_mode,
        "experiments": experiment_details,
    }
    write_json(args.output_dir / "encoder_experiment_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(f"Backend: {summary['backend']}")
    print(f"Scientific result: {summary['scientific_result']}")
    print(f"Test rows accessed: {summary['test_rows_accessed']}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
