from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .common import write_json
from .med_data import deterministic_group_subset, prepare_med_frame
from .trainers import configuration_id, med_feature_names


class FoldSafeZFMIS(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_names: Sequence[str],
        k: int,
        zero_fraction_threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.k = k
        self.zero_fraction_threshold = zero_fraction_threshold
        self.random_state = random_state

    def fit(self, x, y):
        values = np.asarray(x, dtype=float)
        names = np.asarray(list(self.feature_names), dtype=object)
        if values.shape[1] != len(names):
            raise ValueError(f"feature-name mismatch: matrix={values.shape[1]} names={len(names)}")
        self.zero_fraction_ = np.mean(values == 0, axis=0)
        eligible = np.flatnonzero(self.zero_fraction_ <= float(self.zero_fraction_threshold))
        if not len(eligible):
            eligible = np.arange(values.shape[1])
        scores = mutual_info_classif(values[:, eligible], y, random_state=int(self.random_state))
        ordering = sorted(range(len(eligible)), key=lambda index: (-float(scores[index]), str(names[eligible[index]])))
        ranked_eligible = eligible[np.asarray(ordering, dtype=int)]
        actual_k = min(int(self.k), len(ranked_eligible))
        self.selected_indices_ = ranked_eligible[:actual_k]
        self.selected_feature_names_ = names[self.selected_indices_].tolist()
        self.mi_scores_ = np.full(values.shape[1], np.nan, dtype=float)
        self.mi_scores_[eligible] = scores
        self.ranks_ = np.full(values.shape[1], np.nan, dtype=float)
        for rank, feature_index in enumerate(ranked_eligible, start=1):
            self.ranks_[feature_index] = rank
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, x):
        return np.asarray(x, dtype=float)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_feature_names_, dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested, fold-safe SS-VIRULEX feature screening and rule-model evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Nested development-only evaluation; never loads test features.")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--concept-csv", type=Path, required=True)
    evaluate.add_argument("--train-features", type=Path, required=True)
    evaluate.add_argument("--val-features", type=Path, required=True)
    evaluate.add_argument("--feature-set-json", type=Path, required=True)
    evaluate.add_argument("--oof-med-features", type=Path, help="Optional leakage-safe Med-MICN OOF features replacing current Med columns.")
    evaluate.add_argument("--output-dir", type=Path, required=True, help="A new, non-existing output directory.")
    evaluate.add_argument("--outer-folds", type=int, default=5)
    evaluate.add_argument("--inner-folds", type=int, default=3)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--models", nargs="+", choices=["decision_tree", "rulefit"], default=["decision_tree", "rulefit"])
    evaluate.add_argument("--subset-sizes", type=int, nargs="*", help="Optional override; defaults to verified saved hierarchy.")
    evaluate.add_argument("--zero-fraction-threshold", type=float, default=0.5)
    evaluate.add_argument("--smoke", action="store_true")
    evaluate.add_argument("--max-rows", type=int, default=0, help="Group-safe row cap allowed only with --smoke.")

    final = subparsers.add_parser("final-test", help="Fit a locked candidate on full development data and evaluate test once.")
    final.add_argument("--selection-summary", type=Path, required=True)
    final.add_argument("--locked-config-id", required=True)
    final.add_argument("--allow-final-test", action="store_true")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--concept-csv", type=Path, required=True)
    final.add_argument("--train-features", type=Path, required=True)
    final.add_argument("--val-features", type=Path, required=True)
    final.add_argument("--test-features", type=Path, required=True)
    final.add_argument("--oof-med-features", type=Path)
    final.add_argument("--final-test-med-features", type=Path)
    final.add_argument("--output-dir", type=Path, required=True, help="A new, non-existing output directory.")
    final.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_feature_hierarchy(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sizes = []
    for key, columns in data.items():
        if key.startswith("selected_"):
            try:
                nominal = int(key.split("_", 1)[1])
            except ValueError:
                continue
            if len(columns) != nominal:
                raise ValueError(f"saved feature set {key} contains {len(columns)} columns, expected {nominal}")
            sizes.append(nominal)
    expected = [18, 15, 12, 9, 6, 3]
    if sorted(set(sizes), reverse=True) != expected:
        raise ValueError(f"unexpected saved subset hierarchy: {sorted(set(sizes), reverse=True)}")
    return expected


def _load_combined(
    manifest_path: Path,
    concept_csv: Path,
    feature_paths: dict[str, Path],
    med_features: Path | None,
) -> tuple[pd.DataFrame, list[str], str, list[str]]:
    manifest, concept_columns, grouping_mode = prepare_med_frame(manifest_path, concept_csv)
    expected_med_columns = med_feature_names(concept_columns)
    blocks = []
    feature_columns: list[str] | None = None
    for split, path in feature_paths.items():
        rows = manifest[manifest["original_split"] == split].reset_index(drop=True)
        features = pd.read_csv(path).reset_index(drop=True)
        if len(rows) != len(features):
            raise ValueError(f"{split} row mismatch: manifest={len(rows)} features={len(features)}")
        if "label" not in features:
            raise ValueError(f"{path} has no label column")
        if not np.array_equal(features["label"].astype(int).to_numpy(), rows["true_label"].astype(int).to_numpy()):
            raise ValueError(f"{split} manifest/features label misalignment")
        current_columns = [column for column in features.columns if column != "label"]
        if feature_columns is None:
            feature_columns = current_columns
        elif current_columns != feature_columns:
            raise ValueError(f"{split} feature ordering differs")
        metadata = rows[["image_id", "file_name", "image_path", "true_label", "original_split", "group_id"]]
        blocks.append(pd.concat([metadata, features[current_columns]], axis=1))
    combined = pd.concat(blocks, ignore_index=True)
    assert feature_columns is not None
    if med_features:
        _replace_med_features(
            combined,
            feature_columns,
            med_features,
            expected_med_columns=expected_med_columns,
        )
    numeric = combined[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("combined features contain infinite values")
    return combined, feature_columns, grouping_mode, expected_med_columns


def _aligned_med_values(
    target: pd.DataFrame,
    feature_columns: list[str],
    med_features: Path,
    expected_mode: str,
    expected_med_columns: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    with med_features.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    duplicate_headers = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicate_headers:
        raise ValueError(f"{med_features} contains duplicate columns: {duplicate_headers}")
    med = pd.read_csv(med_features)
    required_metadata = {
        "image_id",
        "file_name",
        "image_path",
        "true_label",
        "original_split",
        "fold_id",
        "model_id",
        "feature_generation_mode",
    }
    missing_metadata = sorted(required_metadata - set(med.columns))
    if missing_metadata:
        raise ValueError(f"{med_features} lacks required metadata: {missing_metadata}")
    replacement_columns = [
        column
        for column in med.columns
        if column.startswith("concept_score_") or column.startswith("med_")
    ]
    existing_replacements = [
        column for column in feature_columns if column.startswith("concept_score_") or column.startswith("med_")
    ]
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("predictor feature list contains duplicate columns")
    expected_replacements = list(expected_med_columns or existing_replacements or replacement_columns)
    if existing_replacements and existing_replacements != expected_replacements:
        raise ValueError(
            "existing Med feature schema differs from the expected schema: "
            f"observed={existing_replacements} expected={expected_replacements}"
        )
    if replacement_columns != expected_replacements:
        missing = [column for column in expected_replacements if column not in replacement_columns]
        unexpected = [column for column in replacement_columns if column not in expected_replacements]
        raise ValueError(
            f"Med feature schema/order mismatch in {med_features}: observed={replacement_columns} "
            f"expected={expected_replacements} missing={missing} unexpected={unexpected}"
        )
    if med["image_path"].duplicated().any() or med["image_id"].duplicated().any():
        raise ValueError(f"{med_features} contains duplicate image_path or image_id values")
    lookup = med.set_index("image_path")
    missing = sorted(set(target["image_path"]) - set(lookup.index))
    if missing:
        raise ValueError(f"{med_features} misses {len(missing)} rows; first={missing[0]}")
    aligned = lookup.loc[target["image_path"]]
    if not np.array_equal(
        aligned["true_label"].astype(int).to_numpy(), target["true_label"].astype(int).to_numpy()
    ):
        raise ValueError(f"{med_features} image-to-label misalignment")
    if not np.array_equal(aligned["image_id"].astype(str).to_numpy(), target["image_id"].astype(str).to_numpy()):
        raise ValueError(f"{med_features} image-to-ID misalignment")
    if not np.array_equal(aligned["file_name"].astype(str).to_numpy(), target["file_name"].astype(str).to_numpy()):
        raise ValueError(f"{med_features} image-to-filename misalignment")
    if not np.array_equal(
        aligned["original_split"].astype(str).to_numpy(), target["original_split"].astype(str).to_numpy()
    ):
        raise ValueError(f"{med_features} image-to-split misalignment")
    if set(aligned["feature_generation_mode"].astype(str)) != {expected_mode}:
        raise ValueError(f"unexpected feature-generation mode in {med_features}; expected {expected_mode!r}")
    if aligned["fold_id"].isna().any() or aligned["model_id"].isna().any():
        raise ValueError(f"{med_features} contains missing fold/model provenance")
    values = aligned[replacement_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{med_features} contains non-finite Med features")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{med_features} contains Med probability values outside [0, 1]")
    return replacement_columns, values


def _replace_med_features(
    combined: pd.DataFrame,
    feature_columns: list[str],
    med_features: Path,
    expected_mode: str = "out-of-fold",
    expected_med_columns: Sequence[str] | None = None,
) -> None:
    replacement_columns, values = _aligned_med_values(
        combined,
        feature_columns,
        med_features,
        expected_mode,
        expected_med_columns,
    )
    combined.loc[:, replacement_columns] = values
    for column in replacement_columns:
        if column not in feature_columns:
            feature_columns.append(column)


def _stratified_group_splits(frame: pd.DataFrame, folds: int, seed: int):
    groups_per_class = frame.groupby("true_label")["group_id"].nunique()
    if folds > int(groups_per_class.min()):
        raise ValueError(f"folds={folds} exceeds smallest class group count={int(groups_per_class.min())}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(frame, frame["true_label"], groups=frame["group_id"]))


def _pipeline(feature_columns: list[str], k: int, seed: int, zero_threshold: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "zfmis",
                FoldSafeZFMIS(
                    feature_names=tuple(feature_columns),
                    k=k,
                    zero_fraction_threshold=zero_threshold,
                    random_state=seed,
                ),
            ),
            ("model", DecisionTreeClassifier(random_state=seed)),
        ]
    )


def _grid(smoke: bool) -> dict:
    if smoke:
        return {
            "model__criterion": ["gini"],
            "model__max_depth": [2, 4],
            "model__min_samples_leaf": [2],
            "model__min_samples_split": [2],
            "model__class_weight": [None, "balanced"],
        }
    return {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 8, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__class_weight": [None, "balanced"],
    }


def _calibration_partition(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.5, random_state=seed)
    for calibration, tuning in splitter.split(frame, groups=frame["group_id"]):
        if frame.iloc[calibration]["true_label"].nunique() == 2 and frame.iloc[tuning]["true_label"].nunique() == 2:
            return calibration, tuning
    raise ValueError("could not create group-disjoint calibration/threshold subsets with both classes")


def _fit_sigmoid(raw_probability: np.ndarray, labels: np.ndarray, seed: int):
    model = LogisticRegression(random_state=seed, solver="lbfgs")
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model.fit(logits, labels)
    return model


def _apply_sigmoid(model, raw_probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def _threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), probabilities]))
    best = (float("-inf"), float("-inf"), float("-inf"), -0.5)
    chosen = 0.5
    for candidate in candidates:
        prediction = (probabilities >= candidate).astype(int)
        key = (
            balanced_accuracy_score(labels, prediction),
            f1_score(labels, prediction, zero_division=0),
            accuracy_score(labels, prediction),
            -abs(float(candidate) - 0.5),
        )
        if key > best:
            best, chosen = key, float(candidate)
    return chosen


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _inner_oof_probabilities(
    estimator: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    probabilities = np.full(len(x), np.nan, dtype=float)
    for train_indices, validation_indices in splits:
        fold_estimator = clone(estimator)
        fold_estimator.fit(x.iloc[train_indices], y[train_indices])
        probabilities[validation_indices] = fold_estimator.predict_proba(x.iloc[validation_indices])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("inner OOF probability generation left missing/non-finite values")
    return probabilities


def _ranking_rows(selector: FoldSafeZFMIS, fold: int, k: int, model: str) -> list[dict]:
    selected = set(selector.selected_feature_names_)
    rows = []
    for index, name in enumerate(selector.feature_names):
        rows.append(
            {
                "model": model,
                "subset_size": k,
                "fold_id": fold,
                "feature": name,
                "selected": name in selected,
                "rank": selector.ranks_[index],
                "mutual_information": selector.mi_scores_[index],
                "zero_fraction": selector.zero_fraction_[index],
            }
        )
    return rows


def _rulefit_parameter_grid(smoke: bool) -> list[dict]:
    if smoke:
        return [{"tree_size": 3, "sample_fract": 0.8, "max_rules": 25}]
    return [
        {"tree_size": tree_size, "sample_fract": 0.8, "max_rules": max_rules}
        for tree_size in [3, 4, 5]
        for max_rules in [50, 100, 200]
    ]


def _fit_rulefit_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    k: int,
    params: dict,
    seed: int,
    zero_threshold: float,
):
    from rulefit import RuleFit

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train[feature_columns])
    validation_imputed = imputer.transform(validation[feature_columns])
    selector = FoldSafeZFMIS(tuple(feature_columns), k, zero_threshold, seed).fit(
        train_imputed, train["true_label"].to_numpy(dtype=int)
    )
    train_selected = selector.transform(train_imputed)
    validation_selected = selector.transform(validation_imputed)
    model = RuleFit(rfmode="classify", model_type="r", random_state=seed, **params)
    model.fit(train_selected, train["true_label"].to_numpy(dtype=int), feature_names=selector.selected_feature_names_)
    prediction = np.asarray(model.predict(validation_selected)).astype(int)
    return imputer, selector, model, prediction


def _evaluate_rulefit_candidates(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sizes: list[int],
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    seed: int,
    zero_threshold: float,
    smoke: bool,
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[pd.DataFrame]]:
    metric_rows: list[dict] = []
    ranking_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    rules_dir = output_dir / "rulefit_rules"
    rules_dir.mkdir(parents=True, exist_ok=False)
    for k in sizes:
        for fold_index, (outer_train_indices, outer_validation_indices) in enumerate(outer_splits):
            outer_train = frame.iloc[outer_train_indices].reset_index(drop=True)
            outer_validation = frame.iloc[outer_validation_indices].reset_index(drop=True)
            inner_splits = _stratified_group_splits(outer_train, inner_folds, seed + fold_index + k)
            best_params = None
            best_score = float("-inf")
            for params in _rulefit_parameter_grid(smoke):
                scores = []
                for inner_train_indices, inner_validation_indices in inner_splits:
                    _, _, _, prediction = _fit_rulefit_fold(
                        outer_train.iloc[inner_train_indices],
                        outer_train.iloc[inner_validation_indices],
                        feature_columns,
                        k,
                        params,
                        seed + fold_index,
                        zero_threshold,
                    )
                    scores.append(
                        balanced_accuracy_score(
                            outer_train.iloc[inner_validation_indices]["true_label"], prediction
                        )
                    )
                score = float(np.mean(scores))
                if score > best_score:
                    best_score, best_params = score, params
            assert best_params is not None
            _, selector, model, prediction = _fit_rulefit_fold(
                outer_train,
                outer_validation,
                feature_columns,
                k,
                best_params,
                seed + fold_index,
                zero_threshold,
            )
            labels = outer_validation["true_label"].to_numpy(dtype=int)
            metrics = _metrics(labels, prediction.astype(float), 0.5)
            metrics["roc_auc"] = None
            metric_rows.append(
                {
                    "model": "rulefit",
                    "subset_size": k,
                    "fold_id": fold_index,
                    "outer_train_rows": len(outer_train),
                    "outer_validation_rows": len(outer_validation),
                    "inner_cv_balanced_accuracy": best_score,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "calibration_rows": 0,
                    "threshold_tuning_rows": 0,
                    "threshold": 0.5,
                    **metrics,
                }
            )
            ranking_rows.extend(_ranking_rows(selector, fold_index, k, "rulefit"))
            fold_predictions = outer_validation[["image_id", "image_path", "true_label", "original_split", "group_id"]].copy()
            fold_predictions["model"] = "rulefit"
            fold_predictions["subset_size"] = k
            fold_predictions["fold_id"] = fold_index
            fold_predictions["raw_probability"] = np.nan
            fold_predictions["calibrated_probability"] = prediction.astype(float)
            fold_predictions["threshold"] = 0.5
            fold_predictions["prediction"] = prediction
            prediction_rows.append(fold_predictions)
            rules = model.get_rules()
            rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
            rules.to_csv(rules_dir / f"rulefit_k{k}_fold{fold_index}.csv", index=False)
    return metric_rows, ranking_rows, prediction_rows


def _stability(rankings: pd.DataFrame, folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    jaccard_rows = []
    for (model, k), group in rankings.groupby(["model", "subset_size"]):
        for feature, feature_rows in group.groupby("feature"):
            selected = feature_rows[feature_rows["selected"]]
            rows.append(
                {
                    "model": model,
                    "subset_size": int(k),
                    "feature": feature,
                    "selection_count": int(len(selected)),
                    "selection_frequency": float(len(selected) / folds),
                    "mean_rank": float(feature_rows["rank"].mean()),
                    "std_rank": float(feature_rows["rank"].std(ddof=0)),
                }
            )
        sets = {
            fold: set(fold_rows.loc[fold_rows["selected"], "feature"])
            for fold, fold_rows in group.groupby("fold_id")
        }
        scores = []
        keys = sorted(sets)
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1 :]:
                union = sets[left] | sets[right]
                scores.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
        jaccard_rows.append(
            {
                "model": model,
                "subset_size": int(k),
                "mean_pairwise_jaccard": float(np.mean(scores)) if scores else 1.0,
                "std_pairwise_jaccard": float(np.std(scores)) if scores else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(jaccard_rows)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    sizes = _load_feature_hierarchy(args.feature_set_json)
    if args.subset_sizes:
        invalid = sorted(set(args.subset_sizes) - set(sizes))
        if invalid:
            raise ValueError(f"requested subset sizes are not in the verified hierarchy: {invalid}")
        sizes = args.subset_sizes
    if args.max_rows and not args.smoke:
        raise ValueError("--max-rows requires --smoke")
    frame, feature_columns, grouping_mode, expected_med_columns = _load_combined(
        args.manifest,
        args.concept_csv,
        {"train": args.train_features, "val": args.val_features},
        None,
    )
    if args.max_rows:
        frame = deterministic_group_subset(frame, args.max_rows, args.seed)
    if args.oof_med_features:
        _replace_med_features(
            frame,
            feature_columns,
            args.oof_med_features,
            expected_med_columns=expected_med_columns,
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    skipped = []
    models = list(args.models)
    if "rulefit" in models and importlib.util.find_spec("rulefit") is None:
        skipped.append(
            {
                "model": "rulefit",
                "status": "SKIPPED_DEPENDENCY_MISSING",
                "reason": "Python package 'rulefit' is not importable; no substitute model was reported as RuleFit.",
            }
        )
        models.remove("rulefit")
    if not models:
        raise RuntimeError("no evaluable models remain")

    outer_splits = _stratified_group_splits(frame, args.outer_folds, args.seed)
    metric_rows, ranking_rows, prediction_rows = [], [], []
    if "decision_tree" in models:
        for k in sizes:
          for fold_index, (outer_train_indices, outer_validation_indices) in enumerate(outer_splits):
            outer_train = frame.iloc[outer_train_indices].reset_index(drop=True)
            outer_validation = frame.iloc[outer_validation_indices].reset_index(drop=True)
            if set(outer_train["group_id"]) & set(outer_validation["group_id"]):
                raise RuntimeError(f"outer fold {fold_index} has group overlap")
            x_train = outer_train[feature_columns]
            y_train = outer_train["true_label"].to_numpy(dtype=int)
            inner_splits = _stratified_group_splits(outer_train, args.inner_folds, args.seed + fold_index + k)
            estimator = _pipeline(feature_columns, k, args.seed + fold_index, args.zero_fraction_threshold)
            grid = GridSearchCV(
                estimator,
                _grid(args.smoke),
                scoring="balanced_accuracy",
                cv=inner_splits,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            grid.fit(x_train, y_train)
            best = grid.best_estimator_
            inner_raw = _inner_oof_probabilities(best, x_train, y_train, inner_splits)
            calibration_indices, tuning_indices = _calibration_partition(outer_train, args.seed + 1000 + fold_index + k)
            calibrator = _fit_sigmoid(inner_raw[calibration_indices], y_train[calibration_indices], args.seed + fold_index)
            tuning_probability = _apply_sigmoid(calibrator, inner_raw[tuning_indices])
            threshold = _threshold(y_train[tuning_indices], tuning_probability)
            raw_probability = best.predict_proba(outer_validation[feature_columns])[:, 1]
            calibrated_probability = _apply_sigmoid(calibrator, raw_probability)
            metrics = _metrics(outer_validation["true_label"].to_numpy(dtype=int), calibrated_probability, threshold)
            metric_rows.append(
                {
                    "model": "decision_tree",
                    "subset_size": k,
                    "fold_id": fold_index,
                    "outer_train_rows": len(outer_train),
                    "outer_validation_rows": len(outer_validation),
                    "inner_cv_balanced_accuracy": float(grid.best_score_),
                    "best_params": json.dumps(grid.best_params_, sort_keys=True),
                    "calibration_rows": len(calibration_indices),
                    "threshold_tuning_rows": len(tuning_indices),
                    "threshold": threshold,
                    **metrics,
                }
            )
            selector: FoldSafeZFMIS = best.named_steps["zfmis"]
            ranking_rows.extend(_ranking_rows(selector, fold_index, k, "decision_tree"))
            fold_predictions = outer_validation[["image_id", "image_path", "true_label", "original_split", "group_id"]].copy()
            fold_predictions["model"] = "decision_tree"
            fold_predictions["subset_size"] = k
            fold_predictions["fold_id"] = fold_index
            fold_predictions["raw_probability"] = raw_probability
            fold_predictions["calibrated_probability"] = calibrated_probability
            fold_predictions["threshold"] = threshold
            fold_predictions["prediction"] = (calibrated_probability >= threshold).astype(int)
            prediction_rows.append(fold_predictions)

    if "rulefit" in models:
        try:
            rulefit_metrics, rulefit_rankings, rulefit_predictions = _evaluate_rulefit_candidates(
                frame,
                feature_columns,
                sizes,
                outer_splits,
                args.inner_folds,
                args.seed,
                args.zero_fraction_threshold,
                args.smoke,
                args.output_dir,
            )
            metric_rows.extend(rulefit_metrics)
            ranking_rows.extend(rulefit_rankings)
            prediction_rows.extend(rulefit_predictions)
        except Exception as exc:
            skipped.append(
                {
                    "model": "rulefit",
                    "status": "SKIPPED_RUNTIME_INCOMPATIBLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not metric_rows:
        raise RuntimeError(f"no model completed evaluation; skipped={skipped}")

    metrics = pd.DataFrame(metric_rows)
    rankings = pd.DataFrame(ranking_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    stability, subset_stability = _stability(rankings, args.outer_folds)
    aggregate = (
        metrics.groupby(["model", "subset_size"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            mcc_mean=("mcc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .sort_values(["balanced_accuracy_mean", "f1_mean", "subset_size"], ascending=[False, False, True])
    )
    locked = aggregate.iloc[0].to_dict()
    locked_metrics = metrics[
        (metrics["model"] == locked["model"]) & (metrics["subset_size"] == locked["subset_size"])
    ]
    parameter_mode = Counter(locked_metrics["best_params"]).most_common(1)[0][0]
    lock_payload = {
        "model": locked["model"],
        "subset_size": int(locked["subset_size"]),
        "model_params": json.loads(parameter_mode),
        "zero_fraction_threshold": args.zero_fraction_threshold,
        "feature_columns": feature_columns,
        "upstream_med_features": "oof" if args.oof_med_features else "current_single_checkpoint",
        "seed": args.seed,
    }
    locked_id = configuration_id(lock_payload)
    upstream_scientific = False
    scientific_limitation = "historical single-checkpoint Med training features retain upstream in-sample leakage"
    if args.oof_med_features:
        upstream_summary_path = args.oof_med_features.parent / "oof_summary.json"
        if upstream_summary_path.is_file():
            upstream_summary = json.loads(upstream_summary_path.read_text(encoding="utf-8"))
            upstream_scientific = bool(upstream_summary.get("scientific_result", False))
            scientific_limitation = (
                None
                if upstream_scientific
                else "upstream OOF Med features are explicitly non-scientific smoke outputs"
            )
        else:
            scientific_limitation = "upstream OOF summary is missing, so scientific provenance is unverified"
    if args.smoke:
        scientific_limitation = "fold-safe evaluation used the reduced non-scientific smoke protocol"
    summary = {
        "scientific_result": upstream_scientific and not args.smoke,
        "scientific_limitation": scientific_limitation,
        "evaluation_scope": "development-only nested cross-validation",
        "test_features_loaded": False,
        "test_labels_used": False,
        "grouping_mode": grouping_mode,
        "verified_subset_sizes": _load_feature_hierarchy(args.feature_set_json),
        "evaluated_subset_sizes": sizes,
        "upstream_med_feature_mode": lock_payload["upstream_med_features"],
        "locked_candidate": lock_payload,
        "locked_configuration_id": locked_id,
        "skipped_models": skipped,
    }
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    rankings.to_csv(args.output_dir / "selected_features_by_fold.csv", index=False)
    stability.to_csv(args.output_dir / "feature_selection_stability.csv", index=False)
    subset_stability.to_csv(args.output_dir / "subset_stability.csv", index=False)
    predictions.to_csv(args.output_dir / "development_oof_predictions.csv", index=False)
    aggregate.to_csv(args.output_dir / "model_comparison.csv", index=False)
    write_json(args.output_dir / "skipped_models.json", skipped)
    write_json(args.output_dir / "selection_summary.json", summary)
    return summary


def final_test(args: argparse.Namespace) -> dict:
    if not args.allow_final_test:
        raise PermissionError("final-test evaluation requires --allow-final-test after configuration lock")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    if selection["locked_configuration_id"] != args.locked_config_id:
        raise ValueError("locked configuration ID mismatch")
    locked = selection["locked_candidate"]
    frame, feature_columns, grouping_mode, expected_med_columns = _load_combined(
        args.manifest,
        args.concept_csv,
        {"train": args.train_features, "val": args.val_features, "test": args.test_features},
        None,
    )
    locked_feature_columns = locked.get("feature_columns")
    if not isinstance(locked_feature_columns, list) or not all(
        isinstance(column, str) for column in locked_feature_columns
    ):
        raise ValueError("locked candidate has no valid ordered feature_columns list")
    if len(locked_feature_columns) != len(set(locked_feature_columns)):
        raise ValueError("locked candidate feature_columns contains duplicates")
    development = frame[frame["original_split"].isin(["train", "val"])].reset_index(drop=True)
    test = frame[frame["original_split"] == "test"].reset_index(drop=True)
    if set(development["group_id"]) & set(test["group_id"]):
        raise ValueError("development/test group overlap; final evaluation refused")

    def replace_med(target: pd.DataFrame, path: Path | None, expected_mode: str) -> None:
        if path is None:
            return
        _replace_med_features(
            target,
            feature_columns,
            path,
            expected_mode=expected_mode,
            expected_med_columns=expected_med_columns,
        )

    replace_med(development, args.oof_med_features, "out-of-fold")
    replace_med(test, args.final_test_med_features, "final-test inference")
    if bool(args.oof_med_features) != bool(args.final_test_med_features):
        raise ValueError("OOF development Med features and final-test Med features must be supplied together")
    if feature_columns != locked_feature_columns:
        raise ValueError(
            "final predictor schema differs from the development-locked schema: "
            f"observed={feature_columns} locked={locked_feature_columns}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    k = int(locked["subset_size"])
    labels = development["true_label"].to_numpy(dtype=int)
    if locked["model"] == "decision_tree":
        estimator = _pipeline(feature_columns, k, args.seed, float(locked["zero_fraction_threshold"]))
        estimator.set_params(**locked["model_params"])
        outer_splits = _stratified_group_splits(development, 5, args.seed)
        dev_raw = _inner_oof_probabilities(
            estimator,
            development[feature_columns],
            labels,
            outer_splits,
        )
        calibration_indices, tuning_indices = _calibration_partition(development, args.seed + 5000)
        calibrator = _fit_sigmoid(dev_raw[calibration_indices], labels[calibration_indices], args.seed)
        threshold = _threshold(labels[tuning_indices], _apply_sigmoid(calibrator, dev_raw[tuning_indices]))
        estimator.fit(development[feature_columns], labels)
        test_raw = estimator.predict_proba(test[feature_columns])[:, 1]
        test_probability = _apply_sigmoid(calibrator, test_raw)
        selector: FoldSafeZFMIS = estimator.named_steps["zfmis"]
        ranking = pd.DataFrame(_ranking_rows(selector, 0, k, "decision_tree"))
        rules = export_text(estimator.named_steps["model"], feature_names=selector.selected_feature_names_)
        (args.output_dir / "final_decision_tree_rules.txt").write_text(rules, encoding="utf-8")
        joblib.dump({"estimator": estimator, "calibrator": calibrator, "threshold": threshold}, args.output_dir / "final_model.joblib")
    elif locked["model"] == "rulefit":
        imputer, selector, model, test_prediction = _fit_rulefit_fold(
            development,
            test,
            feature_columns,
            k,
            locked["model_params"],
            args.seed,
            float(locked["zero_fraction_threshold"]),
        )
        threshold = 0.5
        test_raw = np.full(len(test_prediction), np.nan)
        test_probability = test_prediction.astype(float)
        ranking = pd.DataFrame(_ranking_rows(selector, 0, k, "rulefit"))
        rules = model.get_rules()
        rules[rules.coef != 0].sort_values("importance", ascending=False).to_csv(
            args.output_dir / "final_rulefit_rules.csv", index=False
        )
        joblib.dump({"imputer": imputer, "selector": selector, "model": model, "threshold": threshold}, args.output_dir / "final_model.joblib")
    else:
        raise ValueError(f"unsupported locked model: {locked['model']}")
    test_metrics = _metrics(test["true_label"].to_numpy(dtype=int), test_probability, threshold)
    if locked["model"] == "rulefit":
        test_metrics["roc_auc"] = None
    predictions = test[["image_id", "file_name", "image_path", "true_label", "original_split"]].copy()
    predictions["raw_probability"] = test_raw
    predictions["calibrated_probability"] = test_probability
    predictions["threshold"] = threshold
    predictions["prediction"] = (test_probability >= threshold).astype(int)
    predictions.to_csv(args.output_dir / "final_test_predictions.csv", index=False)
    ranking.to_csv(args.output_dir / "final_selected_feature_ranking.csv", index=False)
    summary = {
        "locked_configuration_id": args.locked_config_id,
        "selection_summary": str(args.selection_summary.resolve()),
        "test_evaluation_count": 1,
        "test_used_for_selection": False,
        "grouping_mode": grouping_mode,
        "upstream_med_feature_mode": "oof_and_final" if args.oof_med_features else "current_single_checkpoint",
        "threshold": threshold,
        "metrics": test_metrics,
    }
    write_json(args.output_dir / "final_test_metrics.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = evaluate(args) if args.command == "evaluate" else final_test(args)
    if args.command == "evaluate":
        print(f"Locked configuration ID: {summary['locked_configuration_id']}")
        print(f"Test features loaded: {summary['test_features_loaded']}")
    else:
        print(f"Test evaluation count: {summary['test_evaluation_count']}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
