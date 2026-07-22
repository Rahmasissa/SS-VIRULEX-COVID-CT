import numpy as np
import pandas as pd
import pytest

from ss_virulex_reliability.fold_safe_fusion import (
    FoldSafeZFMIS,
    _aligned_med_values,
    _replace_med_features,
)


def test_zero_filter_and_mi_are_fitted_from_training_fold_only():
    x_train = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.1, 0.2],
            [0.0, 0.9, 0.2],
            [0.0, 1.0, 0.1],
            [0.0, 0.2, 0.3],
            [0.0, 0.8, 0.3],
        ]
    )
    y_train = np.asarray([0, 0, 1, 1, 0, 1])
    selector = FoldSafeZFMIS(("all_zero_train", "informative", "noise"), k=1, random_state=7)
    selector.fit(x_train, y_train)
    assert selector.zero_fraction_[0] == 1.0
    assert selector.selected_feature_names_ == ["informative"]
    x_validation = np.asarray([[999.0, 0.4, 0.5], [999.0, 0.6, 0.5]])
    transformed = selector.transform(x_validation)
    assert transformed.shape == (2, 1)
    assert np.allclose(transformed[:, 0], [0.4, 0.6])


def test_selector_feature_order_is_explicit_and_deterministic():
    x = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
    y = np.asarray([0, 0, 1, 1])
    first = FoldSafeZFMIS(("a", "b"), k=2, random_state=42).fit(x, y)
    second = FoldSafeZFMIS(("a", "b"), k=2, random_state=42).fit(x, y)
    assert first.selected_feature_names_ == second.selected_feature_names_
    assert list(first.get_feature_names_out()) == first.selected_feature_names_


def test_med_replacement_rejects_metadata_misalignment(tmp_path):
    columns = ["concept_score_a", "med_task_prob_covid", "med_neural_prob_covid"]
    target = pd.DataFrame(
        {
            "image_id": ["expected"],
            "file_name": ["image.png"],
            "image_path": ["/image.png"],
            "true_label": [1],
            "original_split": ["train"],
        }
    )
    med = target.copy()
    med["image_id"] = "wrong"
    med["fold_id"] = 0
    med["model_id"] = "model"
    med["feature_generation_mode"] = "out-of-fold"
    med[columns] = [[0.2, 0.3, 0.4]]
    path = tmp_path / "misaligned.csv"
    med.to_csv(path, index=False)
    with pytest.raises(ValueError, match="image-to-ID misalignment"):
        _aligned_med_values(target, columns, path, "out-of-fold")


def _med_frames(columns):
    target = pd.DataFrame(
        {
            "image_id": ["id-1"],
            "file_name": ["image.png"],
            "image_path": ["/image.png"],
            "true_label": [1],
            "original_split": ["train"],
            "mean": [0.25],
        }
    )
    med = target[["image_id", "file_name", "image_path", "true_label", "original_split"]].copy()
    med["fold_id"] = 0
    med["model_id"] = "model"
    med["feature_generation_mode"] = "out-of-fold"
    for index, column in enumerate(columns):
        med[column] = 0.1 * (index + 1)
    return target, med


def test_med_columns_are_inserted_in_explicit_order_when_placeholders_are_absent(tmp_path):
    med_columns = ["concept_score_a", "med_task_prob_covid", "med_neural_prob_covid"]
    target, med = _med_frames(med_columns)
    path = tmp_path / "med.csv"
    med.to_csv(path, index=False)
    feature_columns = ["mean"]

    _replace_med_features(
        target,
        feature_columns,
        path,
        expected_med_columns=med_columns,
    )

    assert feature_columns == ["mean", *med_columns]
    assert list(target[med_columns].iloc[0]) == pytest.approx([0.1, 0.2, 0.3])


def test_med_replacement_rejects_missing_or_misordered_features(tmp_path):
    expected = ["concept_score_a", "med_task_prob_covid", "med_neural_prob_covid"]
    target, med = _med_frames(["med_task_prob_covid", "concept_score_a", "med_neural_prob_covid"])
    misordered = tmp_path / "misordered.csv"
    med.to_csv(misordered, index=False)
    with pytest.raises(ValueError, match="schema/order mismatch"):
        _aligned_med_values(target, ["mean"], misordered, "out-of-fold", expected)

    target, med = _med_frames(expected[:-1])
    missing = tmp_path / "missing.csv"
    med.to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing=.*med_neural_prob_covid"):
        _aligned_med_values(target, ["mean"], missing, "out-of-fold", expected)


def test_med_replacement_rejects_duplicate_csv_headers(tmp_path):
    expected = ["concept_score_a", "med_task_prob_covid", "med_neural_prob_covid"]
    target, med = _med_frames(expected)
    duplicate = tmp_path / "duplicate.csv"
    header = list(med.columns) + ["concept_score_a"]
    row = [str(value) for value in med.iloc[0].tolist()] + ["0.9"]
    duplicate.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate columns"):
        _aligned_med_values(target, ["mean"], duplicate, "out-of-fold", expected)
