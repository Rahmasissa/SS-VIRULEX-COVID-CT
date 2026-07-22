from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from ss_virulex_reliability.oof_med_micn import generate_oof


CONCEPT_NAMES = [
    "peripheral_ground_glass_opacities",
    "bilateral_involvement",
    "multilobar_distribution",
    "crazy_paving_pattern",
    "absence_of_lobar_consolidation",
    "localized_or_diffuse_presentation",
    "increased_density_in_the_lung",
    "ground_glass_appearance",
]


def test_oof_rows_are_unique_aligned_and_group_excluded(tmp_path):
    manifest_rows = []
    concept_rows = []
    for index in range(28):
        label = index % 2
        split = "test" if index >= 24 else "train" if index < 16 else "val"
        file_name = f"image_{index}.png"
        path = tmp_path / file_name
        Image.new("RGB", (20, 20), (30 + index * 3, 60 + label * 80, 90)).save(path)
        manifest_rows.append(
            {"filepath": path, "filename": file_name, "split": split, "class_name": "COVID" if label else "NonCOVID", "label": label, "source": "original"}
        )
        row = {
            "image_id": f"id_{index}",
            "file_name": file_name,
            "class_name": "COVID" if label else "NonCOVID",
            "true_label": label,
            "split": split,
            "image_path": path,
        }
        for concept_index, name in enumerate(CONCEPT_NAMES):
            row[f"concept_{name}_label"] = (label + concept_index) % 2
        concept_rows.append(row)
    manifest_path = tmp_path / "manifest.csv"
    concept_path = tmp_path / "concepts.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(concept_rows).to_csv(concept_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"experiment_id": "test_oof", "enabled": True}), encoding="utf-8")
    output = tmp_path / "oof"
    args = Namespace(
        manifest=manifest_path,
        concept_csv=concept_path,
        encoder_config=config_path,
        med_source_root=tmp_path,
        output_dir=output,
        backend="smoke",
        seed=42,
        folds=2,
        development_splits=["train", "val"],
        max_rows=0,
    )
    summary = generate_oof(args)
    features = pd.read_csv(output / "oof_med_micn_features.csv")
    assert summary["test_rows_accessed"] == 0
    assert len(features) == 24
    assert features["image_id"].is_unique
    assert set(features["feature_generation_mode"]) == {"out-of-fold"}
    assert not features["original_split"].eq("test").any()
    for fold in [0, 1]:
        holdout_ids = set(features.loc[features["fold_id"] == fold, "image_id"])
        provenance = pd.read_csv(output / "checkpoints" / f"fold_{fold}" / "training_provenance.csv")
        assert not holdout_ids.intersection(provenance["image_id"])

    resumed_args = Namespace(**vars(args), resume=True)
    resumed_summary = generate_oof(resumed_args)
    assert all(detail["resumed"] for detail in resumed_summary["fold_details"])

    cached = output / "checkpoints" / "fold_0" / "fold_features.csv"
    corrupted = pd.read_csv(cached)
    corrupted["model_id"] = "wrong-model"
    corrupted.to_csv(cached, index=False)
    with pytest.raises(ValueError, match="different model/configuration"):
        generate_oof(resumed_args)


def test_final_test_requires_explicit_lock_acknowledgement(tmp_path):
    from ss_virulex_reliability.oof_med_micn import generate_final_test

    args = Namespace(allow_final_test=False)
    try:
        generate_final_test(args)
    except PermissionError:
        pass
    else:
        raise AssertionError("final test inference ran without explicit acknowledgement")
