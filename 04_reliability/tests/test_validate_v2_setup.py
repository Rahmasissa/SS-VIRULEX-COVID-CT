from pathlib import Path
from types import SimpleNamespace

import importlib.metadata

import pandas as pd
import pytest
from PIL import Image

from ss_virulex_reliability.validate_v2_setup import (
    COMPLETED_OUTPUT_ROOT,
    EXPECTED_CONCEPT_COLUMNS,
    EXPECTED_CONFIGURATION_ID,
    SetupValidationError,
    parse_exact_requirements,
    validate_locked_config,
    validate_manifest_contract,
    validate_opencv_namespace,
    validate_output_path,
)


def _write_image(path: Path, value: int) -> None:
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


def _development_inputs(tmp_path: Path, second_split: str = "val") -> tuple[Path, Path, Path]:
    train_image = tmp_path / "train.png"
    second_image = tmp_path / "second.png"
    _write_image(train_image, 40)
    _write_image(second_image, 180)
    manifest = pd.DataFrame(
        [
            {
                "filepath": train_image,
                "filename": train_image.name,
                "split": "train",
                "class_name": "NonCOVID",
                "label": 0,
                "source": "original",
            },
            {
                "filepath": second_image,
                "filename": second_image.name,
                "split": second_split,
                "class_name": "COVID",
                "label": 1,
                "source": "original",
            },
        ]
    )
    concepts = pd.DataFrame(
        [
            {
                "image_id": "train-id",
                "file_name": train_image.name,
                "class_name": "NonCOVID",
                "true_label": 0,
                "split": "train",
                "image_path": train_image,
                **{column: 0 for column in EXPECTED_CONCEPT_COLUMNS},
            },
            {
                "image_id": "second-id",
                "file_name": second_image.name,
                "class_name": "COVID",
                "true_label": 1,
                "split": second_split,
                "image_path": second_image,
                **{column: 1 for column in EXPECTED_CONCEPT_COLUMNS},
            },
        ]
    )
    manifest_path = tmp_path / "augmented_splits.csv"
    official_path = tmp_path / "official_splits.csv"
    concept_path = tmp_path / "concept_labels_development.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest.to_csv(official_path, index=False)
    concepts.to_csv(concept_path, index=False)
    return manifest_path, official_path, concept_path


def test_manifest_contract_accepts_eight_concept_group_disjoint_development_data(tmp_path):
    manifest, official, concepts = _development_inputs(tmp_path)
    result = validate_manifest_contract(
        manifest,
        official,
        concepts,
        expected_rows=2,
        expected_split_counts={"train": 1, "val": 1},
    )
    assert result["rows"] == 2
    assert result["concept_count"] == 8
    assert result["group_overlap_count"] == 0
    assert result["official_test_rows_accessed"] == 0


def test_manifest_contract_rejects_test_rows(tmp_path):
    manifest, official, concepts = _development_inputs(tmp_path, second_split="test")
    with pytest.raises(SetupValidationError, match="forbidden split"):
        validate_manifest_contract(
            manifest,
            official,
            concepts,
            expected_rows=2,
            expected_split_counts=None,
        )


def test_locked_v2_baseline_is_the_preserved_configuration():
    repo_root = Path(__file__).resolve().parents[2]
    result = validate_locked_config(repo_root)
    assert result["configuration_id"] == EXPECTED_CONFIGURATION_ID
    assert len(set(Path(path).read_bytes() for path in result["paths"])) == 1


def test_output_path_cannot_overwrite_package_or_existing_run(tmp_path):
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    output = tmp_path / "v2"
    repo.mkdir()
    package.mkdir()
    target = validate_output_path(repo, package, output, "setup-001")
    target.mkdir(parents=True)
    with pytest.raises(SetupValidationError, match="overwrite"):
        validate_output_path(repo, package, output, "setup-001")
    with pytest.raises(SetupValidationError, match="preserved Colab package"):
        validate_output_path(repo, package, package / "outputs", "setup-002")
    with pytest.raises(SetupValidationError, match="completed reliability run"):
        validate_output_path(repo, package, COMPLETED_OUTPUT_ROOT, "setup-003")


def test_output_path_rejects_descendant_of_completed_run(tmp_path):
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    repo.mkdir()
    package.mkdir()
    with pytest.raises(SetupValidationError, match="completed reliability run"):
        validate_output_path(repo, package, COMPLETED_OUTPUT_ROOT / "nested", "setup-nested")


def test_requirements_must_be_exactly_pinned(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==1.26.4\n# supplied separately\n", encoding="utf-8")
    assert parse_exact_requirements(requirements) == {"numpy": "1.26.4"}
    requirements.write_text("numpy>=1.26\n", encoding="utf-8")
    with pytest.raises(SetupValidationError, match="non-exact requirement"):
        parse_exact_requirements(requirements)


def test_opencv_namespace_requires_intended_import_version_only():
    versions = {"opencv-python-headless": "4.10.0.84"}

    def version_getter(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    result = validate_opencv_namespace(
        "4.10.0.84",
        version_getter=version_getter,
        importer=lambda name: SimpleNamespace(__version__="4.10.0", __file__="/runtime/cv2.so"),
    )
    assert result["import_version"] == "4.10.0"
    assert result["conflicting_distributions"] == []


def test_opencv_namespace_rejects_competing_distribution():
    versions = {
        "opencv-python-headless": "4.10.0.84",
        "opencv-python": "5.0.0.93",
    }

    def version_getter(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    with pytest.raises(SetupValidationError, match="conflicting OpenCV distributions"):
        validate_opencv_namespace(
            "4.10.0.84",
            version_getter=version_getter,
            importer=lambda name: SimpleNamespace(__version__="4.10.0", __file__="/runtime/cv2.so"),
        )
