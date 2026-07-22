from argparse import Namespace
from pathlib import Path

import pandas as pd
from PIL import Image

from ss_virulex_reliability.audit_leakage import run_audit


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (24, 24), (value, value, value)).save(path)


def test_audit_detects_cross_split_exact_copy_and_augmented_validation(tmp_path):
    train = tmp_path / "train.png"
    test = tmp_path / "test.png"
    augmented = tmp_path / "augmented.png"
    _image(train, 80)
    test.write_bytes(train.read_bytes())
    _image(augmented, 120)
    manifest = pd.DataFrame(
        [
            {"filepath": train, "filename": train.name, "split": "train", "class_name": "COVID", "label": 1, "source": "original"},
            {"filepath": test, "filename": test.name, "split": "test", "class_name": "COVID", "label": 1, "source": "original"},
            {"filepath": augmented, "filename": augmented.name, "split": "val", "class_name": "COVID", "label": 1, "source": "augmented", "source_filepath": train},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    args = Namespace(
        manifest=manifest_path,
        official_manifest=None,
        metadata_csv=None,
        provenance_json=None,
        output_dir=tmp_path / "audit",
        markdown=tmp_path / "audit.md",
        near_threshold=6,
        near_correlation=0.90,
        near_mae=0.08,
        max_near_pairs=100,
        skip_perceptual=False,
        overwrite=False,
    )
    summary = run_audit(args)
    checks = {item["check"]: item for item in summary["checks"]}
    assert checks["exact_duplicates_cross_split"]["status"] == "FAIL"
    assert checks["augmented_samples_train_only"]["status"] == "FAIL"
    assert checks["patient_overlap"]["finding"].startswith("not verifiable")


def test_audit_refuses_silent_overwrite(tmp_path):
    image = tmp_path / "image.png"
    _image(image, 50)
    manifest = pd.DataFrame(
        [{"filepath": image, "filename": image.name, "split": "train", "class_name": "NonCOVID", "label": 0, "source": "original"}]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output = tmp_path / "audit"
    output.mkdir()
    (output / "existing.csv").write_text("keep\n", encoding="utf-8")
    args = Namespace(
        manifest=manifest_path,
        official_manifest=None,
        metadata_csv=None,
        provenance_json=None,
        output_dir=output,
        markdown=tmp_path / "audit.md",
        near_threshold=6,
        near_correlation=0.90,
        near_mae=0.08,
        max_near_pairs=100,
        skip_perceptual=True,
        overwrite=False,
    )
    try:
        run_audit(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit silently overwrote an existing output directory")
