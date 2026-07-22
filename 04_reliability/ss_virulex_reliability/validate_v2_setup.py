from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


VALIDATED_CHECKPOINT = "78a06737d67b46b3961dab2f8a26c16db7de2b24"
EXPECTED_BRANCH = "reliability-integration"
AUTHORITATIVE_REPO_ROOT = Path("/content/SS-VIRULEX-COVID-CT")
AUTHORITATIVE_PACKAGE_ROOT = Path("/content/drive/MyDrive/colab_package")
AUTHORITATIVE_OUTPUT_ROOT = Path("/content/drive/MyDrive/SS_VIRULEX_Reliability_V2")
COMPLETED_OUTPUT_ROOT = Path("/content/drive/MyDrive/SS_VIRULEX_Reliability")
EXPECTED_CONFIGURATION_ID = "112592618942"
EXPECTED_CONFIG_SHA256 = "ed6a48211eaedb964b0b2fffaded1959165134a2978dd75a913aae427eff43eb"
EXPECTED_DEVELOPMENT_ROWS = 1054
EXPECTED_SPLIT_COUNTS = {"train": 936, "val": 118}
EXPECTED_CONCEPT_COLUMNS = [
    "concept_peripheral_ground_glass_opacities_label",
    "concept_bilateral_involvement_label",
    "concept_multilobar_distribution_label",
    "concept_crazy_paving_pattern_label",
    "concept_absence_of_lobar_consolidation_label",
    "concept_localized_or_diffuse_presentation_label",
    "concept_increased_density_in_the_lung_label",
    "concept_ground_glass_appearance_label",
]
DEVELOPMENT_INPUT_SHA256 = {
    "work/manifests/official_splits.csv": "ac4e1f620cffd8c506cabb4e8669a54765755be1c5bdebe42f19efb7a399cfce",
    "work/manifests/augmented_splits.csv": "47551d3472993f9e987b998e0cc105c1c5f78e01c8499c5ea212055115e8b67b",
    "work/manifests/concept_labels_development.csv": "b8cc912a1cccab0a0a59758fbb24d07ff048aa259100bbe977beef26c7f495c6",
}
LOCKED_CHECKPOINT_PATHS = [
    "04_reliability/configs/encoder_gradual_final_block.yaml",
    "04_reliability/ss_virulex_reliability/audit_leakage.py",
    "04_reliability/ss_virulex_reliability/common.py",
    "04_reliability/ss_virulex_reliability/encoder_experiments.py",
    "04_reliability/ss_virulex_reliability/fold_safe_fusion.py",
    "04_reliability/ss_virulex_reliability/med_data.py",
    "04_reliability/ss_virulex_reliability/oof_med_micn.py",
    "04_reliability/ss_virulex_reliability/trainers.py",
    "results/reliability_colab/final_test/provenance/LOCKED_ENCODER_CONFIG.yaml",
]
REQUIRED_LEAKAGE_PASSES = {
    "official_split_preserved",
    "missing_files",
    "unreadable_images",
    "duplicate_file_paths_cross_split",
    "duplicate_filenames_cross_split",
    "duplicate_image_ids_cross_split",
    "exact_duplicates_cross_split",
    "high_similarity_near_duplicates_cross_split",
    "conflicting_labels",
    "images_assigned_multiple_splits",
    "augmented_samples_train_only",
}
IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "keras": "keras",
    "Pillow": "PIL",
    "ImageHash": "imagehash",
    "PyYAML": "yaml",
    "joblib": "joblib",
    "psutil": "psutil",
    "opencv-python-headless": "cv2",
    "pytest": "pytest",
}
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv-python-headless",
)
INTENDED_OPENCV_DISTRIBUTION = "opencv-python-headless"


class SetupValidationError(RuntimeError):
    """Raised when a mandatory Controlled Colab V2 setup gate fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetupValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_path(repo_root: Path, package_root: Path, output_root: Path, run_id: str) -> Path:
    _require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", run_id)), "run ID must contain only letters, digits, dot, underscore, or dash")
    repo_root = repo_root.resolve()
    package_root = package_root.resolve()
    output_root = output_root.resolve()
    _require(
        not _is_within(output_root, COMPLETED_OUTPUT_ROOT.resolve()),
        "refusing to write setup outputs into the completed reliability run or any descendant",
    )
    _require(not _is_within(output_root, repo_root), "V2 output root must not be inside the Git checkout")
    _require(not _is_within(output_root, package_root), "V2 output root must not be inside the preserved Colab package")
    _require(not _is_within(package_root, output_root), "preserved Colab package must not be inside the V2 output root")
    output_dir = output_root / "00_setup_validation" / run_id
    _require(not output_dir.exists(), f"refusing to overwrite setup output: {output_dir}")
    return output_dir


def _git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def validate_repository(repo_root: Path) -> dict:
    repo_root = repo_root.resolve()
    _require((repo_root / ".git").exists(), f"Git checkout not found: {repo_root}")
    branch = _git(repo_root, "branch", "--show-current").stdout.strip()
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    _require(branch == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}, found {branch or '<detached>'}")
    checkpoint = _git(repo_root, "rev-parse", "--verify", f"{VALIDATED_CHECKPOINT}^{{commit}}").stdout.strip()
    _require(checkpoint == VALIDATED_CHECKPOINT, "validated integration checkpoint is unavailable")
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", VALIDATED_CHECKPOINT, "HEAD", check=False)
    _require(ancestor.returncode == 0, f"validated checkpoint {VALIDATED_CHECKPOINT} is not an ancestor of HEAD {head}")
    status = _git(repo_root, "status", "--porcelain").stdout.strip()
    _require(not status, f"tracked Git changes are present:\n{status}")
    locked_diff = _git(
        repo_root,
        "diff",
        "--exit-code",
        VALIDATED_CHECKPOINT,
        "HEAD",
        "--",
        *LOCKED_CHECKPOINT_PATHS,
        check=False,
    )
    _require(locked_diff.returncode == 0, "validated implementation/config files differ from the integration checkpoint")
    return {"branch": branch, "head": head, "validated_checkpoint": checkpoint, "tracked_status": "clean"}


def parse_exact_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _require("==" in line and line.count("==") == 1, f"non-exact requirement at {path}:{line_number}: {line}")
        name, version = (part.strip() for part in line.split("==", 1))
        _require(bool(name and version), f"invalid requirement at {path}:{line_number}: {line}")
        pins[name] = version
    _require(bool(pins), f"no exact requirements found in {path}")
    return pins


def validate_opencv_namespace(
    expected_distribution_version: str,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    importer: Callable[[str], object] = importlib.import_module,
) -> dict:
    installed: dict[str, str] = {}
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            installed[distribution] = version_getter(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    conflicts = sorted(set(installed) - {INTENDED_OPENCV_DISTRIBUTION})
    _require(
        not conflicts,
        "conflicting OpenCV distributions share the cv2 namespace: "
        + ", ".join(f"{name}=={installed[name]}" for name in conflicts),
    )
    observed_distribution = installed.get(INTENDED_OPENCV_DISTRIBUTION)
    _require(
        observed_distribution == expected_distribution_version,
        f"expected {INTENDED_OPENCV_DISTRIBUTION}=={expected_distribution_version}, found {observed_distribution or 'not installed'}",
    )
    try:
        cv2 = importer("cv2")
    except Exception as exc:
        raise SetupValidationError(f"cv2 import failed: {type(exc).__name__}: {exc}") from exc
    observed_import = str(getattr(cv2, "__version__", ""))
    expected_import = expected_distribution_version.rsplit(".", 1)[0]
    _require(
        observed_import == expected_import,
        f"cv2 import version mismatch: expected {expected_import}, found {observed_import or '<missing __version__>'}",
    )
    return {
        "distribution": INTENDED_OPENCV_DISTRIBUTION,
        "distribution_version": observed_distribution,
        "import_version": observed_import,
        "module_file": str(getattr(cv2, "__file__", "")),
        "conflicting_distributions": [],
    }


def inspect_global_pip_check() -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "returncode": completed.returncode,
        "globally_consistent": completed.returncode == 0,
        "diagnostic_only": True,
        "output": output[:20000],
    }


def validate_dependencies(
    requirements_path: Path,
    setup_tools_path: Path,
    med_source_root: Path,
    require_cuda: bool = True,
) -> dict:
    pins = parse_exact_requirements(requirements_path)
    tool_pins = parse_exact_requirements(setup_tools_path)
    duplicate_pins = set(pins) & set(tool_pins)
    _require(
        all(pins[name] == tool_pins[name] for name in duplicate_pins),
        f"conflicting duplicate dependency pins: {sorted(duplicate_pins)}",
    )
    pins.update(tool_pins)
    resolved: dict[str, str] = {}
    imported: dict[str, dict[str, str]] = {}
    for distribution, expected in pins.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SetupValidationError(f"missing dependency: {distribution}=={expected}") from exc
        _require(observed == expected, f"dependency mismatch for {distribution}: expected {expected}, found {observed}")
        module_name = IMPORT_NAMES.get(distribution)
        if module_name and distribution != INTENDED_OPENCV_DISTRIBUTION:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise SetupValidationError(f"dependency import failed for {distribution} ({module_name}): {type(exc).__name__}: {exc}") from exc
            import_version = str(getattr(module, "__version__", ""))
            _require(
                import_version == expected,
                f"import version mismatch for {module_name}: expected {expected}, found {import_version or '<missing __version__>'}",
            )
            imported[module_name] = {
                "distribution": distribution,
                "version": import_version,
                "module_file": str(getattr(module, "__file__", "")),
            }
        resolved[distribution] = observed

    _require(INTENDED_OPENCV_DISTRIBUTION in pins, "the intended OpenCV distribution is not pinned")
    opencv = validate_opencv_namespace(pins[INTENDED_OPENCV_DISTRIBUTION])
    imported["cv2"] = {
        "distribution": opencv["distribution"],
        "version": opencv["import_version"],
        "module_file": opencv["module_file"],
    }

    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
    except Exception as exc:
        raise SetupValidationError(f"PyTorch runtime import failed: {type(exc).__name__}: {exc}") from exc
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda:
        _require(cuda_available, "CUDA is unavailable; select a Colab GPU runtime before continuing")

    med_source_root = med_source_root.resolve()
    if str(med_source_root) not in sys.path:
        sys.path.insert(0, str(med_source_root))
    try:
        models = importlib.import_module("models")
        torch_explain = importlib.import_module("torch_explain")
        concept_module = importlib.import_module("torch_explain.nn.concepts")
        getattr(models, "Neural_Concat_Model")
        getattr(concept_module, "ConceptReasoningLayer")
    except Exception as exc:
        raise SetupValidationError(f"Med-MICN import failed: {type(exc).__name__}: {exc}") from exc
    _require(_is_within(Path(models.__file__).resolve(), med_source_root), "models imported from outside the authoritative Med-MICN source root")
    _require(_is_within(Path(torch_explain.__file__).resolve(), med_source_root), "torch_explain imported from outside the authoritative Med-MICN source root")

    return {
        "pinned": resolved,
        "imports": imported,
        "opencv_namespace": opencv,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_available": cuda_available,
        "cuda_device": str(torch.cuda.get_device_name(0)) if cuda_available else None,
        "med_source_root": str(med_source_root),
        "global_pip_check": inspect_global_pip_check(),
    }


def validate_required_paths(repo_root: Path, package_root: Path) -> dict:
    required_files = [
        package_root / "requirements_colab.txt",
        package_root / "scripts/runtime_info.py",
        *(package_root / relative for relative in DEVELOPMENT_INPUT_SHA256),
        repo_root / "04_reliability/configs/v2/setup_tools.txt",
        repo_root / "04_reliability/configs/encoder_gradual_final_block.yaml",
        repo_root / "04_reliability/configs/v2/encoder_locked_baseline.yaml",
        repo_root / "results/reliability_colab/final_test/provenance/LOCKED_ENCODER_CONFIG.yaml",
        repo_root / "01_Med-MICN/models.py",
        repo_root / "01_Med-MICN/torch_explain/__init__.py",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    _require(not missing_files, "required files are missing:\n" + "\n".join(missing_files))
    dataset_root = package_root / "assets/COVID-CT-Dataset-master/Images-processed"
    required_directories = [dataset_root / "CT_COVID", dataset_root / "CT_NonCOVID"]
    missing_directories = [str(path) for path in required_directories if not path.is_dir()]
    _require(not missing_directories, "required development image directories are missing:\n" + "\n".join(missing_directories))
    return {
        "required_file_count": len(required_files),
        "development_image_directories": [str(path) for path in required_directories],
    }


def validate_recorded_checksums(package_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in DEVELOPMENT_INPUT_SHA256.items():
        path = package_root / relative
        digest = sha256_file(path)
        _require(digest == expected, f"checksum mismatch for {path}: expected {expected}, found {digest}")
        observed[relative] = digest
    return observed


def _split_values(frame, label: str) -> set[str]:
    _require("split" in frame.columns, f"{label} has no split column")
    values = set(frame["split"].astype(str).str.strip().str.lower())
    _require(values <= {"train", "val"}, f"{label} contains forbidden split values: {sorted(values - {'train', 'val'})}")
    _require("train" in values and "val" in values, f"{label} must contain both train and val rows")
    return values


def validate_manifest_contract(
    manifest_path: Path,
    official_manifest_path: Path,
    concept_path: Path,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
    expected_split_counts: dict[str, int] | None = EXPECTED_SPLIT_COUNTS,
) -> dict:
    import pandas as pd

    from .med_data import prepare_med_frame

    raw_manifest = pd.read_csv(manifest_path)
    raw_official = pd.read_csv(official_manifest_path)
    raw_concepts = pd.read_csv(concept_path)
    split_sets = {
        "augmented_manifest": sorted(_split_values(raw_manifest, "augmented development manifest")),
        "official_manifest": sorted(_split_values(raw_official, "official development manifest")),
        "concept_labels": sorted(_split_values(raw_concepts, "development concept labels")),
    }
    concept_columns = [
        column for column in raw_concepts.columns if column.startswith("concept_") and column.endswith("_label")
    ]
    _require(len(concept_columns) == 8, f"expected exactly 8 concept columns, found {len(concept_columns)}")
    _require(concept_columns == EXPECTED_CONCEPT_COLUMNS, f"concept columns/order mismatch: {concept_columns}")

    frame, prepared_concepts, grouping_mode = prepare_med_frame(manifest_path, concept_path)
    _require(len(frame) == expected_rows, f"expected {expected_rows} development rows, found {len(frame)}")
    _require(prepared_concepts == EXPECTED_CONCEPT_COLUMNS, "prepared concept columns/order mismatch")
    prepared_splits = set(frame["original_split"].astype(str).str.lower())
    _require(prepared_splits == {"train", "val"}, f"prepared frame contains forbidden splits: {sorted(prepared_splits)}")
    counts = frame["original_split"].value_counts().sort_index().astype(int).to_dict()
    if expected_split_counts is not None:
        _require(counts == expected_split_counts, f"development split counts mismatch: expected {expected_split_counts}, found {counts}")
    train_groups = set(frame.loc[frame["original_split"] == "train", "group_id"].astype(str))
    val_groups = set(frame.loc[frame["original_split"] == "val", "group_id"].astype(str))
    overlap = sorted(train_groups & val_groups)
    _require(not overlap, f"train/validation group overlap detected; first={overlap[0]}")
    augmented_outside_train = frame[(frame["source"] == "augmented") & (frame["original_split"] != "train")]
    _require(augmented_outside_train.empty, "augmented rows occur outside the training split")
    return {
        "rows": int(len(frame)),
        "split_counts": counts,
        "split_values": split_sets,
        "concept_columns": prepared_concepts,
        "concept_count": len(prepared_concepts),
        "grouping_mode": grouping_mode,
        "train_group_count": len(train_groups),
        "validation_group_count": len(val_groups),
        "group_overlap_count": 0,
        "official_test_rows_accessed": 0,
    }


def validate_locked_config(repo_root: Path) -> dict:
    import yaml

    from .trainers import configuration_id

    paths = [
        repo_root / "04_reliability/configs/encoder_gradual_final_block.yaml",
        repo_root / "04_reliability/configs/v2/encoder_locked_baseline.yaml",
        repo_root / "results/reliability_colab/final_test/provenance/LOCKED_ENCODER_CONFIG.yaml",
    ]
    contents = [path.read_bytes() for path in paths]
    _require(contents[0] == contents[1] == contents[2], "active, V2 baseline, and preserved locked encoder configs are not byte-identical")
    digest = hashlib.sha256(contents[0]).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, f"locked encoder checksum mismatch: {digest}")
    config = yaml.safe_load(contents[0])
    _require(isinstance(config, dict), "locked encoder YAML is not a mapping")
    config_id = configuration_id(config)
    _require(config_id == EXPECTED_CONFIGURATION_ID, f"expected configuration ID {EXPECTED_CONFIGURATION_ID}, found {config_id}")
    return {"paths": [str(path.resolve()) for path in paths], "sha256": digest, "configuration_id": config_id}


def run_leakage_gate(
    manifest_path: Path,
    official_manifest_path: Path,
    concept_path: Path,
    output_dir: Path,
) -> dict:
    from .audit_leakage import run_audit

    audit_root = output_dir / "leakage_audit"
    summary = run_audit(
        Namespace(
            manifest=manifest_path,
            official_manifest=official_manifest_path,
            metadata_csv=concept_path,
            provenance_json=None,
            output_dir=audit_root / "artifacts",
            markdown=audit_root / "report.md",
            near_threshold=6,
            near_correlation=0.90,
            near_mae=0.08,
            max_near_pairs=10000,
            skip_perceptual=False,
            overwrite=False,
        )
    )
    checks = {item["check"]: item["status"] for item in summary["checks"]}
    missing = sorted(REQUIRED_LEAKAGE_PASSES - set(checks))
    failed = sorted(name for name in REQUIRED_LEAKAGE_PASSES if checks.get(name) != "PASS")
    _require(not missing, f"leakage audit omitted required checks: {missing}")
    _require(not failed, f"leakage audit gate failed: {failed}")
    _require(not any(status == "FAIL" for status in checks.values()), "leakage audit contains one or more FAIL results")
    _require(summary["row_count"] == EXPECTED_DEVELOPMENT_ROWS, "leakage audit row count mismatch")
    return summary


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_setup_validation(
    repo_root: Path,
    package_root: Path,
    output_root: Path,
    run_id: str,
    dependency_validator: Callable[[Path, Path, Path, bool], dict] = validate_dependencies,
) -> dict:
    repo_root = repo_root.resolve()
    package_root = package_root.resolve()
    output_dir = validate_output_path(repo_root, package_root, output_root, run_id)
    report: dict = {
        "status": "RUNNING",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "official_test_rows_accessed": 0,
        "repo_root": str(repo_root),
        "package_root": str(package_root),
        "output_dir": str(output_dir),
    }
    try:
        report["repository"] = validate_repository(repo_root)
        report["required_paths"] = validate_required_paths(repo_root, package_root)
        checksums_before = validate_recorded_checksums(package_root)
        report["dependencies"] = dependency_validator(
            package_root / "requirements_colab.txt",
            repo_root / "04_reliability/configs/v2/setup_tools.txt",
            repo_root / "01_Med-MICN",
            True,
        )
        report["development_contract"] = validate_manifest_contract(
            package_root / "work/manifests/augmented_splits.csv",
            package_root / "work/manifests/official_splits.csv",
            package_root / "work/manifests/concept_labels_development.csv",
        )
        report["locked_encoder"] = validate_locked_config(repo_root)
        output_dir.mkdir(parents=True, exist_ok=False)
        report["leakage_audit"] = run_leakage_gate(
            package_root / "work/manifests/augmented_splits.csv",
            package_root / "work/manifests/official_splits.csv",
            package_root / "work/manifests/concept_labels_development.csv",
            output_dir,
        )
        checksums_after = validate_recorded_checksums(package_root)
        _require(checksums_after == checksums_before, "preserved development inputs changed during validation")
        report["development_input_sha256"] = checksums_after
        report["status"] = "PASS"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "SETUP_VALIDATION_COMPLETE.json", report)
        return report
    except Exception as exc:
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        if output_dir.exists():
            _write_json(output_dir / "SETUP_VALIDATION_FAILED.json", report)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-stop validation for the development-only Controlled Colab V2 setup.")
    parser.add_argument("--repo-root", type=Path, default=AUTHORITATIVE_REPO_ROOT)
    parser.add_argument("--package-root", type=Path, default=AUTHORITATIVE_PACKAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=AUTHORITATIVE_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_setup_validation(args.repo_root, args.package_root, args.output_root, args.run_id)
    except Exception as exc:
        print(f"CONTROLLED COLAB V2 SETUP: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("CONTROLLED COLAB V2 SETUP: PASS")
    print(f"Configuration ID: {report['locked_encoder']['configuration_id']}")
    print(f"Development rows: {report['development_contract']['rows']}")
    print(f"Output: {report['output_dir']}")
    print("STOP: setup validation is complete; no training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
