from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .utils import (
    atomic_write_json,
    canonical_json_bytes,
    find_project_root,
    sha256_bytes,
    sha256_file,
)
from .validation import ValidationReport, validate_submission

FROZEN_FILES = ("library.fasta", "top.fasta", "scores.csv", "manifest.json")


def _covered_hashes(run_dir: Path) -> dict[str, str]:
    missing = [name for name in FROZEN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run directory is missing required files: {missing}")
    return {name: sha256_file(run_dir / name) for name in FROZEN_FILES}


def verify_manifest_hashes(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("outputs", {}).items():
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest output missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest hash mismatch for {name}: {actual} != {expected}")


def verify_manifest_inputs(run_dir: Path) -> list[dict[str, str]]:
    """Verify every release dependency pinned by the run manifest."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{manifest_path}: missing non-empty inputs list")

    verified: list[dict[str, str]] = []
    project_root = find_project_root()
    if not (project_root / "pyproject.toml").is_file():
        project_root = find_project_root(run_dir)
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"{manifest_path}: inputs[{index}] must be an object")
        role = item.get("role")
        path_value = item.get("path")
        expected = item.get("sha256")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{manifest_path}: inputs[{index}].role must be non-empty")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"{manifest_path}: inputs[{index}].path must be non-empty")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"{manifest_path}: inputs[{index}].sha256 is invalid")
        path = Path(path_value)
        resolved_path = path if path.is_absolute() else project_root / path
        if not resolved_path.is_file():
            raise FileNotFoundError(f"manifest input missing: {resolved_path}")
        actual = sha256_file(resolved_path)
        if actual != expected:
            raise ValueError(
                f"manifest input hash mismatch for {resolved_path}: {actual} != {expected}"
            )
        verified.append({"role": role, "path": path_value, "sha256": actual})
    return verified


def freeze_run(
    *,
    run_dir: Path,
    reference_path: Path,
    submission_dir: Path,
    expected_library_size: int,
    expected_top_size: int,
) -> tuple[dict, Path, ValidationReport]:
    report = validate_submission(
        library_path=run_dir / "library.fasta",
        top_path=run_dir / "top.fasta",
        reference_path=reference_path,
        expected_library_size=expected_library_size,
        expected_top_size=expected_top_size,
    )
    report.raise_for_errors()
    verify_manifest_hashes(run_dir)
    inputs = verify_manifest_inputs(run_dir)
    covered = _covered_hashes(run_dir)
    identity_payload = {"schema_version": 1, "files": covered, "inputs": inputs}
    run_id = sha256_bytes(canonical_json_bytes(identity_payload))[:16]
    freeze_path = run_dir / "freeze.json"

    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            existing.get("run_id") != run_id
            or existing.get("files") != covered
            or existing.get("inputs") != inputs
        ):
            raise ValueError(
                f"{freeze_path} describes a different run; generate into a new directory"
            )
        freeze_record = existing
    else:
        freeze_record = {
            **identity_payload,
            "run_id": run_id,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "validation": report.as_dict(),
        }
        atomic_write_json(freeze_path, freeze_record)

    submission_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = submission_dir / f"{run_id}.zip"
    temporary_path = submission_dir / f".{run_id}.zip.tmp"
    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # freeze.json intentionally stays outside the archive: it records the archive
        # digest, so including it would create a recursive and unstable hash.
        for name in FROZEN_FILES:
            data = (run_dir / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    temporary_path.replace(artifact_path)
    freeze_record["artifact"] = {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
    }
    atomic_write_json(freeze_path, freeze_record)
    return freeze_record, artifact_path, report


def verify_frozen_run(run_dir: Path, expected_run_id: str) -> dict:
    freeze_path = run_dir / "freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"run is not frozen: {freeze_path} is missing")
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if record.get("run_id") != expected_run_id:
        raise ValueError(
            f"run ID mismatch: provided {expected_run_id}, frozen {record.get('run_id')}"
        )
    actual = _covered_hashes(run_dir)
    if actual != record.get("files"):
        raise ValueError("frozen run has changed; covered file hashes no longer match")
    verify_manifest_hashes(run_dir)
    inputs = verify_manifest_inputs(run_dir)
    if inputs != record.get("inputs"):
        raise ValueError("frozen run inputs have changed; dependency hashes no longer match")
    return record
