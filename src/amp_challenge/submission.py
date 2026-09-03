from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .freeze import verify_frozen_run
from .utils import sha256_file


def build_kaggle_command(*, competition: str, artifact: Path, message: str) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        str(artifact),
        "-m",
        message,
    ]


def submit_kaggle(
    *,
    run_dir: Path,
    artifact: Path,
    run_id: str,
    message: str,
    competition: str = "amp-challenge",
    execute: bool = False,
) -> str:
    record = verify_frozen_run(run_dir, run_id)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    expected_artifact = record.get("artifact", {})
    expected_digest = expected_artifact.get("sha256")
    actual_digest = sha256_file(artifact)
    if expected_digest and actual_digest != expected_digest:
        raise ValueError("artifact hash does not match the frozen run")

    command = build_kaggle_command(
        competition=competition,
        artifact=artifact,
        message=f"[{run_id}] {message}",
    )
    rendered = shlex.join(command)
    if not execute:
        return f"DRY RUN: {rendered}"
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI is not installed or not on PATH")
    subprocess.run(command, check=True)
    return f"SUBMITTED: {rendered}"
