from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path


def _require_cli() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("Kaggle CLI is not installed; install it with 'uv tool install kaggle'")
    return executable


def build_download_command(*, competition: str, destination: Path) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(destination),
    ]


def run_read_command(
    *,
    action: str,
    competition: str = "amp-challenge",
    destination: Path | None = None,
) -> str:
    executable = _require_cli()
    if action == "files":
        command = [executable, "competitions", "files", "-c", competition]
    elif action == "submissions":
        command = [executable, "competitions", "submissions", "-c", competition]
    elif action == "download":
        if destination is None:
            raise ValueError("destination is required for download")
        destination.mkdir(parents=True, exist_ok=True)
        command = build_download_command(competition=competition, destination=destination)
        command[0] = executable
    else:
        raise ValueError(f"unsupported Kaggle action {action!r}")
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return f"$ {shlex.join(command)}\n{result.stdout}".rstrip()
