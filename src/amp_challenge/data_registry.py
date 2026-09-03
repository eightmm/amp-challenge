from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

from .utils import sha256_file


def load_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported data registry schema")
    return registry


def source_status(*, project_root: Path, registry_path: Path) -> list[dict]:
    registry = load_registry(registry_path)
    results: list[dict] = []
    for name, source in registry["sources"].items():
        destination_value = source.get("destination")
        destination = project_root / destination_value if destination_value else None
        if destination is None:
            status = "manual-review"
            digest = None
        elif not destination.is_file():
            status = "missing"
            digest = None
        else:
            digest = sha256_file(destination)
            status = "ready" if digest == source.get("sha256") else "checksum-mismatch"
        results.append(
            {
                "name": name,
                "access": source.get("access"),
                "status": status,
                "destination": destination_value,
                "sha256": digest,
            }
        )
    return results


def fetch_source(
    *,
    name: str,
    project_root: Path,
    registry_path: Path,
    force: bool = False,
) -> Path:
    registry = load_registry(registry_path)
    try:
        source = registry["sources"][name]
    except KeyError as error:
        raise KeyError(f"unknown data source {name!r}") from error
    if source.get("access") != "automatic":
        raise PermissionError(
            f"{name}: downloader disabled until release/access/license review is recorded"
        )
    destination = project_root / source["destination"]
    expected_digest = source["sha256"]
    if destination.exists() and not force:
        actual = sha256_file(destination)
        if actual == expected_digest:
            return destination
        raise FileExistsError(
            f"{destination} exists with SHA-256 {actual}; pass --force only after review"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            urllib.request.urlopen(source["url"], timeout=60) as response,
            open(file_descriptor, "wb", closefd=True) as handle,
        ):
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        actual = sha256_file(temporary)
        if actual != expected_digest:
            raise ValueError(
                f"{name}: downloaded SHA-256 {actual} does not match {expected_digest}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
