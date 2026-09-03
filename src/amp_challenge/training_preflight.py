from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .splits import audit_cross_fold_similarity
from .training_data import PARSER_COMPONENTS, PARSER_VERSION
from .utils import atomic_write_json, sha256_file

REQUIRED_MIC_COLUMNS = {
    "measurement_id",
    "sequence_id",
    "sequence",
    "n_terminal",
    "c_terminal",
    "organism_name",
    "relation",
    "lower_um",
    "upper_um",
    "lower_log2_um",
    "upper_log2_um",
    "cluster_id",
    "fold",
    "split_role",
}
REQUIRED_HC50_COLUMNS = {
    "measurement_id",
    "sequence_id",
    "sequence",
    "n_terminal",
    "c_terminal",
    "relation",
    "lower_um",
    "upper_um",
    "lower_log2_um",
    "upper_log2_um",
    "cluster_id",
    "fold",
    "split_role",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _check_measurements(
    rows: list[dict[str, str]],
    *,
    task: str,
    split_lookup: dict[str, tuple[str, str, str]],
    errors: list[str],
) -> dict[str, Any]:
    identifiers: set[str] = set()
    folds: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    free_termini = 0
    for index, row in enumerate(rows, start=2):
        identifier = row["measurement_id"]
        if identifier in identifiers:
            errors.append(f"{task}: duplicate measurement_id {identifier}")
        identifiers.add(identifier)
        sequence_id = row["sequence_id"]
        expected_sequence_id = hashlib.sha256(row["sequence"].encode("ascii")).hexdigest()
        if sequence_id != expected_sequence_id:
            errors.append(f"{task}:{index}: sequence_id does not match sequence SHA-256")
        expected = split_lookup.get(sequence_id)
        observed = (row["cluster_id"], row["fold"], row["split_role"])
        if expected is None:
            errors.append(f"{task}:{index}: sequence missing from split table")
        elif expected != observed:
            errors.append(f"{task}:{index}: split fields disagree with sequence_splits.csv")
        lower = float(row["lower_um"]) if row["lower_um"] else None
        upper = float(row["upper_um"]) if row["upper_um"] else None
        if lower is None and upper is None:
            errors.append(f"{task}:{index}: interval has no finite bound")
        if lower is not None and lower <= 0:
            errors.append(f"{task}:{index}: non-positive lower bound")
        if upper is not None and upper <= 0:
            errors.append(f"{task}:{index}: non-positive upper bound")
        if lower is not None and upper is not None and lower > upper:
            errors.append(f"{task}:{index}: descending interval")
        relation = row["relation"]
        expected_bounds = {
            "eq": (True, True),
            "interval": (True, True),
            "lt": (False, True),
            "le": (False, True),
            "gt": (True, False),
            "ge": (True, False),
        }
        if relation not in expected_bounds:
            errors.append(f"{task}:{index}: unsupported relation {relation!r}")
        else:
            needs_lower, needs_upper = expected_bounds[relation]
            if (lower is not None) != needs_lower or (upper is not None) != needs_upper:
                errors.append(f"{task}:{index}: bounds disagree with relation {relation}")
            if relation == "eq" and lower != upper:
                errors.append(f"{task}:{index}: point measurement bounds disagree")
        for bound_name, bound in (("lower", lower), ("upper", upper)):
            log_value = row[f"{bound_name}_log2_um"]
            if bound is None:
                if log_value:
                    errors.append(f"{task}:{index}: log2 value exists without {bound_name} bound")
            elif not log_value or not math.isclose(
                float(log_value),
                math.log2(bound),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append(f"{task}:{index}: {bound_name} log2 value is inconsistent")
        folds[row["fold"]] += 1
        roles[row["split_role"]] += 1
        if row.get("n_terminal") == "free" and row.get("c_terminal") == "free":
            free_termini += 1
    return {
        "rows": len(rows),
        "unique_measurements": len(identifiers),
        "fold_counts": dict(sorted(folds.items())),
        "role_counts": dict(sorted(roles.items())),
        "free_termini_rows": free_termini,
    }


def training_preflight(
    *,
    dataset_dir: Path,
    train_config_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if train_config.get("schema_version") != 1:
        errors.append("unsupported training config schema")
    if train_config.get("dataset_id") != manifest.get("dataset_id"):
        errors.append("training config dataset_id does not match manifest")
    parser = manifest.get("parser", {})
    if parser.get("version") != PARSER_VERSION:
        errors.append("dataset parser version does not match the installed implementation")
    recorded_parser_hashes = parser.get("implementation_sha256", {})
    for name in PARSER_COMPONENTS:
        current_digest = sha256_file(Path(__file__).with_name(name))
        if recorded_parser_hashes.get(name) != current_digest:
            errors.append(f"dataset parser implementation mismatch: {name}")
    revision = train_config.get("backbone", {}).get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("backbone revision must be an immutable 40-character Git commit")
    if train_config.get("backbone", {}).get("frozen") is not True:
        warnings.append("v1 assumes a frozen backbone; config requests trainable weights")

    for filename, metadata in manifest.get("artifacts", {}).items():
        path = dataset_dir / filename
        if not path.is_file():
            errors.append(f"missing artifact {filename}")
        elif sha256_file(path) != metadata["sha256"]:
            errors.append(f"artifact checksum mismatch: {filename}")

    split_fields, split_rows = _read_csv(dataset_dir / "sequence_splits.csv")
    required_split = {"sequence_id", "sequence", "cluster_id", "fold", "split_role"}
    if not required_split.issubset(split_fields):
        errors.append("sequence_splits.csv is missing required columns")
    split_lookup: dict[str, tuple[str, str, str]] = {}
    sequence_lookup: dict[str, str] = {}
    cluster_folds: dict[str, str] = {}
    for row in split_rows:
        sequence_id = row["sequence_id"]
        if sequence_id in split_lookup:
            errors.append(f"duplicate sequence_id in split table: {sequence_id}")
        split_lookup[sequence_id] = (row["cluster_id"], row["fold"], row["split_role"])
        sequence = row["sequence"]
        if sequence in sequence_lookup and sequence_lookup[sequence] != sequence_id:
            errors.append(f"sequence maps to multiple IDs: {sequence}")
        sequence_lookup[sequence] = sequence_id
        prior_fold = cluster_folds.setdefault(row["cluster_id"], row["fold"])
        if prior_fold != row["fold"]:
            errors.append(f"cluster crosses folds: {row['cluster_id']}")

    mic_fields, mic_rows = _read_csv(dataset_dir / "mic_measurements.csv")
    hc50_fields, hc50_rows = _read_csv(dataset_dir / "hc50_measurements.csv")
    if not REQUIRED_MIC_COLUMNS.issubset(mic_fields):
        errors.append("mic_measurements.csv is missing required columns")
    if not REQUIRED_HC50_COLUMNS.issubset(hc50_fields):
        errors.append("hc50_measurements.csv is missing required columns")
    task_reports = {
        "mic": _check_measurements(
            mic_rows,
            task="mic",
            split_lookup=split_lookup,
            errors=errors,
        ),
        "hc50": _check_measurements(
            hc50_rows,
            task="hc50",
            split_lookup=split_lookup,
            errors=errors,
        ),
    }
    required_folds = int(train_config["evaluation"]["folds"])
    observed_folds = {int(row["fold"]) for row in split_rows}
    if observed_folds != set(range(required_folds)):
        errors.append(f"expected folds 0..{required_folds - 1}, observed {sorted(observed_folds)}")
    required_roles = {"train", "calibration", "test"}
    observed_roles = {row["split_role"] for row in split_rows}
    if observed_roles != required_roles:
        errors.append(
            f"expected split roles {sorted(required_roles)}, observed {sorted(observed_roles)}"
        )
    minimums = train_config["minimum_rows"]
    if len(mic_rows) < int(minimums["mic"]):
        errors.append(f"MIC rows below configured minimum {minimums['mic']}")
    if len(hc50_rows) < int(minimums["hc50"]):
        errors.append(f"HC50 rows below configured minimum {minimums['hc50']}")
    clustering = manifest.get("clustering", {})
    if clustering.get("method") != "global_edit_single_linkage":
        errors.append("v1 preflight requires global_edit_single_linkage clustering")
    similarity_audit = audit_cross_fold_similarity(
        {row["sequence"]: int(row["fold"]) for row in split_rows},
        identity_threshold=float(clustering["identity_threshold"]),
        min_coverage=float(clustering["min_coverage"]),
    )
    if similarity_audit["violating_pairs"]:
        errors.append("cross-fold sequence pairs meet or exceed the configured identity threshold")
    warnings.append("The v1 split uses global edit identity, not MMseqs2 local alignment.")

    report = {
        "ready": not errors,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_config_sha256": sha256_file(train_config_path),
        "backbone": train_config.get("backbone"),
        "sequences": len(split_rows),
        "clusters": len(cluster_folds),
        "split_similarity_audit": similarity_audit,
        "tasks": task_reports,
        "errors": errors,
        "warnings": warnings,
        "training_executed": False,
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report
