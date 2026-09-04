from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .physchem import PhyschemFeatures, describe
from .utils import atomic_write_json, canonical_json_bytes, sha256_file

SCHEMA_VERSION = 1
MODEL_TYPE = "physchem-logistic-ensemble"
MODEL_NAME = "linear-physchem-v1"

TEST_FOLD = 0
CALIBRATION_FOLD = 1
SELECTION_TRAIN_FOLDS = (2, 3)
SELECTION_FOLD = 4
REFIT_FOLDS = (2, 3, 4)

FEATURE_NAMES = tuple(PhyschemFeatures.__dataclass_fields__)
DEFAULT_L2_CANDIDATES = (0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True, slots=True)
class LinearOraclePrediction:
    activity: float
    toxicity: float
    uncertainty: float

    def __post_init__(self) -> None:
        for field_name in ("activity", "toxicity", "uncertainty"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class _Example:
    sequence: str
    fold: int
    label: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Scaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} features, received {len(values)}")
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(values, self.mean, self.scale, strict=True)
        )


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite probability {value}")
    return max(0.0, min(1.0, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _logit(value: float) -> float:
    clipped = max(1e-9, min(1.0 - 1e-9, value))
    return math.log(clipped / (1.0 - clipped))


def _feature_vector(sequence: str) -> tuple[float, ...]:
    values = describe(sequence).as_dict()
    return tuple(float(values[name]) for name in FEATURE_NAMES)


def _fit_scaler(examples: Sequence[_Example]) -> _Scaler:
    if not examples:
        raise ValueError("cannot fit a feature scaler without examples")
    width = len(FEATURE_NAMES)
    mean = tuple(
        sum(example.features[index] for example in examples) / len(examples)
        for index in range(width)
    )
    scale_values: list[float] = []
    for index in range(width):
        variance = sum((example.features[index] - mean[index]) ** 2 for example in examples)
        standard_deviation = math.sqrt(variance / len(examples))
        scale_values.append(standard_deviation if standard_deviation > 1e-12 else 1.0)
    return _Scaler(mean, tuple(scale_values))


def _linear_solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("singular logistic-regression Hessian")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0.0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= multiplier * augmented[column][item]
    return [augmented[row][size] for row in range(size)]


def _logistic_objective(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float],
    l2: float,
) -> float:
    loss = 0.0
    for row, label in zip(rows, labels, strict=True):
        linear = sum(value * weight for value, weight in zip(row, weights, strict=True))
        loss += max(linear, 0.0) - label * linear + math.log1p(math.exp(-abs(linear)))
    loss /= len(rows)
    loss += 0.5 * l2 * sum(weight * weight for weight in weights[1:])
    return loss


def _fit_logistic(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float,
    max_iterations: int = 100,
) -> tuple[float, tuple[float, ...]]:
    if len(features) != len(labels) or not features:
        raise ValueError("features and labels must have the same non-zero length")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    classes = set(labels)
    if classes != {0, 1}:
        raise ValueError(f"binary logistic regression requires both classes, observed {classes}")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise ValueError("inconsistent feature widths")
    rows = [[1.0, *(float(value) for value in row)] for row in features]
    positive = sum(labels)
    negative = len(labels) - positive
    weights = [math.log((positive + 0.5) / (negative + 0.5)), *([0.0] * width)]
    current = _logistic_objective(rows, labels, weights, l2)
    for _ in range(max_iterations):
        dimension = width + 1
        gradient = [0.0] * dimension
        hessian = [[0.0] * dimension for _ in range(dimension)]
        for row, label in zip(rows, labels, strict=True):
            linear = sum(value * weight for value, weight in zip(row, weights, strict=True))
            probability = _sigmoid(linear)
            residual = probability - label
            curvature = max(probability * (1.0 - probability), 1e-9)
            for left in range(dimension):
                gradient[left] += residual * row[left] / len(rows)
                for right in range(left + 1):
                    contribution = curvature * row[left] * row[right] / len(rows)
                    hessian[left][right] += contribution
                    if left != right:
                        hessian[right][left] += contribution
        hessian[0][0] += 1e-9
        for index in range(1, dimension):
            gradient[index] += l2 * weights[index]
            hessian[index][index] += l2 + 1e-9
        step = _linear_solve(hessian, gradient)
        step_scale = 1.0
        accepted = False
        while step_scale >= 1e-8:
            candidate = [
                weight - step_scale * change for weight, change in zip(weights, step, strict=True)
            ]
            objective = _logistic_objective(rows, labels, candidate, l2)
            if objective <= current + 1e-12:
                weights = candidate
                accepted = True
                improvement = current - objective
                current = objective
                break
            step_scale *= 0.5
        if not accepted or improvement < 1e-10:
            break
    if not all(math.isfinite(value) for value in weights):
        raise ValueError("logistic-regression optimization produced non-finite weights")
    return weights[0], tuple(weights[1:])


def _predict_logit(
    features: Sequence[float], intercept: float, coefficients: Sequence[float]
) -> float:
    return intercept + sum(
        value * coefficient for value, coefficient in zip(features, coefficients, strict=True)
    )


def _metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("metrics require equally sized non-empty labels and probabilities")
    positives = sum(labels)
    negatives = len(labels) - positives
    clipped = [max(1e-12, min(1.0 - 1e-12, value)) for value in probabilities]
    log_loss = -sum(
        label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
        for label, probability in zip(labels, clipped, strict=True)
    ) / len(labels)
    brier = sum(
        (probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)

    ranked = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    while rank <= len(ranked):
        end = rank
        while end < len(ranked) and ranked[end][0] == ranked[rank - 1][0]:
            end += 1
        average_rank = (rank + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[rank - 1 : end])
        rank = end + 1
    auroc = None
    if positives and negatives:
        auroc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    descending = sorted(zip(probabilities, labels, strict=True), reverse=True)
    true_positives = 0
    seen = 0
    average_precision = 0.0
    start = 0
    while start < len(descending):
        end = start + 1
        while end < len(descending) and descending[end][0] == descending[start][0]:
            end += 1
        group_positives = sum(label for _, label in descending[start:end])
        true_positives += group_positives
        seen = end
        if positives:
            average_precision += (group_positives / positives) * (true_positives / seen)
        start = end

    top_count = max(1, math.ceil(len(labels) * 0.1))
    top_precision = sum(label for _, label in descending[:top_count]) / top_count
    prevalence = positives / len(labels)
    enrichment = top_precision / prevalence if prevalence else None
    return {
        "rows": len(labels),
        "positives": positives,
        "negatives": negatives,
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": average_precision if positives else None,
        "brier": brier,
        "log_loss": log_loss,
        "top_10_percent_count": top_count,
        "top_10_percent_precision": top_precision,
        "top_10_percent_enrichment": enrichment,
    }


def _aggregate_examples(
    path: Path, *, label_column: str, task: str
) -> tuple[list[_Example], dict[str, Any]]:
    required = {"sequence", "fold", "split_role", label_column}
    grouped: dict[str, dict[str, Any]] = {}
    definitive_rows = 0
    ignored_rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_label = row[label_column].strip()
            if raw_label == "":
                ignored_rows += 1
                continue
            if raw_label not in {"0", "1"}:
                raise ValueError(f"{path}:{row_number}: {label_column} must be blank, 0, or 1")
            sequence = row["sequence"].strip().upper()
            features = _feature_vector(sequence)
            try:
                fold = int(row["fold"])
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: invalid fold {row['fold']!r}") from error
            expected_role = (
                "test"
                if fold == TEST_FOLD
                else "calibration"
                if fold == CALIBRATION_FOLD
                else "train"
                if fold in REFIT_FOLDS
                else None
            )
            if expected_role is None:
                raise ValueError(f"{path}:{row_number}: unexpected fold {fold}")
            if row["split_role"].strip() != expected_role:
                raise ValueError(
                    f"{path}:{row_number}: fold {fold} must have role {expected_role!r}"
                )
            label = int(raw_label)
            definitive_rows += 1
            prior = grouped.setdefault(
                sequence,
                {"fold": fold, "labels": [], "features": features, "rows": 0},
            )
            if prior["fold"] != fold:
                raise ValueError(f"{path}:{row_number}: sequence {sequence} crosses folds")
            prior["labels"].append(label)
            prior["rows"] += 1

    examples: list[_Example] = []
    conflicts = 0
    folded: dict[int, dict[str, int]] = {}
    for sequence, record in sorted(grouped.items()):
        labels = record["labels"]
        if len(set(labels)) > 1:
            conflicts += 1
        # Conservative aggregation: activity must hold across all available assays,
        # and safety must hold across all available HC50 observations.
        label = min(labels)
        example = _Example(sequence, record["fold"], label, record["features"])
        examples.append(example)
        counts = folded.setdefault(example.fold, {"sequences": 0, "positive": 0, "negative": 0})
        counts["sequences"] += 1
        counts["positive" if label else "negative"] += 1
    for required_fold in (TEST_FOLD, CALIBRATION_FOLD, *REFIT_FOLDS):
        fold_examples = [example for example in examples if example.fold == required_fold]
        if {example.label for example in fold_examples} != {0, 1}:
            raise ValueError(
                f"{task}: fold {required_fold} must contain both definitive threshold classes"
            )
    report = {
        "task": task,
        "label_column": label_column,
        "source_rows_with_definitive_label": definitive_rows,
        "source_rows_without_definitive_label": ignored_rows,
        "aggregated_sequences": len(examples),
        "conflicting_sequences_resolved_negative": conflicts,
        "aggregation": "minimum_definitive_threshold_label_per_sequence",
        "folds": {str(fold): counts for fold, counts in sorted(folded.items())},
    }
    return examples, report


def _select_examples(examples: Sequence[_Example], folds: Iterable[int]) -> list[_Example]:
    selected = set(folds)
    return [example for example in examples if example.fold in selected]


def _fit_candidate(
    train: Sequence[_Example], validation: Sequence[_Example], l2: float
) -> tuple[dict[str, Any], _Scaler]:
    scaler = _fit_scaler(train)
    transformed = [scaler.transform(example.features) for example in train]
    intercept, coefficients = _fit_logistic(
        transformed, [example.label for example in train], l2=l2
    )
    probabilities = [
        _sigmoid(_predict_logit(scaler.transform(example.features), intercept, coefficients))
        for example in validation
    ]
    return _metrics([example.label for example in validation], probabilities), scaler


def _bootstrap_examples(examples: Sequence[_Example], *, seed: int, member: int) -> list[_Example]:
    if member == 0:
        return list(examples)
    randomizer = random.Random(seed + 104729 * member)
    negative = [example for example in examples if example.label == 0]
    positive = [example for example in examples if example.label == 1]
    sampled = [randomizer.choice(negative) for _ in negative]
    sampled.extend(randomizer.choice(positive) for _ in positive)
    randomizer.shuffle(sampled)
    return sampled


def _raw_ensemble_probabilities(
    examples: Sequence[_Example],
    *,
    scaler: _Scaler,
    members: Sequence[dict[str, Any]],
) -> list[float]:
    probabilities: list[float] = []
    for example in examples:
        transformed = scaler.transform(example.features)
        member_probabilities = [
            _sigmoid(
                _predict_logit(
                    transformed,
                    float(member["intercept"]),
                    [float(value) for value in member["coefficients"]],
                )
            )
            for member in members
        ]
        probabilities.append(statistics.fmean(member_probabilities))
    return probabilities


def _fit_calibration(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[float, float]:
    inputs = [(_logit(probability),) for probability in probabilities]
    intercept, coefficients = _fit_logistic(inputs, labels, l2=0.001)
    return coefficients[0], intercept


def _calibrate(probability: float, *, slope: float, intercept: float) -> float:
    return _sigmoid(intercept + slope * _logit(probability))


def _train_task(
    examples: Sequence[_Example],
    *,
    task: str,
    target: str,
    seed: int,
    ensemble_members: int,
    l2_candidates: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_train = _select_examples(examples, SELECTION_TRAIN_FOLDS)
    selection_validation = _select_examples(examples, (SELECTION_FOLD,))
    trials: list[dict[str, Any]] = []
    for l2 in l2_candidates:
        metrics, scaler = _fit_candidate(selection_train, selection_validation, float(l2))
        trials.append(
            {
                "l2": float(l2),
                "validation_metrics": metrics,
                "scaler_fitted_folds": list(SELECTION_TRAIN_FOLDS),
                "scaler_mean_sha256": hashlib.sha256(
                    canonical_json_bytes(list(scaler.mean))
                ).hexdigest(),
            }
        )
    chosen = min(trials, key=lambda trial: (trial["validation_metrics"]["log_loss"], trial["l2"]))
    selected_l2 = float(chosen["l2"])

    refit = _select_examples(examples, REFIT_FOLDS)
    scaler = _fit_scaler(refit)
    members: list[dict[str, Any]] = []
    task_seed = seed + (0 if task == "activity" else 1_000_003)
    for member_index in range(ensemble_members):
        sampled = _bootstrap_examples(refit, seed=task_seed, member=member_index)
        intercept, coefficients = _fit_logistic(
            [scaler.transform(example.features) for example in sampled],
            [example.label for example in sampled],
            l2=selected_l2,
        )
        members.append(
            {
                "member": member_index,
                "sampling": "all_refit_sequences" if member_index == 0 else "stratified_bootstrap",
                "intercept": intercept,
                "coefficients": list(coefficients),
            }
        )

    calibration_examples = _select_examples(examples, (CALIBRATION_FOLD,))
    calibration_raw = _raw_ensemble_probabilities(
        calibration_examples, scaler=scaler, members=members
    )
    slope, calibration_intercept = _fit_calibration(
        [example.label for example in calibration_examples], calibration_raw
    )
    calibration_probabilities = [
        _calibrate(probability, slope=slope, intercept=calibration_intercept)
        for probability in calibration_raw
    ]

    # Fold 0 is touched only here, once model selection, refit, and calibration are frozen.
    test_examples = _select_examples(examples, (TEST_FOLD,))
    test_raw = _raw_ensemble_probabilities(test_examples, scaler=scaler, members=members)
    test_probabilities = [
        _calibrate(probability, slope=slope, intercept=calibration_intercept)
        for probability in test_raw
    ]
    model = {
        "target": target,
        "feature_names": list(FEATURE_NAMES),
        "scaler": {
            "mean": list(scaler.mean),
            "scale": list(scaler.scale),
            "fitted_folds": list(REFIT_FOLDS),
            "label_independent": True,
        },
        "selected_l2": selected_l2,
        "members": members,
        "calibration": {
            "method": "platt_on_ensemble_probability_logit",
            "slope": slope,
            "intercept": calibration_intercept,
            "fit_fold": CALIBRATION_FOLD,
            "raw_metrics": _metrics(
                [example.label for example in calibration_examples], calibration_raw
            ),
            "calibrated_metrics": _metrics(
                [example.label for example in calibration_examples], calibration_probabilities
            ),
        },
        "uncertainty": {
            "method": "max_of_calibrated_member_std_and_scaled_feature_rms_ood",
            "ood_rms_start": 2.0,
            "ood_rms_span": 4.0,
        },
    }
    report = {
        "model_selection": {
            "train_folds": list(SELECTION_TRAIN_FOLDS),
            "validation_fold": SELECTION_FOLD,
            "criterion": "minimum_log_loss",
            "trials": trials,
            "selected_l2": selected_l2,
        },
        "refit_folds": list(REFIT_FOLDS),
        "calibration_fold": CALIBRATION_FOLD,
        "diagnostic_fold": TEST_FOLD,
        "diagnostic_initially_evaluated_once_after_calibration": True,
        "diagnostic_raw_metrics": _metrics([example.label for example in test_examples], test_raw),
        "diagnostic_calibrated_metrics": _metrics(
            [example.label for example in test_examples], test_probabilities
        ),
    }
    return model, report


def train_linear_oracle(
    *,
    mic_csv: Path,
    hc50_csv: Path,
    checkpoint_path: Path,
    seed: int = 42,
    ensemble_members: int = 5,
    l2_candidates: Sequence[float] = DEFAULT_L2_CANDIDATES,
    dataset_manifest_path: Path | None = None,
    oracle_config_path: Path | None = None,
) -> dict[str, Any]:
    """Train and atomically write a deterministic, dependency-free oracle checkpoint."""

    if ensemble_members < 1:
        raise ValueError("ensemble_members must be positive")
    normalized_l2 = tuple(sorted({float(value) for value in l2_candidates}))
    if not normalized_l2 or any(value < 0.0 or not math.isfinite(value) for value in normalized_l2):
        raise ValueError("l2_candidates must contain finite non-negative values")
    mic_csv = mic_csv.resolve()
    hc50_csv = hc50_csv.resolve()
    activity_examples, activity_data = _aggregate_examples(
        mic_csv, label_column="active_le_16um", task="activity"
    )
    safety_examples, safety_data = _aggregate_examples(
        hc50_csv, label_column="safe_ge_128um", task="safety"
    )
    activity_model, activity_report = _train_task(
        activity_examples,
        task="activity",
        target="conservative_sequence_activity_at_mic_le_16um",
        seed=seed,
        ensemble_members=ensemble_members,
        l2_candidates=normalized_l2,
    )
    safety_model, safety_report = _train_task(
        safety_examples,
        task="safety",
        target="conservative_sequence_safety_at_hc50_ge_128um",
        seed=seed,
        ensemble_members=ensemble_members,
        l2_candidates=normalized_l2,
    )
    trainer_config = {
        "seed": seed,
        "ensemble_members": ensemble_members,
        "l2_candidates": list(normalized_l2),
        "optimizer": "deterministic_newton_irls",
        "aggregation": "minimum_definitive_threshold_label_per_sequence",
        "split_contract": {
            "diagnostic_fold": TEST_FOLD,
            "calibration_fold": CALIBRATION_FOLD,
            "model_selection_train_folds": list(SELECTION_TRAIN_FOLDS),
            "model_selection_validation_fold": SELECTION_FOLD,
            "refit_folds": list(REFIT_FOLDS),
        },
    }
    inputs: dict[str, Any] = {
        "mic_csv": {"sha256": sha256_file(mic_csv), "filename": mic_csv.name},
        "hc50_csv": {"sha256": sha256_file(hc50_csv), "filename": hc50_csv.name},
    }
    for key, path in (
        ("dataset_manifest", dataset_manifest_path),
        ("oracle_config", oracle_config_path),
    ):
        if path is not None:
            resolved = path.resolve()
            inputs[key] = {"sha256": sha256_file(resolved), "filename": resolved.name}
    implementation_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "name": MODEL_NAME,
        "feature_names": list(FEATURE_NAMES),
        "outputs": {
            "activity": "P(conservative MIC <= 16 uM)",
            "toxicity": "1 - P(conservative HC50 >= 128 uM)",
            "uncertainty": "max(activity uncertainty, safety uncertainty)",
        },
        "tasks": {"activity": activity_model, "safety": safety_model},
        "training": {
            "config": trainer_config,
            "config_sha256": hashlib.sha256(canonical_json_bytes(trainer_config)).hexdigest(),
            "data_summary": {"activity": activity_data, "safety": safety_data},
            "reports": {"activity": activity_report, "safety": safety_report},
        },
        "provenance": {
            "inputs": inputs,
            "implementation": {
                "filename": implementation_path.name,
                "sha256": sha256_file(implementation_path),
            },
            "runtime_dependencies": ["python_standard_library", "amp_challenge.physchem"],
        },
        "limitations": [
            "This baseline uses physicochemical features, not protein-language-model embeddings.",
            "DRAMP-only sequence-level labels are sparse and heterogeneous.",
            "Conservative aggregation treats every conflicting sequence as threshold-negative.",
            "Organism, strain, assay, and terminal-modification covariates are not modeled.",
            "Fold-0 metrics informed release head selection and are development diagnostics, "
            "not unbiased final estimates.",
        ],
    }
    payload["checkpoint_id"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    # Validate the exact payload before it reaches disk.
    LinearOracle._from_payload(payload)
    atomic_write_json(checkpoint_path, payload)
    return payload


class LinearOracle:
    """Strict offline inference for an audited linear-oracle JSON checkpoint."""

    def __init__(self, payload: dict[str, Any], *, checkpoint_path: Path | None = None) -> None:
        self._payload = payload
        self.checkpoint_path = checkpoint_path
        self.name = str(payload["name"])

    @classmethod
    def from_checkpoint(cls, path: Path) -> LinearOracle:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load linear-oracle checkpoint {path}: {error}") from error
        oracle = cls._from_payload(payload)
        oracle.checkpoint_path = path
        return oracle

    @classmethod
    def _from_payload(cls, payload: Any) -> LinearOracle:
        if not isinstance(payload, dict):
            raise ValueError("linear-oracle checkpoint root must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported linear-oracle checkpoint schema")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError("unsupported linear-oracle model type")
        if payload.get("name") != MODEL_NAME:
            raise ValueError("unexpected linear-oracle model name")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("checkpoint feature names/order do not match this implementation")
        checkpoint_id = payload.get("checkpoint_id")
        unsigned = dict(payload)
        unsigned.pop("checkpoint_id", None)
        expected_id = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if checkpoint_id != expected_id:
            raise ValueError("linear-oracle checkpoint_id does not match its contents")
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != {"activity", "safety"}:
            raise ValueError("checkpoint must contain exactly activity and safety tasks")
        for task_name, task in tasks.items():
            if not isinstance(task, dict):
                raise ValueError(f"{task_name}: task checkpoint must be an object")
            if task.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError(f"{task_name}: feature names/order mismatch")
            scaler = task.get("scaler")
            if not isinstance(scaler, dict):
                raise ValueError(f"{task_name}: missing scaler")
            means = scaler.get("mean")
            scales = scaler.get("scale")
            if not isinstance(means, list) or not isinstance(scales, list):
                raise ValueError(f"{task_name}: invalid scaler vectors")
            if len(means) != len(FEATURE_NAMES) or len(scales) != len(FEATURE_NAMES):
                raise ValueError(f"{task_name}: scaler width mismatch")
            if not all(isinstance(value, int | float) and math.isfinite(value) for value in means):
                raise ValueError(f"{task_name}: non-finite scaler mean")
            if not all(
                isinstance(value, int | float) and math.isfinite(value) and value > 0
                for value in scales
            ):
                raise ValueError(f"{task_name}: scaler scales must be finite and positive")
            members = task.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(f"{task_name}: ensemble must be non-empty")
            for member in members:
                if not isinstance(member, dict):
                    raise ValueError(f"{task_name}: invalid ensemble member")
                intercept = member.get("intercept")
                coefficients = member.get("coefficients")
                if not isinstance(intercept, int | float) or not math.isfinite(intercept):
                    raise ValueError(f"{task_name}: non-finite intercept")
                if not isinstance(coefficients, list) or len(coefficients) != len(FEATURE_NAMES):
                    raise ValueError(f"{task_name}: coefficient width mismatch")
                if not all(
                    isinstance(value, int | float) and math.isfinite(value)
                    for value in coefficients
                ):
                    raise ValueError(f"{task_name}: non-finite coefficient")
            calibration = task.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError(f"{task_name}: missing calibration")
            for key in ("slope", "intercept"):
                value = calibration.get(key)
                if not isinstance(value, int | float) or not math.isfinite(value):
                    raise ValueError(f"{task_name}: invalid calibration {key}")
        return cls(payload)

    def _task_prediction(self, task_name: str, features: Sequence[float]) -> tuple[float, float]:
        task = self._payload["tasks"][task_name]
        scaler_payload = task["scaler"]
        scaler = _Scaler(
            tuple(float(value) for value in scaler_payload["mean"]),
            tuple(float(value) for value in scaler_payload["scale"]),
        )
        transformed = scaler.transform(features)
        raw_members = [
            _sigmoid(
                _predict_logit(
                    transformed,
                    float(member["intercept"]),
                    [float(value) for value in member["coefficients"]],
                )
            )
            for member in task["members"]
        ]
        raw_mean = statistics.fmean(raw_members)
        calibration = task["calibration"]
        slope = float(calibration["slope"])
        intercept = float(calibration["intercept"])
        probability = _calibrate(raw_mean, slope=slope, intercept=intercept)
        calibrated_members = [
            _calibrate(member_probability, slope=slope, intercept=intercept)
            for member_probability in raw_members
        ]
        disagreement = statistics.pstdev(calibrated_members) if len(calibrated_members) > 1 else 0.0
        rms_z = math.sqrt(sum(value * value for value in transformed) / len(transformed))
        uncertainty_config = task["uncertainty"]
        ood = _clamp_probability(
            (rms_z - float(uncertainty_config["ood_rms_start"]))
            / float(uncertainty_config["ood_rms_span"])
        )
        return _clamp_probability(probability), _clamp_probability(max(disagreement, ood))

    def predict_components(self, sequence: str) -> tuple[float, float, float, float]:
        """Return both biological heads and their head-specific uncertainties."""
        features = _feature_vector(sequence)
        activity, activity_uncertainty = self._task_prediction("activity", features)
        safety, safety_uncertainty = self._task_prediction("safety", features)
        return (
            activity,
            _clamp_probability(1.0 - safety),
            activity_uncertainty,
            safety_uncertainty,
        )

    def predict(self, sequence: str) -> LinearOraclePrediction:
        activity, toxicity, activity_uncertainty, toxicity_uncertainty = self.predict_components(
            sequence
        )
        return LinearOraclePrediction(
            activity=activity,
            toxicity=toxicity,
            uncertainty=_clamp_probability(max(activity_uncertainty, toxicity_uncertainty)),
        )

    def predict_values(self, sequence: str) -> tuple[float, float, float]:
        prediction = self.predict(sequence)
        return prediction.activity, prediction.toxicity, prediction.uncertainty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the dependency-free linear AMP oracle")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--oracle-config", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train and write the checkpoint; otherwise print a dry-run plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dataset_dir": str(arguments.dataset_dir),
                    "output": str(arguments.output),
                    "seed": arguments.seed,
                    "ensemble_members": arguments.ensemble_members,
                    "execute_required": True,
                },
                sort_keys=True,
            )
        )
        return 0
    payload = train_linear_oracle(
        mic_csv=arguments.dataset_dir / "mic_measurements.csv",
        hc50_csv=arguments.dataset_dir / "hc50_measurements.csv",
        checkpoint_path=arguments.output,
        seed=arguments.seed,
        ensemble_members=arguments.ensemble_members,
        dataset_manifest_path=arguments.dataset_manifest,
        oracle_config_path=arguments.oracle_config,
    )
    print(
        json.dumps({"checkpoint": str(arguments.output), "checkpoint_id": payload["checkpoint_id"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
