from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fasta import read_sequences
from .linear_oracle import FEATURE_NAMES
from .physchem import describe
from .utils import atomic_write_json, atomic_write_text, canonical_json_bytes, sha256_file

MODEL_NAME = "esm2-mic16-v2"
MODEL_TYPE = "frozen-esm2-context-logistic-ensemble"
SCHEMA_VERSION = 1
TARGET_COLUMN = "active_le_16um"
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)
FOLDS = tuple(range(5))


@dataclass(frozen=True, slots=True)
class MicExample:
    sequence: str
    fold: int
    organism: str
    gram: str
    n_terminal: str
    c_terminal: str
    label: int


@dataclass(frozen=True, slots=True)
class FeatureState:
    use_esm: bool
    use_physchem: bool
    numeric_mean: tuple[float, ...]
    numeric_scale: tuple[float, ...]
    organisms: tuple[str, ...]
    n_terminals: tuple[str, ...]
    c_terminals: tuple[str, ...]
    rare_organism_min_rows: int

    @property
    def numeric_width(self) -> int:
        return len(self.numeric_mean)

    @property
    def width(self) -> int:
        return (
            self.numeric_width
            + len(self.organisms)
            + 1
            + 3
            + len(self.n_terminals)
            + 1
            + len(self.c_terminals)
            + 1
        )


def _optional_training_imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise RuntimeError("ESM training requires `uv sync --locked --extra train`") from error
    return np, LogisticRegression, (average_precision_score, roc_auc_score)


def _sequence_digest(sequences: list[str]) -> str:
    return hashlib.sha256(("\n".join(sequences) + "\n").encode("ascii")).hexdigest()


def read_split_sequences(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: missing sequence column")
        sequences = sorted({row["sequence"].strip().upper() for row in reader})
    if not sequences:
        raise ValueError(f"{path}: no sequences")
    return sequences


def extract_esm_embeddings(
    *,
    sequences: list[str],
    model_source: str | Path,
    model_name: str,
    revision: str,
    output_path: Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract deterministic masked-mean ESM embeddings into a content-addressed cache."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    np, _, _ = _optional_training_imports()
    ordered = sorted(set(sequences))
    expected_sequence_digest = _sequence_digest(ordered)
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            cached.get("model_name") == model_name
            and cached.get("revision") == revision
            and cached.get("sequence_sha256") == expected_sequence_digest
            and cached.get("artifact", {}).get("sha256") == sha256_file(output_path)
        ):
            return cached
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("embedding extraction requires torch and transformers") from error

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    source = str(model_source)
    local_only = Path(source).exists()
    load_kwargs: dict[str, Any] = {"local_files_only": local_only}
    if not local_only:
        load_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    model = AutoModel.from_pretrained(source, add_pooling_layer=False, **load_kwargs)
    model.eval().to(device)

    batches: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            batch_sequences = ordered[start : start + batch_size]
            encoded = tokenizer(
                batch_sequences,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = encoded.pop("special_tokens_mask").to(device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool() & ~special.bool()
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            batches.append(pooled.cpu().to(torch.float32).numpy())
    embeddings = np.concatenate(batches, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sequences=np.asarray(ordered),
        embeddings=embeddings,
    )
    model_files: dict[str, str] = {}
    model_path = Path(source)
    if model_path.is_dir():
        for name in (
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ):
            candidate = model_path / name
            if candidate.is_file():
                model_files[name] = sha256_file(candidate)
    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "revision": revision,
        "pooling": "attention_and_special-token-masked_mean",
        "sequence_count": len(ordered),
        "sequence_sha256": expected_sequence_digest,
        "embedding_width": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "artifact": {"filename": output_path.name, "sha256": sha256_file(output_path)},
        "local_model_files": model_files,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_embeddings(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    np, _, _ = _optional_training_imports()
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing embedding manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"{path}: embedding artifact hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        sequences = [str(value) for value in payload["sequences"].tolist()]
        matrix = payload["embeddings"].astype(np.float64)
    if len(sequences) != len(set(sequences)) or matrix.shape[0] != len(sequences):
        raise ValueError(f"{path}: invalid sequence/embedding alignment")
    if _sequence_digest(sorted(sequences)) != manifest.get("sequence_sha256"):
        raise ValueError(f"{path}: sequence digest mismatch")
    return dict(zip(sequences, matrix, strict=True)), manifest


def load_mic_examples(path: Path, *, target_column: str = TARGET_COLUMN) -> list[MicExample]:
    required = {
        "sequence",
        "fold",
        "organism_name",
        "gram",
        "n_terminal",
        "c_terminal",
        target_column,
    }
    examples: list[MicExample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            label = row[target_column].strip()
            if label == "":
                continue
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{line}: invalid {target_column}={label!r}")
            fold = int(row["fold"])
            if fold not in FOLDS:
                raise ValueError(f"{path}:{line}: unexpected fold {fold}")
            examples.append(
                MicExample(
                    sequence=row["sequence"].strip().upper(),
                    fold=fold,
                    organism=row["organism_name"].strip() or "unknown",
                    gram=row["gram"].strip() or "unknown",
                    n_terminal=row["n_terminal"].strip() or "unknown",
                    c_terminal=row["c_terminal"].strip() or "unknown",
                    label=int(label),
                )
            )
    if not examples:
        raise ValueError(f"{path}: no definitive {target_column} examples")
    return examples


def _numeric_vector(
    sequence: str,
    embeddings: dict[str, Any],
    *,
    use_esm: bool,
    use_physchem: bool,
) -> Any:
    np, _, _ = _optional_training_imports()
    blocks: list[Any] = []
    if use_esm:
        if sequence not in embeddings:
            raise ValueError(f"missing ESM embedding for {sequence}")
        blocks.append(np.asarray(embeddings[sequence], dtype=np.float64))
    if use_physchem:
        values = describe(sequence).as_dict()
        blocks.append(np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64))
    if not blocks:
        raise ValueError("at least one numeric feature family must be enabled")
    return np.concatenate(blocks)


def _fit_feature_state(
    examples: list[MicExample],
    embeddings: dict[str, Any],
    *,
    use_esm: bool,
    use_physchem: bool,
    rare_organism_min_rows: int,
) -> FeatureState:
    np, _, _ = _optional_training_imports()
    unique_sequences = sorted({example.sequence for example in examples})
    numeric = np.stack(
        [
            _numeric_vector(
                sequence,
                embeddings,
                use_esm=use_esm,
                use_physchem=use_physchem,
            )
            for sequence in unique_sequences
        ]
    )
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-8] = 1.0
    organism_counts = Counter(example.organism for example in examples)
    organisms = tuple(
        sorted(
            organism
            for organism, count in organism_counts.items()
            if count >= rare_organism_min_rows
        )
    )
    return FeatureState(
        use_esm=use_esm,
        use_physchem=use_physchem,
        numeric_mean=tuple(float(value) for value in mean),
        numeric_scale=tuple(float(value) for value in scale),
        organisms=organisms,
        n_terminals=tuple(sorted({example.n_terminal for example in examples})),
        c_terminals=tuple(sorted({example.c_terminal for example in examples})),
        rare_organism_min_rows=rare_organism_min_rows,
    )


def _one_hot(value: str, categories: tuple[str, ...]) -> list[float]:
    output = [0.0] * (len(categories) + 1)
    try:
        index = categories.index(value)
    except ValueError:
        index = len(categories)
    output[index] = 1.0
    return output


def _matrix(examples: list[MicExample], embeddings: dict[str, Any], state: FeatureState) -> Any:
    np, _, _ = _optional_training_imports()
    rows: list[Any] = []
    mean = np.asarray(state.numeric_mean)
    scale = np.asarray(state.numeric_scale)
    for example in examples:
        numeric = _numeric_vector(
            example.sequence,
            embeddings,
            use_esm=state.use_esm,
            use_physchem=state.use_physchem,
        )
        gram = [
            float(example.gram == "negative"),
            float(example.gram == "positive"),
            float(example.gram not in {"negative", "positive"}),
        ]
        context = [
            *_one_hot(example.organism, state.organisms),
            *gram,
            *_one_hot(example.n_terminal, state.n_terminals),
            *_one_hot(example.c_terminal, state.c_terminals),
        ]
        rows.append(np.concatenate(((numeric - mean) / scale, np.asarray(context))))
    output = np.stack(rows)
    if output.shape[1] != state.width:
        raise AssertionError(f"feature width {output.shape[1]} != {state.width}")
    return output


def _labels(examples: list[MicExample]) -> Any:
    np, _, _ = _optional_training_imports()
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _sequence_weights(examples: list[MicExample]) -> Any:
    np, _, _ = _optional_training_imports()
    counts = Counter(example.sequence for example in examples)
    weights = np.asarray([1.0 / counts[example.sequence] for example in examples])
    return weights * len(weights) / weights.sum()


def _sigmoid(values: Any) -> Any:
    np, _, _ = _optional_training_imports()
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_base(matrix: Any, labels: Any, weights: Any, *, c_value: float, seed: int) -> Any:
    _, LogisticRegression, _ = _optional_training_imports()
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=4000,
        random_state=seed,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def _fit_platt(logits: Any, labels: Any, weights: Any, *, seed: int) -> tuple[float, float]:
    np, LogisticRegression, _ = _optional_training_imports()
    calibrator = LogisticRegression(
        C=100.0,
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    calibrator.fit(np.asarray(logits).reshape(-1, 1), labels, sample_weight=weights)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _weighted_top_enrichment(labels: Any, probabilities: Any, weights: Any) -> float:
    np, _, _ = _optional_training_imports()
    order = np.argsort(-probabilities, kind="stable")
    total_weight = float(weights.sum())
    cutoff = 0.1 * total_weight
    selected_positive = 0.0
    selected_weight = 0.0
    for index in order:
        take = min(float(weights[index]), cutoff - selected_weight)
        if take <= 0:
            break
        selected_positive += take * float(labels[index])
        selected_weight += take
    prevalence = float((labels * weights).sum() / total_weight)
    precision = selected_positive / selected_weight
    return precision / prevalence if prevalence > 0 else math.nan


def _metrics(labels: Any, probabilities: Any, weights: Any) -> dict[str, float | int]:
    np, _, metric_functions = _optional_training_imports()
    average_precision_score, roc_auc_score = metric_functions
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    log_loss = -float(
        (weights * (labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))).sum()
        / weights.sum()
    )
    brier = float((weights * (probabilities - labels) ** 2).sum() / weights.sum())
    ece = 0.0
    for start in np.linspace(0.0, 1.0, 11)[:-1]:
        end = start + 0.1
        mask = (probabilities >= start) & (
            (probabilities < end) if end < 1.0 else (probabilities <= end)
        )
        if mask.any():
            bin_weight = float(weights[mask].sum())
            observed = float((weights[mask] * labels[mask]).sum() / bin_weight)
            predicted = float((weights[mask] * probabilities[mask]).sum() / bin_weight)
            ece += bin_weight / float(weights.sum()) * abs(observed - predicted)
    return {
        "rows": int(len(labels)),
        "effective_weighted_rows": float(weights.sum() ** 2 / (weights**2).sum()),
        "prevalence": float((weights * labels).sum() / weights.sum()),
        "auroc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "auprc": float(average_precision_score(labels, probabilities, sample_weight=weights)),
        "log_loss": log_loss,
        "brier": brier,
        "ece_10bin": ece,
        "top_10_percent_enrichment": _weighted_top_enrichment(labels, probabilities, weights),
    }


def _sequence_metrics(examples: list[MicExample], probabilities: Any) -> dict[str, Any]:
    np, _, _ = _optional_training_imports()
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for example, probability in zip(examples, probabilities, strict=True):
        grouped[example.sequence].append((example.label, float(probability)))
    labels = np.asarray([min(label for label, _ in values) for values in grouped.values()])
    conservative = np.asarray(
        [
            float(np.quantile([probability for _, probability in values], 0.2))
            for values in grouped.values()
        ]
    )
    weights = np.ones(len(labels), dtype=np.float64)
    return _metrics(labels, conservative, weights)


def _serialize_state(state: FeatureState) -> dict[str, Any]:
    return {
        "use_esm": state.use_esm,
        "use_physchem": state.use_physchem,
        "numeric_mean": list(state.numeric_mean),
        "numeric_scale": list(state.numeric_scale),
        "organisms": list(state.organisms),
        "n_terminals": list(state.n_terminals),
        "c_terminals": list(state.c_terminals),
        "rare_organism_min_rows": state.rare_organism_min_rows,
        "width": state.width,
    }


def _select_c(
    examples: list[MicExample],
    embeddings: dict[str, Any],
    *,
    folds: tuple[int, ...],
    use_esm: bool,
    use_physchem: bool,
    c_values: tuple[float, ...],
    rare_organism_min_rows: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for c_value in c_values:
        losses: list[float] = []
        for validation_fold in folds:
            train = [
                example
                for example in examples
                if example.fold in folds and example.fold != validation_fold
            ]
            validation = [example for example in examples if example.fold == validation_fold]
            state = _fit_feature_state(
                train,
                embeddings,
                use_esm=use_esm,
                use_physchem=use_physchem,
                rare_organism_min_rows=rare_organism_min_rows,
            )
            model = _fit_base(
                _matrix(train, embeddings, state),
                _labels(train),
                _sequence_weights(train),
                c_value=c_value,
                seed=seed + validation_fold,
            )
            probability = _sigmoid(model.decision_function(_matrix(validation, embeddings, state)))
            losses.append(
                float(
                    _metrics(_labels(validation), probability, _sequence_weights(validation))[
                        "log_loss"
                    ]
                )
            )
        trials.append(
            {
                "c": c_value,
                "inner_log_losses": losses,
                "mean_inner_log_loss": statistics.fmean(losses),
            }
        )
    chosen = min(trials, key=lambda trial: (trial["mean_inner_log_loss"], trial["c"]))
    return float(chosen["c"]), trials


def _cross_validate_family(
    examples: list[MicExample],
    embeddings: dict[str, Any],
    *,
    use_esm: bool,
    use_physchem: bool,
    c_values: tuple[float, ...],
    rare_organism_min_rows: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    np, _, _ = _optional_training_imports()
    oof_examples: list[MicExample] = []
    oof_probabilities: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    selected_cs: list[float] = []
    for outer_fold in FOLDS:
        calibration_fold = (outer_fold + 1) % len(FOLDS)
        train_folds = tuple(fold for fold in FOLDS if fold not in {outer_fold, calibration_fold})
        selected_c, trials = _select_c(
            examples,
            embeddings,
            folds=train_folds,
            use_esm=use_esm,
            use_physchem=use_physchem,
            c_values=c_values,
            rare_organism_min_rows=rare_organism_min_rows,
            seed=seed + 100 * outer_fold,
        )
        selected_cs.append(selected_c)
        train = [example for example in examples if example.fold in train_folds]
        calibration = [example for example in examples if example.fold == calibration_fold]
        test = [example for example in examples if example.fold == outer_fold]
        state = _fit_feature_state(
            train,
            embeddings,
            use_esm=use_esm,
            use_physchem=use_physchem,
            rare_organism_min_rows=rare_organism_min_rows,
        )
        base = _fit_base(
            _matrix(train, embeddings, state),
            _labels(train),
            _sequence_weights(train),
            c_value=selected_c,
            seed=seed + outer_fold,
        )
        calibration_logits = base.decision_function(_matrix(calibration, embeddings, state))
        slope, intercept = _fit_platt(
            calibration_logits,
            _labels(calibration),
            _sequence_weights(calibration),
            seed=seed + 1000 + outer_fold,
        )
        test_logits = base.decision_function(_matrix(test, embeddings, state))
        probabilities = _sigmoid(intercept + slope * test_logits)
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "train_folds": list(train_folds),
                "selected_c": selected_c,
                "inner_trials": trials,
                "measurement_metrics": _metrics(
                    _labels(test), probabilities, _sequence_weights(test)
                ),
                "sequence_broad_spectrum_metrics": _sequence_metrics(test, probabilities),
            }
        )
        oof_examples.extend(test)
        oof_probabilities.extend(float(value) for value in probabilities)
    probabilities = np.asarray(oof_probabilities)
    report = {
        "folds": fold_reports,
        "pooled_measurement_metrics": _metrics(
            _labels(oof_examples), probabilities, _sequence_weights(oof_examples)
        ),
        "pooled_sequence_broad_spectrum_metrics": _sequence_metrics(oof_examples, probabilities),
    }
    selected_c = min(
        sorted(set(selected_cs)),
        key=lambda value: (-selected_cs.count(value), value),
    )
    return report, selected_c


def _fit_deployment_members(
    examples: list[MicExample],
    embeddings: dict[str, Any],
    *,
    c_value: float,
    rare_organism_min_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for calibration_fold in FOLDS:
        train = [example for example in examples if example.fold != calibration_fold]
        calibration = [example for example in examples if example.fold == calibration_fold]
        state = _fit_feature_state(
            train,
            embeddings,
            use_esm=True,
            use_physchem=True,
            rare_organism_min_rows=rare_organism_min_rows,
        )
        model = _fit_base(
            _matrix(train, embeddings, state),
            _labels(train),
            _sequence_weights(train),
            c_value=c_value,
            seed=seed + calibration_fold,
        )
        calibration_logits = model.decision_function(_matrix(calibration, embeddings, state))
        slope, intercept = _fit_platt(
            calibration_logits,
            _labels(calibration),
            _sequence_weights(calibration),
            seed=seed + 2000 + calibration_fold,
        )
        members.append(
            {
                "member": calibration_fold,
                "base_fit_folds": [fold for fold in FOLDS if fold != calibration_fold],
                "calibration_fold": calibration_fold,
                "c": c_value,
                "feature_state": _serialize_state(state),
                "base_intercept": float(model.intercept_[0]),
                "base_coefficients": [float(value) for value in model.coef_[0]],
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
    return members


def train_esm_mic16_oracle(
    *,
    mic_csv: Path,
    embeddings_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    dataset_manifest_path: Path,
    oracle_config_path: Path,
    seed: int = 42,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    rare_organism_min_rows: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark ESM2 against the same-fold physchem baseline and fit deployment members."""

    examples = load_mic_examples(mic_csv)
    embeddings, embedding_manifest = load_embeddings(embeddings_path)
    missing = sorted({example.sequence for example in examples} - embeddings.keys())
    if missing:
        raise ValueError(f"embedding cache misses {len(missing)} training sequences")

    families = {
        "physchem-context": {"use_esm": False, "use_physchem": True},
        "esm2-physchem-context": {"use_esm": True, "use_physchem": True},
    }
    reports: dict[str, Any] = {}
    selected: dict[str, float] = {}
    for family, flags in families.items():
        reports[family], selected[family] = _cross_validate_family(
            examples,
            embeddings,
            c_values=c_values,
            rare_organism_min_rows=rare_organism_min_rows,
            seed=seed,
            **flags,
        )

    baseline = reports["physchem-context"]["pooled_measurement_metrics"]
    candidate = reports["esm2-physchem-context"]["pooled_measurement_metrics"]
    deltas = {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("auroc", "auprc", "top_10_percent_enrichment")
    }
    deltas["log_loss"] = float(candidate["log_loss"]) - float(baseline["log_loss"])
    gate = {
        "kind": "development_gate_not_unbiased_final_evidence",
        "requirements": {
            "auroc_delta_min": 0.02,
            "auprc_delta_min": 0.02,
            "top_10_percent_enrichment_delta_min": 0.05,
            "log_loss_delta_max": 0.02,
        },
    }
    gate["passed"] = bool(
        deltas["auroc"] >= 0.02
        and deltas["auprc"] >= 0.02
        and deltas["top_10_percent_enrichment"] >= 0.05
        and deltas["log_loss"] <= 0.02
    )

    members = _fit_deployment_members(
        examples,
        embeddings,
        c_value=selected["esm2-physchem-context"],
        rare_organism_min_rows=rare_organism_min_rows,
        seed=seed,
    )
    report = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "target": "P(MIC <= 16 uM | peptide, organism, Gram, termini)",
        "evaluation": (
            "five rotating outer cluster folds; separate calibration fold; inner-CV C selection"
        ),
        "weighting": "each sequence has equal total weight within every fit/evaluation split",
        "families": reports,
        "selected_c": selected,
        "candidate_minus_baseline": deltas,
        "development_gate": gate,
        "limitations": [
            "All labels come from one DRAMP snapshot.",
            "The 70% global-edit clusters are weaker than a 40% MMseqs2 family split.",
            "Fold 0 and this source informed v1 decisions, so these are development metrics.",
            "Strain and assay covariates are not modeled in this v2 benchmark.",
        ],
    }
    atomic_write_json(report_path, report)

    checkpoint: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "name": MODEL_NAME,
        "target": "mic_le_16um",
        "backbone": {
            "name": embedding_manifest["model_name"],
            "revision": embedding_manifest["revision"],
            "pooling": embedding_manifest["pooling"],
            "embedding_width": embedding_manifest["embedding_width"],
        },
        "features": ["frozen_esm2", "physchem", "organism", "gram", "termini"],
        "members": members,
        "training": {
            "seed": seed,
            "c_values": list(c_values),
            "selected_c": selected["esm2-physchem-context"],
            "rare_organism_min_rows": rare_organism_min_rows,
            "measurement_rows": len(examples),
            "unique_sequences": len({example.sequence for example in examples}),
            "development_gate_passed": gate["passed"],
        },
        "provenance": {
            "mic_csv": sha256_file(mic_csv),
            "dataset_manifest": sha256_file(dataset_manifest_path),
            "oracle_config": sha256_file(oracle_config_path),
            "embedding_artifact": sha256_file(embeddings_path),
            "embedding_manifest": sha256_file(embeddings_path.with_suffix(".manifest.json")),
            "benchmark_report": sha256_file(report_path),
            "implementation": sha256_file(Path(__file__)),
        },
    }
    checkpoint["checkpoint_id"] = hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest()
    atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint, report


def _state_from_payload(payload: dict[str, Any]) -> FeatureState:
    return FeatureState(
        use_esm=bool(payload["use_esm"]),
        use_physchem=bool(payload["use_physchem"]),
        numeric_mean=tuple(float(value) for value in payload["numeric_mean"]),
        numeric_scale=tuple(float(value) for value in payload["numeric_scale"]),
        organisms=tuple(str(value) for value in payload["organisms"]),
        n_terminals=tuple(str(value) for value in payload["n_terminals"]),
        c_terminals=tuple(str(value) for value in payload["c_terminals"]),
        rare_organism_min_rows=int(payload["rare_organism_min_rows"]),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint schema")
    if payload.get("model_type") != MODEL_TYPE or payload.get("name") != MODEL_NAME:
        raise ValueError(f"{path}: unexpected ESM oracle checkpoint type")
    claimed = payload.get("checkpoint_id")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_id", None)
    if claimed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError(f"{path}: checkpoint_id mismatch")
    if len(payload.get("members", [])) != len(FOLDS):
        raise ValueError(f"{path}: expected {len(FOLDS)} calibrated members")
    return payload


def _read_prefilter_scores(path: Path) -> dict[str, dict[str, float]]:
    required = {"sequence", "activity", "toxicity", "uncertainty", "utility"}
    output: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing score columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            sequence = row["sequence"].strip().upper()
            if sequence in output:
                raise ValueError(f"{path}:{line}: duplicate sequence {sequence}")
            output[sequence] = {
                key: float(row[key]) for key in ("activity", "toxicity", "uncertainty", "utility")
            }
    return output


def _prefilter_sequences(
    sequences: list[str],
    scores: dict[str, dict[str, float]],
    *,
    prefilter_k: int,
    exploration_k: int,
    seed: int,
) -> tuple[list[str], set[str]]:
    if prefilter_k < 1 or exploration_k < 0:
        raise ValueError("prefilter_k must be positive and exploration_k non-negative")
    missing = sorted(set(sequences) - scores.keys())
    if missing:
        raise ValueError(f"prefilter scores miss {len(missing)} candidates")
    ranked = sorted(sequences, key=lambda sequence: (-scores[sequence]["utility"], sequence))
    primary = ranked[: min(prefilter_k, len(ranked))]
    primary_set = set(primary)
    remaining = [sequence for sequence in sequences if sequence not in primary_set]
    explored = sorted(
        remaining,
        key=lambda sequence: hashlib.sha256(f"{seed}:{sequence}".encode()).hexdigest(),
    )[:exploration_k]
    selected = sorted(primary_set | set(explored))
    return selected, set(explored)


def _panel(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported panel schema")
    organisms = payload.get("organisms")
    if not isinstance(organisms, list) or not organisms:
        raise ValueError(f"{path}: panel must contain organisms")
    normalized: list[dict[str, str]] = []
    for organism in organisms:
        if not isinstance(organism, dict) or organism.get("gram") not in {
            "negative",
            "positive",
        }:
            raise ValueError(f"{path}: invalid organism entry")
        normalized.append({"name": str(organism["name"]), "gram": str(organism["gram"])})
    return normalized


def _member_predictions(
    member: dict[str, Any],
    *,
    sequences: list[str],
    organisms: list[dict[str, str]],
    embeddings: dict[str, Any],
) -> tuple[Any, Any]:
    np, _, _ = _optional_training_imports()
    state = _state_from_payload(member["feature_state"])
    examples = [
        MicExample(
            sequence=sequence,
            fold=0,
            organism=organism["name"],
            gram=organism["gram"],
            n_terminal="free",
            c_terminal="free",
            label=0,
        )
        for sequence in sequences
        for organism in organisms
    ]
    matrix = _matrix(examples, embeddings, state)
    coefficients = np.asarray(member["base_coefficients"], dtype=np.float64)
    if matrix.shape[1] != len(coefficients):
        raise ValueError("checkpoint feature width does not match inference features")
    logits = float(member["base_intercept"]) + matrix @ coefficients
    calibrated = _sigmoid(
        float(member["calibration_intercept"]) + float(member["calibration_slope"]) * logits
    ).reshape(len(sequences), len(organisms))
    numeric = matrix.reshape(len(sequences), len(organisms), -1)[:, 0, : state.numeric_width]
    ood_rms = np.sqrt((numeric**2).mean(axis=1))
    return calibrated, ood_rms


def score_esm_mic16_candidates(
    *,
    candidate_fasta: Path,
    prefilter_scores_path: Path,
    checkpoint_path: Path,
    panel_path: Path,
    model_source: str | Path,
    embeddings_path: Path,
    output_path: Path,
    prefilter_k: int,
    exploration_k: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """ESM-score a deterministic coarse subset and emit a full-library activity CSV."""

    np, _, _ = _optional_training_imports()
    sequences = read_sequences(candidate_fasta)
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"{candidate_fasta}: duplicate candidate sequences")
    baseline = _read_prefilter_scores(prefilter_scores_path)
    selected, explored = _prefilter_sequences(
        sequences,
        baseline,
        prefilter_k=prefilter_k,
        exploration_k=exploration_k,
        seed=seed,
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    backbone = checkpoint["backbone"]
    embedding_manifest = extract_esm_embeddings(
        sequences=selected,
        model_source=model_source,
        model_name=backbone["name"],
        revision=backbone["revision"],
        output_path=embeddings_path,
        batch_size=batch_size,
        device=device,
    )
    embeddings, _ = load_embeddings(embeddings_path)
    organisms = _panel(panel_path)
    member_predictions: list[Any] = []
    member_ood: list[Any] = []
    for member in checkpoint["members"]:
        prediction, ood = _member_predictions(
            member,
            sequences=selected,
            organisms=organisms,
            embeddings=embeddings,
        )
        member_predictions.append(prediction)
        member_ood.append(ood)
    predictions = np.stack(member_predictions)
    mean_prediction = predictions.mean(axis=0)
    member_std = predictions.std(axis=0).mean(axis=1)
    negative_indices = [
        index for index, organism in enumerate(organisms) if organism["gram"] == "negative"
    ]
    broad_mean = mean_prediction.mean(axis=1)
    negative_mean = mean_prediction[:, negative_indices].mean(axis=1)
    negative_min = mean_prediction[:, negative_indices].min(axis=1)
    ood_rms = np.stack(member_ood).mean(axis=0)
    ood_penalty = np.clip((ood_rms - 2.0) / 4.0, 0.0, 1.0)
    robust_activity = np.clip(
        0.55 * broad_mean
        + 0.30 * negative_mean
        + 0.15 * negative_min
        - 0.50 * member_std
        - 0.10 * ood_penalty,
        0.0,
        1.0,
    )
    uncertainty = np.clip(0.25 + 2.0 * member_std + 0.50 * ood_penalty, 0.0, 1.0)
    selected_index = {sequence: index for index, sequence in enumerate(selected)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    fields = (
        "sequence",
        "activity",
        "toxicity",
        "uncertainty",
        "esm_scored",
        "exploration_candidate",
        "esm_broad_mean",
        "esm_gram_negative_mean",
        "esm_gram_negative_min",
        "esm_member_std",
        "esm_ood_rms",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for sequence in sequences:
        prior = baseline[sequence]
        row: dict[str, Any] = {
            "sequence": sequence,
            "toxicity": prior["toxicity"],
            "exploration_candidate": int(sequence in explored),
        }
        index = selected_index.get(sequence)
        if index is None:
            row.update(
                {
                    "activity": prior["activity"],
                    "uncertainty": 1.0,
                    "esm_scored": 0,
                    "esm_broad_mean": "",
                    "esm_gram_negative_mean": "",
                    "esm_gram_negative_min": "",
                    "esm_member_std": "",
                    "esm_ood_rms": "",
                }
            )
        else:
            row.update(
                {
                    "activity": float(robust_activity[index]),
                    "uncertainty": float(uncertainty[index]),
                    "esm_scored": 1,
                    "esm_broad_mean": float(broad_mean[index]),
                    "esm_gram_negative_mean": float(negative_mean[index]),
                    "esm_gram_negative_min": float(negative_min[index]),
                    "esm_member_std": float(member_std[index]),
                    "esm_ood_rms": float(ood_rms[index]),
                }
            )
        writer.writerow(row)
    atomic_write_text(output_path, buffer.getvalue())
    report = {
        "status": "scored",
        "candidate_count": len(sequences),
        "esm_scored_count": len(selected),
        "prefilter_count": min(prefilter_k, len(sequences)),
        "exploration_count": len(explored),
        "checkpoint": {"sha256": sha256_file(checkpoint_path), "id": checkpoint["checkpoint_id"]},
        "panel": {"sha256": sha256_file(panel_path), "organisms": organisms},
        "embedding_artifact": embedding_manifest["artifact"],
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "scored_summary": {
            "activity_mean": float(robust_activity.mean()),
            "activity_max": float(robust_activity.max()),
            "uncertainty_mean": float(uncertainty.mean()),
            "ood_rms_mean": float(ood_rms.mean()),
        },
        "limitations": [
            "Only the deterministic coarse subset receives ESM2 inference.",
            "The panel is species-level and cannot resolve challenge strains or MDR phenotype.",
            "Unscored candidates retain their baseline activity and receive uncertainty 1.0.",
        ],
    }
    atomic_write_json(output_path.with_suffix(".manifest.json"), report)
    return report
