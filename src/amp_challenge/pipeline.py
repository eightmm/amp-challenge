from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .constants import MAX_LENGTH, MIN_LENGTH, STANDARD_AMINO_ACIDS
from .fasta import read_sequences, write_fasta
from .generator import generate_heuristic, load_candidates, take_candidates
from .scoring import (
    CSVOracle,
    EnsembleScorer,
    LearnedLinearOracle,
    PhyschemOracle,
    WeightedOracle,
)
from .selector import SelectionResult, select_top
from .similarity import ReferenceIndex
from .utils import atomic_write_json, atomic_write_text, git_commit, sha256_file


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_dir: Path
    library_path: Path
    top_path: Path
    scores_path: Path
    manifest_path: Path
    selection: SelectionResult


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported config schema")
    return value


def _eligible_sequence(sequence: str) -> bool:
    return MIN_LENGTH <= len(sequence) <= MAX_LENGTH and not (set(sequence) - STANDARD_AMINO_ACIDS)


def _config_bool(config: dict, key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"oracle.learned_checkpoint.{key} must be a boolean")
    return value


def _recorded_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _input_record(role: str, path: Path, *, project_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": _recorded_path(resolved, project_root=project_root),
        "sha256": sha256_file(resolved),
    }


def _write_scores(
    path: Path,
    scores: list,
    selected: SelectionResult,
) -> None:
    selected_by_sequence = {
        item.score.sequence: (item.rank, item.max_reference_ratio) for item in selected.selected
    }
    buffer = io.StringIO(newline="")
    first = scores[0].as_flat_dict()
    fieldnames = ["rank", "selected", "max_reference_ratio", *first.keys()]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for score in sorted(scores, key=lambda item: (-item.utility, item.sequence)):
        rank, maximum_ratio = selected_by_sequence.get(score.sequence, ("", ""))
        writer.writerow(
            {
                "rank": rank,
                "selected": bool(rank),
                "max_reference_ratio": maximum_ratio,
                **score.as_flat_dict(),
            }
        )
    atomic_write_text(path, buffer.getvalue())


def run_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    reference_path: Path,
    n_sequences: int,
    top_k: int,
    seed: int,
    candidate_fasta: Path | None = None,
    external_score_paths: list[Path] | None = None,
) -> PipelineResult:
    config = load_config(config_path)
    references_raw = read_sequences(reference_path)
    references = ReferenceIndex(references_raw)

    if candidate_fasta is None:
        generator_config = config["generator"]
        sequences = generate_heuristic(
            n_sequences,
            seed=seed,
            min_length=int(generator_config["min_length"]),
            max_length=int(generator_config["max_length"]),
            forbidden=references.exact,
        )
        candidate_source = {"kind": "heuristic-v0", "sha256": None}
    else:
        candidates = (
            sequence
            for sequence in load_candidates(candidate_fasta, forbidden=references.exact)
            if _eligible_sequence(sequence)
        )
        sequences = take_candidates(candidates, n_sequences)
        candidate_path = candidate_fasta.resolve()
        candidate_source = {
            "kind": "fasta",
            "path": _recorded_path(candidate_path, project_root=project_root),
            "sha256": sha256_file(candidate_path),
        }

    oracle_config = config["oracle"]
    members = [
        WeightedOracle(
            PhyschemOracle(),
            weight=float(oracle_config.get("physchem_weight", 1.0)),
        )
    ]
    learned_checkpoint_record: dict[str, str] | None = None
    learned_config = oracle_config.get("learned_checkpoint")
    if learned_config is not None:
        if not isinstance(learned_config, dict):
            raise ValueError("oracle.learned_checkpoint must be an object or null")
        try:
            learned_path_value = learned_config["path"]
            learned_sha256 = learned_config["sha256"]
        except KeyError as error:
            raise ValueError(
                f"oracle.learned_checkpoint is missing required key: {error.args[0]}"
            ) from error
        learned_path = Path(str(learned_path_value))
        if not learned_path.is_absolute():
            learned_path = project_root / learned_path
        use_activity = _config_bool(learned_config, "use_activity", default=True)
        use_toxicity = _config_bool(learned_config, "use_toxicity", default=True)
        learned_oracle = LearnedLinearOracle(
            learned_path,
            expected_sha256=str(learned_sha256),
            use_activity=use_activity,
            use_toxicity=use_toxicity,
        )
        members.append(
            WeightedOracle(
                learned_oracle,
                weight=float(learned_config.get("weight", 1.0)),
                use_activity=use_activity,
                use_toxicity=use_toxicity,
            )
        )
        learned_checkpoint_record = {
            **_input_record("learned_checkpoint", learned_path, project_root=project_root),
            "source": learned_oracle.name,
            "use_activity": use_activity,
            "use_toxicity": use_toxicity,
        }

    external_oracles: list[CSVOracle] = []
    for score_path in external_score_paths or []:
        external_oracle = CSVOracle(score_path.resolve())
        external_oracle.require_coverage(sequences)
        external_oracles.append(external_oracle)
        members.append(WeightedOracle(external_oracle))
    scorer = EnsembleScorer(
        members,
        activity_weight=float(oracle_config["activity_weight"]),
        toxicity_weight=float(oracle_config["toxicity_weight"]),
        uncertainty_weight=float(oracle_config["uncertainty_weight"]),
    )
    scores = [scorer.score(sequence) for sequence in sequences]

    selector_config = config["selector"]
    selection = select_top(
        scores,
        references,
        top_k=top_k,
        novelty_threshold=float(selector_config["challenge_novelty_threshold"]),
        diversity_threshold=float(selector_config["pairwise_diversity_threshold"]),
        scan_limit=max(top_k, min(len(scores), int(selector_config["candidate_scan_limit"]))),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / "library.fasta"
    top_path = output_dir / "top.fasta"
    scores_path = output_dir / "scores.csv"
    manifest_path = output_dir / "manifest.json"

    write_fasta(sequences, library_path, prefix="amp")
    write_fasta((item.score.sequence for item in selection.selected), top_path, prefix="rank")
    _write_scores(scores_path, scores, selection)

    input_records = [
        _input_record("config", config_path, project_root=project_root),
        _input_record("reference", reference_path, project_root=project_root),
    ]
    if candidate_fasta is not None:
        input_records.append(
            _input_record("candidate_fasta", candidate_fasta, project_root=project_root)
        )
    if learned_checkpoint_record is not None:
        input_records.append(
            {
                key: value
                for key, value in learned_checkpoint_record.items()
                if key in {"role", "path", "sha256"}
            }
        )
    input_records.extend(
        _input_record("external_scores", oracle.path, project_root=project_root)
        for oracle in external_oracles
    )

    manifest = {
        "schema_version": 1,
        "pipeline_version": __version__,
        "git_commit": git_commit(project_root),
        "track": config["track"],
        "seed": seed,
        "library_size": len(sequences),
        "top_size": len(selection.selected),
        "generator": candidate_source,
        "oracle": scorer.name,
        "oracle_sources": list(scorer.sources),
        "learned_checkpoint": learned_checkpoint_record,
        "config": {
            "path": _recorded_path(config_path, project_root=project_root),
            "sha256": sha256_file(config_path),
        },
        "reference": {
            "path": _recorded_path(reference_path, project_root=project_root),
            "sha256": sha256_file(reference_path),
            "sequence_count": len(references.exact),
        },
        "inputs": input_records,
        "selection": {
            "scanned": selection.scanned,
            "rejection_counts": selection.rejection_counts,
            "novelty_threshold": selector_config["challenge_novelty_threshold"],
            "pairwise_diversity_threshold": selector_config["pairwise_diversity_threshold"],
        },
        "external_scores": [
            {
                "path": _recorded_path(oracle.path, project_root=project_root),
                "sha256": sha256_file(oracle.path),
            }
            for oracle in external_oracles
        ],
        "outputs": {
            "library.fasta": sha256_file(library_path),
            "top.fasta": sha256_file(top_path),
            "scores.csv": sha256_file(scores_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return PipelineResult(
        output_dir=output_dir,
        library_path=library_path,
        top_path=top_path,
        scores_path=scores_path,
        manifest_path=manifest_path,
        selection=selection,
    )
