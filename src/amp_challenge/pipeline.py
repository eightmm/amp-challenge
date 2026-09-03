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
from .scoring import CSVOracle, EnsembleScorer, PhyschemOracle, WeightedOracle
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
        candidate_source = {
            "kind": "fasta",
            "path": str(candidate_fasta),
            "sha256": sha256_file(candidate_fasta),
        }

    oracle_config = config["oracle"]
    members = [WeightedOracle(PhyschemOracle())]
    for score_path in external_score_paths or []:
        members.append(WeightedOracle(CSVOracle(score_path)))
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
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "sequence_count": len(references.exact),
        },
        "selection": {
            "scanned": selection.scanned,
            "rejection_counts": selection.rejection_counts,
            "novelty_threshold": selector_config["challenge_novelty_threshold"],
            "pairwise_diversity_threshold": selector_config["pairwise_diversity_threshold"],
        },
        "external_scores": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (external_score_paths or [])
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
