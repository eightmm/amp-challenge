from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .scoring import CandidateScore
from .similarity import ReferenceIndex, ratio


class SelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    rank: int
    score: CandidateScore
    max_reference_ratio: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected: list[SelectedCandidate]
    rejection_counts: dict[str, int]
    scanned: int


def _passes_top_synthesis_gate(score: CandidateScore) -> bool:
    features = score.features
    return (
        features.cysteine_count == 0
        and features.max_homopolymer_run <= 2
        and features.max_hydrophobic_run <= 5
        and 1.5 <= features.net_charge <= 10.5
        and 0.25 <= features.hydrophobic_fraction <= 0.68
        and features.sequence_entropy >= 0.58
    )


def select_top(
    scores: list[CandidateScore],
    references: ReferenceIndex,
    *,
    top_k: int,
    novelty_threshold: float = 0.8,
    diversity_threshold: float = 0.55,
    scan_limit: int = 10_000,
) -> SelectionResult:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 < novelty_threshold <= 1.0:
        raise ValueError("novelty_threshold must be in (0, 1]")
    if not 0.0 < diversity_threshold <= 1.0:
        raise ValueError("diversity_threshold must be in (0, 1]")

    ranked = sorted(scores, key=lambda item: (-item.utility, item.sequence))
    selected: list[SelectedCandidate] = []
    rejected: Counter[str] = Counter()
    scanned = 0

    for candidate in ranked[:scan_limit]:
        scanned += 1
        if not _passes_top_synthesis_gate(candidate):
            rejected["synthesis"] += 1
            continue
        maximum_reference_ratio = references.max_ratio(
            candidate.sequence, threshold=novelty_threshold
        )
        if maximum_reference_ratio > novelty_threshold:
            rejected["reference_novelty"] += 1
            continue
        if any(
            ratio(candidate.sequence, existing.score.sequence) > diversity_threshold
            for existing in selected
        ):
            rejected["pairwise_diversity"] += 1
            continue
        selected.append(
            SelectedCandidate(
                rank=len(selected) + 1,
                score=candidate,
                max_reference_ratio=maximum_reference_ratio,
            )
        )
        if len(selected) == top_k:
            return SelectionResult(selected, dict(rejected), scanned)

    raise SelectionError(
        f"selected only {len(selected)}/{top_k} after scanning {scanned} candidates; "
        f"rejections={dict(rejected)}"
    )
