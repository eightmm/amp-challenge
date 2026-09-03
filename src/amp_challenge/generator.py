from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .constants import MAX_LENGTH, MIN_LENGTH
from .fasta import read_sequences
from .physchem import describe


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    name: str = "heuristic-v0"
    min_length: int = 12
    max_length: int = 32


HYDROPHOBIC_CHOICES = tuple("AILVFMWY")
HYDROPHOBIC_WEIGHTS = (17, 18, 13, 10, 8, 4, 3, 2)
POSITIVE_CHOICES = tuple("KRH")
POSITIVE_WEIGHTS = (58, 37, 5)
POLAR_CHOICES = tuple("GSTNQAP")
POLAR_WEIGHTS = (19, 17, 14, 13, 12, 15, 10)
GENERAL_CHOICES = tuple("ACDEFGHIKLMNPQRSTVWY")
GENERAL_WEIGHTS = (10, 1, 1, 1, 6, 8, 2, 7, 14, 13, 3, 4, 2, 8, 7, 6, 4, 8, 5, 2)


def _weighted_residue(
    rng: random.Random, residues: tuple[str, ...], weights: tuple[int, ...]
) -> str:
    return rng.choices(residues, weights=weights, k=1)[0]


def _draw_candidate(rng: random.Random, min_length: int, max_length: int) -> str:
    length = int(round(rng.triangular(min_length, max_length, 20)))
    phase = rng.randrange(7)
    sequence: list[str] = []
    for index in range(length):
        helical_position = (index + phase) % 7
        chance = rng.random()
        if helical_position in {0, 3} and chance < 0.82:
            residue = _weighted_residue(rng, HYDROPHOBIC_CHOICES, HYDROPHOBIC_WEIGHTS)
        elif helical_position in {1, 4} and chance < 0.86:
            residue = _weighted_residue(rng, POSITIVE_CHOICES, POSITIVE_WEIGHTS)
        elif chance < 0.55:
            residue = _weighted_residue(rng, POLAR_CHOICES, POLAR_WEIGHTS)
        else:
            residue = _weighted_residue(rng, GENERAL_CHOICES, GENERAL_WEIGHTS)
        sequence.append(residue)
    return "".join(sequence)


def _plausible_smoke_candidate(sequence: str) -> bool:
    features = describe(sequence)
    return (
        1.5 <= features.net_charge <= 11.0
        and 0.25 <= features.hydrophobic_fraction <= 0.72
        and features.max_homopolymer_run <= 2
        and features.max_hydrophobic_run <= 6
        and features.sequence_entropy >= 0.55
    )


def generate_heuristic(
    count: int,
    *,
    seed: int,
    min_length: int,
    max_length: int,
    forbidden: frozenset[str] = frozenset(),
) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    if not (MIN_LENGTH <= min_length <= max_length <= MAX_LENGTH):
        raise ValueError(f"length range must be within {MIN_LENGTH}..{MAX_LENGTH}")

    rng = random.Random(seed)
    sequences: list[str] = []
    seen: set[str] = set()
    max_attempts = max(10_000, count * 100)
    for _ in range(max_attempts):
        candidate = _draw_candidate(rng, min_length, max_length)
        if candidate in seen or candidate in forbidden:
            continue
        if not _plausible_smoke_candidate(candidate):
            continue
        seen.add(candidate)
        sequences.append(candidate)
        if len(sequences) == count:
            return sequences
    raise GenerationError(
        f"generated only {len(sequences)} unique valid candidates after {max_attempts} attempts"
    )


def load_candidates(path: Path, *, forbidden: frozenset[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for sequence in read_sequences(path):
        if sequence in seen or sequence in forbidden:
            continue
        seen.add(sequence)
        output.append(sequence)
    return output


def take_candidates(candidates: Iterable[str], count: int) -> list[str]:
    selected: list[str] = []
    for sequence in candidates:
        selected.append(sequence)
        if len(selected) == count:
            return selected
    raise GenerationError(
        f"candidate source has only {len(selected)} usable sequences; need {count}"
    )
