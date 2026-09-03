from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .constants import STANDARD_AMINO_ACIDS

# Kyte-Doolittle residue hydropathy.
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

HYDROPHOBIC = frozenset("AILMFWVY")
AROMATIC = frozenset("FWY")
POSITIVE = frozenset("KR")
NEGATIVE = frozenset("DE")

SIDECHAIN_PKA_POSITIVE = {"K": 10.5, "R": 12.5, "H": 6.0}
SIDECHAIN_PKA_NEGATIVE = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
N_TERMINUS_PKA = 9.69
C_TERMINUS_PKA = 2.34


@dataclass(frozen=True, slots=True)
class PhyschemFeatures:
    length: int
    net_charge: float
    charge_density: float
    mean_hydropathy: float
    hydrophobic_fraction: float
    aromatic_fraction: float
    hydrophobic_moment: float
    isoelectric_point: float
    sequence_entropy: float
    max_hydrophobic_run: int
    max_homopolymer_run: int
    cysteine_count: int
    proline_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence is empty")
    invalid = set(sequence) - STANDARD_AMINO_ACIDS
    if invalid:
        raise ValueError(f"sequence contains non-standard residues: {sorted(invalid)}")


def net_charge(sequence: str, ph: float = 7.4) -> float:
    validate_sequence(sequence)
    positive = 1.0 / (1.0 + 10.0 ** (ph - N_TERMINUS_PKA))
    negative = 1.0 / (1.0 + 10.0 ** (C_TERMINUS_PKA - ph))
    for residue, pka in SIDECHAIN_PKA_POSITIVE.items():
        positive += sequence.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in SIDECHAIN_PKA_NEGATIVE.items():
        negative += sequence.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def isoelectric_point(sequence: str) -> float:
    low, high = 0.0, 14.0
    for _ in range(48):
        midpoint = (low + high) / 2.0
        if net_charge(sequence, midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def hydrophobic_moment(sequence: str, angle_degrees: float = 100.0) -> float:
    validate_sequence(sequence)
    angle = math.radians(angle_degrees)
    x = sum(HYDROPATHY[residue] * math.cos(index * angle) for index, residue in enumerate(sequence))
    y = sum(HYDROPATHY[residue] * math.sin(index * angle) for index, residue in enumerate(sequence))
    return math.hypot(x, y) / len(sequence)


def normalized_entropy(sequence: str) -> float:
    counts = {residue: sequence.count(residue) for residue in set(sequence)}
    entropy = -sum(
        (count / len(sequence)) * math.log(count / len(sequence)) for count in counts.values()
    )
    return entropy / math.log(20.0)


def longest_run(sequence: str, allowed: frozenset[str] | None = None) -> int:
    best = current = 0
    previous: str | None = None
    for residue in sequence:
        if allowed is None:
            current = current + 1 if residue == previous else 1
        else:
            current = current + 1 if residue in allowed else 0
        best = max(best, current)
        previous = residue
    return best


def describe(sequence: str, ph: float = 7.4) -> PhyschemFeatures:
    validate_sequence(sequence)
    length = len(sequence)
    charge = net_charge(sequence, ph)
    return PhyschemFeatures(
        length=length,
        net_charge=charge,
        charge_density=charge / length,
        mean_hydropathy=sum(HYDROPATHY[residue] for residue in sequence) / length,
        hydrophobic_fraction=sum(residue in HYDROPHOBIC for residue in sequence) / length,
        aromatic_fraction=sum(residue in AROMATIC for residue in sequence) / length,
        hydrophobic_moment=hydrophobic_moment(sequence),
        isoelectric_point=isoelectric_point(sequence),
        sequence_entropy=normalized_entropy(sequence),
        max_hydrophobic_run=longest_run(sequence, HYDROPHOBIC),
        max_homopolymer_run=longest_run(sequence),
        cysteine_count=sequence.count("C"),
        proline_fraction=sequence.count("P") / length,
    )
