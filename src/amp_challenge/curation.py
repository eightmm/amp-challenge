from __future__ import annotations

import math
import re

from .constants import STANDARD_AMINO_ACIDS

# Average residue masses after loss of water during peptide-bond formation.
RESIDUE_MASS = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 131.1926,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
WATER_MASS = 18.01528
CENSORED_VALUE = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|≤|≥)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def canonicalize_sequence(raw: str) -> str:
    sequence = "".join(raw.split()).upper()
    invalid = set(sequence) - STANDARD_AMINO_ACIDS
    if not sequence or invalid:
        raise ValueError(f"not a canonical standard-amino-acid sequence: {sorted(invalid)}")
    return sequence


def molecular_weight(sequence: str) -> float:
    sequence = canonicalize_sequence(sequence)
    return WATER_MASS + sum(RESIDUE_MASS[residue] for residue in sequence)


def micrograms_per_ml_to_micromolar(value: float, sequence: str) -> float:
    if value <= 0:
        raise ValueError("concentration must be positive")
    return value * 1000.0 / molecular_weight(sequence)


def parse_censored_value(raw: str) -> tuple[str, float]:
    match = CENSORED_VALUE.match(raw)
    if not match:
        raise ValueError(f"cannot parse censored concentration {raw!r}")
    relation = match.group("relation") or "="
    relation_map = {"=": "eq", "<": "lt", "<=": "le", "≤": "le", ">": "gt", ">=": "ge", "≥": "ge"}
    value = float(match.group("value"))
    if value <= 0 or not math.isfinite(value):
        raise ValueError("concentration must be finite and positive")
    return relation_map[relation], value
