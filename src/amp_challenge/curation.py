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
# Average-mass deltas, consistent with the average residue masses above.
N_ACETYL_MASS_DELTA = 42.03668
C_AMIDE_MASS_DELTA = -0.98476
CENSORED_VALUE = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|≤|≥)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def canonicalize_sequence(raw: str) -> str:
    sequence = "".join(raw.split()).upper()
    invalid = set(sequence) - STANDARD_AMINO_ACIDS
    if not sequence or invalid:
        raise ValueError(f"not a canonical standard-amino-acid sequence: {sorted(invalid)}")
    return sequence


def molecular_weight(
    sequence: str,
    *,
    n_terminal: str = "free",
    c_terminal: str = "free",
) -> float:
    sequence = canonicalize_sequence(sequence)
    if n_terminal not in {"free", "acetylated"}:
        raise ValueError(f"unsupported N-terminal state {n_terminal!r}")
    if c_terminal not in {"free", "amidated"}:
        raise ValueError(f"unsupported C-terminal state {c_terminal!r}")
    mass = WATER_MASS + sum(RESIDUE_MASS[residue] for residue in sequence)
    if n_terminal == "acetylated":
        mass += N_ACETYL_MASS_DELTA
    if c_terminal == "amidated":
        mass += C_AMIDE_MASS_DELTA
    return mass


def micrograms_per_ml_to_micromolar(value: float, sequence: str) -> float:
    if value <= 0:
        raise ValueError("concentration must be positive")
    return value * 1000.0 / molecular_weight(sequence)


def normalize_concentration_unit(raw: str) -> str:
    compact = raw.strip().replace("μ", "u").replace("µ", "u").replace("−", "-").replace("·", "")
    compact = re.sub(r"\s+", "", compact).lower()
    aliases = {
        "um": "uM",
        "microm": "uM",
        "umol/l": "uM",
        "umol/liter": "uM",
        "mm": "mM",
        "mmol/l": "mM",
        "nm": "nM",
        "nmol/l": "nM",
        "pmol/ml": "pmol/mL",
        "ug/ml": "ug/mL",
        "mcg/ml": "ug/mL",
        "mg/l": "mg/L",
        "ng/ml": "ng/mL",
        "mg/ml": "mg/mL",
    }
    try:
        return aliases[compact]
    except KeyError as error:
        raise ValueError(f"unsupported concentration unit {raw!r}") from error


def concentration_to_micromolar(
    value: float,
    unit: str,
    sequence: str,
    *,
    n_terminal: str = "free",
    c_terminal: str = "free",
) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("concentration must be finite and positive")
    normalized = normalize_concentration_unit(unit)
    if normalized == "uM":
        return value
    if normalized == "mM":
        return value * 1000.0
    if normalized in {"nM", "pmol/mL"}:
        return value / 1000.0

    mass = molecular_weight(
        sequence,
        n_terminal=n_terminal,
        c_terminal=c_terminal,
    )
    if normalized in {"ug/mL", "mg/L"}:
        return value * 1000.0 / mass
    if normalized == "ng/mL":
        return value / mass
    if normalized == "mg/mL":
        return value * 1_000_000.0 / mass
    raise AssertionError(f"unhandled normalized unit {normalized}")


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
