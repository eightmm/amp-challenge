from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_LIBRARY_SIZE,
    DEFAULT_TOP_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    STANDARD_AMINO_ACIDS,
)
from .fasta import read_fasta, read_sequences
from .similarity import ReferenceIndex


@dataclass(slots=True)
class ValidationReport:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int | float | str] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def as_dict(self) -> dict:
        return asdict(self)

    def raise_for_errors(self) -> None:
        if self.errors:
            formatted = "\n".join(f"  - {message}" for message in self.errors)
            raise ValueError(f"submission validation failed:\n{formatted}")


def _validate_records(
    records: list[tuple[str, str]],
    *,
    expected_size: int,
    label: str,
    report: ValidationReport,
) -> set[str]:
    if len(records) != expected_size:
        report.error(f"{label}: expected {expected_size} records, found {len(records)}")
    headers: set[str] = set()
    sequences: set[str] = set()
    for index, (header, sequence) in enumerate(records, 1):
        if not header:
            report.error(f"{label} record {index}: empty header")
        elif header in headers:
            report.error(f"{label} record {index}: duplicate header {header!r}")
        headers.add(header)
        invalid = set(sequence) - STANDARD_AMINO_ACIDS
        if invalid:
            report.error(f"{label} record {index}: invalid residues {sorted(invalid)}")
        if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
            report.error(
                f"{label} record {index}: length {len(sequence)} outside {MIN_LENGTH}..{MAX_LENGTH}"
            )
        if sequence in sequences:
            report.error(f"{label} record {index}: duplicate sequence")
        sequences.add(sequence)
    return sequences


def validate_submission(
    *,
    library_path: Path,
    top_path: Path,
    reference_path: Path,
    expected_library_size: int = DEFAULT_LIBRARY_SIZE,
    expected_top_size: int = DEFAULT_TOP_SIZE,
    novelty_threshold: float = 0.8,
) -> ValidationReport:
    report = ValidationReport()
    library_records = read_fasta(library_path)
    top_records = read_fasta(top_path)
    full_sequences = _validate_records(
        library_records,
        expected_size=expected_library_size,
        label="library",
        report=report,
    )
    top_sequences = _validate_records(
        top_records,
        expected_size=expected_top_size,
        label="top",
        report=report,
    )

    missing_from_library = top_sequences - full_sequences
    if missing_from_library:
        report.error(f"top: {len(missing_from_library)} sequences are absent from the library")

    references = ReferenceIndex(read_sequences(reference_path))
    overlaps = full_sequences & references.exact
    if overlaps:
        report.error(f"library: {len(overlaps)} exact overlaps with challenge reference")

    maximum_ratio = 0.0
    violating = 0
    for sequence in top_sequences:
        sequence_ratio = references.max_ratio(sequence, threshold=novelty_threshold)
        maximum_ratio = max(maximum_ratio, sequence_ratio)
        if sequence_ratio > novelty_threshold:
            violating += 1
    if violating:
        report.error(
            f"top: {violating} sequences exceed reference Levenshtein ratio {novelty_threshold}"
        )

    report.metrics.update(
        {
            "library_size": len(library_records),
            "library_unique": len(full_sequences),
            "top_size": len(top_records),
            "top_unique": len(top_sequences),
            "reference_size": len(references.exact),
            "library_reference_overlap": len(overlaps),
            "top_max_reference_ratio_checked": round(maximum_ratio, 8),
            "novelty_threshold": novelty_threshold,
        }
    )
    return report
