from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .utils import atomic_write_text


class FastaFormatError(ValueError):
    pass


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts).upper()))
            header = line[1:].strip()
            parts = []
            continue
        if header is None:
            raise FastaFormatError(f"{path}:{line_number}: sequence appears before a header")
        parts.append(line.replace(" ", ""))

    if header is not None:
        records.append((header, "".join(parts).upper()))
    if not records:
        raise FastaFormatError(f"{path}: no FASTA records found")
    return records


def read_sequences(path: Path) -> list[str]:
    return [sequence for _, sequence in read_fasta(path)]


def write_fasta(
    sequences: Iterable[str],
    path: Path,
    *,
    prefix: str = "seq",
    start: int = 1,
) -> None:
    lines: list[str] = []
    for index, sequence in enumerate(sequences, start=start):
        lines.extend((f">{prefix}{index:06d}", sequence))
    atomic_write_text(path, "\n".join(lines) + "\n")
