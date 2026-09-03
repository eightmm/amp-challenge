from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import Levenshtein


def ratio(left: str, right: str) -> float:
    return float(Levenshtein.ratio(left, right))


class ReferenceIndex:
    """Length-bucketed exact reference index for challenge Levenshtein checks."""

    def __init__(self, sequences: Iterable[str]) -> None:
        buckets: dict[int, list[str]] = defaultdict(list)
        exact: set[str] = set()
        for sequence in sequences:
            buckets[len(sequence)].append(sequence)
            exact.add(sequence)
        self._buckets = dict(buckets)
        self.exact = frozenset(exact)

    def _possible_lengths(self, sequence_length: int, threshold: float) -> Iterable[int]:
        for reference_length in self._buckets:
            upper_bound = (
                2.0 * min(sequence_length, reference_length) / (sequence_length + reference_length)
            )
            if upper_bound > threshold:
                yield reference_length

    def max_ratio(self, sequence: str, *, threshold: float | None = None) -> float:
        best = 0.0
        cutoff = 0.0 if threshold is None else threshold
        lengths = (
            self._buckets.keys()
            if threshold is None
            else self._possible_lengths(len(sequence), threshold)
        )
        for length in lengths:
            for reference in self._buckets[length]:
                score = ratio(sequence, reference)
                if score > best:
                    best = score
                if threshold is not None and best > cutoff:
                    return best
        return best

    def passes(self, sequence: str, *, threshold: float) -> bool:
        return self.max_ratio(sequence, threshold=threshold) <= threshold
