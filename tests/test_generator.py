from __future__ import annotations

import unittest

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.generator import generate_heuristic


class GeneratorTests(unittest.TestCase):
    def test_deterministic_and_unique(self) -> None:
        first = generate_heuristic(100, seed=42, min_length=12, max_length=28)
        second = generate_heuristic(100, seed=42, min_length=12, max_length=28)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(not (set(sequence) - STANDARD_AMINO_ACIDS) for sequence in first))


if __name__ == "__main__":
    unittest.main()
