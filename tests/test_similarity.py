from __future__ import annotations

import unittest

from amp_challenge.similarity import ReferenceIndex, ratio


class SimilarityTests(unittest.TestCase):
    def test_identical_ratio(self) -> None:
        self.assertEqual(ratio("ACDEFGHI", "ACDEFGHI"), 1.0)

    def test_reference_threshold(self) -> None:
        index = ReferenceIndex(["ACDEFGHI", "KKLLAAGG"])
        self.assertFalse(index.passes("ACDEFGHI", threshold=0.8))
        self.assertTrue(index.passes("RRRRVVVV", threshold=0.8))


if __name__ == "__main__":
    unittest.main()
