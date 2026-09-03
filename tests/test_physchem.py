from __future__ import annotations

import unittest

from amp_challenge.physchem import describe, net_charge


class PhyschemTests(unittest.TestCase):
    def test_cationic_sequence_has_positive_charge(self) -> None:
        self.assertGreater(net_charge("KKRLLAAG"), 2.0)

    def test_feature_ranges(self) -> None:
        features = describe("KWKLFKKIGAVLKVL")
        self.assertEqual(features.length, 15)
        self.assertGreaterEqual(features.hydrophobic_fraction, 0.0)
        self.assertLessEqual(features.hydrophobic_fraction, 1.0)
        self.assertGreaterEqual(features.sequence_entropy, 0.0)
        self.assertLessEqual(features.sequence_entropy, 1.0)


if __name__ == "__main__":
    unittest.main()
