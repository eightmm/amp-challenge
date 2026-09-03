from __future__ import annotations

import unittest

from amp_challenge.curation import (
    canonicalize_sequence,
    micrograms_per_ml_to_micromolar,
    molecular_weight,
    parse_censored_value,
)


class CurationTests(unittest.TestCase):
    def test_sequence_and_mass(self) -> None:
        self.assertEqual(canonicalize_sequence(" acd\n"), "ACD")
        self.assertGreater(molecular_weight("ACD"), 300.0)
        self.assertGreater(micrograms_per_ml_to_micromolar(10.0, "ACD"), 1.0)

    def test_censor_parser(self) -> None:
        self.assertEqual(parse_censored_value(">= 64"), ("ge", 64.0))
        self.assertEqual(parse_censored_value("≤4"), ("le", 4.0))
        self.assertEqual(parse_censored_value("8"), ("eq", 8.0))


if __name__ == "__main__":
    unittest.main()
