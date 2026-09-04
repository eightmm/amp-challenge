from __future__ import annotations

import unittest

from amp_challenge.scoring import (
    EnsembleScorer,
    OraclePrediction,
    WeightedOracle,
)


class _FixedOracle:
    def __init__(self, name: str, activity: float, toxicity: float, uncertainty: float) -> None:
        self.name = name
        self._prediction = OraclePrediction(activity, toxicity, uncertainty, name)

    def predict(self, sequence: str) -> OraclePrediction:
        return self._prediction


class ScoringTests(unittest.TestCase):
    def test_disabled_activity_head_does_not_change_activity_or_reduce_uncertainty(self) -> None:
        activity = _FixedOracle("activity", 0.7, 0.4, 0.6)
        safety = _FixedOracle("safety", 0.1, 0.2, 0.05)
        score = EnsembleScorer(
            [
                WeightedOracle(activity),
                WeightedOracle(safety, use_activity=False, use_toxicity=True),
            ]
        ).score("KKLLKKLLKKLL")

        self.assertAlmostEqual(score.activity, 0.7)
        self.assertAlmostEqual(score.toxicity, 0.3)
        self.assertGreaterEqual(score.uncertainty, 0.6)

    def test_member_cannot_disable_both_heads(self) -> None:
        oracle = _FixedOracle("disabled", 0.5, 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "must contribute"):
            WeightedOracle(oracle, use_activity=False, use_toxicity=False)


if __name__ == "__main__":
    unittest.main()
