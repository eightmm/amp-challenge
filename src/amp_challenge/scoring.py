from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .physchem import PhyschemFeatures, describe
from .utils import sha256_file


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _gaussian(value: float, center: float, scale: float) -> float:
    return math.exp(-0.5 * ((value - center) / scale) ** 2)


@dataclass(frozen=True, slots=True)
class OraclePrediction:
    activity: float
    toxicity: float
    uncertainty: float
    source: str

    def __post_init__(self) -> None:
        for field_name in ("activity", "toxicity", "uncertainty"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1], got {value}")


class Oracle(Protocol):
    name: str

    def predict(self, sequence: str) -> OraclePrediction | None: ...


class PhyschemOracle:
    """Transparent infrastructure baseline; not a learned biological oracle."""

    name = "physchem-v0"

    def predict(self, sequence: str) -> OraclePrediction:
        features = describe(sequence)
        charge_term = _gaussian(features.net_charge, center=5.5, scale=3.0)
        hydrophobic_term = _gaussian(features.hydrophobic_fraction, center=0.47, scale=0.16)
        moment_term = _clamp(features.hydrophobic_moment / 1.7)
        length_term = _gaussian(features.length, center=21.0, scale=8.0)
        entropy_term = _clamp((features.sequence_entropy - 0.45) / 0.45)
        activity = _clamp(
            0.28 * charge_term
            + 0.25 * hydrophobic_term
            + 0.20 * moment_term
            + 0.17 * length_term
            + 0.10 * entropy_term
        )

        excessive_hydrophobicity = _clamp((features.hydrophobic_fraction - 0.50) / 0.25)
        excessive_charge = _clamp((features.charge_density - 0.32) / 0.25)
        aromaticity = _clamp((features.aromatic_fraction - 0.12) / 0.22)
        aggregation = _clamp((features.max_hydrophobic_run - 3.0) / 4.0)
        toxicity = _clamp(
            0.35 * excessive_hydrophobicity
            + 0.25 * excessive_charge
            + 0.20 * aromaticity
            + 0.20 * aggregation
        )

        envelope_distance = max(
            0.0,
            abs(features.net_charge - 5.0) / 8.0 - 0.35,
            abs(features.hydrophobic_fraction - 0.47) / 0.42 - 0.35,
            abs(features.length - 21.0) / 25.0 - 0.35,
        )
        uncertainty = _clamp(0.35 + envelope_distance)
        return OraclePrediction(activity, toxicity, uncertainty, self.name)


class CSVOracle:
    """Adapter for normalized external predictions."""

    def __init__(self, path: Path, *, name: str | None = None) -> None:
        self.path = path
        self.name = name or f"csv:{path.stem}"
        self._predictions: dict[str, OraclePrediction] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sequence", "activity", "toxicity", "uncertainty"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row_number, row in enumerate(reader, 2):
                sequence = row["sequence"].strip().upper()
                if sequence in self._predictions:
                    raise ValueError(f"{path}:{row_number}: duplicate sequence {sequence}")
                self._predictions[sequence] = OraclePrediction(
                    activity=float(row["activity"]),
                    toxicity=float(row["toxicity"]),
                    uncertainty=float(row["uncertainty"]),
                    source=self.name,
                )

    def predict(self, sequence: str) -> OraclePrediction | None:
        return self._predictions.get(sequence)

    def require_coverage(self, sequences: Iterable[str]) -> None:
        """Fail closed when a candidate has no prediction in this artifact."""
        missing = sorted(set(sequences) - self._predictions.keys())
        if missing:
            examples = ", ".join(missing[:3])
            raise ValueError(
                f"{self.path}: missing predictions for {len(missing)} candidate sequence(s); "
                f"examples: {examples}"
            )


class LearnedLinearOracle:
    """Adapter for the self-contained JSON linear-oracle checkpoint."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        use_activity: bool = True,
        use_toxicity: bool = True,
    ) -> None:
        if not use_activity and not use_toxicity:
            raise ValueError("learned oracle must enable activity, toxicity, or both")
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                "learned checkpoint sha256 must be 64 lowercase hexadecimal characters"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"learned checkpoint hash mismatch for {path}: {actual_sha256} != {expected_sha256}"
            )

        # Import lazily so the stable scoring contract does not create a cycle with
        # the training/checkpoint implementation.
        from .linear_oracle import LinearOracle

        self.path = path
        self.checkpoint_sha256 = actual_sha256
        self._model = LinearOracle.from_checkpoint(path)
        self.name = self._model.name
        self.use_activity = use_activity
        self.use_toxicity = use_toxicity

    def predict(self, sequence: str) -> OraclePrediction:
        activity, toxicity, activity_uncertainty, toxicity_uncertainty = (
            self._model.predict_components(sequence)
        )
        enabled_uncertainties = []
        if self.use_activity:
            enabled_uncertainties.append(activity_uncertainty)
        if self.use_toxicity:
            enabled_uncertainties.append(toxicity_uncertainty)
        return OraclePrediction(
            activity=activity,
            toxicity=toxicity,
            uncertainty=max(enabled_uncertainties),
            source=self.name,
        )


@dataclass(frozen=True, slots=True)
class WeightedOracle:
    oracle: Oracle
    weight: float = 1.0
    use_activity: bool = True
    use_toxicity: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("oracle weight must be finite and positive")
        if not self.use_activity and not self.use_toxicity:
            raise ValueError("oracle must contribute activity, toxicity, or both")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    sequence: str
    activity: float
    toxicity: float
    uncertainty: float
    utility: float
    oracle_count: int
    oracle_sources: tuple[str, ...]
    features: PhyschemFeatures

    def as_flat_dict(self) -> dict[str, str | int | float]:
        output: dict[str, str | int | float] = {
            "sequence": self.sequence,
            "activity": self.activity,
            "toxicity": self.toxicity,
            "uncertainty": self.uncertainty,
            "utility": self.utility,
            "oracle_count": self.oracle_count,
            "oracle_sources": "|".join(self.oracle_sources),
        }
        output.update(asdict(self.features))
        return output


class EnsembleScorer:
    def __init__(
        self,
        members: list[WeightedOracle] | None = None,
        *,
        activity_weight: float = 1.0,
        toxicity_weight: float = 0.65,
        uncertainty_weight: float = 0.25,
        missing_member_penalty: float = 0.10,
    ) -> None:
        self.members = members or [WeightedOracle(PhyschemOracle())]
        self.activity_weight = activity_weight
        self.toxicity_weight = toxicity_weight
        self.uncertainty_weight = uncertainty_weight
        self.missing_member_penalty = missing_member_penalty

    @property
    def name(self) -> str:
        return "+".join(member.oracle.name for member in self.members)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(member.oracle.name for member in self.members)

    def score(self, sequence: str) -> CandidateScore:
        predictions: list[tuple[OraclePrediction, WeightedOracle]] = []
        missing = 0
        for member in self.members:
            prediction = member.oracle.predict(sequence)
            if prediction is None:
                missing += 1
                continue
            predictions.append((prediction, member))
        if not predictions:
            raise ValueError(f"no oracle produced a prediction for {sequence}")

        activity_predictions = [
            (prediction, member.weight) for prediction, member in predictions if member.use_activity
        ]
        toxicity_predictions = [
            (prediction, member.weight) for prediction, member in predictions if member.use_toxicity
        ]
        if not activity_predictions:
            raise ValueError(f"no oracle produced an activity prediction for {sequence}")
        if not toxicity_predictions:
            raise ValueError(f"no oracle produced a toxicity prediction for {sequence}")

        activity_total = sum(weight for _, weight in activity_predictions)
        toxicity_total = sum(weight for _, weight in toxicity_predictions)
        activity = (
            sum(prediction.activity * weight for prediction, weight in activity_predictions)
            / activity_total
        )
        toxicity = (
            sum(prediction.toxicity * weight for prediction, weight in toxicity_predictions)
            / toxicity_total
        )
        # Adding an oracle must never make a candidate look more certain merely
        # because uncertainties were averaged. Keep the most conservative stated
        # uncertainty and add disagreement for both biological heads.
        stated_uncertainty = max(prediction.uncertainty for prediction, _ in predictions)
        activity_values = [prediction.activity for prediction, _ in activity_predictions]
        toxicity_values = [prediction.toxicity for prediction, _ in toxicity_predictions]
        activity_disagreement = (
            statistics.pstdev(activity_values) if len(activity_values) > 1 else 0.0
        )
        toxicity_disagreement = (
            statistics.pstdev(toxicity_values) if len(toxicity_values) > 1 else 0.0
        )
        uncertainty = _clamp(
            stated_uncertainty
            + max(activity_disagreement, toxicity_disagreement)
            + missing * self.missing_member_penalty
        )
        features = describe(sequence)
        synthesis_penalty = _clamp(
            0.20 * features.cysteine_count
            + 0.10 * max(0, features.max_homopolymer_run - 2)
            + 0.08 * max(0, features.max_hydrophobic_run - 4)
        )
        utility = (
            self.activity_weight * activity
            - self.toxicity_weight * toxicity
            - self.uncertainty_weight * uncertainty
            - synthesis_penalty
        )
        return CandidateScore(
            sequence=sequence,
            activity=activity,
            toxicity=toxicity,
            uncertainty=uncertainty,
            utility=utility,
            oracle_count=len(predictions),
            oracle_sources=tuple(prediction.source for prediction, _ in predictions),
            features=features,
        )
