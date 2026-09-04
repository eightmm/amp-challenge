from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from amp_challenge.linear_oracle import FEATURE_NAMES, LinearOracle, train_linear_oracle


class LinearOracleTests(unittest.TestCase):
    def _sequence(self, task: str, fold: int, index: int, positive: bool) -> str:
        if task == "activity":
            prefix = "KKRLLAAG" if positive else "DDEEGGST"
        else:
            prefix = "STNQGDEP" if positive else "WFLIVVYK"
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        task_offset = 0 if task == "activity" else 11
        suffix = "".join(
            (
                alphabet[(fold + task_offset) % len(alphabet)],
                alphabet[(index + task_offset) % len(alphabet)],
                alphabet[(fold * 6 + index + task_offset) % len(alphabet)],
            )
        )
        return prefix + suffix

    def _write_task(
        self,
        path: Path,
        *,
        task: str,
        label_column: str,
        mutate_test_sequence: bool = False,
        invert_calibration_labels: bool = False,
    ) -> None:
        fields = ["measurement_id", "sequence", "fold", "split_role", label_column]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            measurement = 0
            for fold in range(5):
                role = "test" if fold == 0 else "calibration" if fold == 1 else "train"
                for index in range(6):
                    positive = index % 2 == 0
                    label = not positive if invert_calibration_labels and fold == 1 else positive
                    sequence = self._sequence(task, fold, index, positive)
                    if mutate_test_sequence and fold == 0 and index == 0:
                        sequence = "ACDEFGHIKLMN"
                    measurement += 1
                    writer.writerow(
                        {
                            "measurement_id": f"m{measurement}",
                            "sequence": sequence,
                            "fold": fold,
                            "split_role": role,
                            label_column: int(label),
                        }
                    )
                    # One sequence per task has contradictory definitive measurements.
                    if fold == 2 and index == 0:
                        measurement += 1
                        writer.writerow(
                            {
                                "measurement_id": f"m{measurement}",
                                "sequence": sequence,
                                "fold": fold,
                                "split_role": role,
                                label_column: 0,
                            }
                        )
            writer.writerow(
                {
                    "measurement_id": "unknown",
                    "sequence": "ACDEFGHIK",
                    "fold": 3,
                    "split_role": "train",
                    label_column: "",
                }
            )

    def _train(
        self,
        root: Path,
        *,
        mutate_test_sequence: bool = False,
        invert_calibration_labels: bool = False,
    ) -> tuple[Path, dict]:
        mic = root / "mic.csv"
        hc50 = root / "hc50.csv"
        checkpoint = root / "oracle.json"
        self._write_task(
            mic,
            task="activity",
            label_column="active_le_16um",
            mutate_test_sequence=mutate_test_sequence,
            invert_calibration_labels=invert_calibration_labels,
        )
        self._write_task(
            hc50,
            task="safety",
            label_column="safe_ge_128um",
            mutate_test_sequence=mutate_test_sequence,
            invert_calibration_labels=invert_calibration_labels,
        )
        payload = train_linear_oracle(
            mic_csv=mic,
            hc50_csv=hc50,
            checkpoint_path=checkpoint,
            seed=17,
            ensemble_members=3,
            l2_candidates=(0.01, 0.1),
        )
        return checkpoint, payload

    def test_training_is_deterministic_and_predictions_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first_path, first = self._train(first_dir)
            second_path, second = self._train(second_dir)
            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(first["feature_names"], list(FEATURE_NAMES))
            self.assertEqual(
                first["training"]["data_summary"]["activity"][
                    "conflicting_sequences_resolved_negative"
                ],
                1,
            )

            first_oracle = LinearOracle.from_checkpoint(first_path)
            second_oracle = LinearOracle.from_checkpoint(second_path)
            self.assertEqual(first_oracle.name, "linear-physchem-v1")
            sequence = "KWKLFKKIGAVLKVL"
            self.assertEqual(
                first_oracle.predict_values(sequence), second_oracle.predict_values(sequence)
            )
            activity, toxicity, uncertainty = first_oracle.predict_values(sequence)
            for value in (activity, toxicity, uncertainty):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            components = first_oracle.predict_components(sequence)
            self.assertEqual(components[:2], (activity, toxicity))
            self.assertEqual(uncertainty, max(components[2:]))
            for value in components[2:]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            prediction = first_oracle.predict(sequence)
            self.assertEqual(
                (prediction.activity, prediction.toxicity, prediction.uncertainty),
                (activity, toxicity, uncertainty),
            )

            tampered = json.loads(first_path.read_text(encoding="utf-8"))
            tampered["tasks"]["activity"]["members"][0]["intercept"] += 1.0
            first_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_id"):
                LinearOracle.from_checkpoint(first_path)

    def test_holdout_features_do_not_change_fit_or_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_dir = root / "original"
            changed_dir = root / "changed"
            original_dir.mkdir()
            changed_dir.mkdir()
            _, original = self._train(original_dir)
            _, changed = self._train(changed_dir, mutate_test_sequence=True)
            self.assertEqual(original["tasks"], changed["tasks"])
            self.assertNotEqual(
                original["provenance"]["inputs"]["mic_csv"]["sha256"],
                changed["provenance"]["inputs"]["mic_csv"]["sha256"],
            )
            for task in ("activity", "safety"):
                report = original["training"]["reports"][task]
                self.assertEqual(report["model_selection"]["train_folds"], [2, 3])
                self.assertEqual(report["model_selection"]["validation_fold"], 4)
                self.assertEqual(report["refit_folds"], [2, 3, 4])
                self.assertEqual(report["calibration_fold"], 1)
                self.assertEqual(report["diagnostic_fold"], 0)
                self.assertTrue(report["diagnostic_initially_evaluated_once_after_calibration"])
                self.assertEqual(original["tasks"][task]["scaler"]["fitted_folds"], [2, 3, 4])
                self.assertTrue(original["tasks"][task]["scaler"]["label_independent"])

    def test_calibration_labels_only_change_calibration_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_dir = root / "original"
            changed_dir = root / "changed"
            original_dir.mkdir()
            changed_dir.mkdir()
            _, original = self._train(original_dir)
            _, changed = self._train(changed_dir, invert_calibration_labels=True)
            for task in ("activity", "safety"):
                original_task = original["tasks"][task]
                changed_task = changed["tasks"][task]
                self.assertEqual(original_task["scaler"], changed_task["scaler"])
                self.assertEqual(original_task["members"], changed_task["members"])
                self.assertEqual(original_task["selected_l2"], changed_task["selected_l2"])
                self.assertNotEqual(original_task["calibration"], changed_task["calibration"])


if __name__ == "__main__":
    unittest.main()
