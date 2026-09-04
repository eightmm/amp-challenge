from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from amp_challenge.fasta import write_fasta
from amp_challenge.pipeline import run_pipeline
from amp_challenge.scoring import PhyschemOracle
from amp_challenge.utils import sha256_file
from amp_challenge.validation import validate_submission


class PipelineTests(unittest.TestCase):
    def test_small_pipeline_is_reproducible_and_valid(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            reference = base / "reference.fasta"
            write_fasta(["VNWKKILGKIIKVVK", "GLFDVIKKVASVIGGL"], reference, prefix="ref")
            first_dir = base / "first"
            second_dir = base / "second"
            kwargs = {
                "project_root": project_root,
                "config_path": project_root / "configs/default.json",
                "reference_path": reference,
                "n_sequences": 200,
                "top_k": 10,
                "seed": 42,
            }
            run_pipeline(output_dir=first_dir, **kwargs)
            run_pipeline(output_dir=second_dir, **kwargs)
            for name in ("library.fasta", "top.fasta", "scores.csv", "manifest.json"):
                self.assertEqual((first_dir / name).read_bytes(), (second_dir / name).read_bytes())

            report = validate_submission(
                library_path=first_dir / "library.fasta",
                top_path=first_dir / "top.fasta",
                reference_path=reference,
                expected_library_size=200,
                expected_top_size=10,
            )
            self.assertTrue(report.valid, report.errors)

            manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["oracle_sources"], ["physchem-v0", "linear-physchem-v1"])
            self.assertEqual(
                [item["role"] for item in manifest["inputs"]],
                ["config", "reference", "learned_checkpoint"],
            )
            self.assertEqual(
                [item["path"] for item in manifest["inputs"]],
                [
                    "configs/default.json",
                    str(reference),
                    "checkpoints/linear-physchem-v1.json",
                ],
            )
            with (first_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
                first_score = next(csv.DictReader(handle))
            self.assertEqual(first_score["oracle_sources"], "physchem-v0|linear-physchem-v1")

    def test_external_scores_require_complete_candidate_coverage(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            reference = base / "reference.fasta"
            candidates = base / "candidates.fasta"
            predictions = base / "predictions.csv"
            write_fasta(["VNWKKILGKIIKVVK"], reference, prefix="ref")
            write_fasta(
                ["KKLLKKLLKKLL", "RRIIRRIIRRII"],
                candidates,
                prefix="candidate",
            )
            predictions.write_text(
                "sequence,activity,toxicity,uncertainty\nKKLLKKLLKKLL,0.8,0.2,0.1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing predictions for 1 candidate"):
                run_pipeline(
                    project_root=project_root,
                    config_path=project_root / "configs/default.json",
                    output_dir=base / "run",
                    reference_path=reference,
                    n_sequences=2,
                    top_k=1,
                    seed=42,
                    candidate_fasta=candidates,
                    external_score_paths=[predictions],
                )

    def test_external_activity_scores_do_not_contribute_toxicity(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            reference = base / "reference.fasta"
            candidates = base / "candidates.fasta"
            predictions = base / "activity.csv"
            write_fasta(["VNWKKILGKIIKVVK"], reference, prefix="ref")
            sequences = ["KLGAFRVMSTQK", "RVTKQLAIGMFS"]
            write_fasta(sequences, candidates, prefix="candidate")
            predictions.write_text(
                "sequence,activity,toxicity,uncertainty\n"
                "KLGAFRVMSTQK,0.8,0.99,0.1\n"
                "RVTKQLAIGMFS,0.7,0.99,0.1\n",
                encoding="utf-8",
            )
            config_value = json.loads(
                (project_root / "configs/default.json").read_text(encoding="utf-8")
            )
            config_value["oracle"]["learned_checkpoint"] = None
            config = base / "config.json"
            config.write_text(json.dumps(config_value), encoding="utf-8")
            run_dir = base / "run"
            run_pipeline(
                project_root=project_root,
                config_path=config,
                output_dir=run_dir,
                reference_path=reference,
                n_sequences=2,
                top_k=1,
                seed=42,
                candidate_fasta=candidates,
                external_activity_score_paths=[predictions],
            )
            with (run_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
                rows = {row["sequence"]: row for row in csv.DictReader(handle)}
            physchem = PhyschemOracle().predict("KLGAFRVMSTQK")
            self.assertAlmostEqual(float(rows["KLGAFRVMSTQK"]["toxicity"]), physchem.toxicity)
            self.assertAlmostEqual(
                float(rows["KLGAFRVMSTQK"]["activity"]),
                (0.5 * physchem.activity + 0.8) / 1.5,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["external_scores"][0]["contribution"], "activity_only")
            self.assertIn(
                "external_scores:activity_only",
                [record["role"] for record in manifest["inputs"]],
            )

    def test_configured_learned_checkpoint_is_hashed_loaded_and_reported(self) -> None:
        class FakeLinearOracle:
            name = "fake-linear-v1"

            def predict_components(self, sequence: str) -> tuple[float, float, float, float]:
                return 0.8, 0.15, 0.9, 0.1

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base / "oracle.json"
            checkpoint.write_text("{}\n", encoding="utf-8")
            config_value = json.loads(
                (project_root / "configs/default.json").read_text(encoding="utf-8")
            )
            config_value["oracle"]["learned_checkpoint"] = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "weight": 2.0,
                "use_activity": False,
                "use_toxicity": True,
            }
            config = base / "config.json"
            config.write_text(json.dumps(config_value), encoding="utf-8")
            reference = base / "reference.fasta"
            write_fasta(["VNWKKILGKIIKVVK"], reference, prefix="ref")
            run_dir = base / "run"

            with mock.patch(
                "amp_challenge.linear_oracle.LinearOracle.from_checkpoint",
                return_value=FakeLinearOracle(),
            ):
                run_pipeline(
                    project_root=project_root,
                    config_path=config,
                    output_dir=run_dir,
                    reference_path=reference,
                    n_sequences=20,
                    top_k=3,
                    seed=42,
                )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["oracle_sources"], ["physchem-v0", "fake-linear-v1"])
            self.assertEqual(manifest["learned_checkpoint"]["sha256"], sha256_file(checkpoint))
            self.assertFalse(manifest["learned_checkpoint"]["use_activity"])
            self.assertTrue(manifest["learned_checkpoint"]["use_toxicity"])
            self.assertIn(
                "learned_checkpoint",
                [item["role"] for item in manifest["inputs"]],
            )
            with (run_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
                first_score = next(csv.DictReader(handle))
            self.assertEqual(first_score["oracle_sources"], "physchem-v0|fake-linear-v1")
            physchem = PhyschemOracle().predict(first_score["sequence"])
            self.assertAlmostEqual(float(first_score["activity"]), physchem.activity)
            expected_toxicity = (0.5 * physchem.toxicity + 2.0 * 0.15) / 2.5
            self.assertAlmostEqual(float(first_score["toxicity"]), expected_toxicity)
            self.assertLess(float(first_score["uncertainty"]), 0.9)

    def test_learned_head_flags_require_json_booleans(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config_value = json.loads(
                (project_root / "configs/default.json").read_text(encoding="utf-8")
            )
            config_value["oracle"]["learned_checkpoint"]["use_activity"] = "false"
            config = base / "config.json"
            config.write_text(json.dumps(config_value), encoding="utf-8")
            reference = base / "reference.fasta"
            write_fasta(["VNWKKILGKIIKVVK"], reference, prefix="ref")

            with self.assertRaisesRegex(ValueError, "use_activity must be a boolean"):
                run_pipeline(
                    project_root=project_root,
                    config_path=config,
                    output_dir=base / "run",
                    reference_path=reference,
                    n_sequences=20,
                    top_k=3,
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()
