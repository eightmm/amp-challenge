from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amp_challenge.fasta import write_fasta
from amp_challenge.pipeline import run_pipeline
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


if __name__ == "__main__":
    unittest.main()
