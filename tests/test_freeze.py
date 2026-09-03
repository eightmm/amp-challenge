from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amp_challenge.fasta import write_fasta
from amp_challenge.freeze import freeze_run, verify_frozen_run
from amp_challenge.pipeline import run_pipeline
from amp_challenge.submission import submit_kaggle


class FreezeTests(unittest.TestCase):
    def test_freeze_is_idempotent_and_submit_defaults_to_dry_run(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            reference = base / "reference.fasta"
            write_fasta(["VNWKKILGKIIKVVK"], reference, prefix="ref")
            run_dir = base / "run"
            run_pipeline(
                project_root=project_root,
                config_path=project_root / "configs/default.json",
                output_dir=run_dir,
                reference_path=reference,
                n_sequences=100,
                top_k=5,
                seed=7,
            )
            first, first_artifact, _ = freeze_run(
                run_dir=run_dir,
                reference_path=reference,
                submission_dir=base / "submission",
                expected_library_size=100,
                expected_top_size=5,
            )
            first_bytes = first_artifact.read_bytes()
            second, second_artifact, _ = freeze_run(
                run_dir=run_dir,
                reference_path=reference,
                submission_dir=base / "submission",
                expected_library_size=100,
                expected_top_size=5,
            )
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first_bytes, second_artifact.read_bytes())
            verify_frozen_run(run_dir, first["run_id"])
            result = submit_kaggle(
                run_dir=run_dir,
                artifact=first_artifact,
                run_id=first["run_id"],
                message="unit test",
            )
            self.assertTrue(result.startswith("DRY RUN:"))


if __name__ == "__main__":
    unittest.main()
