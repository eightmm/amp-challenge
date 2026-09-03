from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from amp_challenge.training_data import prepare_training_data
from amp_challenge.training_preflight import training_preflight
from amp_challenge.utils import sha256_file


class TrainingDataTests(unittest.TestCase):
    def _write_fixture(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "general_amps"
        headers = [
            "DRAMP_ID",
            "Sequence",
            "Activity",
            "Target_Organism",
            "Hemolytic_activity",
            "Linear/Cyclic/Branched",
            "N-terminal_Modification",
            "C-terminal_Modification",
            "Other_Modifications",
            "Stereochemistry",
            "Pubmed_ID",
        ]
        sheet.append(headers)
        rows = [
            [
                "D1",
                "KWKLFKKIGAVLKVL",
                "Antibacterial",
                "Gram-negative bacteria: E. coli ATCC 25922 (MIC=8 μM)",
                "HC50>128 μM against human red blood cells",
                "Linear",
                "Free",
                "Amidation",
                "None",
                "L",
                "1",
            ],
            [
                "D2",
                "ACDEFGHIKLMNPQR",
                "Antibacterial",
                "Gram-positive bacteria: S. aureus MRSA (MIC=4-8 μM)",
                "No hemolysis information or data found",
                "Linear",
                "Free",
                "Free",
                "None",
                "L",
                "2",
            ],
            [
                "D3",
                "WWRRWWRRWWRRWWR",
                "Antibacterial",
                "P. aeruginosa PA14 (MIC>64 μM)",
                "No hemolysis information or data found",
                "Linear",
                "Acetylation",
                "Free",
                "None",
                "L",
                "3",
            ],
            [
                "D4",
                "KWKLFKKIGAVLKVA",
                "Antibacterial",
                "C. albicans (MIC=2 μM)",
                "HC50=32 μM",
                "Cyclic",
                "Free",
                "Free",
                "None",
                "L",
                "4",
            ],
        ]
        for row in rows:
            sheet.append(row)
        workbook.save(path)

    def test_prepare_is_reproducible_and_preflight_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "fixture.xlsx"
            self._write_fixture(raw)
            config_dir = root / "configs"
            config_dir.mkdir()
            registry = {
                "schema_version": 1,
                "sources": {
                    "fixture": {
                        "access": "automatic",
                        "destination": "fixture.xlsx",
                        "sha256": sha256_file(raw),
                        "url": "https://example.test/fixture.xlsx",
                        "retrieved_at": "2026-09-03",
                        "license": "CC-BY-4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    }
                },
            }
            data_config = {
                "schema_version": 1,
                "dataset_id": "fixture-v1",
                "source": "fixture",
                "sheet": "general_amps",
                "output_dir": "processed",
                "sequence": {"min": 8, "max": 50},
                "measurements": {
                    "max_concentration_um": 1000000,
                    "taxonomy_confidence": ["high", "medium"],
                },
                "clustering": {
                    "method": "global_edit_single_linkage",
                    "identity_threshold": 0.8,
                    "min_coverage": 0.8,
                },
                "splits": {
                    "folds": 3,
                    "seed": 42,
                    "holdout_folds": {"test": 0, "calibration": 1},
                },
            }
            train_config = {
                "schema_version": 1,
                "dataset_id": "fixture-v1",
                "backbone": {"revision": "a" * 40, "frozen": True},
                "evaluation": {"folds": 3},
                "minimum_rows": {"mic": 3, "hc50": 1},
            }
            registry_path = config_dir / "data_sources.json"
            data_config_path = config_dir / "training_data.json"
            train_config_path = config_dir / "oracle_train.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            data_config_path.write_text(json.dumps(data_config), encoding="utf-8")
            train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
            first = root / "first"
            second = root / "second"
            prepare_training_data(
                project_root=root,
                config_path=data_config_path,
                registry_path=registry_path,
                output_dir=first,
            )
            prepare_training_data(
                project_root=root,
                config_path=data_config_path,
                registry_path=registry_path,
                output_dir=second,
            )
            for name in (
                "mic_measurements.csv",
                "hc50_measurements.csv",
                "sequence_splits.csv",
                "sequence_features.csv",
                "quarantine.csv",
                "manifest.json",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            report = training_preflight(
                dataset_dir=first,
                train_config_path=train_config_path,
            )
            self.assertTrue(report["ready"], report["errors"])
            self.assertEqual(report["tasks"]["mic"]["rows"], 3)
            self.assertEqual(report["tasks"]["hc50"]["rows"], 1)

            manifest_path = first / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["parser"]["implementation_sha256"]["curation.py"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stale_report = training_preflight(
                dataset_dir=first,
                train_config_path=train_config_path,
            )
            self.assertFalse(stale_report["ready"])
            self.assertIn(
                "dataset parser implementation mismatch: curation.py",
                stale_report["errors"],
            )


if __name__ == "__main__":
    unittest.main()
