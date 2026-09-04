from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TRAIN_EXTRA = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(TRAIN_EXTRA, "requires the train extra")
class EsmOracleTests(unittest.TestCase):
    def test_nested_benchmark_is_deterministic_and_uses_all_outer_folds(self) -> None:
        import numpy as np

        from amp_challenge.esm_oracle import train_esm_mic16_oracle
        from amp_challenge.utils import atomic_write_json, sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mic = root / "mic.csv"
            fields = [
                "sequence",
                "fold",
                "organism_name",
                "gram",
                "n_terminal",
                "c_terminal",
                "active_le_16um",
            ]
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
            sequences: list[str] = []
            with mic.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for fold in range(5):
                    for index in range(12):
                        signal = index % 2
                        sequence = ("KKLL" if signal else "DDEE") + "".join(
                            alphabet[(fold * 12 + index + step) % 20] for step in range(8)
                        )
                        sequences.append(sequence)
                        for organism, gram in (
                            ("Escherichia coli", "negative"),
                            ("Staphylococcus aureus", "positive"),
                        ):
                            writer.writerow(
                                {
                                    "sequence": sequence,
                                    "fold": fold,
                                    "organism_name": organism,
                                    "gram": gram,
                                    "n_terminal": "free",
                                    "c_terminal": "free",
                                    "active_le_16um": signal,
                                }
                            )

            embedding_path = root / "embeddings.npz"
            unique = sorted(set(sequences))
            matrix = np.asarray(
                [[float(sequence.startswith("KKLL")), len(sequence) / 20.0] for sequence in unique],
                dtype=np.float32,
            )
            np.savez_compressed(
                embedding_path,
                sequences=np.asarray(unique),
                embeddings=matrix,
            )
            digest = (
                __import__("hashlib").sha256(("\n".join(unique) + "\n").encode("ascii")).hexdigest()
            )
            atomic_write_json(
                embedding_path.with_suffix(".manifest.json"),
                {
                    "schema_version": 1,
                    "model_name": "synthetic-esm",
                    "revision": "test",
                    "pooling": "test",
                    "sequence_count": len(unique),
                    "sequence_sha256": digest,
                    "embedding_width": 2,
                    "dtype": "float32",
                    "artifact": {
                        "filename": embedding_path.name,
                        "sha256": sha256_file(embedding_path),
                    },
                    "local_model_files": {},
                },
            )
            dataset_manifest = root / "dataset.json"
            oracle_config = root / "config.json"
            atomic_write_json(dataset_manifest, {"dataset": "synthetic"})
            atomic_write_json(oracle_config, {"config": "synthetic"})

            outputs = []
            for run in ("first", "second"):
                checkpoint = root / f"{run}.json"
                report = root / f"{run}-report.json"
                payload, metrics = train_esm_mic16_oracle(
                    mic_csv=mic,
                    embeddings_path=embedding_path,
                    checkpoint_path=checkpoint,
                    report_path=report,
                    dataset_manifest_path=dataset_manifest,
                    oracle_config_path=oracle_config,
                    seed=11,
                    c_values=(0.1, 1.0),
                    rare_organism_min_rows=2,
                )
                outputs.append((payload, metrics, json.loads(checkpoint.read_text())))
            self.assertEqual(outputs[0], outputs[1])
            candidate = outputs[0][1]["families"]["esm2-physchem-context"]
            self.assertEqual(
                [fold["outer_fold"] for fold in candidate["folds"]],
                [0, 1, 2, 3, 4],
            )
            self.assertGreater(candidate["pooled_measurement_metrics"]["auroc"], 0.95)
            self.assertEqual(len(outputs[0][0]["members"]), 5)


if __name__ == "__main__":
    unittest.main()
