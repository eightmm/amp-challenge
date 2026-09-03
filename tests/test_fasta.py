from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amp_challenge.fasta import FastaFormatError, read_fasta, write_fasta


class FastaTests(unittest.TestCase):
    def test_round_trip_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "items.fasta"
            write_fasta(["ACDEFGHI", "KKLLAAGG"], path)
            self.assertEqual(
                read_fasta(path),
                [("seq000001", "ACDEFGHI"), ("seq000002", "KKLLAAGG")],
            )
            first = path.read_bytes()
            write_fasta(["ACDEFGHI", "KKLLAAGG"], path)
            self.assertEqual(path.read_bytes(), first)

    def test_sequence_before_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.fasta"
            path.write_text("ACDEFGHI\n", encoding="utf-8")
            with self.assertRaises(FastaFormatError):
                read_fasta(path)


if __name__ == "__main__":
    unittest.main()
