from __future__ import annotations

import unittest
from pathlib import Path

from amp_challenge.kaggle import build_download_command
from amp_challenge.submission import build_kaggle_command


class KaggleCommandTests(unittest.TestCase):
    def test_download_command(self) -> None:
        self.assertEqual(
            build_download_command(
                competition="amp-challenge", destination=Path("data/competition")
            ),
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "amp-challenge",
                "-p",
                "data/competition",
            ],
        )

    def test_submit_command(self) -> None:
        command = build_kaggle_command(
            competition="amp-challenge",
            artifact=Path("submission/run.zip"),
            message="run",
        )
        self.assertEqual(command[0:3], ["kaggle", "competitions", "submit"])


if __name__ == "__main__":
    unittest.main()
