#!/usr/bin/env python3
from __future__ import annotations

import sys

from amp_challenge.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["validate", "--run-dir", "generate_broad_spectrum"])
    main()
