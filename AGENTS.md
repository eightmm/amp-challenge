# Repository instructions

Read `SPEC.md` before changing architecture or scientific defaults.

- Keep `main` as the single integration source of truth.
- Preserve the candidate-FASTA boundary between generators and oracles.
- Do not make `heuristic-v0` sound biologically validated.
- Do not enable a data source until exact version, access terms, checksum, and
  parser provenance are recorded.
- Do not replace censored MIC labels with point estimates.
- Do not use random row/sequence splits as primary model evidence.
- Any default-model change requires leakage-free metrics and calibration evidence.
- `generate` and `generate_broad_spectrum` must retain fixed defaults and
  deterministic output.
- Never invoke Kaggle submission from generation, training, CI, or automation.
- Submission must continue to require both an exact frozen run ID and an explicit
  execute flag.

Before committing, run:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uvx ruff@0.12.12 check .
uvx ruff@0.12.12 format --check .
uv run generate_broad_spectrum --n-sequences 300 --top-k 10
uv run amp validate --run-dir generate_broad_spectrum \
  --expected-library-size 300 --expected-top-size 10
```
