# Training handoff

Status: **ready for implementation/execution of model training; training not run**

Snapshot date: 2026-09-03

## Rebuild and verify

From a clean checkout with Python 3.12 and `uv`:

```bash
uv sync --locked
uv run amp data fetch dramp-general-2026-09-03
uv run amp data prepare
uv run amp train preflight
```

The raw workbook is accepted only when its SHA-256 matches
`1f376e89596a01e0a186258ada2d2c29bcde2e520abfe2969a618b824958f25e`.
Raw and processed tables are intentionally Git-ignored. The processed manifest
records hashes for every generated artifact and the full row funnel.

## Frozen handoff

| Item | Value |
|---|---:|
| Eligible unique sequences | 929 |
| MIC measurements | 5,012 |
| HC50 measurements | 130 |
| Sequence clusters | 417 |
| Largest cluster | 40 |
| Folds | 5 |
| MIC train / calibration / test | 2,972 / 1,053 / 987 |
| HC50 train / calibration / test | 80 / 25 / 25 |
| Free/free-terminus MIC rows | 1,634 |
| Free/free-terminus HC50 rows | 62 |

Generated files under `data/processed/dramp-oracle-v1/`:

- `mic_measurements.csv`: organism-aware, interval/censor-aware MIC target;
- `hc50_measurements.csv`: interval/censor-aware HC50 target;
- `sequence_splits.csv`: stable sequence IDs, clusters, folds, and roles;
- `sequence_features.csv`: deterministic physicochemical covariates;
- `quarantine.csv`: exclusions and parser/taxonomy audit;
- `manifest.json`: source, config, counts, policy, and artifact hashes;
- `preflight.json`: integrity and split-boundary report.

The five-fold measurement counts are:

| Fold | MIC | HC50 | Sequences | Role |
|---:|---:|---:|---:|---|
| 0 | 987 | 25 | 198 | test |
| 1 | 1,053 | 25 | 186 | calibration |
| 2 | 1,006 | 25 | 194 | train |
| 3 | 863 | 32 | 163 | train |
| 4 | 1,103 | 23 | 188 | train |

## Model configuration

`configs/oracle_train.json` pins `facebook/esm2_t12_35M_UR50D` at Git revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`. The backbone remains frozen; the
planned heads are small MLPs for interval-censored log2-micromolar MIC and HC50,
with organism embeddings for MIC and explicit terminal-state covariates.

Training dependencies are isolated from the curation environment:

```bash
uv sync --locked --extra train
```

Installing that extra does not start training. There is intentionally no
`amp train run` command in this handoff: execution begins only after the loss,
metric logging, checkpoint, and resume behavior receive tests and an explicit
training invocation is added.

## Known limits before making model claims

- HC50 is the bottleneck: only 130 rows, and only 62 have free termini at both
  ends. Use strong regularization, report uncertainty, and do not claim a precise
  safety regressor from this source alone.
- The split is deterministic single-linkage global-edit clustering at a 0.70
  threshold with 0.80 minimum coverage. Preflight exhaustively compared 122,480
  eligible cross-fold pairs: there were zero threshold violations and the
  maximum identity was 0.697674. This is still not equivalent to MMseqs2
  local-alignment clustering or a 40%-identity family holdout.
- This is one-source data. Assay/source shift cannot be measured until an
  independently licensed source is added.
- Taxonomy normalization is conservative. Low-confidence organism strings,
  fungal entries, unsupported modifications, and parser failures remain in the
  quarantine audit rather than being guessed into labels.
- APEX is available as an external inference ensemble only; it contributed no
  training labels to this handoff.

The first scientifically defensible next step is a non-neural physicochemical
baseline followed by the frozen-ESM2 small heads, evaluated on the fixed test
clusters and calibrated only on the calibration clusters.
