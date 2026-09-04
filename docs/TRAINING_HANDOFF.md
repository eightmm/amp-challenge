# Training handoff

Status: **handoff verified; dependency-free baseline and frozen-ESM2 MIC16 v2 executed**

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
| MIC train / calibration / diagnostic | 2,972 / 1,053 / 987 |
| HC50 train / calibration / diagnostic | 80 / 25 / 25 |
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
| 0 | 987 | 25 | 198 | development diagnostic (artifact label: `test`); used in release-head review |
| 1 | 1,053 | 25 | 186 | calibration |
| 2 | 1,006 | 25 | 194 | train |
| 3 | 863 | 32 | 163 | train |
| 4 | 1,103 | 23 | 188 | train |

## Executed baseline and frozen-ESM2 successor

The first executable baseline is trained with:

```bash
uv run amp train linear --execute
```

It fits deterministic logistic-ridge ensembles on the fixed cluster folds and
writes `checkpoints/linear-physchem-v1.json`. Fold 2--3 train the model-selection
candidates, fold 4 chooses regularization, folds 2--4 are refit, fold 1 calibrates,
and fold 0 was initially evaluated once as a diagnostic. Its results then
informed the manual release decision to disable the weak activity head and retain
the limited HC50 head only in a safety blend. Fold 0 is therefore development
evidence, not an untouched or unbiased final test. No predeclared numeric gate or
same-fold APEX/HydrAMP/physchem comparison was completed, and no formal promotion
claim is made. Exact metrics and caveats are in `MODEL_CARD.md`.

The first frozen-ESM2 successor is now executed:

`configs/oracle_train.json` pins `facebook/esm2_t12_35M_UR50D` at Git revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`. The backbone remains frozen; the
v2 uses masked-mean ESM2 embeddings plus physicochemical, organism, Gram, and
terminal-state covariates in a calibrated logistic-ridge MIC<=16-uM ensemble.
The planned interval-censored MIC and HC50 heads remain future work.

Training dependencies are isolated from the curation environment:

```bash
uv sync --locked --extra train
```

Installing that extra does not start training. `uv run amp train esm --execute`
creates the content-addressed embedding cache, nested-CV report, and
`checkpoints/esm2-mic16-v2.json`. Metrics and caveats are in
`MODEL_CARD_ESM2_V2.md`.

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

The next scientifically defensible step is an independent-source or newly reserved
family/temporal holdout, followed by interval/ordinal MIC heads and APEX comparison.
Fold 0 cannot serve as a final-test fold because its diagnostics already informed
release decisions.
