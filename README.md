# AMP Challenge 2027 — Oracle-First Pipeline

A reproducible, end-to-end pipeline for designing, ranking, freezing, and
submitting antimicrobial peptides. The first target is the broad-spectrum track;
the modeling priority is a calibrated **MIC + hemolysis oracle ensemble** and a
robust Top-100 selector, with the generator kept replaceable.

> Current status: the repository contains a complete deterministic submission
> path and a deliberately simple physicochemical smoke baseline. It is not yet a
> competitive biological model. AMP-Diffusion/HydrAMP candidate import and
> learned oracles are the next milestones.

## Why this shape

The challenge evaluates a 50,000-sequence library and a ranked Top-100, then
prospectively tests a sample from the selected set. Sparse/noisy MIC and
hemolysis labels make selection quality a larger near-term lever than training a
large generator from scratch.

```text
generator or candidate FASTA
          -> hard sequence gate
          -> MIC / hemolysis / physchem ensemble
          -> uncertainty-aware ranking
          -> novelty + diversity selector
          -> challenge validation
          -> immutable freeze record
          -> explicit Kaggle submit gate
```

The v0 scope intentionally excludes AF3, docking, and membrane MD. Those can be
tested later as Top-100 rerankers only if they add leakage-free prospective value.

## Quick start

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run amp doctor
uv run amp data status

# Fast end-to-end smoke test
uv run generate_broad_spectrum --n-sequences 500 --top-k 20
uv run amp validate \
  --run-dir generate_broad_spectrum \
  --expected-library-size 500 \
  --expected-top-size 20

# Official-sized deterministic output
uv run generate_broad_spectrum
uv run amp validate --run-dir generate_broad_spectrum
```

The default command writes:

```text
generate_broad_spectrum/
  library.fasta
  top.fasta
  scores.csv
  manifest.json
```

`uv run generate` is also provided for compatibility with the minimal official
validator and writes the same contract under `generate/`.

## Use candidates from another generator

Any generator can be connected through FASTA without coupling its environment to
the selector:

```bash
uv run generate_broad_spectrum \
  --candidate-fasta artifacts/ampdiffusion_candidates.fasta
```

This is the first integration path for the official AMP-Diffusion and HydrAMP
starters. Candidate provenance and file digests are recorded in `manifest.json`.

## Freeze and submission safety

Generation never uploads anything. Freeze validates the run and creates a
content-addressed artifact:

```bash
uv run amp freeze --run-dir generate_broad_spectrum
```

Submission is dry-run unless `--execute` is present, and even then requires the
exact frozen run ID:

```bash
uv run amp submit \
  --run-dir generate_broad_spectrum \
  --artifact submission/<run-id>.zip \
  --run-id <run-id> \
  --message "oracle-v1 broad-spectrum" \
  --execute
```

Before API submission, accept the competition rules in the Kaggle web UI once
and configure the official Kaggle CLI. The artifact format must be matched to the
competition's active Kaggle submission page before enabling `--execute`; this
repository does not infer or auto-submit an undocumented payload.

Competition data and current submissions can be accessed through the same CLI
without putting credentials in the repository:

```bash
uv tool install kaggle
uv run amp kaggle files
uv run amp kaggle download --destination data/competition
uv run amp kaggle submissions
```

Kaggle rule acceptance is a one-time web action; API calls fail closed until it
has been completed for the connected account.

## Repository map

```text
configs/                 versioned run and source registries
data/                    challenge reference + provenance notes
docs/                    frozen spec, data/oracle contracts, roadmap
schema/                  machine-readable normalized table schemas
src/amp_challenge/       generation, scoring, selection, validation, gates
tests/                   deterministic unit and integration tests
```

## Design invariants

- One `main` branch is the integration source of truth.
- Fixed seed and byte-identical FASTA output for identical inputs.
- Only 20 standard amino acids, length 8–50, unique linear peptides.
- Full library has no exact challenge-reference overlap.
- Top-100 is a library subset and has no reference similarity above 0.80.
- Random sequence-cluster split is forbidden for oracle evaluation.
- Missing oracle predictions are penalized, never silently imputed as success.
- A frozen run cannot be submitted after any covered file hash changes.
- No network write occurs without the explicit submit command and run ID.

The authoritative implementation plan and acceptance gates are in
[`SPEC.md`](SPEC.md). Upstream contracts are tracked against the
[official challenge repository](https://github.com/szczurek-lab/amp-challenge-2027)
and [Kaggle rules](https://www.kaggle.com/competitions/amp-challenge/rules).

## License

MIT. Upstream models, datasets, and copied assets retain their own licenses; a
source is not enabled for redistribution until its terms are recorded.
