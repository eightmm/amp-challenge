# Deterministic AMP generation with safety-aware DRAMP reranking

## Abstract

We submit a deterministic, fully automated baseline for antimicrobial-peptide
generation and safety-aware portfolio selection. A fixed-seed residue sampler
generates 50,000 unique linear peptides from amphipathic alpha-helical position
priors while enforcing the standard amino-acid alphabet, legal length, free
termini, and exact exclusion of the challenge antibacterial reference. The
activity score is a transparent physicochemical prior. The safety score blends a
deterministic membrane-lytic risk prior with a five-member logistic-ridge
ensemble trained on definitive DRAMP HC50 >= 128 uM labels. The learned MIC <=
16 uM head is retained in the checkpoint for audit but disabled from release
ranking after manual review of weak development diagnostics. Fold 0 informed
that decision, so the reported Fold-0 metrics are not unbiased final performance
estimates. Top-100 selection is fully automatic and applies synthesis-risk,
known-reference novelty, and pairwise diversity constraints. All inputs,
weights, configuration, and outputs are content-addressed, and identical
no-argument runs produce byte-identical FASTA files.

## Repository and outputs

- Repository: https://github.com/eightmm/amp-challenge
- Full library: `generate/library.fasta` (50,000 unique peptides)
- Ranked candidates: `generate/top.fasta` (100 peptides; file order is rank)
- Reproduction: `uv sync --locked && uv run generate`
- License: MIT

The repository includes the trained JSON checkpoint, inference code, fixed seed,
locked environment, model card, source registry, and validation/freeze logic.

## Generation

`heuristic-v0` is a fixed-seed generative sampler. It draws 12--32-residue
sequences from explicit residue distributions with seven-position
amphipathic/helical biases. It then enforces uniqueness, standard amino acids,
free termini by construction, charge and hydrophobicity envelopes, entropy, and
run-length limits. This is deliberately a small transparent baseline rather than
a foundation model.

## Training data and learned component

The learned checkpoint uses the DRAMP 3.0 general AMP workbook retrieved on
2026-09-03 under CC BY 4.0. The content-addressed curation pipeline produced:

- 5,012 bacterial MIC measurements;
- 130 quantitative HC50 measurements;
- 929 eligible unique peptide sequences;
- 417 deterministic sequence clusters.

Concentrations were normalized to micromolar while preserving censoring and
intervals. Definitive MIC <= 16 uM and HC50 >= 128 uM threshold labels were
aggregated conservatively by sequence. Folds 2--3 trained model-selection
candidates, fold 4 selected L2 regularization, folds 2--4 were refit, fold 1 was
used only for Platt calibration, and fold 0 was initially evaluated once as a
diagnostic. Its results then informed the release head-selection decision, so it
is now development evidence rather than an untouched final test. The v1 split
uses 70% global edit identity and is explicitly not claimed to be equivalent to
a 40% MMseqs2 family split.

Development diagnostics were:

| Head | n | AUROC | AUPRC | Top-10% enrichment | Release decision |
|---|---:|---:|---:|---:|---|
| MIC <= 16 uM | 185 | 0.503 | 0.485 | 0.950 | Disabled |
| HC50 >= 128 uM | 18 | 0.822 | 0.956 | 1.200 | Safety-only blend |

The safety result is exploratory because fold 0 contains only 18 sequences and
3 negatives. It is not used as a standalone safety claim: it is blended with the
physicochemical prior and contributes conservative uncertainty. Because these
diagnostics informed which heads are used, they are not unbiased final estimates.
No predeclared numeric promotion threshold or same-fold comparison against APEX,
HydrAMP, or a declared physchem baseline was completed.

This compact baseline does not model organism, strain, assay, or
terminal-modification covariates; those remain targets for the planned neural
oracle benchmark.

## Ranking and selection

Release ranking uses the physicochemical activity score and a safety blend with
relative weights 1:2 for the deterministic and learned safety components. Adding
the learned oracle cannot reduce uncertainty: candidate uncertainty is the
maximum stated member uncertainty plus oracle disagreement. Utility also
penalizes cysteine, long homopolymers, and long hydrophobic runs.

The selector then applies:

- no exact match to the organizer antibacterial reference for all 50,000;
- maximum Top-100 reference Levenshtein ratio <= 0.80;
- maximum Top-100 pairwise ratio <= 0.55;
- synthesis-oriented charge, hydrophobicity, entropy, cysteine, and run gates.

No peptide was manually inserted, removed, or reordered.

## Validation summary

The final local release contains 50,000/50,000 unique library sequences and
100/100 unique ranked sequences. All Top-100 entries are present in the library.
Reference overlap is zero, maximum checked Top-100 reference ratio is 0.65, and
maximum Top-100 pairwise ratio is 0.55. Two consecutive `uv run generate` runs
produced byte-identical library and Top-100 FASTA files.

## External resources and limitations

The organizer reference is used only for exclusion and novelty checks. APEX,
AMP-Diffusion, HydrAMP, DBAASP, APD3, and GRAMPA do not contribute predictions or
training labels to this release. The only manual modeling intervention was the
post-diagnostic release decision to disable the weak activity head and retain the
small safety head only as a blend. This decision is not presented as a passed
promotion gate. Scores must not be interpreted as measured MIC, measured HC50,
clinical efficacy, or exposure safety. Prospective wet-lab validation is
required.

## AI-assistant disclosure

OpenAI Codex assisted with repository scaffolding and implementation, tests,
documentation drafting, and browser-guided submission preparation. Jaemin Sim
set the objectives, reviewed the work, authorized final actions, and remains
responsible for the submitted content and results.
