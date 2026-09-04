# Method abstract

We submit a deterministic, fully automated baseline for antimicrobial-peptide
generation and safety-aware portfolio selection. A fixed-seed residue sampler
generates 50,000 unique linear peptides from amphipathic alpha-helical position
priors while enforcing the 20-residue alphabet, 8--50-residue length contract,
free termini, and exact exclusion of the challenge antibacterial reference.
Candidates are ranked with a hybrid physicochemical oracle. The activity term is
a transparent charge, hydrophobicity, hydrophobic-moment, length, and entropy
prior. The safety term blends a deterministic membrane-lytic risk prior with a
five-member logistic-ridge ensemble trained on definitive DRAMP HC50 >= 128 uM
labels. Model fitting, calibration, and diagnostic scoring use non-overlapping
sequence-cluster folds. A separately trained MIC <= 16 uM head is retained in
the auditable checkpoint but disabled from release ranking after manual review
of weak development diagnostics. Fold 0 informed that decision, so its reported
metrics are not unbiased final performance estimates. No formal
APEX/HydrAMP/physchem promotion comparison was completed. The final Top-100 is
selected without manual peptide choice under synthesis-risk, known-reference
novelty, and pairwise-diversity constraints. Configuration, training inputs,
checkpoint, reference set, and outputs are content-addressed, and two identical
no-argument runs must produce byte-identical FASTA files.

## Data, external resources, and intervention disclosure

- Training labels: DRAMP 3.0 general AMP workbook, retrieved 2026-09-03 under
  CC BY 4.0; 5,012 normalized MIC measurements and 130 HC50 measurements after
  curation.
- Challenge reference: organizer-provided `data/antibacterial.fasta`, used only
  for exact library exclusion and Top-100 novelty checks.
- External learned models/databases in release scoring: none. APEX,
  AMP-Diffusion, HydrAMP, DBAASP, APD3, and GRAMPA are not used.
- Manual intervention: the pipeline-level decision to disable the weak
  learned activity head and retain the limited safety head. No candidate
  sequence or Top-100 member was manually inserted, removed, or reordered.
- Computational filters: standard amino-acid alphabet, legal length, free
  termini by construction, uniqueness, charge/hydrophobicity/entropy/run-length
  gates, synthesis-risk penalty, no exact reference overlap, <=0.80 Top-100
  reference Levenshtein ratio, and <=0.55 pairwise Top-100 ratio.

## AI-assistant disclosure

OpenAI Codex assisted with repository scaffolding and implementation, tests,
documentation drafting, and browser-guided submission preparation. Jaemin Sim
set the objectives, reviewed the work, authorized final actions, and remains
responsible for the submitted content and results.

## Reproduction

```bash
uv sync --locked
uv run generate
```

Outputs are written to `generate/library.fasta` and `generate/top.fasta`.
