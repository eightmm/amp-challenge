# Model card

## Release model

`heuristic-v0` generator with a conservative hybrid selector:

- activity: transparent `physchem-v0` score;
- safety: weighted blend of `physchem-v0` and the trained
  `linear-physchem-v1` HC50-threshold ensemble;
- learned activity head: present in the checkpoint for audit, but disabled in
  release scoring after manual review of weak development diagnostics.

The checkpoint is `checkpoints/linear-physchem-v1.json` (SHA-256
`efaa9236fc93e3b01ac02ffb8a8bceab06fcd351922fb2801ad2b23f6aca6f9a`,
internal checkpoint ID
`6efbe1cde585e6508feae4135697fd4b36b7f7008546da5dfe22615fe3503885`).
It is a 25 KB, standard-library-only logistic-ridge ensemble.

## Intended use

Reproducible participation in the AMP Challenge 2027 computational benchmark.
The model is an execution baseline intended to produce a legal, diverse,
auditable candidate library. It is not a clinical or laboratory safety model.

## Training data

The learned checkpoint uses the pinned DRAMP 3.0 general AMP workbook retrieved
on 2026-09-03 under CC BY 4.0. Curation produced 5,012 MIC measurements and 130
HC50 measurements over 929 eligible sequences. Definitive MIC <= 16 uM and HC50
>= 128 uM threshold labels were conservatively aggregated per sequence.

Sequence clusters, rather than rows, define the split. Folds 2 and 3 are used
for model selection training, fold 4 for regularization selection, folds 2--4
for refitting, and fold 1 for Platt calibration. Fold 0 was initially evaluated
once as a diagnostic, but its results then informed which learned heads were
enabled in release scoring. It is therefore development evidence, not an
untouched final test. The v1 clusters use 70% global edit identity and are not
equivalent to a 40% MMseqs2 family split.

## Development diagnostics (not unbiased final estimates)

| Head | Diagnostic-fold n | Positives | AUROC | AUPRC | Top-10% enrichment | Release use |
|---|---:|---:|---:|---:|---:|---|
| Learned MIC <= 16 uM | 185 | 82 | 0.503 | 0.485 | 0.950 | Disabled |
| Learned HC50 >= 128 uM | 18 | 15 | 0.822 | 0.956 | 1.200 | Safety-only, blended |

The activity result is near random and prompted the conservative manual decision
to disable that head. The safety result is exploratory because the diagnostic
subset contains only 18 sequences and 3 negatives. It is therefore blended with
a deterministic prior and carries an uncertainty penalty rather than being used
alone. Since these diagnostics informed the release configuration, neither row
is an unbiased estimate of final model performance. No predeclared numeric
promotion threshold or same-fold APEX/HydrAMP/physchem comparison was completed,
so this release makes no claim that a formal promotion gate passed.

## Generator and selection

`heuristic-v0` samples standard amino acids from fixed amphipathic/helical
position priors, then applies deterministic sequence and physicochemical gates.
It is a generative computational baseline, not a learned AMP generator. The
Top-100 is selected automatically from the 50,000-member library using activity,
safety, uncertainty, synthesis-risk, reference-novelty, and pairwise-diversity
constraints. No individual peptide is hand-selected.

## Out-of-scope use and limitations

- Do not interpret scores as MIC, HC50, clinical efficacy, or exposure safety.
- DRAMP is the only learned-label source, so source and assay shift are not
  independently measured.
- Sequence-level worst-observation aggregation is conservative but discards
  organism-specific structure.
- Organism, strain, assay, and terminal-modification covariates are not modeled
  by this small baseline.
- No protein-language-model embeddings, APEX, AMP-Diffusion, or HydrAMP outputs
  are used in this release.
- The enabled-head choice used the reported Fold-0 diagnostics; an independently
  reserved evaluation split is required for an unbiased final estimate.
- Prospective wet-lab validation is required before biological claims.
