# ESM2 MIC16 v2 model card

Status: **development checkpoint; passed its same-fold development gate; not yet the
default submission model**

## Purpose

`esm2-mic16-v2` estimates
`P(MIC <= 16 uM | peptide, organism, Gram class, terminal state)`. It corrects the
largest limitation of the v1 activity model: v1 collapsed heterogeneous organism
measurements into a single conservative sequence label and consequently performed
near chance.

The frozen backbone is `facebook/esm2_t12_35M_UR50D` at revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`. Its 480-dimensional masked-mean
embedding is concatenated with deterministic physicochemical features and explicit
organism, Gram, and terminal-state covariates. Five calibrated logistic-ridge
deployment members are stored in `checkpoints/esm2-mic16-v2.json`.

## Evaluation design

All comparisons use the same 4,984 definitive DRAMP MIC-threshold measurements from
864 peptides. Every sequence has equal total weight within a split. For each of five
outer sequence-cluster folds:

1. one fold is the outer evaluation fold;
2. a different fold is reserved for Platt calibration;
3. regularization is selected by inner CV over the remaining three folds;
4. the candidate and baseline receive identical split, weighting, context, and
   calibration treatment.

These are development metrics. The clusters use 70% global-edit identity, all data
come from one DRAMP snapshot, and fold 0 plus this source informed v1 decisions.
They are not an independent final estimate and do not justify biological efficacy
claims.

## Pooled nested-CV results

| Model | AUROC | AUPRC | Log loss | Top-10% enrichment |
|---|---:|---:|---:|---:|
| Physchem + context | 0.6461 | 0.7535 | 0.6331 | 1.1923 |
| Frozen ESM2 + physchem + context | **0.6892** | **0.7872** | **0.6002** | **1.3472** |
| Delta | **+0.0431** | **+0.0338** | **-0.0329** | **+0.1549** |

The development gate required AUROC and AUPRC deltas of at least +0.02,
Top-10% enrichment delta of at least +0.05, and log-loss delta no worse than
+0.02. The checkpoint passed all four requirements.

At the sequence level, using the 20th percentile of organism-conditioned
probabilities as the broad-spectrum proxy, AUROC improved from 0.6122 to 0.6813,
AUPRC from 0.4895 to 0.5626, and Top-10% enrichment from 1.3097 to 1.6637.

## Coarse-to-fine candidate reranking

The v2 scorer applies ESM2 to 3,000 physchem-prefiltered candidates plus 500
deterministic exploration candidates, then emits a full-library activity artifact.
It evaluates a seven-species ESKAPE proxy panel and penalizes ensemble disagreement
and embedding OOD distance. Unscored candidates keep their v1 activity and receive
uncertainty 1.0.

Compared with the submitted v1 Top-100, the development v2 Top-100 has:

| Quantity | v1 Top-100 | v2 Top-100 |
|---|---:|---:|
| Mean robust ESM activity | 0.5603 | **0.8081** |
| Mean Gram-negative probability | 0.5934 | **0.8272** |
| Mean worst Gram-negative probability | 0.5741 | **0.8152** |
| Mean ESM uncertainty | 0.3691 | **0.3176** |
| Maximum pairwise Levenshtein ratio | 0.55 | 0.55 |

Only two sequences overlap between the portfolios; eight v2 selections came from
the deterministic exploration subset. These are model-based comparisons, not wet-lab
results.

## Limitations and promotion boundary

- Species-level conditioning cannot resolve strain or MDR phenotype.
- Exact/interval MIC regression and ordinal 4/16/64 uM consistency remain pending.
- An independent source or temporal/family holdout is required before changing the
  default scientific claim.
- HC50 remains the v1 limited safety blend; v2 improves activity only.
- The Kaggle submission must not be replaced until reproducible batch inference,
  the organizer validator, and the full frozen-run audit pass.
