# Oracle contract

All oracle adapters emit a common record per sequence:

```text
sequence
activity_mean       # [0, 1], higher is better
activity_std        # epistemic/model disagreement
toxicity_mean       # [0, 1], higher is worse
toxicity_std
ood_score           # [0, 1], higher is farther from support
model_id
training_manifest_sha256
```

Strain-specific adapters additionally emit one row per strain with a predictive
distribution over `log2(MIC uM)` and threshold probabilities at 4, 16, and 64 uM.

## CSV adapter

The first external integration boundary is a CSV with at least:

```csv
sequence,activity,toxicity,uncertainty
KWKLFKKIGAVLKVL,0.91,0.18,0.07
```

`activity`, `toxicity`, and `uncertainty` must be in `[0,1]`. Missing sequences
are not assigned a favorable default; the ensemble falls back to the transparent
physchem prior and raises the uncertainty term.

## Ensemble rules

- combine independently trained folds/families, not repeated checkpoints from a
  single leaked split;
- preserve per-member outputs in the score audit;
- use lower confidence for potency and upper confidence for toxicity;
- measure calibration before choosing confidence multipliers;
- reject NaN, infinite, or out-of-range predictions;
- pin model and training-data digests in the run manifest.

Generator candidates may never be optimized against a hidden oracle whose
training provenance is unknown.
