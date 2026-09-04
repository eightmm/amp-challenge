# Oracle contract

The target contract for richer oracle adapters is one common record per
sequence:

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

The current v1 release implements the smaller CSV boundary documented below. Its
`scores.csv` preserves aggregate activity, toxicity, uncertainty, and the names
of every contributing oracle; it does not export each member's raw prediction as
a separate column. Full per-member export remains a future auditability gate.

## CSV adapter

The first external integration boundary is a CSV with at least:

```csv
sequence,activity,toxicity,uncertainty
KWKLFKKIGAVLKVL,0.91,0.18,0.07
```

`activity`, `toxicity`, and `uncertainty` must be in `[0,1]`. Missing sequences
are a release error: once any CSV adapter is supplied, every candidate selected
from the candidate FASTA must have a prediction in every supplied CSV. Extra rows
may be present, but missing candidate rows fail before scoring.

## Learned JSON checkpoint

The default release path can load a self-contained linear-oracle checkpoint from
the run config. Both its repository-relative path and exact SHA-256 are required:

```json
{
  "oracle": {
    "physchem_weight": 1.0,
    "learned_checkpoint": {
      "path": "checkpoints/linear-oracle-v1.json",
      "sha256": "<64 lowercase hexadecimal characters>",
      "weight": 1.0,
      "use_activity": false,
      "use_toxicity": true
    }
  }
}
```

Generation fails if the file is absent, its digest differs, or checkpoint
validation fails. The learned model's source is retained in `scores.csv`, and the
checkpoint path, digest, and source are retained in `manifest.json`.

Each learned head is enabled independently. `use_activity` or `use_toxicity`
records the release configuration, and at least one enabled oracle must remain
for each output. The evidence and decision rule used to choose enabled heads must
be documented. If a diagnostic fold informs that choice, its metrics become
development evidence and must not be reported as an untouched, unbiased final
test. A formal promotion claim additionally requires a predeclared decision rule
and same-fold baseline comparisons; the current v1 release makes no such claim.

## Ensemble rules

- combine independently trained folds/families, not repeated checkpoints from a
  single leaked split;
- preserve contributing source IDs now, and add per-member outputs before making
  formal ensemble-performance claims;
- use lower confidence for potency and upper confidence for toxicity;
- measure calibration before choosing confidence multipliers;
- reject NaN, infinite, or out-of-range predictions;
- pin model and training-data digests in the run manifest.

Generator candidates may never be optimized against a hidden oracle whose
training provenance is unknown.
