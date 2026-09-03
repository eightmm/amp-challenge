# ADR-0001: Oracle-first, replaceable-generator architecture

Status: accepted

## Context

Unlabeled peptide sequences are plentiful, while harmonized strain-specific MIC
and hemolysis measurements are sparse, censored, duplicated, and shifted across
assays. The challenge prospectively tests selected candidates, so ranking errors
and correlated Top-100 failures dominate the practical risk.

## Decision

Keep generation behind a content-addressed FASTA boundary. Spend the first
modeling cycle on leakage-resistant data curation, calibrated potency/safety
ensembles, uncertainty, and portfolio-style selection. Start with official
generators; train a new generator only after oracle/selector baselines are stable.

## Consequences

- AMP-Diffusion and HydrAMP can be compared without dependency co-resolution.
- Oracle reward hacking is visible through cross-model disagreement.
- Full submissions remain reproducible even when training is expensive.
- The bootstrap generator is useful for infrastructure testing but carries no
  claim of biological competitiveness.
- 3D structure methods remain optional rerank ablations rather than critical path.
