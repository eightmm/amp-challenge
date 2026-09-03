# Model card

## Model

`heuristic-v0` generator + `physchem-v0` oracle.

## Intended use

Deterministic infrastructure smoke testing for the AMP Challenge 2027 generation,
selection, validation, freeze, and transport path.

## Out-of-scope use

This version must not be interpreted as a validated antimicrobial activity,
hemolysis, clinical efficacy, or safety predictor. It must not guide treatment or
human/animal exposure.

## Training data

None. The generator samples from explicitly coded amino-acid priors and helical
position biases. The oracle is an explicit physicochemical heuristic.

## Validation

Software invariants only: deterministic outputs, sequence legality, uniqueness,
reference exclusion, Top-K subset, novelty, freeze integrity, and CLI behavior.

## Replacement gate

A learned default requires the evaluation and provenance evidence listed in
`SPEC.md`, including sequence-clustered folds, calibration, Top-K enrichment, and
fully disclosed training data.
