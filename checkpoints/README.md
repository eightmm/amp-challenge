# Checkpoints

The small audited JSON checkpoints are committed directly as
`linear-physchem-v1.json` and `esm2-mic16-v2.json`. Larger learned-model checkpoints must be added through
Git LFS and recorded in a model card containing:

- immutable source/training run identifier;
- SHA-256 digest;
- training data manifest digest;
- architecture and preprocessing version;
- validation split and metrics;
- license and redistribution status.

See `MODEL_CARD.md` and `MODEL_CARD_ESM2_V2.md` for checkpoint digests, split
contracts, metrics, and limitations. The heuristic generator itself has no learned
weights. The v2 JSON stores only small calibrated heads; ESM2 backbone weights are
resolved from their pinned upstream revision and are not redistributed.
