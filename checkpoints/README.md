# Checkpoints

The small audited JSON baseline is committed directly as
`linear-physchem-v1.json`. Larger learned-model checkpoints must be added through
Git LFS and recorded in a model card containing:

- immutable source/training run identifier;
- SHA-256 digest;
- training data manifest digest;
- architecture and preprocessing version;
- validation split and metrics;
- license and redistribution status.

See `MODEL_CARD.md` for the JSON checkpoint digest, split contract, metrics, and
limitations. The heuristic generator itself has no learned weights.
