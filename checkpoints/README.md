# Checkpoints

This directory is intentionally empty in the bootstrap commit. Learned-model
checkpoints must be added through Git LFS and recorded in a model card containing:

- immutable source/training run identifier;
- SHA-256 digest;
- training data manifest digest;
- architecture and preprocessing version;
- validation split and metrics;
- license and redistribution status.

The default `heuristic-v0` smoke baseline has no learned weights.
