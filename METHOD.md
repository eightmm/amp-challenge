# Method abstract (working draft)

This entry uses a generator-agnostic antimicrobial-peptide pipeline in which
candidate sequences are first constrained to the AMP Challenge alphabet, length,
uniqueness, and known-reference exclusion rules, then ranked by an ensemble of
potency, hemolysis, physicochemical, and out-of-distribution estimates. Potency
models are designed as small, calibrated heads over frozen peptide-language-model
representations and are trained with organism/strain conditioning and
censor-aware MIC objectives. Safety is modeled independently to prevent activity
optimization from selecting broadly membrane-lytic peptides. Final selection is
treated as a portfolio problem: lower-confidence potency, upper-confidence
toxicity, novelty, synthesizability, oracle disagreement, and pairwise sequence
diversity jointly determine the ranked Top-100. All candidates are selected by a
fixed-seed automated procedure; inputs, configurations, model/data versions, and
outputs are content-addressed and guarded by an immutable pre-submission freeze.

## Bootstrap implementation disclosure

The current v0.1 code exercises that contract using a non-learned physicochemical
oracle and deterministic heuristic generator. It is an infrastructure baseline,
not the proposed final competition model. The final abstract must be updated with
the exact learned models, data releases, metrics, and weights before submission.
