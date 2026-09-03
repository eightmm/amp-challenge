# Execution roadmap

## Now: make the path boring and reproducible

- [x] Version the official hard constraints.
- [x] Track the challenge reference by SHA-256.
- [x] Implement deterministic FASTA generation and selection.
- [x] Add local validation, manifests, freeze IDs, and a guarded submit adapter.
- [x] Add smoke tests and CI.
- [ ] Run the official 50,000-sequence validator on a GPU/cluster checkout.

## Next: baseline artifacts

- [ ] Pin AMP-Diffusion starter commit and checkpoint hashes.
- [ ] Pin HydrAMP starter commit and checkpoint hashes.
- [ ] Generate surplus pools from both under fixed seeds.
- [ ] Import both through the candidate-FASTA boundary.
- [ ] Record runtime, GPU memory, duplicate rate, rejection funnel, and seqme.

## Data/oracle workstream

- [ ] Resolve license and immutable release for APEX data/weights.
- [ ] Resolve DBAASP/DRAMP/GRAMPA access and redistribution decisions.
- [ ] Build normalized MIC and hemolysis tables.
- [ ] Audit units, censored labels, organism/strain mappings, and conflicts.
- [ ] Create 40%-identity cluster folds with source and temporal holdouts.
- [ ] Train frozen-PLM small heads and non-neural baselines.
- [ ] Calibrate fold ensembles and benchmark Top-K enrichment.

## Selection workstream

- [ ] Add APEX, custom MIC, hemolysis, and OOD CSV adapters.
- [ ] Tune only on nested validation folds.
- [ ] Compare mean score, LCB, Pareto, and quota-constrained portfolios.
- [ ] Stress test oracle disagreement and adversarial/reward-hacked candidates.
- [ ] Freeze one primary and one scientifically distinct backup entry.

## Final gate

- [ ] Confirm active Kaggle payload/API contract and rule acceptance.
- [ ] Publish complete training-data disclosure and model cards.
- [ ] Run generation twice from clean clones; compare bytes.
- [ ] Run the organizer validator from a separate environment.
- [ ] Review Top-100 rejection/provenance audit.
- [ ] Submit only the exact frozen run ID.
