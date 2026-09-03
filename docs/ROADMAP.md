# Execution roadmap

## Now: make the path boring and reproducible

- [x] Version the official hard constraints.
- [x] Track the challenge reference by SHA-256.
- [x] Implement deterministic FASTA generation and selection.
- [x] Add local validation, manifests, freeze IDs, and a guarded submit adapter.
- [x] Add smoke tests and CI.
- [x] Run the official 50,000-sequence validator twice with identical output.

## Next: baseline artifacts

- [ ] Pin AMP-Diffusion starter commit and checkpoint hashes.
- [ ] Pin HydrAMP starter commit and checkpoint hashes.
- [ ] Generate surplus pools from both under fixed seeds.
- [ ] Import both through the candidate-FASTA boundary.
- [ ] Record runtime, GPU memory, duplicate rate, rejection funnel, and seqme.

## Data/oracle workstream

- [x] Pin the APEX MIT code/weight commit; training observations remain absent.
- [x] Enable pinned DRAMP bulk files under CC BY 4.0.
- [x] Keep DBAASP and GRAMPA fail-closed pending acceptable usage terms.
- [x] Build normalized, interval-aware MIC and HC50 tables.
- [x] Audit units, censored labels, termini, taxonomy, and parser failures.
- [x] Create deterministic 70%-threshold single-linkage global-edit folds and
      verify zero cross-fold threshold violations.
- [x] Pin the frozen ESM2-35M revision and pass the training preflight.
- [ ] Add source/temporal holdouts and pinned MMseqs2 cluster evidence.
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
