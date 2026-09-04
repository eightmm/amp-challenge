# AMP Challenge Pipeline Specification

Status: **v0.3 executable baseline release**

Last updated: 2026-09-03

Owner: Jaemin Sim

Integration branch: `main`

## 1. Goal

Produce a reproducible AMP Challenge 2027 entry whose expected prospective
performance is driven by calibrated antimicrobial potency and safety estimates,
not by generator novelty alone.

The system must support the complete path:

1. register and acquire legally usable data;
2. normalize sequence, organism/strain, units, censoring, and assay metadata;
3. build sequence-clustered, leakage-resistant evaluation splits;
4. train and calibrate MIC and hemolysis oracles;
5. ingest candidates from interchangeable generators;
6. rank with ensemble uncertainty and deterministic physicochemical priors;
7. select a diverse Top-100 under challenge constraints;
8. validate, freeze, reproduce, and explicitly submit one model entry.

## 2. Challenge contract

The default category entry point is `uv run generate_broad_spectrum`. A
compatibility alias, `uv run generate`, is also provided.

Default outputs:

- `library.fasta`: exactly 50,000 unique sequences;
- `top.fasta`: exactly 100 ranked sequences, all present in the library;
- `scores.csv`: auditable score components;
- `manifest.json`: input/config/output hashes and implementation identity.

Sequence constraints:

- alphabet: `ACDEFGHIKLMNPQRSTVWY` only;
- length: 8–50 residues;
- linear peptides with free termini;
- no duplicate sequences;
- no exact full-library overlap with `data/antibacterial.fasta`;
- no Top-100 Levenshtein ratio greater than 0.80 to that reference.

## 3. Scientific strategy

### 3.1 Initial track

Start with broad-spectrum generation while explicitly reporting Gram-negative and
MDR-ESKAPE oracle slices. Do not claim a strain-specific track until its target
panel and entry-point contract are confirmed by the organizer.

### 3.2 Generator

The generator is an adapter, not the source of truth. Integration order:

1. `heuristic-v0`: deterministic infrastructure smoke test only;
2. official AMP-Diffusion candidate FASTA;
3. official HydrAMP candidate FASTA as an independent family;
4. only then consider a newly trained generator if selector/oracle gains plateau.

Generation and scoring environments remain separable. A candidate FASTA plus a
content digest is the stable boundary.

### 3.3 Potency oracle

Target interface:

```text
(sequence, organism, strain, resistance phenotype)
    -> distribution over log2(MIC in uM)
```

Required modeling properties:

- frozen or low-rank-adapted peptide language-model representation;
- small shared head with organism/strain embeddings;
- interval/censored likelihood for `<`, `<=`, `>`, and `>=` MIC records;
- ordinal threshold outputs at 4, 16, and 64 uM;
- fold ensemble trained on sequence-identity clusters;
- post-hoc calibration measured only on disjoint calibration clusters.

APEX is an external ensemble member, not ground truth. Generator optimization
against a single oracle is prohibited because it invites reward hacking.

### 3.4 Safety oracle

Predict an HC50 interval where quantitative data exist and a calibrated
hemolysis-risk class otherwise. Include charge, hydrophobic fraction,
hydrophobic moment, aromatic fraction, length, and aggregation proxies alongside
the frozen representation. Missing safety predictions receive an uncertainty
penalty.

### 3.5 Selection objective

For candidate `x`, the initial learned-selector contract is:

```text
utility(x) = potency_LCB(x)
           - lambda_tox * toxicity_UCB(x)
           - lambda_ood * OOD(x)
           - lambda_synth * synthesis_risk(x)
```

`potency_LCB` must reward performance across the requested panel and include a
worst-strain term. Top-100 selection is a constrained portfolio problem, not a
plain top-score sort:

- pass all challenge hard constraints;
- pass the declared novelty threshold;
- cap near-duplicate sequence families;
- preserve multiple generator/model families;
- optimize both expected utility and coverage;
- record every rejection reason.

## 4. Data contract

All quantitative activity is normalized to micromolar before modeling. Original
values and units are retained. Censored measurements are never converted to
point labels. Organisms are normalized at species and strain levels, and the
original text is preserved.

Each processed dataset has a manifest containing:

- source name, immutable version/URL, retrieval time, and license decision;
- raw-file SHA-256 hashes;
- parser version and normalized-schema version;
- row counts before/after every filter;
- exact sequence canonicalization policy;
- duplicate/conflict aggregation policy;
- split-cluster mapping and digest.

See `docs/DATA_CONTRACT.md` and `schema/`.

## 5. Evaluation

Random row or random sequence splits are forbidden as primary evidence.

Required reports:

- identity-cluster split at a declared threshold (target 40%);
- leave-family-out performance;
- per-organism and Gram-positive/negative slices;
- temporal/source holdout where dates permit;
- AUROC/AUPRC for activity thresholds;
- censored/ordinal likelihood and log2-MIC error;
- calibration error and reliability plots;
- enrichment/precision in the Top-K region;
- oracle disagreement and OOD-stratified metrics.

Promotion requires improvement over APEX/HydrAMP/physchem baselines on the same
leakage-free folds, not only random-split averages. The current release has not
completed that comparison and therefore makes no formal promotion claim.

## 6. Run states and authority

```text
draft -> validated -> frozen -> submitted
```

- `draft`: mutable experiment output;
- `validated`: challenge and internal checks pass;
- `frozen`: covered hashes and a run ID are written;
- `submitted`: an explicit user-invoked transport succeeded.

The submit command requires both `--execute` and the exact `--run-id`. A freeze
record is invalid if any covered input/output changed. No training or generation
command can submit implicitly.

## 7. Delivery phases

### Phase 0 — submission skeleton (complete)

- official-sized deterministic generation contract;
- challenge reference registry and checksum;
- physchem oracle and novelty/diversity selector;
- local validation, manifest, freeze, and dry-run submit gate;
- unit/integration tests.

Acceptance: 500-sequence smoke run is byte reproducible and passes all checks;
the default command can produce the official output shape.

### Phase 1 — baseline reproduction

- ingest frozen AMP-Diffusion and HydrAMP outputs;
- run official validator and Phase-1 sequence metrics;
- establish compute/time/memory and score baselines;
- version generator outputs as content-addressed artifacts.

Acceptance: independent reruns have identical FASTA bytes on the pinned runtime.

### Phase 2a — training-data handoff (complete for DRAMP v1)

- pinned, checksum-verified DRAMP general workbook;
- interval-aware bacterial MIC and HC50 parsing with unit normalization;
- explicit terminus state and quarantine audit;
- deterministic 70%-threshold single-linkage global-edit clusters and balanced
  five-fold roles;
- content-addressed artifacts and a fail-closed training preflight;
- frozen ESM2-35M revision and small-head training configuration.

Acceptance: `uv run amp train preflight` reports `ready: true`, every artifact
matches its manifest, and no cluster crosses fold boundaries. This phase does not
claim that training ran or that the current global-edit clustering is equivalent
to MMseqs2 local alignment.

### Phase 2b — oracle benchmark (in progress)

- add independently licensed sources and source/temporal holdouts;
- replace or corroborate the v1 clustering with pinned MMseqs2 runs;
- APEX plus at least two independent learned oracle families;
- calibration and OOD report.

The dependency-free DRAMP physicochemical baseline has been executed. Its MIC
activity head is disabled after manual review of weak development diagnostics,
including Fold 0; because Fold 0 informed that release choice, its metrics are
not an unbiased final performance estimate. The small HC50 head remains an
explicitly limited safety-only ensemble member. Frozen ESM2, independent
oracle-family benchmarks, and the same-fold APEX/HydrAMP/physchem comparison are
still pending. No promotion gate has passed for this release.

Future acceptance (not yet met): a predeclared promotion rule passes on newly
reserved evaluation clusters and Top-K enrichment.

### Phase 3 — robust selector

- ensemble lower-confidence potency score;
- toxicity upper-confidence penalty;
- generator/reference novelty and family quotas;
- Pareto and portfolio ablations.

Acceptance: selector beats each single oracle and naive score sort across bootstrap
resamples and does not collapse Top-100 diversity.

### Phase 4 — candidate freeze

- generate a surplus pool;
- blinded selection rehearsal;
- reproduce full 50,000 + Top-100 twice;
- independent audit against official validator;
- freeze method abstract, provenance, weights, and inference code.

Acceptance: no unresolved license, reproducibility, or submission-schema issue.

## 8. Explicit non-goals for v1

- target-protein docking;
- AF3/ESMFold as a primary activity oracle;
- membrane MD in the default reproducible entry point;
- a new large peptide foundation model;
- hard-coded manual selection of individual Top-100 sequences;
- scraping a database without a documented bulk-use right.

## 9. Promotion checklist

A model/selector can become the default only when all answers are yes:

- Is its training data manifest complete and redistributable as required?
- Are sequence-family leaks excluded from evaluation?
- Is uncertainty calibrated on disjoint calibration clusters?
- Does it improve Top-K enrichment, not only mean loss?
- Does it retain performance across Gram-negative/MDR slices?
- Is full generation deterministic under the pinned environment?
- Are weights, inference code, and default arguments present?
- Does the official validator pass twice with identical outputs?
- Has the exact Kaggle payload/entry type been confirmed?
