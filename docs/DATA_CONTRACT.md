# Data contract

## Canonical sequence

`sequence` is an uppercase string containing only the 20 standard amino acids.
Whitespace and FASTA wrapping are removed; punctuation and chemical
modifications are not silently stripped. Modified/noncanonical records are
quarantined with a reason instead of coerced into linear free-terminus peptides.

The canonical sequence SHA-256 is the stable peptide key. Database identifiers
remain source-specific aliases.

## MIC measurement table

The normalized table follows `schema/mic_measurement.schema.json`. Important
fields are:

| Field | Meaning |
|---|---|
| `sequence` | Canonical peptide sequence |
| `organism_name` | Normalized species name |
| `strain_name` | Preserved strain text when available |
| `resistance_profile` | MDR/clinical phenotype text, preserved and normalized |
| `value_original_low/high` | Original point, range, or censoring boundary |
| `unit_original` | Original concentration unit |
| `relation` | `eq`, `interval`, `lt`, `le`, `gt`, or `ge` |
| `lower_um/upper_um` | Converted interval; an open side is null/blank |
| `lower_log2_um/upper_log2_um` | Log2 interval boundaries |
| `active_le_*um` | 0/1 only when the interval proves the threshold; else null |
| `source_*` | Database and source-record provenance |

Mass-concentration conversion computes peptide molecular weight from residue
masses and the declared terminus state. The free/free form adds water;
N-acetylation and C-amidation use explicit mass deltas:

```text
uM = (ug / mL) * 1000 / molecular_weight_g_per_mol
```

The v1 curation set accepts free or acetylated N termini and free or amidated C
termini as explicit covariates. Other modifications, cyclic/branched peptides,
non-L stereochemistry, noncanonical residues, and lengths outside 8--50 are
quarantined rather than silently transformed. Generator output still requires
linear peptides with free termini.

## Hemolysis table

The normalized table follows `schema/hemolysis_measurement.schema.json`. DRAMP v1
extracts quantitative HC50 observations only, preserves cell-source text and
censoring, and leaves other percent-hemolysis observations out of this target.
They must not be merged with HC50 as if they were the same endpoint.

## Duplicate and conflict policy

The v1 build preserves each source-record observation and collapses only
byte-equivalent parsed duplicates within the same source record, recording a
replicate count. It does not average conflicting studies, infer technical
replicates, or midpoint-impute ranges/censored values. Cross-study aggregation
and source-effect modeling remain future work once a second approved source is
added.

## Split policy

Clusters, not rows, are assigned to folds. The v1 handoff uses deterministic
single-linkage components at 70% global edit identity and 80% minimum length
coverage. Every qualifying pair is joined transitively, so preflight can require
zero cross-fold pairs at or above that threshold. This is not MMseqs2 local
alignment. A pinned 40%-identity MMseqs2 split and source/temporal holdouts remain
required before stronger comparative model claims.

The challenge antibacterial reference is never used as a labeled evaluation
fold. It is a novelty/exclusion reference.

## Source governance

`configs/data_sources.json` is deny-by-default. A source moves from
`manual-review` to `automatic` only after its exact release, access method,
redistribution terms, parser, and SHA-256 expectations are recorded. API keys
and tokens are read only from standard credential stores/environment variables
and are never written into manifests.

The first automatic quantitative source is the DRAMP general workbook retrieved
on 2026-09-03 under CC BY 4.0. The DRAMP patent dataset is excluded. DBAASP and
GRAMPA remain `manual-review` and cannot enter the build. APEX is registered as
an external MIT-licensed model ensemble; its repository does not supply the
training observations used by this data build.
