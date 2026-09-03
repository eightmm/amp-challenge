# Data contract

## Canonical sequence

`sequence` is an uppercase string containing only the 20 standard amino acids.
Whitespace, FASTA wrapping, and terminal punctuation are removed; chemical
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
| `strain_name` | Normalized strain or `null` |
| `taxon_id` | NCBI Taxonomy identifier where resolved |
| `resistance_profile` | MDR/clinical phenotype text, preserved and normalized |
| `value_original` | Numeric boundary as reported |
| `unit_original` | Original concentration unit |
| `relation` | `eq`, `lt`, `le`, `gt`, or `ge` |
| `value_um` | Boundary converted to micromolar |
| `log2_um` | `log2(value_um)` boundary |
| `assay_*` | Medium, temperature, pH, method, incubation metadata |
| `source_*` | Database/publication identity and immutable record version |

Mass-concentration conversion uses the neutral free-terminus peptide molecular
weight computed from residue masses plus water:

```text
uM = (ug / mL) * 1000 / molecular_weight_g_per_mol
```

If the record is amidated or otherwise modified, its reported/formula molecular
weight is required; it is not eligible for challenge generation but may remain
useful as a labeled training record if explicitly represented.

## Hemolysis table

The normalized table follows `schema/hemolysis_measurement.schema.json`. Keep
species/cell source, endpoint (HC10/HC50/% at concentration), exposure time,
buffer, and censoring. Do not merge a percent-hemolysis observation with HC50 as
if they were the same target.

## Duplicate and conflict policy

1. Preserve every raw observation.
2. Collapse exact source duplicates only after provenance audit.
3. Treat technical replicates separately from independent studies.
4. Aggregate compatible two-fold dilution measurements in log2 concentration.
5. Retain conflicting studies and model study/source effects.
6. Never average lower/upper-censored records into an uncensored point.

## Split policy

Clusters, not rows, are assigned to folds. The primary target is MMseqs2 clusters
at 40% identity with coverage declared. Near-identical variants, duplicate
measurements, and the same canonical sequence across sources always share a
fold. A source/temporal holdout is maintained separately to expose assay and
database shift.

The challenge antibacterial reference is never used as a labeled evaluation
fold. It is a novelty/exclusion reference.

## Source governance

`configs/data_sources.json` is deny-by-default. A source moves from
`manual-review` to `automatic` only after its exact release, access method,
redistribution terms, parser, and SHA-256 expectations are recorded. API keys
and tokens are read only from standard credential stores/environment variables
and are never written into manifests.
