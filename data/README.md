# Data

`antibacterial.fasta` is the official AMP Challenge 2027 reference set. Its
expected SHA-256 is pinned in `configs/data_sources.json` and can be checked with:

```bash
uv run amp data status
```

Large raw and derived datasets are intentionally excluded from Git. Every
training run must instead reference a content-addressed data manifest. See
[`docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md) for the normalized MIC and
hemolysis schemas and source-governance rules.

The first approved training-data build is reproduced with:

```bash
uv run amp data fetch dramp-general-2026-09-03
uv run amp data prepare
uv run amp train preflight
```

It writes only under ignored `data/raw/` and `data/processed/` paths. The source
registry pins the DRAMP retrieval URL, date, CC BY 4.0 decision, and raw SHA-256;
the generated manifest pins every derived table.

No database is scraped merely because it is publicly viewable. Sources marked
`manual-review` require an explicit license/terms decision and a pinned release
before a downloader is enabled.
