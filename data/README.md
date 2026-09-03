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

No database is scraped merely because it is publicly viewable. Sources marked
`manual-review` require an explicit license/terms decision and a pinned release
before a downloader is enabled.
