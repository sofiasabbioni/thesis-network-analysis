# S&P 500 constituent list — source and version

This folder holds the **static, committed** stock universe used by the
supplementary large-network extension. It is deliberately a file in the
repository rather than a live scrape: a run six months from now must analyse
exactly the same companies as the run behind the reported numbers.

## Files

| File | Contents |
|---|---|
| `sp500_constituents_2025_12_22.csv` | 503 constituents: `ticker`, `yahoo_ticker`, `company`, `gics_sector`, `gics_sub_industry`, `metadata_source`, `metadata_snapshot_role` |
| `sp500_constituents_2025_12_22_provenance.json` | Machine-readable record of everything below |

## Source

Both membership and classification come from the public dataset

> **https://github.com/fja05680/sp500** — *S&P 500 Historical Components & Changes*
> repository HEAD at retrieval: `c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff` (2026-07-13)

retrieved on 2026-08-28.

### Membership — an end-of-sample point-in-time snapshot

`S&P 500 Historical Components & Changes (Updated).csv` records the full
constituent list on every date index membership changed, from 1996 onwards.
The universe is the last such snapshot **on or before 2025-12-31**, the last
day of the thesis sample:

* **snapshot date: 2025-12-22**
* **503 constituents**
* (the next membership change in the dataset is 2026-01-14, after the sample)

This is the index as it actually stood at the end of the thesis period, rather
than a present-day list projected backwards.

**How this must be described.** The universe is selected on end-of-sample index
membership and is then analysed retrospectively over 2019–2025, so it is a
*point-in-time end-of-sample constituent universe* and the results are
conditional on end-of-period membership:

> The extension uses a point-in-time S&P 500 constituent universe observed at
> the end of the sample period. Consequently, the analysis is conditional on
> end-of-period index membership and should not be interpreted as a
> survivorship-bias-free representation of the S&P 500 throughout 2019–2025.

Companies that were in the index during part of 2019–2025 but had left it by
2025-12-22 are not in the universe, and companies that joined during the window
are included for the whole window (subject to the data-cleaning rules, which
remove any name without a complete price history). A survivorship-bias-free
design would require a time-varying universe, which the threshold-network
methodology used here — one correlation matrix estimated on one fixed common
calendar — does not accommodate.

### GICS classification — pre-sample-end snapshot preferred

`sp500.csv` in the same repository is a capture of the Wikipedia
*List of S&P 500 companies* table (ticker, company, GICS sector, GICS
sub-industry). Because that file is a *current* snapshot, it is read at dated
commits rather than at repository HEAD.

The **primary** source is the last capture dated *before* the end of the thesis
sample, so the classification stays strictly inside the sample period rather
than importing one that only came into force afterwards. The first capture
*after* the sample end is a **fallback**, consulted only for constituents that
the earlier file does not contain — companies added to the index between the
metadata capture and the membership snapshot:

| Commit | Dated | Role | Stocks classified |
|---|---|---|---|
| `58490ec8ea42827b6df808ff1e68dff51144f09f` | 2025-11-16 | **primary** — last capture before the sample end | **498** |
| `c24b12e2b725cb4d18ab6b816fbeb6226280aae6` | 2026-01-17 | **fallback** — first capture after the sample end | **5** |

The five requiring the fallback are `ARES`, `CRH`, `CVNA`, `FIX` and `SNDK`.
Each row of the constituent CSV records which snapshot classified it, in
`metadata_source` and `metadata_snapshot_role`.

Comparing the two captures, **no constituent's GICS sector or sub-industry
differs between them**, so the choice of primary source changes no
classification in this universe — it only makes the sourcing stricter and
auditable.

All 503 constituents resolved to a sourced GICS record; **no sector was
assigned by hand or inferred**. Had any constituent failed to resolve, the
builder would have excluded it and listed it under `unresolved_excluded` in the
provenance JSON rather than guessing.

## Sector distribution (GICS, 11 sectors)

| Sector | n |
|---|---|
| Industrials | 80 |
| Financials | 76 |
| Information Technology | 70 |
| Health Care | 60 |
| Consumer Discretionary | 48 |
| Consumer Staples | 36 |
| Utilities | 31 |
| Real Estate | 31 |
| Materials | 26 |
| Communication Services | 23 |
| Energy | 22 |

Note that these are the **official GICS sector names** (11 sectors). The
original 47-stock thesis universe uses 10 informal sector labels of its own
(`Technology`, `Healthcare`, …) and has no `Materials` bucket. The two
universes are kept separate and are never merged.

## Ticker notation

GICS/Wikipedia write share classes with a dot (`BRK.B`, `BF.B`); Yahoo Finance
expects a hyphen (`BRK-B`, `BF-B`). The canonical symbol is kept in `ticker`
and the download form in `yahoo_ticker`. Two constituents are affected.

## Regenerating this file

```bash
git clone https://github.com/fja05680/sp500 /tmp/sp500
git -C /tmp/sp500 fetch --depth=1000 origin master     # needed for the dated commits
python tools/build_sp500_universe.py --repo /tmp/sp500
```

`--check` compares the committed CSV against what the source produces without
rewriting it.

## Limitations to state in the write-up

**1. Conditional on end-of-period membership.**

> The extension uses a point-in-time S&P 500 constituent universe observed at
> the end of the sample period. Consequently, the analysis is conditional on
> end-of-period index membership and should not be interpreted as a
> survivorship-bias-free representation of the S&P 500 throughout 2019–2025.

**2. Classification is a single snapshot, not a history.** A company
reclassified into a different GICS sector *during* 2019–2025 is labelled by the
sector recorded in the snapshot for the whole period. Since sector metadata is
used only as an external reference *after* the network is built, and never as
an input to graph construction, this affects the interpretation of a handful of
names rather than the structure of the network itself.
