# Large-network extension — supplementary to the 47-stock thesis analysis

This document covers **only** the optional extension. The original thesis
analysis, its methodology and its results are documented in `README.md` and are
untouched by everything described here.

---

## 1. What the extension does

It re-runs the thesis methodology, unchanged, on **503 S&P 500 constituents**
instead of 47, and then uses GICS sector metadata as an **external reference**
to ask whether stocks tend to connect to stocks of their own sector.

It is a scalability and sector-structure robustness check. It does not replace,
revise or invalidate the 47-stock experiment.

**Not done here, deliberately:** portfolio construction, community detection,
rolling windows, machine learning, predictive modelling, or any correlation
measure other than Pearson.

---

## 2. How the original analysis is protected

| Protection | Mechanism |
|---|---|
| Different data files | `config.use_large_network_paths()` rebinds every data path to `data/large/` before any file is opened |
| Different output tree | …and every output path to `outputs_large/` |
| Different universe | `config.apply_universe()` is called **only** by the extension; `main.py` always sees the 47 stocks in `config.STOCKS` |
| Self-test isolated again | `--synthetic` redirects to `data/large_synthetic/` and `outputs_large_synthetic/` |
| Traversal code untouched | `src/traversal_analysis.py` is byte-identical to the thesis version |
| Thesis command unchanged | `python main.py --refresh` runs exactly the code it always ran |

The only edit to an existing analysis module is an **optional `chunk_size`
argument** in `src/data_collection.py`. Its default is `None`, which takes the
identical code path the 47-stock run has always used. See the CHANGELOG.

---

## 3. Commands

### The original thesis analysis (unchanged)

```bash
python main.py --refresh
```

Writes to `data/` and `outputs/` exactly as before.

### The large-network extension

```bash
python large_network_extension.py --refresh
```

Writes to `data/large/` and `outputs_large/`, plus the two committed network
files in `network_exports/`. Expect roughly 10–25 minutes end to end. The
compute stages take about 7½ minutes at 503 stocks (measured on the offline
self-test, whose simulated network is denser than the real one and therefore a
worst case); the rest is the Yahoo Finance download of 503 symbols in batches
of 100. The slowest single step is the average-path-length and diameter
calculation inside the shared `graph_summary`, which is O(V·E) and is retained
so that the large-network table uses exactly the same metric definitions as the
thesis table.

Useful variants:

```bash
python large_network_extension.py --synthetic          # offline self-test, simulated prices
python large_network_extension.py --no-figures         # tables only
python large_network_extension.py --limit 120          # quick check, NOT a real run
python large_network_extension.py --chunk-size 50      # smaller download batches
```

### The network export for the original 47-stock thesis network

```bash
python export_thesis_network.py
```

Reads `outputs/tables/correlation_matrix.csv` (produced by the thesis run),
rebuilds the τ = 0.5 graph with the same code, cross-checks it against the
thesis output tables, and writes the two files into `network_exports/`. It
downloads nothing and writes nothing into `data/` or `outputs/`.

### Regenerating the constituent list

```bash
git clone https://github.com/fja05680/sp500 /tmp/sp500
git -C /tmp/sp500 fetch --depth=1000 origin master
python tools/build_sp500_universe.py --repo /tmp/sp500
```

---

## 4. The stock universe

503 constituents, from a **point-in-time membership snapshot dated
2025-12-22** — the last index change on or before the thesis sample end of
2025-12-31 — with GICS sector and sub-industry attached from a dated capture of
the Wikipedia constituents table.

> **How to describe this universe.** The extension uses a point-in-time S&P 500
> constituent universe observed at the end of the sample period. Consequently,
> the analysis is conditional on end-of-period index membership and should not
> be interpreted as a survivorship-bias-free representation of the S&P 500
> throughout 2019–2025.

Full provenance, including commit hashes and the five tickers whose metadata
came from the post-sample fallback snapshot, is in
[`data/reference/CONSTITUENTS_SOURCE.md`](data/reference/CONSTITUENTS_SOURCE.md).
No sector was assigned by hand.

Note the universe uses the **11 official GICS sector names**
(`Information Technology`, `Health Care`, …, plus `Materials`), which differ
from the 10 informal labels of the thesis universe (`Technology`,
`Healthcare`, …). The two universes are never merged.

---

## 5. Methodology — identical to the thesis

| Choice | Value | Source |
|---|---|---|
| Sample period | 2019-01-02 → 2025-12-31 | `config.START_DATE` / `END_DATE` |
| Prices | Adjusted close | `src/data_collection.py` |
| Calendar alignment | One common calendar, drop-then-ffill | `src/preprocessing.py`, unchanged |
| Max missing fraction | 0.05 | `config.MAX_MISSING_FRACTION` |
| Max forward-fill | 5 days | `config.MAX_FORWARD_FILL_DAYS` |
| Returns | Daily logarithmic | `src/preprocessing.py` |
| Correlation | Pearson | `src/correlation.py` |
| Threshold grid | {0.3, 0.4, 0.5, 0.6, 0.7} | `config.THRESHOLDS` |
| Edge rule | \|C_ij\| ≥ τ | `config.EDGE_FILTER_MODE = "absolute"` |
| Main threshold | 0.5 | `config.MAIN_THRESHOLD` |

**Expected consequence of scaling, not a change of method.** The 5%-missing
rule removes any company that was not listed for the whole 2019–2025 window —
IPOs, spin-offs and recent index additions. That rule was applied unchanged and
was *not* relaxed to retain more stocks; the exact count and the names dropped
at each stage are reported in `outputs_large/logs/large_data_source_report.txt`.

**Sector metadata is never an input to graph construction.** An edge exists if
and only if \|C_ij\| ≥ τ. Sector labels are attached to nodes for description
and are read only after the graphs exist.

---

## 6. Output files

All under `outputs_large/` unless stated.

### `tables/`

| File | Contents |
|---|---|
| `large_threshold_network_metrics.csv` | **Phase 5 table.** One row per τ: `threshold, nodes, edges, density, avg_degree, avg_clustering, components, giant_component_size, giant_component_share, isolated_nodes, same_sector_edge_share, sector_assortativity` |
| `large_threshold_network_metrics_full.csv` | The same graphs in the thesis's own 25-column format (path length, diameter, transitivity, degree assortativity, …) |
| `large_node_sector_alignment.csv` | **Phase 6 table.** One row per non-isolated node at τ = 0.5: official sector, degree, same/cross-sector neighbour counts, same-sector share, its chance benchmark and their ratio, dominant neighbour sector and its share, the full tie information (`dominant_neighbor_sectors_tied`, `n_dominant_tied_sectors`, `dominant_sector_is_tied`, `official_in_dominant_tie`), `dominant_equals_official`, and `meets_cross_sector_rule`. Ranked ascending by `same_sector_share_ratio`, so its first rows are the least sector-aligned stocks whether or not any of them meets the rule |
| `large_sector_alignment_summary.csv` | Overall: same-sector edge share vs its random benchmark, sector assortativity, mean/median same-sector share, proportion of stocks whose own sector is also dominant among neighbours, and the number of dominant-sector ties (and how many include the official sector) |
| `large_sector_alignment_by_sector.csv` | The same statistics per GICS sector |
| `large_cross_sector_oriented_nodes.csv` | **Phase 7 table.** Ranked list of sector-atypical network positions |
| `large_correlation_matrix.csv` | The N×N Pearson matrix |
| `large_sector_correlation_matrix.csv` | Mean correlation within and between the 11 GICS sectors |

### `figures/`

`large_threshold_sensitivity.png`, `large_giant_component_vs_tau.png`,
`large_sector_assortativity_vs_tau.png`, `large_same_sector_share_hist.png`,
`large_sector_alignment_by_sector.png`, `large_network_by_sector.png`
(coloured by sector, **labels omitted** — hundreds of tickers cannot be read).

### `logs/`

`large_data_source_report.txt` — the full provenance record: constituent-list
source and snapshot date, tickers requested / downloaded / retained / excluded
with the reason for each exclusion, observation counts, exact dates, real vs
synthetic, and every configuration parameter.

`large_network_run_log.txt` — the complete run log.

### `network_exports/` (committed to git)

| File | Contents |
|---|---|
| `network_nodes_threshold_0_5.csv` | **Original 47-stock thesis network.** `ticker, company, sector, degree, component, is_isolated, is_articulation_point, network_role, is_mst_leaf` |
| `network_edges_threshold_0_5.csv` | `source, target, correlation, abs_correlation, mantegna_distance, same_sector` |
| `large_network_nodes_threshold_0_5.csv` | The large network, same columns |
| `large_network_edges_threshold_0_5.csv` | The large network, same columns |

Isolated nodes are present in the node files: a stock with no edge at τ = 0.5
is a result, not a row to drop.

---

## 7. How "cross-sector-oriented" is defined

A node is flagged when **all** of these hold:

1. `degree ≥ 5` — below this a share of neighbours is too coarse to read;
2. `same_sector_share ≤ 1.0 × its chance benchmark`, where the benchmark for a
   stock in sector *s* is `(n_s − 1) / (N − 1)`, the same-sector share expected
   if its links were placed at random;
3. there is an **unambiguous** most frequent sector among its neighbours — no
   tie for the maximum;
4. that dominant sector is **not** its own.

### Dominant-sector ties

`dominant_neighbor_sector` shows the alphabetically first of the sectors tied
for the maximum, purely so that one column can be displayed deterministically.
**The alphabetical pick never carries analytical weight.**

- `dominant_neighbor_sectors_tied` lists *every* sector attaining the maximum,
  with `n_dominant_tied_sectors` and the boolean `dominant_sector_is_tied`;
- `official_in_dominant_tie` records whether the stock's own GICS sector is
  among those tied leaders;
- `dominant_equals_official` is **true whenever the official sector is among
  the tied leaders**, not only when it wins the tie-break;
- condition 3 above excludes every tied node from selection.

So a stock whose neighbours split 3–3 between its own sector and another is
never called cross-sector-oriented on the strength of the alphabet, and neither
is one whose neighbours split evenly between two sectors that are both foreign
to it — in that case the neighbourhood has no single orientation to report. The
tie information stays in the exported CSV rather than being filtered away
silently, and the summary reports how many ties occurred and how many included
the official sector.

Condition 2 is a ratio, not an absolute cut-off, because sector sizes in this
universe differ by nearly a factor of four (22 Energy names vs 80 Industrials).
A flat `share ≤ 0.20` would flag a small-sector stock as ordinary and a
large-sector stock as atypical purely because of how big their sectors are.
Both the observed share and the benchmark are exported, so the ratio can be
recomputed by hand from the neighbouring columns.

No composite anomaly score is constructed. Each condition is a single
observable, and the ranking is on the ratio itself.

A strict rule can legitimately select **nothing** — a network in which every
stock is at least as sector-aligned as chance is a finding, not a failed query.
`large_node_sector_alignment.csv` is therefore always ranked by the ratio and
carries a `meets_cross_sector_rule` column, so the least sector-aligned names
are readable from its first rows even when
`large_cross_sector_oriented_nodes.csv` is empty. The run log says so
explicitly when that happens.

**Terminology.** These are *cross-sector-oriented nodes* / *sector-atypical
network positions*, never "misclassified". GICS classifies a company by its
business activity; an edge here records co-movement of returns over one sample
period. The two disagreeing is information about the sample, not an error in
the classification. Sector metadata is an approximate external reference, not a
literal statistical ground truth.

---

## 8. Quality control

`large_network_extension.py` runs these automatically and aborts on failure:

- node counts coherent across all thresholds (thresholding removes edges, never nodes);
- edge counts in the metrics table match the constructed graphs;
- component sizes sum to N at every threshold;
- isolated nodes still represented in the exported node file;
- same-sector + cross-sector neighbours = degree for every node;
- alignment table covers exactly the non-isolated nodes;
- `same_sector_edge_share` equals the share carried by the graph's own edges;
- **the edge set is unchanged when sector labels are randomly permuted between
  stocks** — an empirical proof that sector metadata cannot have influenced
  graph construction. If a label ever leaked into the edge rule, permuting the
  labels would move at least one edge.

`export_thesis_network.py` additionally cross-checks the rebuilt 47-stock
network against the original thesis output tables (component membership,
degrees, articulation points, MST leaves, network roles, and the τ = 0.5 row of
`threshold_network_metrics.csv`).

Synthetic and real results can never mix: a `--synthetic` run writes to a
separate data tree *and* a separate output tree, stamps both with warning
markers, and a real run refuses to reuse a cached panel marked synthetic.
