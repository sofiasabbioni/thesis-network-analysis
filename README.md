# Analysis of Stock Correlation Networks through Graph Traversal Algorithms

Complete, reproducible Python implementation supporting the bachelor thesis
*"Analysis of Stock Correlation Networks through Graph Traversal Algorithms"*
(Computer Science applied to Finance).

The project downloads historical equity prices, estimates the Pearson
correlation matrix of daily log returns, builds correlation graphs, and studies
their structure with **BFS**, **DFS**, **connected components** and a
**Minimum Spanning Tree**. It exports every table, figure and text report that
Chapters 3 and 4 of the thesis need.

> **Scope.** This is an *exploratory, descriptive* study of an estimated
> dependence structure. It does **not** predict returns and does **not**
> produce investment advice. The optional investment section (§15-16 below)
> is explicitly framed as diversification/risk *interpretation*, never as a
> trading strategy.

---

## 1. Installation

Requires **Python 3.9 or newer**.

```bash
# 1. clone / open the project folder
cd thesis-network-analysis

# 2. (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install the dependencies
pip install -r requirements.txt
```

`requirements.txt` pins only lower bounds:
`pandas`, `numpy`, `yfinance`, `networkx`, `matplotlib`, `seaborn`, `scipy`.
No machine-learning or deep-learning library is used anywhere in the project.

---

## 2. Important: Synthetic Mode vs Real Thesis Results

**Read this before generating anything you intend to put in the thesis.**

The project has two modes, and they are kept strictly apart.

| | Real mode | Synthetic mode (`--synthetic`) |
|---|---|---|
| Prices | Downloaded from Yahoo Finance | **Simulated** by a two-factor model |
| Purpose | The thesis | Checking that the pipeline runs offline |
| Outputs go to | `outputs/` | `outputs_synthetic/` |
| Valid for the thesis | Yes | **Never** |

1. **`python main.py --synthetic` is only a self-test.** It exists so that you
   can confirm the code runs - that every table and figure is produced, that
   nothing crashes - without waiting for a download or needing an internet
   connection. It invents prices from a statistical model.
2. **Synthetic results must never appear in the thesis.** They are not data
   about any real market.
3. **Before generating final results, run `python main.py --refresh`.** That
   forces a fresh Yahoo Finance download and writes real results to `outputs/`.
4. **Synthetic data can no longer be reused by accident.** A synthetic run
   marks its data (`data/raw/SYNTHETIC_DATA_WARNING.txt` plus a machine-readable
   `data/raw/data_provenance.json`). A run started *without* `--synthetic`
   detects those marks, refuses the cached panel, logs

   ```
   Synthetic cached data detected. Ignoring cached file and forcing real Yahoo Finance download.
   ```

   and downloads real data instead. The marks are cleared **only after** a real
   download has succeeded, so a failed or interrupted attempt leaves the
   warnings in place rather than promoting simulated data to "real".
5. **Only outputs generated from real Yahoo Finance data belong in the thesis.**
   Check `outputs/logs/data_source_report.txt` before quoting any number: it
   states either

   ```
   DATA SOURCE: REAL YAHOO FINANCE DATA
   ```

   or

   ```
   DATA SOURCE: SYNTHETIC SIMULATED DATA — NOT VALID FOR FINAL THESIS RESULTS
   ```

### Recommended execution order

```bash
python main.py --synthetic --no-figures    # 1. self-test only, NOT results
python main.py --refresh   --no-figures    # 2. real data, tables only
python main.py --refresh                   # 3. real data, full results
```

* **Step 1** is a self-test. It takes a few seconds, needs no internet, and only
  answers the question "does the pipeline run end to end?". Its outputs land in
  `outputs_synthetic/` and are not thesis material.
* **Step 2** is the first *real* run. `--no-figures` skips figure rendering, so
  you find out quickly whether the download worked and whether the numerical
  tables look sensible, before spending time on 31 plots.
* **Step 3** produces the complete set of real results in `outputs/`.

Steps 2 and 3 both write real data; step 3 simply adds the figures. If step 2
looks wrong (missing tickers, an unexpected date range), fix that first - there
is no point rendering figures from a data set you are going to replace.

### If you already ran synthetic mode before this safety fix

Earlier versions of the project wrote synthetic outputs into `outputs/`. If your
`outputs/` folder may contain results from such a run, delete its contents once
before step 2:

```bash
rm -rf outputs/figures/* outputs/tables/* outputs/logs/*
```

A real run rewrites every file it produces, but this guarantees no stale
synthetic file survives.

---

## 3. Running the project

```bash
python main.py
```

That single command runs the whole pipeline (roughly 30-60 seconds once the
data is downloaded) and writes everything to `data/` and `outputs/`.

### Useful options

| Command | Effect |
|---|---|
| `python main.py` | Full run. Reuses `data/raw/adjusted_close_prices.csv` if it exists **and is real**. |
| `python main.py --refresh` | Ignore the cache and download from Yahoo Finance again. Use this for the thesis. |
| `python main.py --no-figures` | Validation run: download and tables only, no figures. |
| `python main.py --synthetic` | **Self-test only.** Simulated prices, no internet; writes to `outputs_synthetic/`. |
| `python main.py --start 2020-01-01 --end 2024-12-31` | Different sample period. |
| `python main.py --main-threshold 0.6 --source MSFT` | Different detailed threshold / BFS source. |
| `python main.py --thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8` | Different threshold grid. |
| `python main.py --no-figures` | Numbers only (fast iteration). |
| `python main.py --no-portfolio` | Skip the optional portfolio illustration. |

**About `--synthetic`.** It generates a price panel from a market-factor +
sector-factor model so that the pipeline can be verified end-to-end without a
network connection. Such a run writes its results to `outputs_synthetic/`
(never to `outputs/`), marks the data it leaves in `data/` as simulated, and
drops `outputs_synthetic/logs/SYNTHETIC_OUTPUTS_WARNING.txt` beside the results.
A later run without the flag detects those marks and refuses to reuse the data.
See §2 above. **Never report numbers from a synthetic run in the thesis.**

**Validating quickly with `--no-figures`.** Rendering the 31 figures is the
slowest part of a run. `--no-figures` produces the download report and all 60
CSV tables and skips the plots, which is the fastest way to check that the data
and the numbers are right before committing to a full run:

```bash
python main.py --synthetic --no-figures    # does the pipeline run at all?
python main.py --refresh   --no-figures    # did the real download work?
```

### If the download fails

`yfinance` occasionally rate-limits or a ticker is temporarily unavailable. The
code retries the batch call three times with exponential backoff, then retries
each missing ticker individually, and finally continues with whatever it has,
reporting exactly which tickers were dropped. If nothing at all can be
downloaded (no internet, corporate proxy, blocked host), the run stops with a
clear message and suggests `--synthetic` for a pipeline check.

---

## 4. Project structure

```
thesis-network-analysis/
├── main.py                      # entry point: runs the 14 pipeline stages
├── config.py                    # ALL parameters live here
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── adjusted_close_prices.csv
│   │   ├── data_provenance.json          # real or synthetic? when? which tickers?
│   │   └── SYNTHETIC_DATA_WARNING.txt    # present only while the data is simulated
│   └── processed/
│       ├── clean_adjusted_close_prices.csv
│       └── log_returns.csv
│
├── outputs/                     # REAL results only
│   ├── figures/                 # 31 PNG figures at 300 DPI
│   ├── tables/                  # 60 CSV tables
│   └── logs/                    # run log, data_source_report.txt, 4 reports
│
├── outputs_synthetic/           # created only by --synthetic runs; never
│                                # thesis material (see README section 2)
│
└── src/
    ├── data_collection.py       # stage 1  - download adjusted closes
    ├── preprocessing.py         # stage 2  - align, clean, log returns
    ├── correlation.py           # stage 3  - Pearson matrix + description
    ├── graph_construction.py    # stages 4-5 - complete & threshold graphs
    ├── metrics.py               # stages 6-7 - node metrics, components
    ├── traversal_analysis.py    # stages 8-9 - BFS, DFS, cut vertices, bridges
    ├── mst_analysis.py          # stage 10 - Mantegna distance + MST
    ├── visualisation.py         # stage 11 - every figure
    ├── interpretation.py        # stages 12/14 - framework + summary report
    ├── portfolio.py             # stage 13 - illustrative portfolios
    └── utils.py                 # paths, logging, table export
```

The requested `interpretation.py` was split in two: the classification framework
and the master report stayed there, while the illustrative portfolio comparison
moved to its own `portfolio.py`. Everything else follows the layout above.

---

## 5. What each module does

### `config.py` - the experimental set-up
Every parameter of the study in one place: sample period, the 47-stock
universe grouped by sector, cleaning rules, thresholds, the main threshold, the
BFS source, MST options, figure style and the random seed. Editing this file
changes the entire experiment; nothing is hard-coded elsewhere.

The universe is **47 large-cap S&P 500 stocks across 10 GICS sectors**
(4-5 per sector). It is balanced by sector on purpose: an unbalanced universe
would mechanically produce a network dominated by the over-represented sector,
which would confound the sectoral-clustering analysis.

### `src/data_collection.py` - stage 1
Downloads **adjusted** closing prices. Adjusted rather than raw closes because
the raw price jumps mechanically on ex-dividend and split dates; those jumps
would appear as large returns and contaminate every correlation. Handles the
several column layouts `yfinance` has used over its versions, retries, recovers
individual failures, caches the panel and reports requested/downloaded/failed
tickers, the date range and the number of trading days.

It also owns the **provenance guards** described in §2: it writes the synthetic
markers and `data_provenance.json`, refuses a cached panel that either of them
flags as simulated, clears those marks only after a real download has reached
disk, and writes `outputs/logs/data_source_report.txt` at the end of a run.

### `src/preprocessing.py` - stage 2
Aligns the trading calendar, drops stocks with more than 5% missing prices,
forward-fills gaps of at most 5 days, drops the remaining incomplete rows, then
computes `r_t = ln(P_t / P_{t-1})`. Exports per-stock descriptive statistics
(mean, standard deviation, min, max, median, skewness, excess kurtosis,
annualised mean and volatility, number of observations).

Three choices are documented in the module docstring for Chapter 3: *why*
aligned calendars are necessary (otherwise each correlation is estimated on a
different sample), *why* missing values matter (forward-filling manufactures
zero returns that bias correlations towards zero), and *why* log returns
(additivity over time, approximate stationarity, symmetry).

### `src/correlation.py` - stage 3
Estimates the N x N Pearson matrix, flattens it into a tidy pair table, computes
the sector-average correlation matrix, and writes a textual interpretation
covering the overall level of co-movement, the strongest and weakest pairs, the
within-sector versus cross-sector gap, and the stocks most and least correlated
with the rest of the universe.

### `src/graph_construction.py` - stages 4-5
Builds the **complete weighted graph** (all N(N-1)/2 pairs; kept as a reference
object) and one **threshold graph** per value of `THRESHOLDS`, keeping an edge
when `|C_ij| >= tau`. Node attributes: ticker, sector, company. Edge attributes:
correlation, |correlation|, weight, Mantegna distance, same-sector flag.

For each graph it computes nodes, edges, density, average degree, average
strength, clustering (plain and weighted), transitivity, connected components,
giant component size and share, isolated nodes, mean edge correlation, the share
of intra-sector edges, sector assortativity, degree assortativity, and the
average path length and diameter *inside the giant component*.

### `src/metrics.py` - stages 6-7
Per node and per threshold: degree, strength, degree/betweenness (plain and
distance-weighted)/closeness/eigenvector centrality, clustering coefficient,
mean neighbour correlation, component id and size, isolation flag, and a
**composite centrality** (the mean of four min-max-normalised centralities,
used only for labelling). Also the "top 10" tables and the connected-component
and sector-composition tables.

### `src/traversal_analysis.py` - stages 8-9
BFS and DFS are each implemented **twice**: a transparent from-scratch version
for Chapter 3, and the NetworkX routine used to validate it. The validation
result is printed and stored (`PASS`/`FAIL` per check).

* **BFS** (`custom_bfs`): FIFO queue, neighbours visited alphabetically for
  determinism, `O(|V|+|E|)`. Returns traversal order, shortest-path distances,
  and the BFS tree. Validated against `nx.single_source_shortest_path_length`
  (BFS distances are canonical, so an exact match is expected).
* **DFS** (`custom_dfs`, `custom_dfs_forest`): explicit LIFO stack rather than
  recursion. Restarting on every unvisited node yields the DFS forest, whose
  trees *are* the connected components. Validated for the visited set, the
  absence of repeats, the defining pre-order property, and against NetworkX's
  components. A DFS pre-order is **not** unique, so the exact sequence is not
  required to match - that is stated in the code and worth a sentence in the
  thesis.
* **Articulation points** and **bridges**, each with the *magnitude* of the
  damage their removal would cause (how many components appear, which stocks
  get detached, how large the two sides of a bridge are).

### `src/mst_analysis.py` - stage 10
Mantegna's transformation `d_ij = sqrt(2(1 - C_ij))` - a proper metric, which is
what makes a minimum-weight spanning tree meaningful. By default the **signed**
correlation enters the formula, so anti-correlated stocks are far apart;
`config.MST_USE_ABS_CORRELATION = True` switches to `|C_ij|` for a robustness
check. The MST is computed with Kruskal's algorithm and described by node
degrees, leaves, betweenness, eccentricity, depth from the hub, total and
normalised tree length, diameter, and - the key result - the share of tree links
that join two stocks of the same sector, compared with the share a random tree
would achieve.

### `src/visualisation.py` - stage 11
All 29 figures, 300 DPI, one fixed sector-colour mapping used everywhere.
Two non-obvious pieces of machinery are worth knowing about, because they are
what makes the network figures legible:

* **Component-packed layout.** A plain spring layout applied to a disconnected
  graph pushes each component apart and shrinks it to a dot. Instead every
  component is laid out on its own, overlapping nodes are relaxed apart, and the
  components are packed on concentric rings around the giant component.
* **Adaptive node sizing.** Marker size is derived from the space the layout
  actually gives each node (the 25th-percentile nearest-neighbour distance
  converted to points), then modulated by degree. Dense components get small
  markers, sparse ones get large markers, and neither overlaps.

### `src/interpretation.py` - stages 12 and 14
Classifies every stock as central / peripheral / bridge / clustered and writes
one descriptive sentence per stock, then assembles the master report
`outputs/logs/summary_interpretation.txt`. Every artefact carries the disclaimer
verbatim.

### `src/portfolio.py` - stage 13
Portfolio A takes the most representative stock of each sector; Portfolio B
greedily picks stocks that are far apart in the network (peripheral, different
components, minimising the *maximum* pairwise correlation); Portfolio C is the
equally weighted universe as a reference. Reports average pairwise correlation,
annualised return, CAGR, annualised volatility, Sharpe ratio (rf = 0) and
maximum drawdown - all **in-sample, equally weighted, no costs**, i.e. an
illustration of how network structure maps onto basket correlation, not a
backtest.

---

## 6. Generated outputs

### Tables (`outputs/tables/`)

| File | Contents |
|---|---|
| `return_summary_statistics.csv` | Per-stock descriptive statistics of daily log returns |
| `correlation_matrix.csv` | The N x N Pearson matrix |
| `correlation_pairs.csv` | One row per stock pair (correlation, sectors, same-sector flag) |
| `sector_correlation_matrix.csv` | Average correlation within and between sectors |
| `full_graph_summary.csv` | Statistics of the complete correlation graph |
| `threshold_network_metrics.csv` | **One row per threshold**: edges, density, degree, clustering, components, giant component, isolated nodes, assortativity, path length |
| `node_metrics_threshold_X.csv` | All node-level metrics, one file per threshold |
| `top_degree_stocks_threshold_X.csv` | Top 10 by degree |
| `top_central_stocks_threshold_X.csv` | Top 10 by composite centrality |
| `top_peripheral_stocks_threshold_X.csv` | Bottom 10 by composite centrality |
| `isolated_stocks_threshold_X.csv` | Stocks with degree 0 |
| `connected_components_threshold_X.csv` | Component id and size per stock |
| `component_sector_composition_threshold_X.csv` | Sector composition of each component |
| `bfs_distances_SOURCE_threshold_X.csv` | BFS distance, visit order, parent, reachability |
| `bfs_layers_SOURCE_threshold_X.csv` | Size and membership of each BFS layer |
| `bfs_reachability_by_threshold_SOURCE.csv` | How the reachable set shrinks with the threshold |
| `dfs_order_threshold_X.csv` | DFS visit order, parent, tree root, component |
| `articulation_points_threshold_X.csv` | Cut vertices + what their removal would detach |
| `bridges_threshold_X.csv` | Cut edges + the size of the two sides |
| `mst_node_metrics.csv` | MST degree, leaf flag, betweenness, eccentricity, depth |
| `mst_edges.csv` | The 46 retained links with correlation and distance |
| `mst_summary.csv` | Tree-level statistics |
| `mst_sector_branches.csv` | Internal/external links per sector |
| `investment_interpretation_framework.csv` | Per-stock role + one-sentence interpretation |
| `investment_framework_summary.csv` | Counts and composition per role |
| `portfolio_comparison.csv` | Risk/return statistics of portfolios A, B, C |
| `portfolio_holdings.csv` | Constituents and why each was selected |

### Figures (`outputs/figures/`)

| File | Contents |
|---|---|
| `correlation_heatmap.png` | Correlation matrix, stocks grouped by sector, sector separators |
| `sector_correlation_heatmap.png` | Sector-average correlations |
| `correlation_distribution.png` | Histogram of pairwise correlations with the thresholds marked, and same-sector vs cross-sector densities |
| `threshold_network_threshold_X.png` | The network at each threshold, coloured by sector |
| `connected_components_threshold_X.png` | Same network coloured by connected component |
| `degree_distribution_threshold_X.png` | Degree histogram + per-stock degree ranking |
| `bfs_tree_SOURCE_threshold_X.png` | BFS tree drawn as explicit distance layers |
| `bfs_distance_map_SOURCE_threshold_X.png` | The whole network coloured by BFS distance |
| `bfs_reachability_SOURCE.png` | Reachable share and mean distance vs threshold |
| `dfs_tree_threshold_X.png` | DFS forest over the network, cut vertices and bridges highlighted |
| `minimum_spanning_tree.png` | MST, colour = sector, size = degree, red links = cross-sector |
| `threshold_edges.png` | Edges vs threshold |
| `threshold_density.png` | Density vs threshold |
| `threshold_components.png` | Number of components vs threshold |
| `threshold_giant_component.png` | Giant component share vs threshold |
| `threshold_clustering.png` | Average clustering vs threshold |
| `threshold_summary_panel.png` | All six threshold curves in one panel |
| `portfolio_cumulative_returns.png` | Cumulative value of the three portfolios |
| `portfolio_risk_comparison.png` | Bar comparison of the headline risk statistics |

### Reports (`outputs/logs/`)

| File | Contents |
|---|---|
| `data_source_report.txt` | **Real or synthetic?** Data source, dates, tickers requested/downloaded/removed, observations, timestamp |
| `run_log.txt` | Full timestamped log of the run |
| `correlation_interpretation.txt` | Narrative reading of the correlation matrix |
| `mst_interpretation.txt` | Narrative reading of the MST |
| `summary_interpretation.txt` | **Master report**: set-up, correlation structure, threshold response, main network, components, BFS, DFS, MST, investment framework, disclaimer, limitations |

---

## 7. How to interpret the main outputs

**Average correlation and its dispersion.** A high positive average reflects the
common market factor. The *dispersion* around it is what the network exploits:
it is the group-specific part of the dependence structure.

**Threshold curves.** Raising `tau` removes weak links: edges and density fall,
the number of components rises and the giant component shrinks. The interesting
range is where the giant component starts to break up - that is where structure
rather than the market factor becomes visible.

**Degree and centrality.** A high-degree stock co-moves strongly with much of
the market, so a position in it carries mostly systematic exposure. A
high-betweenness stock is a *connector* between regions - a different property,
and the two do not always coincide.

**Connected components.** They answer "does the market split into separate
blocks?". Watch whether the blocks that survive at high thresholds coincide with
industry groups.

**BFS distances.** The minimum number of strong correlations needed to travel
from the source to another stock. Distance 1 = direct co-movement; larger
distances = the relationship exists only through intermediaries. Topological
proximity inside an estimated structure - descriptive, not predictive.

**Articulation points and bridges.** Structural chokepoints: positions through
which co-movement passes between otherwise weakly related parts of the market.
A property of an estimated network over one sample - **not** a claim that these
stocks are good investments.

**MST.** The N-1 strongest links that connect every stock without a cycle, with
no threshold at all. That a large majority of them join same-sector stocks -
while no sector information entered the computation - is the clearest evidence
in the study that industry membership is encoded in return co-movement.

---

## 8. How to use these outputs in the thesis

### Chapter 3 - Methodology and Implementation

| Section | Use |
|---|---|
| Data and sample | `config.py` (universe, period), the download report in `run_log.txt`, `data/raw/adjusted_close_prices.csv` |
| Preprocessing | `preprocessing.py` docstrings (aligned calendars, missing values, log returns), `return_summary_statistics.csv` |
| Correlation matrix | `correlation.py`, the definition of `C_ij`, `correlation_matrix.csv` |
| Graph construction | `graph_construction.py`: the `G = (V, E, w)` formalism, node/edge attributes, the threshold rule and the justification of the `absolute` mode |
| BFS implementation | `custom_bfs` - quote the code, the FIFO argument, `O(|V|+|E|)`, and the validation against NetworkX |
| DFS implementation | `custom_dfs` / `custom_dfs_forest` - the LIFO stack, the forest = components argument, and why a DFS pre-order is not unique |
| MST construction | `mst_analysis.py`: why `d = sqrt(2(1-C))` is a metric, the treatment of negative correlations, Kruskal |
| Reproducibility | `RANDOM_SEED`, the version block in `summary_interpretation.txt`, the `--synthetic` self-test |

### Chapter 4 - Results and Analysis

| Section | Use |
|---|---|
| 4.1 Correlation structure | `correlation_heatmap.png`, `sector_correlation_heatmap.png`, `correlation_distribution.png`, `correlation_interpretation.txt` |
| 4.2 The network at the main threshold | `threshold_network_threshold_0_5.png`, `degree_distribution_threshold_0_5.png`, `node_metrics_threshold_0_5.csv`, the top-10 tables |
| 4.3 Threshold sensitivity | `threshold_network_metrics.csv` and the six threshold figures (or the single `threshold_summary_panel.png`) |
| 4.4 Connected components | `connected_components_threshold_0_5.png` (and 0.4 / 0.6), `connected_components_threshold_X.csv`, `component_sector_composition_threshold_X.csv` |
| 4.5 BFS | `bfs_tree_AAPL_threshold_0_5.png`, `bfs_distance_map_...png`, `bfs_reachability_AAPL.png`, `bfs_distances_...csv`, `bfs_layers_...csv` |
| 4.6 DFS, cut vertices and bridges | `dfs_tree_threshold_0_5.png`, `dfs_order_...csv`, `articulation_points_...csv`, `bridges_...csv` |
| 4.7 Minimum Spanning Tree | `minimum_spanning_tree.png`, `mst_node_metrics.csv`, `mst_edges.csv`, `mst_sector_branches.csv`, `mst_interpretation.txt` |
| 4.8 Exploratory investment interpretation | `investment_interpretation_framework.csv`, `investment_framework_summary.csv` |
| 4.9 Illustrative portfolios | `portfolio_comparison.csv`, `portfolio_holdings.csv`, `portfolio_cumulative_returns.png`, `portfolio_risk_comparison.png` |

### Chapter 5 - Discussion and Conclusion

Section 9 of `summary_interpretation.txt` lists the limitations in a form ready
to be expanded:

* Pearson correlation captures **linear** dependence only and is sensitive to
  the fat tails of daily returns;
* one matrix estimated over several years averages over calm and crisis
  regimes - correlations are not stable;
* the threshold is a modelling choice, which is why every result is reported
  across a grid rather than at one value;
* the universe is 47 large-cap US stocks balanced by sector;
* correlation is not causation: an edge records co-movement, not influence;
* nothing here forecasts returns.

Natural extensions to propose: rolling-window networks (crisis vs calm),
partial correlations to strip out the market factor, Planar Maximally Filtered
Graphs as a richer filter than the MST, community detection beyond connected
components, and - clearly labelled as a *possible* future extension - using
network features as inputs to a predictive model.

---

## 9. Reproducibility notes

* Every stochastic component (graph layouts, the synthetic self-test) is seeded
  from `config.RANDOM_SEED`, so re-running on the same data reproduces
  identical figures.
* `data/` and `outputs/` are git-ignored, because a repository should never
  contain results whose provenance (a real download or a `--synthetic`
  self-test) cannot be checked from the commit alone. When the thesis is
  finalised, freeze the exact data set behind it with
  `git add -f data/raw/adjusted_close_prices.csv`.
* `data/raw/adjusted_close_prices.csv` is the immutable input of every later
  stage; `--refresh` replaces it.
* `outputs/logs/run_log.txt` records the full configuration, every dropped
  ticker, and every file written.
* `outputs/logs/data_source_report.txt` certifies which data source the run used.
  It is written last, so its presence also certifies that the run completed.
* `data/raw/data_provenance.json` records when the cached panel was obtained and
  whether it is real or simulated.
* `summary_interpretation.txt` includes the exact Python and library versions
  used for the run.

---

## 10. Disclaimer

This framework is exploratory and supports diversification and risk
interpretation. It does not constitute investment advice and does not predict
future returns.
