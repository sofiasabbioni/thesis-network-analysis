"""
config.py
=========

Central configuration for the thesis project

    "Analysis of Stock Correlation Networks through Graph Traversal Algorithms"

Every parameter that a reader of the thesis might reasonably want to change
(sample period, stock universe, correlation thresholds, BFS source stock, ...)
is declared here and nowhere else.  The analysis modules in ``src/`` import
this file, so a single edit here changes the whole experiment.  This keeps the
empirical work reproducible: the configuration block is literally the
"experimental setup" section of Chapter 3.

Nothing in this file performs computation; it only declares values.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# All paths are derived from the location of this file so that the project can
# be moved or cloned anywhere without editing anything.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Canonical file names for the two persistent data sets.
RAW_PRICES_FILE = os.path.join(RAW_DATA_DIR, "adjusted_close_prices.csv")
LOG_RETURNS_FILE = os.path.join(PROCESSED_DATA_DIR, "log_returns.csv")
CLEAN_PRICES_FILE = os.path.join(PROCESSED_DATA_DIR, "clean_adjusted_close_prices.csv")

ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
]

# --- Provenance guards ------------------------------------------------------
# The self-test mode (``--synthetic``) writes SIMULATED prices to
# RAW_PRICES_FILE.  Three mechanisms make sure such data can never be mistaken
# for, or silently reused as, real Yahoo Finance data:
#
#   1. a human-readable marker file next to the raw panel;
#   2. a machine-readable provenance record read by the cache loader;
#   3. a completely separate output tree, so a synthetic run can never
#      overwrite the figures and tables behind the thesis.
#
# See ``src/data_collection.py`` for the detection logic.
SYNTHETIC_DATA_MARKER = os.path.join(RAW_DATA_DIR, "SYNTHETIC_DATA_WARNING.txt")
SYNTHETIC_PROCESSED_MARKER = os.path.join(PROCESSED_DATA_DIR, "SYNTHETIC_DATA_WARNING.txt")
PROVENANCE_FILE = os.path.join(RAW_DATA_DIR, "data_provenance.json")

# Where a ``--synthetic`` run writes its figures, tables and logs.  Real runs
# always write to ``outputs/``; the two trees never mix.
SYNTHETIC_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_synthetic")
SYNTHETIC_OUTPUTS_MARKER_NAME = "SYNTHETIC_OUTPUTS_WARNING.txt"

# Exact wording used by every synthetic warning, kept here so that it is
# identical in the marker files, the logs and the reports.
SYNTHETIC_OUTPUTS_WARNING = (
    "These outputs were generated using synthetic simulated data. "
    "They must not be used as empirical thesis results."
)


def use_synthetic_output_paths() -> None:
    """Redirect every output directory to ``outputs_synthetic/``.

    Called by ``main.py`` at the very start of a ``--synthetic`` run, before any
    directory is created and before logging is configured, so that *nothing*
    produced by a self-test can land in the real ``outputs/`` tree.  The
    analysis modules read these paths from ``config`` at call time, so this one
    rebinding is enough to move the whole run.
    """
    global OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, ALL_DIRS
    OUTPUT_DIR = SYNTHETIC_OUTPUT_DIR
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    ALL_DIRS = [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]

# ---------------------------------------------------------------------------
# 2. SAMPLE PERIOD
# ---------------------------------------------------------------------------
# The default window covers a deliberately heterogeneous period: the calm
# pre-pandemic market (2019), the COVID-19 crash and recovery (2020-2021), the
# 2022 inflation/rate-hike drawdown and the subsequent recovery.  A correlation
# network estimated over such a period describes the *average* dependence
# structure of the sample; sub-period robustness is discussed in Chapter 5 as a
# limitation and a possible extension.
START_DATE = "2019-01-01"
# yfinance treats the end date as EXCLUSIVE, so an end of "2025-12-31" silently
# drops the last trading day of 2025.  Asking for the first day of the next
# year is the reliable way to obtain the complete final year.
END_DATE = "2026-01-01"

# Number of trading days per year, used to annualise means and volatilities.
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# 3. STOCK UNIVERSE
# ---------------------------------------------------------------------------
# 47 large-capitalisation S&P 500 constituents drawn from ten GICS sectors.
# The universe is intentionally *balanced by sector* (4-5 names per sector)
# rather than capitalisation-weighted: an unbalanced universe would mechanically
# produce a network dominated by whichever sector is over-represented, which
# would confound the "sectoral clustering" analysis in Chapter 4.
#
# Each ticker becomes one node of the correlation graph; the dictionary key
# becomes the node's ``sector`` attribute.
STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ADBE", "CRM"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX"],
    "Industrials": ["BA", "CAT", "GE", "HON", "UNP"],
    "Communication Services": ["GOOGL", "META", "DIS", "VZ", "T"],
    "Utilities": ["NEE", "DUK", "SO", "AEP"],
    "Real Estate": ["AMT", "PLD", "SPG"],
}

# Optional, purely cosmetic: used for the node-attribute table and for figure
# captions.  A missing entry simply falls back to the ticker itself.
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "ADBE": "Adobe",
    "CRM": "Salesforce",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo",
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "MRK": "Merck",
    "UNH": "UnitedHealth", "ABBV": "AbbVie",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "SLB (Schlumberger)", "EOG": "EOG Resources",
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "WMT": "Walmart", "COST": "Costco",
    "AMZN": "Amazon", "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike",
    "SBUX": "Starbucks",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "GE Aerospace",
    "HON": "Honeywell", "UNP": "Union Pacific",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "DIS": "Walt Disney",
    "VZ": "Verizon", "T": "AT&T",
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Company",
    "AEP": "American Electric Power",
    "AMT": "American Tower", "PLD": "Prologis", "SPG": "Simon Property Group",
}

# Flattened, order-preserving list of tickers and the inverse ticker -> sector
# map.  Both are derived, never edited by hand.
TICKERS = [t for sector_tickers in STOCKS.values() for t in sector_tickers]
TICKER_TO_SECTOR = {
    ticker: sector for sector, tickers in STOCKS.items() for ticker in tickers
}
SECTORS = list(STOCKS.keys())

# ---------------------------------------------------------------------------
# 4. DATA CLEANING RULES
# ---------------------------------------------------------------------------
# A stock is discarded if more than this fraction of its observations is
# missing after the trading calendar has been aligned.  5% of ~1 750 trading
# days is roughly 87 days, which is generous for a liquid S&P 500 name; a stock
# exceeding it is almost certainly not listed for the whole sample.
MAX_MISSING_FRACTION = 0.05

# Maximum number of consecutive days that may be forward-filled.  Short gaps
# (a local exchange holiday, a one-day trading halt) are carried forward;
# longer gaps would fabricate a run of exactly-zero returns and are dropped.
MAX_FORWARD_FILL_DAYS = 5

# A stock must have at least this many valid return observations to be kept.
MIN_OBSERVATIONS = 250

# ---------------------------------------------------------------------------
# 5. GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------
# Thresholds used to filter the complete correlation graph.  An edge (i, j)
# survives threshold tau if |C_ij| >= tau (see EDGE_FILTER_MODE below).
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# The threshold used for the "detailed" single-network analyses (BFS, DFS,
# component visualisation, investment framework).  It must be an element of
# THRESHOLDS.
MAIN_THRESHOLD = 0.5

# How an edge is selected:
#   "absolute" -> keep the edge if |C_ij| >= tau  (the specification used in
#                 this thesis: a strong negative co-movement is as informative
#                 a relationship as a strong positive one)
#   "positive" -> keep the edge if  C_ij  >= tau  (common in the econophysics
#                 literature, where only co-movement is treated as proximity)
# The choice is reported in the outputs so that Chapter 3 can justify it.
EDGE_FILTER_MODE = "absolute"

# Source node for the Breadth-First Search analysis.  If the ticker is absent
# from the final universe the code falls back to the highest-degree node and
# says so in the log.
SOURCE_STOCK = "AAPL"

# ---------------------------------------------------------------------------
# 6. MINIMUM SPANNING TREE
# ---------------------------------------------------------------------------
# Mantegna's (1999) transformation from correlation to a metric distance:
#       d_ij = sqrt( 2 * (1 - C_ij) )
# d is 0 for perfectly correlated stocks, sqrt(2) for uncorrelated stocks and 2
# for perfectly anti-correlated stocks.
#
# If MST_USE_ABS_CORRELATION is False (default) the *signed* correlation enters
# the transformation, so an anti-correlated pair is treated as maximally
# distant.  This is the standard choice and it is the one discussed in the
# thesis.  Setting it to True uses |C_ij| instead, which treats strong negative
# dependence as proximity; the flag exists so that the robustness of the MST to
# this modelling choice can be checked.
MST_USE_ABS_CORRELATION = False

# ---------------------------------------------------------------------------
# 7. INVESTMENT-INTERPRETATION FRAMEWORK (exploratory only)
# ---------------------------------------------------------------------------
# Quantile cut-offs used to label a stock "central" or "peripheral".  These are
# descriptive labels attached to network positions, not recommendations.
CENTRAL_QUANTILE = 0.80      # top 20% of the composite centrality score
PERIPHERAL_QUANTILE = 0.25   # bottom 25% of the composite centrality score

# Number of stocks in each illustrative portfolio of Section 16.
PORTFOLIO_SIZE = 10

# Risk-free rate assumed when computing the illustrative Sharpe ratio.
RISK_FREE_RATE = 0.0

# Standard disclaimer, injected verbatim into every output that touches the
# investment interpretation.  Keeping it in the configuration guarantees that
# the wording is identical everywhere.
INVESTMENT_DISCLAIMER = (
    "This framework is exploratory and supports diversification and risk "
    "interpretation. It does not constitute investment advice and does not "
    "predict future returns."
)

# ---------------------------------------------------------------------------
# 8. VISUALISATION
# ---------------------------------------------------------------------------
FIGURE_DPI = 300                 # print quality for a bound thesis
FIGURE_SIZE_STANDARD = (10, 7)   # line/bar charts
FIGURE_SIZE_NETWORK = (14, 11)   # node-link diagrams
FIGURE_SIZE_HEATMAP = (14, 12)   # correlation heat map
FIGURE_FORMAT = "png"
FONT_SIZE_LABEL = 8              # node labels in network figures
MAX_LABELLED_NODES = 60          # above this, only high-degree nodes are labelled

# Colour-blind-friendly qualitative palette, one colour per sector, fixed so
# that a given sector keeps the same colour in every figure of the thesis.
SECTOR_COLORS = {
    "Technology": "#4C72B0",
    "Financials": "#DD8452",
    "Healthcare": "#55A868",
    "Energy": "#C44E52",
    "Consumer Staples": "#8172B3",
    "Consumer Discretionary": "#937860",
    "Industrials": "#DA8BC3",
    "Communication Services": "#8C8C8C",
    "Utilities": "#CCB974",
    "Real Estate": "#64B5CD",
}

# ---------------------------------------------------------------------------
# 9. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# Seed for every stochastic component of the pipeline (graph layout algorithms,
# and the synthetic self-test data set).  Fixing it makes the figures
# byte-identical across runs on the same data.
RANDOM_SEED = 42

# Number of retries when Yahoo Finance refuses or drops a request.
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_BACKOFF = 2.0     # seconds; doubled after each failed attempt

# How many entries the "top-k" tables contain.
TOP_K = 10

# ---------------------------------------------------------------------------
# 10. LARGE-NETWORK EXTENSION  (supplementary; leaves sections 1-9 untouched)
# ---------------------------------------------------------------------------
# Everything below supports the OPTIONAL large-network extension driven by
# ``large_network_extension.py``.  None of it is read by ``main.py``: the
# 47-stock experiment behind the thesis uses only sections 1-9 and its numbers
# are unaffected by any value declared here.
#
# The extension deliberately reuses the thesis methodology unchanged - the same
# sample period (section 2), the same cleaning rules (section 4), the same
# threshold grid and the same absolute-correlation edge rule (section 5).  What
# changes is only the size of the universe.

# Static, committed constituent list.  See data/reference/CONSTITUENTS_SOURCE.md
# for the source, the snapshot date and the derivation.  A file rather than a
# live scrape: the universe must not drift between runs.
LARGE_UNIVERSE_FILE = os.path.join(
    DATA_DIR, "reference", "sp500_constituents_2025_12_22.csv")
LARGE_UNIVERSE_PROVENANCE_FILE = os.path.join(
    DATA_DIR, "reference", "sp500_constituents_2025_12_22_provenance.json")

# Separate data and output trees, so that a large run can never overwrite the
# 47-stock panel in data/ or the thesis results in outputs/.
LARGE_DATA_DIR = os.path.join(DATA_DIR, "large")
LARGE_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_large")
LARGE_SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "large_synthetic")
LARGE_SYNTHETIC_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_large_synthetic")

# Network exports requested by the supervisor.  These are committed to the
# repository (unlike data/ and outputs/, which are generated), so that the
# network itself can be reused directly from GitHub.
NETWORK_EXPORT_DIR = os.path.join(BASE_DIR, "network_exports")

# --- Phase 7 rule: which nodes count as "cross-sector-oriented" -------------
# A purely descriptive filter, stated up front so that the selection cannot be
# tuned after seeing the results.  A node is flagged when ALL of:
#   (i)   its degree is at least LARGE_CROSS_SECTOR_MIN_DEGREE - below this a
#         "share of neighbours" is too noisy to interpret (with 3 neighbours the
#         share can only be 0, 1/3, 2/3 or 1);
#   (ii)  its same-sector neighbour share is at most
#         LARGE_CROSS_SECTOR_MAX_SHARE_RATIO times the share that *chance alone*
#         would give it;
#   (iii) the most frequent sector among its neighbours is NOT its own sector.
#
# Condition (ii) is a ratio rather than an absolute cut-off because an absolute
# one is not comparable across sectors.  With eleven GICS sectors, a stock in a
# 22-name sector out of ~440 could reach a same-sector share of only about 0.05
# by chance, while a stock in an 80-name sector reaches about 0.18; a flat
# "share <= 0.20" would therefore flag the first as ordinary and the second as
# atypical purely because of how big their sectors are.  The benchmark for a
# stock in sector s is the share of same-sector partners available to it,
#         (n_s - 1) / (N - 1),
# i.e. the expected same-sector share if its links were placed at random.  A
# ratio of 1.0 means "no more same-sector neighbours than chance would give".
#
# This is a normalised observable, not an invented anomaly score: both the
# numerator and the denominator are reported in the exported table so that the
# ratio can be recomputed by hand from the columns beside it.
LARGE_CROSS_SECTOR_MIN_DEGREE = 5
LARGE_CROSS_SECTOR_MAX_SHARE_RATIO = 1.0

# Colours for the eleven official GICS sectors.  The ten informal labels of the
# thesis universe keep their own palette in SECTOR_COLORS above; the two maps
# are separate because the two universes are never merged.
GICS_SECTOR_COLORS = {
    "Information Technology": "#4C72B0",
    "Financials": "#DD8452",
    "Health Care": "#55A868",
    "Energy": "#C44E52",
    "Consumer Staples": "#8172B3",
    "Consumer Discretionary": "#937860",
    "Industrials": "#DA8BC3",
    "Communication Services": "#8C8C8C",
    "Utilities": "#CCB974",
    "Real Estate": "#64B5CD",
    "Materials": "#B07AA1",
}


def use_large_network_paths(synthetic: bool = False) -> None:
    """Point every data and output path at the large-network tree.

    Called once by ``large_network_extension.py`` before any directory is
    created and before logging is configured, exactly as
    :func:`use_synthetic_output_paths` is called by ``main.py``.  The analysis
    modules read these paths from ``config`` at call time, so this single
    rebinding moves the whole run - which is what allows the extension to reuse
    ``src/`` without modifying any of it.

    Because ``data/raw`` and ``outputs/`` are never rebound to by this
    function, a large run cannot touch the 47-stock panel or the thesis
    results, whatever happens inside it.
    """
    global DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
    global OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, ALL_DIRS
    global RAW_PRICES_FILE, LOG_RETURNS_FILE, CLEAN_PRICES_FILE
    global SYNTHETIC_DATA_MARKER, SYNTHETIC_PROCESSED_MARKER, PROVENANCE_FILE

    DATA_DIR = LARGE_SYNTHETIC_DATA_DIR if synthetic else LARGE_DATA_DIR
    OUTPUT_DIR = LARGE_SYNTHETIC_OUTPUT_DIR if synthetic else LARGE_OUTPUT_DIR

    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    RAW_PRICES_FILE = os.path.join(RAW_DATA_DIR, "adjusted_close_prices.csv")
    LOG_RETURNS_FILE = os.path.join(PROCESSED_DATA_DIR, "log_returns.csv")
    CLEAN_PRICES_FILE = os.path.join(PROCESSED_DATA_DIR,
                                     "clean_adjusted_close_prices.csv")
    SYNTHETIC_DATA_MARKER = os.path.join(RAW_DATA_DIR, "SYNTHETIC_DATA_WARNING.txt")
    SYNTHETIC_PROCESSED_MARKER = os.path.join(PROCESSED_DATA_DIR,
                                              "SYNTHETIC_DATA_WARNING.txt")
    PROVENANCE_FILE = os.path.join(RAW_DATA_DIR, "data_provenance.json")

    ALL_DIRS = [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]


def apply_universe(tickers_by_sector: dict, company_names: dict) -> None:
    """Replace the stock universe that the ``src/`` modules read.

    ``src/`` resolves ``config.TICKER_TO_SECTOR``, ``config.COMPANY_NAMES``,
    ``config.STOCKS``, ``config.SECTORS`` and ``config.TICKERS`` at call time
    rather than at import time, so rebinding them here swaps the universe for
    the whole run without touching a single line of the analysis modules.

    Used only by the large-network extension.  ``main.py`` never calls it, so
    the 47-stock universe declared in section 3 is what the thesis pipeline
    always sees.
    """
    global STOCKS, COMPANY_NAMES, TICKERS, TICKER_TO_SECTOR, SECTORS
    STOCKS = {sector: list(tickers) for sector, tickers in tickers_by_sector.items()}
    COMPANY_NAMES = dict(company_names)
    TICKERS = [t for sector_tickers in STOCKS.values() for t in sector_tickers]
    TICKER_TO_SECTOR = {t: s for s, tickers in STOCKS.items() for t in tickers}
    SECTORS = list(STOCKS.keys())
