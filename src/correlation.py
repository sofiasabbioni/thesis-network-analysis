"""
correlation.py
==============

Step 3 of the empirical pipeline: the **Pearson correlation matrix** of daily
log returns, which is the object the whole network analysis is built on.

For two return series r_i and r_j,

        C_ij = cov(r_i, r_j) / (sigma_i * sigma_j)   with   C_ij in [-1, 1].

Interpretation used in the thesis: ``C_ij`` measures the strength of the
*linear* co-movement of two stocks.  Values close to +1 mean the two stocks
tend to move together, values close to 0 mean their daily moves are linearly
unrelated, and negative values mean they tend to move in opposite directions.
The matrix is symmetric with a unit diagonal, so it contains N(N-1)/2 distinct
numbers - exactly the potential edges of the correlation graph.

Known limitations (Chapter 5): the Pearson coefficient captures linear
dependence only, it is not robust to the fat tails of daily returns, and a
single coefficient estimated over several years averages over calm and
turbulent regimes.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def compute_correlation_matrix(returns: pd.DataFrame,
                               method: str = "pearson") -> pd.DataFrame:
    """Return the N x N correlation matrix of the return columns.

    Because :func:`preprocessing.align_and_clean_prices` guarantees a complete
    rectangular panel, every coefficient is estimated on the *same* sample of
    dates - see the discussion of aligned calendars in ``preprocessing.py``.
    """
    corr = returns.corr(method=method)

    # Guard against numerical noise on the diagonal and enforce exact symmetry,
    # which some downstream NetworkX/NumPy routines rely on.  The symmetrised
    # matrix is rebuilt from a fresh NumPy array rather than modified in place:
    # under pandas' copy-on-write semantics the array behind a DataFrame is
    # read-only, so an in-place ``fill_diagonal`` would fail.
    values = corr.to_numpy(dtype=float, copy=True)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    corr.index.name = "ticker"
    corr.columns.name = "ticker"

    n = corr.shape[0]
    logging.info("  correlation matrix       : %d x %d (%d distinct pairs)",
                 n, n, n * (n - 1) // 2)
    return corr


def upper_triangle_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Flatten the strict upper triangle into a tidy pair-level table.

    One row per unordered pair, with the two sectors and a flag telling whether
    the pair is intra-sector.  This "long" representation is what feeds the
    edge list of the graph and the descriptive statistics below.
    """
    tickers = list(corr.columns)
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = float(corr.iloc[i, j])
            sector_a = config.TICKER_TO_SECTOR.get(a, "Unknown")
            sector_b = config.TICKER_TO_SECTOR.get(b, "Unknown")
            rows.append({
                "stock_a": a, "stock_b": b,
                "sector_a": sector_a, "sector_b": sector_b,
                "correlation": c,
                "abs_correlation": abs(c),
                "same_sector": sector_a == sector_b,
            })
    return pd.DataFrame(rows)


def sector_correlation_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Average correlation between (and within) sectors.

    The diagonal is the mean correlation of *distinct* pairs inside a sector,
    not 1: this makes the table directly comparable with the off-diagonal
    entries and is the cleanest quantitative evidence of sectoral clustering
    that can be shown before any graph is built.
    """
    pairs = upper_triangle_pairs(corr)
    sectors = [s for s in config.SECTORS
               if any(t in corr.columns for t in config.STOCKS[s])]
    out = pd.DataFrame(index=sectors, columns=sectors, dtype=float)
    for s1 in sectors:
        for s2 in sectors:
            mask = (((pairs["sector_a"] == s1) & (pairs["sector_b"] == s2))
                    | ((pairs["sector_a"] == s2) & (pairs["sector_b"] == s1)))
            values = pairs.loc[mask, "correlation"]
            out.loc[s1, s2] = float(values.mean()) if len(values) else np.nan
    out.index.name = "sector"
    return out


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


def describe_correlations(corr: pd.DataFrame, top_k: int | None = None) -> Dict[str, object]:
    """Compute the headline numbers used in the textual interpretation."""
    top_k = top_k or config.TOP_K
    pairs = upper_triangle_pairs(corr)
    values = pairs["correlation"].to_numpy()

    within = pairs.loc[pairs["same_sector"], "correlation"]
    across = pairs.loc[~pairs["same_sector"], "correlation"]

    # Average correlation of each stock with all the others: a first,
    # matrix-based notion of "centrality" that does not require a graph.
    mean_corr_per_stock = (corr.sum(axis=1) - 1.0) / (corr.shape[0] - 1)

    return {
        "pairs": pairs,
        "n_pairs": int(len(pairs)),
        "mean_correlation": float(values.mean()),
        "median_correlation": float(np.median(values)),
        "mean_abs_correlation": float(np.abs(values).mean()),
        "std_correlation": float(values.std(ddof=1)),
        "min_correlation": float(values.min()),
        "max_correlation": float(values.max()),
        "share_negative": float((values < 0).mean()),
        "mean_within_sector": float(within.mean()) if len(within) else np.nan,
        "mean_across_sector": float(across.mean()) if len(across) else np.nan,
        "n_within_sector": int(len(within)),
        "n_across_sector": int(len(across)),
        "strongest_pairs": pairs.nlargest(top_k, "correlation").reset_index(drop=True),
        "weakest_pairs": pairs.nsmallest(top_k, "correlation").reset_index(drop=True),
        "most_correlated_stocks": mean_corr_per_stock.sort_values(ascending=False),
        "quantiles": pairs["correlation"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
    }


def correlation_interpretation_text(stats: Dict[str, object],
                                    sector_matrix: pd.DataFrame) -> str:
    """Compose the plain-language interpretation of the correlation matrix.

    Written to ``outputs/logs/correlation_interpretation.txt`` and intended to
    be paraphrased in Section 4.1 of the thesis.
    """
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("INTERPRETATION OF THE PEARSON CORRELATION MATRIX")
    add("=" * 78)
    add("")
    add(f"Sample period          : {config.START_DATE} to {config.END_DATE}")
    add(f"Distinct stock pairs   : {stats['n_pairs']}")
    add("")
    add("1. OVERALL LEVEL OF CO-MOVEMENT")
    add("-" * 78)
    add(f"  Average correlation            : {stats['mean_correlation']:+.4f}")
    add(f"  Median correlation             : {stats['median_correlation']:+.4f}")
    add(f"  Average ABSOLUTE correlation   : {stats['mean_abs_correlation']:.4f}")
    add(f"  Standard deviation             : {stats['std_correlation']:.4f}")
    add(f"  Range                          : {stats['min_correlation']:+.4f} to "
        f"{stats['max_correlation']:+.4f}")
    add(f"  Share of negative correlations : {100 * stats['share_negative']:.1f}%")
    add("")
    add("  A positive average correlation is the empirical signature of a common")
    add("  market factor: most large-capitalisation US equities move together with")
    add("  the aggregate market, which is why the correlation graph is dense at low")
    add("  thresholds. The dispersion around that average is what the network")
    add("  analysis exploits: it is the part of the dependence structure that is")
    add("  specific to groups of stocks rather than common to all of them.")
    add("")
    add("2. STRONGEST POSITIVE CORRELATIONS")
    add("-" * 78)
    strongest = stats["strongest_pairs"]
    for _, row in strongest.iterrows():
        tag = "same sector" if row["same_sector"] else f"{row['sector_a']} / {row['sector_b']}"
        add(f"  {row['stock_a']:<6} - {row['stock_b']:<6} {row['correlation']:+.4f}   ({tag})")
    same_share = 100 * float(strongest["same_sector"].mean())
    add("")
    add(f"  {same_share:.0f}% of the {len(strongest)} strongest pairs join two stocks of the "
        "same sector,")
    add("  which is the first indication that the network will exhibit sectoral")
    add("  clustering rather than a homogeneous structure.")
    add("")
    add("3. WEAKEST (OR MOST NEGATIVE) CORRELATIONS")
    add("-" * 78)
    for _, row in stats["weakest_pairs"].iterrows():
        tag = "same sector" if row["same_sector"] else f"{row['sector_a']} / {row['sector_b']}"
        add(f"  {row['stock_a']:<6} - {row['stock_b']:<6} {row['correlation']:+.4f}   ({tag})")
    add("")
    add("  These pairs are the loosest links in the system. In a diversification")
    add("  reading they are the pairs whose combination reduces portfolio variance")
    add("  the most - which is a statement about the estimation sample, not a")
    add("  forecast that the relationship will persist.")
    add("")
    add("4. WITHIN-SECTOR VERSUS CROSS-SECTOR CO-MOVEMENT")
    add("-" * 78)
    add(f"  Mean correlation, same sector      : {stats['mean_within_sector']:+.4f}"
        f"   ({stats['n_within_sector']} pairs)")
    add(f"  Mean correlation, different sectors: {stats['mean_across_sector']:+.4f}"
        f"   ({stats['n_across_sector']} pairs)")
    gap = stats["mean_within_sector"] - stats["mean_across_sector"]
    add(f"  Difference                         : {gap:+.4f}")
    add("")
    add("  Sector-average correlation matrix (diagonal = average of distinct pairs")
    add("  inside the sector):")
    add("")
    for line in sector_matrix.round(3).to_string().splitlines():
        add("    " + line)
    add("")
    diag = pd.Series({s: sector_matrix.loc[s, s] for s in sector_matrix.index}).dropna()
    if not diag.empty:
        add(f"  Most internally cohesive sector : {diag.idxmax()} ({diag.max():+.3f})")
        add(f"  Least internally cohesive sector: {diag.idxmin()} ({diag.min():+.3f})")
    add("")
    add("5. STOCKS MOST AND LEAST CORRELATED WITH THE REST OF THE UNIVERSE")
    add("-" * 78)
    ranked = stats["most_correlated_stocks"]
    add("  Highest average correlation with all other stocks:")
    for ticker, value in ranked.head(config.TOP_K).items():
        add(f"    {ticker:<6} {value:+.4f}   ({config.TICKER_TO_SECTOR.get(ticker, 'Unknown')})")
    add("")
    add("  Lowest average correlation with all other stocks:")
    for ticker, value in ranked.tail(config.TOP_K).iloc[::-1].items():
        add(f"    {ticker:<6} {value:+.4f}   ({config.TICKER_TO_SECTOR.get(ticker, 'Unknown')})")
    add("")
    add("  These two lists anticipate the network result: stocks at the top will")
    add("  tend to become high-degree, central nodes once the graph is built, and")
    add("  stocks at the bottom will tend to be peripheral or isolated.")
    add("")
    add("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_correlation_analysis(returns: pd.DataFrame) -> Dict[str, object]:
    """Estimate, export and describe the correlation matrix.

    Writes
    ------
    ``outputs/tables/correlation_matrix.csv``
    ``outputs/tables/correlation_pairs.csv``
    ``outputs/tables/sector_correlation_matrix.csv``
    ``outputs/logs/correlation_interpretation.txt``
    """
    utils.subsection("Estimating the Pearson correlation matrix")
    corr = compute_correlation_matrix(returns)
    utils.save_table(corr.round(6), "correlation_matrix.csv", index=True)

    stats = describe_correlations(corr)
    utils.save_table(stats["pairs"].sort_values("correlation", ascending=False),
                     "correlation_pairs.csv")

    sector_matrix = sector_correlation_matrix(corr)
    utils.save_table(sector_matrix.round(4), "sector_correlation_matrix.csv", index=True)

    logging.info("  mean correlation         : %+.4f", stats["mean_correlation"])
    logging.info("  mean |correlation|       : %.4f", stats["mean_abs_correlation"])
    logging.info("  range                    : %+.4f to %+.4f",
                 stats["min_correlation"], stats["max_correlation"])
    logging.info("  within-sector mean       : %+.4f", stats["mean_within_sector"])
    logging.info("  cross-sector mean        : %+.4f", stats["mean_across_sector"])
    top = stats["strongest_pairs"].iloc[0]
    logging.info("  strongest pair           : %s-%s (%+.4f)",
                 top["stock_a"], top["stock_b"], top["correlation"])

    text = correlation_interpretation_text(stats, sector_matrix)
    utils.save_text(text, "correlation_interpretation.txt")

    return {"correlation": corr, "stats": stats, "sector_matrix": sector_matrix}
