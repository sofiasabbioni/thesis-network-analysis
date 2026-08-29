"""
portfolio.py
============

Section 16 of the specification: a **purely illustrative** comparison between a
portfolio diversified by sector and a portfolio diversified by network
position.

What this module is
-------------------
A demonstration that the structural information extracted by the graph
algorithms (components, degree, MST leaves) can be *read* in portfolio terms.
It shows that selecting stocks that are far apart in the correlation network
produces a basket with a lower average pairwise correlation than selecting one
stock per sector, and it reports what that did to realised risk over the
estimation sample.

What this module is **not**
---------------------------
It is not a backtest and not a strategy.  Specifically:

* the portfolios are formed using the correlation matrix estimated over the
  *same* period on which they are then evaluated, so the comparison is entirely
  in-sample and cannot be read as evidence of out-of-sample performance;
* weights are equal and rebalanced daily, transaction costs, taxes, liquidity
  and short-sale constraints are all ignored;
* the sample contains a small number of large-capitalisation US stocks over one
  particular period, so nothing here generalises;
* a lower realised volatility over a fixed past window does not imply lower
  future risk, and the network-diversified portfolio is *not* claimed to be
  superior in any sense.

The honest conclusion the thesis can draw from it is narrow and defensible:
network position is informative about the correlation structure of a basket,
and that is a quantity portfolio construction cares about.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Portfolio construction rules
# ---------------------------------------------------------------------------


def select_sector_diversified(corr: pd.DataFrame,
                              node_metrics: pd.DataFrame) -> List[str]:
    """Portfolio A: one stock from every sector.

    Within each sector the *most representative* stock is chosen - the one with
    the highest average correlation to its own sector.  This is the natural
    analogue of picking a sector bellwether, and it makes the rule deterministic
    and independent of the network analysis, so that Portfolio A is a genuine
    benchmark rather than a second network-based portfolio.
    """
    selected: List[str] = []
    for sector in config.SECTORS:
        members = [t for t in config.STOCKS.get(sector, []) if t in corr.columns]
        if not members:
            continue
        if len(members) == 1:
            selected.append(members[0])
            continue
        within = corr.loc[members, members]
        mean_within = (within.sum(axis=1) - 1.0) / (len(members) - 1)
        selected.append(str(mean_within.idxmax()))
    return selected


def select_network_diversified(corr: pd.DataFrame,
                               node_metrics: pd.DataFrame,
                               mst_metrics: pd.DataFrame,
                               size: int | None = None) -> List[str]:
    """Portfolio B: stocks that are far apart in the correlation network.

    A transparent greedy rule, stated so that it can be reproduced by hand:

    1. build a pool of candidates ranked by a *peripherality* score that
       rewards a low composite centrality, a low degree and being a leaf of the
       MST;
    2. seed the portfolio with the most peripheral stock;
    3. repeatedly add the candidate that minimises its **maximum** correlation
       with the stocks already selected, preferring, at equal correlation, a
       stock from a connected component that is not yet represented.

    Minimising the *maximum* pairwise correlation (rather than the average) is
    deliberate: it prevents the greedy rule from accepting one very highly
    correlated pair in exchange for a good average, which is precisely the
    concentration a diversified basket is supposed to avoid.
    """
    size = size or config.PORTFOLIO_SIZE
    universe = [t for t in corr.columns]

    metrics = node_metrics.set_index("ticker")
    mst = mst_metrics.set_index("ticker")

    # Peripherality score in [0, 1]; higher = further from the core.
    centrality = metrics["composite_centrality"].reindex(universe).fillna(0.0)
    degree = metrics["degree"].reindex(universe).fillna(0.0).astype(float)
    degree_norm = degree / degree.max() if degree.max() > 0 else degree
    is_leaf = mst["is_leaf"].reindex(universe).fillna(False).astype(float)
    score = (1 - centrality) * 0.5 + (1 - degree_norm) * 0.3 + is_leaf * 0.2

    component = metrics["component_id"].reindex(universe).fillna(-1).astype(int)

    ranked = list(score.sort_values(ascending=False).index)
    selected = [ranked[0]]
    used_components = {int(component[ranked[0]])}

    while len(selected) < min(size, len(universe)):
        best, best_key = None, None
        for candidate in ranked:
            if candidate in selected:
                continue
            max_corr = float(max(abs(corr.loc[candidate, s]) for s in selected))
            new_component = int(component[candidate]) not in used_components
            # Sort key: prefer an unused component, then a low maximum
            # correlation, then a high peripherality score.
            key = (0 if new_component else 1, round(max_corr, 6), -float(score[candidate]))
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        if best is None:
            break
        selected.append(best)
        used_components.add(int(component[best]))
    return selected


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _simple_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert log returns back to simple returns for portfolio aggregation.

    A portfolio's return is the weighted average of its constituents' *simple*
    returns; log returns are not additive across assets, only across time.
    Aggregating log returns directly would understate the portfolio return.
    """
    return np.exp(log_returns) - 1.0


def portfolio_series(log_returns: pd.DataFrame, tickers: Sequence[str]) -> pd.Series:
    """Daily simple return of an equally weighted, daily rebalanced portfolio."""
    simple = _simple_returns(log_returns[list(tickers)])
    return simple.mean(axis=1)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the cumulative value path (a negative number)."""
    wealth = (1.0 + daily_returns).cumprod()
    running_max = wealth.cummax()
    return float((wealth / running_max - 1.0).min())


def average_pairwise_correlation(corr: pd.DataFrame, tickers: Sequence[str]) -> float:
    """Mean of the off-diagonal entries of the portfolio's correlation sub-matrix."""
    tickers = list(tickers)
    if len(tickers) < 2:
        return np.nan
    block = corr.loc[tickers, tickers].to_numpy()
    return float(block[np.triu_indices_from(block, k=1)].mean())


def evaluate_portfolio(name: str, tickers: Sequence[str],
                       log_returns: pd.DataFrame, corr: pd.DataFrame,
                       node_metrics: pd.DataFrame,
                       mst_metrics: pd.DataFrame) -> Dict[str, object]:
    """Compute the risk/return statistics of one illustrative portfolio."""
    tickers = list(tickers)
    daily = portfolio_series(log_returns, tickers)
    ann = config.TRADING_DAYS_PER_YEAR

    annualised_return = float(daily.mean() * ann)
    annualised_volatility = float(daily.std(ddof=1) * np.sqrt(ann))
    wealth = float((1.0 + daily).prod())
    years = len(daily) / ann
    cagr = float(wealth ** (1.0 / years) - 1.0) if years > 0 and wealth > 0 else np.nan
    sharpe = ((annualised_return - config.RISK_FREE_RATE) / annualised_volatility
              if annualised_volatility > 0 else np.nan)

    metrics = node_metrics.set_index("ticker")
    mst = mst_metrics.set_index("ticker")
    sectors = {config.TICKER_TO_SECTOR.get(t, "Unknown") for t in tickers}
    components = {int(metrics["component_id"].get(t, -1)) for t in tickers}
    pairs = corr.loc[tickers, tickers].to_numpy()
    off_diagonal = pairs[np.triu_indices_from(pairs, k=1)]

    return {
        "portfolio": name,
        "n_stocks": len(tickers),
        "tickers": ", ".join(tickers),
        "n_sectors": len(sectors),
        "n_network_components": len(components),
        "average_pairwise_correlation": average_pairwise_correlation(corr, tickers),
        "max_pairwise_correlation": float(off_diagonal.max()) if len(off_diagonal) else np.nan,
        "min_pairwise_correlation": float(off_diagonal.min()) if len(off_diagonal) else np.nan,
        "mean_degree_of_members": float(metrics["degree"].reindex(tickers).mean()),
        "mean_mst_degree_of_members": float(mst["mst_degree"].reindex(tickers).mean()),
        "n_mst_leaves": int(mst["is_leaf"].reindex(tickers).fillna(False).sum()),
        "annualised_return": annualised_return,
        "cagr": cagr,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio_rf0": sharpe,
        "max_drawdown": max_drawdown(daily),
        "total_return": wealth - 1.0,
        "n_observations": int(len(daily)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_portfolio_illustration(log_returns: pd.DataFrame,
                               corr: pd.DataFrame,
                               node_metrics: pd.DataFrame,
                               mst_metrics: pd.DataFrame) -> Dict[str, object]:
    """Build, evaluate and export the two illustrative portfolios.

    Writes
    ------
    ``outputs/tables/portfolio_comparison.csv``
    ``outputs/tables/portfolio_holdings.csv``
    and returns the cumulative-value paths so that ``visualisation`` can plot
    them.
    """
    utils.subsection("Exploratory portfolio illustration (NOT a trading strategy)")

    sector_portfolio = select_sector_diversified(corr, node_metrics)
    network_portfolio = select_network_diversified(
        corr, node_metrics, mst_metrics, size=len(sector_portfolio))
    benchmark = list(corr.columns)

    logging.info("  Portfolio A (sector-diversified)  : %s", ", ".join(sector_portfolio))
    logging.info("  Portfolio B (network-diversified) : %s", ", ".join(network_portfolio))
    overlap = sorted(set(sector_portfolio) & set(network_portfolio))
    logging.info("  overlap between A and B           : %d stock(s)%s",
                 len(overlap), (" -> " + ", ".join(overlap)) if overlap else "")

    rows = [
        evaluate_portfolio("A - Sector diversified", sector_portfolio,
                           log_returns, corr, node_metrics, mst_metrics),
        evaluate_portfolio("B - Network diversified", network_portfolio,
                           log_returns, corr, node_metrics, mst_metrics),
        evaluate_portfolio("C - Equally weighted universe", benchmark,
                           log_returns, corr, node_metrics, mst_metrics),
    ]
    comparison = pd.DataFrame(rows)
    comparison["disclaimer"] = config.INVESTMENT_DISCLAIMER
    utils.save_table(comparison, "portfolio_comparison.csv")

    # Per-stock holdings table, so the thesis can show *why* each stock was picked.
    metrics = node_metrics.set_index("ticker")
    mst = mst_metrics.set_index("ticker")
    holdings = []
    for label, members in (("A - Sector diversified", sector_portfolio),
                           ("B - Network diversified", network_portfolio)):
        for ticker in members:
            holdings.append({
                "portfolio": label,
                "ticker": ticker,
                "sector": config.TICKER_TO_SECTOR.get(ticker, "Unknown"),
                "weight": 1.0 / len(members),
                "degree": int(metrics["degree"].get(ticker, 0)),
                "composite_centrality": float(metrics["composite_centrality"].get(ticker, np.nan)),
                "component_id": int(metrics["component_id"].get(ticker, -1)),
                "mst_degree": int(mst["mst_degree"].get(ticker, 0)),
                "mst_leaf": bool(mst["is_leaf"].get(ticker, False)),
            })
    utils.save_table(pd.DataFrame(holdings), "portfolio_holdings.csv")

    cumulative = pd.DataFrame({
        "A - Sector diversified": (1 + portfolio_series(log_returns, sector_portfolio)).cumprod(),
        "B - Network diversified": (1 + portfolio_series(log_returns, network_portfolio)).cumprod(),
        "C - Equally weighted universe": (1 + portfolio_series(log_returns, benchmark)).cumprod(),
    })

    for row in rows:
        logging.info("  %-30s corr %.3f | vol %.2f%% | ret %.2f%% | Sharpe %.2f | MDD %.1f%%",
                     row["portfolio"], row["average_pairwise_correlation"],
                     100 * row["annualised_volatility"], 100 * row["annualised_return"],
                     row["sharpe_ratio_rf0"], 100 * row["max_drawdown"])
    logging.warning("  %s", config.INVESTMENT_DISCLAIMER)

    return {"comparison": comparison, "cumulative": cumulative,
            "sector_portfolio": sector_portfolio,
            "network_portfolio": network_portfolio,
            "holdings": pd.DataFrame(holdings)}
