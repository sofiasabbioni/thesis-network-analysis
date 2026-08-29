"""
preprocessing.py
================

Step 2 of the empirical pipeline: turn the raw adjusted-close panel into a
clean, rectangular matrix of **daily logarithmic returns**.

Three modelling choices are made here and each one is defended in Chapter 3.

1. *Why aligned trading dates?*
   A Pearson correlation is only defined for pairs of observations measured at
   the same instant.  If two stocks were compared over partially different
   calendars (for example because one of them was halted for a day), pandas
   would silently drop the mismatched rows *pair by pair*, so every entry of
   the correlation matrix would be estimated on a different sample.  The
   resulting matrix is not guaranteed to be positive semi-definite and the
   network built from it would mix information from different periods.  We
   therefore force one common calendar for the whole universe.

2. *Why do missing values matter?*
   A gap that is filled with the previous price produces a return of exactly
   zero.  A run of artificial zeros lowers the estimated volatility of that
   stock and biases its correlations towards zero, which would push it
   artificially towards the periphery of the network.  Short gaps are
   forward-filled (at most ``config.MAX_FORWARD_FILL_DAYS`` days); a stock with
   too many missing observations is removed from the universe instead of being
   patched.

3. *Why logarithmic returns?*
   With ``r_t = ln(P_t / P_{t-1})``:
   - returns are additive over time, so multi-day returns are simple sums;
   - the series is (approximately) stationary, which a price level is not -
     correlating price *levels* would mostly measure common trends, not
     co-movement;
   - log returns are symmetric around zero (a +10% move followed by a -10%
     move returns to the starting point), which removes the asymmetry of
     simple returns;
   - for the small daily moves considered here, ``ln(1+x) ~ x``, so the
     economic interpretation is unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def align_and_clean_prices(prices: pd.DataFrame,
                           max_missing_fraction: float | None = None,
                           max_ffill_days: int | None = None
                           ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Align the trading calendar and remove unusable stocks.

    The cleaning sequence is deliberately ordered from least to most
    destructive:

    ``sort`` -> ``drop empty rows`` -> ``drop sparse tickers`` ->
    ``limited forward fill`` -> ``drop remaining incomplete rows``

    Returns
    -------
    (clean_prices, report)
        ``report`` records which tickers were dropped and why, so that the
        thesis can state exactly how the final universe was obtained.
    """
    max_missing_fraction = (config.MAX_MISSING_FRACTION
                            if max_missing_fraction is None else max_missing_fraction)
    max_ffill_days = (config.MAX_FORWARD_FILL_DAYS
                      if max_ffill_days is None else max_ffill_days)

    report: Dict[str, List[str]] = {
        "dropped_all_missing": [], "dropped_too_sparse": [],
        "dropped_too_short": [], "kept": [],
    }

    df = prices.copy().sort_index()
    df = df[~df.index.duplicated(keep="first")]
    n_days_initial, n_tickers_initial = df.shape

    # (a) A column that is empty over the whole window is a failed download.
    empty = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty:
        report["dropped_all_missing"] = empty
        df = df.drop(columns=empty)

    # (b) A row where *every* stock is missing is a non-trading day that the
    #     data provider still returned; it carries no information.
    df = df.dropna(axis=0, how="all")

    # (c) Drop stocks whose coverage of the common calendar is too poor.  This
    #     removes companies that were listed (or delisted) inside the window.
    missing_fraction = df.isna().mean()
    too_sparse = missing_fraction[missing_fraction > max_missing_fraction].index.tolist()
    if too_sparse:
        report["dropped_too_sparse"] = too_sparse
        logging.warning("  dropping %d ticker(s) with >%.0f%% missing prices: %s",
                        len(too_sparse), 100 * max_missing_fraction,
                        ", ".join(f"{t} ({100*missing_fraction[t]:.1f}%)" for t in too_sparse))
        df = df.drop(columns=too_sparse)

    # (d) Carry the last known price forward across *short* gaps only.
    df = df.ffill(limit=max_ffill_days)

    # (e) Anything still missing (a leading gap, or a gap longer than the
    #     limit) is removed row-wise, which preserves the alignment property.
    before_rows = len(df)
    df = df.dropna(axis=0, how="any")
    dropped_rows = before_rows - len(df)

    # (f) Final length check.
    if len(df) < config.MIN_OBSERVATIONS:
        raise utils.PipelineError(
            f"Only {len(df)} aligned trading days remain (minimum "
            f"{config.MIN_OBSERVATIONS}). Widen the sample period or relax "
            f"config.MAX_MISSING_FRACTION."
        )

    report["kept"] = list(df.columns)

    logging.info("  raw panel                : %d days x %d tickers",
                 n_days_initial, n_tickers_initial)
    logging.info("  forward-filled gaps      : up to %d consecutive day(s)", max_ffill_days)
    logging.info("  rows dropped (unaligned) : %d", dropped_rows)
    logging.info("  aligned panel            : %d days x %d tickers", *df.shape)
    logging.info("  common calendar          : %s to %s",
                 df.index.min().date(), df.index.max().date())
    return df, report


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily logarithmic returns ``r_t = ln(P_t / P_{t-1})``.

    Non-positive prices (impossible for an adjusted close, but cheap to guard
    against) are masked before the logarithm so that a single corrupt value
    cannot poison a whole column with ``-inf``.
    """
    safe = prices.where(prices > 0)
    returns = np.log(safe / safe.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    returns.index.name = "Date"

    logging.info("  log returns              : %d observations x %d stocks", *returns.shape)
    return returns


def drop_short_series(returns: pd.DataFrame,
                      min_observations: int | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """Remove stocks with too few return observations to estimate correlations."""
    min_observations = (config.MIN_OBSERVATIONS
                        if min_observations is None else min_observations)
    counts = returns.notna().sum()
    too_short = counts[counts < min_observations].index.tolist()
    if too_short:
        logging.warning("  dropping %d ticker(s) with fewer than %d observations: %s",
                        len(too_short), min_observations, ", ".join(too_short))
        returns = returns.drop(columns=too_short)
    return returns, too_short


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-stock descriptive statistics of the daily log returns.

    Besides the four moments required by the specification (mean, standard
    deviation, minimum, maximum, number of observations) the table reports the
    annualised mean and volatility, which are the quantities a finance reader
    expects, plus skewness and excess kurtosis, which document the well-known
    non-normality of daily equity returns and justify treating the Pearson
    correlation as a *linear-dependence* measure rather than as a complete
    description of dependence (a limitation discussed in Chapter 5).
    """
    try:
        from scipy import stats as scipy_stats
        skew = returns.apply(lambda s: float(scipy_stats.skew(s.dropna(), bias=False)))
        kurt = returns.apply(lambda s: float(scipy_stats.kurtosis(s.dropna(), bias=False)))
    except Exception:  # scipy is optional
        skew = returns.skew()
        kurt = returns.kurt()

    ann = config.TRADING_DAYS_PER_YEAR
    table = pd.DataFrame({
        "ticker": returns.columns,
        "sector": [config.TICKER_TO_SECTOR.get(t, "Unknown") for t in returns.columns],
        "company": [config.COMPANY_NAMES.get(t, t) for t in returns.columns],
        "n_observations": returns.notna().sum().values,
        "mean_daily_return": returns.mean().values,
        "std_daily_return": returns.std().values,
        "min_daily_return": returns.min().values,
        "max_daily_return": returns.max().values,
        "median_daily_return": returns.median().values,
        "skewness": skew.values,
        "excess_kurtosis": kurt.values,
        "annualised_mean_return": returns.mean().values * ann,
        "annualised_volatility": returns.std().values * np.sqrt(ann),
    })
    table = table.sort_values("ticker").reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_preprocessing(prices: pd.DataFrame) -> Dict[str, object]:
    """Execute the whole preprocessing stage and export its artefacts.

    Writes
    ------
    ``data/processed/clean_adjusted_close_prices.csv``
    ``data/processed/log_returns.csv``
    ``outputs/tables/return_summary_statistics.csv``
    """
    utils.subsection("Aligning and cleaning the price panel")
    clean_prices, report = align_and_clean_prices(prices)

    utils.subsection("Computing logarithmic returns")
    returns = compute_log_returns(clean_prices)
    returns, too_short = drop_short_series(returns)
    report["dropped_too_short"] = too_short
    if too_short:
        clean_prices = clean_prices.drop(columns=too_short)
        report["kept"] = list(returns.columns)

    if returns.shape[1] < 3:
        raise utils.PipelineError(
            f"Only {returns.shape[1]} stock(s) survived preprocessing; a network "
            "cannot be built. Check the download stage."
        )

    clean_prices.to_csv(config.CLEAN_PRICES_FILE, float_format="%.6f")
    returns.to_csv(config.LOG_RETURNS_FILE, float_format="%.8f")
    logging.info("  clean prices -> %s", utils.relative(config.CLEAN_PRICES_FILE))
    logging.info("  log returns  -> %s", utils.relative(config.LOG_RETURNS_FILE))

    utils.subsection("Descriptive statistics of daily log returns")
    stats = summary_statistics(returns)
    utils.save_table(stats, "return_summary_statistics.csv")

    logging.info("  cross-sectional mean of annualised volatility : %.2f%%",
                 100 * stats["annualised_volatility"].mean())
    logging.info("  most volatile stock  : %s (%.2f%% p.a.)",
                 stats.loc[stats["annualised_volatility"].idxmax(), "ticker"],
                 100 * stats["annualised_volatility"].max())
    logging.info("  least volatile stock : %s (%.2f%% p.a.)",
                 stats.loc[stats["annualised_volatility"].idxmin(), "ticker"],
                 100 * stats["annualised_volatility"].min())

    dropped = (report["dropped_all_missing"] + report["dropped_too_sparse"]
               + report["dropped_too_short"])
    if dropped:
        logging.info("  tickers removed in total : %d -> %s", len(dropped), ", ".join(dropped))
    else:
        logging.info("  tickers removed in total : 0 (the full universe survived)")

    return {"prices": clean_prices, "returns": returns,
            "summary": stats, "report": report}
