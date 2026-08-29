"""
data_collection.py
==================

Step 1 of the empirical pipeline: obtain daily **adjusted closing prices** for
the stock universe declared in ``config.STOCKS``.

Why adjusted closing prices?
----------------------------
The raw ("unadjusted") closing price of a stock jumps mechanically on the
ex-dividend date and on the day of a split, even though no economic gain or
loss occurred.  Such artificial jumps would appear as large returns and would
contaminate every correlation in the matrix - in particular they would create
spurious *idiosyncratic* moves that push a stock towards the periphery of the
network.  The adjusted close restates the historical price series for
dividends and splits, so that the return computed from two consecutive
adjusted closes is the actual total return earned by an investor.  This is the
standard input for correlation-network studies (Mantegna, 1999; Onnela et al.,
2003).

The module is deliberately defensive: a delisted, renamed or temporarily
unavailable ticker must not abort a run that takes several minutes.  Failures
are collected, reported and excluded from the universe, and the pipeline
continues with whatever data is available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from src import utils

# Human-readable marker written next to a SIMULATED price panel.  Its presence
# is the primary signal that ``data/raw/adjusted_close_prices.csv`` must not be
# reused by a real run.  Kept as a module-level alias for readability.
SYNTHETIC_MARKER_FILE = config.SYNTHETIC_DATA_MARKER


# ---------------------------------------------------------------------------
# Yahoo Finance download
# ---------------------------------------------------------------------------


def _extract_adjusted_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Pull the adjusted-close block out of whatever shape yfinance returned.

    ``yfinance`` has changed its return layout several times: with a list of
    tickers it returns a MultiIndex column frame ``(field, ticker)``; with a
    single ticker the columns are flat; and since version 0.2.51 the default
    ``auto_adjust=True`` removes the ``Adj Close`` column altogether because
    ``Close`` is *already* adjusted.  This helper normalises all of those cases
    into a plain ``dates x tickers`` frame.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        fields = list(raw.columns.get_level_values(0).unique())
        field = "Adj Close" if "Adj Close" in fields else "Close"
        prices = raw[field].copy()
    else:
        # Single ticker, flat columns.
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]

    # Keep only the requested tickers, in the configured order, and drop
    # columns that are entirely empty (a silent yfinance failure).
    present = [t for t in tickers if t in prices.columns]
    prices = prices[present]
    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices.sort_index()


def _download_batch(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download all tickers in one vectorised yfinance call, with retries."""
    import yfinance as yf  # imported lazily so that --synthetic works offline

    last_error: Exception | None = None
    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,   # keep the explicit "Adj Close" column
                progress=False,
                group_by="column",
                threads=True,
            )
            prices = _extract_adjusted_close(raw, tickers)
            if not prices.empty:
                return prices
            last_error = RuntimeError("yfinance returned an empty frame")
        except Exception as exc:  # network error, rate limit, API change ...
            last_error = exc
        wait = config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1))
        logging.warning("  batch download attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, config.DOWNLOAD_MAX_RETRIES, last_error, wait)
        time.sleep(wait)

    logging.error("  batch download failed after %d attempts: %s",
                  config.DOWNLOAD_MAX_RETRIES, last_error)
    return pd.DataFrame()


def _download_single(ticker: str, start: str, end: str) -> pd.Series | None:
    """Second-chance download of one ticker through the Ticker API.

    The batch endpoint occasionally drops individual symbols under load; a
    per-ticker retry recovers most of them.
    """
    import yfinance as yf

    try:
        history = yf.Ticker(ticker).history(start=start, end=end,
                                            interval="1d", auto_adjust=False)
        if history is None or history.empty:
            return None
        column = "Adj Close" if "Adj Close" in history.columns else "Close"
        series = history[column].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.name = ticker
        return series
    except Exception as exc:
        logging.debug("  single download of %s failed: %s", ticker, exc)
        return None


def download_prices(tickers: List[str] | None = None,
                    start: str | None = None,
                    end: str | None = None,
                    chunk_size: int | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """Download adjusted closing prices for ``tickers`` between two dates.

    Parameters
    ----------
    tickers : list of str, optional
        Defaults to every ticker in ``config.STOCKS``.
    start, end : str, optional
        ISO dates; default to ``config.START_DATE`` / ``config.END_DATE``.
    chunk_size : int, optional
        Split the request into batches of at most this many tickers.  ``None``
        (the default) sends one single batch, which is the behaviour the
        47-stock thesis run has always used and is unaffected by this
        parameter.  Chunking exists for the large-network extension: a single
        request for several hundred symbols is slow and is frequently
        truncated by the endpoint, and a partial batch response is
        indistinguishable from a genuinely unavailable ticker.

    Returns
    -------
    (prices, failed)
        ``prices`` is a ``dates x tickers`` DataFrame of adjusted closes and
        ``failed`` lists the tickers that could not be retrieved at all.
    """
    tickers = list(tickers or config.TICKERS)
    start = start or config.START_DATE
    end = end or config.END_DATE

    logging.info("Requesting %d tickers from Yahoo Finance (%s to %s)",
                 len(tickers), start, end)

    if chunk_size and len(tickers) > chunk_size:
        n_batches = (len(tickers) + chunk_size - 1) // chunk_size
        logging.info("  splitting into %d batches of at most %d tickers",
                     n_batches, chunk_size)
        frames = []
        for index in range(n_batches):
            chunk = tickers[index * chunk_size:(index + 1) * chunk_size]
            logging.info("  batch %d/%d: %d tickers (%s ... %s)",
                         index + 1, n_batches, len(chunk), chunk[0], chunk[-1])
            frame = _download_batch(chunk, start, end)
            if not frame.empty:
                frames.append(frame)
            else:
                logging.warning("  batch %d/%d returned nothing", index + 1, n_batches)
        # Batches share a trading calendar but may differ by a row if a symbol
        # was halted, so they are joined on the union of dates; the gaps that
        # creates are exactly what the cleaning stage is there to handle.
        prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    else:
        prices = _download_batch(tickers, start, end)

    # Retry, one by one, whatever the batch call did not return.
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logging.warning("  %d ticker(s) missing from the batch response: %s",
                        len(missing), ", ".join(missing))
        recovered = {}
        for ticker in missing:
            series = _download_single(ticker, start, end)
            if series is not None and not series.empty:
                recovered[ticker] = series
        if recovered:
            logging.info("  recovered %d ticker(s) individually: %s",
                         len(recovered), ", ".join(sorted(recovered)))
            prices = pd.concat([prices, pd.DataFrame(recovered)], axis=1)

    failed = [t for t in tickers if t not in prices.columns]
    if not prices.empty:
        prices = prices[[t for t in tickers if t in prices.columns]].sort_index()
    return prices, failed


# ---------------------------------------------------------------------------
# Synthetic data (self-test only - NEVER thesis material)
# ---------------------------------------------------------------------------


def generate_synthetic_prices(tickers: List[str] | None = None,
                              start: str | None = None,
                              end: str | None = None,
                              seed: int | None = None) -> pd.DataFrame:
    """Simulate a price panel with a market factor and sector factors.

    **This function does not produce real data.**  It exists for two practical
    reasons only:

    1. it lets a user verify that the whole pipeline runs (all figures and
       tables are produced) before spending time on a live download, and
    2. it makes the project testable on a machine with no internet access.

    The generating process is a standard two-level factor model

        r_it = beta_i * f_market_t + gamma_i * f_sector(i),t + sigma_i * eps_it

    which reproduces the qualitative feature the thesis studies - a single
    dominant market factor plus sector blocks - without pretending to be a
    calibrated model of any real market.  Every output produced from synthetic
    prices is stamped with a warning banner.
    """
    tickers = list(tickers or config.TICKERS)
    start = start or config.START_DATE
    end = end or config.END_DATE
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    dates = pd.bdate_range(start=start, end=end, name="Date")
    n_days = len(dates)

    sectors = sorted({config.TICKER_TO_SECTOR.get(t, "Unknown") for t in tickers})
    market = rng.normal(0.0003, 0.011, n_days)
    sector_factors = {s: rng.normal(0.0, 0.008, n_days) for s in sectors}

    # Sector-specific loadings: defensive sectors (Utilities, Staples) load
    # weakly on the market and strongly on their own factor, which is what
    # pushes them towards the periphery of a real correlation network.
    beta_by_sector = {
        "Technology": 1.20, "Financials": 1.15, "Healthcare": 0.80,
        "Energy": 0.90, "Consumer Staples": 0.55, "Consumer Discretionary": 1.05,
        "Industrials": 1.05, "Communication Services": 1.00,
        "Utilities": 0.45, "Real Estate": 0.85,
    }
    gamma_by_sector = {
        "Technology": 0.70, "Financials": 0.95, "Healthcare": 0.60,
        "Energy": 1.30, "Consumer Staples": 0.75, "Consumer Discretionary": 0.55,
        "Industrials": 0.60, "Communication Services": 0.50,
        "Utilities": 1.10, "Real Estate": 0.80,
    }

    columns = {}
    for ticker in tickers:
        sector = config.TICKER_TO_SECTOR.get(ticker, "Unknown")
        beta = beta_by_sector.get(sector, 1.0) * rng.uniform(0.85, 1.15)
        gamma = gamma_by_sector.get(sector, 0.7) * rng.uniform(0.85, 1.15)
        idio = rng.normal(0.0, 0.008, n_days)
        returns = beta * market + gamma * sector_factors[sector] + idio
        columns[ticker] = 100.0 * np.exp(np.cumsum(returns))

    prices = pd.DataFrame(columns, index=dates)
    logging.warning("SYNTHETIC DATA GENERATED - %d tickers x %d simulated trading days",
                    prices.shape[1], prices.shape[0])
    logging.warning("These prices are simulated. Do NOT report results based on "
                    "them in the thesis.")
    return prices


def _write_synthetic_markers(prices: pd.DataFrame) -> None:
    """Mark the raw and processed data folders as holding SIMULATED data.

    Two identical human-readable files are written, one beside the raw price
    panel and one beside the processed returns, because both directories are
    overwritten by a self-test run.  The raw one is also what
    :func:`cached_data_is_synthetic` looks for.
    """
    banner = (
        "!!! SYNTHETIC DATA WARNING !!!\n\n"
        "data/raw/adjusted_close_prices.csv and everything derived from it in\n"
        "data/processed/ currently contain SIMULATED prices produced by\n"
        "src/data_collection.generate_synthetic_prices().\n"
        "They were generated by a two-factor model, NOT downloaded from Yahoo Finance.\n\n"
        f"{config.SYNTHETIC_OUTPUTS_WARNING}\n\n"
        "While this file exists, a normal run (without --synthetic) will REFUSE to\n"
        "reuse the cached panel and will download real data instead. The file is\n"
        "removed automatically once a real download has succeeded.\n\n"
        "To obtain real data now, run:\n"
        "    python main.py --refresh\n\n"
        f"Simulated panel shape: {prices.shape[0]} rows x {prices.shape[1]} columns\n"
        f"Written: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    for marker in (config.SYNTHETIC_DATA_MARKER, config.SYNTHETIC_PROCESSED_MARKER):
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(banner)


def _remove_synthetic_markers() -> List[str]:
    """Delete the synthetic markers; return the paths actually removed.

    Called **only** after a real download has succeeded and its prices have been
    written to disk, so the markers can never be cleared while simulated data is
    still the content of ``data/raw``.
    """
    removed = []
    for marker in (config.SYNTHETIC_DATA_MARKER, config.SYNTHETIC_PROCESSED_MARKER):
        if os.path.exists(marker):
            os.remove(marker)
            removed.append(marker)
    return removed


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


def write_provenance(prices: pd.DataFrame, synthetic: bool, source: str,
                     requested: List[str], failed: List[str],
                     start: str, end: str) -> None:
    """Record, next to the raw panel, exactly where that panel came from.

    The human-readable marker file can be deleted by hand; this machine-readable
    record is a second, independent guard, and it is what lets a later run that
    *reuses* the cache still report when and how the data was obtained.
    """
    record = {
        "synthetic": bool(synthetic),
        "source": source,
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": start,
        "end_date": end,
        "n_tickers_requested": len(requested),
        "n_tickers_obtained": int(prices.shape[1]),
        "n_observations": int(prices.shape[0]),
        "tickers_failed": list(failed),
        "first_trading_day": str(prices.index.min().date()) if len(prices) else None,
        "last_trading_day": str(prices.index.max().date()) if len(prices) else None,
    }
    os.makedirs(os.path.dirname(config.PROVENANCE_FILE), exist_ok=True)
    with open(config.PROVENANCE_FILE, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def read_provenance() -> Dict[str, object] | None:
    """Read the provenance record of the cached panel, or ``None``."""
    if not os.path.exists(config.PROVENANCE_FILE):
        return None
    try:
        with open(config.PROVENANCE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.warning("Could not read the provenance record (%s).", exc)
        return None


def cached_data_is_synthetic() -> bool:
    """Is the cached price panel simulated rather than downloaded?

    Two independent signals are checked and **either** is enough to reject the
    cache: the human-readable marker file required by the project convention,
    and the machine-readable provenance record.  Erring towards "synthetic"
    is the safe direction: the cost of a false positive is one extra download,
    the cost of a false negative is a thesis written on simulated data.
    """
    if os.path.exists(config.SYNTHETIC_DATA_MARKER):
        return True
    provenance = read_provenance()
    return bool(provenance and provenance.get("synthetic"))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_cached_prices(path: str | None = None,
                       allow_synthetic: bool = False) -> pd.DataFrame | None:
    """Read a previously downloaded price panel, or return ``None``.

    The cache is **refused** when it holds simulated data and the caller did not
    explicitly ask for it (``allow_synthetic``).  This is the safety property
    the whole module is built around: a run started without ``--synthetic``
    must never silently continue on data left behind by a self-test, because
    the resulting tables and figures would look exactly like real results.

    Returning ``None`` makes :func:`collect_prices` fall through to a fresh
    Yahoo Finance download.
    """
    path = path or config.RAW_PRICES_FILE
    if not os.path.exists(path):
        return None

    if not allow_synthetic and cached_data_is_synthetic():
        logging.warning("")
        logging.warning("Synthetic cached data detected. Ignoring cached file and "
                        "forcing real Yahoo Finance download.")
        logging.warning("  cached file : %s", utils.relative(path))
        logging.warning("  marker      : %s", utils.relative(config.SYNTHETIC_DATA_MARKER))
        logging.warning("  the marker will be removed only once real data has been "
                        "downloaded successfully.")
        logging.warning("")
        return None

    try:
        prices = pd.read_csv(path, index_col=0)
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"
        return prices.sort_index()
    except Exception as exc:
        logging.warning("Could not read the cached price file (%s); re-downloading.", exc)
        return None


def _describe(prices: pd.DataFrame, requested: List[str], failed: List[str]) -> None:
    """Print the download report required by Section 5 of the specification."""
    logging.info("")
    logging.info("Download report")
    logging.info("  tickers requested        : %d", len(requested))
    logging.info("  tickers downloaded       : %d", prices.shape[1])
    logging.info("  tickers unavailable      : %d%s", len(failed),
                 (" -> " + ", ".join(failed)) if failed else "")
    if not prices.empty:
        logging.info("  first trading day        : %s", prices.index.min().date())
        logging.info("  last trading day         : %s", prices.index.max().date())
        logging.info("  number of trading days   : %d", prices.shape[0])
        coverage = prices.notna().mean().mul(100)
        logging.info("  mean per-ticker coverage : %.2f%% of trading days", coverage.mean())
        thin = coverage[coverage < 100.0].sort_values()
        if not thin.empty:
            logging.info("  tickers with gaps        : %s",
                         ", ".join(f"{t} ({c:.1f}%)" for t, c in thin.items()))


def write_synthetic_outputs_marker() -> str:
    """Drop the outputs warning file inside the synthetic output tree.

    A ``--synthetic`` run already writes to ``outputs_synthetic/`` rather than
    ``outputs/``, so the two can never mix.  This file is the belt to that
    pair of braces: it keeps the warning attached to the results themselves, so
    that a figure copied out of the folder can still be traced back.
    """
    banner = (
        "!!! SYNTHETIC OUTPUTS WARNING !!!\n\n"
        f"{config.SYNTHETIC_OUTPUTS_WARNING}\n\n"
        "Every figure, table and report in this folder was produced by\n"
        "    python main.py --synthetic\n"
        "from prices simulated by a two-factor model, not downloaded from\n"
        "Yahoo Finance.\n\n"
        "Real results are written to outputs/ instead. To produce them, run:\n"
        "    python main.py --refresh\n\n"
        f"Written: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    return utils.save_text(banner, config.SYNTHETIC_OUTPUTS_MARKER_NAME)


def write_data_source_report(collected: Dict[str, object],
                             n_observations: int | None = None,
                             dropped_tickers: List[str] | None = None,
                             start: str | None = None,
                             end: str | None = None) -> str:
    """Write ``data_source_report.txt``: what this run's numbers rest on.

    Produced at the end of a successful run so that a reader of the outputs can
    tell, from the outputs alone and without re-reading the log, whether they
    are looking at real Yahoo Finance data or at a self-test.
    """
    synthetic = bool(collected.get("synthetic"))
    prices: pd.DataFrame = collected["prices"]
    failed: List[str] = list(collected.get("failed") or [])
    dropped = sorted(set(dropped_tickers or []) | set(failed))
    provenance = collected.get("provenance") or {}

    headline = ("DATA SOURCE: SYNTHETIC SIMULATED DATA \u2014 NOT VALID FOR FINAL "
                "THESIS RESULTS" if synthetic else
                "DATA SOURCE: REAL YAHOO FINANCE DATA")

    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add(headline)
    add("=" * 78)
    add("")
    add(f"Run timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Data source              : {collected.get('source', 'unknown')}")
    add(f"Synthetic data           : {'YES' if synthetic else 'NO'}")
    add(f"Reused a cached panel    : {'YES' if collected.get('from_cache') else 'NO'}")
    if provenance.get("downloaded_at"):
        add(f"Panel obtained on        : {provenance['downloaded_at']}")
    add("")
    add(f"Start date (requested)   : {start or config.START_DATE}")
    add(f"End date (requested)     : {end or config.END_DATE}")
    if len(prices):
        add(f"First trading day        : {prices.index.min().date()}")
        add(f"Last trading day         : {prices.index.max().date()}")
    add("")
    add(f"Tickers requested        : {len(config.TICKERS)}")
    add(f"Tickers downloaded       : {prices.shape[1]}")
    add(f"Observations (raw days)  : {prices.shape[0]}")
    if n_observations is not None:
        add(f"Observations (log returns after cleaning) : {n_observations}")
    add("")
    if dropped:
        add(f"Tickers removed ({len(dropped)}) : {', '.join(dropped)}")
    else:
        add("Tickers removed          : none (the full universe survived)")
    add("")
    add(f"Outputs written to       : {utils.relative(config.OUTPUT_DIR)}")
    add("")
    add("-" * 78)
    if synthetic:
        add(config.SYNTHETIC_OUTPUTS_WARNING)
        add("")
        add("To produce real thesis results, run:")
        add("    python main.py --refresh")
    else:
        add("These outputs were generated from real Yahoo Finance data and are")
        add("valid material for the thesis, subject to the usual caveats about the")
        add("sample period and the stock universe documented in the README.")
    add("-" * 78)
    return utils.save_text("\n".join(lines), "data_source_report.txt")


def collect_prices(use_cache: bool = True,
                   synthetic: bool = False,
                   tickers: List[str] | None = None,
                   start: str | None = None,
                   end: str | None = None,
                   chunk_size: int | None = None) -> Dict[str, object]:
    """Top-level entry point of the data-collection stage.

    Order of preference:

    1. ``synthetic=True``  -> simulate the panel (self-test mode);
    2. ``use_cache=True`` and a **non-synthetic** cached panel exists
       -> reuse it (so that repeated runs do not hammer the Yahoo endpoint and
       so that the thesis figures are reproduced from a frozen data set);
    3. otherwise -> download from Yahoo Finance.

    Step 2 is the safety-critical one.  A cached panel left behind by a
    ``--synthetic`` self-test is detected (marker file and/or provenance
    record) and refused, and the run falls through to a real download.  The
    synthetic markers are cleared **only after** that download has succeeded
    and its prices have reached disk, so an interrupted or failed real run
    leaves the warnings in place rather than promoting simulated data to
    "real".

    Returns
    -------
    dict
        ``{"prices", "failed", "source", "synthetic", "from_cache", "provenance"}``
    """
    utils.ensure_directories()
    tickers = list(tickers or config.TICKERS)
    start = start or config.START_DATE
    end = end or config.END_DATE
    failed: List[str] = []
    from_cache = False
    removed_markers: List[str] = []

    if synthetic:
        # --- self-test branch: simulate, persist, and mark loudly ----------
        prices = generate_synthetic_prices(tickers, start, end)
        source = "synthetic factor model (SELF-TEST ONLY)"
        _describe(prices, tickers, failed)
        prices.to_csv(config.RAW_PRICES_FILE, float_format="%.6f")
        logging.info("  raw prices -> %s", utils.relative(config.RAW_PRICES_FILE))
        _write_synthetic_markers(prices)
        write_provenance(prices, True, source, tickers, failed, start, end)
        logging.warning("  synthetic marker -> %s",
                        utils.relative(config.SYNTHETIC_DATA_MARKER))
        return {"prices": prices, "failed": failed, "source": source,
                "synthetic": True, "from_cache": False,
                "provenance": read_provenance()}

    # --- real branch ------------------------------------------------------
    cached = load_cached_prices() if use_cache else None
    if cached is not None and not cached.empty:
        # Guaranteed non-synthetic: load_cached_prices refuses a marked panel.
        prices = cached
        from_cache = True
        source = f"cache of real data: {utils.relative(config.RAW_PRICES_FILE)}"
        logging.info("Using the cached price panel (%s).", source)
        provenance = read_provenance()
        if provenance:
            logging.info("  downloaded on %s from %s",
                         provenance.get("downloaded_at", "an unrecorded date"),
                         provenance.get("source", "an unrecorded source"))
        logging.info("  pass --refresh to download again.")
        failed = [t for t in tickers if t not in prices.columns]
        _describe(prices, tickers, failed)
        # The cache is left untouched: rewriting it would only risk corrupting
        # the frozen data set the thesis results are reproduced from.
    else:
        prices, failed = download_prices(tickers, start, end, chunk_size=chunk_size)
        source = "Yahoo Finance (yfinance)"
        if prices.empty:
            raise utils.PipelineError(
                "No price data could be downloaded. Check the network "
                "connection, or run 'python main.py --synthetic' to verify "
                "the pipeline offline with simulated data. Note that any "
                "synthetic warning markers are deliberately left in place, so "
                "the next real run will try to download again rather than "
                "reuse simulated prices."
            )
        _describe(prices, tickers, failed)

        # Persist the raw panel exactly as retrieved: this file is the
        # immutable input of every later stage and makes the run reproducible.
        prices.to_csv(config.RAW_PRICES_FILE, float_format="%.6f")
        logging.info("  raw prices -> %s", utils.relative(config.RAW_PRICES_FILE))

        # Real data is now on disk: and only now may the synthetic warnings go.
        removed_markers = _remove_synthetic_markers()
        write_provenance(prices, False, source, tickers, failed, start, end)
        if removed_markers:
            logging.info("  real data downloaded; removed the synthetic marker(s): %s",
                         ", ".join(utils.relative(m) for m in removed_markers))

    return {"prices": prices, "failed": failed, "source": source,
            "synthetic": False, "from_cache": from_cache,
            "provenance": read_provenance(), "removed_markers": removed_markers}
