"""
large_network_extension.py
==========================

Entry point of the **supplementary large-network extension**.

    python large_network_extension.py --refresh

This is a scalability and sector-structure robustness check, not a replacement
for anything.  It re-runs the thesis methodology - unchanged - on several
hundred S&P 500 constituents instead of 47, and then uses GICS sector metadata
as an external reference to ask whether stocks tend to connect to stocks of
their own sector.

Separation from the thesis run
------------------------------
The original 47-stock experiment behind the thesis is driven by ``main.py`` and
is untouched by this script:

* ``config.use_large_network_paths()`` is called before anything else, so this
  run reads and writes ``data/large/`` and ``outputs_large/`` and can never
  open ``data/raw/adjusted_close_prices.csv`` or write into ``outputs/``;
* the universe is swapped by rebinding ``config`` attributes, which only this
  script does - ``main.py`` always sees the 47 stocks of ``config.STOCKS``;
* a ``--synthetic`` self-test is pushed one level further out again, into
  ``data/large_synthetic/`` and ``outputs_large_synthetic/``.

What is kept identical to the thesis
------------------------------------
Sample period, adjusted closing prices, the calendar-alignment and
missing-value rules, logarithmic returns, the Pearson coefficient, the
threshold grid ``{0.3, 0.4, 0.5, 0.6, 0.7}`` and the ``|C_ij| >= tau`` edge
rule.  All of them are read from the same ``config`` entries the thesis uses,
and the graphs are built by the same ``src.graph_construction`` functions.

What is deliberately NOT done
-----------------------------
No portfolio construction, no community detection, no rolling windows, no
predictive modelling and no correlation measure other than Pearson.  The
extension adds exactly one new scientific ingredient: sector metadata, applied
strictly *after* the network has been built.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import (correlation, data_collection, graph_construction, large_network,
                 large_visualisation, network_export, preprocessing, utils)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_arguments(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary large-network extension of the thesis analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the cached large price panel and download again. "
                             "Use this for the final extension results.")
    parser.add_argument("--synthetic", action="store_true",
                        help="SELF-TEST ONLY: simulated prices, no network needed. "
                             "Writes to outputs_large_synthetic/ and must never be "
                             "reported as an empirical result.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=config.THRESHOLDS,
                        help="Threshold grid; defaults to the thesis grid.")
    parser.add_argument("--main-threshold", type=float, default=config.MAIN_THRESHOLD,
                        help="Threshold used for the sector-alignment analysis and "
                             "the network export.")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Tickers per Yahoo Finance request. A single request for "
                             "several hundred symbols is unreliable.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N constituents. For quick checks; "
                             "a real run must not use it.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figures; produce the tables only.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only warnings and errors on the console.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase 9 - data provenance report
# ---------------------------------------------------------------------------


def write_data_source_report(context: Dict[str, object]) -> str:
    """Write ``large_data_source_report.txt``: what these numbers rest on.

    Deliberately exhaustive about *loss*: the gap between the 503 constituents
    requested and the stocks that reach the network is the single most
    important caveat of the extension, so every stage that drops a ticker is
    reported with its count, its reason and its names.
    """
    synthetic = bool(context["synthetic"])
    universe: pd.DataFrame = context["universe"]
    provenance: Dict[str, object] = context["universe_provenance"]
    report: Dict[str, List[str]] = context["cleaning_report"]

    headline = ("DATA SOURCE: SYNTHETIC SIMULATED DATA — NOT VALID FOR THESIS RESULTS"
                if synthetic else "DATA SOURCE: REAL YAHOO FINANCE DATA")

    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add("LARGE-NETWORK EXTENSION — DATA SOURCE REPORT")
    add("=" * 78)
    add("")
    add(headline)
    add("")
    add(f"Run timestamp              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Data source                : {context['data_source']}")
    add(f"Synthetic data             : {'YES' if synthetic else 'NO'}")
    add(f"Reused a cached panel      : {'YES' if context['from_cache'] else 'NO'}")
    add(f"Output tree                : {utils.relative(config.OUTPUT_DIR)}")
    add("")
    add("-" * 78)
    add("1. CONSTITUENT LIST")
    add("-" * 78)
    add(f"  File                     : {utils.relative(config.LARGE_UNIVERSE_FILE)}")
    if provenance:
        add(f"  Source repository        : {provenance.get('source_repository')}")
        add(f"  Repository HEAD          : {provenance.get('source_repository_head')}")
        add(f"  Membership file          : {provenance.get('membership_file')}")
        add(f"  Membership snapshot date : {provenance.get('membership_snapshot_date')}"
            "   (last index change on or before the sample end)")
        add(f"  GICS metadata file       : {provenance.get('metadata_file')}")
        for entry in provenance.get("metadata_commits", []):
            add(f"     commit {entry['commit'][:12]}  dated {entry['date']}"
                f"  [{entry.get('role', 'unspecified')}]")
        add(f"  Classified from the pre-sample-end snapshot  : "
            f"{provenance.get('n_from_primary_metadata', 'n/a')}")
        fallback = provenance.get("fallback_metadata_tickers") or []
        add(f"  Classified from the post-end fallback        : "
            f"{provenance.get('n_from_fallback_metadata', 'n/a')}"
            + (f" -> {', '.join(fallback)}" if fallback else ""))
        add(f"  Retrieved on             : {provenance.get('generated_at')}")
        add(f"  Constituents in snapshot : {provenance.get('n_constituents_in_snapshot')}")
        unresolved = provenance.get("unresolved_excluded") or []
        add(f"  Excluded, no GICS record : {len(unresolved)}"
            + (f" -> {', '.join(unresolved)}" if unresolved else ""))
    else:
        add("  Provenance record not found; see data/reference/CONSTITUENTS_SOURCE.md")
    add(f"  Sectors represented      : {universe['gics_sector'].nunique()} (GICS)")
    add("")
    add("  UNIVERSE CONSTRUCTION - HOW TO DESCRIBE IT")
    add("  The extension uses a point-in-time S&P 500 constituent universe observed")
    add("  at the end of the sample period. Consequently, the analysis is conditional")
    add("  on end-of-period index membership and should not be interpreted as a")
    add("  survivorship-bias-free representation of the S&P 500 throughout 2019-2025.")
    add("")
    add("-" * 78)
    add("2. SAMPLE PERIOD")
    add("-" * 78)
    add(f"  Start date (requested)   : {config.START_DATE}")
    add(f"  End date (requested)     : {config.END_DATE}   (exclusive, as yfinance treats it)")
    if context["first_trading_day"]:
        add(f"  First trading day        : {context['first_trading_day']}")
        add(f"  Last trading day         : {context['last_trading_day']}")
    add(f"  Raw trading days          : {context['n_raw_days']}")
    add(f"  Aligned trading days      : {context['n_clean_days']}")
    add(f"  Log-return observations   : {context['n_observations']}")
    add("")
    add("-" * 78)
    add("3. TICKER ACCOUNTING")
    add("-" * 78)
    add(f"  Requested                : {context['n_requested']}")
    add(f"  Downloaded               : {context['n_downloaded']}")
    add(f"  Retained in the network  : {context['n_stocks']}")
    add(f"  Excluded in total        : {context['n_requested'] - context['n_stocks']}")
    add("")

    buckets = [
        ("not returned by Yahoo Finance", context["failed"]),
        ("empty over the whole window", report.get("dropped_all_missing", [])),
        (f"more than {100 * config.MAX_MISSING_FRACTION:.0f}% of prices missing "
         "after calendar alignment", report.get("dropped_too_sparse", [])),
        (f"fewer than {config.MIN_OBSERVATIONS} return observations",
         report.get("dropped_too_short", [])),
    ]
    for reason, tickers in buckets:
        add(f"  [{len(tickers):>3}] {reason}")
        if tickers:
            for start in range(0, len(tickers), 10):
                add("        " + ", ".join(sorted(tickers)[start:start + 10]))
    add("")
    add("  Note. The dominant exclusion reason is expected to be the missing-price")
    add("  rule: a company that joined the index, listed or was spun off after")
    add("  2019-01-02 has no price for the early part of the sample and therefore")
    add("  cannot enter a correlation matrix estimated on one common calendar.")
    add("  The rule is the thesis rule, applied unchanged; it was not relaxed to")
    add("  retain more stocks.")
    add("")
    add("-" * 78)
    add("4. CONFIGURATION")
    add("-" * 78)
    add(f"  Thresholds               : {', '.join(f'{t:g}' for t in context['thresholds'])}")
    add(f"  Main threshold           : {context['main_threshold']:g}")
    add(f"  Edge rule                : {config.EDGE_FILTER_MODE}  (|C_ij| >= tau)")
    add("  Correlation              : Pearson, daily log returns")
    add(f"  Max missing fraction     : {config.MAX_MISSING_FRACTION}")
    add(f"  Max forward-fill days    : {config.MAX_FORWARD_FILL_DAYS}")
    add(f"  Min observations         : {config.MIN_OBSERVATIONS}")
    add(f"  Cross-sector rule        : degree >= {config.LARGE_CROSS_SECTOR_MIN_DEGREE}, "
        f"same-sector share <= {config.LARGE_CROSS_SECTOR_MAX_SHARE_RATIO:g}x the "
        "chance benchmark,")
    add("                             and dominant neighbour sector differs from "
        "the official one")
    add("")
    add("-" * 78)
    add("5. USE OF SECTOR METADATA")
    add("-" * 78)
    add("  Sector labels were NOT used in graph construction. An edge exists if and")
    add("  only if |C_ij| >= tau. Sector metadata enters only afterwards, as an")
    add("  external reference for interpretation and validation. It is an")
    add("  approximate reference, not a statistical ground truth: GICS classifies a")
    add("  company by its business activity, while an edge here records co-movement")
    add("  of returns over one sample period.")
    add("")
    add("-" * 78)
    if synthetic:
        add(config.SYNTHETIC_OUTPUTS_WARNING)
        add("")
        add("To produce real extension results, run:")
        add("    python large_network_extension.py --refresh")
    else:
        add("These outputs were generated from real Yahoo Finance data.")
    add("-" * 78)
    return utils.save_text("\n".join(lines), "large_data_source_report.txt")


# ---------------------------------------------------------------------------
# Phase 12 - internal consistency checks
# ---------------------------------------------------------------------------


def verify_outputs(context: Dict[str, object]) -> None:
    """Check the exported tables against each other; raise on any disagreement.

    These are the checks the specification asks for, run automatically rather
    than by eye: node counts coherent, edge counts matching the graphs,
    isolated nodes still represented, component sizes summing to N.
    """
    utils.subsection("Internal consistency checks")
    problems: List[str] = []

    n_stocks = int(context["n_stocks"])
    threshold_table: pd.DataFrame = context["threshold_table"]
    graphs = context["graphs"]

    # Every threshold graph must carry the full node set: thresholding removes
    # edges, never nodes.
    for _, row in threshold_table.iterrows():
        if int(row["nodes"]) != n_stocks:
            problems.append(f"tau={row['threshold']}: {int(row['nodes'])} nodes "
                            f"but {n_stocks} stocks survived cleaning")

    for threshold, graph in graphs.items():
        row = threshold_table.loc[threshold_table["threshold"].round(6)
                                  == round(threshold, 6)]
        if len(row) != 1:
            problems.append(f"tau={threshold}: not exactly one row in the metrics table")
            continue
        row = row.iloc[0]
        if int(row["edges"]) != graph.number_of_edges():
            problems.append(f"tau={threshold}: table says {int(row['edges'])} edges, "
                            f"graph has {graph.number_of_edges()}")
        sizes = [len(c) for c in nx.connected_components(graph)]
        if sum(sizes) != graph.number_of_nodes():
            problems.append(f"tau={threshold}: component sizes sum to {sum(sizes)}, "
                            f"not {graph.number_of_nodes()}")
        if int(row["components"]) != len(sizes):
            problems.append(f"tau={threshold}: table says {int(row['components'])} "
                            f"components, graph has {len(sizes)}")
        if int(row["isolated_nodes"]) != sum(1 for s in sizes if s == 1):
            problems.append(f"tau={threshold}: isolated-node count disagrees with "
                            "the singleton components of the graph")

        # The reported same-sector edge share must equal the share actually
        # carried by the graph's own edges.  This is the one column read out of
        # graph_summary as a percentage and re-expressed as a fraction, so it is
        # worth confirming rather than assuming the conversion.
        edges = [bool(d["same_sector"]) for _, _, d in graph.edges(data=True)]
        observed = (sum(edges) / len(edges)) if edges else float("nan")
        reported = float(row["same_sector_edge_share"])
        if edges and abs(observed - reported) > 1e-6:
            problems.append(f"tau={threshold}: same_sector_edge_share {reported:.6f} "
                            f"but the graph's edges give {observed:.6f}")

    # The alignment table covers exactly the non-isolated nodes of the main graph.
    alignment: pd.DataFrame = context["alignment"]
    main_graph = graphs[context["main_threshold"]]
    expected = {n for n, d in main_graph.degree() if d > 0}
    actual = set(alignment["ticker"])
    if expected != actual:
        problems.append(f"alignment table covers {len(actual)} nodes, expected "
                        f"{len(expected)} non-isolated nodes")
    if len(alignment) and not (
            alignment["same_sector_neighbors"] + alignment["cross_sector_neighbors"]
            == alignment["degree"]).all():
        problems.append("same-sector + cross-sector neighbours != degree for some node")

    # Isolated nodes must still be represented in the exported node file.
    nodes: pd.DataFrame = context["export"]["nodes"]
    if len(nodes) != main_graph.number_of_nodes():
        problems.append(f"node export has {len(nodes)} rows for a graph of "
                        f"{main_graph.number_of_nodes()} nodes")
    n_isolated_graph = sum(1 for _, d in main_graph.degree() if d == 0)
    if int(nodes["is_isolated"].sum()) != n_isolated_graph:
        problems.append("isolated nodes are missing from the node export")

    # Sector metadata must not influence graph construction.  Rather than
    # assert this from reading the code, rebuild the main threshold graph with
    # the sector labels randomly permuted between stocks and check that the
    # edge set is bit-for-bit the same.  If a sector ever leaked into the edge
    # rule, permuting the labels would move at least one edge.
    saved_sectors = dict(config.TICKER_TO_SECTOR)
    try:
        tickers = list(saved_sectors)
        shuffled = list(tickers)
        random.Random(config.RANDOM_SEED).shuffle(shuffled)
        config.TICKER_TO_SECTOR = {t: saved_sectors[other]
                                   for t, other in zip(tickers, shuffled)}
        permuted = graph_construction.build_threshold_graph(
            context["correlation"], context["main_threshold"])
    finally:
        config.TICKER_TO_SECTOR = saved_sectors

    if ({frozenset(e) for e in permuted.edges()}
            != {frozenset(e) for e in main_graph.edges()}):
        problems.append("the edge set changed when sector labels were permuted: "
                        "sector metadata is influencing graph construction")

    if problems:
        raise utils.PipelineError("Consistency checks failed:\n  - "
                                  + "\n  - ".join(problems))

    logging.info("  node counts coherent across all thresholds        : OK")
    logging.info("  edge counts match the constructed graphs          : OK")
    logging.info("  component sizes sum to N at every threshold       : OK")
    logging.info("  isolated nodes represented in the node export     : OK")
    logging.info("  neighbour counts sum to degree in the alignment   : OK")
    logging.info("  same-sector edge share matches the graph's edges  : OK")
    logging.info("  edges unchanged under a sector-label permutation  : OK")
    logging.info("    (sector metadata is used only AFTER construction)")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_extension(args: argparse.Namespace) -> Dict[str, object]:
    started = time.time()
    np.random.seed(config.RANDOM_SEED)

    # Redirect every path BEFORE any directory is created or any log opened, so
    # that nothing this run does can reach the thesis data or outputs.
    config.use_large_network_paths(synthetic=args.synthetic)
    utils.ensure_directories()
    utils.setup_logging("large_network_run_log.txt", verbose=not args.quiet)

    thresholds = sorted(float(t) for t in args.thresholds)
    main_threshold = float(args.main_threshold)
    if main_threshold not in thresholds:
        thresholds = sorted(thresholds + [main_threshold])
        logging.warning("Main threshold %.2f was not in the grid; it has been added.",
                        main_threshold)

    utils.section("Large-network extension of the stock correlation analysis")
    if args.synthetic:
        logging.warning("")
        logging.warning("*** SELF-TEST MODE: prices are SIMULATED, not downloaded.      ***")
        logging.warning("*** No number from this run may be reported as a result.       ***")
        logging.warning("*** Outputs go to %-45s ***",
                        utils.relative(config.OUTPUT_DIR) + "/")

    # -- universe ----------------------------------------------------------
    utils.subsection("Stock universe")
    universe = large_network.load_universe()
    provenance = large_network.universe_provenance()
    if args.limit:
        universe = universe.head(args.limit).reset_index(drop=True)
        logging.warning("  --limit %d in force: this is NOT a full run.", args.limit)
    large_network.apply_universe_to_config(universe)

    logging.info("  constituent file         : %s",
                 utils.relative(config.LARGE_UNIVERSE_FILE))
    logging.info("  membership snapshot      : %s",
                 provenance.get("membership_snapshot_date", "unrecorded"))
    logging.info("  constituents requested   : %d across %d GICS sectors",
                 len(universe), universe["gics_sector"].nunique())
    logging.info("  sample period            : %s to %s", config.START_DATE, config.END_DATE)
    logging.info("  thresholds               : %s", ", ".join(f"{t:g}" for t in thresholds))
    logging.info("  main threshold           : %g", main_threshold)
    logging.info("  edge rule                : %s (|C_ij| >= tau)", config.EDGE_FILTER_MODE)

    # -- 1. data -----------------------------------------------------------
    utils.section("Stage 1 - data collection")
    collected = data_collection.collect_prices(
        use_cache=not args.refresh, synthetic=args.synthetic,
        start=config.START_DATE, end=config.END_DATE,
        chunk_size=args.chunk_size)
    prices = collected["prices"]

    # -- 2. preprocessing ---------------------------------------------------
    utils.section("Stage 2 - preprocessing and log returns")
    processed = preprocessing.run_preprocessing(prices)
    returns = processed["returns"]

    # -- 3. correlation -----------------------------------------------------
    utils.section("Stage 3 - correlation matrix")
    corr = correlation.compute_correlation_matrix(returns)
    utils.save_table(corr.round(6), "large_correlation_matrix.csv", index=True)
    # The pair-level table is deliberately not exported: at this size it has
    # ~100 000 rows and adds nothing the matrix does not already contain.
    sector_matrix = correlation.sector_correlation_matrix(corr)
    utils.save_table(sector_matrix.round(4), "large_sector_correlation_matrix.csv",
                     index=True)

    # -- 4. threshold graphs -------------------------------------------------
    utils.section("Stages 4 and 5 - threshold graphs and network metrics")
    # The complete (unfiltered) graph of the thesis pipeline is skipped here:
    # with several hundred nodes every pair is an edge by construction, so its
    # summary carries no information while its path-length statistics cost
    # minutes. The threshold graphs themselves are built by exactly the same
    # function the thesis uses.
    graphs = {}
    summaries = []
    for threshold in thresholds:
        graph = graph_construction.build_threshold_graph(corr, threshold)
        graphs[threshold] = graph
        summary = graph_construction.graph_summary(graph, threshold=threshold)
        summaries.append(summary)
        logging.info("  tau = %.2f : %d nodes, %d edges, density %.4f, "
                     "giant %d (%.1f%%), isolated %d, assortativity %+.4f",
                     threshold, summary["n_nodes"], summary["n_edges"],
                     summary["density"], summary["largest_component_size"],
                     summary["largest_component_pct"], summary["n_isolated_nodes"],
                     summary["sector_assortativity"])

    threshold_table = large_network.threshold_metrics_table(summaries)
    utils.save_table(threshold_table, "large_threshold_network_metrics.csv")
    utils.save_table(pd.DataFrame(summaries), "large_threshold_network_metrics_full.csv")

    # -- 5. sector alignment (Phases 6 and 7) --------------------------------
    utils.section("Stages 6 and 7 - sector metadata as an external reference")
    main_graph = graphs[main_threshold]
    alignment_results = large_network.run_sector_alignment(main_graph, main_threshold)

    # -- 6. network export (Phase 8) -----------------------------------------
    utils.section("Stage 8 - network export")
    utils.subsection(f"Rebuilding and exporting the network at tau = {main_threshold:.2f}")
    export = network_export.compute_network_export(corr, main_threshold)
    stats = network_export.verify_export(export["nodes"], export["edges"],
                                         export["graph"])
    logging.info("  nodes / edges            : %d / %d", stats["n_nodes"], stats["n_edges"])
    logging.info("  articulation points      : %d", stats["n_articulation_points"])
    export_dir = (config.NETWORK_EXPORT_DIR if not args.synthetic
                  else os.path.join(config.OUTPUT_DIR, "network_export"))
    network_export.write_export(export["nodes"], export["edges"],
                                "large_network", main_threshold, export_dir)

    # -- accumulate ----------------------------------------------------------
    report = processed["report"]
    context: Dict[str, object] = {
        "universe": universe, "universe_provenance": provenance,
        "synthetic": bool(collected["synthetic"]),
        "data_source": collected["source"], "from_cache": bool(collected["from_cache"]),
        "failed": list(collected["failed"]), "cleaning_report": report,
        "n_requested": len(universe), "n_downloaded": int(prices.shape[1]),
        "n_stocks": int(returns.shape[1]), "n_observations": int(returns.shape[0]),
        "n_raw_days": int(prices.shape[0]), "n_clean_days": int(processed["prices"].shape[0]),
        "first_trading_day": (str(prices.index.min().date()) if len(prices) else None),
        "last_trading_day": (str(prices.index.max().date()) if len(prices) else None),
        "thresholds": thresholds, "main_threshold": main_threshold,
        "correlation": corr, "graphs": graphs, "threshold_table": threshold_table,
        "alignment": alignment_results["alignment"],
        "alignment_summary": alignment_results["summary"],
        "by_sector": alignment_results["by_sector"],
        "cross_sector": alignment_results["cross_sector"],
        "export": export,
    }

    # -- 7. consistency checks (Phase 12) ------------------------------------
    utils.section("Stage 9 - quality control")
    verify_outputs(context)

    # -- 8. provenance report (Phase 9) --------------------------------------
    utils.section("Stage 10 - data provenance report")
    write_data_source_report(context)

    # -- 9. figures (Phase 10) -----------------------------------------------
    if not args.no_figures:
        utils.section("Stage 11 - figures")
        context["figures"] = large_visualisation.generate_large_figures(
            threshold_table, context["alignment"], context["by_sector"],
            main_graph, main_threshold)

    # -- summary -------------------------------------------------------------
    utils.section("Run complete")
    logging.info("Constituents requested : %d", context["n_requested"])
    logging.info("Stocks in the network  : %d", context["n_stocks"])
    logging.info("Return observations    : %d", context["n_observations"])
    logging.info("Tables                 : %s", utils.relative(config.TABLES_DIR))
    logging.info("Figures                : %s", utils.relative(config.FIGURES_DIR))
    logging.info("Network export         : %s", utils.relative(export_dir))
    logging.info("Elapsed                : %.1f seconds", time.time() - started)
    if context["synthetic"]:
        logging.warning("")
        logging.warning("DATA SOURCE: SYNTHETIC SIMULATED DATA - NOT VALID AS A RESULT")
        logging.warning("%s", config.SYNTHETIC_OUTPUTS_WARNING)
    else:
        logging.info("")
        logging.info("DATA SOURCE: REAL YAHOO FINANCE DATA")
    return context


def main(argv: List[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        run_extension(args)
    except KeyboardInterrupt:
        logging.error("Interrupted by the user.")
        return 130
    except utils.PipelineError as exc:
        logging.error("")
        logging.error("The extension could not continue:")
        logging.error("  %s", exc)
        return 2
    except Exception as exc:
        logging.exception("The extension stopped with an unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
