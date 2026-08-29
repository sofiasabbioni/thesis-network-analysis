"""
main.py
=======

Entry point of the thesis project

    "Analysis of Stock Correlation Networks through Graph Traversal Algorithms"

Running ``python main.py`` executes the complete empirical pipeline described
in Chapter 3 and writes every table, figure and report that Chapter 4 refers
to.  The stages are:

    1.  download adjusted closing prices                 -> data/raw
    2.  clean, align and compute log returns             -> data/processed
    3.  estimate the Pearson correlation matrix          -> outputs/tables
    4.  build the complete correlation graph
    5.  build the threshold-filtered graphs
    6.  compute node- and graph-level network metrics
    7.  analyse the connected components
    8.  run Breadth-First Search from a source stock
    9.  run Depth-First Search, find articulation points and bridges
    10. build the Minimum Spanning Tree (Mantegna distance)
    11. produce every figure                             -> outputs/figures
    12. build the exploratory investment framework
    13. run the illustrative portfolio comparison
    14. write the summary interpretation                 -> outputs/logs

Every parameter lives in ``config.py``; the command-line options below simply
override a few of them for a single run.

Examples
--------
    python main.py                          # full run, real data (cached if present)
    python main.py --refresh                # force a fresh download
    python main.py --synthetic --no-figures # quick offline self-test (SIMULATED data)
    python main.py --refresh --no-figures   # quick real-data check, tables only
    python main.py --main-threshold 0.6 --source MSFT
    python main.py --start 2020-01-01 --end 2024-12-31
    python main.py --thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List

import numpy as np

import config
from src import (correlation, data_collection, graph_construction, interpretation,
                 metrics, mst_analysis, portfolio, preprocessing, traversal_analysis,
                 utils, visualisation)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_arguments(argv: List[str] | None = None) -> argparse.Namespace:
    """Command-line overrides for the most frequently changed parameters."""
    parser = argparse.ArgumentParser(
        description="Stock correlation networks and graph traversal algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start", default=config.START_DATE,
                        help="First day of the sample (YYYY-MM-DD).")
    parser.add_argument("--end", default=config.END_DATE,
                        help="Last day of the sample (YYYY-MM-DD).")
    parser.add_argument("--thresholds", type=float, nargs="+", default=config.THRESHOLDS,
                        help="Correlation thresholds to examine.")
    parser.add_argument("--main-threshold", type=float, default=config.MAIN_THRESHOLD,
                        help="Threshold used for the detailed BFS/DFS/component analysis.")
    parser.add_argument("--source", default=config.SOURCE_STOCK,
                        help="Source stock for the Breadth-First Search.")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the cached price file and download again. "
                             "Use this for the final thesis results.")
    parser.add_argument("--synthetic", action="store_true",
                        help="SELF-TEST ONLY: run on simulated prices, no network "
                             "needed. Writes to outputs_synthetic/, marks the data "
                             "as synthetic, and must never be used for the thesis.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation: validates the download and the "
                             "numerical tables in a fraction of the time.")
    parser.add_argument("--no-portfolio", action="store_true",
                        help="Skip the optional illustrative portfolio section.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only warnings and errors on the console.")
    return parser.parse_args(argv)


def _apply_overrides(args: argparse.Namespace) -> None:
    """Write the command-line overrides back into ``config`` .

    The analysis modules read their defaults from ``config``, so overriding the
    module attributes keeps a single source of truth for the whole run and
    makes the exported reports describe what was actually executed.
    """
    config.START_DATE = args.start
    config.END_DATE = args.end
    config.THRESHOLDS = sorted(float(t) for t in args.thresholds)
    config.MAIN_THRESHOLD = float(args.main_threshold)
    config.SOURCE_STOCK = args.source

    if config.MAIN_THRESHOLD not in config.THRESHOLDS:
        config.THRESHOLDS = sorted(config.THRESHOLDS + [config.MAIN_THRESHOLD])
        logging.warning("MAIN_THRESHOLD %.2f was not in THRESHOLDS; it has been added.",
                        config.MAIN_THRESHOLD)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def generate_figures(context: Dict[str, object]) -> Dict[str, str]:
    """Produce every figure of the thesis from the accumulated results."""
    utils.section("Stage 11 - figures")
    visualisation.apply_style()
    paths: Dict[str, str] = {}

    corr = context["correlation"]
    main_threshold = context["main_threshold"]
    graphs = context["graphs"]

    # -- correlation ------------------------------------------------------
    paths["correlation_heatmap"] = visualisation.plot_correlation_heatmap(corr)
    paths["sector_correlation_heatmap"] = visualisation.plot_sector_correlation_heatmap(
        context["sector_matrix"])
    paths["correlation_distribution"] = visualisation.plot_correlation_distribution(
        context["correlation_stats"]["pairs"], context["thresholds"])

    # -- one network figure and one degree figure per threshold ------------
    for threshold in context["thresholds"]:
        graph = graphs[threshold]
        tag = utils.threshold_tag(threshold)
        paths[f"network_{tag}"] = visualisation.plot_threshold_network(graph, threshold)
        paths[f"degree_{tag}"] = visualisation.plot_degree_distribution(
            context["metrics"][threshold]["node_metrics"], threshold)

    # -- components (main threshold, plus its neighbours for comparison) ----
    component_thresholds = sorted({main_threshold} | {
        t for t in context["thresholds"] if abs(t - main_threshold) <= 0.1001})
    for threshold in component_thresholds:
        tag = utils.threshold_tag(threshold)
        paths[f"components_{tag}"] = visualisation.plot_connected_components(
            graphs[threshold], threshold)

    # -- traversals --------------------------------------------------------
    bfs = context["traversal"]["bfs"]
    paths["bfs_tree"] = visualisation.plot_bfs_tree(
        graphs[main_threshold], bfs, main_threshold)
    paths["bfs_distance_map"] = visualisation.plot_bfs_distance_map(
        graphs[main_threshold], bfs, main_threshold)
    paths["bfs_reachability"] = visualisation.plot_bfs_reachability(
        context["traversal"]["reachability"], bfs["source"])
    for threshold in component_thresholds:
        tag = utils.threshold_tag(threshold)
        dfs_result = context["traversal"]["dfs_by_threshold"].get(threshold)
        if dfs_result is not None:
            paths[f"dfs_tree_{tag}"] = visualisation.plot_dfs_tree(
                graphs[threshold], dfs_result, threshold)

    # -- MST ---------------------------------------------------------------
    paths["mst"] = visualisation.plot_mst(context["mst"]["mst"],
                                          context["mst"]["node_metrics"])

    # -- threshold comparison ---------------------------------------------
    paths.update({f"threshold_{k}": v for k, v in
                  visualisation.plot_threshold_comparisons(context["threshold_table"]).items()})

    # -- portfolios --------------------------------------------------------
    if context.get("portfolio") is not None:
        paths["portfolio_cumulative"] = visualisation.plot_portfolio_cumulative_returns(
            context["portfolio"]["cumulative"])
        paths["portfolio_bars"] = visualisation.plot_portfolio_comparison_bars(
            context["portfolio"]["comparison"])

    logging.info("")
    logging.info("  %d figures written to %s", len(paths), utils.relative(config.FIGURES_DIR))
    return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _banner(lines: List[str]) -> None:
    """Log a boxed warning whose sides line up whatever the message lengths."""
    width = max(len(line) for line in lines)
    logging.warning("")
    for line in lines:
        logging.warning("*** %s ***", line.ljust(width))


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    """Execute every stage of the analysis and return the accumulated results."""
    started = time.time()
    np.random.seed(config.RANDOM_SEED)

    # A self-test must not be able to touch the real results, so the whole
    # output tree is redirected to outputs_synthetic/ *before* any directory is
    # created and before the log file is opened.
    if args.synthetic:
        config.use_synthetic_output_paths()

    utils.ensure_directories()
    utils.setup_logging(verbose=not args.quiet)
    _apply_overrides(args)

    utils.section("Stock correlation networks through graph traversal algorithms")
    logging.info("Sample period   : %s to %s", config.START_DATE, config.END_DATE)
    logging.info("Universe        : %d tickers across %d sectors",
                 len(config.TICKERS), len(config.SECTORS))
    logging.info("Thresholds      : %s", ", ".join(f"{t:g}" for t in config.THRESHOLDS))
    logging.info("Main threshold  : %g", config.MAIN_THRESHOLD)
    logging.info("BFS source      : %s", config.SOURCE_STOCK)
    logging.info("Edge rule       : %s", config.EDGE_FILTER_MODE)
    logging.info("Output tree     : %s", utils.relative(config.OUTPUT_DIR))
    if args.synthetic:
        _banner([
            "SELF-TEST MODE: prices are SIMULATED, not downloaded.",
            "Do not report any number from this run in the thesis.",
            f"Outputs go to {utils.relative(config.OUTPUT_DIR)}/, not outputs/.",
        ])
    elif data_collection.cached_data_is_synthetic():
        # The previous run was a self-test.  Its price cache will be refused in
        # stage 1, but outputs/ may still hold results from before this project
        # separated the two output trees.
        _banner([
            "The cached price panel is SYNTHETIC. It will be ignored and",
            "real data will be downloaded instead.",
            "If outputs/ holds results from an earlier synthetic run,",
            "delete its contents before writing up the thesis.",
        ])

    # -- 1. Data -----------------------------------------------------------
    utils.section("Stage 1 - data collection")
    collected = data_collection.collect_prices(
        use_cache=not args.refresh, synthetic=args.synthetic,
        start=config.START_DATE, end=config.END_DATE)
    if collected["synthetic"]:
        data_collection.write_synthetic_outputs_marker()

    # -- 2. Preprocessing --------------------------------------------------
    utils.section("Stage 2 - preprocessing and log returns")
    processed = preprocessing.run_preprocessing(collected["prices"])
    returns = processed["returns"]

    # -- 3. Correlation ----------------------------------------------------
    utils.section("Stage 3 - correlation matrix")
    correlation_results = correlation.run_correlation_analysis(returns)
    corr = correlation_results["correlation"]

    # -- 4/5. Graphs -------------------------------------------------------
    utils.section("Stages 4 and 5 - graph construction")
    graph_results = graph_construction.run_graph_construction(corr, config.THRESHOLDS)
    graphs = graph_results["graphs"]

    # -- 6/7. Metrics and components --------------------------------------
    utils.section("Stages 6 and 7 - network metrics and connected components")
    metric_results = metrics.run_metrics(graphs, config.MAIN_THRESHOLD)

    # -- 8/9. Traversals ---------------------------------------------------
    utils.section("Stages 8 and 9 - BFS and DFS")
    traversal_results = traversal_analysis.run_traversal_analysis(
        graphs, config.MAIN_THRESHOLD, config.SOURCE_STOCK)

    # -- 10. MST -----------------------------------------------------------
    utils.section("Stage 10 - minimum spanning tree")
    mst_results = mst_analysis.run_mst_analysis(corr)

    # -- accumulate --------------------------------------------------------
    dropped = (processed["report"]["dropped_all_missing"]
               + processed["report"]["dropped_too_sparse"]
               + processed["report"]["dropped_too_short"]
               + collected["failed"])
    context: Dict[str, object] = {
        "start_date": config.START_DATE, "end_date": config.END_DATE,
        "thresholds": config.THRESHOLDS, "main_threshold": config.MAIN_THRESHOLD,
        "data_source": collected["source"], "synthetic": collected["synthetic"],
        "n_requested": len(config.TICKERS), "n_stocks": returns.shape[1],
        "n_observations": returns.shape[0],
        "dropped_tickers": sorted(set(dropped)),
        "prices": processed["prices"], "returns": returns,
        "summary_statistics": processed["summary"],
        "correlation": corr, "correlation_stats": correlation_results["stats"],
        "sector_matrix": correlation_results["sector_matrix"],
        "full_graph": graph_results["full_graph"],
        "graphs": graphs, "threshold_table": graph_results["threshold_table"],
        "metrics": metric_results, "traversal": traversal_results,
        "mst": mst_results, "portfolio": None,
    }

    # -- 12. Investment framework -----------------------------------------
    utils.section("Stage 12 - exploratory investment interpretation")
    framework_results = interpretation.run_interpretation(
        node_metrics=metric_results[config.MAIN_THRESHOLD]["node_metrics"],
        mst_metrics=mst_results["node_metrics"],
        articulation=traversal_results["dfs"]["articulation"],
        components=metric_results[config.MAIN_THRESHOLD]["components"],
        threshold=config.MAIN_THRESHOLD)
    context["framework"] = framework_results["framework"]

    # -- 13. Portfolios ----------------------------------------------------
    if not args.no_portfolio:
        utils.section("Stage 13 - illustrative portfolio comparison")
        context["portfolio"] = portfolio.run_portfolio_illustration(
            log_returns=returns, corr=corr,
            node_metrics=metric_results[config.MAIN_THRESHOLD]["node_metrics"],
            mst_metrics=mst_results["node_metrics"])

    # -- 11. Figures -------------------------------------------------------
    if not args.no_figures:
        context["figures"] = generate_figures(context)

    # -- 14. Summary report ------------------------------------------------
    utils.section("Stage 14 - summary interpretation")
    report = interpretation.build_summary_report(context)
    utils.save_text(report, "summary_interpretation.txt")

    # Provenance statement: written last, so its presence certifies that the
    # whole run completed on the data source it names.
    data_collection.write_data_source_report(
        collected,
        n_observations=context["n_observations"],
        dropped_tickers=context["dropped_tickers"],
        start=config.START_DATE, end=config.END_DATE)

    elapsed = time.time() - started
    utils.section("Run complete")
    logging.info("Stocks analysed  : %d", context["n_stocks"])
    logging.info("Trading days     : %d", context["n_observations"])
    logging.info("Tables           : %s", utils.relative(config.TABLES_DIR))
    logging.info("Figures          : %s", utils.relative(config.FIGURES_DIR))
    logging.info("Logs and reports : %s", utils.relative(config.LOGS_DIR))
    logging.info("Elapsed          : %.1f seconds", elapsed)
    if context["synthetic"]:
        logging.warning("")
        logging.warning("DATA SOURCE: SYNTHETIC SIMULATED DATA - NOT VALID FOR FINAL "
                        "THESIS RESULTS")
        logging.warning("%s", config.SYNTHETIC_OUTPUTS_WARNING)
        logging.warning("Run 'python main.py --refresh' to produce real results in "
                        "outputs/.")
    else:
        logging.info("")
        logging.info("DATA SOURCE: REAL YAHOO FINANCE DATA")
    return context


def main(argv: List[str] | None = None) -> int:
    """Console entry point; returns a process exit code."""
    args = parse_arguments(argv)
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        logging.error("Interrupted by the user.")
        return 130
    except utils.PipelineError as exc:
        # An anticipated failure: the message is the useful part, not the stack.
        logging.error("")
        logging.error("The pipeline could not continue:")
        logging.error("  %s", exc)
        return 2
    except Exception as exc:
        logging.exception("The pipeline stopped with an unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
