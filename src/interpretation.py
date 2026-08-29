"""
interpretation.py
=================

Sections 15 and 18 of the specification:

* the **exploratory investment interpretation framework**, which attaches a
  descriptive label (hub / peripheral / bridge / clustered) to every stock on
  the basis of its position in the network, and
* the **master summary report** written to
  ``outputs/logs/summary_interpretation.txt``, which collects in one place the
  numbers that Chapter 4 has to quote.

The framework deliberately produces *descriptions*, not recommendations.  A
label such as "hub" says that the stock co-moves strongly with many others over
the estimation sample; it says nothing about whether the stock is cheap, well
managed or likely to rise.  Every exported artefact carries
``config.INVESTMENT_DISCLAIMER`` verbatim.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def build_interpretation_framework(node_metrics: pd.DataFrame,
                                   mst_metrics: pd.DataFrame,
                                   articulation: pd.DataFrame,
                                   components: pd.DataFrame,
                                   threshold: float) -> pd.DataFrame:
    """Classify every stock by its position in the network.

    Four (deliberately non-exclusive) categories:

    ``is_central``    composite centrality above ``config.CENTRAL_QUANTILE``.
                      The stock co-moves with a large part of the market, so a
                      position in it carries a lot of *systematic* exposure and
                      little that is specific to the company.
    ``is_peripheral`` composite centrality below ``config.PERIPHERAL_QUANTILE``,
                      or a leaf of the MST.  Over this sample the stock shared
                      comparatively little variation with the rest of the
                      universe, which is the property a diversifying position
                      is looking for.
    ``is_bridge``     the stock is an articulation point: removing it would
                      disconnect the network.  It links groups that are
                      otherwise only weakly related, so it transmits
                      co-movement between them.
    ``cluster_label`` the connected component the stock belongs to, described by
                      that component's dominant sector.  Two stocks in the same
                      cluster are, by construction, connected by a chain of
                      strong correlations.

    A stock can be both central and a bridge, or peripheral and isolated; the
    flags are properties, not a partition.
    """
    metrics = node_metrics.copy()
    mst = mst_metrics.set_index("ticker")
    bridge_stocks = set(articulation["ticker"]) if len(articulation) else set()

    component_info = (components.drop_duplicates("component_id")
                      .set_index("component_id")) if len(components) else pd.DataFrame()

    central_cut = metrics["composite_centrality"].quantile(config.CENTRAL_QUANTILE)
    peripheral_cut = metrics["composite_centrality"].quantile(config.PERIPHERAL_QUANTILE)

    rows = []
    for _, row in metrics.iterrows():
        ticker = row["ticker"]
        mst_degree = int(mst["mst_degree"].get(ticker, 0))
        mst_leaf = bool(mst["is_leaf"].get(ticker, False))
        component_id = int(row["component_id"])

        is_central = bool(row["composite_centrality"] >= central_cut and row["degree"] > 0)
        is_peripheral = bool(row["composite_centrality"] <= peripheral_cut or mst_leaf)
        is_bridge = ticker in bridge_stocks
        is_isolated = bool(row["is_isolated"])

        dominant = (component_info["component_dominant_sector"].get(component_id, "n/a")
                    if len(component_info) else "n/a")
        size = (int(component_info["component_size"].get(component_id, 0))
                if len(component_info) else 0)
        cluster_label = (f"C{component_id} ({size} stocks, mainly {dominant})"
                         if size > 1 else f"C{component_id} (singleton)")

        rows.append({
            "ticker": ticker,
            "company": row.get("company", ticker),
            "sector": row["sector"],
            "degree": int(row["degree"]),
            "strength": float(row["strength"]),
            "degree_centrality": float(row["degree_centrality"]),
            "betweenness_centrality": float(row["betweenness_centrality"]),
            "closeness_centrality": float(row["closeness_centrality"]),
            "composite_centrality": float(row["composite_centrality"]),
            "clustering_coefficient": float(row["clustering_coefficient"]),
            "component_id": component_id,
            "component_size": int(row["component_size"]),
            "cluster_label": cluster_label,
            "mst_degree": mst_degree,
            "mst_leaf": mst_leaf,
            "is_central": is_central,
            "is_peripheral": is_peripheral,
            "is_bridge": is_bridge,
            "is_isolated": is_isolated,
            "network_role": _role_label(is_central, is_peripheral, is_bridge, is_isolated),
            "interpretation": _interpretation_sentence(
                ticker, row, mst_degree, mst_leaf,
                is_central, is_peripheral, is_bridge, is_isolated, cluster_label),
            "disclaimer": config.INVESTMENT_DISCLAIMER,
        })

    table = pd.DataFrame(rows)
    table.attrs["threshold"] = threshold
    table.attrs["central_cut"] = float(central_cut)
    table.attrs["peripheral_cut"] = float(peripheral_cut)
    return table.sort_values("composite_centrality", ascending=False).reset_index(drop=True)


def _role_label(is_central: bool, is_peripheral: bool,
                is_bridge: bool, is_isolated: bool) -> str:
    """A single headline role, used for grouping in the tables and the report."""
    if is_isolated:
        return "Isolated"
    if is_bridge and is_central:
        return "Central bridge"
    if is_bridge:
        return "Bridge"
    if is_central:
        return "Hub / central"
    if is_peripheral:
        return "Peripheral"
    return "Intermediate"


def _interpretation_sentence(ticker: str, row: pd.Series, mst_degree: int,
                             mst_leaf: bool, is_central: bool, is_peripheral: bool,
                             is_bridge: bool, is_isolated: bool,
                             cluster_label: str) -> str:
    """One descriptive sentence per stock, phrased without any recommendation."""
    parts: List[str] = []
    if is_isolated:
        parts.append(f"No correlation above the threshold: {ticker} moved largely "
                     f"independently of this universe over the sample")
    elif is_central:
        parts.append(f"Highly connected ({int(row['degree'])} strong links, MST degree "
                     f"{mst_degree}): {ticker} co-moves with a large part of the market, "
                     f"so a position in it is dominated by systematic exposure")
    elif is_peripheral:
        parts.append(f"Loosely attached ({int(row['degree'])} strong links"
                     + (", leaf of the MST" if mst_leaf else "")
                     + f"): {ticker} shared comparatively little variation with the rest "
                       f"of the universe over the sample")
    else:
        parts.append(f"Intermediate position ({int(row['degree'])} strong links, MST "
                     f"degree {mst_degree})")

    if is_bridge:
        parts.append("acts as an articulation point, holding otherwise separate groups "
                     "of stocks together, so shocks propagate through it")
    if not is_isolated:
        parts.append(f"belongs to {cluster_label}")
    return "; ".join(parts) + "."


def framework_summary(framework: pd.DataFrame) -> pd.DataFrame:
    """Counts and sector composition of each network role."""
    rows = []
    for role, group in framework.groupby("network_role"):
        sectors = group["sector"].value_counts()
        rows.append({
            "network_role": role,
            "n_stocks": len(group),
            "share_of_universe": len(group) / len(framework),
            "mean_degree": float(group["degree"].mean()),
            "mean_composite_centrality": float(group["composite_centrality"].mean()),
            "mean_mst_degree": float(group["mst_degree"].mean()),
            "dominant_sector": sectors.index[0],
            "tickers": ", ".join(sorted(group["ticker"])),
        })
    return (pd.DataFrame(rows).sort_values("n_stocks", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------


def _rule(character: str = "=", width: int = 78) -> str:
    return character * width


def build_summary_report(context: Dict[str, object]) -> str:
    """Assemble ``outputs/logs/summary_interpretation.txt``.

    ``context`` is the dictionary of results accumulated by ``main.run_pipeline``.
    The report is organised in the order of Chapter 4 so that it can be read
    top-to-bottom while writing the chapter.
    """
    lines: List[str] = []
    add = lines.append

    threshold = context["main_threshold"]
    graph = context["graphs"][threshold]
    node_metrics: pd.DataFrame = context["metrics"][threshold]["node_metrics"]
    components: pd.DataFrame = context["metrics"][threshold]["components"]
    composition: pd.DataFrame = context["metrics"][threshold]["composition"]
    threshold_table: pd.DataFrame = context["threshold_table"]
    corr_stats: Dict[str, object] = context["correlation_stats"]
    bfs = context["traversal"]["bfs"]
    dfs = context["traversal"]["dfs"]
    mst_summary: Dict[str, object] = context["mst"]["summary"]
    mst_metrics: pd.DataFrame = context["mst"]["node_metrics"]
    framework: pd.DataFrame = context["framework"]

    add(_rule())
    add("ANALYSIS OF STOCK CORRELATION NETWORKS THROUGH GRAPH TRAVERSAL ALGORITHMS")
    add("Summary of empirical results")
    add(_rule())
    add("")
    if context.get("synthetic"):
        add("*" * 78)
        add("WARNING: THIS RUN USED SIMULATED PRICES (--synthetic).")
        add("The numbers below are a pipeline self-test and must NOT be reported")
        add("as empirical results in the thesis.")
        add("*" * 78)
        add("")

    # -- 0. Setup ----------------------------------------------------------
    add("0. EXPERIMENTAL SET-UP")
    add(_rule("-"))
    add(f"  Data source            : {context['data_source']}")
    add(f"  Sample period          : {context['start_date']} to {context['end_date']}")
    add(f"  Trading days           : {context['n_observations']}")
    add(f"  Stocks requested       : {context['n_requested']}")
    add(f"  Stocks analysed        : {context['n_stocks']} "
        f"across {node_metrics['sector'].nunique()} sectors")
    if context.get("dropped_tickers"):
        add(f"  Stocks removed         : {', '.join(context['dropped_tickers'])}")
    add("  Returns                : daily logarithmic, r_t = ln(P_t / P_(t-1))")
    add(f"  Edge rule              : |C| >= tau  (mode: {config.EDGE_FILTER_MODE})")
    add(f"  Thresholds examined    : {', '.join(f'{t:g}' for t in context['thresholds'])}")
    add(f"  Main threshold         : {threshold:g}")
    add(f"  BFS source stock       : {bfs['source']}")
    add("")
    add(utils.environment_report())
    add("")

    # -- 1. Correlation ----------------------------------------------------
    add("1. CORRELATION STRUCTURE")
    add(_rule("-"))
    add(f"  Distinct pairs                 : {corr_stats['n_pairs']}")
    add(f"  Mean correlation               : {corr_stats['mean_correlation']:+.4f}")
    add(f"  Mean absolute correlation      : {corr_stats['mean_abs_correlation']:.4f}")
    add(f"  Range                          : {corr_stats['min_correlation']:+.4f} to "
        f"{corr_stats['max_correlation']:+.4f}")
    add(f"  Negative correlations          : {100 * corr_stats['share_negative']:.1f}% of pairs")
    add(f"  Mean within-sector correlation : {corr_stats['mean_within_sector']:+.4f}")
    add(f"  Mean cross-sector correlation  : {corr_stats['mean_across_sector']:+.4f}")
    strongest = corr_stats["strongest_pairs"].iloc[0]
    weakest = corr_stats["weakest_pairs"].iloc[0]
    add(f"  Strongest pair                 : {strongest['stock_a']}-{strongest['stock_b']} "
        f"({strongest['correlation']:+.4f})")
    add(f"  Weakest pair                   : {weakest['stock_a']}-{weakest['stock_b']} "
        f"({weakest['correlation']:+.4f})")
    add("")
    add("  Reading: a positive average correlation reflects the common market factor;")
    add("  the gap between the within-sector and cross-sector averages is the part of")
    add("  the structure the network analysis is designed to expose.")
    add("")

    # -- 2. Threshold comparison ------------------------------------------
    add("2. RESPONSE OF THE NETWORK TO THE THRESHOLD")
    add(_rule("-"))
    display = threshold_table[["threshold", "n_edges", "density", "average_degree",
                               "average_clustering", "n_connected_components",
                               "largest_component_size", "largest_component_pct",
                               "n_isolated_nodes"]]
    for line in utils.format_table(display, max_rows=len(display)).splitlines():
        add("  " + line)
    add("")
    first, last = threshold_table.iloc[0], threshold_table.iloc[-1]
    add(f"  Raising tau from {first['threshold']:g} to {last['threshold']:g} removes "
        f"{int(first['n_edges'] - last['n_edges'])} of the {int(first['n_edges'])} edges "
        f"({100 * (1 - last['n_edges'] / max(1, first['n_edges'])):.1f}%),")
    add(f"  takes the density from {first['density']:.3f} to {last['density']:.3f}, and takes the")
    add(f"  giant component from {first['largest_component_pct']:.1f}% to "
        f"{last['largest_component_pct']:.1f}% of the universe, leaving "
        f"{int(last['n_isolated_nodes'])} isolated stock(s).")
    add("")
    add("  Reading: the threshold acts as a magnifying glass. Low values retain the")
    add("  common market factor and produce a single dense blob; high values retain")
    add("  only the strongest, mostly sector-internal relationships and the network")
    add("  fragments. The interesting range is the one in which the giant component")
    add("  starts to break up, because that is where the *structure* rather than the")
    add("  market factor becomes visible.")
    add("")

    # -- 3. Main network ---------------------------------------------------
    main_row = threshold_table.loc[threshold_table["threshold"] == threshold].iloc[0]
    add(f"3. THE NETWORK AT THE MAIN THRESHOLD (tau = {threshold:g})")
    add(_rule("-"))
    add(f"  Nodes / edges                  : {int(main_row['n_nodes'])} / "
        f"{int(main_row['n_edges'])}")
    add(f"  Density                        : {main_row['density']:.4f}")
    add(f"  Average degree                 : {main_row['average_degree']:.2f}")
    add(f"  Average clustering coefficient : {main_row['average_clustering']:.4f}")
    add(f"  Connected components           : {int(main_row['n_connected_components'])}")
    add(f"  Giant component                : {int(main_row['largest_component_size'])} stocks "
        f"({main_row['largest_component_pct']:.1f}%)")
    add(f"  Isolated stocks                : {int(main_row['n_isolated_nodes'])}"
        + (f" -> {main_row['isolated_nodes']}" if main_row["isolated_nodes"] else ""))
    add(f"  Intra-sector edges             : {main_row['intra_sector_edge_pct']:.1f}%")
    add(f"  Sector assortativity           : {main_row['sector_assortativity']:+.4f}")
    if main_row["avg_shortest_path_giant"] == main_row["avg_shortest_path_giant"]:
        add(f"  Average path length (giant)    : {main_row['avg_shortest_path_giant']:.3f}")
        add(f"  Diameter (giant)               : {int(main_row['diameter_giant'])}")
    add("")
    add("  Most central stocks (composite of degree, betweenness, closeness and")
    add("  eigenvector centrality):")
    for _, row in node_metrics.nlargest(config.TOP_K, "composite_centrality").iterrows():
        add(f"    {row['ticker']:<6} degree {int(row['degree']):>3}  "
            f"betweenness {row['betweenness_centrality']:.3f}  "
            f"closeness {row['closeness_centrality']:.3f}  ({row['sector']})")
    add("")
    add("  Most peripheral stocks:")
    for _, row in node_metrics.nsmallest(config.TOP_K, "composite_centrality").iterrows():
        add(f"    {row['ticker']:<6} degree {int(row['degree']):>3}  "
            f"composite {row['composite_centrality']:.3f}  ({row['sector']})")
    add("")

    # -- 4. Components -----------------------------------------------------
    add("4. CONNECTED COMPONENTS")
    add(_rule("-"))
    sizes = components.drop_duplicates("component_id")[
        ["component_id", "component_size", "component_dominant_sector",
         "component_dominant_sector_share"]]
    for _, row in sizes.iterrows():
        members = ", ".join(sorted(
            components.loc[components["component_id"] == row["component_id"], "ticker"]))
        add(f"  Component {int(row['component_id'])}: {int(row['component_size'])} stock(s), "
            f"dominant sector {row['component_dominant_sector']} "
            f"({100 * row['component_dominant_sector_share']:.0f}%)")
        add(f"    {members}")
    add("")
    pure = composition[(composition["share_of_component"] == 1.0)
                       & (composition["component_size"] > 1)]
    if len(pure):
        add("  Components made of a single sector:")
        for _, row in pure.iterrows():
            add(f"    C{int(row['component_id'])}: {row['sector']} -> {row['tickers']}")
    else:
        add("  No component is made of a single sector at this threshold: every group")
        add("  of connected stocks mixes at least two industries.")
    add("")
    add("  Reading: components answer the question 'does the market break into")
    add("  separate blocks of co-moving stocks?'. A single dominant component means")
    add("  the market is cohesive at this level of correlation; several components,")
    add("  or isolated stocks, mean part of the universe detaches once weak links")
    add("  are removed.")
    add("")

    # -- 5. BFS ------------------------------------------------------------
    add(f"5. BREADTH-FIRST SEARCH FROM {bfs['source']}")
    add(_rule("-"))
    add(f"  Reachable stocks               : {len(bfs['reachable']) - 1} of "
        f"{graph.number_of_nodes() - 1}")
    add(f"  Unreachable stocks             : {len(bfs['unreachable'])}"
        + (f" -> {', '.join(bfs['unreachable'])}" if bfs["unreachable"] else ""))
    add(f"  Eccentricity of the source     : {bfs['eccentricity']} edges")
    add(f"  Mean distance from the source  : {bfs['mean_distance']:.3f} edges")
    add("")
    for _, row in bfs["layers"].iterrows():
        add(f"    distance {int(row['bfs_distance'])}: {int(row['n_stocks']):>2} stock(s)")
        add(f"      {row['tickers']}")
    add("")
    add(f"  Validation of the custom BFS against NetworkX: "
        f"{'all checks passed' if all(bfs['validation'].values()) else 'MISMATCH - see the log'}")
    add("")
    add("  Reading: the BFS distance is the minimum number of strong correlations")
    add("  needed to travel from the source to another stock. Distance 1 means direct")
    add("  co-movement; larger distances mean the relationship exists only through")
    add("  intermediaries. This is topological proximity within an estimated")
    add("  dependence structure - it is descriptive, not predictive.")
    add("")

    # -- 6. DFS ------------------------------------------------------------
    add("6. DEPTH-FIRST SEARCH, ARTICULATION POINTS AND BRIDGES")
    add(_rule("-"))
    add(f"  DFS forest                     : {len(dfs['components'])} tree(s) covering "
        f"{len(dfs['order'])} stocks")
    add(f"  Articulation points            : {len(dfs['articulation'])}")
    if len(dfs["articulation"]):
        for _, row in dfs["articulation"].iterrows():
            add(f"    {row['ticker']:<6} ({row['sector']}) - removing it takes the network "
                f"from {int(row['components_before'])} to {int(row['components_after'])} "
                f"component(s)")
            if row["detached_stocks"]:
                add(f"      would detach: {row['detached_stocks']}")
    add(f"  Bridges                        : {len(dfs['bridges'])}")
    if len(dfs["bridges"]):
        for _, row in dfs["bridges"].head(config.TOP_K).iterrows():
            add(f"    {row['stock_a']:<6} - {row['stock_b']:<6} C = {row['correlation']:+.3f}  "
                f"({'cross-sector' if row['cross_sector'] else 'same sector'}), "
                f"splits off {int(row['smaller_side_size'])} stock(s)")
    structural = context["traversal"].get("structural")
    if structural is not None and len(structural):
        add("")
        add("  Structural fragility across the whole threshold grid:")
        for _, row in structural.iterrows():
            add(f"    tau = {row['threshold']:.2f} : {int(row['n_edges']):>4} edge(s), "
                f"{int(row['n_components']):>2} component(s), "
                f"{int(row['n_articulation_points']):>2} articulation point(s), "
                f"{int(row['n_bridges']):>2} bridge(s)"
                + (f"   -> {row['articulation_points']}"
                   if row["articulation_points"] else ""))
        add("")
        add("  A dense network has no articulation points *by construction*: with many")
        add("  redundant paths, removing any single stock leaves the rest connected.")
        add("  Cut vertices therefore appear only once the threshold is high enough for")
        add("  the network to become sparse, and the threshold at which the first one")
        add("  appears is itself a measure of how redundant the correlation structure is.")
    add("")
    add("  Reading: an articulation point is a stock whose removal would disconnect")
    add("  the network, and a bridge is a single relationship holding two groups")
    add("  together. Structurally they are the channels through which co-movement")
    add("  passes between otherwise weakly related parts of the market. This is a")
    add("  property of an estimated network over one sample. It is NOT a statement")
    add("  that these stocks are good or bad investments.")
    add("")

    # -- 7. MST ------------------------------------------------------------
    add("7. MINIMUM SPANNING TREE")
    add(_rule("-"))
    add(f"  Links retained                 : {mst_summary['n_edges']} of "
        f"{mst_summary['n_nodes'] * (mst_summary['n_nodes'] - 1) // 2} possible pairs")
    add(f"  Hub of the tree                : {mst_summary['hub_stock']} "
        f"({mst_summary['hub_sector']}), degree {mst_summary['max_degree']}")
    add(f"  Leaves                         : {mst_summary['n_leaves']} "
        f"({100 * mst_summary['leaf_share']:.1f}%)")
    add(f"  Normalised tree length         : {mst_summary['normalised_tree_length']:.4f}")
    add(f"  Diameter / average path        : {mst_summary['diameter']} / "
        f"{mst_summary['average_shortest_path']:.3f} links")
    add(f"  Intra-sector links             : "
        f"{100 * mst_summary['intra_sector_edge_share']:.1f}% "
        f"(random benchmark {100 * mst_summary['random_benchmark_share']:.1f}%, "
        f"ratio {mst_summary['sector_clustering_ratio']:.2f}x)")
    add("")
    add("  Most central stocks of the backbone:")
    for _, row in mst_metrics.head(config.TOP_K).iterrows():
        add(f"    {row['ticker']:<6} MST degree {int(row['mst_degree']):>2}  "
            f"betweenness {row['mst_betweenness']:.3f}  ({row['sector']})")
    add("")
    add("  Reading: the MST keeps the N-1 strongest links that connect every stock")
    add("  without a cycle, and it does so without any threshold. The fact that a")
    add("  large majority of those links join stocks of the same sector - while no")
    add("  sector information entered the computation - is the clearest evidence in")
    add("  this study that industry membership is encoded in return co-movement.")
    add("")

    # -- 8. Investment framework -------------------------------------------
    add("8. EXPLORATORY INVESTMENT INTERPRETATION")
    add(_rule("-"))
    summary = framework_summary(framework)
    for _, row in summary.iterrows():
        add(f"  {row['network_role']:<16} {int(row['n_stocks']):>2} stock(s) "
            f"({100 * row['share_of_universe']:.0f}%), mean degree "
            f"{row['mean_degree']:.1f}")
        add(f"    {row['tickers']}")
    add("")
    if context.get("portfolio") is not None:
        comparison: pd.DataFrame = context["portfolio"]["comparison"]
        add("  Illustrative portfolios (equally weighted, in-sample, no costs):")
        display_cols = ["portfolio", "n_stocks", "average_pairwise_correlation",
                        "annualised_return", "annualised_volatility",
                        "sharpe_ratio_rf0", "max_drawdown"]
        for line in utils.format_table(comparison[display_cols],
                                       max_rows=len(comparison)).splitlines():
            add("    " + line)
        add("")
        add(f"    A holds: {context['portfolio']['sector_portfolio']}")
        add(f"    B holds: {context['portfolio']['network_portfolio']}")
        add("")
        add("    The two portfolios are formed and evaluated on the SAME sample, so the")
        add("    comparison shows only that network position captures information about")
        add("    the correlation structure of a basket. It is not evidence of")
        add("    out-of-sample performance and it is not a trading strategy.")
        add("")
    add("  " + _rule("!"))
    add("  " + config.INVESTMENT_DISCLAIMER)
    add("  " + _rule("!"))
    add("")

    # -- 9. Limitations ----------------------------------------------------
    add("9. LIMITATIONS (for Chapter 5)")
    add(_rule("-"))
    add("  - The Pearson coefficient measures LINEAR dependence only and is sensitive")
    add("    to the fat tails of daily returns; non-linear or tail dependence is not")
    add("    captured.")
    add("  - A single correlation matrix estimated over several years averages over")
    add("    calm and turbulent regimes. Correlations are known to rise sharply in")
    add("    crises, so the network is a period average, not a stable object.")
    add("  - The threshold is a modelling choice. Section 2 above shows how strongly")
    add("    the conclusions depend on it, which is why every result is reported for a")
    add("    range of thresholds rather than for a single one.")
    add("  - The universe is 47 large-capitalisation US stocks chosen to balance")
    add("    sectors. A different or larger universe would produce a different network.")
    add("  - Correlation is not causation: an edge records co-movement, not influence.")
    add("  - Nothing in this study forecasts returns. Predictive modelling on top of")
    add("    the network features is listed as a possible extension, not as a result.")
    add("")
    add(_rule())
    add("End of report.")
    add(_rule())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_interpretation(node_metrics: pd.DataFrame,
                       mst_metrics: pd.DataFrame,
                       articulation: pd.DataFrame,
                       components: pd.DataFrame,
                       threshold: float) -> Dict[str, object]:
    """Build and export the investment interpretation framework.

    Writes
    ------
    ``outputs/tables/investment_interpretation_framework.csv``
    ``outputs/tables/investment_framework_summary.csv``
    """
    utils.subsection("Exploratory investment interpretation framework")
    framework = build_interpretation_framework(node_metrics, mst_metrics,
                                               articulation, components, threshold)
    utils.save_table(framework, "investment_interpretation_framework.csv")

    summary = framework_summary(framework)
    utils.save_table(summary, "investment_framework_summary.csv")

    for _, row in summary.iterrows():
        logging.info("  %-16s %2d stock(s) : %s", row["network_role"],
                     int(row["n_stocks"]), row["tickers"][:90])
    logging.warning("  %s", config.INVESTMENT_DISCLAIMER)

    return {"framework": framework, "summary": summary}
