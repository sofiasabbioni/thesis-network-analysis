"""
graph_construction.py
=====================

Step 4 of the empirical pipeline: turn the correlation matrix into **graphs**.

Formal set-up used in Chapter 3
-------------------------------
Let ``G = (V, E, w)`` be an undirected weighted graph where

* ``V`` is the set of stocks (one node per ticker, carrying a ``sector``
  attribute);
* ``E`` is a set of unordered pairs of stocks;
* ``w`` assigns to each edge the Pearson correlation of the two stocks.

Two families of graphs are built.

1. **The complete correlation graph** keeps all N(N-1)/2 pairs.  It contains
   the full information of the correlation matrix but is useless as a picture:
   every node is adjacent to every other, so degree, distance and connectivity
   are constant by construction.  It is retained as the reference object from
   which the filtered graphs are derived.

2. **Threshold-filtered graphs** keep only the pairs whose dependence exceeds a
   cut-off ``tau``.  This is the standard way of extracting a sparse, readable
   structure from a dense correlation matrix.  With
   ``config.EDGE_FILTER_MODE == "absolute"`` an edge survives when
   ``|C_ij| >= tau``: a strongly *negative* relationship is treated as an
   equally informative link, because for the purposes of risk and
   diversification a pair of stocks that systematically moves in opposite
   directions is strongly related, not unrelated.  The alternative
   ``"positive"`` mode (keep the edge when ``C_ij >= tau``) is available in the
   configuration for robustness checks.

The threshold is a genuine modelling parameter, not a nuisance: the whole of
Section 4.3 studies how the structure of the network responds to it.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def correlation_to_distance(correlation: float, use_abs: bool = False) -> float:
    """Mantegna's (1999) correlation-to-distance transformation.

        d_ij = sqrt( 2 * (1 - C_ij) )

    ``d`` is a proper metric on the set of stocks: it is non-negative, it is
    zero only for perfectly correlated series, it is symmetric and it satisfies
    the triangle inequality.  That is precisely what makes it a legitimate edge
    weight for a Minimum Spanning Tree.  See ``mst_analysis.py``.
    """
    c = abs(correlation) if use_abs else correlation
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.sqrt(2.0 * (1.0 - c)))


def _add_nodes(graph: nx.Graph, tickers: List[str]) -> None:
    """Add every stock as a node with its descriptive attributes.

    Nodes are added *before* any edge so that a stock with no surviving
    correlation still appears in the graph as an isolated node.  Isolation is a
    result to report, not a stock to silently drop.
    """
    for ticker in tickers:
        graph.add_node(
            ticker,
            ticker=ticker,
            sector=config.TICKER_TO_SECTOR.get(ticker, "Unknown"),
            company=config.COMPANY_NAMES.get(ticker, ticker),
        )


def build_full_graph(corr: pd.DataFrame,
                     use_abs_for_distance: bool | None = None) -> nx.Graph:
    """Build the complete weighted correlation graph.

    Edge attributes
    ---------------
    ``correlation``     signed Pearson coefficient
    ``abs_correlation`` its absolute value
    ``weight``          alias of ``abs_correlation`` (NetworkX's default weight
                        key, so weighted routines measure strength of
                        association)
    ``distance``        Mantegna distance, used by the MST
    ``same_sector``     True when both endpoints share a sector
    """
    use_abs = (config.MST_USE_ABS_CORRELATION
               if use_abs_for_distance is None else use_abs_for_distance)
    tickers = list(corr.columns)
    graph = nx.Graph(name="complete_correlation_graph")
    _add_nodes(graph, tickers)

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = float(corr.iloc[i, j])
            graph.add_edge(
                a, b,
                correlation=c,
                abs_correlation=abs(c),
                weight=abs(c),
                distance=correlation_to_distance(c, use_abs=use_abs),
                same_sector=(config.TICKER_TO_SECTOR.get(a) ==
                             config.TICKER_TO_SECTOR.get(b)),
            )
    return graph


def build_threshold_graph(corr: pd.DataFrame,
                          threshold: float,
                          mode: str | None = None) -> nx.Graph:
    """Build the graph that keeps only edges above ``threshold``.

    Parameters
    ----------
    corr : DataFrame
        Correlation matrix.
    threshold : float
        Cut-off ``tau``.
    mode : {"absolute", "positive"}, optional
        Filtering rule; defaults to ``config.EDGE_FILTER_MODE``.
    """
    mode = (mode or config.EDGE_FILTER_MODE).lower()
    if mode not in ("absolute", "positive"):
        raise ValueError(f"Unknown EDGE_FILTER_MODE {mode!r}; use 'absolute' or 'positive'.")

    tickers = list(corr.columns)
    graph = nx.Graph(name=f"threshold_graph_{utils.threshold_tag(threshold)}")
    graph.graph["threshold"] = float(threshold)
    graph.graph["filter_mode"] = mode
    _add_nodes(graph, tickers)

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = float(corr.iloc[i, j])
            keep = (abs(c) >= threshold) if mode == "absolute" else (c >= threshold)
            if keep:
                graph.add_edge(
                    a, b,
                    correlation=c,
                    abs_correlation=abs(c),
                    weight=abs(c),
                    distance=correlation_to_distance(c),
                    same_sector=(config.TICKER_TO_SECTOR.get(a) ==
                                 config.TICKER_TO_SECTOR.get(b)),
                )
    return graph


# ---------------------------------------------------------------------------
# Graph-level description
# ---------------------------------------------------------------------------


def graph_summary(graph: nx.Graph, threshold: float | None = None) -> Dict[str, object]:
    """Compute every graph-level statistic reported in the threshold table.

    All quantities are defined so that they remain meaningful for a *possibly
    disconnected* graph, which the high-threshold networks certainly are:
    average path length and diameter are computed inside the largest connected
    component only, and that restriction is stated in the exported table.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    possible = n * (n - 1) / 2 if n > 1 else 0

    degrees = dict(graph.degree())
    strengths = dict(graph.degree(weight="weight"))

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    giant = components[0] if components else set()
    isolated = sorted(nx.isolates(graph))

    edge_corrs = [d["correlation"] for _, _, d in graph.edges(data=True)]
    same_sector_edges = [d["same_sector"] for _, _, d in graph.edges(data=True)]

    # Path-based statistics: only defined on the giant component and only
    # informative when it has at least two nodes.
    if len(giant) > 1:
        sub = graph.subgraph(giant)
        avg_path = float(nx.average_shortest_path_length(sub))
        diameter = int(nx.diameter(sub))
    else:
        avg_path, diameter = np.nan, np.nan

    # Assortativity by sector ("homophily"): the tendency of a stock to be
    # linked to stocks of its own sector.  Positive values are direct evidence
    # of sectoral clustering.
    try:
        sector_assortativity = float(
            nx.attribute_assortativity_coefficient(graph, "sector")) if m > 0 else np.nan
    except Exception:
        sector_assortativity = np.nan
    try:
        degree_assortativity = float(
            nx.degree_assortativity_coefficient(graph)) if m > 1 else np.nan
    except Exception:
        degree_assortativity = np.nan

    return {
        "threshold": float(threshold) if threshold is not None
        else graph.graph.get("threshold", np.nan),
        "n_nodes": n,
        "n_edges": m,
        "possible_edges": int(possible),
        "edge_retention_pct": 100.0 * m / possible if possible else np.nan,
        "density": float(nx.density(graph)),
        "average_degree": float(np.mean(list(degrees.values()))) if n else np.nan,
        "max_degree": int(max(degrees.values())) if n else 0,
        "average_strength": float(np.mean(list(strengths.values()))) if n else np.nan,
        "average_clustering": float(nx.average_clustering(graph)) if n > 2 else np.nan,
        "average_clustering_weighted": (float(nx.average_clustering(graph, weight="weight"))
                                        if n > 2 else np.nan),
        "transitivity": float(nx.transitivity(graph)) if n > 2 else np.nan,
        "n_connected_components": len(components),
        "largest_component_size": len(giant),
        "largest_component_pct": 100.0 * len(giant) / n if n else np.nan,
        "second_component_size": len(components[1]) if len(components) > 1 else 0,
        "n_isolated_nodes": len(isolated),
        "isolated_nodes": ", ".join(isolated) if isolated else "",
        "mean_edge_correlation": float(np.mean(edge_corrs)) if edge_corrs else np.nan,
        "min_edge_correlation": float(np.min(edge_corrs)) if edge_corrs else np.nan,
        "intra_sector_edge_pct": (100.0 * float(np.mean(same_sector_edges))
                                  if same_sector_edges else np.nan),
        "sector_assortativity": sector_assortativity,
        "degree_assortativity": degree_assortativity,
        "avg_shortest_path_giant": avg_path,
        "diameter_giant": diameter,
    }


def _log_summary(summary: Dict[str, object], label: str) -> None:
    """Human-readable one-block report of a graph summary."""
    logging.info("  %s", label)
    logging.info("    nodes / edges          : %d / %d  (%.1f%% of all possible pairs)",
                 summary["n_nodes"], summary["n_edges"],
                 summary["edge_retention_pct"] if summary["edge_retention_pct"] == summary["edge_retention_pct"] else float("nan"))
    logging.info("    density                : %.4f", summary["density"])
    logging.info("    average degree         : %.2f", summary["average_degree"])
    logging.info("    avg clustering coeff.  : %.4f", summary["average_clustering"])
    logging.info("    connected components   : %d", summary["n_connected_components"])
    logging.info("    largest component      : %d nodes (%.1f%%)",
                 summary["largest_component_size"], summary["largest_component_pct"])
    logging.info("    isolated nodes         : %d%s", summary["n_isolated_nodes"],
                 (" -> " + summary["isolated_nodes"]) if summary["isolated_nodes"] else "")
    if summary["intra_sector_edge_pct"] == summary["intra_sector_edge_pct"]:
        logging.info("    intra-sector edges     : %.1f%%", summary["intra_sector_edge_pct"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_graph_construction(corr: pd.DataFrame,
                           thresholds: List[float] | None = None
                           ) -> Dict[str, object]:
    """Build the complete graph and one graph per threshold; export the summary.

    Writes
    ------
    ``outputs/tables/full_graph_summary.csv``
    ``outputs/tables/threshold_network_metrics.csv``
    """
    thresholds = list(thresholds if thresholds is not None else config.THRESHOLDS)

    utils.subsection("Complete (unfiltered) correlation graph")
    full_graph = build_full_graph(corr)
    full_summary = graph_summary(full_graph, threshold=0.0)
    _log_summary(full_summary, "complete weighted correlation graph")
    logging.info("    (every pair is an edge by construction, so degree and density")
    logging.info("     carry no information; the graph is kept as a reference object)")
    utils.save_table(pd.DataFrame([full_summary]), "full_graph_summary.csv")

    utils.subsection(f"Threshold-filtered graphs (rule: {config.EDGE_FILTER_MODE})")
    graphs: Dict[float, nx.Graph] = {}
    summaries = []
    for threshold in thresholds:
        graph = build_threshold_graph(corr, threshold)
        graphs[threshold] = graph
        summary = graph_summary(graph, threshold=threshold)
        summaries.append(summary)
        _log_summary(summary, f"tau = {threshold:.2f}")

    table = pd.DataFrame(summaries)
    utils.save_table(table, "threshold_network_metrics.csv")

    return {"full_graph": full_graph, "full_summary": full_summary,
            "graphs": graphs, "threshold_table": table}
