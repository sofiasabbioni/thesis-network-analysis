"""
metrics.py
==========

Steps 9 and 10 of the pipeline: **node-level network metrics** and **connected
component analysis** for each threshold graph.

Every measure below answers one specific question of Chapter 4.

``degree``                 With how many other stocks does this stock co-move
                           strongly?  The most direct measure of how "typical"
                           a stock is of the market.
``strength``               Same, but each link counts its |correlation|, so a
                           few very strong links can outweigh many weak ones.
``degree_centrality``      Degree normalised by (N-1); comparable across graphs.
``betweenness_centrality`` How often the stock lies on the shortest path
                           between two other stocks.  High values identify
                           *connectors* between otherwise distant regions of the
                           network, not necessarily strongly correlated stocks.
``closeness_centrality``   Inverse of the mean topological distance to the
                           other reachable stocks: how quickly the rest of the
                           market is "reached" from this node.
``eigenvector_centrality`` A recursive notion: a stock is central if it is
                           connected to other central stocks.
``clustering_coefficient`` The fraction of a stock's neighbours that are
                           themselves correlated - i.e. whether the stock sits
                           inside a tight cluster or bridges separate groups.
``component_id``           Which connected component the stock belongs to
                           (0 = largest, the "giant component").

A methodological caveat that belongs in Chapter 3: closeness and betweenness
are computed on a possibly disconnected graph.  NetworkX's closeness uses the
Wasserman-Faust correction, which evaluates the measure inside the node's own
component and rescales it by the component's relative size, so isolated nodes
receive 0 rather than an undefined value.  Betweenness is likewise summed over
pairs that are actually connected.  Both are therefore comparable *within* a
given threshold graph but should not be compared across thresholds without
remembering that the underlying component structure changed.
"""

from __future__ import annotations

import logging
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Node-level metrics
# ---------------------------------------------------------------------------


def _safe_eigenvector_centrality(graph: nx.Graph) -> Dict[str, float]:
    """Eigenvector centrality that degrades gracefully.

    The power iteration does not converge on graphs with isolated nodes or
    several components of similar size, which is exactly the situation at high
    thresholds.  We try the exact (NumPy/eigendecomposition) solver first,
    compute it component-by-component if that fails, and fall back to zeros as
    a last resort so that a single ill-conditioned graph cannot abort the run.
    """
    if graph.number_of_edges() == 0:
        return {node: 0.0 for node in graph.nodes()}
    try:
        return dict(nx.eigenvector_centrality_numpy(graph, weight="weight"))
    except Exception:
        pass
    values = {node: 0.0 for node in graph.nodes()}
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        if sub.number_of_edges() == 0:
            continue
        try:
            local = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except Exception:
            try:
                local = nx.eigenvector_centrality(sub, max_iter=1000, tol=1e-6,
                                                  weight="weight")
            except Exception:
                continue
        values.update({k: float(v) for k, v in local.items()})
    return values


def _component_labels(graph: nx.Graph) -> Dict[str, Dict[str, object]]:
    """Map every node to its component id (0 = largest) and component size."""
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    labels: Dict[str, Dict[str, object]] = {}
    for index, component in enumerate(components):
        for node in component:
            labels[node] = {"component_id": index, "component_size": len(component),
                            "in_giant_component": index == 0}
    return labels


def compute_node_metrics(graph: nx.Graph) -> pd.DataFrame:
    """Return one row per stock with all node-level network statistics."""
    nodes = sorted(graph.nodes())
    degrees = dict(graph.degree())
    strengths = dict(graph.degree(weight="weight"))
    labels = _component_labels(graph)

    degree_centrality = nx.degree_centrality(graph)
    # Unweighted betweenness measures purely topological brokerage; the
    # distance-weighted variant uses the Mantegna distance so that a path
    # through strongly correlated stocks is considered "shorter".
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    try:
        betweenness_w = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    except Exception:
        betweenness_w = {node: np.nan for node in graph.nodes()}
    closeness = nx.closeness_centrality(graph)
    eigenvector = _safe_eigenvector_centrality(graph)
    clustering = nx.clustering(graph)
    clustering_w = nx.clustering(graph, weight="weight")

    # Average correlation of a node with its neighbours: the finance-facing
    # complement of the purely topological measures above.
    mean_neighbour_corr = {}
    for node in nodes:
        corrs = [graph[node][nb]["correlation"] for nb in graph.neighbors(node)]
        mean_neighbour_corr[node] = float(np.mean(corrs)) if corrs else np.nan

    table = pd.DataFrame({
        "ticker": nodes,
        "sector": [graph.nodes[n].get("sector", "Unknown") for n in nodes],
        "company": [graph.nodes[n].get("company", n) for n in nodes],
        "degree": [degrees[n] for n in nodes],
        "strength": [strengths[n] for n in nodes],
        "degree_centrality": [degree_centrality[n] for n in nodes],
        "betweenness_centrality": [betweenness[n] for n in nodes],
        "betweenness_centrality_weighted": [betweenness_w.get(n, np.nan) for n in nodes],
        "closeness_centrality": [closeness[n] for n in nodes],
        "eigenvector_centrality": [eigenvector.get(n, 0.0) for n in nodes],
        "clustering_coefficient": [clustering[n] for n in nodes],
        "clustering_coefficient_weighted": [clustering_w[n] for n in nodes],
        "mean_neighbour_correlation": [mean_neighbour_corr[n] for n in nodes],
        "component_id": [labels.get(n, {}).get("component_id", -1) for n in nodes],
        "component_size": [labels.get(n, {}).get("component_size", 0) for n in nodes],
        "in_giant_component": [labels.get(n, {}).get("in_giant_component", False)
                               for n in nodes],
        "is_isolated": [degrees[n] == 0 for n in nodes],
    })

    table["composite_centrality"] = composite_centrality(table)
    table = table.sort_values(["degree", "strength"], ascending=False).reset_index(drop=True)
    table["centrality_rank"] = table["composite_centrality"].rank(
        ascending=False, method="min").astype(int)
    return table


def composite_centrality(node_metrics: pd.DataFrame) -> pd.Series:
    """Average of four min-max-normalised centrality measures.

    No single centrality measure is "the" right one: degree rewards local
    density, betweenness rewards brokerage, closeness rewards global proximity
    and eigenvector rewards being attached to important nodes.  Averaging their
    normalised values gives a single, transparent ranking that does not depend
    on the arbitrary choice of one measure, and it is used only to *label*
    stocks as central or peripheral in the exploratory framework of Section 15.
    """
    columns = ["degree_centrality", "betweenness_centrality",
               "closeness_centrality", "eigenvector_centrality"]
    normalised = []
    for column in columns:
        values = node_metrics[column].astype(float)
        spread = values.max() - values.min()
        normalised.append((values - values.min()) / spread if spread > 0
                          else pd.Series(0.0, index=values.index))
    return pd.concat(normalised, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Top-k extraction
# ---------------------------------------------------------------------------


def top_k_tables(node_metrics: pd.DataFrame, k: int | None = None
                 ) -> Dict[str, pd.DataFrame]:
    """Extract the "top 10" tables required by Section 9 of the specification."""
    k = k or config.TOP_K
    display = ["ticker", "sector", "degree", "strength", "degree_centrality",
               "betweenness_centrality", "closeness_centrality",
               "eigenvector_centrality", "clustering_coefficient",
               "composite_centrality", "component_id"]
    display = [c for c in display if c in node_metrics.columns]

    highest_degree = node_metrics.nlargest(k, "degree")[display].reset_index(drop=True)
    most_central = node_metrics.nlargest(k, "composite_centrality")[display].reset_index(drop=True)
    most_between = node_metrics.nlargest(k, "betweenness_centrality")[display].reset_index(drop=True)
    # "Peripheral" = lowest composite centrality; ties are broken by degree so
    # that isolated stocks always come first.
    peripheral = node_metrics.sort_values(
        ["composite_centrality", "degree"]).head(k)[display].reset_index(drop=True)
    isolated = node_metrics.loc[node_metrics["is_isolated"], display].reset_index(drop=True)

    return {"top_degree": highest_degree, "top_central": most_central,
            "top_betweenness": most_between, "most_peripheral": peripheral,
            "isolated": isolated}


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def connected_component_table(graph: nx.Graph) -> pd.DataFrame:
    """One row per stock: which component it belongs to and how big that is.

    Connected components answer the question "is the market one single block of
    co-moving stocks, or does it break into separate groups?".  At a low
    threshold almost every stock sits in one giant component; as the threshold
    rises the network fragments, and the way it fragments (by sector? into a
    core and a set of isolated defensive names?) is the substantive result.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    rows = []
    for index, component in enumerate(components):
        members = sorted(component)
        sectors = [graph.nodes[n].get("sector", "Unknown") for n in members]
        dominant = pd.Series(sectors).value_counts()
        for node in members:
            rows.append({
                "ticker": node,
                "sector": graph.nodes[node].get("sector", "Unknown"),
                "company": graph.nodes[node].get("company", node),
                "component_id": index,
                "component_size": len(component),
                "is_giant_component": index == 0,
                "component_n_sectors": int(dominant.shape[0]),
                "component_dominant_sector": dominant.index[0],
                "component_dominant_sector_share": float(dominant.iloc[0] / len(members)),
                "degree": int(graph.degree(node)),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["component_id", "degree", "ticker"],
                                  ascending=[True, False, True]).reset_index(drop=True)
    return table


def component_sector_composition(graph: nx.Graph) -> pd.DataFrame:
    """Sector composition of every connected component.

    ``share_of_component`` answers "how sector-pure is this component?" and
    ``share_of_sector`` answers "how much of this sector ended up here?".  A
    component that is 100% one sector *and* contains 100% of that sector is a
    cleanly detached sector cluster - the clearest possible evidence of
    sector-driven fragmentation.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    sector_totals = pd.Series(
        [graph.nodes[n].get("sector", "Unknown") for n in graph.nodes()]
    ).value_counts()

    rows = []
    for index, component in enumerate(components):
        sectors = pd.Series(
            [graph.nodes[n].get("sector", "Unknown") for n in component]
        ).value_counts()
        for sector, count in sectors.items():
            members = sorted(n for n in component
                             if graph.nodes[n].get("sector") == sector)
            rows.append({
                "component_id": index,
                "component_size": len(component),
                "sector": sector,
                "n_stocks": int(count),
                "share_of_component": float(count / len(component)),
                "share_of_sector": float(count / sector_totals.get(sector, count)),
                "tickers": ", ".join(members),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_metrics_for_threshold(graph: nx.Graph, threshold: float,
                              export_top_k: bool = True) -> Dict[str, object]:
    """Compute, export and log every metric for one threshold graph.

    Writes (``X`` = threshold tag, e.g. ``0_5``)
    ------------------------------------------
    ``outputs/tables/node_metrics_threshold_X.csv``
    ``outputs/tables/connected_components_threshold_X.csv``
    ``outputs/tables/component_sector_composition_threshold_X.csv``
    and, when ``export_top_k`` is set, the four "top 10" tables.
    """
    tag = utils.threshold_tag(threshold)

    node_metrics = compute_node_metrics(graph)
    utils.save_table(node_metrics, f"node_metrics_threshold_{tag}.csv")

    components = connected_component_table(graph)
    utils.save_table(components, f"connected_components_threshold_{tag}.csv")

    composition = component_sector_composition(graph)
    utils.save_table(composition, f"component_sector_composition_threshold_{tag}.csv")

    tops = top_k_tables(node_metrics)
    if export_top_k:
        utils.save_table(tops["top_degree"], f"top_degree_stocks_threshold_{tag}.csv")
        utils.save_table(tops["top_central"], f"top_central_stocks_threshold_{tag}.csv")
        utils.save_table(tops["most_peripheral"], f"top_peripheral_stocks_threshold_{tag}.csv")
        utils.save_table(tops["isolated"], f"isolated_stocks_threshold_{tag}.csv")

    if not node_metrics.empty:
        best = tops["top_degree"].head(5)
        logging.info("  highest-degree stocks    : %s",
                     ", ".join(f"{r.ticker} ({int(r.degree)})" for r in best.itertuples()))
        worst = tops["most_peripheral"].head(5)
        logging.info("  most peripheral stocks   : %s",
                     ", ".join(f"{r.ticker} ({int(r.degree)})" for r in worst.itertuples()))
        n_iso = int(node_metrics["is_isolated"].sum())
        logging.info("  isolated stocks          : %d%s", n_iso,
                     (" -> " + ", ".join(node_metrics.loc[node_metrics["is_isolated"],
                                                          "ticker"])) if n_iso else "")

    return {"node_metrics": node_metrics, "components": components,
            "composition": composition, "tops": tops}


def run_metrics(graphs: Dict[float, nx.Graph],
                main_threshold: float | None = None) -> Dict[float, Dict[str, object]]:
    """Run :func:`run_metrics_for_threshold` for every threshold graph."""
    main_threshold = config.MAIN_THRESHOLD if main_threshold is None else main_threshold
    results: Dict[float, Dict[str, object]] = {}
    for threshold, graph in graphs.items():
        utils.subsection(f"Node metrics and components, tau = {threshold:.2f}"
                         + ("   [MAIN THRESHOLD]" if threshold == main_threshold else ""))
        results[threshold] = run_metrics_for_threshold(graph, threshold)
    return results
