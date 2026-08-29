"""
mst_analysis.py
===============

Step 13 of the pipeline: the **Minimum Spanning Tree** of the correlation
network.

From correlation to distance
----------------------------
A correlation is a similarity, not a distance, so it cannot be used directly as
an edge weight in a minimum-weight problem.  Mantegna (1999) showed that

        d_ij = sqrt( 2 * (1 - C_ij) )

is a proper metric: ``d = 0`` when C = +1 (identical behaviour), ``d = sqrt(2)``
when C = 0 (unrelated) and ``d = 2`` when C = -1 (opposite behaviour).  The
transformation is monotonically decreasing in C, so "minimising total distance"
is the same as "keeping the strongest correlations".

Treatment of negative correlations
----------------------------------
With the signed correlation in the formula (the default,
``config.MST_USE_ABS_CORRELATION = False``), an anti-correlated pair is placed
*far apart*: the MST will connect stocks that move **together**, which is the
convention of the econophysics literature and the one used in this thesis.  If
one instead considers any strong linear relationship - positive or negative -
to be a form of proximity, ``|C_ij|`` can be substituted by setting that flag
to ``True``.  The choice matters only if strong negative correlations exist in
the sample; the exported summary reports how many pairs are negative so that
the thesis can state whether the question is empirically relevant at all.

What the MST is, and what it is not
-----------------------------------
The MST keeps exactly ``N-1`` of the ``N(N-1)/2`` correlations - the smallest
set of links that still connects every stock - and it does so **without any
arbitrary threshold**.  It is therefore the natural complement to the
threshold-filtered graphs: it extracts the *dominant correlation backbone* of
the market.  It is a description of an estimated dependence structure over a
fixed sample.  It is not a forecast, and a stock's position in the tree is not
a recommendation.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils
from src.graph_construction import correlation_to_distance


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_distance_graph(corr: pd.DataFrame,
                         use_abs: bool | None = None) -> nx.Graph:
    """Complete weighted graph whose edge weight is the Mantegna distance.

    Both the distance and the original correlation are stored on every edge, so
    that the MST can be interpreted in correlation terms after it has been
    computed in distance terms.
    """
    use_abs = config.MST_USE_ABS_CORRELATION if use_abs is None else use_abs
    tickers = list(corr.columns)
    graph = nx.Graph(name="distance_graph")
    for ticker in tickers:
        graph.add_node(ticker,
                       ticker=ticker,
                       sector=config.TICKER_TO_SECTOR.get(ticker, "Unknown"),
                       company=config.COMPANY_NAMES.get(ticker, ticker))
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = float(corr.iloc[i, j])
            graph.add_edge(a, b,
                           correlation=c,
                           abs_correlation=abs(c),
                           distance=correlation_to_distance(c, use_abs=use_abs),
                           same_sector=(config.TICKER_TO_SECTOR.get(a) ==
                                        config.TICKER_TO_SECTOR.get(b)))
    return graph


def build_mst(distance_graph: nx.Graph) -> nx.Graph:
    """Minimum Spanning Tree of the distance graph (Kruskal's algorithm).

    ``nx.minimum_spanning_tree`` defaults to Kruskal: sort all edges by weight
    and add an edge whenever it joins two different components.  Applied to the
    Mantegna distance this greedily keeps the strongest correlations that do
    not create a cycle.
    """
    mst = nx.minimum_spanning_tree(distance_graph, weight="distance", algorithm="kruskal")
    mst.graph["name"] = "minimum_spanning_tree"
    return mst


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


def mst_node_metrics(mst: nx.Graph) -> pd.DataFrame:
    """Node-level statistics of the MST.

    ``mst_degree``    number of tree links; the classical measure of how
                      "central" a stock is in the correlation backbone.
    ``is_leaf``       degree 1: the stock hangs off the tree and is a candidate
                      *peripheral* name in the diversification reading.
    ``eccentricity``  longest tree path starting at the stock.
    ``depth_from_hub``tree distance (in links) to the highest-degree stock.
    """
    degrees = dict(mst.degree())
    betweenness = nx.betweenness_centrality(mst, normalized=True)
    closeness = nx.closeness_centrality(mst)
    eccentricity = nx.eccentricity(mst) if nx.is_connected(mst) else {
        n: np.nan for n in mst.nodes()}

    hub = max(sorted(degrees), key=lambda n: degrees[n])
    depth = nx.single_source_shortest_path_length(mst, hub)

    rows = []
    for node in sorted(mst.nodes()):
        neighbours = list(mst.neighbors(node))
        corrs = [mst[node][nb]["correlation"] for nb in neighbours]
        dists = [mst[node][nb]["distance"] for nb in neighbours]
        same_sector = [mst[node][nb]["same_sector"] for nb in neighbours]
        rows.append({
            "ticker": node,
            "sector": mst.nodes[node].get("sector", "Unknown"),
            "company": mst.nodes[node].get("company", node),
            "mst_degree": int(degrees[node]),
            "is_leaf": degrees[node] == 1,
            "mst_betweenness": float(betweenness[node]),
            "mst_closeness": float(closeness[node]),
            "mst_eccentricity": float(eccentricity.get(node, np.nan)),
            "depth_from_hub": int(depth.get(node, -1)),
            "mean_neighbour_correlation": float(np.mean(corrs)) if corrs else np.nan,
            "mean_neighbour_distance": float(np.mean(dists)) if dists else np.nan,
            "same_sector_links": int(np.sum(same_sector)),
            "same_sector_link_share": float(np.mean(same_sector)) if same_sector else np.nan,
            "neighbours": ", ".join(sorted(neighbours)),
        })
    table = pd.DataFrame(rows)
    table = table.sort_values(["mst_degree", "mst_betweenness"],
                              ascending=False).reset_index(drop=True)
    table.attrs["hub"] = hub
    return table


def mst_edge_table(mst: nx.Graph) -> pd.DataFrame:
    """Edge list of the MST, sorted from the strongest link to the weakest."""
    rows = []
    for a, b, data in mst.edges(data=True):
        rows.append({
            "stock_a": a, "stock_b": b,
            "sector_a": mst.nodes[a].get("sector", "Unknown"),
            "sector_b": mst.nodes[b].get("sector", "Unknown"),
            "same_sector": bool(data["same_sector"]),
            "correlation": float(data["correlation"]),
            "distance": float(data["distance"]),
        })
    return (pd.DataFrame(rows)
            .sort_values("correlation", ascending=False)
            .reset_index(drop=True))


def mst_summary(mst: nx.Graph, node_metrics: pd.DataFrame,
                edges: pd.DataFrame) -> Dict[str, object]:
    """Tree-level summary statistics.

    ``normalised_tree_length`` (total distance divided by the number of links)
    is a single number summarising how tightly the market is coupled: it falls
    when correlations rise, so comparing it across sub-periods is a standard
    way of measuring market-wide "coupling" - suggested in Chapter 5 as an
    extension.

    ``intra_sector_edge_share`` is the key clustering statistic: the fraction of
    the N-1 backbone links that join two stocks of the same sector.  Compared
    with the share expected if links were placed at random, it quantifies how
    strongly the backbone follows industry lines.
    """
    degrees = dict(mst.degree())
    total_length = float(sum(d["distance"] for _, _, d in mst.edges(data=True)))
    leaves = sorted(n for n, d in degrees.items() if d == 1)

    # Share of same-sector pairs among *all* possible pairs = the benchmark a
    # random tree would achieve.
    sectors = pd.Series({n: mst.nodes[n].get("sector", "Unknown") for n in mst.nodes()})
    counts = sectors.value_counts()
    n = len(sectors)
    random_same_sector = float((counts * (counts - 1) / 2).sum() / (n * (n - 1) / 2))

    return {
        "n_nodes": mst.number_of_nodes(),
        "n_edges": mst.number_of_edges(),
        "total_tree_length": total_length,
        "normalised_tree_length": total_length / max(1, mst.number_of_edges()),
        "mean_edge_correlation": float(edges["correlation"].mean()),
        "min_edge_correlation": float(edges["correlation"].min()),
        "max_edge_correlation": float(edges["correlation"].max()),
        "max_degree": int(max(degrees.values())),
        "hub_stock": node_metrics.iloc[0]["ticker"],
        "hub_sector": node_metrics.iloc[0]["sector"],
        "n_leaves": len(leaves),
        "leaf_share": len(leaves) / mst.number_of_nodes(),
        "leaves": ", ".join(leaves),
        "diameter": int(nx.diameter(mst)),
        "average_shortest_path": float(nx.average_shortest_path_length(mst)),
        "intra_sector_edge_share": float(edges["same_sector"].mean()),
        "random_benchmark_share": random_same_sector,
        "sector_clustering_ratio": (float(edges["same_sector"].mean()) / random_same_sector
                                    if random_same_sector > 0 else np.nan),
        "distance_uses_abs_correlation": bool(config.MST_USE_ABS_CORRELATION),
    }


def mst_sector_branch_table(mst: nx.Graph) -> pd.DataFrame:
    """Sector-level view of the tree: cohesion and external connections.

    For every sector the table reports how many MST links stay inside the
    sector, how many leave it, and which sectors they lead to.  A sector whose
    links are almost all internal forms a compact branch of the tree - the
    tree-based counterpart of a connected component in the threshold graphs.
    """
    rows = []
    for sector in sorted({mst.nodes[n].get("sector", "Unknown") for n in mst.nodes()}):
        members = sorted(n for n in mst.nodes()
                         if mst.nodes[n].get("sector") == sector)
        internal, external, partners = 0, 0, []
        for node in members:
            for neighbour in mst.neighbors(node):
                if mst.nodes[neighbour].get("sector") == sector:
                    internal += 1               # counted twice, halved below
                else:
                    external += 1
                    partners.append(mst.nodes[neighbour].get("sector", "Unknown"))
        internal //= 2
        rows.append({
            "sector": sector,
            "n_stocks": len(members),
            "internal_links": internal,
            "external_links": external,
            "cohesion": internal / max(1, internal + external),
            "max_possible_internal": len(members) - 1,
            "is_connected_branch": internal == len(members) - 1 and len(members) > 1,
            "connects_to": ", ".join(sorted(set(partners))),
            "tickers": ", ".join(members),
        })
    return pd.DataFrame(rows)


def mst_interpretation_text(summary: Dict[str, object],
                            node_metrics: pd.DataFrame,
                            edges: pd.DataFrame,
                            branches: pd.DataFrame) -> str:
    """Plain-language interpretation of the MST for Section 4.5 of the thesis."""
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("INTERPRETATION OF THE MINIMUM SPANNING TREE")
    add("=" * 78)
    add("")
    add("Construction")
    add("-" * 78)
    add("  Distance     : d_ij = sqrt(2 * (1 - C_ij))   [Mantegna, 1999]")
    add(f"  Correlation  : {'|C_ij| (absolute)' if summary['distance_uses_abs_correlation'] else 'C_ij (signed)'}")
    add("  Algorithm    : Kruskal's minimum spanning tree")
    add(f"  Result       : {summary['n_edges']} links retained out of "
        f"{summary['n_nodes'] * (summary['n_nodes'] - 1) // 2} possible pairs "
        f"({100 * summary['n_edges'] / (summary['n_nodes'] * (summary['n_nodes'] - 1) / 2):.1f}%)")
    add("")
    add("Tree statistics")
    add("-" * 78)
    add(f"  Total tree length            : {summary['total_tree_length']:.4f}")
    add(f"  Normalised tree length       : {summary['normalised_tree_length']:.4f}")
    add(f"  Mean correlation of a link   : {summary['mean_edge_correlation']:+.4f}")
    add(f"  Weakest link retained        : {summary['min_edge_correlation']:+.4f}")
    add(f"  Diameter                     : {summary['diameter']} links")
    add(f"  Average path length          : {summary['average_shortest_path']:.3f} links")
    add("")
    add("  A short normalised tree length means the backbone is made of strongly")
    add("  correlated links, i.e. a tightly coupled market. Comparing this number")
    add("  across sub-periods (calm versus crisis) is a natural extension of the")
    add("  thesis and is discussed in Chapter 5.")
    add("")
    add("Central stocks of the backbone")
    add("-" * 78)
    for _, row in node_metrics.head(config.TOP_K).iterrows():
        add(f"  {row['ticker']:<6} degree {int(row['mst_degree']):>2}   "
            f"betweenness {row['mst_betweenness']:.3f}   ({row['sector']})")
    add("")
    add(f"  The hub of the tree is {summary['hub_stock']} ({summary['hub_sector']}) with")
    add(f"  {summary['max_degree']} direct links. A high MST degree means the stock acts as the")
    add("  reference around which other stocks organise themselves: many companies")
    add("  have their single strongest surviving relationship with it. This is a")
    add("  statement about the estimated dependence structure, not about the")
    add("  quality of the company or its expected return.")
    add("")
    add("Peripheral stocks (leaves of the tree)")
    add("-" * 78)
    add(f"  {summary['n_leaves']} of {summary['n_nodes']} stocks are leaves "
        f"({100 * summary['leaf_share']:.1f}%).")
    leaves = node_metrics.loc[node_metrics["is_leaf"]].sort_values("mean_neighbour_correlation")
    for _, row in leaves.head(config.TOP_K).iterrows():
        add(f"  {row['ticker']:<6} attached to {row['neighbours']:<12} "
            f"C = {row['mean_neighbour_correlation']:+.3f}   ({row['sector']})")
    add("")
    add("  Leaves attached by a weak link sit at the edge of the correlation")
    add("  structure: over this sample they shared comparatively little variation")
    add("  with the rest of the universe. In a diversification reading they are the")
    add("  natural candidates for reducing portfolio concentration - subject to the")
    add("  standard caveat that correlations are unstable over time.")
    add("")
    add("Sectoral organisation of the backbone")
    add("-" * 78)
    add(f"  Links joining two stocks of the same sector : "
        f"{100 * summary['intra_sector_edge_share']:.1f}%")
    add(f"  Share expected under random attachment      : "
        f"{100 * summary['random_benchmark_share']:.1f}%")
    add(f"  Ratio (observed / random)                   : "
        f"{summary['sector_clustering_ratio']:.2f}x")
    add("")
    for _, row in branches.sort_values("cohesion", ascending=False).iterrows():
        flag = " [fully connected branch]" if row["is_connected_branch"] else ""
        add(f"  {row['sector']:<24} internal {int(row['internal_links']):>2} / "
            f"external {int(row['external_links']):>2}   "
            f"cohesion {row['cohesion']:.2f}{flag}")
    add("")
    add("  A ratio well above 1 shows that the backbone follows industry lines even")
    add("  though no sector information whatsoever entered its construction: the")
    add("  sectors are *recovered* from prices alone. This is the central empirical")
    add("  result of the MST section.")
    add("")
    add("Caveats")
    add("-" * 78)
    add("  - The MST retains only N-1 links; strong correlations that would create")
    add("    a cycle are discarded, so the tree understates the true density of the")
    add("    dependence structure. It is a backbone, not a complete description.")
    add("  - The tree is estimated on one fixed sample. A different period can")
    add("    produce a different hub and different branches.")
    add("  - Nothing in this section predicts returns.")
    add("")
    add("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_mst_analysis(corr: pd.DataFrame) -> Dict[str, object]:
    """Build the MST, export its tables and its interpretation.

    Writes
    ------
    ``outputs/tables/mst_node_metrics.csv``
    ``outputs/tables/mst_edges.csv``
    ``outputs/tables/mst_summary.csv``
    ``outputs/tables/mst_sector_branches.csv``
    ``outputs/logs/mst_interpretation.txt``
    """
    utils.subsection("Minimum Spanning Tree (Mantegna distance)")
    distance_graph = build_distance_graph(corr)
    mst = build_mst(distance_graph)

    node_metrics = mst_node_metrics(mst)
    edges = mst_edge_table(mst)
    summary = mst_summary(mst, node_metrics, edges)
    branches = mst_sector_branch_table(mst)

    utils.save_table(node_metrics, "mst_node_metrics.csv")
    utils.save_table(edges, "mst_edges.csv")
    utils.save_table(pd.DataFrame([summary]), "mst_summary.csv")
    utils.save_table(branches, "mst_sector_branches.csv")

    logging.info("  nodes / links            : %d / %d", summary["n_nodes"], summary["n_edges"])
    logging.info("  hub of the tree          : %s (%s), degree %d",
                 summary["hub_stock"], summary["hub_sector"], summary["max_degree"])
    logging.info("  leaves                   : %d (%.1f%% of the universe)",
                 summary["n_leaves"], 100 * summary["leaf_share"])
    logging.info("  normalised tree length   : %.4f", summary["normalised_tree_length"])
    logging.info("  intra-sector links       : %.1f%% (random benchmark %.1f%%, ratio %.2fx)",
                 100 * summary["intra_sector_edge_share"],
                 100 * summary["random_benchmark_share"],
                 summary["sector_clustering_ratio"])
    logging.info("  diameter                 : %d links", summary["diameter"])

    text = mst_interpretation_text(summary, node_metrics, edges, branches)
    utils.save_text(text, "mst_interpretation.txt")

    return {"distance_graph": distance_graph, "mst": mst,
            "node_metrics": node_metrics, "edges": edges,
            "summary": summary, "branches": branches}
