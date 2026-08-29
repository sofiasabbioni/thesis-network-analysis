"""
traversal_analysis.py
=====================

Steps 11 and 12 of the pipeline: **Breadth-First Search** and **Depth-First
Search** on the threshold-filtered correlation network.  These two algorithms
are the methodological core of the thesis, so each one is implemented twice:

* a **transparent, from-scratch implementation** (``custom_bfs`` /
  ``custom_dfs``) whose code can be shown line by line in Chapter 3, and
* the corresponding **NetworkX routine**, used to *validate* the custom code.

The validation step is itself a result worth reporting: it shows that the
hand-written traversals reproduce a reference implementation exactly, which is
what allows the rest of the analysis to rely on them.

Financial reading of the two traversals
---------------------------------------
BFS explores the network *layer by layer*, so the BFS distance from a source
stock is the minimum number of "strong correlation" links needed to reach
another stock.  A stock at distance 1 co-moves with the source directly; a
stock at distance 3 is connected to it only through a chain of intermediaries.
Distance is therefore a measure of **topological proximity inside the
dependence structure**, not a forecast: it says how information about one stock
propagates through the estimated correlation network, and nothing about future
prices.

DFS explores one path as deeply as possible before backtracking.  On its own
the visiting order is of limited economic interest (it depends on the order in
which neighbours happen to be stored), but the DFS *skeleton* yields two
structurally important objects: **articulation points**, i.e. stocks whose
removal disconnects the network, and **bridges**, i.e. single links whose
removal disconnects it.  Both identify positions that hold otherwise separate
correlation clusters together.

Important caveat, repeated in the exported outputs: an articulation point is a
*structural* property of an estimated network over a specific sample.  It is
not a quality judgement about the company and certainly not a buy signal.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Hashable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Custom (educational) implementations
# ---------------------------------------------------------------------------


def custom_bfs(graph: nx.Graph, source: Hashable
               ) -> Tuple[List[Hashable], Dict[Hashable, int], Dict[Hashable, Hashable | None]]:
    """Breadth-First Search implemented from first principles.

    The algorithm maintains a FIFO queue of discovered-but-unexplored nodes.
    Because every edge has unit cost in the *unweighted* threshold graph, the
    order in which BFS dequeues nodes is exactly the order of increasing
    distance from the source, which is why BFS also solves the shortest-path
    problem here.

    Neighbours are visited in alphabetical order so that the traversal is
    deterministic and reproducible across runs and machines.

    Complexity: O(|V| + |E|) time, O(|V|) space.

    Returns
    -------
    (order, distance, parent)
        ``order``    nodes in the order they were dequeued;
        ``distance`` number of edges from the source (missing = unreachable);
        ``parent``   predecessor of each node in the BFS tree.
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} is not in the graph.")

    visited = {source}
    distance: Dict[Hashable, int] = {source: 0}
    parent: Dict[Hashable, Hashable | None] = {source: None}
    order: List[Hashable] = []
    queue: deque = deque([source])

    while queue:
        node = queue.popleft()          # FIFO -> breadth first
        order.append(node)
        for neighbour in sorted(graph.neighbors(node)):
            if neighbour not in visited:
                visited.add(neighbour)          # mark on discovery, not on visit,
                distance[neighbour] = distance[node] + 1   # so no node is queued twice
                parent[neighbour] = node
                queue.append(neighbour)
    return order, distance, parent


def custom_dfs(graph: nx.Graph, source: Hashable
               ) -> Tuple[List[Hashable], Dict[Hashable, Hashable | None]]:
    """Depth-First Search implemented iteratively (pre-order).

    An explicit LIFO stack is used instead of recursion: the recursion depth of
    a natural recursive DFS is bounded by |V|, which is safe for 47 stocks but
    not for a larger universe, and an explicit stack makes the "go deep, then
    backtrack" behaviour visible in the code.

    Neighbours are pushed in reverse alphabetical order so that they are popped
    alphabetically, matching the deterministic convention used in ``custom_bfs``.

    Complexity: O(|V| + |E|) time, O(|V|) space.
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} is not in the graph.")

    visited = set()
    parent: Dict[Hashable, Hashable | None] = {source: None}
    order: List[Hashable] = []
    stack: List[Hashable] = [source]

    while stack:
        node = stack.pop()              # LIFO -> depth first
        if node in visited:
            continue                    # already reached by a deeper branch
        visited.add(node)
        order.append(node)
        for neighbour in sorted(graph.neighbors(node), reverse=True):
            if neighbour not in visited:
                parent.setdefault(neighbour, node)
                stack.append(neighbour)
    return order, parent


def custom_dfs_forest(graph: nx.Graph) -> Tuple[List[Hashable], Dict[Hashable, Hashable | None],
                                                List[List[Hashable]]]:
    """Run DFS from every not-yet-visited node to cover the whole graph.

    A single DFS only reaches the component of its source.  Restarting on every
    unvisited node produces a **DFS forest** whose trees are exactly the
    connected components of the graph - which is the classical way of computing
    connected components with a traversal, and the way it is presented in
    Chapter 3.
    """
    visited = set()
    order: List[Hashable] = []
    parent: Dict[Hashable, Hashable | None] = {}
    components: List[List[Hashable]] = []

    for root in sorted(graph.nodes()):
        if root in visited:
            continue
        local_order, local_parent = custom_dfs(graph, root)
        visited.update(local_order)
        order.extend(local_order)
        parent.update(local_parent)
        components.append(local_order)
    return order, parent, components


def parents_to_tree(parent: Dict[Hashable, Hashable | None]) -> nx.Graph:
    """Turn a ``child -> parent`` map into the corresponding tree (or forest)."""
    tree = nx.Graph()
    tree.add_nodes_from(parent.keys())
    for child, father in parent.items():
        if father is not None:
            tree.add_edge(father, child)
    return tree


# ---------------------------------------------------------------------------
# Validation against NetworkX
# ---------------------------------------------------------------------------


def validate_bfs(graph: nx.Graph, source: Hashable,
                 order: Sequence[Hashable], distance: Dict[Hashable, int]) -> Dict[str, bool]:
    """Check the custom BFS against NetworkX's reference implementation.

    Two properties are checked:

    * the set of reached nodes equals the source's connected component, and
    * every distance equals ``nx.single_source_shortest_path_length``.

    BFS distances are canonical (they do not depend on tie-breaking), so an
    exact match is expected and any mismatch is a genuine bug.
    """
    reference_distance = dict(nx.single_source_shortest_path_length(graph, source))
    reachable_reference = set(reference_distance)

    checks = {
        "same_reached_set": set(order) == reachable_reference,
        "same_distances": distance == reference_distance,
        "order_is_non_decreasing": all(distance[a] <= distance[b]
                                       for a, b in zip(order, order[1:])),
    }
    for name, ok in checks.items():
        logging.log(logging.INFO if ok else logging.ERROR,
                    "    BFS validation - %-24s : %s", name, "PASS" if ok else "FAIL")
    return checks


def validate_dfs(graph: nx.Graph, source: Hashable,
                 order: Sequence[Hashable]) -> Dict[str, bool]:
    """Check the custom DFS against NetworkX's reference implementation.

    Only the *set* of visited nodes and the length of the traversal are
    compared.  Unlike BFS distances, a DFS pre-order is **not** unique: it
    depends on the order in which each node's neighbours are examined, so two
    correct implementations can legitimately produce different sequences.  We
    additionally verify the defining structural property of a DFS pre-order -
    every visited node after the first is adjacent to some earlier node - which
    is what actually certifies the traversal.
    """
    reference = list(nx.dfs_preorder_nodes(graph, source=source))
    visited_prefix = {order[0]} if order else set()
    connected_prefix = True
    for node in order[1:]:
        if not any(neighbour in visited_prefix for neighbour in graph.neighbors(node)):
            connected_prefix = False
            break
        visited_prefix.add(node)

    checks = {
        "same_visited_set": set(order) == set(reference),
        "same_length": len(order) == len(reference),
        "no_repeated_visits": len(order) == len(set(order)),
        "valid_preorder": connected_prefix,
    }
    for name, ok in checks.items():
        logging.log(logging.INFO if ok else logging.ERROR,
                    "    DFS validation - %-24s : %s", name, "PASS" if ok else "FAIL")
    return checks


# ---------------------------------------------------------------------------
# BFS analysis
# ---------------------------------------------------------------------------


def resolve_source(graph: nx.Graph, requested: str | None = None) -> str:
    """Pick the BFS source, falling back to the highest-degree node.

    If the configured source is missing from the universe (a failed download)
    or is isolated at this threshold (so that BFS would reach nothing), the
    highest-degree stock is used instead and the substitution is logged, so the
    thesis can state exactly which node the reported traversal started from.
    """
    requested = requested or config.SOURCE_STOCK
    if requested in graph and graph.degree(requested) > 0:
        return requested

    degrees = dict(graph.degree())
    if not degrees or max(degrees.values()) == 0:
        raise utils.PipelineError("The graph has no edges: BFS has nothing to "
                                  "traverse. Lower the threshold.")
    fallback = max(sorted(degrees), key=lambda n: degrees[n])
    if requested not in graph:
        logging.warning("  source %s is not in the universe; using %s instead.",
                        requested, fallback)
    else:
        logging.warning("  source %s is isolated at this threshold; using %s instead.",
                        requested, fallback)
    return fallback


def bfs_analysis(graph: nx.Graph, source: str, threshold: float) -> Dict[str, object]:
    """Full BFS analysis from ``source`` on one threshold graph."""
    order, distance, parent = custom_bfs(graph, source)
    checks = validate_bfs(graph, source, order, distance)

    reachable = [n for n in order]
    unreachable = sorted(set(graph.nodes()) - set(reachable))
    tree = parents_to_tree(parent)

    distances_only = [d for n, d in distance.items() if n != source]
    eccentricity = max(distances_only) if distances_only else 0
    mean_distance = float(np.mean(distances_only)) if distances_only else np.nan

    rows = []
    for rank, node in enumerate(order):
        rows.append({
            "ticker": node,
            "sector": graph.nodes[node].get("sector", "Unknown"),
            "company": graph.nodes[node].get("company", node),
            "bfs_distance": int(distance[node]),
            "bfs_visit_order": rank,
            "bfs_parent": parent[node] if parent[node] is not None else "",
            "reachable": True,
            "degree": int(graph.degree(node)),
            "correlation_with_parent": (
                float(graph[node][parent[node]]["correlation"])
                if parent[node] is not None else np.nan),
        })
    for node in unreachable:
        rows.append({
            "ticker": node,
            "sector": graph.nodes[node].get("sector", "Unknown"),
            "company": graph.nodes[node].get("company", node),
            "bfs_distance": np.nan,
            "bfs_visit_order": np.nan,
            "bfs_parent": "",
            "reachable": False,
            "degree": int(graph.degree(node)),
            "correlation_with_parent": np.nan,
        })
    table = pd.DataFrame(rows)

    layer_sizes = (table.loc[table["reachable"], "bfs_distance"]
                   .value_counts().sort_index().astype(int))
    layers = pd.DataFrame({
        "bfs_distance": layer_sizes.index.astype(int),
        "n_stocks": layer_sizes.values,
        "share_of_universe": layer_sizes.values / graph.number_of_nodes(),
        "tickers": [", ".join(sorted(table.loc[(table["bfs_distance"] == d)
                                               & table["reachable"], "ticker"]))
                    for d in layer_sizes.index],
    })

    logging.info("  BFS source               : %s (%s)", source,
                 graph.nodes[source].get("sector", "Unknown"))
    logging.info("  reachable / unreachable  : %d / %d", len(reachable), len(unreachable))
    logging.info("  eccentricity of source   : %d (within its component)", eccentricity)
    logging.info("  mean distance from source: %.3f edges", mean_distance)
    for _, row in layers.iterrows():
        tickers = row["tickers"]
        logging.info("    layer %d : %2d stock(s)  %s", int(row["bfs_distance"]),
                     int(row["n_stocks"]),
                     tickers if len(tickers) <= 70 else tickers[:67] + "...")
    if unreachable:
        logging.info("  unreachable from source  : %s", ", ".join(unreachable))

    return {"source": source, "threshold": threshold, "order": order,
            "distance": distance, "parent": parent, "tree": tree,
            "table": table, "layers": layers, "reachable": reachable,
            "unreachable": unreachable, "eccentricity": eccentricity,
            "mean_distance": mean_distance, "validation": checks}


def bfs_reachability_across_thresholds(graphs: Dict[float, nx.Graph],
                                       source: str) -> pd.DataFrame:
    """How far the source stock "sees" as the threshold rises.

    A compact robustness table: it shows in one line per threshold how the
    source's neighbourhood shrinks, which complements the aggregate
    fragmentation curves of Section 14 with a node-level view.
    """
    rows = []
    for threshold in sorted(graphs):
        graph = graphs[threshold]
        if source not in graph:
            continue
        distances = dict(nx.single_source_shortest_path_length(graph, source))
        others = [d for node, d in distances.items() if node != source]
        rows.append({
            "threshold": threshold,
            "degree_of_source": int(graph.degree(source)),
            "n_reachable": len(others),
            "share_reachable": len(others) / max(1, graph.number_of_nodes() - 1),
            "eccentricity": int(max(others)) if others else 0,
            "mean_distance": float(np.mean(others)) if others else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DFS analysis
# ---------------------------------------------------------------------------


def articulation_point_table(graph: nx.Graph) -> pd.DataFrame:
    """Articulation points and the damage their removal would cause.

    For every cut vertex the table reports how many components the graph would
    break into once the stock is removed, and how large the biggest surviving
    piece would be.  That converts a binary structural property into a
    magnitude, which is what makes the finding interpretable: a stock whose
    removal detaches one isolated name is far less structurally important than
    one whose removal splits the network in half.
    """
    articulation = sorted(nx.articulation_points(graph))
    if not articulation:
        return pd.DataFrame(columns=[
            "ticker", "sector", "company", "degree", "betweenness_centrality",
            "components_before", "components_after", "new_components",
            "largest_component_after", "detached_stocks"])

    betweenness = nx.betweenness_centrality(graph, normalized=True)
    before = nx.number_connected_components(graph)

    rows = []
    for node in articulation:
        reduced = graph.copy()
        reduced.remove_node(node)
        components = sorted(nx.connected_components(reduced), key=len, reverse=True)
        detached = sorted(set().union(*components[1:])) if len(components) > 1 else []
        rows.append({
            "ticker": node,
            "sector": graph.nodes[node].get("sector", "Unknown"),
            "company": graph.nodes[node].get("company", node),
            "degree": int(graph.degree(node)),
            "betweenness_centrality": float(betweenness[node]),
            "components_before": int(before),
            "components_after": int(len(components)),
            "new_components": int(len(components) - before),
            "largest_component_after": int(len(components[0])) if components else 0,
            "detached_stocks": ", ".join(detached),
        })
    return (pd.DataFrame(rows)
            .sort_values(["new_components", "betweenness_centrality"], ascending=False)
            .reset_index(drop=True))


def bridge_table(graph: nx.Graph) -> pd.DataFrame:
    """Bridges (cut edges) with the size of the two sides they hold together."""
    columns = ["stock_a", "stock_b", "sector_a", "sector_b", "cross_sector",
               "correlation", "side_a_size", "side_b_size", "smaller_side_size",
               "smaller_side_stocks"]
    bridges = sorted(tuple(sorted(edge)) for edge in nx.bridges(graph))
    if not bridges:
        # An explicit empty frame keeps the exported CSV self-documenting: the
        # header is present even when the network happens to have no bridge.
        return pd.DataFrame(columns=columns)
    rows = []
    for a, b in bridges:
        reduced = graph.copy()
        reduced.remove_edge(a, b)
        side_a = nx.node_connected_component(reduced, a)
        side_b = nx.node_connected_component(reduced, b)
        sector_a = graph.nodes[a].get("sector", "Unknown")
        sector_b = graph.nodes[b].get("sector", "Unknown")
        rows.append({
            "stock_a": a, "stock_b": b,
            "sector_a": sector_a, "sector_b": sector_b,
            "cross_sector": sector_a != sector_b,
            "correlation": float(graph[a][b]["correlation"]),
            "side_a_size": len(side_a),
            "side_b_size": len(side_b),
            "smaller_side_size": min(len(side_a), len(side_b)),
            "smaller_side_stocks": ", ".join(sorted(side_a if len(side_a) <= len(side_b)
                                                    else side_b)),
        })
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("smaller_side_size", ascending=False)
            .reset_index(drop=True))


def structural_summary_across_thresholds(graphs: Dict[float, nx.Graph]) -> pd.DataFrame:
    """Count cut vertices and cut edges at every threshold.

    Articulation points and bridges only exist in a network that is sparse
    enough to have them: a dense low-threshold graph has many redundant paths,
    so removing any single stock leaves it connected.  Reporting the counts
    across the whole threshold grid turns "there are no articulation points at
    tau = 0.5" from an awkward non-result into an informative one - it shows at
    which level of correlation the network stops being structurally redundant.
    """
    rows = []
    for threshold in sorted(graphs):
        graph = graphs[threshold]
        articulation = sorted(nx.articulation_points(graph))
        bridges = list(nx.bridges(graph))
        rows.append({
            "threshold": threshold,
            "n_edges": graph.number_of_edges(),
            "n_components": nx.number_connected_components(graph),
            "n_articulation_points": len(articulation),
            "n_bridges": len(bridges),
            "articulation_points": ", ".join(articulation),
            "share_of_edges_that_are_bridges": (len(bridges) / graph.number_of_edges()
                                                if graph.number_of_edges() else np.nan),
        })
    return pd.DataFrame(rows)


def dfs_analysis(graph: nx.Graph, threshold: float,
                 source: str | None = None) -> Dict[str, object]:
    """Full DFS analysis of one threshold graph.

    Runs a DFS forest over the whole graph (so that every component is
    covered), validates the traversal that starts at ``source`` against
    NetworkX, and extracts articulation points and bridges.
    """
    order, parent, dfs_components = custom_dfs_forest(graph)
    forest = parents_to_tree(parent)

    checks: Dict[str, bool] = {}
    if source is not None and source in graph and graph.degree(source) > 0:
        local_order, _ = custom_dfs(graph, source)
        checks = validate_dfs(graph, source, local_order)

    # DFS-derived components must coincide with NetworkX's connected components.
    nx_components = [set(c) for c in nx.connected_components(graph)]
    dfs_sets = [set(c) for c in dfs_components]
    components_match = sorted(map(sorted, dfs_sets)) == sorted(map(sorted, nx_components))
    logging.log(logging.INFO if components_match else logging.ERROR,
                "    DFS validation - %-24s : %s", "components_match_networkx",
                "PASS" if components_match else "FAIL")
    checks["components_match_networkx"] = components_match

    component_of = {node: index for index, component in enumerate(dfs_components)
                    for node in component}
    rows = []
    for rank, node in enumerate(order):
        father = parent.get(node)
        rows.append({
            "dfs_visit_order": rank,
            "ticker": node,
            "sector": graph.nodes[node].get("sector", "Unknown"),
            "company": graph.nodes[node].get("company", node),
            "dfs_parent": father if father is not None else "",
            "is_tree_root": father is None,
            "dfs_component_id": component_of[node],
            "degree": int(graph.degree(node)),
            "correlation_with_parent": (float(graph[node][father]["correlation"])
                                        if father is not None else np.nan),
        })
    order_table = pd.DataFrame(rows)

    articulation = articulation_point_table(graph)
    bridges = bridge_table(graph)

    logging.info("  DFS visited              : %d node(s) in %d tree(s)",
                 len(order), len(dfs_components))
    logging.info("  articulation points      : %d%s", len(articulation),
                 (" -> " + ", ".join(articulation["ticker"])) if len(articulation) else "")
    logging.info("  bridges                  : %d", len(bridges))
    if len(bridges):
        top = bridges.head(5)
        logging.info("    largest split(s)       : %s",
                     "; ".join(f"{r.stock_a}-{r.stock_b} (isolates {int(r.smaller_side_size)})"
                               for r in top.itertuples()))

    return {"order": order, "parent": parent, "forest": forest,
            "order_table": order_table, "articulation": articulation,
            "bridges": bridges, "components": dfs_components,
            "validation": checks, "threshold": threshold}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_traversal_analysis(graphs: Dict[float, nx.Graph],
                           main_threshold: float | None = None,
                           source_stock: str | None = None) -> Dict[str, object]:
    """Run BFS and DFS on the main threshold graph and export every artefact.

    Writes (``X`` = threshold tag, ``S`` = source ticker)
    ----------------------------------------------------
    ``outputs/tables/bfs_distances_S_threshold_X.csv``
    ``outputs/tables/bfs_layers_S_threshold_X.csv``
    ``outputs/tables/bfs_reachability_by_threshold_S.csv``
    ``outputs/tables/dfs_order_threshold_X.csv``
    ``outputs/tables/articulation_points_threshold_X.csv``
    ``outputs/tables/bridges_threshold_X.csv``
    """
    main_threshold = config.MAIN_THRESHOLD if main_threshold is None else main_threshold
    if main_threshold not in graphs:
        raise KeyError(f"MAIN_THRESHOLD={main_threshold} is not among the thresholds "
                       f"{sorted(graphs)}.")
    graph = graphs[main_threshold]
    tag = utils.threshold_tag(main_threshold)

    utils.subsection(f"Breadth-First Search, tau = {main_threshold:.2f}")
    source = resolve_source(graph, source_stock)
    bfs = bfs_analysis(graph, source, main_threshold)
    utils.save_table(bfs["table"], f"bfs_distances_{source}_threshold_{tag}.csv")
    utils.save_table(bfs["layers"], f"bfs_layers_{source}_threshold_{tag}.csv")

    reachability = bfs_reachability_across_thresholds(graphs, source)
    utils.save_table(reachability, f"bfs_reachability_by_threshold_{source}.csv")

    utils.subsection(f"Depth-First Search, tau = {main_threshold:.2f}")
    dfs = dfs_analysis(graph, main_threshold, source=source)
    utils.save_table(dfs["order_table"], f"dfs_order_threshold_{tag}.csv")
    utils.save_table(dfs["articulation"], f"articulation_points_threshold_{tag}.csv")
    utils.save_table(dfs["bridges"], f"bridges_threshold_{tag}.csv")

    # The same DFS analysis on the neighbouring thresholds, so that the thesis
    # can show where the network becomes structurally fragile.
    others = {t: dfs_analysis(graphs[t], t)
              for t in sorted(graphs) if t != main_threshold}
    for threshold, result in others.items():
        other_tag = utils.threshold_tag(threshold)
        if len(result["articulation"]):
            utils.save_table(result["articulation"],
                             f"articulation_points_threshold_{other_tag}.csv")
        if len(result["bridges"]):
            utils.save_table(result["bridges"], f"bridges_threshold_{other_tag}.csv")

    structural = structural_summary_across_thresholds(graphs)
    utils.save_table(structural, "articulation_bridges_by_threshold.csv")
    logging.info("")
    logging.info("  Structural fragility across thresholds:")
    for _, row in structural.iterrows():
        logging.info("    tau = %.2f : %3d edge(s), %2d component(s), "
                     "%2d articulation point(s), %2d bridge(s)",
                     row["threshold"], int(row["n_edges"]), int(row["n_components"]),
                     int(row["n_articulation_points"]), int(row["n_bridges"]))

    return {"bfs": bfs, "dfs": dfs, "dfs_by_threshold": {main_threshold: dfs, **others},
            "source": source, "structural": structural,
            "reachability": reachability, "threshold": main_threshold}
