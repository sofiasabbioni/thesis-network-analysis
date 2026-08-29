"""
network_export.py
=================

Exports a threshold-filtered correlation network as a **node list** and an
**edge list**, so that the network itself - not only the statistics computed
from it - can be reused directly from the repository.

The same code serves both networks, which is the point: the original 47-stock
thesis network and the supplementary large network are described by files with
identical columns, so anything that reads one reads the other.  The two are
kept apart by their file names (``network_*`` versus ``large_network_*``), never
by their format.

Nothing here re-implements an analysis.  Every column is produced by the module
that already owns it in the thesis pipeline:

``degree``, ``component``      ``src.metrics``
``is_articulation_point``      ``src.traversal_analysis``
``is_mst_leaf``                ``src.mst_analysis``
``network_role``               ``src.interpretation``
edge attributes                ``src.graph_construction``

so an exported file cannot drift away from the network the thesis analysed: it
is rebuilt from the same correlation matrix with the same functions.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import networkx as nx
import pandas as pd

import config
from src import (graph_construction, interpretation, metrics, mst_analysis,
                 traversal_analysis, utils)

NODE_COLUMNS = ["ticker", "company", "sector", "degree", "component",
                "is_isolated", "is_articulation_point", "network_role",
                "is_mst_leaf"]

EDGE_COLUMNS = ["source", "target", "correlation", "abs_correlation",
                "mantegna_distance", "same_sector"]


def build_edge_table(graph: nx.Graph) -> pd.DataFrame:
    """Edge list of ``graph``, one row per undirected edge.

    ``source`` and ``target`` are ordered alphabetically within the pair and
    the table is sorted by descending ``abs_correlation``, so that the file is
    deterministic: the same network always produces a byte-identical export.
    """
    rows = []
    for a, b, data in graph.edges(data=True):
        source, target = (a, b) if a <= b else (b, a)
        rows.append({
            "source": source,
            "target": target,
            "correlation": float(data["correlation"]),
            "abs_correlation": float(data["abs_correlation"]),
            "mantegna_distance": float(data["distance"]),
            "same_sector": bool(data["same_sector"]),
        })
    table = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    if not table.empty:
        table = table.sort_values(["abs_correlation", "source", "target"],
                                  ascending=[False, True, True]).reset_index(drop=True)
    return table


def build_node_table(graph: nx.Graph,
                     node_metrics: pd.DataFrame,
                     articulation: pd.DataFrame,
                     mst_metrics: pd.DataFrame,
                     framework: pd.DataFrame | None) -> pd.DataFrame:
    """Node list of ``graph``, one row per stock.

    **Every** node appears, including isolated ones: a stock that survived data
    cleaning but has no edge at this threshold is a result of the analysis, so
    dropping it from the export would misrepresent the network.  ``degree`` is
    read from the graph itself rather than from a joined table, so the export
    is internally consistent by construction.
    """
    articulation_set = set(articulation["ticker"]) if len(articulation) else set()
    mst_leaf = (mst_metrics.set_index("ticker")["is_leaf"].to_dict()
                if len(mst_metrics) else {})
    roles = (framework.set_index("ticker")["network_role"].to_dict()
             if framework is not None and len(framework) else {})
    component = node_metrics.set_index("ticker")["component_id"].to_dict()

    rows = []
    for node in sorted(graph.nodes()):
        degree = int(graph.degree(node))
        rows.append({
            "ticker": node,
            "company": graph.nodes[node].get("company", node),
            "sector": graph.nodes[node].get("sector", "Unknown"),
            "degree": degree,
            "component": int(component.get(node, -1)),
            "is_isolated": degree == 0,
            "is_articulation_point": node in articulation_set,
            "network_role": roles.get(node, ""),
            "is_mst_leaf": bool(mst_leaf.get(node, False)),
        })
    return pd.DataFrame(rows, columns=NODE_COLUMNS)


def compute_network_export(corr: pd.DataFrame, threshold: float,
                           with_roles: bool = True) -> Dict[str, object]:
    """Rebuild the threshold network from ``corr`` and derive both tables.

    The graph is rebuilt with :func:`graph_construction.build_threshold_graph`
    rather than read from a cached object, which is what guarantees that the
    export describes exactly the network the rest of the pipeline analysed at
    this threshold: same correlation matrix, same edge rule, same code.

    ``with_roles=False`` skips the interpretation framework (and the MST it
    depends on).  The role labels are then left blank.
    """
    graph = graph_construction.build_threshold_graph(corr, threshold)

    node_metrics = metrics.compute_node_metrics(graph)
    components = metrics.connected_component_table(graph)
    articulation = traversal_analysis.articulation_point_table(graph)

    mst_metrics = pd.DataFrame(columns=["ticker", "is_leaf"])
    framework = None
    if with_roles:
        mst = mst_analysis.build_mst(mst_analysis.build_distance_graph(corr))
        mst_metrics = mst_analysis.mst_node_metrics(mst)
        framework = interpretation.build_interpretation_framework(
            node_metrics=node_metrics, mst_metrics=mst_metrics,
            articulation=articulation, components=components, threshold=threshold)

    nodes = build_node_table(graph, node_metrics, articulation, mst_metrics, framework)
    edges = build_edge_table(graph)
    return {"graph": graph, "nodes": nodes, "edges": edges,
            "node_metrics": node_metrics, "components": components,
            "articulation": articulation, "mst_metrics": mst_metrics,
            "framework": framework}


def verify_export(nodes: pd.DataFrame, edges: pd.DataFrame,
                  graph: nx.Graph) -> Dict[str, object]:
    """Internal consistency checks; raises if the export contradicts the graph.

    Cheap, but worth running every time: an export is the artefact other people
    will actually reuse, and a silent mismatch between the node file, the edge
    file and the graph they claim to describe would be invisible downstream.
    """
    problems = []

    if len(nodes) != graph.number_of_nodes():
        problems.append(f"node rows {len(nodes)} != graph nodes {graph.number_of_nodes()}")
    if len(edges) != graph.number_of_edges():
        problems.append(f"edge rows {len(edges)} != graph edges {graph.number_of_edges()}")
    if nodes["ticker"].duplicated().any():
        problems.append("duplicate tickers in the node file")

    # Every endpoint of every edge must be declared in the node file.
    known = set(nodes["ticker"])
    unknown = (set(edges["source"]) | set(edges["target"])) - known
    if unknown:
        problems.append(f"edge endpoints missing from the node file: {sorted(unknown)}")

    # Degree in the node file must equal the degree implied by the edge file.
    implied = pd.concat([edges["source"], edges["target"]]).value_counts()
    stated = nodes.set_index("ticker")["degree"]
    mismatched = [t for t in known
                  if int(implied.get(t, 0)) != int(stated.get(t, 0))]
    if mismatched:
        problems.append(f"degree disagrees with the edge list for: {sorted(mismatched)[:10]}")

    # Isolation flag must agree with degree, and component sizes must sum to N.
    if not (nodes["is_isolated"] == (nodes["degree"] == 0)).all():
        problems.append("is_isolated disagrees with degree")
    component_total = int(nodes["component"].value_counts().sum())
    if component_total != len(nodes):
        problems.append(f"component membership covers {component_total} of {len(nodes)} nodes")

    if problems:
        raise utils.PipelineError(
            "Network export failed its consistency checks:\n  - " + "\n  - ".join(problems))

    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_isolated": int(nodes["is_isolated"].sum()),
        "n_components": int(nodes["component"].nunique()),
        "n_articulation_points": int(nodes["is_articulation_point"].sum()),
        "same_sector_edge_share": (float(edges["same_sector"].mean())
                                   if len(edges) else float("nan")),
    }


def write_export(nodes: pd.DataFrame, edges: pd.DataFrame, prefix: str,
                 threshold: float, directory: str | None = None) -> Tuple[str, str]:
    """Write ``<prefix>_nodes_threshold_X.csv`` and ``<prefix>_edges_threshold_X.csv``.

    Defaults to ``network_exports/``, which - unlike ``data/`` and ``outputs/`` -
    is tracked by git, because the supervisor asked for the network itself to be
    available from the repository.
    """
    directory = directory or config.NETWORK_EXPORT_DIR
    os.makedirs(directory, exist_ok=True)
    tag = utils.threshold_tag(threshold)

    node_path = os.path.join(directory, f"{prefix}_nodes_threshold_{tag}.csv")
    edge_path = os.path.join(directory, f"{prefix}_edges_threshold_{tag}.csv")
    nodes.to_csv(node_path, index=False, float_format="%.6f")
    edges.to_csv(edge_path, index=False, float_format="%.6f")

    logging.info("  nodes  -> %s  (%d rows)", utils.relative(node_path), len(nodes))
    logging.info("  edges  -> %s  (%d rows)", utils.relative(edge_path), len(edges))
    return node_path, edge_path
