"""
large_visualisation.py
======================

Figures for the **large-network extension**.

Deliberately few.  A node-link diagram of several hundred stocks with a ticker
on every node is unreadable and tells a reader nothing that the tables do not
say better, so it is not produced.  What is produced is the small set of
pictures that answer a question the numbers alone answer awkwardly:

``large_threshold_sensitivity``      four panels: how nodes, edges, density and
                                     average degree respond to tau.
``large_giant_component_vs_tau``     at which threshold the network fragments.
``large_sector_assortativity_vs_tau``whether sector structure strengthens as
                                     weak edges are removed - the extension's
                                     central claim, in one line.
``large_same_sector_share_hist``     the distribution behind the mean: is
                                     sector alignment typical of most stocks or
                                     driven by a minority?
``large_sector_alignment_by_sector`` which sectors cohere and which do not.
``large_network_by_sector``          the network itself, coloured by sector and
                                     **unlabelled**, as a qualitative check
                                     that the blocks the statistics describe
                                     are visible.

Every figure uses the eleven-colour GICS palette from ``config``, which is
separate from the ten-colour palette of the thesis universe.
"""

from __future__ import annotations

import logging
from typing import Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils, visualisation


def _gics_colour(sector: str) -> str:
    """Colour of a GICS sector, with a neutral fallback."""
    return config.GICS_SECTOR_COLORS.get(sector, "#999999")


def plot_threshold_sensitivity(table: pd.DataFrame,
                               name: str = "large_threshold_sensitivity") -> str:
    """Four-panel view of how the network responds to the threshold."""
    panels = [
        ("edges", "Number of edges", "Edges surviving the threshold"),
        ("density", "Density", "Density"),
        ("avg_degree", "Average degree", "Average degree"),
        ("avg_clustering", "Average clustering coefficient", "Average clustering"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, ylabel, title) in zip(axes.ravel(), panels):
        ax.plot(table["threshold"], table[column], marker="o",
                color="#4C72B0", linewidth=2)
        ax.set_xlabel(r"threshold $\tau$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(table["threshold"])
    fig.suptitle(f"Large network: threshold sensitivity "
                 f"({int(table['nodes'].iloc[0])} stocks)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return visualisation._save(fig, name)


def plot_giant_component(table: pd.DataFrame,
                         name: str = "large_giant_component_vs_tau") -> str:
    """Giant-component share and isolated-node count against tau.

    The two series are the two halves of the same story - as the threshold
    rises the giant component sheds nodes and the isolated count absorbs them -
    so they share an x-axis and are drawn on twin y-axes.
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    ax.plot(table["threshold"], 100 * table["giant_component_share"],
            marker="o", color="#4C72B0", linewidth=2, label="giant component")
    ax.set_xlabel(r"threshold $\tau$")
    ax.set_ylabel("Giant component (% of nodes)", color="#4C72B0")
    ax.set_ylim(0, 105)
    ax.set_xticks(table["threshold"])
    ax.tick_params(axis="y", labelcolor="#4C72B0")

    twin = ax.twinx()
    twin.plot(table["threshold"], table["isolated_nodes"],
              marker="s", color="#C44E52", linewidth=2, linestyle="--",
              label="isolated nodes")
    twin.set_ylabel("Isolated nodes (count)", color="#C44E52")
    twin.tick_params(axis="y", labelcolor="#C44E52")
    twin.grid(False)

    handles = [plt.Line2D([], [], color="#4C72B0", marker="o", label="giant component (%)"),
               plt.Line2D([], [], color="#C44E52", marker="s", linestyle="--",
                          label="isolated nodes")]
    ax.legend(handles=handles, loc="center left")
    ax.set_title("Large network: fragmentation as the threshold rises")
    fig.tight_layout()
    return visualisation._save(fig, name)


def plot_sector_assortativity(table: pd.DataFrame,
                              name: str = "large_sector_assortativity_vs_tau") -> str:
    """Sector assortativity against tau, with the zero line for reference."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    ax.plot(table["threshold"], table["sector_assortativity"],
            marker="o", color="#55A868", linewidth=2)
    ax.axhline(0.0, color="#666666", linewidth=1, linestyle=":")
    ax.annotate("no sector structure", xy=(table["threshold"].iloc[0], 0.0),
                xytext=(0, 6), textcoords="offset points", fontsize=9, color="#666666")
    ax.set_xlabel(r"threshold $\tau$")
    ax.set_ylabel("Sector assortativity coefficient")
    ax.set_xticks(table["threshold"])
    ax.set_title("Large network: sector assortativity against the threshold")
    fig.tight_layout()
    return visualisation._save(fig, name)


def plot_same_sector_share_distribution(
        alignment: pd.DataFrame, threshold: float,
        name: str = "large_same_sector_share_hist") -> str:
    """Distribution of the per-stock same-sector neighbour share."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    values = alignment["same_sector_share"].to_numpy()
    ax.hist(values, bins=np.linspace(0, 1, 21), color="#4C72B0",
            edgecolor="white", alpha=0.9)
    mean, median = float(np.mean(values)), float(np.median(values))
    ax.axvline(mean, color="#C44E52", linewidth=2,
               label=f"mean = {mean:.3f}")
    ax.axvline(median, color="#DD8452", linewidth=2, linestyle="--",
               label=f"median = {median:.3f}")
    ax.set_xlabel("Same-sector share of a stock's neighbours")
    ax.set_ylabel("Number of stocks")
    ax.set_title(f"Large network: same-sector neighbour share "
                 f"(tau = {threshold:.2f}, {len(alignment)} non-isolated stocks)")
    ax.legend()
    fig.tight_layout()
    return visualisation._save(fig, name)


def plot_alignment_by_sector(by_sector: pd.DataFrame, threshold: float,
                             name: str = "large_sector_alignment_by_sector") -> str:
    """Mean same-sector neighbour share for each GICS sector."""
    ordered = by_sector.sort_values("mean_same_sector_share")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ordered["sector"], ordered["mean_same_sector_share"],
            color=[_gics_colour(s) for s in ordered["sector"]], edgecolor="white")
    for y, (value, count) in enumerate(zip(ordered["mean_same_sector_share"],
                                           ordered["n_stocks_with_neighbours"])):
        ax.text(value + 0.005, y, f"{value:.3f}  (n={int(count)})",
                va="center", fontsize=9)
    ax.set_xlabel("Mean same-sector share of neighbours")
    ax.set_xlim(0, min(1.0, ordered["mean_same_sector_share"].max() * 1.35 + 0.05))
    ax.set_title(f"Large network: sector alignment by GICS sector (tau = {threshold:.2f})")
    fig.tight_layout()
    return visualisation._save(fig, name)


def plot_network_by_sector(graph: nx.Graph, threshold: float,
                           name: str = "large_network_by_sector") -> str:
    """The large network, coloured by sector, with **no node labels**.

    Hundreds of tickers cannot be read on one page, so none are drawn: the
    figure is a qualitative check that the sector blocks the statistics
    describe are actually visible, not a lookup table.  Isolated nodes are
    drawn too, ringed in grey, because their number is part of the result.

    Layout, canvas shape and marker size all come from the helpers the thesis
    figures already use.  That matters here more than it does at 47 nodes: a
    plain spring layout of a graph with a dense giant component and a scatter
    of detached stocks collapses the interesting part into a dot and spends the
    rest of the page on whitespace.  ``visualisation._layout`` instead lays out
    each component separately and packs them, and ``_fit_axes_to_layout``
    frames the axes on the result.
    """
    if graph.number_of_nodes() == 0:
        raise utils.PipelineError("Cannot draw an empty graph.")

    pos = visualisation._layout(graph, kind="spring", weight="weight")
    fig, ax = visualisation._figure_for_layout(pos, plot_area=150.0,
                                               legend_width=3.4,
                                               side_bounds=(8.0, 20.0))
    scale = visualisation._fit_axes_to_layout(fig, ax, pos)
    sizes = visualisation._auto_node_sizes(graph, pos, scale,
                                           degree_range=(0.5, 1.4),
                                           bounds=(3.0, 14.0))

    degrees = dict(graph.degree())
    colours = [_gics_colour(graph.nodes[n].get("sector", "Unknown"))
               for n in graph.nodes()]
    isolated = [n for n in graph.nodes() if degrees[n] == 0]

    # Edges are drawn very faint: at several thousand edges the ink of the
    # links would otherwise bury the nodes the figure is about.
    alpha = float(np.clip(1200.0 / max(1, graph.number_of_edges()), 0.02, 0.30))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=alpha, width=0.35,
                           edge_color="#444444")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=colours,
                           linewidths=0.3, edgecolors="white")
    if isolated:
        isolated_sizes = [sizes[i] for i, n in enumerate(graph.nodes()) if n in set(isolated)]
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=isolated,
                               node_size=isolated_sizes,
                               node_color=[_gics_colour(graph.nodes[n].get("sector"))
                                           for n in isolated],
                               linewidths=0.8, edgecolors="#888888")

    sectors = sorted({graph.nodes[n].get("sector", "Unknown") for n in graph.nodes()})
    handles = [mpatches.Patch(color=_gics_colour(s), label=s) for s in sectors]
    ax.legend(handles=handles, title="GICS sector", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=True, fontsize=9)

    ax.set_title(f"Large correlation network at tau = {threshold:.2f}\n"
                 f"{graph.number_of_nodes()} stocks, {graph.number_of_edges()} edges "
                 f"({len(isolated)} isolated) - node size = degree, labels omitted",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    return visualisation._save(fig, name)


def generate_large_figures(threshold_table: pd.DataFrame,
                           alignment: pd.DataFrame,
                           by_sector: pd.DataFrame,
                           main_graph: nx.Graph,
                           main_threshold: float) -> Dict[str, str]:
    """Produce every large-network figure; never abort the run on a failure.

    A figure is a presentation of results that already exist in the exported
    tables, so a plotting error must not cost the numerical outputs of a run
    that may have taken a long time to download.
    """
    utils.subsection("Figures")
    visualisation.apply_style()
    paths: Dict[str, str] = {}

    jobs = [
        ("threshold_sensitivity", lambda: plot_threshold_sensitivity(threshold_table)),
        ("giant_component", lambda: plot_giant_component(threshold_table)),
        ("sector_assortativity", lambda: plot_sector_assortativity(threshold_table)),
        ("same_sector_share", lambda: plot_same_sector_share_distribution(
            alignment, main_threshold)),
        ("alignment_by_sector", lambda: plot_alignment_by_sector(by_sector, main_threshold)),
        ("network_by_sector", lambda: plot_network_by_sector(main_graph, main_threshold)),
    ]
    for key, job in jobs:
        try:
            paths[key] = job()
        except Exception as exc:
            logging.warning("  figure '%s' could not be produced (%s); continuing.",
                            key, exc)

    logging.info("  %d figure(s) written to %s", len(paths),
                 utils.relative(config.FIGURES_DIR))
    return paths
