"""
visualisation.py
================

Every figure exported to ``outputs/figures``.

House style (Section 17 of the specification)
---------------------------------------------
* one consistent figure size per figure family, 300 DPI, PNG;
* a fixed sector -> colour mapping (``config.SECTOR_COLORS``) so that a sector
  keeps the same colour in every figure of the thesis;
* a colour-blind-safe qualitative palette and a diverging palette centred on
  zero for correlations;
* titles state the sample and the threshold, so a figure remains readable when
  it is extracted from the document;
* labels are drawn for every node when the graph is small enough, and only for
  the highest-degree nodes when it is not.

All functions take the objects produced by the analysis modules and return the
path of the file they wrote, so ``main.py`` can log a complete manifest.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")          # non-interactive backend: works on a server too

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import utils


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Set the global matplotlib/seaborn style used by every figure."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 110,                 # on-screen; savefig uses FIGURE_DPI
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": False,
    })


def _save(fig: plt.Figure, name: str) -> str:
    """Write a figure to ``outputs/figures`` and close it."""
    path = utils.figure_path(name)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info("  figure -> %s", utils.relative(path))
    return path


def _sector_colour(sector: str) -> str:
    """Colour of a sector, with a neutral fallback for unknown sectors."""
    return config.SECTOR_COLORS.get(sector, "#999999")


def _sector_legend(ax: plt.Axes, sectors: Iterable[str], title: str = "Sector",
                   loc: str = "upper left") -> None:
    """Attach a sector colour legend outside the plotting area."""
    handles = [mpatches.Patch(color=_sector_colour(s), label=s)
               for s in sorted(set(sectors))]
    ax.legend(handles=handles, title=title, loc=loc,
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=True)


def _labels_for(graph: nx.Graph, max_labels: int | None = None) -> Dict[str, str]:
    """Label every node, or only the highest-degree ones on a crowded graph."""
    max_labels = max_labels or config.MAX_LABELLED_NODES
    if graph.number_of_nodes() <= max_labels:
        return {n: n for n in graph.nodes()}
    ranked = sorted(graph.nodes(), key=lambda n: (-graph.degree(n), n))
    return {n: n for n in ranked[:max_labels]}


def _normalise_positions(pos: Dict[str, np.ndarray], radius: float,
                         centre: tuple = (0.0, 0.0)) -> Dict[str, np.ndarray]:
    """Centre a layout on ``centre`` and rescale it to fit inside ``radius``."""
    if not pos:
        return {}
    points = np.array([pos[n] for n in pos], dtype=float)
    points = points - points.mean(axis=0)
    extent = float(np.abs(points).max())
    if extent > 0:
        points = points / extent * radius
    return {node: points[index] + np.array(centre, dtype=float)
            for index, node in enumerate(pos)}


def _relax_overlaps(pos: Dict[str, np.ndarray], min_distance: float,
                    iterations: int = 80) -> Dict[str, np.ndarray]:
    """Push apart nodes that a force-directed layout piled on top of each other.

    A spring layout of a *dense* subgraph (which is what a low-threshold
    correlation network is) collapses its core into a blob, because every node
    attracts every other one.  A few iterations of pure repulsion between pairs
    closer than ``min_distance`` separate them again while preserving the
    overall shape the spring layout found.
    """
    nodes = list(pos)
    if len(nodes) < 2:
        return pos
    rng = np.random.default_rng(config.RANDOM_SEED)
    points = np.array([pos[n] for n in nodes], dtype=float)
    # Nudge exactly coincident nodes so that the direction of repulsion exists.
    points = points + rng.normal(0.0, min_distance * 1e-3, points.shape)

    for _ in range(iterations):
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((deltas ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        overlapping = distances < min_distance
        if not overlapping.any():
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            directions = deltas / distances[:, :, None]
        directions = np.nan_to_num(directions)
        push = np.where(overlapping, min_distance - distances, 0.0)
        points = points + 0.5 * (directions * push[:, :, None]).sum(axis=1)
    return {node: points[index] for index, node in enumerate(nodes)}


def _layout(graph: nx.Graph, kind: str = "spring", weight: str | None = "weight"):
    """Deterministic node layout that stays readable on a *disconnected* graph.

    A plain force-directed layout is unusable here: applied to a graph with
    several components it pushes each component apart and then shrinks it to a
    dot, which is exactly the situation at high thresholds.  Instead each
    connected component is laid out **on its own** and the components are then
    packed: the giant component keeps the centre of the canvas and the smaller
    ones (down to isolated stocks) are arranged on concentric rings around it,
    each scaled by the square root of its size.

    The seed is fixed in ``config.RANDOM_SEED`` so that re-running the project
    reproduces byte-identical figures - a small but real reproducibility point
    for Chapter 3.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if kind == "kamada_kawai" and graph.number_of_edges() > 0:
        try:
            return nx.kamada_kawai_layout(graph, weight=weight)
        except Exception:
            pass

    def spring(subgraph: nx.Graph) -> Dict[str, np.ndarray]:
        n = subgraph.number_of_nodes()
        if n == 1:
            return {next(iter(subgraph.nodes())): np.zeros(2)}
        k = 1.6 / np.sqrt(n)
        pos = nx.spring_layout(subgraph, seed=config.RANDOM_SEED, k=k,
                               iterations=300, weight=weight)
        # Target separation if the n nodes were spread evenly over the unit
        # disc the layout is normalised to.
        return _relax_overlaps(_normalise_positions(pos, radius=1.0),
                               min_distance=1.55 / np.sqrt(n))

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) == 1:
        return _normalise_positions(spring(graph), radius=1.0)

    positions: Dict[str, np.ndarray] = {}
    giant = components[0]
    giant_size = len(giant)

    # Radius allotted to a component, proportional to the square root of its
    # size so that node *density* is comparable across components.
    def radius_of(size: int) -> float:
        if size <= 1:
            return 0.05
        return float(max(0.14, np.sqrt(size / giant_size)))

    if giant_size > 1:
        positions.update(_normalise_positions(spring(graph.subgraph(giant)), radius=1.0))
        satellites = components[1:]
        ring_start = 1.0
    else:
        # No component has an edge: every stock is isolated. Lay all of them
        # out on rings from the centre outwards.
        satellites = components
        ring_start = 0.0

    # Pack the satellites on concentric rings, largest first.
    index, ring, ring_radius = 0, 0, ring_start
    while index < len(satellites):
        biggest = radius_of(len(satellites[index]))
        ring_radius += biggest + (0.30 if ring == 0 else 0.24)
        # How many components fit on this ring without overlapping.
        capacity = max(1, int(np.floor(np.pi / max(1e-6,
                       np.arcsin(min(1.0, (biggest + 0.07) / ring_radius))))))
        on_ring = satellites[index:index + capacity]
        for slot, component in enumerate(on_ring):
            angle = 2 * np.pi * slot / len(on_ring) + (0.35 * ring)
            centre = (ring_radius * np.cos(angle), ring_radius * np.sin(angle))
            local_radius = radius_of(len(component))
            positions.update(_normalise_positions(
                spring(graph.subgraph(component)), radius=local_radius, centre=centre))
        ring_radius += biggest
        index += len(on_ring)
        ring += 1
    return positions


def _figure_for_layout(pos: Dict[str, np.ndarray], plot_area: float = 99.0,
                       legend_width: float = 3.0,
                       side_bounds: tuple = (6.0, 18.0)) -> tuple:
    """Create a figure whose shape matches the shape of the node layout.

    A packed layout of several components is typically much wider than it is
    tall.  Drawing it on a fixed 14x11 canvas with an equal aspect ratio leaves
    a third of the page blank, so the canvas is instead derived from the
    bounding box of the layout, at roughly constant plotting area.
    """
    if not pos:
        return plt.subplots(figsize=config.FIGURE_SIZE_NETWORK)
    points = np.array(list(pos.values()), dtype=float)
    width, height = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    aspect = float(np.clip(width / height, 0.35, 3.2))

    plot_height = float(np.clip(np.sqrt(plot_area / aspect), *side_bounds))
    plot_width = float(np.clip(aspect * plot_height, *side_bounds))
    return plt.subplots(figsize=(plot_width + legend_width, plot_height))


def _fit_axes_to_layout(fig: plt.Figure, ax: plt.Axes, pos: Dict[str, np.ndarray],
                        margin: float = 0.09) -> float:
    """Frame the axes on the layout and return the data-to-points scale.

    Node markers are sized in *points*, while the layout lives in arbitrary data
    units, so a marker that looks right on a sparse tree swallows a dense
    component.  Fixing the aspect ratio and the axis limits here makes the
    conversion between the two well defined, which is what
    :func:`_auto_node_sizes` needs to choose a marker size that always fits the
    space actually available to each node.
    """
    if not pos:
        return 1.0
    points = np.array(list(pos.values()), dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    extent = np.maximum(high - low, 1e-6) * (1.0 + 2 * margin)

    # The axes never fill the whole figure (title, colour bar, external legend),
    # so the usable fraction is approximated conservatively.
    width_inches, height_inches = fig.get_size_inches()
    usable_width, usable_height = width_inches * 0.74, height_inches * 0.84

    # Pad the *shorter* side of the layout until it matches the aspect ratio of
    # the canvas.  Without this the equal-aspect axes letterbox a wide layout
    # and half the figure is blank.
    canvas_aspect = usable_width / usable_height
    if extent[0] / extent[1] < canvas_aspect:
        extent[0] = extent[1] * canvas_aspect
    else:
        extent[1] = extent[0] / canvas_aspect

    ax.set_aspect("equal")
    ax.set_xlim(centre[0] - extent[0] / 2, centre[0] + extent[0] / 2)
    ax.set_ylim(centre[1] - extent[1] / 2, centre[1] + extent[1] / 2)
    return usable_width * 72.0 / extent[0]        # points per data unit


def _auto_node_sizes(graph: nx.Graph, pos: Dict[str, np.ndarray], scale: float,
                     degree_range: tuple = (0.55, 1.35),
                     bounds: tuple = (12.0, 38.0)) -> List[float]:
    """Marker areas that adapt to how much room the layout gives each node.

    The base diameter is a fraction of the typical nearest-neighbour distance in
    the layout, so a dense component gets small markers and a sparse one gets
    large markers, and neither overlaps.  Degree then modulates the diameter
    within ``degree_range`` so that the "size = connectedness" encoding survives.
    """
    nodes = list(graph.nodes())
    if not nodes:
        return []
    points = np.array([pos[n] for n in nodes], dtype=float)
    if len(nodes) > 1:
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((deltas ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        typical = float(np.percentile(nearest[np.isfinite(nearest)], 25))
    else:
        typical = 1.0

    base_diameter = float(np.clip(0.62 * typical * scale, *bounds))
    degrees = np.array([graph.degree(n) for n in nodes], dtype=float)
    spread = degrees.max() - degrees.min()
    relative = (degrees - degrees.min()) / spread if spread > 0 else np.zeros_like(degrees)
    factors = degree_range[0] + (degree_range[1] - degree_range[0]) * relative
    return [(base_diameter * f) ** 2 for f in factors]


def _auto_font_size(n_nodes: int) -> float:
    """Label size that shrinks as the drawing gets busier."""
    return float(np.clip(120.0 / max(8, n_nodes), 4.5, config.FONT_SIZE_LABEL))


# ---------------------------------------------------------------------------
# 1. Correlation figures
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(corr: pd.DataFrame,
                             name: str = "correlation_heatmap") -> str:
    """Heat map of the correlation matrix, with stocks grouped by sector.

    Sorting the axes by sector (rather than alphabetically) is what makes the
    block structure visible: sectoral clustering appears as bright squares on
    the diagonal *before* any graph algorithm is applied.  Thin separating
    lines mark the sector boundaries and the sector names are printed along the
    top axis.
    """
    order = [t for s in config.SECTORS for t in config.STOCKS[s] if t in corr.columns]
    order += [t for t in corr.columns if t not in order]
    ordered = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_HEATMAP)
    sns.heatmap(ordered, cmap="RdBu_r", center=0.0, vmin=-1.0, vmax=1.0,
                square=True, linewidths=0.0, ax=ax,
                cbar_kws={"label": "Pearson correlation of daily log returns",
                          "shrink": 0.75})
    ax.set_xticks(np.arange(len(order)) + 0.5)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, rotation=0, fontsize=7)

    # Sector separators and sector names on the secondary axis.
    boundaries, ticks, names = [], [], []
    position = 0
    for sector in config.SECTORS:
        members = [t for t in config.STOCKS[sector] if t in corr.columns]
        if not members:
            continue
        boundaries.append(position)
        ticks.append(position + len(members) / 2)
        names.append(sector)
        position += len(members)
    boundaries.append(position)
    for edge in boundaries[1:-1]:
        ax.axhline(edge, color="black", linewidth=1.1)
        ax.axvline(edge, color="black", linewidth=1.1)

    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels(names, rotation=45, ha="left", fontsize=8, fontweight="bold")
    top.tick_params(length=0)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Correlation matrix of daily log returns, stocks grouped by sector\n"
                 f"{len(order)} S&P 500 stocks, {config.START_DATE} to {config.END_DATE}",
                 pad=52)
    return _save(fig, name)


def plot_sector_correlation_heatmap(sector_matrix: pd.DataFrame,
                                    name: str = "sector_correlation_heatmap") -> str:
    """Average correlation between (and within) sectors."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(sector_matrix.astype(float), cmap="RdBu_r", center=0.0,
                vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", annot_kws={"size": 8},
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Average Pearson correlation", "shrink": 0.8})
    ax.set_title("Average correlation within and between sectors\n"
                 "(diagonal: average over distinct pairs inside the sector)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, name)


def plot_correlation_distribution(pairs: pd.DataFrame,
                                  thresholds: Sequence[float] | None = None,
                                  name: str = "correlation_distribution") -> str:
    """Distribution of the pairwise correlations, with the thresholds marked.

    This figure justifies the choice of ``config.THRESHOLDS``: it shows what
    fraction of the pairs each cut-off retains, which is the information a
    reader needs to judge whether a threshold is "high" or "low" *for this
    sample* rather than in the abstract.
    """
    thresholds = list(thresholds if thresholds is not None else config.THRESHOLDS)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.histplot(pairs["correlation"], bins=40, kde=True, ax=ax_left,
                 color="#4C72B0", edgecolor="white")
    for threshold in thresholds:
        ax_left.axvline(threshold, color="#C44E52", linestyle="--", linewidth=1.2)
        ax_left.text(threshold, ax_left.get_ylim()[1] * 0.95, f" {threshold:g}",
                     color="#C44E52", fontsize=8, rotation=90, va="top")
    ax_left.axvline(float(pairs["correlation"].mean()), color="black", linewidth=1.5,
                    label=f"mean = {pairs['correlation'].mean():.3f}")
    ax_left.set_xlabel("Pearson correlation")
    ax_left.set_ylabel("Number of stock pairs")
    ax_left.set_title("Distribution of pairwise correlations")
    ax_left.legend()

    # Same-sector versus different-sector distributions.
    sns.kdeplot(data=pairs, x="correlation", hue="same_sector", ax=ax_right,
                fill=True, common_norm=False, alpha=0.35,
                palette={True: "#55A868", False: "#8C8C8C"})
    ax_right.set_xlabel("Pearson correlation")
    ax_right.set_ylabel("Density")
    ax_right.set_title("Same-sector versus cross-sector pairs")
    legend = ax_right.get_legend()
    if legend is not None:
        legend.set_title("Same sector")
    fig.suptitle(f"Pairwise correlation structure, {config.START_DATE} to {config.END_DATE}",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 2. Network figures
# ---------------------------------------------------------------------------


def plot_threshold_network(graph: nx.Graph, threshold: float,
                           name: str | None = None) -> str:
    """Node-link diagram of a threshold graph, nodes coloured by sector.

    Node size is proportional to degree and edge width to |correlation|, so the
    picture encodes three quantities at once (sector, connectedness, strength of
    the relationships) without becoming decorative.
    """
    tag = utils.threshold_tag(threshold)
    name = name or f"threshold_network_threshold_{tag}"

    if graph.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_NETWORK)
        ax.text(0.5, 0.5, f"No edge survives the threshold tau = {threshold:g}:\n"
                          f"all {graph.number_of_nodes()} stocks are isolated.",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        ax.set_title(f"Correlation network, |C| >= {threshold:g}")
        return _save(fig, name)

    pos = _layout(graph, "spring")
    fig, ax = _figure_for_layout(pos)
    scale = _fit_axes_to_layout(fig, ax, pos)
    degrees = dict(graph.degree())
    sizes = _auto_node_sizes(graph, pos, scale)
    colours = [_sector_colour(graph.nodes[n].get("sector", "Unknown")) for n in graph.nodes()]
    widths = [0.3 + 1.6 * (abs(d["correlation"]) - threshold) / max(1e-9, 1 - threshold)
              for _, _, d in graph.edges(data=True)]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.22,
                           edge_color="#555555")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=colours,
                           edgecolors="white", linewidths=0.6)
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax,
                            font_size=_auto_font_size(graph.number_of_nodes()))

    isolated = [n for n in graph.nodes() if degrees[n] == 0]
    subtitle = (f"{graph.number_of_nodes()} stocks, {graph.number_of_edges()} edges, "
                f"density {nx.density(graph):.3f}, "
                f"{nx.number_connected_components(graph)} component(s)"
                + (f", {len(isolated)} isolated" if isolated else ""))
    ax.set_title(f"Stock correlation network, threshold |C| >= {threshold:g}\n{subtitle}")
    ax.set_axis_off()
    _sector_legend(ax, [graph.nodes[n].get("sector", "Unknown") for n in graph.nodes()])
    return _save(fig, name)


def plot_connected_components(graph: nx.Graph, threshold: float,
                              name: str | None = None) -> str:
    """Node-link diagram with nodes coloured by **connected component**.

    Compare this figure with :func:`plot_threshold_network`: the same layout
    coloured by sector and by component makes it immediately visible whether
    the components the algorithm found do or do not coincide with industry
    groups.
    """
    tag = utils.threshold_tag(threshold)
    name = name or f"connected_components_threshold_{tag}"

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if graph.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_NETWORK)
        ax.text(0.5, 0.5, f"At tau = {threshold:g} every stock forms its own component "
                          f"({graph.number_of_nodes()} singletons).",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        ax.set_title(f"Connected components, |C| >= {threshold:g}")
        return _save(fig, name)

    palette = sns.color_palette("tab20", max(3, min(20, len(components))))
    component_of, colour_of = {}, {}
    for index, component in enumerate(components):
        colour = palette[index % len(palette)] if len(component) > 1 else (0.72, 0.72, 0.72)
        for node in component:
            component_of[node] = index
            colour_of[node] = colour

    pos = _layout(graph, "spring")
    fig, ax = _figure_for_layout(pos, legend_width=4.0)
    scale = _fit_axes_to_layout(fig, ax, pos)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.6, alpha=0.22, edge_color="#555555")
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                           node_size=_auto_node_sizes(graph, pos, scale),
                           node_color=[colour_of[n] for n in graph.nodes()],
                           edgecolors="white", linewidths=0.6)
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax,
                            font_size=_auto_font_size(graph.number_of_nodes()))

    # Multi-stock components are listed individually; the singletons (isolated
    # stocks) are collapsed into one entry, otherwise the legend of a
    # high-threshold network is longer than the figure.
    handles = []
    multi = [(i, c) for i, c in enumerate(components) if len(c) > 1]
    singletons = [(i, c) for i, c in enumerate(components) if len(c) == 1]
    for index, component in multi[:12]:
        sectors = pd.Series([graph.nodes[n].get("sector", "Unknown") for n in component])
        dominant = sectors.value_counts()
        handles.append(mpatches.Patch(
            color=palette[index % len(palette)],
            label=f"C{index} - {len(component)} stocks - "
                  f"{dominant.index[0]} {100 * dominant.iloc[0] / len(component):.0f}%"))
    if len(multi) > 12:
        handles.append(mpatches.Patch(color="white",
                                      label=f"... and {len(multi) - 12} more components"))
    if singletons:
        isolated = sorted(next(iter(c)) for _, c in singletons)
        shown = ", ".join(isolated[:8]) + ("..." if len(isolated) > 8 else "")
        handles.append(mpatches.Patch(
            color=(0.72, 0.72, 0.72),
            label=f"{len(singletons)} isolated stock(s): {shown}"))
    ax.legend(handles=handles, title="Connected component", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)

    giant = len(components[0])
    ax.set_title(f"Connected components of the correlation network, |C| >= {threshold:g}\n"
                 f"{len(components)} component(s); largest holds {giant} of "
                 f"{graph.number_of_nodes()} stocks "
                 f"({100 * giant / graph.number_of_nodes():.1f}%)")
    ax.set_axis_off()
    return _save(fig, name)


def plot_degree_distribution(node_metrics: pd.DataFrame, threshold: float,
                             name: str | None = None) -> str:
    """Degree histogram plus the degree ranking of the individual stocks."""
    tag = utils.threshold_tag(threshold)
    name = name or f"degree_distribution_threshold_{tag}"

    fig, (ax_hist, ax_rank) = plt.subplots(1, 2, figsize=(14, 5.8),
                                           gridspec_kw={"width_ratios": [1, 1.4]})
    degrees = node_metrics["degree"].astype(int)
    bins = range(int(degrees.min()), int(degrees.max()) + 2)
    ax_hist.hist(degrees, bins=bins, color="#4C72B0", edgecolor="white", align="left")
    ax_hist.axvline(float(degrees.mean()), color="#C44E52", linestyle="--",
                    label=f"mean = {degrees.mean():.2f}")
    ax_hist.set_xlabel("Degree (number of strongly correlated partners)")
    ax_hist.set_ylabel("Number of stocks")
    ax_hist.set_title("Degree distribution")
    ax_hist.legend()

    ranked = node_metrics.sort_values("degree", ascending=True)
    ax_rank.barh(ranked["ticker"], ranked["degree"],
                 color=[_sector_colour(s) for s in ranked["sector"]])
    ax_rank.set_xlabel("Degree")
    ax_rank.set_ylabel("")
    ax_rank.tick_params(axis="y", labelsize=6.5)
    ax_rank.set_title("Degree by stock (colour = sector)")
    _sector_legend(ax_rank, ranked["sector"])

    fig.suptitle(f"Connectivity of individual stocks, threshold |C| >= {threshold:g}",
                 fontsize=13, fontweight="bold", y=1.01)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 3. Traversal figures
# ---------------------------------------------------------------------------


def _layered_positions(tree: nx.Graph, distance: Dict[str, int],
                       parent: Dict[str, str | None]) -> Dict[str, tuple]:
    """Place BFS nodes on horizontal layers, ordered to reduce edge crossings.

    Within a layer the nodes are sorted by the horizontal position of their
    parent (a one-pass barycentre heuristic).  Children of the same parent
    therefore stay together and the tree edges hardly cross, which is what makes
    a wide BFS layer readable.
    """
    layers: Dict[int, List[str]] = {}
    for node, d in distance.items():
        layers.setdefault(int(d), []).append(node)

    positions: Dict[str, tuple] = {}
    for depth in sorted(layers):
        if depth == 0:
            ordered = sorted(layers[depth])
        else:
            ordered = sorted(
                layers[depth],
                key=lambda n: (positions.get(parent.get(n), (0.5, 0))[0], n))
        count = len(ordered)
        for index, node in enumerate(ordered):
            x = 0.5 if count == 1 else index / (count - 1)
            positions[node] = (x, -float(depth))
    return positions


def plot_bfs_tree(graph: nx.Graph, bfs: Dict[str, object], threshold: float,
                  name: str | None = None) -> str:
    """BFS tree drawn as explicit layers of increasing distance from the source.

    A layered layout is used rather than a force-directed one because the whole
    point of BFS is the layer structure: the vertical axis *is* the
    shortest-path distance from the source.  Labels are rotated in crowded
    layers so that every ticker stays readable at print size.
    """
    source = bfs["source"]
    tag = utils.threshold_tag(threshold)
    name = name or f"bfs_tree_{source}_threshold_{tag}"

    tree: nx.Graph = bfs["tree"]
    distance: Dict[str, int] = bfs["distance"]
    parent: Dict[str, str | None] = bfs["parent"]

    if tree.number_of_nodes() <= 1:
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.text(0.5, 0.5, f"{source} has no neighbour at tau = {threshold:g}.",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return _save(fig, name)

    pos = _layered_positions(tree, distance, parent)
    max_layer = max(distance.values())
    widest = max(sum(1 for d in distance.values() if d == layer)
                 for layer in range(max_layer + 1))

    # Grow the canvas with the widest layer so that nodes never overlap.
    width = float(np.clip(1.2 + 0.34 * widest, 12.0, 26.0))
    height = float(np.clip(4.5 + 1.9 * (max_layer + 1), 7.0, 16.0))
    fig, ax = plt.subplots(figsize=(width, height))

    cmap = matplotlib.colormaps["viridis"]
    colours = [cmap(distance[n] / max(1, max_layer)) for n in tree.nodes()]
    node_size = 520 if widest <= 12 else (300 if widest <= 24 else 190)

    nx.draw_networkx_edges(tree, pos, ax=ax, width=1.0, alpha=0.55,
                           edge_color="#444444")
    nx.draw_networkx_nodes(
        tree, pos, ax=ax,
        node_size=[node_size * (1.8 if n == source else 1.0) for n in tree.nodes()],
        node_color=colours,
        edgecolors=["#C44E52" if n == source else "white" for n in tree.nodes()],
        linewidths=[2.6 if n == source else 0.8 for n in tree.nodes()])

    # Labels: horizontal above the node when the layer is sparse, rotated below
    # it when the layer is crowded.
    for node, (x, y) in pos.items():
        layer_size = sum(1 for d in distance.values() if d == distance[node])
        crowded = layer_size > 10
        ax.text(x, y - (0.14 if crowded else 0.16), node,
                rotation=90 if crowded else 0,
                ha="center", va="top",
                fontsize=7.5 if crowded else 9,
                fontweight="bold" if node == source else "normal")

    for layer in range(max_layer + 1):
        members = [n for n in tree.nodes() if distance[n] == layer]
        ax.text(-0.09, -float(layer), f"d = {layer}\n({len(members)})",
                fontsize=10, fontweight="bold", ha="right", va="center")

    unreachable: List[str] = list(bfs["unreachable"])
    if not unreachable:
        note = f"all {graph.number_of_nodes()} stocks reachable"
    elif len(unreachable) <= 12:
        note = f"unreachable: {', '.join(unreachable)}"
    else:
        note = (f"unreachable: {', '.join(unreachable[:12])} "
                f"(+{len(unreachable) - 12} more)")
    ax.set_title(f"Breadth-First Search tree from {source}, threshold |C| >= {threshold:g}\n"
                 f"eccentricity {bfs['eccentricity']}, mean distance "
                 f"{bfs['mean_distance']:.2f} edges - {note}")
    ax.set_axis_off()
    ax.set_xlim(-0.16, 1.06)
    ax.set_ylim(-max_layer - 0.75, 0.45)

    normaliser = matplotlib.colors.Normalize(vmin=0, vmax=max_layer)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normaliser, cmap=cmap),
                       ax=ax, shrink=0.55, pad=0.015, aspect=28)
    bar.set_label("BFS distance from the source (number of edges)")
    bar.set_ticks(range(max_layer + 1))
    return _save(fig, name)


def plot_bfs_distance_map(graph: nx.Graph, bfs: Dict[str, object], threshold: float,
                          name: str | None = None) -> str:
    """The whole network laid out normally, but coloured by BFS distance.

    Complements the layered tree: it shows *where* in the network each BFS layer
    sits, and which stocks the search never reaches.
    """
    source = bfs["source"]
    tag = utils.threshold_tag(threshold)
    name = name or f"bfs_distance_map_{source}_threshold_{tag}"

    distance: Dict[str, int] = bfs["distance"]
    if graph.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_NETWORK)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, name)

    pos = _layout(graph, "spring")
    fig, ax = _figure_for_layout(pos)
    scale = _fit_axes_to_layout(fig, ax, pos)
    max_layer = max(distance.values()) if distance else 0
    cmap = matplotlib.colormaps["viridis"]
    colours = [cmap(distance[n] / max(1, max_layer)) if n in distance else "#DDDDDD"
               for n in graph.nodes()]
    sizes = _auto_node_sizes(graph, pos, scale)
    sizes = [size * (2.2 if node == source else 1.0)
             for node, size in zip(graph.nodes(), sizes)]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.5, alpha=0.16, edge_color="#555555")
    tree_edges = [(a, b) for a, b in bfs["tree"].edges()]
    nx.draw_networkx_edges(graph, pos, edgelist=tree_edges, ax=ax, width=1.4,
                           alpha=0.85, edge_color="#C44E52")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors=["#C44E52" if n == source else "white"
                                       for n in graph.nodes()],
                           linewidths=[2.2 if n == source else 0.6 for n in graph.nodes()])
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax,
                            font_size=_auto_font_size(graph.number_of_nodes()))

    ax.set_title(f"Topological proximity to {source}, threshold |C| >= {threshold:g}\n"
                 f"colour = BFS distance; red edges = BFS tree; grey nodes = unreachable")
    ax.set_axis_off()
    normaliser = matplotlib.colors.Normalize(vmin=0, vmax=max_layer)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normaliser, cmap=cmap),
                       ax=ax, shrink=0.6, pad=0.02)
    bar.set_label("BFS distance from the source")
    return _save(fig, name)


def plot_dfs_tree(graph: nx.Graph, dfs: Dict[str, object], threshold: float,
                  name: str | None = None) -> str:
    """The DFS forest drawn on top of the network, with cut vertices and bridges.

    Thin grey edges are the edges of the graph that the DFS did not use ("back
    edges"); thick dark edges are the DFS tree edges.  Red-outlined nodes are
    articulation points and red dashed edges are bridges - the two structural
    objects the DFS section of the thesis is about.
    """
    tag = utils.threshold_tag(threshold)
    name = name or f"dfs_tree_threshold_{tag}"

    if graph.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_NETWORK)
        ax.text(0.5, 0.5, f"No edge at tau = {threshold:g}: the DFS forest is a set of "
                          f"{graph.number_of_nodes()} isolated nodes.",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return _save(fig, name)

    forest: nx.Graph = dfs["forest"]
    articulation = set(dfs["articulation"]["ticker"]) if len(dfs["articulation"]) else set()
    bridges = set()
    if len(dfs["bridges"]):
        bridges = {tuple(sorted((row.stock_a, row.stock_b)))
                   for row in dfs["bridges"].itertuples()}

    pos = _layout(graph, "spring")
    fig, ax = _figure_for_layout(pos, legend_width=3.4)
    scale = _fit_axes_to_layout(fig, ax, pos)
    tree_edges = [tuple(sorted(e)) for e in forest.edges()]
    back_edges = [tuple(sorted(e)) for e in graph.edges()
                  if tuple(sorted(e)) not in set(tree_edges)]

    nx.draw_networkx_edges(graph, pos, edgelist=back_edges, ax=ax, width=0.5,
                           alpha=0.14, edge_color="#777777")
    nx.draw_networkx_edges(graph, pos, edgelist=tree_edges, ax=ax, width=1.5,
                           alpha=0.85, edge_color="#333333")
    if bridges:
        nx.draw_networkx_edges(graph, pos, edgelist=sorted(bridges), ax=ax, width=2.2,
                               alpha=0.95, edge_color="#C44E52", style="dashed")

    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_size=_auto_node_sizes(graph, pos, scale),
        node_color=[_sector_colour(graph.nodes[n].get("sector", "Unknown"))
                    for n in graph.nodes()],
        edgecolors=["#C44E52" if n in articulation else "white" for n in graph.nodes()],
        linewidths=[2.2 if n in articulation else 0.6 for n in graph.nodes()])
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax,
                            font_size=_auto_font_size(graph.number_of_nodes()))

    handles = [
        mpatches.Patch(color="#333333", label="DFS tree edge"),
        mpatches.Patch(color="#777777", label="edge not used by the DFS"),
        mpatches.Patch(color="#C44E52", label=f"bridge ({len(bridges)})"),
        mpatches.Patch(facecolor="white", edgecolor="#C44E52", linewidth=2,
                       label=f"articulation point ({len(articulation)})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, fontsize=9)

    ax.set_title(f"Depth-First Search forest, threshold |C| >= {threshold:g}\n"
                 f"{len(dfs['components'])} tree(s) covering {graph.number_of_nodes()} "
                 f"stocks; {len(articulation)} articulation point(s), "
                 f"{len(bridges)} bridge(s)")
    ax.set_axis_off()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 4. MST figure
# ---------------------------------------------------------------------------


def plot_mst(mst: nx.Graph, node_metrics: pd.DataFrame,
             name: str = "minimum_spanning_tree") -> str:
    """The Minimum Spanning Tree: colour = sector, size = degree.

    The layout is Kamada-Kawai on the Mantegna distances, so geometric
    proximity on the page approximates correlation distance: stocks drawn close
    together really are strongly correlated.
    """
    pos = _layout(mst, "kamada_kawai", weight="distance")
    fig, ax = _figure_for_layout(pos, plot_area=140.0, legend_width=3.2,
                                 side_bounds=(8.0, 18.0))
    scale = _fit_axes_to_layout(fig, ax, pos)

    hub = node_metrics.iloc[0]["ticker"]
    correlations = [d["correlation"] for _, _, d in mst.edges(data=True)]
    widths = [0.8 + 3.2 * max(0.0, c) for c in correlations]

    nx.draw_networkx_edges(mst, pos, ax=ax, width=widths, alpha=0.45,
                           edge_color=["#C44E52" if not d["same_sector"] else "#444444"
                                       for _, _, d in mst.edges(data=True)])
    nx.draw_networkx_nodes(mst, pos, ax=ax,
                           node_size=_auto_node_sizes(mst, pos, scale,
                                                      degree_range=(0.62, 1.55),
                                                      bounds=(14.0, 34.0)),
                           node_color=[_sector_colour(mst.nodes[n].get("sector", "Unknown"))
                                       for n in mst.nodes()],
                           edgecolors=["black" if n == hub else "white" for n in mst.nodes()],
                           linewidths=[2.2 if n == hub else 0.9 for n in mst.nodes()])
    labels = nx.draw_networkx_labels(mst, pos, labels={n: n for n in mst.nodes()},
                                     ax=ax, font_size=8, font_weight="bold")
    # A translucent halo keeps the labels legible where the tree is dense
    # around the hub, without hiding the edges underneath.
    for text in labels.values():
        text.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none",
                           boxstyle="round,pad=0.12"))

    intra = float(np.mean([d["same_sector"] for _, _, d in mst.edges(data=True)]))
    ax.set_title(
        "Minimum Spanning Tree of the stock correlation network\n"
        f"Mantegna distance d = sqrt(2(1-C)); {mst.number_of_edges()} links retained; "
        f"hub = {hub}; {100 * intra:.0f}% of links join two stocks of the same sector\n"
        f"node size = MST degree; grey links = same sector, red links = cross-sector")
    ax.set_axis_off()
    _sector_legend(ax, [mst.nodes[n].get("sector", "Unknown") for n in mst.nodes()])
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 5. Threshold comparison figures
# ---------------------------------------------------------------------------


def _threshold_line(table: pd.DataFrame, column: str, ylabel: str, title: str,
                    name: str, colour: str = "#4C72B0",
                    percentage: bool = False, annotate: bool = True) -> str:
    """Shared helper for the five single-metric threshold plots."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    ax.plot(table["threshold"], table[column], marker="o", markersize=8,
            linewidth=2.2, color=colour)
    if annotate:
        for _, row in table.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            label = f"{value:.1f}%" if percentage else (
                f"{int(value)}" if float(value).is_integer() else f"{value:.3f}")
            ax.annotate(label, (row["threshold"], value), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9)
    ax.set_xlabel("Correlation threshold  |C| >= tau")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(table["threshold"])
    ax.margins(y=0.16)
    return _save(fig, name)


def plot_threshold_comparisons(table: pd.DataFrame) -> Dict[str, str]:
    """The five figures required by Section 14, plus a combined panel."""
    paths = {
        "edges": _threshold_line(
            table, "n_edges", "Number of edges",
            "Number of edges as a function of the correlation threshold",
            "threshold_edges", "#4C72B0"),
        "density": _threshold_line(
            table, "density", "Network density",
            "Network density as a function of the correlation threshold",
            "threshold_density", "#DD8452"),
        "components": _threshold_line(
            table, "n_connected_components", "Number of connected components",
            "Fragmentation of the network as the threshold rises",
            "threshold_components", "#55A868"),
        "giant": _threshold_line(
            table, "largest_component_pct", "Share of stocks in the largest component (%)",
            "Size of the giant component as a function of the threshold",
            "threshold_giant_component", "#C44E52", percentage=True),
        "clustering": _threshold_line(
            table, "average_clustering", "Average clustering coefficient",
            "Average clustering coefficient as a function of the threshold",
            "threshold_clustering", "#8172B3"),
    }

    # Combined panel: convenient as a single figure in the thesis.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    specs = [
        ("n_edges", "Number of edges", "#4C72B0"),
        ("density", "Density", "#DD8452"),
        ("n_connected_components", "Connected components", "#55A868"),
        ("largest_component_pct", "Giant component (% of stocks)", "#C44E52"),
        ("average_clustering", "Average clustering coefficient", "#8172B3"),
        ("n_isolated_nodes", "Isolated stocks", "#937860"),
    ]
    for ax, (column, label, colour) in zip(axes.flat, specs):
        ax.plot(table["threshold"], table[column], marker="o", linewidth=2, color=colour)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("threshold tau")
        ax.set_xticks(table["threshold"])
    fig.suptitle("Response of the correlation network to the filtering threshold\n"
                 f"{int(table['n_nodes'].iloc[0])} stocks, "
                 f"{config.START_DATE} to {config.END_DATE}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths["panel"] = _save(fig, "threshold_summary_panel")
    return paths


def plot_bfs_reachability(reachability: pd.DataFrame, source: str,
                          name: str | None = None) -> str:
    """How the source stock's reachable set shrinks as the threshold rises."""
    name = name or f"bfs_reachability_{source}"
    fig, ax_left = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    ax_left.plot(reachability["threshold"], 100 * reachability["share_reachable"],
                 marker="o", linewidth=2.2, color="#4C72B0", label="stocks reachable (%)")
    ax_left.set_xlabel("Correlation threshold  |C| >= tau")
    ax_left.set_ylabel("Share of the universe reachable from the source (%)",
                       color="#4C72B0")
    ax_left.tick_params(axis="y", labelcolor="#4C72B0")
    ax_left.set_xticks(reachability["threshold"])

    ax_right = ax_left.twinx()
    ax_right.plot(reachability["threshold"], reachability["mean_distance"],
                  marker="s", linewidth=2.2, color="#C44E52", linestyle="--",
                  label="mean BFS distance")
    ax_right.set_ylabel("Mean BFS distance to the reachable stocks (edges)",
                        color="#C44E52")
    ax_right.tick_params(axis="y", labelcolor="#C44E52")
    ax_right.grid(False)

    lines = ax_left.get_lines() + ax_right.get_lines()
    ax_left.legend(lines, [line.get_label() for line in lines], loc="center left")
    ax_left.set_title(f"Reachability from {source} as the correlation threshold rises")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 6. Portfolio figure
# ---------------------------------------------------------------------------


def plot_portfolio_cumulative_returns(cumulative: pd.DataFrame,
                                      name: str = "portfolio_cumulative_returns") -> str:
    """Cumulative growth of one unit invested in each illustrative portfolio."""
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = ["#4C72B0", "#C44E52", "#8C8C8C", "#55A868"]
    for index, column in enumerate(cumulative.columns):
        ax.plot(cumulative.index, cumulative[column], linewidth=1.9,
                color=palette[index % len(palette)], label=column)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of one unit invested at the start of the sample")
    ax.set_title("Illustrative portfolios: cumulative performance\n"
                 "equally weighted, daily rebalanced, no transaction costs - "
                 "an in-sample illustration, not a trading strategy")
    ax.legend(loc="upper left")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    return _save(fig, name)


def plot_portfolio_comparison_bars(comparison: pd.DataFrame,
                                   name: str = "portfolio_risk_comparison") -> str:
    """Side-by-side bars for the four headline risk statistics."""
    metrics = [("average_pairwise_correlation", "Average pairwise correlation"),
               ("annualised_volatility", "Annualised volatility"),
               ("annualised_return", "Annualised return"),
               ("max_drawdown", "Maximum drawdown")]
    available = [(c, l) for c, l in metrics if c in comparison.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 5))
    axes = np.atleast_1d(axes)
    palette = ["#4C72B0", "#C44E52", "#8C8C8C", "#55A868"]
    for ax, (column, label) in zip(axes, available):
        values = comparison[column].astype(float)
        ax.bar(comparison["portfolio"], values,
               color=[palette[i % len(palette)] for i in range(len(comparison))])
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Illustrative portfolio comparison (in-sample, equally weighted)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, name)
