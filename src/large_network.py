"""
large_network.py
================

Analysis code specific to the **supplementary large-network extension**.

The extension asks a different question from the thesis.  The 47-stock study
asks what a correlation network *looks like* and how graph traversal algorithms
read it.  The extension asks whether the structure the thesis found survives at
several hundred nodes, and it uses GICS sector metadata as an **external
reference** against which the network can be checked.

The methodological rule that governs this whole module
------------------------------------------------------
Sector metadata is used **only after the network exists**.  Nothing in
``src.graph_construction`` consults a sector when deciding whether an edge
survives a threshold: an edge is kept when ``|C_ij| >= tau`` and for no other
reason.  Every quantity computed here is therefore a comparison between two
independently obtained objects - a network estimated from returns alone, and a
classification produced by S&P/MSCI - rather than a property the construction
put there.

That is also why the language of this module is deliberately neutral.  A stock
whose neighbours are mostly outside its own GICS sector is described as
*cross-sector-oriented*, never as "misclassified": GICS classifies a company by
its business activity, while an edge here records co-movement of returns over
one particular sample.  The two disagreeing is information about the sample,
not an error in the classification.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import config
from src import utils


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def load_universe(path: str | None = None) -> pd.DataFrame:
    """Read the static, committed S&P 500 constituent file.

    A file rather than a live scrape, so that the universe cannot drift between
    runs.  See ``data/reference/CONSTITUENTS_SOURCE.md`` for the source and the
    snapshot date.
    """
    path = path or config.LARGE_UNIVERSE_FILE
    if not os.path.exists(path):
        raise utils.PipelineError(
            f"Constituent file not found: {path}\n"
            "Regenerate it with:\n"
            "    git clone https://github.com/fja05680/sp500 /tmp/sp500\n"
            "    git -C /tmp/sp500 fetch --depth=1000 origin master\n"
            "    python tools/build_sp500_universe.py --repo /tmp/sp500")

    universe = pd.read_csv(path)
    required = {"ticker", "yahoo_ticker", "company", "gics_sector"}
    missing = required - set(universe.columns)
    if missing:
        raise utils.PipelineError(f"{path} is missing column(s): {sorted(missing)}")

    universe = universe.dropna(subset=["ticker", "gics_sector"])
    universe = universe.drop_duplicates(subset="ticker").reset_index(drop=True)
    return universe


def universe_provenance(path: str | None = None) -> Dict[str, object]:
    """Read the provenance record written beside the constituent file."""
    path = path or config.LARGE_UNIVERSE_PROVENANCE_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.warning("Could not read the universe provenance record (%s).", exc)
        return {}


def apply_universe_to_config(universe: pd.DataFrame) -> None:
    """Make ``src/`` see the large universe instead of the 47-stock one.

    The analysis modules read ``config.TICKER_TO_SECTOR`` and friends at call
    time, so this single rebinding is what lets the extension reuse the thesis
    code unchanged.  ``main.py`` never calls it.

    Stocks are keyed by their **Yahoo** symbol, because that is the column name
    the downloaded price panel will carry.  The canonical GICS symbol differs
    only for share classes (``BRK.B`` -> ``BRK-B``).
    """
    by_sector: Dict[str, List[str]] = {}
    for sector, group in universe.groupby("gics_sector", sort=True):
        by_sector[sector] = sorted(group["yahoo_ticker"])
    companies = dict(zip(universe["yahoo_ticker"], universe["company"]))
    config.apply_universe(by_sector, companies)


# ---------------------------------------------------------------------------
# Phase 5 - threshold metrics
# ---------------------------------------------------------------------------

# The column names the extension specification asks for, mapped onto the keys
# that ``graph_construction.graph_summary`` already produces.  Reusing that
# function rather than recomputing is deliberate: it guarantees that "density"
# or "average clustering" means exactly what it means in the thesis tables.
_THRESHOLD_COLUMNS = [
    ("threshold", "threshold"),
    ("nodes", "n_nodes"),
    ("edges", "n_edges"),
    ("density", "density"),
    ("avg_degree", "average_degree"),
    ("avg_clustering", "average_clustering"),
    ("components", "n_connected_components"),
    ("giant_component_size", "largest_component_size"),
    ("giant_component_share", "largest_component_pct"),
    ("isolated_nodes", "n_isolated_nodes"),
    ("same_sector_edge_share", "intra_sector_edge_pct"),
    ("sector_assortativity", "sector_assortativity"),
]

# Columns that ``graph_summary`` reports as a percentage but that this table
# reports as a fraction in [0, 1].  Converting here keeps the shared function -
# and therefore the thesis tables it also feeds - untouched.
_THRESHOLD_PERCENT_COLUMNS = ("giant_component_share", "same_sector_edge_share")


def threshold_metrics_table(summaries: List[Dict[str, object]]) -> pd.DataFrame:
    """Build ``large_threshold_network_metrics.csv`` from graph summaries.

    ``same_sector_edge_share`` is the share of surviving edges whose two
    endpoints carry the same GICS sector.  It is read straight from the value
    ``graph_summary`` already computes (``intra_sector_edge_pct``), so it is the
    same quantity the thesis tables report and no graph is rebuilt to obtain it.
    Read together with ``sector_assortativity`` it separates two different
    questions: the share says how much of the network is intra-sector, the
    assortativity says how much more that is than chance would give.
    """
    rows = []
    for summary in summaries:
        row = {name: summary[key] for name, key in _THRESHOLD_COLUMNS}
        for column in _THRESHOLD_PERCENT_COLUMNS:
            value = row[column]
            row[column] = float(value) / 100.0 if value == value else value
        rows.append(row)
    return pd.DataFrame(rows, columns=[name for name, _ in _THRESHOLD_COLUMNS])


# ---------------------------------------------------------------------------
# Phase 6 - node-level sector alignment
# ---------------------------------------------------------------------------


def _dominant_sector(sectors: List[str]) -> Tuple[str, List[str], float, bool]:
    """Most frequent sector(s) among a node's neighbours.

    Returns ``(display, tied, share, is_tied)`` where ``tied`` is **every**
    sector attaining the maximum neighbour count and ``display`` is the
    alphabetically first of them.

    The alphabetical pick exists only so that a single column can be shown
    deterministically.  It must never carry analytical weight: when two sectors
    are equally represented among a stock's neighbours there is no dominant
    sector, and letting the alphabet decide would make a stock look
    cross-sector-oriented purely because a rival sector's name sorts earlier
    than its own.  Callers therefore work from ``tied`` and ``is_tied``, not
    from ``display``.
    """
    counts = pd.Series(sectors).value_counts()
    top = counts.max()
    tied = sorted(counts[counts == top].index)
    return tied[0], tied, float(top / len(sectors)), len(tied) > 1


def node_sector_alignment(graph: nx.Graph) -> pd.DataFrame:
    """One row per **non-isolated** node describing its neighbourhood's sectors.

    Isolated nodes are excluded because every quantity in the table is a share
    of a node's neighbours, which is undefined when there are none.  They are
    still counted, and reported separately, by the summary below and they
    remain present in the network export.
    """
    # Sector sizes over the whole node set, used for the per-stock chance
    # benchmark below.  Isolated nodes are counted: they were available as
    # potential partners even though no edge reached them.
    sector_sizes = pd.Series(
        [graph.nodes[n].get("sector", "Unknown") for n in graph.nodes()]).value_counts()
    n_nodes = int(sector_sizes.sum())

    rows = []
    for node in sorted(graph.nodes()):
        neighbours = list(graph.neighbors(node))
        if not neighbours:
            continue
        own_sector = graph.nodes[node].get("sector", "Unknown")
        neighbour_sectors = [graph.nodes[nb].get("sector", "Unknown") for nb in neighbours]
        same = sum(1 for s in neighbour_sectors if s == own_sector)
        dominant, tied_sectors, dominant_share, tied = _dominant_sector(neighbour_sectors)
        # The official sector counts as dominant whenever it is among the sectors
        # tied for the maximum, not only when it wins the alphabetical tie-break.
        official_in_tie = own_sector in tied_sectors
        observed_share = same / len(neighbours)

        # Share of same-sector partners this stock could have reached at all:
        # the expected same-sector share if its links were placed at random.
        # It differs sector by sector, which is exactly why the raw share is
        # not comparable across stocks.
        available = int(sector_sizes.get(own_sector, 1)) - 1
        expected_share = available / (n_nodes - 1) if n_nodes > 1 else np.nan
        ratio = (observed_share / expected_share
                 if expected_share and expected_share > 0 else np.nan)

        rows.append({
            "ticker": node,
            "company": graph.nodes[node].get("company", node),
            "official_sector": own_sector,
            "degree": len(neighbours),
            "same_sector_neighbors": same,
            "cross_sector_neighbors": len(neighbours) - same,
            "same_sector_share": observed_share,
            "expected_same_sector_share": expected_share,
            "same_sector_share_ratio": ratio,
            "dominant_neighbor_sector": dominant,
            "dominant_neighbor_sector_share": dominant_share,
            "dominant_neighbor_sectors_tied": ", ".join(tied_sectors),
            "n_dominant_tied_sectors": len(tied_sectors),
            "dominant_sector_is_tied": tied,
            "official_in_dominant_tie": official_in_tie,
            "dominant_equals_official": official_in_tie,
            "n_neighbor_sectors": len(set(neighbour_sectors)),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["same_sector_share_ratio", "degree"],
                                  ascending=[True, False]).reset_index(drop=True)
    return table


def sector_alignment_summary(graph: nx.Graph, alignment: pd.DataFrame) -> Dict[str, object]:
    """Overall measures of how far the network agrees with the classification.

    ``sector_assortativity`` is Newman's attribute assortativity: the
    correlation of the sector labels across edges, normalised so that 0 is what
    a random attachment would give and 1 is a network whose every edge stays
    inside one sector.  It is the single number that answers the supervisor's
    question most directly; the neighbour-share statistics beside it describe
    the same phenomenon node by node, which is what makes individual stocks
    comparable.

    ``random_benchmark_same_sector_share`` is the share of same-sector edges
    that would be expected if edges were placed at random between the observed
    nodes, so that the observed share can be read against something.
    """
    edges = list(graph.edges(data=True))
    same_sector_edges = [bool(d["same_sector"]) for _, _, d in edges]

    sector_counts = pd.Series(
        [graph.nodes[n].get("sector", "Unknown") for n in graph.nodes()]).value_counts()
    n = int(sector_counts.sum())
    # Probability that a uniformly random unordered pair of distinct nodes
    # falls inside one sector.
    within_pairs = float(sum(c * (c - 1) / 2 for c in sector_counts))
    all_pairs = n * (n - 1) / 2 if n > 1 else np.nan
    random_benchmark = within_pairs / all_pairs if all_pairs else np.nan

    try:
        assortativity = (float(nx.attribute_assortativity_coefficient(graph, "sector"))
                         if edges else np.nan)
    except Exception:
        assortativity = np.nan

    observed = float(np.mean(same_sector_edges)) if same_sector_edges else np.nan
    isolated = int(sum(1 for _, d in graph.degree() if d == 0))

    return {
        "threshold": float(graph.graph.get("threshold", np.nan)),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_isolated_nodes": isolated,
        "n_nodes_with_neighbours": int(len(alignment)),
        "same_sector_edge_share": observed,
        "random_benchmark_same_sector_share": random_benchmark,
        "same_sector_ratio_vs_random": (observed / random_benchmark
                                        if random_benchmark else np.nan),
        "sector_assortativity": assortativity,
        "mean_same_sector_share": (float(alignment["same_sector_share"].mean())
                                   if len(alignment) else np.nan),
        "median_same_sector_share": (float(alignment["same_sector_share"].median())
                                     if len(alignment) else np.nan),
        "share_dominant_equals_official": (float(alignment["dominant_equals_official"].mean())
                                           if len(alignment) else np.nan),
        "n_dominant_equals_official": (int(alignment["dominant_equals_official"].sum())
                                       if len(alignment) else 0),
        "n_dominant_sector_tied": (int(alignment["dominant_sector_is_tied"].sum())
                                   if len(alignment) else 0),
        "n_official_in_dominant_tie": (int((alignment["dominant_sector_is_tied"]
                                            & alignment["official_in_dominant_tie"]).sum())
                                       if len(alignment) else 0),
    }


def sector_alignment_by_sector(alignment: pd.DataFrame,
                               graph: nx.Graph) -> pd.DataFrame:
    """The same statistics broken down by official GICS sector.

    ``n_isolated`` is taken from the graph rather than from ``alignment``, which
    by construction contains no isolated node, so that each sector's row still
    accounts for all of its stocks.
    """
    isolated_by_sector = pd.Series(
        [graph.nodes[n].get("sector", "Unknown")
         for n, d in graph.degree() if d == 0]).value_counts()

    rows = []
    for sector, group in alignment.groupby("official_sector", sort=True):
        rows.append({
            "sector": sector,
            "n_stocks_with_neighbours": len(group),
            "n_isolated": int(isolated_by_sector.get(sector, 0)),
            "mean_degree": float(group["degree"].mean()),
            "mean_same_sector_share": float(group["same_sector_share"].mean()),
            "median_same_sector_share": float(group["same_sector_share"].median()),
            "min_same_sector_share": float(group["same_sector_share"].min()),
            "max_same_sector_share": float(group["same_sector_share"].max()),
            "share_dominant_equals_official": float(group["dominant_equals_official"].mean()),
            "n_dominant_equals_official": int(group["dominant_equals_official"].sum()),
        })
    return pd.DataFrame(rows).sort_values("mean_same_sector_share",
                                          ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 7 - cross-sector-oriented nodes
# ---------------------------------------------------------------------------

CROSS_SECTOR_COLUMNS = ["ticker", "company", "official_sector", "degree",
                        "same_sector_neighbors", "cross_sector_neighbors",
                        "same_sector_share", "dominant_neighbor_sector",
                        "dominant_neighbor_sector_share"]


def cross_sector_oriented_nodes(alignment: pd.DataFrame,
                                min_degree: int | None = None,
                                max_share_ratio: float | None = None) -> pd.DataFrame:
    """Stocks whose neighbourhood is unusually cross-sector-oriented.

    The rule is the conjunction of three directly readable conditions, fixed in
    ``config`` before the results are seen:

    1. ``degree >= min_degree`` - below this a share of neighbours is too
       coarse to interpret;
    2. ``same_sector_share_ratio <= max_share_ratio``, i.e. the stock has no
       more same-sector neighbours than chance alone would give it;
    3. there is an **unambiguous** most frequent sector among the neighbours
       (no tie for the maximum) and it is not the stock's own.

    Condition 2 is deliberately a *ratio to a per-stock benchmark* rather than
    an absolute cut-off: sectors differ in size by nearly a factor of four in
    this universe, so a flat cut-off on the raw share would select stocks for
    belonging to a small sector rather than for being cross-sector-oriented.
    Both the observed share and the benchmark are exported alongside, so the
    ratio can be checked by hand.

    Condition 3 requires the absence of a tie because a tie means there is no
    dominant sector to compare against.  A stock whose neighbours split evenly
    between its own sector and another must not be called cross-sector-oriented
    on the strength of an alphabetical tie-break, and neither must one whose
    neighbours split evenly between two sectors that are both foreign to it -
    in that case the neighbourhood simply has no single orientation to report.
    Ties are kept visible in the exported table rather than filtered away
    silently.

    No composite score is invented: every condition is a single observable, and
    the ranking is on the ratio itself, with degree breaking ties so that the
    better-evidenced cases come first.

    These are *sector-atypical network positions*, not misclassifications: the
    network is estimated from return co-movement over one sample, while GICS
    classifies a company by what it sells.
    """
    min_degree = config.LARGE_CROSS_SECTOR_MIN_DEGREE if min_degree is None else min_degree
    max_share_ratio = (config.LARGE_CROSS_SECTOR_MAX_SHARE_RATIO
                       if max_share_ratio is None else max_share_ratio)

    if alignment.empty:
        return pd.DataFrame(columns=CROSS_SECTOR_COLUMNS)

    selected = alignment[
        (alignment["degree"] >= min_degree)
        & (alignment["same_sector_share_ratio"] <= max_share_ratio)
        & (~alignment["dominant_equals_official"])
        & (~alignment["dominant_sector_is_tied"])
    ].copy()

    selected = selected.sort_values(["same_sector_share_ratio", "degree"],
                                    ascending=[True, False]).reset_index(drop=True)
    columns = (CROSS_SECTOR_COLUMNS
               + ["expected_same_sector_share", "same_sector_share_ratio",
                  "dominant_neighbor_sectors_tied", "n_dominant_tied_sectors",
                  "dominant_sector_is_tied", "official_in_dominant_tie",
                  "n_neighbor_sectors"])
    return selected[[c for c in columns if c in selected.columns]]


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------


def run_sector_alignment(graph: nx.Graph, threshold: float) -> Dict[str, object]:
    """Compute and export every Phase 6/7 artefact for one threshold graph.

    Writes
    ------
    ``large_node_sector_alignment.csv``
    ``large_sector_alignment_summary.csv``
    ``large_sector_alignment_by_sector.csv``
    ``large_cross_sector_oriented_nodes.csv``
    """
    utils.subsection(f"Sector alignment of the large network (tau = {threshold:.2f})")

    alignment = node_sector_alignment(graph)
    summary = sector_alignment_summary(graph, alignment)
    by_sector = sector_alignment_by_sector(alignment, graph)
    cross_sector = cross_sector_oriented_nodes(alignment)

    # Carry the flag back onto the full table.  ``alignment`` is already sorted
    # by ``same_sector_share_ratio`` ascending, so it *is* the ranked list of
    # candidates whether or not any of them meets the rule; the flag says which
    # ones do.  This matters because a strict rule can legitimately select
    # nothing - a network in which every stock is at least as sector-aligned as
    # chance is a finding, not a failed query - and the reader still needs
    # somewhere to see who came closest.
    flagged = set(cross_sector["ticker"]) if len(cross_sector) else set()
    alignment = alignment.copy()
    alignment["meets_cross_sector_rule"] = alignment["ticker"].isin(flagged)

    utils.save_table(alignment, "large_node_sector_alignment.csv")
    utils.save_table(pd.DataFrame([summary]), "large_sector_alignment_summary.csv")
    utils.save_table(by_sector, "large_sector_alignment_by_sector.csv")
    utils.save_table(cross_sector, "large_cross_sector_oriented_nodes.csv")

    logging.info("  nodes with neighbours    : %d (of %d; %d isolated)",
                 summary["n_nodes_with_neighbours"], summary["n_nodes"],
                 summary["n_isolated_nodes"])
    logging.info("  same-sector edge share   : %.1f%%  (random benchmark %.1f%%, "
                 "ratio %.2fx)",
                 100 * summary["same_sector_edge_share"],
                 100 * summary["random_benchmark_same_sector_share"],
                 summary["same_sector_ratio_vs_random"])
    logging.info("  sector assortativity     : %+.4f", summary["sector_assortativity"])
    logging.info("  same-sector share, mean  : %.4f", summary["mean_same_sector_share"])
    logging.info("  same-sector share, median: %.4f", summary["median_same_sector_share"])
    logging.info("  own sector also dominant : %d of %d (%.1f%%)  "
                 "[counts ties in which the official sector is among the leaders]",
                 summary["n_dominant_equals_official"],
                 summary["n_nodes_with_neighbours"],
                 100 * summary["share_dominant_equals_official"])
    logging.info("  dominant-sector ties     : %d (%d of them include the "
                 "official sector)",
                 summary["n_dominant_sector_tied"],
                 summary["n_official_in_dominant_tie"])
    logging.info("  cross-sector-oriented    : %d node(s)  (degree >= %d, "
                 "same-sector share <= %.2fx chance, dominant sector differs)",
                 len(cross_sector), config.LARGE_CROSS_SECTOR_MIN_DEGREE,
                 config.LARGE_CROSS_SECTOR_MAX_SHARE_RATIO)
    if len(cross_sector):
        for _, row in cross_sector.head(5).iterrows():
            logging.info("      %-6s %-24s %-24s share %.3f (%.2fx chance), "
                         "neighbours mostly %s",
                         row["ticker"], str(row["company"])[:24],
                         row["official_sector"][:24], row["same_sector_share"],
                         row["same_sector_share_ratio"],
                         row["dominant_neighbor_sector"])
    else:
        logging.info("      no stock met all three conditions; the least "
                     "sector-aligned names are the first rows of")
        logging.info("      large_node_sector_alignment.csv, which is ranked by "
                     "same_sector_share_ratio")

    return {"alignment": alignment, "summary": summary,
            "by_sector": by_sector, "cross_sector": cross_sector}
