"""
export_thesis_network.py
========================

Exports the **original 47-stock thesis network** as a node list and an edge
list, as requested by the supervisor so that the network itself is available
directly from the repository.

    python export_thesis_network.py

Writes, into ``network_exports/``:

    network_nodes_threshold_0_5.csv
    network_edges_threshold_0_5.csv

What this script is careful *not* to do
---------------------------------------
It does not download anything, it does not re-run the thesis pipeline and it
does not write into ``data/`` or ``outputs/``.  It reads the correlation matrix
that the thesis run already produced -
``outputs/tables/correlation_matrix.csv`` - and rebuilds the threshold graph
from it with the same ``src.graph_construction`` code the pipeline used.  The
exported files therefore describe *the exact network already used in the
thesis*, and running this script can never change a thesis number.

To prove that rather than assert it, the script also cross-checks the rebuilt
network against the tables the original run wrote next to that correlation
matrix (component membership, articulation points, MST leaves, network roles,
and the threshold summary row).  Any disagreement is reported and the script
exits non-zero.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

import pandas as pd

import config
from src import network_export, utils


def load_correlation_matrix(path: str) -> pd.DataFrame:
    """Read the exported correlation matrix back into a square DataFrame."""
    if not os.path.exists(path):
        raise utils.PipelineError(
            f"Correlation matrix not found at {path}.\n"
            "Run the thesis pipeline first:\n"
            "    python main.py --refresh\n"
            "or point --correlation-matrix at an existing copy.")
    corr = pd.read_csv(path, index_col=0)
    corr.index.name = "ticker"
    corr.columns.name = "ticker"
    if list(corr.index) != list(corr.columns):
        raise utils.PipelineError(
            f"{path} is not square: {corr.shape[0]} rows vs {corr.shape[1]} columns.")
    return corr


def cross_check(result: Dict[str, object], tables_dir: str,
                threshold: float) -> List[str]:
    """Compare the rebuilt network with the tables of the original run.

    Every check is skipped silently when its reference file is absent, so the
    script still works on a fresh clone; when the file is there, a mismatch is
    a hard error.  Returns the list of checks that actually ran.
    """
    tag = utils.threshold_tag(threshold)
    nodes: pd.DataFrame = result["nodes"]
    problems: List[str] = []
    performed: List[str] = []

    def reference(name: str) -> pd.DataFrame | None:
        path = os.path.join(tables_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    # -- component membership --------------------------------------------
    components = reference(f"connected_components_threshold_{tag}.csv")
    if components is not None:
        performed.append("component membership")
        merged = nodes.merge(components[["ticker", "component_id", "degree"]],
                             on="ticker", how="outer", indicator=True)
        if (merged["_merge"] != "both").any():
            missing = merged.loc[merged["_merge"] != "both", "ticker"].tolist()
            problems.append(f"ticker set differs from the original run: {missing}")
        else:
            bad = merged.loc[merged["component"] != merged["component_id"], "ticker"].tolist()
            if bad:
                problems.append(f"component id differs for {bad}")
            bad = merged.loc[merged["degree_x"] != merged["degree_y"], "ticker"].tolist()
            if bad:
                problems.append(f"degree differs for {bad}")

    # -- articulation points ----------------------------------------------
    articulation = reference(f"articulation_points_threshold_{tag}.csv")
    if articulation is not None:
        performed.append("articulation points")
        expected = set(articulation["ticker"]) if len(articulation) else set()
        actual = set(nodes.loc[nodes["is_articulation_point"], "ticker"])
        if expected != actual:
            problems.append(f"articulation points differ: original {sorted(expected)}, "
                            f"rebuilt {sorted(actual)}")

    # -- MST leaves ---------------------------------------------------------
    mst = reference("mst_node_metrics.csv")
    if mst is not None:
        performed.append("MST leaves")
        expected = set(mst.loc[mst["is_leaf"].astype(bool), "ticker"])
        actual = set(nodes.loc[nodes["is_mst_leaf"], "ticker"])
        if expected != actual:
            problems.append(f"MST leaves differ: only in original "
                            f"{sorted(expected - actual)}, only in rebuilt "
                            f"{sorted(actual - expected)}")

    # -- network roles ------------------------------------------------------
    framework = reference("investment_interpretation_framework.csv")
    if framework is not None and "network_role" in framework.columns:
        performed.append("network roles")
        merged = nodes.merge(framework[["ticker", "network_role"]],
                             on="ticker", how="inner", suffixes=("", "_original"))
        bad = merged.loc[merged["network_role"] != merged["network_role_original"],
                         "ticker"].tolist()
        if bad:
            problems.append(f"network_role differs for {bad}")

    # -- headline threshold row ---------------------------------------------
    summary = reference("threshold_network_metrics.csv")
    if summary is not None:
        row = summary.loc[summary["threshold"].round(6) == round(threshold, 6)]
        if len(row) == 1:
            performed.append("threshold summary row")
            row = row.iloc[0]
            if int(row["n_nodes"]) != len(nodes):
                problems.append(f"node count {len(nodes)} != original {int(row['n_nodes'])}")
            if int(row["n_edges"]) != len(result["edges"]):
                problems.append(f"edge count {len(result['edges'])} != original "
                                f"{int(row['n_edges'])}")
            if int(row["n_isolated_nodes"]) != int(nodes["is_isolated"].sum()):
                problems.append("isolated-node count differs from the original run")

    if problems:
        raise utils.PipelineError(
            "The rebuilt network does NOT match the original thesis run:\n  - "
            + "\n  - ".join(problems))
    return performed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the original 47-stock thesis network as node and edge lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--correlation-matrix",
                        default=os.path.join(config.TABLES_DIR, "correlation_matrix.csv"),
                        help="Correlation matrix written by the thesis pipeline.")
    parser.add_argument("--threshold", type=float, default=config.MAIN_THRESHOLD,
                        help="Threshold to export.")
    parser.add_argument("--output-dir", default=config.NETWORK_EXPORT_DIR,
                        help="Where to write the two CSV files.")
    parser.add_argument("--prefix", default="network",
                        help="File-name prefix; keeps the thesis network distinct "
                             "from the large-network export.")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="Skip the comparison against the original run's tables.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        utils.section("Exporting the original 47-stock thesis network")
        logging.info("Correlation matrix : %s", args.correlation_matrix)
        logging.info("Threshold          : tau = %.2f", args.threshold)

        corr = load_correlation_matrix(args.correlation_matrix)
        logging.info("Universe           : %d stocks", corr.shape[0])

        # Guard against exporting a network built on a universe that is not the
        # thesis universe (for instance a large-network matrix passed by mistake).
        unexpected = [t for t in corr.columns if t not in config.TICKER_TO_SECTOR]
        if unexpected:
            logging.warning("")
            logging.warning("%d ticker(s) in this matrix are not in the 47-stock thesis "
                            "universe declared in config.py:", len(unexpected))
            logging.warning("  %s", ", ".join(unexpected[:20]))
            logging.warning("They will be exported with sector 'Unknown'. If you meant to "
                            "export the large network, use large_network_extension.py.")

        utils.subsection("Rebuilding the threshold graph from the correlation matrix")
        result = network_export.compute_network_export(corr, args.threshold)
        stats = network_export.verify_export(result["nodes"], result["edges"],
                                             result["graph"])
        logging.info("  nodes / edges          : %d / %d", stats["n_nodes"], stats["n_edges"])
        logging.info("  isolated nodes         : %d", stats["n_isolated"])
        logging.info("  connected components   : %d", stats["n_components"])
        logging.info("  articulation points    : %d", stats["n_articulation_points"])
        logging.info("  same-sector edge share : %.1f%%",
                     100 * stats["same_sector_edge_share"])

        if not args.no_cross_check:
            utils.subsection("Cross-checking against the original thesis outputs")
            tables_dir = os.path.dirname(os.path.abspath(args.correlation_matrix))
            performed = cross_check(result, tables_dir, args.threshold)
            if performed:
                for check in performed:
                    logging.info("  OK  %s matches the original run", check)
            else:
                logging.warning("  no reference tables found next to the correlation "
                                "matrix; nothing could be cross-checked")

        utils.subsection("Writing the export")
        network_export.write_export(result["nodes"], result["edges"],
                                    args.prefix, args.threshold, args.output_dir)
        logging.info("")
        logging.info("Done. These files describe the exact network analysed in the thesis.")
        return 0
    except utils.PipelineError as exc:
        logging.error("")
        logging.error("Export failed:")
        logging.error("  %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
