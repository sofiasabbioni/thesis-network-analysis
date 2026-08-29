"""
src
===

Analysis modules for the thesis "Analysis of Stock Correlation Networks
through Graph Traversal Algorithms".

Each module corresponds to one step of the empirical pipeline described in
Chapter 3 and is importable and testable in isolation:

    data_collection      download raw adjusted closing prices
    preprocessing        align, clean and transform prices into log returns
    correlation          Pearson correlation matrix and its description
    graph_construction   complete and threshold-filtered correlation graphs
    metrics              node- and graph-level network statistics
    traversal_analysis   Breadth-First Search and Depth-First Search
    mst_analysis         Mantegna distance and Minimum Spanning Tree
    visualisation        every figure exported to outputs/figures
    portfolio            illustrative (non-prescriptive) portfolio comparison
    interpretation       plain-language summaries written to outputs/logs
    utils                shared helpers (paths, logging, table export)
"""

__all__ = [
    "data_collection",
    "preprocessing",
    "correlation",
    "graph_construction",
    "metrics",
    "traversal_analysis",
    "mst_analysis",
    "visualisation",
    "portfolio",
    "interpretation",
    "utils",
]
