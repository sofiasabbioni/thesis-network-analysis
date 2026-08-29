"""
utils.py
========

Small shared helpers used by every other module: directory creation, logging,
deterministic file naming and table export.

Keeping these here avoids repeating boilerplate in the analysis modules and
guarantees that every table and figure produced by the project is written in
exactly the same way (same encoding, same float precision, same DPI).
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime
from typing import Iterable

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """A failure the pipeline anticipated and can explain to the user.

    Raised for conditions such as "no price data could be downloaded" or "too
    few stocks survived cleaning", where a full Python traceback would only
    obscure an actionable message.  ``main.main`` prints these without a
    traceback and keeps the traceback for genuinely unexpected exceptions.
    """


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def ensure_directories(dirs: Iterable[str] | None = None) -> None:
    """Create every output/data directory if it does not already exist.

    Called once at the start of :func:`main.run_pipeline` so that the project
    can be cloned without its (empty) data and output folders.

    Parameters
    ----------
    dirs : iterable of str, optional
        Directories to create.  Defaults to ``config.ALL_DIRS``.
    """
    for directory in dirs if dirs is not None else config.ALL_DIRS:
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def threshold_tag(threshold: float) -> str:
    """Turn a threshold into a file-name-safe tag.

    ``0.5 -> "0_5"``, ``0.35 -> "0_35"``.  Used so that the exported files are
    named ``node_metrics_threshold_0_5.csv`` and so on, exactly as referenced
    in the thesis text.
    """
    return ("%g" % float(threshold)).replace(".", "_").replace("-", "m")


def table_path(name: str) -> str:
    """Absolute path of a CSV table inside ``outputs/tables``."""
    if not name.endswith(".csv"):
        name = name + ".csv"
    return os.path.join(config.TABLES_DIR, name)


def figure_path(name: str) -> str:
    """Absolute path of a figure inside ``outputs/figures``."""
    if "." not in os.path.basename(name):
        name = f"{name}.{config.FIGURE_FORMAT}"
    return os.path.join(config.FIGURES_DIR, name)


def log_path(name: str) -> str:
    """Absolute path of a text log inside ``outputs/logs``."""
    if not name.endswith(".txt"):
        name = name + ".txt"
    return os.path.join(config.LOGS_DIR, name)


def relative(path: str) -> str:
    """Path relative to the project root, for compact log messages."""
    try:
        return os.path.relpath(path, config.BASE_DIR)
    except ValueError:  # pragma: no cover - different drive on Windows
        return path


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def save_table(df: pd.DataFrame, name: str, index: bool = False,
               float_format: str = "%.6f") -> str:
    """Write a DataFrame to ``outputs/tables`` and log the fact.

    A single choke-point for CSV export keeps the numerical precision of every
    exported table identical, which matters when the tables are pasted into the
    thesis.

    Returns
    -------
    str
        The absolute path that was written.
    """
    path = table_path(name)
    df.to_csv(path, index=index, float_format=float_format)
    logging.info("  table  -> %s  (%d rows x %d cols)",
                 relative(path), len(df), df.shape[1])
    return path


def save_text(text: str, name: str) -> str:
    """Write a plain-text report to ``outputs/logs`` and log the fact."""
    path = log_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    logging.info("  report -> %s  (%d characters)", relative(path), len(text))
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: str = "run_log.txt", verbose: bool = True) -> None:
    """Configure logging to the console *and* to ``outputs/logs/run_log.txt``.

    The file copy is what makes a run auditable months later, when the thesis
    is being written and the exact universe/period of a figure must be checked.
    """
    ensure_directories([config.LOGS_DIR])
    path = log_path(log_file)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove handlers left over from a previous call (e.g. in a notebook).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # Third-party libraries are chatty; keep the thesis log readable.
    for noisy in ("matplotlib", "yfinance", "peewee", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def section(title: str) -> None:
    """Print a visually separated section header into the log."""
    logging.info("")
    logging.info("=" * 78)
    logging.info(title.upper())
    logging.info("=" * 78)


def subsection(title: str) -> None:
    """Print a lighter sub-section header into the log."""
    logging.info("")
    logging.info("--- %s " + "-" * max(0, 70 - len(title)), title)


# ---------------------------------------------------------------------------
# Run metadata (reproducibility)
# ---------------------------------------------------------------------------


def environment_report() -> str:
    """Return a text block describing the software environment of the run.

    Written into ``outputs/logs/summary_interpretation.txt`` so that the
    "reproducibility" paragraph of Chapter 3 can quote exact versions.
    """
    lines = [
        f"Run timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python               : {platform.python_version()} ({platform.system()})",
    ]
    for module_name in ("pandas", "numpy", "networkx", "matplotlib", "seaborn",
                        "yfinance", "scipy"):
        try:
            module = __import__(module_name)
            lines.append(f"{module_name:<21}: {getattr(module, '__version__', 'n/a')}")
        except Exception:  # pragma: no cover - optional dependency absent
            lines.append(f"{module_name:<21}: not installed")
    return "\n".join(lines)


def format_table(df: pd.DataFrame, max_rows: int = 15, float_format: str = "%.4f") -> str:
    """Render a DataFrame as fixed-width text for the plain-text reports."""
    with pd.option_context("display.max_rows", max_rows,
                           "display.width", 120,
                           "display.max_columns", 20):
        return df.head(max_rows).to_string(index=False, float_format=lambda v: float_format % v)
