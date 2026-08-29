"""
build_sp500_universe.py
=======================

Regenerates the static S&P 500 constituent file used by the large-network
extension:

    data/reference/sp500_constituents_2025_12_22.csv

Why a static file?
------------------
The extension must be reproducible from the repository alone.  Scraping
Wikipedia on every run would make the universe silently drift: a rerun in six
months would analyse a different set of companies and would no longer
reproduce the numbers reported in the thesis.  The constituent list is
therefore resolved **once**, written to a CSV that is committed to the
repository, and read from disk by every later run.  This script exists so that
the CSV is not a hand-typed artefact: it documents, and can re-execute, the
exact derivation.

Data source
-----------
Everything comes from the public repository

    https://github.com/fja05680/sp500

which is the standard open dataset for point-in-time S&P 500 membership.  Two
files inside it are combined:

``S&P 500 Historical Components & Changes (Updated).csv``
    One row per date on which index membership changed, with the full list of
    constituent tickers on that date.  This gives a **point-in-time snapshot**:
    the script takes the last row whose date is on or before the cut-off
    (2025-12-31, the last day of the thesis sample), i.e. the index as it
    actually stood at the end of the sample period.

    Note how this must be described.  The universe is selected on end-of-sample
    membership and then analysed retrospectively over 2019-2025, so results are
    conditional on end-of-period index membership and are NOT a
    survivorship-bias-free representation of the S&P 500 across the window.

``sp500.csv``
    The Wikipedia "List of S&P 500 companies" table (ticker, company name,
    GICS sector, GICS sub-industry), as captured by that repository.  Because
    it is a *current* snapshot, it is read at dated commits rather than at HEAD.
    The primary source is the last capture dated **before** the end of the
    thesis sample, so the classification stays inside the sample period; the
    first capture **after** the sample end is consulted only for constituents
    the earlier file does not contain.  The provenance record reports how many
    stocks came from each.

Membership and classification are therefore both sourced and dated.  Nothing
in this file is invented: a constituent for which no GICS record can be found
in the dataset is reported and excluded rather than assigned a guessed sector.

Usage
-----
    git clone https://github.com/fja05680/sp500 /tmp/sp500
    python tools/build_sp500_universe.py --repo /tmp/sp500

Run with ``--check`` to verify the committed CSV still matches what the source
produces, without rewriting it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(BASE_DIR, "data", "reference")

# Last day of the thesis sample; the membership snapshot is the last one on or
# before this date.
CUTOFF_DATE = "2025-12-31"

HISTORICAL_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
METADATA_FILE = "sp500.csv"

# Commits of fja05680/sp500 from which the GICS metadata table is read, in
# order of preference.
#
# The PRIMARY source is the last metadata snapshot dated *before* the end of the
# thesis sample.  Using a pre-end snapshot keeps the classification strictly
# within the sample period rather than importing a classification that only came
# into force afterwards.
#
# The FALLBACK is the first snapshot *after* the sample end, consulted only for
# a constituent that is present in the 2025-12-22 membership snapshot but absent
# from the earlier metadata file - typically a company added to the index
# between the metadata snapshot and the membership snapshot.  Nothing is ever
# guessed: a constituent that appears in neither file is excluded and reported.
METADATA_COMMITS = [
    ("58490ec8ea42827b6df808ff1e68dff51144f09f", "2025-11-16", "primary (before sample end)"),
    ("c24b12e2b725cb4d18ab6b816fbeb6226280aae6", "2026-01-17", "fallback (after sample end)"),
]

# Yahoo Finance writes share-class suffixes with a hyphen where GICS/Wikipedia
# use a dot (BRK.B -> BRK-B).  The canonical GICS symbol stays in ``ticker``;
# ``yahoo_ticker`` is what the download stage asks Yahoo for.
def to_yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def _git_show(repo: str, ref: str, path: str) -> str:
    """Return the contents of ``path`` at ``ref`` in the clone at ``repo``."""
    result = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Could not read '{path}' at {ref[:12]} from {repo}.\n"
            f"git said: {result.stderr.strip()}\n"
            "The clone may be shallow; deepen it with\n"
            f"    git -C {repo} fetch --depth=1000 origin master"
        )
    return result.stdout


def membership_snapshot(repo: str, cutoff: str) -> tuple[list[str], str]:
    """Point-in-time constituent list on the last change date <= ``cutoff``."""
    path = os.path.join(repo, HISTORICAL_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"{HISTORICAL_FILE} not found in {repo}")
    history = pd.read_csv(path)
    history["date"] = pd.to_datetime(history["date"])
    eligible = history[history["date"] <= pd.Timestamp(cutoff)]
    if eligible.empty:
        raise SystemExit(f"No membership snapshot on or before {cutoff}.")
    row = eligible.iloc[-1]
    tickers = sorted({t.strip() for t in str(row["tickers"]).split(",") if t.strip()})
    return tickers, str(row["date"].date())


def gics_metadata(repo: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """GICS table assembled from the dated commits in ``METADATA_COMMITS``.

    Later commits never overwrite an earlier one: the first commit in the list
    is authoritative, the rest only fill in symbols it does not contain.  Since
    the list is ordered primary-first, this means the pre-sample-end snapshot
    supplies every classification it can and the post-end snapshot is consulted
    only for what is genuinely missing from it.

    Returns the combined table, a per-ticker human-readable source string, and a
    per-ticker role (``primary`` / ``fallback``) so that the two can be counted.
    """
    frames = []
    origin: dict[str, str] = {}
    role: dict[str, str] = {}
    for commit, date, label in METADATA_COMMITS:
        blob = _git_show(repo, commit, METADATA_FILE)
        frame = pd.read_csv(io.StringIO(blob))
        frame = frame[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
        frame = frame.rename(columns={
            "Symbol": "ticker", "Security": "company",
            "GICS Sector": "gics_sector", "GICS Sub-Industry": "gics_sub_industry"})
        for symbol in frame["ticker"]:
            if symbol not in origin:
                origin[symbol] = f"{METADATA_FILE}@{commit[:12]} ({date})"
                role[symbol] = label.split()[0]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="ticker", keep="first")
    return combined, origin, role


def build(repo: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    tickers, snapshot_date = membership_snapshot(repo, cutoff)
    metadata, origin, role = gics_metadata(repo)

    universe = pd.DataFrame({"ticker": tickers})
    universe = universe.merge(metadata, on="ticker", how="left")

    unresolved = sorted(universe.loc[universe["gics_sector"].isna(), "ticker"])
    universe = universe.dropna(subset=["gics_sector"]).reset_index(drop=True)

    universe["yahoo_ticker"] = universe["ticker"].map(to_yahoo_symbol)
    universe["metadata_source"] = universe["ticker"].map(origin)
    universe["metadata_snapshot_role"] = universe["ticker"].map(role)
    universe = universe[["ticker", "yahoo_ticker", "company", "gics_sector",
                         "gics_sub_industry", "metadata_source",
                         "metadata_snapshot_role"]]
    universe = universe.sort_values("ticker").reset_index(drop=True)

    from_primary = sorted(universe.loc[universe["metadata_snapshot_role"] == "primary",
                                       "ticker"])
    from_fallback = sorted(universe.loc[universe["metadata_snapshot_role"] == "fallback",
                                        "ticker"])

    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    provenance = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_repository": "https://github.com/fja05680/sp500",
        "source_repository_head": head,
        "membership_file": HISTORICAL_FILE,
        "membership_snapshot_date": snapshot_date,
        "membership_cutoff_requested": cutoff,
        "metadata_file": METADATA_FILE,
        "metadata_commits": [{"commit": c, "date": d, "role": r}
                             for c, d, r in METADATA_COMMITS],
        "n_constituents_in_snapshot": len(tickers),
        "n_with_gics_metadata": int(len(universe)),
        "n_from_primary_metadata": len(from_primary),
        "n_from_fallback_metadata": len(from_fallback),
        "fallback_metadata_tickers": from_fallback,
        "n_unresolved_excluded": len(unresolved),
        "unresolved_excluded": unresolved,
        "n_sectors": int(universe["gics_sector"].nunique()),
        "sectors": sorted(universe["gics_sector"].unique()),
    }
    return universe, provenance


def output_paths(snapshot_date: str) -> tuple[str, str]:
    tag = snapshot_date.replace("-", "_")
    return (os.path.join(REFERENCE_DIR, f"sp500_constituents_{tag}.csv"),
            os.path.join(REFERENCE_DIR, f"sp500_constituents_{tag}_provenance.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--repo", required=True,
                        help="Path to a clone of https://github.com/fja05680/sp500")
    parser.add_argument("--cutoff", default=CUTOFF_DATE,
                        help="Membership snapshot is the last one on or before this date.")
    parser.add_argument("--check", action="store_true",
                        help="Compare with the committed CSV instead of rewriting it.")
    args = parser.parse_args(argv)

    universe, provenance = build(args.repo, args.cutoff)
    csv_path, json_path = output_paths(provenance["membership_snapshot_date"])

    if args.check:
        if not os.path.exists(csv_path):
            print(f"MISSING: {csv_path}")
            return 1
        committed = pd.read_csv(csv_path)
        same = committed.equals(universe)
        print(("MATCH" if same else "DIFFERS") + f": {os.path.relpath(csv_path, BASE_DIR)}")
        return 0 if same else 1

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    universe.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print(f"Membership snapshot : {provenance['membership_snapshot_date']} "
          f"({provenance['n_constituents_in_snapshot']} constituents)")
    print(f"With GICS metadata  : {provenance['n_with_gics_metadata']}")
    print(f"  from primary snapshot ({METADATA_COMMITS[0][1]}, before sample end) : "
          f"{provenance['n_from_primary_metadata']}")
    print(f"  from fallback snapshot ({METADATA_COMMITS[1][1]}, after sample end)  : "
          f"{provenance['n_from_fallback_metadata']} "
          f"{provenance['fallback_metadata_tickers']}")
    print(f"Excluded (no GICS)  : {provenance['n_unresolved_excluded']} "
          f"{provenance['unresolved_excluded']}")
    print(f"Sectors             : {provenance['n_sectors']}")
    print(f"-> {os.path.relpath(csv_path, BASE_DIR)}")
    print(f"-> {os.path.relpath(json_path, BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
