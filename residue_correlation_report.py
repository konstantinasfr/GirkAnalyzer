"""
analysis/residue_correlation_report.py
========================================
Summarizes PIP2-gate Spearman correlations for a specific residue number
(e.g. 197) across all RUN* folders, for HBC, upper G-loop and lower G-loop gates.

For each RUN and each gate:
  - Reads pip_gate_correlation/<gate>/correlation_all.csv
  - Filters rows where atom == "P5" AND residue contains the target number
    (catches 197.2a, 197.2b, 197.2c, 197.G1 etc.)
  - Finds the row with highest max(|r_AC|, |r_BD|) -> always included
  - Also includes any other rows where max(|r_AC|, |r_BD|) >= secondary_threshold

Output (saved in base_path/residue_correlation_report/<residue_query>/):
    report_<resnum>_<gate>.csv                    -- all selected rows (any subunit)
    report_<resnum>_<gate>_highest_only.csv        -- one row per RUN, global highest
    report_<resnum>_<gate>_matching_subunit.csv    -- best subunit-matching combo per RUN
                                                       (computed independently, see below)

IMPORTANT FIX (see notes):
    The matching_subunit report is now computed INDEPENDENTLY per RUN, directly
    from the raw residue/PIP table -- NOT by filtering the already-restricted
    "highest or above threshold" table. Previously, a RUN whose subunit-matching
    combo wasn't also the single global highest (across all 4 subunits) and
    wasn't above secondary_threshold would silently vanish from the
    matching_subunit file, even if a valid matching combo existed. Now each RUN
    is evaluated on its own: restrict to rows where residue-subunit == PIP-subunit
    FIRST, then pick the highest within that matching subset (+ anything above
    threshold). A RUN is only absent from this file if it has ZERO subunit-matching
    combos at all.

Standalone script -- run directly:

    python3 residue_correlation_report.py \\
        --base_path /path/to/GIRK12_WT/ \\
        --residue_number 197 \\
        --secondary_threshold 0.4
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


GATES = ["hbc", "upper_gl", "lower_gl"]


def _subunit_of_residue(res_label: str) -> str:
    """Extract subunit from residue label e.g. GLN197.2b -> 2b"""
    parts = str(res_label).strip().split(".")
    return parts[-1].strip() if len(parts) > 1 else ""


def _subunit_of_pip(pip_label: str) -> str:
    """Extract subunit from PIP label e.g. PIP_1301_2b -> 2b"""
    parts = str(pip_label).strip().split("_")
    return parts[-1].strip() if len(parts) >= 3 else ""


def _load_residue_rows(
    run_dir:        Path,
    gate:           str,
    residue_query:  str,
) -> Optional[pd.DataFrame]:
    """
    Load and filter correlation_all.csv for one RUN+gate down to the
    P5 / target-residue rows. Returns None if the file is missing or
    no matching rows exist.
    """
    csv_path = run_dir / "pip_gate_correlation" / gate / "correlation_all.csv"

    if not csv_path.exists():
        print(f"    WARNING: {csv_path} not found -- skipping.")
        return None

    df = pd.read_csv(csv_path)

    specific_subunit = "." in str(residue_query)
    residue_number   = residue_query.split(".")[0]

    if specific_subunit:
        mask = (
            (df["atom"] == "P5") &
            (df["residue"].astype(str).str.endswith(residue_query))
        )
    else:
        mask = (
            (df["atom"] == "P5") &
            (df["residue"].astype(str).str.contains(f"\\.?{residue_number}\\.", regex=True) |
             df["residue"].astype(str).str.endswith(str(residue_number)))
        )
    sub = df[mask].copy()

    if sub.empty:
        print(f"    INFO: no P5 rows matching '{residue_query}' in {run_dir.name}/{gate}")
        return None

    return sub


def _process_run_gate(
    run_dir:             Path,
    gate:                str,
    residue_query:       str,
    secondary_threshold: float,
) -> List[dict]:
    """
    GLOBAL selection (any subunit combo): highest overall + anything above
    threshold. Used for report_<res>_<gate>.csv and the highest_only file.
    """
    sub = _load_residue_rows(run_dir, gate, residue_query)
    if sub is None:
        return []

    sub = sub.copy()
    sub["max_abs_r"] = sub.apply(
        lambda r: max(abs(r["r_AC"]), abs(r["r_BD"])), axis=1
    )

    rows = []
    best_idx = sub["max_abs_r"].idxmax()

    for idx, row in sub.iterrows():
        is_highest = idx == best_idx
        above_threshold = row["max_abs_r"] >= secondary_threshold

        if is_highest or above_threshold:
            rows.append({
                "run":         run_dir.name,
                "gate":        gate,
                "pip":         row["pip"],
                "atom":        row["atom"],
                "residue":     row["residue"],
                "r_AC":        row["r_AC"],
                "p_AC":        row["p_AC"],
                "r_BD":        row["r_BD"],
                "p_BD":        row["p_BD"],
                "max_abs_r":   round(float(row["max_abs_r"]), 4),
                "is_highest":  is_highest,
                "above_threshold": above_threshold,
            })

    return rows


def _process_run_gate_matching(
    run_dir:             Path,
    gate:                str,
    residue_query:       str,
    secondary_threshold: float,
) -> List[dict]:
    """
    INDEPENDENT matching-subunit selection: restrict to rows where
    residue-subunit == PIP-subunit FIRST, then pick the highest within
    that matching subset (+ anything above threshold). Does NOT depend
    on the global (all-subunit) selection above.
    """
    sub = _load_residue_rows(run_dir, gate, residue_query)
    if sub is None:
        return []

    sub = sub.copy()
    sub["res_subunit"] = sub["residue"].apply(_subunit_of_residue)
    sub["pip_subunit"] = sub["pip"].apply(_subunit_of_pip)
    sub_match = sub[sub["res_subunit"] == sub["pip_subunit"]].copy()

    if sub_match.empty:
        return []  # genuinely no subunit-matching combo for this RUN

    sub_match["max_abs_r"] = sub_match.apply(
        lambda r: max(abs(r["r_AC"]), abs(r["r_BD"])), axis=1
    )

    best_idx = sub_match["max_abs_r"].idxmax()
    rows = []
    for idx, row in sub_match.iterrows():
        is_highest = idx == best_idx
        above_threshold = row["max_abs_r"] >= secondary_threshold
        if is_highest or above_threshold:
            rows.append({
                "run":         run_dir.name,
                "gate":        gate,
                "pip":         row["pip"],
                "atom":        row["atom"],
                "residue":     row["residue"],
                "r_AC":        row["r_AC"],
                "p_AC":        row["p_AC"],
                "r_BD":        row["r_BD"],
                "p_BD":        row["p_BD"],
                "max_abs_r":   round(float(row["max_abs_r"]), 4),
                "is_highest":  is_highest,
            })
    return rows


def run_residue_correlation_report(
    base_path:            Path,
    residue_query:        str,    # e.g. "197" (all subunits) or "197.G1" (specific)
    secondary_threshold:  float = 0.4,
):
    base_path        = Path(base_path)
    specific_subunit = "." in str(residue_query)
    residue_number   = residue_query.split(".")[0]
    out_dir          = base_path / "residue_correlation_report" / str(residue_query).replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("RESIDUE CORRELATION REPORT")
    print(f"  base_path           : {base_path}")
    print(f"  residue_query       : {residue_query} ({'specific subunit' if specific_subunit else 'all subunits'})")
    print(f"  secondary_threshold : {secondary_threshold}")
    print(f"  gates               : {GATES}")
    print(f"  atom filter         : P5 only")
    print("=" * 70)

    # find all RUN<digits> folders sorted (excludes things like "run_variability")
    run_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and re.match(r"^RUN\d+$", d.name.upper())],
        key=lambda d: int(re.match(r"^RUN(\d+)$", d.name.upper()).group(1))
    )

    if not run_dirs:
        print(f"\n  ERROR: no RUN<digits> folders found in {base_path}")
        return

    print(f"\n  Found {len(run_dirs)} RUN folders: {[d.name for d in run_dirs]}")

    for gate in GATES:
        print(f"\n  Processing gate: {gate.upper()}")
        all_rows = []
        all_matching_rows = []

        for run_dir in run_dirs:
            rows = _process_run_gate(
                run_dir              = run_dir,
                gate                 = gate,
                residue_query        = residue_query,
                secondary_threshold  = secondary_threshold,
            )
            if rows:
                print(f"    {run_dir.name}: {len(rows)} rows included")
                for r in rows:
                    flag = "* highest" if r["is_highest"] else f"  >={secondary_threshold}"
                    print(f"      {flag}  {r['residue']:<20} "
                          f"r_AC={r['r_AC']:>7.3f}  r_BD={r['r_BD']:>7.3f}  "
                          f"max|r|={r['max_abs_r']:.3f}")
                all_rows.extend(rows)
            else:
                print(f"    {run_dir.name}: no rows found")

            match_rows = _process_run_gate_matching(
                run_dir              = run_dir,
                gate                 = gate,
                residue_query        = residue_query,
                secondary_threshold  = secondary_threshold,
            )
            all_matching_rows.extend(match_rows)

        if all_rows:
            df_out = pd.DataFrame(all_rows)
            df_out = df_out.sort_values("max_abs_r", ascending=False)
            df_out = df_out.drop(columns=["gate", "max_abs_r", "above_threshold"])

            # CSV 1 -- highest + secondary threshold, sorted (any subunit combo)
            out_path = out_dir / f"report_{str(residue_query).replace('.', '_')}_{gate}.csv"
            df_out.to_csv(out_path, index=False)
            print(f"\n  Saved -> {out_path.name}  ({len(df_out)} rows)")

            # CSV 2 -- highest only (one row per RUN), sorted
            df_highest = df_out[df_out["is_highest"]].copy()
            out_path_highest = out_dir / f"report_{str(residue_query).replace('.', '_')}_{gate}_highest_only.csv"
            df_highest.to_csv(out_path_highest, index=False)
            print(f"  Saved -> {out_path_highest.name}  ({len(df_highest)} rows)")
        else:
            print(f"\n  No data found for {gate} -- skipping report/highest_only CSVs.")

        # CSV 3 -- matching subunit only, computed INDEPENDENTLY per RUN
        if all_matching_rows:
            df_matching = pd.DataFrame(all_matching_rows)
            df_matching = df_matching.sort_values("max_abs_r", ascending=False)
            df_matching = df_matching.drop(columns=["gate", "max_abs_r"])
        else:
            df_matching = pd.DataFrame(columns=[
                "run", "pip", "atom", "residue",
                "r_AC", "p_AC", "r_BD", "p_BD", "is_highest"
            ])

        out_path_matching = out_dir / f"report_{str(residue_query).replace('.', '_')}_{gate}_matching_subunit.csv"
        df_matching.to_csv(out_path_matching, index=False)
        print(f"  Saved -> {out_path_matching.name}  ({len(df_matching)} rows, matching subunit only)")

    print(f"\nResidue correlation report complete.")
    print(f"  Results in: {out_dir}")


# -----------------------------------------------------------------------------
# Standalone entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize PIP2-gate correlations for a specific residue across RUN* folders."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to folder containing RUN1, RUN2, ... subfolders",
    )
    parser.add_argument(
        "--residue_query",
        type=str,
        required=True,
        help="Residue to search for: '197' matches all subunits, '197.G1' matches only that subunit",
    )
    parser.add_argument(
        "--secondary_threshold",
        type=float,
        default=0.4,
        help="Min |r| to include non-highest correlations (default: 0.4)",
    )

    args = parser.parse_args()

    run_residue_correlation_report(
        base_path           = args.base_path,
        residue_query       = args.residue_query,
        secondary_threshold = args.secondary_threshold,
    )

'''
python3 residue_correlation_report.py \
    --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --residue_query 197

python3 residue_correlation_report.py \
    --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --residue_query 197
'''