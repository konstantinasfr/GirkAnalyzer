"""
analysis/pip_gate_correlation.py
==================================
Computes Spearman correlation between PIP2 residue distances (from pip_proximity_X)
and HBC gate distances (A_C and B_D) from hbc_distances.csv.

For every PIP, every atom (P1/P4/P5):
  - Reads proximity_summary.csv to find residues ever < 4A (close residues)
  - Reads distances_all_pdb.csv for the actual per-frame distances
  - Correlates each residue distance vs A_C and B_D gate distances

Mandatory residues per atom (PDB numbering):
  P1: 90, 91, 92
  P4: 60, 64, 199, 200, 194, 90
  P5: 60, 64, 199, 200, 194, 90

Outputs (saved in results_dir/pip_gate_correlation/):
    correlation_all.csv                  -- all residues, sorted by max |r|
    correlation_filtered_r{min_r}.csv    -- filtered by |r| >= min_r

    correlation_filtered_r{min_r}.xlsx        -- all residues, 3 sort sheets
    correlation_close_r{min_r}.xlsx           -- ever < 4A residues only
    correlation_mandatory_r{min_r}.xlsx       -- mandatory residues only
    correlation_combined_r{min_r}.xlsx        -- union of close + mandatory

Called from main.py:

    from analysis.pip_gate_correlation import run_pip_gate_correlation

    run_pip_gate_correlation(
        results_dir      = results_dir,
        prescreen_factor = 5.0,
        min_r            = 0.3,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Mandatory residues per atom (PDB numbering) -- same as pip_proximity_analysis
# -----------------------------------------------------------------------------
MANDATORY_RESIDUES = {
    "P1": [90, 91, 92],
    "P4": [60, 64, 199, 200, 194, 90],
    "P5": [60, 64, 199, 200, 194, 90],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_pip_proximity_dir(results_dir: Path, prescreen_factor: float) -> Optional[Path]:
    factor_str = (
        str(int(prescreen_factor))
        if prescreen_factor == int(prescreen_factor)
        else str(prescreen_factor)
    )
    folder = results_dir / f"pip_proximity_{factor_str}"
    return folder if folder.exists() else None


def _find_gate_csvs(results_dir: Path) -> dict:
    """Find all gate distance CSVs. Returns dict of {name: path}."""
    gates = {}
    for name, filename in [
        ("hbc",      "hbc_distances.csv"),
        ("upper_gl", "upper_gl_distances.csv"),
        ("lower_gl", "lower_gl_distances.csv"),
    ]:
        p = results_dir / filename
        if p.exists():
            gates[name] = p
        else:
            print(f"  INFO: {filename} not found -- skipping {name} correlations.")
    return gates


def _extract_subunit(pip_folder_name: str) -> Optional[str]:
    """e.g. 'PIP_1301_2b' -> '2b', 'PIP_1302_G1' -> 'G1'"""
    parts = pip_folder_name.split("_")
    return parts[-1] if len(parts) >= 3 else None


def _get_close_residues(proximity_csv: Path) -> Set[str]:
    """Return set of pdb_labels that had frames_close > 0."""
    if not proximity_csv.exists():
        return set()
    df = pd.read_csv(proximity_csv)
    if "pdb_label" not in df.columns:
        return set()
    return set(df["pdb_label"].astype(str).tolist())


def _get_mandatory_cols(all_cols: List[str], atom_name: str,
                        subunit: Optional[str]) -> tuple[List[str], List[str]]:
    """
    Return (found_cols, missing_nums) for mandatory residues of this atom.
    Matches column names ending in '{num}.{subunit}' (or just '{num}' if no subunit).
    """
    mandatory_nums = MANDATORY_RESIDUES.get(atom_name, [])
    found_cols     = []
    missing_nums   = []
    for num in mandatory_nums:
        if subunit:
            matches = [c for c in all_cols if c.endswith(f"{num}.{subunit}")]
        else:
            matches = [c for c in all_cols if c.endswith(str(num))]
        if matches:
            found_cols.extend(matches)
        else:
            missing_nums.append(str(num))
    return found_cols, missing_nums


def _write_excel_3sheets(df: pd.DataFrame, path: Path, clean_cols: List[str]):
    """Write DataFrame to Excel with 3 sort sheets."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df[clean_cols].to_excel(
            writer, sheet_name="sorted_max_r", index=False)
        df.sort_values("abs_r_AC", ascending=False)[clean_cols].to_excel(
            writer, sheet_name="sorted_r_AC", index=False)
        df.sort_values("abs_r_BD", ascending=False)[clean_cols].to_excel(
            writer, sheet_name="sorted_r_BD", index=False)


# -----------------------------------------------------------------------------
# Core correlation
# -----------------------------------------------------------------------------

def _correlate_pip_with_gate(
    pip_proximity_dir: Path,
    hbc_df:            pd.DataFrame,
) -> pd.DataFrame:
    """
    For each PIP/atom combination:
      1. Load proximity_summary.csv -> close residues (ever < 4A)
      2. Load distances_all_pdb.csv -> per-frame distances
      3. Compute Spearman r vs A_C and B_D for every residue
      4. Tag each row with: is_close, is_mandatory
    """
    rows = []

    pip_folders = sorted(pip_proximity_dir.glob("PIP_*"))
    if not pip_folders:
        print(f"  WARNING: no PIP_* folders found in {pip_proximity_dir}")
        return pd.DataFrame()

    for pip_folder in pip_folders:
        pip_label = pip_folder.name
        subunit   = _extract_subunit(pip_label)

        for atom_name in ("P1", "P4", "P5"):
            atom_folder = pip_folder / atom_name
            if not atom_folder.is_dir():
                continue

            dist_csv    = atom_folder / "distances_all_pdb.csv"
            prox_csv    = atom_folder / "proximity_summary.csv"

            if not dist_csv.exists():
                print(f"  WARNING: {dist_csv} not found -- skipping.")
                continue

            # load distances
            pip_df = pd.read_csv(dist_csv)

            # align frames with hbc
            merged = pd.merge(pip_df, hbc_df[["frame", "A_C", "B_D"]],
                              on="frame", how="inner")
            if len(merged) < 10:
                print(f"  WARNING: too few overlapping frames for "
                      f"{pip_label}/{atom_name} -- skipping.")
                continue

            all_cols = [c for c in pip_df.columns if c != "frame"]

            # close residues from proximity_summary.csv
            close_set = _get_close_residues(prox_csv)

            # mandatory cols for this atom/subunit
            mandatory_cols, missing = _get_mandatory_cols(all_cols, atom_name, subunit)
            if missing:
                print(f"  INFO: mandatory residues not in data for "
                      f"{pip_label}/{atom_name}: {missing}")
            mandatory_set = set(mandatory_cols)

            # correlate all residues
            for col in all_cols:
                vals = merged[col].values
                ac   = merged["A_C"].values
                bd   = merged["B_D"].values

                mask_ac = ~(np.isnan(vals) | np.isnan(ac))
                mask_bd = ~(np.isnan(vals) | np.isnan(bd))

                if mask_ac.sum() < 10 or mask_bd.sum() < 10:
                    continue

                r_ac, p_ac = spearmanr(vals[mask_ac], ac[mask_ac])
                r_bd, p_bd = spearmanr(vals[mask_bd], bd[mask_bd])

                rows.append({
                    "pip":         pip_label,
                    "atom":        atom_name,
                    "residue":     col,
                    "r_AC":        round(float(r_ac), 4),
                    "p_AC":        float(p_ac),
                    "r_BD":        round(float(r_bd), 4),
                    "p_BD":        float(p_bd),
                    "abs_r_AC":    round(abs(float(r_ac)), 4),
                    "abs_r_BD":    round(abs(float(r_bd)), 4),
                    "max_abs_r":   round(max(abs(float(r_ac)), abs(float(r_bd))), 4),
                    "is_close":    col in close_set,
                    "is_mandatory": col in mandatory_set,
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_pip_gate_correlation(
    results_dir:      Path,
    prescreen_factor: float = 5.0,
    min_r:            float = 0.3,
):
    results_dir = Path(results_dir)
    out_dir     = results_dir / "pip_gate_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PIP-GATE CORRELATION ANALYSIS")
    print(f"  results_dir      : {results_dir}")
    print(f"  prescreen_factor : {prescreen_factor}")
    print(f"  min_r threshold  : {min_r}")
    print("=" * 70)

    # --- check pip proximity folder ---
    pip_dir = _find_pip_proximity_dir(results_dir, prescreen_factor)
    if pip_dir is None:
        factor_str = (
            str(int(prescreen_factor))
            if prescreen_factor == int(prescreen_factor)
            else str(prescreen_factor)
        )
        print(f"\n  ERROR: pip_proximity_{factor_str}/ not found in {results_dir}")
        print("  Please run run_pip_proximity_analysis first.")
        return

    print(f"  PIP proximity : {pip_dir}")

    # --- find all gate CSVs ---
    gate_csvs = _find_gate_csvs(results_dir)
    if not gate_csvs:
        print("\n  ERROR: no gate distance CSVs found.")
        return

    # --- run correlations for each gate ---
    for gate_name, gate_path in gate_csvs.items():
        print(f"\n{'='*50}")
        print(f"  Gate: {gate_name.upper()} ({gate_path.name})")
        print(f"{'='*50}")

        gate_df = pd.read_csv(gate_path)
        if not {"frame", "A_C", "B_D"}.issubset(gate_df.columns):
            print(f"  ERROR: {gate_path.name} must have columns: frame, A_C, B_D")
            print(f"  Found: {list(gate_df.columns)}")
            continue

        print(f"  Frames: {len(gate_df)}")

        # subfolder per gate
        gate_out_dir = out_dir / gate_name
        gate_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Computing Spearman correlations...")
        corr_df = _correlate_pip_with_gate(pip_dir, gate_df)

        if corr_df.empty:
            print(f"  No correlations computed for {gate_name}.")
            continue

        corr_df = corr_df.sort_values("max_abs_r", ascending=False).reset_index(drop=True)
        clean_cols = ["pip", "atom", "residue", "r_AC", "p_AC", "r_BD", "p_BD"]

        # save CSVs
        corr_df[clean_cols].to_csv(gate_out_dir / "correlation_all.csv", index=False)
        print(f"  Saved all correlations ({len(corr_df)} rows)")

        filtered_df = corr_df[corr_df["max_abs_r"] >= min_r].copy()
        filtered_df[clean_cols].to_csv(
            gate_out_dir / f"correlation_filtered_r{min_r}.csv", index=False)
        print(f"  Saved filtered CSV ({len(filtered_df)} rows)")

        # save Excels
        _write_excel_3sheets(filtered_df, 
            gate_out_dir / f"correlation_filtered_r{min_r}.xlsx", clean_cols)
        _write_excel_3sheets(filtered_df[filtered_df["is_close"]].copy(),
            gate_out_dir / f"correlation_close_r{min_r}.xlsx", clean_cols)
        _write_excel_3sheets(filtered_df[filtered_df["is_mandatory"]].copy(),
            gate_out_dir / f"correlation_mandatory_r{min_r}.xlsx", clean_cols)
        _write_excel_3sheets(
            filtered_df[filtered_df["is_close"] | filtered_df["is_mandatory"]].copy(),
            gate_out_dir / f"correlation_combined_r{min_r}.xlsx", clean_cols)
        print(f"  Saved 4 Excel files -> {gate_out_dir}")

        # terminal summary
        print(f"\n  Top 10 by max |r| ({gate_name}):")
        print(f"  {'PIP':<20} {'atom':<6} {'residue':<20} {'r_AC':>8} {'r_BD':>8}")
        print(f"  {'-'*65}")
        for _, row in corr_df.head(10).iterrows():
            print(f"  {row['pip']:<20} {row['atom']:<6} {row['residue']:<20} "
                  f"{row['r_AC']:>8.3f} {row['r_BD']:>8.3f}")

    print(f"\n✓ PIP-gate correlation analysis complete.")
    print(f"  Results in: {out_dir}")