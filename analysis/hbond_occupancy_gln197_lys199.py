"""
hbond_occupancy_gln197_lys199.py

For a single RUN, computes per-frame occupancy of specific H-bond pairs
across subunits (GLN197/GLN186 and LYS199/LYS188 groups).

Checks if hbond_per_frame.csv files exist before running.
Outputs per col_type (sc_any, sc_sc, bb_sc, bb_bb):
  - 3 detailed CSVs (one per table: GLN197_ALA, GLN197_GLU, LYS199_GLU)
  - 1 compact summary CSV (5 rows x 4 cols) including 194↔P5 and 194↔197 rows

Called from main.py.
"""

import os
import numpy as np
import pandas as pd


# ── Table definitions ─────────────────────────────────────────────────────────

TABLES = {
    "GLN197_ALA": [
        ("197_2a", "GLN197.2a", "THR305.G1"),
        ("197_2b", "GLN197.2b", "ALA316.2a"),
        ("197_2c", "GLN197.2c", "ALA316.2b"),
        ("186_G1", "GLN186.G1", "ALA316.2c"),
    ],
    "GLN197_GLU": [
        ("197_2a", "GLN197.2a", "GLU304.G1"),
        ("197_2b", "GLN197.2b", "GLU315.2a"),
        ("197_2c", "GLN197.2c", "GLU315.2b"),
        ("186_G1", "GLN186.G1", "GLU315.2c"),
    ],
    "LYS199_GLU": [
        ("199_2a", "LYS199.2a", "GLU304.G1"),
        ("199_2b", "LYS199.2b", "GLU315.2a"),
        ("199_2c", "LYS199.2c", "GLU315.2b"),
        ("188_G1", "LYS188.G1", "GLU315.2c"),
    ],
}

# 194↔197 hbond: folder -> partner label to look for
LYS194_GLN197_PAIRS = [
    ("197_2a", "LYS194.2a"),
    ("197_2b", "LYS194.2b"),
    ("197_2c", "LYS194.2c"),
    ("186_G1", "LYS183.G1"),
]

# 199↔197 hbond: folder -> partner label to look for
LYS199_GLN197_PAIRS = [
    ("197_2a", "LYS199.2a"),
    ("197_2b", "LYS199.2b"),
    ("197_2c", "LYS199.2c"),
    ("186_G1", "LYS188.G1"),
]

# 194↔P5 distance: PIP resid -> pdb_label to look for in proximity_summary.csv
# column order matches SUMMARY_COLS: 2a↔G1, 2b↔2a, 2c↔2b, G1↔2c
LYS194_P5_PAIRS = [
    (1304, "2a", "LYS194.2a"),
    (1301, "2b", "LYS194.2b"),
    (1303, "2c", "LYS194.2c"),
    (1302, "G1", "LYS183.G1"),
]

# Row and column labels for the compact summary
SUMMARY_ROWS = ["197↔316/305", "197↔315/304", "199↔315/304", "194↔197", "199↔197", "194↔P5", "194↔P5_min", "194↔P5_max"]
SUMMARY_COLS = ["2a↔G1", "2b↔2a", "2c↔2b", "G1↔2c"]

COLUMNS = ["sc_any_partners", "sc_sc_partners", "bb_sc_partners", "bb_bb_partners"]

REQUIRED_FOLDERS = ["197_2a", "197_2b", "197_2c", "186_G1",
                    "199_2a", "199_2b", "199_2c", "188_G1"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hbond_csv_path(results_dir, folder_suffix):
    return os.path.join(
        results_dir, "residue_neighbors", folder_suffix, "hbonds", "hbond_per_frame.csv"
    )


def _check_files(results_dir):
    missing = []
    for suffix in REQUIRED_FOLDERS:
        p = _hbond_csv_path(results_dir, suffix)
        if not os.path.isfile(p):
            missing.append(p)
    return missing


def _partner_occupancy(df, col, partner_label):
    """Returns (occupancy, raw_frame_count) for partner_label in col."""
    if col not in df.columns:
        return np.nan, np.nan
    total = len(df)
    if total == 0:
        return np.nan, np.nan
    count = int(df[col].fillna("").apply(
        lambda cell: partner_label in cell.split("|")
    ).sum())
    return round(float(count) / total, 4), count


def _get_p5_stats(results_dir, pip_resid, subunit, pdb_label):
    """Read mean±std, min, max of distance for pdb_label from distances_all_pdb.csv in P5 folder.
    Returns (mean_std_str, min_val, max_val)."""
    path = os.path.join(
        results_dir, "pip_proximity_5",
        f"PIP_{pip_resid}_{subunit}", "P5", "distances_all_pdb.csv"
    )
    if not os.path.isfile(path):
        print(f"    [194↔P5] Missing: {path}")
        return "N/A", np.nan, np.nan
    df = pd.read_csv(path)
    if pdb_label not in df.columns:
        print(f"    [194↔P5] {pdb_label} not found as column in {path}")
        return "N/A", np.nan, np.nan
    vals = df[pdb_label].dropna()
    mean = round(float(vals.mean()), 4)
    std  = round(float(vals.std()),  4)
    vmin = round(float(vals.min()),  4)
    vmax = round(float(vals.max()),  4)
    return f"{mean} ± {std}", vmin, vmax


# ── Main entry point ──────────────────────────────────────────────────────────

def run_hbond_occupancy_gln197_lys199(results_dir):
    print("\n" + "=" * 60)
    print("H-BOND OCCUPANCY COMPARISON (per subunit)")
    print("=" * 60)

    # ── Check required files ──────────────────────────────────────────────────
    missing = _check_files(results_dir)
    if missing:
        print("\n[hbond_occupancy_gln197_lys199] Missing required files:")
        for m in missing:
            print(f"  {m}")
        print("\n  → Run Neighboring Residues analysis first.")
        return

    # ── Load all hbond CSVs once ──────────────────────────────────────────────
    dfs = {}
    for suffix in REQUIRED_FOLDERS:
        path = _hbond_csv_path(results_dir, suffix)
        dfs[suffix] = pd.read_csv(path)
        print(f"  Loaded: {path}")

    # ── Compute 194↔P5 distances (same for all col_types) ────────────────────
    p5_row     = []
    p5_min_row = []
    p5_max_row = []
    for (pip_resid, subunit, pdb_label) in LYS194_P5_PAIRS:
        mean_std, vmin, vmax = _get_p5_stats(results_dir, pip_resid, subunit, pdb_label)
        p5_row.append(mean_std)
        p5_min_row.append(vmin)
        p5_max_row.append(vmax)
    print(f"  194↔P5 stats: {p5_row}")

    # ── Output dir ────────────────────────────────────────────────────────────
    out_base = os.path.join(results_dir, "hbond_occupancy_gln197_lys199")
    os.makedirs(out_base, exist_ok=True)

    # ── One CSV per table per col_type + compact summary ─────────────────────
    for col in COLUMNS:
        col_name = col.replace("_partners", "")
        out_dir = os.path.join(out_base, col_name)
        os.makedirs(out_dir, exist_ok=True)

        summary_occ = {}

        for table_name, pairs in TABLES.items():
            bond_labels = [f"{src}↔{prt}" for (_, src, prt) in pairs]
            occ_vals, raw_vals, tot_vals = [], [], []

            for (suffix, src, prt) in pairs:
                occ, raw = _partner_occupancy(dfs[suffix], col, prt)
                occ_vals.append(occ)
                raw_vals.append(raw)
                tot_vals.append(len(dfs[suffix]))

            summary_occ[table_name] = occ_vals

            # Detailed CSV
            df_out = pd.DataFrame(
                [occ_vals, raw_vals, tot_vals],
                columns=bond_labels,
                index=["occupancy", "n_frames_with_bond", "n_frames_total"]
            )
            csv_path = os.path.join(out_dir, f"{table_name}.csv")
            df_out.to_csv(csv_path)
            print(f"  Saved [{col_name}/{table_name}]: {csv_path}")
            print(df_out.to_string())
            print()

        # ── 194↔197 hbond occupancy ───────────────────────────────────────────
        lys194_row = []
        for (suffix, partner) in LYS194_GLN197_PAIRS:
            occ, _ = _partner_occupancy(dfs[suffix], col, partner)
            lys194_row.append(occ)

        # ── 199↔197 hbond occupancy ───────────────────────────────────────────
        lys199_row = []
        for (suffix, partner) in LYS199_GLN197_PAIRS:
            occ, _ = _partner_occupancy(dfs[suffix], col, partner)
            lys199_row.append(occ)

        # ── Compact summary CSV ───────────────────────────────────────────────
        df_summary = pd.DataFrame(
            [summary_occ["GLN197_ALA"],
             summary_occ["GLN197_GLU"],
             summary_occ["LYS199_GLU"],
             lys194_row,
             lys199_row,
             p5_row,
             p5_min_row,
             p5_max_row],
            columns=SUMMARY_COLS,
            index=SUMMARY_ROWS,
        )
        summary_path = os.path.join(out_dir, "summary.csv")
        df_summary.to_csv(summary_path)
        print(f"  Saved [{col_name}/summary]:\n{df_summary.to_string()}\n")

    print("\n  Done. Results in:", out_base)
    print("=" * 60)