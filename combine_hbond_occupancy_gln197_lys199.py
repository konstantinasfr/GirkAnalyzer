"""
combine_hbond_occupancy_gln197_lys199.py

Aggregates per-RUN hbond occupancy results for GLN197/GLN186 and LYS199/LYS188
across all RUN folders.

Produces per col_type (sc_any, sc_sc, bb_sc, bb_bb):
  - Table 1: GLN197 ↔ ALA316/THR305
  - Table 2: GLN197 ↔ GLU315/GLU304
  - Table 3: LYS199 ↔ GLU315/GLU304

Each table has:
  - One row per RUN
  - 4 columns (one per subunit)
  - mean ± std row
  - Pooled occupancy row (sum raw frames / sum total frames across all RUNs)

Usage:
    python3 combine_hbond_occupancy_gln197_lys199.py \
        --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
        --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_hbond_occupancy_gln197_lys199
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Table definitions ─────────────────────────────────────────────────────────
# Each table: { subunit_label: (source_label, partner_label) }

TABLES = {
    "GLN197_ALA": {
        "title":   "GLN197 / GLN186  ↔  ALA316 / THR305",
        "subunits": {
            "GLN197.2a↔THR305.G1": ("GLN197.2a", "THR305.G1"),
            "GLN197.2b↔ALA316.2a": ("GLN197.2b", "ALA316.2a"),
            "GLN197.2c↔ALA316.2b": ("GLN197.2c", "ALA316.2b"),
            "GLN186.G1↔ALA316.2c": ("GLN186.G1", "ALA316.2c"),
        },
    },
    "GLN197_GLU": {
        "title":   "GLN197 / GLN186  ↔  GLU315 / GLU304",
        "subunits": {
            "GLN197.2a↔GLU304.G1": ("GLN197.2a", "GLU304.G1"),
            "GLN197.2b↔GLU315.2a": ("GLN197.2b", "GLU315.2a"),
            "GLN197.2c↔GLU315.2b": ("GLN197.2c", "GLU315.2b"),
            "GLN186.G1↔GLU315.2c": ("GLN186.G1", "GLU315.2c"),
        },
    },
    "LYS199_GLU": {
        "title":   "LYS199 / LYS188  ↔  GLU315 / GLU304",
        "subunits": {
            "LYS199.2a↔GLU304.G1": ("LYS199.2a", "GLU304.G1"),
            "LYS199.2b↔GLU315.2a": ("LYS199.2b", "GLU315.2a"),
            "LYS199.2c↔GLU315.2b": ("LYS199.2c", "GLU315.2b"),
            "LYS188.G1↔GLU315.2c": ("LYS188.G1", "GLU315.2c"),
        },
    },
}

COLUMNS    = ["sc_any", "sc_sc", "bb_sc", "bb_bb"]
SUBUNITS   = list(list(TABLES.values())[0]["subunits"].keys())  # derived from table keys
COLORS     = ["#2c5f8a", "#c0392b", "#27ae60", "#8e44ad"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_run_dirs(base_dir):
    runs = []
    for name in sorted(os.listdir(base_dir)):
        if name.startswith("RUN") and os.path.isdir(os.path.join(base_dir, name)):
            runs.append((name, os.path.join(base_dir, name)))
    return runs


def load_occupancy_csvs(run_dir, col_type):
    """Load all 3 table CSVs for a RUN and col_type into a dict keyed by bond label."""
    result = {}
    for table_name in ["GLN197_ALA", "GLN197_GLU", "LYS199_GLU"]:
        path = os.path.join(
            run_dir, "hbond_occupancy_gln197_lys199", col_type, f"{table_name}.csv"
        )
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, index_col=0)
        # df has rows: occupancy, n_frames_with_bond, n_frames_total
        # columns: bond labels like "GLN197.2a↔THR305.G1"
        for bond_col in df.columns:
            result[bond_col] = {
                "occupancy":          float(df.loc["occupancy", bond_col]),
                "n_frames_with_bond": float(df.loc["n_frames_with_bond", bond_col]),
                "n_frames_total":     float(df.loc["n_frames_total", bond_col]),
            }
    return result if result else None


def lookup(bond_dict, source_label, partner_label):
    """Return (occupancy, n_frames_with_bond, n_frames_total) for a bond."""
    if bond_dict is None:
        return np.nan, np.nan, np.nan
    bond_key = f"{source_label}↔{partner_label}"
    entry = bond_dict.get(bond_key)
    if entry is None:
        return np.nan, np.nan, np.nan
    return entry["occupancy"], entry["n_frames_with_bond"], entry["n_frames_total"]


# ── Build table for one col_type + one table definition ──────────────────────

def build_table(run_dirs, col_type, table_def):
    subunits = table_def["subunits"]
    rows_occ   = []   # occupancy per RUN
    rows_raw   = []   # raw frames per RUN
    rows_total = []   # total frames per RUN

    run_labels = []
    for run_name, run_dir in run_dirs:
        df = load_occupancy_csvs(run_dir, col_type)
        if df is None:
            continue
        run_labels.append(run_name)
        occ_row   = {"RUN": run_name}
        raw_row   = {"RUN": run_name}
        total_row = {"RUN": run_name}
        for su, (src, prt) in subunits.items():
            occ, raw, tot = lookup(df, src, prt)
            occ_row[su]   = occ
            raw_row[su]   = raw
            total_row[su] = tot
        rows_occ.append(occ_row)
        rows_raw.append(raw_row)
        rows_total.append(total_row)

    if not rows_occ:
        return None, None, None, None

    df_occ   = pd.DataFrame(rows_occ).set_index("RUN")
    df_raw   = pd.DataFrame(rows_raw).set_index("RUN")
    df_total = pd.DataFrame(rows_total).set_index("RUN")

    cols = list(subunits.keys())

    # ── mean ± std ────────────────────────────────────────────────────────────
    means  = df_occ[cols].mean()
    stds   = df_occ[cols].std()
    mean_row = {su: f"{means[su]:.4f} ± {stds[su]:.4f}" for su in cols}

    # ── pooled occupancy (treat all RUNs as one continuous simulation) ────────
    pooled_raw   = df_raw[cols].sum()
    pooled_total = df_total[cols].sum()
    pooled_occ   = pooled_raw / pooled_total
    pooled_row   = {su: f"{pooled_occ[su]:.4f} ({int(pooled_raw[su])}/{int(pooled_total[su])})"
                    for su in cols}

    # Assemble full display table
    df_display = df_occ[cols].copy().astype(str)
    # format floats
    for su in cols:
        df_display[su] = df_occ[su].apply(
            lambda v: f"{v:.4f}" if not np.isnan(v) else "N/A"
        )
    df_display.loc["mean ± std"] = mean_row
    df_display.loc["pooled"]     = pooled_row

    return df_display, means, stds, pooled_occ


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(df_occ, means, stds, pooled_occ, table_def, col_type, out_path):
    """Grouped bar chart: one group per subunit, one bar per RUN + mean + pooled."""
    run_names = [r for r in df_occ.index if r not in ("mean ± std", "pooled")]
    n_runs    = len(run_names)
    subunits  = list(table_def["subunits"].keys())

    x      = np.arange(len(subunits))
    n_bars = n_runs + 2   # RUNs + mean + pooled
    width  = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))

    run_colors  = plt.cm.tab20(np.linspace(0, 0.8, n_runs))
    mean_color  = "#e67e22"
    pooled_color = "#2c3e50"

    for i, run in enumerate(run_names):
        vals = [
            float(df_occ.loc[run, su]) if df_occ.loc[run, su] != "N/A" else 0
            for su in subunits
        ]
        ax.bar(x + i * width, vals, width, label=run,
               color=run_colors[i], alpha=0.8, edgecolor="black", linewidth=0.5)

    # mean ± std bar
    mean_vals   = [means[su] for su in subunits]
    std_vals    = [stds[su]  for su in subunits]
    ax.bar(x + n_runs * width, mean_vals, width,
           yerr=std_vals, label="mean ± std",
           color=mean_color, alpha=0.9, edgecolor="black", linewidth=0.8,
           capsize=4, error_kw={"elinewidth": 1.5})

    # pooled bar
    pooled_vals = [pooled_occ[su] for su in subunits]
    ax.bar(x + (n_runs + 1) * width, pooled_vals, width, label="pooled",
           color=pooled_color, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x + (n_bars - 1) * width / 2)
    ax.set_xticklabels(subunits, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Occupancy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title(f"{table_def['title']}\n[{col_type}]", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved plot: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Path containing RUN1, RUN2, ... folders")
    parser.add_argument("--out", required=True,
                        help="Output directory for combined results")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.base_dir)
    if not run_dirs:
        print(f"No RUN* folders found in {args.base_dir}")
        return
    print(f"Found {len(run_dirs)} RUN folders: {[r for r,_ in run_dirs]}")

    os.makedirs(args.out, exist_ok=True)

    # Summary row/col labels
    SUMMARY_ROWS = ["197↔316/305", "197↔315/304", "199↔315/304"]
    SUMMARY_COLS = ["2a↔G1", "2b↔2a", "2c↔2b", "G1↔2c"]
    TABLE_ORDER  = ["GLN197_ALA", "GLN197_GLU", "LYS199_GLU"]

    for col_type in COLUMNS:
        print(f"\n{'='*60}")
        print(f"Column type: {col_type}")
        print(f"{'='*60}")

        col_out = os.path.join(args.out, col_type)
        os.makedirs(col_out, exist_ok=True)

        # Collect mean/std/pooled per table for compact summaries
        summary_mean_std = {}
        summary_pooled   = {}

        for table_name, table_def in TABLES.items():
            print(f"\n  Table: {table_name}")

            df_display, means, stds, pooled_occ = build_table(
                run_dirs, col_type, table_def
            )
            if df_display is None:
                print(f"    No data found — skipping.")
                continue

            # Save detailed CSV
            csv_path = os.path.join(col_out, f"{table_name}.csv")
            df_display.to_csv(csv_path)
            print(f"    Saved table: {csv_path}")
            print(df_display.to_string())

            # Save plot
            plot_path = os.path.join(col_out, f"{table_name}.png")
            df_occ_only = df_display[
                ~df_display.index.isin(["mean ± std", "pooled"])
            ]
            make_plot(df_occ_only, means, stds, pooled_occ,
                      table_def, col_type, plot_path)

            # Collect for compact summaries
            cols = list(table_def["subunits"].keys())

            # Re-derive pooled_raw and pooled_total for the pooled string
            pooled_raw_vals   = []
            pooled_total_vals = []
            for run_name, run_dir in run_dirs:
                df_run = load_occupancy_csvs(run_dir, col_type)
                if df_run is None:
                    continue
                for su, (src, prt) in table_def["subunits"].items():
                    _, raw, tot = lookup(df_run, src, prt)
                    break  # just need one to get totals per col; do all below

            # Rebuild pooled properly per column
            pooled_raw_per_col   = {su: 0.0 for su in cols}
            pooled_total_per_col = {su: 0.0 for su in cols}
            for run_name, run_dir in run_dirs:
                df_run = load_occupancy_csvs(run_dir, col_type)
                if df_run is None:
                    continue
                for su, (src, prt) in table_def["subunits"].items():
                    _, raw, tot = lookup(df_run, src, prt)
                    if not np.isnan(raw):
                        pooled_raw_per_col[su]   += raw
                    if not np.isnan(tot):
                        pooled_total_per_col[su] += tot

            summary_mean_std[table_name] = [
                f"{means[su]:.4f} ± {stds[su]:.4f}" for su in cols
            ]
            summary_pooled[table_name] = [
                f"{pooled_raw_per_col[su]/pooled_total_per_col[su]:.4f} "
                f"({int(pooled_raw_per_col[su])}/{int(pooled_total_per_col[su])})"
                if pooled_total_per_col[su] > 0 else "N/A"
                for su in cols
            ]

        # Write compact summary CSVs
        if summary_mean_std:
            df_ms = pd.DataFrame(
                [summary_mean_std[t] for t in TABLE_ORDER if t in summary_mean_std],
                columns=SUMMARY_COLS,
                index=[r for r, t in zip(SUMMARY_ROWS, TABLE_ORDER) if t in summary_mean_std]
            )
            ms_path = os.path.join(col_out, "summary_mean_std.csv")
            df_ms.to_csv(ms_path)
            print(f"\n  Saved summary_mean_std:\n{df_ms.to_string()}\n")

        if summary_pooled:
            df_po = pd.DataFrame(
                [summary_pooled[t] for t in TABLE_ORDER if t in summary_pooled],
                columns=SUMMARY_COLS,
                index=[r for r, t in zip(SUMMARY_ROWS, TABLE_ORDER) if t in summary_pooled]
            )
            po_path = os.path.join(col_out, "summary_pooled.csv")
            df_po.to_csv(po_path)
            print(f"  Saved summary_pooled:\n{df_po.to_string()}\n")

    print(f"\nAll done. Results in: {args.out}")


if __name__ == "__main__":
    main()
    
'''
python3 combine_hbond_occupancy_gln197_lys199.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_hbond_occupancy_gln197_lys199
'''