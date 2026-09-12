"""
global_hbond_summary.py
=========================
Aggregates H-bond analysis results across all RUN* folders for a specific residue.

Part 1 — Global H-bond summary (all frames):
    Reads residue_neighbors/<resid_folder>/hbonds/hbond_residue_summary.csv
    and hbond_type_per_frame.csv from pair folders.
    Computes mean ± std % across RUNs per partner per bond type.

Part 2 — Global HBC threshold analysis:
    Reads residue_neighbors/<resid_folder>/hbonds/hbc_threshold_analysis/AC|BD/data_<bt>.csv
    Computes mean ± std % per partner per threshold across RUNs.
    Missing partners treated as 0%.

Output (saved in base_path/global_hbond_summary/<resid_folder>/):
    Part 1:
        global_all.csv / barplot_all.png
        global_bb_bb.csv / barplot_bb_bb.png
        global_sc_sc.csv / barplot_sc_sc.png
        global_bb_sc.csv / barplot_bb_sc.png
        global_sc_any.csv / barplot_sc_any.png
    Part 2:
        hbc_threshold/AC/
            global_all.csv / plot_all.png
            global_bb_bb.csv / plot_bb_bb.png
            ...
        hbc_threshold/BD/
            same files

Standalone — run directly:

    python3 global_hbond_summary.py \\
        --base_path /path/to/G12/ \\
        --resid_folder 197_2b

    python3 global_hbond_summary.py \
        --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
        --resid_folder 197_2d


    python3 global_hbond_summary.py \
        --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
        --resid_folder 199_2b
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BB_ATOMS = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3"}

BOND_TYPES = {
    "all":    "All H-bonds",
    "bb_bb":  "Backbone-Backbone",
    "sc_sc":  "Sidechain-Sidechain",
    "bb_sc":  "Backbone-Sidechain",
    "sc_any": "Sidechain Involved",
}

COLORS = {
    "all":    "steelblue",
    "bb_bb":  "steelblue",
    "sc_sc":  "darkorange",
    "bb_sc":  "mediumseagreen",
    "sc_any": "mediumpurple",
}

THRESHOLDS = [5, 6, 7, 8, 9, 10, 11, 12, 13]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _barplot(partner_stats, title, out_path, color, bt_label):
    """Bar plot of mean % with std shown as text label."""
    if not partner_stats:
        return
    partners = sorted(partner_stats.keys(), key=lambda p: -partner_stats[p]["mean"])
    means    = [partner_stats[p]["mean"] for p in partners]
    stds     = [partner_stats[p]["std"]  for p in partners]
    n_runs   = partner_stats[partners[0]]["n_runs"]

    fig_w = max(6, len(partners) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x    = np.arange(len(partners))
    bars = ax.bar(x, means, color=color, edgecolor="white", alpha=0.85)

    max_val = max(means) if means else 1
    for bar, mean, std in zip(bars, means, stds):
        # only label bars >= 5% to avoid crowding
        if mean >= 5.0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    f"{mean:.1f}%\n±{std:.1f}",
                    ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Mean % frames with H-bond", fontsize=15)
    ax.set_xlabel("Partner residue", fontsize=15)
    ax.set_title(f"{title}\n(n={n_runs} RUNs, mean ± std)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, min(130, max_val * 1.6))
    plt.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def _lineplot_threshold(data, title, out_path, color):
    """
    Line plot: x=threshold, y=mean%, shaded std.
    data: {partner: {"means": [...], "stds": [...]}}
    """
    if not data:
        return

    # only plot partners reaching >= 5% at any threshold
    to_plot = {p: v for p, v in data.items() if max(v["means"]) >= 5.0}
    if not to_plot:
        to_plot = dict(list(data.items())[:10])

    n = len(to_plot)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (partner, vals) in enumerate(to_plot.items()):
        means = np.array(vals["means"])
        stds  = np.array(vals["stds"])
        color_i = cmap(i)
        ax.plot(THRESHOLDS, means, marker="o", linewidth=2.0,
                label=partner, color=color_i, alpha=0.9)
        ax.errorbar(THRESHOLDS, means, yerr=stds,
                    fmt="none", ecolor=color_i, elinewidth=1.0,
                    capsize=3, alpha=0.5)

    ax.set_xlabel("HBC threshold (Å)", fontsize=13)
    ax.set_ylabel("Mean % frames with H-bond", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(THRESHOLDS)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.6,
              ncol=max(1, n // 8 + 1))
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Part 1 — Global H-bond summary
# -----------------------------------------------------------------------------

def _part1_global_hbond(
    run_dirs:     List[Path],
    resid_folder: str,
    out_dir:      Path,
    pdb_label:    str,
):
    print("\n--- Part 1: Global H-bond summary ---")

    # collect per-bond-type data across RUNs
    # {bt: {partner: [pct_run1, pct_run2, ...]}}
    all_data = {bt: {} for bt in BOND_TYPES}

    bond_cols = {
        "all":    "partners",
        "bb_bb":  "bb_bb_partners",
        "sc_sc":  "sc_sc_partners",
        "bb_sc":  "bb_sc_partners",
        "sc_any": "sc_any_partners",
    }

    n_valid = 0
    for run_dir in run_dirs:
        hb_dir = run_dir / "residue_neighbors" / resid_folder / "hbonds"

        # --- all types from hbond_residue_summary.csv ---
        res_csv = hb_dir / "hbond_residue_summary.csv"
        if not res_csv.exists():
            continue
        n_valid += 1
        df_res = pd.read_csv(res_csv)
        n_total = df_res["n_total_frames"].max() if "n_total_frames" in df_res.columns else None

        for _, row in df_res.iterrows():
            p   = str(row["partner"])
            pct = float(row["pct_frames"])
            if p not in all_data["all"]:
                all_data["all"][p] = []
            all_data["all"][p].append(pct)

        # --- bond type breakdown from hbond_type_per_frame.csv in pair folders ---
        per_frame_csv = hb_dir / "hbond_per_frame.csv"
        if not per_frame_csv.exists():
            continue
        df_pf = pd.read_csv(per_frame_csv)
        if "bb_bb_partners" not in df_pf.columns:
            continue
        n_frames = len(df_pf)

        for bt in ["bb_bb", "sc_sc", "bb_sc", "sc_any"]:
            col = bond_cols[bt]
            if col not in df_pf.columns:
                continue
            # count frames per partner for this bond type
            partner_counts = {}
            for _, row in df_pf.iterrows():
                s = str(row[col])
                if s == "nan" or not s.strip():
                    continue
                for p in s.split("|"):
                    p = p.strip()
                    if p:
                        partner_counts[p] = partner_counts.get(p, 0) + 1
            for p, cnt in partner_counts.items():
                pct = round(100.0 * cnt / n_frames, 2)
                if p not in all_data[bt]:
                    all_data[bt][p] = []
                all_data[bt][p].append(pct)

    if n_valid == 0:
        print(f"  No valid RUNs found for {resid_folder}")
        return

    print(f"  Aggregating across {n_valid} RUNs...")

    # fill missing RUNs with 0
    for bt in BOND_TYPES:
        for p in all_data[bt]:
            while len(all_data[bt][p]) < n_valid:
                all_data[bt][p].append(0.0)

    # compute mean ± std and save
    for bt, bt_label in BOND_TYPES.items():
        stats = {}
        rows  = []
        for p, vals in all_data[bt].items():
            vals_arr = np.array(vals)
            mean = float(np.mean(vals_arr))
            std  = float(np.std(vals_arr))
            if mean < 0.1:
                continue
            stats[p] = {"mean": mean, "std": std, "n_runs": n_valid}
            rows.append({"partner": p, "mean_pct": round(mean, 2),
                         "std_pct": round(std, 2), "n_runs": n_valid})

        if rows:
            df_out = (pd.DataFrame(rows)
                      .sort_values("mean_pct", ascending=False)
                      .reset_index(drop=True))
            df_out.to_csv(out_dir / f"global_{bt}.csv", index=False)
            print(f"  Saved -> global_{bt}.csv  ({len(df_out)} partners)")

            _barplot(stats,
                     f"{pdb_label} — {bt_label} (global, {n_valid} RUNs)",
                     out_dir / f"barplot_{bt}.png",
                     COLORS[bt], bt_label)


# -----------------------------------------------------------------------------
# Part 2 — Global HBC threshold analysis
# -----------------------------------------------------------------------------

def _part2_hbc_threshold(
    run_dirs:     List[Path],
    resid_folder: str,
    out_dir:      Path,
    pdb_label:    str,
):
    print("\n--- Part 2: Global HBC threshold analysis ---")

    thr_out = out_dir / "hbc_threshold"
    thr_out.mkdir(exist_ok=True)

    for gate in ["AC", "BD"]:
        gate_out = thr_out / gate
        gate_out.mkdir(exist_ok=True)

        for bt in BOND_TYPES:
            # {partner: {thr: [pct_run1, pct_run2, ...]}}
            data = {}
            n_valid = 0

            for run_dir in run_dirs:
                data_csv = (run_dir / "residue_neighbors" / resid_folder /
                            "hbonds" / "hbc_threshold_analysis" / gate /
                            f"data_{bt}.csv")
                if not data_csv.exists():
                    continue
                n_valid += 1

                df = pd.read_csv(data_csv, index_col=0)
                # index = threshold, columns = partners
                for partner in df.columns:
                    if partner not in data:
                        data[partner] = {t: [] for t in THRESHOLDS}
                    for thr in THRESHOLDS:
                        thr_str = str(thr)
                        thr_int = int(thr)
                        val = 0.0
                        if thr_int in df.index:
                            val = float(df.loc[thr_int, partner])
                        elif thr_str in df.index:
                            val = float(df.loc[thr_str, partner])
                        data[partner][thr_int].append(val)

            if n_valid == 0:
                continue

            # fill missing RUNs with 0
            for partner in data:
                for thr in THRESHOLDS:
                    while len(data[partner][thr]) < n_valid:
                        data[partner][thr].append(0.0)

            # compute mean ± std per threshold per partner
            agg = {}  # {partner: {"means": [...], "stds": [...]}}
            csv_rows = []
            all_partners = list(data.keys())

            for partner in all_partners:
                means = []
                stds  = []
                for thr in THRESHOLDS:
                    vals = np.array(data[partner][thr])
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                # skip if never above 1%
                if max(means) < 1.0:
                    continue
                agg[partner] = {"means": means, "stds": stds}
                row = {"partner": partner}
                for i, thr in enumerate(THRESHOLDS):
                    row[f"thr_{thr}_mean"] = round(means[i], 2)
                    row[f"thr_{thr}_std"]  = round(stds[i],  2)
                csv_rows.append(row)

            if not csv_rows:
                continue

            # save CSV
            df_out = pd.DataFrame(csv_rows).sort_values(
                f"thr_{THRESHOLDS[0]}_mean", ascending=False).reset_index(drop=True)
            csv_path = gate_out / f"global_{bt}.csv"
            df_out.to_csv(csv_path, index=False)
            print(f"  Saved -> hbc_threshold/{gate}/global_{bt}.csv  ({len(df_out)} partners)")

            # line plot
            _lineplot_threshold(
                data   = agg,
                title  = (f"{pdb_label} — {BOND_TYPES[bt]}\n"
                          f"{gate} gate >= threshold  (n={n_valid} RUNs, mean ± std)"),
                out_path = gate_out / f"plot_{bt}.png",
                color    = COLORS[bt],
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_global_hbond_summary(base_path: Path, resid_folder: str):
    base_path = Path(base_path)
    out_dir   = base_path / "global_hbond_summary" / resid_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # find all RUN* folders
    run_dirs = sorted([
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.upper().startswith("RUN")
    ])
    if not run_dirs:
        print(f"ERROR: no RUN* folders found in {base_path}")
        return

    # pdb label from folder e.g. "197_2b" -> "197.2b"
    pdb_label = resid_folder.replace("_", ".", 1) if "_" in resid_folder else resid_folder

    print("=" * 70)
    print("GLOBAL H-BOND SUMMARY")
    print(f"  base_path    : {base_path}")
    print(f"  resid_folder : {resid_folder}  ({pdb_label})")
    print(f"  RUNs found   : {len(run_dirs)}")
    print(f"  output       : {out_dir}")
    print("=" * 70)

    _part1_global_hbond(run_dirs, resid_folder, out_dir, pdb_label)
    _part2_hbc_threshold(run_dirs, resid_folder, out_dir, pdb_label)

    print(f"\n✓ Global H-bond summary complete.")
    print(f"  Results in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate H-bond results across RUN* folders."
    )
    parser.add_argument("--base_path", type=str, required=True,
                        help="Folder containing RUN1, RUN2, ...")
    parser.add_argument("--resid_folder", type=str, required=True,
                        help="Residue folder name e.g. '197_2b'")
    args = parser.parse_args()

    run_global_hbond_summary(
        base_path    = args.base_path,
        resid_folder = args.resid_folder,
    )