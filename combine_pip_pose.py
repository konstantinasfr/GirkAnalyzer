"""
combine_pip_pose.py
====================
Combines pip_pose results across multiple RUN folders into global summaries.

Reads per-RUN output files and aggregates by summing counts, then
recalculating percentages from combined totals.

Files combined:
  - subunit_*/summary.csv                      -> state distribution per subunit
  - lys90_per_state.csv                        -> LYS90 co-occurrence per state
  - unique_residue_count_distribution.csv      -> residue count distribution
  - marginal_per_p_atom.csv                    -> P4/P5 independent pose distribution

All repeated for gate subfolders:
  asn_gate_frames/     (end2  -- ASN gate crossing)
  asn_gate_m1_frames/  (end2-1 -- frame before ASN gate)
  hbc_gate_frames/     (end3  -- HBC gate crossing)
  hbc_gate_m1_frames/  (end3-1 -- frame before HBC gate)

Usage
-----
    python combine_pip_pose.py \
        --runs_dir     /path/to/G12 \
        --channel_type G12 \
        --output_dir   /path/to/G12/combined_pip_pose

Output structure
----------------
    output_dir/
        subunit_2a/
            global_summary.csv
            barplot_states.png
            marginal_P4_pose.png
            marginal_P5_pose.png
        subunit_2b/ ...
        lys90_per_state.csv
        lys90_per_state_2a.png ...
        unique_residue_count_distribution.csv
        marginal_pose_distribution.csv
        count_distribution_per_state/
        asn_gate_frames/      (same structure)
        asn_gate_m1_frames/   (same structure)
        hbc_gate_frames/      (same structure)
        hbc_gate_m1_frames/   (same structure)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



ALL_STATES = [
    "p4_nterm_p5_nterm", "p4_nterm_p5_cterm", "p4_nterm_p5_none",
    "p4_cterm_p5_nterm", "p4_cterm_p5_cterm", "p4_cterm_p5_none",
    "p4_none_p5_nterm",  "p4_none_p5_cterm",  "p4_none_p5_none",
]

STATE_COLORS = {
    "p4_nterm_p5_nterm": "#1565C0",
    "p4_nterm_p5_cterm": "#4CAF50",
    "p4_nterm_p5_none":  "#A5D6A7",
    "p4_cterm_p5_nterm": "#AB47BC",
    "p4_cterm_p5_cterm": "#B71C1C",
    "p4_cterm_p5_none":  "#EF9A9A",
    "p4_none_p5_nterm":  "#90CAF9",
    "p4_none_p5_cterm":  "#FFCDD2",
    "p4_none_p5_none":   "#9E9E9E",
}

# Gate subfolders: (folder_name, human_label)
GATE_SUBSETS = [
    ("asn_gate_frames",    "ASN gate (end2)"),
    ("asn_gate_m1_frames", "ASN gate -1 (pre-end2)"),
    ("hbc_gate_frames",    "HBC gate (end3)"),
    ("hbc_gate_m1_frames", "HBC gate -1 (pre-end3)"),
]

STATE_LABEL = lambda s: s.replace("p4_", "").replace("_p5_", "+")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sorted_run_dirs(runs_dir: Path, exclude_runs: list[str] | None = None) -> list:
    exclude = {r.upper() for r in (exclude_runs or [])}
    return sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.upper().startswith("RUN")
         and d.name.upper() not in exclude],
        key=lambda d: d.name,
    )


def _barplot(labels, values, ylabel, title, out_path,
             colors=None, figsize=None, fontsize=18, rotation=30):
    if not labels:
        return
    fig_w = figsize[0] if figsize else max(8, len(labels) * 0.9)
    fig_h = figsize[1] if figsize else 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(len(labels))
    c = colors if colors else ["#7E57C2"] * len(labels)
    bars = ax.bar(x, values, color=c, edgecolor="white", alpha=0.85)
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    f"{val:.1f}%", va="bottom", ha="center", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 3)
    ax.set_ylim(0, min(115, max_val * 1.35) if max_val > 0 else 10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _double_barplot(labels, vals_p4, vals_p5, title, out_path, fontsize=18):
    if not labels:
        return
    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    b1 = ax.bar(x - width / 2, vals_p4, width, label="P4",
                color="#FF9800", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, vals_p5, width, label="P5",
                color="#2196F3", alpha=0.85, edgecolor="white")
    max_val = max(vals_p4 + vals_p5) if vals_p4 + vals_p5 else 1
    for bar, val in zip(b1, vals_p4):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    f"{val:.0f}%", va="bottom", ha="center",
                    fontsize=fontsize - 1, color="#E65100")
    for bar, val in zip(b2, vals_p5):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    f"{val:.0f}%", va="bottom", ha="center",
                    fontsize=fontsize - 1, color="#0D47A1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel("% frames LYS90 close", fontsize=fontsize + 1)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize)
    ax.set_ylim(0, min(115, max_val * 1.35) if max_val > 0 else 10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Combine state summary per subunit
# -----------------------------------------------------------------------------

def _combine_state_summaries(runs_dir: Path, out_dir: Path,
                              channel_type: str, subset: str = None,
                              label: str = None, exclude_runs: list = None):
    run_dirs  = _sorted_run_dirs(runs_dir, exclude_runs)
    base_path = "pip_pose" + (f"/{subset}" if subset else "")
    label     = label or subset or "all frames"

    print(f"\n  State summaries [{label}]...")

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    sub_data = {s: [] for s in SUBUNIT_ORDER}
    for run_dir in run_dirs:
        for sub in SUBUNIT_ORDER:
            csv = run_dir / base_path / f"subunit_{sub}" / "summary.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if df.empty:
                continue
            sub_data[sub].append(df)
            print(f"    {run_dir.name} / subunit_{sub}: {len(df)} states")

    for sub in SUBUNIT_ORDER:
        dfs = sub_data[sub]
        if not dfs:
            continue

        all_rows = pd.concat(dfs, ignore_index=True)
        agg = (all_rows.groupby("state")
                       .agg(count=("count", "sum"))
                       .reset_index())
        total          = agg["count"].sum()
        agg["pct"]     = (agg["count"] / total * 100).round(3)
        agg["total_frames"] = int(total)
        agg["label"]   = agg["state"].apply(STATE_LABEL)
        agg["n_runs"]  = len(dfs)
        agg = agg.sort_values("count", ascending=False)

        sub_out = out_dir / f"subunit_{sub}"
        sub_out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(sub_out / "global_summary.csv", index=False)

        states_present = [s for s in ALL_STATES if s in agg["state"].values]
        pcts   = [agg.loc[agg["state"] == s, "pct"].values[0]
                  if s in agg["state"].values else 0.0 for s in states_present]
        colors = [STATE_COLORS.get(s, "#9E9E9E") for s in states_present]
        labels_plot = [STATE_LABEL(s) for s in states_present]
        _barplot(labels_plot, pcts, "% of frames",
                 f"PIP2 Pose -- {channel_type} -- Subunit {sub}\n"
                 f"[{label}, {len(dfs)} runs, {total} frames]",
                 sub_out / "barplot_states.png",
                 colors=colors, fontsize=18)
        print(f"    Subunit {sub}: {total} total frames, {len(dfs)} runs -> saved")

    # -- summary table: rows=subunits, cols=states -------------------------
    subs_for_table = [s for s in SUBUNIT_ORDER if sub_data[s]]
    if subs_for_table:
        table_rows = []
        for sub in subs_for_table:
            dfs = sub_data[sub]
            all_rows_t = pd.concat(dfs, ignore_index=True)
            agg_t = (all_rows_t.groupby("state")
                               .agg(count=("count", "sum"))
                               .reset_index())
            total_t = agg_t["count"].sum()
            agg_t["pct"] = (agg_t["count"] / total_t * 100).round(1)
            row = {"subunit": sub}
            for state in ALL_STATES:
                val = agg_t.loc[agg_t["state"] == state, "pct"].values
                row[STATE_LABEL(state)] = float(val[0]) if len(val) > 0 else 0.0
            table_rows.append(row)
        # transpose: rows=states, cols=subunits
        df_joint_table = pd.DataFrame(table_rows).set_index("subunit").T
        df_joint_table.index.name = "state"
        df_joint_table.to_csv(out_dir / "summary_table_joint.csv")
        try:
            excel_path = out_dir / "summary_table_joint.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df_joint_table.to_excel(writer, sheet_name="joint_states")
            print(f"  Joint summary table -> {excel_path}")
        except Exception as e:
            print(f"  Excel export failed: {e}")

    # -- combined plot: all subunits side by side -----------------------------
    subs_found = [s for s in SUBUNIT_ORDER if sub_data[s]]
    n_subs     = len(subs_found)
    if n_subs > 0:
        # collect union of states present across all subunits
        all_states_present = []
        for s in ALL_STATES:
            for sub in subs_found:
                dfs = sub_data[sub]
                if not dfs:
                    continue
                all_rows = pd.concat(dfs, ignore_index=True)
                if s in all_rows["state"].values:
                    if s not in all_states_present:
                        all_states_present.append(s)
                    break

        fig, axes = plt.subplots(1, n_subs,
                                 figsize=(max(6, len(all_states_present)*0.45) * n_subs, 6),
                                 sharey=False)
        if n_subs == 1: axes = [axes]

        for ax, sub in zip(axes, subs_found):
            dfs = sub_data[sub]
            all_rows = pd.concat(dfs, ignore_index=True)
            agg_s = (all_rows.groupby("state")
                             .agg(count=("count", "sum"))
                             .reset_index())
            total_s = agg_s["count"].sum()
            agg_s["pct"] = (agg_s["count"] / total_s * 100).round(3)

            states_s = [s for s in ALL_STATES if s in agg_s["state"].values]
            pcts_s   = [agg_s.loc[agg_s["state"]==s, "pct"].values[0]
                        if s in agg_s["state"].values else 0.0 for s in states_s]
            colors_s = [STATE_COLORS.get(s, "#9E9E9E") for s in states_s]
            labels_s = [STATE_LABEL(s) for s in states_s]

            x       = np.arange(len(states_s))
            max_val = max(pcts_s) if pcts_s else 1
            bars = ax.bar(x, pcts_s, width=0.65, color=colors_s,
                          edgecolor="white", alpha=0.85)
            for bar, val in zip(bars, pcts_s):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max_val*0.02,
                            f"{val:.1f}%", va="bottom", ha="center", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=17)
            ax.tick_params(axis="y", labelsize=17)
            ax.set_ylabel("% of frames", fontsize=18)
            ax.set_title(f"Subunit {sub}", fontsize=19)
            ax.set_ylim(0, min(115, max_val*1.35) if max_val > 0 else 10)

        n_runs_total = max(len(sub_data[s]) for s in subs_found)
        plt.suptitle(f"PIP2 Pose -- {channel_type} [{label}, {n_runs_total} runs]",
                     fontsize=17)
        plt.tight_layout()
        fig.savefig(out_dir / "barplot_states_all_subunits.png", dpi=300)
        plt.close(fig)
        print(f"  Combined subunit barplot -> barplot_states_all_subunits.png")


# -----------------------------------------------------------------------------
# Combine LYS90 per state
# -----------------------------------------------------------------------------

def _combine_lys90(runs_dir: Path, out_dir: Path,
                   channel_type: str, subset: str = None,
                   label: str = None, exclude_runs: list = None):
    run_dirs  = _sorted_run_dirs(runs_dir, exclude_runs)
    base_path = "pip_pose" + (f"/{subset}" if subset else "")
    label     = label or subset or "all frames"

    print(f"\n  LYS90 per state [{label}]...")

    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / base_path / "lys90_per_state.csv"
        if not csv.exists():
            continue
        dfs.append(pd.read_csv(csv))

    if not dfs:
        print(f"    No lys90_per_state.csv found.")
        return

    all_rows = pd.concat(dfs, ignore_index=True)
    agg = (all_rows.groupby(["subunit", "state"])
                   .agg(
                       n_frames         = ("n_frames",         "sum"),
                       n_p4_lys90_close = ("n_p4_lys90_close", "sum"),
                       n_p5_lys90_close = ("n_p5_lys90_close", "sum"),
                   )
                   .reset_index())
    agg["pct_p4_lys90"] = (agg["n_p4_lys90_close"] / agg["n_frames"] * 100).round(2)
    agg["pct_p5_lys90"] = (agg["n_p5_lys90_close"] / agg["n_frames"] * 100).round(2)
    agg.to_csv(out_dir / "lys90_per_state.csv", index=False)

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    for sub in SUBUNIT_ORDER:
        df_s = agg[agg["subunit"] == sub]
        if df_s.empty:
            continue
        states_s = [s for s in ALL_STATES if s in df_s["state"].values]
        if not states_s:
            continue
        labels_plot = [STATE_LABEL(s) for s in states_s]
        vals_p4 = [df_s.loc[df_s["state"] == s, "pct_p4_lys90"].values[0]
                   if s in df_s["state"].values else 0.0 for s in states_s]
        vals_p5 = [df_s.loc[df_s["state"] == s, "pct_p5_lys90"].values[0]
                   if s in df_s["state"].values else 0.0 for s in states_s]
        _double_barplot(labels_plot, vals_p4, vals_p5,
                        f"LYS90 proximity per state -- {channel_type}\n"
                        f"Subunit {sub} [{label}, {len(dfs)} runs]",
                        out_dir / f"lys90_per_state_{sub}.png")
    print(f"    LYS90 per state saved ({len(dfs)} runs)")


# -----------------------------------------------------------------------------
# Combine unique residue count distribution
# -----------------------------------------------------------------------------

def _combine_count_distribution(runs_dir: Path, out_dir: Path,
                                 channel_type: str, subset: str = None,
                                 label: str = None, exclude_runs: list = None):
    run_dirs  = _sorted_run_dirs(runs_dir, exclude_runs)
    base_path = "pip_pose" + (f"/{subset}" if subset else "")
    label     = label or subset or "all frames"

    print(f"\n  Count distribution [{label}]...")

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / base_path / "unique_residue_count_distribution.csv"
        if not csv.exists():
            continue
        dfs.append(pd.read_csv(csv))

    if not dfs:
        print(f"    No unique_residue_count_distribution.csv found.")
        return

    all_rows = pd.concat(dfs, ignore_index=True)
    agg = (all_rows.groupby(["subunit", "state", "n_residues"])
                   .agg(frames=("frames", "sum"))
                   .reset_index())
    totals = agg.groupby(["subunit", "state"])["frames"].sum().reset_index()
    totals.rename(columns={"frames": "total"}, inplace=True)
    agg = agg.merge(totals, on=["subunit", "state"])
    agg["pct"] = (agg["frames"] / agg["total"] * 100).round(3)
    agg.drop(columns="total", inplace=True)
    agg.to_csv(out_dir / "unique_residue_count_distribution.csv", index=False)

    dist_dir = out_dir / "count_distribution_per_state"
    dist_dir.mkdir(exist_ok=True)

    for state in ALL_STATES:
        df_st = agg[agg["state"] == state]
        if df_st.empty:
            continue
        subs_in_state = [s for s in SUBUNIT_ORDER if s in df_st["subunit"].values]
        n_subs = len(subs_in_state)
        if n_subs == 0:
            continue
        fig, axes = plt.subplots(1, n_subs, figsize=(6 * n_subs, 6), sharey=False)
        if n_subs == 1: axes = [axes]
        for ax, sub in zip(axes, subs_in_state):
            df_sub = df_st[df_st["subunit"] == sub].sort_values("n_residues")
            xlabels = [str(int(r)) for r in df_sub["n_residues"]]
            pcts    = df_sub["pct"].tolist()
            max_val = max(pcts) if pcts else 1
            x = np.arange(len(xlabels))
            bars = ax.bar(x, pcts, width=0.5, color="#7E57C2", alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.02,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=18)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=18)
            ax.tick_params(axis="y", labelsize=18)
            ax.set_xlabel("Number of unique residues", fontsize=18)
            ax.set_ylabel("% of frames", fontsize=18)
            ax.set_title(f"Subunit {sub}", fontsize=19)
            ax.set_ylim(0, min(110, max_val * 1.3))
        plt.suptitle(f"Unique residues -- {STATE_LABEL(state)}\n"
                     f"{channel_type} [{label}, {len(dfs)} runs]", fontsize=19)
        plt.tight_layout()
        fig.savefig(dist_dir / f"count_dist_{state}.png", dpi=100)
        plt.close(fig)
    print(f"    Count distribution saved ({len(dfs)} runs)")


# -----------------------------------------------------------------------------
# Combine global count distribution
# -----------------------------------------------------------------------------

def _combine_global_count_distribution(runs_dir: Path, out_dir: Path,
                                        channel_type: str, subset: str = None,
                                        label: str = None, exclude_runs: list = None):
    run_dirs  = _sorted_run_dirs(runs_dir, exclude_runs)
    base_path = "pip_pose" + (f"/{subset}" if subset else "")
    label     = label or subset or "all frames"

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    print(f"\n  Global count distribution [{label}]...")

    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / base_path / "global_count_distribution.csv"
        if not csv.exists():
            continue
        dfs.append(pd.read_csv(csv))

    if not dfs:
        print(f"    No global_count_distribution.csv found.")
        return

    all_rows = pd.concat(dfs, ignore_index=True)
    agg = (all_rows.groupby(["subunit", "n_residues"])
                   .agg(frames=("frames", "sum"))
                   .reset_index())
    totals = agg.groupby("subunit")["frames"].sum().reset_index()
    totals.rename(columns={"frames": "total"}, inplace=True)
    agg = agg.merge(totals, on="subunit")
    agg["pct"] = (agg["frames"] / agg["total"] * 100).round(3)
    agg.drop(columns="total", inplace=True)
    agg.to_csv(out_dir / "global_count_distribution.csv", index=False)

    subs_present = [s for s in SUBUNIT_ORDER if s in agg["subunit"].values]
    n_subs = len(subs_present)
    if n_subs == 0:
        return

    fig, axes = plt.subplots(1, n_subs, figsize=(6 * n_subs, 6), sharey=False)
    if n_subs == 1: axes = [axes]
    for ax, sub in zip(axes, subs_present):
        df_s    = agg[agg["subunit"] == sub].sort_values("n_residues")
        xlabels = [str(int(r)) for r in df_s["n_residues"]]
        pcts    = df_s["pct"].tolist()
        max_val = max(pcts) if pcts else 1
        bars = ax.bar(range(len(xlabels)), pcts,
                      color="#5C6BC0", alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    f"{val:.1f}%", va="bottom", ha="center", fontsize=18)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel("% of frames", fontsize=19)
        ax.set_title(f"Subunit {sub}", fontsize=19)
        ax.set_ylim(0, min(110, max_val * 1.25))
    plt.suptitle(f"Unique residues close to P4+P5 (all states)\n"
                 f"{channel_type} [{label}, {len(dfs)} runs]", fontsize=19)
    plt.tight_layout()
    fig.savefig(out_dir / "global_count_distribution.png", dpi=100)
    plt.close(fig)
    print(f"    Global count distribution saved ({len(dfs)} runs)")


# -----------------------------------------------------------------------------
# Marginal P4 / P5 pose distributions
# -----------------------------------------------------------------------------

def _marginal_pose_plots(runs_dir: Path, out_dir: Path,
                          channel_type: str, subset: str = None,
                          label: str = None, exclude_runs: list = None):
    label = label or subset or "all frames"
    print(f"\n  Marginal pose distributions [{label}]...")

    poses  = ["nterm", "cterm", "none"]
    colors = {"nterm": "#2196F3", "cterm": "#F44336", "none": "#9E9E9E"}

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]
        
    subs_found = [s for s in SUBUNIT_ORDER
                  if (out_dir / f"subunit_{s}" / "global_summary.csv").exists()]
    if not subs_found:
        print(f"    No global_summary.csv found -- skipping marginal plots.")
        return

    results = []
    for sub in subs_found:
        df = pd.read_csv(out_dir / f"subunit_{sub}" / "global_summary.csv")
        if df.empty:
            continue
        total = df["count"].sum()
        for p_atom in ["P4", "P5"]:
            for pose in poses:
                if p_atom == "P4":
                    mask = df["state"].str.startswith(f"p4_{pose}_")
                else:
                    mask = df["state"].str.endswith(f"_p5_{pose}")
                count = df.loc[mask, "count"].sum()
                results.append({
                    "subunit": sub,
                    "p_atom":  p_atom,
                    "pose":    pose,
                    "count":   int(count),
                    "pct":     round(count / total * 100, 3),
                })

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "marginal_pose_distribution.csv", index=False)

    for p_atom in ["P4", "P5"]:
        df_p   = df_res[df_res["p_atom"] == p_atom]
        n_subs = len(subs_found)
        fig, axes = plt.subplots(1, n_subs, figsize=(5 * n_subs, 6), sharey=False)
        if n_subs == 1: axes = [axes]

        for ax, sub in zip(axes, subs_found):
            df_s    = df_p[df_p["subunit"] == sub]
            pcts    = [df_s.loc[df_s["pose"] == p, "pct"].values[0]
                       if p in df_s["pose"].values else 0.0 for p in poses]
            cols    = [colors[p] for p in poses]
            max_val = max(pcts) if pcts else 1
            bars = ax.bar(range(len(poses)), pcts, color=cols,
                          edgecolor="white", alpha=0.85, width=0.5)
            for bar, val in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.02,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=18)
            ax.set_xticks(range(len(poses)))
            ax.set_xticklabels(poses, fontsize=18)
            ax.tick_params(axis="y", labelsize=18)
            ax.set_ylabel("% of frames", fontsize=18)
            ax.set_title(f"Subunit {sub}", fontsize=19)
            ax.set_ylim(0, min(115, max_val * 1.35))

        plt.suptitle(f"{p_atom} pose distribution -- {channel_type}\n[{label}]",
                     fontsize=19)
        plt.tight_layout()
        fig.savefig(out_dir / f"marginal_{p_atom}_pose.png", dpi=100)
        plt.close(fig)

        for sub in subs_found:
            df_s    = df_p[df_p["subunit"] == sub]
            pcts    = [df_s.loc[df_s["pose"] == p, "pct"].values[0]
                       if p in df_s["pose"].values else 0.0 for p in poses]
            cols    = [colors[p] for p in poses]
            max_val = max(pcts) if pcts else 1
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            bars = ax2.bar(range(len(poses)), pcts, color=cols,
                           edgecolor="white", alpha=0.85, width=0.5)
            for bar, val in zip(bars, pcts):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max_val * 0.02,
                         f"{val:.1f}%", va="bottom", ha="center", fontsize=18)
            ax2.set_xticks(range(len(poses)))
            ax2.set_xticklabels(poses, fontsize=18)
            ax2.tick_params(axis="y", labelsize=18)
            ax2.set_ylabel("% of frames", fontsize=18)
            ax2.set_title(f"{p_atom} pose -- {channel_type}\n"
                          f"Subunit {sub} [{label}]", fontsize=19)
            ax2.set_ylim(0, min(115, max_val * 1.35))
            plt.tight_layout()
            sub_dir = out_dir / f"subunit_{sub}"
            sub_dir.mkdir(exist_ok=True)
            fig2.savefig(sub_dir / f"marginal_{p_atom}_pose.png", dpi=100)
            plt.close(fig2)

        print(f"  Marginal {p_atom} plots saved (combined + per subunit)")

    # -- summary tables: one per P atom, rows=poses, cols=subunits -----------
    print(f"  Marginal summary tables [{label}]...")
    try:
        excel_path = out_dir / "summary_tables.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for p_atom in ["P4", "P5"]:
                df_p = df_res[df_res["p_atom"] == p_atom]
                # build dict: pose -> {subunit: pct}
                data = {}
                for pose in poses:
                    data[f"{pose}_%"] = {}
                    for sub in subs_found:
                        df_s = df_p[df_p["subunit"] == sub]
                        val  = df_s.loc[df_s["pose"] == pose, "pct"].values
                        data[f"{pose}_%"][sub] = round(float(val[0]), 1) if len(val) > 0 else 0.0
                df_table = pd.DataFrame(data).T  # rows=poses, cols=subunits
                df_table.index.name = "pose"
                df_table.to_csv(out_dir / f"summary_table_{p_atom}.csv")
                df_table.to_excel(writer, sheet_name=p_atom)
                print(f"  Summary table {p_atom} -> {out_dir / f'summary_table_{p_atom}.csv'}")
        print(f"  Excel summary -> {excel_path}")
    except Exception as e:
        print(f"  Table export failed: {e}")


# -----------------------------------------------------------------------------
# Run all for one subset
# -----------------------------------------------------------------------------

def _run_all_for_subset(runs_dir: Path, out_dir: Path,
                         channel_type: str, subset: str = None,
                         label: str = None, exclude_runs: list = None):
    label = label or subset or "all frames"
    print(f"\n{'='*60}")
    print(f"  Processing: {label}")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _combine_state_summaries(runs_dir, out_dir, channel_type, subset, label, exclude_runs)
    _marginal_pose_plots(runs_dir, out_dir, channel_type, subset, label, exclude_runs)
    _combine_lys90(runs_dir, out_dir, channel_type, subset, label, exclude_runs)
    _combine_count_distribution(runs_dir, out_dir, channel_type, subset, label, exclude_runs)
    _combine_global_count_distribution(runs_dir, out_dir, channel_type, subset, label, exclude_runs)


# -----------------------------------------------------------------------------
# Delta analysis + heatmap
# -----------------------------------------------------------------------------

def _delta_analysis(output_dir: Path, channel_type: str):
    """
    For each P atom (P4, P5), load summary_table_*.csv from all frames and
    each gate subset, compute deltas vs all frames, save delta CSVs and heatmaps.
    """
    try:
        import matplotlib.colors as mcolors
    except ImportError:
        pass

    gate_labels = {
        "asn_gate_frames":    "ASN gate",
        "asn_gate_m1_frames": "ASN gate -1",
        "hbc_gate_frames":    "HBC gate",
        "hbc_gate_m1_frames": "HBC gate -1",
    }

    delta_dir = output_dir / "delta_analysis"
    delta_dir.mkdir(exist_ok=True)

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    for p_atom in ["P4", "P5"]:
        # load all-frames baseline
        baseline_csv = output_dir / f"summary_table_{p_atom}.csv"
        if not baseline_csv.exists():
            print(f"  Delta: missing baseline {baseline_csv} -- skipping {p_atom}")
            continue
        df_base = pd.read_csv(baseline_csv, index_col=0)

        subunits = [c for c in df_base.columns if c in SUBUNIT_ORDER]
        poses    = df_base.index.tolist()

        # subfolder per P atom
        p_dir = delta_dir / p_atom
        p_dir.mkdir(exist_ok=True)

        # collect deltas for each gate subset
        all_deltas = {}   # {gate_label: df_delta}
        for folder, human_label in gate_labels.items():
            gate_csv = output_dir / folder / f"summary_table_{p_atom}.csv"
            if not gate_csv.exists():
                continue
            df_gate  = pd.read_csv(gate_csv, index_col=0)
            df_delta = df_gate[subunits] - df_base[subunits]
            df_delta = df_delta.round(2)
            all_deltas[human_label] = df_delta

            # save delta CSV
            df_delta.to_csv(p_dir / f"delta_{p_atom}_{folder}.csv")

        if not all_deltas:
            continue

        # ── heatmaps: one combined + two split (crossing vs pre-crossing) ────
        vmax = max(df.abs().max().max() for df in all_deltas.values())
        vmax = max(vmax, 5.0)
        n_subs  = len(subunits)
        n_poses = len(poses)

        def _make_heatmap(deltas_dict, fname, title):
            n_g = len(deltas_dict)
            if n_g == 0:
                return
            fig, axes = plt.subplots(n_poses, n_g,
                                     figsize=(3.5 * n_g, 2.5 * n_poses),
                                     sharey=False)
            if n_poses == 1: axes = [axes]
            if n_g == 1:     axes = [[ax] for ax in axes]

            for gi, (gate_lbl, df_delta) in enumerate(deltas_dict.items()):
                for pi, pose in enumerate(poses):
                    ax   = axes[pi][gi]
                    vals = df_delta.loc[pose, subunits].values.reshape(1, -1).astype(float)
                    im   = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                     aspect="auto")
                    ax.set_xticks(range(n_subs))
                    ax.set_xticklabels(subunits, fontsize=13)
                    ax.set_yticks([0])
                    ax.set_yticklabels([pose.replace("_%", "")], fontsize=13)
                    for xi, val in enumerate(vals[0]):
                        color = "white" if abs(val) > vmax * 0.6 else "black"
                        ax.text(xi, 0, f"{val:+.1f}%", ha="center", va="center",
                                fontsize=12, color=color, fontweight="bold")
                    if pi == 0:
                        ax.set_title(gate_lbl, fontsize=13, pad=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(f"{p_atom} pose delta vs all frames -- {channel_type} | {title}",
                         fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            fig.savefig(p_dir / fname, dpi=300)
            plt.close(fig)
            print(f"  Delta heatmap {p_atom} [{title}] -> {p_dir / fname}")

        # all gates combined
        _make_heatmap(all_deltas,
                      f"heatmap_delta_{p_atom}_all.png",
                      "all gate subsets")

        # crossing frames only (ASN gate + HBC gate)
        crossing = {k: v for k, v in all_deltas.items() if "-1" not in k}
        _make_heatmap(crossing,
                      f"heatmap_delta_{p_atom}_crossing.png",
                      "crossing frames")

        # pre-crossing frames only (ASN gate -1 + HBC gate -1)
        pre = {k: v for k, v in all_deltas.items() if "-1" in k}
        _make_heatmap(pre,
                      f"heatmap_delta_{p_atom}_pre_crossing.png",
                      "pre-crossing frames")

        # ── combined delta table: all gates stacked ───────────────────────────
        stack_rows = []
        for gate_lbl, df_delta in all_deltas.items():
            for pose in poses:
                row = {"gate": gate_lbl, "pose": pose.replace("_%", "")}
                for sub in subunits:
                    row[sub] = df_delta.loc[pose, sub]
                stack_rows.append(row)
        pd.DataFrame(stack_rows).to_csv(
            p_dir / f"delta_all_gates_{p_atom}.csv", index=False)

    # ── joint state delta analysis ───────────────────────────────────────────
    print("  Joint state delta analysis...")
    baseline_joint = output_dir / "summary_table_joint.csv"
    if baseline_joint.exists():
        df_base_j = pd.read_csv(baseline_joint, index_col=0)
        subunits_j = [c for c in df_base_j.columns if c in SUBUNIT_ORDER]
        states_j   = df_base_j.index.tolist()

        joint_dir = delta_dir / "joint"
        joint_dir.mkdir(exist_ok=True)

        all_deltas_j = {}
        for folder, human_label in gate_labels.items():
            gate_csv = output_dir / folder / "summary_table_joint.csv"
            if not gate_csv.exists():
                continue
            df_gate_j  = pd.read_csv(gate_csv, index_col=0)
            df_delta_j = (df_gate_j[subunits_j] - df_base_j[subunits_j]).round(2)
            all_deltas_j[human_label] = df_delta_j
            df_delta_j.to_csv(joint_dir / f"delta_joint_{folder}.csv")

        if all_deltas_j:
            # stacked CSV
            stack_j = []
            for gate_lbl, df_d in all_deltas_j.items():
                for state in states_j:
                    row = {"gate": gate_lbl, "state": state}
                    for sub in subunits_j:
                        row[sub] = df_d.loc[state, sub]
                    stack_j.append(row)
            pd.DataFrame(stack_j).to_csv(joint_dir / "delta_all_gates_joint.csv", index=False)

            vmax_j = max(df.abs().max().max() for df in all_deltas_j.values())
            vmax_j = max(vmax_j, 5.0)
            n_subs_j  = len(subunits_j)
            n_states_j = len(states_j)

            def _make_joint_heatmap(deltas_dict, fname, title):
                n_g = len(deltas_dict)
                if n_g == 0:
                    return
                gate_items = list(deltas_dict.items())
                n_cols_per_gate = 3   # 3 state columns per gate
                n_rows          = 3   # 3 state rows
                # total cols: 3 per gate + 1 spacer between gates
                total_cols = n_cols_per_gate * n_g + (n_g - 1)

                fig = plt.figure(figsize=(4.0 * n_cols_per_gate * n_g, 2.8 * n_rows))
                # use gridspec for spacer columns
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(n_rows, total_cols,
                                       wspace=0.55, hspace=0.7,
                                       left=0.08,
                                       width_ratios=[1,1,1, 0.3, 1,1,1] if n_g==2 else None)

                for gi, (gate_lbl, df_d) in enumerate(gate_items):
                    col_offset = gi * (n_cols_per_gate + 1)  # +1 for spacer
                    for si, state in enumerate(states_j):
                        row = si // n_cols_per_gate
                        col = col_offset + (si % n_cols_per_gate)
                        ax  = fig.add_subplot(gs[row, col])
                        vals = df_d.loc[state, subunits_j].values.reshape(1, -1).astype(float)
                        im   = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax_j, vmax=vmax_j,
                                         aspect="auto")
                        ax.set_xticks(range(n_subs_j))
                        ax.set_xticklabels(subunits_j, fontsize=13)
                        ax.set_yticks([0])
                        ax.set_yticklabels([state.replace("+", "+\n")], fontsize=11,
                                           rotation=0, ha="right", va="center",
                                           multialignment="right")
                        ax.tick_params(axis="y", pad=6)
                        for xi, val in enumerate(vals[0]):
                            color = "white" if abs(val) > vmax_j * 0.6 else "black"
                            ax.text(xi, 0, f"{val:+.1f}%", ha="center", va="center",
                                    fontsize=12, color=color, fontweight="bold")
                        ax.set_title(state, fontsize=12, pad=6)
                        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

                    # big gate label centered above its 3x3 block
                    # use figure text positioned above the middle column of this gate
                    mid_col  = col_offset + 1
                    ax_mid   = fig.add_subplot(gs[0, mid_col])
                    ax_mid_pos = ax_mid.get_position()
                    fig.text(ax_mid_pos.x0 + ax_mid_pos.width / 2,
                             0.97, gate_lbl,
                             ha="center", va="top", fontsize=16, fontweight="bold",
                             transform=fig.transFigure)
                    ax_mid.remove()
                    # re-add the actual subplot
                    ax_re = fig.add_subplot(gs[0, mid_col])
                    si    = 1  # state index for middle top cell
                    state = states_j[si]
                    vals  = gate_items[gi][1].loc[state, subunits_j].values.reshape(1, -1).astype(float)
                    im    = ax_re.imshow(vals, cmap="RdBu_r", vmin=-vmax_j, vmax=vmax_j,
                                         aspect="auto")
                    ax_re.set_xticks(range(n_subs_j))
                    ax_re.set_xticklabels(subunits_j, fontsize=13)
                    ax_re.set_yticks([0])
                    ax_re.set_yticklabels([state.replace("+", "+\n")], fontsize=11,
                                          rotation=0, ha="right", va="center",
                                          multialignment="right")
                    ax_re.tick_params(axis="y", pad=6)
                    for xi, val in enumerate(vals[0]):
                        color = "white" if abs(val) > vmax_j * 0.6 else "black"
                        ax_re.text(xi, 0, f"{val:+.1f}%", ha="center", va="center",
                                   fontsize=12, color=color, fontweight="bold")
                    ax_re.set_title(state, fontsize=12, pad=6)
                    plt.colorbar(im, ax=ax_re, fraction=0.03, pad=0.02)

                plt.suptitle(f"Joint state delta vs all frames -- {channel_type} | {title}",
                             fontsize=15, y=1.03)
                fig.savefig(joint_dir / fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  Joint delta heatmap [{title}] -> {joint_dir / fname}")

            _make_joint_heatmap(all_deltas_j, "heatmap_delta_joint_all.png", "all gate subsets")
            crossing_j = {k: v for k, v in all_deltas_j.items() if "-1" not in k}
            _make_joint_heatmap(crossing_j, "heatmap_delta_joint_crossing.png", "crossing frames")
            pre_j = {k: v for k, v in all_deltas_j.items() if "-1" in k}
            _make_joint_heatmap(pre_j, "heatmap_delta_joint_pre_crossing.png", "pre-crossing frames")

    print(f"  Delta analysis complete -> {delta_dir}")


# -----------------------------------------------------------------------------
# Per-run breakdown
# -----------------------------------------------------------------------------

def _per_run_breakdown(runs_dir: Path, output_dir: Path,
                        channel_type: str,
                        exclude_runs: list[str] | None = None,
                        permeation_csv: str | None = None):
    """
    For each subunit, create a CSV with one row per run showing:
    nterm%, cterm%, none% for P4 and P5, plus LYS90 close % for P4 and P5.
    """
    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs)
    poses    = ["nterm", "cterm", "none"]

    if args.channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    for sub in SUBUNIT_ORDER:
        rows = []
        for run_dir in run_dirs:
            csv = run_dir / "pip_pose" / f"subunit_{sub}" / "summary.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if df.empty:
                continue
            total = df["count"].sum()

            row = {"run": run_dir.name}

            # P4 and P5 pose percentages
            for p_atom in ["P4", "P5"]:
                for pose in poses:
                    if p_atom == "P4":
                        mask = df["state"].str.startswith(f"p4_{pose}_")
                    else:
                        mask = df["state"].str.endswith(f"_p5_{pose}")
                    count = df.loc[mask, "count"].sum()
                    row[f"{p_atom}_{pose}_%"] = round(count / total * 100, 2) if total > 0 else 0.0

            # LYS90 close % -- raw % of all frames where P atom is close to LYS90
            # read from per_frame CSV directly
            pf_csv = run_dir / "pip_pose" / f"per_frame_{sub}.csv"
            if pf_csv.exists():
                df_pf = pd.read_csv(pf_csv)
                n_total_pf = len(df_pf)
                for p_atom, lys90_col in [
                    ("P4", "p4_lys90_close"),
                    ("P5", "p5_lys90_close"),
                ]:
                    if lys90_col in df_pf.columns and n_total_pf > 0:
                        n_close = int(df_pf[lys90_col].sum())
                        row[f"{p_atom}_LYS90_%"] = round(n_close / n_total_pf * 100, 2)
                    else:
                        row[f"{p_atom}_LYS90_%"] = float("nan")
            else:
                row["P4_LYS90_%"] = float("nan")
                row["P5_LYS90_%"] = float("nan")

            rows.append(row)

        if not rows:
            continue

        df_out = pd.DataFrame(rows)

        # merge permeation data if provided
        if permeation_csv:
            try:
                df_perm = pd.read_csv(permeation_csv)
                # normalize run name column
                run_col = [c for c in df_perm.columns if c.lower() == "run"][0]
                df_perm = df_perm.rename(columns={run_col: "run"})
                # strip leading zeros and uppercase: RUN01 -> RUN1
                df_perm["run"] = df_perm["run"].str.upper().str.replace(r"RUN0*", "RUN", regex=True)
                df_out["run"]  = df_out["run"].str.upper().str.replace(r"RUN0*", "RUN", regex=True)
                # keep only columns that exist
                perm_cols = ["run"] + [c for c in ["HBC_Permeations", "Final_Permeations",
                                                     "HBC_Per_Frame", "Final_Per_Frame"]
                                        if c in df_perm.columns]
                df_out = df_out.merge(df_perm[perm_cols], on="run", how="left")
                n_matched = df_out["HBC_Permeations"].notna().sum() if "HBC_Permeations" in df_out else 0
                print(f"  Permeation data merged: {n_matched}/{len(df_out)} runs matched")
            except Exception as e:
                print(f"  Could not merge permeation data: {e}")
        # reorder columns nicely
        col_order = ["run", "HBC_Permeations", "Final_Permeations",
                     "HBC_Per_Frame", "Final_Per_Frame"]
        for p_atom in ["P4", "P5"]:
            for pose in poses:
                col_order.append(f"{p_atom}_{pose}_%")
            col_order.append(f"{p_atom}_LYS90_%")
        col_order = [c for c in col_order if c in df_out.columns]
        df_out    = df_out[col_order]

        out_path = output_dir / f"per_run_breakdown_subunit_{sub}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  Per-run breakdown subunit {sub} -> {out_path}")
        print(df_out.to_string(index=False))


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_combine_pip_pose(
    runs_dir:       str | Path,
    channel_type:   str,
    output_dir:     Optional[str | Path] = None,
    exclude_runs:   list[str] | None = None,
    permeation_csv: str | None = None,
):


    if channel_type == "G2":
        SUBUNIT_ORDER = ["2a", "2b", "2c", "2d"]
    else:
        SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

    runs_dir = Path(runs_dir)
    if output_dir is None:
        output_dir = runs_dir / "combined_pip_pose"
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("COMBINE PIP POSE -- ACROSS RUNS")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Output dir   : {output_dir}")
    if exclude_runs:
        print(f"  Excluding    : {exclude_runs}")
    print("=" * 60)

    # all frames
    _run_all_for_subset(runs_dir, output_dir, channel_type,
                        subset=None, label="all frames",
                        exclude_runs=exclude_runs)

    # gate subsets
    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs)
    for folder_name, human_label in GATE_SUBSETS:
        has_subset = any(
            (rd / "pip_pose" / folder_name / "summary.csv").exists()
            for rd in run_dirs
        )
        if has_subset:
            _run_all_for_subset(
                runs_dir,
                output_dir / folder_name,
                channel_type,
                subset=folder_name,
                label=human_label,
                exclude_runs=exclude_runs,
            )
        else:
            print(f"\n  [{human_label}] not found in any RUN -- skipping.")

    # delta analysis across gate subsets
    print("\n  Running delta analysis...")
    _delta_analysis(output_dir, channel_type)

    # per-run breakdown table
    print("\n  Making per-run breakdown tables...")
    _per_run_breakdown(runs_dir, output_dir, channel_type, exclude_runs,
                       permeation_csv=permeation_csv)

    print(f"\n✓ Combined pip pose results -> {output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine pip_pose results across multiple RUN folders.")
    parser.add_argument("--runs_dir",     required=True)
    parser.add_argument("--channel_type", required=True)
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--exclude_runs", nargs="+", default=None,
                        help="Run folders to exclude, e.g. --exclude_runs RUN4 RUN7")
    parser.add_argument("--permeation_csv", default=None,
                        help="Path to permeation CSV with columns Run, HBC_Permeations etc.")
    args = parser.parse_args()

    run_combine_pip_pose(
        runs_dir       = args.runs_dir,
        channel_type   = args.channel_type,
        output_dir     = args.output_dir,
        exclude_runs   = args.exclude_runs,
        permeation_csv = args.permeation_csv,
    )

# python3 combine_pip_pose.py --runs_dir .../G12     --channel_type G12
# python3 combine_pip_pose.py --runs_dir .../G12_ML  --channel_type G12_ML  --exclude_runs RUN4
# python3 combine_pip_pose.py --runs_dir .../G12_GAT --channel_type G12_GAT

'''
python3 combine_pip_pose.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --permeation_csv /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/permeation_summary/permeation_per_run.csv 

python3 combine_pip_pose.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML \
    --exclude_runs RUN4 \
    --permeation_csv /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/permeation_summary/permeation_per_run.csv

python3 combine_pip_pose.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT \
    --permeation_csv /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/permeation_summary/permeation_per_run.csv


python3 combine_pip_pose.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --permeation_csv /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/permeation_summary/permeation_per_run.csv 

'''