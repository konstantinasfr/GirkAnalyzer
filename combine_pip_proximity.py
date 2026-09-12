"""
combine_pip_proximity.py
=========================
Combines pip_proximity results across multiple RUN folders into global summaries.

Processes both the main proximity_summary.csv and the 8 gate subset folders:
    asn_gate_frames/       (ion crosses ASN gate, region 2->3)
    non_asn_gate_frames/
    asn_gate_m1_frames/    (frame before ASN gate crossing)
    non_asn_gate_m1_frames/
    hbc_gate_frames/       (ion crosses HBC gate, region 3->4)
    non_hbc_gate_frames/
    hbc_gate_m1_frames/    (frame before HBC gate crossing)
    non_hbc_gate_m1_frames/

Usage
-----
    python combine_pip_proximity.py \
        --runs_dir     /path/to/G12 \
        --channel_type G12 \
        --output_dir   /path/to/G12/combined_pip_proximity \
        --exclude_runs RUN4

    # G2 example (custom folder name, resids auto-detected):
    python combine_pip_proximity.py \
        --runs_dir     /path/to/G2 \
        --channel_type G2 \
        --pip_folder   pip_proximity_5

Output structure
----------------
    output_dir/
        PIP_1301/P4/
            global_summary_raw.csv
            ...
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 8 gate subsets: (folder_name, human_label)
SUBSET_FOLDERS = [
    ("asn_gate_frames",         "ASN gate (end2)"),
    ("non_asn_gate_frames",     "non-ASN gate (end2)"),
    ("asn_gate_m1_frames",      "ASN gate -1 (pre-end2)"),
    ("non_asn_gate_m1_frames",  "non-ASN gate -1 (pre-end2)"),
    ("hbc_gate_frames",         "HBC gate (end3)"),
    ("non_hbc_gate_frames",     "non-HBC gate (end3)"),
    ("hbc_gate_m1_frames",      "HBC gate -1 (pre-end3)"),
    ("non_hbc_gate_m1_frames",  "non-HBC gate -1 (pre-end3)"),
]

ATOM_BAR_COLORS = {
    "P1": "mediumseagreen",
    "P4": "darkorange",
    "P5": "steelblue",
}

def _atom_color(atom_name: str) -> str:
    return ATOM_BAR_COLORS.get(atom_name, "steelblue")


# -----------------------------------------------------------------------------
# Auto-detect PIP resids
# -----------------------------------------------------------------------------

def _detect_pip_resids(runs_dir: Path, pip_folder: str) -> list[int]:
    """
    Scan all RUN* subdirectories for folders matching PIP_<resid>_* or PIP_<resid>
    inside pip_folder, and return sorted unique resids.
    """
    resids = set()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.upper().startswith("RUN"):
            continue
        pip_dir = run_dir / pip_folder
        if not pip_dir.exists():
            continue
        for entry in pip_dir.iterdir():
            if not entry.is_dir():
                continue
            # match PIP_<resid> or PIP_<resid>_<subunit>
            parts = entry.name.split("_")
            if len(parts) >= 2 and parts[0] == "PIP":
                try:
                    resids.add(int(parts[1]))
                except ValueError:
                    pass
    if resids:
        detected = sorted(resids)
        print(f"  Auto-detected PIP resids: {detected}")
        return detected
    # fallback
    print("  Could not auto-detect PIP resids, falling back to [1301,1302,1303,1304]")
    return [1301, 1302, 1303, 1304]


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


def _get_total_frames_main(csv_dir: Path) -> int:
    summary = csv_dir / "proximity_summary.csv"
    if not summary.exists():
        return 0
    df = pd.read_csv(summary)
    if "n_total_frames" in df.columns and not df.empty:
        return int(df["n_total_frames"].iloc[0])
    return _total_from_stats(df)


def _total_from_stats(df: pd.DataFrame) -> int:
    valid = df[(df["frames_close"] > 0) & (df["pct_close"] > 0)]
    if valid.empty:
        return 0
    best = valid.loc[valid["frames_close"].idxmax()]
    return int(round(best["frames_close"] / (best["pct_close"] / 100.0)))


def _make_barplot(labels, values, ylabel, title, out_path, color="steelblue"):
    if not labels:
        return
    pairs       = sorted(zip(values, labels), reverse=True)
    vals_sorted = [p[0] for p in pairs]
    labs_sorted = [p[1] for p in pairs]

    fig_w = max(7, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x_pos = np.arange(len(labs_sorted))
    bars  = ax.bar(x_pos, vals_sorted, color=color, edgecolor="white", alpha=0.85)

    max_val = max(vals_sorted) if vals_sorted else 1
    for bar, val in zip(bars, vals_sorted):
        fmt = f"{val:.1f}" if isinstance(val, float) else str(val)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                fmt, va="bottom", ha="center", fontsize=16)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labs_sorted, rotation=30, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(ylabel, fontsize=18)
    if "\n" in title:
        parts    = title.split("\n", 1)
        line1    = parts[0]
        subtitle = parts[1] if len(parts) > 1 else ""
        n_sub    = subtitle.count("\n") + 1
        pad      = 14 + n_sub * 16
        ax.set_title(line1, fontsize=20, pad=pad)
        ax.text(0.5, 1.01 + (n_sub - 1) * 0.045, subtitle,
                transform=ax.transAxes,
                fontsize=13, ha="center", va="bottom", linespacing=1.6)
    else:
        ax.set_title(title, fontsize=20)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Subunit detection
# -----------------------------------------------------------------------------

def _detect_subunit(records: list) -> str:
    """
    Auto-detect subunit from pdb_label column in loaded records.
    Labels look like ARG81.G1, LYS90.2a etc.
    Returns subunit string or 'unknown'.
    """
    subunit_map = {"G1": "G1", "2a": "2a", "2b": "2b", "2c": "2c", "2d": "2d"}
    for rec in records:
        df = rec["df"]
        if "pdb_label" not in df.columns:
            continue
        for lbl in df["pdb_label"].dropna():
            parts = str(lbl).split(".")
            if len(parts) >= 2:
                suffix = parts[-1]
                if suffix in subunit_map:
                    return subunit_map[suffix]
    return "unknown"


# -----------------------------------------------------------------------------
# Core combine logic
# -----------------------------------------------------------------------------

def _combine_summaries(records, out_dir, title_base,
                        threshold=4.0, label="", atom="P4"):
    if not records:
        print(f"    [{label}] No data -- skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    total_all_runs = sum(r["total_frames"] for r in records)

    # per-run breakdown
    breakdown_rows = []
    for rec in records:
        for _, row in rec["df"].iterrows():
            breakdown_rows.append({
                "run":                 rec["run"],
                "total_frames_in_run": rec["total_frames"],
                "raw_label":           row.get("raw_label", ""),
                "pdb_label":           row.get("pdb_label", ""),
                "frames_close":        int(row["frames_close"]),
                "pct_close":           float(row["pct_close"]),
                "min_dist":            row.get("min_dist", np.nan),
                "mean_dist":           row.get("mean_dist", np.nan),
                "max_dist":            row.get("max_dist", np.nan),
            })

    df_breakdown = pd.DataFrame(breakdown_rows)
    for label_col, suffix in [("raw_label", "raw"), ("pdb_label", "pdb")]:
        cols = ["run", "total_frames_in_run", label_col,
                "frames_close", "pct_close", "min_dist", "mean_dist", "max_dist"]
        (df_breakdown[cols]
            .sort_values([label_col, "run"])
            .to_csv(out_dir / f"per_run_breakdown_{suffix}.csv", index=False))

    # global summary
    agg = (df_breakdown
           .groupby("raw_label")
           .agg(
               pdb_label          = ("pdb_label",    "first"),
               frames_close_total = ("frames_close", "sum"),
               min_dist           = ("min_dist",      "min"),
               mean_dist          = ("mean_dist",     "mean"),
               max_dist           = ("max_dist",      "max"),
               n_runs             = ("run",            "nunique"),
           )
           .reset_index())
    agg["pct_close_global"]      = (agg["frames_close_total"] / total_all_runs * 100.0).round(3)
    agg["total_frames_all_runs"] = total_all_runs
    agg = agg.sort_values("frames_close_total", ascending=False)
    for col in ["min_dist", "mean_dist", "max_dist"]:
        agg[col] = agg[col].round(3)

    for label_col, suffix in [("raw_label", "raw"), ("pdb_label", "pdb")]:
        cols = [label_col, "frames_close_total", "pct_close_global",
                "total_frames_all_runs", "n_runs",
                "min_dist", "mean_dist", "max_dist"]
        agg[cols].to_csv(out_dir / f"global_summary_{suffix}.csv", index=False)

    # bar plots
    n_runs    = len(records)
    info      = (f"[{label}, {n_runs} runs, {total_all_runs} frames]"
                 if label else f"[{n_runs} runs, {total_all_runs} frames]")
    agg_nz    = agg[agg["pct_close_global"] >= 0.1]

    for label_col, suffix in [("raw_label", "raw"), ("pdb_label", "pdb")]:
        _make_barplot(agg_nz[label_col].tolist(),
                      agg_nz["frames_close_total"].tolist(),
                      "Total frames close (all runs)",
                      f"{title_base}\n{info}\nframes within {threshold} A",
                      out_dir / f"barplot_frames_{suffix}.png", _atom_color(atom))
        _make_barplot(agg_nz[label_col].tolist(),
                      agg_nz["pct_close_global"].tolist(),
                      "% of total frames (all runs)",
                      f"{title_base}\n{info}\n% frames within {threshold} A",
                      out_dir / f"barplot_percentage_{suffix}.png", _atom_color(atom))

    print(f"    [{label or 'all frames'}] done  "
          f"(runs={n_runs}, total_frames={total_all_runs})")


# -----------------------------------------------------------------------------
# Data loading  (pip_folder passed through everywhere)
# -----------------------------------------------------------------------------

def _load_records_main(runs_dir, pip_resid, atom, pip_folder, exclude_runs=None):
    records  = []
    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs)
    for run_dir in run_dirs:
        csv_dir  = run_dir / pip_folder / f"PIP_{pip_resid}" / atom
        # also try PIP_<resid>_<subunit> layout
        if not csv_dir.exists():
            matches = list((run_dir / pip_folder).glob(f"PIP_{pip_resid}_*/{atom}"))
            if matches:
                csv_dir = matches[0]
        csv_path = csv_dir / "proximity_summary.csv"
        if not csv_path.exists():
            print(f"      MISSING: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        tf = _get_total_frames_main(csv_dir)
        records.append({"run": run_dir.name, "total_frames": tf, "df": df})
    print(f"      -> {len(records)} runs loaded (main)")
    return records


def _load_records_subset(runs_dir, pip_resid, atom, subset, pip_folder, exclude_runs=None):
    records  = []
    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs)
    for run_dir in run_dirs:
        base = run_dir / pip_folder / f"PIP_{pip_resid}" / atom
        if not base.exists():
            matches = list((run_dir / pip_folder).glob(f"PIP_{pip_resid}_*/{atom}"))
            if matches:
                base = matches[0]
        csv_path = base / subset / "proximity_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        tf = _total_from_stats(df)
        records.append({"run": run_dir.name, "total_frames": tf, "df": df})
    print(f"      -> {len(records)} runs loaded ({subset})")
    return records


def _load_combination_records(runs_dir, pip_resid, atom, pip_folder,
                               subset=None, exclude_runs=None):
    records  = []
    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs)
    for run_dir in run_dirs:
        base = run_dir / pip_folder / f"PIP_{pip_resid}" / atom
        if not base.exists():
            matches = list((run_dir / pip_folder).glob(f"PIP_{pip_resid}_*/{atom}"))
            if matches:
                base = matches[0]
        if subset:
            base = base / subset
        combo_dir  = base / "combination_analysis"
        combo_raw  = combo_dir / "combination_counts_raw.csv"
        combo_pdb  = combo_dir / "combination_counts_pdb.csv"
        count_dist = combo_dir / "count_distribution.csv"

        if not combo_raw.exists() or not count_dist.exists():
            continue
        df_raw  = pd.read_csv(combo_raw)
        df_pdb  = pd.read_csv(combo_pdb) if combo_pdb.exists() else df_raw.copy()
        df_dist = pd.read_csv(count_dist)
        if df_raw.empty or df_dist.empty:
            continue
        tf = int(df_dist["frames"].sum())
        if tf == 0:
            continue
        records.append({"run": run_dir.name, "total_frames": tf,
                         "combo_raw": df_raw, "combo_pdb": df_pdb,
                         "count_dist": df_dist})
    print(f"      -> {len(records)} runs loaded (combo{' '+subset if subset else ''})")
    return records


# -----------------------------------------------------------------------------
# Combination aggregation
# -----------------------------------------------------------------------------

def _aggregate_combinations(records, out_dir, title_base, threshold,
                              atom, label="", top_n=15):
    if not records:
        print(f"    [{label}] No combination data -- skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    total_all = sum(r["total_frames"] for r in records)
    color     = _atom_color(atom)
    title_info = (f"[{label}, {len(records)} runs, {total_all} frames]"
                  if label else f"[{len(records)} runs, {total_all} frames]")

    for key, suffix in [("combo_raw", "raw"), ("combo_pdb", "pdb")]:
        all_rows = []
        for rec in records:
            for _, row in rec[key].iterrows():
                all_rows.append({"combination": row["combination"],
                                  "n_residues": int(row["n_residues"]),
                                  "frames":     int(row["frames"])})
        if not all_rows:
            continue
        df  = pd.DataFrame(all_rows)
        agg = (df.groupby(["combination", "n_residues"])
                 .agg(frames=("frames", "sum"))
                 .reset_index())
        agg["pct"]                  = (agg["frames"] / total_all * 100).round(3)
        agg["total_frames_all_runs"] = total_all
        agg = agg.sort_values("frames", ascending=False)
        agg.to_csv(out_dir / f"global_combination_counts_{suffix}.csv", index=False)

        top    = agg.head(top_n)
        labels = top["combination"].tolist()
        pcts   = top["pct"].tolist()
        frames = top["frames"].tolist()

        for vals, ylabel, fsuffix in [
            (pcts,   "% of frames",      "pct"),
            (frames, "Number of frames", "frames"),
        ]:
            if not labels:
                continue
            fig_w = max(8, len(labels) * 0.7)
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            bars = ax.bar(range(len(labels)), vals, color=color,
                          edgecolor="white", alpha=0.85)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.1f}%" if fsuffix == "pct" else str(val)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        fmt, va="bottom", ha="center", fontsize=11)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
            ax.tick_params(axis="y", labelsize=13)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_title(f"{title_base}\n{title_info}\n"
                         f"Top combinations (threshold {threshold} A)", fontsize=14)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
            plt.tight_layout()
            fig.savefig(out_dir / f"barplot_top_combinations_{suffix}_{fsuffix}.png", dpi=150)
            plt.close(fig)

    # count distribution
    all_dist = []
    for rec in records:
        for _, row in rec["count_dist"].iterrows():
            all_dist.append({"n_residues_close": int(row["n_residues_close"]),
                              "frames":           int(row["frames"])})
    df_dist  = pd.DataFrame(all_dist)
    agg_dist = (df_dist.groupby("n_residues_close")
                       .agg(frames=("frames", "sum"))
                       .reset_index())
    agg_dist["pct"]                  = (agg_dist["frames"] / total_all * 100).round(3)
    agg_dist["total_frames_all_runs"] = total_all
    agg_dist = agg_dist.sort_values("n_residues_close")
    agg_dist.to_csv(out_dir / "global_count_distribution.csv", index=False)

    x_labels    = [str(k) for k in agg_dist["n_residues_close"].tolist()]
    pcts_dist   = agg_dist["pct"].tolist()
    frames_dist = agg_dist["frames"].tolist()

    for vals, ylabel, fsuffix in [
        (pcts_dist,   "% of frames",      "pct"),
        (frames_dist, "Number of frames", "frames"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.8), 5))
        bars = ax.bar(x_labels, vals, color=color, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1f}%" if fsuffix == "pct" else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    fmt, va="bottom", ha="center", fontsize=13)
        ax.set_xlabel("Number of residues within threshold", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f"{title_base}\nCount distribution (threshold {threshold} A)",
                     fontsize=16)
        ax.tick_params(labelsize=13)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
        plt.tight_layout()
        fig.savefig(out_dir / f"barplot_count_distribution_{fsuffix}.png", dpi=150)
        plt.close(fig)

    print(f"    [{label or 'all frames'}] combination done "
          f"(runs={len(records)}, total={total_all})")


# -----------------------------------------------------------------------------
# Per-PIP/atom orchestration
# -----------------------------------------------------------------------------

def _combine_for_pip_atom(runs_dir, pip_resid, atom, channel_type,
                           out_dir, pip_folder, threshold=4.0, exclude_runs=None):
    print(f"\n  PIP_{pip_resid} / {atom}")

    # main (all frames)
    records = _load_records_main(runs_dir, pip_resid, atom, pip_folder, exclude_runs)

    # auto-detect subunit and rename output folder
    subunit = _detect_subunit(records)
    if subunit != "unknown":
        out_dir = out_dir.parent.parent / f"subunit_{subunit}" / atom
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> detected subunit {subunit}, output: {out_dir}")

    title_base = f"PIP{pip_resid}:{atom} [subunit {subunit}] [{channel_type}]"

    _combine_summaries(records, out_dir, title_base,
                       threshold=threshold, label="all frames", atom=atom)
    combo_main = _load_combination_records(runs_dir, pip_resid, atom, pip_folder,
                                           exclude_runs=exclude_runs)
    _aggregate_combinations(combo_main, out_dir / "combination_analysis",
                            title_base, threshold, atom, label="all frames")
    _aggregate_combinations(combo_main, out_dir / "combination_analysis_top10",
                            title_base, threshold, atom,
                            label="all frames", top_n=10)

    # 8 gate subsets
    for folder_name, human_label in SUBSET_FOLDERS:
        sub_records = _load_records_subset(runs_dir, pip_resid, atom,
                                           folder_name, pip_folder, exclude_runs)
        _combine_summaries(sub_records, out_dir / folder_name, title_base,
                           threshold=threshold, label=human_label, atom=atom)
        combo_sub = _load_combination_records(runs_dir, pip_resid, atom, pip_folder,
                                              subset=folder_name,
                                              exclude_runs=exclude_runs)
        _aggregate_combinations(combo_sub,
                                out_dir / folder_name / "combination_analysis",
                                title_base, threshold, atom, label=human_label)
        _aggregate_combinations(combo_sub,
                                out_dir / folder_name / "combination_analysis_top10",
                                title_base, threshold, atom,
                                label=human_label, top_n=10)


# -----------------------------------------------------------------------------
# Delta analysis
# -----------------------------------------------------------------------------

def _delta_analysis_proximity(output_dir: Path, channel_type: str,
                               pip_resids: list, atoms: list):
    """
    For each subunit/atom, compute delta of pct_close_global vs all frames
    for each gate subset. Saves CSVs and heatmaps.
    """
    gate_folders = [
        ("asn_gate_frames",    "ASN gate"),
        ("asn_gate_m1_frames", "ASN gate -1"),
        ("hbc_gate_frames",    "HBC gate"),
        ("hbc_gate_m1_frames", "HBC gate -1"),
    ]

    # collect all subunit dirs
    subunit_dirs = sorted([d for d in output_dir.iterdir()
                           if d.is_dir() and d.name.startswith("subunit_")])
    if not subunit_dirs:
        # fallback to PIP dirs
        subunit_dirs = sorted([d for d in output_dir.iterdir()
                               if d.is_dir() and d.name.startswith("PIP_")])

    sub_results = []   # collect per-subunit data for combined plots

    for sub_dir in subunit_dirs:
        for atom in atoms:
            atom_dir = sub_dir / atom
            if not atom_dir.exists():
                continue

            delta_dir = atom_dir / "delta_analysis"
            delta_dir.mkdir(exist_ok=True)

            # load baseline
            baseline_csv = atom_dir / "global_summary_pdb.csv"
            if not baseline_csv.exists():
                continue
            df_base = pd.read_csv(baseline_csv).set_index("pdb_label")
            if "pct_close_global" not in df_base.columns:
                continue

            all_deltas     = {}
            all_deltas_combo = {}

            for folder, human_label in gate_folders:
                gate_csv = atom_dir / folder / "global_summary_pdb.csv"
                if not gate_csv.exists():
                    continue
                df_gate  = pd.read_csv(gate_csv).set_index("pdb_label")
                common   = df_base.index.intersection(df_gate.index)
                if common.empty:
                    continue
                delta    = (df_gate.loc[common, "pct_close_global"]
                            - df_base.loc[common, "pct_close_global"]).round(2)
                delta.name = human_label
                all_deltas[human_label] = delta
                delta.to_frame().to_csv(
                    delta_dir / f"delta_residues_{folder}.csv")

                combo_base = atom_dir / "combination_analysis" / "global_combination_counts_pdb.csv"
                combo_gate = atom_dir / folder / "combination_analysis" / "global_combination_counts_pdb.csv"
                if combo_base.exists() and combo_gate.exists():
                    cb = pd.read_csv(combo_base).set_index("combination")
                    cg = pd.read_csv(combo_gate).set_index("combination")
                    common_c = cb.index.intersection(cg.index)
                    if not common_c.empty:
                        dc = (cg.loc[common_c, "pct"] - cb.loc[common_c, "pct"]).round(2)
                        dc.name = human_label
                        all_deltas_combo[human_label] = dc
                        dc.to_frame().to_csv(
                            delta_dir / f"delta_combinations_{folder}.csv")

            if not all_deltas:
                continue

            df_stack = pd.DataFrame(all_deltas)
            df_stack.index.name = "residue"
            df_stack.to_csv(delta_dir / "delta_all_gates_residues.csv")

            if all_deltas_combo:
                df_stack_c = pd.DataFrame(all_deltas_combo)
                df_stack_c.index.name = "combination"
                df_stack_c.to_csv(delta_dir / "delta_all_gates_combinations.csv")

            base_title = f"Residue contact delta vs all frames | {sub_dir.name}/{atom} [{channel_type}]"
            _plot_delta_heatmap(df_stack,
                                delta_dir / "heatmap_delta_residues_all.png",
                                base_title)
            crossing_r = {k: v for k, v in all_deltas.items() if "-1" not in k}
            pre_r      = {k: v for k, v in all_deltas.items() if "-1" in k}
            if crossing_r:
                _plot_delta_heatmap(pd.DataFrame(crossing_r),
                                    delta_dir / "heatmap_delta_residues_crossing.png",
                                    f"Residue delta | crossing frames | {sub_dir.name}/{atom} [{channel_type}]")
            if pre_r:
                _plot_delta_heatmap(pd.DataFrame(pre_r),
                                    delta_dir / "heatmap_delta_residues_pre_crossing.png",
                                    f"Residue delta | pre-crossing frames | {sub_dir.name}/{atom} [{channel_type}]")

            if all_deltas_combo:
                df_stack_c = pd.DataFrame(all_deltas_combo)
                base_title_c = f"Combination delta vs all frames | {sub_dir.name}/{atom} [{channel_type}]"
                _plot_delta_heatmap(df_stack_c,
                                    delta_dir / "heatmap_delta_combinations_all.png",
                                    base_title_c, max_rows=20)
                crossing_c = {k: v for k, v in all_deltas_combo.items() if "-1" not in k}
                pre_c      = {k: v for k, v in all_deltas_combo.items() if "-1" in k}
                if crossing_c:
                    _plot_delta_heatmap(pd.DataFrame(crossing_c),
                                        delta_dir / "heatmap_delta_combinations_crossing.png",
                                        f"Combination delta | crossing frames | {sub_dir.name}/{atom} [{channel_type}]",
                                        max_rows=20)
                if pre_c:
                    _plot_delta_heatmap(pd.DataFrame(pre_c),
                                        delta_dir / "heatmap_delta_combinations_pre_crossing.png",
                                        f"Combination delta | pre-crossing frames | {sub_dir.name}/{atom} [{channel_type}]",
                                        max_rows=20)

            print(f"  Delta analysis -> {delta_dir}")
            sub_results.append({
                "subunit": sub_dir.name,
                "atom":    atom,
                "residues":     all_deltas,
                "combinations": all_deltas_combo,
            })

    # combined multi-subunit heatmaps
    print("  Making combined multi-subunit heatmaps...")
    subunit_order = ["subunit_2a", "subunit_2b", "subunit_2c", "subunit_2d"]

    for atom in atoms:
        for kind, key, max_r in [
            ("Residue",     "residues",     30),
            ("Combination", "combinations", 20),
        ]:
            for fsuffix, subset_label, subset_filter in [
                ("crossing",     "crossing frames",     lambda k: "-1" not in k),
                ("pre_crossing", "pre-crossing frames", lambda k: "-1" in k),
            ]:
                sub_data = {}
                for sub in subunit_order:
                    csv_path = output_dir / sub / atom / "delta_analysis" / f"delta_all_gates_{key}.csv"
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path, index_col=0)
                    cols = [c for c in df.columns if subset_filter(c)]
                    if not cols:
                        continue
                    sub_data[sub] = df[cols]

                if len(sub_data) < 1:
                    continue

                all_max = pd.concat(sub_data.values()).abs().max(axis=1)
                thresh  = 2.0 if (all_max >= 2.0).any() else 1.0
                keep    = set()
                for df_s in sub_data.values():
                    mx = df_s.abs().max(axis=1)
                    keep.update(mx[mx >= thresh].nlargest(max_r).index.tolist())
                keep = sorted(keep)
                if not keep:
                    continue

                n_subs  = len(sub_data)
                n_cols  = len(next(iter(sub_data.values())).columns)
                n_rows  = len(keep)
                vmax    = max(max(df.abs().max().max() for df in sub_data.values()), 5.0)

                fig, axes = plt.subplots(
                    1, n_subs,
                    figsize=(4 * n_cols * n_subs, max(3, n_rows * 0.65)),
                    sharey=True)
                if n_subs == 1:
                    axes = [axes]

                for ax, (sub_name, df_s) in zip(axes, sub_data.items()):
                    df_s = df_s.reindex(keep).fillna(0.0)
                    im   = ax.imshow(df_s.values, cmap="RdBu_r",
                                     vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set_xticks(range(n_cols))
                    ax.set_xticklabels(df_s.columns, fontsize=14,
                                       rotation=25, ha="right")
                    ax.set_yticks(range(n_rows))
                    ax.set_yticklabels(df_s.index, fontsize=12)
                    ax.set_title(sub_name.replace("subunit_", "Subunit "),
                                 fontsize=15, fontweight="bold", pad=8)
                    for i in range(n_rows):
                        for j in range(n_cols):
                            val   = df_s.values[i, j]
                            color = "white" if abs(val) > vmax * 0.6 else "black"
                            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                                    fontsize=12, color=color, fontweight="bold")
                    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                                 label="Δ% frames")

                plt.suptitle(
                    f"{kind} contact delta vs all frames | {atom} | {subset_label} | {channel_type}",
                    fontsize=16, y=1.02)
                plt.tight_layout()
                out_path = output_dir / f"combined_{kind.lower()}_{atom}_{fsuffix}.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  Combined heatmap -> {out_path}")


def _plot_delta_heatmap(df: pd.DataFrame, out_path: Path, title: str,
                         max_rows: int = 30):
    max_abs = df.abs().max(axis=1)
    threshold_pct = 2.0 if (max_abs >= 2.0).any() else 1.0
    df_plot = df[max_abs >= threshold_pct].copy()
    if df_plot.empty:
        df_plot = df
    if len(df_plot) > max_rows:
        df_plot = df_plot.loc[max_abs[df_plot.index].nlargest(max_rows).index]
    df_plot = df_plot.sort_values(df_plot.columns[0], ascending=False)

    n_rows = len(df_plot)
    n_cols = len(df_plot.columns)
    vmax   = max(df_plot.abs().max().max(), 5.0)

    row_h  = 0.7
    fig, ax = plt.subplots(figsize=(4.5 * n_cols, max(3, n_rows * row_h)))
    im = ax.imshow(df_plot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df_plot.columns, fontsize=15, rotation=25, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df_plot.index, fontsize=13)

    for i in range(n_rows):
        for j in range(n_cols):
            val   = df_plot.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Δ% frames")
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_combine_pip_proximity(
    runs_dir:     str | Path,
    channel_type: str,
    output_dir:   Optional[str | Path] = None,
    pip_resids:   List[int]  = None,
    atoms:        List[str]  = None,
    threshold:    float      = 4.0,
    exclude_runs: list[str] | None = None,
    pip_folder:   str        = "pip_proximity",
):
    runs_dir = Path(runs_dir)
    if atoms      is None: atoms      = ["P1", "P4", "P5"]
    if output_dir is None: output_dir = runs_dir / "combined_pip_proximity"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # auto-detect resids if not provided
    if pip_resids is None:
        pip_resids = _detect_pip_resids(runs_dir, pip_folder)

    print("\n" + "=" * 70)
    print("COMBINE PIP PROXIMITY -- ACROSS RUNS")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  PIP folder   : {pip_folder}")
    print(f"  PIP resids   : {pip_resids}")
    print(f"  Atoms        : {atoms}")
    print(f"  Output dir   : {output_dir}")
    if exclude_runs:
        print(f"  Excluding    : {exclude_runs}")
    print(f"  Subsets      : {[f for f, _ in SUBSET_FOLDERS]}")
    print("=" * 70)

    for pip_resid in pip_resids:
        for atom in atoms:
            _combine_for_pip_atom(
                runs_dir     = runs_dir,
                pip_resid    = pip_resid,
                atom         = atom,
                channel_type = channel_type,
                out_dir      = output_dir / f"PIP_{pip_resid}" / atom,
                pip_folder   = pip_folder,
                threshold    = threshold,
                exclude_runs = exclude_runs,
            )

    print("\n  Running delta analysis...")
    _delta_analysis_proximity(output_dir, channel_type, pip_resids, atoms)

    print(f"\n✓ Combined results saved to: {output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine pip_proximity results across multiple RUN folders.")
    parser.add_argument("--runs_dir",     required=True)
    parser.add_argument("--channel_type", required=True)
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--pip_resids",   nargs="+", type=int, default=None,
                        help="PIP residue IDs to process. If omitted, auto-detected from folder names.")
    parser.add_argument("--pip_folder",   default="pip_proximity",
                        help="Name of the pip_proximity subfolder inside each RUN dir "
                             "(default: pip_proximity). E.g. pip_proximity_5 for G2.")
    parser.add_argument("--atoms",        nargs="+", default=["P1", "P4", "P5"])
    parser.add_argument("--threshold",    type=float, default=4.0)
    parser.add_argument("--exclude_runs", nargs="+", default=None,
                        help="Run folders to exclude, e.g. --exclude_runs RUN4 RUN7")
    args = parser.parse_args()

    run_combine_pip_proximity(
        runs_dir     = args.runs_dir,
        channel_type = args.channel_type,
        output_dir   = args.output_dir,
        pip_resids   = args.pip_resids,
        atoms        = args.atoms,
        threshold    = args.threshold,
        exclude_runs = args.exclude_runs,
        pip_folder   = args.pip_folder,
    )

# ── Examples ──────────────────────────────────────────────────────────────────
# G12 (default folder, default resids):
# python3 combine_pip_proximity.py --runs_dir .../G12 --channel_type G12
#
# G12_ML (exclude RUN4):
# python3 combine_pip_proximity.py --runs_dir .../G12_ML --channel_type G12_ML --exclude_runs RUN4
#
# G2 (custom folder, resids auto-detected):
# python3 combine_pip_proximity.py --runs_dir .../G2 --channel_type G2 --pip_folder pip_proximity_5
#
# G2 (explicit resids):
# python3 combine_pip_proximity.py --runs_dir .../G2 --channel_type G2 \
#     --pip_folder pip_proximity_5 --pip_resids 1313 1314 1315 1316

'''
python3 combine_pip_proximity.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML \
    --exclude_runs RUN4

python3 combine_pip_proximity.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT

python3 combine_pip_proximity.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12

python3 combine_pip_proximity.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 --pip_folder pip_proximity_5

'''