"""
combine_p4p5_proximity.py

Aggregates per-RUN P4/P5 proximity category results (produced by
residue_pip_p4p5_proximity.py) across all RUNs of a channel type, and
generates combined mean ± std bar plots plus a summary CSV/txt.

Expected input structure (one folder per RUN, each containing the output
of residue_pip_p4p5_proximity.py):

    <base_path>/RUN1/residue_pip_p4p5_proximity/category_counts.csv
    <base_path>/RUN2/residue_pip_p4p5_proximity/category_counts.csv
    ...

For each (pip_resid, residue_resid) pair, computes across all RUNs:
    mean ± std of pct_P4_only, pct_P5_only, pct_Both, pct_None

Output saved to <base_path>/combined_p4p5_proximity/:
    combined_category_counts.csv   — mean ± std per pair, plus n_runs
    combined_category_counts.txt   — same, human-readable
    barplot_pct_mean_std.png       — grouped bar plot, mean % with std error bars
    barplot_count_mean_std.png     — grouped bar plot, mean frame count with std error bars
    barplot_count_total.png        — grouped bar plot, raw counts summed across all RUNs
    barplot_pct_pooled.png         — grouped bar plot, pooled % across all RUNs

GATE THRESHOLD PROGRESSION (optional, one gate at a time)
------------------------------------------------------------
If you've also run residue_pip_p4p5_proximity.py with gate_filters (e.g.
"hbc", "upper_gl"), each RUN folder will contain <gate>_<threshold>/
sub-folders (hbc_5, hbc_6, ..., upper_gl_5, ...). Calling
run_gate_threshold_progression(base_path, channel_type, gate_name="hbc")
collects those across all RUNs and all thresholds, and produces ONE line
plot PER (pip_resid, residue_resid) pair, with:
    x-axis = gate threshold (Å)
    y-axis = % of frames
    4 lines = P4_only, P5_only, Both, None

Output saved to <base_path>/combined_p4p5_proximity/gate_progression/<gate_name>/:
    <gate_name>_progression.csv                       — full aggregated table
    PIP{pip}_{residue_pdb}_{gate_name}_progression.png — one line plot per pair

Standalone usage:
    # combined bar plots only:
    python3 combine_p4p5_proximity.py \\
        --base_path /path/to/GIRK12_WT \\
        --channel_type G12 \\
        --run_glob "RUN*"

    # combined bar plots + gate threshold progression line plots for hbc and upper_gl:
    python3 combine_p4p5_proximity.py \\
        --base_path /path/to/GIRK12_WT \\
        --channel_type G12 \\
        --run_glob "RUN*" \\
        --gate_names hbc upper_gl \\
        --gate_thresholds 5 6 7 8 9 10 11 12
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.converter import to_pdb

warnings.filterwarnings("ignore")

CATEGORY_COLORS = {
    "P4_only": "steelblue",
    "P5_only": "darkorange",
    "Both": "mediumseagreen",
    "None": "lightgray",
}
CATEGORIES = ["P4_only", "P5_only", "Both", "None"]


# -----------------------------------------------------------------------------
# Collect per-RUN data
# -----------------------------------------------------------------------------

def _collect_run_csvs(base_path: Path, run_glob: str) -> pd.DataFrame:
    """
    Reads category_counts.csv from every matching RUN folder and
    concatenates into one long DataFrame, tagging each row with its run.
    """
    rows = []
    run_dirs = sorted(base_path.glob(run_glob))
    if not run_dirs:
        print(f"[WARN] No run folders matched '{run_glob}' under {base_path}")

    for run_dir in run_dirs:
        csv_path = run_dir / "residue_pip_p4p5_proximity" / "category_counts.csv"
        if not csv_path.exists():
            print(f"  [WARN] Missing {csv_path} — skipping")
            continue
        df = pd.read_csv(csv_path)
        df["run"] = run_dir.name
        rows.append(df)
        print(f"  Loaded {csv_path} ({len(df)} pairs)")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def _aggregate(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by (pip_resid, residue_resid) and computes, across RUNs:
      - mean ± std of each category's percentage
      - mean ± std of each category's raw frame count
      - total (summed) raw frame count across all RUNs
      - total n_frames summed across all RUNs
    """
    group_cols = ["pip_resid", "residue_resid"]
    pct_cols = [f"pct_{c}" for c in CATEGORIES]
    count_cols = [f"count_{c}" for c in CATEGORIES]

    agg_rows = []
    for (pip_resid, residue_resid), group in all_df.groupby(group_cols):
        row = {
            "pip_resid": pip_resid,
            "residue_resid": residue_resid,
            "n_runs": group["run"].nunique(),
            "total_n_frames": int(group["n_frames"].sum()),
        }
        for col in pct_cols:
            row[f"{col}_mean"] = round(group[col].mean(), 2)
            row[f"{col}_std"] = round(group[col].std(ddof=0), 2)
        for col in count_cols:
            row[f"{col}_mean"] = round(group[col].mean(), 2)
            row[f"{col}_std"] = round(group[col].std(ddof=0), 2)
            row[f"{col}_total"] = int(group[col].sum())

        # pooled percentage: sum of raw frames in category across all RUNs,
        # divided by sum of total frames across all RUNs
        for cat in CATEGORIES:
            total_count = row[f"count_{cat}_total"]
            row[f"pct_{cat}_pooled"] = round(100.0 * total_count / row["total_n_frames"], 2) if row["total_n_frames"] else 0.0

        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(["pip_resid", "residue_resid"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _pair_label(pip_resid: int, residue_resid: int, channel_type: str) -> str:
    pdb_label = to_pdb(residue_resid, channel_type)
    return f"PIP{pip_resid}\n{pdb_label}"


def _make_combined_barplot(df: pd.DataFrame, out_path: Path, channel_type: str,
                            value_prefix: str, ylabel: str, title: str):
    """
    Grouped bar plot: one group per pair, one bar per category, mean height
    with std error bars, and mean±std printed above each bar.

    value_prefix: "pct" or "count" — selects which aggregated columns to plot.
    """
    labels = [_pair_label(r.pip_resid, r.residue_resid, channel_type) for r in df.itertuples()]
    n_pairs = len(df)
    n_cats = len(CATEGORIES)

    x = np.arange(n_pairs)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.8 + 2), 5.5))

    for i, cat in enumerate(CATEGORIES):
        means = df[f"{value_prefix}_{cat}_mean"].values
        stds = df[f"{value_prefix}_{cat}_std"].values
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width=width, yerr=stds, capsize=3,
            label=cat, color=CATEGORY_COLORS.get(cat), edgecolor="black", linewidth=0.6,
        )
        fmt = "{:.1f}" if value_prefix == "pct" else "{:.0f}"
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + (0.5 if value_prefix == "pct" else max(means.max() * 0.01, 1)),
                f"{fmt.format(m)}\n±{fmt.format(s)}",
                ha="center", va="bottom", fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_cats)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_pooled_barplot(df: pd.DataFrame, out_path: Path, channel_type: str,
                          value_col_fmt: str, ylabel: str, title: str, as_pct: bool):
    """
    Grouped bar plot of pooled (summed across all RUNs) values — no error
    bars, since these are single aggregate totals, not per-RUN means.

    value_col_fmt: format string producing the column name per category,
                   e.g. "count_{}_total" or "pct_{}_pooled"
    """
    labels = [_pair_label(r.pip_resid, r.residue_resid, channel_type) for r in df.itertuples()]
    n_pairs = len(df)
    n_cats = len(CATEGORIES)

    x = np.arange(n_pairs)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.8 + 2), 5.5))

    for i, cat in enumerate(CATEGORIES):
        values = df[value_col_fmt.format(cat)].values
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width=width,
            label=cat, color=CATEGORY_COLORS.get(cat), edgecolor="black", linewidth=0.6,
        )
        fmt = "{:.1f}" if as_pct else "{:.0f}"
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (values.max() * 0.01 if values.max() else 0.1),
                fmt.format(v),
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_cats)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_txt(df: pd.DataFrame, out_path: Path, channel_type: str):
    with open(out_path, "w") as f:
        f.write(f"Combined P4/P5 proximity summary — {channel_type}\n")
        f.write("=" * 60 + "\n\n")
        for r in df.itertuples():
            pdb_label = to_pdb(r.residue_resid, channel_type)
            f.write(f"PIP {r.pip_resid} <-> residue {pdb_label} "
                    f"(raw resid {r.residue_resid}, n_runs={r.n_runs}, "
                    f"total_n_frames={r.total_n_frames})\n")
            for cat in CATEGORIES:
                pct_mean = getattr(r, f"pct_{cat}_mean")
                pct_std = getattr(r, f"pct_{cat}_std")
                count_mean = getattr(r, f"count_{cat}_mean")
                count_std = getattr(r, f"count_{cat}_std")
                count_total = getattr(r, f"count_{cat}_total")
                pct_pooled = getattr(r, f"pct_{cat}_pooled")
                f.write(f"  {cat:8s}: {pct_mean:5.1f}% ± {pct_std:4.1f}% (per-run mean)   "
                        f"| pooled: {pct_pooled:5.1f}%   "
                        f"| frames: {count_mean:7.1f} ± {count_std:6.1f}   "
                        f"| total across runs: {count_total}\n")
            f.write("\n")


# -----------------------------------------------------------------------------
# Gate-threshold progression (line plots: % vs threshold, per pair, per gate)
# -----------------------------------------------------------------------------

DEFAULT_GATE_THRESHOLDS = [5, 6, 7, 8, 9, 10, 11, 12]


def _collect_gate_threshold_csvs(base_path: Path, run_glob: str, gate_name: str,
                                  thresholds: List[float]) -> pd.DataFrame:
    """
    Reads category_counts.csv from every RUN's <gate_name>_<threshold> folder,
    for every threshold, and concatenates into one long DataFrame tagged with
    'run' and 'threshold'.

    Expected per-RUN structure:
        <base_path>/RUN1/residue_pip_p4p5_proximity/hbc_5/category_counts.csv
        <base_path>/RUN1/residue_pip_p4p5_proximity/hbc_6/category_counts.csv
        ...
    """
    rows = []
    run_dirs = sorted(base_path.glob(run_glob))
    if not run_dirs:
        print(f"[WARN] No run folders matched '{run_glob}' under {base_path}")

    for threshold in thresholds:
        threshold_tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        for run_dir in run_dirs:
            csv_path = (run_dir / "residue_pip_p4p5_proximity"
                        / f"{gate_name}_{threshold_tag}" / "category_counts.csv")
            if not csv_path.exists():
                print(f"  [WARN] Missing {csv_path} — skipping")
                continue
            df = pd.read_csv(csv_path)
            df["run"] = run_dir.name
            df["threshold"] = threshold
            rows.append(df)
            print(f"  Loaded {csv_path} ({len(df)} pairs)")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _aggregate_by_threshold(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by (pip_resid, residue_resid, threshold) and computes, across RUNs:
      - mean ± std of each category's percentage
      - pooled percentage (sum of category frames / sum of total frames, across all RUNs)
    """
    group_cols = ["pip_resid", "residue_resid", "threshold"]
    pct_cols = [f"pct_{c}" for c in CATEGORIES]
    count_cols = [f"count_{c}" for c in CATEGORIES]

    agg_rows = []
    for (pip_resid, residue_resid, threshold), group in all_df.groupby(group_cols):
        row = {
            "pip_resid": pip_resid,
            "residue_resid": residue_resid,
            "threshold": threshold,
            "n_runs": group["run"].nunique(),
            "total_n_frames": int(group["n_frames"].sum()),
        }
        for col in pct_cols:
            row[f"{col}_mean"] = round(group[col].mean(), 2)
            row[f"{col}_std"] = round(group[col].std(ddof=0), 2)
        for col in count_cols:
            row[f"{col}_total"] = int(group[col].sum())
        for cat in CATEGORIES:
            total_count = row[f"count_{cat}_total"]
            row[f"pct_{cat}_pooled"] = (
                round(100.0 * total_count / row["total_n_frames"], 2) if row["total_n_frames"] else 0.0
            )
        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(["pip_resid", "residue_resid", "threshold"]).reset_index(drop=True)


def _get_grand_total_frames(base_path: Path, run_glob: str) -> Optional[int]:
    """
    Sums the (unfiltered) n_frames across all RUNs, using the full-run
    category_counts.csv (not gate-filtered). Since n_frames is the same for
    every pair within a RUN, only the first pair's value per RUN is counted.
    """
    full_df = _collect_run_csvs(base_path, run_glob)
    if full_df.empty:
        return None
    per_run = full_df.groupby("run")["n_frames"].first()
    return int(per_run.sum())


def _get_frames_used_per_threshold(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each threshold, sums the gate-filtered n_frames across all RUNs
    (using the first pair's value per RUN/threshold, since n_frames is the
    same for every pair given the same gate mask).

    Returns a DataFrame with columns: threshold, frames_used.
    """
    per_run_threshold = all_df.groupby(["threshold", "run"])["n_frames"].first().reset_index()
    frames_used = per_run_threshold.groupby("threshold")["n_frames"].sum().reset_index()
    frames_used = frames_used.rename(columns={"n_frames": "frames_used"})
    return frames_used


def _make_gate_progression_lineplot(df_pair: pd.DataFrame, pip_resid: int, residue_resid: int,
                                     gate_name: str, channel_type: str, out_path: Path,
                                     value_suffix: str = "pooled",
                                     frames_used_df: Optional[pd.DataFrame] = None,
                                     grand_total_frames: Optional[int] = None):
    """
    Line plot for ONE (pip_resid, residue_resid) pair: x-axis = gate threshold,
    y-axis = % of frames, one line per category (P4_only, P5_only, Both, None).

    value_suffix: "pooled" (uses pct_{cat}_pooled) or "mean" (uses pct_{cat}_mean,
    with std shaded/error-barred).

    If frames_used_df and grand_total_frames are given, each x-tick label is
    annotated with the raw frame count that passed the gate at that
    threshold, and what % of the total (unfiltered) frames that represents.
    """
    df_pair = df_pair.sort_values("threshold")
    thresholds = df_pair["threshold"].values
    pdb_label = to_pdb(residue_resid, channel_type)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cat in CATEGORIES:
        y = df_pair[f"pct_{cat}_{value_suffix}"].values
        color = CATEGORY_COLORS.get(cat)
        if value_suffix == "mean":
            yerr = df_pair[f"pct_{cat}_std"].values
            ax.errorbar(thresholds, y, yerr=yerr, marker="o", capsize=3,
                        label=cat, color=color, linewidth=1.8)
        else:
            ax.plot(thresholds, y, marker="o", label=cat, color=color, linewidth=1.8)

        for xt, yt in zip(thresholds, y):
            ax.annotate(f"{yt:.1f}", (xt, yt), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=color)

    ax.set_xlabel(f"{gate_name.upper()} gate threshold (Å)", fontsize=10)
    ax.set_ylabel("% of frames", fontsize=10)
    ax.set_title(f"{channel_type} — PIP{pip_resid} / {pdb_label} — {gate_name.upper()} threshold progression",
                 fontsize=10.5, fontweight="bold")
    ax.set_xticks(thresholds)

    if frames_used_df is not None and grand_total_frames:
        tick_labels = []
        for t in thresholds:
            match = frames_used_df.loc[frames_used_df["threshold"] == t, "frames_used"]
            if len(match):
                n = int(match.iloc[0])
                pct = 100.0 * n / grand_total_frames
                tick_labels.append(f"{t:g}\nn={n}\n({pct:.1f}%)")
            else:
                tick_labels.append(f"{t:g}")
        ax.set_xticklabels(tick_labels, fontsize=7.5)
    else:
        ax.set_xticklabels([f"{t:g}" for t in thresholds])

    ax.set_ylim(0, 108)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_gate_threshold_progression(
    base_path: str,
    channel_type: str,
    gate_name: str,
    run_glob: str = "RUN*",
    thresholds: List[float] = DEFAULT_GATE_THRESHOLDS,
    value_suffix: str = "pooled",
) -> pd.DataFrame:
    """
    For a given gate (e.g. "hbc" or "upper_gl"), collects category_counts.csv
    across all RUNs and all thresholds, aggregates per (pair, threshold), and
    produces ONE line plot per (pip_resid, residue_resid) pair showing how
    P4_only / P5_only / Both / None % progresses as the gate threshold increases.

    Output saved to <base_path>/combined_p4p5_proximity/gate_progression/<gate_name>/:
        <gate_name>_progression.csv   — full aggregated table (all pairs, all thresholds,
                                         plus frames_used and pct_of_total_frames per threshold)
        PIP{pip}_{residue_pdb}_{gate_name}_progression.png   — one plot per pair, with
                                         raw frame count + % of total frames shown per threshold
    """
    base_path = Path(base_path)
    out_dir = base_path / "combined_p4p5_proximity" / "gate_progression" / gate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {gate_name} category_counts.csv across thresholds {thresholds} "
          f"under {base_path} (glob='{run_glob}')...")
    all_df = _collect_gate_threshold_csvs(base_path, run_glob, gate_name, thresholds)
    if all_df.empty:
        print("No data found — nothing to aggregate.")
        return pd.DataFrame()

    agg_df = _aggregate_by_threshold(all_df)

    # frame-count context: how many frames passed the gate at each threshold,
    # and what % of the total (unfiltered) frames that represents
    grand_total_frames = _get_grand_total_frames(base_path, run_glob)
    frames_used_df = _get_frames_used_per_threshold(all_df)
    if grand_total_frames:
        frames_used_df["pct_of_total_frames"] = round(
            100.0 * frames_used_df["frames_used"] / grand_total_frames, 2
        )
        agg_df = agg_df.merge(frames_used_df, on="threshold", how="left")
        print(f"Grand total frames across all RUNs (unfiltered): {grand_total_frames}")
        for _, r in frames_used_df.iterrows():
            print(f"  threshold {r['threshold']:g}: {int(r['frames_used'])} frames "
                  f"({r['pct_of_total_frames']:.1f}% of total)")
    else:
        print("[WARN] Could not determine grand total frame count — "
              "frame-count annotations will be skipped on plots")

    csv_path = out_dir / f"{gate_name}_progression.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    for (pip_resid, residue_resid), df_pair in agg_df.groupby(["pip_resid", "residue_resid"]):
        pdb_label = to_pdb(residue_resid, channel_type).replace(".", "_")
        plot_path = out_dir / f"PIP{pip_resid}_{pdb_label}_{gate_name}_progression.png"
        _make_gate_progression_lineplot(
            df_pair, pip_resid, residue_resid, gate_name, channel_type, plot_path,
            value_suffix=value_suffix,
            frames_used_df=frames_used_df if grand_total_frames else None,
            grand_total_frames=grand_total_frames,
        )
        print(f"Saved -> {plot_path}")

    return agg_df


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_combine_p4p5_proximity(base_path: str, channel_type: str, run_glob: str = "RUN*") -> pd.DataFrame:
    base_path = Path(base_path)
    out_dir = base_path / "combined_p4p5_proximity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting per-RUN category_counts.csv under {base_path} (glob='{run_glob}')...")
    all_df = _collect_run_csvs(base_path, run_glob)
    if all_df.empty:
        print("No data found — nothing to aggregate.")
        return pd.DataFrame()

    agg_df = _aggregate(all_df)

    csv_path = out_dir / "combined_category_counts.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    txt_path = out_dir / "combined_category_counts.txt"
    _write_summary_txt(agg_df, txt_path, channel_type)
    print(f"Saved -> {txt_path}")

    plot_pct_path = out_dir / "barplot_pct_mean_std.png"
    _make_combined_barplot(
        agg_df, plot_pct_path, channel_type,
        value_prefix="pct",
        ylabel="Mean % of frames (± std across RUNs)",
        title=f"{channel_type} — P4 / P5 / Both / None proximity, % of frames (mean ± std)",
    )
    print(f"Saved -> {plot_pct_path}")

    plot_count_path = out_dir / "barplot_count_mean_std.png"
    _make_combined_barplot(
        agg_df, plot_count_path, channel_type,
        value_prefix="count",
        ylabel="Mean frame count per RUN (± std)",
        title=f"{channel_type} — P4 / P5 / Both / None proximity, frame counts (mean ± std)",
    )
    print(f"Saved -> {plot_count_path}")

    plot_total_count_path = out_dir / "barplot_count_total.png"
    _make_pooled_barplot(
        agg_df, plot_total_count_path, channel_type,
        value_col_fmt="count_{}_total",
        ylabel="Total frame count (summed across all RUNs)",
        title=f"{channel_type} — P4 / P5 / Both / None proximity, raw counts summed across all RUNs",
        as_pct=False,
    )
    print(f"Saved -> {plot_total_count_path}")

    plot_pooled_pct_path = out_dir / "barplot_pct_pooled.png"
    _make_pooled_barplot(
        agg_df, plot_pooled_pct_path, channel_type,
        value_col_fmt="pct_{}_pooled",
        ylabel="% of frames (pooled: total category frames / total frames across all RUNs)",
        title=f"{channel_type} — P4 / P5 / Both / None proximity, pooled % across all RUNs",
        as_pct=True,
    )
    print(f"Saved -> {plot_pooled_pct_path}")

    return agg_df


# -----------------------------------------------------------------------------
# Standalone CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate P4/P5 proximity category results across multiple RUNs."
    )
    ap.add_argument("--base_path", required=True, help="Path containing RUN1, RUN2, ... folders")
    ap.add_argument("--channel_type", required=True, help="Channel type for to_pdb() conversion, e.g. G12, G2")
    ap.add_argument("--run_glob", default="RUN*", help="Glob pattern for run folders (default: RUN*)")
    ap.add_argument("--gate_names", nargs="*", default=[],
                     help="Gate names to also produce threshold-progression line plots for, "
                          "e.g. --gate_names hbc upper_gl")
    ap.add_argument("--gate_thresholds", nargs="*", type=float, default=DEFAULT_GATE_THRESHOLDS,
                     help="Thresholds to use for the gate progression plots (default: 5-12)")
    args = ap.parse_args()

    run_combine_p4p5_proximity(
        base_path=args.base_path,
        channel_type=args.channel_type,
        run_glob=args.run_glob,
    )

    for gate_name in args.gate_names:
        run_gate_threshold_progression(
            base_path=args.base_path,
            channel_type=args.channel_type,
            gate_name=gate_name,
            run_glob=args.run_glob,
            thresholds=args.gate_thresholds,
        )


if __name__ == "__main__":
    main()
'''
python3 combine_p4p5_proximity.py \
    --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --run_glob "RUN*" \
    --gate_names hbc upper_gl \
    --gate_thresholds 5 6 7 8 9 10 11 12
'''