"""
combine_two_residues_proximity.py

Aggregates per-RUN two-residue proximity category results (produced by
residue_pip_two_residues_proximity.py) across all RUNs of a channel type,
and generates combined mean ± std bar plots plus pooled bar plots and a
summary CSV/txt. Also supports gate-threshold progression line plots,
exactly like combine_p4p5_proximity.py.

Expected input structure (one folder per RUN):

    <base_path>/RUN1/residue_pip_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/category_counts.csv
    <base_path>/RUN2/residue_pip_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/category_counts.csv
    ...

e.g. for phosphate="P5", res_a_label="199", res_b_label="64":
    <base_path>/RUN1/residue_pip_two_residues_proximity/P5_64_199/category_counts.csv

For each (pip_resid, residue_a_resid, residue_b_resid) triple, computes
across all RUNs:
    mean ± std of pct_{res_a_label}_only, pct_{res_b_label}_only, pct_Both, pct_None
    pooled % (sum of category frames / sum of total frames, across all RUNs)

Output saved to <base_path>/combined_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/:
    combined_category_counts.csv   — mean ± std + pooled % per triple, plus n_runs
    combined_category_counts.txt   — same, human-readable
    barplot_pct_mean_std.png       — grouped bar plot, mean % with std error bars
    barplot_count_mean_std.png     — grouped bar plot, mean frame count with std error bars
    barplot_count_total.png        — grouped bar plot, raw counts summed across all RUNs
    barplot_pct_pooled.png         — grouped bar plot, pooled % across all RUNs

GATE THRESHOLD PROGRESSION (optional, one gate at a time)
------------------------------------------------------------
If you've also run residue_pip_two_residues_proximity.py with gate_filters,
each RUN's phosphate folder will contain <gate>_<threshold>/ sub-folders.
Calling run_gate_threshold_progression(..., gate_name="hbc") collects those
across all RUNs and all thresholds, and produces ONE line plot PER
(pip_resid, residue_a_resid, residue_b_resid) triple, with:
    x-axis = gate threshold (Å)
    y-axis = % of frames
    4 lines = {res_a_label}_only, {res_b_label}_only, Both, None
    x-tick labels also show the raw frame count + % of total frames used

Output saved to <base_path>/combined_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/gate_progression/<gate_name>/:
    <gate_name>_progression.csv
    PIP{pip}_{residue_pdb}_{gate_name}_progression.png

Standalone usage:
    # combined bar plots only:
    python3 combine_two_residues_proximity.py \\
        --base_path /path/to/GIRK12_WT \\
        --channel_type G12 \\
        --run_glob "RUN*" \\
        --phosphate P5 \\
        --res_a_label 199 \\
        --res_b_label 64

    # combined bar plots + gate threshold progression for hbc and upper_gl:
    python3 combine_two_residues_proximity.py \\
        --base_path /path/to/GIRK12_WT \\
        --channel_type G12 \\
        --run_glob "RUN*" \\
        --phosphate P5 \\
        --res_a_label 199 \\
        --res_b_label 64 \\
        --gate_names hbc upper_gl \\
        --gate_thresholds 5 6 7 8 9 10 11 12
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.converter import to_pdb

warnings.filterwarnings("ignore")

DEFAULT_GATE_THRESHOLDS = [5, 6, 7, 8, 9, 10, 11, 12]
CATEGORY_COLOR_CYCLE = ["steelblue", "darkorange", "mediumseagreen", "lightgray"]


def _categories(res_a_label: str, res_b_label: str) -> List[str]:
    return [f"{res_a_label}_only", f"{res_b_label}_only", "Both", "None"]


def _category_colors(categories: List[str]) -> dict:
    return dict(zip(categories, CATEGORY_COLOR_CYCLE))


def _phosphate_folder(phosphate: str, res_a_label: str, res_b_label: str) -> str:
    return f"{phosphate}_{res_b_label}_{res_a_label}"


# -----------------------------------------------------------------------------
# Collect per-RUN data
# -----------------------------------------------------------------------------

def _collect_run_csvs(base_path: Path, run_glob: str, phosphate_folder: str) -> pd.DataFrame:
    """
    Reads category_counts.csv from every matching RUN folder's phosphate
    sub-folder, and concatenates into one long DataFrame tagged with 'run'.
    """
    rows = []
    run_dirs = sorted(base_path.glob(run_glob))
    if not run_dirs:
        print(f"[WARN] No run folders matched '{run_glob}' under {base_path}")

    for run_dir in run_dirs:
        csv_path = run_dir / "residue_pip_two_residues_proximity" / phosphate_folder / "category_counts.csv"
        if not csv_path.exists():
            print(f"  [WARN] Missing {csv_path} — skipping")
            continue
        df = pd.read_csv(csv_path)
        df["run"] = run_dir.name
        rows.append(df)
        print(f"  Loaded {csv_path} ({len(df)} triples)")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _collect_gate_threshold_csvs(base_path: Path, run_glob: str, phosphate_folder: str,
                                  gate_name: str, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    run_dirs = sorted(base_path.glob(run_glob))
    if not run_dirs:
        print(f"[WARN] No run folders matched '{run_glob}' under {base_path}")

    for threshold in thresholds:
        threshold_tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        for run_dir in run_dirs:
            csv_path = (run_dir / "residue_pip_two_residues_proximity" / phosphate_folder
                        / f"{gate_name}_{threshold_tag}" / "category_counts.csv")
            if not csv_path.exists():
                print(f"  [WARN] Missing {csv_path} — skipping")
                continue
            df = pd.read_csv(csv_path)
            df["run"] = run_dir.name
            df["threshold"] = threshold
            rows.append(df)
            print(f"  Loaded {csv_path} ({len(df)} triples)")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def _aggregate(all_df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    group_cols = ["pip_resid", "residue_a_resid", "residue_b_resid"]
    pct_cols = [f"pct_{c}" for c in categories]
    count_cols = [f"count_{c}" for c in categories]

    agg_rows = []
    for keys, group in all_df.groupby(group_cols):
        pip_resid, residue_a_resid, residue_b_resid = keys
        row = {
            "pip_resid": pip_resid,
            "residue_a_resid": residue_a_resid,
            "residue_b_resid": residue_b_resid,
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
        for cat in categories:
            total_count = row[f"count_{cat}_total"]
            row[f"pct_{cat}_pooled"] = (
                round(100.0 * total_count / row["total_n_frames"], 2) if row["total_n_frames"] else 0.0
            )
        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(
        ["pip_resid", "residue_a_resid", "residue_b_resid"]
    ).reset_index(drop=True)


def _aggregate_by_threshold(all_df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    group_cols = ["pip_resid", "residue_a_resid", "residue_b_resid", "threshold"]
    pct_cols = [f"pct_{c}" for c in categories]
    count_cols = [f"count_{c}" for c in categories]

    agg_rows = []
    for keys, group in all_df.groupby(group_cols):
        pip_resid, residue_a_resid, residue_b_resid, threshold = keys
        row = {
            "pip_resid": pip_resid,
            "residue_a_resid": residue_a_resid,
            "residue_b_resid": residue_b_resid,
            "threshold": threshold,
            "n_runs": group["run"].nunique(),
            "total_n_frames": int(group["n_frames"].sum()),
        }
        for col in pct_cols:
            row[f"{col}_mean"] = round(group[col].mean(), 2)
            row[f"{col}_std"] = round(group[col].std(ddof=0), 2)
        for col in count_cols:
            row[f"{col}_total"] = int(group[col].sum())
        for cat in categories:
            total_count = row[f"count_{cat}_total"]
            row[f"pct_{cat}_pooled"] = (
                round(100.0 * total_count / row["total_n_frames"], 2) if row["total_n_frames"] else 0.0
            )
        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(
        ["pip_resid", "residue_a_resid", "residue_b_resid", "threshold"]
    ).reset_index(drop=True)


def _get_grand_total_frames(base_path: Path, run_glob: str, phosphate_folder: str) -> Optional[int]:
    full_df = _collect_run_csvs(base_path, run_glob, phosphate_folder)
    if full_df.empty:
        return None
    per_run = full_df.groupby("run")["n_frames"].first()
    return int(per_run.sum())


def _get_frames_used_per_threshold(all_df: pd.DataFrame) -> pd.DataFrame:
    per_run_threshold = all_df.groupby(["threshold", "run"])["n_frames"].first().reset_index()
    frames_used = per_run_threshold.groupby("threshold")["n_frames"].sum().reset_index()
    frames_used = frames_used.rename(columns={"n_frames": "frames_used"})
    return frames_used


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _aa_label(resid: int, channel_type: str, aa_name: str, aa_overrides: Optional[dict]) -> str:
    """
    Builds a label like "LYS199.2b". aa_overrides is keyed by the
    PDB-converted residue NUMBER (e.g. 52 for the G1 copy), not the raw
    MDAnalysis resid.
    """
    pdb_label = to_pdb(resid, channel_type)
    pdb_number_str = pdb_label.split(".")[0]
    try:
        pdb_number = int(pdb_number_str)
    except ValueError:
        pdb_number = None
    aa = (aa_overrides or {}).get(pdb_number, aa_name)
    return f"{aa}{pdb_label}"


def _triple_label(pip_resid: int, residue_a_resid: int, residue_b_resid: int, channel_type: str,
                   res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                   res_b_aa_overrides: Optional[dict] = None) -> str:
    label_a = _aa_label(residue_a_resid, channel_type, res_a_aa, None)
    label_b = _aa_label(residue_b_resid, channel_type, res_b_aa, res_b_aa_overrides)
    return f"PIP{pip_resid}\n{label_a}\n{label_b}"


def _make_combined_barplot(df: pd.DataFrame, out_path: Path, channel_type: str,
                            categories: List[str], category_colors: dict,
                            value_prefix: str, ylabel: str, title: str,
                            res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                            res_b_aa_overrides: Optional[dict] = None):
    labels = [_triple_label(r.pip_resid, r.residue_a_resid, r.residue_b_resid, channel_type,
                             res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides)
              for r in df.itertuples()]
    n_triples = len(df)
    n_cats = len(categories)

    x = np.arange(n_triples)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_triples * 1.8 + 2), 5.5))

    for i, cat in enumerate(categories):
        means = df[f"{value_prefix}_{cat}_mean"].values
        stds = df[f"{value_prefix}_{cat}_std"].values
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width=width, yerr=stds, capsize=3,
            label=cat, color=category_colors.get(cat), edgecolor="black", linewidth=0.6,
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
                          categories: List[str], category_colors: dict,
                          value_col_fmt: str, ylabel: str, title: str, as_pct: bool,
                          res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                          res_b_aa_overrides: Optional[dict] = None):
    labels = [_triple_label(r.pip_resid, r.residue_a_resid, r.residue_b_resid, channel_type,
                             res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides)
              for r in df.itertuples()]
    n_triples = len(df)
    n_cats = len(categories)

    x = np.arange(n_triples)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_triples * 1.8 + 2), 5.5))

    for i, cat in enumerate(categories):
        values = df[value_col_fmt.format(cat)].values
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width=width,
            label=cat, color=category_colors.get(cat), edgecolor="black", linewidth=0.6,
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


def _make_gate_progression_lineplot(df_triple: pd.DataFrame, pip_resid: int,
                                     residue_a_resid: int, residue_b_resid: int,
                                     gate_name: str, channel_type: str, out_path: Path,
                                     categories: List[str], category_colors: dict,
                                     value_suffix: str = "pooled",
                                     frames_used_df: Optional[pd.DataFrame] = None,
                                     grand_total_frames: Optional[int] = None,
                                     res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                                     res_b_aa_overrides: Optional[dict] = None):
    df_triple = df_triple.sort_values("threshold")
    thresholds = df_triple["threshold"].values
    label_a = _aa_label(residue_a_resid, channel_type, res_a_aa, None)
    label_b = _aa_label(residue_b_resid, channel_type, res_b_aa, res_b_aa_overrides)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cat in categories:
        y = df_triple[f"pct_{cat}_{value_suffix}"].values
        color = category_colors.get(cat)
        if value_suffix == "mean":
            yerr = df_triple[f"pct_{cat}_std"].values
            ax.errorbar(thresholds, y, yerr=yerr, marker="o", capsize=3,
                        label=cat, color=color, linewidth=1.8)
        else:
            ax.plot(thresholds, y, marker="o", label=cat, color=color, linewidth=1.8)

        for xt, yt in zip(thresholds, y):
            ax.annotate(f"{yt:.1f}", (xt, yt), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=color)

    ax.set_xlabel(f"{gate_name.upper()} gate threshold (Å)", fontsize=10)
    ax.set_ylabel("% of frames", fontsize=10)
    ax.set_title(f"{channel_type} — PIP{pip_resid} / {label_a} / {label_b} — {gate_name.upper()} threshold progression",
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


def _write_summary_txt(df: pd.DataFrame, out_path: Path, channel_type: str, categories: List[str]):
    with open(out_path, "w") as f:
        f.write(f"Combined two-residue proximity summary — {channel_type}\n")
        f.write("=" * 60 + "\n\n")
        for r in df.itertuples():
            pdb_a = to_pdb(r.residue_a_resid, channel_type)
            pdb_b = to_pdb(r.residue_b_resid, channel_type)
            f.write(f"PIP {r.pip_resid} <-> {pdb_a} / {pdb_b} "
                    f"(n_runs={r.n_runs}, total_n_frames={r.total_n_frames})\n")
            for cat in categories:
                pct_mean = getattr(r, f"pct_{cat}_mean")
                pct_std = getattr(r, f"pct_{cat}_std")
                count_mean = getattr(r, f"count_{cat}_mean")
                count_std = getattr(r, f"count_{cat}_std")
                count_total = getattr(r, f"count_{cat}_total")
                pct_pooled = getattr(r, f"pct_{cat}_pooled")
                f.write(f"  {cat:12s}: {pct_mean:5.1f}% ± {pct_std:4.1f}% (per-run mean)   "
                        f"| pooled: {pct_pooled:5.1f}%   "
                        f"| frames: {count_mean:7.1f} ± {count_std:6.1f}   "
                        f"| total across runs: {count_total}\n")
            f.write("\n")


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

def run_combine_two_residues_proximity(
    base_path: str,
    channel_type: str,
    phosphate: str = "P5",
    res_a_label: str = "199",
    res_b_label: str = "64",
    res_a_aa: str = "LYS",
    res_b_aa: str = "LYS",
    res_b_aa_overrides: Optional[dict] = None,
    run_glob: str = "RUN*",
) -> pd.DataFrame:
    base_path = Path(base_path)
    phosphate_folder = _phosphate_folder(phosphate, res_a_label, res_b_label)
    categories = _categories(res_a_label, res_b_label)
    category_colors = _category_colors(categories)

    out_dir = base_path / "combined_two_residues_proximity" / phosphate_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting per-RUN category_counts.csv under {base_path} "
          f"(glob='{run_glob}', phosphate_folder='{phosphate_folder}')...")
    all_df = _collect_run_csvs(base_path, run_glob, phosphate_folder)
    if all_df.empty:
        print("No data found — nothing to aggregate.")
        return pd.DataFrame()

    agg_df = _aggregate(all_df, categories)

    csv_path = out_dir / "combined_category_counts.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    txt_path = out_dir / "combined_category_counts.txt"
    _write_summary_txt(agg_df, txt_path, channel_type, categories)
    print(f"Saved -> {txt_path}")

    # amino-acid-labeled title fragment, matching the per-RUN script's style, e.g. "K199 / K64"
    title_a = f"{res_a_aa}{res_a_label}"
    title_b = f"{res_b_aa}{res_b_label}"

    plot_pct_path = out_dir / "barplot_pct_mean_std.png"
    _make_combined_barplot(
        agg_df, plot_pct_path, channel_type, categories, category_colors,
        value_prefix="pct",
        ylabel="Mean % of frames (± std across RUNs)",
        title=f"{channel_type} — {phosphate} close to {title_a} / {title_b} / Both / None (mean ± std)",
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {plot_pct_path}")

    plot_count_path = out_dir / "barplot_count_mean_std.png"
    _make_combined_barplot(
        agg_df, plot_count_path, channel_type, categories, category_colors,
        value_prefix="count",
        ylabel="Mean frame count per RUN (± std)",
        title=f"{channel_type} — {phosphate} close to {title_a} / {title_b} / Both / None, frame counts (mean ± std)",
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {plot_count_path}")

    plot_total_count_path = out_dir / "barplot_count_total.png"
    _make_pooled_barplot(
        agg_df, plot_total_count_path, channel_type, categories, category_colors,
        value_col_fmt="count_{}_total",
        ylabel="Total frame count (summed across all RUNs)",
        title=f"{channel_type} — {phosphate} close to {title_a} / {title_b} / Both / None, raw counts summed across all RUNs",
        as_pct=False,
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {plot_total_count_path}")

    plot_pooled_pct_path = out_dir / "barplot_pct_pooled.png"
    _make_pooled_barplot(
        agg_df, plot_pooled_pct_path, channel_type, categories, category_colors,
        value_col_fmt="pct_{}_pooled",
        ylabel="% of frames (pooled: total category frames / total frames across all RUNs)",
        title=f"{channel_type} — {phosphate} close to {title_a} / {title_b} / Both / None, pooled % across all RUNs",
        as_pct=True,
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {plot_pooled_pct_path}")

    return agg_df


def run_gate_threshold_progression(
    base_path: str,
    channel_type: str,
    gate_name: str,
    phosphate: str = "P5",
    res_a_label: str = "199",
    res_b_label: str = "64",
    res_a_aa: str = "LYS",
    res_b_aa: str = "LYS",
    res_b_aa_overrides: Optional[dict] = None,
    run_glob: str = "RUN*",
    thresholds: List[float] = DEFAULT_GATE_THRESHOLDS,
    value_suffix: str = "pooled",
) -> pd.DataFrame:
    base_path = Path(base_path)
    phosphate_folder = _phosphate_folder(phosphate, res_a_label, res_b_label)
    categories = _categories(res_a_label, res_b_label)
    category_colors = _category_colors(categories)

    out_dir = base_path / "combined_two_residues_proximity" / phosphate_folder / "gate_progression" / gate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {gate_name} category_counts.csv across thresholds {thresholds} "
          f"under {base_path} (glob='{run_glob}', phosphate_folder='{phosphate_folder}')...")
    all_df = _collect_gate_threshold_csvs(base_path, run_glob, phosphate_folder, gate_name, thresholds)
    if all_df.empty:
        print("No data found — nothing to aggregate.")
        return pd.DataFrame()

    agg_df = _aggregate_by_threshold(all_df, categories)

    grand_total_frames = _get_grand_total_frames(base_path, run_glob, phosphate_folder)
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

    for (pip_resid, residue_a_resid, residue_b_resid), df_triple in agg_df.groupby(
        ["pip_resid", "residue_a_resid", "residue_b_resid"]
    ):
        pdb_a = to_pdb(residue_a_resid, channel_type).replace(".", "_")
        plot_path = out_dir / f"PIP{pip_resid}_{pdb_a}_{gate_name}_progression.png"
        _make_gate_progression_lineplot(
            df_triple, pip_resid, residue_a_resid, residue_b_resid, gate_name, channel_type, plot_path,
            categories=categories, category_colors=category_colors,
            value_suffix=value_suffix,
            frames_used_df=frames_used_df if grand_total_frames else None,
            grand_total_frames=grand_total_frames,
            res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
        )
        print(f"Saved -> {plot_path}")

    return agg_df


# -----------------------------------------------------------------------------
# Standalone CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate two-residue proximity category results across multiple RUNs."
    )
    ap.add_argument("--base_path", required=True, help="Path containing RUN1, RUN2, ... folders")
    ap.add_argument("--channel_type", required=True, help="Channel type for to_pdb() conversion, e.g. G12, G2")
    ap.add_argument("--phosphate", default="P5", help="Which phosphate's folder to read (P4 or P5)")
    ap.add_argument("--res_a_label", default="199", help="Label used for residue A in filenames/columns")
    ap.add_argument("--res_b_label", default="64", help="Label used for residue B in filenames/columns")
    ap.add_argument("--res_a_aa", default="LYS", help="Amino acid code for residue A (default LYS)")
    ap.add_argument("--res_b_aa", default="LYS", help="Amino acid code for residue B (default LYS)")
    ap.add_argument("--res_b_aa_overrides", nargs="*", default=[],
                     help="Per-PDB-number amino acid overrides for residue B, e.g. --res_b_aa_overrides 52:ARG")
    ap.add_argument("--run_glob", default="RUN*", help="Glob pattern for run folders (default: RUN*)")
    ap.add_argument("--gate_names", nargs="*", default=[],
                     help="Gate names to also produce threshold-progression line plots for, "
                          "e.g. --gate_names hbc upper_gl")
    ap.add_argument("--gate_thresholds", nargs="*", type=float, default=DEFAULT_GATE_THRESHOLDS,
                     help="Thresholds to use for the gate progression plots (default: 5-12)")
    args = ap.parse_args()

    res_b_aa_overrides = {}
    for item in args.res_b_aa_overrides:
        resid_str, aa = item.split(":")
        res_b_aa_overrides[int(resid_str)] = aa

    run_combine_two_residues_proximity(
        base_path=args.base_path,
        channel_type=args.channel_type,
        phosphate=args.phosphate,
        res_a_label=args.res_a_label,
        res_b_label=args.res_b_label,
        res_a_aa=args.res_a_aa,
        res_b_aa=args.res_b_aa,
        res_b_aa_overrides=res_b_aa_overrides or None,
        run_glob=args.run_glob,
    )

    for gate_name in args.gate_names:
        run_gate_threshold_progression(
            base_path=args.base_path,
            channel_type=args.channel_type,
            gate_name=gate_name,
            phosphate=args.phosphate,
            res_a_label=args.res_a_label,
            res_b_label=args.res_b_label,
            res_a_aa=args.res_a_aa,
            res_b_aa=args.res_b_aa,
            res_b_aa_overrides=res_b_aa_overrides or None,
            run_glob=args.run_glob,
            thresholds=args.gate_thresholds,
        )


if __name__ == "__main__":
    main()

'''
python3 combine_two_residues_proximity.py \
    --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --run_glob "RUN*" \
    --phosphate P5 \
    --res_a_label 199 \
    --res_b_label 64 \
    --res_b_aa_overrides 52:ARG \
    --gate_names hbc upper_gl \
    --gate_thresholds 5 6 7 8 9 10 11 12

python3 combine_two_residues_proximity.py \
    --base_path /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --run_glob "RUN*" \
    --phosphate P4 \
    --res_a_label 199 \
    --res_b_label 64 \
    --res_b_aa_overrides 52:ARG \
    --gate_names hbc upper_gl \
    --gate_thresholds 5 6 7 8 9 10 11 12
'''