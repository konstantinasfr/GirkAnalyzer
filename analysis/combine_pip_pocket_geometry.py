"""
combine_pip_pocket_geometry.py
================================
Combines pip_pocket_geometry results across multiple RUN folders.
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

SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]

DIST_COLS = {
    "d12_Nterm_Cterm": "N-term ↔ C-term",
    "d13_Nterm_LYS90": "N-term ↔ LYS90",
    "d23_Cterm_LYS90": "C-term ↔ LYS90",
}
DIST_COLORS = {
    "d12_Nterm_Cterm": "#E53935",
    "d13_Nterm_LYS90": "#1E88E5",
    "d23_Cterm_LYS90": "#43A047",
}


def _sorted_run_dirs(runs_dir: Path) -> list:
    return sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.upper().startswith("RUN") and d.name[3:].isdigit()],
        key=lambda d: int(d.name[3:]),
    )


def _load_all_frames(runs_dir: Path, subunit: str) -> pd.DataFrame:
    """Load and concatenate per_frame.csv from all runs for a subunit."""
    run_dirs = _sorted_run_dirs(runs_dir)
    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / "pip_pocket_geometry" / f"subunit_{subunit}" / "per_frame.csv"
        if not csv.exists():
            print(f"    MISSING: {csv}")
            continue
        df = pd.read_csv(csv)
        df["run"] = run_dir.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f"    Subunit {subunit}: {len(dfs)} runs, {len(combined)} total frames")
    return combined


def _make_summary(df: pd.DataFrame, n_runs: int) -> pd.DataFrame:
    rows = []
    for col, label in DIST_COLS.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        rows.append({
            "distance":  col,
            "label":     label,
            "n_frames":  len(vals),
            "n_runs":    n_runs,
            "mean":      round(float(vals.mean()), 3),
            "median":    round(float(np.median(vals)), 3),
            "std":       round(float(vals.std()), 3),
            "min":       round(float(vals.min()), 3),
            "max":       round(float(vals.max()), 3),
            "q25":       round(float(np.percentile(vals, 25)), 3),
            "q75":       round(float(np.percentile(vals, 75)), 3),
        })
    return pd.DataFrame(rows)


def _boxplot(df: pd.DataFrame, title: str, out_path: Path):
    cols   = [c for c in DIST_COLS if c in df.columns]
    values = [df[c].dropna().values for c in cols]
    labels = [DIST_COLS[c] for c in cols]
    colors = [DIST_COLORS[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(values, patch_artist=True, labels=labels,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker=".", markersize=1, alpha=0.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, vals in enumerate(values):
        mean   = np.mean(vals)
        median = np.median(vals)
        ax.scatter(i+1, mean, marker="D", color="white", s=50,
                   zorder=5, edgecolors="black", linewidths=1)
        ax.text(i+1+0.15, mean,   f"μ={mean:.1f}Å",   va="center", fontsize=10)
        ax.text(i+1+0.15, median, f"m={median:.1f}Å", va="center", fontsize=10, color="grey")

    ax.set_ylabel("Distance (Å)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _violinplot(df: pd.DataFrame, title: str, out_path: Path):
    cols   = [c for c in DIST_COLS if c in df.columns]
    values = [df[c].dropna().values for c in cols]
    labels = [DIST_COLS[c] for c in cols]
    colors = [DIST_COLORS[c] for c in cols]

    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot(values, positions=range(1, len(cols)+1),
                          showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")

    ax.set_xticks(range(1, len(cols)+1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Distance (Å)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_combine_pip_pocket_geometry(
    runs_dir:    str,
    channel_type: str,
    output_dir:  Optional[str] = None,
):
    runs_dir = Path(runs_dir)
    if output_dir is None:
        output_dir = runs_dir / "combined_pip_pocket_geometry"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _sorted_run_dirs(runs_dir)
    n_runs   = len(run_dirs)

    print("\n" + "=" * 60)
    print("COMBINE PIP POCKET GEOMETRY")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Runs found   : {n_runs}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60)

    for sub in SUBUNIT_ORDER:
        df = _load_all_frames(runs_dir, sub)
        if df.empty:
            continue

        sub_dir = output_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)

        # save combined per-frame CSV
        df.to_csv(sub_dir / "per_frame_all_runs.csv", index=False)

        # summary
        summary = _make_summary(df, n_runs)
        summary.to_csv(sub_dir / "summary.csv", index=False)
        print(f"\n  Subunit {sub} summary:")
        print(summary[["distance","mean","median","std"]].to_string(index=False))

        # plots
        title = f"PIP2 pocket geometry — {channel_type} | Subunit {sub}\n[{n_runs} runs, {len(df)} frames]"
        _boxplot(df, title, sub_dir / "boxplot.png")
        _violinplot(df, title, sub_dir / "violin.png")
        print(f"  Plots saved → {sub_dir}")

    print(f"\n✓ Combined pocket geometry → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",     required=True)
    parser.add_argument("--channel_type", required=True)
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    run_combine_pip_pocket_geometry(
        runs_dir     = args.runs_dir,
        channel_type = args.channel_type,
        output_dir   = args.output_dir,
    )

'''
python3 combine_pip_pocket_geometry.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12

python3 combine_pip_pocket_geometry.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML

python3 combine_pip_pocket_geometry.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT
'''