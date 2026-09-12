"""
combine_lys90_cooccurrence.py
==============================
Combines lys90_cooccurrence results across multiple RUN folders.
Reads summary.csv from each RUN, sums counts, recalculates percentages.
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
MAIN_CONDITIONS = ["close_P1", "close_P5", "close_P1_and_P5"]
COLORS = {
    "close_P1":        "#4CAF50",
    "close_P5":        "#2196F3",
    "close_P1_and_P5": "#9C27B0",
}
LABELS = {
    "close_P1":        "Close to P1",
    "close_P5":        "Close to P5",
    "close_P1_and_P5": "Close to\nP1 AND P5",
}


def _sorted_run_dirs(runs_dir: Path) -> list:
    return sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.upper().startswith("RUN") and d.name[3:].isdigit()],
        key=lambda d: int(d.name[3:]),
    )


def _barplot(summary: pd.DataFrame, title: str, out_path: Path, n_runs: int, total: int):
    df_plot = summary[summary["condition"].isin(MAIN_CONDITIONS)]
    if df_plot.empty:
        return

    pcts    = [df_plot.loc[df_plot["condition"]==c, "pct"].values[0]
               if c in df_plot["condition"].values else 0.0
               for c in MAIN_CONDITIONS]
    max_val = max(pcts) if pcts else 1

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(len(MAIN_CONDITIONS)), pcts,
                  color=[COLORS[c] for c in MAIN_CONDITIONS],
                  edgecolor="white", alpha=0.85, width=0.5)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max_val*0.02,
                f"{val:.1f}%", va="bottom", ha="center", fontsize=16)
    ax.set_xticks(np.arange(len(MAIN_CONDITIONS)))
    ax.set_xticklabels([LABELS[c] for c in MAIN_CONDITIONS], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("% of frames", fontsize=16)
    ax.set_title(f"{title}\n[{n_runs} runs, {total} frames]", fontsize=15)
    ax.set_ylim(0, min(115, max_val*1.35) if max_val > 0 else 10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Plot → {out_path.name}")


def _combine_summaries(run_dirs: list, base_path: str,
                       out_dir: Path, channel_type: str,
                       subunit: Optional[str] = None,
                       label: str = "all frames"):
    """Load and combine summary.csv files, save global_summary.csv and barplot."""
    rows = []
    total = 0  # sum total_frames from each run
    for run_dir in run_dirs:
        if subunit:
            csv = run_dir / base_path / f"subunit_{subunit}" / "summary.csv"
        else:
            csv = run_dir / base_path / "summary.csv"
        if not csv.exists():
            print(f"    MISSING: {csv}")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        # get this run's total_frames — same for all rows, take first
        if "total_frames" in df.columns:
            total += int(df["total_frames"].iloc[0])
        elif df["frames"].sum() > 0 and df["pct"].sum() > 0:
            # back-calculate from one row
            v = df[(df["frames"] > 0) & (df["pct"] > 0)]
            best = v.loc[v["frames"].idxmax()]
            total += int(round(best["frames"] / (best["pct"] / 100)))
        rows.append(df)

    if not rows:
        print(f"    No data found for [{label}]{' subunit ' + subunit if subunit else ''}")
        return

    all_df = pd.concat(rows, ignore_index=True)
    n_runs = len(rows)

    # print verification example for close_P1
    ex = all_df[all_df["condition"] == "close_P1"]
    if not ex.empty:
        print(f"    VERIFICATION (close_P1):")
        print(f"      frames per run: {ex['frames'].tolist()}")
        print(f"      sum of frames : {ex['frames'].sum()}")
        print(f"      total_frames  : {total}")
        print(f"      pct           : {ex['frames'].sum() / total * 100:.2f}%")

    # sum frames per condition across all runs
    agg = (all_df.groupby("condition")
                 .agg(frames=("frames", "sum"),
                      label=("label", "first"),
                      threshold_A=("threshold_A", "first"))
                 .reset_index())

    agg["pct"]          = (agg["frames"] / total * 100).round(3) if total > 0 else 0.0
    agg["total_frames"] = total
    agg["n_runs"]       = n_runs
    agg = agg.sort_values("frames", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "global_summary.csv", index=False)
    print(f"    CSV  → {out_dir / 'global_summary.csv'} ({n_runs} runs, {total} frames)")

    sub_label = f"Subunit {subunit} — " if subunit else ""
    _barplot(agg, f"LYS90 co-occurrence — {channel_type}\n{sub_label}{label}",
             out_dir / "barplot.png", n_runs, total)


def _run_for_subset(runs_dir: Path, out_dir: Path,
                    channel_type: str, threshold: int,
                    subset: Optional[str] = None):
    run_dirs  = _sorted_run_dirs(runs_dir)
    base_path = f"lys90_cooccurrence/threshold_{threshold}" + (f"/{subset}" if subset else "")
    label     = subset or "all frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{label}] — threshold {threshold} Å")

    # per subunit only
    for sub in SUBUNIT_ORDER:
        sub_out = out_dir / f"subunit_{sub}"
        _combine_summaries(run_dirs, base_path, sub_out,
                           channel_type, subunit=sub, label=label)


def run_combine_lys90_cooccurrence(
    runs_dir:     str,
    channel_type: str,
    threshold:    int = 4,
    output_dir:   Optional[str] = None,
):
    runs_dir = Path(runs_dir)
    if output_dir is None:
        output_dir = runs_dir / "combined_lys90_cooccurrence"
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("COMBINE LYS90 CO-OCCURRENCE")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Threshold    : {threshold} Å")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60)

    thresh_out = output_dir / f"threshold_{threshold}"

    # all frames
    _run_for_subset(runs_dir, thresh_out, channel_type, threshold, subset=None)

    # end2 / end3
    run_dirs = _sorted_run_dirs(runs_dir)
    for subset in ["end2_frames", "end3_frames"]:
        has = any(
            (rd / f"lys90_cooccurrence/threshold_{threshold}" / subset / "summary.csv").exists()
            for rd in run_dirs
        )
        if has:
            _run_for_subset(runs_dir, thresh_out / subset,
                            channel_type, threshold, subset=subset)
        else:
            print(f"\n  [{subset}] not found — skipping.")

    print(f"\n✓ Combined LYS90 co-occurrence → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",     required=True)
    parser.add_argument("--channel_type", required=True)
    parser.add_argument("--threshold",    type=int, default=4)
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    run_combine_lys90_cooccurrence(
        runs_dir     = args.runs_dir,
        channel_type = args.channel_type,
        threshold    = args.threshold,
        output_dir   = args.output_dir,
    )

'''
python3 combine_lys90_cooccurrence.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 --threshold 4

python3 combine_lys90_cooccurrence.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML --threshold 4

python3 combine_lys90_cooccurrence.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT --threshold 4
'''