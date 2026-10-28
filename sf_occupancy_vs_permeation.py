#!/usr/bin/env python3
"""
sf_occupancy_vs_permeation.py
=============================

Occupancy of the selectivity filter (SF) with respect to conduction.

For every RUN* it counts, per frame, how many K+ sit inside channel 1
(region 1 = the SF cylinder of IonPermeationAnalysis), averages that over the
run, and plots it against the run's permeation count: one point per run.

Nothing is read from the trajectory. Every ion that ever enters region 1 is
tracked frame by frame in ion_region_tracking, so the number of ions with
region == 1 at a frame IS the SF occupancy at that frame. Per run it reads,
in this order:

  1. <run>/analyzer_state.pkl           (ion_region_tracking + permeation_events)
  2. <run>/ion_region_tracking.json  +  <run>/permeation_table.json

USAGE
-----
    python sf_occupancy_vs_permeation.py /media/.../girk_analyser_results/G12

OUTPUTS  (default: <base_dir>/sf_occupancy/)
-------
    sf_occupancy_per_run.csv            one row per run: frames, mean / std SF
                                        occupancy, permeation count
    sf_occupancy_vs_permeation.png      x = permeations, y = mean K+ in the SF,
                                        with Spearman / Pearson in the corner
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Large fonts: these end up on slides
FS_TITLE  = 24
FS_AXIS   = 22
FS_TICK   = 18
FS_LEGEND = 16
FS_ANNOT  = 13

POINT_COLOR = "#2a78d6"
INK   = "#0b0b0b"
INK_2 = "#52514e"


def run_number(run_dir):
    digits = "".join(c for c in run_dir.name if c.isdigit())
    return int(digits) if digits else -1


def load_run(run_dir):
    """Return (ion_region_tracking, n_permeations, start_frame, end_frame) or None."""
    state_file = run_dir / "analyzer_state.pkl"
    if state_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        tracking = {int(i): {int(fr): int(r) for fr, r in d.items()}
                    for i, d in state["ion_region_tracking"].items()}
        return (tracking, len(state["permeation_events"]),
                int(state["start_frame"]), int(state["end_frame"]))

    tracking_file = run_dir / "ion_region_tracking.json"
    if not tracking_file.exists():
        return None
    with open(tracking_file) as f:
        tracking = {int(i): {int(fr): int(r) for fr, r in d.items()}
                    for i, d in json.load(f).items()}
    perm_file = run_dir / "permeation_table.json"
    n_perm = 0
    if perm_file.exists():
        with open(perm_file) as f:
            n_perm = len(json.load(f))
    frames = [fr for d in tracking.values() for fr in d]
    if not frames:
        return None
    # The JSON has no frame range: take it from the tracked frames
    # (every ion is tracked to the last frame once it has entered region 1).
    return tracking, n_perm, min(frames), max(frames)


def sf_occupancy(tracking, start_frame, end_frame):
    """Number of ions in region 1 (the SF) at every frame start_frame..end_frame."""
    occ = np.zeros(end_frame - start_frame + 1, dtype=int)
    for regions in tracking.values():
        for frame, region in regions.items():
            if region == 1 and start_frame <= frame <= end_frame:
                occ[frame - start_frame] += 1
    return occ


def plot_scatter(df, channel, out_file):
    x = df["Permeations"].to_numpy(float)
    y = df["SF_occ_mean"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(x, y, s=120, color=POINT_COLOR, edgecolor="white",
               linewidth=2, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(str(r["Run_number"]), (r["Permeations"], r["SF_occ_mean"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=FS_ANNOT, color=INK_2)

    subtitle = f"n = {len(df)} runs"
    if len(df) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + icpt, color=POINT_COLOR, linewidth=2,
                alpha=0.5, zorder=2)
        sr, sp = stats.spearmanr(x, y)
        pr, pp = stats.pearsonr(x, y)
        subtitle = (f"Spearman ρ = {sr:.2f} (p = {sp:.2g}),  "
                    f"Pearson r = {pr:.2f} (p = {pp:.2g}),  {subtitle}")

    ax.set_xlabel("Permeation count", fontsize=FS_AXIS)
    ax.set_ylabel("Mean K⁺ in SF", fontsize=FS_AXIS)
    # stats as a subtitle so they never sit on top of a point
    ax.set_title(f"{channel}: SF occupancy vs permeation\n", fontsize=FS_TITLE)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FS_LEGEND, color=INK_2)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Mean SF (channel 1) occupancy vs permeation count, one point per run")
    parser.add_argument("base_dir", help="Results directory containing RUN*")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Only these run numbers")
    parser.add_argument("-channel", default=None,
                        help="Name in the plot title (default: base_dir name)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <base_dir>/sf_occupancy)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    channel = args.channel or base.name
    out_dir = Path(args.out) if args.out else base / "sf_occupancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted((d for d in base.glob("RUN*") if d.is_dir()), key=run_number)
    if args.runs is not None:
        run_dirs = [d for d in run_dirs if run_number(d) in args.runs]

    rows = []
    for run_dir in run_dirs:
        loaded = load_run(run_dir)
        if loaded is None:
            print(f"  {run_dir.name}: no analyzer_state.pkl / ion_region_tracking.json, skipped")
            continue
        tracking, n_perm, start_frame, end_frame = loaded
        occ = sf_occupancy(tracking, start_frame, end_frame)
        rows.append({
            "Run": run_dir.name,
            "Run_number": run_number(run_dir),
            "Frames": len(occ),
            "SF_occ_mean": occ.mean(),
            "SF_occ_std": occ.std(),
            "Permeations": n_perm,
        })
        print(f"  {run_dir.name}: {len(occ)} frames, mean SF occupancy "
              f"{occ.mean():.2f}, {n_perm} permeations")

    if not rows:
        print("ERROR: no runs with ion_region_tracking found.")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sf_occupancy_per_run.csv", index=False)
    if df["Frames"].max() - df["Frames"].min() > 0.05 * df["Frames"].max():
        print("\nWARNING: run lengths differ by more than 5% - longer runs will "
              "have more permeations regardless of occupancy.")
    plot_scatter(df, channel, out_dir / "sf_occupancy_vs_permeation.png")
    print(f"\nResults in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
