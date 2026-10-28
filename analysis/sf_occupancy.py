"""
SF occupancy (one run)
======================
How many K+ sit in the selectivity filter at every frame, and when the
permeating ions leave it.

SF = channel 1 (region 1 of IonPermeationAnalysis). Every ion that ever enters
region 1 is tracked frame by frame in analyzer.ion_region_tracking, so the
number of ions with region == 1 at a frame IS the SF occupancy at that frame;
no trajectory reading. A permeation is placed in time at the frame its ion
leaves the SF (end_1).

Results (results_dir/sf_occupancy/):
    sf_occupancy_per_frame.csv   frame, time_ns, sf_occupancy, sf_exits
                                 (permeating ions with end_1 at that frame)
    sf_occupancy_summary.json    run, frames, frame_duration, mean / std / median
                                 occupancy, % of frames with 0, 1, 2, 3, 4+ K+,
                                 permeations
    sf_occupancy_timeseries.png  occupancy over time, SF exits marked on top

Aggregate over runs with aggregate_sf_occupancy.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OCC_LABELS = ["0", "1", "2", "3", "4+"]
OUT_FOLDER = "sf_occupancy"
PER_FRAME_CSV = "sf_occupancy_per_frame.csv"
SUMMARY_JSON = "sf_occupancy_summary.json"


def sf_occupancy(ion_region_tracking, start_frame, end_frame):
    """Number of ions in region 1 (the SF) at every frame start_frame..end_frame."""
    occ = np.zeros(end_frame - start_frame + 1, dtype=int)
    for regions in ion_region_tracking.values():
        for frame, region in regions.items():
            frame, region = int(frame), int(region)
            if region == 1 and start_frame <= frame <= end_frame:
                occ[frame - start_frame] += 1
    return occ


def occupancy_fractions(occ):
    """{'0': frac, ..., '4+': frac} of frames with that many K+ in the SF."""
    return {label: float((occ >= k).mean() if label.endswith("+") else (occ == k).mean())
            for k, label in enumerate(OCC_LABELS)}


def plot_timeseries(per_frame, run, frame_duration, smooth_ns, out_file):
    t_us = per_frame["time_ns"].to_numpy() / 1000.0
    occ = per_frame["sf_occupancy"].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_us, occ, color="#86b6ef", linewidth=0.6, label="per frame")
    smooth = max(1, int(round(smooth_ns / frame_duration)))
    if smooth > 1 and len(occ) > smooth:
        sm = pd.Series(occ).rolling(smooth, center=True, min_periods=1).mean()
        ax.plot(t_us, sm, color="#1c5cab", linewidth=2,
                label=f"running mean ({smooth_ns:g} ns)")
    exits = per_frame["sf_exits"].to_numpy()
    t_exit = np.repeat(t_us, exits)
    top = max(occ.max(), 1) + 0.6
    ax.plot(t_exit, np.full(len(t_exit), top), linestyle="none", marker="v",
            markersize=9, color="#eb6834", markeredgecolor="white",
            label=f"permeating K⁺ leaves SF ({len(t_exit)})")
    ax.set_ylim(-0.3, top + 0.6)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Time (µs)", fontsize=22)
    ax.set_ylabel("K⁺ in SF", fontsize=22)
    ax.set_title(f"RUN{run}: SF occupancy (mean {occ.mean():.2f})", fontsize=24)
    ax.tick_params(labelsize=18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=14, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sf_occupancy_analysis(ion_region_tracking, permeation_events, start_frame, end_frame,
                              run=1, results_dir=".", frame_duration=0.075, smooth_ns=50.0,
                              force_rerun=False):
    """
    SF occupancy per frame of one run. See the module docstring.

    ion_region_tracking, permeation_events : from IonPermeationAnalysis (or its saved state)
    start_frame, end_frame                 : the analysed frame range
    frame_duration                         : ns per frame (0.075 G12 systems, 0.4 G2)
    smooth_ns                              : running mean in the time-series plot
    Results go to results_dir/sf_occupancy/

    Call from main.py:
        from analysis.sf_occupancy import run_sf_occupancy_analysis
        run_sf_occupancy_analysis(analyzer.ion_region_tracking, analyzer.permeation_events,
                                  start_frame, end_frame, run_number, results_dir, frame_duration)
    """
    output_dir = Path(results_dir) / OUT_FOLDER
    out_csv = output_dir / PER_FRAME_CSV
    if out_csv.exists() and not force_rerun:
        print(f"  {out_csv} already exists - skipping (use force_rerun=True to overwrite).")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    occ = sf_occupancy(ion_region_tracking, start_frame, end_frame)
    exits = np.zeros(len(occ), dtype=int)
    for e in permeation_events:
        f = int(e["end_1"]) - start_frame
        if 0 <= f < len(occ):
            exits[f] += 1

    per_frame = pd.DataFrame({"frame": np.arange(start_frame, end_frame + 1),
                              "time_ns": np.arange(len(occ)) * frame_duration,
                              "sf_occupancy": occ,
                              "sf_exits": exits})
    per_frame.to_csv(out_csv, index=False)

    summary = {
        "run": run,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frames": int(len(occ)),
        "frame_duration_ns": frame_duration,
        "sf_occ_mean": float(occ.mean()),
        "sf_occ_std": float(occ.std()),
        "sf_occ_median": float(np.median(occ)),
        "sf_occ_max": int(occ.max()),
        "frac_frames": occupancy_fractions(occ),
        "permeations": len(permeation_events),
    }
    with open(output_dir / SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    plot_timeseries(per_frame, run, frame_duration, smooth_ns,
                    output_dir / "sf_occupancy_timeseries.png")
    print(f"  SF occupancy RUN{run}: mean {occ.mean():.2f} K+, "
          f"{len(permeation_events)} permeations -> {output_dir}")
