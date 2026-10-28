#!/usr/bin/env python3
"""
aggregate_sf_occupancy.py
=========================

Occupancy of the selectivity filter (SF) with respect to conduction,
aggregated over all runs of one system (per-run part: analysis/sf_occupancy.py,
called from main.py).

SF = channel 1 (region 1 of IonPermeationAnalysis). Per run it reads, in this order:

  1. <run>/sf_occupancy/sf_occupancy_per_frame.csv   (written by main.py; its
                                                      time_ns column sets the
                                                      frame duration of that run)
  2. <run>/analyzer_state.pkl                         (ion_region_tracking +
                                                      permeation_events)
  3. <run>/ion_region_tracking.json + <run>/permeation_table.json

so it also works for runs main.py has not been rerun on. 2 and 3 use
--frame_duration for the window length.

USAGE
-----
    python aggregate_sf_occupancy.py /media/.../girk_analyser_results/G12

    # G2 saves a frame every 0.4 ns (G12 systems 0.075, the default)
    python aggregate_sf_occupancy.py /media/.../girk_analyser_results/G2 --frame_duration 0.4

OUTPUTS  (default: <base_dir>/aggregated_sf_occupancy/)
-------
    sf_occupancy_per_run.csv            one row per run: frames, mean / std SF
                                        occupancy, permeation count
    sf_occupancy_vs_permeation.png      x = permeations, y = mean K+ in the SF,
                                        with Spearman / Pearson under the title
    sf_occupancy_distribution.png       % of frames with 0, 1, 2, 3, 4+ K+ in
                                        the SF, runs sorted by permeation count
    sf_occupancy_windows_2d.png         each run cut into --window_ns windows:
                                        mean SF occupancy of the window vs the
                                        permeating ions that left the SF (end_1)
                                        in it, all runs pooled, as a 2D histogram
    sf_occupancy_windows.csv            the windows behind that plot
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

from analysis.sf_occupancy import (sf_occupancy, occupancy_fractions, OCC_LABELS,
                                   OUT_FOLDER, PER_FRAME_CSV)


# Large fonts: these end up on slides
FS_TITLE  = 24
FS_AXIS   = 22
FS_TICK   = 18
FS_LEGEND = 16
FS_ANNOT  = 13

POINT_COLOR = "#2a78d6"
TREND_COLOR = "#eb6834"
# 0, 1, 2, 3, 4+ ions in the SF: one hue, light -> dark
OCC_COLORS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK   = "#0b0b0b"
INK_2 = "#52514e"


def run_number(run_dir):
    digits = "".join(c for c in run_dir.name if c.isdigit())
    return int(digits) if digits else -1


def load_run(run_dir):
    """Return (sf_occupancy per frame, sf_exits per frame, ns per frame or None) or None.

    sf_exits: permeating ions leaving the SF (end_1) at each frame.
    """
    per_frame_csv = run_dir / OUT_FOLDER / PER_FRAME_CSV
    if per_frame_csv.exists():
        pf = pd.read_csv(per_frame_csv)
        t = pf["time_ns"].to_numpy()
        ns = float(t[1] - t[0]) if len(t) > 1 else None
        return pf["sf_occupancy"].to_numpy(int), pf["sf_exits"].to_numpy(int), ns

    state_file = run_dir / "analyzer_state.pkl"
    if state_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        tracking = state["ion_region_tracking"]
        exit_frames = [int(e["end_1"]) for e in state["permeation_events"]]
        start_frame, end_frame = int(state["start_frame"]), int(state["end_frame"])
    else:
        tracking_file = run_dir / "ion_region_tracking.json"
        if not tracking_file.exists():
            return None
        with open(tracking_file) as f:
            tracking = json.load(f)
        perm_file = run_dir / "permeation_table.json"
        exit_frames = []
        if perm_file.exists():
            with open(perm_file) as f:
                exit_frames = [int(e["R1_end"]) for e in json.load(f)]
        frames = [int(fr) for d in tracking.values() for fr in d]
        if not frames:
            return None
        # The JSON has no frame range: take it from the tracked frames
        # (every ion is tracked to the last frame once it has entered region 1).
        start_frame, end_frame = min(frames), max(frames)

    occ = sf_occupancy(tracking, start_frame, end_frame)
    exits = np.zeros(len(occ), dtype=int)
    for f in exit_frames:
        if 0 <= f - start_frame < len(occ):
            exits[f - start_frame] += 1
    return occ, exits, None


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


def plot_occupancy_distribution(df, channel, out_file):
    sub = df.sort_values("Permeations").reset_index(drop=True)
    fracs = sub[[f"Frac_{l}_ions" for l in OCC_LABELS]].to_numpy() * 100
    height = max(5, 0.45 * len(sub) + 2.5)
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(sub))
    left = np.zeros(len(sub))
    for k, label in enumerate(OCC_LABELS):
        ax.barh(y, fracs[:, k], left=left, height=0.8, color=OCC_COLORS[k],
                edgecolor="white", linewidth=2, label=f"{label} K⁺")
        left += fracs[:, k]
    ax.set_yticks(y)
    ax.set_yticklabels(sub["Run"], fontsize=FS_TICK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of frames", fontsize=FS_AXIS)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    # permeation count next to each run, mean occupancy in brackets
    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.text(101.5, yi, f"{r['Permeations']:.0f}  (⟨n⟩={r['SF_occ_mean']:.2f})",
                va="center", fontsize=FS_ANNOT + 2, color=INK)
    ax.text(101.5, len(sub) - 0.2, "Permeations", fontsize=FS_ANNOT + 2,
            color=INK_2, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{channel}: K⁺ in the SF, runs sorted by permeation",
                 fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=len(OCC_LABELS),
              loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / height))
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_windows_2d(wdf, channel, window_ns, out_file):
    x = wdf["SF_occ_mean"].to_numpy()
    y = wdf["Permeations"].to_numpy()
    # 0.25-ion occupancy bins, one row per permeation count
    x_edges = np.arange(np.floor(x.min() * 4) / 4, np.ceil(x.max() * 4) / 4 + 0.25, 0.25)
    y_edges = np.arange(-0.5, y.max() + 1.5, 1.0)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(H.T, 0), cmap="Blues",
                         shading="flat", edgecolors="white", linewidth=1.5)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Number of windows", fontsize=FS_AXIS - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 2)

    # mean permeations per occupancy bin, only bins with >= 3 windows,
    # so single windows do not make the trend jump
    centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    idx = np.clip(np.digitize(x, x_edges) - 1, 0, len(centers) - 1)
    means = [y[idx == b].mean() if np.sum(idx == b) >= 3 else np.nan for b in range(len(centers))]
    ax.plot(centers, means, color=TREND_COLOR, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, label="mean per bin (≥3 windows)")

    subtitle = f"n = {len(wdf)} windows"
    if len(wdf) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        sr, sp = stats.spearmanr(x, y)
        subtitle = f"Spearman ρ = {sr:.2f} (p = {sp:.2g}),  {subtitle}"
    ax.set_title(f"{channel}: {window_ns:g} ns windows, all runs\n", fontsize=FS_TITLE)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FS_LEGEND, color=INK_2)
    ax.set_xlabel("Mean SF occupancy in window (K⁺)", fontsize=FS_AXIS)
    ax.set_ylabel("Permeations in window", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Mean SF (channel 1) occupancy vs permeation count, one point per run")
    parser.add_argument("base_dir", help="Results directory containing RUN*")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Only these run numbers")
    parser.add_argument("--frame_duration", "-ns_per_frame", type=float, default=0.075,
                        help="ns per saved frame, for the window length of runs without "
                             "sf_occupancy/ from main.py: 0.075 for G12 / G12_ML / G12_GAT "
                             "(default), 0.4 for G2")
    parser.add_argument("--window_ns", type=float, default=50.0,
                        help="Window length for the 2D window plot (default 50 ns)")
    parser.add_argument("-channel", default=None,
                        help="Name in the plot title (default: base_dir name)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <base_dir>/aggregated_sf_occupancy)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    channel = args.channel or base.name
    out_dir = Path(args.out) if args.out else base / "aggregated_sf_occupancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted((d for d in base.glob("RUN*") if d.is_dir()), key=run_number)
    if args.runs is not None:
        run_dirs = [d for d in run_dirs if run_number(d) in args.runs]

    rows, windows = [], []
    for run_dir in run_dirs:
        loaded = load_run(run_dir)
        if loaded is None:
            print(f"  {run_dir.name}: no sf_occupancy / analyzer_state.pkl / "
                  f"ion_region_tracking.json, skipped")
            continue
        occ, exits, ns = loaded
        window_frames = max(1, int(round(args.window_ns / (ns or args.frame_duration))))
        n_perm = int(exits.sum())
        row = {
            "Run": run_dir.name,
            "Run_number": run_number(run_dir),
            "Frames": len(occ),
            "SF_occ_mean": occ.mean(),
            "SF_occ_std": occ.std(),
            "Permeations": n_perm,
        }
        for label, frac in occupancy_fractions(occ).items():
            row[f"Frac_{label}_ions"] = frac
        rows.append(row)

        for w0 in range(0, len(occ) - window_frames + 1, window_frames):
            w1 = w0 + window_frames
            windows.append({
                "Run": run_dir.name,
                "Window_first_frame_index": w0,
                "Window_last_frame_index": w1 - 1,
                "SF_occ_mean": occ[w0:w1].mean(),
                "Permeations": int(exits[w0:w1].sum()),
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
    plot_occupancy_distribution(df, channel, out_dir / "sf_occupancy_distribution.png")
    wdf = pd.DataFrame(windows)
    if wdf.empty:
        print(f"No complete {args.window_ns:g} ns window in any run - window plot skipped.")
    else:
        wdf.to_csv(out_dir / "sf_occupancy_windows.csv", index=False)
        plot_windows_2d(wdf, channel, args.window_ns, out_dir / "sf_occupancy_windows_2d.png")
    print(f"\nResults in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
