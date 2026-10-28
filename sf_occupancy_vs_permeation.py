#!/usr/bin/env python3
"""
sf_occupancy_vs_permeation.py
=============================

Occupancy of the selectivity filter (SF) with respect to conduction.

For every frame of every RUN* it counts how many K+ sit inside channel 1
(region 1 = the SF cylinder of IonPermeationAnalysis) and relates that SF
occupancy to the number of permeation events.

Nothing is read from the trajectory. Every ion that ever enters region 1 is
tracked frame by frame in ion_region_tracking, so the number of ions with
region == 1 at a frame IS the SF occupancy at that frame. Per run it reads,
in this order:

  1. <run>/analyzer_state.pkl           (ion_region_tracking + permeation_events)
  2. <run>/ion_region_tracking.json  +  <run>/permeation_table.json

Two levels of correlation:

  * per run     mean SF occupancy of the whole run vs its permeation count
                (one point per run)
  * per window  each run is cut into windows of --window_ns; for every window
                the mean SF occupancy vs the number of permeating ions that
                LEFT the SF (end_1) inside that window, pooled over all runs
                and shown as a 2D histogram

USAGE
-----
    python sf_occupancy_vs_permeation.py /media/.../girk_analyser_results/G12 \
        --frame_duration 0.075

Several systems on the same per-run scatter (one frame duration per system,
or one value for all); use --per_us when the run lengths differ (e.g. G2):

    python sf_occupancy_vs_permeation.py .../G12 .../G12_ML .../G2 \
        --frame_duration 0.075 0.075 0.4 --per_us \
        --out .../girk_analyser_results/sf_occupancy_comparison

OUTPUTS  (default: <first base_dir>/sf_occupancy/)
-------
    sf_occupancy_per_run.csv            one row per run: mean / std / median SF
                                        occupancy, fraction of frames with
                                        0, 1, 2, 3, 4+ ions, permeation count
                                        and rate
    sf_occupancy_windows.csv            one row per window (all runs)
    per_frame/<system>_<run>.csv        SF occupancy of every frame
    scatter_sf_occupancy_vs_perm.png    per-run mean occupancy vs permeations
    <system>_occupancy_distribution.png fraction of frames with n ions in the
                                        SF, runs sorted by permeation count
    <system>_windows_2d.png             per-window 2D histogram
    <system>_timeseries/<run>.png       SF occupancy over time, permeating
                                        ions leaving the SF marked on top
    correlations.txt                    Pearson / Spearman with p-values
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
from matplotlib.colors import ListedColormap, BoundaryNorm


# ─────────────────────────────────────────────
# Plot styling (large fonts: these end up on slides)
# ─────────────────────────────────────────────
FS_TITLE  = 24
FS_AXIS   = 22
FS_TICK   = 18
FS_LEGEND = 16
FS_ANNOT  = 13

SYSTEM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 0, 1, 2, 3, 4+ ions in the SF: one hue, light -> dark
OCC_COLORS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
OCC_LABELS = ["0", "1", "2", "3", "4+"]
INK = "#0b0b0b"
INK_2 = "#52514e"


def style_axes(ax):
    ax.tick_params(labelsize=FS_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def run_number(run_dir):
    digits = "".join(c for c in run_dir.name if c.isdigit())
    return int(digits) if digits else -1


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_run(run_dir):
    """Return (ion_region_tracking, permeation_events, start_frame, end_frame) or None.

    permeation_events use the analyzer keys (start_1, end_1, ..., end_4).
    """
    state_file = run_dir / "analyzer_state.pkl"
    if state_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        tracking = {int(i): {int(fr): int(r) for fr, r in d.items()}
                    for i, d in state["ion_region_tracking"].items()}
        return (tracking, state["permeation_events"],
                int(state["start_frame"]), int(state["end_frame"]))

    tracking_file = run_dir / "ion_region_tracking.json"
    perm_file = run_dir / "permeation_table.json"
    if not tracking_file.exists():
        return None
    with open(tracking_file) as f:
        tracking = {int(i): {int(fr): int(r) for fr, r in d.items()}
                    for i, d in json.load(f).items()}
    events = []
    if perm_file.exists():
        with open(perm_file) as f:
            events = [{"ion_id": e["ion_id"], "start_1": e["R1_start"],
                       "end_1": e["R1_end"], "end_4": e["R4_end"]}
                      for e in json.load(f)]
    frames = [fr for d in tracking.values() for fr in d]
    if not frames:
        return None
    # The JSON has no frame range: take it from the tracked frames
    # (every ion is tracked to the last frame once it has entered region 1).
    return tracking, events, min(frames), max(frames)


def sf_occupancy(tracking, start_frame, end_frame):
    """Number of ions in region 1 (the SF) at every frame start_frame..end_frame."""
    occ = np.zeros(end_frame - start_frame + 1, dtype=int)
    for regions in tracking.values():
        for frame, region in regions.items():
            if region == 1 and start_frame <= frame <= end_frame:
                occ[frame - start_frame] += 1
    return occ


# ─────────────────────────────────────────────
# Per-run / per-window tables
# ─────────────────────────────────────────────

def summarize_run(system, run_dir, ns_per_frame, window_frames):
    loaded = load_run(run_dir)
    if loaded is None:
        print(f"  {system} {run_dir.name}: no analyzer_state.pkl / ion_region_tracking.json, skipped")
        return None, None, None
    tracking, events, start_frame, end_frame = loaded
    occ = sf_occupancy(tracking, start_frame, end_frame)
    n_frames = len(occ)
    n_perm = len(events)
    length_us = n_frames * ns_per_frame / 1000.0

    row = {
        "System": system,
        "Run": run_dir.name,
        "Run_number": run_number(run_dir),
        "Frames": n_frames,
        "Length_us": length_us,
        "SF_occ_mean": occ.mean(),
        "SF_occ_std": occ.std(),
        "SF_occ_median": float(np.median(occ)),
        "SF_occ_max": int(occ.max()),
    }
    for k, label in enumerate(OCC_LABELS):
        frac = (occ >= k).mean() if label.endswith("+") else (occ == k).mean()
        row[f"Frac_{label}_ions"] = frac
    row["Permeations"] = n_perm
    row["Permeations_per_us"] = n_perm / length_us if length_us else np.nan

    # Each permeation is placed in time by the frame its ion leaves the SF (end_1)
    exit_frames = np.array([int(e["end_1"]) - start_frame for e in events], dtype=int)

    windows = []
    for w0 in range(0, n_frames - window_frames + 1, window_frames):
        w1 = w0 + window_frames
        windows.append({
            "System": system,
            "Run": run_dir.name,
            "Window_start_frame": w0 + start_frame,
            "Window_end_frame": w1 - 1 + start_frame,
            "SF_occ_mean": occ[w0:w1].mean(),
            "Permeations": int(((exit_frames >= w0) & (exit_frames < w1)).sum()),
        })

    per_frame = pd.DataFrame({"frame": np.arange(start_frame, end_frame + 1),
                              "time_ns": np.arange(n_frames) * ns_per_frame,
                              "sf_occupancy": occ})
    print(f"  {system} {run_dir.name}: {n_frames} frames, mean SF occupancy "
          f"{row['SF_occ_mean']:.2f}, {n_perm} permeations")
    return row, windows, (per_frame, exit_frames + start_frame)


# ─────────────────────────────────────────────
# Correlations
# ─────────────────────────────────────────────

def correlate(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return dict(n=len(x), pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp)


def fmt_corr(c):
    return (f"n={c['n']}, Spearman ρ={c['spearman_rho']:.2f} (p={c['spearman_p']:.2g}), "
            f"Pearson r={c['pearson_r']:.2f} (p={c['pearson_p']:.2g})")


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_run_scatter(df, systems, y_col, y_label, out_file):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i, system in enumerate(systems):
        sub = df[df["System"] == system]
        if sub.empty:
            continue
        color = SYSTEM_COLORS[i % len(SYSTEM_COLORS)]
        c = correlate(sub["SF_occ_mean"], sub[y_col])
        label = system if np.isnan(c["spearman_rho"]) else \
            f"{system}  (ρ={c['spearman_rho']:.2f}, p={c['spearman_p']:.2g})"
        ax.scatter(sub["SF_occ_mean"], sub[y_col], s=110, color=color,
                   edgecolor="white", linewidth=2, zorder=3, label=label)
        if len(sub) >= 3 and np.ptp(sub["SF_occ_mean"]) > 0:
            slope, icpt = np.polyfit(sub["SF_occ_mean"], sub[y_col], 1)
            xs = np.linspace(sub["SF_occ_mean"].min(), sub["SF_occ_mean"].max(), 50)
            ax.plot(xs, slope * xs + icpt, color=color, linewidth=2, alpha=0.6, zorder=2)
        for _, r in sub.iterrows():
            ax.annotate(str(r["Run_number"]), (r["SF_occ_mean"], r[y_col]),
                        xytext=(6, 5), textcoords="offset points",
                        fontsize=FS_ANNOT, color=INK_2)
    ax.set_xlabel("Mean SF occupancy (K⁺ in channel 1)", fontsize=FS_AXIS)
    ax.set_ylabel(y_label, fontsize=FS_AXIS)
    ax.set_title("SF occupancy vs permeation (one point per run)", fontsize=FS_TITLE)
    style_axes(ax)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_occupancy_distribution(sub, system, y_col, y_label, out_file):
    sub = sub.sort_values(y_col).reset_index(drop=True)
    fracs = sub[[f"Frac_{l}_ions" for l in OCC_LABELS]].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(sub) + 2.5)))
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
    # permeation value next to each run, mean occupancy in brackets
    fmt = "{:.0f}" if y_col == "Permeations" else "{:.2f}"
    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.text(101.5, yi, f"{fmt.format(r[y_col])}  (⟨n⟩={r['SF_occ_mean']:.2f})",
                va="center", fontsize=FS_ANNOT + 2, color=INK)
    ax.text(101.5, len(sub) - 0.2, y_label, fontsize=FS_ANNOT + 2,
            color=INK_2, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{system}: K⁺ in the SF, runs sorted by permeation",
                 fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=len(OCC_LABELS),
              loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / (0.45 * len(sub) + 2.5)))
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_windows_2d(wsub, system, window_ns, out_file):
    x = wsub["SF_occ_mean"].to_numpy()
    y = wsub["Permeations"].to_numpy()
    # 0.25-ion occupancy bins, one row per permeation count
    x_edges = np.arange(np.floor(x.min() * 4) / 4, np.ceil(x.max() * 4) / 4 + 0.25, 0.25)
    y_edges = np.arange(-0.5, y.max() + 1.5, 1.0)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    Hm = np.ma.masked_equal(H.T, 0)
    mesh = ax.pcolormesh(x_edges, y_edges, Hm, cmap="Blues", shading="flat",
                         edgecolors="white", linewidth=1.5)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Number of windows", fontsize=FS_AXIS - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 2)

    # mean permeations per occupancy bin, to see the trend through the cloud
    centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    idx = np.clip(np.digitize(x, x_edges) - 1, 0, len(centers) - 1)
    # only bins with >= 3 windows, so single windows do not make the trend jump
    means = [y[idx == b].mean() if np.sum(idx == b) >= 3 else np.nan for b in range(len(centers))]
    ax.plot(centers, means, color="#eb6834", linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, label="mean per bin (≥3 windows)")

    c = correlate(x, y)
    ax.set_xlabel("Mean SF occupancy in window (K⁺)", fontsize=FS_AXIS)
    ax.set_ylabel("Permeations in window", fontsize=FS_AXIS)
    ax.set_title(f"{system}: {window_ns:g} ns windows, all runs", fontsize=FS_TITLE)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style_axes(ax)
    ax.grid(False)
    if not np.isnan(c["spearman_rho"]):
        ax.text(0.02, 0.97, f"ρ={c['spearman_rho']:.2f} (p={c['spearman_p']:.2g}), n={c['n']}",
                transform=ax.transAxes, fontsize=FS_LEGEND, va="top", color=INK)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_timeseries(per_frame, exit_frames, system, run_name, ns_per_frame,
                    smooth_frames, smooth_ns, out_file):
    t_us = per_frame["time_ns"].to_numpy() / 1000.0
    occ = per_frame["sf_occupancy"].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_us, occ, color="#86b6ef", linewidth=0.6, label="per frame")
    if smooth_frames > 1 and len(occ) > smooth_frames:
        sm = pd.Series(occ).rolling(smooth_frames, center=True, min_periods=1).mean()
        ax.plot(t_us, sm, color="#1c5cab", linewidth=2,
                label=f"running mean ({smooth_ns:g} ns)")
    start = per_frame["frame"].iloc[0]
    t_exit = (np.asarray(exit_frames) - start) * ns_per_frame / 1000.0
    top = max(occ.max(), 1) + 0.6
    ax.plot(t_exit, np.full(len(t_exit), top), linestyle="none", marker="v",
            markersize=9, color="#eb6834", markeredgecolor="white",
            label=f"permeating K⁺ leaves SF ({len(t_exit)})")
    ax.set_ylim(-0.3, top + 0.6)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Time (µs)", fontsize=FS_AXIS)
    ax.set_ylabel("K⁺ in SF", fontsize=FS_AXIS)
    ax.set_title(f"{system} {run_name}: SF occupancy (mean {occ.mean():.2f})",
                 fontsize=FS_TITLE)
    style_axes(ax)
    ax.legend(fontsize=FS_LEGEND - 2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SF (channel 1) occupancy vs permeation across runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1].split("OUTPUTS")[0],
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="One results directory per system, each containing RUN*")
    parser.add_argument("--frame_duration", "-ns_per_frame", type=float, nargs="+",
                        default=[0.075],
                        help="ns per frame: one value for all systems or one per "
                             "base_dir (0.075 for G12 / G12_ML / G12_GAT, 0.4 for G2)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="System names for legends (default: base_dir names)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Only these run numbers")
    parser.add_argument("--window_ns", type=float, default=50.0,
                        help="Window length for the per-window correlation (default 50 ns)")
    parser.add_argument("--per_us", action="store_true",
                        help="Plot permeations per µs instead of the raw count "
                             "(use when the runs/systems have different lengths)")
    parser.add_argument("--no_timeseries", action="store_true",
                        help="Skip the per-run occupancy time-series plots")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <first base_dir>/sf_occupancy)")
    args = parser.parse_args()

    base_dirs = [Path(b) for b in args.base_dirs]
    durations = args.frame_duration
    if len(durations) == 1:
        durations = durations * len(base_dirs)
    if len(durations) != len(base_dirs):
        parser.error("--frame_duration needs one value, or one per base_dir")
    labels = args.labels or [b.name for b in base_dirs]
    if len(labels) != len(base_dirs):
        parser.error("--labels needs one name per base_dir")

    out_dir = Path(args.out) if args.out else base_dirs[0] / "sf_occupancy"
    (out_dir / "per_frame").mkdir(parents=True, exist_ok=True)

    y_col = "Permeations_per_us" if args.per_us else "Permeations"
    y_label = "Permeations per µs" if args.per_us else "Permeations"

    rows, windows = [], []
    for base, system, ns in zip(base_dirs, labels, durations):
        print(f"\n{system}  ({base}, {ns} ns/frame)")
        run_dirs = sorted((d for d in base.glob("RUN*") if d.is_dir()), key=run_number)
        if args.runs is not None:
            run_dirs = [d for d in run_dirs if run_number(d) in args.runs]
        window_frames = max(1, int(round(args.window_ns / ns)))
        for run_dir in run_dirs:
            row, wins, frame_data = summarize_run(system, run_dir, ns, window_frames)
            if row is None:
                continue
            rows.append(row)
            windows.extend(wins)
            per_frame, exit_frames = frame_data
            per_frame.to_csv(out_dir / "per_frame" / f"{system}_{run_dir.name}.csv", index=False)
            if not args.no_timeseries:
                ts_dir = out_dir / f"{system}_timeseries"
                ts_dir.mkdir(exist_ok=True)
                plot_timeseries(per_frame, exit_frames, system, run_dir.name, ns,
                                window_frames, args.window_ns, ts_dir / f"{run_dir.name}.png")

    if not rows:
        print("ERROR: no runs with ion_region_tracking found.")
        return 1

    df = pd.DataFrame(rows)
    wdf = pd.DataFrame(windows)
    df.to_csv(out_dir / "sf_occupancy_per_run.csv", index=False)
    wdf.to_csv(out_dir / "sf_occupancy_windows.csv", index=False)

    lengths = df.groupby("System")["Frames"].agg(["min", "max"])
    if not args.per_us and ((lengths["max"] - lengths["min"]) > 0.05 * lengths["max"]).any():
        print("\nWARNING: run lengths differ by more than 5% within a system - "
              "consider --per_us so long runs do not look more conductive.")

    present = [s for s in labels if s in set(df["System"])]
    plot_run_scatter(df, present, y_col, y_label, out_dir / "scatter_sf_occupancy_vs_perm.png")

    lines = [f"SF occupancy vs permeation  (windows: {args.window_ns:g} ns, "
             f"permeation placed at the frame the ion leaves the SF)\n"]
    for system in present:
        sub = df[df["System"] == system]
        plot_occupancy_distribution(sub, system, y_col, y_label,
                                    out_dir / f"{system}_occupancy_distribution.png")
        wsub = wdf[wdf["System"] == system]
        if not wsub.empty:
            plot_windows_2d(wsub, system, args.window_ns, out_dir / f"{system}_windows_2d.png")

        lines.append(f"[{system}]")
        lines.append(f"  mean SF occupancy over runs: {sub['SF_occ_mean'].mean():.3f} "
                     f"± {sub['SF_occ_mean'].std():.3f} (SD over {len(sub)} runs)")
        lines.append(f"  per run, mean occupancy vs {y_col}: "
                     f"{fmt_corr(correlate(sub['SF_occ_mean'], sub[y_col]))}")
        if not wsub.empty:
            lines.append(f"  per window, mean occupancy vs permeations: "
                         f"{fmt_corr(correlate(wsub['SF_occ_mean'], wsub['Permeations']))}")
        lines.append("")
    if len(present) > 1:
        lines.append(f"[all systems pooled, per run]  "
                     f"{fmt_corr(correlate(df['SF_occ_mean'], df[y_col]))}")
    text = "\n".join(lines)
    (out_dir / "correlations.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nAll results in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
