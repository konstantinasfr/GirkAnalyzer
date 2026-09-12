"""
dwell_time_analysis.py

Computes ion dwell time in the GLU/ASN cavity for ONE simulation system.

Dwell time definition:
    dwell_time = (R2_end - R1_end) x frame_duration

Where:
    R1_end = last frame the ion is in region 1 (selectivity filter)
    R2_end = last frame the ion is in region 2 (GLU/ASN cavity)
    --> (R2_end - R1_end) = number of frames the ion spends in the
        GLU/ASN cavity after leaving the selectivity filter

Reads:   <base_dir>/RUN*/permeation_table.json
Outputs:
    <label>_summary.csv          -- one row per run + a TOTAL row
    <label>_all_events.csv       -- one row per permeation event
    <label>_dwell_bar.png        -- mean +/- std bar, annotated
    <label>_dwell_box.png        -- box/whisker + jittered dots, annotated
    <label>_dwell_histogram.png  -- histogram with mean & median lines

Usage:
    python3 dwell_time_analysis.py /path/to/G12
    python3 dwell_time_analysis.py /path/to/G12_ML --frame_duration 0.075
    python3 dwell_time_analysis.py /path/to/G12_ML --frame_duration 0.075 --output /path/to/out
"""

import json
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(base_dir: Path, frame_duration: float):
    """
    Walk all RUN* subdirectories, load permeation_table.json.

    Returns
    -------
    all_events   : list of dict  -- one entry per valid permeation event
    runs_detail  : list of dict  -- per-run summary
    total_frames : int           -- sum across runs with known frame counts
    """
    all_events   = []
    runs_detail  = []
    total_frames = 0

    run_dirs = sorted(base_dir.glob("RUN*"))
    if not run_dirs:
        raise FileNotFoundError(f"No RUN* directories found in {base_dir}")

    for run_dir in run_dirs:
        perm_file = run_dir / "permeation_table.json"
        if not perm_file.exists():
            print(f"  [SKIP] {run_dir.name} -- no permeation_table.json")
            continue

        with open(perm_file) as f:
            events = json.load(f)

        run_frames = _read_frame_count(run_dir)
        if run_frames:
            total_frames += run_frames

        run_valid   = 0
        run_skipped = 0
        run_dwells  = []

        for ev in events:
            e1 = ev.get("R1_end")
            e2 = ev.get("R2_end")
            if e1 is not None and e2 is not None and e2 > e1:
                dwell_ns = (e2 - e1) * frame_duration
                record = {
                    "run":          run_dir.name,
                    "ion_id":       ev.get("ion_id"),
                    "R1_end":       e1,
                    "R2_end":       e2,
                    "dwell_frames": e2 - e1,
                    "dwell_ns":     dwell_ns,
                }
                all_events.append(record)
                run_dwells.append(dwell_ns)
                run_valid += 1
            else:
                run_skipped += 1

        st_ns = run_frames * frame_duration if run_frames else None
        st_us = st_ns / 1000 if st_ns else None
        pc_per_half_us = (run_valid / (st_us / 0.5)) if (st_us and st_us > 0) else None

        runs_detail.append({
            "run":             run_dir.name,
            "frames":          run_frames,
            "st_ns":           st_ns,
            "st_us":           st_us,
            "n_events":        len(events),
            "n_valid":         run_valid,
            "n_skipped":       run_skipped,
            "pc_per_half_us":  pc_per_half_us,
            "mean_dwell_ns":   float(np.mean(run_dwells))   if run_dwells else None,
            "std_dwell_ns":    float(np.std(run_dwells))    if run_dwells else None,
            "median_dwell_ns": float(np.median(run_dwells)) if run_dwells else None,
        })

        msg = (f"  {run_dir.name}: {len(events)} events, {run_valid} valid"
               + (f", {run_skipped} skipped (R1_end >= R2_end)" if run_skipped else "")
               + (f", {run_frames} frames" if run_frames else ""))
        print(msg)

    return all_events, runs_detail, total_frames


def _read_frame_count(run_dir: Path):
    """Try to read total frame count from the saved analyzer state."""
    try:
        import pickle
        state_file = run_dir / "analyzer_state.pkl"
        if state_file.exists():
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            if "end_frame" in state:
                return int(state["end_frame"]) + 1
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(label, all_events, runs_detail, total_frames, frame_duration):
    """Return a dict of pooled summary statistics."""
    d = np.array([e["dwell_ns"] for e in all_events])

    st_ns = total_frames * frame_duration if total_frames else None
    st_us = st_ns / 1000 if st_ns else None
    pc    = len(d)
    rate_per_ns      = (pc / st_ns)          if (st_ns and st_ns > 0) else None
    rate_per_us      = (pc / st_us)          if (st_us and st_us > 0) else None
    rate_per_half_us = (pc / (st_us / 0.5))  if (st_us and st_us > 0) else None

    return {
        "label":           label,
        "n_runs":          len(runs_detail),
        "total_frames":    total_frames,
        "st_ns":           st_ns,
        "st_us":           st_us,
        "pc":              pc,
        "pc_per_ns":       rate_per_ns,
        "pc_per_us":       rate_per_us,
        "pc_per_half_us":  rate_per_half_us,
        "mean_ns":         float(np.mean(d))   if len(d) else None,
        "std_ns":          float(np.std(d))    if len(d) else None,
        "median_ns":       float(np.median(d)) if len(d) else None,
        "min_ns":          float(np.min(d))    if len(d) else None,
        "max_ns":          float(np.max(d))    if len(d) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "n/a"


def print_summary(s: dict, runs_detail: list):
    w = 72
    print()
    print("=" * w)
    print(f"  DWELL TIME SUMMARY -- {s['label']}")
    print("=" * w)
    print(f"  {'Runs loaded':<32}: {s['n_runs']}")
    if s["st_ns"]:
        print(f"  {'Total sim. time (ST)':<32}: {s['st_ns']:.1f} ns  ({s['st_us']:.4f} us)")
    else:
        print(f"  {'Total sim. time (ST)':<32}: unknown (no analyzer_state.pkl)")
    print(f"  {'Permeation count (PC)':<32}: {s['pc']}")
    if s["pc_per_ns"]:
        print(f"  {'PC per ns':<32}: {s['pc_per_ns']:.4f}")
    if s["pc_per_us"]:
        print(f"  {'PC per us':<32}: {s['pc_per_us']:.2f}")
    if s["pc_per_half_us"]:
        print(f"  {'PC per 0.5 us':<32}: {s['pc_per_half_us']:.2f}")
    print()
    if s["mean_ns"] is not None:
        print(f"  Dwell time  (R2_end - R1_end) x frame_duration:")
        print(f"    {'Mean':<12}: {s['mean_ns']:.4f} ns")
        print(f"    {'Std':<12}: {s['std_ns']:.4f} ns")
        print(f"    {'Median':<12}: {s['median_ns']:.4f} ns")
        print(f"    {'Min':<12}: {s['min_ns']:.4f} ns")
        print(f"    {'Max':<12}: {s['max_ns']:.4f} ns")
    print()
    # Per-run table
    print(f"  {'Run':<10} {'Frames':>8} {'ST(ns)':>10} {'PC':>5} "
          f"{'PC/0.5us':>10} {'Mean(ns)':>10} {'Std(ns)':>10} {'Median(ns)':>11}")
    print("  " + "-" * (w - 2))
    for r in runs_detail:
        frames_s = str(r["frames"]) if r["frames"] else "?"
        st_s     = f"{r['st_ns']:.1f}" if r["st_ns"] else "?"
        pc05_s   = f"{r['pc_per_half_us']:.2f}" if r["pc_per_half_us"] else "?"
        mean_s   = f"{r['mean_dwell_ns']:.4f}"   if r["mean_dwell_ns"]   is not None else "--"
        std_s    = f"{r['std_dwell_ns']:.4f}"    if r["std_dwell_ns"]    is not None else "--"
        med_s    = f"{r['median_dwell_ns']:.4f}" if r["median_dwell_ns"] is not None else "--"
        print(f"  {r['run']:<10} {frames_s:>8} {st_s:>10} {r['n_valid']:>5} "
              f"{pc05_s:>10} {mean_s:>10} {std_s:>10} {med_s:>11}")
    print("=" * w)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(s: dict, runs_detail: list, out_path: Path):
    """Write per-run rows + a TOTAL row to CSV."""
    rows = []
    for r in runs_detail:
        rows.append({
            "run":             r["run"],
            "frames":          r["frames"] if r["frames"] else "",
            "st_ns":           f"{r['st_ns']:.2f}"  if r["st_ns"] else "",
            "st_us":           f"{r['st_us']:.2f}"  if r["st_us"] else "",
            "PC":              r["n_valid"],
            "PC_per_ns":       f"{r['n_valid'] / r['st_ns']:.2f}" if r["st_ns"] else "",
            "PC_per_0.5us":    f"{r['pc_per_half_us']:.2f}" if r["pc_per_half_us"] else "",
            "mean_dwell_ns":   f"{r['mean_dwell_ns']:.2f}"   if r["mean_dwell_ns"]   is not None else "",
            "std_dwell_ns":    f"{r['std_dwell_ns']:.2f}"    if r["std_dwell_ns"]    is not None else "",
            "median_dwell_ns": f"{r['median_dwell_ns']:.2f}" if r["median_dwell_ns"] is not None else "",
            "min_dwell_ns":    "",
            "max_dwell_ns":    "",
        })
    # TOTAL row
    rows.append({
        "run":             "TOTAL",
        "frames":          s["total_frames"] if s["total_frames"] else "",
        "st_ns":           f"{s['st_ns']:.2f}"  if s["st_ns"] else "",
        "st_us":           f"{s['st_us']:.2f}"  if s["st_us"] else "",
        "PC":              s["pc"],
        "PC_per_ns":       f"{s['pc_per_ns']:.2f}"  if s["pc_per_ns"]  else "",
        "PC_per_0.5us":    f"{s['pc_per_half_us']:.2f}" if s["pc_per_half_us"] else "",
        "mean_dwell_ns":   f"{s['mean_ns']:.2f}"   if s["mean_ns"]   is not None else "",
        "std_dwell_ns":    f"{s['std_ns']:.2f}"    if s["std_ns"]    is not None else "",
        "median_dwell_ns": f"{s['median_ns']:.2f}" if s["median_ns"] is not None else "",
        "min_dwell_ns":    f"{s['min_ns']:.2f}"    if s["min_ns"]    is not None else "",
        "max_dwell_ns":    f"{s['max_ns']:.2f}"    if s["max_ns"]    is not None else "",
    })

    fields = ["run", "frames", "st_ns", "st_us", "PC", "PC_per_ns", "PC_per_0.5us",
              "mean_dwell_ns", "std_dwell_ns", "median_dwell_ns", "min_dwell_ns", "max_dwell_ns"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def save_events_csv(all_events: list, out_path: Path):
    """Write one row per permeation event."""
    if not all_events:
        return
    fields = ["run", "ion_id", "R1_end", "R2_end", "dwell_frames", "dwell_ns"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_events)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _stats_box(s):
    return (f"n = {s['pc']}\n"
            f"Mean   = {_fmt(s['mean_ns'], 2)} ns\n"
            f"Std    = {_fmt(s['std_ns'], 2)} ns\n"
            f"Median = {_fmt(s['median_ns'], 2)} ns")


def plot_bar(label, all_events, s, out_path, frame_duration):
    """Mean +/- std bar with jittered dots and annotated stats."""
    d = np.array([e["dwell_ns"] for e in all_events])
    if len(d) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 6))

    ax.bar([0], [s["mean_ns"]], yerr=[s["std_ns"]], capsize=9, width=0.42,
           color="#4878CF", alpha=0.72, edgecolor="black", linewidth=0.9,
           error_kw=dict(elinewidth=1.8, ecolor="black"))

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.13, 0.13, size=len(d))
    ax.scatter(jitter, d, color="#4878CF", alpha=0.45, s=18,
               edgecolors="none", zorder=3)

    ax.hlines(s["median_ns"], -0.23, 0.23, colors="#C0392B", linewidths=1.8,
              linestyles="--", zorder=4, label=f"Median = {_fmt(s['median_ns'], 2)} ns")

    ax.text(0.97, 0.97, _stats_box(s), transform=ax.transAxes,
            fontsize=13, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.legend(fontsize=13, loc="upper left")
    ax.set_xticks([0])
    ax.set_xticklabels([label], fontsize=15)
    ax.set_ylabel("Dwell time in cavity (ns)", fontsize=15)
    ax.set_title(f"{label} — Ion dwell time in cavity", fontsize=16, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_box(label, all_events, s, out_path, frame_duration):
    """Box/whisker + jittered dots with annotated stats."""
    d = np.array([e["dwell_ns"] for e in all_events])
    if len(d) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 6))

    bp = ax.boxplot([d], patch_artist=True, notch=False, widths=0.42,
                    medianprops=dict(color="#C0392B", linewidth=2.2),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    flierprops=dict(marker="", linestyle="none"))
    bp["boxes"][0].set_facecolor("#4878CF")
    bp["boxes"][0].set_alpha(0.62)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(d))
    ax.scatter(1 + jitter, d, color="#4878CF", alpha=0.45,
               s=18, edgecolors="none", zorder=3)

    ax.text(0.97, 0.97, _stats_box(s), transform=ax.transAxes,
            fontsize=13, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xticks([1])
    ax.set_xticklabels([label], fontsize=15)
    ax.set_ylabel("Dwell time in cavity (ns)", fontsize=15)
    ax.set_title(f"{label} — Ion dwell time in cavity", fontsize=16, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_histogram(label, all_events, s, out_path, frame_duration, bins=100):
    """Histogram of dwell times with mean and median vertical lines."""
    d = np.array([e["dwell_ns"] for e in all_events])
    if len(d) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(d, bins=bins, color="#4878CF", alpha=0.72,
            edgecolor="white", linewidth=0.5)

    ax.axvline(s["mean_ns"],   color="#E74C3C", linewidth=2.2,
               linestyle="-",  label=f"Mean = {_fmt(s['mean_ns'], 2)} ns")
    ax.axvline(s["median_ns"], color="#2ECC71", linewidth=2.2,
               linestyle="--", label=f"Median = {_fmt(s['median_ns'], 2)} ns")
    ax.axvspan(s["mean_ns"] - s["std_ns"], s["mean_ns"] + s["std_ns"],
               alpha=0.12, color="#E74C3C",
               label=f"+/-1 std ({_fmt(s['std_ns'], 2)} ns)")

    ax.legend(fontsize=13)
    ax.set_xlabel("Dwell time in cavity (ns)", fontsize=15)
    ax.set_ylabel("Number of events", fontsize=15)
    ax.set_title(f"{label} — Dwell time histogram  (n = {s['pc']})", fontsize=16, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dwell time analysis for one GIRK simulation system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dwell_time_analysis.py /path/to/G12
  python3 dwell_time_analysis.py /path/to/G12_ML --frame_duration 0.075
  python3 dwell_time_analysis.py /path/to/G12_ML --frame_duration 0.075 --output /path/to/out
        """
    )
    parser.add_argument("base_dir",
                        help="Directory containing RUN1, RUN2, ... subdirectories")
    parser.add_argument("--frame_duration", type=float, default=0.075,
                        help="Duration of one frame in ns (default: 0.075)")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of bins in the dwell time histogram (default: 100)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <base_dir>/dwell_time_analysis)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return 1

    out_dir = Path(args.output) if args.output else base_dir / "dwell_time_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    label = base_dir.name

    print(f"\nSystem         : {label}")
    print(f"Directory      : {base_dir}")
    print(f"Frame duration : {args.frame_duration} ns")
    print(f"Histogram bins : {args.bins}")
    print(f"Dwell def.     : (R2_end - R1_end) x {args.frame_duration} ns")
    print(f"Output dir     : {out_dir}\n")

    # Load
    all_events, runs_detail, total_frames = load_data(base_dir, args.frame_duration)

    if not all_events:
        print("ERROR: No valid dwell times found. Check that permeation_table.json "
              "files exist and contain R1_end / R2_end fields.")
        return 1

    # Summary stats
    s = compute_summary(label, all_events, runs_detail, total_frames, args.frame_duration)

    # Print to terminal
    print_summary(s, runs_detail)

    # CSV files
    print("Saving CSV files...")
    save_summary_csv(s, runs_detail, out_dir / f"{label}_summary.csv")
    save_events_csv(all_events,      out_dir / f"{label}_all_events.csv")

    # Plots
    print("Generating plots...")
    plot_bar(      label, all_events, s,
                   out_dir / f"{label}_dwell_bar.png",       args.frame_duration)
    plot_box(      label, all_events, s,
                   out_dir / f"{label}_dwell_box.png",       args.frame_duration)
    plot_histogram(label, all_events, s,
                   out_dir / f"{label}_dwell_histogram.png", args.frame_duration, args.bins)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())

"""
python3 dwell_time_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dwell_time_analysis --frame_duration 0.1

python3 dwell_time_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dwell_time_analysis

python3 dwell_time_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dwell_time_analysis

python3 dwell_time_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dwell_time_analysis
"""