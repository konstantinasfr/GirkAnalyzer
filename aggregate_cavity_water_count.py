#!/usr/bin/env python3
"""
Aggregate Cavity Water Count — across multiple runs
=====================================================
For every RUN* directory:
  1. Reads  cavity_water_count/cavity_water_count.pkl  (water count per frame)
  2. Reads  permeation_table.json                      (permeation events)
  3. Extracts R2_end frames from the permeation table
     — these are exactly the frames when a K+ ion leaves the GLU/ASN cavity
  4. Splits all frames into:
       • permeation frames   : frames that appear as R2_end
       • non-permeation frames: everything else

Then produces:
  1. aggregate_cavity_water_count.png
       — one grouped bar pair (non-perm vs perm) aggregated over ALL runs
         with mean + std error bars
  2. aggregate_cavity_water_count_per_run.png
       — same grouped bars but one pair per run
  3. aggregate_cavity_water_count.txt
       — full numerical summary

How to run (command line)
--------------------------
    python aggregate_cavity_water_count.py \\
        --base_dir /path/to/results/G12 \\
        --output_dir /path/to/results/G12/aggregate \\
        --channel_type G12

Or import:
    from aggregate_cavity_water_count import run_aggregate_water_analysis
    run_aggregate_water_analysis(
        base_dir     = "/path/to/results/G12",
        output_dir   = "/path/to/results/G12/aggregate",
        channel_type = "G12",
    )
"""

import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_water_pkl(run_dir: Path):
    """Load cavity_water_count.pkl from a run directory."""
    pkl_path = run_dir / "cavity_water_count" / "cavity_water_count.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {run_dir.name}: pkl not found ({pkl_path})")
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def _load_r2_end_frames(run_dir: Path):
    """
    Read permeation_table.json and return a set of R2_end frame numbers.
    R2_end is the frame the ion leaves the GLU/ASN cavity (ch2).
    """
    json_path = run_dir / "permeation_table.json"
    if not json_path.exists():
        print(f"  [WARN] {run_dir.name}: permeation_table.json not found")
        return set()
    with open(json_path, "r") as f:
        events = json.load(f)

    r2_end_frames = set()
    for event in events:
        # permeation_table.json is a list of dicts with keys like R2_end
        val = event.get("R2_end")
        if val is not None:
            r2_end_frames.add(int(val))

    return r2_end_frames


def _split_frames(water_pkl, r2_end_frames):
    """
    Split water counts into permeation vs non-permeation.

    A frame is a 'permeation frame' if its absolute frame number appears
    in r2_end_frames (i.e. it is an R2_end frame from permeation_table.json).
    All other frames are 'non-permeation frames'.

    Returns
    -------
    perm_counts     : list[int]  — water counts at R2_end frames
    non_perm_counts : list[int]  — water counts at all other frames
    """
    frame_indices = water_pkl["frame_indices"]   # absolute frame numbers
    water_counts  = water_pkl["water_counts"]    # count per frame

    perm_counts     = []
    non_perm_counts = []

    for frame_abs, wc in zip(frame_indices, water_counts):
        if int(frame_abs) in r2_end_frames:
            perm_counts.append(int(wc))
        else:
            non_perm_counts.append(int(wc))

    return perm_counts, non_perm_counts


# ── plotting ───────────────────────────────────────────────────────────────────

def _bar_plot(means, stds, ns, labels, colors, title, ylabel, out_path):
    """
    Simple grouped bar chart with mean ± std and n annotations.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stds,
        color=colors, width=0.5,
        capsize=7,
        error_kw={"linewidth": 1.8, "ecolor": "black"},
        edgecolor="black", linewidth=0.9,
    )

    # annotate with mean and n
    y_offset = max(stds) * 0.15 if max(stds) > 0 else 0.3
    for bar, mean, std, n in zip(bars, means, stds, ns):
        top = bar.get_height() + std
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + y_offset,
            f"mean={mean:.2f}\nstd={std:.2f}\nn={n:,}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.5)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _per_run_bar_plot(run_names, run_means_non, run_stds_non,
                      run_means_perm, run_stds_perm,
                      channel_type, out_path):
    """
    Grouped bar chart: one pair of bars per run.
    """
    x     = np.arange(len(run_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(run_names) * 1.6), 5))

    ax.bar(x - width / 2, run_means_non, width,
           yerr=run_stds_non,
           label="Non-permeation frames",
           color="#2196F3", capsize=5,
           error_kw={"linewidth": 1.2, "ecolor": "black"},
           edgecolor="black", linewidth=0.7)

    ax.bar(x + width / 2, run_means_perm, width,
           yerr=run_stds_perm,
           label="Permeation frames",
           color="#E53935", capsize=5,
           error_kw={"linewidth": 1.2, "ecolor": "black"},
           edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean # water molecules in GLU/ASN cavity", fontsize=11)
    ax.set_title(
        f"GLU/ASN cavity waters per run — {channel_type}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def run_aggregate_water_analysis(
    base_dir,
    output_dir,
    channel_type="G12",
):
    base_dir   = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("AGGREGATE CAVITY WATER COUNT")
    print("=" * 70)
    print(f"  Base dir     : {base_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Output dir   : {output_dir}")

    # ── find RUN directories ───────────────────────────────────────────────────
    run_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("RUN")
    )
    if not run_dirs:
        print(f"  [ERROR] No RUN* directories found in {base_dir}")
        return

    print(f"\n  Found {len(run_dirs)} RUN directories\n")

    # ── load each run ──────────────────────────────────────────────────────────
    all_perm     = []   # all permeation-frame water counts across runs
    all_non_perm = []   # all non-permeation-frame water counts across runs

    run_names      = []
    run_means_non  = []
    run_stds_non   = []
    run_means_perm = []
    run_stds_perm  = []
    total_r2_events = 0

    # for the txt summary
    per_run_summary = []

    for run_dir in run_dirs:
        run_name = run_dir.name

        # load water counts
        water_pkl = _load_water_pkl(run_dir)
        if water_pkl is None:
            continue

        # load R2_end frames from permeation_table.json
        r2_end_frames = _load_r2_end_frames(run_dir)

        # filter R2_end frames to only those within the analysed range
        valid_r2 = {
            f for f in r2_end_frames
            if water_pkl["start_frame"] <= f <= water_pkl["end_frame"]
        }

        print(f"  {run_name}: {water_pkl['n_frames']} frames | "
              f"{len(r2_end_frames)} R2_end events in json | "
              f"{len(valid_r2)} within analysed range")

        # split
        perm, non_perm = _split_frames(water_pkl, valid_r2)
        total_r2_events += len(valid_r2)

        all_perm.extend(perm)
        all_non_perm.extend(non_perm)

        # per-run stats
        m_non = np.mean(non_perm) if non_perm else 0.0
        s_non = np.std(non_perm,  ddof=1) if len(non_perm) > 1 else 0.0
        m_per = np.mean(perm)     if perm     else 0.0
        s_per = np.std(perm,      ddof=1) if len(perm) > 1 else 0.0

        run_names.append(run_name)
        run_means_non.append(m_non)
        run_stds_non.append(s_non)
        run_means_perm.append(m_per)
        run_stds_perm.append(s_per)

        per_run_summary.append({
            "run"         : run_name,
            "n_non_perm"  : len(non_perm),
            "mean_non"    : m_non,
            "std_non"     : s_non,
            "n_perm"      : len(perm),
            "mean_perm"   : m_per,
            "std_perm"    : s_per,
            "r2_events"   : len(valid_r2),
        })

    if not run_names:
        print("  [ERROR] No data loaded.")
        return

    # ── overall statistics ─────────────────────────────────────────────────────
    mean_non = np.mean(all_non_perm) if all_non_perm else 0.0
    std_non  = np.std(all_non_perm, ddof=1) if len(all_non_perm) > 1 else 0.0
    mean_per = np.mean(all_perm)     if all_perm     else 0.0
    std_per  = np.std(all_perm, ddof=1) if len(all_perm) > 1 else 0.0

    print(f"\n  ── Overall ──────────────────────────────────────────")
    print(f"  Non-permeation : mean={mean_non:.2f}  std={std_non:.3f}"
          f"  n={len(all_non_perm):,} frames")
    print(f"  Permeation     : mean={mean_per:.2f}  std={std_per:.3f}"
          f"  n={len(all_perm):,} frames  ({total_r2_events} R2_end events)")

    # ── aggregate bar plot ─────────────────────────────────────────────────────
    _bar_plot(
        means  = [mean_non, mean_per],
        stds   = [std_non,  std_per],
        ns     = [len(all_non_perm), len(all_perm)],
        labels = ["Non-permeation\nframes", "Permeation\nframes"],
        colors = ["#2196F3", "#E53935"],
        title  = (f"GLU/ASN cavity waters — {channel_type}\n"
                  f"({len(run_names)} runs, {total_r2_events} K⁺ exit events)"),
        ylabel = "Mean # water molecules in GLU/ASN cavity",
        out_path = output_dir / "aggregate_cavity_water_count.png",
    )

    # ── per-run bar plot ───────────────────────────────────────────────────────
    _per_run_bar_plot(
        run_names      = run_names,
        run_means_non  = run_means_non,
        run_stds_non   = run_stds_non,
        run_means_perm = run_means_perm,
        run_stds_perm  = run_stds_perm,
        channel_type   = channel_type,
        out_path       = output_dir / "aggregate_cavity_water_count_per_run.png",
    )

    # ── save txt summary ───────────────────────────────────────────────────────
    txt_path = output_dir / "aggregate_cavity_water_count.txt"
    with open(txt_path, "w") as f:
        f.write("AGGREGATE CAVITY WATER COUNT SUMMARY\n")
        f.write(f"Channel type        : {channel_type}\n")
        f.write(f"Runs loaded         : {len(run_names)}\n")
        f.write(f"Total R2_end events : {total_r2_events}\n")
        f.write("=" * 65 + "\n\n")

        f.write("OVERALL\n")
        f.write(f"  Non-permeation : mean={mean_non:.3f}  std={std_non:.4f}"
                f"  n={len(all_non_perm):,} frames\n")
        f.write(f"  Permeation     : mean={mean_per:.3f}  std={std_per:.4f}"
                f"  n={len(all_perm):,} frames\n\n")

        f.write("PER RUN\n")
        hdr = (f"{'Run':<10} {'n_non':>8} {'mean_non':>10} {'std_non':>9} "
               f"{'n_perm':>8} {'mean_perm':>11} {'std_perm':>9} "
               f"{'R2_events':>10}\n")
        f.write(hdr)
        f.write("-" * 80 + "\n")
        for r in per_run_summary:
            f.write(
                f"{r['run']:<10} {r['n_non_perm']:>8} {r['mean_non']:>10.3f} "
                f"{r['std_non']:>9.4f} {r['n_perm']:>8} {r['mean_perm']:>11.3f} "
                f"{r['std_perm']:>9.4f} {r['r2_events']:>10}\n"
            )
    print(f"  ✅ Summary txt saved: {txt_path}")
    print("\n  Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate GLU/ASN cavity water count across runs."
    )
    parser.add_argument("--base_dir",     required=True,
                        help="Directory containing RUN1, RUN2, ...")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save outputs")
    parser.add_argument("--channel_type", default="G12",
                        help="Channel label for plot titles (default: G12)")
    args = parser.parse_args()

    run_aggregate_water_analysis(
        base_dir     = args.base_dir,
        output_dir   = args.output_dir,
        channel_type = args.channel_type,
    )

"""
python3 aggregate_cavity_water_count.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/aggregate_cavity_water \
--channel_type G2

python3 aggregate_cavity_water_count.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_cavity_water \
--channel_type G12

python3 aggregate_cavity_water_count.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/aggregate_cavity_water \
--channel_type G12_ML

python3 aggregate_cavity_water_count.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/aggregate_cavity_water \
--channel_type G12_GAT

"""