#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Aggregate Cavity Water Residence Analysis
==========================================
Loads cavity_water_residence.pkl from every RUN* subdirectory and
combines all residence time data into a single set of plots and a
summary file.

What it produces
-----------------
1. aggregate_water_residence.pkl     — combined results dict
2. aggregate_water_residence.txt     — per-run and overall summary
3. aggregate_water_residence.png     — three-panel plot:
     left  : bar = mean ± std of per-water means across all runs,
              dots = every individual water from every run
     middle: histogram of all individual residence event durations
              combined across all runs
     right : per-run comparison — one bar per run showing mean ± std

How to run (command line)
--------------------------
    python aggregate_cavity_water_residence.py \\
        --base_dir  /path/to/results/G12 \\
        --output_dir /path/to/results/G12/aggregate \\
        --channel_type G12

Or import:
    from aggregate_cavity_water_residence import run_aggregate_residence
    run_aggregate_residence(
        base_dir     = "/path/to/results/G12",
        output_dir   = "/path/to/results/G12/aggregate",
        channel_type = "G12",
    )
"""

import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── loader ────────────────────────────────────────────────────────────────────

def _load_run(run_dir: Path):
    """Load cavity_water_residence.pkl from one RUN directory."""
    pkl_path = run_dir / "cavity_water_residence" / "cavity_water_residence.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {run_dir.name}: pkl not found")
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        n_waters = data["n_waters_visited"]
        n_events = len(data["all_durations_flat"])
        print(f"  [OK]   {run_dir.name}: "
              f"{n_waters} waters  |  {n_events} events  |  "
              f"mean = {data['ev_mean']:.1f} ± {data['ev_std']:.1f} fr")
        return data
    except Exception as e:
        print(f"  [ERR]  {run_dir.name}: {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def run_aggregate_residence(
    base_dir,
    output_dir,
    channel_type="G12",
):
    base_dir   = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("AGGREGATE CAVITY WATER RESIDENCE")
    print("=" * 70)
    print(f"  Base dir     : {base_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Output dir   : {output_dir}")

    # ── find and load all RUN directories ─────────────────────────────────────
    run_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("RUN")
    )
    if not run_dirs:
        print(f"  [ERROR] No RUN* directories found in {base_dir}")
        return

    print(f"\n  Found {len(run_dirs)} RUN directories:\n")
    loaded = []
    for run_dir in run_dirs:
        data = _load_run(run_dir)
        if data is not None:
            loaded.append((run_dir.name, data))

    if not loaded:
        print("  [ERROR] No pkl files loaded.")
        return

    n_runs = len(loaded)
    print(f"\n  Loaded: {n_runs} runs")

    # ── combine across all runs ───────────────────────────────────────────────

    # all individual event durations combined
    all_durations_combined = []
    # all per-water means combined (one value per unique water per run)
    all_pw_means_combined  = []

    # per-run statistics for comparison plot
    run_names    = []
    run_ev_means = []
    run_ev_stds  = []
    run_n_waters = []
    run_n_events = []

    for run_name, data in loaded:
        run_names.append(run_name)
        run_ev_means.append(data["ev_mean"])
        run_ev_stds.append(data["ev_std"])
        run_n_waters.append(data["n_waters_visited"])
        run_n_events.append(len(data["all_durations_flat"]))

        all_durations_combined.extend(data["all_durations_flat"])
        all_pw_means_combined.extend(list(data["per_water_mean"].values()))

    # overall statistics
    all_durs = np.array(all_durations_combined)
    all_pw   = np.array(all_pw_means_combined)

    # per-water mean stats (for bar+dot plot)
    pw_mean = float(np.mean(all_pw))
    pw_std  = float(np.std(all_pw,  ddof=1)) if len(all_pw) > 1 else 0.0

    # individual event stats (for histogram)
    ev_mean   = float(np.mean(all_durs))
    ev_std    = float(np.std(all_durs,  ddof=1)) if len(all_durs) > 1 else 0.0
    ev_median = float(np.median(all_durs))
    ev_max    = int(all_durs.max())

    total_waters = len(all_pw_means_combined)
    total_events = len(all_durations_combined)

    print(f"\n  Total unique waters (all runs) : {total_waters}")
    print(f"  Total residence events         : {total_events}")
    print(f"  Per-water mean  (mean ± std)   : {pw_mean:.2f} ± {pw_std:.2f} frames")
    print(f"  Event mean ± std               : {ev_mean:.2f} ± {ev_std:.2f} frames")
    print(f"  Event median                   : {ev_median:.2f} frames")
    print(f"  Longest single residence       : {ev_max} frames")

    # ── save pkl ──────────────────────────────────────────────────────────────
    results = {
        "channel_type"          : channel_type,
        "n_runs"                : n_runs,
        "run_names"             : run_names,
        "all_durations_combined": all_durations_combined,
        "all_pw_means_combined" : all_pw_means_combined,
        "pw_mean"               : pw_mean,
        "pw_std"                : pw_std,
        "ev_mean"               : ev_mean,
        "ev_std"                : ev_std,
        "ev_median"             : ev_median,
        "ev_max"                : ev_max,
        "total_waters"          : total_waters,
        "total_events"          : total_events,
        "run_ev_means"          : run_ev_means,
        "run_ev_stds"           : run_ev_stds,
        "run_n_waters"          : run_n_waters,
        "run_n_events"          : run_n_events,
    }
    pkl_path = output_dir / "aggregate_water_residence.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  ✅ Saved: {pkl_path}")

    # ── save txt ──────────────────────────────────────────────────────────────
    txt_path = output_dir / "aggregate_water_residence.txt"
    with open(txt_path, "w") as f:
        f.write("AGGREGATE CAVITY WATER RESIDENCE\n")
        f.write(f"Channel type     : {channel_type}\n")
        f.write(f"Runs loaded      : {n_runs}\n")
        f.write(f"Total waters     : {total_waters}\n")
        f.write(f"Total events     : {total_events}\n")
        f.write(f"Per-water mean   : {pw_mean:.3f} ± {pw_std:.3f} frames\n")
        f.write(f"Event mean       : {ev_mean:.3f} ± {ev_std:.3f} frames\n")
        f.write(f"Event median     : {ev_median:.3f} frames\n")
        f.write(f"Longest residence: {ev_max} frames\n")
        f.write("=" * 60 + "\n\n")
        f.write("PER RUN SUMMARY\n")
        f.write(f"{'Run':<10} {'N waters':>10} {'N events':>10} "
                f"{'Event mean':>12} {'Event std':>11}\n")
        f.write("-" * 56 + "\n")
        for rn, nm, ne, rm, rs in zip(
            run_names, run_n_waters, run_n_events,
            run_ev_means, run_ev_stds
        ):
            f.write(f"{rn:<10} {nm:>10} {ne:>10} "
                    f"{rm:>12.3f} {rs:>11.3f}\n")
    print(f"  ✅ Saved: {txt_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    _plot(results, output_dir)
    _plot_simple(results, output_dir)
    print("\n  Done.")


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot(results, out_dir):
    channel_type = results["channel_type"]
    all_pw       = np.array(results["all_pw_means_combined"])
    all_durs     = np.array(results["all_durations_combined"])
    pw_mean      = results["pw_mean"]
    pw_std       = results["pw_std"]
    ev_mean      = results["ev_mean"]
    ev_std       = results["ev_std"]
    ev_median    = results["ev_median"]
    ev_max       = results["ev_max"]
    run_names    = results["run_names"]
    run_ev_means = results["run_ev_means"]
    run_ev_stds  = results["run_ev_stds"]
    n_runs       = results["n_runs"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              gridspec_kw={"width_ratios": [1, 2, 1.5]})
    fig.suptitle(
        f"GLU/ASN cavity water residence times — {channel_type}  "
        f"({n_runs} runs)\n"
        f"Per-water mean = {pw_mean:.1f} ± {pw_std:.1f} fr  |  "
        f"Event mean = {ev_mean:.1f} ± {ev_std:.1f} fr  |  "
        f"{results['total_waters']} unique waters  |  "
        f"{results['total_events']} events",
        fontsize=11, fontweight="bold"
    )

    # ── panel 1: bar + dots (per-water means) ─────────────────────────────────
    ax = axes[0]
    ax.bar([0], [pw_mean], yerr=[pw_std],
           color="#2196F3", width=0.45, capsize=10,
           error_kw={"linewidth": 2.0, "ecolor": "#0D47A1"},
           edgecolor="#0D47A1", linewidth=0.8, zorder=2,
           label=f"Mean ± std\n({pw_mean:.1f} ± {pw_std:.1f} fr)")

    np.random.seed(42)
    jitter = np.random.uniform(-0.18, 0.18, size=len(all_pw))
    ax.scatter(jitter, all_pw,
               color="#E53935", s=14, alpha=0.45, zorder=5,
               label=f"Each water (n={len(all_pw)})")

    ax.set_xticks([0])
    ax.set_xticklabels(["All runs combined"], fontsize=9)
    ax.set_ylabel("Mean residence time per water (frames)", fontsize=10)
    ax.set_xlim(-0.5, 0.5)
    y_top = max(all_pw.max() * 1.15, pw_mean + pw_std * 2.5, 2)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title("Per-water mean residence\n(one dot = one water molecule)",
                 fontsize=9)

    # ── panel 2: histogram of all individual event durations ──────────────────
    ax2 = axes[1]

    cutoff    = min(ev_max, max(int(ev_mean + 6 * ev_std) + 1, 10))
    bin_edges = np.arange(0.5, cutoff + 1.5, 1.0)    # cutoff+1 edges → cutoff bins
    bar_x     = np.arange(1, cutoff + 1)              # cutoff values

    vals_trunc  = [d for d in all_durs if d <= cutoff]
    n_truncated = len(all_durs) - len(vals_trunc)
    counts, _   = np.histogram(vals_trunc, bins=bin_edges)

    ax2.bar(bar_x, counts,
            color="#2196F3", edgecolor="#0D47A1", linewidth=0.4, width=0.8)

    if cutoff <= 40:
        for x, c in zip(bar_x, counts):
            if c > 0:
                ax2.text(x, c + max(counts) * 0.01, str(c),
                         ha="center", va="bottom", fontsize=6)

    ax2.axvline(ev_mean, color="#E53935", linewidth=2.0,
                linestyle="--", label=f"Mean = {ev_mean:.1f} fr")
    ax2.axvline(ev_median, color="#FF9800", linewidth=1.5,
                linestyle=":", label=f"Median = {ev_median:.1f} fr")

    ax2.set_xticks(bar_x[::max(1, len(bar_x) // 20)])
    ax2.set_xlabel(
        "Residence duration (frames)"
        + (f"   [+{n_truncated} events > {cutoff} fr not shown]"
           if n_truncated > 0 else ""),
        fontsize=10
    )
    ax2.set_ylabel("Number of residence events", fontsize=10)
    ax2.set_title(
        f"All individual residence events combined\n"
        f"(n = {len(all_durs)} events from {n_runs} runs)",
        fontsize=9
    )
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # ── panel 3: per-run comparison ───────────────────────────────────────────
    ax3 = axes[2]
    x  = np.arange(n_runs)
    ax3.bar(x, run_ev_means, yerr=run_ev_stds,
            color="#2196F3", width=0.6, capsize=6,
            error_kw={"linewidth": 1.5, "ecolor": "#0D47A1"},
            edgecolor="#0D47A1", linewidth=0.7)

    # annotate with n_events
    for xi, (m, ne) in enumerate(zip(run_ev_means, results["run_n_events"])):
        ax3.text(xi, m + max(run_ev_stds) * 0.15,
                 f"n={ne}", ha="center", va="bottom", fontsize=7)

    ax3.axhline(ev_mean, color="#E53935", linewidth=1.5,
                linestyle="--", label=f"Overall mean = {ev_mean:.1f} fr")
    ax3.set_xticks(x)
    ax3.set_xticklabels(run_names, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Mean event residence (frames)", fontsize=10)
    ax3.set_title("Per-run comparison\n(mean ± std of event durations)",
                  fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax3.set_ylim(0, max(
        [m + s for m, s in zip(run_ev_means, run_ev_stds)] + [ev_mean * 1.3]
    ) * 1.15)

    plt.tight_layout()
    out_path = out_dir / "aggregate_water_residence.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")



def _plot_simple(results, out_dir):
    """
    Clean two-panel plot — no dots on the bar, bigger font sizes.
    Left  : bar only (mean ± std of per-water means)
    Right : histogram of all individual residence event durations
    """
    channel_type = results["channel_type"]
    pw_mean      = results["pw_mean"]
    pw_std       = results["pw_std"]
    ev_mean      = results["ev_mean"]
    ev_std       = results["ev_std"]
    ev_median    = results["ev_median"]
    ev_max       = results["ev_max"]
    all_durs     = np.array(results["all_durations_combined"])
    n_runs       = results["n_runs"]

    FS_TITLE  = 14   # panel title font size
    FS_LABEL  = 13   # axis label font size
    FS_TICK   = 12   # tick label font size
    FS_LEGEND = 11   # legend font size
    FS_ANNOT  = 10   # bar count annotations

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(
        f"GLU/ASN cavity water residence — {channel_type}  ({n_runs} runs)\n"
        f"Per-water mean = {pw_mean:.1f} ± {pw_std:.1f} fr  |  "
        f"Event mean = {ev_mean:.1f} ± {ev_std:.1f} fr  |  "
        f"{results['total_waters']} unique waters  |  "
        f"{results['total_events']} events",
        fontsize=FS_TITLE, fontweight="bold"
    )

    # ── left: bar only ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar([0], [pw_mean], yerr=[pw_std],
           color="#2196F3", width=0.45, capsize=12,
           error_kw={"linewidth": 2.2, "ecolor": "#0D47A1"},
           edgecolor="#0D47A1", linewidth=0.9, zorder=2,
           label=f"Mean ± std\n({pw_mean:.1f} ± {pw_std:.1f} frames)")

    ax.set_xticks([0])
    ax.set_xticklabels(["All runs combined"], fontsize=FS_TICK)
    ax.set_ylabel("Mean residence time per water (frames)", fontsize=FS_LABEL)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, (pw_mean + pw_std * 3) * 1.2)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title("Per-water mean residence time", fontsize=FS_TITLE)

    # ── right: histogram ──────────────────────────────────────────────────────
    ax2 = axes[1]

    cutoff    = min(ev_max, max(int(ev_mean + 6 * ev_std) + 1, 10))
    bin_edges = np.arange(0.5, cutoff + 1.5, 1.0)
    bar_x     = np.arange(1, cutoff + 1)

    vals_trunc  = [d for d in all_durs if d <= cutoff]
    n_truncated = len(all_durs) - len(vals_trunc)
    counts, _   = np.histogram(vals_trunc, bins=bin_edges)

    ax2.bar(bar_x, counts,
            color="#2196F3", edgecolor="#0D47A1", linewidth=0.5, width=0.8)

    if cutoff <= 40:
        for x, c in zip(bar_x, counts):
            if c > 0:
                ax2.text(x, c + max(counts) * 0.012, str(c),
                         ha="center", va="bottom", fontsize=FS_ANNOT)

    ax2.axvline(ev_mean, color="#E53935", linewidth=2.0,
                linestyle="--", label=f"Mean = {ev_mean:.1f} fr")
    ax2.axvline(ev_median, color="#FF9800", linewidth=1.8,
                linestyle=":", label=f"Median = {ev_median:.1f} fr")

    ax2.set_xticks(bar_x[::max(1, len(bar_x) // 20)])
    ax2.tick_params(axis="both", labelsize=FS_TICK)
    ax2.set_xlabel(
        "Residence duration (frames)"
        + (f"   [+{n_truncated} events > {cutoff} fr not shown]"
           if n_truncated > 0 else ""),
        fontsize=FS_LABEL
    )
    ax2.set_ylabel("Number of residence events", fontsize=FS_LABEL)
    ax2.set_title(
        f"Distribution of all residence event durations\n"
        f"(n = {len(all_durs)} events from {n_runs} runs)",
        fontsize=FS_TITLE
    )
    ax2.legend(fontsize=FS_LEGEND)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path = out_dir / "aggregate_water_residence_simple.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate cavity water residence times across runs."
    )
    parser.add_argument("--base_dir",     required=True,
                        help="Directory containing RUN1, RUN2, ...")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save outputs")
    parser.add_argument("--channel_type", default="G12",
                        help="Channel label for plot titles (default: G12)")
    args = parser.parse_args()

    run_aggregate_residence(
        base_dir     = args.base_dir,
        output_dir   = args.output_dir,
        channel_type = args.channel_type,
    )

"""
python3 aggregate_cavity_water_residence.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
    --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/aggregate_cavity_water \
    --channel_type G2

python3 aggregate_cavity_water_residence.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_cavity_water \
--channel_type G12

python3 aggregate_cavity_water_residence.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/aggregate_cavity_water \
--channel_type G12_ML


python3 aggregate_cavity_water_residence.py \
--base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/aggregate_cavity_water \
--channel_type G12_GAT

"""