"""
aggregate_untracked_cavity_ions.py
===================================
Reads the outputs of untracked_cavity_ions.py from every RUN* subfolder
and produces pooled statistics and plots across all runs.

Expects this folder structure:
    <base_dir>/
        RUN1/untracked_cavity_ions/untracked_ions.csv
        RUN1/untracked_cavity_ions/summary.csv
        RUN2/untracked_cavity_ions/...
        ...

Outputs (saved to <base_dir>/untracked_cavity_ions_aggregated/):
    all_observations.csv      every untracked (frame, ion) row, tagged with run
    all_ion_summaries.csv     per-ion summary rows from all runs, tagged with run
    aggregated_summary.csv    global stats: total events, unique ions, mean time etc.
    untracked_count.png       time-series of untracked ion count per frame (all runs)
    top_ions_bar.png          bar chart of top untracked ions by total frames in cavity

Usage:
    python3 aggregate_untracked_cavity_ions.py /path/to/G12
    python3 aggregate_untracked_cavity_ions.py /path/to/G12 --label G12
    python3 aggregate_untracked_cavity_ions.py /path/to/G12 --output /path/to/out


    python3 aggregate_untracked_cavity_ions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/untracked_cavity_ion \
--label G2 

python3 aggregate_untracked_cavity_ions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/untracked_cavity_ion \
--label G12 

python3 aggregate_untracked_cavity_ions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/untracked_cavity_ion \
--label G12_ML 

python3 aggregate_untracked_cavity_ions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/untracked_cavity_ion \
--label G12_GAT 
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── loading ───────────────────────────────────────────────────────────────────

def load_all_runs(base_dir: Path, frame_duration: float):
    """
    Walk every RUN* subfolder, read:
      - untracked_ions.csv   → per-frame observations
      - summary.csv          → per-ion summaries
    """
    all_observations = []   # list of dicts: {run, frame, ion_resid, x, y, z}
    all_summaries    = []   # list of dicts: {run, ion_resid, n_frames, ...}

    run_dirs = sorted(base_dir.glob("RUN*"))
    if not run_dirs:
        raise FileNotFoundError(f"No RUN* directories found in {base_dir}")

    for run_dir in run_dirs:
        ana_dir = run_dir / "untracked_cavity_ions"

        obs_csv = ana_dir / "untracked_ions.csv"
        sum_csv = ana_dir / "summary.csv"

        if not obs_csv.exists() and not sum_csv.exists():
            print(f"  [SKIP] {run_dir.name} — no untracked_cavity_ions output")
            continue

        run_obs = 0
        if obs_csv.exists():
            with open(obs_csv, newline="") as f:
                for row in csv.DictReader(f):
                    converted = {}
                    for k, v in row.items():
                        try:
                            converted[k] = float(v) if "." in v else int(v)
                        except (ValueError, TypeError):
                            converted[k] = v
                    converted["run"] = run_dir.name
                    all_observations.append(converted)
                    run_obs += 1

        run_ions = 0
        if sum_csv.exists():
            with open(sum_csv, newline="") as f:
                for row in csv.DictReader(f):
                    converted = {}
                    for k, v in row.items():
                        try:
                            converted[k] = float(v) if "." in str(v) else int(v)
                        except (ValueError, TypeError):
                            converted[k] = v
                    converted["run"] = run_dir.name
                    all_summaries.append(converted)
                    run_ions += 1

        print(f"  {run_dir.name}: {run_obs} observations, {run_ions} unique untracked ions")

    return all_observations, all_summaries


# ── saving ────────────────────────────────────────────────────────────────────

def save_all_observations(rows, out_path):
    if not rows:
        return
    fields = ["run"] + [f for f in rows[0].keys() if f != "run"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


def save_all_summaries(rows, out_path):
    if not rows:
        return
    fields = ["run"] + [f for f in rows[0].keys() if f != "run"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


def save_aggregated_summary(observations, summaries, label,
                            n_runs, frames_per_run, frame_duration, out_path):
    """Global statistics across all runs."""

    # unique ions pooled across runs (identified by run+resid)
    unique_ions = set((r["run"], r["ion_resid"]) for r in summaries)
    n_frames_per_ion = [r["n_frames"] for r in summaries]
    time_per_ion     = [r["time_ns"]  for r in summaries]

    # per-run untracked ion counts
    runs_with_untracked = set(r["run"] for r in summaries)

    # total simulation frames across all runs
    total_sim_frames = n_runs * frames_per_run

    # frames that had at least one untracked ion (unique run+frame pairs)
    frames_with_untracked = set(
        (obs["run"], int(obs["frame"])) for obs in observations
    )
    n_frames_with_untracked = len(frames_with_untracked)
    pct_frames_with_untracked = (
        100.0 * n_frames_with_untracked / total_sim_frames
        if total_sim_frames > 0 else 0.0
    )

    rows = [
        {"metric": "system",                       "value": label,             "unit": ""},
        {"metric": "n_runs_loaded",                 "value": n_runs,            "unit": ""},
        {"metric": "frames_per_run",                "value": frames_per_run,    "unit": "frames"},
        {"metric": "total_simulation_frames",       "value": total_sim_frames,  "unit": "frames"},
        {"metric": "n_runs_with_untracked_ions",    "value": len(runs_with_untracked), "unit": ""},
        {"metric": "unique_untracked_ions",         "value": len(unique_ions),  "unit": ""},
        {"metric": "total_untracked_observations",  "value": len(observations), "unit": "frames"},
        {"metric": "frames_with_untracked_ions",    "value": n_frames_with_untracked,
         "unit": f"out of {total_sim_frames}"},
        {"metric": "pct_frames_with_untracked_ions","value": f"{pct_frames_with_untracked:.3f}",
         "unit": "%"},
    ]

    if n_frames_per_ion:
        rows += [
            {"metric": "mean_frames_per_ion", "value": f"{np.mean(n_frames_per_ion):.1f}", "unit": "frames"},
            {"metric": "std_frames_per_ion",  "value": f"{np.std(n_frames_per_ion):.1f}",  "unit": "frames"},
            {"metric": "max_frames_per_ion",  "value": f"{np.max(n_frames_per_ion)}",       "unit": "frames"},
            {"metric": "mean_time_per_ion",   "value": f"{np.mean(time_per_ion):.2f}",      "unit": "ns"},
            {"metric": "std_time_per_ion",    "value": f"{np.std(time_per_ion):.2f}",       "unit": "ns"},
            {"metric": "max_time_per_ion",    "value": f"{np.max(time_per_ion):.2f}",       "unit": "ns"},
        ]

    # per-frame counts from observations
    if observations:
        frame_counts = defaultdict(int)
        for obs in observations:
            frame_counts[(obs["run"], obs["frame"])] += 1
        counts_per_frame = list(frame_counts.values())
        rows += [
            {"metric": "mean_untracked_ions_per_frame",
             "value": f"{np.mean(counts_per_frame):.3f}", "unit": "ions/frame"},
            {"metric": "max_untracked_ions_in_one_frame",
             "value": f"{np.max(counts_per_frame)}", "unit": "ions"},
        ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "unit"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_time_series(observations, label, frame_duration, out_path):
    """
    For each run separately, plot untracked ion count per frame over time.
    All runs shown as faint lines, mean across runs as bold line.
    """
    if not observations:
        print("  No observations to plot.")
        return

    # group by run
    by_run = defaultdict(lambda: defaultdict(int))
    for obs in observations:
        by_run[obs["run"]][int(obs["frame"])] += 1

    fig, ax = plt.subplots(figsize=(14, 5))

    all_series = []
    for run, frame_dict in sorted(by_run.items()):
        frames = sorted(frame_dict.keys())
        counts = [frame_dict[f] for f in frames]
        t = np.array(frames) * frame_duration
        ax.plot(t, counts, color="#C0392B", alpha=0.20, linewidth=0.7)
        all_series.append((frames, counts))

    # mean across runs on a common frame grid
    if len(all_series) > 1:
        all_frames = sorted(set(f for frames, _ in all_series for f in frames))
        mat = []
        for frames, counts in all_series:
            d = dict(zip(frames, counts))
            mat.append([d.get(f, 0) for f in all_frames])
        mean_counts = np.mean(mat, axis=0)
        t_mean = np.array(all_frames) * frame_duration
        ax.plot(t_mean, mean_counts, color="#C0392B", linewidth=2.2,
                label=f"Mean across {len(all_series)} runs")

    ax.set_xlabel("Time (ns)", fontsize=15)
    ax.set_ylabel("Untracked K⁺ ions in ch2", fontsize=15)
    ax.set_title(
        f"Untracked ions in GLU/ASN cavity per frame\n"
        f"{label}  ({len(by_run)} runs)",
        fontsize=15, pad=12)
    ax.tick_params(labelsize=13)
    if len(all_series) > 1:
        ax.legend(fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_top_ions(summaries, label, out_path, top_n=20):
    """
    Bar chart of the top N untracked ions by total frames in cavity,
    coloured by run.
    """
    if not summaries:
        print("  No summaries to plot.")
        return

    # sort by n_frames descending, take top_n
    sorted_rows = sorted(summaries, key=lambda r: r["n_frames"], reverse=True)[:top_n]

    ion_labels = [f"{r['run']}\n{r['ion_resid']}" for r in sorted_rows]
    n_frames   = [r["n_frames"] for r in sorted_rows]
    time_ns    = [r["time_ns"]  for r in sorted_rows]

    # colour by run
    runs       = sorted(set(r["run"] for r in sorted_rows))
    cmap       = plt.cm.get_cmap("tab10", len(runs))
    run_color  = {run: cmap(i) for i, run in enumerate(runs)}
    colors     = [run_color[r["run"]] for r in sorted_rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(sorted_rows) * 0.9), 6))

    for ax, vals, ylabel, title_suffix in [
        (axes[0], n_frames, "Frames in cavity", "by frame count"),
        (axes[1], time_ns,  "Time in cavity (ns)", "by time (ns)"),
    ]:
        bars = ax.bar(range(len(ion_labels)), vals, color=colors,
                      edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.set_xticks(range(len(ion_labels)))
        ax.set_xticklabels(ion_labels, fontsize=10, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f"Top {len(sorted_rows)} untracked ions — {title_suffix}\n{label}",
                     fontsize=14, pad=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)

        # annotate
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if ylabel.startswith("Frames") else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)

    # legend for runs
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=run_color[r], label=r) for r in runs]
    axes[1].legend(handles=legend_handles, title="Run", fontsize=10,
                   title_fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── terminal summary ──────────────────────────────────────────────────────────

def print_summary(observations, summaries, label, n_runs, frames_per_run):
    unique_ions      = set((r["run"], r["ion_resid"]) for r in summaries)
    runs_with_ions   = set(r["run"] for r in summaries)
    n_frames_per_ion = [r["n_frames"] for r in summaries]
    time_per_ion     = [r["time_ns"]  for r in summaries]

    total_sim_frames        = n_runs * frames_per_run
    frames_with_untracked   = set((obs["run"], int(obs["frame"])) for obs in observations)
    n_frames_with_untracked = len(frames_with_untracked)
    pct = 100.0 * n_frames_with_untracked / total_sim_frames if total_sim_frames > 0 else 0.0

    print()
    print(f"  {'='*58}")
    print(f"  AGGREGATED SUMMARY — {label}")
    print(f"  {'='*58}")
    print(f"  Runs loaded                    : {n_runs}")
    print(f"  Frames per run                 : {frames_per_run}")
    print(f"  Total simulation frames        : {total_sim_frames}")
    print(f"  Runs with untracked ions       : {len(runs_with_ions)}")
    print(f"  Unique untracked ions          : {len(unique_ions)}")
    print(f"  Total observations             : {len(observations)}")
    print(f"  Frames with untracked ions     : {n_frames_with_untracked}"
          f"  /  {total_sim_frames}  ({pct:.2f}%)")
    if n_frames_per_ion:
        print(f"  Frames per ion  mean ± std     : "
              f"{np.mean(n_frames_per_ion):.1f} ± {np.std(n_frames_per_ion):.1f}")
        print(f"  Time per ion    mean ± std (ns): "
              f"{np.mean(time_per_ion):.2f} ± {np.std(time_per_ion):.2f}")
        print(f"  Max frames (single ion)        : {np.max(n_frames_per_ion)}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate untracked_cavity_ions results across all RUN folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 aggregate_untracked_cavity_ions.py /path/to/G12
  python3 aggregate_untracked_cavity_ions.py /path/to/G12_ML --label G12_ML
  python3 aggregate_untracked_cavity_ions.py /path/to/G12 --output /path/to/out
        """
    )
    parser.add_argument("base_dir",
                        help="Directory containing RUN1, RUN2, ... subdirectories")
    parser.add_argument("--label", default=None,
                        help="Label for plots and CSV (default: folder name)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <base_dir>/untracked_cavity_ions_aggregated)")
    parser.add_argument("--frame_duration", type=float, default=0.075,
                        help="ns per frame (default: 0.075)")
    parser.add_argument("--frames_per_run", type=int, default=6800,
                        help="Number of frames per simulation run (default: 6800)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top ions to show in bar chart (default: 20)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return 1

    label   = args.label or base_dir.name
    out_dir = Path(args.output) if args.output else \
              base_dir / "untracked_cavity_ions_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSystem          : {label}")
    print(f"Base dir        : {base_dir}")
    print(f"Output          : {out_dir}")
    print(f"Frame duration  : {args.frame_duration} ns")
    print(f"Frames per run  : {args.frames_per_run}\n")

    # ── load ──────────────────────────────────────────────────────────────
    observations, summaries = load_all_runs(base_dir, args.frame_duration)
    n_runs = len(sorted(base_dir.glob("RUN*")))

    print_summary(observations, summaries, label, n_runs, args.frames_per_run)

    # ── save CSVs ─────────────────────────────────────────────────────────
    save_all_observations(observations, out_dir / "all_observations.csv")
    save_all_summaries(summaries,       out_dir / "all_ion_summaries.csv")
    save_aggregated_summary(observations, summaries, label, n_runs,
                            args.frames_per_run, args.frame_duration,
                            out_dir / "aggregated_summary.csv")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_time_series(observations, label, args.frame_duration,
                     out_dir / "untracked_count.png")
    plot_top_ions(summaries, label,
                  out_dir / "top_ions_bar.png", top_n=args.top_n)

    print(f"\nDone. All outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())