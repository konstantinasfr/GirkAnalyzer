"""
aggregate_ion_ion_permeation.py
================================
Reads per_event.csv produced by ion_ion_permeation_analysis.py from every
RUN* subfolder and produces pooled statistics and plots across all runs.

Expects this folder structure:
    <base_dir>/
        RUN1/ion_ion_permeation/per_event.csv
        RUN2/ion_ion_permeation/per_event.csv
        ...

Outputs (saved to <base_dir>/ion_ion_permeation_aggregated/):
    all_events.csv          all per-event rows from every run (with run label)
    aggregated_summary.csv  pooled mean +/- std / min / max across all events
    dist_bar.png            bar chart of distance metrics
    dist_hist.png           histograms + KDE for distance metrics
    force_mag_bar.png       bar chart of force magnitudes
    force_mag_hist.png      histograms + KDE for force magnitudes
    force_axial_bar.png     bar chart of axial forces
    force_axial_hist.png    histograms + KDE for axial forces

Usage:
    python3 aggregate_ion_ion_permeation.py /path/to/G12
    python3 aggregate_ion_ion_permeation.py /path/to/G12 --label G12
    python3 aggregate_ion_ion_permeation.py /path/to/G12 --output /path/to/out

python3 aggregate_ion_ion_permeation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ion_ion_permeation \
--label G2 

python3 aggregate_ion_ion_permeation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ion_ion_permeation \
--label G12 

python3 aggregate_ion_ion_permeation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ion_ion_permeation \
--label G12_ML 

python3 aggregate_ion_ion_permeation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ion_ion_permeation \
--label G12_GAT 

"""

"""
aggregate_ion_ion_permeation.py
================================
Reads per_event.csv produced by ion_ion_permeation_analysis.py from every
RUN* subfolder and produces pooled statistics and plots across all runs.

Expects this folder structure:
    <base_dir>/
        RUN1/ion_ion_permeation/per_event.csv
        RUN2/ion_ion_permeation/per_event.csv
        ...

Outputs (saved to <base_dir>/ion_ion_permeation_aggregated/):
    all_events.csv          all per-event rows from every run (with run label)
    aggregated_summary.csv  pooled mean +/- std / min / max across all events
    dist_bar.png            bar chart of distance metrics
    dist_hist.png           histograms + KDE for distance metrics
    force_mag_bar.png       bar chart of force magnitudes
    force_mag_hist.png      histograms + KDE for force magnitudes
    force_axial_bar.png     bar chart of axial forces
    force_axial_hist.png    histograms + KDE for axial forces

Usage:
    python3 aggregate_ion_ion_permeation.py /path/to/G12
    python3 aggregate_ion_ion_permeation.py /path/to/G12 --label G12
    python3 aggregate_ion_ion_permeation.py /path/to/G12 --output /path/to/out
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── column definitions (must match ion_ion_permeation_analysis.py) ───────────

DIST_METRICS = ["mean_dist_all", "mean_dist_sf", "mean_dist_cavity",
                 "min_dist_all", "dist_closest_above"]
DIST_NAMES   = ["Mean\n(all)", "Mean\n(SF)", "Mean\n(cavity)",
                "Min\n(closest)", "Closest\nabove"]

FMAG_METRICS = ["force_all_mag", "force_sf_mag",
                 "force_cavity_mag", "force_closest_mag"]
FMAG_NAMES   = ["Total\n(SF+cavity)", "SF\nonly",
                "Cavity\nonly", "Closest\nion above"]

FAX_METRICS  = ["force_all_axial", "force_sf_axial",
                "force_cavity_axial", "force_closest_axial"]
FAX_NAMES    = ["Total\n(SF+cavity)", "SF\nonly",
                "Cavity\nonly", "Closest\nion above"]

ALL_NUMERIC  = DIST_METRICS + FMAG_METRICS + FAX_METRICS


# ── loading ───────────────────────────────────────────────────────────────────

def load_all_runs(base_dir: Path):
    """
    Walk every RUN* subfolder, read ion_ion_permeation/per_event.csv,
    tag each row with the run name, return combined list of dicts.
    """
    all_rows = []
    run_dirs = sorted(base_dir.glob("RUN*"))

    if not run_dirs:
        raise FileNotFoundError(f"No RUN* directories found in {base_dir}")

    for run_dir in run_dirs:
        csv_path = run_dir / "ion_ion_permeation" / "per_event.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {run_dir.name} — no per_event.csv")
            continue

        run_rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                converted = {}
                for k, v in row.items():
                    try:
                        converted[k] = float(v)
                    except (ValueError, TypeError):
                        converted[k] = v if v != "" else None
                converted["run"] = run_dir.name
                run_rows.append(converted)

        all_rows.extend(run_rows)
        print(f"  {run_dir.name}: {len(run_rows)} events")

    return all_rows


# ── saving ────────────────────────────────────────────────────────────────────

def save_all_events(rows, out_path):
    if not rows:
        return
    fields = ["run"] + [f for f in rows[0].keys() if f != "run"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


def save_summary(rows, label, out_path):
    summary = [
        {"metric": "system",   "mean": label,    "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "n_events", "mean": len(rows), "std": "", "min": "", "max": "", "n": len(rows)},
    ]
    for col in ALL_NUMERIC:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        summary.append({
            "metric": col,
            "mean":   f"{np.mean(vals):.3f}",
            "std":    f"{np.std(vals):.3f}",
            "min":    f"{np.min(vals):.3f}",
            "max":    f"{np.max(vals):.3f}",
            "n":      len(vals),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"  Saved: {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def _vals(rows, col):
    return [r[col] for r in rows if r.get(col) is not None]


def bar_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    """Bar chart: mean ± std with jittered individual event dots."""
    means, stds, all_vals = [], [], []
    for m in metrics:
        v = _vals(rows, m)
        means.append(np.mean(v) if v else 0)
        stds.append(np.std(v)   if v else 0)
        all_vals.append(v)

    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 3), 7))
    x   = np.arange(len(names))
    rng = np.random.default_rng(42)

    ax.bar(x, means, yerr=stds, capsize=10, width=0.55,
           color=color, alpha=0.75, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=2.0, ecolor="black"), zorder=2)

    for i, v in enumerate(all_vals):
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(i + jitter, v, color=color, alpha=0.25, s=28,
                   edgecolors="none", zorder=3)

    data_top = max((max(v) for v in all_vals if v), default=1)
    data_top = max(data_top, max(m + s for m, s in zip(means, stds)))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.2f}\n±{s:.2f}",
                ha="center", va="bottom", fontsize=16)

    ax.set_ylim(bottom=0, top=data_top * 1.40)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} events)", fontsize=20, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=17)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def hist_plot(rows, metrics, names, label, xlabel, out_path, color):
    """Histograms + KDE, one panel per metric."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m, name in zip(axes, metrics, names):
        v = np.array(_vals(rows, m))
        if len(v) == 0:
            ax.set_title(name, fontsize=18)
            continue

        ax.hist(v, bins=max(8, len(v) // 3), color=color, alpha=0.55,
                edgecolor="white", linewidth=0.5, density=True)

        if len(v) >= 3:
            try:
                kde = gaussian_kde(v, bw_method="scott")
                xg  = np.linspace(v.min(), v.max(), 200)
                ax.plot(xg, kde(xg), color=color, linewidth=2.5)
            except Exception:
                pass

        ax.axvline(np.mean(v), color="black", linewidth=1.6,
                   linestyle="--", label=f"mean = {np.mean(v):.2f}")
        ax.set_xlabel(name, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=15)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{label} — {xlabel}", fontsize=20, y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def box_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    """
    Boxplot for each metric. X-axis tick labels show:
        metric name
        mean = X.XXX
        median = X.XXX
    """
    all_vals = [_vals(rows, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 7))

    ax.boxplot(
        all_vals,
        positions=range(len(names)),
        widths=0.50,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.35,
                        markerfacecolor=color, markeredgecolor="none"),
        medianprops=dict(color="black", linewidth=2.5),
        boxprops=dict(facecolor=color, alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
    )

    # x-tick labels: name + mean + median
    tick_labels = []
    for name, v in zip(names, all_vals):
        if v:
            mn  = np.mean(v)
            med = np.median(v)
            tick_labels.append(f"{name}\nmean={mn:.2f}\nmedian={med:.2f}")
        else:
            tick_labels.append(name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} events)",
                 fontsize=20, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── terminal summary ──────────────────────────────────────────────────────────

def print_summary(rows, label):
    print()
    print(f"  {'='*58}")
    print(f"  AGGREGATED SUMMARY — {label}  (n = {len(rows)} events)")
    print(f"  {'='*58}")
    print("  DISTANCES (Å):")
    for col, name in zip(DIST_METRICS, DIST_NAMES):
        v = _vals(rows, col)
        if v:
            name_clean = name.replace("\n", " ")
            print(f"    {name_clean:<28}  {np.mean(v):6.2f} +/- {np.std(v):.2f}")
    print("  FORCE MAGNITUDES (kcal/(mol·Å)):")
    for col, name in zip(FMAG_METRICS, FMAG_NAMES):
        v = _vals(rows, col)
        if v:
            name_clean = name.replace("\n", " ")
            print(f"    {name_clean:<28}  {np.mean(v):6.2f} +/- {np.std(v):.2f}")
    print("  AXIAL FORCES (kcal/(mol·Å), +ve = toward HBC):")
    for col, name in zip(FAX_METRICS, FAX_NAMES):
        v = _vals(rows, col)
        if v:
            name_clean = name.replace("\n", " ")
            print(f"    {name_clean:<28}  {np.mean(v):+6.2f} +/- {np.std(v):.2f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ion_ion_permeation_analysis results across all RUN folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 aggregate_ion_ion_permeation.py /path/to/G12
  python3 aggregate_ion_ion_permeation.py /path/to/G12_ML --label G12_ML
  python3 aggregate_ion_ion_permeation.py /path/to/G12 --output /path/to/out
        """
    )
    parser.add_argument("base_dir",
                        help="Directory containing RUN1, RUN2, ... subdirectories")
    parser.add_argument("--label", default=None,
                        help="Label for plots and CSV (default: folder name)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <base_dir>/ion_ion_permeation_aggregated)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return 1

    label   = args.label or base_dir.name
    out_dir = Path(args.output) if args.output else base_dir / "ion_ion_permeation_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSystem   : {label}")
    print(f"Base dir : {base_dir}")
    print(f"Output   : {out_dir}\n")

    # ── load ──────────────────────────────────────────────────────────────
    rows = load_all_runs(base_dir)
    if not rows:
        print("ERROR: No event data found.")
        return 1
    print(f"\nTotal events loaded: {len(rows)}")

    # ── print ─────────────────────────────────────────────────────────────
    print_summary(rows, label)

    # ── save CSVs ─────────────────────────────────────────────────────────
    save_all_events(rows, out_dir / "all_events.csv")
    save_summary(rows, label, out_dir / "aggregated_summary.csv")

    # ── distance plots ────────────────────────────────────────────────────
    bar_plot(rows, DIST_METRICS, DIST_NAMES, label,
             "Distance (Å)",
             "Ion–ion distances at moment of permeation",
             out_dir / "dist_bar.png", color="#4878CF")
    hist_plot(rows, DIST_METRICS, DIST_NAMES, label,
              "Distribution of ion–ion distances at moment of permeation",
              out_dir / "dist_hist.png", color="#4878CF")
    box_plot(rows, DIST_METRICS, DIST_NAMES, label,
             "Distance (Å)",
             "Ion–ion distances at moment of permeation",
             out_dir / "dist_box.png", color="#4878CF")

    # ── force magnitude plots ─────────────────────────────────────────────
    bar_plot(rows, FMAG_METRICS, FMAG_NAMES, label,
             "Coulomb force magnitude (kcal/(mol·Å))",
             "Total Coulomb force magnitude on permeating ion",
             out_dir / "force_mag_bar.png", color="#C0392B")
    hist_plot(rows, FMAG_METRICS, FMAG_NAMES, label,
              "Distribution of Coulomb force magnitude on permeating ion",
              out_dir / "force_mag_hist.png", color="#C0392B")
    box_plot(rows, FMAG_METRICS, FMAG_NAMES, label,
             "Coulomb force magnitude (kcal/(mol·Å))",
             "Total Coulomb force magnitude on permeating ion",
             out_dir / "force_mag_box.png", color="#C0392B")

    # ── axial force plots ─────────────────────────────────────────────────
    bar_plot(rows, FAX_METRICS, FAX_NAMES, label,
             "Axial Coulomb force (kcal/(mol·Å)) ",
             "Axial ion–ion force on permeating ion at moment of permeation\n"
             "(positive = pushes toward HBC = favors permeation)",
             out_dir / "force_axial_bar.png", color="#E67E22")
    hist_plot(rows, FAX_METRICS, FAX_NAMES, label,
              "Distribution of axial ion–ion force at moment of permeation\n"
              "(positive = pushes toward HBC)",
              out_dir / "force_axial_hist.png", color="#E67E22")
    box_plot(rows, FAX_METRICS, FAX_NAMES, label,
             "Axial Coulomb force (kcal/(mol·Å)) ",
             "Axial ion–ion force on permeating ion at moment of permeation\n"
             "(positive = pushes toward HBC = favors permeation)",
             out_dir / "force_axial_box.png", color="#E67E22")

    print(f"\nDone. All outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())