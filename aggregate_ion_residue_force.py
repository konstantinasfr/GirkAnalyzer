"""
aggregate_ion_residue_force.py
================================
Reads per_event.csv produced by ion_residue_force_analysis.py from every
RUN* subfolder and produces pooled statistics and plots across all runs.

Expects this folder structure:
    <base_dir>/
        RUN1/ion_residue_force/per_event.csv
        RUN2/ion_residue_force/per_event.csv
        ...

Outputs (saved to <base_dir>/ion_residue_force_aggregated/):
    all_events.csv              all per-event rows from every run (with run label)
    aggregated_summary.csv      pooled mean ± std / min / max across all events
    force_mag_bar.png           bar chart of force magnitudes
    force_mag_hist.png          histograms + KDE for force magnitudes
    force_axial_bar.png         bar chart of axial forces
    force_axial_hist.png        histograms + KDE for axial forces

Ligand-related bars (Ligand only, GLU+ASN+ligand+ions) are added
automatically only when at least one pooled event has lig_present=True
(i.e. only for channels like G12_ML / G12_GAT). Channels with no ligand
(G2, G12) get the same plots as before, unaffected.

Usage:
    python3 aggregate_ion_residue_force.py /path/to/G12
    python3 aggregate_ion_residue_force.py /path/to/G12 --label G12
    python3 aggregate_ion_residue_force.py /path/to/G12 --output /path/to/out
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── column definitions ────────────────────────────────────────────────────────

FMAG_METRICS_BASE = ["force_asn_mag", "force_glu_mag",
                      "force_glu_asn_mag", "force_ions_mag"]
FMAG_NAMES_BASE   = ["ASN\nonly", "GLU\nonly",
                      "GLU + ASN", "All ions"]

FAX_METRICS_BASE  = ["force_asn_axial", "force_glu_axial",
                      "force_glu_asn_axial", "force_ions_axial",
                      "force_glu_asn_ions_axial"]
FAX_NAMES_BASE    = ["ASN\nonly", "GLU\nonly",
                      "GLU + ASN", "All ions",
                      "GLU + ASN\n+ ions"]

# extra columns/names inserted only when ligand events are present
LIG_MAG_METRIC, LIG_MAG_NAME = "force_lig_mag", "Ligand\nonly"
LIG_AX_METRIC,  LIG_AX_NAME  = "force_lig_axial", "Ligand\nonly"
TOTAL_AX_METRIC, TOTAL_AX_NAME = "force_total_axial", "GLU + ASN\n+ ligand + ions"

# font sizes
FS_TITLE  = 22
FS_LABEL  = 20
FS_TICK   = 18
FS_ANNOT  = 16
FS_LEGEND = 16


# ── loading ───────────────────────────────────────────────────────────────────

def load_all_runs(base_dir: Path):
    all_rows = []
    run_dirs = sorted(base_dir.glob("RUN*"))

    if not run_dirs:
        raise FileNotFoundError(f"No RUN* directories found in {base_dir}")

    for run_dir in run_dirs:
        csv_path = run_dir / "ion_residue_force" / "per_event.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {run_dir.name} — no per_event.csv")
            continue

        run_rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                converted = {}
                for k, v in row.items():
                    if k == "lig_present":
                        converted[k] = str(v).strip().lower() in ("true", "1")
                        continue
                    try:
                        converted[k] = float(v)
                    except (ValueError, TypeError):
                        converted[k] = v if v != "" else None
                converted["run"] = run_dir.name
                run_rows.append(converted)

        all_rows.extend(run_rows)
        print(f"  {run_dir.name}: {len(run_rows)} events")

    return all_rows


def has_ligand_events(rows):
    return any(r.get("lig_present") is True for r in rows)


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


def save_summary(rows, label, out_path, numeric_cols):
    summary = [
        {"metric": "system",   "mean": label,    "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "n_events", "mean": len(rows), "std": "", "min": "", "max": "", "n": len(rows)},
    ]
    for col in numeric_cols:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        summary.append({
            "metric": col,
            "mean":   f"{np.mean(vals):.4f}",
            "std":    f"{np.std(vals):.4f}",
            "min":    f"{np.min(vals):.4f}",
            "max":    f"{np.max(vals):.4f}",
            "n":      len(vals),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"  Saved: {out_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _vals(rows, col):
    return [r[col] for r in rows if r.get(col) is not None]


# ── plots ─────────────────────────────────────────────────────────────────────

def bar_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    """Bar chart: mean ± std with jittered individual event dots."""
    means, stds, all_vals = [], [], []
    for m in metrics:
        v = _vals(rows, m)
        means.append(np.mean(v) if v else 0)
        stds.append(np.std(v)   if v else 0)
        all_vals.append(v)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 7))
    x   = np.arange(len(names))
    rng = np.random.default_rng(42)

    ax.bar(x, means, yerr=stds, capsize=10, width=0.60,
           color=color, alpha=0.75, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=2.0, ecolor="black"), zorder=2)

    for i, v in enumerate(all_vals):
        jitter = rng.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(i + jitter, v, color=color, alpha=0.25, s=28,
                   edgecolors="none", zorder=3)

    data_top = max((max(v) for v in all_vals if v), default=1)
    data_bot = min((min(v) for v in all_vals if v), default=0)
    data_top = max(data_top, max(m + s for m, s in zip(means, stds)))
    data_bot = min(data_bot, min(m - s for m, s in zip(means, stds)))

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=FS_ANNOT)

    pad = (data_top - data_bot) * 0.40 if data_top != data_bot else 1
    ax.set_ylim(bottom=data_bot - pad * 0.1, top=data_top + pad)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} events)",
                 fontsize=FS_TITLE, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def hist_plot(rows, metrics, names, label, suptitle, out_path, color):
    """Histograms + KDE, one panel per metric."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m, name in zip(axes, metrics, names):
        v = np.array(_vals(rows, m))
        if len(v) == 0:
            ax.set_title(name, fontsize=FS_LABEL)
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
                   linestyle="--", label=f"mean = {np.mean(v):.3f}")
        ax.axvline(0, color="gray", linewidth=1.0, linestyle=":", alpha=0.6)
        ax.set_xlabel(name, fontsize=FS_LABEL)
        ax.set_ylabel("Density", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{suptitle}\n{label}  (n = {len(rows)} events)",
                 fontsize=FS_TITLE, y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def box_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    """Boxplot for each metric. X-axis labels show name + mean + median."""
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

    tick_labels = []
    for name, v in zip(names, all_vals):
        if v:
            tick_labels.append(
                f"{name}\nmean={np.mean(v):.3f}\nmedian={np.median(v):.3f}"
            )
        else:
            tick_labels.append(name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK - 2)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} events)",
                 fontsize=FS_TITLE, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── terminal summary ──────────────────────────────────────────────────────────


def print_summary(rows, label, fax_metrics, fax_names, fmag_metrics, fmag_names):
    print()
    print(f"  {'='*60}")
    print(f"  AGGREGATED SUMMARY — {label}  (n = {len(rows)} events)")
    print(f"  {'='*60}")
    print("  AXIAL FORCES  kcal/(mol·Å)   +ve = toward HBC:")
    for col, name in zip(fax_metrics, fax_names):
        v = _vals(rows, col)
        if v:
            name_clean = name.replace("\n", " ")
            print(f"    {name_clean:<28}  {np.mean(v):+7.3f} ± {np.std(v):.3f}")
    print("  FORCE MAGNITUDES  kcal/(mol·Å):")
    for col, name in zip(fmag_metrics, fmag_names):
        v = _vals(rows, col)
        if v:
            name_clean = name.replace("\n", " ")
            print(f"    {name_clean:<28}  {np.mean(v):7.3f} ± {np.std(v):.3f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ion_residue_force_analysis results across all RUN folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 aggregate_ion_residue_force.py /path/to/G12
  python3 aggregate_ion_residue_force.py /path/to/G12_ML --label G12_ML
  python3 aggregate_ion_residue_force.py /path/to/G12 --output /path/to/out
        """
    )
    parser.add_argument("base_dir",
                        help="Directory containing RUN1, RUN2, ... subdirectories")
    parser.add_argument("--label", default=None,
                        help="Label for plots and CSV (default: folder name)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <base_dir>/ion_residue_force_aggregated)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return 1

    label   = args.label or base_dir.name
    out_dir = Path(args.output) if args.output else \
              base_dir / "ion_residue_force_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSystem   : {label}")
    print(f"Base dir : {base_dir}")
    print(f"Output   : {out_dir}\n")

    # load
    rows = load_all_runs(base_dir)
    if not rows:
        print("ERROR: No event data found.")
        return 1
    print(f"\nTotal events loaded: {len(rows)}")

    # detect ligand presence in the pooled data (e.g. G12_ML, G12_GAT)
    has_lig = has_ligand_events(rows)
    print(f"Ligand events present: {has_lig}")

    # build metric/name lists, adding ligand bars only when relevant
    fmag_metrics = list(FMAG_METRICS_BASE)
    fmag_names   = list(FMAG_NAMES_BASE)
    if has_lig:
        fmag_metrics.insert(2, LIG_MAG_METRIC)   # after GLU only, before GLU+ASN
        fmag_names.insert(2, LIG_MAG_NAME)

    fax_metrics = list(FAX_METRICS_BASE)
    fax_names   = list(FAX_NAMES_BASE)
    if has_lig:
        fax_metrics.insert(2, LIG_AX_METRIC)     # after GLU only
        fax_names.insert(2, LIG_AX_NAME)
        fax_metrics.append(TOTAL_AX_METRIC)      # extra bar at the end: + ligand
        fax_names.append(TOTAL_AX_NAME)

    numeric_cols = sorted(set(fmag_metrics + fax_metrics))

    # print summary
    print_summary(rows, label, fax_metrics, fax_names, fmag_metrics, fmag_names)

    # save CSVs
    save_all_events(rows, out_dir / "all_events.csv")
    save_summary(rows, label, out_dir / "aggregated_summary.csv", numeric_cols)

    # force magnitude plots
    bar_plot(rows, fmag_metrics, fmag_names, label,
             "Force magnitude  (kcal/(mol·Å))",
             "Coulomb force magnitude on permeating ion at moment of permeation",
             out_dir / "force_mag_bar.png", color="#8E44AD")
    hist_plot(rows, fmag_metrics, fmag_names, label,
              "Distribution of Coulomb force magnitude on permeating ion",
              out_dir / "force_mag_hist.png", color="#8E44AD")
    box_plot(rows, fmag_metrics, fmag_names, label,
             "Force magnitude  (kcal/(mol·Å))",
             "Coulomb force magnitude on permeating ion at moment of permeation",
             out_dir / "force_mag_box.png", color="#8E44AD")

    # axial force plots
    bar_plot(rows, fax_metrics, fax_names, label,
             "Axial force  (kcal/(mol·Å))",
             "Axial Coulomb force on permeating ion\n"
             "(positive = pushes toward HBC = favors permeation)",
             out_dir / "force_axial_bar.png", color="#E67E22")
    hist_plot(rows, fax_metrics, fax_names, label,
              "Distribution of axial force on permeating ion\n"
              "(positive = toward HBC, negative = back toward SF)",
              out_dir / "force_axial_hist.png", color="#E67E22")
    box_plot(rows, fax_metrics, fax_names, label,
             "Axial force  (kcal/(mol·Å))",
             "Axial Coulomb force on permeating ion\n"
             "(positive = pushes toward HBC = favors permeation)",
             out_dir / "force_axial_box.png", color="#E67E22")

    print(f"\nDone. All outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

"""
    python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ion_residue_force \
--label G2 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ion_residue_force \
--label G12 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ion_residue_force \
--label G12_ML 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ion_residue_force \
--label G12_GAT 
"""


"""
    python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ion_residue_force \
--label G2 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ion_residue_force \
--label G12 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ion_residue_force \
--label G12_ML 

python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ion_residue_force \
--label G12_GAT 

    python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ion_residue_force \
--label G12_D173N 


    python3 aggregate_ion_residue_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ion_residue_force \
--label G12_E141Q 
"""