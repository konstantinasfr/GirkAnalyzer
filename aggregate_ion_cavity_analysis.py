"""
aggregate_ion_cavity_analysis.py
=================================
Reads the per-event CSV and axial traces CSV produced by ion_cavity_analysis.py
from every RUN* subfolder and produces pooled statistics and plots across all runs.

Expects this folder structure:
    <base_dir>/
        RUN1/ion_cavity_analysis/contacts_per_event.csv
        RUN1/ion_cavity_analysis/axial_traces.csv
        RUN1/ion_cavity_analysis/contacts_summary.csv   (for landmark positions)
        RUN2/ion_cavity_analysis/...
        ...

Outputs (saved to <base_dir>/ion_cavity_analysis_aggregated/):
    all_events.csv          all per-event rows from every run, with run label
    aggregated_summary.csv  pooled mean +/- std / min / max across all events
    axial_traces.png        normalised trajectory plot (all events pooled)
    axial_pmf.png           PMF along channel axis (all events pooled)
    zones_bar_ns.png        compartment time in ns
    zones_bar_pct.png       compartment time in %
    closest_bar_ns.png      closest-group time in ns
    closest_bar_pct.png     closest-group time in %

Usage:
    python3 aggregate_ion_cavity_analysis.py /path/to/G12
    python3 aggregate_ion_cavity_analysis.py /path/to/G12 --output /path/to/out
    python3 aggregate_ion_cavity_analysis.py /path/to/G12 --label G12
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ZONE_NAMES  = ["GLU/ASN cavity", "ASN/HBC cavity"]
ZONE_KEYS   = ["glu_asn",        "asn_hbc"]
ZONE_COLORS = ["#E67E22",        "#2980B9"]

CLOSE_NAMES  = ["Closest to GLU", "Closest to ASN", "Closest to HBC"]
CLOSE_KEYS   = ["close_glu",      "close_asn",      "close_hbc"]
CLOSE_COLORS = ["#C0392B",        "#E67E22",         "#2C3E50"]

kB   = 0.001987
TEMP = 310.0
kT   = kB * TEMP


# ── loading ───────────────────────────────────────────────────────────────────

def load_all_runs(base_dir: Path):
    """
    Walk every RUN* subfolder, read:
      - contacts_per_event.csv  → contact rows
      - axial_traces.csv        → per-frame d values grouped by event
      - contacts_summary.csv    → landmark positions (glu_d_A, asn_d_A, hbc_d_A)
    """
    all_contact_rows = []
    all_traces       = []   # list of {"d": np.array, "run": str}
    all_d_flat       = []   # flat list of all d values for PMF
    glu_d_vals, asn_d_vals, hbc_d_vals = [], [], []

    run_dirs = sorted(base_dir.glob("RUN*"))
    if not run_dirs:
        raise FileNotFoundError(f"No RUN* directories found in {base_dir}")

    for run_dir in run_dirs:
        ana_dir = run_dir / "ion_cavity_analysis"

        # ── contacts per event ────────────────────────────────────────────
        contacts_csv = ana_dir / "contacts_per_event.csv"
        if not contacts_csv.exists():
            print(f"  [SKIP] {run_dir.name} — no contacts_per_event.csv")
            continue

        run_contact_rows = []
        with open(contacts_csv, newline="") as f:
            for row in csv.DictReader(f):
                converted = {}
                for k, v in row.items():
                    try:
                        converted[k] = float(v)
                    except (ValueError, TypeError):
                        converted[k] = v
                converted["run"] = run_dir.name
                run_contact_rows.append(converted)
        all_contact_rows.extend(run_contact_rows)

        # ── axial traces ──────────────────────────────────────────────────
        traces_csv = ana_dir / "axial_traces.csv"
        if traces_csv.exists():
            # group d_A values by event_idx
            event_d = defaultdict(list)
            with open(traces_csv, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        event_d[row["event_idx"]].append(float(row["d_A"]))
                    except (ValueError, KeyError):
                        continue
            for d_list in event_d.values():
                if len(d_list) >= 2:
                    all_traces.append({"d": np.array(d_list), "run": run_dir.name})
                    all_d_flat.extend(d_list)

        # ── landmark positions from summary ───────────────────────────────
        summary_csv = ana_dir / "contacts_summary.csv"
        if summary_csv.exists():
            with open(summary_csv, newline="") as f:
                for row in csv.DictReader(f):
                    m = row.get("metric", "")
                    try:
                        val = float(row.get("mean", ""))
                    except (ValueError, TypeError):
                        continue
                    if m == "glu_d_A":   glu_d_vals.append(val)
                    elif m == "asn_d_A": asn_d_vals.append(val)
                    elif m == "hbc_d_A": hbc_d_vals.append(val)

        print(f"  {run_dir.name}: {len(run_contact_rows)} events, "
              f"{sum(1 for t in all_traces if t['run'] == run_dir.name)} traces")

    mean_glu_d = float(np.mean(glu_d_vals)) if glu_d_vals else 0.0
    mean_asn_d = float(np.mean(asn_d_vals)) if asn_d_vals else 0.0
    mean_hbc_d = float(np.mean(hbc_d_vals)) if hbc_d_vals else 0.0

    return all_contact_rows, all_traces, np.array(all_d_flat), mean_glu_d, mean_asn_d, mean_hbc_d


# ── saving ────────────────────────────────────────────────────────────────────

def save_all_events(rows, out_path):
    if not rows:
        return
    fields = list(rows[0].keys())
    # ensure run is first
    if "run" in fields:
        fields = ["run"] + [f for f in fields if f != "run"]
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
    for key in ZONE_KEYS + CLOSE_KEYS:
        for suffix in ("ns", "pct"):
            col = f"{key}_{suffix}"
            vals = [r[col] for r in rows if col in r]
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

def plot_traces(traces, label, out_path, glu_d, asn_d, hbc_d):
    """Normalised trajectory plot — SF at top, HBC at bottom."""
    if not traces:
        print("  No traces to plot.")
        return
    N = 200
    grid = np.linspace(0, 1, N)
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = []
    for tr in traces:
        d_ev = tr["d"]
        prog = np.linspace(0, 1, len(d_ev))
        ax.plot(prog, d_ev, color="#4878CF", alpha=0.25, linewidth=0.8, zorder=1)
        mat.append(np.interp(grid, prog, d_ev))
    mat    = np.array(mat)
    mean_d = np.mean(mat, axis=0)
    std_d  = np.std(mat,  axis=0)
    ax.plot(grid, mean_d, color="#C0392B", linewidth=2.4, label="Mean", zorder=3)
    ax.fill_between(grid, mean_d - std_d, mean_d + std_d,
                    color="#C0392B", alpha=0.15, label="+/-1 std", zorder=2)
    ax.axhline(0,     color="#666666", linewidth=1.2, linestyle=":", alpha=0.9, label="SF (0 Å)")
    ax.axhline(glu_d, color="#C0392B", linewidth=1.2, linestyle=":", alpha=0.9, label=f"GLU ({glu_d:.1f} Å)")
    ax.axhline(asn_d, color="#E67E22", linewidth=1.2, linestyle=":", alpha=0.9, label=f"ASN ({asn_d:.1f} Å)")
    ax.axhline(hbc_d, color="#2C3E50", linewidth=1.2, linestyle=":", alpha=0.9, label=f"HBC ({hbc_d:.1f} Å)")
    ax.invert_yaxis()
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%\n(SF exit)", "25%", "50%", "75%", "100%\n(HBC exit)"], fontsize=11)
    ax.set_xlabel("Normalised permeation progress", fontsize=14)
    ax.set_ylabel("Axial position (Å from SF,  ↓ toward HBC)", fontsize=14)
    ax.set_title(f"{label} — Ion axial trajectory  (n = {len(traces)} events)\n"
                 f"SF = 0 Å (top)   HBC = {hbc_d:.1f} Å (bottom)", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pmf(all_d, label, out_path, glu_d, asn_d, hbc_d, n_bins=60):
    """PMF along channel axis, clipped to [0, hbc_d], lightly smoothed."""
    if len(all_d) == 0:
        print("  No d values for PMF.")
        return
    from scipy.ndimage import gaussian_filter1d
    clipped = all_d[(all_d >= 0) & (all_d <= hbc_d)]
    if len(clipped) < 10:
        clipped = all_d
    counts, edges = np.histogram(clipped, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth  = gaussian_filter1d(counts.astype(float), sigma=1.0)
    smooth  = np.where(smooth > 0, smooth, np.nan)
    pmf     = -kT * np.log(smooth)
    pmf    -= np.nanmin(pmf)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, pmf, color="#4878CF", linewidth=2.2, zorder=3)
    ax.fill_between(centers, 0, pmf, alpha=0.15, color="#4878CF", zorder=2)
    ax.axvline(0,     color="#666666", linewidth=1.2, linestyle=":", alpha=0.8, label="SF (d=0)")
    ax.axvline(glu_d, color="#C0392B", linewidth=1.2, linestyle=":", alpha=0.8, label=f"GLU ({glu_d:.1f} Å)")
    ax.axvline(asn_d, color="#E67E22", linewidth=1.2, linestyle=":", alpha=0.8, label=f"ASN ({asn_d:.1f} Å)")
    ax.axvline(hbc_d, color="#2C3E50", linewidth=1.2, linestyle=":", alpha=0.8, label=f"HBC ({hbc_d:.1f} Å)")
    ax.set_xlim(0, hbc_d * 1.02)
    ymax = np.nanmax(pmf[np.isfinite(pmf)]) * 1.05
    ax.set_ylim(bottom=0)
    ax.text(0,     ymax, "SF",  ha="center", va="bottom", fontsize=11, color="#666666")
    ax.text(hbc_d, ymax, "HBC", ha="center", va="bottom", fontsize=11, color="#2C3E50")
    ax.set_xlabel("Axial position (Å from SF,  SF left → HBC right)", fontsize=14)
    ax.set_ylabel("PMF (kcal/mol)", fontsize=14)
    ax.set_title(f"{label} — PMF along channel axis  (n_frames = {len(clipped)})", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def bar_plot(rows, label, out_path, mode, keys, names, colors):
    ylabel = "Mean time (ns)" if mode == "ns" else "Mean time (%)"

    all_vals = []
    means, stds = [], []
    for key in keys:
        col = f"{key}_{mode}"
        vals = [r[col] for r in rows if col in r]
        all_vals.append(vals)
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals)   if vals else 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(names))

    ax.bar(x, means, yerr=stds, capsize=7, width=0.55,
           color=colors, alpha=0.80, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=2)

    # faint jittered dots for every event
    rng = np.random.default_rng(42)
    for i, (vals, col) in enumerate(zip(all_vals, colors)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, alpha=0.25, s=16,
                   edgecolors="none", zorder=3)

    # annotation above error bar
    data_top = max(
        (max(v) for v in all_vals if v),
        default=1
    )
    data_top = max(data_top, max(m + s for m, s in zip(means, stds)))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.2f}\n±{s:.2f}",
                ha="center", va="bottom", fontsize=11)

    ax.set_ylim(bottom=0, top=data_top * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"{label}  (n = {len(rows)} events)", fontsize=14, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── terminal summary ──────────────────────────────────────────────────────────

def print_summary(rows, label):
    print()
    print(f"  {'='*55}")
    print(f"  AGGREGATED SUMMARY — {label}  (n = {len(rows)} events)")
    print(f"  {'='*55}")
    print("  COMPARTMENTS (axial position):")
    for key, name in zip(ZONE_KEYS, ZONE_NAMES):
        ns_v  = [r[f"{key}_ns"]  for r in rows if f"{key}_ns"  in r]
        pct_v = [r[f"{key}_pct"] for r in rows if f"{key}_pct" in r]
        print(f"    {name:<22}  {np.mean(ns_v):6.2f} +/- {np.std(ns_v):.2f} ns  "
              f"({np.mean(pct_v):5.1f} +/- {np.std(pct_v):.1f} %)")
    print("  CLOSEST GROUP (no cutoff):")
    for key, name in zip(CLOSE_KEYS, CLOSE_NAMES):
        ns_v  = [r[f"{key}_ns"]  for r in rows if f"{key}_ns"  in r]
        pct_v = [r[f"{key}_pct"] for r in rows if f"{key}_pct" in r]
        print(f"    {name:<22}  {np.mean(ns_v):6.2f} +/- {np.std(ns_v):.2f} ns  "
              f"({np.mean(pct_v):5.1f} +/- {np.std(pct_v):.1f} %)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ion_cavity_analysis results across all RUN folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 aggregate_ion_cavity_analysis.py /path/to/G12
  python3 aggregate_ion_cavity_analysis.py /path/to/G12_ML --label G12_ML
  python3 aggregate_ion_cavity_analysis.py /path/to/G12 --output /path/to/out
        """
    )
    parser.add_argument("base_dir",
                        help="Directory containing RUN1, RUN2, ... subdirectories")
    parser.add_argument("--label", default=None,
                        help="Label for plots and CSV (default: folder name)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <base_dir>/ion_cavity_analysis_aggregated)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return 1

    label   = args.label or base_dir.name
    out_dir = Path(args.output) if args.output else base_dir / "ion_cavity_analysis_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSystem    : {label}")
    print(f"Base dir  : {base_dir}")
    print(f"Output    : {out_dir}\n")

    # load
    rows, traces, all_d, mean_glu_d, mean_asn_d, mean_hbc_d = load_all_runs(base_dir)
    if not rows:
        print("ERROR: No event data found.")
        return 1
    print(f"\nTotal events loaded : {len(rows)}")
    print(f"Total traces loaded : {len(traces)}")
    print(f"Mean landmarks      : GLU={mean_glu_d:.2f} Å  ASN={mean_asn_d:.2f} Å  HBC={mean_hbc_d:.2f} Å")

    # print
    print_summary(rows, label)

    # save CSVs
    save_all_events(rows, out_dir / "all_events.csv")
    save_summary(rows, label, out_dir / "aggregated_summary.csv")

    # trajectory and PMF plots
    plot_traces(traces, label, out_dir / "axial_traces.png",
                mean_glu_d, mean_asn_d, mean_hbc_d)
    plot_pmf(all_d, label, out_dir / "axial_pmf.png",
             mean_glu_d, mean_asn_d, mean_hbc_d)

    # bar plots
    bar_plot(rows, f"{label} — Cavity compartment time",
             out_dir / "zones_bar_ns.png",  mode="ns",
             keys=ZONE_KEYS, names=ZONE_NAMES, colors=ZONE_COLORS)
    bar_plot(rows, f"{label} — Cavity compartment time",
             out_dir / "zones_bar_pct.png", mode="pct",
             keys=ZONE_KEYS, names=ZONE_NAMES, colors=ZONE_COLORS)
    bar_plot(rows, f"{label} — Time closest to each residue group",
             out_dir / "closest_bar_ns.png",  mode="ns",
             keys=CLOSE_KEYS, names=CLOSE_NAMES, colors=CLOSE_COLORS)
    bar_plot(rows, f"{label} — Time closest to each residue group",
             out_dir / "closest_bar_pct.png", mode="pct",
             keys=CLOSE_KEYS, names=CLOSE_NAMES, colors=CLOSE_COLORS)

    print(f"\nDone. All outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())


"""
python3 aggregate_ion_cavity_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ion_cavity_analysis \
--label G12_GAT 

python3 aggregate_ion_cavity_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ion_cavity_analysis \
--label G12 

python3 aggregate_ion_cavity_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ion_cavity_analysis \
--label G12_ML 

python3 aggregate_ion_cavity_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ion_cavity_analysis \
--label G2 
"""