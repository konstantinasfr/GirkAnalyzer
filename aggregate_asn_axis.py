"""
aggregate_asn_axis.py
=====================
Aggregates ASN axis proximity data from multiple MD runs and produces
pooled plots identical in style to the per-run asn_axis_analysis plots.

Usage
-----
    python aggregate_asn_axis.py --topdir /path/to/runs --outdir /path/to/output

The topdir should contain subdirectories like RUN1, RUN2, RUN3 ... each
containing:
    <topdir>/RUN*/asn_axis_analysis/asn_axis_distances.csv

Output (saved to outdir/):
    - winner_counts.png        bar chart: OD1 vs ND2 closer, per residue
    - delta_histograms.png     Δd = d(OD1) − d(ND2) histograms per residue
    - hist_OD1_ND2.png         overlaid OD1 / ND2 distance histograms
    - summary.txt              text summary with counts and percentages
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Panel layout ──────────────────────────────────────────────────────────────
# 184.B  184.C
# 184.A  173.G1
PANEL_ORDER = ["184.B", "184.C", "184.A", "173.G1"]

TITLE_SIZE  = 26
LABEL_SIZE  = 22
TICK_SIZE   = 19
LEGEND_SIZE = 18


# =============================================================================
# Data loading
# =============================================================================

def load_all_runs(topdir: Path) -> pd.DataFrame:
    """
    Walk topdir looking for RUN*/asn_axis_analysis/asn_axis_distances.csv
    and concatenate all found files into one DataFrame.
    """
    topdir = Path(topdir)
    csv_files = sorted(topdir.glob("*/asn_axis_analysis/asn_axis_distances.csv"))

    if not csv_files:
        # Also try one level deeper (e.g. channel_type/RUN*/...)
        csv_files = sorted(topdir.glob("*/*/asn_axis_analysis/asn_axis_distances.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No asn_axis_distances.csv files found under {topdir}\n"
            f"Expected pattern: <topdir>/RUN*/asn_axis_analysis/asn_axis_distances.csv"
        )

    print(f"\n{'='*60}")
    print(f"LOADING DATA FROM {len(csv_files)} RUNS")
    print(f"{'='*60}")

    dfs = []
    for csv_path in csv_files:
        run_name = csv_path.parts[-3]   # e.g. "RUN1"
        df = pd.read_csv(csv_path)
        df["run"] = run_name
        dfs.append(df)
        print(f"  {run_name:10s}: {len(df):>8,} rows  ({csv_path})")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows: {len(combined):,}")
    print(f"  PDB labels: {sorted(combined['pdb_label'].unique())}")
    return combined


# =============================================================================
# Helpers
# =============================================================================

def _sorted_labels(df: pd.DataFrame) -> list:
    """Return PDB labels present in df, sorted by PANEL_ORDER."""
    present = df["pdb_label"].unique()
    ordered = [lbl for lbl in PANEL_ORDER if lbl in present]
    # Append any unknown labels at the end
    ordered += [lbl for lbl in present if lbl not in PANEL_ORDER]
    return ordered


def _make_grid(n: int, ncols: int = 2):
    """Create a tight grid of n axes."""
    import math
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5.5 * nrows),
                             squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    return fig, ax_flat[:n]


# =============================================================================
# Plot 1 — Winner counts bar chart
# =============================================================================

def plot_winner_counts(df: pd.DataFrame, outdir: Path) -> None:
    labels = _sorted_labels(df)
    fig, axes = _make_grid(len(labels))

    for ax, lbl in zip(axes, labels):
        sub   = df[df["pdb_label"] == lbl]
        total = len(sub)

        od1_count = (sub["closer_atom"] == "OD1").sum()
        nd2_count = (sub["closer_atom"] == "ND2").sum()

        bars = ax.bar(
            ["OD1 closer\n(O toward axis)", "ND2 closer\n(N toward axis)"],
            [od1_count, nd2_count],
            color=["#e74c3c", "#3498db"],
            alpha=0.85, edgecolor="black", linewidth=1.5,
        )
        for bar, count in zip(bars, [od1_count, nd2_count]):
            pct = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold",
            )

        ax.set_title(f"ASN {lbl}",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_ylabel("Number of frames", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        # Add 25% headroom above the tallest bar so labels don't overlap title
        ax.set_ylim(0, max(od1_count, nd2_count) * 1.25)

    fig.suptitle("ASN Amide Orientation Relative to Pore Axis  (all runs pooled)",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = outdir / "winner_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")



# =============================================================================
# Plot 2 — Delta histograms  (Δd = d_OD1 − d_ND2)
# =============================================================================

def plot_delta_histograms(df: pd.DataFrame, outdir: Path) -> None:
    labels = _sorted_labels(df)
    fig, axes = _make_grid(len(labels))

    for ax, lbl in zip(axes, labels):
        sub = df[df["pdb_label"] == lbl]

        ax.hist(sub["delta"], bins=80, color="#8e44ad", alpha=0.75, density=True)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="0 (equal)")
        ax.axvline(sub["delta"].mean(), color="red", linestyle="--",
                   linewidth=2, label=f"mean = {sub['delta'].mean():.2f} Å")

        ax.axvspan(sub["delta"].min(), 0,
                   alpha=0.07, color="#e74c3c", label="OD1 closer (O→axis)")
        ax.axvspan(0, sub["delta"].max(),
                   alpha=0.07, color="#3498db", label="ND2 closer (N→axis)")

        ax.set_title(f"ASN {lbl}  —  Δd = d(OD1) − d(ND2)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Δd  (Å)     [ negative = OD1 faces axis ]",
                      fontsize=LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE - 1)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("ASN Amide Asymmetry  Δd = d_OD1 − d_ND2  (all runs pooled)",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = outdir / "delta_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot 3 — Overlaid OD1 / ND2 distance histograms
# =============================================================================

def plot_distance_histograms(df: pd.DataFrame, outdir: Path) -> None:
    labels = _sorted_labels(df)
    fig, axes = _make_grid(len(labels))

    for ax, lbl in zip(axes, labels):
        sub = df[df["pdb_label"] == lbl]

        ax.hist(sub["d_OD1"], bins=80, alpha=0.6, color="#e74c3c",
                label="OD1", density=True)
        ax.hist(sub["d_ND2"], bins=80, alpha=0.6, color="#3498db",
                label="ND2", density=True)

        ax.axvline(sub["d_OD1"].mean(), color="#e74c3c", linestyle="--",
                   linewidth=2,
                   label=f"OD1 mean = {sub['d_OD1'].mean():.2f} Å")
        ax.axvline(sub["d_ND2"].mean(), color="#3498db", linestyle="--",
                   linewidth=2,
                   label=f"ND2 mean = {sub['d_ND2'].mean():.2f} Å")

        ax.set_title(f"ASN {lbl}",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Perpendicular distance to axis  (Å)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("OD1 vs ND2 Distance to Pore Axis  (all runs pooled)",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = outdir / "hist_OD1_ND2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Summary text
# =============================================================================

def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    labels  = _sorted_labels(df)
    runs    = sorted(df["run"].unique())
    n_total = len(df)

    lines = [
        "ASN AXIS PROXIMITY ANALYSIS — AGGREGATED SUMMARY",
        "=" * 65,
        f"Runs included : {', '.join(runs)}",
        f"Total frames  : {n_total:,}",
        "",
        "Δd = d(OD1) − d(ND2)",
        "  Δd < 0 → OD1 is closer to axis (carbonyl O faces pore)",
        "  Δd > 0 → ND2 is closer to axis (amide N/HD faces pore)",
        "",
        "=" * 65,
        "",
    ]

    for lbl in labels:
        sub   = df[df["pdb_label"] == lbl]
        total = len(sub)
        od1_c = (sub["closer_atom"] == "OD1").sum()
        nd2_c = (sub["closer_atom"] == "ND2").sum()

        lines += [
            f"ASN {lbl}",
            f"  Frames            : {total:,}",
            f"  OD1 mean dist     : {sub['d_OD1'].mean():.3f} ± {sub['d_OD1'].std():.3f} Å",
            f"  ND2 mean dist     : {sub['d_ND2'].mean():.3f} ± {sub['d_ND2'].std():.3f} Å",
            f"  Mean Δd           : {sub['delta'].mean():.3f} ± {sub['delta'].std():.3f} Å",
            f"  OD1 closer        : {od1_c:>8,} frames ({100*od1_c/total:.1f}%)",
            f"  ND2 closer        : {nd2_c:>8,} frames ({100*nd2_c/total:.1f}%)",
            f"  Dominant          : {'OD1 faces pore (O inward)' if od1_c > nd2_c else 'ND2 faces pore (N/HD inward)'}",
            "",
        ]

    path = outdir / "summary.txt"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ASN axis analysis from multiple MD runs."
    )
    parser.add_argument(
        "--topdir", required=True, type=Path,
        help="Top-level directory containing RUN1, RUN2, ... subdirectories.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory (default: <topdir>/asn_axis_aggregated).",
    )
    args = parser.parse_args()

    outdir = args.outdir or (args.topdir / "asn_axis_aggregated")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ASN AXIS AGGREGATED ANALYSIS")
    print(f"{'='*60}")
    print(f"  Top dir : {args.topdir}")
    print(f"  Out dir : {outdir}")

    # Load all runs
    df = load_all_runs(args.topdir)

    # Generate plots
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    plot_winner_counts(df, outdir)
    plot_delta_histograms(df, outdir)
    plot_distance_histograms(df, outdir)
    write_summary(df, outdir)

    print(f"\n{'='*60}")
    print(f"✓  Done.  Results in: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# python3 aggregate_asn_axis.py \
#     --topdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
#     --outdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/asn_axis_aggregated

# python3 aggregate_asn_axis.py \
#     --topdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
#     --outdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/asn_axis_aggregated

# python3 aggregate_asn_axis.py \
#     --topdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
#     --outdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/asn_axis_aggregated

# python3 aggregate_asn_axis.py \
#     --topdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
#     --outdir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/asn_axis_aggregated