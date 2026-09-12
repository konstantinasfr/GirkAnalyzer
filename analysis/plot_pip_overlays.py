"""
plot_pip_overlays.py
====================
Standalone script to make KDE overlay plots from already-computed
per_frame_all_runs.csv files.

Plots Nterm vs Cterm vs LYS90 KDE density for each P atom per subunit.

Usage
-----
    python plot_pip_overlays.py \
        --combined_dir /path/to/combined_pip_headgroup_distances \
        --channel_type G12
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]
P_ATOMS       = ["P1", "P4", "P5"]

CLUSTER_COLORS = {
    "Nterm":  "#1565C0",
    "Cterm":  "#B71C1C",
    "LYS90":  "#FF6F00",
}
LINESTYLES = {"Nterm": "-", "Cterm": "-", "LYS90": "-"}


def make_overlays(combined_dir: Path, channel_type: str, subset_label: str = "all frames"):
    for sub in SUBUNIT_ORDER:
        csv = combined_dir / f"subunit_{sub}" / "per_frame_all_runs.csv"
        if not csv.exists():
            print(f"  Missing: {csv}")
            continue

        df = pd.read_csv(csv)
        n_runs = df["run"].nunique() if "run" in df.columns else "?"
        print(f"\n  Subunit {sub} ({n_runs} runs)...")

        overlay_dir = combined_dir / f"subunit_{sub}" / "gaussians" / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for p_atom in P_ATOMS:
            fig, ax = plt.subplots(figsize=(6, 4))
            plotted = False

            for cluster in ["Nterm", "Cterm", "LYS90"]:
                col = f"{p_atom}_{cluster}"
                if col not in df.columns:
                    continue
                vals = df[col].dropna().values
                if len(vals) == 0:
                    continue

                # subsample for speed
                if len(vals) > 50000:
                    vals = np.random.default_rng(42).choice(vals, 50000, replace=False)

                x_range  = np.linspace(vals.min(), vals.max(), 500)
                kde_vals = gaussian_kde(vals, bw_method=0.18)(x_range)

                ax.fill_between(x_range, kde_vals,
                                color=CLUSTER_COLORS[cluster], alpha=0.1)
                ax.plot(x_range, kde_vals,
                        color=CLUSTER_COLORS[cluster],
                        linewidth=2,
                        linestyle=LINESTYLES[cluster],
                        label=cluster)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("Distance (Å)", fontsize=15)
            ax.set_ylabel("Density", fontsize=15)
            ax.set_title(f"{p_atom} | Nterm vs Cterm vs LYS90\n"
                         f"{channel_type} | Subunit {sub} [{n_runs} runs] | {subset_label}",
                         fontsize=14)
            ax.legend(fontsize=13)
            ax.tick_params(labelsize=14)
            plt.tight_layout()
            fname = f"overlay_{p_atom}_Nterm_vs_Cterm_vs_LYS90.png"
            fig.savefig(overlay_dir / fname, dpi=300)
            plt.close(fig)
            print(f"    {p_atom} -> {fname}")


GATE_SUBSETS = [
    "asn_gate_frames",
    "asn_gate_m1_frames",
    "hbc_gate_frames",
    "hbc_gate_m1_frames",
]


def make_all_overlays(combined_dir: Path, channel_type: str):
    # all frames
    print("\n=== All frames ===")
    make_overlays(combined_dir, channel_type, subset_label="all frames")

    # gate subsets
    gate_labels = {
        "asn_gate_frames":    "ASN gate",
        "asn_gate_m1_frames": "ASN gate -1",
        "hbc_gate_frames":    "HBC gate",
        "hbc_gate_m1_frames": "HBC gate -1",
    }
    for subset in GATE_SUBSETS:
        subset_dir = combined_dir / subset
        if not subset_dir.exists():
            print(f"  Skipping {subset} -- not found")
            continue
        print(f"\n=== {subset} ===")
        make_overlays(subset_dir, channel_type, subset_label=gate_labels[subset])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_dir", required=True)
    parser.add_argument("--channel_type", required=True)
    args = parser.parse_args()

    make_all_overlays(Path(args.combined_dir), args.channel_type)
    print("\n✓ Done")


"""
python3 plot_pip_overlays.py \
    --combined_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_headgroup_distances \
    --channel_type G12

python3 plot_pip_overlays.py \
    --combined_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_headgroup_distances \
    --channel_type G12_ML

python3 plot_pip_overlays.py \
    --combined_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_headgroup_distances \
    --channel_type G12_GAT
"""