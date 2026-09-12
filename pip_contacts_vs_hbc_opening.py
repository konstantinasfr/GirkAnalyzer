"""
pip_contacts_vs_hbc_opening.py

Standalone script. For every PIP and every phosphate (P1, P4, P5),
reads the combined CSVs across multiple threshold folders and plots
how contact_pct changes as threshold increases.

Usage:
    python3 pip_contacts_vs_hbc_opening.py \
        --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_hbc_open_pip \
        --thresholds 5 6 7 8 9 10 11 12 13

Output:
    base_dir/pip_contacts_vs_hbc_opening/
        PIP_1301_2b/
            P1_sweep.png
            P1_sweep.csv
            P4_sweep.png
            ...
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


PHOSPHATE_COLORS_MAIN = {
    "P1": "#1a7a2e",
    "P4": "#b85c00",
    "P5": "#1a3a7a",
}


def load_sweep_data(base_dir: str, thresholds: list, pip_name: str, phosphate: str):
    """
    For a given PIP and phosphate, load contact_pct for each threshold.
    Returns a DataFrame: rows=residues, columns=thresholds.
    Missing residues at a threshold get 0.
    """
    data = {}  # threshold -> {residue: pct}

    for thr in thresholds:
        csv_path = os.path.join(base_dir, str(thr), pip_name, f"{phosphate}_combined.csv")
        if not os.path.isfile(csv_path):
            # try int threshold folder name
            csv_path = os.path.join(base_dir, str(int(thr)), pip_name, f"{phosphate}_combined.csv")
        if not os.path.isfile(csv_path):
            print(f"  [sweep] Missing: {csv_path}")
            data[thr] = {}
            continue

        df = pd.read_csv(csv_path, index_col="residue")
        data[thr] = df["contact_pct"].to_dict()

    # collect all residues that appear at least once
    all_residues = set()
    for d in data.values():
        all_residues.update(d.keys())

    if not all_residues:
        return None

    # build wide DataFrame: residues x thresholds
    records = {}
    for residue in all_residues:
        records[residue] = [data[thr].get(residue, 0.0) for thr in thresholds]

    df_wide = pd.DataFrame(records, index=thresholds).T
    df_wide.index.name = "residue"
    df_wide.columns = [str(t) for t in thresholds]

    # sort by mean pct descending
    df_wide["_mean"] = df_wide.mean(axis=1)
    df_wide = df_wide.sort_values("_mean", ascending=False).drop(columns="_mean")

    return df_wide


def plot_sweep(df_wide: pd.DataFrame, thresholds: list, title: str,
               out_path: str, top_n: int = 15):
    """Line plot: x=threshold, y=contact_pct, one line per residue."""
    # only plot top_n residues by mean pct
    top = df_wide.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 4))

    # use a colormap for distinct line colors
    colors = cm.tab20(np.linspace(0, 1, len(top)))

    x = [str(int(t)) if t == int(t) else str(t) for t in thresholds]

    for (residue, row), color in zip(top.iterrows(), colors):
        ax.plot(x, row.values, marker="o", linewidth=2.5,
                markersize=7, label=residue, color=color)

    ax.set_xlabel("HBC threshold (Å)", fontsize=18)
    ax.set_ylabel("Contact % of open frames", fontsize=18)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0, frameon=True)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.savefig(out_path, dpi=60, bbox_inches="tight")
    plt.close()


def run(base_dir: str, thresholds: list, top_n: int = 15):
    out_dir = os.path.join(base_dir, "pip_contacts_vs_hbc_opening")
    os.makedirs(out_dir, exist_ok=True)

    # find all PIP names from first available threshold
    pip_names = set()
    for thr in thresholds:
        thr_dir = os.path.join(base_dir, str(thr))
        if not os.path.isdir(thr_dir):
            thr_dir = os.path.join(base_dir, str(int(thr)))
        for pip_folder in glob.glob(os.path.join(thr_dir, "PIP_*")):
            pip_names.add(os.path.basename(pip_folder))

    if not pip_names:
        print(f"No PIP_* folders found under any threshold in {base_dir}")
        return

    pip_names = sorted(pip_names)
    print(f"PIPs: {pip_names}")
    print(f"Thresholds: {thresholds}")

    for pip_name in pip_names:
        pip_out = os.path.join(out_dir, pip_name)
        os.makedirs(pip_out, exist_ok=True)

        for phosphate in ["P1", "P4", "P5"]:
            print(f"  {pip_name} / {phosphate} ...")

            df_wide = load_sweep_data(base_dir, thresholds, pip_name, phosphate)
            if df_wide is None:
                print(f"    no data, skipping.")
                continue

            # save wide csv
            csv_out = os.path.join(pip_out, f"{phosphate}_sweep.csv")
            df_wide.to_csv(csv_out)

            title = f"{pip_name} | {phosphate} — contact % vs HBC threshold"
            plot_sweep(
                df_wide, thresholds,
                title=title,
                out_path=os.path.join(pip_out, f"{phosphate}_sweep.png"),
                top_n=top_n,
            )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Path to combined_hbc_open_pip folder")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[5, 6, 7, 8, 9, 10, 11, 12, 13],
                        help="Threshold values to sweep")
    parser.add_argument("--top_n", type=int, default=15,
                        help="Max residues to show per plot")
    args = parser.parse_args()
    run(args.base_dir, args.thresholds, args.top_n)