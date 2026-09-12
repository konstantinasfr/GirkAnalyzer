"""
plot_residue_correlations.py
============================
For a given residue number (e.g. 197), loads correlation_all.csv from
every RUN across all gate folders, filters to matched-subunit rows, and
produces boxplots with individual run dots.

Usage:
    python plot_residue_correlations.py \
        --base_dir /media/ziyue/.../G2 \
        --residue 197 \
        --out_dir ./plots_197
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


GATE_FOLDERS = ["hbc", "upper_gl", "lower_gl"]
ATOMS        = ["P1", "P4", "P5"]
SUBUNIT_ORDER_MAP = {
    "G2":  ["2a", "2b", "2c", "2d"],
    "G12": ["2a", "2b", "2c", "G1"],
}

GATE_LABELS = {
    "hbc":      "HBC gate",
    "upper_gl": "Upper G-loop gate",
    "lower_gl": "Lower G-loop gate",
}

SUBUNIT_COLORS = {
    "2a": "#4C72B0",
    "2b": "#DD8452",
    "2c": "#55A868",
    "2d": "#C44E52",
    "G1": "#9467BD",
}


def extract_subunit_from_pip(pip_label: str) -> str:
    """PIP_1316_2a -> 2a"""
    return pip_label.split("_")[-1]


def extract_subunit_from_residue(residue: str) -> str:
    """GLN197.2a -> 2a"""
    if "." in residue:
        return residue.split(".")[-1]
    return ""


def load_close_residues(base_dir: Path, subunit_order: List[str]) -> dict:
    """
    Load global_summary_pdb.csv for each subunit x atom from combined_pip_proximity.
    Returns dict: {(subunit, atom): set_of_pdb_labels}
    """
    close = {}
    prox_dir = base_dir / "combined_pip_proximity"
    for subunit in subunit_order:
        sub_dir = prox_dir / f"subunit_{subunit}"
        for atom in ATOMS:
            csv = sub_dir / atom / "global_summary_pdb.csv"
            if not csv.exists():
                print(f"  WARNING: {csv} not found -- no filter for ({subunit}, {atom})")
                close[(subunit, atom)] = None  # None = no filter
                continue
            df = pd.read_csv(csv)
            if "pdb_label" not in df.columns:
                close[(subunit, atom)] = None
                continue
            close[(subunit, atom)] = set(df["pdb_label"].astype(str).tolist())
    return close


def load_all_runs(base_dir: Path, gate: str, residue_num: str,
                  close_residues: dict = None) -> pd.DataFrame:
    """Load and stack correlation_all.csv from all RUNs for a gate."""
    dfs = []
    run_dirs = sorted(base_dir.glob("RUN*"))
    for run_dir in run_dirs:
        csv = run_dir / "pip_gate_correlation" / gate / "correlation_all.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["run"] = run_dir.name

        # filter to rows containing the residue number
        df = df[df["residue"].str.contains(rf"\.?{residue_num}\.", regex=True) |
                df["residue"].str.endswith(residue_num)]

        # extract subunits
        df["pip_subunit"] = df["pip"].apply(extract_subunit_from_pip)
        df["res_subunit"] = df["residue"].apply(extract_subunit_from_residue)

        # keep only matched subunit rows
        df = df[df["pip_subunit"] == df["res_subunit"]]

        # filter to close residues from global_summary_pdb if available
        if close_residues is not None:
            def _in_close(row):
                key = (row["res_subunit"], row["atom"])
                allowed = close_residues.get(key)
                if allowed is None:
                    return True  # no filter
                return row["residue"] in allowed
            df = df[df.apply(_in_close, axis=1)]

        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_boxplot(ax, data_by_subunit: dict, gate_col: str, title: str, subunit_order: List[str]):
    """Draw boxplot + jittered dots on ax."""
    subunits = [s for s in subunit_order if s in data_by_subunit and len(data_by_subunit[s]) > 0]
    if not subunits:
        ax.set_visible(False)
        return

    positions = list(range(len(subunits)))
    box_data  = [data_by_subunit[s][gate_col].values for s in subunits]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=""),
        showfliers=False,
    )

    for patch, subunit in zip(bp["boxes"], subunits):
        color = SUBUNIT_COLORS.get(subunit, "#888888")
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # jittered dots
    rng = np.random.default_rng(42)
    for i, (subunit, vals) in enumerate(zip(subunits, box_data)):
        color  = SUBUNIT_COLORS.get(subunit, "#888888")
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            i + jitter, vals,
            color=color, edgecolors="black", linewidths=0.5,
            s=40, zorder=3, alpha=0.85,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(subunits, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Spearman r", fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.3)


def plot_residue(base_dir: Path, residue_num: str, out_dir: Path, subunit_order: List[str], close_residues: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    for gate in GATE_FOLDERS:
        print(f"  Loading {gate}...")
        df = load_all_runs(base_dir, gate, residue_num, close_residues)
        if df.empty:
            print(f"    No data found for residue {residue_num} in {gate}")
            continue

        atoms_found = [a for a in ATOMS if a in df["atom"].values]
        if not atoms_found:
            continue

        # one figure per gate: rows=atoms, cols=r_AC / r_BD
        n_rows = len(atoms_found)
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(10, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Residue {residue_num} — {GATE_LABELS.get(gate, gate)}\n"
            f"(matched subunit, {df['run'].nunique()} RUNs)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for row_idx, atom in enumerate(atoms_found):
            atom_df = df[df["atom"] == atom]

            data_by_subunit = {
                s: atom_df[atom_df["res_subunit"] == s]
                for s in subunit_order
            }

            make_boxplot(
                axes[row_idx, 0], data_by_subunit,
                "r_AC",
                f"{atom} — r_AC (A_C gate)",
                subunit_order,
            )
            make_boxplot(
                axes[row_idx, 1], data_by_subunit,
                "r_BD",
                f"{atom} — r_BD (B_D gate)",
                subunit_order,
            )

        # legend
        patches = [
            mpatches.Patch(color=SUBUNIT_COLORS[s], label=s, alpha=0.7)
            for s in subunit_order
        ]
        fig.legend(handles=patches, title="Subunit", loc="upper right",
                   bbox_to_anchor=(1.12, 1.0), fontsize=9)

        plt.tight_layout()
        out_path = out_dir / f"residue{residue_num}_{gate}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out_path}")


def top4_per_run(base_dir: Path, residue_num: str, out_dir: Path, subunit_order: List[str], close_residues: dict = None):
    """
    For each RUN x atom x gate, find top 4 residues by max(|r_AC|, |r_BD|).
    Matched subunit only. Save one CSV with all results.
    """
    rows = []
    run_dirs = sorted(base_dir.glob("RUN*"))

    for run_dir in run_dirs:
        run_name = run_dir.name
        for gate in GATE_FOLDERS:
            csv = run_dir / "pip_gate_correlation" / gate / "correlation_all.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)

            # filter to residue number
            df = df[df["residue"].str.contains(rf"\.?{residue_num}\.", regex=True) |
                    df["residue"].str.endswith(residue_num)]
            if df.empty:
                continue

            # matched subunit only
            df["pip_subunit"] = df["pip"].apply(extract_subunit_from_pip)
            df["res_subunit"] = df["residue"].apply(extract_subunit_from_residue)
            df = df[df["pip_subunit"] == df["res_subunit"]]
            if df.empty:
                continue

            # filter to close residues from global_summary_pdb if available
            if close_residues is not None:
                def _in_close(row):
                    key = (row["res_subunit"], row["atom"])
                    allowed = close_residues.get(key)
                    if allowed is None:
                        return True
                    return row["residue"] in allowed
                df = df[df.apply(_in_close, axis=1)]
            if df.empty:
                continue

            # best_r = max(|r_AC|, |r_BD|), note which gate
            df["abs_r_AC"] = df["r_AC"].abs()
            df["abs_r_BD"] = df["r_BD"].abs()
            df["best_r"]   = df[["abs_r_AC", "abs_r_BD"]].max(axis=1)
            df["best_gate"] = df.apply(
                lambda r: "r_AC" if r["abs_r_AC"] >= r["abs_r_BD"] else "r_BD", axis=1
            )
            df["best_r_signed"] = df.apply(
                lambda r: r["r_AC"] if r["best_gate"] == "r_AC" else r["r_BD"], axis=1
            )

            for atom in ATOMS:
                atom_df = df[df["atom"] == atom].copy()
                if atom_df.empty:
                    continue
                top4 = atom_df.nlargest(4, "best_r")
                for _, row in top4.iterrows():
                    rows.append({
                        "run":       run_name,
                        "gate":      gate,
                        "atom":      atom,
                        "residue":   row["residue"],
                        "best_gate": row["best_gate"],
                        "best_r":    round(row["best_r_signed"], 4),
                        "r_AC":      round(row["r_AC"], 4),
                        "r_BD":      round(row["r_BD"], 4),
                    })

    if not rows:
        print("  No data found for top4 analysis.")
        return

    out_df  = pd.DataFrame(rows)
    top_dir = out_dir / f"top4_per_run_residue{residue_num}"

    for gate in GATE_FOLDERS:
        gate_df = out_df[out_df["gate"] == gate]
        if gate_df.empty:
            continue
        gate_dir = top_dir / gate
        gate_dir.mkdir(parents=True, exist_ok=True)
        for atom in ATOMS:
            atom_df = gate_df[gate_df["atom"] == atom]
            if atom_df.empty:
                continue
            out_path = gate_dir / f"{atom}.csv"
            atom_df.to_csv(out_path, index=False)
            print(f"  Saved {gate}/{atom} -> {out_path} ({len(atom_df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Path to results dir containing RUN0..RUN19")
    parser.add_argument("--residue", required=True,
                        help="Residue number to plot, e.g. 197")
    parser.add_argument("--channel_type", default="G2", choices=["G2", "G12"],
                        help="Channel type: G2 or G12 (affects subunit order)")
    parser.add_argument("--out_dir", default="./residue_corr_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    base_dir      = Path(args.base_dir)
    out_dir       = Path(args.out_dir)
    subunit_order = SUBUNIT_ORDER_MAP[args.channel_type]

    print(f"\nResidueCorrelationPlotter")
    print(f"  base_dir     : {base_dir}")
    print(f"  channel_type : {args.channel_type}")
    print(f"  subunit_order: {subunit_order}")
    print(f"  residue      : {args.residue}")
    print(f"  out_dir      : {out_dir}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    close_residues = load_close_residues(base_dir, subunit_order)
    plot_residue(base_dir, args.residue, out_dir, subunit_order, close_residues)
    top4_per_run(base_dir, args.residue, out_dir, subunit_order, close_residues)
    print("\nDone.")


if __name__ == "__main__":
    main()

'''
# G2
python3 plot_residue_correlations.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2  \
    --residue 90 --channel_type G2 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/residue_correlations/plots_90

# G12
python plot_residue_correlations.py \
    --base_dir .../G12 --residue 197 --channel_type G12 --out_dir ./plots_197_g12
'''