"""
top4_close_residues.py
======================
For each RUN x gate x atom, find top 4 residues by max(|r_AC|, |r_BD|).
Only considers residues in combined_pip_proximity global_summary_pdb.csv (close whitelist).
Matched subunit only (PIP subunit == residue subunit).

Output folder structure:
    out_dir/
        hbc/P1.csv, P4.csv, P5.csv
        upper_gl/P1.csv, P4.csv, P5.csv
        lower_gl/P1.csv, P4.csv, P5.csv

Usage:
    python top4_close_residues.py \
        --base_dir /path/to/G2 \
        --channel_type G2 \
        --out_dir ./top4_results
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


GATE_FOLDERS  = ["hbc", "upper_gl", "lower_gl"]
ATOMS         = ["P1", "P4", "P5"]
SUBUNIT_ORDER_MAP = {
    "G2":  ["2a", "2b", "2c", "2d"],
    "G12": ["2a", "2b", "2c", "G1"],
}


def extract_subunit_from_pip(pip_label: str) -> str:
    return pip_label.split("_")[-1]


def extract_subunit_from_residue(residue: str) -> str:
    if "." in residue:
        return residue.split(".")[-1]
    return ""


def load_close_residues(base_dir: Path, subunit_order: List[str]) -> dict:
    """
    Load global_summary_pdb.csv for each subunit x atom.
    Returns {(subunit, atom): set_of_pdb_labels}
    """
    close = {}
    prox_dir = base_dir / "combined_pip_proximity"
    for subunit in subunit_order:
        sub_dir = prox_dir / f"subunit_{subunit}"
        for atom in ATOMS:
            csv = sub_dir / atom / "global_summary_pdb.csv"
            if not csv.exists():
                print(f"  WARNING: {csv} not found -- no filter for ({subunit}, {atom})")
                close[(subunit, atom)] = None
                continue
            df = pd.read_csv(csv)
            if "pdb_label" not in df.columns:
                close[(subunit, atom)] = None
                continue
            close[(subunit, atom)] = set(df["pdb_label"].astype(str).tolist())
            print(f"  Loaded {len(close[(subunit, atom)])} close residues for {subunit}/{atom}")
    return close


def _make_barplots(plot_data: dict, plot_dir: Path, topn: int = 5):
    """For each gate x atom: bar plot of residue appearance count + mean |best_r| on top."""
    for (gate, atom), df in plot_data.items():
        # strip subunit: GLN197.2a -> GLN197
        df = df.copy()
        df["residue_base"] = df["residue"].apply(
            lambda r: r if r.endswith(".G1") else (r.split(".")[0] if "." in r else r)
        )
        df["abs_best_r"] = df["best_r"].abs()

        counts   = df.groupby("residue_base").size().reset_index(name="count")
        mean_r   = df.groupby("residue_base")["abs_best_r"].mean().reset_index(name="mean_abs_r")
        summary  = counts.merge(mean_r, on="residue_base").sort_values("count", ascending=False)

        fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.8 + 2), 5))
        bars = ax.bar(
            range(len(summary)),
            summary["count"],
            color="#4C72B0", edgecolor="black", linewidth=0.7, alpha=0.8,
        )
        # mean |r| on top of each bar
        for bar, (_, row) in zip(bars, summary.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{row['mean_abs_r']:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["residue_base"], rotation=45, ha="right", fontsize=9)
        n_runs       = df["run"].nunique()
        max_possible = n_runs * topn
        ax.set_ylabel("Times in top-N (across runs)", fontsize=10)
        ax.set_title(
            f"{gate} | {atom} — residue appearance count "
            f"(max possible = {n_runs} runs x {topn} = {max_possible})\n"
            f"number above bar = mean |r|",
            fontsize=11, fontweight="bold")
        ax.set_ylim(0, summary["count"].max() + 2)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = plot_dir / f"{gate}_{atom}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Barplot -> {out_path}")


def run_top4(base_dir: Path, out_dir: Path, subunit_order: List[str], topn: int = 5):
    close_residues = load_close_residues(base_dir, subunit_order)

    run_dirs = sorted(base_dir.glob("RUN*"))
    all_rows = {gate: [] for gate in GATE_FOLDERS}

    for run_dir in run_dirs:
        run_name = run_dir.name
        for gate in GATE_FOLDERS:
            csv = run_dir / "pip_gate_correlation" / gate / "correlation_all.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)

            # extract subunits
            df["pip_subunit"] = df["pip"].apply(extract_subunit_from_pip)
            df["res_subunit"] = df["residue"].apply(extract_subunit_from_residue)

            # matched subunit only
            df = df[df["pip_subunit"] == df["res_subunit"]]
            if df.empty:
                continue

            # filter to close residues whitelist
            def _in_close(row):
                key = (row["res_subunit"], row["atom"])
                allowed = close_residues.get(key)
                if allowed is None:
                    return True
                return row["residue"] in allowed
            df = df[df.apply(_in_close, axis=1)]
            if df.empty:
                continue

            # compute best_r
            df["abs_r_AC"]     = df["r_AC"].abs()
            df["abs_r_BD"]     = df["r_BD"].abs()
            df["best_abs_r"]   = df[["abs_r_AC", "abs_r_BD"]].max(axis=1)
            df["best_gate"]    = df.apply(
                lambda r: "r_AC" if r["abs_r_AC"] >= r["abs_r_BD"] else "r_BD", axis=1
            )
            df["best_r"] = df.apply(
                lambda r: r["r_AC"] if r["best_gate"] == "r_AC" else r["r_BD"], axis=1
            )

            for atom in ATOMS:
                atom_df = df[df["atom"] == atom].copy()
                if atom_df.empty:
                    continue
                top4 = atom_df.nlargest(topn, "best_abs_r")
                for _, row in top4.iterrows():
                    all_rows[gate].append({
                        "run":       run_name,
                        "atom":      atom,
                        "residue":   row["residue"],
                        "best_gate": row["best_gate"],
                        "best_r":    round(float(row["best_r"]), 4),
                        "r_AC":      round(float(row["r_AC"]), 4),
                        "r_BD":      round(float(row["r_BD"]), 4),
                    })

    # save CSVs and collect for plots
    plot_data = {}  # {(gate, atom): DataFrame}
    for gate in GATE_FOLDERS:
        if not all_rows[gate]:
            print(f"  No data for {gate}")
            continue
        gate_df  = pd.DataFrame(all_rows[gate])
        gate_dir = out_dir / gate
        gate_dir.mkdir(parents=True, exist_ok=True)
        for atom in ATOMS:
            atom_df = gate_df[gate_df["atom"] == atom]
            if atom_df.empty:
                continue
            out_path = gate_dir / f"{atom}.csv"
            atom_df.to_csv(out_path, index=False)
            print(f"  Saved {gate}/{atom} -> {out_path} ({len(atom_df)} rows)")
            plot_data[(gate, atom)] = atom_df.copy()

    # bar plots
    plot_dir = out_dir / "barplots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _make_barplots(plot_data, plot_dir, topn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--channel_type", default="G2", choices=["G2", "G12"])
    parser.add_argument("--topn", type=int, default=5,
                        help="How many top residues per RUN x gate x atom (default 5)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    base_dir      = Path(args.base_dir)
    out_dir       = Path(args.out_dir)
    subunit_order = SUBUNIT_ORDER_MAP[args.channel_type]

    print(f"\ntop4_close_residues")
    print(f"  base_dir     : {base_dir}")
    print(f"  channel_type : {args.channel_type}")
    print(f"  out_dir      : {out_dir}")
    print()

    run_top4(base_dir, out_dir, subunit_order, args.topn)
    print("\nDone.")


if __name__ == "__main__":
    main()

'''

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/top1_correlation_results \
    --topn 1

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/top3_correlation_results \
    --topn 3

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/top5_correlation_results \
    --topn 5

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/top10_correlation_results \
    --topn 10


python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/top1_correlation_results \
    --topn 1

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/top3_correlation_results \
    --topn 3

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/top5_correlation_results \
    --topn 5

python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/top10_correlation_results \
    --topn 10

    python top4_close_residues.py \
    --base_dir .../G2 \
    --channel_type G2 \
    --out_dir ./top4_results \
    --topn 5


python3 top4_close_residues.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12 \
    --out_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/top3_correlation_results \
    --topn 3

    '''