"""
combine_switch_pairs.py

Combines switch_mechanism_pairs.csv files across all RUNs and subunit combinations,
finds pairs that appear most consistently, and reports mean lag +/- std.

Usage:
    python3 combine_switch_pairs.py \
        --base /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
        --runs 1 20 \
        --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_switch_pairs
"""
"""
combine_switch_pairs.py

Combines switch_mechanism_pairs.csv files across all RUNs and subunit combinations,
finds pairs that appear most consistently, and reports mean lag +/- std.

Usage:
    python3 combine_switch_pairs.py \
        --base /path/to/G12 \
        --runs 1 20 \
        --out ./combined_switch_output
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUBUNIT_COMBOS = [
    "197_2b_199_2b_315_2a",
    "197_2c_199_2c_315_2b",
    "197_2a_199_2a_304_G1",
    "186_G1_188_G1_315_2c",
]

ANALYSES = ["all_interactions", "sc_any_only", "sc_sc_only"]


def main():
    parser = argparse.ArgumentParser(description="Combine switch pairs across RUNs")
    parser.add_argument("--base", required=True,
                        help="Base path to G12 folder")
    parser.add_argument("--runs", nargs=2, type=int, default=[1, 20],
                        metavar=("START", "END"),
                        help="RUN range inclusive (default: 1 20)")
    parser.add_argument("--min_runs", type=int, default=3,
                        help="Min RUNs a pair must appear in to be reported (default: 3)")
    parser.add_argument("--out", default="./combined_switch_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_start, run_end = args.runs

    for analysis in ANALYSES:
        print(f"\n{'='*60}")
        print(f"Analysis: {analysis}")
        print(f"{'='*60}")

        all_dfs = []

        for run_n in range(run_start, run_end + 1):
            for combo in SUBUNIT_COMBOS:
                csv_path = os.path.join(
                    args.base, f"RUN{run_n}", "switch_output",
                    combo, analysis, "switch_mechanism_pairs.csv"
                )
                if not os.path.exists(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    df["run"]   = run_n
                    df["combo"] = combo
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: could not read {csv_path}: {e}")

        if not all_dfs:
            print(f"  No data found for {analysis}")
            continue

        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"  Loaded {len(df_all)} pairs from {df_all['run'].nunique()} RUNs")

        # ── Group and count ───────────────────────────────────────────────────
        group_cols = ["bond_A", "event_A", "bond_B", "event_B", "pair_type"]

        agg = df_all.groupby(group_cols).agg(
            n_runs      = ("run",        "nunique"),
            runs_list   = ("run",        lambda x: sorted(x.unique().tolist())),
            combos_list = ("combo",      lambda x: sorted(x.unique().tolist())),
            all_lags    = ("lag_frames", lambda x: sorted(x.tolist())),
            mean_lag    = ("lag_frames", "mean"),
            std_lag     = ("lag_frames", "std"),
            min_lag     = ("lag_frames", "min"),
            max_lag     = ("lag_frames", "max"),
            n_total     = ("lag_frames", "count"),
        ).reset_index()

        agg["mean_lag"] = agg["mean_lag"].round(1)
        agg["std_lag"]  = agg["std_lag"].round(1)
        agg = agg.sort_values("n_runs", ascending=False)

        # filter by min_runs
        agg_filtered = agg[agg["n_runs"] >= args.min_runs]

        # Save full and filtered
        out_dir = os.path.join(args.out, analysis)
        os.makedirs(out_dir, exist_ok=True)

        agg.to_csv(os.path.join(out_dir, "combined_pairs_all.csv"), index=False)
        agg_filtered.to_csv(os.path.join(out_dir, "combined_pairs_conserved.csv"), index=False)

        print(f"  {len(agg_filtered)} conserved pairs (appearing in >= {args.min_runs} RUNs)")
        print(f"\n  TOP 20 CONSERVED PAIRS:")
        print(f"  {'bond_A':<35} {'evt_A':<12} {'bond_B':<35} {'evt_B':<12} {'type':<22} {'n_runs':>6}  {'mean_lag':>9}  {'std_lag':>8}")
        print(f"  {'-'*140}")
        for _, row in agg_filtered.head(20).iterrows():
            print(f"  {row['bond_A']:<35} {row['event_A']:<12} {row['bond_B']:<35} {row['event_B']:<12} "
                  f"{row['pair_type']:<22} {row['n_runs']:>6}  {row['mean_lag']:>9}  {row['std_lag']:>8}")

        # ── Plot: top conserved pairs ─────────────────────────────────────────
        top = agg_filtered.head(20).copy()
        top["label"] = top["bond_A"].str.split("↔").str[1] + " " + top["event_A"] + \
                       " → " + top["bond_B"].str.split("↔").str[1] + " " + top["event_B"]

        fig, axes = plt.subplots(1, 2, figsize=(18, max(4, len(top) * 0.4 + 2)))

        # Left: bar chart of n_runs
        ax = axes[0]
        colors = plt.cm.RdYlGn(top["n_runs"] / (run_end - run_start + 1))
        bars = ax.barh(range(len(top)), top["n_runs"], color=colors,
                       edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["label"], fontsize=7)
        ax.set_xlabel("Number of RUNs", fontsize=10)
        ax.set_title("Pair frequency across RUNs", fontsize=10, fontweight="bold")
        ax.axvline(args.min_runs, color="red", lw=1, ls="--", alpha=0.5, label=f"min={args.min_runs}")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        for bar, (_, row) in zip(bars, top.iterrows()):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{row['n_runs']}/{run_end-run_start+1}",
                    va="center", fontsize=7)

        # Right: lag distribution (mean ± std)
        ax2 = axes[1]
        y = range(len(top))
        ax2.barh(y, top["mean_lag"], xerr=top["std_lag"].fillna(0),
                 color="#5b9bd5", edgecolor="black", linewidth=0.5,
                 error_kw={"ecolor": "black", "capsize": 3})
        ax2.set_yticks(y)
        ax2.set_yticklabels(top["label"], fontsize=7)
        ax2.set_xlabel("Mean lag (frames) ± std", fontsize=10)
        ax2.set_title("Lag between events", fontsize=10, fontweight="bold")
        ax2.axvline(0, color="gray", lw=0.8, ls="--")
        ax2.invert_yaxis()

        fig.suptitle(f"Conserved Switch Pairs — {analysis}\n"
                     f"RUN{run_start}–RUN{run_end}, min_runs={args.min_runs}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "conserved_pairs.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {out_dir}/conserved_pairs.png")
        print(f"  Saved: {out_dir}/combined_pairs_conserved.csv")
        print(f"  Saved: {out_dir}/combined_pairs_all.csv")

    print("\nAll done.")


if __name__ == "__main__":
    main()