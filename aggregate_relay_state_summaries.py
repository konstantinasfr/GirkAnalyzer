"""
analysis/aggregate_relay_state_summaries.py
=============================================
Combines the per-run relay_state_summary.csv files (already produced
by relay_state_classification.py, one per RUN) across all runs and
channel types into a single aggregated view.

Reads:
    <results_root>/<channel_type>/RUN<n>/network_199_315_324_233/
        <channel_type>_RUN<n>_relay_state_summary.csv

Usage:
    python -m analysis.aggregate_relay_state_summaries \
        --results_root /media/ziyue/.../girk_analyser_results \
        --channel_types G12 G12_ML G12_GAT \
        --n_runs 20

Outputs (saved under results_root):
    relay_state_all_runs_combined.csv         (every subunit-run row, concatenated)
    relay_switching_summary_by_channel.csv    (switching % by channel_type)
    relay_state_breakdown_by_subunit.csv      (mean open/closed/ambiguous per subunit, across runs)
"""

import argparse
from pathlib import Path

import pandas as pd

NETWORK_FOLDER = "network_199_315_324_233"


def collect_summaries(results_root, channel_types, n_runs):
    results_root = Path(results_root)
    dfs = []

    for channel_type in channel_types:
        for run_num in range(1, n_runs + 1):
            csv_path = (
                results_root / channel_type / f"RUN{run_num}" / NETWORK_FOLDER
                / f"{channel_type}_RUN{run_num}_relay_state_summary.csv"
            )
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            dfs.append(df)

    if not dfs:
        print("No relay_state_summary.csv files found -- check results_root / channel_types / n_runs.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} per-run summary files, {len(combined)} subunit-run rows total.")
    return combined


def summarize_switching(df):
    total = len(df)
    n_switching = int(df["is_switching"].sum())
    print(f"\nOverall: {n_switching} / {total} subunit-run combinations show switching "
          f"({100.0 * n_switching / total:.1f}%)")

    by_channel = df.groupby("channel_type").agg(
        n_subunit_runs=("is_switching", "size"),
        n_switching=("is_switching", "sum"),
    )
    by_channel["pct_switching"] = 100.0 * by_channel["n_switching"] / by_channel["n_subunit_runs"]
    print("\nBy channel type:")
    print(by_channel.to_string())

    by_subunit = df.groupby(["channel_type", "subunit"]).agg(
        n_subunit_runs=("is_switching", "size"),
        n_switching=("is_switching", "sum"),
    )
    by_subunit["pct_switching"] = 100.0 * by_subunit["n_switching"] / by_subunit["n_subunit_runs"]
    print("\nBy channel type + subunit:")
    print(by_subunit.to_string())

    return by_channel, by_subunit


def classify_run_category(row, min_occupancy_pct):
    """Every run falls into exactly one of these 4 mutually exclusive categories."""
    open_ok = row["pct_open"] >= min_occupancy_pct
    closed_ok = row["pct_closed"] >= min_occupancy_pct
    if open_ok and closed_ok:
        return "switching"
    elif open_ok:
        return "open_only"
    elif closed_ok:
        return "closed_only"
    else:
        return "ambiguous_only"


def summarize_run_categories(df, min_occupancy_pct):
    df = df.copy()
    df["run_category"] = df.apply(classify_run_category, axis=1, min_occupancy_pct=min_occupancy_pct)

    counts = df.groupby(["channel_type", "subunit", "run_category"]).size().unstack(fill_value=0)
    for cat in ["switching", "open_only", "closed_only", "ambiguous_only"]:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[["switching", "open_only", "closed_only", "ambiguous_only"]]

    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0).mul(100).round(1)

    print(f"\nFull run categorization (occupancy floor = {min_occupancy_pct}%):")
    print(f"{'':>17} {'switching':>10} {'open_only':>10} {'closed_only':>12} {'ambiguous_only':>15}   n_runs")
    for (channel_type, subunit) in counts.index:
        row_n = counts.loc[(channel_type, subunit)]
        row_p = pct.loc[(channel_type, subunit)]
        n = totals.loc[(channel_type, subunit)]
        print(f"  {channel_type:>7} - {subunit:>3}  "
              f"{row_n['switching']:>4} ({row_p['switching']:>4.0f}%)  "
              f"{row_n['open_only']:>4} ({row_p['open_only']:>4.0f}%)  "
              f"{row_n['closed_only']:>5} ({row_p['closed_only']:>4.0f}%)  "
              f"{row_n['ambiguous_only']:>7} ({row_p['ambiguous_only']:>4.0f}%)   n={n}")

    return counts, pct, df


def summarize_state_breakdown(df):
    breakdown = df.groupby(["channel_type", "subunit"]).agg(
        n_runs=("run", "nunique"),
        mean_pct_open=("pct_open", "mean"),
        mean_pct_closed=("pct_closed", "mean"),
        mean_pct_ambiguous=("pct_ambiguous", "mean"),
        mean_n_transitions=("n_transitions", "mean"),
    ).round(1)

    print("\nMean state occupancy per subunit (averaged across runs):")
    for (channel_type, subunit), row in breakdown.iterrows():
        print(f"  {channel_type:>8} - {subunit:>3}  (n={int(row['n_runs'])} runs): "
              f"open {row['mean_pct_open']:>5.1f}%   "
              f"closed {row['mean_pct_closed']:>5.1f}%   "
              f"ambiguous {row['mean_pct_ambiguous']:>5.1f}%   "
              f"transitions~{row['mean_n_transitions']:.1f}")

    return breakdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per-run relay state summaries.")
    parser.add_argument("--results_root", required=True,
                         help="e.g. /media/ziyue/.../girk_analyser_results")
    parser.add_argument("--channel_types", nargs="+", default=["G12", "G12_ML", "G12_GAT"])
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--output_dir", default=None,
                         help="Where to save results. Defaults to "
                              "<results_root>/aggregate_relay_state_summaries/ if not given.")
    parser.add_argument("--min_occupancy_pct", type=float, default=2.0,
                         help="Must match the value used in relay_state_classification.py")
    args = parser.parse_args()

    combined = collect_summaries(args.results_root, args.channel_types, args.n_runs)
    if combined.empty:
        raise SystemExit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / "aggregate_relay_state_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / "relay_state_all_runs_combined.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"\nSaved combined per subunit-run table -> {combined_csv}")

    by_channel, by_subunit = summarize_switching(combined)
    by_channel_csv = out_dir / "relay_switching_summary_by_channel.csv"
    by_channel.to_csv(by_channel_csv)
    print(f"Saved channel-level switching summary -> {by_channel_csv}")

    by_subunit_csv = out_dir / "relay_switching_summary_by_channel_subunit.csv"
    by_subunit.to_csv(by_subunit_csv)
    print(f"Saved channel+subunit switching summary -> {by_subunit_csv}")

    breakdown = summarize_state_breakdown(combined)
    breakdown_csv = out_dir / "relay_state_breakdown_by_subunit.csv"
    breakdown.to_csv(breakdown_csv)
    print(f"Saved state breakdown by subunit -> {breakdown_csv}")

    counts, pct, combined_with_category = summarize_run_categories(combined, args.min_occupancy_pct)
    counts_csv = out_dir / "relay_run_category_counts.csv"
    pct_csv = out_dir / "relay_run_category_pct.csv"
    counts.to_csv(counts_csv)
    pct.to_csv(pct_csv)
    print(f"Saved run category counts -> {counts_csv}")
    print(f"Saved run category percentages -> {pct_csv}")


'''
python3 aggregate_relay_state_summaries.py \
    --results_root /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results \
    --channel_types G12 \
    --n_runs 20 \
    --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_relay_state_summaries
'''