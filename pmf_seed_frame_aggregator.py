"""
pmf_seed_frame_aggregator.py

Pools the per-run PMF seed-frame candidate CSVs (written by
pmf_seed_frame_finder.py, under each RUN's
results_dir/pmf_seed_frames/<case_label>.csv) across every RUN of a
channel type, into one combined CSV per case.

Follows the "pool events not means" convention used everywhere else in
this pipeline: rows are concatenated as-is (each run already ranked its
own candidates), a "run" column is added, and the pooled table is
re-sorted GLOBALLY across all runs.

Sort order: highest sf_fraction first — i.e. the ion sitting closest to
the TOP of the SF (the upper1 / bulk-entry boundary), NOT the deepest one.
(Recall: within a single frame, the target ion is chosen as the *deepest*
of however many ions are in the SF, per your rule that any other SF ion
must be above it — that per-frame choice is unchanged. This aggregator
just ranks the resulting candidate frames, across every run, by how close
that deepest/target ion still gets to the top of the SF.)

Usage (standalone):
    python3 -m analysis.pmf_seed_frame_aggregator \
        --channel_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12

Usage (import):
    from analysis.pmf_seed_frame_aggregator import run_pmf_seed_frame_aggregation
    run_pmf_seed_frame_aggregation(
        channel_dir="/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12"
    )
"""

import argparse
import csv
import re
from pathlib import Path

CASE_LABELS = ["case0_sf_only", "case1_sf_plus1cavity", "case2_sf_plus2cavity"]


def _run_sort_key(run_dir_name):
    """Natural sort: RUN2 before RUN10 (not lexicographic)."""
    m = re.search(r"(\d+)$", run_dir_name)
    return int(m.group(1)) if m else run_dir_name


def _find_run_dirs(channel_dir):
    run_dirs = [p for p in channel_dir.glob("RUN*") if p.is_dir()]
    run_dirs.sort(key=lambda p: _run_sort_key(p.name))
    return run_dirs


def run_pmf_seed_frame_aggregation(
    channel_dir,
    force_rerun=False,
):
    """
    channel_dir: path to the folder that directly contains RUN1, RUN2, ...
                 e.g. ".../girk_analyser_results/G12"

    Reads {channel_dir}/RUN*/pmf_seed_frames/<case>.csv for every RUN,
    pools rows per case, sorts each pooled table by sf_fraction descending
    (best = closest to the top/upper1 side of the SF), and writes:

        {channel_dir}/pmf_seed_frames_aggregated/<case>_pooled.csv
        {channel_dir}/pmf_seed_frames_aggregated/summary.txt
    """
    channel_dir = Path(channel_dir)
    channel_type = channel_dir.name  # just for display
    out_dir = channel_dir / "pmf_seed_frames_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.txt"
    if summary_path.exists() and not force_rerun:
        print(f"{summary_path} already exists — skipping (force_rerun=False).")
        return

    run_dirs = _find_run_dirs(channel_dir)
    if not run_dirs:
        print(f"No RUN* folders found under {channel_dir}")
        return

    print("\n" + "=" * 80)
    print(f"PMF SEED FRAME AGGREGATION — {channel_type}  ({len(run_dirs)} runs)")
    print("=" * 80)

    summary_lines = [f"PMF seed frame aggregation — {channel_type}",
                      f"Runs scanned: {len(run_dirs)}", ""]

    for case_label in CASE_LABELS:
        pooled_rows = []
        fieldnames = None
        per_run_counts = {}

        for run_dir in run_dirs:
            csv_path = run_dir / "pmf_seed_frames" / f"{case_label}.csv"
            if not csv_path.exists():
                per_run_counts[run_dir.name] = 0
                continue

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                rows = list(reader)

            per_run_counts[run_dir.name] = len(rows)
            for row in rows:
                row["run"] = run_dir.name
                pooled_rows.append(row)

        n_with_data = sum(1 for v in per_run_counts.values() if v > 0)
        print(f"\n  [{case_label}] {n_with_data}/{len(run_dirs)} runs had "
              f"candidates, {len(pooled_rows)} rows pooled")

        if not pooled_rows:
            summary_lines.append(f"[{case_label}] — 0 rows pooled (no runs had candidates)")
            summary_lines.append("")
            continue

        # Best = highest sf_fraction = closest to the TOP of the SF
        # (upper1 / bulk-entry side), longest dwell as tiebreak.
        pooled_rows.sort(
            key=lambda r: (-float(r["sf_fraction"]), -int(r["dwell_length_frames"]))
        )

        out_fieldnames = ["global_rank", "run"] + [f for f in fieldnames if f != "rank"]
        out_csv = out_dir / f"{case_label}_pooled.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            for rank, row in enumerate(pooled_rows, start=1):
                out_row = {"global_rank": rank, "run": row["run"]}
                out_row.update({k: row[k] for k in fieldnames if k != "rank"})
                writer.writerow(out_row)

        best = pooled_rows[0]
        n_meets_cutoff = sum(1 for r in pooled_rows if r.get("meets_cutoff") == "True")
        print(f"    Saved: {out_csv}")
        print(f"    BEST overall: run={best['run']}  frame={best['best_frame']}  "
              f"sf_fraction={best['sf_fraction']}  "
              f"dist_from_upper1_A={best.get('dist_from_upper1_A')}  "
              f"meets_cutoff={best.get('meets_cutoff')}")
        print(f"    {n_meets_cutoff}/{len(pooled_rows)} pooled rows meet the cutoff "
              f"used when each run was screened")

        summary_lines.append(f"[{case_label}] — {len(pooled_rows)} rows pooled from "
                              f"{n_with_data}/{len(run_dirs)} runs")
        summary_lines.append(
            f"  BEST OVERALL: run={best['run']}, frame={best['best_frame']} "
            f"(t={best['best_frame_t_ns']} ns), ion={best['ion_resid']}, "
            f"sf_fraction={best['sf_fraction']}, "
            f"dist_from_upper1_A={best.get('dist_from_upper1_A')}, "
            f"n_sf_ions={best.get('n_sf_ions_at_best_frame')}, "
            f"dwell={best['dwell_length_frames']} frames"
        )
        summary_lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n  Summary written to {summary_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-run PMF seed-frame candidates into pooled, "
                     "globally-ranked CSVs per case."
    )
    parser.add_argument("--channel_dir", required=True,
                         help="Folder that directly contains RUN1, RUN2, ... "
                              "e.g. .../girk_analyser_results/G12")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    run_pmf_seed_frame_aggregation(
        channel_dir=args.channel_dir,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()


'''
python3 pmf_seed_frame_aggregator.py --channel_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12
'''