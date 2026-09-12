"""
aggregate_residence_time_stratified.py

Pools per-run residence-time-stratified-by-occupancy results (produced by
residence_time_stratified_analysis.py) across all RUN* directories under a
channel's results folder, e.g.:

    /media/.../girk_analyser_results/G12/RUN1/residence_time_stratified/
    /media/.../girk_analyser_results/G12/RUN2/residence_time_stratified/
    ...

Follows the "pool events, not means" principle used throughout the
pipeline: individual permeation events from every run are concatenated
into one pooled list BEFORE computing any statistics or plots — never
by averaging per-run summaries.

This script does NOT recompute anything from ion_region_tracking.json; it
simply reads the already-computed per-run CSVs (residence_time_stratified_
results.csv and intermediate_occupancy_per_event.csv) and pools them. Run
residence_time_stratified_analysis.py on each RUN first.

Reuses (does not duplicate) the summary/plotting functions already defined
in residence_time_stratified_analysis.py — pooled data is just fed into
the same functions used for a single run.

Outputs (saved to <base_dir>/residence_time_stratified_aggregated/):
  pooled_residence_time_stratified_results.csv   — all events, all runs, with a 'run' column
  group_summary.csv                              — mean/median/std/N per final-N group, pooled
  residence_time_boxplot.png                     — pooled boxplot
  pooled_intermediate_occupancy_per_event.csv    — all intermediate-occupancy rows, all runs
  intermediate_occupancy_group_summary.csv       — pooled occurrence/pct stats
  intermediate_occupancy_finalN_<N>.png          — pooled barplots, one per final N

Usage:
    python3 aggregate_residence_time_stratified.py \\
        --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \\
        --channel_type G12
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

# Reuse the exact same summary/plot logic used for a single run — no
# duplicated statistics code, so pooled results are guaranteed consistent
# with what a single-run analysis would produce.
from analysis.residence_time_stratified_analysis import (
    save_group_summary_csv,
    plot_boxplot,
    print_summary,
    aggregate_intermediate_by_final_group,
    save_intermediate_group_summary_csv,
    plot_intermediate_occupancy_barplots,
)


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + loading
# ─────────────────────────────────────────────────────────────────────────────

def find_run_dirs(base_dir):
    """Return sorted list of RUN* directories under base_dir."""
    base_dir = Path(base_dir)
    run_dirs = sorted(
        [p for p in base_dir.glob("RUN*") if p.is_dir()],
        key=lambda p: p.name,
    )
    return run_dirs


def load_pooled_residence_events(base_dir):
    """
    Read residence_time_stratified_results.csv from every RUN* dir and
    pool all events into one list of dicts, tagged with which run they
    came from.

    Returns
    -------
    list of dicts, each with keys: run, ion_id, end_1, end_2, end_3,
    residence_frames, residence_ns, n_ions_at_end3, ion_ids_at_end3,
    perm_ion_region_at_end3
    """
    run_dirs = find_run_dirs(base_dir)
    pooled = []

    for run_dir in run_dirs:
        csv_path = run_dir / "residence_time_stratified" / "residence_time_stratified_results.csv"
        if not csv_path.exists():
            print(f"  [skip] {run_dir.name}: no residence_time_stratified_results.csv found")
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            n_events = 0
            for row in reader:
                pooled.append({
                    "run":                     run_dir.name,
                    "ion_id":                  int(row["ion_id"]),
                    "end_1":                   int(row["end_1"]),
                    "end_2":                   row["end_2"],
                    "end_3":                   int(row["end_3"]),
                    "residence_frames":        int(row["residence_frames"]),
                    "n_ions_at_end3":          int(row["n_ions_at_end3"]),
                    "ion_ids_at_end3":         row["ion_ids_at_end3"],
                    "perm_ion_region_at_end3": row["perm_ion_region_at_end3"],
                })
                n_events += 1
        print(f"  [ok]   {run_dir.name}: {n_events} events")

    return pooled


def load_pooled_intermediate_events(base_dir):
    """
    Read intermediate_occupancy_per_event.csv (long format: one row per
    event per intermediate_N) from every RUN* dir, pool all rows, and
    reconstruct per-event dicts of the form:
        {"run": ..., "ion_id": ..., "end_1": ..., "end_3": ...,
         "final_N": ..., "pct_dict": {intermediate_N: pct, ...}}
    which is exactly what aggregate_intermediate_by_final_group() expects.

    Returns
    -------
    (pooled_rows, pooled_events)
        pooled_rows   : flat list of dicts (one per CSV row, tagged with run) — for writing the combined CSV
        pooled_events : list of per-event dicts with reconstructed pct_dict — for aggregation/plotting
    """
    run_dirs = find_run_dirs(base_dir)
    pooled_rows = []
    # key = (run, ion_id, end_1, end_3) -> {"final_N":..., "pct_dict": {...}}
    events_by_key = {}

    for run_dir in run_dirs:
        csv_path = run_dir / "residence_time_stratified" / "intermediate_occupancy_per_event.csv"
        if not csv_path.exists():
            print(f"  [skip] {run_dir.name}: no intermediate_occupancy_per_event.csv found")
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            n_rows = 0
            for row in reader:
                run = run_dir.name
                ion_id = int(row["ion_id"])
                end_1  = int(row["end_1"])
                end_3  = int(row["end_3"])
                final_N_raw = row["final_N"]
                final_N = int(final_N_raw) if final_N_raw not in ("", "None") else None
                n_inter = int(row["intermediate_N"])
                pct     = float(row["pct_of_transit"])

                pooled_rows.append({
                    "run":              run,
                    "ion_id":           ion_id,
                    "end_1":            end_1,
                    "end_3":            end_3,
                    "final_N":          final_N,
                    "residence_frames": row["residence_frames"],
                    "residence_ns":     row["residence_ns"],
                    "intermediate_N":   n_inter,
                    "frames_at_N":      row["frames_at_N"],
                    "pct_of_transit":   pct,
                })

                key = (run, ion_id, end_1, end_3)
                if key not in events_by_key:
                    events_by_key[key] = {"final_N": final_N, "pct_dict": {}}
                events_by_key[key]["pct_dict"][n_inter] = pct
                n_rows += 1
        print(f"  [ok]   {run_dir.name}: {n_rows} intermediate-occupancy rows")

    pooled_events = [
        {"run": run, "ion_id": ion_id, "end_1": e1, "end_3": e3,
         "final_N": v["final_N"], "pct_dict": v["pct_dict"]}
        for (run, ion_id, e1, e3), v in events_by_key.items()
    ]

    return pooled_rows, pooled_events


# ─────────────────────────────────────────────────────────────────────────────
# Save pooled CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_pooled_residence_csv(pooled, output_dir):
    out = output_dir / "pooled_residence_time_stratified_results.csv"
    fieldnames = [
        "run", "ion_id", "end_1", "end_2", "end_3",
        "residence_frames", "n_ions_at_end3", "ion_ids_at_end3",
        "perm_ion_region_at_end3",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pooled)
    print(f"  Saved: {out}")
    return out


def save_pooled_intermediate_csv(pooled_rows, output_dir):
    out = output_dir / "pooled_intermediate_occupancy_per_event.csv"
    fieldnames = [
        "run", "ion_id", "end_1", "end_3", "final_N",
        "residence_frames", "residence_ns", "intermediate_N",
        "frames_at_N", "pct_of_transit",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pooled_rows)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_aggregate_residence_time_stratified(base_dir, channel_type="",
                                            frame_duration=0.075, force_rerun=False):
    base_dir   = Path(base_dir)
    output_dir = base_dir / "residence_time_stratified_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_marker = output_dir / "pooled_residence_time_stratified_results.csv"
    if out_marker.exists() and not force_rerun:
        print(f"  pooled_residence_time_stratified_results.csv already exists — skipping "
              f"(use force_rerun=True to overwrite).")
        return

    print("\n" + "="*70)
    print(f"AGGREGATING RESIDENCE TIME STRATIFIED RESULTS — {channel_type}")
    print(f"Base directory: {base_dir}")
    print("="*70)

    # ── Main residence-time-at-permeation results ──
    print("\nLoading per-run residence_time_stratified_results.csv...")
    pooled = load_pooled_residence_events(base_dir)
    if not pooled:
        print("  No events found across any RUN — aborting.")
        return
    print(f"\nTotal pooled events: {len(pooled)}")

    save_pooled_residence_csv(pooled, output_dir)
    save_group_summary_csv(pooled, output_dir, frame_duration)
    print_summary(pooled, frame_duration)
    plot_boxplot(pooled, output_dir, channel_type, frame_duration)

    # ── Intermediate occupancy results ──
    print("\nLoading per-run intermediate_occupancy_per_event.csv...")
    pooled_rows, pooled_events = load_pooled_intermediate_events(base_dir)
    if pooled_rows:
        print(f"\nTotal pooled intermediate-occupancy rows: {len(pooled_rows)} "
              f"({len(pooled_events)} events)")
        save_pooled_intermediate_csv(pooled_rows, output_dir)
        grouped = aggregate_intermediate_by_final_group(pooled_events)
        save_intermediate_group_summary_csv(grouped, pooled_events, output_dir)
        plot_intermediate_occupancy_barplots(grouped, output_dir, channel_type)
    else:
        print("  No intermediate-occupancy data found across any RUN — skipping that part.")

    print(f"\n✓ All aggregated outputs saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Pool residence_time_stratified_analysis.py results across all RUN* dirs."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Path to channel results dir containing RUN1, RUN2, ... subfolders")
    parser.add_argument("--channel_type", default="",
                        help="Channel type label for plot titles, e.g. G12")
    parser.add_argument("--frame_duration", type=float, default=0.075,
                        help="ns per frame (default 0.075)")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Overwrite existing aggregated outputs")
    args = parser.parse_args()

    run_aggregate_residence_time_stratified(
        base_dir=args.base_dir,
        channel_type=args.channel_type,
        frame_duration=args.frame_duration,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()

'''
python3 aggregate_residence_time_stratified.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12 \
    --force_rerun

python3 aggregate_residence_time_stratified.py \
    --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2 \
    --force_rerun
'''