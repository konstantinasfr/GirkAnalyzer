"""
aggregate_state_chain.py
========================
Pools analysis/state_chain.py over all runs of one or more systems: the frames
CSVs of every RUN* are joined and the frame-by-frame state chain is analysed on
the pooled events (time per region, states, exit rates by K+ above x below with
the control, Markov tests, chain against real events, plots).

Input:  <base>/RUN*/state_chain/<settings>/frames_run*.csv and events_run*.csv
Output: <base>/aggregated_state_chain/<settings>/<analysis>/
        analysis = <level>_<region_rule>_h<min_hold>[_sf], e.g. region_last_h4
        plus frames_all.csv / events_all.csv in <settings>/
<settings>: t<cutoff>_d<min_duration>, e.g. t3.5_d10

USAGE
-----
    python3 aggregate_state_chain.py /path/girk_analyser_results/G12 \\
        [/path/girk_analyser_results/G12_E141Q ...] [--level region] [--region_rule last]
        [--max_above 3] [--max_below 2] [--min_hold 4] [--include_sf] [--exclude_frame0]
Each system folder is aggregated on its own, for every <settings> folder found in its runs.
"""

import argparse
from pathlib import Path

import pandas as pd

from aggregate_duration_results import run_sort_key
from analysis.state_chain import (OUT_FOLDER, add_analysis_args, analysis_folder, analyze_states,
                                  default_min_hold)

FRAME_DURATION = {"G2": 0.4}                 # ns per frame; everything else 0.075


def settings_folders(base):
    return sorted({d.name for run_dir in Path(base).glob("RUN*")
                   for d in (run_dir / OUT_FOLDER).glob("t*_d*") if d.is_dir()})


def load(base, settings):
    frames, events = [], []
    for run_dir in sorted(Path(base).glob("RUN*"), key=lambda p: run_sort_key(p.name)):
        d = run_dir / OUT_FOLDER / settings
        for f_csv in d.glob("frames_run*.csv"):
            e_csv = d / f_csv.name.replace("frames_", "events_")
            if not e_csv.exists():
                print(f"  [SKIP] {f_csv} has no {e_csv.name}")
                continue
            frames.append(pd.read_csv(f_csv, keep_default_na=False))
            events.append(pd.read_csv(e_csv))
    if not events:
        return None, None
    return pd.concat(frames, ignore_index=True), pd.concat(events, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="+", help="system folders holding RUN*/")
    add_analysis_args(p)
    p.add_argument("--exclude_frame0", action="store_true",
                   help="leave out events whose ion was in the SF in the starting structure (end_1 == 0)")
    args = p.parse_args()

    for base in args.bases:
        base = Path(base)
        system = base.name
        dt = FRAME_DURATION.get(system.upper(), 0.075)
        min_hold = default_min_hold(system) if args.min_hold is None else args.min_hold
        for settings in settings_folders(base):
            frames, events = load(base, settings)
            if events is None:
                continue
            if args.exclude_frame0:
                keep = events["end_1"] != 0
                print(f"{system} {settings}: leaving out {int((~keep).sum())} events with end_1 == 0")
                events = events[keep]
                kept = set(zip(events["run"], events["event"]))
                frames = frames[[(r, e) in kept for r, e in zip(frames["run"], frames["event"])]]
            out = base / f"aggregated_{OUT_FOLDER}" / settings
            out.mkdir(parents=True, exist_ok=True)
            frames.to_csv(out / "frames_all.csv", index=False)
            events.to_csv(out / "events_all.csv", index=False)
            analyze_states(frames, events,
                           out / analysis_folder(args.level, args.region_rule, min_hold, args.include_sf),
                           f"{system} ({events['run'].nunique()} runs)", dt, level=args.level,
                           region_rule=args.region_rule, max_above=args.max_above,
                           max_below=args.max_below, min_hold=min_hold, include_sf=args.include_sf)


if __name__ == "__main__":
    main()
