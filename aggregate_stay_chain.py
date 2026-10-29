"""
aggregate_stay_chain.py
=======================
Pools analysis/stay_chain.py over all runs of one or more systems: the stays
CSVs of every RUN* are joined and the stay-to-stay chain is analysed on the
pooled events (tables, Markov tests, chain against real events, plots), with
the per-run mean +/- SEM of P(out next) in the summary.

Input:  <base>/RUN*/stay_chain/<settings>/stays_run*.csv and events_run*.csv
Output: <base>/aggregated_stay_chain/<settings>/<level>/   (level = ring, residue)
        plus stays_all.csv / events_all.csv in <settings>/
<settings>: t<cutoff>_d<min_duration>, e.g. t3.5_d10

USAGE
-----
    python3 aggregate_stay_chain.py /path/girk_analyser_results/G12 \\
        [/path/girk_analyser_results/G12_E141Q ...] [--max_bin 4] [--exclude_frame0]
Each system folder is aggregated on its own, for every <settings> folder found in its runs.
"""

import argparse
from pathlib import Path

import pandas as pd

from aggregate_duration_results import run_sort_key
from analysis.stay_chain import LEVELS, OUT_FOLDER, analyze_chain

FRAME_DURATION = {"G2": 0.4}                 # ns per frame; everything else 0.075


def settings_folders(base):
    return sorted({d.name for run_dir in Path(base).glob("RUN*")
                   for d in (run_dir / OUT_FOLDER).glob("t*_d*") if d.is_dir()})


def load(base, settings):
    stays, events = [], []
    for run_dir in sorted(Path(base).glob("RUN*"), key=lambda p: run_sort_key(p.name)):
        d = run_dir / OUT_FOLDER / settings
        for s_csv in d.glob("stays_run*.csv"):
            e_csv = d / s_csv.name.replace("stays_", "events_")
            if not e_csv.exists():
                print(f"  [SKIP] {s_csv} has no {e_csv.name}")
                continue
            stays.append(pd.read_csv(s_csv))
            events.append(pd.read_csv(e_csv))
    if not events:
        return None, None
    return (pd.concat(stays, ignore_index=True) if stays else pd.DataFrame(),
            pd.concat(events, ignore_index=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="+", help="system folders holding RUN*/")
    p.add_argument("--max_bin", type=int, default=4)
    p.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS)
    p.add_argument("--exclude_frame0", action="store_true",
                   help="leave out events whose ion was in the SF in the starting structure (end_1 == 0)")
    args = p.parse_args()

    for base in args.bases:
        base = Path(base)
        system = base.name
        dt = FRAME_DURATION.get(system.upper(), 0.075)
        for settings in settings_folders(base):
            stays, events = load(base, settings)
            if events is None:
                continue
            if args.exclude_frame0:
                keep = events["end_1"] != 0
                print(f"{system} {settings}: leaving out {int((~keep).sum())} events with end_1 == 0")
                drop = set(zip(events.loc[~keep, "run"], events.loc[~keep, "event"]))
                events = events[keep]
                if len(stays):
                    stays = stays[[(r, e) not in drop for r, e in zip(stays["run"], stays["event"])]]
            out = base / f"aggregated_{OUT_FOLDER}" / settings
            out.mkdir(parents=True, exist_ok=True)
            stays.to_csv(out / "stays_all.csv", index=False)
            events.to_csv(out / "events_all.csv", index=False)
            n_runs = events["run"].nunique()
            for level in args.levels:
                analyze_chain(stays, events, out / level, f"{system} ({n_runs} runs)", dt,
                              level=level, max_bin=args.max_bin)


if __name__ == "__main__":
    main()
