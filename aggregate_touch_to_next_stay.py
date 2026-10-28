"""
aggregate_touch_to_next_stay.py
===============================
Aggregates analysis/touch_to_next_stay.py over all runs of one or more
systems: for every permeation event whose first touch after the SF is on the
lower ring (ASP or ASN), the next residue the ion stays on (or 'out'), split
by how many GLUs were occupied by other K+ when it left the touched residue
(results of load_mode='glu'), or by how many K+ were above it in the cavity
(load_mode='above', <settings>_above folders).
Events with end_1 == 0 (ion in the SF in the starting structure) are counted
and kept; --exclude_frame0 leaves them out of the plots and the counts tables.

Every figure is made twice:
    pooled    every event of every run counted once; bars = % of the panel's events (y 0-100 %),
              labels = % (number of events)
    per_run   % in each run, then mean ± SEM over runs, dots = runs (runs with
              no event in the panel are left out of that panel)

Figures (one set per system, 9 bars per panel: the 4 GLU, the 4 ASN/ASP, out):
    fig1_next_stay_overall_<kind>.png        first touch = ASP | first touch = ASN
    fig2_asp_by_glu_occupancy_<kind>.png     first touch = ASP, one panel per number of GLUs
                                             occupied at the leaving frame (1-4; 0 is left out)
    asp_by_glu_occupancy/asp_<k>glu_occupied_<kind>.png   each panel on its own
    fig3_asn_by_glu_occupancy_<kind>.png     the same for first touch = ASN
    asn_by_glu_occupancy/asn_<k>glu_occupied_<kind>.png
  load_mode='above': figures 2-3 split by K+ above the ion in the cavity, one panel per value
  >= 1 that occurs among that figure's events (ASP or ASN; 0 and values with no event get no panel):
    fig2_asp_by_ions_above_<kind>.png, asp_by_ions_above/asp_<k>above_<kind>.png
    fig3_asn_by_ions_above_<kind>.png, asn_by_ions_above/asn_<k>above_<kind>.png
  G2 has no ASP: figure 1 shows only ASN and figure 2 (ASP) is not made.
Tables:
    events.csv          every event of every run (not_lower events included)
    counts_pooled.csv   one row per panel and bar: count, % of the panel
    per_run.csv         one row per run, panel and bar: count, % of the run's events in the panel

Input:  <base>/RUN*/touch_to_next_stay/<settings>/touch_to_next_stay_run*.csv (+ settings_run*.json)
Output: <base>/aggregated_touch_to_next_stay/<settings>/
<settings>: t<cutoff>_w<window>_d<min_duration>, e.g. t3.5_wno_d10 (no first-touch window,
stays >= 10 frames), t3.5_wno_d5, t3.5_w10_d10, t3.5_wno_d10_above

USAGE
-----
    python3 aggregate_touch_to_next_stay.py /path/girk_analyser_results/G12 \\
        /path/girk_analyser_results/G12_E141Q /path/girk_analyser_results/G12_D173N \\
        /path/girk_analyser_results/G2
Each system folder is aggregated on its own, for every <settings> folder found in its runs.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_duration_results import run_sort_key
from analysis.touch_to_next_stay import (NOT_LOWER, CLASSES, bar_labels, pooled_counts, per_run_pct,
                                         write_plots, load_split, split_events, frame0_events,
                                         plotted_classes)


SYSTEM_NAMES = {'G12': 'WT', 'G12_E141Q': 'E141Q', 'G12_D173N': 'D173N', 'G2': 'G2'}
KINDS = ['pooled', 'per_run']


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def settings_folders(base):
    """Every touch_to_next_stay/<settings>/ folder name found in the runs of one system."""
    return sorted({d.name for run_dir in Path(base).glob("RUN*")
                   for d in (run_dir / "touch_to_next_stay").glob("t*_w*_d*") if d.is_dir()})


def load_system(base, subdir):
    """(events DataFrame, settings of the first run) of every RUN*, or (None, None)."""
    frames, settings = [], []
    for run_dir in sorted((d for d in Path(base).iterdir() if d.is_dir() and d.name.startswith('RUN')),
                          key=lambda d: run_sort_key(d.name)):
        folder = run_dir / "touch_to_next_stay" / subdir
        for csv in sorted(folder.glob("touch_to_next_stay_run*.csv")):
            df = pd.read_csv(csv)
            df['run'] = run_dir.name
            frames.append(df)
            s = folder / csv.name.replace("touch_to_next_stay_", "settings_").replace(".csv", ".json")
            if s.exists():
                with open(s) as f:
                    settings.append(json.load(f))
    if not frames:
        return None, None
    keys = ('cutoff', 'window', 'min_duration', 'frame_duration')
    differ = {k for k in keys if len({s.get(k) for s in settings}) > 1}
    if differ:
        print(f"  ⚠ runs differ in {', '.join(sorted(differ))}; titles use the first run's settings")
    if not settings:
        raise SystemExit(f"No settings_run*.json next to the CSVs in {base}/RUN*/touch_to_next_stay/{subdir}")
    return pd.concat(frames, ignore_index=True), settings[0]


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def panels(df, load_mode, glu_name):
    """(class, panel title or 'all', events) of every panel, as in figures 1-3."""
    for cls in CLASSES:
        sub = df[df['first_touch_class'] == cls]
        yield cls, 'all', sub
        column, bins = load_split(load_mode, glu_name, sub)[:2]
        for title, _, test in bins:
            yield cls, title, split_events(sub, column, test)


def write_tables(all_events, df, labels, types, load_mode, glu_name, output_dir):
    """events.csv: every event; counts_pooled.csv / per_run.csv: the plotted events (df)."""
    all_events.assign(_order=all_events['run'].map(lambda r: run_sort_key(r)[0])) \
        .sort_values(['_order', 'end_2']).drop(columns='_order').to_csv(output_dir / "events.csv", index=False)
    pooled, per_run = [], []
    for cls, k, sub in panels(df, load_mode, glu_name):
        for l, c in zip(labels, pooled_counts(sub, labels)):
            pooled.append({'first_touch_class': cls, 'panel': k, 'next_stay': l,
                           'next_stay_type': types[l], 'count': c, 'n_events': len(sub),
                           'pct': 100.0 * c / len(sub) if len(sub) else np.nan})
        pct = per_run_pct(sub, labels)
        n_run = sub.groupby('run').size()
        for run, row in pct.iterrows():
            for l in labels:
                per_run.append({'run': run, 'first_touch_class': cls, 'panel': k, 'next_stay': l,
                                'count': int((sub[sub['run'] == run]['next_stay_residue'] == l).sum()),
                                'n_events_run': int(n_run[run]), 'pct': row[l]})
    pd.DataFrame(pooled).to_csv(output_dir / "counts_pooled.csv", index=False)
    pd.DataFrame(per_run).to_csv(output_dir / "per_run.csv", index=False)
    print(f"✓ Tables saved in {output_dir}")


# ----------------------------------------------------------------------

def aggregate(base, subdir, channel_type=None, output_dir=None, exclude_frame0=False):
    base = Path(base)
    name = channel_type or base.name
    head = SYSTEM_NAMES.get(name, name)
    df, settings = load_system(base, subdir)
    if df is None:
        print(f"\n{base}: no touch_to_next_stay/{subdir} results")
        return
    output_dir = Path(output_dir) if output_dir else base / "aggregated_touch_to_next_stay" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {head} ({base}){' ' + subdir if subdir else ''}: {df['run'].nunique()} runs, "
          f"{len(df)} events -> {output_dir}")
    all_events = df
    df = frame0_events(df, exclude_frame0)
    counts = df['first_touch_class'].value_counts()
    print("  First touch: " + ", ".join(f"{c}: {counts.get(c, 0)}" for c in CLASSES + [NOT_LOWER]))
    labels, types, glu_name = bar_labels(settings)
    load_mode = settings.get('load_mode', 'glu')
    for cls in plotted_classes(settings):
        sub = df[df['first_touch_class'] == cls]
        column, bins = load_split(load_mode, glu_name, sub)[:2]
        print(f"  {cls} by {column}: " + (", ".join(f"{title}: {len(split_events(sub, column, test))}"
                                                    for title, _, test in bins) or "no events"))

    unknown = set(df.loc[df['first_touch_class'].isin(CLASSES), 'next_stay_residue']) - set(labels)
    if unknown:
        print(f"  ⚠ next stays not on the 9 bars (left out of the plots): {sorted(unknown)}")
    write_plots(df, settings, head, output_dir, KINDS, exclude_frame0=None)   # already handled above
    write_tables(all_events, df, labels, types, load_mode, glu_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Aggregate touch_to_next_stay results over runs, per system")
    parser.add_argument("base_directories", nargs='+',
                        help="System folders containing RUN* (e.g. .../G12 .../G12_E141Q .../G12_D173N .../G2)")
    parser.add_argument("--settings", default=None,
                        help="Only this settings folder (e.g. t3.5_wno_d10); default: every one found")
    parser.add_argument("--exclude_frame0", action="store_true",
                        help="Leave events with end_1 == 0 (starting structure) out of the plots and counts")
    parser.add_argument("--channel_type", default=None,
                        help="Name for the titles when only one folder is given (default: folder name)")
    args = parser.parse_args()
    for base in args.base_directories:
        subdirs = settings_folders(base) if args.settings is None else [args.settings]
        if not subdirs:
            print(f"\n{base}: no RUN*/touch_to_next_stay/t*_w*_d*/ results")
        for subdir in subdirs:
            aggregate(base, subdir, args.channel_type if len(args.base_directories) == 1 else None,
                      exclude_frame0=args.exclude_frame0)


if __name__ == "__main__":
    main()


'''
python3 aggregate_touch_to_next_stay.py  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --exclude_frame0
python3 aggregate_touch_to_next_stay.py  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2
python3 aggregate_touch_to_next_stay.py  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q
python3 aggregate_touch_to_next_stay.py  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N
'''
