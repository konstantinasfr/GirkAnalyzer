"""
collect_duration_results.py
===========================
Collects the contact analyses of analysis/longest_duration_analysis.py from
ALL channels into ONE Excel file, for comparing channels:

    last     LAST residue/subunit contacted before leaving the GLU/ASN cavity
    longest  LONGEST contact before leaving the cavity
    first    FIRST residue/subunit contacted after leaving the SF
    touch    FIRST residue/subunit within the threshold in the first --window
             frames after leaving the SF (run_first_touch_analysis)

each in residue and subunit mode, for every run group:

    ALL                  every run
    HIGH/TYPICAL/LOW     permeation categories (permeation_summary/run_categories_*.csv)
    D/E                  ligand categories, G12_ML only (ligand_inter_class/ligand_categories.csv)

It reads the per-run JSONs directly (same loading as aggregate_duration_results.py),
so the aggregator does not need to be run first.

Sheets of the output file:
    README        what every column means, and the parameters used
    summary       one row per channel x contact x mode x group x residue/subunit:
                  per-run mean % +/- SEM, median, min, max, pooled count and %,
                  mean/median contact duration
    ring_summary  the same, with residues grouped into the upper ring (152/141),
                  the lower ring (184/173) and No contact  (residue mode only)
    per_run       one row per channel x run x contact x mode x residue/subunit:
                  count and % of that run's events
    events        one row per event (all channels)
    coverage      runs found per channel, runs without a category, and residue
                  labels outside the expected positions

USAGE
-----
    python3 collect_duration_results.py /path/to/girk_analyser_results --threshold 3.5 --min_duration 5

Channels are all sub-folders containing RUN* folders, or pass --channels G12 G12_ML ...
Output: <results_root>/duration_comparison_t3.5_d5.xlsx (or --output).
"""

import io
import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_duration_results import (DurationResultsAggregator, load_categories,
                                        load_ligand_categories, run_sort_key,
                                        NO_CONTACT, LIGAND_CHANNELS)


# Contact kinds: (analysis passed to the aggregator, side inside it)
CONTACTS = [('last', 'last'), ('last', 'longest'), ('first', 'first'), ('touch', 'touch')]
MODES = ['residue', 'subunit']
RINGS = {'152': 'upper ring (152/141)', '141': 'upper ring (152/141)',
         '184': 'lower ring (184/173)', '173': 'lower ring (184/173)'}
RING_ORDER = ['upper ring (152/141)', 'lower ring (184/173)', NO_CONTACT]


def ring_of(label):
    if label == NO_CONTACT:
        return NO_CONTACT
    return RINGS.get(str(label).split('.')[0], 'other')


def find_channels(root):
    return sorted(d.name for d in Path(root).iterdir()
                  if d.is_dir() and any(c.is_dir() and c.name.startswith('RUN') for c in d.iterdir()))


def load_channel(channel_dir, threshold, min_duration, window=10):
    """Events of one channel (DataFrame) + run categories + coverage notes."""
    channel = channel_dir.name
    rows, coverage = [], []
    runs_found = {}

    for analysis, side in CONTACTS:
        agg = DurationResultsAggregator(channel_dir, threshold, min_duration, channel, analysis, window)
        with contextlib.redirect_stdout(io.StringIO()):
            agg.load_all_runs()
        for mode in MODES:
            runs_found[(side, mode)] = agg.runs_loaded[mode]
            for e in agg.events[mode][side]:
                rows.append({'channel': channel, 'contact': side, 'mode': mode, **e})

    events = pd.DataFrame(rows)

    # Run categories
    categories, ligand, ligand_left_out = {}, {}, {}
    cat_file = channel_dir / "permeation_summary" / "run_categories_full_permeations_total.csv"
    if cat_file.exists():
        categories = load_categories(cat_file)
    if channel.upper() in LIGAND_CHANNELS:
        lig_file = channel_dir / "ligand_inter_class" / "ligand_categories.csv"
        if lig_file.exists():
            ligand, ligand_left_out = load_ligand_categories(lig_file)

    all_runs = sorted({r for runs in runs_found.values() for r in runs}, key=run_sort_key)
    for (side, mode), runs in runs_found.items():
        coverage.append({'channel': channel, 'what': f'runs with {side}/{mode} results',
                         'n': len(runs), 'runs': ', '.join(sorted(runs, key=run_sort_key))})
    coverage.append({'channel': channel, 'what': 'permeation categories file',
                     'n': len(categories), 'runs': str(cat_file) if categories else 'not found'})
    no_cat = [r for r in all_runs if r not in categories]
    if categories and no_cat:
        coverage.append({'channel': channel, 'what': 'runs without a permeation category (only in ALL)',
                         'n': len(no_cat), 'runs': ', '.join(no_cat)})
    if channel.upper() in LIGAND_CHANNELS:
        coverage.append({'channel': channel, 'what': 'ligand categories (D/E)', 'n': len(ligand),
                         'runs': ', '.join(f"{r}:{c}" for r, c in sorted(ligand.items(), key=lambda x: run_sort_key(x[0])))
                         or 'not found'})
        if ligand_left_out:
            coverage.append({'channel': channel, 'what': 'runs left out of D/E', 'n': len(ligand_left_out),
                             'runs': ', '.join(f"{r} ({c or 'no category'})" for r, c in ligand_left_out.items())})

    # Runs with results but no permeation events drop out of the per-run averages
    with_events = set(events['run']) if not events.empty else set()
    no_events = [r for r in all_runs if r not in with_events]
    if no_events:
        coverage.append({'channel': channel, 'what': 'runs with 0 permeation events (not in per-run averages)',
                         'n': len(no_events), 'runs': ', '.join(no_events)})

    if not events.empty:
        events.insert(1, 'category', events['run'].map(lambda r: categories.get(r, '')))
        events.insert(2, 'ligand', events['run'].map(lambda r: ligand.get(r, '')))
        res = events[events['mode'] == 'residue']
        odd = sorted(set(l for l in res['label'] if ring_of(l) == 'other'))
        if odd:
            bad_runs = sorted(set(res.loc[res['label'].isin(odd), 'run']), key=run_sort_key)
            coverage.append({'channel': channel, 'what': 'UNEXPECTED residue labels (check numbering)',
                             'n': len(odd), 'runs': f"labels {', '.join(odd)} in {', '.join(bad_runs)}"})
    return events, coverage


def groups_of(events):
    """[(group name, boolean mask)] for one channel's events."""
    out = [('ALL', np.ones(len(events), bool))]
    for cat in ['HIGH', 'TYPICAL', 'LOW']:
        m = (events['category'] == cat).values
        if m.any():
            out.append((cat, m))
    for cat in ['D', 'E']:
        m = (events['ligand'] == cat).values
        if m.any():
            out.append((f'ligand {cat}', m))
    return out


def summarise(df, label_col):
    """Per-run and pooled statistics of one channel x contact x mode x group."""
    counts = df.groupby(['run', label_col]).size().unstack(fill_value=0)
    pct = 100.0 * counts.div(counts.sum(axis=1), axis=0)
    rows = []
    for label in counts.columns:
        sel = df[df[label_col] == label]
        v = pct[label]
        rows.append({
            'label': label,
            'n_runs': len(v),
            'mean_pct_per_run': v.mean(),
            'sem_pct_per_run': v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
            'median_pct_per_run': v.median(),
            'min_pct_run': v.min(),
            'max_pct_run': v.max(),
            'n_runs_with_label': int((counts[label] > 0).sum()),
            'pooled_count': int(counts[label].sum()),
            'pooled_pct': 100.0 * counts[label].sum() / counts.values.sum(),
            'mean_duration_fr': sel['duration'].mean() if label != NO_CONTACT else np.nan,
            'median_duration_fr': sel['duration'].median() if label != NO_CONTACT else np.nan,
            'mean_gap_fr': sel['gap'].mean() if label != NO_CONTACT else np.nan,
        })
    return rows, counts, pct


def main():
    parser = argparse.ArgumentParser(description="Collect last/longest/first contact results of all channels "
                                                 "into one Excel file")
    parser.add_argument("results_root", help="Folder containing one folder per channel (G12, G12_ML, ...)")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold used (e.g. 3.5)")
    parser.add_argument("--min_duration", type=int, required=True, help="min_duration used (e.g. 5)")
    parser.add_argument("--window", type=int, default=10,
                        help="Window of the first-touch analysis (frames after end_1, default 10)")
    parser.add_argument("--channels", nargs='+', default=None,
                        help="Channels to include (default: every sub-folder with RUN* folders)")
    parser.add_argument("--output", default=None,
                        help="Output .xlsx (default: <results_root>/duration_comparison_t*_d*.xlsx)")
    args = parser.parse_args()

    root = Path(args.results_root)
    channels = args.channels or find_channels(root)
    t = int(args.threshold) if float(args.threshold).is_integer() else args.threshold
    tag = f"t{t}_d{args.min_duration}"
    output = Path(args.output) if args.output else root / f"duration_comparison_{tag}.xlsx"

    all_events, coverage = [], []
    for ch in channels:
        ev, cov = load_channel(root / ch, args.threshold, args.min_duration, args.window)
        coverage += cov
        if ev.empty:
            print(f"  ✗ {ch}: no contact results for threshold={args.threshold}, min_duration={args.min_duration}")
            continue
        all_events.append(ev)
        n = {c: ev.loc[ev['contact'] == c, 'run'].nunique() for c in ['last', 'first', 'touch']}
        print(f"  ✓ {ch}: last/longest from {n['last']} runs, first from {n['first']} runs, "
              f"touch from {n['touch']} runs")
    if not all_events:
        print("No results found. Exiting.")
        return
    events = pd.concat(all_events, ignore_index=True)
    events['ring'] = [ring_of(l) if m == 'residue' else '' for l, m in zip(events['label'], events['mode'])]

    summary, ring_summary, per_run = [], [], []
    for (ch, contact, mode), df in events.groupby(['channel', 'contact', 'mode'], sort=False):
        for group, mask in groups_of(df):
            g = df[mask]
            key = {'channel': ch, 'contact': contact, 'mode': mode, 'group': group,
                   'n_events': len(g)}
            rows, counts, pct = summarise(g, 'label')
            types = g.dropna(subset=['type']).groupby('label')['type'].first().to_dict()
            for r in rows:
                summary.append({**key, 'ring': ring_of(r['label']) if mode == 'residue' else '',
                                'type': types.get(r['label'], ''), **r})
            if group == 'ALL':
                for run in counts.index:
                    for label in counts.columns:
                        per_run.append({'channel': ch, 'run': run,
                                        'category': g.loc[g['run'] == run, 'category'].iloc[0],
                                        'ligand': g.loc[g['run'] == run, 'ligand'].iloc[0],
                                        'contact': contact, 'mode': mode,
                                        'n_events_run': int(counts.loc[run].sum()), 'label': label,
                                        'count': int(counts.loc[run, label]), 'pct': pct.loc[run, label]})
            if mode == 'residue':
                rows, _, _ = summarise(g, 'ring')
                for r in rows:
                    ring_summary.append({**key, **r})

    summary = pd.DataFrame(summary)
    ring_summary = pd.DataFrame(ring_summary)
    ring_summary['_o'] = ring_summary['label'].map(lambda l: RING_ORDER.index(l) if l in RING_ORDER else 9)
    ring_summary = ring_summary.sort_values(['channel', 'contact', 'mode', 'group', '_o']).drop(columns='_o')
    per_run = pd.DataFrame(per_run)

    readme = pd.DataFrame({'item': [
        'threshold (A)', 'min_duration (frames)', 'channels', 'contact = last',
        'contact = longest', 'contact = first', 'contact = touch', 'mode', 'group', 'label',
        'mean_pct_per_run / sem / median / min / max',
        'n_runs_with_label', 'pooled_count / pooled_pct', 'mean/median_duration_fr',
        'mean_gap_fr', 'ring', 'No contact', 'durations',
    ], 'meaning': [
        args.threshold, f"{args.min_duration} (touch: none, window = {args.window} frames)", ', '.join(channels),
        'LAST residue/subunit the ion contacted before leaving the GLU/ASN cavity (searched end_2 -> end_1)',
        'LONGEST contact before leaving the cavity (end_1 -> end_2)',
        'FIRST residue/subunit contacted after leaving the SF (end_1 -> end_2, same window as last)',
        'FIRST residue/subunit within the threshold in the first <window> frames after end_1 '
        '(any duration; duration = frames in a row in the window)',
        'residue = single residues; subunit = chain (its GLU/GLN + ASN/ASP together)',
        'ALL runs; HIGH/TYPICAL/LOW = permeation categories; ligand D/E = G12_ML ligand categories',
        'residue position (e.g. 173.G1) or subunit (G2.a, G1); compare positions, not names, across mutants',
        '% of a run\'s events with this label, computed per run, then over the runs of the group '
        '(every run weighs the same)',
        'number of runs in which this label occurred at least once',
        'all events of the group pooled (runs with more events weigh more)',
        'contact length in frames (consecutive frames closest and within the threshold)',
        'last/longest: frames from the end of the contact to end_2; first: frames from end_1 to the contact',
        'upper ring = 152/141 (GLU/GLN), lower ring = 184/173 (ASN/ASP)',
        'the ion never stayed within the threshold for min_duration frames',
        'in frames; convert with each channel\'s time per frame before comparing channels in ns',
    ]})

    events_out = events[['channel', 'run', 'category', 'ligand', 'contact', 'mode', 'ion_id',
                         'end_1_frame', 'end_2_frame', 'label', 'type', 'ring', 'duration',
                         'start_frame', 'end_frame', 'gap']]
    with pd.ExcelWriter(output) as xl:
        readme.to_excel(xl, sheet_name='README', index=False)
        summary.to_excel(xl, sheet_name='summary', index=False, float_format='%.3f')
        ring_summary.to_excel(xl, sheet_name='ring_summary', index=False, float_format='%.3f')
        per_run.to_excel(xl, sheet_name='per_run', index=False, float_format='%.3f')
        events_out.to_excel(xl, sheet_name='events', index=False)
        pd.DataFrame(coverage).to_excel(xl, sheet_name='coverage', index=False)

    odd = [c for c in coverage if c['what'].startswith('UNEXPECTED')]
    for c in odd:
        print(f"  ⚠ {c['channel']}: {c['runs']}")
    print(f"\n✓ {len(events_out)} events from {events['channel'].nunique()} channels -> {output}")


if __name__ == "__main__":
    main()


# USAGE EXAMPLE:
'''
 python3 collect_duration_results.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results \
     --threshold 3.5 --min_duration 5
'''
