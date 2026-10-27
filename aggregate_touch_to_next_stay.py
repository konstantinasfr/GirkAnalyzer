"""
aggregate_touch_to_next_stay.py
===============================
Aggregates analysis/touch_to_next_stay.py over all runs of one or more
systems: for every permeation event whose first touch after the SF is on the
lower ring (ASP or ASN), the next residue the ion stays on (or 'out'), split
by how many GLUs were occupied by other K+ when it left the touched residue.

Every figure is made twice:
    pooled    every event of every run counted once; bars = counts, labels = count (% of the panel)
    per_run   % in each run, then mean ± SEM over runs, dots = runs (runs with
              no event in the panel are left out of that panel)

Figures (one set per system, 9 bars per panel: the 4 GLU, the 4 ASN/ASP, out):
    fig1_next_stay_overall_<kind>.png        first touch = ASP | first touch = ASN
    fig2_asp_by_glu_occupancy_<kind>.png     first touch = ASP, one panel per number of GLUs
                                             occupied at the leaving frame (0-4)
    asp_by_glu_occupancy/asp_<k>glu_occupied_<kind>.png   each panel on its own
    fig3_asn_by_glu_occupancy_<kind>.png     the same for first touch = ASN
    asn_by_glu_occupancy/asn_<k>glu_occupied_<kind>.png
Tables:
    events.csv          every event of every run (not_lower events included)
    counts_pooled.csv   one row per panel and bar: count, % of the panel
    per_run.csv         one row per run, panel and bar: count, % of the run's events in the panel

Input:  <base>/RUN*/touch_to_next_stay/<settings>/touch_to_next_stay_run*.csv (+ settings_run*.json)
Output: <base>/aggregated_touch_to_next_stay/<settings>/
<settings>: t<cutoff>_w<window>_d<min_duration>, e.g. t3.5_wno_d10 (no first-touch window,
stays >= 10 frames), t3.5_wno_d5, t3.5_w10_d10

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from aggregate_duration_results import COLOR_MAP, TYPE_ORDER, NO_CONTACT_COLOR, ONE_LETTER, run_sort_key
from analysis.touch_to_next_stay import OUT, NOT_LOWER


SYSTEM_NAMES = {'G12': 'WT', 'G12_E141Q': 'E141Q', 'G12_D173N': 'D173N', 'G2': 'G2'}
CLASSES = ['ASP', 'ASN']
N_GLU = range(5)                       # 0-4 GLUs occupied
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


def bar_labels(settings):
    """The 9 bars: GLU ring, then the ASN/ASP ring (ASN before ASP), then out; and their types."""
    ring = settings['ring_residues']
    glu = sorted((r for r in ring if r['ring'] == 'GLU'), key=lambda r: r['pdb'])
    low = sorted((r for r in ring if r['ring'] == 'ASN'),
                 key=lambda r: (TYPE_ORDER.index(r['residue_type']) if r['residue_type'] in TYPE_ORDER
                                else len(TYPE_ORDER), r['pdb']))
    labels = [r['pdb'] for r in glu + low] + [OUT]
    types = {r['pdb']: r['residue_type'] for r in glu + low}
    types[OUT] = OUT
    glu_name = '/'.join(sorted({r['residue_type'] for r in glu}))
    return labels, types, glu_name


def rule_text(s):
    fd = s['frame_duration']
    touch = ("first touch anywhere in the cavity (end_1 → end_2)" if s['window'] is None
             else f"first-touch window {s['window']} fr ({s['window'] * fd:g} ns)")
    return f"cutoff {s['cutoff']} Å · {touch} · stay ≥ {s['min_duration']} fr ({s['min_duration'] * fd:g} ns)"


# ----------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------

def pooled_counts(sub, labels):
    return np.array([int((sub['next_stay_residue'] == l).sum()) for l in labels])


def per_run_pct(sub, labels):
    """DataFrame run x label of % of the run's events in this panel (runs with none left out)."""
    if sub.empty:
        return pd.DataFrame(columns=labels)
    counts = sub.groupby(['run', 'next_stay_residue']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=labels, fill_value=0)
    counts = counts.loc[sorted(counts.index, key=run_sort_key)]
    return 100.0 * counts.div(counts.sum(axis=1), axis=0)


# ----------------------------------------------------------------------
# Plots (style of create_bar_plot / aggregate_duration_results.py)
# ----------------------------------------------------------------------

def _style(ax):
    ax.grid(True, alpha=0.25, axis='y', zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.1)


def _tick(label, types):
    return label if label == OUT else f"{ONE_LETTER.get(types[label], '')}{label}"


def _color(label, types):
    return NO_CONTACT_COLOR if label == OUT else COLOR_MAP.get(types[label], COLOR_MAP['Unknown'])


def legend_handles(labels, types):
    present = {types[l] for l in labels if l != OUT}
    handles = [Patch(facecolor=COLOR_MAP[t], edgecolor='black', label=t) for t in TYPE_ORDER if t in present]
    handles.append(Patch(facecolor=NO_CONTACT_COLOR, edgecolor='black', hatch='//', label='out (no stay)'))
    return handles


def draw_panel(ax, sub, kind, labels, types, title, small=False):
    """One panel: the 9 next-stay bars of the events in `sub`."""
    fs = {'text': 10 if small else 13, 'tick': 12 if small else 16, 'axis': 13 if small else 18,
          'title': 13 if small else 16}
    x = np.arange(len(labels))
    n_events, n_runs = len(sub), sub['run'].nunique()
    colors = [_color(l, types) for l in labels]

    if kind == 'pooled':
        counts = pooled_counts(sub, labels)
        bars = ax.bar(x, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2,
                      width=0.65, zorder=3)
        top = max(int(counts.max()), 1)
        ax.set_ylim(0, top * 1.35)
        for bar, c in zip(bars, counts):
            text = f"{c}\n({100.0 * c / n_events:.0f}%)" if n_events else "0"
            ax.text(bar.get_x() + bar.get_width() / 2., c + top * 0.03, text, ha='center', va='bottom',
                    fontsize=fs['text'], fontweight='bold', zorder=4)
        ax.set_ylabel('Number of events', fontsize=fs['axis'], fontweight='bold')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        n_text = f"n = {n_events} events, {n_runs} runs"
    else:
        pct = per_run_pct(sub, labels)
        means = pct.mean(axis=0).reindex(labels).fillna(0).values
        sems = (pct.std(axis=0, ddof=1) / np.sqrt(len(pct))).reindex(labels).fillna(0).values \
            if len(pct) > 1 else np.zeros(len(labels))
        bars = ax.bar(x, means, yerr=sems, capsize=5 if small else 6, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.2, width=0.65, zorder=3,
                      error_kw={'elinewidth': 1.4, 'zorder': 4})
        rng = np.random.default_rng(0)
        for i, l in enumerate(labels):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(pct)), pct[l].values, s=18 if small else 26,
                       color='black', alpha=0.65, zorder=5, linewidths=0)
        top = max(float(pct.values.max()) if len(pct) else 0.0, 1.0)
        ax.set_ylim(0, top * 1.25)
        for bar, m, e in zip(bars, means, sems):
            if len(pct):
                ax.text(bar.get_x() + bar.get_width() / 2., top * 1.22, f"{m:.0f}±{e:.0f}",
                        ha='center', va='top', fontsize=fs['text'] - 1, fontweight='bold', zorder=6)
        ax.set_ylabel('% of events in a run', fontsize=fs['axis'], fontweight='bold')
        n_text = f"n = {n_events} events, {len(pct)} runs"
    bars[-1].set_hatch('//')

    if n_events == 0:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes, ha='center', va='center',
                fontsize=fs['title'], fontweight='bold', color='0.4')
    ax.set_xticks(x)
    ax.set_xticklabels([_tick(l, types) for l in labels], fontsize=fs['tick'], fontweight='bold',
                       rotation=40, ha='right')
    ax.set_xlabel('Next stay', fontsize=fs['axis'], fontweight='bold')
    ax.tick_params(axis='y', labelsize=fs['tick'] - 2)
    ax.set_title(f"{title}\n({n_text})", fontsize=fs['title'], fontweight='bold', pad=12)
    _style(ax)


def kind_text(kind):
    return ("pooled: every event counted once" if kind == 'pooled'
            else "per run: mean ± SEM of % per run, dots = runs")


def _save(fig, output_file):
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {output_file}")


def figure_overall(df, kind, labels, types, head, rule, output_file):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5))
    for ax, cls in zip(axes, CLASSES):
        draw_panel(ax, df[df['first_touch_class'] == cls], kind, labels, types, f"First touch = {cls}")
    axes[-1].legend(handles=legend_handles(labels, types), loc='upper left', bbox_to_anchor=(1.01, 1.0),
                    fontsize=13, frameon=True, framealpha=0.9)
    fig.suptitle(f"{head}: next stay after the first touch on the lower ring ({kind_text(kind)})\n{rule}",
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_file)


def figure_by_occupancy(df, cls, kind, labels, types, glu_name, head, rule, output_file, single_dir):
    sub_cls = df[df['first_touch_class'] == cls]
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    for k, ax in zip(N_GLU, axes.flat):
        draw_panel(ax, sub_cls[sub_cls['n_glu_occupied'] == k], kind, labels, types,
                   f"{k} {glu_name} occupied", small=True)
    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_handles(labels, types), loc='center', fontsize=14, frameon=True,
                     title=f"First touch = {cls}\nn = {len(sub_cls)} events\n\nNext stay",
                     title_fontsize=14)
    fig.suptitle(f"{head}: next stay after first touch = {cls}, by {glu_name} occupied "
                 f"when the ion leaves {cls} ({kind_text(kind)})\n{rule}",
                 fontsize=17, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_file)

    single_dir.mkdir(parents=True, exist_ok=True)
    for k in N_GLU:
        fig, ax = plt.subplots(figsize=(11, 7))
        draw_panel(ax, sub_cls[sub_cls['n_glu_occupied'] == k], kind, labels, types,
                   f"{head}: first touch = {cls}, {k} {glu_name} occupied at leaving\n"
                   f"{rule}\n{kind_text(kind)}")
        ax.legend(handles=legend_handles(labels, types), loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=12, frameon=True, framealpha=0.9)
        plt.tight_layout()
        _save(fig, single_dir / f"{cls.lower()}_{k}glu_occupied_{kind}.png")


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def panels(df):
    """(class, n_glu or 'all', events) of every panel."""
    for cls in CLASSES:
        sub = df[df['first_touch_class'] == cls]
        yield cls, 'all', sub
        for k in N_GLU:
            yield cls, k, sub[sub['n_glu_occupied'] == k]


def write_tables(df, labels, types, output_dir):
    df.assign(_order=df['run'].map(lambda r: run_sort_key(r)[0])).sort_values(['_order', 'end_2']) \
      .drop(columns='_order').to_csv(output_dir / "events.csv", index=False)
    pooled, per_run = [], []
    for cls, k, sub in panels(df):
        for l, c in zip(labels, pooled_counts(sub, labels)):
            pooled.append({'first_touch_class': cls, 'n_glu_occupied': k, 'next_stay': l,
                           'next_stay_type': types[l], 'count': c, 'n_events': len(sub),
                           'pct': 100.0 * c / len(sub) if len(sub) else np.nan})
        pct = per_run_pct(sub, labels)
        n_run = sub.groupby('run').size()
        for run, row in pct.iterrows():
            for l in labels:
                per_run.append({'run': run, 'first_touch_class': cls, 'n_glu_occupied': k, 'next_stay': l,
                                'count': int((sub[sub['run'] == run]['next_stay_residue'] == l).sum()),
                                'n_events_run': int(n_run[run]), 'pct': row[l]})
    pd.DataFrame(pooled).to_csv(output_dir / "counts_pooled.csv", index=False)
    pd.DataFrame(per_run).to_csv(output_dir / "per_run.csv", index=False)
    print(f"✓ Tables saved in {output_dir}")


# ----------------------------------------------------------------------

def aggregate(base, subdir, channel_type=None, output_dir=None):
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
    counts = df['first_touch_class'].value_counts()
    print("  First touch: " + ", ".join(f"{c}: {counts.get(c, 0)}" for c in CLASSES + [NOT_LOWER]))
    for cls in CLASSES:
        sub = df[df['first_touch_class'] == cls]
        occ = sub['n_glu_occupied'].value_counts()
        print(f"  {cls} by GLUs occupied: " + ", ".join(f"{k}: {occ.get(k, 0)}" for k in N_GLU))

    labels, types, glu_name = bar_labels(settings)
    unknown = set(df.loc[df['first_touch_class'].isin(CLASSES), 'next_stay_residue']) - set(labels)
    if unknown:
        print(f"  ⚠ next stays not on the 9 bars (left out of the plots): {sorted(unknown)}")
    rule = rule_text(settings)
    for kind in KINDS:
        figure_overall(df, kind, labels, types, head, rule, output_dir / f"fig1_next_stay_overall_{kind}.png")
        for fig_no, cls in ((2, 'ASP'), (3, 'ASN')):
            figure_by_occupancy(df, cls, kind, labels, types, glu_name, head, rule,
                                output_dir / f"fig{fig_no}_{cls.lower()}_by_glu_occupancy_{kind}.png",
                                output_dir / f"{cls.lower()}_by_glu_occupancy")
    write_tables(df, labels, types, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Aggregate touch_to_next_stay results over runs, per system")
    parser.add_argument("base_directories", nargs='+',
                        help="System folders containing RUN* (e.g. .../G12 .../G12_E141Q .../G12_D173N .../G2)")
    parser.add_argument("--settings", default=None,
                        help="Only this settings folder (e.g. t3.5_wno_d10); default: every one found")
    parser.add_argument("--channel_type", default=None,
                        help="Name for the titles when only one folder is given (default: folder name)")
    args = parser.parse_args()
    for base in args.base_directories:
        subdirs = settings_folders(base) if args.settings is None else [args.settings]
        if not subdirs:
            print(f"\n{base}: no RUN*/touch_to_next_stay/t*_w*_d*/ results")
        for subdir in subdirs:
            aggregate(base, subdir, args.channel_type if len(args.base_directories) == 1 else None)


if __name__ == "__main__":
    main()
