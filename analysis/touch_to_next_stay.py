"""
touch_to_next_stay.py
=====================
Where does a permeating ion stay next after it first touches the lower ring
(ASN/ASP), and does that depend on how many glutamates are already occupied
by other K+ when it leaves the touched residue?

Everything except the leaving frame is computed by existing code:

    first touch   FirstTouchAnalysis (longest_duration_analysis.py), residue
                  mode, threshold = cutoff, first `window` frames after end_1
                  (no_window=True: anywhere from end_1 to end_2, the same search
                  as glu_occupancy_at_exit.py with window=None)
    stays         LastAndLongestDurationAnalysis.find_last_and_longest_duration
                  (ion_id, end_2, end_1)['all_durations']: every stay of
                  >= min_duration frames between end_1 and end_2, the same
                  segmentation as the first-stay and last-contact plots
    occupancy     GluOccupancyAtExitAnalysis.occupancy (glu_occupancy_at_exit.py):
                  a GLU is occupied when another K+ (not the permeating ion)
                  is <= cutoff from any atom of the residue

Per permeation event:
    1. first touch on a residue of asn_residues -> class = its topology
       name (ASP or ASN); a GLU first touch or no contact -> 'not_lower'
       (kept in the CSV, left out of the plots)
    2. leave_frame: from touch_frame, the ion is followed forward (never past
       end_2) while the touched residue stays the closest residue within the
       cutoff (_closest_from_row); the last such frame
    3. load at leave_frame (load_mode picks which one splits figures 2-3; all are stored):
         n_glu_occupied  GLUs (0-4) with another K+ <= cutoff        (load_mode='glu', default)
         n_above         other K+ inside region 2 (the cavity, main.py's ch2 cylinder)
                         further toward the SF than the permeating ion along the
                         ch2 axis                                    (load_mode='above')
         n_below         other K+ inside region 2 further from the SF than the ion
         n_sf            other K+ inside region 1 (the SF, main.py's ch1 cylinder)
       Regions: Channel.compute_geometry + is_within_cylinder every frame, as in
       ion_state_sequence.py. "Toward the SF" is the sign of the ch2 axis that
       points from the cavity centre to the SF centre, found every frame.
    4. next stay: the earliest stay with end_frame >= touch_frame (may be the
       touched residue itself), or 'out' if there is none

Events with end_1 == 0 (the ion was in the SF in the starting structure) are
flagged starts_at_frame0; they stay in the plots unless exclude_frame0=True.

window and min_duration default to 2 frames for G2 and 10 frames for G12
systems (default_frames).

Results (results_dir/touch_to_next_stay/<settings_folder>/, e.g. t3.5_wno_d10 = cutoff 3.5 Å,
no first-touch window, stays >= 10 frames; t3.5_w10_d10 with a 10-frame window;
<settings_folder>_above/ with load_mode='above'):
    touch_to_next_stay_run<run>.csv   one row per event
    settings_run<run>.json            cutoff, window, min_duration, frame_duration,
                                      ring residues and the first-touch count check
    plots/                            figures 1-3 of this run (pooled counts and %), see write_plots
With load_mode='above' the run also checks that every column but the n_above /
n_below / n_sf split matches the load_mode='glu' CSV of the same settings, if
that exists, and prints the n_above / n_below / n_sf distributions.
Plots over runs: aggregate_touch_to_next_stay.py (repo root).
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.longest_duration_analysis import LastAndLongestDurationAnalysis, FirstTouchAnalysis
from analysis.glu_occupancy_at_exit import GluOccupancyAtExitAnalysis, ION_SELECTION
from analysis.closest_residue_analysis import ClosestResidueAnalysis, ONE_LETTER
from analysis.converter import convert_to_pdb_numbering


OUT = 'out'
NOT_LOWER = 'not_lower'
COLUMNS = ['run', 'ion_id', 'end_1', 'end_2', 'starts_at_frame0', 'first_touch_residue',
           'first_touch_class', 'touch_frame', 'leave_frame', 'n_glu_occupied', 'n_above',
           'n_below', 'n_sf', 'next_stay_residue', 'next_stay_type', 'next_stay_start',
           'next_stay_frames']
INT_COLUMNS = ['touch_frame', 'leave_frame', 'n_glu_occupied', 'n_above', 'n_below', 'n_sf',
               'next_stay_start', 'next_stay_frames']
# columns that must be the same in both load modes (the check in load_mode='above')
SHARED = ['ion_id', 'end_1', 'end_2', 'first_touch_residue', 'first_touch_class', 'touch_frame',
          'leave_frame', 'n_glu_occupied', 'next_stay_residue', 'next_stay_start', 'next_stay_frames']
LOAD_MODES = ('glu', 'above')
LEAVE_CHUNK = 50                         # frames read at a time while following the ion
CLASSES = ['ASP', 'ASN']                 # first-touch classes that are plotted
N_GLU = range(1, 5)                      # 1-4 GLUs occupied (figures 2-3 leave out the 0 case)
COLOR_MAP = ClosestResidueAnalysis.COLOR_MAP
TYPE_ORDER = ClosestResidueAnalysis.TYPE_ORDER
NO_CONTACT_COLOR = LastAndLongestDurationAnalysis.NO_CONTACT_COLOR


def default_frames(channel_type):
    """First-touch window and minimum stay: 2 frames for G2, 10 for the G12 systems."""
    return 2 if str(channel_type).upper().startswith('G2') else 10


def settings_folder(cutoff, window, min_duration, load_mode='glu'):
    """'t3.5_wno_d10': cutoff, first-touch window (no = none) and minimum stay in frames;
    '_above' appended for load_mode='above'."""
    return (f"t{cutoff:g}_w{'no' if window is None else window}_d{min_duration}"
            + ('_above' if load_mode == 'above' else ''))


def csv_name(run):
    return f"touch_to_next_stay_run{run}.csv"


def leave_frame(touch, ion_id, resid, touch_frame, end_2_frame):
    """
    Last frame, from touch_frame on (never past end_2), at which residue
    `resid` is still the closest residue within the cutoff. `touch` is the
    FirstTouchAnalysis object (its distances and _closest_from_row).
    """
    last = touch_frame
    for start in range(touch_frame, end_2_frame + 1, LEAVE_CHUNK):
        frames, D = touch.distances.min_distances(ion_id, start, min(start + LEAVE_CHUNK - 1, end_2_frame))
        if D is None:
            return last
        for frame, row in zip(frames, D):
            info, _ = touch._closest_from_row(row)
            if not info or info['residue_id'] != resid:
                return last
            last = int(frame)
    return last


def next_stay(stays, touch_frame):
    """The earliest stay with end_frame >= touch_frame, or None."""
    later = [s for s in stays if s['end_frame'] >= touch_frame]
    return min(later, key=lambda s: s['start_frame']) if later else None


def ions_along_pore(ions, ion_id, channel1, channel2):
    """
    At the current frame: other K+ in the cavity (region 2) above / below the
    permeating ion along the ch2 axis, and other K+ in the SF (region 1).
    Both regions are recomputed with compute_geometry and tested with
    is_within_cylinder, as in ion_state_sequence.py. "Above" = toward the SF:
    the sign of the ch2 axis that points from the cavity centre to the SF
    centre. Returns ({'n_above', 'n_below', 'n_sf'}, toward-SF unit vector).
    """
    channel1.compute_geometry(1)
    channel2.compute_geometry(2)
    axis = channel2.channel_axis
    to_sf = axis if np.dot(channel1.channel_center - channel2.channel_center, axis) > 0 else -axis
    pos = ions.positions
    other = ions.resids != int(ion_id)
    t = (pos - channel2.channel_center) @ to_sf          # position along the pore, + = toward the SF
    t_ion = t[~other][0]
    in_cav = np.array([channel2.is_within_cylinder(p) for p in pos], dtype=bool)
    in_sf = np.array([channel1.is_within_cylinder(p) for p in pos], dtype=bool)
    return {'n_above': int((other & in_cav & (t > t_ion)).sum()),
            'n_below': int((other & in_cav & (t < t_ion)).sum()),
            'n_sf': int((other & in_sf).sum())}, to_sf


def check_modes(df, glu_csv):
    """
    load_mode='above' check: every SHARED column equals the load_mode='glu'
    CSV of the same settings. Returns a dict for settings_run<run>.json.
    """
    if not glu_csv.exists():
        print(f"  Load-mode check: no load_mode='glu' results at {glu_csv} - not compared")
        return {'compared_with': None}
    glu = pd.read_csv(glu_csv)
    cols = [c for c in SHARED if c in glu.columns]
    a, b = df[cols].reset_index(drop=True), glu[cols].reset_index(drop=True)
    same_shape = len(a) == len(b)
    # two missing values (no touch, no stay) count as equal
    differ = {c: int((~((a[c] == b[c]) | (a[c].isna() & b[c].isna()))).sum()) for c in cols} if same_shape else {}
    ok = same_shape and not any(differ.values())
    print(f"  Load-mode check vs {glu_csv}: "
          + (f"identical ({len(a)} events; {', '.join(cols)})" if ok else
             f"{len(a)} events here vs {len(b)} there" if not same_shape else
             "DIFFERENT in " + ", ".join(f"{c} ({n} events)" for c, n in differ.items() if n)))
    return {'compared_with': str(glu_csv), 'identical': ok, 'columns': cols, 'n_differ': differ}


def check_first_touch(touch_results, results_dir, cutoff, window):
    """
    Compare the first touch of every event with saved results at the same
    settings, if they exist: FirstTouchAnalysis (duration_touch_residue/
    <cutoff>_w<window>/), or without a window (window=None) the GLU/ASN
    occupancy-at-exit analysis (glu_occupancy_exit/<cutoff>_no_w/).
    Returns a dict for settings_run<run>.json.
    """
    if window is None:
        where = "glu_occupancy_exit/{t}_no_w/glu_occupancy_exit_complete.json"
    else:
        where = f"duration_touch_residue/{{t}}_w{window}/touch_duration_residue.json"
    found = None
    for t in dict.fromkeys((str(cutoff), f"{cutoff:g}")):
        f = Path(results_dir) / where.format(t=t)
        if f.exists():
            found = f
            break
    if found is None:
        print(f"  First-touch check: no saved results at {where.format(t=cutoff)} - not compared")
        return {'compared_with': None}
    with open(found) as fh:
        saved = {(int(e['ion_id']), int(e['end_2_frame'])):
                 (e.get('identifier', e.get('touch_identifier')), e.get('touch_frame'))
                 for e in json.load(fh)['results']}
    mismatches = []
    for r in touch_results:
        key = (r['ion_id'], r['end_2_frame'])
        if key not in saved or saved[key] != (r['touch_identifier'], r['touch_frame']):
            mismatches.append({'ion_id': r['ion_id'], 'end_2': r['end_2_frame'],
                               'here': [r['touch_identifier'], r['touch_frame']],
                               'saved': list(saved[key]) if key in saved else None})
    n_saved = len(saved)
    ok = not mismatches and n_saved == len(touch_results)
    print(f"  First-touch check vs {found}: "
          + ("all events match" if ok else
             f"{len(mismatches)} mismatches, {len(touch_results)} events here vs {n_saved} saved"))
    for m in mismatches[:10]:
        print(f"    ion {m['ion_id']} end_2 {m['end_2']}: here {m['here']}, saved {m['saved']}")
    return {'compared_with': str(found), 'n_events': len(touch_results), 'n_saved': n_saved,
            'all_match': ok, 'mismatches': mismatches}


# ----------------------------------------------------------------------
# Plots (style of create_bar_plot; shared with aggregate_touch_to_next_stay.py)
# ----------------------------------------------------------------------

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
    counts = counts.loc[sorted(counts.index, key=lambda r: (int(''.join(c for c in str(r) if c.isdigit()) or 0), str(r)))]
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


# Font sizes (pt) for every plot. They are large compared with the panel size, so the
# text stays readable when a figure is shrunk onto a slide; 'count' is the smaller (n).
FONT = {'suptitle': 27, 'title': 26, 'axis': 26, 'tick': 22, 'pct': 18, 'count': 14,
        'legend': 22, 'note': 20}
PANEL_SIZE = (9.5, 7)                    # inches per panel
DPI = 150
LABEL_PAD = 3                            # points between a bar (or the 100 % line) and its label


def n_label(n_events, n_runs):
    """'n = 5 events, 2 runs' (singular for 1)."""
    return (f"n = {n_events} event{'' if n_events == 1 else 's'}, "
            f"{n_runs} run{'' if n_runs == 1 else 's'}")


def _stack_labels(ax, x, y, lines):
    """
    Labels stacked upwards above data point (x, y), offset in points so their
    spacing does not depend on the axis; lines = [(text, fontsize, bold)],
    bottom line first. Returns the height used, in points.
    """
    offset = LABEL_PAD
    for text, size, bold in lines:
        ax.annotate(text, (x, y), xytext=(0, offset), textcoords='offset points', ha='center',
                    va='bottom', fontsize=size, fontweight='bold' if bold else 'normal',
                    annotation_clip=False, zorder=6)
        offset += size * 1.2
    return offset


def draw_panel(ax, sub, kind, labels, types, title):
    """
    One panel: the 9 next-stay bars of the events in `sub`, as % on a 0-100 %
    axis, labelled with the % and, smaller, the number of events in brackets.
    Bars with no events get no label.
    pooled : % of the panel's events, labels just above each bar
    per_run: mean ± SEM of the % in each run, dots = runs, labels in a row above 100 %
    """
    x = np.arange(len(labels))
    n_events, n_runs = len(sub), sub['run'].nunique()
    counts = pooled_counts(sub, labels)
    colors = [_color(l, types) for l in labels]
    label_height = 0

    if kind == 'pooled':
        pct = 100.0 * counts / n_events if n_events else np.zeros(len(labels))
        bars = ax.bar(x, pct, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5,
                      width=0.75, zorder=3)
        for xi, p, c in zip(x, pct, counts):
            if c:
                h = _stack_labels(ax, xi, p, [(f"({c})", FONT['count'], False), (f"{p:.0f}%", FONT['pct'], True)])
                if p >= 99.5:
                    label_height = max(label_height, h)
        ax.set_ylabel('% of events', fontsize=FONT['axis'], fontweight='bold')
        n_text = n_label(n_events, n_runs)
    else:
        runs = per_run_pct(sub, labels)
        means = runs.mean(axis=0).reindex(labels).fillna(0).values
        sems = (runs.std(axis=0, ddof=1) / np.sqrt(len(runs))).reindex(labels).fillna(0).values \
            if len(runs) > 1 else np.zeros(len(labels))
        bars = ax.bar(x, means, yerr=sems, capsize=6, color=colors, alpha=0.85, edgecolor='black',
                      linewidth=1.5, width=0.75, zorder=3, error_kw={'elinewidth': 2, 'capthick': 2, 'zorder': 4})
        rng = np.random.default_rng(0)
        for i, l in enumerate(labels):
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(runs)), runs[l].values, s=40,
                       color='black', alpha=0.65, zorder=5, linewidths=0)
        for xi, m, e, c in zip(x, means, sems, counts):
            if c:
                label_height = max(label_height, _stack_labels(
                    ax, xi, 100, [(f"({c})", FONT['count'], False), (f"±{e:.0f}", FONT['count'], False),
                                  (f"{m:.0f}%", FONT['pct'], True)]))
        ax.set_ylabel('% of events in a run', fontsize=FONT['axis'], fontweight='bold')
        n_text = n_label(n_events, len(runs))
    bars[-1].set_hatch('//')

    # y axis always 0-100 %; labels above 100 % sit between the axis and the panel title
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    if n_events == 0:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes, ha='center', va='center',
                fontsize=FONT['title'], fontweight='bold', color='0.4')
    ax.set_xticks(x)
    ax.set_xticklabels([_tick(l, types) for l in labels], fontsize=FONT['tick'], fontweight='bold',
                       rotation=50, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Next stay', fontsize=FONT['axis'], fontweight='bold')
    ax.tick_params(axis='y', labelsize=FONT['tick'])
    ax.tick_params(axis='both', width=1.5, length=6)
    ax.set_title(f"{title}\n({n_text})", fontsize=FONT['title'], fontweight='bold', pad=8 + label_height)
    _style(ax)


def kind_text(kind):
    return ("pooled: every event counted once" if kind == 'pooled'
            else "per run: mean ± SEM of % per run, dots = runs")


def _save(fig, output_file):
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {output_file}")


def _wrap(text, fig, fontsize):
    """Wrap each line of text to the figure's width, so titles never run past the panels."""
    width = max(20, int(fig.get_figwidth() * 72 / (fontsize * 0.55)))
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def _figure_legend(fig, labels, types, title=None):
    """Legend outside the panels, on the right of the figure."""
    fig.legend(handles=legend_handles(labels, types), loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=FONT['legend'], title=title, title_fontsize=FONT['legend'], frameon=True)


def plotted_classes(settings):
    """First-touch classes that get plots: G2 has no ASP, so only ASN; every other system both."""
    return ['ASN'] if str(settings.get('channel_type', '')).upper().startswith('G2') else CLASSES


def figure_overall(df, kind, labels, types, head, rule, output_file, classes=CLASSES):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, len(classes), figsize=(len(classes) * PANEL_SIZE[0], PANEL_SIZE[1] + 1.5),
                             squeeze=False, layout='constrained')
    for ax, cls in zip(axes.flat, classes):
        draw_panel(ax, df[df['first_touch_class'] == cls], kind, labels, types, f"First touch = {cls}")
    _figure_legend(fig, labels, types, "Next stay")
    fig.suptitle(_wrap(f"{head}: next stay after the first touch on the lower ring\n{rule}\n({kind_text(kind)})",
                       fig, FONT['suptitle']), fontsize=FONT['suptitle'], fontweight='bold')
    _save(fig, output_file)


def load_split(load_mode, glu_name, df=None):
    """
    How figures 2-3 split the events: (column, [(panel title, file tag, value
    test)] one per panel, sub-folder tag, how the suptitle names it).
    'glu': 1-4 GLUs occupied. 'above': one panel per value of n_above >= 1
    that occurs among the events in df (values with no event get no panel).
    The 0 case (no GLU occupied / no K+ above) is never shown in figures 2-3.
    """
    if load_mode == 'above':
        values = pd.to_numeric(df['n_above'], errors='coerce').dropna() if df is not None else []
        present = sorted({int(v) for v in values if v >= 1})
        bins = [(f"{k} ions above", f"{k}above", lambda v, k=k: v == k) for k in present]
        return 'n_above', bins, 'ions_above', "the number of K+ above the ion in the cavity"
    bins = [(f"{k} {glu_name} occupied", f"{k}glu_occupied", lambda v, k=k: v == k) for k in N_GLU]
    return 'n_glu_occupied', bins, 'glu_occupancy', f"{glu_name} occupied"


def split_events(sub, column, test):
    """Events of `sub` whose `column` passes `test` (missing values never do)."""
    v = pd.to_numeric(sub[column], errors='coerce')
    return sub[v.notna() & test(v)]


def figure_by_load(df, cls, kind, labels, types, load_mode, glu_name, head, rule, output_file, single_dir):
    """Figure 2 / 3 for first touch = cls: one panel per load value (see load_split), plus each panel alone."""
    sub_cls = df[df['first_touch_class'] == cls]
    column, bins, _, what = load_split(load_mode, glu_name, sub_cls)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    if not bins:                                          # 'above' with no ASP/ASN events at all
        bins = [("no events", "none", lambda v: v != v)]
    ncols = 2 if len(bins) == 4 else min(3, len(bins))      # 4 panels as 2 x 2
    nrows = -(-len(bins) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_SIZE[0] * ncols, PANEL_SIZE[1] * nrows + 2),
                             squeeze=False, layout='constrained')
    for (title, _, test), ax in zip(bins, axes.flat):
        draw_panel(ax, split_events(sub_cls, column, test), kind, labels, types, title)
    for ax in axes.flat[len(bins):]:
        ax.axis('off')
    _figure_legend(fig, labels, types, f"First touch = {cls}\nn = {len(sub_cls)} events\n\nNext stay")
    fig.suptitle(_wrap(f"{head}: next stay after first touch = {cls},\nby {what} when it leaves {cls}\n"
                       f"{rule}\n({kind_text(kind)})", fig, FONT['suptitle']),
                 fontsize=FONT['suptitle'], fontweight='bold')
    _save(fig, output_file)

    single_dir.mkdir(parents=True, exist_ok=True)
    for title, tag, test in bins:
        fig, ax = plt.subplots(figsize=PANEL_SIZE, layout='constrained')
        draw_panel(ax, split_events(sub_cls, column, test), kind, labels, types,
                   f"{head}: first touch = {cls}, {title}")
        _figure_legend(fig, labels, types, "Next stay")
        fig.text(0.5, -0.02, _wrap(f"{rule}\n{kind_text(kind)}", fig, FONT['note']), ha='center', va='top',
                 fontsize=FONT['note'])
        _save(fig, single_dir / f"{cls.lower()}_{tag}_{kind}.png")


def frame0_events(df, exclude):
    """
    Prints how many starting-structure events (end_1 == 0) there are per run;
    returns df without them when exclude, else df unchanged.
    """
    frame0 = pd.to_numeric(df['end_1'], errors='coerce') == 0
    per_run = df.loc[frame0, 'run'].value_counts()
    print(f"  Starting-structure events (end_1 == 0): {int(frame0.sum())}"
          + (" (" + ", ".join(f"{r}: {n}" for r, n in sorted(per_run.items(), key=lambda x: str(x[0]))) + ")"
             if len(per_run) else "")
          + (" - left out of the plots" if exclude else " - kept in the plots"))
    return df[~frame0] if exclude else df


def write_plots(df, settings, head, output_dir, kinds=('pooled',), exclude_frame0=False):
    """
    Figures 1-3 (and the single panels of 2-3) for the events in df, in
    output_dir: one run (main.py, kinds=('pooled',)) or a group of runs
    (aggregate_touch_to_next_stay.py, pooled and per_run). Figures 2-3 are
    split by settings['load_mode'] (n_glu_occupied, or n_above). G2 (no ASP):
    ASN only (plotted_classes).
    exclude_frame0: True leaves end_1 == 0 events out; None = the caller already decided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if exclude_frame0 is not None:
        df = frame0_events(df, exclude_frame0)
    labels, types, glu_name = bar_labels(settings)
    rule = rule_text(settings)
    load_mode = settings.get('load_mode', 'glu')
    folder = load_split(load_mode, glu_name)[2]
    classes = plotted_classes(settings)
    for kind in kinds:
        figure_overall(df, kind, labels, types, head, rule, output_dir / f"fig1_next_stay_overall_{kind}.png",
                       classes)
        for fig_no, cls in ((2, 'ASP'), (3, 'ASN')):
            if cls not in classes:
                continue
            figure_by_load(df, cls, kind, labels, types, load_mode, glu_name, head, rule,
                           output_dir / f"fig{fig_no}_{cls.lower()}_by_{folder}_{kind}.png",
                           output_dir / f"{cls.lower()}_by_{folder}")


def run_touch_to_next_stay_analysis(universe, permeation_events, asn_residues, glu_residues,
                                    channel_type="G12", run=1, results_dir=".", frame_duration=0.075,
                                    cutoff=3.5, window=None, min_duration=None,
                                    analysis_start_frame=0, ion_selection=ION_SELECTION,
                                    no_window=False, load_mode='glu', channel1=None, channel2=None,
                                    exclude_frame0=False, force_rerun=False):
    """
    First touch on the lower ring -> next stay, with the number of occupied
    GLUs when the ion leaves the touched residue. See the module docstring.

    window / min_duration : frames; None = default_frames(channel_type)
    no_window             : first touch anywhere from end_1 to end_2 (window ignored)
    load_mode             : 'glu' (figures 2-3 split by n_glu_occupied) or 'above' (by n_above);
                            n_glu_occupied, n_above, n_below and n_sf are stored in both
    channel1, channel2    : main.py's ch1 (SF, region 1) and ch2 (cavity, region 2) Channel objects
    exclude_frame0        : True: leave events with end_1 == 0 out of the plots (always kept in the CSV)
    Results go to results_dir/touch_to_next_stay/<settings_folder(cutoff, window, min_duration)>/
    (plots of the run in plots/; a run skipped because its CSV exists still gets them if missing)
    results_dir           : the run's results folder (RUN<run>/)

    Call from main.py:
        from analysis.touch_to_next_stay import run_touch_to_next_stay_analysis
        run_touch_to_next_stay_analysis(u, analyzer.permeation_events, asn_residues, glu_residues,
                                        channel_type, run_number, results_dir, frame_duration,
                                        channel1=ch1, channel2=ch2)
    """
    if load_mode not in LOAD_MODES:
        raise ValueError(f"load_mode must be one of {LOAD_MODES}, not {load_mode!r}")
    window = None if no_window else default_frames(channel_type) if window is None else int(window)
    min_duration = default_frames(channel_type) if min_duration is None else int(min_duration)
    results_dir = Path(results_dir)
    output_dir = results_dir / "touch_to_next_stay" / settings_folder(cutoff, window, min_duration, load_mode)
    out_csv = output_dir / csv_name(run)
    if out_csv.exists() and not force_rerun:
        print(f"  {out_csv} already exists - skipping (use force_rerun=True to overwrite).")
        df = pd.read_csv(out_csv)
        settings_file = output_dir / f"settings_run{run}.json"
        if not (output_dir / "plots").exists() and settings_file.exists():
            with open(settings_file) as f:
                write_plots(df, json.load(f), f"{channel_type} RUN{run}", output_dir / "plots",
                            exclude_frame0=exclude_frame0)
        return df
    if channel1 is None or channel2 is None:
        raise ValueError("run_touch_to_next_stay_analysis needs channel1=ch1 and channel2=ch2 "
                         "(main.py's region-1 and region-2 Channel objects) for n_above / n_below / n_sf")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TOUCH -> NEXT STAY (cutoff {cutoff} Å, "
          + ("first touch anywhere end_1 -> end_2" if no_window else f"first-touch window {window} fr")
          + f", stay >= {min_duration} fr, load_mode='{load_mode}')")
    print("=" * 80)

    # 1. first touch: the existing analysis, residue mode, both rings compete.
    # No window: a window longer than the trajectory, so the search runs end_1+1 ... end_2
    touch = FirstTouchAnalysis(universe, permeation_events, asn_residues, glu_residues,
                               channel_type, threshold=cutoff,
                               window=len(universe.trajectory) if no_window else window,
                               analysis_start_frame=analysis_start_frame,
                               residue_types=['ASN', 'GLU'], subunit_groups=None,
                               results_dir=output_dir)
    touch.run_analysis()
    # stays: the existing segmentation, every stay of >= min_duration frames
    stays = LastAndLongestDurationAnalysis(universe, permeation_events, asn_residues, glu_residues,
                                           channel_type, threshold=cutoff, min_duration=min_duration,
                                           analysis_start_frame=analysis_start_frame,
                                           residue_types=['ASN', 'GLU'], subunit_groups=None,
                                           results_dir=output_dir)
    # occupancy by other K+: the GLU/ASN occupancy-at-exit code (no events of its own)
    occupancy = GluOccupancyAtExitAnalysis(universe, [], asn_residues, glu_residues, channel_type,
                                           threshold=cutoff, window=window,
                                           analysis_start_frame=analysis_start_frame,
                                           ion_selection=ion_selection, results_dir=output_dir)

    lower = {int(r) for r in asn_residues}
    rows, to_sf_z = [], []
    for t in touch.results:
        ion_id, end_1, end_2 = t['ion_id'], t['end_1_frame'], t['end_2_frame']
        info = t['touch_residues'][0] if t['touch_residues'] else None
        row = dict.fromkeys(COLUMNS)
        row.update(run=run, ion_id=ion_id, end_1=end_1, end_2=end_2, starts_at_frame0=end_1 == 0,
                   first_touch_residue=info['residue_pdb'] if info else None)
        if not info or info['residue_id'] not in lower:
            row['first_touch_class'] = NOT_LOWER
            rows.append(row)
            continue

        row['first_touch_class'] = info['residue_type']
        row['touch_frame'] = t['touch_frame']
        row['leave_frame'] = leave_frame(touch, ion_id, info['residue_id'], t['touch_frame'], end_2)
        row['n_glu_occupied'] = occupancy.occupancy(row['leave_frame'], ion_id)['n_glu_occupied']
        universe.trajectory[row['leave_frame']]
        along, to_sf = ions_along_pore(occupancy.ions, ion_id, channel1, channel2)
        row.update(along)
        if not to_sf_z:
            print(f"  Toward the SF (from the region centres, frame {row['leave_frame']}): "
                  f"({to_sf[0]:+.2f}, {to_sf[1]:+.2f}, {to_sf[2]:+.2f}) = "
                  f"{'-' if np.dot(to_sf, channel2.channel_axis) < 0 else '+'}ch2 axis; "
                  f"SF {'above' if to_sf[2] > 0 else 'below'} the cavity in z")
        to_sf_z.append(float(to_sf[2]))
        stay = next_stay(stays.find_last_and_longest_duration(ion_id, end_2, end_1)['all_durations'],
                         t['touch_frame'])
        if stay is None:
            row.update(next_stay_residue=OUT, next_stay_type=OUT)
        else:
            res = stay['residues'][0]
            row.update(next_stay_residue=res['residue_pdb'], next_stay_type=res['residue_type'],
                       next_stay_start=stay['start_frame'], next_stay_frames=stay['duration'])
        rows.append(row)
        print(f"Ion {ion_id}: first touch {row['first_touch_class']} {row['first_touch_residue']} "
              f"at {row['touch_frame']}, leaves at {row['leave_frame']} "
              f"({row['n_glu_occupied']} GLU occupied, {row['n_above']} K+ above, {row['n_below']} below, "
              f"{row['n_sf']} in SF) -> next stay {row['next_stay_residue']}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in INT_COLUMNS:
        df[col] = df[col].astype('Int64')
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {len(df)} events saved to: {out_csv}")
    counts = df['first_touch_class'].value_counts().to_dict()
    print("  First touch: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    flips = sum(1 for z in to_sf_z if np.sign(z) != np.sign(to_sf_z[0])) if to_sf_z else 0
    if flips:
        print(f"  ⚠ the toward-SF direction changes sign in z in {flips} of {len(to_sf_z)} leaving frames")
    lower_rows = df[df['first_touch_class'] != NOT_LOWER]
    for col in ('n_above', 'n_below', 'n_sf'):
        dist = lower_rows[col].value_counts().sort_index()
        print(f"  {col} at leave_frame ({len(lower_rows)} lower-ring events): "
              + (", ".join(f"{int(k)}: {v}" for k, v in dist.items()) or "-"))
    mode_check = (check_modes(pd.read_csv(out_csv), results_dir / "touch_to_next_stay" / settings_folder(cutoff, window, min_duration)
                              / csv_name(run)) if load_mode == 'above' else None)

    # Ring residues (in the order of the lists) so the aggregator draws every bar, also empty ones
    ring_residues = [{'pdb': convert_to_pdb_numbering(r, channel_type), 'resid': int(r), 'ring': ring,
                      'residue_type': touch.resnames.get(int(r), 'Unknown')}
                     for ring, resids in (('GLU', glu_residues), ('ASN', asn_residues)) for r in resids]
    settings = {
        'channel_type': channel_type, 'run': run, 'cutoff': cutoff, 'window': window,
        'min_duration': min_duration, 'frame_duration': frame_duration, 'load_mode': load_mode,
        'exclude_frame0': exclude_frame0, 'n_starts_at_frame0': int(df['starts_at_frame0'].sum()),
        'to_sf_z_flips': flips, 'load_mode_check': mode_check,
        'ion_selection': ion_selection, 'ring_residues': ring_residues,
        'first_touch_counts': counts,
        'first_touch_check': check_first_touch(touch.results, results_dir, cutoff, window),
    }
    with open(output_dir / f"settings_run{run}.json", 'w') as f:
        json.dump(settings, f, indent=2)
    write_plots(df, settings, f"{channel_type} RUN{run}", output_dir / "plots", exclude_frame0=exclude_frame0)
    return df
