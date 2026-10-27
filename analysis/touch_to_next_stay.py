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
    3. n_glu_occupied (0-4) at leave_frame
    4. next stay: the earliest stay with end_frame >= touch_frame (may be the
       touched residue itself), or 'out' if there is none

window and min_duration default to 2 frames for G2 and 10 frames for G12
systems (default_frames).

Results (results_dir/touch_to_next_stay/<settings_folder>/, e.g. t3.5_wno_d10 = cutoff 3.5 Å,
no first-touch window, stays >= 10 frames; t3.5_w10_d10 with a 10-frame window):
    touch_to_next_stay_run<run>.csv   one row per event
    settings_run<run>.json            cutoff, window, min_duration, frame_duration,
                                      ring residues and the first-touch count check
    plots/                            figures 1-3 of this run (pooled counts and %), see write_plots
Plots over runs: aggregate_touch_to_next_stay.py (repo root).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from analysis.longest_duration_analysis import LastAndLongestDurationAnalysis, FirstTouchAnalysis
from analysis.glu_occupancy_at_exit import GluOccupancyAtExitAnalysis, ION_SELECTION
from analysis.closest_residue_analysis import ClosestResidueAnalysis, ONE_LETTER
from analysis.converter import convert_to_pdb_numbering


OUT = 'out'
NOT_LOWER = 'not_lower'
COLUMNS = ['run', 'ion_id', 'end_1', 'end_2', 'first_touch_residue', 'first_touch_class',
           'touch_frame', 'leave_frame', 'n_glu_occupied', 'next_stay_residue',
           'next_stay_type', 'next_stay_start', 'next_stay_frames']
LEAVE_CHUNK = 50                         # frames read at a time while following the ion
CLASSES = ['ASP', 'ASN']                 # first-touch classes that are plotted
N_GLU = range(5)                         # 0-4 GLUs occupied
COLOR_MAP = ClosestResidueAnalysis.COLOR_MAP
TYPE_ORDER = ClosestResidueAnalysis.TYPE_ORDER
NO_CONTACT_COLOR = LastAndLongestDurationAnalysis.NO_CONTACT_COLOR


def default_frames(channel_type):
    """First-touch window and minimum stay: 2 frames for G2, 10 for the G12 systems."""
    return 2 if str(channel_type).upper().startswith('G2') else 10


def settings_folder(cutoff, window, min_duration):
    """'t3.5_wno_d10': cutoff, first-touch window (no = none) and minimum stay in frames."""
    return f"t{cutoff:g}_w{'no' if window is None else window}_d{min_duration}"


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


def n_label(n_events, n_runs):
    """'n = 5 events, 2 runs' (singular for 1)."""
    return (f"n = {n_events} event{'' if n_events == 1 else 's'}, "
            f"{n_runs} run{'' if n_runs == 1 else 's'}")


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
        n_text = n_label(n_events, n_runs)
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
        n_text = n_label(n_events, len(pct))
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


def write_plots(df, settings, head, output_dir, kinds=('pooled',)):
    """
    Figures 1-3 (and the single occupancy panels) for the events in df, in
    output_dir: one run (main.py, kinds=('pooled',)) or a group of runs
    (aggregate_touch_to_next_stay.py, pooled and per_run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, types, glu_name = bar_labels(settings)
    rule = rule_text(settings)
    for kind in kinds:
        figure_overall(df, kind, labels, types, head, rule, output_dir / f"fig1_next_stay_overall_{kind}.png")
        for fig_no, cls in ((2, 'ASP'), (3, 'ASN')):
            figure_by_occupancy(df, cls, kind, labels, types, glu_name, head, rule,
                                output_dir / f"fig{fig_no}_{cls.lower()}_by_glu_occupancy_{kind}.png",
                                output_dir / f"{cls.lower()}_by_glu_occupancy")


def run_touch_to_next_stay_analysis(universe, permeation_events, asn_residues, glu_residues,
                                    channel_type="G12", run=1, results_dir=".", frame_duration=0.075,
                                    cutoff=3.5, window=None, min_duration=None,
                                    analysis_start_frame=0, ion_selection=ION_SELECTION,
                                    no_window=False, force_rerun=False):
    """
    First touch on the lower ring -> next stay, with the number of occupied
    GLUs when the ion leaves the touched residue. See the module docstring.

    window / min_duration : frames; None = default_frames(channel_type)
    no_window             : first touch anywhere from end_1 to end_2 (window ignored)
    Results go to results_dir/touch_to_next_stay/<settings_folder(cutoff, window, min_duration)>/
    (plots of the run in plots/; a run skipped because its CSV exists still gets them if missing)
    results_dir           : the run's results folder (RUN<run>/)

    Call from main.py:
        from analysis.touch_to_next_stay import run_touch_to_next_stay_analysis
        run_touch_to_next_stay_analysis(u, analyzer.permeation_events, asn_residues, glu_residues,
                                        channel_type, run_number, results_dir, frame_duration)
    """
    window = None if no_window else default_frames(channel_type) if window is None else int(window)
    min_duration = default_frames(channel_type) if min_duration is None else int(min_duration)
    results_dir = Path(results_dir)
    output_dir = results_dir / "touch_to_next_stay" / settings_folder(cutoff, window, min_duration)
    out_csv = output_dir / csv_name(run)
    if out_csv.exists() and not force_rerun:
        print(f"  {out_csv} already exists - skipping (use force_rerun=True to overwrite).")
        df = pd.read_csv(out_csv)
        settings_file = output_dir / f"settings_run{run}.json"
        if not (output_dir / "plots").exists() and settings_file.exists():
            with open(settings_file) as f:
                write_plots(df, json.load(f), f"{channel_type} RUN{run}", output_dir / "plots")
        return df
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TOUCH -> NEXT STAY (cutoff {cutoff} Å, "
          + ("first touch anywhere end_1 -> end_2" if no_window else f"first-touch window {window} fr")
          + f", stay >= {min_duration} fr)")
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
    rows = []
    for t in touch.results:
        ion_id, end_1, end_2 = t['ion_id'], t['end_1_frame'], t['end_2_frame']
        info = t['touch_residues'][0] if t['touch_residues'] else None
        row = dict.fromkeys(COLUMNS)
        row.update(run=run, ion_id=ion_id, end_1=end_1, end_2=end_2,
                   first_touch_residue=info['residue_pdb'] if info else None)
        if not info or info['residue_id'] not in lower:
            row['first_touch_class'] = NOT_LOWER
            rows.append(row)
            continue

        row['first_touch_class'] = info['residue_type']
        row['touch_frame'] = t['touch_frame']
        row['leave_frame'] = leave_frame(touch, ion_id, info['residue_id'], t['touch_frame'], end_2)
        row['n_glu_occupied'] = occupancy.occupancy(row['leave_frame'], ion_id)['n_glu_occupied']
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
              f"({row['n_glu_occupied']} GLU occupied) -> next stay {row['next_stay_residue']}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ['touch_frame', 'leave_frame', 'n_glu_occupied', 'next_stay_start', 'next_stay_frames']:
        df[col] = df[col].astype('Int64')
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {len(df)} events saved to: {out_csv}")
    counts = df['first_touch_class'].value_counts().to_dict()
    print("  First touch: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    # Ring residues (in the order of the lists) so the aggregator draws every bar, also empty ones
    ring_residues = [{'pdb': convert_to_pdb_numbering(r, channel_type), 'resid': int(r), 'ring': ring,
                      'residue_type': touch.resnames.get(int(r), 'Unknown')}
                     for ring, resids in (('GLU', glu_residues), ('ASN', asn_residues)) for r in resids]
    settings = {
        'channel_type': channel_type, 'run': run, 'cutoff': cutoff, 'window': window,
        'min_duration': min_duration, 'frame_duration': frame_duration,
        'ion_selection': ion_selection, 'ring_residues': ring_residues,
        'first_touch_counts': counts,
        'first_touch_check': check_first_touch(touch.results, results_dir, cutoff, window),
    }
    with open(output_dir / f"settings_run{run}.json", 'w') as f:
        json.dump(settings, f, indent=2)
    write_plots(df, settings, f"{channel_type} RUN{run}", output_dir / "plots")
    return df
