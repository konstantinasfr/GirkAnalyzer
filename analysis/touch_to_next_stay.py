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
Plots over runs: aggregate_touch_to_next_stay.py (repo root).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.longest_duration_analysis import LastAndLongestDurationAnalysis, FirstTouchAnalysis
from analysis.glu_occupancy_at_exit import GluOccupancyAtExitAnalysis, ION_SELECTION
from analysis.converter import convert_to_pdb_numbering


OUT = 'out'
NOT_LOWER = 'not_lower'
COLUMNS = ['run', 'ion_id', 'end_1', 'end_2', 'first_touch_residue', 'first_touch_class',
           'touch_frame', 'leave_frame', 'n_glu_occupied', 'next_stay_residue',
           'next_stay_type', 'next_stay_start', 'next_stay_frames']
LEAVE_CHUNK = 50                         # frames read at a time while following the ion


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
        return pd.read_csv(out_csv)
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
    return df
