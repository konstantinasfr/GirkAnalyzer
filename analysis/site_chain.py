"""
site_chain.py
=============
Site-to-site chain of permeation through the GLU/ASN cavity: every permeating
K+ is followed from the moment it leaves the SF (end_1) to its exit (end_2) as
a series of visits to the cavity residues,

    SF|n -> E152.2a|2a1b -> N184.2a|3a0b -> Exit

and all ions' transitions are pooled into one chain.

VISIT    the ion binds a residue: a stay (>= min_duration frames closest to that
         residue within `cutoff`, LastAndLongestDurationAnalysis, the same stays
         as the other analyses). Stays on the same residue in a row are one visit:
         a short loss of contact and return is not a step. The visit runs from
         its first bound frame to its last one before the ion stays on another
         residue or exits (residue time); contact time = the frames of it that
         are within the cutoff.
STATE    residue | K+ above | K+ below WHEN THE ION LEAVES: the most common
         (above, below) over the last `window` bound frames of the visit (4 for
         G12, 1 for G2; a tie goes to the later value). K+ above / below = the
         other K+ in the cavity (main.py's ch2 cylinder) further toward / away
         from the SF along the pore axis. Bins 0 .. max_above-1, max_above+
         (default 0-4, 5+) and 0 .. max_below-1, max_below+ (default 0-2, 3+).
START    SF|n, n = the other K+ in the cavity at frame end_1 (bins 0-4, 5+).
END      Exit (after end_2).
MOVING   frames not inside a visit: SF -> first visit, between visits, last
         visit -> Exit. Kept per arrow and per ion; residue time + moving time =
         end_2 - end_1 for every ion. Residues touched while moving (contacts
         too short to be a stay) are listed in the CSVs.

TWO MODELS (same visits)
    residue   each residue its own site (8 in G12)
    grouped   residues grouped by sequence number (G12: E141.G1, E152 x3,
              D173.G1, N184 x3); a hop between two residues of one group is a
              loop (E152 -> E152), the residues stay separate visits.

CONTROL  the same states with the other ions taken from frames 1/4, 1/2, 3/4 of
         the run later (the ion itself at the real frame): keeps what the ion's
         own position does to the counts, removes any real effect of the other
         ions.

IONS PRESENT AT FRAME 0 (end_1 == 0: their entry was not seen) are kept in the
CSVs and left out of the analysis unless include_frame0=True.

EQUILIBRATION  only ions that leave the SF after the protein has settled are
analysed (unless include_preequil=True; the others stay in the CSVs). Per run:
the Ca RMSD against frame 0 after superposition (protein_rmsd_analysis.
fitted_ca_rmsd), smoothed over 7.5 ns; start = where it stops rising: the start
of the first 30 ns window (eq_window_ns) in which the smoothed RMSD is flat,
|slope| <= 0.5 A per 100 ns (eq_max_slope), with its first point at the window's
level. Later jumps of the RMSD (e.g. a conformational change late in the run) do
not move it; if no window is that flat, nothing is cut (never_flat in
equilibration.csv). A start frame given per run (start_frame=, or
--start_frames run,start_frame CSV in the aggregator) replaces the automatic one.

OUTPUTS   <results_dir>/site_chain/<settings>/           (settings e.g. t3.5_d10)
    rmsd_run<run>.csv    fitted / unfitted Ca RMSD of every frame
    equilibration.csv, equilibration.png   start frame per run (automatic, given, used)
    frames_run<run>.csv  per frame end_1 .. end_2 of every event: residue in
                         contact (closest within the cutoff), residues occupied by
                         the other K+ in the cavity (same rule; ';'-joined), K+ above /
                         below / in the SF, the same counts for the 3 control shifts
                         (frames from an older version without 'occupied' are collected again)
    stays_run<run>.csv, events_run<run>.csv
    <model>_a<max_above>b<max_below>[_frame0][_preeq]/
        summary.txt
        visits.csv        one row per visit (times, counts when leaving, next, touches)
        events.csv        one row per ion (SF count, residue / moving time, path)
        states.csv        per state: departures, P(next site / Exit), times
        transitions.csv   per arrow: count, P, moving time
        exit_by_counts.csv  P(Exit next) per site x K+ above x K+ below, real and control
        exit_by_cavity_k.csv  P(Exit next) by the number of other K+ in the cavity, real and control
        crowding_within_blocks.csv, crowding_per_run.csv   the crowding checks of the summary
        network.png          P(next site | site), all K+ counts together
        network_crowding.png the same, split: departures with <= 3 / >= 4 other K+ in the cavity
        network_net.png      net flow (hops a -> b minus b -> a) per permeating ion
        exit_sources.png     last site before Exit, all and by cavity filling
        exit_heatmaps.png, first_step.png, time_budget.png,
        time_to_exit.png, prediction.png
        visits/visits_histogram.png  visits per ion, all ions and by the other K+ at end_1
        visits/leaving_by_visit.csv, .png   per visit k (1 = no stop .. 10+): % of the
                             permeating ions leaving at visit k and their mean transit
                             end_1 -> end_2 +- SE (compare systems: compare_leaving_by_visit.py)
        visits/visit_<k>.csv, visits/visit_<k>[_all|_by_k].png   k = 1-4 (counted from 1;
                             visits.csv counts from 0): of the ions with a visit k-1, the residue of
                             visit k or out (left before it); all ions and one panel per
                             other K+ in the cavity (visit 1: at end_1; visit k: above +
                             below when leaving visit k-1). _all only all ions, _by_k only
                             the K+ panels
        visits/visit_<k>_summary.csv, .png   visit k, one row per other-K+ count: ions,
                             % of the ions, visit time, % out / to each residue group
        visits/visit_<k>_k_time.csv, .png   visit k, per other-K+ count: % of the ions
                             and how long visit k lasted (mean +- SE; ions that go out
                             have no visit and count only in the %); out time = how long the
                             ions going out at visit k took to leave from where they were
                             (also in the visit plot titles and the summary table)
        occupancy/           residues holding another K+ at each step (visit 1: when the ion
                             leaves the SF; visit k: when it leaves visit k-1):
            visit_<k>_occupied_by_k.csv, .png    % of the ions seeing each residue group
                             occupied, per other-K+ count
            visit_<k>_free_vs_occupied.csv, .png  per residue group and K+ count: % going to
                             it and % going out when it is free vs occupied
            visit_<k>_summary_occupancy.csv, .png  the visit summary table with, next to each
                             residue group's % going to it, the % seeing it occupied (one number
                             per residue for E152 / N184)
            occupied = another K+ has the residue as its closest one within the cutoff for at
                             least as many frames as a stay (10 for G12, 2 for G2; all frames
                             if fewer) while the ion moves to its next visit or the exit
            visit_<k>_free_vs_occupied_residue.csv, visit_<k>_free_vs_occupied_<E152|N184>.png
                             the same test for each residue against itself (E152.2a free vs
                             occupied -> going to E152.2a, ...)
            visit_<k>_free_vs_occupied_table.png  the same per-residue test as one table:
                             to it / out, free -> occupied, for all ions and per K+ count
            visit_<k>_free_vs_occupied_visits.png  the same table, the "-> it" cells as: of the ions
                             that went to the residue, % that went while it held another K+
                             (in brackets: % of all ions seeing it occupied)
            robustness/visit_<k>_free_vs_occupied_rules.csv, .png  the same table with two other
                             occupancy rules, to check the result does not come from the trip length:
                             half of trip (another K+ there in >= half of the trip's frames, whatever
                             its length), fixed frames (another K+ there in >= half of the same 10 / 2 frames
                             right after leaving, for every ion) and leave frame (the frame it leaves)
            arrival_sharing.csv   visits starting while the residue holds another K+, and
                             how often that K+ has gone when the ion leaves (knock-on)
            occupancy_summary.txt
        handover/                    hand-over at the residues: arrival_sharing_by_k.csv (shared arrivals
                             by the other K+ in the cavity) and, with occupant identities in the frames
                             (main.py run with this version), the ion tracking of the K+ sitting at a
                             residue while the new ion travels there: when it leaves (relative to the
                             arrival) and where it goes (handover_visits.csv, handover_summary.csv,
                             handover.png, handover.txt)
        presentation/                (residue model) the visit-1 story for slides, one message per figure:
                             01 first stop by K+, 02 who holds a K+, 03 free vs occupied, 04 the K+
                             already there (pushed or gone), 05 where the pushed K+ goes, 06 storage
                             seats push ions out, 07 the bottom of the cavity; speech_numbers.txt
            presentation/tables/     the tables of the slides (PNG + CSV): first stop and occupancy by
                             K+, free -> occupied (with robustness), Glu / bottom tracking (the K+ already there,
                             where it goes), after a stop at each residue, last stop, last stop by the visit it was
                             (06b: of the ions leaving after visit 1, 2, ... % per residue), asparagines by K+
        stops/stops.csv, .png, .txt   per residue (a stop): visits, time there, next stop (out / GLU
                             ring / lower ring), time still needed to leave the cavity from the visit,
                             and the SF -> exit time of the ions stopping there vs never (is it a trap?)
Over runs: aggregate_site_chain.py (repo root).

USAGE - from main.py (after ch1, ch2 and analyzer exist)
    from analysis.site_chain import run_site_chain
    run_site_chain(u, analyzer.permeation_events, asn_residues, glu_residues,
                   channel1=ch1, channel2=ch2, results_dir=results_dir,
                   run=run_number, channel_type=args.channel_type,
                   frame_duration=frame_duration)
CLI (G12 runs 1-20 built in):
    python3 -m analysis.site_chain --channel_type G12 --run 3
"""

import argparse
import contextlib
import io
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 12, "legend.title_fontsize": 13,
                     "figure.titlesize": 16})   # large enough to read when the figures are shrunk

from analysis.stay_chain import (PoreFrames, CONTROL_SHIFTS, default_min_duration, settings_folder,
                                 ks_gap, _Report)

OUT_FOLDER = "site_chain"
SF, EXIT = "SF", "Exit"
MODELS = ("residue", "grouped")
ONE_LETTER = {"GLU": "E", "GLH": "E", "GLN": "Q", "ASP": "D", "ASH": "D", "ASN": "N"}
COLORS = {"E": "#e8766f", "Q": "#f2a35e", "N": "#5aa5d6", "D": "#5cbf6a"}
GREY = "#9a9a9a"
MIN_TEST = 20                     # departures needed for the memory test of a state
N_SIM = 20000
LEAVE_VISITS = 10                 # leaving_by_visit: visits 1 .. 9 and 10+
PLOT_VISITS = (0, 1, 2, 3)            # visits/visit_<k>*.png: visits.csv numbering, 0 = first visit
FROM_NEXT = "from \\ next"
EQ_MAX_SLOPE = 0.5                # A per 100 ns: a smoothed RMSD this flat over EQ_WINDOW_NS = settled
EQ_WINDOW_NS = 30.0
EQ_SMOOTH_NS = 7.5
N_C = len(CONTROL_SHIFTS)


def default_crowded(channel_type):
    """Other K+ in the cavity counted as 'crowded': 3 for G2 (its cavity holds one K+ fewer), else 4."""
    return 3 if str(channel_type).upper().startswith("G2") else 4


def default_window(channel_type):
    """Bound frames the counts at leaving are taken over: 1 for G2 (0.4 ns frames), 4 for G12."""
    return 1 if str(channel_type).upper().startswith("G2") else 4


def analysis_folder(model, max_above, max_below, include_frame0, include_preequil=False):
    return (f"{model}_a{max_above}b{max_below}" + ("_frame0" if include_frame0 else "")
            + ("_preeq" if include_preequil else ""))


def bin_label(n, top):
    return f"{top}+" if n >= top else str(int(n))


def site_label(residue_pdb, resname):
    return f"{ONE_LETTER.get(str(resname).upper(), '')}{residue_pdb}"


def group_label(site):
    return site.split(".")[0]


def group_names(sites):
    """{site: name in the grouped model}: residues with the same sequence number
    together (E152.2a/b/c -> E152); a residue alone keeps its full name (E141.G1)."""
    members = {}
    for s in set(sites):
        members.setdefault(group_label(s), []).append(s)
    return {s: (g if len(m) > 1 else s) for g, m in members.items() for s in m}


def site_color(site):
    return COLORS.get(site[:1], GREY)


# ─────────────────────────────────────────────────────────────────────────────
# Collecting the data of one run (needs the trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def collect_run(universe, permeation_events, asn_residues, glu_residues, channel1, channel2,
                run, channel_type, cutoff, min_duration, analysis_start_frame=0):
    """(frames, stays, events) DataFrames of one run; see the module docstring."""
    from analysis.longest_duration_analysis import LastAndLongestDurationAnalysis

    with contextlib.redirect_stdout(io.StringIO()):            # its constructor prints
        dur = LastAndLongestDurationAnalysis(
            universe, permeation_events, list(asn_residues), list(glu_residues), channel_type,
            threshold=cutoff, min_duration=min_duration, analysis_start_frame=analysis_start_frame,
            residue_types=["ASN", "GLU"], subunit_groups=None, results_dir=Path("."))
    pore = PoreFrames(universe, channel1, channel2)
    n_frames = len(universe.trajectory)
    shifts = [int(f * n_frames) for f in CONTROL_SHIFTS]
    glu = {int(r) for r in glu_residues}

    events = []
    for k, ev in enumerate(permeation_events):
        end_1 = int(ev["end_1"] if "end_1" in ev else ev["R1_end"])
        end_2 = int(ev["end_2"] if "end_2" in ev else ev["R2_end"])
        events.append(dict(run=run, event=k, ion_id=int(ev["ion_id"]), end_1=end_1, end_2=end_2,
                           starts_at_frame0=end_1 == 0))
    wanted, real = set(), set()
    for e in events:
        for f in range(e["end_1"], e["end_2"] + 1):
            wanted.add(f)
            real.add(f)
            wanted.update((f + s) % n_frames for s in shifts)
    occ = {}                                                   # frame -> {K+ resid: residue it occupies}
    cav_ions, sf_ions = {}, {}                                 # frame -> {resid: position along the pore}, [resid]
    from tqdm import tqdm
    for f in tqdm(sorted(wanted), desc=f"site_chain RUN{run}: reading frames", unit="frame"):  # once, in order
        t_ax, cav, in_sf = pore.get(f)
        if f in real:
            occ[f] = _frame_occupancy(dur, pore.ions, cav)
            cav_ions[f] = {int(r): float(z) for r, z in zip(pore.ions.resids[cav], t_ax[cav])}
            sf_ions[f] = [int(r) for r in pore.ions.resids[in_sf]]

    frames, stays = [], []
    for e in tqdm(events, desc=f"site_chain RUN{run}: contacts per ion", unit="ion"):
        ion, end_1, end_2 = e["ion_id"], e["end_1"], e["end_2"]
        if end_2 < end_1:
            continue
        found = dur.find_last_and_longest_duration(ion, end_2, end_1)["all_durations"]
        for d in sorted(found, key=lambda d: d["start_frame"]):
            res = d["residues"][0]
            stays.append(dict(run=run, event=e["event"], residue_id=int(res["residue_id"]),
                              residue_pdb=res["residue_pdb"], resname=res["residue_type"],
                              ring="GLU" if int(res["residue_id"]) in glu else "LOWER",
                              start_frame=int(d["start_frame"]), end_frame=int(d["end_frame"])))
        fr, D = dur.distances.min_distances(ion, end_1, end_2)
        for i, f in enumerate(range(end_1, end_2 + 1)):
            info = dur._closest_from_row(D[i])[0] if D is not None else None
            row = dict(run=run, event=e["event"], frame=f,
                       contact=site_label(info["residue_pdb"], info["residue_type"]) if info else "",
                       occupied=";".join(sorted(lab for r, lab in occ.get(f, {}).items() if r != ion)),
                       occupants=";".join(f"{lab}:{r}" for r, lab in sorted(occ.get(f, {}).items()) if r != ion),
                       cavity_ions=";".join(f"{r}:{z:.1f}" for r, z in sorted(cav_ions.get(f, {}).items())
                                            if r != ion),
                       sf_ions=";".join(str(r) for r in sf_ions.get(f, []) if r != ion))
            row["n_above"], row["n_below"], row["n_sf"] = pore.neighbours(ion, f)
            for j, s in enumerate(shifts, start=1):
                row[f"n_above_c{j}"], row[f"n_below_c{j}"], _ = pore.neighbours(ion, f, (f + s) % n_frames)
            frames.append(row)
    stay_cols = ["run", "event", "residue_id", "residue_pdb", "resname", "ring", "start_frame", "end_frame"]
    frame_cols = ["run", "event", "frame", "contact", "occupied", "occupants", "cavity_ions", "sf_ions",
                  "n_above", "n_below", "n_sf"] + \
                 [f"n_{k}_c{j}" for j in range(1, N_C + 1) for k in ("above", "below")]
    event_cols = ["run", "event", "ion_id", "end_1", "end_2", "starts_at_frame0"]
    return (pd.DataFrame(frames, columns=frame_cols), pd.DataFrame(stays, columns=stay_cols),
            pd.DataFrame(events, columns=event_cols))


def _frame_occupancy(dur, ions, in_cavity):
    """{resid: site} of the K+ in the cavity bound to a residue at the current frame: the closest
    residue within the cutoff (the same rule as the permeating ion's contact)."""
    dist = dur.distances
    idx = np.where(in_cavity)[0]
    if not len(idx) or not dist.resids:
        return {}
    d = np.linalg.norm(dist.atoms.positions[None, :, :] - ions.positions[idx][:, None, :], axis=2)
    D = np.minimum.reduceat(d, dist.starts, axis=1)
    out = {}
    for i, row in zip(idx, D):
        info = dur._closest_from_row(row)[0]
        if info:
            out[int(ions.resids[i])] = site_label(info["residue_pdb"], info["residue_type"])
    return out


def read_csv_or_empty(path, **kw):
    """pd.read_csv, or an empty DataFrame for a file without columns (a run without permeation)."""
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Visits (shared by both models)
# ─────────────────────────────────────────────────────────────────────────────

def _mode_last(pairs):
    """Most common value; a tie goes to the one seen last."""
    c = Counter(pairs)
    best = max(c.values())
    for p in reversed(pairs):
        if c[p] == best:
            return p


def _touches(contacts, skip=""):
    c = Counter(x for x in contacts if x and x != skip)
    return ";".join(f"{k}:{v}" for k, v in sorted(c.items()))


def _window_occupancy(occupied, f0, a, b, decision, need):
    """Residues holding another K+ while the ion moves (frames a..b): those occupied in at least
    `need` of these frames (all of them when the window is shorter); an empty window (the ion
    goes straight to its next visit) uses the frame it leaves from (`decision`)."""
    frames = range(a, b + 1) if b >= a else [decision]
    need = min(need, len(frames))
    cnt = Counter(r for f in frames for r in _occ_set(occupied[f - f0]))
    return ";".join(sorted(r for r, c in cnt.items() if c >= need))


def _fixed_occupancy(occupied, f0, leave, last, n):
    """Robustness rule, the same frames for every ion whatever it does next: residues holding another
    K+ in at least half of the n frames right after it leaves (leave+1..leave+n; cut at the frame it
    leaves the cavity, then half of those; the frame it leaves when none are left)."""
    b = min(leave + n, last, f0 + len(occupied) - 1)
    return _window_occupancy(occupied, f0, leave + 1, b, leave, (max(b - leave, 1) + 1) // 2)


def _half_occupancy(occupied, f0, a, b, decision):
    """Robustness rule over the whole trip: residues holding another K+ in at least half of the frames
    a..b, whatever the trip length (the frame it leaves when the trip is empty)."""
    return _window_occupancy(occupied, f0, a, b, decision, (max(b - a + 1, 1) + 1) // 2)


def build_visits(frames, stays, events, window, occ_frames=10):
    """
    (visits, events) DataFrames: visits one row per visit (see the module
    docstring), counts when leaving for the real ions (above, below) and the
    controls (above_c1, below_c1, ...); events with the SF count, residue and
    moving frames, and the touches before the first visit. Occupancy (when the
    frames have it): residues holding another K+ for >= occ_frames of the frames
    the ion moves before its next visit (all of them if fewer): occupied_sf
    (SF -> visit 1 / Exit), occupied_leave (this visit -> next / Exit); the
    single-frame snapshots are kept as occupied_sf_frame, occupied_leave_frame,
    occupied_arrive (first frame of the visit).
    """
    fr_by = {k: g.sort_values("frame") for k, g in frames.groupby(["run", "event"])}
    st_by = {k: g.sort_values("start_frame") for k, g in stays.groupby(["run", "event"])} if len(stays) else {}
    a_cols = ["n_above"] + [f"n_above_c{j}" for j in range(1, N_C + 1)]
    b_cols = ["n_below"] + [f"n_below_c{j}" for j in range(1, N_C + 1)]
    v_rows, e_rows = [], []
    for ev in events.to_dict("records"):
        key = (ev["run"], ev["event"])
        g = fr_by.get(key)
        if g is None or ev["end_2"] <= ev["end_1"]:
            continue
        end_1, end_2 = int(ev["end_1"]), int(ev["end_2"])
        f0 = int(g["frame"].iloc[0])
        A = {c: g[c].to_numpy() for c in a_cols + b_cols + ["n_sf"]}
        contact = g["contact"].fillna("").astype(str).to_numpy()
        has_occ = "occupied" in g and g["occupied"].notna().all()   # runs collected before occupancy: none
        occupied = g["occupied"].astype(str).to_numpy() if has_occ else None

        def at(f, col):
            return int(A[col][f - f0])

        # stays inside the event, consecutive ones on one residue joined
        visits = []
        s = st_by.get(key)
        for r in ([] if s is None else s.to_dict("records")):
            start, end = max(int(r["start_frame"]), end_1 + 1), min(int(r["end_frame"]), end_2)
            if end < start:
                continue
            site = site_label(r["residue_pdb"], r["resname"])
            if visits and visits[-1]["site"] == site:
                visits[-1].update(end=end, last_stay_start=start)
            else:
                visits.append(dict(site=site, ring=r["ring"], start=start, end=end, last_stay_start=start))

        e_out = dict(ev, transit_frames=end_2 - end_1, n_visits=len(visits))
        e_out["sf_count"] = at(end_1, "n_above") + at(end_1, "n_below")
        for j in range(1, N_C + 1):
            e_out[f"sf_count_c{j}"] = at(end_1, f"n_above_c{j}") + at(end_1, f"n_below_c{j}")
        first = visits[0]["start"] if visits else end_2 + 1
        e_out["moving_first"] = first - end_1 - 1
        e_out["touches_first"] = _touches(contact[end_1 + 1 - f0:first - f0])
        e_out["occupied_sf"] = (_window_occupancy(occupied, f0, end_1 + 1, first - 1, end_1, occ_frames)
                                if occupied is not None else np.nan)
        e_out["occupied_sf_frame"] = occupied[end_1 - f0] if occupied is not None else np.nan
        e_out["occupied_sf_half"] = (_half_occupancy(occupied, f0, end_1 + 1, first - 1, end_1)
                                     if occupied is not None else np.nan)
        e_out["occupied_sf_fixed"] = (_fixed_occupancy(occupied, f0, end_1, end_2, occ_frames)
                                      if occupied is not None else np.nan)

        for i, v in enumerate(visits):
            nxt = visits[i + 1] if i + 1 < len(visits) else None
            lo = max(v["last_stay_start"], v["end"] - window + 1)
            row = dict(run=ev["run"], event=ev["event"], ion_id=ev["ion_id"], visit=i, site=v["site"],
                       ring=v["ring"], start=v["start"], end=v["end"],
                       residue_frames=v["end"] - v["start"] + 1,
                       contact_frames=int(np.sum(contact[v["start"] - f0:v["end"] - f0 + 1] == v["site"])),
                       touches_inside=_touches(contact[v["start"] - f0:v["end"] - f0 + 1], skip=v["site"]),
                       n_sf=at(v["end"], "n_sf"),
                       prev_site=visits[i - 1]["site"] if i else SF,
                       next_site=nxt["site"] if nxt else EXIT,
                       moving_after=(nxt["start"] if nxt else end_2 + 1) - v["end"] - 1,
                       to_exit_frames=end_2 - v["start"] + 1)
            row["touches_after"] = _touches(contact[v["end"] + 1 - f0:(nxt["start"] if nxt else end_2 + 1) - f0])
            if occupied is not None:                      # residues holding another K+ (';'-joined)
                row["occupied_arrive"] = occupied[v["start"] - f0]
                row["occupied_leave_frame"] = occupied[v["end"] - f0]
                row["occupied_leave"] = _window_occupancy(occupied, f0, v["end"] + 1,
                                                          (nxt["start"] if nxt else end_2 + 1) - 1, v["end"],
                                                          occ_frames)
                row["occupied_leave_half"] = _half_occupancy(occupied, f0, v["end"] + 1,
                                                             (nxt["start"] if nxt else end_2 + 1) - 1, v["end"])
                row["occupied_leave_fixed"] = _fixed_occupancy(occupied, f0, v["end"], end_2, occ_frames)
            for j, (ac, bc) in enumerate(zip(a_cols, b_cols)):
                pair = _mode_last([(at(f, ac), at(f, bc)) for f in range(lo, v["end"] + 1)])
                suffix = "" if j == 0 else f"_c{j}"
                row[f"above{suffix}"], row[f"below{suffix}"] = pair
            v_rows.append(row)
        e_out["residue_frames"] = sum(v["end"] - v["start"] + 1 for v in visits)
        e_out["moving_frames"] = e_out["transit_frames"] - e_out["residue_frames"]
        e_rows.append(e_out)
    return pd.DataFrame(v_rows), pd.DataFrame(e_rows)


# ─────────────────────────────────────────────────────────────────────────────
# One model: states and steps
# ─────────────────────────────────────────────────────────────────────────────

def steps_table(visits, events, model, max_above, max_below, max_sf, control=0):
    """
    One row per arrow: from_state, to_state, from_site, to_site, prev_site,
    moving (frames), residue_frames of the from visit (NaN for SF), run, event,
    frame (when the ion leaves: end_1 for SF, else the last bound frame of the visit).
    control=j uses the counts of control shift j.
    """
    sfx = "" if control == 0 else f"_c{control}"
    groups = group_names(visits["site"]) if len(visits) else {}
    name = (lambda s: s) if model == "residue" else groups.get
    by_event = {k: g.sort_values("visit") for k, g in visits.groupby(["run", "event"])} if len(visits) else {}
    rows = []
    for ev in events.to_dict("records"):
        g = by_event.get((ev["run"], ev["event"]))
        vs = [] if g is None else g.to_dict("records")
        state = f"{SF}|{bin_label(ev['sf_count' + sfx], max_sf)}"
        site, prev, res_frames, t = SF, None, np.nan, ev["end_1"]
        moving = ev["moving_first"]
        for v in vs:
            s_new = name(v["site"])
            st_new = f"{s_new}|{bin_label(v['above' + sfx], max_above)}a{bin_label(v['below' + sfx], max_below)}b"
            rows.append((ev["run"], ev["event"], state, st_new, site, s_new, prev, moving, res_frames, t))
            prev, state, site = site, st_new, s_new
            moving, res_frames, t = v["moving_after"], v["residue_frames"], v["end"]
        rows.append((ev["run"], ev["event"], state, EXIT, site, EXIT, prev, moving, res_frames, t))
    return pd.DataFrame(rows, columns=["run", "event", "from_state", "to_state", "from_site", "to_site",
                                       "prev_site", "moving", "residue_frames", "frame"])


def site_order(visits, model):
    """Sites: GLU ring first, then the lower ring, by residue number and subunit."""
    if not len(visits):
        return []
    ring = {}
    groups = group_names(visits["site"])
    for s, r in zip(visits["site"], visits["ring"]):
        ring[s if model == "residue" else groups[s]] = r

    def key(s):
        num = "".join(ch for ch in s.split(".")[0] if ch.isdigit())
        return (ring[s] != "GLU", int(num) if num else 0, s)
    return sorted(ring, key=key), ring


def time_to_exit_model(steps, dt):
    """Mean time to Exit from every state of the semi-Markov chain (ns):
    T(i) = residue time(i) + sum_j P(i -> j) (moving(i -> j) + T(j))."""
    states = sorted(steps["from_state"].unique())
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    M, b = np.eye(n), np.zeros(n)
    for s, g in steps.groupby("from_state"):
        i = idx[s]
        b[i] = (0 if s.startswith(SF + "|") else np.nanmean(g["residue_frames"])) + g["moving"].mean()
        for t, c in g["to_state"].value_counts().items():
            if t != EXIT:
                M[i, idx[t]] -= c / len(g)
    try:
        T = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        T = np.linalg.lstsq(M, b, rcond=None)[0]
    return {s: T[idx[s]] * dt for s in states}


def simulate(steps, n_sim, rng):
    """
    The chain run forward from the observed SF states: next state from the
    observed arrows; residue time drawn from the visits of the state; moving
    time from the arrow (fallback: the site pair, then all). Returns
    (transit frames, visits per ion, moving frames per ion).
    """
    nxt, res, mov, mov_site, mov_all = {}, {}, {}, {}, steps["moving"].to_numpy()
    for s, g in steps.groupby("from_state"):
        c = g["to_state"].value_counts()
        nxt[s] = (list(c.index), c.to_numpy() / c.sum())
        res[s] = g["residue_frames"].dropna().to_numpy()
    for k, g in steps.groupby(["from_state", "to_state"]):
        mov[k] = g["moving"].to_numpy()
    for k, g in steps.groupby(["from_site", "to_site"]):
        mov_site[k] = g["moving"].to_numpy()
    site_of = dict(zip(steps["to_state"], steps["to_site"]))
    site_of.update(zip(steps["from_state"], steps["from_site"]))
    starts = steps.loc[steps["from_state"].str.startswith(SF + "|"), "from_state"].to_numpy()
    transit = np.zeros(n_sim)
    n_vis = np.zeros(n_sim, dtype=int)
    moving = np.zeros(n_sim)
    for k in range(n_sim):
        s = starts[rng.integers(len(starts))]
        for _ in range(100000):
            if not s.startswith(SF + "|") and len(res.get(s, [])):
                transit[k] += res[s][rng.integers(len(res[s]))]
            targets, probs = nxt[s]
            t = targets[rng.choice(len(targets), p=probs)]
            m = mov.get((s, t))
            if m is None or len(m) < 3:
                m = mov_site.get((site_of[s], site_of[t]), mov_all)
            dm = m[rng.integers(len(m))]
            transit[k] += dm
            moving[k] += dm
            if t == EXIT or t not in nxt:
                break
            n_vis[k] += 1
            s = t
    return transit, n_vis, moving


def _chi2_p(stat):
    """p of a chi-square statistic with 1 degree of freedom."""
    import math
    return math.erfc(math.sqrt(max(stat, 0) / 2))


def crowding_within_blocks(steps, dt, block_ns, crowded=4):
    """
    Crowded (>= crowded other K+) vs uncrowded departures compared only within the
    same block of block_ns of the same run (Cochran-Mantel-Haenszel). Returns dict:
    blocks used, crowded / uncrowded departures and their exit % in those blocks,
    crowded exits observed vs expected if crowding did not matter within blocks, p.
    """
    d = steps.assign(crowd=steps["from_state"].map(crowding) >= crowded, ex=steps["to_site"] == EXIT,
                     block=(steps["frame"] * dt // block_ns).astype(int))
    obs = exp = var = 0.0
    n1 = n0 = e1 = e0 = blocks = 0
    for _, g in d.groupby(["run", "block"]):
        c, u = g[g["crowd"]], g[~g["crowd"]]
        if not len(c) or not len(u):
            continue
        N, m1 = len(g), int(g["ex"].sum())
        blocks += 1
        n1, n0, e1, e0 = n1 + len(c), n0 + len(u), e1 + int(c["ex"].sum()), e0 + int(u["ex"].sum())
        obs += c["ex"].sum()
        exp += len(c) * m1 / N
        if N > 1:
            var += len(c) * len(u) * m1 * (N - m1) / (N ** 2 * (N - 1))
    stat = (abs(obs - exp) - 0.5) ** 2 / var if var > 0 else np.nan
    return dict(block_ns=block_ns, blocks=blocks, crowded_n=n1, crowded_exit_pct=100 * e1 / n1 if n1 else np.nan,
                uncrowded_n=n0, uncrowded_exit_pct=100 * e0 / n0 if n0 else np.nan,
                crowded_exits=int(obs), expected=exp, p=_chi2_p(stat) if np.isfinite(stat) else np.nan)


def crowding_per_run(steps, crowded=4, min_n=3):
    """Per run: exit % of crowded and uncrowded departures (runs with >= min_n of each)."""
    d = steps.assign(crowd=steps["from_state"].map(crowding) >= crowded, ex=steps["to_site"] == EXIT)
    rows = []
    for run, g in d.groupby("run"):
        c, u = g[g["crowd"]], g[~g["crowd"]]
        rows.append(dict(run=run, crowded_n=len(c), crowded_exit_pct=100 * c["ex"].mean() if len(c) else np.nan,
                         uncrowded_n=len(u), uncrowded_exit_pct=100 * u["ex"].mean() if len(u) else np.nan,
                         compared=len(c) >= min_n and len(u) >= min_n))
    return pd.DataFrame(rows)


def memory_test(steps, min_row=MIN_TEST):
    """Per state with >= min_row departures: chi-square of previous site x next site."""
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        chi2_contingency = None
    out = []
    for s, g in steps[~steps["from_state"].str.startswith(SF + "|")].groupby("from_state"):
        if len(g) < min_row:
            continue
        t = pd.crosstab(g["prev_site"].fillna(SF), g["to_site"])
        tk = t[t.sum(axis=1) >= 5]
        tk = tk.loc[:, tk.sum(axis=0) > 0]
        p = np.nan
        if chi2_contingency is not None and tk.shape[0] >= 2 and tk.shape[1] >= 2:
            p = float(chi2_contingency(tk.to_numpy())[1])
        out.append(dict(state=s, n=len(g), p=p, table=t))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _layout(sites, ring):
    """GLU ring on top, lower ring below; one column per subunit when every site has one."""
    pos = {}
    suffixes = [s.split(".")[1] if "." in s else None for s in sites]
    if all(suffixes):
        cols = sorted(set(suffixes), key=lambda x: (not x.upper().startswith("G"), x))
        col = {c: i for i, c in enumerate(cols)}
        for s, sx in zip(sites, suffixes):
            pos[s] = ((col[sx] - (len(cols) - 1) / 2) * 2.4, 2.7 if ring[s] == "GLU" else 0.5)
    else:
        for r, y in (("GLU", 2.6), ("LOWER", 0.6)):
            row = [s for s in sites if (ring[s] == "GLU") == (r == "GLU")]
            for i, s in enumerate(row):
                pos[s] = ((i - (len(row) - 1) / 2) * 3.0, y)
    pos[SF] = (0.0, 4.3)
    pos[EXIT] = (0.0, -1.1)
    return pos


def crowding(state):
    """Other K+ in the cavity of a state: n of SF|n, above + below of site|XaYb (bins: lower bound)."""
    lab = state.split("|", 1)[1] if "|" in state else ""
    if state.startswith(SF + "|"):
        return int(lab.rstrip("+"))
    a, b = lab[:-1].split("a")
    return int(a.rstrip("+")) + int(b.rstrip("+"))


def draw_network(ax, steps, sites, ring, dt, title, min_prob=0.05, label_prob=0.10, net=False, n_ions=None):
    """
    Sites as nodes, one panel. net=False: arrow width = P(next site | site) over the
    departures in steps. net=True: arrow = net flow a -> b (hops a -> b minus b -> a)
    per permeating ion (n_ions), drawn in the net direction; min_prob / label_prob
    are then thresholds in hops per ion. Nodes: departures (= visits), median residue
    time, P(Exit next).
    """
    pos = _layout(sites, ring)
    counts = steps.groupby(["from_site", "to_site"]).size()
    totals = steps.groupby("from_site").size()
    med = steps.groupby("from_site")["residue_frames"].median() * dt
    n_ions = n_ions or max(int(totals.get(SF, 0)), 1)
    top = max([totals.get(s, 0) for s in sites] + [1])
    size = {s: 5000 + 4000 * np.sqrt(totals.get(s, 0) / top) for s in sites}   # big enough for the node text
    size[SF] = size[EXIT] = 5000
    radius = {s: np.sqrt(a) / 2 for s, a in size.items()}
    arrows = []
    for (a, b), c in counts.items():
        if a == b or a not in pos or b not in pos:
            continue
        if net:
            back = counts.get((b, a), 0)
            if c <= back:
                continue
            w = (c - back) / n_ions
            arrows.append((a, b, w, min(w / 0.3, 1), f"{w:.2f}"))
        else:
            w = c / totals[a]
            arrows.append((a, b, w, min(w / 0.4, 1), f"{100 * w:.0f}%"))
    for a, b, w, strength, text in arrows:
        if w < min_prob:
            continue
        color = site_color(a) if a != SF else GREY
        straight = a == SF and b == EXIT
        lw = 0.8 + (12 * w if not net else 25 * min(w, 0.6))
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=12 + 16 * min(w, 1),
                                     lw=lw, color=color, alpha=0.35 + 0.55 * strength,
                                     connectionstyle="arc3,rad=0.0" if straight else "arc3,rad=0.15",
                                     shrinkA=radius[a], shrinkB=radius[b], zorder=1))
        if w >= label_prob:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if not straight:
                mx, my = mx + 0.075 * (y2 - y1), my - 0.075 * (x2 - x1)
            ax.text(mx, my, text, fontsize=12, ha="center", va="center", zorder=4, color=color,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    exits = int(counts.xs(EXIT, level=1).sum()) if len(counts) and EXIT in counts.index.get_level_values(1) else 0
    for s, (x, y) in pos.items():
        ax.scatter([x], [y], s=size[s], color=site_color(s) if s not in (SF, EXIT) else GREY,
                   edgecolor="black", linewidth=1.2, zorder=3)
        if s == SF:
            text = f"SF\n{int(totals.get(SF, 0))} ions"
        elif s == EXIT:
            text = f"Exit\n{exits} exits"
        else:
            dep = totals.get(s, 0)
            loop = counts.get((s, s), 0)
            m = med.get(s, np.nan)
            text = (f"{s}\n{int(dep)} visits\n" + (f"{m:.1f} ns" if np.isfinite(m) else "-")
                    + f"\nexit {100 * counts.get((s, EXIT), 0) / max(dep, 1):.0f}%"
                    + (f"\nloop {100 * loop / max(dep, 1):.0f}%" if loop else ""))
        ax.text(x, y, text, ha="center", va="center", fontsize=11, fontweight="bold", zorder=5)
    xs = [x for x, _ in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(-2.4, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)


def plot_network(steps, sites, ring, dt, title, out_file, min_prob=0.05, label_prob=0.10):
    """Sites as nodes; arrows: P(next site | site), all K+ counts together."""
    fig, ax = plt.subplots(figsize=(17, 12))
    draw_network(ax, steps, sites, ring, dt,
                 f"{title}\narrow width = P(next site | site), all K+ counts together, shown if "
                 f">= {100 * min_prob:.0f}%; node size = visits; in the node: visits, median residue "
                 "time, P(Exit next)", min_prob, label_prob)
    fig.tight_layout()
    fig.savefig(out_file, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_network_crowding(steps, sites, ring, dt, title, out_file, crowded=4, min_prob=0.05, label_prob=0.10):
    """Two networks: departures with < crowded other K+ in the cavity, and with >= crowded."""
    k = steps["from_state"].map(crowding)
    fig, axes = plt.subplots(2, 1, figsize=(17, 23))   # stacked: text stays readable when shrunk
    for ax, sel, name in ((axes[0], k < crowded, f"{crowded - 1} or fewer other K+"),
                          (axes[1], k >= crowded, f"{crowded} or more other K+")):
        g = steps[sel]
        ax_title = (f"departures with {name} in the cavity ({len(g)} departures, "
                    f"{100 * (g['to_site'] == EXIT).mean():.0f}% to Exit)")
        draw_network(ax, g, sites, ring, dt, ax_title, min_prob, label_prob)
    fig.suptitle(f"{title}\nP(next site | site) split by how full the cavity is when the ion leaves "
                 f"(SF: when it enters); arrows >= {100 * min_prob:.0f}%", fontsize=16)
    fig.subplots_adjust(hspace=0.1, top=0.92)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network_net(steps, sites, ring, dt, title, out_file, min_flow=0.02, label_flow=0.05):
    """Net flow: hops a -> b minus b -> a, per permeating ion."""
    n_ions = int((steps["from_site"] == SF).sum())
    fig, ax = plt.subplots(figsize=(17, 12))
    draw_network(ax, steps, sites, ring, dt,
                 f"{title}\nNET FLOW: arrow = (hops a -> b minus b -> a) per permeating ion ({n_ions} ions), "
                 f"in the net direction, shown if >= {min_flow:g}; back-and-forth exchange cancels",
                 min_flow, label_flow, net=True, n_ions=n_ions)
    fig.tight_layout()
    fig.savefig(out_file, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_exit_sources(steps, sites, title, out_file, crowded=4):
    """Where the exits come from: last site before Exit, all and split by cavity filling."""
    ex = steps[steps["to_site"] == EXIT]
    k = ex["from_state"].map(crowding)
    order = [SF] + sites
    labels = ["no binding\n(SF)"] + sites
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    for i, (sel, name, c) in enumerate(((k == k, f"all ({len(ex)})", "#555555"),
                                        (k < crowded, f"{crowded - 1} or fewer other K+ ({int((k < crowded).sum())})", "#7fa7d6"),
                                        (k >= crowded, f"{crowded}+ other K+ ({int((k >= crowded).sum())})", "#c0504d"))):
        g = ex[sel]
        frac = [100 * (g["from_site"] == s).mean() if len(g) else 0 for s in order]
        ax.bar(x + (i - 1) * 0.27, frac, 0.27, color=c, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% of exits")
    ax.legend(title="exits", fontsize=12)
    ax.set_title(f"{title}: where the exits come from (last site before Exit)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def exit_grid(steps, site, max_above, max_below):
    """(P(Exit next) %, departures) arrays [above, below] for one site."""
    g = steps[steps["from_site"] == site]
    P = np.full((max_above + 1, max_below + 1), np.nan)
    N = np.zeros_like(P)
    if not len(g):
        return P, N
    lab = g["from_state"].str.split("|").str[1]
    for a in range(max_above + 1):
        for b in range(max_below + 1):
            key = f"{bin_label(a, max_above)}a{bin_label(b, max_below)}b"
            sel = (lab == key).to_numpy()
            N[a, b] = sel.sum()
            if sel.sum():
                P[a, b] = 100 * (g["to_site"].to_numpy()[sel] == EXIT).mean()
    return P, N


def plot_exit_heatmaps(real, ctrl, sites, max_above, max_below, title, out_file):
    if not sites:
        return
    fig, axes = plt.subplots(2, len(sites), figsize=(3.8 * len(sites) + 1, 10), squeeze=False)
    a_lab = [bin_label(a, max_above) for a in range(max_above + 1)]
    b_lab = [bin_label(b, max_below) for b in range(max_below + 1)]
    for i, (steps, name) in enumerate(((real, "real"), (ctrl, "control"))):
        for j, s in enumerate(sites):
            ax = axes[i, j]
            P, N = exit_grid(steps, s, max_above, max_below)
            ax.set_facecolor("#eeeeee")
            im = ax.imshow(P, origin="lower", cmap="Reds", vmin=0, vmax=100, aspect="auto")
            for a in range(max_above + 1):
                for b in range(max_below + 1):
                    if N[a, b]:
                        ax.text(b, a, f"{P[a, b]:.0f}%\nn={int(N[a, b])}", ha="center", va="center",
                                fontsize=11, color="black" if N[a, b] >= 5 else "#777777")
            ax.set_xticks(range(max_below + 1))
            ax.set_xticklabels(b_lab)
            ax.set_yticks(range(max_above + 1))
            ax.set_yticklabels(a_lab)
            ax.set_xlabel("K+ below")
            if j == 0:
                ax.set_ylabel(f"{name}\nK+ above")
            ax.set_title(s, fontsize=14)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.colorbar(im, ax=axes, fraction=0.02, label="% going to Exit next")
    fig.suptitle(f"{title}\nP(Exit next | site, K+ above, K+ below when leaving); n = departures "
                 "(grey text: n < 5; empty: never seen)")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_first_step(steps, sites, max_sf, title, out_file):
    g = steps[steps["from_site"] == SF]
    if not len(g):
        return
    bins = [bin_label(k, max_sf) for k in range(max_sf + 1)]
    targets = sites + [EXIT]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(bins))
    bottom = np.zeros(len(bins))
    lab = g["from_state"].str.split("|").str[1]
    n = np.array([(lab == b).sum() for b in bins])
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(targets):
        vals = np.array([100 * ((lab == b) & (g["to_site"] == t)).sum() / max((lab == b).sum(), 1)
                         for b in bins])
        ax.bar(x, vals, bottom=bottom, color="#bbbbbb" if t == EXIT else cmap(i % 20), edgecolor="white",
               hatch="//" if t == EXIT else None, label=t)
        bottom += vals
    for xi, ni in zip(x, n):
        ax.text(xi, 101, f"n={ni}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 110)
    ax.set_xlabel("other K+ in the cavity when the ion leaves the SF")
    ax.set_ylabel("% of ions")
    ax.legend(title="first site", fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def visit_table(events, visits_site, sites, max_k, visit=0):
    """
    Where the ions go at visit `visit` (visits.csv numbering, 0 = first visit).
    Ions: those with n_visits >= visit; target = the residue of that visit, or
    Exit (n_visits == visit: left the cavity without it). Split by the other K+
    in the cavity at the step: visit 0 at end_1 (sf_count), visit k when the ion
    leaves visit k-1 (above + below of that row). Columns: start (all,
    0 .. max_k+), n_ions, n_runs, target, count, pct.
    """
    ev = events[events["n_visits"] >= visit]
    v = visits_site.set_index(["run", "event", "visit"]) if len(visits_site) else None
    keys = list(zip(ev["run"], ev["event"]))
    here = v["site"] if v is not None else {}
    target = [here.get((r, e, visit), EXIT) for r, e in keys]
    if visit == 0:
        total = ev["sf_count"].to_numpy()
    else:
        tot = v["above"] + v["below"]
        total = [tot[(r, e, visit - 1)] for r, e in keys]
    ev = ev.assign(start=[bin_label(k, max_k) for k in total], target=target)
    rows = []
    for start in ["all"] + [bin_label(k, max_k) for k in range(max_k + 1)]:
        g = ev if start == "all" else ev[ev["start"] == start]
        c = g["target"].value_counts()
        for t in sites + [EXIT]:
            rows.append(dict(start=start, n_ions=len(g), n_runs=g["run"].nunique(), target=t,
                             count=int(c.get(t, 0)), pct=100 * c.get(t, 0) / len(g) if len(g) else np.nan))
    return pd.DataFrame(rows, columns=["start", "n_ions", "n_runs", "target", "count", "pct"])


def _bold(text):
    """Bold part of a matplotlib title (mathtext)."""
    t = text.replace("%", r"\%").replace("±", r"\pm").replace(" ", r"\ ")
    return r"$\bf{" + t + "}$"


def _mean_sem(n, mean, sem):
    """'9.4 ± 1.1 ns' (or '9.4 ns' for one value)."""
    return f"{mean:.1f} ± {sem:.1f} ns" if n > 1 and np.isfinite(sem) else f"{mean:.1f} ns"


def _pooled(ktime, n_col, mean_col):
    d = ktime[ktime[n_col] > 0]
    n = int(d[n_col].sum())
    return n, ((d[mean_col] * d[n_col]).sum() / n if n else np.nan)


def _ktime_text(ktime, start):
    """'40% of the ions · visit 7.7 ± 1.2 ns (101)' and on a second line 'out 3.1 ± 0.4 ns (65)' ('all': every ion)."""
    if start == "all":
        pct = 100.0
        nv, mean = _pooled(ktime, "n_visits", "mean_ns")
        no, omean = _pooled(ktime, "n_out", "out_mean_ns")
        sem = osem = np.nan
    else:
        r = ktime[ktime["start"] == start]
        if not len(r):
            return ""
        r = r.iloc[0]
        pct, nv, mean, sem = r["pct"], int(r["n_visits"]), r["mean_ns"], r["sem_ns"]
        no, omean, osem = int(r["n_out"]), r["out_mean_ns"], r["out_sem_ns"]
    line = f"{_bold(f'{pct:.0f}%')} of the ions · " + (f"visit {_bold(_mean_sem(nv, mean, sem))} ({nv})" if nv else "no visit")
    return line + (f"\nout {_bold(_mean_sem(no, omean, osem))} ({no})" if no else "")


def visit_summary_table(table, ktime):
    """
    One row per K+ count at the step (and all ions): n_ions, pct_ions, visit time
    (n_visits, mean_ns, sem_ns), and the % of the ions going out / to each residue group
    (E141, E152, D173, N184, ... by sequence number).
    """
    groups = list(dict.fromkeys(group_label(t) for t in table["target"] if t != EXIT))
    rows = []
    for start in table["start"].unique():
        g = table[table["start"] == start]
        n = int(g["n_ions"].iloc[0])
        if not n:
            continue
        row = dict(start=start, n_ions=n)
        if start == "all":
            nv, mean = _pooled(ktime, "n_visits", "mean_ns")
            no, omean = _pooled(ktime, "n_out", "out_mean_ns")
            row.update(pct_ions=100.0, n_visits=nv, mean_ns=mean, sem_ns=np.nan,
                       n_out=no, out_mean_ns=omean, out_sem_ns=np.nan)
        else:
            r = ktime[ktime["start"] == start].iloc[0]
            row.update(pct_ions=r["pct"], n_visits=int(r["n_visits"]), mean_ns=r["mean_ns"], sem_ns=r["sem_ns"],
                       n_out=int(r["n_out"]), out_mean_ns=r["out_mean_ns"], out_sem_ns=r["out_sem_ns"])
        row["out"] = 100 * g.loc[g["target"] == EXIT, "count"].sum() / n
        for grp in groups:
            row[grp] = 100 * g.loc[g["target"].map(group_label) == grp, "count"].sum() / n
        rows.append(row)
    return pd.DataFrame(rows)


def plot_visit_summary(summary, title, out_file, visit=0, min_n=15):
    """The summary table as a figure; rows with fewer than min_n ions in grey."""
    if not len(summary):
        return
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    groups = [c for c in summary.columns if c not in ("start", "n_ions", "pct_ions", "n_visits", "mean_ns", "sem_ns",
                                                      "n_out", "out_mean_ns", "out_sem_ns")]
    cols = ["other K+", "ions", "% of ions", f"visit {visit + 1} time (ns)", "out time (ns)"] + groups
    cells = []
    for r in summary.to_dict("records"):
        t = "–" if not r["n_visits"] else (f"{r['mean_ns']:.1f} ± {r['sem_ns']:.1f}" if np.isfinite(r["sem_ns"])
                                           else f"{r['mean_ns']:.1f}")
        o = "–" if not r["n_out"] else (f"{r['out_mean_ns']:.1f} ± {r['out_sem_ns']:.1f}"
                                         if np.isfinite(r["out_sem_ns"]) else f"{r['out_mean_ns']:.1f}")
        cells.append(["all ions" if r["start"] == "all" else r["start"], str(r["n_ions"]), f"{r['pct_ions']:.0f}%", t, o]
                     + [f"{r[g]:.0f}%" for g in groups])
    widths = [1.3, 1.0, 1.3, 2.3, 2.1] + [1.15] * len(groups)
    fig, ax = plt.subplots(figsize=(1.1 * sum(widths), 0.8 + 0.42 * (len(cells) + 1)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[x / sum(widths) for x in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(16)
    tb.scale(1, 2.2)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#C9D3D8")
        if i == 0:
            c.set_facecolor("#123B4A")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        else:
            if summary.iloc[i - 1]["start"] == "all" or j in (2, 3, 4):   # all ions, % of ions, times
                c.get_text().set_fontweight("bold")
            if summary.iloc[i - 1]["n_ions"] < min_n:
                c.get_text().set_color("#9a9a9a")
    ax.set_title(f"{title}\nvisit {visit + 1}: where the ions go (% of the ions), by the other K+ in the cavity {when}"
                 f"\n(grey: fewer than {min_n} ions)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def panel_grid(n, max_col=4):
    """(nrow, ncol, legend_inside) with the fewest empty panels: the legend fills one empty
    panel when there is one; among equally full grids the one closest to 3:2 wins."""
    best = None
    for ncol in range(1, min(max_col, n) + 1):
        nrow = int(np.ceil(n / ncol))
        empty = nrow * ncol - n
        waste = max(empty - 1, 0)
        key = (waste, abs(ncol / nrow - 1.5))
        if best is None or key < best[0]:
            best = (key, nrow, ncol, empty >= 1)
    return best[1], best[2], best[3]


def plot_visit(table, sites, title, out_file, visit=0, which=None, ktime=None):
    """One panel for all ions and one per K+ count at the step: % of the ions whose visit `visit` is each
    residue, or out. which: the panels to draw (e.g. ["all"]); default all of them. ktime
    (visit_k_time_table): adds the % of the ions at that K+ count and how long the visit lasted."""
    starts = [s for s in table["start"].unique() if table.loc[table["start"] == s, "n_ions"].iloc[0] > 0
              and (which is None or s in which)]
    if not starts:
        return
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"   # labels count from 1
    targets = sites + [EXIT]
    nrow, ncol, legend_inside = panel_grid(len(starts))
    size = (9, 6.4) if len(starts) == 1 else (6.2 * ncol, 5.0 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
    for ax in axes.flat[len(starts):]:
        ax.axis("off")
    x = np.arange(len(targets))
    top = table.loc[table["start"].isin(starts), "pct"].max()      # same y scale in every panel
    ymax = min(135, (top if np.isfinite(top) else 100) + 22)
    for ax, start in zip(axes.flat, starts):
        g = table[table["start"] == start].set_index("target").reindex(targets)
        ax.bar(x, g["pct"], color=[site_color(t) if t != EXIT else "#d9d9d9" for t in targets],
               hatch=["////" if t == EXIT else None for t in targets], edgecolor="#555555", linewidth=0.5)
        for xi, (c, p) in enumerate(zip(g["count"], g["pct"])):
            if c:
                ax.text(xi, p + 1, f"{p:.0f}%\n({int(c)})", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([t if t != EXIT else "out" for t in targets], rotation=60, ha="right", fontsize=12)
        ax.set_ylim(0, ymax)
        ax.set_yticks(range(0, int(min(ymax - 10, 100)) + 1, 20))
        ax.set_ylabel("% of ions")
        ax.set_xlabel(f"visit {visit + 1}")
        name = "all ions" if start == "all" else f"{start} other K+ in the cavity {when}"
        line3 = "" if ktime is None else "\n" + _ktime_text(ktime, start)
        ax.set_title(f"{name}\n(n = {int(g['n_ions'].iloc[0])} ions, {int(g['n_runs'].iloc[0])} runs){line3}", fontsize=14)
    from matplotlib.patches import Patch
    kinds = [(k, v) for k, v in (("GLU", "E"), ("GLN", "Q"), ("ASP", "D"), ("ASN", "N"))
             if any(t[:1] == v for t in sites)]
    handles = [Patch(color=COLORS[v], label=k) for k, v in kinds] + \
              [Patch(facecolor="#d9d9d9", hatch="////", edgecolor="#555555", label="out")]
    split = "" if starts == ["all"] else f", by the other K+ in the cavity {when}"
    fig.suptitle(f"{title}\nvisit {visit + 1}{split}", fontsize=17)
    fig.tight_layout()
    if legend_inside:                                            # in the first empty panel
        axes.flat[len(starts)].legend(handles=handles, title=f"visit {visit + 1}", loc="center",
                                      fontsize=16, title_fontsize=17)
    else:
        fig.legend(handles=handles, title=f"visit {visit + 1}", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visit_k_time_table(events, visits_site, max_k, dt, visit=0):
    """
    Visit `visit`, per other-K+ count at the step (as in visit_table): ions (n_ions, pct of
    the ions reaching this visit), how long the visit lasted for those that made it
    (n_visits, mean_ns, sem_ns, median_ns of residue_frames), and for those that went out
    instead how long they took to leave the cavity from where they were - the SF (visit 0)
    or visit-1 (n_out, out_mean_ns, out_sem_ns, out_median_ns).
    """
    ev = events[events["n_visits"] >= visit]
    v = visits_site.set_index(["run", "event", "visit"]) if len(visits_site) else None
    keys = list(zip(ev["run"], ev["event"]))
    if visit == 0:
        total = ev["sf_count"].to_numpy()
    else:
        tot = v["above"] + v["below"]
        total = [tot[(r, e, visit - 1)] for r, e in keys]
    res = v["residue_frames"] if v is not None else {}
    dur = np.array([res.get((r, e, visit), np.nan) for r, e in keys], dtype=float) * dt
    if visit == 0:
        out_f = np.where(ev["n_visits"] == 0, ev["moving_first"], np.nan)
    else:
        mov = v["moving_after"]
        out_f = np.array([mov[(r, e, visit - 1)] if n == visit else np.nan
                          for (r, e), n in zip(keys, ev["n_visits"])], dtype=float)
    out_t = np.asarray(out_f, dtype=float) * dt
    lab = np.array([bin_label(k, max_k) for k in total])
    rows = []
    for s in [bin_label(k, max_k) for k in range(max_k + 1)]:
        m = lab == s
        d = dur[m][np.isfinite(dur[m])]
        o = out_t[m][np.isfinite(out_t[m])]
        rows.append(dict(start=s, n_ions=int(m.sum()), pct=100 * m.sum() / max(len(lab), 1), n_visits=len(d),
                         mean_ns=d.mean() if len(d) else np.nan,
                         sem_ns=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                         median_ns=np.median(d) if len(d) else np.nan,
                         n_out=len(o), out_mean_ns=o.mean() if len(o) else np.nan,
                         out_sem_ns=o.std(ddof=1) / np.sqrt(len(o)) if len(o) > 1 else np.nan,
                         out_median_ns=np.median(o) if len(o) else np.nan))
    return pd.DataFrame(rows)


def plot_visit_k_time(tab, title, out_file, visit=0):
    """Per other-K+ count: % of the ions (left axis) and how long visit `visit` lasted, mean +- SE (right axis)."""
    tab = tab[tab["n_ions"] > 0].reset_index(drop=True)
    if not len(tab):
        return
    x = np.arange(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, tab["pct"], w, color="#7f7f7f", label="% of ions")
    b2 = ax2.bar(x + w / 2, tab["mean_ns"], w, yerr=tab["sem_ns"], capsize=4, color="#4f81bd",
                 error_kw=dict(lw=1.5), label=f"visit {visit + 1} time (mean ± SE)")
    top = max(tab["pct"].max(), 10)
    for xi, (p, c) in enumerate(zip(tab["pct"], tab["n_ions"])):
        ax.text(xi - w / 2, p + 0.01 * top, f"({c})", ha="center", va="bottom", fontsize=12)
        ax.text(xi - w / 2, p + 0.065 * top, f"{p:.0f}%", ha="center", va="bottom", fontsize=17, fontweight="bold")
    t_top = np.nanmax((tab["mean_ns"] + tab["sem_ns"].fillna(0)).to_numpy()) if tab["mean_ns"].notna().any() else 1
    for xi, (m, e, nv) in enumerate(zip(tab["mean_ns"], tab["sem_ns"].fillna(0), tab["n_visits"])):
        if np.isfinite(m):
            ax2.text(xi + w / 2, m + e + 0.01 * t_top, f"{_ns_label(m)}\n({nv})", ha="center", va="bottom",
                     fontsize=12, color="#2f5597", fontweight="bold")
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    ax.set_xticks(x)
    ax.set_xticklabels(tab["start"], fontsize=17)
    ax.set_xlabel(f"other K+ in the cavity {when}", fontsize=17)
    ax.set_ylabel("% of ions", fontsize=19)
    ax2.set_ylabel(f"visit {visit + 1} time (ns)", color="#4f81bd", fontsize=19)
    ax.tick_params(axis="y", labelsize=17)
    ax2.tick_params(axis="y", colors="#4f81bd", labelsize=17)
    ax.set_ylim(0, top * 1.5)
    ax2.set_ylim(0, t_top * 1.6)
    ax.legend(handles=[b1, b2], loc="upper left", fontsize=15)
    ax.set_title(f"{title}\nvisit {visit + 1}: ions by the other K+ in the cavity, and how long the visit lasted",
                 fontsize=17)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _occ_set(text):
    """Residues holding another K+ ('E141.G1;D173.G1' -> {'E141.G1', 'D173.G1'})."""
    return set(t for t in str(text).split(";") if t and t != "nan")


def occupancy_steps(events, visits_raw, visits_site, max_k, visit=0, rule=""):
    """
    One row per ion reaching visit `visit` (as in visit_table): start (other K+ bin at the
    step), occupied (residue groups holding another K+ at that moment: when the ion leaves
    the SF for visit 0, when it leaves visit-1 otherwise) and target (residue group of the
    visit, or Exit). rule: occupancy column suffix ("" trip window, "_half" half of the trip, "_fixed" the same fixed
    frames after leaving for every ion, "_frame" the frame it leaves).
    """
    ev = events[events["n_visits"] >= visit]
    keys = list(zip(ev["run"], ev["event"]))
    raw = visits_raw.set_index(["run", "event", "visit"]) if len(visits_raw) else None
    here = visits_site.set_index(["run", "event", "visit"])["site"] if len(visits_site) else {}
    here_raw = raw["site"] if raw is not None else {}
    if visit == 0:
        if f"occupied_sf{rule}" not in ev:
            return pd.DataFrame()
        total, occ = ev["sf_count"].to_numpy(), ev[f"occupied_sf{rule}"].to_numpy()
    else:
        tot = raw["above"] + raw["below"]
        total = [tot[(r, e, visit - 1)] for r, e in keys]
        if f"occupied_leave{rule}" not in raw:
            return pd.DataFrame()
        occ = [raw[f"occupied_leave{rule}"][(r, e, visit - 1)] for r, e in keys]
    st = pd.DataFrame(dict(
        start=[bin_label(k, max_k) for k in total],
        known=[isinstance(o, str) for o in occ],
        occupied=[{group_label(x) for x in _occ_set(o)} if isinstance(o, str) else set() for o in occ],
        occupied_sites=[_occ_set(o) if isinstance(o, str) else set() for o in occ],
        target=[group_label(here.get((r, e, visit), EXIT)) for r, e in keys],
        target_site=[here_raw.get((r, e, visit), EXIT) for r, e in keys]))
    return st[st["known"]].drop(columns="known")             # ions from runs without occupancy left out


def occupancy_tables(st, groups, max_k, members=None):
    """(occupied, test): % of the ions seeing each group occupied, per K+ bin; and per group and
    K+ bin, % going to the group and % going out when the group is free vs occupied."""
    bins = ["all"] + [bin_label(k, max_k) for k in range(max_k + 1)]
    occ_rows, test_rows = [], []
    for b in bins:
        g = st if b == "all" else st[st["start"] == b]
        if not len(g):
            continue
        row = dict(start=b, n_ions=len(g))
        for grp in groups:
            row[grp] = 100 * g["occupied"].map(lambda o: grp in o).mean()
        for site in (members or []):                            # each residue on its own
            row[site] = 100 * g["occupied_sites"].map(lambda o: site in o).mean()
        occ_rows.append(row)
        for grp in groups:
            busy = g["occupied"].map(lambda o: grp in o)
            r = dict(group=grp, start=b)
            for name, m in (("free", ~busy), ("occupied", busy)):
                sub = g[m]
                r[f"n_{name}"] = len(sub)
                r[f"to_group_{name}"] = 100 * (sub["target"] == grp).mean() if len(sub) else np.nan
                r[f"out_{name}"] = 100 * (sub["target"] == EXIT).mean() if len(sub) else np.nan
            test_rows.append(r)
    return pd.DataFrame(occ_rows), pd.DataFrame(test_rows)


def residue_test(st, residues, max_k):
    """Per residue and K+ bin: % going to that residue and % going out when that same residue is free
    vs held by another K+ (same columns as occupancy_tables' test, 'group' = the residue)."""
    rows = []
    for b in ["all"] + [bin_label(k, max_k) for k in range(max_k + 1)]:
        g = st if b == "all" else st[st["start"] == b]
        if not len(g):
            continue
        for res in residues:
            busy = g["occupied_sites"].map(lambda o: res in o)
            r = dict(group=res, start=b)
            for name, m in (("free", ~busy), ("occupied", busy)):
                sub = g[m]
                r[f"n_{name}"] = len(sub)
                r[f"to_group_{name}"] = 100 * (sub["target_site"] == res).mean() if len(sub) else np.nan
                r[f"out_{name}"] = 100 * (sub["target_site"] == EXIT).mean() if len(sub) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def plot_occupancy_by_k(occ, groups, title, out_file, visit=0):
    """x = other K+ in the cavity, one bar per residue group = % of the ions that see it occupied."""
    occ = occ[occ["start"] != "all"].reset_index(drop=True)
    if not len(occ):
        return
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    x = np.arange(len(occ))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(occ) + 3), 6.5))
    for i, grp in enumerate(groups):
        off = x - 0.4 + w * (i + 0.5)
        ax.bar(off, occ[grp], w, color=site_color(grp), edgecolor="#555555", linewidth=0.5,
               alpha=1 - 0.18 * (i % 2), label=grp)
        for xi, v in zip(off, occ[grp]):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(occ["start"], occ["n_ions"])], fontsize=15)
    ax.set_xlabel(f"other K+ in the cavity {when}", fontsize=17)
    ax.set_ylabel("% of ions seeing it occupied", fontsize=18)
    ax.set_ylim(0, min(110, occ[groups].to_numpy().max() + 18))
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(title="residue holding another K+", fontsize=14, title_fontsize=14, ncol=len(groups),
              loc="upper left")
    ax.set_title(f"{title}\nvisit {visit + 1}: which residues already hold another K+", fontsize=17)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_free_vs_occupied(test, groups, title, out_file, visit=0, min_n=5):
    """Per residue group (columns): % of the ions going to it (top) and going out (bottom) when it is
    free vs occupied by another K+, per other-K+ count; bars from fewer than min_n ions left out."""
    t = test[test["start"] != "all"]
    starts = [b for b in dict.fromkeys(t["start"]) if
              (t.loc[t["start"] == b, ["n_free", "n_occupied"]].to_numpy() >= min_n).any()]
    if not starts:
        return
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    fig, axes = plt.subplots(2, len(groups), figsize=(5.0 * len(groups), 10), squeeze=False, sharey="row")
    x = np.arange(len(starts))
    w = 0.38
    for j, grp in enumerate(groups):
        g = t[t["group"] == grp].set_index("start").reindex(starts)
        for i, (what, ylab) in enumerate((("to_group", "% going to it"), ("out", "% going out"))):
            ax = axes[i, j]
            for k, (name, alpha, hatch) in enumerate((("free", 0.35, None), ("occupied", 1.0, "//"))):
                n = g[f"n_{name}"].fillna(0).to_numpy()
                v = np.where(n >= min_n, g[f"{what}_{name}"], np.nan)
                col = site_color(grp) if what == "to_group" else "#7f7f7f"
                ax.bar(x - w / 2 + k * w, v, w, color=col, alpha=alpha, hatch=hatch, edgecolor="#333333",
                       linewidth=0.5, label=f"{grp} {name}")
                for xi, vi, ni in zip(x - w / 2 + k * w, v, n):
                    if np.isfinite(vi):
                        ax.text(xi, vi + 1, f"{vi:.0f}%\n({int(ni)})", ha="center", va="bottom", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(starts, fontsize=14)
            if i == 0:
                ax.set_title(grp, fontsize=18, fontweight="bold")
            else:
                ax.set_xlabel("other K+", fontsize=15)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=16)
            ax.tick_params(axis="y", labelsize=13)
            ax.set_ylim(0, 118)
            ax.set_xlim(-0.6, len(starts) - 0.4)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#bbbbbb", alpha=0.35, edgecolor="#333333", label="residue free"),
                        Patch(facecolor="#555555", hatch="//", edgecolor="#333333", label="residue holds another K+")],
               loc="upper center", fontsize=14, ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}\nvisit {visit + 1}: residue free vs already holding another K+, by the other K+ in "
                 f"the cavity {when}\n(bars from < {min_n} ions left out; (n) = ions)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_free_vs_occupied_table(rtest, residues, title, out_file, visit=0, min_bin=15, min_n=5, note=None,
                                visits=False):
    """Table, one row per residue: % of the ions going to it and % going out when that residue is free
    -> holds another K+, for all ions and per other-K+ count (counts with >= min_bin ions); cells from
    fewer than min_n ions on a side in grey; effects from all ions (going to it: occupied / free >= 1.5
    attracts, <= 0.67 blocks; going out: a change of 8 points or more). rtest can also be a dict
    {occupancy rule: table}: one row per residue and rule (the robustness comparison).
    visits=True: the "-> it" cells instead give, of the ions that went to the residue, the % that went
    while it held another K+ (in brackets: the % of all ions seeing it occupied, the value expected
    if occupancy made no difference); grey when fewer than min_n ions went there."""
    tests = rtest if isinstance(rtest, dict) else {None: rtest}
    tests = {k: v[v["group"].isin(residues)] for k, v in tests.items()}
    t = next(iter(tests.values()))
    if not len(t):
        return
    n_bin = t.groupby("start")[["n_free", "n_occupied"]].first().sum(axis=1)
    bins = ["all"] + [b for b in dict.fromkeys(t["start"]) if b != "all" and n_bin.get(b, 0) >= min_bin]
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    lab = lambda b: "all ions" if b == "all" else f"{b} other K+"
    multi = len(tests) > 1
    cols = ["residue"] + (["occupancy\nrule"] if multi else []) + ["ions\nfree / occ."] + \
           (["went there\nfree / occ."] if visits else []) + \
           [(f"went there: occ.\n{lab(b)}" if visits else f"→ it\n{lab(b)}") for b in bins] + \
           [f"out\n{lab(b)}" for b in bins] + ["K+ there:\ngoing to it", "K+ there:\ngoing out"]
    cells, grey, shade = [], [], []
    for ri, (res, rule) in enumerate((r, k) for r in residues for k in tests):
        tt = tests[rule]
        g = tt[tt["group"] == res].set_index("start")
        if "all" not in g.index:
            continue
        a = g.loc["all"]
        first = rule == next(iter(tests))
        c = [res if first else ""] + ([rule] if multi else []) + [f"{int(a['n_free'])} / {int(a['n_occupied'])}"]
        if visits:
            c.append(f"{a['n_free'] * a['to_group_free'] / 100 if a['n_free'] else 0:.0f} / "
                     f"{a['n_occupied'] * a['to_group_occupied'] / 100 if a['n_occupied'] else 0:.0f}")
        gr = [False] * len(c)
        shade.append(residues.index(res) % 2 == 1)
        for what in ("to_group", "out"):
            for b in bins:
                if visits and what == "to_group":
                    r = g.loc[b] if b in g.index else None
                    nf, no = (r["n_free"], r["n_occupied"]) if r is not None else (0, 0)
                    vf = nf * r["to_group_free"] / 100 if nf else 0
                    vo = no * r["to_group_occupied"] / 100 if no else 0
                    if vf + vo < 0.5:
                        c.append("–"); gr.append(False); continue
                    c.append(f"{100 * vo / (vf + vo):.0f}% ({100 * no / (nf + no):.0f}%)")
                    gr.append(vf + vo < min_n)
                    continue
                if b not in g.index or not g.loc[b, "n_free"] or not g.loc[b, "n_occupied"]:
                    c.append("–"); gr.append(False); continue
                r = g.loc[b]
                c.append(f"{r[f'{what}_free']:.0f}% → {r[f'{what}_occupied']:.0f}%")
                gr.append(min(r["n_free"], r["n_occupied"]) < min_n)
        f, o = a["to_group_free"], a["to_group_occupied"]
        if not (a["n_free"] and a["n_occupied"]) or not np.isfinite(f) or not np.isfinite(o) or max(f, o) < 5:
            eff = "–"
        elif o >= 1.5 * max(f, 0.1):
            eff = "attracts"
        elif o <= f / 1.5:
            eff = "blocks"
        else:
            eff = "no clear effect"
        fo, oo = a["out_free"], a["out_occupied"]
        if not (a["n_free"] and a["n_occupied"]) or not np.isfinite(fo) or not np.isfinite(oo):
            eff_out = "–"
        elif oo <= fo - 8:
            eff_out = "fewer leave"
        elif oo >= fo + 8:
            eff_out = "more leave"
        else:
            eff_out = "no clear effect"
        c += [eff, eff_out]; gr += [False, False]
        cells.append(c); grey.append(gr)
    if not cells:
        return
    off = (1 if multi else 0) + (1 if visits else 0)            # extra "occupancy rule" / "went there" columns
    widths = [1.6] + ([2.3] if multi else []) + [1.6] + ([1.6] if visits else []) + \
             [2.1 if visits else 2.0] * len(bins) + [2.0] * len(bins) + [2.2, 2.2]
    fig, ax = plt.subplots(figsize=(1.0 * sum(widths), 1.0 + 0.42 * (len(cells) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[x / sum(widths) for x in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(16)
    tb.scale(1, 2.3)
    out_cols = set(range(2 + off + len(bins), 2 + off + 2 * len(bins)))
    eff_cols = (len(cols) - 2, len(cols) - 1)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#C9D3D8")
        if i == 0:
            cell.set_facecolor("#5B6770" if j in out_cols else "#123B4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 1.9)
            continue
        if j in out_cols:
            cell.set_facecolor("#EEF3F5")
        elif multi and shade[i - 1]:
            cell.set_facecolor("#F6F6F6")
        if j in (0, 2 + off, 2 + off + len(bins)) or j in eff_cols:
            cell.get_text().set_fontweight("bold")
        if j in eff_cols:
            cell.get_text().set_color({"attracts": "#1a7f37", "blocks": "#c0392b", "more leave": "#1a7f37",
                                       "fewer leave": "#c0392b"}.get(cells[i - 1][j], "#333333"))
        if grey[i - 1][j]:
            cell.get_text().set_color("#9a9a9a")
    head = (f"visit {visit + 1}: of the ions that went to each residue, % that went while it already held another "
            f"K+ (in brackets: % of all ions seeing it occupied, expected if it made no difference),\nand % going out "
            f"when the residue is free → holds another K+, by the other K+ in the cavity {when}\n") if visits else (
            f"visit {visit + 1}: % of the ions going to each residue (→ it) and going out, when that "
            f"residue is free → already holds another K+, by the other K+ in the cavity {when}\n")
    ax.set_title(f"{title}\n" + head +
                 f"(K+ there, from all ions: going to it occupied ≥ 1.5× free attracts, ≤ free / 1.5 blocks (– below 5%); "
                 f"going out ± 8 points; grey: fewer than "
                 f"{min_n} ions on a side)" + (f"\n{note}" if note else ""), fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def arrival_sharing(visits_raw, model_names):
    """Per site: visits, % arriving while the same residue already holds another K+, and of those
    the % where it is free again when the ion leaves (the other K+ has moved: knock-on)."""
    rows = []
    if "occupied_arrive" not in visits_raw:
        return pd.DataFrame(rows)
    for site, g in visits_raw[visits_raw["occupied_arrive"].map(lambda o: isinstance(o, str))].groupby("site"):
        shared = g["occupied_arrive"].map(lambda o: site in _occ_set(o))
        leave = "occupied_leave_frame" if "occupied_leave_frame" in g else "occupied_leave"   # same rule as arrival
        still = g.loc[shared, leave].map(lambda o: site in _occ_set(o))
        rows.append(dict(site=site, visits=len(g), pct_shared_at_arrival=100 * shared.mean(),
                         n_shared=int(shared.sum()),
                         pct_other_gone_at_leave=100 * (~still).mean() if len(still) else np.nan))
    return pd.DataFrame(rows)


def plot_occupancy_summary(summary, occ, groups, title, out_file, visit=0, min_n=15, members=None, site_pct=None):
    """The visit summary table with, next to each residue group's % of the ions going to it, the % of
    the ions that saw it already holding another K+ (from the ions of runs with occupancy)."""
    occ = occ.set_index("start")
    when = "when the ion leaves the SF" if visit == 0 else f"when leaving visit {visit}"
    cols = ["other K+", "ions", "% of ions", f"visit {visit + 1}\ntime (ns)", "out\ntime (ns)", "out"]
    members = members or {}
    def sub(g):
        return [m for m in members.get(g, []) if m in occ.columns]
    def sub_to(g):                                              # residues with their own visit %
        return [m for m in members.get(g, []) if site_pct is not None and m in site_pct.columns]
    for g in groups:
        ms = sub(g)
        tag = " / ".join(m.split(".", 1)[1] for m in ms) if len(ms) > 1 else ""
        mt = sub_to(g)
        tag_to = " / ".join(m.split(".", 1)[1] for m in mt) + " / all" if len(mt) > 1 else ""
        cols += [f"→ {g}\n{tag_to}" if tag_to else f"→ {g}", f"{g} occupied\n{tag}" if tag else f"{g}\noccupied"]
    cols.append("ions with\noccupancy")
    rows, cells = [], []
    for r in summary.to_dict("records"):
        o = occ.loc[r["start"]] if r["start"] in occ.index else None
        t = "–" if not r["n_visits"] else (f"{r['mean_ns']:.1f} ± {r['sem_ns']:.1f}" if np.isfinite(r["sem_ns"])
                                           else f"{r['mean_ns']:.1f}")
        ot = "–" if not r["n_out"] else (f"{r['out_mean_ns']:.1f} ± {r['out_sem_ns']:.1f}"
                                         if np.isfinite(r["out_sem_ns"]) else f"{r['out_mean_ns']:.1f}")
        c = ["all ions" if r["start"] == "all" else r["start"], str(r["n_ions"]), f"{r['pct_ions']:.0f}%", t, ot,
             f"{r['out']:.0f}%"]
        row = dict(start=r["start"], n_ions=r["n_ions"], pct_ions=r["pct_ions"], out=r["out"])
        for g in groups:
            occ_v = o[g] if o is not None else np.nan
            ms = sub(g)
            if o is None:
                txt = "–"
            elif len(ms) > 1:                                   # one number per residue of the group
                txt = " / ".join(f"{o[m]:.0f}" for m in ms) + "%"
                for m in ms:
                    row[f"{m}_occupied"] = o[m]
            else:
                txt = f"{occ_v:.0f}%"
            mt = sub_to(g)
            if len(mt) > 1 and r["start"] in site_pct.index:     # one number per residue, then the group
                sp = site_pct.loc[r["start"]]
                to_txt = " / ".join(f"{sp[m]:.0f}" for m in mt) + f" / {r.get(g, 0):.0f}%"
                for m in mt:
                    row[f"to_{m}"] = sp[m]
            else:
                to_txt = f"{r.get(g, 0):.0f}%"
            c += [to_txt, txt]
            row[f"to_{g}"], row[f"{g}_occupied"] = r.get(g, np.nan), occ_v
        n_occ = int(o["n_ions"]) if o is not None else 0
        c.append(str(n_occ))
        row["n_ions_occupancy"] = n_occ
        cells.append(c)
        rows.append(row)
    widths = [1.3, 0.9, 1.15, 1.9, 1.7, 0.95] + sum(([2.6 if len(sub_to(g)) > 1 else 1.0,
                                                      2.1 if len(sub(g)) > 1 else 1.25] for g in groups), []) + [1.35]
    fig, ax = plt.subplots(figsize=(1.0 * sum(widths), 0.9 + 0.36 * (len(cells) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[x / sum(widths) for x in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(15)
    tb.scale(1, 2.2)
    occ_cols = {6 + 2 * i + 1 for i in range(len(groups))}
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#C9D3D8")
        if i == 0:
            cell.set_facecolor("#5B6770" if j in occ_cols else "#123B4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 2.1)
            continue
        if j in occ_cols:
            cell.set_facecolor("#EEF3F5")
        if rows[i - 1]["start"] == "all" or j in (2, 3, 4):
            cell.get_text().set_fontweight("bold")
        if rows[i - 1]["n_ions"] < min_n:
            cell.get_text().set_color("#9a9a9a")
    ax.set_title(f"{title}\nvisit {visit + 1}: where the ions go (→, % of the ions) and which residues already hold "
                 f"another K+ (occupied, % of the ions), by the other K+ in the cavity {when}\n"
                 f"(occupied from the ions of runs with occupancy; grey: fewer than {min_n} ions)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def occupancy_analysis(events, visits_raw, visits_site, sites, max_k, title, out_dir, dt=None, occ_frames=10):
    """Residue occupancy by the other K+ at each step, written to out_dir/occupancy/."""
    if "occupied_sf" not in events or events["occupied_sf"].isna().all():
        print("site_chain: no residue occupancy in the frames (collected with an older version); "
              "run the site chain from main.py again to add it")
        return
    missing = sorted(events.loc[events["occupied_sf"].isna(), "run"].unique())
    if missing:
        print(f"site_chain: runs without residue occupancy (run main.py again for them): "
              f"{', '.join(f'RUN{r}' for r in missing)}; their ions are left out of occupancy/")
    od = Path(out_dir) / "occupancy"
    od.mkdir(exist_ok=True)
    groups = list(dict.fromkeys(group_label(s) for s in sites))
    raw_sites = set(visits_raw["site"]) if len(visits_raw) else set()
    for col, df in (("occupied_sf", events), ("occupied_leave", visits_raw)):   # also residues never visited
        if col in df:
            raw_sites |= set().union(*[_occ_set(o) for o in df[col] if isinstance(o, str)] or [set()])
    members = {g: sorted(m for m in raw_sites if group_label(m) == g) for g in groups}
    all_members = [m for g in groups for m in members[g]]
    rep = _Report()
    rep(f"RESIDUE OCCUPANCY  {title}")
    rep("occupied = another K+ in the cavity has the residue as its closest one within the cutoff (the same rule as a\n"
        "contact) for at least as many frames as a stay (or all of them when shorter) while the ion moves from where it\n"
        "leaves - the SF (visit 1) or its previous visit (visit k) - to its next visit or the exit; if it goes straight\n"
        "on, at the frame it leaves")
    rep_rob = []
    for k in PLOT_VISITS:
        st = occupancy_steps(events, visits_raw, visits_site, max_k, k)
        if not len(st):
            continue
        occ, test = occupancy_tables(st, groups, max_k, all_members)
        name = f"visit_{k + 1}"
        occ.to_csv(od / f"{name}_occupied_by_k.csv", index=False, float_format="%.2f")
        test.to_csv(od / f"{name}_free_vs_occupied.csv", index=False, float_format="%.2f")
        plot_occupancy_by_k(occ, groups, title, od / f"{name}_occupied_by_k.png", k)
        plot_free_vs_occupied(test, groups, title, od / f"{name}_free_vs_occupied.png", k)
        rtest = residue_test(st, all_members, max_k)          # each residue against itself
        rtest.to_csv(od / f"{name}_free_vs_occupied_residue.csv", index=False, float_format="%.2f")
        plot_free_vs_occupied_table(rtest, all_members, title, od / f"{name}_free_vs_occupied_table.png", k)
        plot_free_vs_occupied_table(rtest, all_members, title, od / f"{name}_free_vs_occupied_visits.png", k,
                                    visits=True)
        rob = {"trip (main)": rtest}                           # robustness: other occupancy rules
        for rule, lab in (("_half", "half of trip"), ("_fixed", "fixed frames"), ("_frame", "leave frame")):
            st_r = occupancy_steps(events, visits_raw, visits_site, max_k, k, rule)
            if len(st_r):
                rob[lab] = residue_test(st_r, all_members, max_k)
        if len(rob) > 1:
            rd = od / "robustness"
            rd.mkdir(exist_ok=True)
            pd.concat([v.assign(rule=lab) for lab, v in rob.items()]).to_csv(
                rd / f"{name}_free_vs_occupied_rules.csv", index=False, float_format="%.2f")
            when = "the SF" if k == 0 else f"visit {k}"
            plot_free_vs_occupied_table(
                rob, all_members, title, rd / f"{name}_free_vs_occupied_rules.png", k,
                note=f"occupancy rules: trip (main) = another K+ there ≥ {occ_frames} frames of the "
                     f"trip to the next stop (all if shorter); half of trip = there in ≥ half of the trip's frames; fixed frames = there in ≥ half of the same "
                     f"{occ_frames} frames after leaving {when} for every ion; leave frame = there "
                     f"at the frame it leaves {when}")
            rep_rob.append((k, rob))
        for g in groups:
            if len(members[g]) > 1:
                plot_free_vs_occupied(rtest, members[g], title, od / f"{name}_free_vs_occupied_{g}.png", k)
        if dt is not None:
            vt = visit_table(events, visits_site, sites, max_k, k)
            site_pct = vt.pivot_table(index="start", columns="target", values="pct", aggfunc="first")
            summ = visit_summary_table(vt, visit_k_time_table(events, visits_site, max_k, dt, k))
            plot_occupancy_summary(summ, occ, groups, title, od / f"{name}_summary_occupancy.png", k,
                                   members=members, site_pct=site_pct).to_csv(
                od / f"{name}_summary_occupancy.csv", index=False, float_format="%.2f")
        rep(f"\nVISIT {k + 1}: % of the ions seeing each residue group occupied, by other K+")
        rep("    other K+     n  " + "".join(f"{g:>8}" for g in groups))
        for r in occ.to_dict("records"):
            rep(f"    {r['start']:>8} {r['n_ions']:>5}  " + "".join(f"{r[g]:>7.0f}%" for g in groups))
        rep("  going to the group / going out, when it is free vs occupied (all K+ counts together):")
        for r in test[test["start"] == "all"].to_dict("records"):
            rep(f"    {r['group']:>8}: free (n={r['n_free']:>3}) to it {r['to_group_free']:5.1f}%  out "
                f"{r['out_free']:5.1f}%   |   occupied (n={r['n_occupied']:>3}) to it {r['to_group_occupied']:5.1f}%"
                f"  out {r['out_occupied']:5.1f}%")
        rep("  each residue against itself: going to it / going out when that residue is free vs occupied (all K+):")
        for r in rtest[rtest["start"] == "all"].to_dict("records"):
            rep(f"    {r['group']:>8}: free (n={r['n_free']:>3}) to it {r['to_group_free']:5.1f}%  out "
                f"{r['out_free']:5.1f}%   |   occupied (n={r['n_occupied']:>3}) to it {r['to_group_occupied']:5.1f}%"
                f"  out {r['out_occupied']:5.1f}%")
    if rep_rob:
        rep(f"\nROBUSTNESS: the same per-residue test with other occupancy rules (all K+ counts together;\n"
            f"trip (main) = another K+ there >= {occ_frames} frames of the trip to the next stop, all if shorter;\n"
            f"half of trip = there in >= half of the frames of the trip, whatever its length;\n"
            f"fixed frames = there in >= half of the same {occ_frames} frames after leaving, for every ion;\n"
            f"leave frame = there at the frame the ion leaves); to it / out: free -> occupied")
        for k, rob in rep_rob:
            rep(f"  VISIT {k + 1}")
            rep("    residue    " + "".join(f"{lab:>34}" for lab in rob))
            for res in all_members:
                cells = []
                for v in rob.values():
                    r = v[(v["group"] == res) & (v["start"] == "all")]
                    if not len(r):
                        cells.append(f"{'-':>34}"); continue
                    r = r.iloc[0]
                    cells.append(f"{r['to_group_free']:>5.1f} -> {r['to_group_occupied']:>5.1f} | "
                                 f"{r['out_free']:>5.1f} -> {r['out_occupied']:>5.1f}".rjust(34))
                rep(f"    {res:<11}" + "".join(cells))
    sh = arrival_sharing(visits_raw, None)
    sh.to_csv(od / "arrival_sharing.csv", index=False, float_format="%.2f")
    rep("\nSHARED ARRIVALS: visits that start while the same residue already holds another K+, and of those the %\n"
        "where that residue is free of other K+ when the ion leaves (the other K+ moved on: knock-on)")
    for r in sh.to_dict("records"):
        rep(f"    {r['site']:>9}: {r['visits']:>4} visits, shared at arrival {r['pct_shared_at_arrival']:5.1f}% "
            f"({r['n_shared']}), other K+ gone when leaving {r['pct_other_gone_at_leave']:5.1f}%")
    rep.write(od / "occupancy_summary.txt")


def _mse(x):
    """'mean ± SE (median)' of a series (ns), '–' when empty."""
    x = pd.Series(x).dropna()
    if not len(x):
        return "–", np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    txt = f"{x.mean():.1f} ± {se:.1f} ({x.median():.1f})" if np.isfinite(se) else f"{x.mean():.1f} ({x.median():.1f})"
    return txt, x.mean(), se, x.median()


def stop_table(events, visits, sites, ring, dt):
    """Per site (a stop): its visits, how long ions stay, where they go next (out, the GLU ring, the
    lower ring), how long they still need to leave the cavity from the start of the visit, and the
    SF -> exit time of the ions stopping there at least once vs never."""
    raw_ring = dict(zip(visits["site"], visits["ring"]))        # raw residue names (next_site is raw)
    nxt_ring = visits["next_site"].map(lambda x: "Exit" if x == EXIT else raw_ring.get(x, "LOWER"))
    v = visits.assign(next_ring=nxt_ring.values,
                      site=visits["model_site"] if "model_site" in visits else visits["site"])
    transit = events.set_index(["run", "event"])["transit_frames"] * dt
    n_ions = len(events)
    rows = []
    for site in sites:
        g = v[v["site"] == site]
        if not len(g):
            continue
        ions = set(zip(g["run"], g["event"]))
        at = transit[[k in ions for k in transit.index]]
        never = transit[[k not in ions for k in transit.index]]
        nv = events.set_index(["run", "event"])["n_visits"]
        row = dict(site=site, ring=ring.get(site, ""), visits=len(g), ions=len(ions), pct_ions=100 * len(ions) / n_ions)
        for col, x in (("stay", g["residue_frames"] * dt), ("to_exit", g["to_exit_frames"] * dt),
                       ("transit_stop", at), ("transit_never", never)):
            row[f"{col}_txt"], row[f"{col}_mean_ns"], row[f"{col}_sem_ns"], row[f"{col}_median_ns"] = _mse(x)
        for t, name in (("Exit", "out"), ("GLU", "to_glu"), ("LOWER", "to_lower")):
            row[f"pct_{name}"] = 100 * (g["next_ring"] == t).mean()
        row["visits_per_ion_stop"] = nv[[k in ions for k in nv.index]].mean()
        row["visits_per_ion_never"] = nv[[k not in ions for k in nv.index]].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stop_table(tab, title, out_file, min_n=15):
    """The stop table as a figure; rows from fewer than min_n visits in grey."""
    if not len(tab):
        return
    cols = ["stop", "visits", "ions\n(% of ions)", "time there (ns)\nmean ± SE (median)", "next:\nout",
            "next:\nGLU ring", "next:\nlower ring", "time to leave the cavity\nfrom the visit (ns)",
            "SF → exit (ns)\nions stopping here", "SF → exit (ns)\nions never here"]
    cells = [[r["site"], str(r["visits"]), f"{r['ions']} ({r['pct_ions']:.0f}%)", r["stay_txt"],
              f"{r['pct_out']:.0f}%", f"{r['pct_to_glu']:.0f}%", f"{r['pct_to_lower']:.0f}%", r["to_exit_txt"],
              r["transit_stop_txt"], r["transit_never_txt"]] for r in tab.to_dict("records")]
    widths = [1.5, 1.0, 1.5, 2.6, 1.0, 1.2, 1.2, 3.0, 2.6, 2.6]
    fig, ax = plt.subplots(figsize=(1.0 * sum(widths), 1.0 + 0.42 * (len(cells) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[x / sum(widths) for x in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(16)
    tb.scale(1, 2.3)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#C9D3D8")
        if i == 0:
            cell.set_facecolor("#5B6770" if j in (4, 5, 6) else "#123B4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 1.9)
            continue
        r = tab.iloc[i - 1]
        if j in (4, 5, 6):
            cell.set_facecolor("#EEF3F5")
        if j in (0, 3, 7):
            cell.get_text().set_fontweight("bold")
        if r["visits"] < min_n:
            cell.get_text().set_color("#9a9a9a")
    ax.set_title(f"{title}\nstops: how long ions stay at each residue, where they go next, and how long they still need "
                 f"to leave the cavity (from the start of the visit)\n(SF → exit: whole cavity time of the ions that "
                 f"stop here at least once vs never; grey: fewer than {min_n} visits)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stop_analysis(events, visits, sites, ring, title, out_dir, dt):
    """Is a residue a trap? Writes out_dir/stops/stops.csv, .png, .txt."""
    if not len(visits):
        return
    od = Path(out_dir) / "stops"
    od.mkdir(exist_ok=True)
    tab = stop_table(events, visits, sites, ring, dt)
    keep = [c for c in tab.columns if not c.endswith("_txt")]
    tab[keep].to_csv(od / "stops.csv", index=False, float_format="%.2f")
    plot_stop_table(tab, title, od / "stops.png")
    rep = _Report()
    rep(f"STOPS  {title}")
    rep("per residue: visits; ions stopping there (% of ions); time there; next stop; time still needed to leave the\n"
        "cavity from the start of the visit; SF -> exit time of the ions stopping there at least once vs never\n"
        "(with their visits per ion: ions with more stops take longer anyway)")
    for r in tab.to_dict("records"):
        rep(f"  {r['site']:<9} {r['visits']:>4} visits, {r['ions']:>4} ions ({r['pct_ions']:4.0f}%)  there "
            f"{r['stay_txt']:<22} next out {r['pct_out']:3.0f}%  GLU {r['pct_to_glu']:3.0f}%  lower "
            f"{r['pct_to_lower']:3.0f}%  to leave {r['to_exit_txt']:<22} SF->exit {r['transit_stop_txt']} "
            f"({r['visits_per_ion_stop']:.1f} visits/ion) vs never {r['transit_never_txt']} "
            f"({r['visits_per_ion_never']:.1f})")
    rep.write(od / "stops.txt")


def _step_k(events, visits_raw, max_k):
    """{(run, event, visit): other-K+ bin at the step that leads to that visit} (visit 0: SF count at
    end_1; visit k: K+ above + below when leaving visit k-1)."""
    out = {}
    sf = dict(zip(zip(events["run"], events["event"]), events["sf_count"]))
    prev = {(r, e, v): a + b for r, e, v, a, b in
            zip(visits_raw["run"], visits_raw["event"], visits_raw["visit"], visits_raw["above"], visits_raw["below"])}
    for r, e, v in zip(visits_raw["run"], visits_raw["event"], visits_raw["visit"]):
        k = sf.get((r, e)) if v == 0 else prev.get((r, e, v - 1))
        if k is not None and np.isfinite(k):
            out[(r, e, v)] = bin_label(int(k), max_k)
    return out


def arrival_sharing_by_k(events, visits_raw, max_k):
    """arrival_sharing split by the other K+ in the cavity at the step leading to the visit."""
    if "occupied_arrive" not in visits_raw:
        return pd.DataFrame()
    kmap = _step_k(events, visits_raw, max_k)
    v = visits_raw[visits_raw["occupied_arrive"].map(lambda o: isinstance(o, str))].copy()
    v["k"] = [kmap.get(key) for key in zip(v["run"], v["event"], v["visit"])]
    v = v[v["k"].notna()]
    leave = "occupied_leave_frame" if "occupied_leave_frame" in v else "occupied_leave"
    rows = []
    for (site, k), g in v.groupby(["site", "k"]):
        shared = g["occupied_arrive"].map(lambda o: site in _occ_set(o))
        still = g.loc[shared, leave].map(lambda o: site in _occ_set(o))
        rows.append(dict(site=site, start=k, visits=len(g), n_shared=int(shared.sum()),
                         pct_shared_at_arrival=100 * shared.mean(),
                         pct_other_gone_at_leave=100 * (~still).mean() if len(still) else np.nan))
    return pd.DataFrame(rows)


def _parse_occupants(x):
    """'E141.G1:1234;D173.G1:1250' -> {resid: site}."""
    out = {}
    if isinstance(x, str) and x:
        for item in x.split(";"):
            site, _, r = item.rpartition(":")
            if site and r.lstrip("-").isdigit():
                out[int(r)] = site
    return out


def _parse_ids(x):
    """'1234:3.2;1250:-5.1' or '1300;1301' -> set of resids."""
    if not isinstance(x, str) or not x:
        return set()
    return {int(i.split(":")[0]) for i in x.split(";") if i.split(":")[0].lstrip("-").isdigit()}


def handover_table(events, frames, visits_raw, max_k, dt, events_all=None, visits_all=None):
    """Ion tracking at each visit: the K+ that sat at the residue during the ion's trip there (the
    resident: the one seen there last before the arrival), when it left the residue relative to the
    arrival (ns, < 0 before), and where it went next within the ion's own transit (another residue of
    the GLU ring / the lower ring, the SF, out of the cavity, still there, or free in the cavity until the ion leaves). One row per
    visit; visits whose residue had no other K+ during the trip have resident = NaN.
    With events_all / visits_all (every permeation of the runs): whether the pushed resident itself
    permeates in the same cavity stay (a permeation of that K+ whose transit covers the moment it is
    pushed), how long after being pushed it leaves the cavity, whether it leaves before the new ion,
    and its next stops after being pushed (from its own visits)."""
    need = {"occupants", "cavity_ions", "sf_ions"}
    if not need <= set(frames.columns) or not len(visits_raw):
        return pd.DataFrame()
    ring = dict(zip(visits_raw["site"], visits_raw["ring"]))
    kmap = _step_k(events, visits_raw, max_k)
    end1 = dict(zip(zip(events["run"], events["event"]), events["end_1"]))
    rows = []
    perm = events_all if events_all is not None else events
    perm_by = {k: g for k, g in perm.groupby(["run", "ion_id"])}
    va = visits_all if visits_all is not None else visits_raw
    va_by = {k: g.sort_values("visit") for k, g in va.groupby(["run", "event"])} if len(va) else {}
    end2 = dict(zip(zip(perm["run"], perm["event"]), perm["end_2"]))
    fr_by = {k: g.sort_values("frame") for k, g in frames.groupby(["run", "event"])}
    vis_by = {k: g.sort_values("visit") for k, g in visits_raw.groupby(["run", "event"])}
    for key, vv in vis_by.items():
        g = fr_by.get(key)
        if g is None or key not in end1 or g["occupants"].isna().all():   # run collected without identities
            continue
        f0 = int(g["frame"].iloc[0])
        occ = [_parse_occupants(x) for x in g["occupants"]]
        cav = [_parse_ids(x) for x in g["cavity_ions"]]
        sf = [_parse_ids(x) for x in g["sf_ions"]]
        n = len(occ)
        prev_end = end1[key]
        for v in vv.to_dict("records"):
            site, a = v["site"], int(v["start"])
            lo, hi = prev_end + 1, a                              # the trip there, and the arrival frame
            prev_end = int(v["end"])
            row = dict(run=key[0], event=key[1], visit=v["visit"], site=site, ring=v["ring"],
                       start=kmap.get((key[0], key[1], v["visit"])), resident=np.nan, leave_lag_ns=np.nan,
                       resident_went="", resident_went_to="", resident_permeates=np.nan,
                       resident_exit_ns=np.nan, resident_exits_first=np.nan, resident_next="")
            res, last = None, None
            for f in range(min(hi, f0 + n - 1), max(lo, f0) - 1, -1):   # last K+ seen at the residue
                who = [r for r, s_ in occ[f - f0].items() if s_ == site]
                if who:
                    res, last = who[0], f
                    break
            if res is None:
                rows.append(row)
                continue
            row["resident"] = res
            f = last
            while f + 1 < f0 + n and occ[f + 1 - f0].get(res) == site:     # until it leaves the residue
                f += 1
            if f + 1 >= f0 + n:
                row["leave_lag_ns"] = np.nan
                row["resident_went"] = "still there"
                rows.append(row)
                continue
            leave = f + 1
            row["leave_lag_ns"] = (leave - a) * dt
            went = "free in cavity"
            for h in range(leave, f0 + n):                       # where it goes next
                i = h - f0
                s_ = occ[i].get(res)
                if s_ and s_ != site:
                    went = "GLU ring" if ring.get(s_, "LOWER") == "GLU" else "lower ring"
                    row["resident_went_to"] = s_
                    break
                if res in sf[i]:
                    went = "SF"
                    break
                if res not in cav[i]:
                    went = "out of the cavity"
                    break
            row["resident_went"] = went
            pr = perm_by.get((key[0], res))                       # does the pushed K+ permeate?
            hit = pr[(pr["end_1"] <= leave) & (pr["end_2"] >= leave)] if pr is not None else []
            row["resident_permeates"] = bool(len(hit))
            if len(hit):
                h = hit.iloc[0]
                row["resident_exit_ns"] = (int(h["end_2"]) - leave) * dt
                row["resident_exits_first"] = bool(int(h["end_2"]) < end2.get(key, np.inf))
                rv = va_by.get((key[0], int(h["event"])))
                if rv is not None:
                    nxt = rv[rv["start"] >= leave]["site"].tolist()[:4]
                    row["resident_next"] = ">".join(nxt + ["Exit"]) if len(nxt) < 4 else ">".join(nxt) + ">..."
            rows.append(row)
    return pd.DataFrame(rows)


DEST = ["GLU ring", "lower ring", "out of the cavity", "SF", "still there", "free in cavity"]


def handover_summary(ho, sites, push_ns=1.0):
    """Per site and K+ bin (+ all): visits, % with a resident during the trip, of those % leaving the
    residue within push_ns of the arrival, median lag (ns) and where the resident went (%)."""
    rows = []
    if not len(ho):
        return pd.DataFrame()
    for site in sites:
        g0 = ho[ho["site"] == site]
        if not len(g0):
            continue
        for b in ["all"] + sorted(x for x in g0["start"].dropna().unique()):
            g = g0 if b == "all" else g0[g0["start"] == b]
            r_ = g[g["resident"].notna()]
            row = dict(site=site, start=b, visits=len(g), n_resident=len(r_),
                       pct_resident=100 * len(r_) / len(g) if len(g) else np.nan,
                       median_lag_ns=r_["leave_lag_ns"].median() if len(r_) else np.nan,
                       pct_leave_near_arrival=100 * (r_["leave_lag_ns"].abs() <= push_ns).mean() if len(r_) else np.nan,
                       pct_leave_before=100 * (r_["leave_lag_ns"] < -push_ns).mean() if len(r_) else np.nan,
                       pct_leave_after=100 * (r_["leave_lag_ns"] > push_ns).mean() if len(r_) else np.nan)
            for d in DEST:
                row[f"pct_{d.replace(' ', '_')}"] = 100 * (r_["resident_went"] == d).mean() if len(r_) else np.nan
            moved = r_[r_["resident_went"] != "still there"] if len(r_) else r_
            pm = moved["resident_permeates"].astype(float) if len(moved) and "resident_permeates" in moved else pd.Series(dtype=float)
            row["n_moved"] = len(moved)
            row["pct_resident_permeates"] = 100 * pm.mean() if len(pm.dropna()) else np.nan
            ex = moved.loc[moved["resident_permeates"] == True] if len(moved) else moved
            row["median_resident_exit_ns"] = ex["resident_exit_ns"].median() if len(ex) else np.nan
            row["pct_resident_exits_first"] = 100 * ex["resident_exits_first"].astype(float).mean() if len(ex) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_handover_table(summ, title, out_file, push_ns=1.0, min_n=5, scope="all visits"):
    """Table per residue (all ions): visits, % with a resident K+, when it left, where it went."""
    t = summ[summ["start"] == "all"]
    if not len(t):
        return
    cols = ["residue", "visits", "with a K+ there\nduring the trip", f"it leaves\n> {push_ns:g} ns before",
            f"it leaves within\n± {push_ns:g} ns of arrival", f"it leaves\n> {push_ns:g} ns after",
            "median\n(ns)", "it goes to:\nGLU ring", "it goes to:\nlower ring", "it goes:\nout of cavity",
            "it goes:\nup to the SF", "still there /\nfree in cavity", "it then\npermeates",
            "its exit after\nthe push (ns)", "it exits before\nthe new ion"]
    cells, grey = [], []
    for r in t.to_dict("records"):
        ok = r["n_resident"] > 0
        f = (lambda x: f"{x:.0f}%") if ok else (lambda x: "–")
        cells.append([r["site"], str(r["visits"]), f"{r['n_resident']} ({r['pct_resident']:.0f}%)",
                      f(r["pct_leave_before"]), f(r["pct_leave_near_arrival"]), f(r["pct_leave_after"]),
                      f"{r['median_lag_ns']:+.1f}" if ok else "–", f(r["pct_GLU_ring"]), f(r["pct_lower_ring"]),
                      f(r["pct_out_of_the_cavity"]), f(r["pct_SF"]),
                      f(r["pct_still_there"] + r["pct_free_in_cavity"]) if ok else "–",
                      f"{r['pct_resident_permeates']:.0f}%" if np.isfinite(r.get("pct_resident_permeates", np.nan)) else "–",
                      f"{r['median_resident_exit_ns']:.1f}" if np.isfinite(r.get("median_resident_exit_ns", np.nan)) else "–",
                      f"{r['pct_resident_exits_first']:.0f}%" if np.isfinite(r.get("pct_resident_exits_first", np.nan)) else "–"])
        grey.append(r["n_resident"] < min_n)
    widths = [1.5, 1.0, 1.9, 1.6, 1.9, 1.6, 1.1, 1.4, 1.4, 1.5, 1.4, 1.5, 1.4, 1.7, 1.7]
    fig, ax = plt.subplots(figsize=(1.0 * sum(widths), 1.0 + 0.42 * (len(cells) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[x / sum(widths) for x in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(16)
    tb.scale(1, 2.3)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#C9D3D8")
        if i == 0:
            cell.set_facecolor("#1F6F50" if j >= 12 else "#5B6770" if j >= 7 else "#123B4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(cell.get_height() * 1.9)
            continue
        if 7 <= j < 12:
            cell.set_facecolor("#EEF3F5")
        elif j >= 12:
            cell.set_facecolor("#EAF4EE")
        if j in (0, 4, 12):
            cell.get_text().set_fontweight("bold")
        if grey[i - 1]:
            cell.get_text().set_color("#9a9a9a")
    ax.set_title(f"{title}\nion tracking: the K+ sitting at the residue while the new ion travels there (resident), "
                 f"when it leaves the residue relative to the new ion's arrival, and where it goes next\n"
                 f"({scope}; next stop tracked within the new ion's transit; 'it then permeates' = the pushed K+ leaves "
                 f"the cavity through the gate in that same stay; grey: fewer than {min_n} residents)",
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def handover_analysis(events, frames, visits_raw, sites, max_k, title, out_dir, dt, events_all=None,
                      visits_all=None):
    """Writes out_dir/handover/: shared arrivals by K+ (from the occupancy columns) and, when the
    frames carry occupant identities, the ion tracking of the residents."""
    od = Path(out_dir) / "handover"
    od.mkdir(exist_ok=True)
    rep = _Report()
    rep(f"HAND-OVER  {title}")
    sh = arrival_sharing_by_k(events, visits_raw, max_k)
    if len(sh):
        sh.to_csv(od / "arrival_sharing_by_k.csv", index=False, float_format="%.2f")
        rep("\nSHARED ARRIVALS BY THE OTHER K+ IN THE CAVITY: visits starting while the residue already holds another\n"
            "K+ (single frame at arrival), and of those the % where it is free again when the ion leaves")
        for site in sites:
            g = sh[sh["site"] == site]
            if len(g):
                rep(f"  {site:<9} " + "   ".join(f"{r['start']} other K+: {r['pct_shared_at_arrival']:4.1f}% of "
                                                f"{r['visits']} (gone {r['pct_other_gone_at_leave']:.0f}%)"
                                                if r["n_shared"] else f"{r['start']} other K+: 0% of {r['visits']}"
                                                for r in g.to_dict("records")))
    ho = handover_table(events, frames, visits_raw, max_k, dt, events_all, visits_all)
    if not len(ho):
        rep("\nION TRACKING: not available - the frames have no occupant identities (run main.py again)")
        print("site_chain: no occupant identities in the frames: run main.py again for the ion tracking")
    else:
        no_id = sorted(set(events["run"]) - set(ho["run"]))
        if no_id:
            rep("\nruns without occupant identities, left out of the ion tracking (run main.py again for them): "
                + ", ".join(f"RUN{r}" for r in no_id))
            print("site_chain: ion tracking without " + ", ".join(f"RUN{r}" for r in no_id)
                  + " (no occupant identities: run main.py again for them)")
        ho.to_csv(od / "handover_visits.csv", index=False, float_format="%.2f")
        summ = handover_summary(ho, sites)
        summ.to_csv(od / "handover_summary.csv", index=False, float_format="%.2f")
        plot_handover_table(summ, title, od / "handover.png")
        ho1 = ho[ho["visit"] == 0]                                # the new ion's first stop after the SF
        summ1 = handover_summary(ho1, sites)
        if len(summ1):
            summ1.to_csv(od / "handover_visit_1_summary.csv", index=False, float_format="%.2f")
            plot_handover_table(summ1, title, od / "handover_visit_1.png", scope="visit 1: the new ion's first stop")
        rep("\nION TRACKING: the resident = the K+ seen last at the residue during the new ion's trip there;\n"
            "lag = when it leaves the residue minus the new ion's arrival (ns); destination = where it goes next")
        for r in summ.to_dict("records"):
            if not r["n_resident"]:
                rep(f"  {r['site']:<9} {str(r['start']):>4}: {r['visits']:>4} visits, no resident")
                continue
            rep(f"  {r['site']:<9} {str(r['start']):>4}: {r['visits']:>4} visits, resident {r['pct_resident']:4.0f}%; "
                f"leaves before {r['pct_leave_before']:3.0f}% / near arrival {r['pct_leave_near_arrival']:3.0f}% / "
                f"after {r['pct_leave_after']:3.0f}% (median {r['median_lag_ns']:+.1f} ns); goes to GLU "
                f"{r['pct_GLU_ring']:3.0f}%, lower {r['pct_lower_ring']:3.0f}%, out {r['pct_out_of_the_cavity']:3.0f}%,"
                f" SF {r['pct_SF']:3.0f}%, still/free {r['pct_still_there'] + r['pct_free_in_cavity']:3.0f}%")
        rep("\nWHERE THE RESIDENT GOES, by residue (all visits; the next residue it sits at, % of the residents "
            "that move to another residue)")
        for site in sites:
            g = ho[(ho["site"] == site) & (ho["resident_went_to"] != "")]
            if len(g):
                vc = g["resident_went_to"].value_counts()
                rep(f"  {site:<9} -> " + ", ".join(f"{k} {100 * n / len(g):.0f}%" for k, n in vc.items())
                    + f"   (n={len(g)})")
        for scope, hh in (("ALL VISITS", ho), ("VISIT 1", ho[ho["visit"] == 0])):
            m = hh[hh["resident"].notna() & (hh["resident_went"] != "still there")]
            if not len(m):
                continue
            rep(f"\nTHE PUSHED K+ ({scope}): % that then permeate (leave through the gate in the same stay), median time"
                "\nfrom the push to its exit, % leaving before the new ion, and its commonest next stops")
            for site in sites:
                g = m[m["site"] == site]
                if not len(g):
                    continue
                pm = g["resident_permeates"].astype(float)
                ex = g[g["resident_permeates"] == True]
                paths = ex["resident_next"].replace("", np.nan).dropna().value_counts().head(3)
                rep(f"  {site:<9} n={len(g):>3}: permeates {100 * pm.mean():3.0f}%"
                    + (f", exit {ex['resident_exit_ns'].median():5.1f} ns after the push, before the new ion "
                       f"{100 * ex['resident_exits_first'].astype(float).mean():3.0f}%" if len(ex) else "")
                    + ("; paths: " + ", ".join(f"{p_} ({n})" for p_, n in paths.items()) if len(paths) else ""))
        res_n = ho[ho["resident"].notna()].groupby("site")["resident"].nunique()
        if len(res_n):
            rep("\nDIFFERENT RESIDENT K+ per residue (a few K+ coming back again and again = long-lived residents): "
                + ", ".join(f"{k} {int(v)}" for k, v in res_n.items()))
    rep.write(od / "handover.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Presentation figures: the story of visit 1, one message per figure
# ─────────────────────────────────────────────────────────────────────────────

PRES_COLORS = {"E141": "#C0392B", "E152": "#D98A1C", "D173": "#13896B", "N184": "#3B7DD8", EXIT: "#6B6B6B"}
PRES_FONT = dict(title=22, label=19, tick=18, value=17, legend=17)


def _pres_color(site):
    return PRES_COLORS.get(group_label(site), "#8C8C8C")


def _pres_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pres_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=PRES_FONT["tick"])


def _pres_bins(vt, min_n=15):
    """K+ bins of visit_1.csv with at least min_n ions, in order: [(bin, n_ions)]."""
    b = vt[vt["start"] != "all"].groupby("start", sort=False)["n_ions"].first()
    return [(k, int(n)) for k, n in b.items() if n >= min_n]


def _k_label(k, n):
    return f"{k} other K+\n({n} ions)"


def pres_first_stop(vt, out, lines):
    """01: where a new ion goes first, by K+ in the cavity: 100% bars of E141 / E152 / D173 / N184 / out."""
    bins = _pres_bins(vt)
    if not bins:
        return
    groups = [g for g in ("E141", "E152", "D173", "N184") if (vt["target"].map(group_label) == g).any()] + [EXIT]
    fig, ax = plt.subplots(figsize=(13, 1.6 + 1.25 * len(bins)))
    for i, (k, n) in enumerate(bins):
        g = vt[vt["start"] == k]
        left = 0
        vals = {gr: g.loc[g["target"].map(group_label) == gr, "pct"].sum() for gr in groups}
        for gr in groups:
            v = vals[gr]
            ax.barh(i, v, left=left, color=PRES_COLORS[gr], edgecolor="white", linewidth=2, height=0.62,
                    label=("out of the cavity" if gr == EXIT else gr) if i == 0 else None)
            if v >= 5:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white",
                        fontsize=PRES_FONT["value"], fontweight="bold")
            left += v
        lines.append(f"  {k} other K+ ({n} ions): " + ", ".join(f"{'out' if gr == EXIT else gr} {vals[gr]:.0f}%"
                                                              for gr in groups))
    ax.set_yticks(range(len(bins)))
    ax.set_yticklabels([_k_label(k, n) for k, n in bins], fontsize=PRES_FONT["tick"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of new ions", fontsize=PRES_FONT["label"])
    ax.legend(ncol=len(groups), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"],
              frameon=False)
    ax.set_title("Where a new ion goes first after the SF", fontsize=PRES_FONT["title"], pad=48)
    _pres_axes(ax)
    _pres_save(fig, out)


def pres_occupancy(occ, vt, sites, out, lines, min_mean=30):
    """02: % of new ions finding each residue already holding a K+, by K+ in the cavity."""
    bins = [k for k, _ in _pres_bins(vt)]
    occ = occ.set_index("start")
    res = [s_ for s_ in sites if s_ in occ.columns and (group_label(s_) != "N184" or occ.loc["all", s_] >= min_mean)]
    bins = [k for k in bins if k in occ.index]
    if not res or not bins:
        return
    ramp = {1: ["#2F5E9E"], 2: ["#8DB4E6", "#1E4E8C"], 3: ["#A9C6EC", "#4F84C9", "#1E4E8C"],
            4: ["#C4D8F2", "#7FA7DD", "#3D6FB8", "#173A66"]}
    shades = ramp.get(len(bins), ["#D6E4F5", "#A9C6EC", "#6F9FD8", "#2F5E9E", "#173A66"][:len(bins)])
    fig, ax = plt.subplots(figsize=(max(12, 2.1 * len(res) + 2), 6.8))
    w = 0.8 / len(bins)
    x = np.arange(len(res))
    for j, k in enumerate(bins):
        v = [occ.loc[k, r] for r in res]
        ax.bar(x - 0.4 + w * (j + 0.5), v, w * 0.92, color=shades[j], label=f"{k} other K+")
        for xi, vi in zip(x - 0.4 + w * (j + 0.5), v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=PRES_FONT["value"] - 2)
    for r in res:
        lines.append(f"  {r}: " + " / ".join(f"{occ.loc[k, r]:.0f}%" for k in bins) + f"  ({' / '.join(bins)} other K+)")
    ax.set_xticks(x)
    ax.set_xticklabels(res, fontsize=PRES_FONT["tick"], fontweight="bold")
    for t, r in zip(ax.get_xticklabels(), res):
        t.set_color(_pres_color(r))
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of new ions finding it\nholding another K+", fontsize=PRES_FONT["label"])
    ax.legend(ncol=len(bins), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"], frameon=False)
    ax.set_title("Which residues already hold a K+ as the cavity fills", fontsize=PRES_FONT["title"], pad=44)
    _pres_axes(ax)
    _pres_save(fig, out)


def _pres_pair(ax, res, a, b, la, lb, ylab):
    x = np.arange(len(res))
    for off, vals, lab, light in ((-0.2, a, la, True), (0.2, b, lb, False)):
        cols = [_pres_color(r) for r in res]
        ax.bar(x + off, vals, 0.38, color=cols, alpha=0.38 if light else 1.0, edgecolor=cols, linewidth=2,
               label=lab)
        for xi, vi in zip(x + off, vals):
            ax.text(xi, vi + 1, f"{vi:.0f}%", ha="center", va="bottom", fontsize=PRES_FONT["value"],
                    fontweight="bold" if not light else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(res, fontsize=PRES_FONT["tick"], fontweight="bold")
    ax.set_ylabel(ylab, fontsize=PRES_FONT["label"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#9A9A9A", alpha=0.38, edgecolor="#9A9A9A", label=la),
                       Patch(facecolor="#4A4A4A", label=lb)], ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"], frameon=False)
    _pres_axes(ax)


def pres_free_vs_occupied(rt, out, lines, min_visits=15):
    """03: % of new ions going to a residue when it is free vs already holding a K+ (all ions)."""
    a = rt[rt["start"] == "all"].copy()
    a["visits"] = a["n_free"] * a["to_group_free"] / 100 + a["n_occupied"] * a["to_group_occupied"] / 100
    a = a[(a["visits"] >= min_visits) & (a["n_free"] > 0) & (a["n_occupied"] > 0)]
    if not len(a):
        return
    res = a["group"].tolist()
    fig, ax = plt.subplots(figsize=(max(11, 2.2 * len(res) + 2), 6.5))
    _pres_pair(ax, res, a["to_group_free"].tolist(), a["to_group_occupied"].tolist(), "residue free",
               "residue already holds a K+", "% of new ions\ngoing to the residue")
    ax.set_ylim(0, max(a[["to_group_free", "to_group_occupied"]].to_numpy().max() * 1.25, 10))
    ax.set_title("Does a K+ already sitting there change where the new ion goes?", fontsize=PRES_FONT["title"], pad=44)
    for r in a.to_dict("records"):
        lines.append(f"  {r['group']}: free {r['to_group_free']:.0f}% -> holds a K+ {r['to_group_occupied']:.0f}%"
                     f"  ({int(r['n_free'])} / {int(r['n_occupied'])} ions)")
    _pres_save(fig, out)


def pres_resident_at_arrival(hs, out, lines, min_res=5):
    """04: when a new ion lands, was a K+ there, and was it pushed off at that moment or had it left before?"""
    a = hs[(hs["start"] == "all") & (hs["n_resident"] >= min_res)]
    if not len(a):
        return
    res = a["site"].tolist()
    fig, ax = plt.subplots(figsize=(13.5, 1.6 + 1.05 * len(res)))
    for i, r in enumerate(a.to_dict("records")):
        c = _pres_color(r["site"])
        near, before = r["pct_leave_near_arrival"], r["pct_leave_before"] + r["pct_leave_after"]
        ax.barh(i, near, color=c, edgecolor="white", linewidth=2, height=0.6)
        ax.barh(i, before, left=near, color=c, alpha=0.3, edgecolor="white", linewidth=2, height=0.6)
        if near >= 8:
            ax.text(near / 2, i, f"{near:.0f}%", ha="center", va="center", color="white",
                    fontsize=PRES_FONT["value"], fontweight="bold")
        if before >= 8:
            ax.text(near + before / 2, i, f"{before:.0f}%", ha="center", va="center", color="#222222",
                    fontsize=PRES_FONT["value"])
        ax.text(102, i, f"K+ there in {r['pct_resident']:.0f}%\nof arrivals", va="center", fontsize=PRES_FONT["value"] - 2)
        lines.append(f"  {r['site']}: K+ there in {r['pct_resident']:.0f}% of arrivals; pushed off as the new ion lands "
                     f"{near:.0f}%, had left before {before:.0f}% (median {r['median_lag_ns']:+.1f} ns)")
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res, fontsize=PRES_FONT["tick"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the K+ sitting there when the new ion lands", fontsize=PRES_FONT["label"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4A4A4A", label="pushed off as the new ion lands (±1 ns)"),
                       Patch(facecolor="#4A4A4A", alpha=0.3, label="had already left (seat refilled)")],
              ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"], frameon=False)
    ax.set_title("What happens to the K+ already sitting there?", fontsize=PRES_FONT["title"], pad=44)
    _pres_axes(ax)
    _pres_save(fig, out)


def pres_where_pushed(hv, hs, out, lines, min_res=15, min_push=60):
    """05: for the hand-over residues, where the pushed K+ goes next, and whether it then permeates."""
    a = hs[(hs["start"] == "all") & (hs["n_resident"] >= min_res) & (hs["pct_leave_near_arrival"] >= min_push)]
    if not len(a) or not len(hv):
        return
    rows = []
    for r in a.to_dict("records"):
        g = hv[(hv["site"] == r["site"]) & hv["resident"].notna()].copy()
        dest = g["resident_went_to"].fillna("").astype(str)
        dest = dest.where(dest != "", g["resident_went"].map(lambda x: EXIT if x == "out of the cavity" else ""))
        dest = dest[dest != ""]
        if len(dest):
            rows.append((r, dest.value_counts(normalize=True) * 100))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(14, 1.8 + 1.25 * len(rows)))
    for i, (r, vc) in enumerate(rows):
        groups = {}                                            # residues of the same group together
        for d, v in vc.items():
            groups.setdefault(group_label(d) if d != EXIT else EXIT, []).append((d, v))
        left = 0
        order = [g_ for g_ in ("E141", "E152", "D173", "N184") if g_ in groups] + ([EXIT] if EXIT in groups else [])
        parts = []
        for g_ in order:
            v = sum(x for _, x in groups[g_])
            name = groups[g_][0][0] if len(groups[g_]) == 1 else g_
            ax.barh(i, v, left=left, color=PRES_COLORS.get(g_, "#8C8C8C"), edgecolor="white", linewidth=2, height=0.6)
            lab = "out" if g_ == EXIT else name
            if v >= 9:
                ax.text(left + v / 2, i, f"{lab}\n{v:.0f}%", ha="center", va="center", color="white",
                        fontsize=PRES_FONT["value"] - 1, fontweight="bold")
            parts.append(f"{lab} {v:.0f}%")
            left += v
        ax.text(102, i, f"{r['n_resident']:.0f} K+", va="center", fontsize=PRES_FONT["value"] - 2)
        lines.append(f"  K+ pushed off {r['site']} ({r['n_resident']:.0f} K+): " + ", ".join(parts))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"pushed off\n{r['site']}" for r, _ in rows], fontsize=PRES_FONT["tick"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the pushed K+, by the next residue it sits at", fontsize=PRES_FONT["label"])
    from matplotlib.patches import Patch
    used = [g_ for g_ in ("E141", "E152", "D173", "N184") if any(group_label(d) == g_ for _, vc in rows for d in vc.index)]
    ax.legend(handles=[Patch(facecolor=PRES_COLORS[g_], label=g_) for g_ in used]
              + [Patch(facecolor=PRES_COLORS[EXIT], label="out of the cavity")], ncol=len(used) + 1,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"], frameon=False)
    ax.set_title("Where the pushed K+ goes next", fontsize=PRES_FONT["title"], pad=44)
    _pres_axes(ax)
    _pres_save(fig, out)


def pres_storage_push(rt, out, lines, min_n=30):
    """06: storage residues (fewer new ions go there when occupied): % of new ions leaving the cavity
    directly, residue free vs holding a K+."""
    a = rt[rt["start"] == "all"]
    a = a[(a["n_free"] >= min_n) & (a["n_occupied"] >= min_n) & (a["to_group_free"] >= 5)
          & (a["to_group_occupied"] <= a["to_group_free"] / 1.5)]
    if not len(a):
        return
    res = a["group"].tolist()
    fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(res) + 3), 6.5))
    _pres_pair(ax, res, a["out_free"].tolist(), a["out_occupied"].tolist(), "storage seat free",
               "storage seat holds a K+", "% of new ions leaving\nthe cavity directly")
    ax.set_ylim(0, max(a[["out_free", "out_occupied"]].to_numpy().max() * 1.25, 10))
    ax.set_title("Does a full storage seat push the new ion out?", fontsize=PRES_FONT["title"], pad=44)
    for r in a.to_dict("records"):
        lines.append(f"  {r['group']}: new ions going there free {r['to_group_free']:.0f}% -> holds a K+ "
                     f"{r['to_group_occupied']:.0f}%; leaving the cavity free {r['out_free']:.0f}% -> holds a K+ "
                     f"{r['out_occupied']:.0f}%")
    _pres_save(fig, out)


def pres_lower_ring(st, out, lines, min_visits=15):
    """07: after a stop at each residue (all visits): next to a Glu (upper ring), the lower ring, or out."""
    a = st[st["visits"] >= min_visits]
    if not len(a):
        return
    res = a["site"].tolist()
    fig, ax = plt.subplots(figsize=(13, 1.6 + 0.85 * len(res)))
    parts = (("pct_to_glu", "next: a Glu (upper ring)", PRES_COLORS["E152"]),
             ("pct_to_lower", "next: D173 / Asn (lower ring)", PRES_COLORS["N184"]),
             ("pct_out", "next: out of the cavity", PRES_COLORS[EXIT]))
    for i, r in enumerate(a.to_dict("records")):
        left = 0
        for col, lab, c in parts:
            v = r[col]
            ax.barh(i, v, left=left, color=c, edgecolor="white", linewidth=2, height=0.62, label=lab if i == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white",
                        fontsize=PRES_FONT["value"], fontweight="bold")
            left += v
        ax.text(102, i, f"{r['visits']} stops", va="center", fontsize=PRES_FONT["value"] - 3)
        lines.append(f"  after a stop at {r['site']} ({r['visits']} stops): a Glu {r['pct_to_glu']:.0f}%, lower ring "
                     f"{r['pct_to_lower']:.0f}%, out {r['pct_out']:.0f}%")
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res, fontsize=PRES_FONT["tick"], fontweight="bold")
    for t, r in zip(ax.get_yticklabels(), res):
        t.set_color(_pres_color(r))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the stops, by where the ion goes next", fontsize=PRES_FONT["label"])
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=PRES_FONT["legend"] - 2, frameon=False)
    ax.set_title("After a stop at each residue: where the ion goes next", fontsize=PRES_FONT["title"], pad=44)
    _pres_axes(ax)
    _pres_save(fig, out)


# ── slide tables (presentation/tables/): the tables used on the slides, as PNG + CSV ──
PRES_TINT = {"E141": "F6DEDB", "E152": "F9E8D2", "D173": "D6EFE7", "N184": "DCE8F8", EXIT: "E6E6E6"}


def _ptable(cells, cols, widths, out, title, head_colors=None, fills=None, bold=None, grey=None, note=None, head_size=12):
    """A slide-style table: navy (or per-column coloured) header, tinted key cells, bold cells, a note."""
    fills, bold, grey = fills or {}, bold or set(), grey or set()
    fig, ax = plt.subplots(figsize=(max(8.0, 0.95 * sum(widths)), 0.9 + 0.36 * (len(cells) + 2) + (0.25 if note else 0)))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[w / sum(widths) for w in widths])
    tb.auto_set_font_size(False)
    tb.set_fontsize(14)
    tb.scale(1, 2.1)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#C9D3D8")
        if i == 0:
            c.set_facecolor("#" + (head_colors[j] if head_colors and head_colors[j] else "123B4A"))
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
            c.get_text().set_fontsize(head_size)
            c.set_height(c.get_height() * 1.7)
            continue
        if (i - 1, j) in fills:
            c.set_facecolor("#" + fills[(i - 1, j)])
        if (i - 1, j) in bold or j == 0:
            c.get_text().set_fontweight("bold")
        if (i - 1, j) in grey:
            c.get_text().set_color("#9a9a9a")
    ax.set_title(title, fontsize=15)
    if note:
        fig.text(0.01, 0.01, note, fontsize=11, color="#5A6570", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(cells, columns=[c.replace("\n", " ") for c in cols]).to_csv(Path(out).with_suffix(".csv"), index=False)


def pres_table_load(vt, occ, rt, kt, rules, sites, tdir, title):
    """slide 'as the cavity fills': per K+ bin, for every residue → (first stop) and occ. (already holds
    a K+), out and time to leave; 01b: new ions going to it when the residue is free → occupied."""
    bins = [k for k, _ in _pres_bins(vt)] + ["all"]
    occ_i, kt_i = occ.set_index("start"), kt.set_index("start") if len(kt) else pd.DataFrame()
    cols, widths, heads = ["other K+", "ions"], [1.1, 0.8], [None, None]
    for s_ in sites:
        cols += [f"{s_}\n→", f"{s_}\nocc."]
        widths += [1.05, 1.05]
        c = PRES_COLORS.get(group_label(s_), "#8C8C8C").lstrip("#")
        heads += [c, "5B6770"]
    cols += ["out of\nthe cavity", "time to\nleave (ns)"]
    widths += [1.15, 1.15]
    heads += [PRES_COLORS[EXIT].lstrip("#"), None]
    cells, fills, bold = [], {}, set()
    for i, k in enumerate(bins):
        g = vt[vt["start"] == k].set_index("target")
        if not len(g):
            continue
        row = [k, str(int(g["n_ions"].iloc[0]))]
        for j, s_ in enumerate(sites):
            row += [f"{g['pct'].get(s_, 0):.0f}%", f"{occ_i.loc[k, s_]:.0f}%" if k in occ_i.index and s_ in occ_i else "–"]
            fills[(len(cells), 3 + 2 * j)] = "E3EAEE"
        row += [f"{g['pct'].get(EXIT, 0):.0f}%"]
        t = kt_i.loc[k, "out_mean_ns"] if k in kt_i.index and "out_mean_ns" in kt_i else np.nan
        if k == "all" and len(kt):
            w = kt[kt["start"] != "all"]
            t = (w["out_mean_ns"] * w["n_out"]).sum() / max(w["n_out"].sum(), 1)
        row += [f"{t:.1f}" if np.isfinite(t) else "–"]
        if k == "all":
            bold |= {(len(cells), j) for j in range(len(row))}
        cells.append(row)
    _ptable(cells, cols, widths, tdir / "01_first_stop_and_occupancy.png",
            f"{title}\nvisit 1: where new ions go first (→) and which residues already hold another K+ (occ.), by the other K+ in the cavity",
            head_colors=heads, fills=fills, bold=bold, head_size=11,
            note="→ = % of new ions stopping there first; occ. = % of new ions finding it already holding another K+")
    a = rt[rt["start"] == "all"].set_index("group")
    rows, f2 = [], {}
    for s_ in sites:
        if s_ not in a.index:
            continue
        rows.append([s_, f"{int(a.loc[s_, 'n_free'])} / {int(a.loc[s_, 'n_occupied'])}",
                     f"{a.loc[s_, 'to_group_free']:.0f}% → {a.loc[s_, 'to_group_occupied']:.0f}%"])
        tf, to = a.loc[s_, "to_group_free"], a.loc[s_, "to_group_occupied"]
        if max(tf, to) >= 5 and (to >= 1.5 * tf or to <= tf / 1.5):
            f2[(len(rows) - 1, 2)] = PRES_TINT.get(group_label(s_), "E6E6E6")
    if rows:
        # 'leaving the cavity: free -> occupied' is left out: under the trip rule short (direct-exit) trips
        # are labelled free more often, and the other occupancy rules flip its direction
        _ptable(rows, ["residue", "ions\nfree / occupied", "new ions going there:\nfree → occupied"], [1.8, 1.9, 2.8],
                tdir / "01b_free_vs_occupied.png",
                f"{title}\nvisit 1: does a K+ already sitting at the residue change where the new ion goes? (all ions)", fills=f2)


def _dest_table(hv, hs, residues, sites):
    """Where the K+ pushed off each residue goes next: per group, '%' (with the per-residue split)."""
    groups = [g_ for g_ in ("E141", "E152", "D173", "N184") if any(group_label(x) == g_ for x in sites)]
    rows = []
    for res, label, sub in residues:
        g = hv[(hv["site"] == res) & hv["resident"].notna()]
        if sub is not None:
            g = g[g["start"].astype(str) == sub]
        n = len(g)
        if not n:
            continue
        dest = g["resident_went_to"].fillna("").astype(str)
        dest = dest.where(dest != "", g["resident_went"].map(lambda x: EXIT if x == "out of the cavity" else "other"))
        row = [label, str(n)]
        for gr in groups:
            m = dest[dest.map(lambda d: group_label(d) == gr)]
            if group_label(res) == gr and len(set(x for x in sites if group_label(x) == gr)) == 1:
                row.append("–")
                continue
            v = 100 * len(m) / n
            members = sorted(x for x in sites if group_label(x) == gr)
            split = " / ".join(f"{100 * (m == x).sum() / n:.0f}" for x in members) if len(members) > 1 and len(m) else ""
            row.append(f"{v:.0f}%" + (f" ({split})" if split and v > 0 else ""))
        row.append(f"{100 * (dest == EXIT).sum() / n:.0f}%")
        rows.append(row)
    cols = ["K+ from", "K+"] + [f"→ {gr}" + ("\n(" + " / ".join(x.split(".", 1)[1] for x in sorted(y for y in sites if group_label(y) == gr)) + ")"
                                       if len([y for y in sites if group_label(y) == gr]) > 1 else "") for gr in groups] \
           + ["straight\nout"]
    return rows, cols


def pres_table_tracking(hs, hv, sites, tdir, title, which, min_res=5):
    """slides 'ion tracking': the K+ already sitting there (pushed off at arrival or gone before) and where
    it goes next; which = 'GLU' (upper ring) or 'LOWER'."""
    ring_of = lambda x: "GLU" if group_label(x) in ("E141", "E152") or x.startswith("E") else "LOWER"
    a = hs[(hs["start"] == "all") & (hs["n_resident"] >= min_res) & hs["site"].map(lambda x: ring_of(x) == which)]
    if not len(a):
        return
    cells = [[r["site"], f"{r['pct_resident']:.0f}%", f"{r['pct_leave_near_arrival']:.0f}%",
              f"{r['pct_leave_before'] + r['pct_leave_after']:.0f}%", f"{r['median_lag_ns']:+.1f} ns"] for r in a.to_dict("records")]
    fills = {}
    for i, r in enumerate(a.to_dict("records")):
        fills[(i, 2 if r["pct_leave_near_arrival"] >= 60 else 3)] = PRES_TINT.get(group_label(r["site"]), "E6E6E6")
    name = "02_glu_tracking" if which == "GLU" else "04_bottom_tracking"
    _ptable(cells, ["residue", "K+ sitting there\nwhen the new ion lands", "that K+ is pushed off\nas the new ion lands",
                    "that K+ had already\nleft (> 1 ns before)", "median: when it left,\nrelative to arrival"],
            [1.6, 3.0, 3.0, 3.0, 2.8], tdir / f"{name}_the_K_already_there.png",
            f"{title}\nvisit 1, ion tracking: the K+ already sitting at the residue when the new ion lands", fills=fills)
    res = [(r["site"], r["site"], None) for r in a.to_dict("records")]
    rows, cols = _dest_table(hv, hs, res, sites)
    if rows:
        _ptable(rows, cols, [1.9, 1.0] + [2.4 if "(" in c else 1.3 for c in cols[2:-1]] + [1.3], tdir / f"{name}_where_it_goes.png",
                f"{title}\nvisit 1, ion tracking: where the K+ pushed off (or gone from) the residue goes next",
                note="a residue group with several members gives the total and, in brackets, the split per member (%)")


def last_stop_counts(st, vt, sites):
    """Per residue: ions whose last stop before leaving the cavity is that residue (stops x % out), and
    the ions leaving without any stop; [(label, ions, % of ions)] in site order, then 'no stop'."""
    a = vt[vt["start"] == "all"]
    if not len(a) or not len(st):
        return [], 0
    n = int(a["n_ions"].iloc[0])
    exits = {r["site"]: r["visits"] * r["pct_out"] / 100 for r in st.to_dict("records")}
    rows = [(x, exits.get(x, 0.0), 100 * exits.get(x, 0.0) / n) for x in sites]
    nost = float(a.set_index("target")["count"].get(EXIT, 0))
    rows.append(("no stop", nost, 100 * nost / n))
    return rows, n


def last_stop_by_visit(visits, n_ions, n_nostop, sites, bins=((1, 1), (2, 2), (3, 3), (4, 6), (7, None))):
    """Ions leaving the cavity, grouped by the visit their last stop was (visit 1 = they left after their
    first stop, ...): per group the ions (% of all ions) and the % whose last stop was each residue;
    first row the ions leaving without a stop. visits = visits.csv (counted from 0)."""
    if not len(visits) or not n_ions:
        return [], []
    last = visits.sort_values("visit").groupby(["run", "event"]).tail(1)
    last = last[last["next_site"] == EXIT]
    k = last["visit"] + 1
    rows = [["no stop", n_nostop, 100 * n_nostop / n_ions] + [None] * len(sites)]
    for lo, hi in bins:
        g = last[(k >= lo) & ((k <= hi) if hi else True)]
        lab = f"visit {lo}" if lo == hi else (f"visit {lo}-{hi}" if hi else f"visit {lo}+")
        c = g["site"].value_counts()
        rows.append([lab, len(g), 100 * len(g) / n_ions] + [100 * c.get(x, 0) / len(g) if len(g) else 0 for x in sites])
    c = last["site"].value_counts()
    rows.append(["all", len(last), 100 * len(last) / n_ions] + [100 * c.get(x, 0) / len(last) if len(last) else 0 for x in sites])
    return rows, ["last stop", "ions", "% of ions"] + list(sites)


def pres_last_stop_by_visit(visits, vt, sites, tdir, title):
    """table: of the ions leaving after visit k, which residue was the last stop (row %)."""
    a = vt[vt["start"] == "all"]
    if not len(a):
        return
    n = int(a["n_ions"].iloc[0])
    nost = int(a.set_index("target")["count"].get(EXIT, 0))
    rows, cols = last_stop_by_visit(visits, n, nost, sites)
    if not rows:
        return
    cells = [[r[0], str(r[1]), f"{r[2]:.0f}%"] + ["" if v is None else f"{v:.0f}%" for v in r[3:]] for r in rows]
    heads = [None, None, None] + [PRES_COLORS.get(group_label(x), "123B4A").lstrip("#") for x in sites]
    _ptable(cells, ["ions left\nafter", "ions", "% of\nall ions"] + [x.split(".G")[0].replace(".", "\n") for x in sites],
            [1.6, 0.9, 1.0] + [1.0] * len(sites), tdir / "06b_last_stop_by_visit.png",
            f"{title}\nions leaving the cavity: which residue was the last stop, by the visit it was (row %)",
            head_colors=heads, bold={(len(cells) - 1, j) for j in range(len(cols))}, head_size=11)


def pres_table_bottom(st, vt, occ, sites, tdir, title, min_visits=15, visits=None):
    """slide 'after a stop': for every residue (all visits) where the ion goes next (another GLU-ring
    residue / the lower ring / out), the last stop before leaving per residue, and the asparagines as
    first stop / occupied by K+ in the cavity."""
    if len(st):
        a = st[st["visits"] >= min_visits]
        _ptable([[r["site"], str(r["visits"]), f"{r['pct_to_glu']:.0f}%", f"{r['pct_to_lower']:.0f}%", f"{r['pct_out']:.0f}%"]
                 for r in a.to_dict("records")],
                ["after a stop at", "stops", "next: a Glu\n(upper ring)", "next: D173 / Asn\n(lower ring)", "next: out of\nthe cavity"],
                [2.0, 1.1, 1.9, 2.1, 1.8], tdir / "05_after_a_stop.png",
                f"{title}\nall visits: where the ion goes after a stop at each residue",
                fills={(i, 4): PRES_TINT[EXIT] for i in range(len(a))})
    rows, n = last_stop_counts(st, vt, sites)
    if rows:
        _ptable([[r[0], f"{r[1]:.0f}", f"{r[2]:.0f}%"] for r in rows], ["last stop before leaving", "ions", "% of ions"],
                [3.0, 1.2, 1.4], tdir / "06_last_stop_before_leaving.png",
                f"{title}\nlast stop before leaving the cavity ({n} ions; exits = stops x % out, all visits)")
    if visits is not None and len(visits):
        pres_last_stop_by_visit(visits, vt, sites, tdir, title)
    asn = [x for x in sites if group_label(x) == "N184"]
    bins = [k for k, _ in _pres_bins(vt)]
    if asn and bins and len(occ):
        o = occ.set_index("start")
        rows = []
        for k in bins:
            g = vt[vt["start"] == k].set_index("target")["pct"]
            rows.append([k, " / ".join(f"{g.get(x, 0):.0f}" for x in asn) + "%",
                         " / ".join(f"{o.loc[k, x]:.0f}" for x in asn if k in o.index) + "%"])
        sub = " / ".join(x.split(".", 1)[1] for x in asn)
        _ptable(rows, ["other K+", f"new ions → N184\n({sub})", f"N184 occupied\n({sub})"], [1.4, 2.8, 2.8],
                tdir / "07_asparagines_by_load.png", f"{title}\nvisit 1: the asparagines as first stop and occupied, by the other K+ in the cavity")


def presentation_tables(out_dir, title, sites):
    """out_dir/presentation/tables/: the tables of the slides (PNG + CSV), from visits/, occupancy/,
    handover/ and stops/."""
    out_dir = Path(out_dir)
    tdir = out_dir / "presentation" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    rd = lambda p_: read_csv_or_empty(p_) if Path(p_).exists() else pd.DataFrame()
    vt, occ = rd(out_dir / "visits" / "visit_1.csv"), rd(out_dir / "occupancy" / "visit_1_occupied_by_k.csv")
    rt, kt = rd(out_dir / "occupancy" / "visit_1_free_vs_occupied_residue.csv"), rd(out_dir / "visits" / "visit_1_k_time.csv")
    rules = rd(out_dir / "occupancy" / "robustness" / "visit_1_free_vs_occupied_rules.csv")
    hs, hv = rd(out_dir / "handover" / "handover_visit_1_summary.csv"), rd(out_dir / "handover" / "handover_visits.csv")
    st = rd(out_dir / "stops" / "stops.csv")
    visits = rd(out_dir / "visits.csv")
    for d in (vt, occ, rt, kt, rules, hs):
        if len(d) and "start" in d:
            d["start"] = d["start"].astype(str)
    if len(rules) and "rule" in rules:
        rules = rules[rules["rule"] != "trip (main)"].pipe(lambda x: pd.concat([rules[rules["rule"] == "trip (main)"], x]))
    if len(hv):
        hv = hv[hv["visit"] == 0].copy()
        hv["start"] = hv["start"].astype(str)
    jobs = [("first stop and occupancy", lambda: pres_table_load(vt, occ, rt, kt, rules, sites, tdir, title) if len(vt) and len(occ) and len(rt) else None),
            ("glu tracking", lambda: pres_table_tracking(hs, hv, sites, tdir, title, "GLU") if len(hs) else None),
            ("bottom tracking", lambda: pres_table_tracking(hs, hv, sites, tdir, title, "LOWER") if len(hs) else None),
            ("bottom", lambda: pres_table_bottom(st, vt, occ, sites, tdir, title, visits=visits))]
    for name, fn in jobs:
        try:
            fn()
        except Exception as e:                                 # one table failing must not stop the others
            print(f"site_chain: presentation table '{name}' not drawn: {e}")


def presentation_figures(out_dir, title, sites):
    """out_dir/presentation/: the visit-1 story, one message per figure, from the tables written by
    the analysis (visits/, occupancy/, handover/, stops/), plus speech_numbers.txt with the numbers
    to say for each figure."""
    out_dir = Path(out_dir)
    pdir = out_dir / "presentation"
    pdir.mkdir(exist_ok=True)
    rd = lambda p_: read_csv_or_empty(p_) if Path(p_).exists() else pd.DataFrame()
    vt = rd(out_dir / "visits" / "visit_1.csv")
    occ = rd(out_dir / "occupancy" / "visit_1_occupied_by_k.csv")
    rt = rd(out_dir / "occupancy" / "visit_1_free_vs_occupied_residue.csv")
    hs = rd(out_dir / "handover" / "handover_visit_1_summary.csv")
    hv = rd(out_dir / "handover" / "handover_visits.csv")
    st = rd(out_dir / "stops" / "stops.csv")
    for d in (vt, occ, rt, hs):
        if len(d) and "start" in d:
            d["start"] = d["start"].astype(str)
    if len(hv):
        hv = hv[hv["visit"] == 0]
    rep = _Report()
    rep(f"PRESENTATION NUMBERS  {title}  (visit 1 = the new ion's first stop after the SF)")
    steps = [
        ("01_first_stop.png", "Where a new ion goes first, by the other K+ in the cavity",
         lambda L: pres_first_stop(vt, pdir / "01_first_stop.png", L) if len(vt) else None),
        ("02_who_holds_a_K.png", "% of new ions finding each residue already holding a K+",
         lambda L: pres_occupancy(occ, vt, sites, pdir / "02_who_holds_a_K.png", L) if len(occ) and len(vt) else None),
        ("03_free_vs_occupied.png", "% of new ions going to the residue: free vs already holding a K+",
         lambda L: pres_free_vs_occupied(rt, pdir / "03_free_vs_occupied.png", L) if len(rt) else None),
        ("04_K_already_there.png", "When the new ion lands: the K+ already there pushed off at that moment, or gone before",
         lambda L: pres_resident_at_arrival(hs, pdir / "04_K_already_there.png", L) if len(hs) else None),
        ("05_where_pushed_K_goes.png", "Where the pushed K+ goes next",
         lambda L: pres_where_pushed(hv, hs, pdir / "05_where_pushed_K_goes.png", L) if len(hs) else None),
        ("06_storage_push.png", "Storage seats: new ions avoid them when full, and more leave the cavity",
         lambda L: pres_storage_push(rt, pdir / "06_storage_push.png", L) if len(rt) else None),
        ("07_after_a_stop.png", "After a stop at each residue: to a Glu, the lower ring, or out (all visits)",
         lambda L: pres_lower_ring(st, pdir / "07_after_a_stop.png", L) if len(st) else None),
    ]
    for fname, what, fn in steps:
        lines = []
        try:
            fn(lines)
        except Exception as e:                                 # one figure failing must not stop the others
            lines.append(f"  (not drawn: {e})")
        rep(f"\n{fname}: {what}")
        for l_ in lines or ["  (not drawn: the data it needs are missing)"]:
            rep(l_)
    rep.write(pdir / "speech_numbers.txt")
    presentation_tables(out_dir, title, sites)


def leaving_by_visit_table(events, dt, max_visit=LEAVE_VISITS):
    """
    The ions grouped by the visit at which they leave the cavity: visit k = left
    with k-1 visits (visit 1 = no visit, SF -> Exit), the last group max_visit+ =
    max_visit-1 or more visits. Columns: visit (label), n_visits_min, n_ions,
    pct (of all permeating ions), mean_ns / sem_ns / sd_ns / median_ns of the
    transit end_1 -> end_2.
    """
    n = events["n_visits"].astype(int).to_numpy()
    t = events["transit_frames"].to_numpy() * dt
    rows = []
    for k in range(1, max_visit + 1):
        last = k == max_visit
        m = n >= k - 1 if last else n == k - 1
        x = t[m]
        rows.append(dict(visit=f"{k}+" if last else str(k), n_visits_min=k - 1, n_ions=int(m.sum()),
                         pct=100 * m.sum() / max(len(n), 1),
                         mean_ns=x.mean() if len(x) else np.nan,
                         sem_ns=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                         sd_ns=x.std(ddof=1) if len(x) > 1 else np.nan,
                         median_ns=np.median(x) if len(x) else np.nan))
    return pd.DataFrame(rows)


def _ns_label(t):
    return f"{t:.1f} ns" if t < 10 else f"{t:.0f} ns"


def plot_leaving_by_visit(tab, title, out_file):
    """Per visit k: % of the permeating ions that leave the cavity at visit k (left axis) and their
    mean time from leaving the SF to leaving the cavity, +- standard error (right axis)."""
    x = np.arange(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, tab["pct"], w, color="#7f7f7f", label="% of ions leaving")
    b2 = ax2.bar(x + w / 2, tab["mean_ns"], w, yerr=tab["sem_ns"], capsize=4, color="#4f81bd",
                 error_kw=dict(lw=1.5), label="time SF to exit (mean ± SE)")
    top = max(tab["pct"].max(), 10)
    for xi, (p, c) in enumerate(zip(tab["pct"], tab["n_ions"])):
        if c:
            ax.text(xi - w / 2, p + 0.01 * top, f"({c})", ha="center", va="bottom", fontsize=12)
            ax.text(xi - w / 2, p + 0.065 * top, f"{p:.0f}%", ha="center", va="bottom", fontsize=17,
                    fontweight="bold")
    t_top = np.nanmax((tab["mean_ns"] + tab["sem_ns"].fillna(0)).to_numpy())
    for xi, (m, e) in enumerate(zip(tab["mean_ns"], tab["sem_ns"].fillna(0))):
        if np.isfinite(m):
            ax2.text(xi + w / 2, m + e + 0.01 * t_top, _ns_label(m), ha="center", va="bottom", fontsize=13,
                     color="#2f5597", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"visit {v}" for v in tab["visit"]], rotation=30, ha="right", fontsize=17)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("% of permeating ions leaving", fontsize=19)
    ax2.set_ylabel("time SF to exit (ns)", color="#4f81bd", fontsize=19)
    ax2.tick_params(axis="y", colors="#4f81bd", labelsize=17)
    ax.set_ylim(0, top * 1.4)
    ax2.set_ylim(0, t_top * 1.25)
    ax.legend(handles=[b1, b2], loc="upper center", fontsize=16)
    ax.set_title(f"{title}\nions leaving the cavity at each visit (visit 1 = no stop)", fontsize=19)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_visit_histogram(events, max_sf, title, out_file):
    """Number of visits per ion: all ions (counts), and % per K+ count at end_1."""
    n = events["n_visits"].astype(int).to_numpy()
    if not len(n):
        return
    hi = int(n.max())
    b = np.arange(hi + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    cnt = np.bincount(n, minlength=hi + 1)
    ax1.bar(b, cnt, color="#555555")
    for bi, c in zip(b, cnt):
        if c:
            ax1.text(bi, c, str(c), ha="center", va="bottom", fontsize=11)
    ax1.set_xlabel("visits per ion")
    ax1.set_ylabel("ions")
    ax1.set_title(f"all ions (n={len(n)}, mean {n.mean():.2f}, median {np.median(n):.0f})")
    lab = events["sf_count"].map(lambda v: bin_label(v, max_sf)).to_numpy()
    cmap = plt.get_cmap("viridis")
    bins = [bin_label(k, max_sf) for k in range(max_sf + 1)]
    groups = [("0", n == 0), ("1", n == 1), ("2", n == 2), ("3", n == 3), ("4", n == 4),
              ("5-9", (n >= 5) & (n <= 9)), ("10+", n >= 10)]
    used = [s for s in bins if (lab == s).any()]
    w = 0.8 / max(len(used), 1)
    for i, s in enumerate(used):
        m = lab == s
        ax2.bar(np.arange(len(groups)) - 0.4 + w * (i + 0.5), [100 * (g & m).sum() / m.sum() for _, g in groups],
                width=w, color=cmap(bins.index(s) / max(len(bins) - 1, 1)), label=f"{s} (n={m.sum()})")
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels([g for g, _ in groups])
    ax2.set_xlabel("visits per ion")
    ax2.set_ylabel("% of ions")
    ax2.legend(title="other K+ when leaving the SF", fontsize=12)
    ax2.set_title("by the other K+ in the cavity when the ion leaves the SF")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_time_budget(ev, visits_site, sites, dt, title, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for col, c, lab in (("residue_frames", "#c0504d", "residue time"), ("moving_frames", "#4f81bd", "moving time"),
                        ("transit_frames", "black", "whole transit")):
        x = np.sort(ev[col].to_numpy() * dt)
        ax1.step(np.maximum(x, dt / 2), 1 - np.arange(len(x)) / len(x), where="post", color=c,
                 label=f"{lab} (mean {x.mean():.1f} ns)")
    ax1.set_xscale("log")
    ax1.set_xlabel("time per ion (ns)")
    ax1.set_ylabel("fraction of ions with >= t")
    ax1.legend(fontsize=12)
    per_ion = {s: visits_site.loc[visits_site["site"] == s, "residue_frames"].sum() * dt / len(ev) for s in sites}
    names = sites + ["moving"]
    vals = [per_ion[s] for s in sites] + [ev["moving_frames"].mean() * dt]
    ax2.bar(range(len(names)), vals, color=[site_color(s) for s in sites] + ["#4f81bd"])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("mean time per ion (ns)")
    ax2.set_title("where the transit time goes")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_time_to_exit(st, title, out_file):
    st = st[np.isfinite(st["model_to_exit_ns"]) & np.isfinite(st["mean_to_exit_ns"])]
    if not len(st):
        return
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for s, g in st.groupby("site"):
        ax.scatter(g["model_to_exit_ns"], g["mean_to_exit_ns"], s=12 + 8 * np.sqrt(g["departures"]),
                   color=site_color(s) if s != SF else GREY, edgecolor="black", alpha=0.8, label=s)
    lim = [0, max(st["model_to_exit_ns"].max(), st["mean_to_exit_ns"].max()) * 1.05]
    ax.plot(lim, lim, color="grey", ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean time to Exit, chain (ns)")
    ax.set_ylabel("mean time to Exit, measured (ns)")
    ax.legend(fontsize=11, title="site (size: departures)", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_prediction(obs_t, sim_t, obs_n, sim_n, title, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for arr, c, lab in ((obs_t, "black", "real"), (sim_t, "#1c5cab", "chain")):
        x = np.sort(np.maximum(arr, 1e-3))
        ax1.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, label=lab)
    ax1.set_xscale("log")
    ax1.set_xlabel("transit time, SF to exit (ns)")
    ax1.set_ylabel("fraction of ions out by t")
    ax1.legend()
    hi = int(max(obs_n.max(initial=0), np.percentile(sim_n, 99.5))) + 1
    b = np.arange(hi + 1)
    ax2.bar(b - 0.2, np.bincount(np.minimum(obs_n, hi), minlength=hi + 1) / len(obs_n), width=0.4,
            color="black", label="real")
    ax2.bar(b + 0.2, np.bincount(np.minimum(sim_n, hi), minlength=hi + 1) / len(sim_n), width=0.4,
            color="#1c5cab", label="chain")
    ax2.set_xlabel("visits before Exit")
    ax2.set_ylabel("fraction of ions")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis of one model (one run, or pooled runs)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_model(visits, events, out_dir, head, frame_duration, model="residue", max_above=5,
                  max_below=3, max_sf=5, include_frame0=False, include_preequil=False, crowded=4,
                  n_sim=N_SIM, seed=0, occ_frames=10, frames=None):
    """Tables, tests and plots of one model; see the module docstring. events may carry
    eq_start_frame (per run): ions leaving the SF before it are left out unless include_preequil."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(frame_duration)
    rng = np.random.default_rng(seed)
    rep = _Report()
    events_all, visits_all = events, visits                  # every permeation, to follow a pushed K+
    n_all = len(events)
    n_frame0 = n_pre = 0
    eq_report = excluded_ids = None
    if len(events):
        eq = events["eq_start_frame"] if "eq_start_frame" in events else pd.Series(0, index=events.index)
        frame0 = events["starts_at_frame0"].astype(bool)
        pre = (events["end_1"] < eq) & ~frame0
        eq_report = (events.assign(pre=pre, eq=eq).groupby("run")
                     .agg(start=("eq", "first"), ions=("pre", "size"), before=("pre", "sum")))
        excluded_ids = events.loc[pre, "ion_id"].value_counts()
        drop = pd.Series(False, index=events.index)
        if not include_frame0:
            drop |= frame0
            n_frame0 = int(frame0.sum())
        if not include_preequil:
            drop |= pre
            n_pre = int(pre.sum())
        events = events[~drop]
        keep = set(zip(events["run"], events["event"]))
        visits = visits[[(r, e) in keep for r, e in zip(visits["run"], visits["event"])]] if len(visits) else visits
    if not len(events):
        rep(f"site_chain: no events for {head}")
        rep.write(out_dir / "summary.txt")
        return None
    visits = visits.copy()
    if len(visits):
        visits["model_site"] = visits["site"] if model == "residue" else visits["site"].map(group_names(visits["site"]))
    sites, ring = site_order(visits, model) if len(visits) else ([], {})
    vs = visits.assign(site=visits["model_site"]) if len(visits) else visits

    steps = steps_table(visits, events, model, max_above, max_below, max_sf)
    ctrl = pd.concat([steps_table(visits, events, model, max_above, max_below, max_sf, control=j)
                      for j in range(1, N_C + 1)], ignore_index=True)

    rep("=" * 78)
    rep(f"SITE-TO-SITE CHAIN  {head}   model: {model}")
    rep(f"  states: site | K+ above | K+ below when leaving (bins 0..{max_above}+ / 0..{max_below}+); "
        f"start SF|other K+ in the cavity at end_1 (0..{max_sf}+)")
    rep("=" * 78)
    rep(f"runs: {events['run'].nunique()}; ions: {len(events)} of {n_all}"
        + f" ({n_frame0} present at frame 0 and {n_pre} entering before equilibration left out)"
        + f"; visits: {len(visits)}; states seen: {steps['from_state'].nunique()}")
    if eq_report is not None and (eq_report["start"] > 0).any():
        rep("  equilibration start per run (ns) and ions entering before it"
            + (" (kept: include_preequil)" if include_preequil else " (left out)") + ":")
        rep("    " + "  ".join(f"RUN{r}: {d['start'] * dt:.0f} ns, {int(d['before'])}/{int(d['ions'])}"
                               for r, d in eq_report.iterrows()))
        if excluded_ids is not None and len(excluded_ids):
            rep("    ions most often entering before equilibration: "
                + ", ".join(f"{i} ({n}x)" for i, n in excluded_ids.head(6).items()))
    rep(f"ions with no visit (SF -> Exit): {int((events['n_visits'] == 0).sum())} "
        f"({100 * (events['n_visits'] == 0).mean():.0f}%); visits per ion: mean "
        f"{events['n_visits'].mean():.2f}, median {events['n_visits'].median():.0f}, max {events['n_visits'].max()}")

    # ── time budget ──
    rep("")
    rep("TIME BUDGET (ns per ion)")
    for col, name in (("transit_frames", "transit"), ("residue_frames", "at residues"),
                      ("moving_frames", "moving")):
        x = events[col] * dt
        rep(f"    {name:<14} mean {x.mean():8.2f}   median {x.median():8.2f}")
    rep(f"    moving = {100 * events['moving_frames'].sum() / max(events['transit_frames'].sum(), 1):.0f}% "
        "of all transit time")
    if len(vs):
        rep(f"    {'site':<12}{'visits':>8}{'ions':>7}{'mean ns/ion':>13}{'median visit ns':>17}"
            f"{'contact %':>11}")
        for s in sites:
            g = vs[vs["site"] == s]
            rep(f"    {s:<12}{len(g):>8}{g[['run', 'event']].drop_duplicates().shape[0]:>7}"
                f"{g['residue_frames'].sum() * dt / len(events):>13.2f}{g['residue_frames'].median() * dt:>17.2f}"
                f"{100 * g['contact_frames'].sum() / max(g['residue_frames'].sum(), 1):>10.0f}%")

    # ── paths ──
    paths = steps.groupby(["run", "event"])["to_site"].apply(lambda x: ">".join([SF] + list(x)))
    pc = paths.value_counts()
    rep("")
    rep("COMMONEST PATHS (sites only)")
    for p_, n in pc.head(12).items():
        rep(f"    {100 * n / len(paths):5.1f}%  ({n:>3})  {p_}")

    # ── first step ──
    rep("")
    rep("FIRST STEP BY THE OTHER K+ IN THE CAVITY AT end_1   % of ions (n)")
    first = steps[steps["from_site"] == SF]
    targets = sites + [EXIT]
    rep(f"    {'start':<8}{'n':>5}" + "".join(f"{t:>10}" for t in targets))
    for k in range(max_sf + 1):
        st_ = f"{SF}|{bin_label(k, max_sf)}"
        g = first[first["from_state"] == st_]
        if len(g):
            rep(f"    {st_:<8}{len(g):>5}" + "".join(f"{100 * (g['to_site'] == t).mean():>9.0f}%" for t in targets))

    # ── states ──
    model_t = time_to_exit_model(steps, dt)
    st_rows = []
    for s, g in steps.groupby("from_state"):
        site = s.split("|")[0]
        row = dict(state=s, site=site, departures=len(g))
        for t in targets:
            row[f"P_{t}"] = (g["to_site"] == t).mean()
        row["P_loop"] = (g["to_site"] == site).mean()
        row["median_residue_ns"] = np.nanmedian(g["residue_frames"]) * dt if g["residue_frames"].notna().any() else np.nan
        row["mean_residue_ns"] = np.nanmean(g["residue_frames"]) * dt if g["residue_frames"].notna().any() else np.nan
        row["median_moving_next_ns"] = g["moving"].median() * dt
        row["model_to_exit_ns"] = model_t.get(s, np.nan)
        st_rows.append(row)
    st = pd.DataFrame(st_rows)
    # measured time to exit from the start of the visits of each state
    meas = {}
    if len(vs):
        lab = (vs["site"] + "|" + vs["above"].map(lambda a: bin_label(a, max_above)) + "a"
               + vs["below"].map(lambda b: bin_label(b, max_below)) + "b")
        meas = (vs.assign(state=lab).groupby("state")["to_exit_frames"].mean() * dt).to_dict()
        cont = (vs.assign(state=lab).groupby("state")[["contact_frames", "residue_frames"]].sum())
        st["contact_pct"] = st["state"].map(lambda s: 100 * cont.loc[s, "contact_frames"] / cont.loc[s, "residue_frames"]
                                            if s in cont.index else np.nan)
    sf_meas = (events.assign(state=SF + "|" + events["sf_count"].map(lambda n: bin_label(n, max_sf)))
               .groupby("state")["transit_frames"].mean() * dt).to_dict()
    meas.update(sf_meas)
    st["mean_to_exit_ns"] = st["state"].map(meas)
    order = {s: i for i, s in enumerate([SF] + sites)}
    st["o"] = st["site"].map(order)
    st = st.sort_values(["o", "state"]).drop(columns="o")
    st.to_csv(out_dir / "states.csv", index=False, float_format="%.4f")
    rep("")
    rep("STATES: P(next site | state) % of departures; residue time of the visits; time to Exit "
        "(measured from the start of the visit / chain)")
    rep(f"    {'state':<20}{'n':>5}" + "".join(f"{t:>9}" for t in targets)
        + f"{'res ns':>8}{'to Exit':>9}{'chain':>7}")
    for d in st.to_dict("records"):
        rep(f"    {d['state']:<20}{d['departures']:>5}" + "".join(f"{100 * d['P_' + t]:>8.0f}%" for t in targets)
            + (f"{d['median_residue_ns']:>8.2f}" if np.isfinite(d["median_residue_ns"]) else f"{'':>8}")
            + f"{d['mean_to_exit_ns']:>9.1f}{d['model_to_exit_ns']:>7.1f}")
    rep("    (res ns: median residue time of the visits that left in this state; each row is shown "
        "whatever its n)")

    # ── transitions ──
    tr = (steps.groupby(["from_state", "to_state", "from_site", "to_site"])
          .agg(count=("moving", "size"), median_moving_ns=("moving", "median"), mean_moving_ns=("moving", "mean"))
          .reset_index())
    tr["p"] = tr["count"] / tr["from_state"].map(steps.groupby("from_state").size())
    tr["median_moving_ns"] *= dt
    tr["mean_moving_ns"] *= dt
    tr.to_csv(out_dir / "transitions.csv", index=False, float_format="%.4f")
    hops = (steps.groupby(["from_site", "to_site"]).agg(count=("moving", "size"), median=("moving", "median"))
            .reset_index())
    rep("")
    rep("HOPS BETWEEN SITES (all K+ counts together): count, P(to | from), median moving time")
    tot = steps.groupby("from_site").size()
    rank = {s: i for i, s in enumerate([SF] + sites + [EXIT])}
    hops["o"] = hops["from_site"].map(rank)
    for d in hops.sort_values(["o", "count"], ascending=[True, False]).to_dict("records"):
        rep(f"    {d['from_site']:>10} -> {d['to_site']:<10} {d['count']:>5}  {100 * d['count'] / tot[d['from_site']]:5.1f}%"
            f"  {d['median'] * dt:7.2f} ns")

    # ── exit by counts, real and control ──
    rows = []
    rep("")
    rep("P(EXIT NEXT | site, K+ above x K+ below when leaving): real (n) and control (n); "
        "control = other ions from frames 1/4, 1/2, 3/4 of the run later")
    for s in sites:
        P, N = exit_grid(steps, s, max_above, max_below)
        Pc, Nc = exit_grid(ctrl, s, max_above, max_below)
        rep(f"  {s}")
        for a in range(max_above + 1):
            for b in range(max_below + 1):
                if N[a, b] or Nc[a, b]:
                    rows.append(dict(site=s, above=bin_label(a, max_above), below=bin_label(b, max_below),
                                     n=int(N[a, b]), p_exit=P[a, b], n_control=int(Nc[a, b]), p_exit_control=Pc[a, b]))
                    rep(f"    {bin_label(a, max_above):>2} above {bin_label(b, max_below):>2} below:  real "
                        + (f"{P[a, b]:5.1f}%" if N[a, b] else "    -") + f" (n={int(N[a, b]):>3})   control "
                        + (f"{Pc[a, b]:5.1f}%" if Nc[a, b] else "    -") + f" (n={int(Nc[a, b]):>3})")
        for name, axis, top in (("above", 1, max_above), ("below", 0, max_below)):
            ex = np.nan_to_num(P * N / 100).sum(axis=axis)
            exc = np.nan_to_num(Pc * Nc / 100).sum(axis=axis)
            n1, nc1 = N.sum(axis=axis), Nc.sum(axis=axis)
            rep(f"    by K+ {name}:" + "".join(
                f"  {bin_label(k, top)}: {100 * ex[k] / n1[k]:.0f}% (n={int(n1[k])}; ctrl {100 * exc[k] / nc1[k]:.0f}%)"
                for k in range(top + 1) if n1[k]))
    pd.DataFrame(rows).to_csv(out_dir / "exit_by_counts.csv", index=False, float_format="%.2f")

    # ── memory ──
    rep("")
    rep("EXIT BY OTHER K+ IN THE CAVITY (when the ion leaves; SF: when it enters), real vs control")
    rep(f"    {'':<14}{'other K+':>9}{'departures':>12}{'exit %':>8}{'control %':>11}")
    by_tot = []
    for part, r_sub, c_sub in (("SF entries", steps[steps["from_site"] == SF], ctrl[ctrl["from_site"] == SF]),
                               ("site visits", steps[steps["from_site"] != SF], ctrl[ctrl["from_site"] != SF])):
        kr, kc = r_sub["from_state"].map(crowding).clip(upper=5), c_sub["from_state"].map(crowding).clip(upper=5)
        for k in range(6):
            g, gc = r_sub[kr == k], c_sub[kc == k]
            if not len(g):
                continue
            pr, pc = 100 * (g["to_site"] == EXIT).mean(), 100 * (gc["to_site"] == EXIT).mean() if len(gc) else np.nan
            by_tot.append(dict(part=part, other_k=f"{k}" if k < 5 else "5+", n=len(g), p_exit=pr,
                               n_control=len(gc), p_exit_control=pc))
            rep(f"    {part:<14}{by_tot[-1]['other_k']:>9}{len(g):>12}{pr:>7.0f}%{pc:>10.0f}%")
    pd.DataFrame(by_tot).to_csv(out_dir / "exit_by_cavity_k.csv", index=False, float_format="%.2f")
    rep("")
    rep(f"CROWDING CHECKS: is 'more K+ in the cavity -> more exits' real?   crowded = {crowded}+ other K+ in the "
        f"cavity when the ion leaves (SF: when it enters), uncrowded = {crowded - 1} or fewer (--crowded)")
    rep("  1 within the same time: crowded vs uncrowded departures compared only inside the same block of "
        "the same run (Cochran-Mantel-Haenszel), so slow changes of the protein cannot cause it")
    rep(f"    {'':<14}{'block':>7}{'blocks':>8}{'crowded':>9}{'exit %':>8}{'uncrowded':>11}{'exit %':>8}"
        f"{'crowded exits: real / expected':>32}{'p':>9}")
    rows_chk = []
    for part, sub in (("all", steps), ("SF entries", steps[steps["from_site"] == SF]),
                      ("site visits", steps[steps["from_site"] != SF])):
        for block in (25, 50, 100):
            r = crowding_within_blocks(sub, dt, block, crowded)
            rows_chk.append(dict(part=part, **r))
            rep(f"    {part:<14}{block:>5} ns{r['blocks']:>8}{r['crowded_n']:>9}{r['crowded_exit_pct']:>7.0f}%"
                f"{r['uncrowded_n']:>11}{r['uncrowded_exit_pct']:>7.0f}%"
                f"{r['crowded_exits']:>20} / {r['expected']:>9.1f}{r['p']:>9.2g}")
    pd.DataFrame(rows_chk).to_csv(out_dir / "crowding_within_blocks.csv", index=False, float_format="%.4f")
    per = crowding_per_run(steps, crowded)
    per.to_csv(out_dir / "crowding_per_run.csv", index=False, float_format="%.2f")
    cmp_ = per[per["compared"]]
    if len(cmp_):
        k = int((cmp_["crowded_exit_pct"] > cmp_["uncrowded_exit_pct"]).sum())
        from math import comb
        p_sign = sum(comb(len(cmp_), i) for i in range(k, len(cmp_) + 1)) / 2 ** len(cmp_)
        rep(f"  2 run by run (runs with >= 3 crowded and >= 3 uncrowded departures): crowded departures exit more "
            f"often in {k} of {len(cmp_)} runs (sign test p = {p_sign:.2g}; crowding_per_run.csv)")
        rep("    " + "  ".join(f"RUN{d['run']}: {d['crowded_exit_pct']:.0f}% vs {d['uncrowded_exit_pct']:.0f}%"
                               for d in cmp_.to_dict("records")))
    rep("")
    rep("MARKOV TEST 1: DOES THE NEXT SITE DEPEND ON THE SITE THE ION CAME FROM?")
    rep(f"  states with >= {MIN_TEST} departures; chi-square of previous site x next site (rows with >= 5); "
        "p < 0.05 = memory; tables for p < 0.05")
    mem = memory_test(steps)
    for m in mem:
        rep(f"  {m['state']}  (n={m['n']}, p = {m['p']:.3g})" if np.isfinite(m["p"])
            else f"  {m['state']}  (n={m['n']}, not testable)")
        if np.isfinite(m["p"]) and m["p"] < 0.05:
            t = m["table"]
            rep("      " + f"{FROM_NEXT:<12}" + "".join(f"{c:>9}" for c in t.columns))
            for idx, row in t.iterrows():
                rep(f"      {idx:<12}" + "".join(f"{100 * v / row.sum():>8.0f}%" for v in row) + f"   (n={row.sum()})")
    tested = [m for m in mem if np.isfinite(m["p"])]
    if tested:
        rep(f"  {sum(m['p'] < 0.05 for m in tested)} of {len(tested)} testable states have p < 0.05 "
            f"(about {0.05 * len(tested):.1f} expected by chance)")

    # ── residue times ──
    rep("")
    rep("MARKOV TEST 2: RESIDUE TIMES PER STATE (SD/mean = 1 for an exponential, i.e. memoryless)")
    sdm = [np.nanstd(g["residue_frames"]) / np.nanmean(g["residue_frames"])
           for _, g in steps.dropna(subset=["residue_frames"]).groupby("from_state") if len(g) >= MIN_TEST]
    if sdm:
        rep(f"    {len(sdm)} states with >= {MIN_TEST} visits: SD/mean median {np.median(sdm):.2f} "
            f"(range {min(sdm):.2f} - {max(sdm):.2f})")

    # ── prediction ──
    rep("")
    rep(f"MARKOV TEST 3: THE CHAIN RUN FORWARD ({n_sim} simulated ions) AGAINST THE REAL IONS")
    sim_t, sim_n, sim_m = simulate(steps, n_sim, rng)
    obs_t = events["transit_frames"].to_numpy()
    obs_n = events["n_visits"].to_numpy()
    rep(f"    transit:          real median {np.median(obs_t) * dt:.2f} ns mean {obs_t.mean() * dt:.2f}   chain "
        f"median {np.median(sim_t) * dt:.2f} mean {sim_t.mean() * dt:.2f}   KS {ks_gap(obs_t, sim_t):.2f} "
        f"(~{1.36 / np.sqrt(len(obs_t)):.2f} is the 5% level)")
    rep(f"    visits per ion:   real mean {obs_n.mean():.2f}   chain {sim_n.mean():.2f}   KS {ks_gap(obs_n, sim_n):.2f}")
    rep(f"    no visit:         real {100 * np.mean(obs_n == 0):.1f}%   chain {100 * np.mean(sim_n == 0):.1f}%")
    rep(f"    moving per ion:   real mean {events['moving_frames'].mean() * dt:.2f} ns   chain {sim_m.mean() * dt:.2f} ns")
    ok = st.dropna(subset=["mean_to_exit_ns", "model_to_exit_ns"])
    ok = ok[ok["departures"] >= MIN_TEST]
    if len(ok) > 2:
        r = np.corrcoef(ok["mean_to_exit_ns"], ok["model_to_exit_ns"])[0, 1]
        rep(f"    time to Exit per state (states with >= {MIN_TEST} departures): measured vs chain r = {r:.2f}, "
            f"median |difference| {np.median(np.abs(ok['mean_to_exit_ns'] - ok['model_to_exit_ns'])):.1f} ns")

    # ── CSVs and plots ──
    visits.to_csv(out_dir / "visits.csv", index=False)
    ev_out = events.copy()
    ev_out["path"] = ev_out.apply(lambda e: paths.get((e["run"], e["event"]), ""), axis=1)
    ev_out["transit_ns"] = ev_out["transit_frames"] * dt
    ev_out["residue_ns"] = ev_out["residue_frames"] * dt
    ev_out["moving_ns"] = ev_out["moving_frames"] * dt
    ev_out.to_csv(out_dir / "events.csv", index=False, float_format="%.3f")
    n_perm = f"{len(events)} permeating ions"
    head = f"{head[:-1]}, {n_perm})" if head.endswith(")") else f"{head} ({n_perm})"   # plot titles
    busy = len(sites) > 5                        # 8 residues: only the main hops drawn
    mp, lp = (0.10, 0.15) if busy else (0.05, 0.10)
    plot_network(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network.png", mp, lp)
    plot_network_crowding(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network_crowding.png", crowded,
                          min_prob=mp, label_prob=lp)
    plot_network_net(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network_net.png",
                     min_flow=0.03 if busy else 0.02, label_flow=0.06 if busy else 0.05)
    plot_exit_sources(steps, sites, f"{head}: {model} model", out_dir / "exit_sources.png", crowded)
    plot_exit_heatmaps(steps, ctrl, sites, max_above, max_below, f"{head}: {model} model",
                       out_dir / "exit_heatmaps.png")
    plot_first_step(steps, sites, max_sf, f"{head}: first site after the SF", out_dir / "first_step.png")
    vis_dir = out_dir / "visits"
    vis_dir.mkdir(exist_ok=True)
    plot_visit_histogram(events, max_sf, f"{head}: visits per ion", vis_dir / "visits_histogram.png")
    leave = leaving_by_visit_table(events, dt)
    leave.to_csv(vis_dir / "leaving_by_visit.csv", index=False, float_format="%.3f")
    plot_leaving_by_visit(leave, f"{head}", vis_dir / "leaving_by_visit.png")
    for k in PLOT_VISITS:
        tab = visit_table(events, vs, sites, max_sf, k)
        kt = visit_k_time_table(events, vs, max_sf, dt, k)
        name = f"visit_{k + 1}"                   # files and labels count visits from 1
        tab.to_csv(vis_dir / f"{name}.csv", index=False, float_format="%.2f")
        kt.to_csv(vis_dir / f"{name}_k_time.csv", index=False, float_format="%.3f")
        head_k = f"{head}: {model} model"
        plot_visit(tab, sites, head_k, vis_dir / f"{name}.png", k, ktime=kt)
        plot_visit(tab, sites, head_k, vis_dir / f"{name}_all.png", k, which=["all"], ktime=kt)
        plot_visit(tab, sites, head_k, vis_dir / f"{name}_by_k.png", k,
                   which=[b for b in tab["start"].unique() if b != "all"], ktime=kt)
        summ = visit_summary_table(tab, kt)
        summ.to_csv(vis_dir / f"{name}_summary.csv", index=False, float_format="%.2f")
        plot_visit_summary(summ, head_k, vis_dir / f"{name}_summary.png", k)
        plot_visit_k_time(kt, head_k, vis_dir / f"{name}_k_time.png", k)
    occupancy_analysis(events, visits, vs, sites, max_sf, f"{head}: {model} model", out_dir, dt, occ_frames)
    stop_analysis(events, visits, sites, ring, f"{head}: {model} model", out_dir, dt)
    if frames is not None and len(visits):
        handover_analysis(events, frames, visits, sorted(set(visits["site"]), key=lambda x: (ring.get(x) != "GLU", x))
                          if model != "residue" else sites, max_sf, f"{head}: {model} model", out_dir, dt,
                          events_all, visits_all)
    if model == "residue":
        presentation_figures(out_dir, head, sites)
    plot_time_budget(events, vs, sites, dt, f"{head}: time budget", out_dir / "time_budget.png")
    plot_time_to_exit(st, f"{head}: mean time to Exit per state", out_dir / "time_to_exit.png")
    plot_prediction(obs_t * dt, sim_t * dt, obs_n, sim_n, f"{head}: chain against real ions",
                    out_dir / "prediction.png")
    rep.write(out_dir / "summary.txt")
    print(f"\nsite_chain: results in {out_dir}")
    return st


def equilibration_start(frames, rmsd, dt, max_slope=EQ_MAX_SLOPE, window_ns=EQ_WINDOW_NS,
                        smooth_ns=EQ_SMOOTH_NS):
    """
    (start frame, level, smoothed RMSD, flat) of one run: the start of the first
    window of window_ns in which the smoothed RMSD is flat - |linear slope| <=
    max_slope (A per 100 ns), and the window's first point no further from the
    window's mean than a line of that slope would be (so a window straddling the
    end of the rise does not count); level = the window's mean RMSD. Later jumps
    do not move it. If no window is flat: start 0 (nothing cut), flat = False.
    """
    frames, rmsd = np.asarray(frames), np.asarray(rmsd, dtype=float)
    sm = pd.Series(rmsd).rolling(max(1, int(round(smooth_ns / dt))), center=True, min_periods=1).mean().to_numpy()
    w = max(3, int(round(window_ns / dt)))
    if len(sm) <= w:
        return int(frames[0]), float(sm.mean()), sm, False
    Y = np.lib.stride_tricks.sliding_window_view(sm, w)
    x = np.arange(w) - (w - 1) / 2
    slope = np.abs((Y - Y.mean(axis=1, keepdims=True)) @ x / (x @ x)) / dt * 100   # A per 100 ns
    mean = Y.mean(axis=1)
    near = np.abs(sm[:len(mean)] - mean) <= max_slope / 100 * window_ns / 2
    ok = np.flatnonzero((slope <= max_slope) & near)
    if not len(ok):
        return int(frames[0]), float(mean[0]), sm, False
    i = int(ok[0])
    return int(frames[i]), float(mean[i]), sm, True


def equilibration(rmsd, dt, out_base, max_slope=EQ_MAX_SLOPE, window_ns=EQ_WINDOW_NS, start_frames=None):
    """{run: start frame used}; writes equilibration.csv / .png in out_base."""
    start_frames = {int(k): int(v) for k, v in (start_frames or {}).items()}
    rows, curves = [], {}
    for run, g in (rmsd.groupby("run") if rmsd is not None and len(rmsd) else []):
        g = g.sort_values("frame")
        auto, plateau, sm, flat = equilibration_start(g["frame"], g["rmsd_fit_A"], dt, max_slope, window_ns)
        if not flat:
            print(f"site_chain: RUN{run}: the RMSD never gets flat; nothing cut (give a start with --start_frames)")
        used = start_frames.get(int(run), auto)
        rows.append(dict(run=run, auto_start_frame=auto, given_start_frame=start_frames.get(int(run)),
                         used_start_frame=used, used_start_ns=used * dt, rmsd_at_start_A=plateau,
                         never_flat=not flat))
        curves[run] = (g["frame"].to_numpy(), g["rmsd_fit_A"].to_numpy(), sm, flat, auto, used)
    for run, f in start_frames.items():
        if run not in curves:
            rows.append(dict(run=run, auto_start_frame=None, given_start_frame=f, used_start_frame=f,
                             used_start_ns=f * dt, rmsd_at_start_A=np.nan))
    if not rows:
        return {}
    table = pd.DataFrame(rows).sort_values("run")
    table.to_csv(Path(out_base) / "equilibration.csv", index=False, float_format="%.3f")
    if curves:
        n = len(curves)
        cols = min(4, n)
        fig, axes = plt.subplots(-(-n // cols), cols, figsize=(4.2 * cols, 2.8 * -(-n // cols)), squeeze=False)
        for ax, (run, (f, r, sm, flat, auto, used)) in zip(axes.flat, sorted(curves.items())):
            ax.plot(f * dt, r, lw=0.4, color="#9ab8d8")
            ax.plot(f * dt, sm, lw=1.5, color="#1c5cab")
            ax.axvspan(auto * dt, auto * dt + window_ns, color="#dddddd", zorder=0)
            ax.axvline(used * dt, color="#c0504d", lw=1.5, label=f"start {used * dt:.0f} ns")
            if used != auto:
                ax.axvline(auto * dt, color="#c0504d", lw=1, ls="--", label=f"automatic {auto * dt:.0f} ns")
            ax.set_title(f"RUN{run}" + ("" if flat else "  (never flat: nothing cut)"), fontsize=13)
            ax.legend(fontsize=11)
            ax.set_xlabel("time (ns)", fontsize=12)
            ax.set_ylabel("Ca RMSD, fitted (A)", fontsize=12)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"Equilibration: start of the first {window_ns:g} ns window (grey) in which\n"
                     f"the smoothed RMSD is flat (|slope| <= {max_slope:g} A per 100 ns)", fontsize=14)
        fig.tight_layout()
        fig.savefig(Path(out_base) / "equilibration.png", dpi=130)
        plt.close(fig)
    return dict(zip(table["run"], table["used_start_frame"]))


def analyze_all(frames, stays, events, out_base, head, frame_duration, window, models=MODELS,
                max_above=5, max_below=3, max_sf=5, include_frame0=False, rmsd=None, eq_max_slope=EQ_MAX_SLOPE,
                eq_window_ns=EQ_WINDOW_NS,
                start_frames=None, include_preequil=False, crowded=4, occ_frames=10):
    eq = equilibration(rmsd, frame_duration, out_base, eq_max_slope, eq_window_ns, start_frames)
    events = events.assign(eq_start_frame=events["run"].map(eq).fillna(0).astype(int))
    visits, ev = build_visits(frames, stays, events, window, occ_frames)
    for model in models:
        analyze_model(visits, ev, Path(out_base) / analysis_folder(model, max_above, max_below, include_frame0,
                                                                     include_preequil),
                      head, frame_duration, model=model, max_above=max_above, max_below=max_below,
                      max_sf=max_sf, include_frame0=include_frame0, include_preequil=include_preequil,
                      crowded=crowded, occ_frames=occ_frames, frames=frames)
    return visits, ev


def collect_rmsd(universe, run):
    """Fitted and unfitted Ca RMSD of every frame against frame 0."""
    from analysis.protein_rmsd_analysis import fitted_ca_rmsd

    frames, fit, nofit = fitted_ca_rmsd(universe, 0, len(universe.trajectory) - 1)
    return pd.DataFrame(dict(run=run, frame=frames, rmsd_fit_A=fit, rmsd_nofit_A=nofit))


# ─────────────────────────────────────────────────────────────────────────────
# One run
# ─────────────────────────────────────────────────────────────────────────────

def run_site_chain(universe, permeation_events, asn_residues, glu_residues, channel1, channel2,
                   results_dir, run, channel_type="G12", frame_duration=0.075, cutoff=3.5,
                   min_duration=None, window=None, models=MODELS, max_above=5, max_below=3, max_sf=5,
                   include_frame0=False, start_frame=None, eq_max_slope=EQ_MAX_SLOPE,
                   eq_window_ns=EQ_WINDOW_NS, include_preequil=False, crowded=None,
                   analysis_start_frame=0, force_rerun=False):
    """Collect the data of one run (cached in CSVs) and analyse it; see the module docstring.
    start_frame: equilibration start of this run, instead of the one found from the RMSD."""
    min_duration = default_min_duration(channel_type) if min_duration is None else int(min_duration)
    window = default_window(channel_type) if window is None else int(window)
    crowded = default_crowded(channel_type) if crowded is None else int(crowded)
    out_dir = Path(results_dir) / OUT_FOLDER / settings_folder(cutoff, min_duration)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{k}_run{run}.csv" for k in ("frames", "stays", "events")}
    if all(p.exists() for p in paths.values()) and not force_rerun:
        print(f"site_chain: using {paths['frames']}")
        frames = read_csv_or_empty(paths["frames"], keep_default_na=False)
        stays, events = read_csv_or_empty(paths["stays"]), read_csv_or_empty(paths["events"])
    recollect = force_rerun or not all(p.exists() for p in paths.values())
    if not recollect and len(frames) and ("occupied" not in frames.columns or "occupants" not in frames.columns):
        print("site_chain: frames without residue occupancy / occupant identities (older version): collecting again")
        recollect = True
    if recollect:
        frames, stays, events = collect_run(universe, permeation_events, asn_residues, glu_residues,
                                            channel1, channel2, run, channel_type, cutoff, min_duration,
                                            analysis_start_frame)
        for k, d in (("frames", frames), ("stays", stays), ("events", events)):
            d.to_csv(paths[k], index=False)
    if not len(events) or not len(frames):
        print(f"site_chain: RUN{run} has no permeation events: nothing to analyse")
        return frames, stays, events
    rmsd_csv = out_dir / f"rmsd_run{run}.csv"
    if rmsd_csv.exists() and not force_rerun:
        rmsd = pd.read_csv(rmsd_csv)
    else:
        print("site_chain: computing the fitted Ca RMSD")
        rmsd = collect_rmsd(universe, run)
        rmsd.to_csv(rmsd_csv, index=False, float_format="%.4f")
    analyze_all(frames, stays, events, out_dir, f"{channel_type} RUN{run}", frame_duration, window,
                models=models, max_above=max_above, max_below=max_below, max_sf=max_sf,
                include_frame0=include_frame0, rmsd=rmsd, eq_max_slope=eq_max_slope, eq_window_ns=eq_window_ns,
                start_frames=None if start_frame is None else {run: start_frame},
                include_preequil=include_preequil, crowded=crowded, occ_frames=min_duration)
    return frames, stays, events


def add_analysis_args(p):
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--max_above", type=int, default=5, help="K+ above binned 0 .. max_above-1, max_above+")
    p.add_argument("--max_below", type=int, default=3, help="K+ below binned 0 .. max_below-1, max_below+")
    p.add_argument("--max_sf", type=int, default=5, help="K+ in the cavity at end_1 binned 0 .. max_sf-1, max_sf+")
    p.add_argument("--window", type=int, help="bound frames the counts at leaving are taken over "
                                              "(default 4 for G12, 1 for G2)")
    p.add_argument("--include_frame0", action="store_true",
                   help="also analyse the ions present at frame 0 (end_1 == 0)")
    p.add_argument("--eq_max_slope", type=float, default=EQ_MAX_SLOPE,
                   help="A per 100 ns: the smoothed RMSD this flat counts as equilibrated")
    p.add_argument("--eq_window_ns", type=float, default=EQ_WINDOW_NS,
                   help="ns: length of the window that has to be that flat")
    p.add_argument("--crowded", type=int, help="other K+ in the cavity counted as crowded in the crowding checks "
                                               "(default 3 for G2, 4 for the other systems)")
    p.add_argument("--include_preequil", action="store_true",
                   help="also analyse the ions entering before equilibration")


def main(argv=None):
    import MDAnalysis as mda
    from analysis.channels import Channel
    from analysis.markov_check import G12_CONFIG, _int_list

    p = argparse.ArgumentParser(description="Site-to-site chain of one run.")
    p.add_argument("--channel_type", default="G12")
    p.add_argument("--run", type=int, default=3)
    p.add_argument("--top")
    p.add_argument("--traj")
    p.add_argument("--results_dir", help="the RUN dir of main.py (holds analyzer_state.pkl)")
    for name in ("upper1", "lower1", "upper2", "lower2", "glu_residues", "asn_residues"):
        p.add_argument(f"--{name}", type=_int_list)
    p.add_argument("--frame_duration", type=float)
    p.add_argument("--cutoff", type=float, default=3.5)
    p.add_argument("--min_duration", type=int, help="frames of a stay (default 10 for G12, 2 for G2)")
    add_analysis_args(p)
    p.add_argument("--start_frame", type=int, help="equilibration start of this run (default: from the RMSD)")
    p.add_argument("--force_rerun", action="store_true", help="recollect the data from the trajectory")
    args = p.parse_args(argv)

    keys = ("top", "traj", "results_dir", "upper1", "lower1", "upper2", "lower2",
            "glu_residues", "asn_residues", "frame_duration")
    cfg = {}
    if args.channel_type == "G12" and args.run <= 20:
        cfg = {k: (v.format(run=args.run) if isinstance(v, str) else v) for k, v in G12_CONFIG.items()}
    cfg.update({k: getattr(args, k) for k in keys if getattr(args, k) is not None})
    missing = [k for k in keys if k not in cfg]
    if missing:
        sys.exit(f"site_chain: no built-in settings for {args.channel_type} RUN{args.run}; "
                 f"pass --{' --'.join(missing)}")
    results_dir = Path(cfg["results_dir"])
    state_file = results_dir / "analyzer_state.pkl"
    if not state_file.exists():
        sys.exit(f"site_chain: {state_file} not found; run main.py for this run first")
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    u = mda.Universe(cfg["top"], cfg["traj"])
    ch1 = Channel(u, cfg["upper1"], cfg["lower1"], num=1, radius=2.5)
    ch2 = Channel(u, cfg["upper2"], cfg["lower2"], num=2, radius=15.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_site_chain(u, state["permeation_events"], cfg["asn_residues"], cfg["glu_residues"],
                       ch1, ch2, results_dir, args.run, channel_type=args.channel_type,
                       frame_duration=cfg["frame_duration"], cutoff=args.cutoff,
                       min_duration=args.min_duration, window=args.window, models=args.models,
                       max_above=args.max_above, max_below=args.max_below, max_sf=args.max_sf,
                       include_frame0=args.include_frame0, start_frame=args.start_frame,
                       eq_max_slope=args.eq_max_slope, eq_window_ns=args.eq_window_ns,
                       include_preequil=args.include_preequil, crowded=args.crowded,
                       force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
