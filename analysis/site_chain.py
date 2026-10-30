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
                         contact (closest within the cutoff), K+ above / below /
                         in the SF, the same counts for the 3 control shifts
    stays_run<run>.csv, events_run<run>.csv
    <model>_a<max_above>b<max_below>[_frame0][_preeq]/
        summary.txt
        visits.csv        one row per visit (times, counts when leaving, next, touches)
        events.csv        one row per ion (SF count, residue / moving time, path)
        states.csv        per state: departures, P(next site / Exit), times
        transitions.csv   per arrow: count, P, moving time
        exit_by_counts.csv  P(Exit next) per site x K+ above x K+ below, real and control
        crowding_within_blocks.csv, crowding_per_run.csv   the crowding checks of the summary
        network.png          P(next site | site), all K+ counts together
        network_crowding.png the same, split: departures with <= 3 / >= 4 other K+ in the cavity
        network_net.png      net flow (hops a -> b minus b -> a) per permeating ion
        exit_sources.png     last site before Exit, all and by cavity filling
        exit_heatmaps.png, first_step.png, time_budget.png,
        time_to_exit.png, prediction.png
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
FROM_NEXT = "from \\ next"
EQ_MAX_SLOPE = 0.5                # A per 100 ns: a smoothed RMSD this flat over EQ_WINDOW_NS = settled
EQ_WINDOW_NS = 30.0
EQ_SMOOTH_NS = 7.5
N_C = len(CONTROL_SHIFTS)


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
    wanted = set()
    for e in events:
        for f in range(e["end_1"], e["end_2"] + 1):
            wanted.add(f)
            wanted.update((f + s) % n_frames for s in shifts)
    for f in sorted(wanted):                                   # read the trajectory once, in order
        pore.get(f)

    frames, stays = [], []
    for e in events:
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
                       contact=site_label(info["residue_pdb"], info["residue_type"]) if info else "")
            row["n_above"], row["n_below"], row["n_sf"] = pore.neighbours(ion, f)
            for j, s in enumerate(shifts, start=1):
                row[f"n_above_c{j}"], row[f"n_below_c{j}"], _ = pore.neighbours(ion, f, (f + s) % n_frames)
            frames.append(row)
    stay_cols = ["run", "event", "residue_id", "residue_pdb", "resname", "ring", "start_frame", "end_frame"]
    return pd.DataFrame(frames), pd.DataFrame(stays, columns=stay_cols), pd.DataFrame(events)


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


def build_visits(frames, stays, events, window):
    """
    (visits, events) DataFrames: visits one row per visit (see the module
    docstring), counts when leaving for the real ions (above, below) and the
    controls (above_c1, below_c1, ...); events with the SF count, residue and
    moving frames, and the touches before the first visit.
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
            pos[s] = ((col[sx] - (len(cols) - 1) / 2) * 2.0, 2.6 if ring[s] == "GLU" else 0.6)
    else:
        for r, y in (("GLU", 2.6), ("LOWER", 0.6)):
            row = [s for s in sites if (ring[s] == "GLU") == (r == "GLU")]
            for i, s in enumerate(row):
                pos[s] = ((i - (len(row) - 1) / 2) * 2.6, y)
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
    size = {s: 1700 + 2800 * np.sqrt(totals.get(s, 0) / top) for s in sites}
    size[SF] = size[EXIT] = 2600
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
            ax.text(mx, my, text, fontsize=8, ha="center", va="center", zorder=4, color=color,
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
        ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
    xs = [x for x, _ in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(-2.2, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def plot_network(steps, sites, ring, dt, title, out_file, min_prob=0.05, label_prob=0.10):
    """Sites as nodes; arrows: P(next site | site), all K+ counts together."""
    fig, ax = plt.subplots(figsize=(13, 9))
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
    fig, axes = plt.subplots(1, 2, figsize=(9 + 1.3 * len(sites), 9.5))
    for ax, sel, name in ((axes[0], k < crowded, f"{crowded - 1} or fewer other K+"),
                          (axes[1], k >= crowded, f"{crowded} or more other K+")):
        g = steps[sel]
        ax_title = (f"departures with {name} in the cavity ({len(g)} departures, "
                    f"{100 * (g['to_site'] == EXIT).mean():.0f}% to Exit)")
        draw_network(ax, g, sites, ring, dt, ax_title, min_prob, label_prob)
    fig.suptitle(f"{title}\nP(next site | site) split by how full the cavity is when the ion leaves "
                 f"(SF: when it enters); arrows >= {100 * min_prob:.0f}%", fontsize=12)
    fig.subplots_adjust(wspace=0.02)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network_net(steps, sites, ring, dt, title, out_file, min_flow=0.02, label_flow=0.05):
    """Net flow: hops a -> b minus b -> a, per permeating ion."""
    n_ions = int((steps["from_site"] == SF).sum())
    fig, ax = plt.subplots(figsize=(13, 9))
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
    ax.legend(title="exits", fontsize=8)
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
    fig, axes = plt.subplots(2, len(sites), figsize=(2.9 * len(sites) + 1, 8.5), squeeze=False)
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
                                fontsize=7, color="black" if N[a, b] >= 5 else "#777777")
            ax.set_xticks(range(max_below + 1))
            ax.set_xticklabels(b_lab)
            ax.set_yticks(range(max_above + 1))
            ax.set_yticklabels(a_lab)
            ax.set_xlabel("K+ below")
            if j == 0:
                ax.set_ylabel(f"{name}\nK+ above")
            ax.set_title(s, fontsize=10)
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
        ax.text(xi, 101, f"n={ni}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 110)
    ax.set_xlabel("other K+ in the cavity when the ion leaves the SF (end_1)")
    ax.set_ylabel("% of ions")
    ax.legend(title="first site", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title)
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
    ax1.legend(fontsize=8)
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
    ax.legend(fontsize=7, title="site (size: departures)", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_prediction(obs_t, sim_t, obs_n, sim_n, title, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for arr, c, lab in ((obs_t, "black", "real"), (sim_t, "#1c5cab", "chain")):
        x = np.sort(np.maximum(arr, 1e-3))
        ax1.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, label=lab)
    ax1.set_xscale("log")
    ax1.set_xlabel("transit time end_1 -> end_2 (ns)")
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
                  max_below=3, max_sf=5, include_frame0=False, include_preequil=False, n_sim=N_SIM,
                  seed=0):
    """Tables, tests and plots of one model; see the module docstring. events may carry
    eq_start_frame (per run): ions leaving the SF before it are left out unless include_preequil."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(frame_duration)
    rng = np.random.default_rng(seed)
    rep = _Report()
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
    rep("CROWDING CHECKS: is 'more K+ in the cavity -> more exits' real?   crowded = 4+ other K+ in the "
        "cavity when the ion leaves (SF: when it enters), uncrowded = 3 or fewer")
    rep("  1 within the same time: crowded vs uncrowded departures compared only inside the same block of "
        "the same run (Cochran-Mantel-Haenszel), so slow changes of the protein cannot cause it")
    rep(f"    {'':<14}{'block':>7}{'blocks':>8}{'crowded':>9}{'exit %':>8}{'uncrowded':>11}{'exit %':>8}"
        f"{'crowded exits: real / expected':>32}{'p':>9}")
    rows_chk = []
    for part, sub in (("all", steps), ("SF entries", steps[steps["from_site"] == SF]),
                      ("site visits", steps[steps["from_site"] != SF])):
        for block in (25, 50, 100):
            r = crowding_within_blocks(sub, dt, block)
            rows_chk.append(dict(part=part, **r))
            rep(f"    {part:<14}{block:>5} ns{r['blocks']:>8}{r['crowded_n']:>9}{r['crowded_exit_pct']:>7.0f}%"
                f"{r['uncrowded_n']:>11}{r['uncrowded_exit_pct']:>7.0f}%"
                f"{r['crowded_exits']:>20} / {r['expected']:>9.1f}{r['p']:>9.2g}")
    pd.DataFrame(rows_chk).to_csv(out_dir / "crowding_within_blocks.csv", index=False, float_format="%.4f")
    per = crowding_per_run(steps)
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
    busy = len(sites) > 5                        # 8 residues: only the main hops drawn
    mp, lp = (0.10, 0.15) if busy else (0.05, 0.10)
    plot_network(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network.png", mp, lp)
    plot_network_crowding(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network_crowding.png",
                          min_prob=mp, label_prob=lp)
    plot_network_net(steps, sites, ring, dt, f"{head}: {model} model", out_dir / "network_net.png",
                     min_flow=0.03 if busy else 0.02, label_flow=0.06 if busy else 0.05)
    plot_exit_sources(steps, sites, f"{head}: {model} model", out_dir / "exit_sources.png")
    plot_exit_heatmaps(steps, ctrl, sites, max_above, max_below, f"{head}: {model} model",
                       out_dir / "exit_heatmaps.png")
    plot_first_step(steps, sites, max_sf, f"{head}: first site after the SF", out_dir / "first_step.png")
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
            ax.set_title(f"RUN{run}" + ("" if flat else "  (never flat: nothing cut)"), fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("time (ns)", fontsize=8)
            ax.set_ylabel("Ca RMSD, fitted (A)", fontsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"Equilibration: start of the first {window_ns:g} ns window (grey) in which\n"
                     f"the smoothed RMSD is flat (|slope| <= {max_slope:g} A per 100 ns)", fontsize=10)
        fig.tight_layout()
        fig.savefig(Path(out_base) / "equilibration.png", dpi=130)
        plt.close(fig)
    return dict(zip(table["run"], table["used_start_frame"]))


def analyze_all(frames, stays, events, out_base, head, frame_duration, window, models=MODELS,
                max_above=5, max_below=3, max_sf=5, include_frame0=False, rmsd=None, eq_max_slope=EQ_MAX_SLOPE,
                eq_window_ns=EQ_WINDOW_NS,
                start_frames=None, include_preequil=False):
    eq = equilibration(rmsd, frame_duration, out_base, eq_max_slope, eq_window_ns, start_frames)
    events = events.assign(eq_start_frame=events["run"].map(eq).fillna(0).astype(int))
    visits, ev = build_visits(frames, stays, events, window)
    for model in models:
        analyze_model(visits, ev, Path(out_base) / analysis_folder(model, max_above, max_below, include_frame0,
                                                                     include_preequil),
                      head, frame_duration, model=model, max_above=max_above, max_below=max_below,
                      max_sf=max_sf, include_frame0=include_frame0, include_preequil=include_preequil)
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
                   eq_window_ns=EQ_WINDOW_NS, include_preequil=False,
                   analysis_start_frame=0, force_rerun=False):
    """Collect the data of one run (cached in CSVs) and analyse it; see the module docstring.
    start_frame: equilibration start of this run, instead of the one found from the RMSD."""
    min_duration = default_min_duration(channel_type) if min_duration is None else int(min_duration)
    window = default_window(channel_type) if window is None else int(window)
    out_dir = Path(results_dir) / OUT_FOLDER / settings_folder(cutoff, min_duration)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{k}_run{run}.csv" for k in ("frames", "stays", "events")}
    if all(p.exists() for p in paths.values()) and not force_rerun:
        print(f"site_chain: using {paths['frames']}")
        frames = pd.read_csv(paths["frames"], keep_default_na=False)
        stays, events = pd.read_csv(paths["stays"]), pd.read_csv(paths["events"])
    else:
        frames, stays, events = collect_run(universe, permeation_events, asn_residues, glu_residues,
                                            channel1, channel2, run, channel_type, cutoff, min_duration,
                                            analysis_start_frame)
        for k, d in (("frames", frames), ("stays", stays), ("events", events)):
            d.to_csv(paths[k], index=False)
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
                include_preequil=include_preequil)
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
                       include_preequil=args.include_preequil,
                       force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
