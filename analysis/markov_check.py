"""
markov_check.py
===============
Is ion movement through the SF and the GLU/ASN cavity Markovian? Five state
models are tested before building a full Markov model: three geometric ones
(M1-M3, split at z_mid) and two built on binding (M4, M5).

Standalone: nothing in main.py or in the other modules is changed. Call
run_markov_check(...) from main.py, or run this file from the command line
(see CLI below).

STATES (every frame, every K+ while it is in the SF or the cavity)
------------------------------------------------------------------
Regions are the ones of main.py, tested exactly like
IonPermeationAnalysis._determine_ion_region (region 1 first, then region 2):
    ch1 = Channel(u, upper1, lower1, num=1, radius=2.5)    region 1 = SF
    ch2 = Channel(u, upper2, lower2, num=2, radius=15.0)   region 2 = cavity
with compute_geometry(1) / compute_geometry(2) every frame.

Per frame:
    z_glu   mean z of the CD atoms of glu_residues (GLU or GLN)
    z_low   mean z of the CG atoms of asn_residues (ASN or ASP)
    z_mid   (z_glu + z_low) / 2

    SF      inside region 1
    G       inside region 2, on the SF side of z_mid
    A       inside region 2, on the other side of z_mid
    Exit    after end_2 of a permeation event of this ion (the ion has left
            the cavity through the gate). Absorbing: the ion is not tracked
            after it.
"On the SF side" uses the sign of (z of the ch1 centre - z_mid) of that
frame, so it is "above z_mid" when the SF is above the cavity; frames where
the SF is below are counted and reported.

A TRACKED SEGMENT is a run of consecutive frames in which the ion is in
region 1 or 2. It ends with Exit when its last frame is an end_2 of the ion
(if an end_2 falls inside a segment, the segment is cut there and the
mismatch is reported). Any other end (the ion leaves the SF upwards, leaves
the cavity sideways, makes an excursion below the cavity and comes back, or
the trajectory ends) is censored: no transition is counted past it.

PERMEATING IONS ONLY (permeating_only=True, --permeating_only)
-----------------------------------------------------------
Only the permeation events, each from end_1 + 1 (the ion has left the SF
for the last time) to end_2 (its last cavity frame), Exit right after: one
segment per event, nothing censored. There is no SF state. Frames of the
window outside region 2 (an excursion out of the cavity cylinder, mostly
below the lower ring, and back) are the state Out in every model. M3 still
counts every other K+ in the cavity. The transition
probabilities are then conditioned on the ion leaving through the gate:
they describe how permeating ions move, not the equilibrium of all cavity
ions. Results go to markov_check/RUN<run>_permeating/; the trajectory pass
is shared with the all-ions mode (cached in markov_check/RUN<run>/).

Models:
    M1   SF, G, A, Exit
    M2   SF, G, A_asp, A_other, Exit
         A_asp: in A and within asp_enter (3.5 A) of a side-chain O (OD1/OD2)
         of D173.G1; once in A_asp the ion stays there while it is in A and
         within asp_leave (4.5 A) (hysteresis). Leaving A resets it.
    M3   M2 x number of OTHER K+ in region 2 at that frame (<=2, 3, >=4),
         counting every K+ of the system; Exit is one state.
    M4   SF, Glu, Asp, Free_up, Free_low, Exit (binding, any z)
         Glu: within glu_enter (3.5 A) of a side-chain O/N of the glu ring
         (GLU OE1/OE2, GLN OE1/NE2), kept until glu_leave (4.5 A). Asp: the
         same for D173.G1 with asp_enter / asp_leave. If both are entered on
         the same frame the closer one wins. Otherwise free: Free_up /
         Free_low by the side of z_mid, switched only once the ion is
         z_buffer (1 A) past z_mid on the other side (no flicker at z_mid).
    M5   SF, Entry, Glu, Asp, Exit: core sets (milestoning). SF, Glu, Asp
         (as in M4) and Out are cores; every other frame carries the last
         core visited, or Entry before the first one. A free ion is thus
         counted where it came from until it reaches the next core, so
         recrossings of a free region are not transitions.
    In permeating_only, every model also has Out and none has SF.

TESTS
-----
Transition matrices    for each lag tau, counts state(t) -> state(t + tau)
                       over all tracked segments (t + tau past the exit
                       frame of an exited segment counts as Exit), row
                       normalised. Exit row = absorbing. Rows with fewer
                       than min_counts counts are flagged.
1 implied timescales   eigenvalues of the matrix restricted to the
                       non-absorbing states with counts, t_i = -tau / ln|l_i|
                       for the n_timescales slowest. Levelling off: the
                       first lag after which a timescale changes by less
                       than plateau_rtol between consecutive lags, for at
                       least two steps, judged only at lags shorter than the
                       timescale (a shorter one is not resolved). The
                       model's lag is that of the slowest timescale.
2 Chapman-Kolmogorov   base lag tau0 = base_lag (10 frames) and, if
                       different, the lag where the timescales level off.
                       Predicted P(tau0)^k vs observed P(k tau0), k = 1..10.
                       Mean absolute difference over all start / end states
                       and k = 2..10 (k = 1 is the same matrix by
                       construction), and for the Exit column alone. The
                       observed values get an SD from n_boot bootstrap
                       resamples of the tracked segments; the share of
                       points within 2 SD is the verdict to read, since a
                       Markovian model still has a nonzero difference.
3 exit times           M2, M4 and M5 at tau0. M2: mean first-passage time G -> Exit,
                       m = tau0 (I - Q)^-1 1, and the predicted first-passage
                       distribution f_k = (Q^(k-1) r)_G. Compared with the
                       observed end_1 -> end_2 transit times of this run.
                       The same prediction starting from the observed
                       state at end_1 + 1 of each event (every model) is
                       also reported, with the KS gap (largest distance
                       between observed and predicted cumulative curves),
                       and the first passage G -> Exit among paths that do
                       not return to the SF first: end_1 is the LAST frame
                       in the SF, so the observed transits are of that kind
                       and are shorter than the MFPT when ions go back up.
Dwell times            runs of consecutive frames in a state, per model.
                       Complete runs (a transition at both ends) are plotted
                       as a log-scale histogram and survival function with a
                       geometric (exponential) reference of the same mean.
                       Flicker check: % of all runs that last 1-2 frames.

OUTPUTS   <results_dir>/markov_check/RUN<run>/
-----------------------------------------------
    summary.txt                          every printed number
    implied_timescales_M{1..5}.png
    ck_test_M{1..5}_lag<tau0>.png
    exit_times_M{2,4,5}_lag<tau0>.png
    dwell_times_M{1..5}.png
    matrices/transition_M<m>_lag<tau>.csv  row-normalised, plus row counts
                                           and a low_counts flag
    matrices/counts_M<m>_lag<tau>.csv      raw counts
    frame_data_cache.npz                 the trajectory pass (reused unless
                                         force_rerun or settings change)

USAGE - from main.py (after ch1, ch2 and analyzer exist)
--------------------------------------------------------
    from analysis.markov_check import run_markov_check
    run_markov_check(
        universe            = u,
        permeation_events   = analyzer.permeation_events,
        channel1            = ch1,
        channel2            = ch2,
        glu_residues        = glu_residues,
        asn_residues        = asn_residues,
        results_dir         = results_dir,
        run                 = run_number,
        channel_type        = args.channel_type,
        frame_duration      = frame_duration,
        start_frame         = start_frame,
        end_frame           = end_frame,
        ion_region_tracking = analyzer.ion_region_tracking,
    )

CLI
---
    python3 -m analysis.markov_check --channel_type G12 --run 3
    python3 -m analysis.markov_check --channel_type G12 --run 3 --permeating_only
reads analyzer_state.pkl (written by main.py) from the results dir for the
permeation events. The paths and residues of G12 runs 1-20 are those of
main.py; any of them can be overridden (--top, --traj, --results_dir,
--upper1, ..., --asp_resid).
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm optional
    def tqdm(x, **kwargs):
        return x

ION_SELECTION = "resname K+ K"
GLU_POLAR = {"GLU": ("OE1", "OE2"), "GLH": ("OE1", "OE2"), "GLN": ("OE1", "NE2")}
OUT_FOLDER = "markov_check"
CACHE_FILE = "frame_data_cache.npz"
CACHE_VERSION = 2                                   # 2: + d_glu

DEFAULT_LAGS = (1, 2, 5, 10, 20, 40, 80, 160, 320)
LOAD_LABELS = ["<=2", "3", ">=4"]

# region codes of the frame data
OUTSIDE, SF_REGION, CAVITY_REGION = 0, 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# Output helper: print and keep every line for summary.txt
# ─────────────────────────────────────────────────────────────────────────────

class _Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def _inside_cylinder(channel, positions):
    """Vectorised Channel.is_within_cylinder (same formulas, same <= tests)."""
    rel = positions - channel.channel_center
    axial = rel @ channel.channel_axis
    radial = np.linalg.norm(rel - axial[:, None] * channel.channel_axis, axis=1)
    return (radial <= channel.radius) & (np.abs(axial) <= channel.channel_length / 2)


def _ring_atoms(u, residues, names, what):
    """One atom per residue (first match of `names`), in residue order."""
    idx = []
    for resid in residues:
        sel = u.select_atoms(f"resid {resid} and name {' '.join(names)}")
        if len(sel) == 0:
            raise ValueError(f"{what}: resid {resid} has no atom named {'/'.join(names)}")
        idx.append(sel[0].index)
    return u.atoms[idx]


def collect_frame_data(u, channel1, channel2, glu_residues, asn_residues, asp_resid,
                       start_frame, end_frame):
    """
    One pass over the trajectory. Returns a dict of arrays:
        ion_resids  (n_ions,)
        frames      (n_frames,)
        region      (n_frames, n_ions) int8   0 outside, 1 SF, 2 cavity
        z           (n_frames, n_ions) float32
        d_asp       (n_frames, n_ions) float32  min distance to D173 OD1/OD2
        d_glu       (n_frames, n_ions) float32  min distance to the side-chain
                    O/N of the Glu ring (GLU/GLH OE1 OE2, GLN OE1 NE2)
        z_glu, z_low, z_sf  (n_frames,)  z_sf = z of the ch1 centre
    """
    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        raise ValueError(f"no ions match '{ION_SELECTION}'")
    glu_cd = _ring_atoms(u, glu_residues, ["CD"], "glu_residues")
    low_cg = _ring_atoms(u, asn_residues, ["CG"], "asn_residues")
    asp_o = u.select_atoms(f"resid {asp_resid} and name OD1 OD2")
    if len(asp_o) == 0:
        raise ValueError(f"asp_resid {asp_resid} has no OD1/OD2 atoms")
    glu_polar = []
    for resid in glu_residues:
        res = u.select_atoms(f"resid {resid}")
        names = GLU_POLAR.get(res.residues[0].resname if len(res) else "", ("OE1", "OE2"))
        sel = res.select_atoms(f"name {' '.join(names)}")
        if len(sel) == 0:
            raise ValueError(f"glu_residues: resid {resid} has no {'/'.join(names)} atoms")
        glu_polar.append(sel)
    glu_polar = sum(glu_polar[1:], glu_polar[0])

    frames = np.arange(start_frame, end_frame + 1)
    n_f, n_i = len(frames), len(ions)
    region = np.zeros((n_f, n_i), dtype=np.int8)
    z = np.zeros((n_f, n_i), dtype=np.float32)
    d_asp = np.zeros((n_f, n_i), dtype=np.float32)
    d_glu = np.zeros((n_f, n_i), dtype=np.float32)
    z_glu = np.zeros(n_f)
    z_low = np.zeros(n_f)
    z_sf = np.zeros(n_f)

    for k, ts in enumerate(tqdm(u.trajectory[start_frame:end_frame + 1], total=n_f,
                                desc="markov_check frames", unit="frame")):
        channel1.compute_geometry(1)
        channel2.compute_geometry(2)
        pos = ions.positions
        in1 = _inside_cylinder(channel1, pos)
        in2 = _inside_cylinder(channel2, pos) & ~in1
        region[k, in1] = SF_REGION
        region[k, in2] = CAVITY_REGION
        z[k] = pos[:, 2]
        d_asp[k] = distance_array(pos, asp_o.positions, box=ts.dimensions).min(axis=1)
        d_glu[k] = distance_array(pos, glu_polar.positions, box=ts.dimensions).min(axis=1)
        z_glu[k] = glu_cd.positions[:, 2].mean()
        z_low[k] = low_cg.positions[:, 2].mean()
        z_sf[k] = channel1.channel_center[2]

    return dict(ion_resids=ions.resids.astype(int), frames=frames, region=region, z=z,
                d_asp=d_asp, d_glu=d_glu, z_glu=z_glu, z_low=z_low, z_sf=z_sf)


def _cache_key(glu_residues, asn_residues, asp_resid, start_frame, end_frame, channel1, channel2):
    return json.dumps(dict(version=CACHE_VERSION, glu=list(map(int, glu_residues)),
                           asn=list(map(int, asn_residues)), asp=int(asp_resid),
                           start=int(start_frame), end=int(end_frame),
                           ch1=[list(map(int, channel1.upper_gate_residues)),
                                list(map(int, channel1.lower_gate_residues)), channel1.radius],
                           ch2=[list(map(int, channel2.upper_gate_residues)),
                                list(map(int, channel2.lower_gate_residues)), channel2.radius]),
                      sort_keys=True)


def load_or_collect(u, channel1, channel2, glu_residues, asn_residues, asp_resid,
                    start_frame, end_frame, out_dir, force_rerun=False):
    cache = Path(out_dir) / CACHE_FILE
    key = _cache_key(glu_residues, asn_residues, asp_resid, start_frame, end_frame,
                     channel1, channel2)
    if cache.exists() and not force_rerun:
        with np.load(cache) as f:
            if str(f["key"]) == key:
                print(f"markov_check: using cached frame data {cache}")
                return {k: f[k] for k in f.files if k != "key"}
        print("markov_check: cached frame data has other settings, recomputing")
    data = collect_frame_data(u, channel1, channel2, glu_residues, asn_residues, asp_resid,
                              start_frame, end_frame)
    np.savez_compressed(cache, key=key, **data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Segments and state assignment
# ─────────────────────────────────────────────────────────────────────────────

def _end1(ev):
    return int(ev["end_1"] if "end_1" in ev else ev["R1_end"])


def _end2(ev):
    return int(ev["end_2"] if "end_2" in ev else ev["R2_end"])


def build_segments(data, permeation_events):
    """
    Tracked segments: runs of consecutive frames in region 1 or 2, per ion.
    Returns (segments, info). A segment is a dict with ion (column index),
    resid, i0, i1 (frame indices into data['frames'], inclusive) and exited.
    """
    frames = data["frames"]
    f0 = int(frames[0])
    col_of = {int(r): c for c, r in enumerate(data["ion_resids"])}
    end2_by_ion = {}
    for ev in permeation_events:
        end2_by_ion.setdefault(int(ev["ion_id"]), []).append(_end2(ev) - f0)

    segments = []
    info = dict(end2_found=0, end2_cut=0, end2_missing=[], unknown_ions=set())
    for ion_id in end2_by_ion:
        if ion_id not in col_of:
            info["unknown_ions"].add(ion_id)

    tracked = np.isin(data["region"], (SF_REGION, CAVITY_REGION))
    for c in np.flatnonzero(tracked.any(axis=0)):
        resid = int(data["ion_resids"][c])
        col = tracked[:, c].astype(np.int8)
        edges = np.diff(np.concatenate(([0], col, [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        end2s = sorted(e for e in end2_by_ion.get(resid, []) if 0 <= e < len(frames))
        used = set()
        for i0, i1 in zip(starts, ends):
            i0, i1 = int(i0), int(i1)
            while True:
                inside = [e for e in end2s if i0 <= e <= i1 and e not in used]
                if not inside:
                    segments.append(dict(ion=int(c), resid=resid, i0=i0, i1=i1, exited=False))
                    break
                e = inside[0]
                used.add(e)
                if e == i1:
                    info["end2_found"] += 1
                else:
                    info["end2_cut"] += 1
                segments.append(dict(ion=int(c), resid=resid, i0=i0, i1=e, exited=True))
                if e == i1:
                    break
                i0 = e + 1
        for e in end2s:
            if e not in used:
                info["end2_missing"].append((resid, e + f0))
    return segments, info


def build_permeating_segments(data, permeation_events):
    """
    permeating_only: one segment per permeation event, end_1 + 1 .. end_2
    (the ion has left the SF for the last time -> its last cavity frame),
    ending in Exit. Frames of the window outside region 2 (an excursion out
    of the cavity cylinder and back) are kept: they are the Out state.
    Returns (segments, info).
    """
    f0 = int(data["frames"][0])
    n_f = len(data["frames"])
    col_of = {int(r): c for c, r in enumerate(data["ion_resids"])}
    segments = []
    info = dict(events=0, window_frames=0, non_cavity_frames=0, sf_frames=0,
                events_with_out=0, no_cavity=[], unknown_ions=set(), outside_traj=0)
    for ev in permeation_events:
        ion_id = int(ev["ion_id"])
        c = col_of.get(ion_id)
        if c is None:
            info["unknown_ions"].add(ion_id)
            continue
        i0, i1 = _end1(ev) + 1 - f0, _end2(ev) - f0
        if i0 < 0 or i1 >= n_f:
            info["outside_traj"] += 1
            continue
        info["events"] += 1
        reg = data["region"][max(i0, 0):i1 + 1, c]
        if i1 < i0 or not np.any(reg == CAVITY_REGION):
            info["no_cavity"].append((ion_id, _end2(ev)))
            continue
        info["window_frames"] += len(reg)
        n_out = int((reg != CAVITY_REGION).sum())
        info["non_cavity_frames"] += n_out
        info["sf_frames"] += int((reg == SF_REGION).sum())
        info["events_with_out"] += int(n_out > 0)
        segments.append(dict(ion=int(c), resid=ion_id, i0=i0, i1=i1, exited=True))
    return segments, info


class StateModel:
    def __init__(self, name, labels):
        self.name = name
        self.labels = list(labels) + ["Exit"]
        self.n = len(self.labels)
        self.exit = self.n - 1


def make_models(with_sf=True, with_out=False):
    """
    M1..M5. with_sf: the SF is a state (all-ions mode). with_out: the Out
    state, frames of a permeation window outside the cavity (permeating_only).
    """
    sf = ["SF"] if with_sf else []
    out = ["Out"] if with_out else []
    m2_core = ["G", "A_asp", "A_other"]
    return [
        StateModel("M1", sf + ["G", "A"] + out),
        StateModel("M2", sf + m2_core + out),
        StateModel("M3", [f"{x}|{l}" for x in sf + m2_core for l in LOAD_LABELS] + out),
        StateModel("M4", sf + ["Glu", "Asp", "Free_up", "Free_low"] + out),
        StateModel("M5", sf + ["Entry", "Glu", "Asp"] + out),
    ]


def assign_states(data, segments, models, asp_enter, asp_leave, glu_enter, glu_leave,
                  z_buffer):
    """
    Per segment, the state index of every frame under each model (Exit not
    included: it is implied after an exited segment). Returns
    ({model name: [int array per segment]}, stats).

    Frame classes: region 1 -> SF, region 2 -> cavity, anything else -> Out
    (only inside a permeation window; in the all-ions mode segments hold
    regions 1 and 2 only). In permeating_only there is no SF state and a
    region-1 frame would also count as Out (none are expected).

    M1/M2/M3  G / A split at z_mid; A_asp with the asp_enter / asp_leave
              hysteresis, only while in A.
    M4        Glu: within glu_enter of a Glu-ring O/N, kept until glu_leave;
              Asp: the same for D173 with asp_enter / asp_leave; if both are
              entered on the same frame the closer one wins. Otherwise free,
              Free_up / Free_low by the side of z_mid, switched only once the
              ion is z_buffer beyond z_mid on the other side.
    M5        core sets: SF, Glu, Asp (as in M4) and Out are cores; every
              other frame carries the last core visited, or Entry before the
              first one of the segment.
    """
    region, z, d_asp, d_glu = data["region"], data["z"], data["d_asp"], data["d_glu"]
    z_mid = (data["z_glu"] + data["z_low"]) / 2
    sf_sign = np.sign(data["z_sf"] - z_mid)
    low_side = (data["z_low"] - z_mid) * sf_sign       # z_low on the "SF side" axis (< 0)
    n_cav = (region == CAVITY_REGION).sum(axis=1)
    index = {m.name: {l: i for i, l in enumerate(m.labels)} for m in models}
    with_sf = "SF" in index["M1"]

    out = {m.name: [] for m in models}
    stats = dict(n_sf_below=0, out_frames=0, out_below_low=0)
    for seg in segments:
        sl = slice(seg["i0"], seg["i1"] + 1)
        c = seg["ion"]
        reg = region[sl, c]
        sv = (z[sl, c] - z_mid[sl]) * sf_sign[sl]      # > 0: SF side of z_mid
        da, dg = d_asp[sl, c], d_glu[sl, c]
        stats["n_sf_below"] += int((sf_sign[sl] < 0).sum())
        others = n_cav[sl] - (reg == CAVITY_REGION)
        load = np.clip(others - 2, 0, 2)                # <=2 -> 0, 3 -> 1, >=4 -> 2

        labels = {m.name: [] for m in models}
        in_asp = False
        bound = free_side = last_core = None
        for t in range(len(reg)):
            if reg[t] == CAVITY_REGION:
                base = None
            elif reg[t] == SF_REGION and with_sf:
                base = "SF"
            else:
                base = "Out"
                stats["out_frames"] += 1
                stats["out_below_low"] += int(sv[t] < low_side[sl][t])

            # M1, M2, M3
            if base:
                m1 = m2 = base
                in_asp = False
            elif sv[t] > 0:
                m1 = m2 = "G"
                in_asp = False
            else:
                m1 = "A"
                in_asp = da[t] <= asp_leave if in_asp else da[t] < asp_enter
                m2 = "A_asp" if in_asp else "A_other"
            m3 = m2 if m2 == "Out" else f"{m2}|{LOAD_LABELS[load[t]]}"

            # M4
            if base:
                m4 = base
                bound = free_side = None
            else:
                if not ((bound == "Glu" and dg[t] <= glu_leave)
                        or (bound == "Asp" and da[t] <= asp_leave)):
                    near_g, near_a = dg[t] < glu_enter, da[t] < asp_enter
                    if near_g and near_a:
                        bound = "Glu" if dg[t] <= da[t] else "Asp"
                    else:
                        bound = "Glu" if near_g else ("Asp" if near_a else None)
                if bound:
                    m4 = bound
                    free_side = None
                else:
                    if (free_side is None or (free_side == "up" and sv[t] < -z_buffer)
                            or (free_side == "low" and sv[t] > z_buffer)):
                        free_side = "up" if sv[t] > 0 else "low"
                    m4 = "Free_up" if free_side == "up" else "Free_low"

            # M5
            core = base or bound
            if core:
                last_core = core
            m5 = last_core or "Entry"

            for name, lab in (("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4), ("M5", m5)):
                labels[name].append(index[name][lab])
        for name in out:
            out[name].append(np.asarray(labels[name], dtype=np.int16))
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrices
# ─────────────────────────────────────────────────────────────────────────────

def count_matrix(seqs, exited, n_states, lag):
    """Counts state(t) -> state(t + lag); past the end of an exited segment = Exit."""
    C = np.zeros((n_states, n_states), dtype=np.int64)
    ex = n_states - 1
    for s, e in zip(seqs, exited):
        L = len(s)
        if L > lag:
            np.add.at(C, (s[:-lag], s[lag:]), 1)
        if e:
            src = s[max(0, L - lag):]
            np.add.at(C, (src, np.full(len(src), ex)), 1)
    return C


def segment_count_matrices(seqs, exited, n_states, lag):
    """count_matrix per segment: (n_segments, n_states, n_states)."""
    out = np.zeros((len(seqs), n_states, n_states), dtype=np.int32)
    for k, (s, e) in enumerate(zip(seqs, exited)):
        out[k] = count_matrix([s], [e], n_states, lag)
    return out


def bootstrap_sd(per_segment, n_boot, rng):
    """SD of the row-normalised matrix over bootstrap resamples of segments."""
    n_seg = per_segment.shape[0]
    flat = per_segment.reshape(n_seg, -1).astype(float)
    reps = []
    for _ in range(n_boot):
        w = rng.multinomial(n_seg, np.full(n_seg, 1.0 / n_seg))
        C = (w @ flat).reshape(per_segment.shape[1:])
        rows = C.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps.append(np.where(rows > 0, C / rows, np.nan))
    with warnings.catch_warnings():                    # entries NaN in every resample
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(np.array(reps), axis=0)


def transition_matrix(C):
    """Row-normalised; rows without counts are 0; the Exit row is absorbing."""
    T = np.zeros(C.shape, dtype=float)
    rows = C.sum(axis=1)
    ok = rows > 0
    T[ok] = C[ok] / rows[ok, None]
    T[-1] = 0.0
    T[-1, -1] = 1.0
    return T, rows


def _active(rows):
    """Non-absorbing states with counts."""
    return np.flatnonzero(rows[:-1] > 0)


def implied_timescales(T, rows, lag, n_its):
    act = _active(rows)
    if len(act) == 0:
        return np.full(n_its, np.nan)
    lam = np.sort(np.abs(np.linalg.eigvals(T[np.ix_(act, act)])))[::-1]
    out = np.full(n_its, np.nan)
    for i, l in enumerate(lam[:n_its]):
        if l >= 1.0:
            out[i] = np.inf
        elif l > 0:
            out[i] = -lag / np.log(l)
        else:
            out[i] = 0.0
    return out


def find_plateau(lags, ts, rtol):
    """
    First lag after which ts changes by < rtol between consecutive lags, for
    at least two steps. Only lags where the timescale is longer than the lag
    are judged (a shorter one is not resolved at that lag). Returns
    (plateau lag or None, last lag judged or None).
    """
    lags = np.asarray(lags)
    ts = np.asarray(ts, dtype=float)
    ok = np.isfinite(ts) & (ts > lags)
    n_ok = len(ok) if ok.all() else int(np.argmin(ok))     # leading resolvable lags
    last = int(lags[n_ok - 1]) if n_ok else None
    for k in range(n_ok - 2):
        tail = ts[k:n_ok]
        if np.all(np.abs(np.diff(tail)) / tail[:-1] < rtol):
            return int(lags[k]), last
    return None, last


def save_matrix_csvs(model, lag, C, T, rows, min_counts, mat_dir):
    pd.DataFrame(C, index=model.labels, columns=model.labels).to_csv(
        mat_dir / f"counts_{model.name}_lag{lag}.csv", index_label="from")
    df = pd.DataFrame(T, index=model.labels, columns=model.labels)
    df["row_counts"] = rows
    df["low_counts"] = [(r < min_counts) if i < model.exit else False
                        for i, r in enumerate(rows)]
    df.to_csv(mat_dir / f"transition_{model.name}_lag{lag}.csv", index_label="from",
              float_format="%.6f")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_implied_timescales(model, lags, its, dt, plateau, out_file):
    fig, ax = plt.subplots(figsize=(7, 5))
    lag_ns = np.asarray(lags) * dt
    for i in range(its.shape[1]):
        y = np.where(np.isfinite(its[:, i]), its[:, i] * dt, np.nan)
        ax.plot(lag_ns, y, marker="o", label=f"t{i + 1}")
    ax.plot(lag_ns, lag_ns, color="grey", linestyle=":", label="t = lag")
    if plateau is not None:
        ax.axvline(plateau * dt, color="black", linestyle="--", linewidth=1,
                   label=f"levels off at {plateau} frames")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("implied timescale (ns)")
    ax.set_title(f"{model.name}: implied timescales\n"
                 f"({' / '.join(model.labels[:-1]) if model.n <= 6 else f'{model.n - 1} transient states'})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_ck(model, tau0, ks, pred, obs, obs_sd, dt, out_file):
    """pred/obs/obs_sd: (len(ks), n, n). Full grid for small models, self + Exit otherwise."""
    starts = [i for i in range(model.n - 1) if np.isfinite(obs[0, i]).all()]
    full = model.n <= 6
    t_ns = np.asarray(ks) * tau0 * dt
    if not starts:
        return
    cols = list(range(model.n)) if full else None
    n_cols = model.n if full else 2
    fig, axes = plt.subplots(len(starts), n_cols, figsize=(2.6 * n_cols, 2.1 * len(starts)),
                             squeeze=False, sharex=True, sharey=True)
    for r, i in enumerate(starts):
        targets = cols if full else [i, model.exit]
        for cidx, j in enumerate(targets):
            ax = axes[r, cidx]
            ax.plot(t_ns, pred[:, i, j], color="#1c5cab", label="predicted P(tau0)^k")
            ax.errorbar(t_ns, obs[:, i, j], yerr=2 * obs_sd[:, i, j], linestyle="none",
                        marker="o", markersize=3, color="#d1495b", elinewidth=0.8,
                        label="observed P(k tau0) +/- 2 SD")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{model.labels[i]} -> {model.labels[j]}", fontsize=8)
            if r == len(starts) - 1:
                ax.set_xlabel("time (ns)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{model.name}: Chapman-Kolmogorov test, tau0 = {tau0} frames "
                 f"({tau0 * dt:g} ns)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def plot_exit_times(model_name, tau0, dt, f_g, f_entry, f_cond, observed_ns, out_file):
    """Left: fraction of exits per log-spaced time bin. Right: cumulative."""
    width = tau0 * dt
    curves = [(f, color, label) for f, color, label in (
        (f_g, "#1c5cab", "predicted, start in G"),
        (f_entry, "#d1495b", "predicted, observed entry states"),
        (f_cond, "#2a9d4b", "predicted, start in G, no return to SF")) if f is not None]
    hi = width
    if len(observed_ns):
        hi = max(hi, np.max(observed_ns))
    for f, _, _ in curves:                              # 99.9% of the predicted mass
        hi = max(hi, (np.searchsorted(np.cumsum(f), 0.999) + 1) * width)
    lo = min(dt, np.min(observed_ns[observed_ns > 0])) if np.any(observed_ns > 0) else dt
    bins = np.geomspace(lo * 0.999, hi * 1.001, 26)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if len(observed_ns):
        h, _ = np.histogram(observed_ns, bins=bins)
        ax.bar(bins[:-1], h / len(observed_ns), width=np.diff(bins), align="edge",
               color="#bbbbbb", edgecolor="white",
               label=f"observed end_1 -> end_2 (n = {len(observed_ns)})")
        x = np.sort(observed_ns)
        ax2.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="black",
                 label="observed")
    for f, color, label in curves:
        t = np.arange(1, len(f) + 1) * width
        mass, _ = np.histogram(t, bins=bins, weights=f)
        ax.step(bins[:-1], mass, where="post", color=color, linewidth=1.5, label=label)
        ax2.step(t, np.cumsum(f), where="post", color=color, linewidth=1.5, label=label)
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_xlim(bins[0], bins[-1])
        a.set_xlabel("time to Exit (ns)")
    ax.set_ylabel("fraction of exits per bin")
    ax2.set_ylabel("fraction exited by t")
    ax2.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{model_name} exit times, tau0 = {tau0} frames ({width:g} ns); "
                 f"predictions are on a {width:g} ns grid")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_dwell(model, dwell, dt, out_file):
    states = [s for s in range(model.n - 1) if len(dwell[s]["complete"])]
    if not states:
        return
    fig, axes = plt.subplots(2, len(states), figsize=(3.2 * len(states), 6), squeeze=False)
    for c, s in enumerate(states):
        d = np.asarray(dwell[s]["complete"])
        mean = d.mean()
        p = 1.0 / mean
        lengths = np.arange(1, d.max() + 1)
        counts = np.bincount(d, minlength=d.max() + 1)[1:]
        ax = axes[0, c]
        width = max(1, int(np.ceil(d.max() / 30)))       # frames per bin
        edges = np.arange(1, d.max() + width + 1, width)
        hist, _ = np.histogram(d, bins=edges)
        ax.bar(edges[:-1] * dt, hist, width=width * dt, align="edge", color="#86b6ef",
               edgecolor="white", linewidth=0.3)
        ref = len(d) * ((1 - p) ** (edges[:-1] - 1) - (1 - p) ** (edges[1:] - 1))
        ax.plot((edges[:-1] + width / 2) * dt, ref, color="grey", linestyle="--",
                label="exponential, same mean")
        ax.set_yscale("log")
        ax.set_ylim(0.5, hist.max() * 2)
        ax.set_title(f"{model.labels[s]}  (n = {len(d)})", fontsize=8)
        ax.set_xlabel("dwell (ns)", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("count")
            ax.legend(fontsize=6)
        ax = axes[1, c]
        surv = 1.0 - np.concatenate(([0], np.cumsum(counts)[:-1])) / len(d)
        ax.step(lengths * dt, surv, where="post")
        ax.plot(lengths * dt, (1 - p) ** (lengths - 1), color="grey", linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(0.5 / len(d), 1.5)
        ax.set_xlabel("dwell (ns)", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("P(dwell >= t)")
    fig.suptitle(f"{model.name}: dwell times (complete visits; straight line = exponential)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def dwell_times(seqs, exited, n_transient):
    """Per state: all run lengths and complete ones (a transition at both ends)."""
    out = {s: {"all": [], "complete": []} for s in range(n_transient)}
    for s, e in zip(seqs, exited):
        change = np.flatnonzero(np.diff(s)) + 1
        bounds = np.concatenate(([0], change, [len(s)]))
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            state, length = int(s[a]), int(b - a)
            out[state]["all"].append(length)
            if a > 0 and (b < len(s) or e):
                out[state]["complete"].append(length)
    return out


def first_passage(T, rows, start, tau0, exit_idx, max_steps=100000, tol=1e-6):
    """
    MFPT (frames) to Exit and the first-passage distribution f_k from `start`
    (a state index or a probability vector over all states), on the active
    transient states.
    """
    act = _active(rows)
    Q = T[np.ix_(act, act)]
    r = T[act, exit_idx]
    if np.isscalar(start):
        if start not in act:
            return np.nan, None
        p = (act == start).astype(float)
    else:
        p = np.asarray(start, dtype=float)[act]
        if p.sum() == 0:
            return np.nan, None
        p = p / p.sum()
    try:
        m = np.linalg.solve(np.eye(len(act)) - Q, np.ones(len(act))) * tau0
    except np.linalg.LinAlgError:
        return np.inf, None
    mfpt = float(p @ m)
    f = []
    surv = 1.0
    while surv > tol and len(f) < max_steps:
        f.append(float(p @ r))
        p = p @ Q
        surv = p.sum()
    return mfpt, np.asarray(f)


def first_passage_no_return(T, rows, start, avoid, exit_idx, max_steps=100000, tol=1e-9):
    """
    First passage to Exit from `start` among the paths that never visit
    `avoid` (the SF) first: the counterpart of the observed end_1 -> end_2
    transit, where end_1 is the LAST frame in the SF. Returns
    (P(Exit before avoid), mean time in frames of tau0 steps, normalised f_k).
    """
    act = [i for i in _active(rows) if i != avoid]
    if start not in act:
        return np.nan, np.nan, None
    Q = T[np.ix_(act, act)]
    r = T[act, exit_idx]
    p = (np.asarray(act) == start).astype(float)
    f = []
    while p.sum() > tol and len(f) < max_steps:
        f.append(float(p @ r))
        p = p @ Q
    f = np.asarray(f)
    reach = f.sum()
    if reach == 0:
        return 0.0, np.nan, None
    f = f / reach
    return float(reach), float((np.arange(1, len(f) + 1) * f).sum()), f


def _median_from_pmf(f, tau0):
    cdf = np.cumsum(f)
    k = int(np.searchsorted(cdf, 0.5 * cdf[-1]))
    return (k + 1) * tau0


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_markov_check(universe, permeation_events, channel1, channel2,
                     glu_residues, asn_residues, results_dir, run,
                     channel_type="G12", frame_duration=0.075,
                     start_frame=0, end_frame=None, asp_resid=None,
                     lags=DEFAULT_LAGS, base_lag=10, ck_steps=10,
                     asp_enter=3.5, asp_leave=4.5, glu_enter=3.5, glu_leave=4.5,
                     z_buffer=1.0, min_counts=50,
                     n_timescales=3, plateau_rtol=0.1, n_boot=200, seed=0,
                     permeating_only=False,
                     ion_region_tracking=None, force_rerun=False):
    """Run the Markov checks for one run; see the module docstring."""
    u = universe
    if end_frame is None:
        end_frame = len(u.trajectory) - 1
    cache_dir = Path(results_dir) / OUT_FOLDER / f"RUN{run}"
    out_dir = cache_dir.parent / f"RUN{run}_permeating" if permeating_only else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = out_dir / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    dt = float(frame_duration)
    lags = [int(l) for l in lags]
    rep = _Report()

    if asp_resid is None:
        from analysis.converter import from_pdb
        asp_resid = from_pdb(173, "G1", channel_type)
        if asp_resid is None:
            raise ValueError(f"no D173.G1 for channel_type {channel_type}; pass asp_resid")
    asp_res = u.select_atoms(f"resid {asp_resid}").residues
    asp_name = asp_res[0].resname if len(asp_res) else "?"

    rep("=" * 78)
    rep(f"MARKOV CHECK  {channel_type} RUN{run}"
        + ("  (permeating ions only, end_1 + 1 -> end_2)" if permeating_only else ""))
    rep("=" * 78)
    rep(f"frames {start_frame}-{end_frame}, frame time {dt:g} ns")
    rep(f"glu ring {list(glu_residues)} (CD), lower ring {list(asn_residues)} (CG)")
    rep(f"D173.G1 = resid {asp_resid} ({asp_name}); A_asp / Asp enter < {asp_enter} A, "
        f"leave > {asp_leave} A")
    rep(f"M4/M5 Glu: side-chain O/N of the glu ring, enter < {glu_enter} A, leave > "
        f"{glu_leave} A; Free_up / Free_low switch {z_buffer} A past z_mid")
    if asp_name not in ("ASP", "ASH"):
        rep(f"WARNING: resid {asp_resid} is {asp_name}, not ASP")
    if int(asp_resid) not in [int(r) for r in asn_residues]:
        rep(f"WARNING: resid {asp_resid} is not one of asn_residues")
    rep(f"lags {lags} frames, base lag {base_lag}, CK k = 1..{ck_steps}, "
        f"rows with < {min_counts} counts flagged")
    rep(f"output {out_dir}")

    data = load_or_collect(u, channel1, channel2, glu_residues, asn_residues, asp_resid,
                           start_frame, end_frame, cache_dir, force_rerun)

    # ── consistency with the analyzer's own region tracking ──
    if ion_region_tracking:
        col_of = {int(r): c for c, r in enumerate(data["ion_resids"])}
        f0 = int(data["frames"][0])
        agree = total = 0
        for ion_id, regs in ion_region_tracking.items():
            c = col_of.get(int(ion_id))
            if c is None:
                continue
            for fr, reg in regs.items():
                k = int(fr) - f0
                if 0 <= k < len(data["frames"]) and int(reg) in (0, 1, 2):
                    total += 1
                    agree += int(data["region"][k, c] == int(reg))
        if total:
            rep(f"region check vs ion_region_tracking (regions 0-2): "
                f"{agree}/{total} frames agree ({100 * agree / total:.2f}%)")

    models = make_models(with_sf=not permeating_only, with_out=permeating_only)
    by_name = {m.name: m for m in models}
    if permeating_only:
        segments, seg_info = build_permeating_segments(data, permeation_events)
    else:
        segments, seg_info = build_segments(data, permeation_events)
    exited = [s["exited"] for s in segments]
    states, st = assign_states(data, segments, models, asp_enter, asp_leave,
                               glu_enter, glu_leave, z_buffer)
    n_sf_below = st["n_sf_below"]
    n_frames_tracked = sum(s["i1"] - s["i0"] + 1 for s in segments)

    rep("")
    rep("TRACKING")
    if permeating_only:
        rep(f"  permeation events: {len(permeation_events)}; used: {seg_info['events']}; "
            f"outside the analysed frames: {seg_info['outside_traj']}")
        rep(f"  window frames (end_1 + 1 .. end_2): {seg_info['window_frames']}; not in the "
            f"cavity: {seg_info['non_cavity_frames']} (of them in the SF: "
            f"{seg_info['sf_frames']}) = state Out, in {seg_info['events_with_out']} "
            f"events; Out frames beyond the lower ring: "
            f"{st['out_below_low']}/{st['out_frames']}")
        if seg_info["no_cavity"]:
            rep(f"    events without cavity frames (ion, end_2): {seg_info['no_cavity'][:10]}")
        rep(f"  tracked segments: {len(segments)} ({n_frames_tracked} ion-frames); "
            f"ending in Exit: {sum(exited)}; censored: {len(segments) - sum(exited)}")
    else:
        rep(f"  K+ in the system: {len(data['ion_resids'])}; "
            f"ever in SF/cavity: {len({s['ion'] for s in segments})}")
        rep(f"  tracked segments: {len(segments)} ({n_frames_tracked} ion-frames); "
            f"ending in Exit: {sum(exited)}; censored: {len(segments) - sum(exited)}")
        rep(f"  permeation events: {len(permeation_events)}; end_2 at a segment end: "
            f"{seg_info['end2_found']}; end_2 inside a segment (cut there): "
            f"{seg_info['end2_cut']}; end_2 outside every segment: "
            f"{len(seg_info['end2_missing'])}")
        if seg_info["end2_missing"]:
            rep(f"    (ion, end_2) not matched: {seg_info['end2_missing'][:10]}")
    if seg_info["unknown_ions"]:
        rep(f"    permeating ions not found among K+: {sorted(seg_info['unknown_ions'])}")
    if n_sf_below:
        rep(f"  WARNING: in {n_sf_below} tracked ion-frames the SF centre was below z_mid")
    for m in models:
        occ = np.bincount(np.concatenate(states[m.name]), minlength=m.n - 1)
        rep(f"  {m.name} occupancy: " + ", ".join(
            f"{m.labels[s]} {100 * occ[s] / max(occ.sum(), 1):.1f}%" for s in range(m.n - 1)))

    # ── transition matrices at all lags ──
    rep("")
    rep("TRANSITION MATRICES (rows: counts; * = fewer than "
        f"{min_counts} counts)")
    cache_T = {}

    def get_T(model, lag):
        if (model.name, lag) not in cache_T:
            C = count_matrix(states[model.name], exited, model.n, lag)
            T, rows = transition_matrix(C)
            cache_T[(model.name, lag)] = (C, T, rows)
        return cache_T[(model.name, lag)]

    for m in models:
        rep(f"  {m.name}")
        for lag in lags:
            C, T, rows = get_T(m, lag)
            save_matrix_csvs(m, lag, C, T, rows, min_counts, mat_dir)
            low = [m.labels[i] for i in range(m.n - 1) if rows[i] < min_counts]
            rep(f"    lag {lag:>3}: " + ", ".join(
                f"{m.labels[i]} {rows[i]}{'*' if rows[i] < min_counts else ''}"
                for i in range(m.n - 1)))
            if low:
                rep(f"             low-count rows: {', '.join(low)}")
        if m.n <= 6:
            _, T, _ = get_T(m, base_lag)
            rep(f"    P(lag {base_lag}):")
            rep("      " + "".join(f"{l:>10}" for l in m.labels))
            for i in range(m.n):
                rep("      " + "".join(f"{T[i, j]:>10.4f}" for j in range(m.n))
                    + f"   <- {m.labels[i]}")

    # ── test 1: implied timescales ──
    rep("")
    rep(f"TEST 1: IMPLIED TIMESCALES (ns; levelling off = the timescale changes "
        f"< {100 * plateau_rtol:g}% between consecutive lags for >= 2 steps, judged only "
        f"at lags shorter than the timescale; the model's lag is that of t1)")
    plateaus = {}
    for m in models:
        its = np.array([implied_timescales(get_T(m, lag)[1], get_T(m, lag)[2], lag,
                                           n_timescales) for lag in lags])
        plateau, _ = find_plateau(lags, its[:, 0], plateau_rtol)
        plateaus[m.name] = plateau
        rep(f"  {m.name}")
        for k, lag in enumerate(lags):
            rep(f"    lag {lag:>3} ({lag * dt:6.3f} ns): " + "  ".join(
                f"t{i + 1} = {its[k, i] * dt:9.3f}" for i in range(n_timescales)))
        for i in range(n_timescales):
            p_i, last = find_plateau(lags, its[:, i], plateau_rtol)
            judged = (f"judged up to lag {last}" if last is not None
                      else "shorter than every lag, not resolved")
            rep(f"    t{i + 1}: " + (f"levels off from lag {p_i} frames ({p_i * dt:g} ns), {judged}"
                                     if p_i is not None else f"never levels off ({judged})"))
        plot_implied_timescales(m, lags, its, dt, plateau,
                                out_dir / f"implied_timescales_{m.name}.png")

    # ── test 2: Chapman-Kolmogorov ──
    rep("")
    rep(f"TEST 2: CHAPMAN-KOLMOGOROV (k = 1..{ck_steps}; mean |predicted - observed| over "
        f"k = 2..{ck_steps}, start states with >= {min_counts} counts; SD of observed from "
        f"{n_boot} bootstrap resamples of the tracked segments)")
    ks = np.arange(1, ck_steps + 1)
    rng = np.random.default_rng(seed)
    base_lags_by_model = {}
    for m in models:
        bl = [int(base_lag)]
        if plateaus[m.name] is not None and plateaus[m.name] != base_lag:
            bl.append(plateaus[m.name])
        base_lags_by_model[m.name] = bl
        for tau0 in bl:
            _, T0, rows0 = get_T(m, tau0)
            pred = np.empty((len(ks), m.n, m.n))
            obs = np.full((len(ks), m.n, m.n), np.nan)
            obs_sd = np.full((len(ks), m.n, m.n), np.nan)
            Pk = np.eye(m.n)
            for a, k in enumerate(ks):
                Pk = Pk @ T0
                pred[a] = Pk
                _, Tk, rowsk = get_T(m, int(k * tau0))
                sd = bootstrap_sd(segment_count_matrices(states[m.name], exited, m.n,
                                                         int(k * tau0)), n_boot, rng)
                for i in range(m.n - 1):
                    if rowsk[i] >= min_counts and rows0[i] >= min_counts:
                        obs[a, i] = Tk[i]
                        obs_sd[a, i] = sd[i]
            diff = np.abs(pred - obs)[1:, :-1, :]
            mad = np.nanmean(diff) if np.isfinite(diff).any() else np.nan
            mad_exit = np.nanmean(diff[:, :, -1]) if np.isfinite(diff[:, :, -1]).any() else np.nan
            sd2 = obs_sd[1:, :-1, :]
            judged = np.isfinite(diff) & np.isfinite(sd2)
            within = int(np.sum(diff[judged] <= 2 * sd2[judged] + 1e-12))
            judged_e = judged[:, :, -1]
            within_e = int(np.sum(diff[:, :, -1][judged_e] <= 2 * sd2[:, :, -1][judged_e] + 1e-12))
            rep(f"  {m.name} tau0 = {tau0} frames ({tau0 * dt:g} ns): mean abs diff {mad:.4f}, "
                f"Exit column {mad_exit:.4f}; within 2 SD: {within}/{int(judged.sum())} "
                f"({100 * within / max(judged.sum(), 1):.0f}%), Exit column "
                f"{within_e}/{int(judged_e.sum())}")
            for i in range(m.n - 1):
                if np.isfinite(obs[0, i]).all():
                    d_i = np.nanmean(diff[:, i, :])
                    d_ie = np.nanmean(diff[:, i, -1])
                    rep(f"      from {m.labels[i]:<14} mean abs diff {d_i:.4f}, "
                        f"P(Exit) {d_ie:.4f}  "
                        f"(P(Exit) at k={ck_steps}: predicted {pred[-1, i, -1]:.3f}, "
                        f"observed {obs[-1, i, -1]:.3f} +/- {obs_sd[-1, i, -1]:.3f})")
            plot_ck(m, tau0, ks, pred, obs, obs_sd, dt,
                    out_dir / f"ck_test_{m.name}_lag{tau0}.png")

    # ── test 3: exit times, M2 / M4 / M5 ──
    rep("")
    rep("TEST 3: EXIT TIMES (M2, M4, M5; KS = largest gap between the observed and predicted "
        "cumulative distributions)")
    f0 = int(data["frames"][0])
    observed = np.array([(_end2(ev) - _end1(ev)) * dt for ev in permeation_events])
    n_frame0 = sum(1 for ev in permeation_events if _end1(ev) == f0)
    if len(observed):
        rep(f"  observed end_1 -> end_2: n = {len(observed)}, mean {observed.mean():.3f} ns, "
            f"median {np.median(observed):.3f} ns ({n_frame0} events with end_1 = "
            f"first frame)")
    else:
        rep("  no permeation events in this run")
    if permeating_only:
        rep("  note: every window starts after end_1 and ends in Exit, so at lag 1 the predicted "
            "mean from the observed entry states equals the observed mean by construction; "
            "judge the median, the KS gap and longer lags")

    seg_lookup = {}
    for si, s in enumerate(segments):
        seg_lookup.setdefault(s["resid"], []).append(si)

    def ks_gap(f, tau0):
        if f is None or not len(observed):
            return np.nan
        t = np.arange(1, len(f) + 1) * tau0 * dt
        x = np.sort(observed)
        pred = np.concatenate(([0.0], np.cumsum(f)))[np.searchsorted(t, x, side="right")]
        obs_hi = np.arange(1, len(x) + 1) / len(x)
        return float(max(np.max(np.abs(obs_hi - pred)), np.max(np.abs(obs_hi - 1 / len(x) - pred))))

    for m in (by_name["M2"], by_name["M4"], by_name["M5"]):
        entry = np.zeros(m.n)                          # observed state at end_1 + 1
        for ev in permeation_events:
            k = _end1(ev) + 1 - f0
            for si in seg_lookup.get(int(ev["ion_id"]), []):
                sg = segments[si]
                if sg["i0"] <= k <= sg["i1"]:
                    entry[states[m.name][si][k - sg["i0"]]] += 1
                    break
        rep(f"  {m.name}" + (": state at end_1 + 1: " + ", ".join(
            f"{m.labels[i]} {int(entry[i])}" for i in range(m.n - 1) if entry[i])
            if entry.sum() else ""))
        for tau0 in base_lags_by_model[m.name]:
            _, T0, rows0 = get_T(m, tau0)
            mfpt_e, f_e = (first_passage(T0, rows0, entry, tau0, m.exit)
                           if entry.sum() else (np.nan, None))
            f_g = f_c = None
            if "G" in m.labels:
                mfpt_g, f_g = first_passage(T0, rows0, m.labels.index("G"), tau0, m.exit)
                line = f"    tau0 = {tau0:>3}: MFPT G -> Exit {mfpt_g * dt:9.3f} ns"
                if f_g is not None:
                    line += (f" (median {_median_from_pmf(f_g, tau0) * dt:.3f} ns, "
                             f"KS {ks_gap(f_g, tau0):.2f})")
                rep(line)
                if "SF" in m.labels:
                    reach, mean_c, f_c = first_passage_no_return(
                        T0, rows0, m.labels.index("G"), m.labels.index("SF"), m.exit)
                    if f_c is not None:
                        rep(f"                 G -> Exit without returning to the SF (the observed "
                            f"end_1 -> end_2 is of this kind): P = {reach:.3f}, mean "
                            f"{mean_c * tau0 * dt:.3f} ns (median "
                            f"{_median_from_pmf(f_c, tau0) * dt:.3f} ns, KS {ks_gap(f_c, tau0):.2f})")
            if f_e is not None:
                rep(f"    tau0 = {tau0:>3}: from the observed entry states: mean "
                    f"{mfpt_e * dt:9.3f} ns, median {_median_from_pmf(f_e, tau0) * dt:.3f} ns, "
                    f"KS {ks_gap(f_e, tau0):.2f}"
                    + (f"; predicted / observed mean {mfpt_e * dt / observed.mean():.2f}, "
                       f"median {_median_from_pmf(f_e, tau0) * dt / np.median(observed):.2f}"
                       if len(observed) else ""))
            plot_exit_times(m.name, tau0, dt, f_g, f_e, f_c, observed,
                            out_dir / f"exit_times_{m.name}_lag{tau0}.png")

    # ── dwell times ──
    rep("")
    rep("DWELL TIMES (all visits; complete = a transition at both ends)")
    for m in models:
        dw = dwell_times(states[m.name], exited, m.n - 1)
        rep(f"  {m.name}")
        for s in range(m.n - 1):
            a = np.asarray(dw[s]["all"])
            c = np.asarray(dw[s]["complete"])
            if len(a) == 0:
                continue
            short = 100 * np.mean(a <= 2)
            rep(f"    {m.labels[s]:<14} visits {len(a):>6} (complete {len(c):>6}), "
                f"1-2 frames {short:5.1f}%, mean complete "
                f"{(c.mean() * dt if len(c) else np.nan):8.3f} ns")
        plot_dwell(m, dw, dt, out_dir / f"dwell_times_{m.name}.png")

    rep.write(out_dir / "summary.txt")
    print(f"\nmarkov_check: results in {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# main.py, G12 runs 1-20
G12_CONFIG = dict(
    upper1=[106, 431, 755, 1080],
    lower1=[101, 426, 750, 1075],
    upper2=[101, 426, 750, 1075],
    lower2=[131, 456, 780, 1105],
    glu_residues=[99, 424, 748, 1073],
    asn_residues=[131, 456, 780, 1105],
    frame_duration=0.075,
    top="/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run}/com_4fs.prmtop",
    traj="/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run}/protein.nc",
    results_dir="/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN{run}",
)


def _int_list(text):
    return [int(x) for x in text.replace(",", " ").split()]


def main(argv=None):
    import MDAnalysis as mda
    from analysis.channels import Channel

    p = argparse.ArgumentParser(description="Markov checks of ion movement in the SF/cavity.")
    p.add_argument("--channel_type", default="G12")
    p.add_argument("--run", type=int, default=3)
    p.add_argument("--top")
    p.add_argument("--traj")
    p.add_argument("--results_dir", help="the RUN dir of main.py (holds analyzer_state.pkl)")
    for name in ("upper1", "lower1", "upper2", "lower2", "glu_residues", "asn_residues"):
        p.add_argument(f"--{name}", type=_int_list)
    p.add_argument("--asp_resid", type=int, help="D173.G1 (default: from converter.from_pdb)")
    p.add_argument("--frame_duration", type=float)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, help="default: last frame")
    p.add_argument("--lags", type=_int_list, default=list(DEFAULT_LAGS))
    p.add_argument("--base_lag", type=int, default=10)
    p.add_argument("--min_counts", type=int, default=50)
    p.add_argument("--glu_enter", type=float, default=3.5, help="M4/M5 Glu binding, A")
    p.add_argument("--glu_leave", type=float, default=4.5, help="M4/M5 Glu unbinding, A")
    p.add_argument("--z_buffer", type=float, default=1.0,
                   help="M4 Free_up / Free_low switch this far past z_mid, A")
    p.add_argument("--permeating_only", action="store_true",
                   help="only the permeating ions, end_1 + 1 -> end_2 of each event "
                        "(results in markov_check/RUN<run>_permeating/)")
    p.add_argument("--force_rerun", action="store_true", help="ignore the cached trajectory pass")
    args = p.parse_args(argv)

    cfg = {}
    if args.channel_type == "G12" and args.run <= 20:
        cfg = {k: (v.format(run=args.run) if isinstance(v, str) else v)
               for k, v in G12_CONFIG.items()}
    for k in ("top", "traj", "results_dir", "upper1", "lower1", "upper2", "lower2",
              "glu_residues", "asn_residues", "frame_duration"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    missing = [k for k in ("top", "traj", "results_dir", "upper1", "lower1", "upper2", "lower2",
                           "glu_residues", "asn_residues", "frame_duration") if k not in cfg]
    if missing:
        sys.exit(f"markov_check: no built-in settings for {args.channel_type} RUN{args.run}; "
                 f"pass --{' --'.join(missing)}")

    results_dir = Path(cfg["results_dir"])
    state_file = results_dir / "analyzer_state.pkl"
    if not state_file.exists():
        sys.exit(f"markov_check: {state_file} not found; run main.py for this run first")
    with open(state_file, "rb") as f:
        state = pickle.load(f)

    u = mda.Universe(cfg["top"], cfg["traj"])
    ch1 = Channel(u, cfg["upper1"], cfg["lower1"], num=1, radius=2.5)
    ch2 = Channel(u, cfg["upper2"], cfg["lower2"], num=2, radius=15.0)
    end_frame = args.end_frame if args.end_frame is not None else len(u.trajectory) - 1

    run_markov_check(
        universe=u,
        permeation_events=state["permeation_events"],
        channel1=ch1,
        channel2=ch2,
        glu_residues=cfg["glu_residues"],
        asn_residues=cfg["asn_residues"],
        results_dir=results_dir,
        run=args.run,
        channel_type=args.channel_type,
        frame_duration=cfg["frame_duration"],
        start_frame=args.start_frame,
        end_frame=end_frame,
        asp_resid=args.asp_resid,
        lags=args.lags,
        base_lag=args.base_lag,
        min_counts=args.min_counts,
        glu_enter=args.glu_enter,
        glu_leave=args.glu_leave,
        z_buffer=args.z_buffer,
        permeating_only=args.permeating_only,
        ion_region_tracking=state.get("ion_region_tracking"),
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
