#!/usr/bin/env python3
"""
ion_state_sequence.py
=====================
A short "state story" for every permeating K+ ion through the GLU/ASN cavity,
plus how loaded the cavity was when the ion left the SF and when it left the
cavity.

Tests two ideas:
  (a) load-and-repel: a loaded GLU ring repels incoming ions, so they drop
      straight to the ASN/ASP ring and get pushed out;
  (b) crowding knock-on: many free ions in the cavity push the ion out,
      whatever the GLUs are doing.

WINDOW
------
Frames end_1 -> end_2 of each permeation event (ion has left the SF -> ion
has left the GLU/ASN cavity), both inclusive. Older runs store end_1 as
'R1_end' and end_2 as 'R2_end'; both spellings are accepted.

PER-FRAME STATE OF THE PERMEATING ION
-------------------------------------
    U   upper ring: min distance ion -> side-chain polar atoms of any
        glu_residues <= cutoff
    L   lower ring: the same for asn_residues
    F   free: touching neither ring
If both rings are within cutoff the closer one wins. Polar atoms are chosen
from each residue's resname in the topology, so mutants need no special case:
    GLU (GLH) OE1 OE2     GLN OE1 NE2     ASN OD1 ND2     ASP (ASH) OD1 OD2
Distances are minimum-image (box dimensions of each frame). Every U/L frame
also keeps the residue that was closest, as a converter.to_pdb label
(e.g. 141.G1, 173.G1, 152.2b).

CLEANING
--------
1. Consecutive frames with the same state are one visit. The residue of a
   visit is the residue closest for most of its frames.
2. Visits shorter than round(min_visit_ns / frame_duration) frames (minimum
   1; G12 0.4/0.075 -> 5 frames, G2 0.4/0.4 -> 1 frame) are removed one at a
   time, shortest first; after each removal the neighbours on either side are
   merged if they now have the same state (the merged visit spans both
   neighbours and the removed blip between them). Shortest-first means a
   1-frame blip inside a long visit is removed before the two halves are
   judged, so they are rejoined rather than both dropped.
3. "out" is appended:   L>U>L>out
ever_upper, n_bounces, last_state_before_out and longest_visit_* use the
cleaned story; time_in_U/L/F_ns count the raw frames, so they add up to the
whole window. last_residue_before_out is the residue of the last U or L
visit (the last ring the ion was on, even if it then went F before leaving).

CAVITY LOAD (at SF exit = end_1 and at cavity exit = end_2)
-----------------------------------------------------------
    n_glu_occupied     GLU-ring residues (0-4) with ANY K+ other than the
                       permeating ion within cutoff of their polar atoms
    which_glu_occupied their PDB labels, ";"-joined
    n_asn_occupied     the same for the ASN/ASP ring
    n_free_cavity      other K+ inside the cavity, within cutoff of neither ring
    n_cavity_total     all other K+ inside the cavity
Single-frame values are at end_1 / end_2. *_n_glu_occupied_mean averages the
load_frames frames ending at that frame.

The CAVITY is region 2 of the permeation analysis: the ch2 cylinder that
main.py builds as Channel(u, upper2, lower2, num=2, radius=15.0), with its
geometry recomputed every frame by compute_geometry(2) and membership tested
with Channel.is_within_cylinder. It is the same region that defines end_1 and
end_2, and the one cavity_occupancy_analysis.py and untracked_cavity_ions.py
use. Pass that object as channel2.

BACKGROUND
----------
The same load quantities at N_BACKGROUND random frames of the trajectory
(fixed seed), so the aggregator can tell whether exits happen when the cavity
is unusually loaded. Exit counts leave out the permeating ion, so to match
them each background frame leaves out ONE ion too: the K+ inside the cavity
with the lowest z (the one closest to leaving, when the SF is above the
cavity). If the cavity is empty nothing is left out. Its resid is in
excluded_ion_id. The same quantities counting every K+ are kept for
reference in the *_all columns.
"Lowest z = closest to leaving" assumes the SF sits above the cavity; the
ch2 axis is checked every frame and a warning is printed (and written to the
provenance file) if it points the other way in any background frame.

OUTPUTS   <results_dir>/ion_state_sequence/
------------------------------------------
    ion_state_sequence.csv             one row per permeation event
    ion_state_sequence_frames.csv      raw per-frame state of every event
    ion_state_background.csv           load at random frames
    ion_state_sequence_provenance.txt  every parameter used
    raster_exit_aligned.png            one row per ion, aligned at end_2
    raster_by_glu_load.png             the same, entry GLU load <=1 vs >=2
    transit_vs_entry_glu_load.png      transit time vs entry GLU load
    transit_vs_entry_free_ions.png     transit time vs entry free cavity ions

USAGE - from main.py
--------------------
    from analysis.ion_state_sequence import run_ion_state_sequence

    run_ion_state_sequence(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        run               = run_number,
        channel2          = ch2,
        force_rerun       = False,
    )
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from analysis.converter import to_pdb
except ImportError:
    try:
        from converter import to_pdb
    except ImportError:
        sys.exit("ERROR: could not import to_pdb from analysis/converter.py or "
                 "converter.py. Run from the pipeline root.")

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm optional
    def tqdm(x, **kwargs):
        return x

SCRIPT_VERSION = "2026-10-21b  (background leaves out the lowest-z cavity ion)"
MODULE_DIR = "ion_state_sequence"

ION_SELECTION = "resname K+ K"

# Side-chain polar atoms per residue name (protonated variants included)
POLAR_ATOMS = {
    "GLU": ("OE1", "OE2"), "GLH": ("OE1", "OE2"), "GLN": ("OE1", "NE2"),
    "ASN": ("OD1", "ND2"), "ASP": ("OD1", "OD2"), "ASH": ("OD1", "OD2"),
}

N_BACKGROUND = 500
BACKGROUND_SEED = 0

STATE_COLOR = {"U": "#E8746B", "L": "#5AA9C9", "F": "#D9D9D9"}
STATE_NAME = {"U": "U  upper ring (GLU)", "L": "L  lower ring (ASN/ASP)", "F": "F  free"}
FIGSIZE = (8, 6.6)

CSV_NAME = "ion_state_sequence.csv"
FRAMES_NAME = "ion_state_sequence_frames.csv"
BACKGROUND_NAME = "ion_state_background.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  Pure logic — no trajectory needed, so it can be tested on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def classify_frame(d_upper, d_lower, cutoff):
    """
    State of one frame from the ion's minimum distance to each ring residue.
    d_upper / d_lower: arrays, one value per residue. Returns (state, index)
    with index into the ring that won (None for F).
    """
    iu, il = int(np.argmin(d_upper)), int(np.argmin(d_lower))
    du, dl = float(d_upper[iu]), float(d_lower[il])
    up, lo = du <= cutoff, dl <= cutoff
    if up and (not lo or du <= dl):
        return "U", iu
    if lo:
        return "L", il
    return "F", None


def run_length(states, residues):
    """Consecutive identical states -> visits [{state, residue, start, n}]."""
    visits = []
    for i, (s, r) in enumerate(zip(states, residues)):
        if visits and visits[-1]["state"] == s:
            visits[-1]["n"] += 1
            visits[-1]["_res"].append(r)
        else:
            visits.append({"state": s, "start": i, "n": 1, "_res": [r]})
    for v in visits:
        v["residue"] = _dominant(v.pop("_res"))
    return visits


def _dominant(labels):
    """Most frequent non-empty label (first one on ties), '' if none."""
    labels = [l for l in labels if l]
    if not labels:
        return ""
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    best = max(counts.values())
    return next(l for l in labels if counts[l] == best)


def clean_visits(states, residues, min_frames):
    """
    Cleaned story: run-length visits, visits shorter than min_frames removed,
    neighbours with the same state merged (spanning the removed blip). The
    merged visit's residue is the dominant residue of all its frames.
    """
    merged = run_length(states, residues)
    # Remove the SHORTEST short visit first and merge its neighbours, then
    # repeat: a 1-frame blip splitting a long visit is removed before the two
    # halves are judged, so the halves are rejoined instead of both dropped.
    while True:
        short = [i for i, v in enumerate(merged) if v["n"] < min_frames]
        if not short:
            break
        i = min(short, key=lambda j: (merged[j]["n"], j))
        merged.pop(i)
        if 0 < i < len(merged) and merged[i - 1]["state"] == merged[i]["state"]:
            left, right = merged[i - 1], merged.pop(i)
            left["n"] = right["start"] + right["n"] - left["start"]
    for m in merged:
        span = range(m["start"], m["start"] + m["n"])
        # only frames with the visit's own state, so a removed blip on the
        # other ring cannot decide which residue the visit was on
        m["residue"] = _dominant([residues[i] for i in span if states[i] == m["state"]]) \
            if m["state"] != "F" else ""
    return merged


def story_fields(visits, frame_duration):
    """Sequence strings and the story-level columns of one event."""
    seq = ">".join([v["state"] for v in visits] + ["out"])
    seq_res = ">".join([v["residue"] if v["state"] != "F" else "F" for v in visits] + ["out"])
    rings = [v["state"] for v in visits if v["state"] != "F"]
    bounces = sum(1 for a, b in zip(rings, rings[1:]) if a != b)
    ring_visits = [v for v in visits if v["state"] != "F"]
    longest = max(visits, key=lambda v: v["n"]) if visits else None
    return {
        "sequence": seq,
        "sequence_residues": seq_res,
        "n_visits": len(visits),
        "n_bounces": bounces,
        "ever_upper": "U" in rings,
        "last_state_before_out": visits[-1]["state"] if visits else "",
        "last_residue_before_out": ring_visits[-1]["residue"] if ring_visits else "",
        "longest_visit_ns": round(longest["n"] * frame_duration, 4) if longest else 0.0,
        "longest_visit_state": longest["state"] if longest else "",
    }


def background_exclusion(in_cavity, z):
    """
    Index of the ion a background frame leaves out: the cavity ion with the
    lowest z, or None when the cavity is empty.
    """
    idx = np.flatnonzero(in_cavity)
    if idx.size == 0:
        return None
    return int(idx[np.argmin(np.asarray(z)[idx])])


def load_counts(occ_upper, occ_lower, in_cavity, other, upper_labels):
    """
    Cavity load at one frame.
    occ_upper / occ_lower : bool arrays (n_ions x n_residues), ion within
                            cutoff of that residue's polar atoms
    in_cavity             : bool array (n_ions,)
    other                 : bool array (n_ions,), True for every ion except
                            the permeating one
    """
    up = occ_upper & other[:, None]
    lo = occ_lower & other[:, None]
    glu_occ = up.any(axis=0)
    touching = up.any(axis=1) | lo.any(axis=1)
    cav = in_cavity & other
    return {
        "n_glu_occupied": int(glu_occ.sum()),
        "which_glu_occupied": ";".join(l for l, o in zip(upper_labels, glu_occ) if o),
        "n_asn_occupied": int(lo.any(axis=0).sum()),
        "n_free_cavity": int((cav & ~touching).sum()),
        "n_cavity_total": int(cav.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Topology
# ─────────────────────────────────────────────────────────────────────────────

def polar_groups(u, resids):
    """[(resid, resname, AtomGroup of side-chain polar atoms)], from the topology."""
    out = []
    for resid in resids:
        res = u.select_atoms(f"resid {resid}")
        if len(res) == 0:
            sys.exit(f"ERROR: resid {resid} matched 0 atoms — wrong channel_type?")
        resname = str(res[0].resname).upper()
        names = POLAR_ATOMS.get(resname)
        if names is None:
            sys.exit(f"ERROR: resid {resid} is {resname}; no polar atoms defined "
                     f"(known: {', '.join(POLAR_ATOMS)})")
        ag = u.select_atoms(f"resid {resid} and name {' '.join(names)}")
        if len(ag) == 0:
            sys.exit(f"ERROR: resid {resid} ({resname}) has none of {names}")
        out.append((int(resid), resname, ag))
    return out


def _event_value(ev, *keys):
    for k in keys:
        if k in ev and ev[k] is not None:
            try:
                return int(ev[k])
            except (TypeError, ValueError):
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def scan_frames(u, frames, ions, upper, lower, channel2, cutoff):
    """
    One pass over the needed frames (sorted). Per frame: minimum distance of
    every ion to every ring residue (minimum image), cavity membership, every
    ion's z, and whether the ch2 axis points down (SF above the cavity).
    """
    from MDAnalysis.lib.distances import distance_array

    up_atoms = sum((g for _, _, g in upper[1:]), upper[0][2])
    lo_atoms = sum((g for _, _, g in lower[1:]), lower[0][2])
    up_split = np.cumsum([0] + [len(g) for _, _, g in upper[:-1]])
    lo_split = np.cumsum([0] + [len(g) for _, _, g in lower[:-1]])

    data = {}
    for ts in tqdm(u.trajectory[list(frames)], total=len(frames), desc="  ion state frames"):
        box = ts.dimensions
        pos = ions.positions
        d_up = np.minimum.reduceat(distance_array(pos, up_atoms.positions, box=box), up_split, axis=1)
        d_lo = np.minimum.reduceat(distance_array(pos, lo_atoms.positions, box=box), lo_split, axis=1)
        channel2.compute_geometry(2)
        in_cav = np.array([channel2.is_within_cylinder(p) for p in pos], dtype=bool)
        axis_down = bool(channel2.channel_axis[2] < 0)
        data[int(ts.frame)] = (d_up.astype(np.float32), d_lo.astype(np.float32), in_cav,
                               pos[:, 2].astype(np.float32), axis_down)
    return data


def load_block(data, frames, perm_idx, cutoff, upper_labels):
    """Load at each of `frames` (those present in data)."""
    out = []
    for f in frames:
        if f not in data:
            continue
        d_up, d_lo, in_cav = data[f][:3]
        other = np.ones(len(in_cav), dtype=bool)
        if perm_idx is not None:
            other[perm_idx] = False
        out.append(load_counts(d_up <= cutoff, d_lo <= cutoff, in_cav, other, upper_labels))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, path):
    if not rows:
        print(f"  [warn] no rows for {path.name}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


def write_provenance(path, **params):
    with open(path, "w") as f:
        f.write("ion_state_sequence.py — parameters used\n")
        f.write("=" * 50 + "\n")
        for k, v in params.items():
            f.write(f"{k:>22} : {v}\n")
    print(f"  Saved: {path}")


def _legend_outside(ax, handles):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=True, fontsize=11)


def _state_handles():
    return [Patch(facecolor=STATE_COLOR[s], edgecolor="black", label=STATE_NAME[s]) for s in "ULF"]


def _draw_raster(ax, events, frame_duration):
    """events: list of (row, visits); rows already sorted."""
    for y, (row, visits) in enumerate(events):
        e1, e2 = row["end_1"], row["end_2"]
        for v in visits:
            # frame f covers ((f - end_2 - 1), (f - end_2)] * dt, so end_2 ends at 0
            x0 = (e1 + v["start"] - e2 - 1) * frame_duration
            ax.broken_barh([(x0, v["n"] * frame_duration)], (y - 0.4, 0.8),
                           facecolors=STATE_COLOR[v["state"]], edgecolors="none")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_ylim(-0.6, max(len(events), 1) - 0.4)
    ax.set_xlabel("Time before leaving the cavity (ns)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Ion (sorted by transit time)", fontsize=13, fontweight="bold")
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_raster(events, frame_duration, title, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _draw_raster(ax, events, frame_duration)
    ax.set_title(title, fontsize=14, fontweight="bold")
    _legend_outside(ax, _state_handles())
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_raster_split(events, frame_duration, title, out_path):
    groups = [("entry GLU load ≤ 1", [e for e in events if e[0]["entry_n_glu_occupied"] <= 1]),
              ("entry GLU load ≥ 2", [e for e in events if e[0]["entry_n_glu_occupied"] >= 2])]
    xmin = min([-(r["end_2"] - r["end_1"] + 1) * frame_duration for r, _ in events] + [-frame_duration])
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
                             gridspec_kw={"height_ratios": [max(len(g), 1) for _, g in groups]})
    for ax, (name, evs) in zip(axes, groups):
        _draw_raster(ax, evs, frame_duration)
        ax.set_xlim(xmin * 1.02, -xmin * 0.02)
        ax.set_title(f"{name}  (n = {len(evs)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ion", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _legend_outside(axes[0], _state_handles())
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_box_strip(rows, key, xlabel, title, out_path, group_max=None):
    """transit_ns grouped by an integer column; median green, mean red dashed."""
    def grp(v):
        return f"≥{group_max}" if group_max is not None and v >= group_max else str(v)
    vals = sorted({min(r[key], group_max) if group_max is not None else r[key] for r in rows})
    labels = [grp(v) for v in vals]
    data = [[r["transit_ns"] for r in rows
             if (min(r[key], group_max) if group_max is not None else r[key]) == v] for v in vals]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if data:
        ax.boxplot(data, positions=range(len(data)), widths=0.55, showfliers=False,
                   medianprops=dict(color="#2E7D32", linewidth=2.2),
                   showmeans=True, meanline=True,
                   meanprops=dict(color="#C62828", linestyle="--", linewidth=2.0))
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=22, color="black",
                       alpha=0.55, zorder=3, linewidths=0)
            if d:
                ax.text(i, max(d), f"n={len(d)}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("Transit time end_1 → end_2 (ns)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    from matplotlib.lines import Line2D
    _legend_outside(ax, [Line2D([], [], color="#2E7D32", linewidth=2.2, label="median"),
                         Line2D([], [], color="#C62828", linestyle="--", linewidth=2.0, label="mean"),
                         Line2D([], [], marker="o", color="black", linestyle="", alpha=0.55, label="ion")])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline entry point — call this from main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_ion_state_sequence(universe, permeation_events, glu_residues, asn_residues,
                           results_dir, channel_type, frame_duration, run,
                           cutoff=3.5, min_visit_ns=0.4, load_frames=3,
                           force_rerun=False, channel2=None):
    """
    Build the per-ion state story and cavity load. Returns the list of row
    dicts, or None if skipped. channel2 is main.py's ch2 (the region-2
    cylinder that defines end_1/end_2); it is needed for the cavity counts.
    """
    u = universe
    results_dir = Path(results_dir)
    out_dir = results_dir / MODULE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / CSV_NAME

    if out_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {out_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None
    if channel2 is None:
        sys.exit("ERROR: run_ion_state_sequence needs channel2=ch2 (the region-2 cylinder "
                 "from main.py) for the cavity counts.")

    min_frames = max(1, int(round(min_visit_ns / frame_duration)))
    print("\n" + "=" * 70)
    print("ION STATE SEQUENCE")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  cutoff={cutoff} A  frame_duration={frame_duration} ns")
    print(f"  min_visit={min_visit_ns} ns -> {min_frames} frames   load_frames={load_frames}")

    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        sys.exit(f"ERROR: ion selection '{ION_SELECTION}' matched 0 atoms.")
    ion_index = {int(r): i for i, r in enumerate(ions.resids)}
    upper = polar_groups(u, glu_residues)
    lower = polar_groups(u, asn_residues)
    upper_labels = [to_pdb(r, channel_type) for r, _, _ in upper]
    lower_labels = [to_pdb(r, channel_type) for r, _, _ in lower]
    print(f"  {len(ions)} ions")
    print(f"  upper ring: " + ", ".join(f"{n} {l}" for (_, n, _), l in zip(upper, upper_labels)))
    print(f"  lower ring: " + ", ".join(f"{n} {l}" for (_, n, _), l in zip(lower, lower_labels)))

    n_traj = len(u.trajectory)
    events = []
    for ev in permeation_events or []:
        ion_id = _event_value(ev, "ion_id")
        e1 = _event_value(ev, "end_1", "R1_end")
        e2 = _event_value(ev, "end_2", "R2_end")
        if ion_id is None or e1 is None or e2 is None or e2 < e1:
            print(f"  [warn] event {ev.get('ion_id')}: missing or inverted end_1/end_2 — skipped")
            continue
        if ion_id not in ion_index:
            print(f"  [warn] ion {ion_id} not in '{ION_SELECTION}' — skipped")
            continue
        events.append((ion_id, e1, e2))
    print(f"  {len(events)} permeation events")

    needed = set()
    for _, e1, e2 in events:
        needed.update(range(max(e1 - load_frames + 1, 0), min(e2, n_traj - 1) + 1))
        needed.update(range(max(e2 - load_frames + 1, 0), min(e2, n_traj - 1) + 1))
    rng = np.random.default_rng(BACKGROUND_SEED)
    background = sorted(rng.choice(n_traj, size=min(N_BACKGROUND, n_traj), replace=False).tolist())
    needed.update(background)
    data = scan_frames(u, sorted(needed), ions, upper, lower, channel2, cutoff)

    rows, frame_rows, raster = [], [], []
    for ion_id, e1, e2 in events:
        idx = ion_index[ion_id]
        states, residues = [], []
        for f in range(e1, e2 + 1):
            if f not in data:
                continue
            d_up, d_lo = data[f][:2]
            s, k = classify_frame(d_up[idx], d_lo[idx], cutoff)
            label = (upper_labels[k] if s == "U" else lower_labels[k]) if k is not None else ""
            states.append(s)
            residues.append(label)
            frame_rows.append({"channel_type": channel_type, "run": run, "ion_id": ion_id,
                               "end_1": e1, "end_2": e2, "frame": f,
                               "t_before_exit_ns": round((f - e2) * frame_duration, 4),
                               "state": s, "residue": label})
        visits = clean_visits(states, residues, min_frames)

        entry = load_block(data, range(e1 - load_frames + 1, e1 + 1), idx, cutoff, upper_labels)
        exit_ = load_block(data, range(e2 - load_frames + 1, e2 + 1), idx, cutoff, upper_labels)
        en, ex = (entry[-1] if entry else {}), (exit_[-1] if exit_ else {})

        row = {
            "channel_type": channel_type, "run": run, "ion_id": ion_id,
            "end_1": e1, "end_2": e2,
            "transit_ns": round((e2 - e1) * frame_duration, 4),
            "entry_n_glu_occupied": en.get("n_glu_occupied", ""),
            "entry_n_glu_occupied_mean": round(float(np.mean([b["n_glu_occupied"] for b in entry])), 3) if entry else "",
            "entry_which_glu_occupied": en.get("which_glu_occupied", ""),
            "entry_n_asn_occupied": en.get("n_asn_occupied", ""),
            "entry_n_free_cavity": en.get("n_free_cavity", ""),
            "entry_n_cavity_total": en.get("n_cavity_total", ""),
            "exit_n_glu_occupied": ex.get("n_glu_occupied", ""),
            "exit_n_glu_occupied_mean": round(float(np.mean([b["n_glu_occupied"] for b in exit_])), 3) if exit_ else "",
            "exit_which_glu_occupied": ex.get("which_glu_occupied", ""),
            "exit_n_asn_occupied": ex.get("n_asn_occupied", ""),
            "exit_n_free_cavity": ex.get("n_free_cavity", ""),
            "exit_n_cavity_total": ex.get("n_cavity_total", ""),
        }
        story = story_fields(visits, frame_duration)
        row.update({k: story[k] for k in ("sequence", "sequence_residues", "n_visits", "n_bounces",
                                          "ever_upper", "last_state_before_out", "last_residue_before_out")})
        for s in "ULF":
            row[f"time_in_{s}_ns"] = round(states.count(s) * frame_duration, 4)
        row["longest_visit_ns"] = story["longest_visit_ns"]
        row["longest_visit_state"] = story["longest_visit_state"]
        rows.append(row)
        raster.append((row, visits))

    bg_rows, n_axis_up = [], 0
    for f in background:
        if f not in data:
            continue
        _, _, in_cav, z, axis_down = data[f]
        n_axis_up += not axis_down
        excl = background_exclusion(in_cav, z)
        matched = load_block(data, [f], excl, cutoff, upper_labels)[0]
        every = load_block(data, [f], None, cutoff, upper_labels)[0]
        bg_rows.append({"channel_type": channel_type, "run": run, "frame": f,
                        "excluded_ion_id": int(ions.resids[excl]) if excl is not None else "",
                        **matched, **{f"{k}_all": v for k, v in every.items()}})
    if n_axis_up:
        print(f"  [warn] ch2 axis points UP (SF below the cavity) in {n_axis_up} background frames: "
              f"there the lowest-z cavity ion is the one closest to the SF, not to the exit")

    write_csv(rows, out_csv)
    write_csv(frame_rows, out_dir / FRAMES_NAME)
    write_csv(bg_rows, out_dir / BACKGROUND_NAME)
    write_provenance(out_dir / "ion_state_sequence_provenance.txt",
                     script_version=SCRIPT_VERSION, channel_type=channel_type, run=run,
                     cutoff_A=cutoff, frame_duration_ns=frame_duration,
                     min_visit_ns=min_visit_ns, min_visit_frames=min_frames,
                     load_frames=load_frames, n_events=len(rows),
                     n_background_frames=len(bg_rows), background_seed=BACKGROUND_SEED,
                     background_excluded="cavity K+ with the lowest z per frame (none if cavity empty); "
                                         "*_all columns count every K+",
                     background_frames_axis_up=n_axis_up,
                     upper_ring=", ".join(f"{r}:{n}:{l}" for (r, n, _), l in zip(upper, upper_labels)),
                     lower_ring=", ".join(f"{r}:{n}:{l}" for (r, n, _), l in zip(lower, lower_labels)),
                     cavity="region 2: main.py ch2 cylinder, compute_geometry(2) every frame")

    if rows:
        raster.sort(key=lambda e: e[0]["transit_ns"])
        tag = f"{channel_type} {run}".strip()
        plot_raster(raster, frame_duration, f"{tag}: state story, aligned at cavity exit",
                    out_dir / "raster_exit_aligned.png")
        usable = [e for e in raster if e[0]["entry_n_glu_occupied"] != ""]
        plot_raster_split(usable, frame_duration, f"{tag}: state story by entry GLU load",
                          out_dir / "raster_by_glu_load.png")
        box_rows = [r for r in rows if r["entry_n_glu_occupied"] != ""]
        plot_box_strip(box_rows, "entry_n_glu_occupied", "GLU-ring residues occupied at SF exit",
                       f"{tag}: transit time vs entry GLU load", out_dir / "transit_vs_entry_glu_load.png")
        plot_box_strip(box_rows, "entry_n_free_cavity", "Free K⁺ in the cavity at SF exit",
                       f"{tag}: transit time vs free cavity ions", out_dir / "transit_vs_entry_free_ions.png",
                       group_max=2)
    return rows
