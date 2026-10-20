#!/usr/bin/env python3
"""
permeation_ion_table.py
=======================
Per-ion permeation table, ranked by permeation order.

One row per PERMEATION EVENT, numbered 1..N in the order the ions completed
their permeation (i.e. sorted by end_3). An ion that permeates twice gets two
rows with two different ranks; the `ion_id` column lets you spot repeats.

Columns
-------
  ion_rank            1..N, ordered by end_3
  ion_id              MDAnalysis resid of the K+ ion
  run, channel_type   provenance
  end_1, end_2, end_3, start_key_used
  cavity_frames       (b) end_3 - end_1   (SF exit -> past the gate)
  res_<LABEL>         (a) frames this ion was within CUTOFF of that residue,
                          counted over end_1..end_3 inclusive     [8 columns]
  frames_window          frames actually scanned for this ion (end_1..end_3)
  frames_any_contact     frames in contact with AT LEAST ONE of the 8 residues
  frames_no_contact      frames touching nothing — the mobility measure
  nions_<LABEL>       (c) number of OTHER ions within CUTOFF of that residue
                          at frame end_3                          [8 columns]
  n_other_ions_contact   distinct other ions touching any of the 8 residues at end_3
  n_cavity_region        ions in region 2 or 3 at end_3, incl. the permeating
                         ion (only if ion_region_tracking.json is present)
  incomplete_range       True if end_1..end_3 falls outside the scanned frames

Two CSVs are written with identical layout:
    permeation_ion_table_frames.csv   — durations in frames  (primary)
    permeation_ion_table_ns.csv       — the same table in ns
plus permeation_ion_table_provenance.txt recording every parameter used.

This script is deliberately SELF-CONTAINED: it does its own trajectory pass and
its own contact measurement, and imports nothing from the analysis package
except the residue-numbering converter (the single shared source of truth).

USAGE — from main.py (primary)
------------------------------
    from analysis.permeation_ion_table import run_permeation_ion_table

    run_permeation_ion_table(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        cutoff            = 3.5,
        run               = run_number,
    )

USAGE — standalone (re-run one RUN without the full pipeline)
-------------------------------------------------------------
    python3 permeation_ion_table.py \
        --top   /path/com_4fs.prmtop \
        --traj  /path/05_ProdN.nc \
        --results_dir /path/girk_analyser_results/G12/RUN3 \
        --channel_type G12 --run 3 --cutoff 3.5
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# ── the ONE shared import ────────────────────────────────────────────────────
try:
    from analysis.converter import to_pdb
except ImportError:
    try:
        from converter import to_pdb
    except ImportError:
        sys.exit("ERROR: could not import to_pdb from analysis/converter.py or "
                 "converter.py. Run from the pipeline root, or put converter.py "
                 "next to this script.")


# ── channel configuration (mirrors main.py) ──────────────────────────────────
CHANNEL_CONFIG = {
    "G2":      dict(glu=[98, 426, 754, 1082], asn=[130, 458, 786, 1114],
                    frame_duration=0.4),
    "G12":     dict(glu=[99, 424, 748, 1073], asn=[131, 456, 780, 1105],
                    frame_duration=0.075),
    "G12_ML":  dict(glu=[99, 424, 749, 1074], asn=[131, 456, 781, 1106],
                    frame_duration=0.075),
    "G12_GAT": dict(glu=[98, 422, 747, 1073], asn=[130, 454, 779, 1105],
                    frame_duration=0.075),
}

ION_SELECTION = "resname K+ K"

# Bump this when the column layout or a definition changes — it is printed at
# run time and written into the provenance file, so you can always tell which
# version of the script produced a given CSV.
SCRIPT_VERSION = "2026-09-26c  (end_1 entry; pre-loaded ions; any/no-contact columns)"


# ─────────────────────────────────────────────────────────────────────────────
#  Event handling
# ─────────────────────────────────────────────────────────────────────────────

def load_permeation_events(results_dir):
    """Read analyzer.permeation_events out of analyzer_state.pkl."""
    state_file = Path(results_dir) / "analyzer_state.pkl"
    if not state_file.is_file():
        sys.exit(f"ERROR: {state_file} not found. Run main.py for this RUN first "
                 f"— the permeation events cannot be reconstructed here.")
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    events = state.get("permeation_events")
    if not events:
        sys.exit(f"ERROR: analyzer_state.pkl contains no permeation events.")
    print(f"  Loaded {len(events)} permeation events from {state_file.name}")
    return events, state


def resolve_cavity_start(event):
    """
    Cavity-entry frame for (b): the frame the ion leaves the selectivity filter.
    Prefers end_1 (stored as 'R1_end' in older runs), falling back to start_2,
    which marks the same transition from the region-2 side.

    Returns (frame, key_name). If NO entry frame exists the ion was already
    inside the cavity when the analysed window began — it never crossed in from
    the SF — and (None, None) is returned so the caller can substitute the
    first analysed frame and mark the row as left-censored.
    """
    for key in ("end_1", "R1_end", "start_2", "start_1"):
        if key in event and event[key] is not None:
            try:
                return int(event[key]), key
            except (TypeError, ValueError):
                continue
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Contact map — one vectorized trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def build_contact_map(u, residues, cutoff, lo, hi):
    """
    contacts[frame][resid] = set of ion resids within `cutoff` of ANY atom of
    that residue. Minimum-distance criterion, all residue atoms, PBC-aware.

    One pass over frames lo..hi inclusive.
    """
    from MDAnalysis.analysis import distances as mda_distances

    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        sys.exit(f"ERROR: ion selection '{ION_SELECTION}' matched 0 atoms.")
    ion_resids = np.array([a.resid for a in ions])

    res_groups = {}
    for resid in residues:
        ag = u.select_atoms(f"resid {resid}")
        if len(ag) == 0:
            sys.exit(f"ERROR: resid {resid} matched 0 atoms — wrong channel_type?")
        res_groups[resid] = ag

    print(f"  {len(ions)} ions, {len(res_groups)} residues, "
          f"cutoff {cutoff} A, frames {lo}..{hi}")

    contacts = {}
    n = 0
    for ts in u.trajectory[lo:hi + 1]:
        frame = ts.frame
        box = ts.dimensions
        per_frame = {}
        ion_pos = ions.positions
        for resid, ag in res_groups.items():
            d = mda_distances.distance_array(ion_pos, ag.positions, box=box)
            close = ion_resids[d.min(axis=1) <= cutoff]
            per_frame[resid] = set(int(r) for r in close)
        contacts[frame] = per_frame
        n += 1
        if n % 500 == 0:
            print(f"    frame {frame}/{hi}", end="\r")
    print(f"    scanned {n} frames." + " " * 20)
    return contacts


# ─────────────────────────────────────────────────────────────────────────────
#  Region-based cavity count (optional)
# ─────────────────────────────────────────────────────────────────────────────

def load_region_tracking(results_dir):
    path = Path(results_dir) / "ion_region_tracking.json"
    if not path.is_file():
        print(f"  [info] {path.name} not found — n_cavity_region will be blank.")
        return None
    with open(path) as f:
        raw = json.load(f)
    tracking = {int(ion): sorted((int(fr), int(rg)) for fr, rg in seq.items())
                for ion, seq in raw.items()}
    print(f"  Loaded region tracking for {len(tracking)} ions")
    return tracking


def region_at_frame(frames_sorted, target):
    """Last known region at or before `target`; None if the ion has no record yet."""
    region = None
    for fr, rg in frames_sorted:
        if fr > target:
            break
        region = rg
    return region


def count_cavity_region(tracking, perm_ion, end_3):
    """Ions in region 2 or 3 at end_3, plus the permeating ion itself."""
    n = 0
    for ion_id, seq in tracking.items():
        if ion_id == perm_ion:
            continue
        if region_at_frame(seq, end_3) in (2, 3):
            n += 1
    return n + 1


# ─────────────────────────────────────────────────────────────────────────────
#  Residue labelling
# ─────────────────────────────────────────────────────────────────────────────

def residue_labels(u, residues, channel_type):
    """
    label = "<RESNAME><pdb>" e.g. E141.G1, D173.G1, N184.2c.
    Residue type is read from the TOPOLOGY, never hardcoded — this is what
    prevents the ASP/ASN mislabelling that bit the other modules.
    """
    one_letter = {"GLU": "E", "ASP": "D", "ASN": "N", "GLN": "Q"}
    labels = {}
    for resid in residues:
        ag = u.select_atoms(f"resid {resid}")
        resname = ag[0].resname if len(ag) else "?"
        pdb = to_pdb(resid, channel_type)
        labels[resid] = f"{one_letter.get(resname, resname)}{pdb}"
    return labels


# ─────────────────────────────────────────────────────────────────────────────
#  Table construction
# ─────────────────────────────────────────────────────────────────────────────

def build_table(events, contacts, residues, labels, tracking,
                channel_type, run, lo, hi):
    rows = []
    usable = []
    n_censored = 0
    for ev in events:
        start, key = resolve_cavity_start(ev)
        end_3 = int(ev["end_3"]) if ev.get("end_3") is not None else None
        if end_3 is None:
            print(f"  [warn] ion {ev.get('ion_id')}: no end_3 — skipped")
            continue
        if start is None:
            # Ion was already inside the cavity when the window opened. Its true
            # entry frame is unknown, so residence is measured from the first
            # analysed frame and is a LOWER BOUND.
            start, key = lo, "trajectory_start"
            n_censored += 1
        usable.append((ev, start, key, end_3))

    if n_censored:
        print(f"  {n_censored} ion(s) were already in the cavity at frame {lo} "
              f"— included, with start_censored=True (residence is a lower bound)")

    usable.sort(key=lambda t: t[3])          # rank by end_3

    for rank, (ev, start, key, end_3) in enumerate(usable, start=1):
        ion_id = int(ev["ion_id"])
        incomplete = (start < lo) or (end_3 > hi)
        s, e = max(start, lo), min(end_3, hi)

        row = {
            "ion_rank": rank,
            "ion_id": ion_id,
            "run": run,
            "channel_type": channel_type,
            "end_1": start,
            "start_key_used": key,
            "start_censored": (key == "trajectory_start"),
            "end_2": int(ev["end_2"]) if ev.get("end_2") is not None else "",
            "end_3": end_3,
            "cavity_frames": end_3 - start,
        }

        # (a) per-residue residence, summed over start..end_3 inclusive.
        # Also track, per frame, whether the ion touched ANY of the 8 residues —
        # this is the mobility measure: frames in contact with something vs
        # frames free in the cavity touching nothing.
        n_window = 0
        n_any = 0
        per_res_counts = {resid: 0 for resid in residues}
        for f in range(s, e + 1):
            per_frame = contacts.get(f)
            if per_frame is None:
                continue
            n_window += 1
            touched = False
            for resid in residues:
                if ion_id in per_frame[resid]:
                    per_res_counts[resid] += 1
                    touched = True
            if touched:
                n_any += 1
        for resid in residues:
            row[f"res_{labels[resid]}"] = per_res_counts[resid]

        row["frames_window"] = n_window
        row["frames_any_contact"] = n_any
        row["frames_no_contact"] = n_window - n_any

        # (c) other ions in contact with each residue at end_3
        at_end3 = contacts.get(end_3)
        others_any = set()
        for resid in residues:
            if at_end3 is None:
                row[f"nions_{labels[resid]}"] = ""
                continue
            others = at_end3[resid] - {ion_id}
            row[f"nions_{labels[resid]}"] = len(others)
            others_any |= others
        row["n_other_ions_contact"] = len(others_any) if at_end3 is not None else ""

        row["n_cavity_region"] = (count_cavity_region(tracking, ion_id, end_3)
                                  if tracking else "")
        row["incomplete_range"] = incomplete
        rows.append(row)

    return rows


DURATION_PREFIXES = ("cavity_frames", "res_")


def to_ns_rows(rows, frame_duration):
    """Convert every duration column to ns; counts and frame indices unchanged."""
    out = []
    for r in rows:
        nr = dict(r)
        for k, v in r.items():
            if (k in ("cavity_frames", "frames_window",
                      "frames_any_contact", "frames_no_contact")
                    or k.startswith("res_")):
                if v != "" and v is not None:
                    nr[k] = round(float(v) * frame_duration, 4)
        nr["cavity_ns"] = nr.pop("cavity_frames")
        out.append(nr)
    return out


def write_csv(rows, path):
    if not rows:
        print("  [warn] no rows to write")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


def write_provenance(path, **params):
    with open(path, "w") as f:
        f.write("permeation_ion_table.py — parameters used\n")
        f.write("=" * 50 + "\n")
        for k, v in params.items():
            f.write(f"{k:>20} : {v}\n")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline entry point — call this from main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_permeation_ion_table(universe, permeation_events, glu_residues,
                             asn_residues, results_dir, channel_type,
                             frame_duration=None, start_frame=0, end_frame=None,
                             cutoff=3.5, run="", force_rerun=False):
    """
    Build the per-ion permeation table. Returns the list of row dicts
    (frames version), or None if skipped / nothing to do.

    Parameters mirror the other analysis modules: pass the live Universe and
    analyzer.permeation_events straight through — no pickle round-trip.
    """
    u = universe
    residues = list(glu_residues) + list(asn_residues)
    if frame_duration is None:
        frame_duration = CHANNEL_CONFIG.get(channel_type, {}).get("frame_duration", 0.075)

    results_dir = Path(results_dir)
    out_dir = results_dir / "permeation_ion_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_csv = out_dir / "permeation_ion_table_frames.csv"

    if frames_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {frames_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None

    print("\n" + "=" * 70)
    print("PERMEATION ION TABLE")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  cutoff={cutoff} A  "
          f"frame_duration={frame_duration} ns")

    if not permeation_events:
        print("  No permeation events — nothing to tabulate.")
        return None
    print(f"  {len(permeation_events)} permeation events")

    # ── DEBUG: what keys do the events actually carry? ───────────────────────
    print("\n  [debug] keys present in event[0]:")
    print(f"          {sorted(permeation_events[0].keys())}")
    print("  [debug] frame keys for the first 5 events "
          "(which one is used for cavity entry?):")
    print(f"    {'ion_id':>8} {'end_1':>8} {'R1_end':>8} {'start_2':>8} "
          f"{'start_1':>8} {'end_2':>8} {'end_3':>8}  -> chosen")
    for ev in permeation_events[:5]:
        chosen, key = resolve_cavity_start(ev)
        def _g(k):
            v = ev.get(k)
            return "-" if v is None else str(v)
        print(f"    {str(ev.get('ion_id')):>8} {_g('end_1'):>8} {_g('R1_end'):>8} "
              f"{_g('start_2'):>8} {_g('start_1'):>8} {_g('end_2'):>8} "
              f"{_g('end_3'):>8}  -> {key}={chosen}")

    n_same = sum(1 for ev in permeation_events
                 if ev.get("end_1") is not None and ev.get("start_2") is not None
                 and int(ev["end_1"]) == int(ev["start_2"]))
    n_both = sum(1 for ev in permeation_events
                 if ev.get("end_1") is not None and ev.get("start_2") is not None)
    if n_both:
        print(f"  [debug] end_1 == start_2 in {n_same}/{n_both} events "
              f"— if all, switching between them changes nothing.")
    else:
        print(f"  [debug] end_1 and start_2 are not both present; "
              f"no comparison possible.")

    tracking = load_region_tracking(results_dir)

    labels = residue_labels(u, residues, channel_type)
    print("  Residues tracked:")
    for r in residues:
        print(f"    resid {r:>5} -> {labels[r]}")

    # Frame window: the events' own span, clipped to the analysed range
    resolved = [resolve_cavity_start(e)[0] for e in permeation_events]
    starts = [s for s in resolved if s is not None]
    has_censored = any(s is None for s in resolved)
    ends = [int(e["end_3"]) for e in permeation_events
            if e.get("end_3") is not None]
    if not ends:
        print("  ERROR: no usable end_3 frames in the events — skipping.")
        return None

    traj_last = len(u.trajectory) - 1
    hard_end = traj_last if end_frame is None else min(int(end_frame), traj_last)
    # Ions already in the cavity are measured from the first analysed frame,
    # so the scan has to start there rather than at the earliest SF crossing.
    lo = int(start_frame) if has_censored else max(int(start_frame),
                                                  min(starts) if starts else int(start_frame))
    hi = min(hard_end, max(ends))

    print(f"\n  [debug] events with a cavity-entry frame: {len(starts)}/"
          f"{len(permeation_events)}   already-in-cavity: "
          f"{len(permeation_events) - len(starts)}")
    print(f"  [debug] scan window {lo}..{hi}  "
          f"(start_frame={start_frame}, end_frame={end_frame}, "
          f"traj_last={traj_last})")
    if starts:
        print(f"  [debug] earliest entry frame {min(starts)}, "
              f"latest end_3 {max(ends)}")
    if hi < lo:
        print(f"  ERROR: empty frame window ({lo}..{hi}) — skipping.")
        return None

    contacts = build_contact_map(u, residues, cutoff, lo, hi)

    rows = build_table(permeation_events, contacts, residues, labels, tracking,
                       channel_type, run, lo, hi)
    write_csv(rows, frames_csv)
    write_csv(to_ns_rows(rows, frame_duration),
              out_dir / "permeation_ion_table_ns.csv")
    write_provenance(
        out_dir / "permeation_ion_table_provenance.txt",
        script_version=SCRIPT_VERSION,
        cavity_entry_key="end_1 (fallback R1_end -> start_2 -> start_1)",
        channel_type=channel_type, run=run, cutoff_A=cutoff,
        frame_duration_ns=frame_duration, ion_selection=ION_SELECTION,
        contact_criterion="min distance, ion to ANY atom of residue, PBC-aware",
        glu_residues=list(glu_residues), asn_residues=list(asn_residues),
        labels={r: labels[r] for r in residues},
        frames_scanned=f"{lo}..{hi}",
        topology=getattr(u, "filename", ""),
        trajectory=getattr(u.trajectory, "filename", ""),
        n_events=len(rows),
        n_incomplete=sum(1 for r in rows if r["incomplete_range"]),
    )

    n_bad = sum(1 for r in rows if r["incomplete_range"])
    if n_bad:
        print(f"  [warn] {n_bad} row(s) span frames outside {lo}..{hi} — "
              f"their res_* counts are truncated (incomplete_range=True).")
    print(f"\n✓ Done. {len(rows)} ions ranked 1..{len(rows)}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True, help="Topology (com_4fs.prmtop)")
    ap.add_argument("--traj", required=True,
                    help="Production trajectory (NOT the stripped protein.nc)")
    ap.add_argument("--results_dir", required=True,
                    help="RUN results dir holding analyzer_state.pkl")
    ap.add_argument("--channel_type", required=True, choices=list(CHANNEL_CONFIG))
    ap.add_argument("--run", default="", help="Run label for the run column")
    ap.add_argument("--cutoff", type=float, default=3.5)
    ap.add_argument("--frame_duration", type=float, default=None,
                    help="ns per frame (default: channel-specific from main.py)")
    ap.add_argument("--glu", nargs="+", type=int, default=None)
    ap.add_argument("--asn", nargs="+", type=int, default=None)
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("ERROR: MDAnalysis required — activate the Amber conda env.")

    cfg = CHANNEL_CONFIG[args.channel_type]
    glu = args.glu or cfg["glu"]
    asn = args.asn or cfg["asn"]

    results_dir = Path(args.results_dir)
    events, state = load_permeation_events(results_dir)

    u = mda.Universe(args.top, args.traj)

    run_permeation_ion_table(
        universe=u,
        permeation_events=events,
        glu_residues=glu,
        asn_residues=asn,
        results_dir=results_dir,
        channel_type=args.channel_type,
        frame_duration=args.frame_duration or cfg["frame_duration"],
        start_frame=state.get("start_frame", 0) or 0,
        end_frame=state.get("end_frame"),
        cutoff=args.cutoff,
        run=args.run,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()