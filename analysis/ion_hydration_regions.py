#!/usr/bin/env python3
"""
ion_hydration_regions.py
========================
How many waters each K+ ion carries, broken down by which cavity region the
ion is in — and, at the two permeation moments, how the permeating ion's own
hydration compares with everyone else's.

WHAT IS COUNTED
---------------
An ion's hydration number = water oxygens within --shell of the ion
(default 3.4 A, the end of the first K+-O coordination peak; bulk K+ carries
about 6-8). PBC-aware, computed with a grid search so it stays cheap.

PER FRAME
---------
For each of the three regions (same geometry as cavity_water_regions.py, so
the definitions cannot drift apart):

    GLU/ASN   between the GLU ring and the ASN/ASP ring
    ASN/HBC   between the ASN/ASP ring and the HBC gate
    GLU/HBC   between the GLU ring and the HBC gate  (union of the two above)

the ions inside that region are found, each one's waters are counted, and the
frame gets: n_ions, mean, median, min, max of those counts. A frame with no
ion in a region is written as blank, not as zero — no ion is not the same as
an ion with no water.

AT THE PERMEATION MOMENTS
-------------------------
    end_2   the frame the ion leaves the GLU/ASN cavity
    end_3   the frame the ion passes the HBC gate

For each event, at both frames:
    - the PERMEATING ion's own hydration number (one number, no averaging)
    - the mean / median hydration of the OTHER ions in each region

Note the permeating ion's own count is measured directly, not conditioned on
a region: at end_3 it has already crossed the HBC and is in none of them.
"Others" always excludes the permeating ion, so the comparison is clean.

OUTPUTS   <results_dir>/water_analysis/ion_hydration_regions/
------------------------------------------------------------
    ion_hydration_per_frame.csv      per frame x region: n_ions, mean, median
    ion_hydration_events.csv         per event: permeating ion vs others
    ion_hydration_events_summary.csv pooled over events, the headline numbers
    ion_hydration_per_ion.csv        every (frame, ion, region, n_waters)
                                     -- only with --dump_per_ion, it is large
    ion_hydration_regions.pkl        everything, for a later aggregator
    ion_hydration_timeseries.png     3 panels: mean waters/ion vs frame
    ion_hydration_mean_bars.png      3 bars, all frames
    ion_hydration_end2_bars.png      permeating ion vs others, at end_2
    ion_hydration_end3_bars.png      permeating ion vs others, at end_3
    ion_hydration_event_hist.png     distribution of the permeating ion's
                                     hydration at end_2 and at end_3

USAGE - from main.py (primary)
------------------------------
    from analysis.ion_hydration_regions import run_ion_hydration_regions

    run_ion_hydration_regions(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        radius            = 12.0,
        shell             = 3.4,
        run               = run_number,
        force_rerun       = False,
    )

USAGE - standalone
------------------
    python3 ion_hydration_regions.py \
        --top  /path/com_4fs.prmtop \
        --traj /path/05_ProdN.nc \
        --results_dir /path/girk_analyser_results/G12/RUN3 \
        --channel_type G12 --run 3 --shell 3.4
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Region geometry, channel rings and event parsing are shared with the
# sibling module on purpose: if the cavity definition ever changes it must
# change in exactly one place.
try:
    from analysis.cavity_water_regions import (
        CHANNEL_CONFIG, WATER_SELECTION, WATER_ROOT, REGIONS, REGION_COLOR,
        count_in_slab, region_bounds, event_frames, load_events_from_disk,
    )
except ImportError:
    try:
        from cavity_water_regions import (
            CHANNEL_CONFIG, WATER_SELECTION, WATER_ROOT, REGIONS, REGION_COLOR,
            count_in_slab, region_bounds, event_frames, load_events_from_disk,
        )
    except ImportError:
        sys.exit("ERROR: cavity_water_regions.py must sit next to this file "
                 "(or in analysis/) — the cavity definitions are shared.")

SCRIPT_VERSION = "2026-10-06c  (+ residue-contact split, + overview figure)"
MODULE_DIR = "ion_hydration_regions"

ION_SELECTION = "resname K+ K"
DEFAULT_SHELL = 3.4
# Contact cutoff for "this ion is sitting on that residue", measured as the
# MINIMUM distance from the ion to any atom of the residue -- the same
# criterion permeation_ion_table.py uses, so the two are comparable.
DEFAULT_CONTACT = 3.5
# The GLU/ASN cavity is where the residue split is computed: those eight
# residues line that region and nothing else.
CONTACT_REGION = "GLU/ASN"

PERM_COLOR = "#C0392B"
OTHER_COLOR = "#90A4AE"



# ── progress bar ─────────────────────────────────────────────────────────────

def _progress(iterable, total=None, desc="frames"):
    """
    tqdm bar when tqdm is installed, plain pass-through otherwise, so the
    module never hard-depends on it.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="frame",
                dynamic_ncols=True, leave=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Core counting — numpy only, testable without a trajectory
# ─────────────────────────────────────────────────────────────────────────────

def residue_labels(u, residues, channel_type):
    """
    "E141.G1", "D173.G1", ... Residue TYPE comes from the topology, never a
    hardcoded list -- that is what keeps ASP from being mislabelled ASN.
    """
    try:
        from analysis.converter import to_pdb
    except ImportError:
        try:
            from converter import to_pdb
        except ImportError:
            to_pdb = None
    one = {"GLU": "E", "ASP": "D", "ASN": "N", "GLN": "Q"}
    out = {}
    for resid in residues:
        ag = u.select_atoms(f"resid {resid}")
        rn = ag[0].resname if len(ag) else "?"
        pdb = to_pdb(resid, channel_type) if to_pdb else str(resid)
        out[resid] = f"{one.get(rn, rn)}{pdb}"
    return out


def ions_in_region(ion_pos, axis_xy, z_lo, z_hi, radius):
    """Indices of ions inside the region cylinder (same test as the waters)."""
    if ion_pos.size == 0:
        return np.array([], dtype=int)
    z = ion_pos[:, 2]
    in_z = (z >= z_lo) & (z <= z_hi)
    idx = np.flatnonzero(in_z)
    if idx.size == 0:
        return idx
    dxy = ion_pos[idx, :2] - np.asarray(axis_xy, dtype=float)
    keep = np.einsum("ij,ij->i", dxy, dxy) <= radius * radius
    return idx[keep]


def shell_counts(ion_pos, water_pos, shell, box=None):
    """
    Waters within `shell` of each ion, as an int array aligned with ion_pos.
    Uses MDAnalysis' grid search so it does not build the full ion x water
    distance matrix.
    """
    from MDAnalysis.lib.distances import capped_distance
    if ion_pos.size == 0:
        return np.zeros(0, dtype=int)
    pairs, _ = capped_distance(ion_pos, water_pos, max_cutoff=shell,
                               box=box, return_distances=True)
    if len(pairs) == 0:
        return np.zeros(len(ion_pos), dtype=int)
    return np.bincount(pairs[:, 0], minlength=len(ion_pos)).astype(int)


def describe(values):
    """mean / median / min / max / n for a set of per-ion counts."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return dict(n_ions=0, mean="", median="", minimum="", maximum="")
    return dict(n_ions=int(v.size),
                mean=round(float(v.mean()), 3),
                median=round(float(np.median(v)), 3),
                minimum=int(v.min()), maximum=int(v.max()))


# ─────────────────────────────────────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────────────────────────────────────

def ranked_events(permeation_events):
    """
    [(rank, ion_id, end_2, end_3)] sorted by end_3, so the rank matches the
    ion_rank in permeation_ion_table.py (the professor's Ion 1..N).
    """
    out = []
    for ev in permeation_events or []:
        ion = ev.get("ion_id")
        if ion is None:
            continue
        def _g(*keys):
            for k in keys:
                if k in ev and ev[k] is not None:
                    try:
                        return int(ev[k])
                    except (TypeError, ValueError):
                        pass
            return None
        e3 = _g("end_3", "R3_end")
        if e3 is None:
            continue
        out.append([None, int(ion), _g("end_2", "R2_end"), e3])
    out.sort(key=lambda t: t[3])
    for i, rec in enumerate(out, start=1):
        rec[0] = i
    return [tuple(r) for r in out]


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def scan_trajectory(u, glu, asn, hbc, radius, shell, lo, hi,
                    want_frames, dump_per_ion=False, contact=DEFAULT_CONTACT,
                    labels=None):
    """
    One pass. `want_frames` maps frame -> list of permeating ion resids whose
    own hydration must be recorded at that frame (end_2 and end_3 frames).

    When `labels` is given (resid -> label for the 8 GLU/ASN residues), each
    GLU/ASN-cavity ion is also classified per frame as in contact with at
    least one of those residues or touching none of them, and the hydration
    of each group is recorded -- plus, per residue, the hydration of the ions
    sitting on it.
    """
    waters = u.select_atoms(WATER_SELECTION)
    if len(waters) == 0:
        sys.exit(f"ERROR: water selection matched 0 atoms.\n"
                 f"  tried: {WATER_SELECTION}\n"
                 f"  Is this the SOLVATED trajectory, not protein.nc?")
    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        sys.exit(f"ERROR: ion selection '{ION_SELECTION}' matched 0 atoms.")
    ion_resids = np.array([a.resid for a in ions])

    rings = {}
    for name, resids in (("glu", glu), ("asn", asn), ("hbc", hbc)):
        ag = u.select_atoms("resid " + " ".join(str(r) for r in resids))
        if len(ag) == 0:
            sys.exit(f"ERROR: {name} ring {resids} matched 0 atoms — "
                     f"wrong channel_type?")
        rings[name] = ag

    res_groups = {}
    if labels:
        for resid in labels:
            ag = u.select_atoms(f"resid {resid}")
            if len(ag) == 0:
                sys.exit(f"ERROR: resid {resid} matched 0 atoms — "
                         f"wrong channel_type?")
            res_groups[resid] = ag

    print(f"  {len(ions)} ions, {len(waters)} water oxygens, "
          f"shell {shell} A, radius {radius} A, frames {lo}..{hi}")
    if res_groups:
        print(f"  contact split in {CONTACT_REGION} at {contact} A over "
              f"{len(res_groups)} residues: "
              f"{', '.join(labels[r] for r in sorted(labels))}")

    per_frame = []       # one dict per frame
    per_ion_rows = []    # optional long dump
    event_hits = {}      # (frame, ion_resid) -> hydration number
    n = 0

    for ts in _progress(u.trajectory[lo:hi + 1], total=hi - lo + 1,
                        desc="ion hydration"):
        frame = int(ts.frame)
        box = ts.dimensions
        z_glu = float(rings["glu"].center_of_mass()[2])
        z_asn = float(rings["asn"].center_of_mass()[2])
        z_hbc = float(rings["hbc"].center_of_mass()[2])
        axis_xy = np.mean([rings["glu"].center_of_mass()[:2],
                           rings["asn"].center_of_mass()[:2],
                           rings["hbc"].center_of_mass()[:2]], axis=0)
        bounds = region_bounds(z_glu, z_asn, z_hbc)

        ion_pos = ions.positions
        water_pos = waters.positions

        # Which ions do we need a hydration number for this frame?
        # The union of (ions in GLU/HBC, the widest region) and any permeating
        # ion this frame — the latter may already be past the gate.
        needed = set(ions_in_region(ion_pos, axis_xy, *bounds["GLU/HBC"],
                                    radius).tolist())
        for resid in want_frames.get(frame, []):
            w = np.flatnonzero(ion_resids == resid)
            needed.update(w.tolist())

        needed = np.array(sorted(needed), dtype=int)
        counts = (shell_counts(ion_pos[needed], water_pos, shell, box)
                  if needed.size else np.zeros(0, dtype=int))
        count_of = {int(i): int(c) for i, c in zip(needed, counts)}

        row = dict(frame=frame, z_glu=z_glu, z_asn=z_asn, z_hbc=z_hbc)
        for region in REGIONS:
            idx = ions_in_region(ion_pos, axis_xy, *bounds[region], radius)
            vals = [count_of[int(i)] for i in idx if int(i) in count_of]
            d = describe(vals)
            row[region] = dict(d, ion_resids=[int(ion_resids[i]) for i in idx],
                               counts=list(vals))

            # ---- residue-contact split, GLU/ASN only ----
            # One distance_array per residue per frame, reused for both the
            # per-residue tally and the any/none split -- no second pass.
            if res_groups and region == CONTACT_REGION:
                from MDAnalysis.analysis import distances as mda_dist
                keep = [int(i) for i in idx if int(i) in count_of]
                per_res = {r: [] for r in res_groups}
                touch_any, touch_none = [], []
                if keep:
                    pos_keep = ion_pos[keep]
                    any_flag = np.zeros(len(keep), dtype=bool)
                    for resid, ag in res_groups.items():
                        dmin = mda_dist.distance_array(
                            pos_keep, ag.positions, box=box).min(axis=1)
                        close = dmin <= contact
                        any_flag |= close
                        for k, c in zip(keep, close):
                            if c:
                                per_res[resid].append(count_of[k])
                    for k, fl in zip(keep, any_flag):
                        (touch_any if fl else touch_none).append(count_of[k])
                row["contact"] = dict(
                    describe(touch_any),
                    per_residue={labels[r]: describe(v)
                                 for r, v in per_res.items()})
                row["no_contact"] = describe(touch_none)
                # raw per-ion values, for pooling over ion-frames later
                row["_contact_vals"] = list(touch_any)
                row["_no_contact_vals"] = list(touch_none)
                row["_per_res_vals"] = {labels[r]: list(v)
                                        for r, v in per_res.items()}
            if dump_per_ion:
                for i, c in zip(idx, vals):
                    per_ion_rows.append(dict(frame=frame, region=region,
                                             ion_id=int(ion_resids[i]),
                                             n_waters=int(c)))
        per_frame.append(row)

        for resid in want_frames.get(frame, []):
            w = np.flatnonzero(ion_resids == resid)
            if w.size:
                event_hits[(frame, int(resid))] = count_of.get(int(w[0]), 0)

        n += 1
    print(f"    scanned {n} frames.")
    return per_frame, per_ion_rows, event_hits


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def write_per_frame_csv(per_frame, path, frame_duration, res_labels=None):
    cols = ["frame", "time_ns"]
    for region in REGIONS:
        tag = region.replace("/", "_")
        cols += [f"n_ions_{tag}", f"mean_{tag}", f"median_{tag}",
                 f"min_{tag}", f"max_{tag}"]
    has_contact = any("contact" in r for r in per_frame)
    if has_contact:
        cols += ["n_contact", "mean_contact", "n_no_contact", "mean_no_contact"]
        if res_labels:
            for lab in res_labels:
                cols += [f"n_{lab}", f"mean_{lab}"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in per_frame:
            out = [r["frame"], round(r["frame"] * frame_duration, 4)]
            for region in REGIONS:
                d = r[region]
                out += [d["n_ions"], d["mean"], d["median"],
                        d["minimum"], d["maximum"]]
            if has_contact:
                c = r.get("contact", {})
                nc = r.get("no_contact", {})
                out += [c.get("n_ions", ""), c.get("mean", ""),
                        nc.get("n_ions", ""), nc.get("mean", "")]
                if res_labels:
                    pr = c.get("per_residue", {})
                    for lab in res_labels:
                        d = pr.get(lab, {})
                        out += [d.get("n_ions", ""), d.get("mean", "")]
            w.writerow(out)
    print(f"    Saved: {Path(path).name}  ({len(per_frame)} rows)")


def build_event_rows(events, per_frame_by_frame, event_hits, channel_type, run):
    """
    One row per permeation event, with the permeating ion's own hydration and
    the others' mean/median in each region, at end_2 and at end_3.
    """
    rows = []
    for rank, ion_id, e2, e3 in events:
        row = dict(channel_type=channel_type, run=run, ion_rank=rank,
                   ion_id=ion_id, end_2=e2 if e2 is not None else "", end_3=e3)
        for moment, frame in (("end2", e2), ("end3", e3)):
            if frame is None or frame not in per_frame_by_frame:
                row[f"perm_waters_{moment}"] = ""
                for region in REGIONS:
                    tag = region.replace("/", "_")
                    row[f"others_n_{moment}_{tag}"] = ""
                    row[f"others_mean_{moment}_{tag}"] = ""
                    row[f"others_median_{moment}_{tag}"] = ""
                continue
            row[f"perm_waters_{moment}"] = event_hits.get((frame, ion_id), "")
            fr = per_frame_by_frame[frame]
            for region in REGIONS:
                tag = region.replace("/", "_")
                resids = fr[region]["ion_resids"]
                counts = fr[region]["counts"]
                others = [c for rid, c in zip(resids, counts) if rid != ion_id]
                d = describe(others)
                row[f"others_n_{moment}_{tag}"] = d["n_ions"]
                row[f"others_mean_{moment}_{tag}"] = d["mean"]
                row[f"others_median_{moment}_{tag}"] = d["median"]
        rows.append(row)
    return rows


def summarise_events(event_rows, channel_type, run):
    """Pooled over all events: permeating ion vs others, at each moment."""
    out = []
    for moment in ("end2", "end3"):
        perm = [float(r[f"perm_waters_{moment}"]) for r in event_rows
                if str(r[f"perm_waters_{moment}"]).strip() != ""]
        rec = dict(channel_type=channel_type, run=run, moment=moment,
                   who="permeating_ion", region="(any)", **describe(perm))
        rec["n_events"] = rec.pop("n_ions")
        out.append(rec)
        for region in REGIONS:
            tag = region.replace("/", "_")
            # Pool the per-event MEANS: each event contributes one number, so
            # an event with 5 ions in the cavity does not outvote one with 1.
            vals = [float(r[f"others_mean_{moment}_{tag}"]) for r in event_rows
                    if str(r[f"others_mean_{moment}_{tag}"]).strip() != ""]
            rec = dict(channel_type=channel_type, run=run, moment=moment,
                       who="other_ions", region=region, **describe(vals))
            rec["n_events"] = rec.pop("n_ions")
            out.append(rec)
    return out


def summarise_contacts(per_frame, channel_type, run, res_labels):
    """
    Pooled over every ion-frame in the GLU/ASN cavity:
      in_contact / no_contact / all   -> mean hydration of each group
      one row per residue             -> hydration of the ions sitting on it

    Pooling is over ION-FRAMES, not over frame means, so a frame with four
    cavity ions weighs four times a frame with one. That is the right unit:
    the question is about ions, not about frames.
    """
    out = []
    touch, free = [], []
    per_res = {lab: [] for lab in (res_labels or [])}
    for r in per_frame:
        c = r.get("contact")
        if c is None:
            continue
        # describe() keeps the raw values out of the row, so re-read them
        touch.extend(r.get("_contact_vals", []))
        free.extend(r.get("_no_contact_vals", []))
        for lab, vals in r.get("_per_res_vals", {}).items():
            per_res.setdefault(lab, []).extend(vals)
    groups = [("in_contact", "(any residue)", touch),
              ("no_contact", "(none)", free),
              ("all_cavity_ions", "(any)", touch + free)]
    for name, scope, vals in groups:
        d = describe(vals)
        out.append(dict(channel_type=channel_type, run=run,
                        moment="glu_asn_contact", who=name, region=scope,
                        n_events=d.pop("n_ions"), **d))
    for lab in (res_labels or []):
        d = describe(per_res.get(lab, []))
        out.append(dict(channel_type=channel_type, run=run,
                        moment="glu_asn_contact", who="per_residue",
                        region=lab, n_events=d.pop("n_ions"), **d))
    return out


def summarise_all_frames(per_frame, channel_type, run):
    """Pooled over every frame: mean waters per ion, per region."""
    out = []
    for region in REGIONS:
        vals = []
        for r in per_frame:
            vals.extend(r[region]["counts"])
        d = describe(vals)
        out.append(dict(channel_type=channel_type, run=run, moment="all_frames",
                        who="all_ions", region=region, n_events=d.pop("n_ions"),
                        **d))
    return out


def write_rows_csv(rows, path):
    if not rows:
        print(f"    [warn] nothing to write to {Path(path).name}")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {Path(path).name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _rolling(a, w=50):
    a = np.asarray(a, dtype=float)
    if a.size < w or w < 2:
        return a
    return np.convolve(a, np.ones(w) / w, mode="same")


def plot_timeseries(per_frame, e2, e3, out_path, title, roll=50):
    frames = np.array([r["frame"] for r in per_frame])
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(14, 9), sharex=True)
    for ax, region in zip(axes, REGIONS):
        y = np.array([r[region]["mean"] if r[region]["n_ions"] else np.nan
                      for r in per_frame], dtype=float)
        color = REGION_COLOR[region]
        ax.plot(frames, y, linewidth=0.5, alpha=0.35, color=color,
                label=f"Mean waters per ion — {region}")
        ok = ~np.isnan(y)
        if ok.sum() > roll:
            ax.plot(frames[ok], _rolling(y[ok], roll), linewidth=1.8,
                    color=color, label=f"Rolling mean ({roll} frames)")
        for i, f in enumerate(e2):
            ax.axvline(f, color="#E67E22", linestyle="--", linewidth=0.9,
                       alpha=0.6, label="end_2" if i == 0 else None)
        for i, f in enumerate(e3):
            ax.axvline(f, color=PERM_COLOR, linestyle=":", linewidth=1.1,
                       alpha=0.7, label="end_3" if i == 0 else None)
        ax.set_ylabel(f"waters/ion\n{region}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Frame", fontsize=12)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_all_frames_bars(rows, out_path, title):
    sel = [r for r in rows if r["moment"] == "all_frames"]
    if not sel:
        return
    sel = sorted(sel, key=lambda r: REGIONS.index(r["region"]))
    labels = [r["region"] for r in sel]
    means = [r["mean"] for r in sel]
    meds = [r["median"] for r in sel]
    ns = [r["n_events"] for r in sel]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.bar(x, means, width=0.55, color=[REGION_COLOR[l] for l in labels],
           alpha=0.85, edgecolor="black", linewidth=0.8)
    for xi, m, md, n in zip(x, means, meds, ns):
        ax.text(xi, m, f"mean={m:.2f}\nmedian={md:.1f}\nn={n:,} ion-frames",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean waters per ion", fontsize=11)
    ax.set_ylim(0, max(means) * 1.35 if max(means) else 1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_event_bars(summary, moment, out_path, title):
    """Permeating ion (one bar) next to the other ions in each region."""
    perm = next((r for r in summary if r["moment"] == moment
                 and r["who"] == "permeating_ion"), None)
    others = [r for r in summary if r["moment"] == moment
              and r["who"] == "other_ions"]
    if perm is None and not others:
        print(f"    [skip] no event data for {moment}")
        return
    others = sorted(others, key=lambda r: REGIONS.index(r["region"]))

    labels, means, errs, ns, colors = [], [], [], [], []
    if perm and perm["mean"] != "":
        labels.append("Permeating ion")
        means.append(perm["mean"])
        errs.append(0.0)
        ns.append(perm["n_events"])
        colors.append(PERM_COLOR)
    for r in others:
        if r["mean"] == "":
            continue
        labels.append(f"Others in\n{r['region']}")
        means.append(r["mean"])
        errs.append(0.0)
        ns.append(r["n_events"])
        colors.append(OTHER_COLOR)
    if not labels:
        return
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, means, width=0.6, color=colors, alpha=0.9,
           edgecolor="black", linewidth=0.8)
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m, f"{m:.2f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean hydration number (waters)", fontsize=11)
    ax.set_ylim(0, max(means) * 1.35)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")



def plot_overview(summary, out_path, title, contact):
    """
    Everything the module measures on one axis: baseline hydration per region,
    the residue-contact split, the eight per-residue values, and the two
    permeation moments. Sections are separated and colour-coded, and every bar
    carries its own n, because the groups differ in sample size by orders of
    magnitude (tens of thousands of ion-frames vs tens of events).
    """
    sections = []

    base = [r for r in summary if r["moment"] == "all_frames"
            and r["mean"] != ""]
    base.sort(key=lambda r: REGIONS.index(r["region"])
              if r["region"] in REGIONS else 99)
    if base:
        sections.append(("All frames\n(per region)", "#2980B9",
                         [(r["region"], r["mean"], r["n_events"]) for r in base]))

    grp = {r["who"]: r for r in summary if r["moment"] == "glu_asn_contact"
           and r["who"] in ("in_contact", "no_contact", "all_cavity_ions")
           and r["mean"] != ""}
    order = [("in_contact", f"in contact\n(≤{contact} Å)"),
             ("no_contact", "touching\nnothing"),
             ("all_cavity_ions", "all cavity\nions")]
    bars = [(lab, grp[k]["mean"], grp[k]["n_events"])
            for k, lab in order if k in grp]
    if bars:
        sections.append(("GLU/ASN cavity\nby residue contact", "#16A085", bars))

    per_res = [r for r in summary if r["moment"] == "glu_asn_contact"
               and r["who"] == "per_residue" and r["mean"] != ""]
    if per_res:
        sections.append(("Ions sitting on\neach residue", "#8E44AD",
                         [(r["region"], r["mean"], r["n_events"])
                          for r in per_res]))

    for moment, label, color in (("end2", "end_2\n(leaves cavity)", "#E67E22"),
                                 ("end3", "end_3\n(past the gate)", PERM_COLOR)):
        rows = [r for r in summary if r["moment"] == moment and r["mean"] != ""]
        if not rows:
            continue
        bars = []
        perm = next((r for r in rows if r["who"] == "permeating_ion"), None)
        if perm:
            bars.append(("permeating\nion", perm["mean"], perm["n_events"]))
        for r in sorted((q for q in rows if q["who"] == "other_ions"),
                        key=lambda q: REGIONS.index(q["region"])
                        if q["region"] in REGIONS else 99):
            bars.append((f"others in\n{r['region']}", r["mean"], r["n_events"]))
        sections.append((label, color, bars))

    if not sections:
        return
    n_bars = sum(len(b) for _, _, b in sections)
    fig, ax = plt.subplots(figsize=(max(12, 0.72 * n_bars + 3), 6.2))

    x = 0
    ticks, labels, edges, centres = [], [], [], []
    top = 0
    for name, color, bars in sections:
        start = x
        for lab, mean, n in bars:
            ax.bar(x, mean, 0.72, color=color, alpha=0.88, edgecolor="black",
                   linewidth=0.6)
            ax.text(x, mean, f"{mean:.2f}\nn={n:,}", ha="center", va="bottom",
                    fontsize=7.2)
            ticks.append(x)
            labels.append(lab)
            top = max(top, mean)
            x += 1
        centres.append(((start + x - 1) / 2, name))
        edges.append(x - 0.5)
        x += 0.8
    for e in edges[:-1]:
        ax.axvline(e, color="grey", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, top * 1.38)
    for cx, name in centres:
        ax.text(cx, top * 1.30, name, ha="center", va="center", fontsize=9.5,
                fontweight="bold")
    ax.set_ylabel("Mean waters per ion", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_contact_split(summary, out_path, title, contact):
    """
    Left: hydration of cavity ions in contact with a residue vs touching none.
    Right: one bar per residue -- the hydration of the ions sitting on it.
    """
    rows = [r for r in summary if r["moment"] == "glu_asn_contact"]
    if not rows:
        return
    grp = [r for r in rows if r["who"] in
           ("in_contact", "no_contact", "all_cavity_ions")]
    res = [r for r in rows if r["who"] == "per_residue" and r["mean"] != ""]
    if not grp:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    order = ["in_contact", "no_contact", "all_cavity_ions"]
    labels = {"in_contact": f"In contact\n(≤{contact} Å)",
              "no_contact": "Touching\nnothing",
              "all_cavity_ions": "All cavity\nions"}
    colors = {"in_contact": "#C0392B", "no_contact": "#2980B9",
              "all_cavity_ions": "#90A4AE"}
    sel = [next((r for r in grp if r["who"] == k), None) for k in order]
    sel = [r for r in sel if r and r["mean"] != ""]
    x = np.arange(len(sel))
    means = [r["mean"] for r in sel]
    ax1.bar(x, means, 0.55, color=[colors[r["who"]] for r in sel], alpha=0.9,
            edgecolor="black", linewidth=0.7)
    for xi, r in zip(x, sel):
        ax1.text(xi, r["mean"], f"{r['mean']:.2f}\nn={r['n_events']:,}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[r["who"]] for r in sel], fontsize=9)
    ax1.set_ylabel("Mean waters per ion", fontsize=10)
    ax1.set_ylim(0, max(means) * 1.3)
    ax1.set_title("GLU/ASN cavity ions", fontsize=11, fontweight="bold")

    if res:
        xr = np.arange(len(res))
        mr = [r["mean"] for r in res]
        ax2.bar(xr, mr, 0.6, color="#C0392B", alpha=0.85, edgecolor="black",
                linewidth=0.6)
        for xi, r in zip(xr, res):
            ax2.text(xi, r["mean"], f"{r['mean']:.2f}\nn={r['n_events']:,}",
                     ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(xr)
        ax2.set_xticklabels([r["region"] for r in res], rotation=45,
                            ha="right", fontsize=9)
        ax2.set_ylim(0, max(mr) * 1.3)
        ax2.set_title("Hydration of the ions sitting on each residue",
                      fontsize=11, fontweight="bold")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_event_hist(event_rows, out_path, title):
    """Distribution of the permeating ion's own hydration at end_2 and end_3."""
    series = {}
    for moment, label, color in (("end2", "at end_2 (leaves cavity)", "#E67E22"),
                                 ("end3", "at end_3 (past the gate)", PERM_COLOR)):
        vals = [float(r[f"perm_waters_{moment}"]) for r in event_rows
                if str(r[f"perm_waters_{moment}"]).strip() != ""]
        if vals:
            series[label] = (np.array(vals), color)
    if not series:
        return
    allv = np.concatenate([v for v, _ in series.values()])
    bins = np.arange(allv.min() - 0.5, allv.max() + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, (vals, color) in series.items():
        ax.hist(vals, bins=bins, alpha=0.55, color=color, edgecolor="black",
                linewidth=0.6,
                label=f"{label}  (n={vals.size}, mean={vals.mean():.2f})")
    ax.set_xlabel("Hydration number of the permeating ion", fontsize=11)
    ax.set_ylabel("Number of permeation events", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline entry point — call this from main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_ion_hydration_regions(universe, permeation_events, results_dir,
                              channel_type, glu_residues=None,
                              asn_residues=None, hbc_residues=None,
                              frame_duration=None, start_frame=0,
                              end_frame=None, radius=12.0,
                              shell=DEFAULT_SHELL, run="", force_rerun=False,
                              roll=50, dump_per_ion=False,
                              contact=DEFAULT_CONTACT, contact_split=True):
    """
    Per-ion hydration by cavity region, plus the permeating ion's hydration at
    end_2 and end_3. Returns the event summary rows, or None if skipped.

    Outputs go to <results_dir>/water_analysis/ion_hydration_regions/.
    """
    u = universe
    cfg = CHANNEL_CONFIG.get(channel_type, {})
    glu = list(glu_residues) if glu_residues else cfg.get("glu")
    asn = list(asn_residues) if asn_residues else cfg.get("asn")
    hbc = list(hbc_residues) if hbc_residues else cfg.get("hbc")
    if not (glu and asn and hbc):
        sys.exit(f"ERROR: no ring definitions for channel_type "
                 f"'{channel_type}' — pass glu/asn/hbc_residues explicitly.")
    if frame_duration is None:
        frame_duration = cfg.get("frame_duration", 0.075)

    out_dir = Path(results_dir) / WATER_ROOT / MODULE_DIR
    per_frame_csv = out_dir / "ion_hydration_per_frame.csv"
    if per_frame_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {per_frame_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ION HYDRATION BY CAVITY REGION")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  shell={shell} A  "
          f"radius={radius} A  frame_duration={frame_duration} ns")

    events = ranked_events(permeation_events)
    e2, e3 = event_frames(permeation_events)
    print(f"  {len(events)} permeation events  "
          f"({len(e2)} end_2 frames, {len(e3)} end_3 frames)")

    # frame -> permeating ions whose own hydration is needed there
    want = {}
    for rank, ion_id, ev2, ev3 in events:
        for fr in (ev2, ev3):
            if fr is not None:
                want.setdefault(fr, []).append(ion_id)

    traj_last = len(u.trajectory) - 1
    lo = max(0, int(start_frame))
    hi = traj_last if end_frame is None else min(int(end_frame), traj_last)
    if hi < lo:
        print(f"  ERROR: empty frame window ({lo}..{hi}) — skipping.")
        return None

    labels = None
    if contact_split:
        labels = residue_labels(u, list(glu) + list(asn), channel_type)

    per_frame, per_ion_rows, event_hits = scan_trajectory(
        u, glu, asn, hbc, radius, shell, lo, hi, want, dump_per_ion,
        contact=contact, labels=labels)
    res_labels = ([labels[r] for r in list(glu) + list(asn)]
                  if labels else None)

    # ---- tables ----
    write_per_frame_csv(per_frame, per_frame_csv, frame_duration, res_labels)

    by_frame = {r["frame"]: r for r in per_frame}
    event_rows = build_event_rows(events, by_frame, event_hits,
                                  channel_type, run)
    write_rows_csv(event_rows, out_dir / "ion_hydration_events.csv")

    summary = (summarise_all_frames(per_frame, channel_type, run)
               + summarise_contacts(per_frame, channel_type, run, res_labels)
               + summarise_events(event_rows, channel_type, run))
    write_rows_csv(summary, out_dir / "ion_hydration_events_summary.csv")

    if dump_per_ion and per_ion_rows:
        write_rows_csv(per_ion_rows, out_dir / "ion_hydration_per_ion.csv")

    with open(out_dir / "ion_hydration_regions.pkl", "wb") as f:
        pickle.dump(dict(script_version=SCRIPT_VERSION,
                         channel_type=channel_type, run=run,
                         shell=shell, radius=radius,
                         frame_duration=frame_duration,
                         glu=glu, asn=asn, hbc=hbc,
                         frames_scanned=(lo, hi),
                         events=events, event_rows=event_rows,
                         summary=summary, per_frame=per_frame), f)
    print(f"    Saved: ion_hydration_regions.pkl")

    # ---- plots ----
    tag = f"{channel_type}" + (f" RUN{run}" if str(run) != "" else "")
    plot_timeseries(per_frame, e2, e3,
                    out_dir / "ion_hydration_timeseries.png",
                    f"{tag} — mean waters per K⁺ ion, by cavity region "
                    f"(shell {shell:g} Å)", roll=roll)
    plot_all_frames_bars(summary, out_dir / "ion_hydration_mean_bars.png",
                         f"{tag} — mean hydration per ion, all frames")
    plot_event_bars(summary, "end2", out_dir / "ion_hydration_end2_bars.png",
                    f"{tag} — hydration at end_2 (K⁺ leaves the cavity)")
    plot_event_bars(summary, "end3", out_dir / "ion_hydration_end3_bars.png",
                    f"{tag} — hydration at end_3 (K⁺ past the gate)")
    plot_event_hist(event_rows, out_dir / "ion_hydration_event_hist.png",
                    f"{tag} — permeating ion's hydration at the two moments")
    plot_overview(summary, out_dir / "ion_hydration_overview.png",
                  f"{tag} — all hydration measures (shell {shell:g} Å)",
                  contact)
    if contact_split:
        plot_contact_split(summary,
                           out_dir / "ion_hydration_contact_split.png",
                           f"{tag} — hydration by residue contact in the "
                           f"GLU/ASN cavity", contact)

    # ---- console ----
    print(f"\n  {'moment':<11} {'who':<15} {'region':<10} {'mean':>7} "
          f"{'median':>7} {'n':>8}")
    for r in summary:
        m = r["mean"] if r["mean"] != "" else float("nan")
        md = r["median"] if r["median"] != "" else float("nan")
        print(f"  {r['moment']:<11} {r['who']:<15} {r['region']:<10} "
              f"{m:>7.2f} {md:>7.2f} {r['n_events']:>8,}")

    print(f"\n✓ Done. Outputs in {out_dir}/")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True,
                    help="SOLVATED production trajectory (not protein.nc)")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--channel_type", required=True, choices=list(CHANNEL_CONFIG))
    ap.add_argument("--run", default="")
    ap.add_argument("--shell", type=float, default=DEFAULT_SHELL,
                    help=f"K+ first-shell cutoff in A (default {DEFAULT_SHELL})")
    ap.add_argument("--radius", type=float, default=12.0,
                    help="Radial cutoff from the pore axis, A (default 12)")
    ap.add_argument("--frame_duration", type=float, default=None)
    ap.add_argument("--glu", nargs="+", type=int, default=None)
    ap.add_argument("--asn", nargs="+", type=int, default=None)
    ap.add_argument("--hbc", nargs="+", type=int, default=None)
    ap.add_argument("--roll", type=int, default=50)
    ap.add_argument("--contact", type=float, default=DEFAULT_CONTACT,
                    help=f"Ion-to-residue contact cutoff in A "
                         f"(default {DEFAULT_CONTACT})")
    ap.add_argument("--no_contact_split", action="store_true",
                    help="Skip the GLU/ASN residue-contact analysis")
    ap.add_argument("--dump_per_ion", action="store_true",
                    help="Also write every (frame, ion, region, n_waters) row")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("ERROR: MDAnalysis required — activate the Amber conda env.")

    results_dir = Path(args.results_dir)
    events, state = load_events_from_disk(results_dir)
    u = mda.Universe(args.top, args.traj)

    run_ion_hydration_regions(
        universe=u,
        permeation_events=events,
        results_dir=results_dir,
        channel_type=args.channel_type,
        glu_residues=args.glu, asn_residues=args.asn, hbc_residues=args.hbc,
        frame_duration=args.frame_duration,
        start_frame=state.get("start_frame", 0) or 0,
        end_frame=state.get("end_frame"),
        radius=args.radius, shell=args.shell,
        run=args.run, force_rerun=args.force_rerun,
        roll=args.roll, dump_per_ion=args.dump_per_ion,
        contact=args.contact, contact_split=not args.no_contact_split,
    )


if __name__ == "__main__":
    main()