#!/usr/bin/env python3
"""
cavity_mobility.py
==================
How much do ions and waters actually move inside the cavity?

For every frame a particle spends in the cavity, the distance it travelled
since the previous frame is recorded. Each particle ends up with a set of
step lengths; its mean step is one number, and the distribution of those
per-particle means is what the plots show.

    IONS    tracked from end_1 (leaves the selectivity filter) to end_3
            (passes the HBC gate) -- the whole GLU/HBC transit, one series
            per permeation event.
    WATERS  tracked for as long as they stay inside GLU/HBC. A water that
            leaves and comes back is two separate visits, because the gap
            in between is not cavity motion.

READ THIS BEFORE INTERPRETING THE NUMBERS
-----------------------------------------
A step length is only meaningful relative to the frame spacing. Free water at
310 K diffuses roughly sqrt(6*D*dt): about 10 A in 75 ps and 24 A in 400 ps.
Both are larger than the cavity itself, so an unbound water's "step" is not a
diffusion measurement -- it saturates at roughly the distance across the
accessible space.

That makes this a CONFINEMENT measure, not a diffusion coefficient. A small
mean step means the particle was held; a large one means it was free to move
as far as the geometry allows. Comparisons are valid between particles in the
same run, and between runs of the same frame spacing. They are NOT valid
between G12 (0.075 ns/frame) and G2 (0.4 ns/frame) -- a step per frame and a
step per ns are both reported, but neither makes those two comparable, since
the saturation differs. Compare channels only by re-running one of them with
matched output spacing.

Displacements use the minimum image convention, so a particle crossing the
periodic boundary does not register a box-sized jump.

OUTPUTS   <results_dir>/water_analysis/cavity_mobility/
--------------------------------------------------------
    ion_mobility_per_ion.csv       one row per permeation event
    ion_mobility_steps.csv         every individual step (--dump_steps)
    water_mobility_per_visit.csv   one row per water visit to the cavity
    water_mobility_summary.csv     the mean of means, ions and waters
    cavity_mobility.pkl            everything, for the aggregator
    ion_mobility_box.png           boxplot + histogram of per-ion mean steps
    water_mobility_box.png         the same for water visits
    cavity_mobility_overview.png   ions and waters side by side
    water_residence_hist.png       how long waters stay in the cavity

USAGE - from main.py
--------------------
    from analysis.cavity_mobility import run_cavity_mobility

    run_cavity_mobility(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        run               = run_number,
        force_rerun       = False,
    )

USAGE - standalone
------------------
    python3 cavity_mobility.py \
        --top  /path/com_4fs.prmtop \
        --traj /path/05_ProdN.nc \
        --results_dir /path/girk_analyser_results/G12/RUN3 \
        --channel_type G12 --run 3
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

try:
    from analysis.cavity_water_regions import (
        CHANNEL_CONFIG, WATER_SELECTION, WATER_ROOT,
        count_in_slab, region_bounds, load_events_from_disk, _progress,
    )
except ImportError:
    try:
        from cavity_water_regions import (
            CHANNEL_CONFIG, WATER_SELECTION, WATER_ROOT,
            count_in_slab, region_bounds, load_events_from_disk, _progress,
        )
    except ImportError:
        sys.exit("ERROR: cavity_water_regions.py must sit next to this file "
                 "(or in analysis/) — the cavity definitions are shared.")

SCRIPT_VERSION = "2026-10-06a"
MODULE_DIR = "cavity_mobility"

ION_SELECTION = "resname K+ K"
TRACK_REGION = "GLU/HBC"      # the whole cavity, GLU ring down to the gate

ION_COLOR = "#C0392B"
WATER_COLOR = "#2980B9"


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────────────

def minimum_image(delta, box):
    """
    Wrap a displacement into the primary cell, so a particle crossing the
    boundary registers its real step and not a box-length jump. Handles the
    orthorhombic case; a triclinic box falls back to the raw delta with a
    warning from the caller.
    """
    if box is None:
        return delta
    L = np.asarray(box[:3], dtype=float)
    if np.any(L <= 0):
        return delta
    return delta - L * np.round(delta / L)


def inside_cavity(pos, axis_xy, z_lo, z_hi, radius):
    """Boolean mask: inside the cavity cylinder."""
    if pos.size == 0:
        return np.zeros(0, dtype=bool)
    z = pos[:, 2]
    ok = (z >= z_lo) & (z <= z_hi)
    if not ok.any():
        return ok
    dxy = pos[:, :2] - np.asarray(axis_xy, dtype=float)
    r2 = np.einsum("ij,ij->i", dxy, dxy)
    return ok & (r2 <= radius * radius)


def describe_steps(steps, frame_duration):
    """mean / median / sd / max / total path for one particle's step list."""
    a = np.asarray(steps, dtype=float)
    if a.size == 0:
        return dict(n_steps=0, mean_step_A="", median_step_A="",
                    sd_step_A="", max_step_A="", total_path_A="",
                    mean_step_A_per_ns="")
    return dict(
        n_steps=int(a.size),
        mean_step_A=round(float(a.mean()), 4),
        median_step_A=round(float(np.median(a)), 4),
        sd_step_A=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        max_step_A=round(float(a.max()), 4),
        total_path_A=round(float(a.sum()), 3),
        mean_step_A_per_ns=round(float(a.mean() / frame_duration), 4)
        if frame_duration else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────────────────────────────────────

def ion_windows(permeation_events):
    """
    [(rank, ion_id, end_1, end_3)] sorted by end_3, matching the ion_rank in
    permeation_ion_table.py. Events with no usable entry frame are dropped,
    because a transit with an unknown start is not a transit.
    """
    out = []
    for ev in permeation_events or []:
        ion = ev.get("ion_id")
        if ion is None:
            continue

        def g(*keys):
            for k in keys:
                if k in ev and ev[k] is not None:
                    try:
                        return int(ev[k])
                    except (TypeError, ValueError):
                        pass
            return None

        e1 = g("end_1", "R1_end", "start_2", "start_1")
        e3 = g("end_3", "R3_end")
        if e1 is None or e3 is None or e3 <= e1:
            continue
        out.append([None, int(ion), e1, e3])
    out.sort(key=lambda t: t[3])
    for i, rec in enumerate(out, start=1):
        rec[0] = i
    return [tuple(r) for r in out]


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def scan(u, glu, asn, hbc, radius, lo, hi, windows, track_waters=True):
    """
    One pass. Returns (ion_steps, water_visits, n_frames).

      ion_steps    {rank: [step lengths]} for frames inside end_1..end_3
      water_visits [dict(water_id, first_frame, last_frame, steps=[...])]
    """
    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        sys.exit(f"ERROR: ion selection '{ION_SELECTION}' matched 0 atoms.")
    ion_resids = np.array([a.resid for a in ions])
    idx_of_resid = {int(r): i for i, r in enumerate(ion_resids)}

    waters = u.select_atoms(WATER_SELECTION) if track_waters else None
    if track_waters and len(waters) == 0:
        sys.exit(f"ERROR: water selection matched 0 atoms.\n"
                 f"  tried: {WATER_SELECTION}\n"
                 f"  Is this the SOLVATED trajectory, not protein.nc?")
    water_ids = (np.array([a.resid for a in waters]) if track_waters
                 else np.zeros(0, dtype=int))

    rings = {}
    for name, resids in (("glu", glu), ("asn", asn), ("hbc", hbc)):
        ag = u.select_atoms("resid " + " ".join(str(r) for r in resids))
        if len(ag) == 0:
            sys.exit(f"ERROR: {name} ring {resids} matched 0 atoms — "
                     f"wrong channel_type?")
        rings[name] = ag

    # frame -> ranks whose window covers it
    active = {}
    for rank, ion_id, e1, e3 in windows:
        for f in range(max(e1, lo), min(e3, hi) + 1):
            active.setdefault(f, []).append((rank, ion_id))

    print(f"  {len(ions)} ions, "
          f"{len(waters) if track_waters else 0} water oxygens, "
          f"{len(windows)} ion windows, frames {lo}..{hi}")

    ion_steps = {rank: [] for rank, _, _, _ in windows}
    prev_ion_pos = None          # positions at the previous frame
    prev_water_pos = None
    open_visits = {}             # water_id -> dict(first_frame, steps)
    water_visits = []
    n = 0
    warned_triclinic = False

    for ts in _progress(u.trajectory[lo:hi + 1], total=hi - lo + 1,
                        desc="cavity mobility"):
        frame = int(ts.frame)
        box = ts.dimensions
        if (box is not None and len(box) >= 6
                and not np.allclose(box[3:6], 90.0) and not warned_triclinic):
            print("  [warn] non-orthorhombic box — minimum-image wrapping is "
                  "approximate; steps near the boundary may be overestimated.")
            warned_triclinic = True

        z_glu = float(rings["glu"].center_of_mass()[2])
        z_asn = float(rings["asn"].center_of_mass()[2])
        z_hbc = float(rings["hbc"].center_of_mass()[2])
        axis_xy = np.mean([rings["glu"].center_of_mass()[:2],
                           rings["asn"].center_of_mass()[:2],
                           rings["hbc"].center_of_mass()[:2]], axis=0)
        z_lo, z_hi = region_bounds(z_glu, z_asn, z_hbc)[TRACK_REGION]

        ion_pos = ions.positions.copy()

        # ---- ions: step only if the previous frame was also in the window ----
        if prev_ion_pos is not None:
            for rank, ion_id in active.get(frame, []):
                if frame - 1 not in active:
                    continue
                if (rank, ion_id) not in active.get(frame - 1, []):
                    continue
                i = idx_of_resid.get(int(ion_id))
                if i is None:
                    continue
                d = minimum_image(ion_pos[i] - prev_ion_pos[i], box)
                ion_steps[rank].append(float(np.linalg.norm(d)))
        prev_ion_pos = ion_pos

        # ---- waters: contiguous visits to the cavity ----
        if track_waters:
            wpos = waters.positions.copy()
            mask = inside_cavity(wpos, axis_xy, z_lo, z_hi, radius)
            here = set(int(w) for w in water_ids[mask])

            for wid in list(open_visits):
                if wid not in here:
                    v = open_visits.pop(wid)
                    v["last_frame"] = frame - 1
                    water_visits.append(v)

            if prev_water_pos is not None:
                pos_of = {int(w): k for k, w in enumerate(water_ids)}
                for wid in here:
                    k = pos_of.get(wid)
                    if k is None:
                        continue
                    if wid in open_visits:
                        d = minimum_image(wpos[k] - prev_water_pos[k], box)
                        open_visits[wid]["steps"].append(
                            float(np.linalg.norm(d)))
                    else:
                        open_visits[wid] = dict(water_id=wid,
                                                first_frame=frame, steps=[])
            else:
                for wid in here:
                    open_visits[wid] = dict(water_id=wid, first_frame=frame,
                                            steps=[])
            prev_water_pos = wpos

        n += 1

    for wid, v in open_visits.items():
        v["last_frame"] = hi
        water_visits.append(v)

    print(f"    scanned {n} frames, "
          f"{sum(len(v) for v in ion_steps.values())} ion steps, "
          f"{len(water_visits)} water visits.")
    return ion_steps, water_visits, n


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def ion_rows(windows, ion_steps, channel_type, run, frame_duration):
    rows = []
    for rank, ion_id, e1, e3 in windows:
        steps = ion_steps.get(rank, [])
        rows.append(dict(channel_type=channel_type, run=run, ion_rank=rank,
                         ion_id=ion_id, end_1=e1, end_3=e3,
                         cavity_frames=e3 - e1,
                         cavity_ns=round((e3 - e1) * frame_duration, 4),
                         **describe_steps(steps, frame_duration)))
    return rows


def water_rows(visits, channel_type, run, frame_duration, min_steps=1):
    rows = []
    for i, v in enumerate(visits, start=1):
        if len(v["steps"]) < min_steps:
            continue
        n_frames = v["last_frame"] - v["first_frame"] + 1
        rows.append(dict(channel_type=channel_type, run=run, visit=i,
                         water_id=v["water_id"], first_frame=v["first_frame"],
                         last_frame=v["last_frame"],
                         residence_frames=n_frames,
                         residence_ns=round(n_frames * frame_duration, 4),
                         **describe_steps(v["steps"], frame_duration)))
    return rows


def mean_of_means(rows, key="mean_step_A"):
    vals = [float(r[key]) for r in rows
            if str(r.get(key, "")).strip() != ""]
    if not vals:
        return None
    a = np.array(vals)
    return dict(n_particles=int(a.size),
                mean_of_means=round(float(a.mean()), 4),
                median_of_means=round(float(np.median(a)), 4),
                sd_of_means=round(float(a.std(ddof=1))
                                  if a.size > 1 else 0.0, 4),
                min_of_means=round(float(a.min()), 4),
                max_of_means=round(float(a.max()), 4))


def write_csv(rows, path):
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

def _vals(rows, key="mean_step_A"):
    return np.array([float(r[key]) for r in rows
                     if str(r.get(key, "")).strip() != ""], dtype=float)


def plot_box_hist(rows, out_path, title, color, what, frame_duration):
    vals = _vals(rows)
    if vals.size == 0:
        print(f"    [skip] no values for {what}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    bp = ax1.boxplot(vals, widths=0.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.65)
    rng = np.random.default_rng(0)
    ax1.scatter(1 + rng.uniform(-0.12, 0.12, vals.size), vals, s=16,
                color="#222222", alpha=0.45, zorder=3)
    ax1.axhline(vals.mean(), color=color, linestyle="--", linewidth=1.6,
                label=f"mean of means = {vals.mean():.3f} Å")
    ax1.set_xticks([])
    ax1.set_ylabel("Mean step per frame (Å)", fontsize=10)
    ax1.set_title(f"{what}  (n={vals.size})", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.hist(vals, bins=min(40, max(8, vals.size // 3)), color=color,
             alpha=0.75, edgecolor="black", linewidth=0.5)
    ax2.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.6,
                label=f"mean {vals.mean():.3f} Å")
    ax2.axvline(np.median(vals), color="black", linestyle=":", linewidth=1.6,
                label=f"median {np.median(vals):.3f} Å")
    ax2.set_xlabel("Mean step per frame (Å)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_overview(ion_rows_, water_rows_, out_path, title, frame_duration):
    iv, wv = _vals(ion_rows_), _vals(water_rows_)
    data = [(d, lab, c) for d, lab, c in
            ((iv, f"Ions\n(n={iv.size})", ION_COLOR),
             (wv, f"Waters\n(n={wv.size})", WATER_COLOR)) if d.size]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bp = ax.boxplot([d for d, _, _ in data], widths=0.5, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for box, (_, _, c) in zip(bp["boxes"], data):
        box.set_facecolor(c)
        box.set_alpha(0.65)
    rng = np.random.default_rng(0)
    for i, (d, _, c) in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.11, 0.11, d.size), d, s=12,
                   color="#222222", alpha=0.35, zorder=3)
        ax.text(i, d.max(), f"mean {d.mean():.2f} Å", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([lab for _, lab, _ in data], fontsize=10)
    ax.set_ylabel("Mean step per frame (Å)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_residence(rows, out_path, title, frame_duration):
    vals = np.array([float(r["residence_ns"]) for r in rows
                     if str(r.get("residence_ns", "")).strip() != ""])
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(vals, bins=min(60, max(10, vals.size // 20)), color=WATER_COLOR,
            alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean {vals.mean():.3f} ns")
    ax.axvline(np.median(vals), color="black", linestyle=":", linewidth=1.5,
               label=f"median {np.median(vals):.3f} ns")
    ax.set_xlabel("Time a water stays in the cavity (ns)", fontsize=11)
    ax.set_ylabel("Number of visits (log)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cavity_mobility(universe, permeation_events, results_dir, channel_type,
                        glu_residues=None, asn_residues=None,
                        hbc_residues=None, frame_duration=None, start_frame=0,
                        end_frame=None, radius=12.0, run="",
                        force_rerun=False, track_waters=True,
                        min_water_steps=1, dump_steps=False):
    """
    Per-frame displacement of ions (end_1 to end_3) and waters (while in the
    GLU/HBC cavity). Returns the summary rows, or None if skipped.
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
    ion_csv = out_dir / "ion_mobility_per_ion.csv"
    if ion_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {ion_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CAVITY MOBILITY")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  radius={radius} A  "
          f"frame_duration={frame_duration} ns")

    windows = ion_windows(permeation_events)
    print(f"  {len(windows)} permeation events with a usable end_1..end_3 window")
    if not windows:
        print("  No usable ion windows — nothing to do.")
        return None

    traj_last = len(u.trajectory) - 1
    lo = max(0, int(start_frame))
    hi = traj_last if end_frame is None else min(int(end_frame), traj_last)
    if hi < lo:
        print(f"  ERROR: empty frame window ({lo}..{hi}) — skipping.")
        return None

    ion_steps, visits, n_frames = scan(u, glu, asn, hbc, radius, lo, hi,
                                       windows, track_waters)

    irows = ion_rows(windows, ion_steps, channel_type, run, frame_duration)
    wrows = water_rows(visits, channel_type, run, frame_duration,
                       min_water_steps)
    write_csv(irows, ion_csv)
    if wrows:
        write_csv(wrows, out_dir / "water_mobility_per_visit.csv")

    if dump_steps:
        steps_rows = [dict(ion_rank=rank, step_index=i + 1, step_A=round(s, 4))
                      for rank, lst in ion_steps.items()
                      for i, s in enumerate(lst)]
        write_csv(steps_rows, out_dir / "ion_mobility_steps.csv")

    summary = []
    for label, rows in (("ions", irows), ("waters", wrows)):
        m = mean_of_means(rows)
        if m is None:
            continue
        summary.append(dict(channel_type=channel_type, run=run, species=label,
                            frame_duration_ns=frame_duration,
                            radius_A=radius, region=TRACK_REGION, **m,
                            mean_of_means_per_ns=round(
                                m["mean_of_means"] / frame_duration, 4)
                            if frame_duration else ""))
    if wrows:
        res = np.array([float(r["residence_ns"]) for r in wrows])
        summary.append(dict(channel_type=channel_type, run=run,
                            species="water_residence", region=TRACK_REGION,
                            n_particles=int(res.size),
                            mean_of_means=round(float(res.mean()), 4),
                            median_of_means=round(float(np.median(res)), 4),
                            sd_of_means=round(float(res.std(ddof=1))
                                              if res.size > 1 else 0.0, 4),
                            min_of_means=round(float(res.min()), 4),
                            max_of_means=round(float(res.max()), 4),
                            note="values are ns in the cavity, not Angstroms"))
    write_csv(summary, out_dir / "water_mobility_summary.csv")

    with open(out_dir / "cavity_mobility.pkl", "wb") as f:
        pickle.dump(dict(script_version=SCRIPT_VERSION,
                         channel_type=channel_type, run=run,
                         frame_duration=frame_duration, radius=radius,
                         region=TRACK_REGION, frames_scanned=(lo, hi),
                         ion_rows=irows, water_rows=wrows,
                         summary=summary), f)
    print(f"    Saved: cavity_mobility.pkl")

    tag = f"{channel_type}" + (f" RUN{run}" if str(run) != "" else "")
    plot_box_hist(irows, out_dir / "ion_mobility_box.png",
                  f"{tag} — how far each permeating K⁺ moves per frame "
                  f"({frame_duration:g} ns)", ION_COLOR, "Ions",
                  frame_duration)
    if wrows:
        plot_box_hist(wrows, out_dir / "water_mobility_box.png",
                      f"{tag} — how far each cavity water moves per frame "
                      f"({frame_duration:g} ns)", WATER_COLOR, "Water visits",
                      frame_duration)
        plot_residence(wrows, out_dir / "water_residence_hist.png",
                       f"{tag} — water residence in the {TRACK_REGION} cavity",
                       frame_duration)
    plot_overview(irows, wrows, out_dir / "cavity_mobility_overview.png",
                  f"{tag} — mobility in the {TRACK_REGION} cavity",
                  frame_duration)

    print(f"\n  {'species':<18}{'n':>8}{'mean of means':>16}{'median':>10}"
          f"{'sd':>9}")
    for r in summary:
        unit = "ns" if r["species"] == "water_residence" else "Å"
        print(f"  {r['species']:<18}{r['n_particles']:>8,}"
              f"{r['mean_of_means']:>14.3f} {unit}{r['median_of_means']:>10.3f}"
              f"{r['sd_of_means']:>9.3f}")
    print(f"\n  Reminder: a step at {frame_duration:g} ns spacing saturates at "
          f"roughly the cavity width for an unbound particle — read these as "
          f"confinement, not diffusion.")
    print(f"\n✓ Done. Outputs in {out_dir}/")
    return summary


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
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--frame_duration", type=float, default=None)
    ap.add_argument("--glu", nargs="+", type=int, default=None)
    ap.add_argument("--asn", nargs="+", type=int, default=None)
    ap.add_argument("--hbc", nargs="+", type=int, default=None)
    ap.add_argument("--no_waters", action="store_true",
                    help="Track only the ions (much faster)")
    ap.add_argument("--min_water_steps", type=int, default=1,
                    help="Drop water visits with fewer steps than this "
                         "(default 1; raise it to ignore brief dips in)")
    ap.add_argument("--dump_steps", action="store_true",
                    help="Also write every individual ion step")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("ERROR: MDAnalysis required — activate the Amber conda env.")

    results_dir = Path(args.results_dir)
    events, state = load_events_from_disk(results_dir)
    u = mda.Universe(args.top, args.traj)

    run_cavity_mobility(
        universe=u, permeation_events=events, results_dir=results_dir,
        channel_type=args.channel_type, glu_residues=args.glu,
        asn_residues=args.asn, hbc_residues=args.hbc,
        frame_duration=args.frame_duration,
        start_frame=state.get("start_frame", 0) or 0,
        end_frame=state.get("end_frame"), radius=args.radius, run=args.run,
        force_rerun=args.force_rerun, track_waters=not args.no_waters,
        min_water_steps=args.min_water_steps, dump_steps=args.dump_steps,
    )


if __name__ == "__main__":
    main()