#!/usr/bin/env python3
"""
cavity_water_regions.py
=======================
Counts the water molecules in each of the three cavity sub-regions, every
frame, and summarises them at the two permeation moments.

REGIONS
-------
Defined by the z of three residue rings, recomputed EVERY FRAME from the
rings' centres of mass (the protein drifts and breathes, so fixed z-planes
would slowly misreport):

    GLU/ASN   between the GLU ring and the ASN/ASP ring   (= ch2)
    ASN/HBC   between the ASN/ASP ring and the HBC gate    (= ch3)
    GLU/HBC   between the GLU ring and the HBC gate        (= ch2 + ch3)

GLU/HBC is the union of the other two, so its count is their sum by
construction — it is reported because it is the whole conduction cavity, not
because it is independent information.

A water counts if its oxygen is inside the z-slab AND within --radius of the
pore axis (default 12 A, matching the existing cavity_water_count module).
The pore axis is the x,y centre of the three rings, also per frame.

EVENT SLICES
------------
    end_2   the frame the ion leaves the GLU/ASN cavity
    end_3   the frame the ion passes the HBC gate (permeation completed)

For each, the same three regions are summarised over just those frames, so
you can see whether the cavity is wetter or drier at the moment an ion moves.

OUTPUTS   <results_dir>/water_analysis/cavity_water_regions/
----------------------------------------------------------
    cavity_water_regions_per_frame.csv   frame, time_ns, the 3 counts, ring z
    cavity_water_regions_summary.csv     region x group (all / end_2 / end_3)
    cavity_water_regions.pkl             same data, for the aggregator
    cavity_water_regions_timeseries.png  3 stacked panels vs frame + event lines
    cavity_water_regions_mean_bars.png   3 bars, all frames
    cavity_water_regions_end2_bars.png   3 bars, end_2 frames only
    cavity_water_regions_end3_bars.png   3 bars, end_3 frames only
    cavity_water_regions_comparison.png  all vs end_2 vs end_3, grouped

The parent water_analysis/ folder is shared: other water modules write their
own subfolder beside this one, so every water result for a run sits together.

USAGE — from main.py (primary)
------------------------------
    from analysis.cavity_water_regions import run_cavity_water_regions

    run_cavity_water_regions(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        radius            = 12.0,
        run               = run_number,
        force_rerun       = False,
    )

    # glu_residues / asn_residues / hbc_residues are optional — the
    # channel-specific defaults below are used when they are not passed.

USAGE — standalone
------------------
    python3 cavity_water_regions.py \
        --top  /path/com_4fs.prmtop \
        --traj /path/05_ProdN.nc \
        --results_dir /path/girk_analyser_results/G12/RUN3 \
        --channel_type G12 --run 3 --radius 12
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-06a"

# Everything water-related for a run lives under this one folder.
WATER_ROOT = "water_analysis"
MODULE_DIR = "cavity_water_regions"

# Ring definitions per channel (same numbering used by plot_pmf_with_lines.py
# and permeation_ion_table.py).
CHANNEL_CONFIG = {
    "G2": dict(glu=[98, 426, 754, 1082], asn=[130, 458, 786, 1114],
               hbc=[138, 466, 794, 1122], frame_duration=0.4),
    "G12": dict(glu=[99, 424, 748, 1073], asn=[131, 456, 780, 1105],
                hbc=[139, 464, 788, 1113], frame_duration=0.075),
    "G12_ML": dict(glu=[99, 424, 749, 1074], asn=[131, 456, 781, 1106],
                   hbc=[139, 464, 789, 1114], frame_duration=0.075),
    "G12_GAT": dict(glu=[98, 422, 747, 1073], asn=[130, 454, 779, 1105],
                    hbc=[138, 462, 787, 1113], frame_duration=0.075),
}

WATER_SELECTION = "resname WAT HOH SOL TIP3 and name O OW OH2"

REGIONS = ["GLU/ASN", "ASN/HBC", "GLU/HBC"]
REGION_KEY = {"GLU/ASN": "n_glu_asn", "ASN/HBC": "n_asn_hbc",
              "GLU/HBC": "n_glu_hbc"}
REGION_COLOR = {"GLU/ASN": "#2980B9", "ASN/HBC": "#27AE60",
                "GLU/HBC": "#8E44AD"}

GROUP_COLOR = {"all": "#90A4AE", "end_2": "#E67E22", "end_3": "#C0392B"}



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
#  Counting — pure numpy, so it can be tested without a trajectory
# ─────────────────────────────────────────────────────────────────────────────

def count_in_slab(pos, axis_xy, z_lo, z_hi, radius):
    """
    Number of positions inside the cylinder: z in [z_lo, z_hi] and radial
    distance from `axis_xy` <= radius.

    No PBC wrapping on the radial term: the cavity sits in the middle of the
    protein, far from the box edge, so the minimum image never differs. If
    the trajectory is wrapped so the channel straddles a boundary, image it
    first with cpptraj (the same requirement the other modules have).
    """
    if pos.size == 0:
        return 0
    z = pos[:, 2]
    in_z = (z >= z_lo) & (z <= z_hi)
    if not in_z.any():
        return 0
    dxy = pos[in_z, :2] - np.asarray(axis_xy, dtype=float)
    return int((np.einsum("ij,ij->i", dxy, dxy) <= radius * radius).sum())


def region_bounds(z_glu, z_asn, z_hbc):
    """
    (lo, hi) z-limits for each region, ordered so lo < hi regardless of which
    way the channel points in this topology.
    """
    return {
        "GLU/ASN": (min(z_glu, z_asn), max(z_glu, z_asn)),
        "ASN/HBC": (min(z_asn, z_hbc), max(z_asn, z_hbc)),
        "GLU/HBC": (min(z_glu, z_hbc), max(z_glu, z_hbc)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Event frames
# ─────────────────────────────────────────────────────────────────────────────

def _event_value(ev, *keys):
    for k in keys:
        if k in ev and ev[k] is not None:
            try:
                return int(ev[k])
            except (TypeError, ValueError):
                continue
    return None


def event_frames(permeation_events):
    """
    (end_2_frames, end_3_frames) as sorted lists.

    Accepts both the modern key names (end_2 / end_3) and the older ones
    (R2_end / R3_end) written by earlier versions of the analyser.
    """
    e2, e3 = [], []
    for ev in permeation_events or []:
        v2 = _event_value(ev, "end_2", "R2_end")
        v3 = _event_value(ev, "end_3", "R3_end")
        if v2 is not None:
            e2.append(v2)
        if v3 is not None:
            e3.append(v3)
    return sorted(e2), sorted(e3)


def load_events_from_disk(results_dir):
    """analyzer_state.pkl first, permeation_table.json as fallback."""
    results_dir = Path(results_dir)
    pkl = results_dir / "analyzer_state.pkl"
    if pkl.is_file():
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        events = state.get("permeation_events") or []
        if events:
            print(f"  Loaded {len(events)} events from analyzer_state.pkl")
            return events, state
    js = results_dir / "permeation_table.json"
    if js.is_file():
        with open(js) as f:
            events = json.load(f)
        print(f"  Loaded {len(events)} events from permeation_table.json")
        return events, {}
    sys.exit(f"ERROR: no analyzer_state.pkl or permeation_table.json in "
             f"{results_dir} — run main.py for this RUN first.")


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory pass
# ─────────────────────────────────────────────────────────────────────────────

def scan_trajectory(u, glu, asn, hbc, radius, lo, hi):
    """One pass. Returns a dict of per-frame arrays."""
    waters = u.select_atoms(WATER_SELECTION)
    if len(waters) == 0:
        sys.exit(f"ERROR: water selection matched 0 atoms.\n"
                 f"  tried: {WATER_SELECTION}\n"
                 f"  Check the topology's water resname/atom name — and make "
                 f"sure this is the SOLVATED trajectory, not protein.nc.")

    rings = {}
    for name, resids in (("glu", glu), ("asn", asn), ("hbc", hbc)):
        sel = "resid " + " ".join(str(r) for r in resids)
        ag = u.select_atoms(sel)
        if len(ag) == 0:
            sys.exit(f"ERROR: {name} ring {resids} matched 0 atoms — "
                     f"wrong channel_type?")
        rings[name] = ag

    print(f"  {len(waters)} water oxygens, radius {radius} A, frames {lo}..{hi}")

    out = {k: [] for k in ("frame", "n_glu_asn", "n_asn_hbc", "n_glu_hbc",
                           "z_glu", "z_asn", "z_hbc")}
    n = 0
    for ts in _progress(u.trajectory[lo:hi + 1], total=hi - lo + 1,
                        desc="cavity waters"):
        z_glu = float(rings["glu"].center_of_mass()[2])
        z_asn = float(rings["asn"].center_of_mass()[2])
        z_hbc = float(rings["hbc"].center_of_mass()[2])
        axis_xy = np.mean([rings["glu"].center_of_mass()[:2],
                           rings["asn"].center_of_mass()[:2],
                           rings["hbc"].center_of_mass()[:2]], axis=0)

        pos = waters.positions
        bounds = region_bounds(z_glu, z_asn, z_hbc)
        out["frame"].append(int(ts.frame))
        for region in REGIONS:
            z_lo, z_hi = bounds[region]
            out[REGION_KEY[region]].append(
                count_in_slab(pos, axis_xy, z_lo, z_hi, radius))
        out["z_glu"].append(z_glu)
        out["z_asn"].append(z_asn)
        out["z_hbc"].append(z_hbc)

        n += 1
    print(f"    scanned {n} frames.")
    return {k: np.array(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise(data, frame_sets, channel_type, run, frame_duration, radius):
    """
    One row per (group, region). Groups are 'all', 'end_2', 'end_3'.
    frame_sets maps group -> set of absolute frame numbers (None = all).
    """
    frames = data["frame"]
    rows = []
    for group, wanted in frame_sets.items():
        if wanted is None:
            mask = np.ones(frames.shape, dtype=bool)
        else:
            mask = np.isin(frames, list(wanted))
        for region in REGIONS:
            vals = data[REGION_KEY[region]][mask]
            if vals.size == 0:
                print(f"  [warn] {group} / {region}: no frames matched")
                continue
            rows.append(dict(
                channel_type=channel_type, run=run, group=group,
                region=region, n_frames=int(vals.size),
                mean=round(float(vals.mean()), 3),
                std=round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 3),
                sem=round(float(vals.std(ddof=1) / np.sqrt(vals.size))
                          if vals.size > 1 else 0.0, 3),
                median=round(float(np.median(vals)), 3),
                minimum=int(vals.min()), maximum=int(vals.max()),
                radius_A=radius, frame_duration_ns=frame_duration,
            ))
    return rows


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

def _rolling(a, w=50):
    if a.size < w or w < 2:
        return a.astype(float)
    k = np.ones(w) / w
    return np.convolve(a.astype(float), k, mode="same")


def plot_timeseries(data, e2, e3, out_path, title, roll=50):
    """Three stacked panels, one per region, with event lines."""
    frames = data["frame"]
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(14, 9), sharex=True)
    for ax, region in zip(axes, REGIONS):
        y = data[REGION_KEY[region]]
        color = REGION_COLOR[region]
        ax.plot(frames, y, linewidth=0.5, alpha=0.35, color=color,
                label=f"Waters in {region}")
        ax.plot(frames, _rolling(y, roll), linewidth=1.8, color=color,
                label=f"Rolling mean ({roll} frames)")
        for i, f in enumerate(e2):
            ax.axvline(f, color="#E67E22", linestyle="--", linewidth=0.9,
                       alpha=0.7, label="K⁺ exits cavity (end_2)" if i == 0 else None)
        for i, f in enumerate(e3):
            ax.axvline(f, color="#C0392B", linestyle=":", linewidth=1.1,
                       alpha=0.8, label="K⁺ past gate (end_3)" if i == 0 else None)
        ax.set_ylabel(f"# waters\n{region}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Frame", fontsize=12)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_three_bars(rows, group, out_path, title):
    """Three bars (one per region) for a single group."""
    sel = [r for r in rows if r["group"] == group]
    if not sel:
        print(f"    [skip] no rows for group '{group}'")
        return
    sel = sorted(sel, key=lambda r: REGIONS.index(r["region"]))
    labels = [r["region"] for r in sel]
    means = [r["mean"] for r in sel]
    stds = [r["std"] for r in sel]
    ns = [r["n_frames"] for r in sel]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x, means, yerr=stds, width=0.55, capsize=7,
           color=[REGION_COLOR[l] for l in labels], alpha=0.85,
           edgecolor="black", linewidth=0.8,
           error_kw={"linewidth": 1.5, "ecolor": "black"})
    top = max(m + s for m, s in zip(means, stds))
    for xi, m, s, n in zip(x, means, stds, ns):
        ax.text(xi, m + s + top * 0.03,
                f"mean={m:.2f}\nstd={s:.2f}\nn={n:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("# water molecules", fontsize=11)
    ax.set_ylim(0, top * 1.35)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_comparison(rows, out_path, title):
    """All frames vs end_2 vs end_3, grouped by region."""
    groups = [g for g in ("all", "end_2", "end_3")
              if any(r["group"] == g for r in rows)]
    if not groups:
        return
    x = np.arange(len(REGIONS))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, g in enumerate(groups):
        means, errs, ns = [], [], []
        for region in REGIONS:
            r = next((q for q in rows
                      if q["group"] == g and q["region"] == region), None)
            means.append(r["mean"] if r else 0.0)
            # SEM on the bars here: these groups have very different n
            # (thousands of frames vs tens of events), so std would make the
            # event groups look noisier than their means actually are.
            errs.append(r["sem"] if r else 0.0)
            ns.append(r["n_frames"] if r else 0)
        pos = x + i * width - 0.4 + width / 2
        ax.bar(pos, means, width * 0.9, yerr=errs, capsize=4,
               color=GROUP_COLOR.get(g, f"C{i}"), alpha=0.85,
               edgecolor="black", linewidth=0.7,
               error_kw={"linewidth": 1.2, "ecolor": "black"},
               label=f"{g}  (n={ns[0]:,} frames)")
        for p, m in zip(pos, means):
            ax.text(p, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, fontsize=11)
    ax.set_ylabel("Mean # water molecules (± SEM)", fontsize=11)
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

def run_cavity_water_regions(universe, permeation_events, results_dir,
                             channel_type, glu_residues=None,
                             asn_residues=None, hbc_residues=None,
                             frame_duration=None, start_frame=0, end_frame=None,
                             radius=12.0, run="", force_rerun=False,
                             roll=50):
    """
    Per-frame water counts in the three cavity regions, plus summaries at the
    end_2 and end_3 permeation moments. Returns the summary rows, or None if
    skipped.

    Outputs go to <results_dir>/water_analysis/cavity_water_regions/.
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
    per_frame_csv = out_dir / "cavity_water_regions_per_frame.csv"

    if per_frame_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {per_frame_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CAVITY WATER REGIONS")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  radius={radius} A  "
          f"frame_duration={frame_duration} ns")
    print(f"  GLU ring: {glu}")
    print(f"  ASN ring: {asn}")
    print(f"  HBC ring: {hbc}")

    e2, e3 = event_frames(permeation_events)
    print(f"  {len(e2)} end_2 frames, {len(e3)} end_3 frames")

    traj_last = len(u.trajectory) - 1
    lo = max(0, int(start_frame))
    hi = traj_last if end_frame is None else min(int(end_frame), traj_last)
    if hi < lo:
        print(f"  ERROR: empty frame window ({lo}..{hi}) — skipping.")
        return None

    data = scan_trajectory(u, glu, asn, hbc, radius, lo, hi)

    # ---- per-frame CSV ----
    with open(per_frame_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "n_glu_asn", "n_asn_hbc", "n_glu_hbc",
                    "z_glu", "z_asn", "z_hbc", "is_end_2", "is_end_3"])
        e2set, e3set = set(e2), set(e3)
        for i, fr in enumerate(data["frame"]):
            w.writerow([int(fr), round(float(fr) * frame_duration, 4),
                        int(data["n_glu_asn"][i]), int(data["n_asn_hbc"][i]),
                        int(data["n_glu_hbc"][i]),
                        round(float(data["z_glu"][i]), 3),
                        round(float(data["z_asn"][i]), 3),
                        round(float(data["z_hbc"][i]), 3),
                        int(fr) in e2set, int(fr) in e3set])
    print(f"    Saved: {per_frame_csv.name}  ({len(data['frame'])} rows)")

    # ---- summary ----
    frame_sets = {"all": None, "end_2": set(e2), "end_3": set(e3)}
    rows = summarise(data, frame_sets, channel_type, run, frame_duration, radius)
    write_csv(rows, out_dir / "cavity_water_regions_summary.csv")

    with open(out_dir / "cavity_water_regions.pkl", "wb") as f:
        pickle.dump(dict(script_version=SCRIPT_VERSION,
                         channel_type=channel_type, run=run, radius=radius,
                         frame_duration=frame_duration,
                         glu=glu, asn=asn, hbc=hbc,
                         frames_scanned=(lo, hi),
                         end_2_frames=e2, end_3_frames=e3,
                         data={k: v for k, v in data.items()},
                         summary=rows), f)
    print(f"    Saved: cavity_water_regions.pkl")

    # ---- plots ----
    tag = f"{channel_type}" + (f" RUN{run}" if str(run) != "" else "")
    plot_timeseries(data, e2, e3,
                    out_dir / "cavity_water_regions_timeseries.png",
                    f"{tag} — waters per cavity region (r = {radius:g} Å)",
                    roll=roll)
    plot_three_bars(rows, "all",
                    out_dir / "cavity_water_regions_mean_bars.png",
                    f"{tag} — mean waters per region, all frames")
    plot_three_bars(rows, "end_2",
                    out_dir / "cavity_water_regions_end2_bars.png",
                    f"{tag} — waters per region at end_2 "
                    f"(K⁺ leaves the cavity, n = {len(e2)})")
    plot_three_bars(rows, "end_3",
                    out_dir / "cavity_water_regions_end3_bars.png",
                    f"{tag} — waters per region at end_3 "
                    f"(K⁺ past the gate, n = {len(e3)})")
    plot_comparison(rows, out_dir / "cavity_water_regions_comparison.png",
                    f"{tag} — all frames vs permeation moments")

    # ---- console summary ----
    print(f"\n  {'group':<8} {'region':<10} {'mean':>8} {'std':>8} {'n':>8}")
    for r in rows:
        print(f"  {r['group']:<8} {r['region']:<10} {r['mean']:>8.2f} "
              f"{r['std']:>8.2f} {r['n_frames']:>8,}")

    print(f"\n✓ Done. Outputs in {out_dir}/")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True, help="Topology (com_4fs.prmtop)")
    ap.add_argument("--traj", required=True,
                    help="SOLVATED production trajectory (not protein.nc)")
    ap.add_argument("--results_dir", required=True,
                    help="RUN results dir holding analyzer_state.pkl")
    ap.add_argument("--channel_type", required=True, choices=list(CHANNEL_CONFIG))
    ap.add_argument("--run", default="")
    ap.add_argument("--radius", type=float, default=12.0,
                    help="Radial cutoff from the pore axis, A (default 12)")
    ap.add_argument("--frame_duration", type=float, default=None)
    ap.add_argument("--glu", nargs="+", type=int, default=None)
    ap.add_argument("--asn", nargs="+", type=int, default=None)
    ap.add_argument("--hbc", nargs="+", type=int, default=None)
    ap.add_argument("--roll", type=int, default=50,
                    help="Rolling-mean window for the time series (default 50)")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("ERROR: MDAnalysis required — activate the Amber conda env.")

    results_dir = Path(args.results_dir)
    events, state = load_events_from_disk(results_dir)
    u = mda.Universe(args.top, args.traj)

    run_cavity_water_regions(
        universe=u,
        permeation_events=events,
        results_dir=results_dir,
        channel_type=args.channel_type,
        glu_residues=args.glu, asn_residues=args.asn, hbc_residues=args.hbc,
        frame_duration=args.frame_duration,
        start_frame=state.get("start_frame", 0) or 0,
        end_frame=state.get("end_frame"),
        radius=args.radius,
        run=args.run,
        force_rerun=args.force_rerun,
        roll=args.roll,
    )


if __name__ == "__main__":
    main()