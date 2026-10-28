#!/usr/bin/env python3
"""
Cavity Water Flux & Residence Analysis
======================================
Counts waters crossing each boundary of the conduction cavity (SF, ASN ring,
HBC gate) and measures how long every water stays in each cavity region.
One pass over the trajectory.

How it works
------------
Every frame, every water oxygen gets a region label using the SAME four
Channel cylinders the ion analysis uses (ch1..ch4, recomputed per frame,
first match wins, exactly like IonPermeationAnalysis._determine_ion_region):

    1 = SF (ch1)   2 = GLU/ASN (ch2)   3 = ASN/HBC (ch3)
    4 = below HBC (ch4)               0 = none of them

The cavity is regions 2 + 3. Comparing each water's label with its label in
the previous frame gives the crossings:

    sf_in      1   -> 2/3    influx from the SF
    sf_back    2/3 -> 1      backflow up into the SF
    asn_down   2   -> 3      past the ASN ring, downward
    asn_up     3   -> 2      past the ASN ring, upward
    hbc_out    2/3 -> 4      outflux through the HBC gate
    hbc_back   4   -> 2/3    re-entry from below the HBC
    side_in    0   -> 2/3    entered the cavity any other way
    side_out   2/3 -> 0      left the cavity any other way

A jump that skips regions between two saved frames counts every boundary it
passed: 1 -> 4 is sf_in + asn_down + hbc_out, 1 -> 3 is sf_in + asn_down,
2 -> 4 is asn_down + hbc_out (and the reverse for upward jumps).

Residence
---------
A visit = one unbroken run of frames a water spends inside a region. It is
tracked for four regions:

    SF        region 1
    GLU/ASN   region 2
    ASN/HBC   region 3
    GLU/HBC   region 2 or 3 (whole cavity: a 2 <-> 3 move does NOT end it)

Residence time = number of frames inside x frame_duration, so a water seen
in a single frame has residence 1 frame. A water already inside at the first
frame is treated as entering at that frame, and its visit counts like any
other. A visit still running at the last frame has no exit, so its true
length is unknown: it is saved with censored = 1 but left out of the
mean / median / histograms / boxplots.

Full SF transits
----------------
The sf_in / sf_back counts above include every water stepping in and out of
the SF cylinder's ends. A FULL TRANSIT is stricter: a water that enters the
SF cylinder at one end and leaves it at the other. Which end is decided by the
water's position along the SF axis in the frame before it entered and the
frame after it left (top half = top end).

    transit_down    entered at the top, left at the bottom
    transit_up      entered at the bottom, left at the top
    return_top      entered and left at the top    (boundary play)
    return_bottom   entered and left at the bottom (boundary play)
    jump_down/up    never seen inside: just above the SF mouth in one frame,
                    in the cavity the next (or the reverse). Counted
                    separately; at 0.4 ns/frame some real transits look
                    like this.

A water already inside the SF at the first frame has no known entry end, so
its first exit is not counted as a transit.

Caveats
-------
- No commit rule: a water flickering across a boundary is counted every
  time, and a one-frame excursion splits a visit in two. Gross crossing
  counts are inflated by that; NET flux (in - back) is not.
- Anything faster than one frame is missed. At 0.4 ns/frame (G2) a water
  can go from the cavity to past ch4 unseen and is counted as side_out, so
  G2 hbc_out is an underestimate, and residence times have a 0.4 ns
  resolution. Do not compare G12 and G2 absolute numbers.
- ch1 has a 2.5 A radius, so only waters in the single-file SF count as
  region 1.

What it produces   <results_dir>/water_analysis/cavity_water_flux/
-----------------------------------------------------------------
1. cavity_water_flux_per_frame.csv        frame, time_ns, n per region, 8 crossing counts
2. cavity_water_flux_summary.txt          crossings + residence stats
3. cavity_water_residence_visits.csv      every visit: region, water, start, end, length
4. cavity_water_residence_per_water.csv   per water and region: n visits, total, mean
5. cavity_water_flux.pkl                  all of the above, for an aggregator
6. cavity_water_flux_cumulative.png       cumulative crossings + K+ end_2/end_3 lines
7. cavity_water_flux_rates.png            forward / back / net per ns, per boundary
8. cavity_water_flux_timeseries.png       net flux per ns in time bins
9. cavity_water_residence_hist.png        histogram per region, mean + median marked
10. cavity_water_residence_box.png        boxplot per region
11. cavity_water_sf_transits.csv          every full SF transit: water, direction, frames
12. cavity_water_sf_transits.png          cumulative SF transits next to cumulative K+
                                          permeations (end_1)

How to call it from main.py
----------------------------
    from analysis.cavity_water_flux import run_cavity_water_flux

    run_cavity_water_flux(
        u=u,
        channels=(ch1, ch2, ch3, ch4),
        permeation_events=analyzer.permeation_events,
        results_dir=results_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_duration=frame_duration,
    )
"""

import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


_WATER_SEL = "resname WAT HOH SOL TIP3 and name O OW OH2"

CROSSINGS = ("sf_in", "sf_back", "asn_down", "asn_up",
             "hbc_out", "hbc_back", "side_in", "side_out")

# boundary name -> (forward crossing, backward crossing), in pore order
BOUNDARIES = {
    "SF":       ("sf_in", "sf_back"),
    "ASN ring": ("asn_down", "asn_up"),
    "HBC gate": ("hbc_out", "hbc_back"),
}

# residence region -> labels that count as inside it
RES_REGIONS = {
    "SF":      (1,),
    "GLU/ASN": (2,),
    "ASN/HBC": (3,),
    "GLU/HBC": (2, 3),
}
RES_COLORS = {"SF": "#5E35B1", "GLU/ASN": "#1E88E5", "ASN/HBC": "#FB8C00", "GLU/HBC": "#43A047"}


def label_regions(pos, channels):
    """
    Region label (0-4) for every position, vectorised.

    channels : (ch1, ch2, ch3, ch4) with geometry already computed this frame.
    Filled in reverse so the lowest-numbered matching channel wins, same as
    the ion code's first-match order.
    """
    labels = np.zeros(len(pos), dtype=np.int8)
    for num in (4, 3, 2, 1):
        ch = channels[num - 1]
        rel = pos - ch.channel_center
        axial = rel @ ch.channel_axis
        radial2 = np.einsum("ij,ij->i", rel, rel) - axial * axial
        inside = (np.abs(axial) <= ch.channel_length / 2) & (radial2 <= ch.radius ** 2)
        labels[inside] = num
    return labels


def count_crossings(prev, cur):
    """Crossing counts between two consecutive label arrays."""
    was_cav = (prev == 2) | (prev == 3)
    now_cav = (cur == 2) | (cur == 3)
    jump_down = (prev == 1) & (cur == 4)
    jump_up = (prev == 4) & (cur == 1)
    # past the ASN ring: from above it (1, 2) to below it (3, 4), or back
    above_prev = (prev == 1) | (prev == 2)
    below_prev = (prev == 3) | (prev == 4)
    above_cur = (cur == 1) | (cur == 2)
    below_cur = (cur == 3) | (cur == 4)
    return {
        "sf_in":    int(np.sum((prev == 1) & now_cav) + np.sum(jump_down)),
        "sf_back":  int(np.sum(was_cav & (cur == 1)) + np.sum(jump_up)),
        "asn_down": int(np.sum(above_prev & below_cur)),
        "asn_up":   int(np.sum(below_prev & above_cur)),
        "hbc_out":  int(np.sum(was_cav & (cur == 4)) + np.sum(jump_down)),
        "hbc_back": int(np.sum((prev == 4) & now_cav) + np.sum(jump_up)),
        "side_in":  int(np.sum((prev == 0) & now_cav)),
        "side_out": int(np.sum(was_cav & (cur == 0))),
    }


class ResidenceTracker:
    """
    Run-length tracking of visits to one region, vectorised over waters.

    Only the entry frame of each ongoing visit is kept; a visit is written
    out the frame the water leaves. Frames are absolute trajectory frames.
    """

    def __init__(self, n_waters, labels_inside):
        self.labels_inside = labels_inside
        self.entry = np.full(n_waters, -1, dtype=np.int64)  # -1 = not inside
        self.visits = []  # (water_i, start, end, censored) arrays per frame

    def inside(self, labels):
        return np.isin(labels, self.labels_inside)

    def update(self, labels, frame):
        now = self.inside(labels)
        was = self.entry >= 0
        left = np.nonzero(was & ~now)[0]
        if len(left):
            # last frame inside was the previous one
            self._record(left, frame - 1)
            self.entry[left] = -1
        # on the first frame every water inside is a new entry at that frame
        self.entry[now & ~was] = frame

    def finish(self, last_frame):
        open_ = np.nonzero(self.entry >= 0)[0]
        if len(open_):
            self._record(open_, last_frame, end_censored=True)
            self.entry[open_] = -1

    def _record(self, idx, end, end_censored=False):
        self.visits.append((idx, self.entry[idx], np.full(len(idx), end),
                            np.full(len(idx), end_censored)))

    def result(self):
        if not self.visits:
            empty = np.zeros(0, dtype=np.int64)
            return {"water_i": empty, "start": empty, "end": empty,
                    "n_frames": empty, "censored": np.zeros(0, dtype=bool)}
        w, s, e, c = (np.concatenate(x) for x in zip(*self.visits))
        order = np.lexsort((w, s))
        w, s, e, c = w[order], s[order], e[order], c[order]
        return {"water_i": w, "start": s, "end": e,
                "n_frames": e - s + 1, "censored": c}


class SFTransitTracker:
    """
    Follows waters through the SF cylinder (region 1) and classifies each
    visit by the end it came in at and the end it left by.

    Ends are read from the position along ch1's axis, which points from the
    upper gate to the lower one: negative = top end, positive = bottom end.
    """

    COUNTS = ("transit_down", "transit_up", "return_top", "return_bottom",
              "jump_down", "jump_up")

    def __init__(self, n_waters, jump_radius=5.0, jump_reach=5.0):
        self.side = np.zeros(n_waters, dtype=np.int8)   # 0 out, -1 top, 1 bottom, 2 unknown
        self.entry = np.full(n_waters, -1, dtype=np.int64)
        self.jump_r2 = jump_radius ** 2
        self.jump_reach = jump_reach
        self.prev = None          # (labels, s, r2, half) of the previous frame
        self.events = []          # (water_i, entry, last_inside, direction +1 down / -1 up)
        self.n_unknown = 0        # visits that started before the first frame
        self.n_open = 0           # visits still running at the last frame

    def _axial(self, pos, ch):
        rel = pos - ch.channel_center
        s = rel @ ch.channel_axis
        r2 = np.einsum("ij,ij->i", rel, rel) - s * s
        return s, r2, ch.channel_length / 2

    def update(self, labels, pos, ch1, frame):
        s, r2, half = self._axial(pos, ch1)
        now = labels == 1
        out = dict.fromkeys(self.COUNTS, 0)

        if self.prev is None:
            self.side[now] = 2
            self.entry[now] = frame
        else:
            p_lab, p_s, p_r2, p_half = self.prev
            was = self.side != 0

            left = np.nonzero(was & ~now)[0]
            if len(left):
                ent = self.side[left]
                ext = np.where(s[left] < 0, -1, 1).astype(np.int8)
                known = ent != 2
                self.n_unknown += int(np.sum(~known))
                through = known & (ext != ent)
                out["transit_down"] = int(np.sum(through & (ent == -1)))
                out["transit_up"] = int(np.sum(through & (ent == 1)))
                out["return_top"] = int(np.sum(known & (ext == ent) & (ent == -1)))
                out["return_bottom"] = int(np.sum(known & (ext == ent) & (ent == 1)))
                t = left[through]
                if len(t):
                    self.events.append((t, self.entry[t], np.full(len(t), frame - 1),
                                        np.where(self.side[t] == -1, 1, -1)))
                self.side[left] = 0
                self.entry[left] = -1

            came = np.nonzero(now & ~was)[0]
            self.side[came] = np.where(p_s[came] < 0, -1, 1)
            self.entry[came] = frame

            # never seen inside: just above the SF mouth <-> in the cavity
            p_mouth = ((p_lab == 0) & (p_s < -p_half) & (p_s >= -p_half - self.jump_reach)
                       & (p_r2 <= self.jump_r2))
            mouth = ((labels == 0) & (s < -half) & (s >= -half - self.jump_reach)
                     & (r2 <= self.jump_r2))
            p_cav = (p_lab == 2) | (p_lab == 3)
            cav = (labels == 2) | (labels == 3)
            out["jump_down"] = int(np.sum(p_mouth & cav))
            out["jump_up"] = int(np.sum(p_cav & mouth))

        self.prev = (labels, s, r2, half)
        return out

    def finish(self):
        self.n_open = int(np.sum(self.side != 0))

    def result(self):
        if not self.events:
            empty = np.zeros(0, dtype=np.int64)
            return {"water_i": empty, "entry": empty, "last_inside": empty,
                    "direction": empty, "n_frames": empty}
        w, e, l, d = (np.concatenate(x) for x in zip(*self.events))
        order = np.argsort(e, kind="stable")
        w, e, l, d = w[order], e[order], l[order], d[order]
        return {"water_i": w, "entry": e, "last_inside": l, "direction": d,
                "n_frames": l - e + 1}


def describe(values):
    """n, mean, median, std, min, max, q1, q3 of a 1-D array (nan if empty)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {k: (0 if k == "n" else float("nan"))
                for k in ("n", "mean", "median", "std", "min", "max", "q1", "q3")}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}


def run_cavity_water_flux(
    u,
    channels,
    permeation_events,
    results_dir,
    start_frame=0,
    end_frame=None,
    frame_duration=0.075,
    water_selection=_WATER_SEL,
    flux_bin_ns=5.0,
):
    """
    Main entry point — water crossings and residence, save results & plots.

    Parameters
    ----------
    u                 : MDAnalysis Universe
    channels          : (ch1, ch2, ch3, ch4) Channel objects, as in main.py
    permeation_events : list of dicts with 'end_2' / 'end_3' keys
    results_dir       : Path
    start_frame       : int
    end_frame         : int or None
    frame_duration    : float  ns per frame (0.075 G12, 0.4 G2)
    water_selection   : str  MDAnalysis selection for water oxygens
    flux_bin_ns       : float  bin width (ns) of the flux time-series plot
    """
    out_dir = Path(results_dir) / "water_analysis" / "cavity_water_flux"
    out_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1
    n_frames = end_frame - start_frame + 1

    print("\n" + "=" * 70)
    print("CAVITY WATER FLUX & RESIDENCE ANALYSIS")
    print("=" * 70)

    waters = u.select_atoms(water_selection)
    if len(waters) == 0:
        print(f"  [ERROR] No water atoms found with: '{water_selection}'")
        print(f"  Available resnames: {sorted(set(u.atoms.resnames))}")
        return None

    print(f"  Water O atoms in system : {len(waters)}")
    print(f"  Channel radii (ch1-ch4) : {[ch.radius for ch in channels]} Å")
    print(f"  Frames                  : {start_frame} → {end_frame}  ({n_frames} total)")

    # ── per-frame loop ─────────────────────────────────────────────────────────
    frame_indices = np.arange(start_frame, end_frame + 1, dtype=int)
    n_in = {r: np.zeros(n_frames, dtype=int) for r in RES_REGIONS}
    counts = {k: np.zeros(n_frames, dtype=int) for k in CROSSINGS}
    trackers = {r: ResidenceTracker(len(waters), labs) for r, labs in RES_REGIONS.items()}
    sf_tracker = SFTransitTracker(len(waters))
    sf_counts = {k: np.zeros(n_frames, dtype=int) for k in SFTransitTracker.COUNTS}
    prev = None

    for fi, ts in enumerate(tqdm(u.trajectory[start_frame: end_frame + 1],
                                 total=n_frames, desc="  Water flux",
                                 unit="frame", dynamic_ncols=True)):
        for num, ch in enumerate(channels, start=1):
            ch.compute_geometry(num)

        pos = waters.positions
        cur = label_regions(pos, channels)
        for k, v in sf_tracker.update(cur, pos, channels[0], ts.frame).items():
            sf_counts[k][fi] = v
        for r, tr in trackers.items():
            tr.update(cur, ts.frame)
            n_in[r][fi] = int(np.sum(tr.entry >= 0))

        # first frame has no previous frame, so its crossings stay 0;
        # waters already inside are initial residents, never counted as influx
        if prev is not None:
            for k, v in count_crossings(prev, cur).items():
                counts[k][fi] = v
        prev = cur

    for tr in trackers.values():
        tr.finish(end_frame)
    sf_tracker.finish()

    # ── crossings summary ─────────────────────────────────────────────────────
    totals = {k: int(v.sum()) for k, v in counts.items()}
    sim_ns = (n_frames - 1) * frame_duration
    n_perm = len(permeation_events)
    net = {b: totals[f] - totals[k] for b, (f, k) in BOUNDARIES.items()}

    end2 = sorted(int(ev["end_2"]) for ev in permeation_events
                  if ev.get("end_2") is not None
                  and start_frame <= int(ev["end_2"]) <= end_frame)
    end3 = sorted(int(ev["end_3"]) for ev in permeation_events
                  if ev.get("end_3") is not None
                  and start_frame <= int(ev["end_3"]) <= end_frame)

    lines = [
        "CAVITY WATER FLUX & RESIDENCE",
        f"Frames          : {start_frame} → {end_frame}  ({sim_ns:.2f} ns, {frame_duration} ns/frame)",
        f"Channel radii   : {[ch.radius for ch in channels]} Å",
        f"K+ permeation events  : {n_perm}",
        "",
        f"{'region':<10} {'initial':>8} {'mean n':>8}",
    ]
    for r in RES_REGIONS:
        lines.append(f"{r:<10} {n_in[r][0]:>8} {n_in[r].mean():>8.2f}")
    lines += [
        "",
        "CROSSINGS",
        "=" * 60,
        f"{'crossing':<10} {'total':>8} {'per ns':>10} {'per ion':>10}",
        "-" * 60,
    ]
    for k in CROSSINGS:
        per_ns = totals[k] / sim_ns if sim_ns > 0 else float("nan")
        per_ion = totals[k] / n_perm if n_perm else float("nan")
        lines.append(f"{k:<10} {totals[k]:>8} {per_ns:>10.3f} {per_ion:>10.2f}")
    lines.append("-" * 60)
    for b, (f, k) in BOUNDARIES.items():
        lines.append(f"net {b:<9} ({f} - {k}) : {net[b]}")

    # ── residence summary ─────────────────────────────────────────────────────
    visits = {r: tr.result() for r, tr in trackers.items()}
    res_stats = {}
    lines += [
        "",
        "RESIDENCE (ns; censored = still inside at the last frame, excluded)",
        "=" * 78,
        f"{'region':<10} {'n':>7} {'censored':>9} {'mean':>8} {'median':>8} "
        f"{'std':>8} {'q1':>7} {'q3':>7} {'max':>8}",
        "-" * 78,
    ]
    for r, v in visits.items():
        t = v["n_frames"][~v["censored"]] * frame_duration
        st = describe(t)
        st["n_censored"] = int(v["censored"].sum())
        res_stats[r] = st
        lines.append(f"{r:<10} {st['n']:>7} {st['n_censored']:>9} {st['mean']:>8.3f} "
                     f"{st['median']:>8.3f} {st['std']:>8.3f} {st['q1']:>7.3f} "
                     f"{st['q3']:>7.3f} {st['max']:>8.3f}")
    lines.append(f"(one frame = {frame_duration} ns, the shortest possible residence)")

    sf_tot = {k: int(v.sum()) for k, v in sf_counts.items()}
    sf_events = sf_tracker.result()
    net_transit = sf_tot["transit_down"] - sf_tot["transit_up"]
    t_ns = sf_events["n_frames"] * frame_duration
    lines += [
        "",
        "FULL SF TRANSITS (enter one end of the SF, leave by the other)",
        "=" * 60,
        f"{'':<16} {'total':>8} {'per ns':>10} {'per ion':>10}",
        "-" * 60,
    ]
    for k in SFTransitTracker.COUNTS:
        per_ns = sf_tot[k] / sim_ns if sim_ns > 0 else float("nan")
        per_ion = sf_tot[k] / n_perm if n_perm else float("nan")
        lines.append(f"{k:<16} {sf_tot[k]:>8} {per_ns:>10.4f} {per_ion:>10.2f}")
    lines += [
        "-" * 60,
        f"net transits (down - up)      : {net_transit}"
        + (f"   ({net_transit / n_perm:.2f} per K+ permeation)" if n_perm else ""),
        f"time inside per transit       : median {np.median(t_ns) if t_ns.size else float('nan'):.2f} ns, "
        f"mean {t_ns.mean() if t_ns.size else float('nan'):.2f} ns",
        f"not classified                : {sf_tracker.n_unknown} inside at the first frame, "
        f"{sf_tracker.n_open} still inside at the last",
    ]
    print("\n".join("  " + s for s in lines))

    (out_dir / "cavity_water_flux_summary.txt").write_text("\n".join(lines) + "\n")

    # ── csv ───────────────────────────────────────────────────────────────────
    with open(out_dir / "cavity_water_flux_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", *(f"n_{r}" for r in RES_REGIONS), *CROSSINGS])
        for i, fr in enumerate(frame_indices):
            w.writerow([fr, round(fr * frame_duration, 4),
                        *(n_in[r][i] for r in RES_REGIONS),
                        *(counts[k][i] for k in CROSSINGS)])

    water_index = waters.indices
    with open(out_dir / "cavity_water_residence_visits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "water_atom_index", "water_resid", "start_frame",
                    "end_frame", "n_frames", "residence_ns", "censored"])
        for r, v in visits.items():
            for wi, s, e, n, c in zip(v["water_i"], v["start"], v["end"],
                                      v["n_frames"], v["censored"]):
                w.writerow([r, water_index[wi], waters.resids[wi], s, e, n,
                            round(n * frame_duration, 4), int(c)])

    with open(out_dir / "cavity_water_residence_per_water.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "water_atom_index", "water_resid", "n_visits",
                    "n_censored", "total_ns", "mean_complete_ns", "max_ns"])
        for r, v in visits.items():
            for wi in np.unique(v["water_i"]):
                m = v["water_i"] == wi
                t = v["n_frames"][m] * frame_duration
                done = t[~v["censored"][m]]
                w.writerow([r, water_index[wi], waters.resids[wi], int(m.sum()),
                            int(v["censored"][m].sum()), round(t.sum(), 4),
                            round(done.mean(), 4) if done.size else "",
                            round(t.max(), 4)])

    with open(out_dir / "cavity_water_sf_transits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["water_atom_index", "water_resid", "direction", "entry_frame",
                    "last_frame_inside", "n_frames", "time_inside_ns"])
        for wi, e, l, d, n in zip(sf_events["water_i"], sf_events["entry"],
                                  sf_events["last_inside"], sf_events["direction"],
                                  sf_events["n_frames"]):
            w.writerow([water_index[wi], waters.resids[wi], "down" if d > 0 else "up",
                        e, l, n, round(n * frame_duration, 4)])

    pkl_data = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_duration": frame_duration,
        "frame_indices": frame_indices,
        "n_in_region": n_in,
        "counts": counts,
        "totals": totals,
        "net": net,
        "n_permeation_events": n_perm,
        "end2_frames": end2,
        "end3_frames": end3,
        "sf_transits": {"counts": sf_counts, "totals": sf_tot, "events": sf_events,
                        "net": net_transit, "n_unknown": sf_tracker.n_unknown,
                        "n_open": sf_tracker.n_open},
        "residence_visits": visits,
        "residence_stats": res_stats,
        "water_atom_indices": water_index,
        "channel_radii": [ch.radius for ch in channels],
        "water_selection": water_selection,
    }
    with open(out_dir / "cavity_water_flux.pkl", "wb") as f:
        pickle.dump(pkl_data, f)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_cumulative(frame_indices, counts, end2, end3,
                     out_dir / "cavity_water_flux_cumulative.png")
    _plot_rates(totals, sim_ns, out_dir / "cavity_water_flux_rates.png")
    end1 = sorted(int(ev["end_1"]) for ev in permeation_events
                  if ev.get("end_1") is not None
                  and start_frame <= int(ev["end_1"]) <= end_frame)
    _plot_sf_transits(frame_indices, sf_counts, end1, frame_duration,
                      out_dir / "cavity_water_sf_transits.png")
    _plot_flux_timeseries(frame_indices, counts, frame_duration, flux_bin_ns,
                          end2, end3, out_dir / "cavity_water_flux_timeseries.png")
    complete = {r: v["n_frames"][~v["censored"]] * frame_duration
                for r, v in visits.items()}
    _plot_residence_hist(complete, res_stats, frame_duration,
                         out_dir / "cavity_water_residence_hist.png")
    _plot_residence_box(complete, res_stats,
                        out_dir / "cavity_water_residence_box.png")

    print(f"  ✅ All outputs in: {out_dir}")
    return pkl_data


# ── plotting helpers ───────────────────────────────────────────────────────────

def _event_lines(ax, end2, end3, to_x=lambda f: f):
    for ef in end2:
        ax.axvline(to_x(ef), color="#E53935", linewidth=0.8, linestyle="--", alpha=0.6)
    for ef in end3:
        ax.axvline(to_x(ef), color="#8E24AA", linewidth=0.8, linestyle="--", alpha=0.6)


def _plot_cumulative(frame_indices, counts, end2, end3, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    colors = ("#1E88E5", "#FB8C00", "#43A047")
    for ax, (b, (fwd, back)), color in zip(axes, BOUNDARIES.items(), colors):
        ax.plot(frame_indices, np.cumsum(counts[fwd]), color=color,
                linewidth=1.6, label=fwd)
        ax.plot(frame_indices, np.cumsum(counts[back]), color=color,
                linewidth=1.2, linestyle=":", label=back)
        ax.plot(frame_indices, np.cumsum(counts[fwd] - counts[back]),
                color="black", linewidth=1.4, label="net")
        _event_lines(ax, end2, end3)
        ax.set_ylabel("cumulative waters", fontsize=11)
        ax.set_title(b, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")

    # dashed event lines explained once, under the plot
    axes[-1].set_xlabel("Frame   (red dashed = K⁺ end_2, purple dashed = K⁺ end_3)",
                        fontsize=11)
    axes[-1].set_xlim(frame_indices[0], frame_indices[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_sf_transits(frame_indices, sf_counts, end1, frame_duration, out_path):
    """Cumulative full SF transits next to cumulative K+ leaving the SF (end_1)."""
    t = frame_indices * frame_duration
    down = np.cumsum(sf_counts["transit_down"])
    up = np.cumsum(sf_counts["transit_up"])
    ions = np.searchsorted(np.asarray(end1), frame_indices, side="right")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, down, color="#1E88E5", linewidth=1.6, label="water transits down")
    ax.plot(t, up, color="#1E88E5", linewidth=1.2, linestyle=":", label="water transits up")
    ax.plot(t, down - up, color="black", linewidth=1.6, label="net water transits")
    ax.plot(t, ions, color="#E53935", linewidth=1.6, drawstyle="steps-post",
            label="K⁺ permeations (end_1)")
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("cumulative count", fontsize=11)
    ax.set_title("Full SF transits: water entering one end and leaving the other",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(t[0], t[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_rates(totals, sim_ns, out_path):
    """Forward / backward / net crossings per ns at each boundary."""
    if sim_ns <= 0:
        return
    names = list(BOUNDARIES)
    fwd = np.array([totals[f] for f, _ in BOUNDARIES.values()]) / sim_ns
    back = np.array([totals[k] for _, k in BOUNDARIES.values()]) / sim_ns
    x = np.arange(len(names))
    wdt = 0.27

    fig, ax = plt.subplots(figsize=(9, 5))
    for off, vals, label, color in ((-wdt, fwd, "forward (down)", "#1E88E5"),
                                    (0, back, "backward (up)", "#90CAF9"),
                                    (wdt, fwd - back, "net", "#212121")):
        bars = ax.bar(x + off, vals, wdt, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, fontsize=11)
    ax.set_ylabel("waters per ns", fontsize=11)
    ax.set_title("Water crossings per boundary", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_flux_timeseries(frame_indices, counts, frame_duration, bin_ns,
                          end2, end3, out_path):
    """Net crossings per ns in bins of bin_ns, one line per boundary."""
    per_bin = max(1, int(round(bin_ns / frame_duration)))
    n_bins = len(frame_indices) // per_bin
    if n_bins < 2:
        return
    span = n_bins * per_bin
    t_mid = (frame_indices[:span].reshape(n_bins, per_bin).mean(axis=1)
             * frame_duration)
    width_ns = per_bin * frame_duration

    fig, ax = plt.subplots(figsize=(14, 5))
    for (b, (fwd, back)), color in zip(BOUNDARIES.items(),
                                       ("#1E88E5", "#FB8C00", "#43A047")):
        net = (counts[fwd] - counts[back])[:span].reshape(n_bins, per_bin).sum(axis=1)
        ax.plot(t_mid, net / width_ns, color=color, linewidth=1.5, marker="o",
                markersize=3, label=b)
    ax.axhline(0, color="black", linewidth=0.8)
    _event_lines(ax, end2, end3, to_x=lambda f: f * frame_duration)
    ax.set_xlabel("Time (ns)   (red dashed = K⁺ end_2, purple dashed = K⁺ end_3)",
                  fontsize=11)
    ax.set_ylabel("net waters per ns (downward +)", fontsize=11)
    ax.set_title(f"Net water flux, {width_ns:g} ns bins", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_residence_hist(complete, stats, frame_duration, out_path, title=None):
    """One histogram per region; residence times are multiples of one frame."""
    ncol = 2 if len(complete) == 4 else len(complete)
    nrow = int(np.ceil(len(complete) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.5 * nrow), squeeze=False)
    for ax in axes.flat[len(complete):]:
        ax.set_visible(False)
    for ax, (r, t) in zip(axes.flat, complete.items()):
        if t.size == 0:
            ax.set_title(f"{r}  (no complete visits)", fontsize=12)
            continue
        # cap at the 99th percentile so the tail does not squash the bulk
        hi = max(np.percentile(t, 99), 2 * frame_duration)
        n_frames_hi = int(np.ceil(hi / frame_duration))
        step = max(1, n_frames_hi // 60)
        edges = (np.arange(0, n_frames_hi + step + 1, step) + 0.5) * frame_duration
        ax.hist(np.minimum(t, edges[-1] - 1e-9), bins=edges,
                color=RES_COLORS[r], alpha=0.8)
        ax.axvline(stats[r]["mean"], color="black", linewidth=1.4,
                   label=f"mean {stats[r]['mean']:.3f} ns")
        ax.axvline(stats[r]["median"], color="black", linewidth=1.4, linestyle="--",
                   label=f"median {stats[r]['median']:.3f} ns")
        ax.set_yscale("log")
        ax.set_xlabel("residence time (ns)  — last bin includes the tail", fontsize=10)
        ax.set_ylabel("visits", fontsize=10)
        ax.set_title(f"{r}  (n = {t.size})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_residence_box(complete, stats, out_path, title=None):
    names = [r for r in complete if complete[r].size]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot([complete[r] for r in names], showmeans=True, patch_artist=True,
                    flierprops={"markersize": 2, "alpha": 0.3},
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black"})
    for patch, r in zip(bp["boxes"], names):
        patch.set_facecolor(RES_COLORS[r])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(names) + 1),
                  [f"{r}\nmean {stats[r]['mean']:.3f}\nmedian {stats[r]['median']:.3f}"
                   for r in names], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("residence time (ns, log)", fontsize=11)
    ax.set_title(title or "Water residence per region (◇ = mean, line = median)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")
