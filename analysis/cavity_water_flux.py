#!/usr/bin/env python3
"""
Cavity Water Flux Analysis
==========================
Counts waters entering the GLU/ASN cavity from the selectivity filter and
waters leaving the cavity through the HBC gate. One pass over the
trajectory, no per-water storage.

How it works
------------
Every frame, every water oxygen gets a region label using the SAME four
Channel cylinders the ion analysis uses (ch1..ch4, recomputed per frame,
first match wins, exactly like IonPermeationAnalysis._determine_ion_region):

    1 = SF (ch1)   2 = GLU/ASN (ch2)   3 = ASN/HBC (ch3)
    4 = below HBC (ch4)               0 = none of them

The cavity is regions 2 + 3. Comparing each water's label with its label in
the previous frame gives the crossings:

    sf_in      1   -> 2/3    influx from the SF             (asked for)
    sf_back    2/3 -> 1      backflow up into the SF
    hbc_out    2/3 -> 4      outflux through the HBC gate   (asked for)
    hbc_back   4   -> 2/3    re-entry from below the HBC
    side_in    0   -> 2/3    entered the cavity any other way
    side_out   2/3 -> 0      left the cavity any other way

A 1 -> 4 jump (straight through the cavity between two saved frames) counts
as both sf_in and hbc_out.

Caveats
-------
- No commit rule: a water flickering across a boundary is counted every
  time. Gross counts are inflated by that; NET flux (in - back) is not.
- Crossings faster than one frame are missed. At 0.4 ns/frame (G2) a water
  can go from the cavity to past ch4 unseen and is counted as side_out, so
  G2 hbc_out is an underestimate. Do not compare G12 and G2 absolute rates.
- ch1 has a 2.5 A radius, so only waters in the single-file SF count as
  region 1.

What it produces   <results_dir>/water_analysis/cavity_water_flux/
-----------------------------------------------------------------
1. cavity_water_flux_per_frame.csv   frame, time_ns, n_cavity, 6 crossing counts
2. cavity_water_flux_summary.txt     totals, net flux, rates per ns, per ion
3. cavity_water_flux.pkl             same data, for an aggregator
4. cavity_water_flux_cumulative.png  cumulative crossings + K+ end_2/end_3 lines

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

CROSSINGS = ("sf_in", "sf_back", "hbc_out", "hbc_back", "side_in", "side_out")


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
    jump = (prev == 1) & (cur == 4)
    return {
        "sf_in":    int(np.sum((prev == 1) & now_cav) + np.sum(jump)),
        "sf_back":  int(np.sum(was_cav & (cur == 1))),
        "hbc_out":  int(np.sum(was_cav & (cur == 4)) + np.sum(jump)),
        "hbc_back": int(np.sum((prev == 4) & now_cav)),
        "side_in":  int(np.sum((prev == 0) & now_cav)),
        "side_out": int(np.sum(was_cav & (cur == 0))),
    }


def run_cavity_water_flux(
    u,
    channels,
    permeation_events,
    results_dir,
    start_frame=0,
    end_frame=None,
    frame_duration=0.075,
    water_selection=_WATER_SEL,
):
    """
    Main entry point — count water crossings per frame, save results & plots.

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
    """
    out_dir = Path(results_dir) / "water_analysis" / "cavity_water_flux"
    out_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1
    n_frames = end_frame - start_frame + 1

    print("\n" + "=" * 70)
    print("CAVITY WATER FLUX ANALYSIS")
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
    n_cavity = np.zeros(n_frames, dtype=int)
    counts = {k: np.zeros(n_frames, dtype=int) for k in CROSSINGS}
    prev = None

    for fi, ts in enumerate(tqdm(u.trajectory[start_frame: end_frame + 1],
                                 total=n_frames, desc="  Water flux",
                                 unit="frame", dynamic_ncols=True)):
        for num, ch in enumerate(channels, start=1):
            ch.compute_geometry(num)

        cur = label_regions(waters.positions, channels)
        n_cavity[fi] = int(np.sum((cur == 2) | (cur == 3)))

        # first frame has no previous frame, so its crossings stay 0;
        # waters already inside are initial residents, never counted as influx
        if prev is not None:
            for k, v in count_crossings(prev, cur).items():
                counts[k][fi] = v
        prev = cur

    # ── summary ───────────────────────────────────────────────────────────────
    totals = {k: int(v.sum()) for k, v in counts.items()}
    sim_ns = (n_frames - 1) * frame_duration
    n_perm = len(permeation_events)
    net_sf = totals["sf_in"] - totals["sf_back"]
    net_hbc = totals["hbc_out"] - totals["hbc_back"]

    end2 = sorted(int(ev["end_2"]) for ev in permeation_events
                  if ev.get("end_2") is not None
                  and start_frame <= int(ev["end_2"]) <= end_frame)
    end3 = sorted(int(ev["end_3"]) for ev in permeation_events
                  if ev.get("end_3") is not None
                  and start_frame <= int(ev["end_3"]) <= end_frame)

    lines = [
        "CAVITY WATER FLUX",
        f"Frames          : {start_frame} → {end_frame}  ({sim_ns:.2f} ns, {frame_duration} ns/frame)",
        f"Channel radii   : {[ch.radius for ch in channels]} Å",
        f"Initial residents in cavity (frame {start_frame}): {n_cavity[0]}",
        f"Mean waters in cavity : {n_cavity.mean():.2f}",
        f"K+ permeation events  : {n_perm}",
        "=" * 60,
        f"{'crossing':<10} {'total':>8} {'per ns':>10} {'per ion':>10}",
        "-" * 60,
    ]
    for k in CROSSINGS:
        per_ns = totals[k] / sim_ns if sim_ns > 0 else float("nan")
        per_ion = totals[k] / n_perm if n_perm else float("nan")
        lines.append(f"{k:<10} {totals[k]:>8} {per_ns:>10.3f} {per_ion:>10.2f}")
    lines += [
        "-" * 60,
        f"net SF influx  (sf_in - sf_back)    : {net_sf}",
        f"net HBC efflux (hbc_out - hbc_back) : {net_hbc}",
    ]
    print("\n".join("  " + s for s in lines))

    txt_path = out_dir / "cavity_water_flux_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n")

    csv_path = out_dir / "cavity_water_flux_per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "n_cavity", *CROSSINGS])
        for i, fr in enumerate(frame_indices):
            w.writerow([fr, round(fr * frame_duration, 4), n_cavity[i],
                        *(counts[k][i] for k in CROSSINGS)])

    pkl_data = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_duration": frame_duration,
        "frame_indices": frame_indices,
        "n_cavity": n_cavity,
        "counts": counts,
        "totals": totals,
        "net_sf": net_sf,
        "net_hbc": net_hbc,
        "n_permeation_events": n_perm,
        "end2_frames": end2,
        "end3_frames": end3,
        "channel_radii": [ch.radius for ch in channels],
        "water_selection": water_selection,
    }
    pkl_path = out_dir / "cavity_water_flux.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)

    _plot_cumulative(frame_indices, counts, end2, end3,
                     out_dir / "cavity_water_flux_cumulative.png")

    print(f"  ✅ All outputs in: {out_dir}")
    return pkl_data


# ── plotting helper ────────────────────────────────────────────────────────────

def _plot_cumulative(frame_indices, counts, end2, end3, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    panels = (
        (axes[0], "SF → cavity", ("sf_in", "sf_back"), "#1E88E5"),
        (axes[1], "cavity → HBC", ("hbc_out", "hbc_back"), "#43A047"),
    )
    for ax, title, (fwd, back), color in panels:
        ax.plot(frame_indices, np.cumsum(counts[fwd]), color=color,
                linewidth=1.6, label=fwd)
        ax.plot(frame_indices, np.cumsum(counts[back]), color=color,
                linewidth=1.2, linestyle=":", label=back)
        ax.plot(frame_indices, np.cumsum(counts[fwd] - counts[back]),
                color="black", linewidth=1.4, label="net")
        for ef in end2:
            ax.axvline(ef, color="#E53935", linewidth=0.8, linestyle="--", alpha=0.6)
        for ef in end3:
            ax.axvline(ef, color="#8E24AA", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel("cumulative waters", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")

    # dashed event lines explained once, under the plot
    axes[1].set_xlabel("Frame   (red dashed = K⁺ end_2, purple dashed = K⁺ end_3)",
                       fontsize=11)
    axes[1].set_xlim(frame_indices[0], frame_indices[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")
