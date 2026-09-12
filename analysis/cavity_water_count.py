#!/usr/bin/env python3
"""
Cavity Water Count Analysis
============================
Counts the number of water molecules inside ch2 (the Glu/Asn cavity)
at every frame of the trajectory.

The cylinder geometry (center, axis, length) is taken from ch2 and updated
every frame. The radius used for water counting can be set independently
via `water_radius` (default 12 Å) so the ch2 object itself is never modified.

What it produces
----------------
1. cavity_water_count.pkl               — raw per-frame water counts
2. cavity_water_count.txt               — plain-text table: frame | count
3. cavity_water_count.png               — water count vs frame (clean)
4. cavity_water_count_with_events.png   — same + red dashed lines at end_2

How to call it from main.py
----------------------------
    from analysis.cavity_water_count import run_cavity_water_count

    run_cavity_water_count(
        u=u,
        channel2=ch2,
        permeation_events=analyzer.permeation_events,
        results_dir=results_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        water_radius=12.0,   # override radius for water counting only
    )
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm


_WATER_SEL = "resname WAT SOL TIP3 and name O OW"


def _count_waters_in_cylinder(water_positions, center, axis, half_length, radius):
    """
    Vectorised cylinder membership test — no Python loop over atoms.

    Parameters
    ----------
    water_positions : (N, 3) float array  — positions of all water O atoms
    center          : (3,)   float array  — cylinder centre point
    axis            : (3,)   float array  — unit vector along cylinder axis
    half_length     : float               — half the cylinder length (Å)
    radius          : float               — cylinder radius to use (Å)
                      NOTE: this is passed explicitly and does NOT have to
                      match channel2.radius — so ch2 is never modified.

    Returns
    -------
    int — number of water O atoms inside the cylinder
    """
    rel        = water_positions - center          # (N, 3)
    axial      = rel @ axis                        # (N,)
    radial_vec = rel - axial[:, np.newaxis] * axis # (N, 3)
    radial     = np.linalg.norm(radial_vec, axis=1)# (N,)

    inside = (np.abs(axial) <= half_length) & (radial <= radius)
    return int(np.sum(inside))


def run_cavity_water_count(
    u,
    channel2,
    permeation_events,
    results_dir,
    start_frame=0,
    end_frame=None,
    water_selection=_WATER_SEL,
    water_radius=12.0,          # ← radius used ONLY for water counting
):
    """
    Main entry point — count waters in ch2 per frame, save results & plots.

    Parameters
    ----------
    u                 : MDAnalysis Universe
    channel2          : your Channel object for ch2
    permeation_events : list of dicts with 'end_2' key
    results_dir       : Path
    start_frame       : int
    end_frame         : int or None
    water_selection   : str  MDAnalysis selection for water oxygens
    water_radius      : float  radius (Å) used for water counting only.
                        Does NOT modify channel2.radius.
                        Default 12.0 Å — tighter than ch2's 15 Å so we
                        capture only waters directly in the Glu/Asn cavity.
    """
    results_dir = Path(results_dir) / "cavity_water_count"
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    n_frames = end_frame - start_frame + 1

    print("\n" + "=" * 70)
    print("CAVITY WATER COUNT ANALYSIS")
    print("=" * 70)

    # ── select water oxygen atoms ──────────────────────────────────────────────
    waters = u.select_atoms(water_selection)
    if len(waters) == 0:
        print(f"  [ERROR] No water atoms found with: '{water_selection}'")
        print(f"  Available resnames: {sorted(set(u.atoms.resnames))}")
        return None

    print(f"  Water O atoms in system  : {len(waters)}")
    print(f"  ch2 radius (channel obj) : {channel2.radius:.1f} Å  (NOT changed)")
    print(f"  Water counting radius    : {water_radius:.1f} Å")
    print(f"  Frames                   : {start_frame} → {end_frame}  ({n_frames} total)")
    print(f"  Water selection          : {water_selection}")

    # ── per-frame loop ─────────────────────────────────────────────────────────
    frame_indices = np.arange(start_frame, end_frame + 1, dtype=int)
    water_counts  = np.zeros(n_frames, dtype=int)

    with tqdm(total=n_frames, desc="  Cavity waters",
              unit="frame", dynamic_ncols=True) as pbar:

        for fi, ts in enumerate(u.trajectory[start_frame: end_frame + 1]):

            # Recompute ch2 geometry from current frame's CA positions
            # This updates channel_center, channel_axis, channel_length
            # It does NOT change channel2.radius — that stays at 15 Å
            channel2.compute_geometry(gate_num=2)

            # Count waters using water_radius (12 Å), NOT channel2.radius (15 Å)
            water_counts[fi] = _count_waters_in_cylinder(
                water_positions = waters.positions,
                center          = channel2.channel_center,
                axis            = channel2.channel_axis,
                half_length     = channel2.channel_length / 2.0,
                radius          = water_radius,              # ← 12 Å here
            )

            pbar.update(1)

    print(f"\n  Mean waters in cavity : {water_counts.mean():.2f}")
    print(f"  Min                   : {water_counts.min()}")
    print(f"  Max                   : {water_counts.max()}")

    # ── collect permeation exit frames ────────────────────────────────────────
    exit_frames = sorted({
        int(ev["end_2"])
        for ev in permeation_events
        if ev.get("end_2") is not None
        and start_frame <= int(ev["end_2"]) <= end_frame
    })
    print(f"  K⁺ exit frames (end_2) : {exit_frames}")

    # ── save pkl ───────────────────────────────────────────────────────────────
    pkl_data = {
        "start_frame"    : start_frame,
        "end_frame"      : end_frame,
        "n_frames"       : n_frames,
        "frame_indices"  : frame_indices,
        "water_counts"   : water_counts,
        "exit_frames"    : exit_frames,
        "water_selection": water_selection,
        "water_radius"   : water_radius,
        "ch2_radius"     : channel2.radius,
        "ch2_length"     : channel2.channel_length,
    }
    pkl_path = results_dir / "cavity_water_count.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"  ✅ Saved: {pkl_path}")

    # ── save txt ───────────────────────────────────────────────────────────────
    txt_path = results_dir / "cavity_water_count.txt"
    with open(txt_path, "w") as f:
        f.write("CAVITY WATER COUNT\n")
        f.write(f"ch2 radius (channel obj) : {channel2.radius:.1f} Å  (unchanged)\n")
        f.write(f"Water counting radius    : {water_radius:.1f} Å\n")
        f.write(f"Water sel  : {water_selection}\n")
        f.write(f"Frames     : {start_frame} → {end_frame}\n")
        f.write(f"Exit frames: {exit_frames}\n")
        f.write("=" * 30 + "\n")
        f.write(f"{'Frame':>10}  {'Waters':>8}\n")
        f.write("-" * 22 + "\n")
        for fr, wc in zip(frame_indices, water_counts):
            f.write(f"{fr:>10}  {wc:>8}\n")
    print(f"  ✅ Saved: {txt_path}")

    # ── plots ──────────────────────────────────────────────────────────────────
    _plot_water_count(
        frame_indices, water_counts,
        exit_frames=[],
        title=f"Water molecules in GLU/ASN cavity (r={water_radius:.0f} Å)",
        out_path=results_dir / "cavity_water_count.png",
    )
    _plot_water_count(
        frame_indices, water_counts,
        exit_frames=exit_frames,
        title=f"Water molecules in GLU/ASN cavity (r={water_radius:.0f} Å) — with K⁺ exit events",
        out_path=results_dir / "cavity_water_count_with_events.png",
    )
    print(f"  ✅ All outputs in: {results_dir}")
    return pkl_data


# ── plotting helper ────────────────────────────────────────────────────────────

def _plot_water_count(frame_indices, water_counts, exit_frames, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(frame_indices, water_counts,
            color="#2196F3", linewidth=0.7, alpha=0.55,
            label="Waters in ch2")

    if len(water_counts) >= 50:
        window   = 50
        kernel   = np.ones(window) / window
        smoothed = np.convolve(water_counts.astype(float), kernel, mode="same")
        half     = window // 2
        ax.plot(frame_indices[half:-half], smoothed[half:-half],
                color="#0D47A1", linewidth=1.8, alpha=0.9,
                label=f"Rolling mean ({window} frames)")

    first = True
    for ef in exit_frames:
        ax.axvline(x=ef, color="#E53935", linewidth=1.2,
                   linestyle="--", alpha=0.85,
                   label="K⁺ exits ch2 (end_2)" if first else "_nolegend_")
        first = False

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("# water molecules", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(frame_indices[0], frame_indices[-1])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")