"""
analysis/chi_angle_analysis.py
================================
Computes chi1 and chi2 dihedral angles for specified residues over the
MD trajectory. Useful for tracking sidechain reorientation events.

Chi angle definitions (same as neighbor_analysis.py):
    chi1: N  - CA - CB - CG  (or equivalent)
    chi2: CA - CB - CG - CD  (or equivalent)

Output (saved in results_dir/chi_angles/<resid_pdb>/):
    chi_angles.csv              -- per-frame chi1, chi2
    chi_angles.png              -- raw trajectory plot
    chi_angles_smooth<W>.png    -- smoothed trajectory plot
    chi_histogram.png           -- distribution of chi1/chi2 angles

Called from main.py:

    from analysis.chi_angle_analysis import run_chi_angle_analysis

    run_chi_angle_analysis(
        u                = u,
        results_dir      = results_dir,
        channel_type     = args.channel_type,
        resids           = [469],        # raw resids
        start_frame      = start_frame,
        end_frame        = end_frame,
        smoothing_window = 100,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import calc_dihedrals
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Chi angle atom definitions (from neighbor_analysis.py)
# -----------------------------------------------------------------------------
CHI_ATOMS = {
    "GLU": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "ASP": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","OD1"]},
    "ASN": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","OD1"]},
    "GLN": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "LYS": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "ARG": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "SER": {"chi1": ["N","CA","CB","OG"],  "chi2": None},
    "THR": {"chi1": ["N","CA","CB","OG1"], "chi2": None},
    "PHE": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "TYR": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "TRP": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "HIS": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","ND1"]},
    "HID": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","ND1"]},
    "HIE": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","ND1"]},
    "HIP": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","ND1"]},
    "ILE": {"chi1": ["N","CA","CB","CG1"], "chi2": ["CA","CB","CG1","CD"]},
    "LEU": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "MET": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","SD"]},
    "VAL": {"chi1": ["N","CA","CB","CG1"], "chi2": None},
    "CYS": {"chi1": ["N","CA","CB","SG"],  "chi2": None},
    "PRO": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
}


def _get_chi(residue, chi_name: str) -> float:
    """Compute chi angle for a residue. Returns np.nan if not computable."""
    rn = residue.resname.upper()
    if rn not in CHI_ATOMS:
        return np.nan
    names = CHI_ATOMS[rn].get(chi_name)
    if names is None:
        return np.nan
    pts = []
    for n in names:
        sel = residue.atoms.select_atoms(f"name {n}")
        if len(sel) == 0:
            return np.nan
        pts.append(sel.positions[0])
    a1, a2, a3, a4 = [p[np.newaxis, :] for p in pts]
    return float(np.degrees(calc_dihedrals(a1, a2, a3, a4)[0]))


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def _smooth(vals: np.ndarray, window: int) -> np.ndarray:
    return (pd.Series(vals)
            .rolling(window=window, center=True, min_periods=1)
            .mean().values)


def _plot_chi_trajectory(
    frames:           np.ndarray,
    chi1:             np.ndarray,
    chi2:             np.ndarray,
    pdb_label:        str,
    out_path:         Path,
    smoothing_window: Optional[int] = None,
):
    """Plot chi1 and chi2 over trajectory — one panel each."""
    has_chi2 = not np.all(np.isnan(chi2))
    n_panels = 2 if has_chi2 else 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, vals, label, color in zip(
        axes,
        [chi1, chi2[:len(chi1)] if has_chi2 else None],
        ["χ1", "χ2"],
        ["steelblue", "darkorange"],
    ):
        if vals is None:
            continue
        smooth = smoothing_window and smoothing_window > 1
        ax.plot(frames, vals, linewidth=0.5, alpha=0.3,
                color=color, label="raw")
        if smooth:
            ax.plot(frames, _smooth(vals, smoothing_window),
                    linewidth=2.0, alpha=0.9, color=color,
                    label=f"smooth (w={smoothing_window})")
        ax.axhline(0,    color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(180,  color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-180, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"{label} (°)", fontsize=12)
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
        ax.legend(fontsize=8, loc="upper right", framealpha=0.6)
        ax.tick_params(labelsize=11)

    axes[-1].set_xlabel("Frame", fontsize=13)
    fig.suptitle(f"Sidechain chi angles — {pdb_label}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


def _plot_chi_polar(
    chi1:      np.ndarray,
    chi2:      np.ndarray,
    pdb_label: str,
    out_path:  Path,
):
    """
    Polar rose plot of chi1 and chi2 angle distributions.
    Chi angles are circular (-180 to +180) so a polar histogram
    is more appropriate than a linear one.
    """
    has_chi2 = not np.all(np.isnan(chi2))
    n_panels = 2 if has_chi2 else 1

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5),
                             subplot_kw={"projection": "polar"})
    if n_panels == 1:
        axes = [axes]

    for ax, vals, label, color in zip(
        axes,
        [chi1, chi2 if has_chi2 else None],
        ["χ1", "χ2"],
        ["steelblue", "darkorange"],
    ):
        if vals is None:
            continue
        valid = vals[~np.isnan(vals)]

        # bin into 36 bins of 10° each
        n_bins  = 36
        bins    = np.linspace(-np.pi, np.pi, n_bins + 1)
        counts, _ = np.histogram(np.radians(valid), bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width   = 2 * np.pi / n_bins

        bars = ax.bar(bin_centers, counts, width=width,
                      color=color, edgecolor="white",
                      alpha=0.85, linewidth=0.5)

        # mark mean angle
        mean_rad = np.arctan2(
            np.nanmean(np.sin(np.radians(valid))),
            np.nanmean(np.cos(np.radians(valid)))
        )
        ax.plot([mean_rad, mean_rad], [0, counts.max()],
                color="red", linewidth=2.0,
                label=f"mean={np.degrees(mean_rad):.1f}°")

        # labels at cardinal angles
        ax.set_xticks(np.radians([-180, -120, -60, 0, 60, 120]))
        ax.set_xticklabels(["-180°", "-120°", "-60°", "0°", "60°", "120°"],
                           fontsize=9)
        ax.set_title(f"{pdb_label} — {label}", fontsize=12, pad=15)
        ax.legend(fontsize=8, loc="upper right",
                  bbox_to_anchor=(1.3, 1.1))

    fig.suptitle(f"Chi angle distributions — {pdb_label}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Main per-residue analysis
# -----------------------------------------------------------------------------

def _analyse_chi(
    u,
    resid:            int,
    pdb_label:        str,
    start_frame:      int,
    end_frame:        int,
    out_dir:          Path,
    smoothing_window: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # get residue object
    sel = u.select_atoms(f"resid {resid}")
    if len(sel) == 0:
        print(f"  WARNING: resid {resid} not found -- skipping.")
        return

    residue  = sel.residues[0]
    resname  = residue.resname.upper()
    n_frames = end_frame - start_frame + 1
    frames   = np.arange(start_frame, end_frame + 1)

    if resname not in CHI_ATOMS:
        print(f"  [{pdb_label}] {resname} has no chi angles defined -- skipping.")
        return

    has_chi2 = CHI_ATOMS[resname]["chi2"] is not None
    chi1_vals = np.full(n_frames, np.nan)
    chi2_vals = np.full(n_frames, np.nan)

    print(f"  [{pdb_label}] Computing chi angles ({resname}, "
          f"chi1={'yes'}, chi2={'yes' if has_chi2 else 'no'})...")

    for fi, ts in enumerate(tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames,
        desc=f"  {pdb_label} chi",
        leave=False,
    )):
        chi1_vals[fi] = _get_chi(residue, "chi1")
        if has_chi2:
            chi2_vals[fi] = _get_chi(residue, "chi2")

    # save CSV
    df = pd.DataFrame({"frame": frames, "chi1": chi1_vals})
    if has_chi2:
        df["chi2"] = chi2_vals
    df.to_csv(out_dir / "chi_angles.csv", index=False)
    print(f"    Saved -> chi_angles.csv")

    # summary stats
    print(f"    chi1: mean={np.nanmean(chi1_vals):.1f}°  "
          f"std={np.nanstd(chi1_vals):.1f}°  "
          f"min={np.nanmin(chi1_vals):.1f}°  "
          f"max={np.nanmax(chi1_vals):.1f}°")
    if has_chi2:
        print(f"    chi2: mean={np.nanmean(chi2_vals):.1f}°  "
              f"std={np.nanstd(chi2_vals):.1f}°  "
              f"min={np.nanmin(chi2_vals):.1f}°  "
              f"max={np.nanmax(chi2_vals):.1f}°")

    # trajectory plot
    _plot_chi_trajectory(
        frames           = frames,
        chi1             = chi1_vals,
        chi2             = chi2_vals,
        pdb_label        = pdb_label,
        out_path         = out_dir / "chi_angles.png",
        smoothing_window = smoothing_window,
    )

    # histogram
    _plot_chi_polar(
        chi1      = chi1_vals,
        chi2      = chi2_vals,
        pdb_label = pdb_label,
        out_path  = out_dir / "chi_polar.png",
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_chi_angle_analysis(
    u,
    results_dir:      Path,
    channel_type:     str,
    resids:           List[int],   # raw resids
    start_frame:      int,
    end_frame:        int,
    smoothing_window: int = 100,
):
    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    results_dir = Path(results_dir)
    out_base    = results_dir / "chi_angles"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CHI ANGLE ANALYSIS")
    print(f"  results_dir   : {results_dir}")
    print(f"  channel_type  : {channel_type}")
    print(f"  resids (raw)  : {resids}")
    print(f"  frames        : {start_frame} -> {end_frame}")
    print(f"  smoothing     : {smoothing_window} frames")
    print("=" * 70)

    for resid in resids:
        pdb_label   = to_pdb(resid, channel_type)
        folder_name = pdb_label.replace(".", "_")
        out_dir     = out_base / folder_name

        print(f"\n  Processing resid {resid} -> {pdb_label}")

        _analyse_chi(
            u                = u,
            resid            = resid,
            pdb_label        = pdb_label,
            start_frame      = start_frame,
            end_frame        = end_frame,
            out_dir          = out_dir,
            smoothing_window = smoothing_window,
        )

    print(f"\n✓ Chi angle analysis complete.")
    print(f"  Results in: {out_base}")