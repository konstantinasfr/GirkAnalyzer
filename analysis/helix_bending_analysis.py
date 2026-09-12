"""
analysis/helix_bending_analysis.py
=====================================
Computes the bending angle of the M2 (TM2) helix at the conserved glycine
hinge (G169) for each subunit over the MD trajectory.

Method:
    For each frame, per subunit:
    1. Take Cα positions of upper segment (gly_resid - window to gly_resid - 1)
       axis_upper = vector from first to last CA, pointing TOWARD hinge (DOWN)
    2. Take Cα positions of lower segment (gly_resid + 1 to gly_resid + window)
       axis_lower = vector from last to first CA, pointing TOWARD hinge (UP)
    3. Compute angle between axis_upper and axis_lower
       → straight helix: ~180°, open state bend: ~157° (~23° bend)

    Also computes bending_degree = 180 - angle
       → straight helix: ~0°, open state: ~23°

Reference:
    Meng et al., Scientific Reports 2016 (doi: 10.1038/srep29399)

Output (saved in results_dir/helix_bending_analysis/window_<N>/):
    bending_angles.csv                   -- angle and bending_degree per subunit
    bending_angles.png                   -- combined raw, angle convention
    bending_angles_smooth<W>.png         -- combined smoothed, angle convention
    bending_degree.png                   -- combined raw, bending degree (180-angle)
    bending_degree_smooth<W>.png         -- combined smoothed, bending degree
    bending_<subunit>.png                -- individual subunit, both conventions

Called from main.py:

    from analysis.helix_bending_analysis import run_helix_bending_analysis

    run_helix_bending_analysis(
        u                = u,
        results_dir      = results_dir,
        channel_type     = args.channel_type,
        gly_residues     = gly_residues,
        start_frame      = start_frame,
        end_frame        = end_frame,
        window           = 5,
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
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Helix axis and angle
# -----------------------------------------------------------------------------

def _helix_axis(positions: np.ndarray) -> np.ndarray:
    """
    Vector from first to last Cα — overall direction of the helix segment.
    Normalized to unit vector.
    """
    axis = positions[-1] - positions[0]
    return axis / np.linalg.norm(axis)


def _compute_bending_angle(u, gly_resid: int, window: int) -> float:
    """
    Compute M2 bending angle at gly_resid for the current trajectory frame.

    Upper segment: residues [gly_resid - window ... gly_resid - 1]
        axis_upper = pos[last] - pos[first]  → points TOWARD hinge (DOWN)

    Lower segment: residues [gly_resid + 1 ... gly_resid + window]
        axis_lower = pos[first] - pos[last]  → points TOWARD hinge (UP)
                     (reversed so both vectors point toward the hinge)

    Angle between them:
        ~180° = straight helix
        ~157° = open state (23° bend, as per Meng et al. 2016)

    Returns angle in degrees.
    """
    upper_resids = list(range(gly_resid - window, gly_resid))
    lower_resids = list(range(gly_resid + 1, gly_resid + window + 1))

    upper_sel   = " or ".join([f"resid {r}" for r in upper_resids])
    lower_sel   = " or ".join([f"resid {r}" for r in lower_resids])
    upper_atoms = u.select_atoms(f"({upper_sel}) and name CA")
    lower_atoms = u.select_atoms(f"({lower_sel}) and name CA")

    if len(upper_atoms) < 2 or len(lower_atoms) < 2:
        return np.nan

    # upper: first→last = pointing DOWN toward hinge
    axis_upper = _helix_axis(upper_atoms.positions)

    # lower: last→first = pointing UP toward hinge (reversed!)
    axis_lower = -_helix_axis(lower_atoms.positions)

    cos_angle = np.dot(axis_upper, axis_lower) / (
        np.linalg.norm(axis_upper) * np.linalg.norm(axis_lower)
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _smooth(series: np.ndarray, window: int) -> np.ndarray:
    return (pd.Series(series)
            .rolling(window=window, center=True, min_periods=1)
            .mean().values)


def _plot_combined(
    df:               pd.DataFrame,
    subunit_labels:   List[str],
    title:            str,
    out_path:         Path,
    ylabel:           str,
    ref_lines:        List[tuple],   # list of (value, color, style, label)
    smoothing_window: Optional[int] = None,
):
    """All subunits on one combined plot."""
    n      = len(subunit_labels)
    cmap   = plt.cm.get_cmap("tab10", max(n, 1))
    frames = df["frame"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(subunit_labels):
        if col not in df.columns:
            continue
        vals = df[col].values
        lw   = 0.8
        if smoothing_window and smoothing_window > 1:
            vals = _smooth(vals, smoothing_window)
            lw   = 2.0
        ax.plot(frames, vals, label=col, linewidth=lw, alpha=0.9, color=cmap(i))

    for val, color, style, label in ref_lines:
        ax.axhline(val, color=color, linestyle=style, linewidth=1.2, label=label)

    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    legend = ax.legend(fontsize=9, loc="upper right", framealpha=0.6)
    if smoothing_window and smoothing_window > 1:
        for leg in legend.get_lines():
            leg.set_linewidth(2.0)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def _plot_single(
    frames:           np.ndarray,
    vals_angle:       np.ndarray,
    col:              str,
    channel_type:     str,
    out_path:         Path,
    smoothing_window: Optional[int] = None,
):
    """One subunit — two panels: angle (top) and bending degree 180-angle (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    vals_degree = 180.0 - vals_angle

    for ax, vals, ylabel, ref_lines in [
        (ax1, vals_angle,  "Bending angle (°)",
         [(180, "gray", ":", "180° (straight)"), (157, "red", "--", "157° (~23° bend)")]),
        (ax2, vals_degree, "Bending degree (°)",
         [(0,  "gray", ":", "0° (straight)"),    (23,  "red", "--", "23° (open state)")]),
    ]:
        ax.plot(frames, vals, linewidth=0.6, alpha=0.5,
                color="steelblue", label="raw")
        if smoothing_window and smoothing_window > 1:
            ax.plot(frames, _smooth(vals, smoothing_window),
                    linewidth=2.0, alpha=0.9, color="steelblue",
                    label=f"smooth (w={smoothing_window})")
        for val, color, style, label in ref_lines:
            ax.axhline(val, color=color, linestyle=style,
                       linewidth=1.2, label=label)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.6)
        ax.tick_params(labelsize=11)

    ax2.set_xlabel("Frame", fontsize=13)
    fig.suptitle(f"M2 bending at {col} — {channel_type}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_helix_bending_analysis(
    u,
    results_dir:      Path,
    channel_type:     str,
    gly_residues:     List[int],
    start_frame:      int,
    end_frame:        int,
    window:           int = 5,
    smoothing_window: int = 100,
):
    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    results_dir = Path(results_dir)
    out_dir     = results_dir / "helix_bending_analysis" / f"window_{window}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("HELIX BENDING ANALYSIS")
    print(f"  results_dir   : {results_dir}")
    print(f"  channel_type  : {channel_type}")
    print(f"  gly_residues  : {gly_residues}")
    print(f"  window        : ±{window} residues around hinge")
    print(f"  frames        : {start_frame} -> {end_frame}")
    print(f"  smoothing     : {smoothing_window} frames")
    print("=" * 70)

    subunit_labels = [to_pdb(r, channel_type) for r in gly_residues]
    print(f"\n  Hinge residues (PDB labels): {subunit_labels}")
    for r, lbl in zip(gly_residues, subunit_labels):
        print(f"  {lbl}: upper [{r-window}..{r-1}], hinge [{r}], lower [{r+1}..{r+window}]")

    n_frames   = end_frame - start_frame + 1
    all_angles = np.full((n_frames, len(gly_residues)), np.nan)

    print(f"\n  Computing bending angles over {n_frames} frames...")
    for fi, ts in enumerate(tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames,
        desc="  Helix bending",
        leave=False,
    )):
        for si, gly_resid in enumerate(gly_residues):
            all_angles[fi, si] = _compute_bending_angle(u, gly_resid, window)

    # build DataFrame — store both angle and bending degree
    frames = np.arange(start_frame, end_frame + 1)
    df     = pd.DataFrame({"frame": frames})
    df_deg = pd.DataFrame({"frame": frames})

    for i, col in enumerate(subunit_labels):
        df[col]     = all_angles[:, i]
        df_deg[col] = 180.0 - all_angles[:, i]

    # save CSV with both
    csv_path = out_dir / "bending_angles.csv"
    df_combined = pd.DataFrame({"frame": frames})
    for col in subunit_labels:
        df_combined[f"{col}_angle"]   = df[col]
        df_combined[f"{col}_bending"] = df_deg[col]
    df_combined.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV -> {csv_path.name}")

    # summary
    print(f"\n  Summary per subunit:")
    for col in subunit_labels:
        a = df[col].dropna()
        b = df_deg[col].dropna()
        print(f"    {col}: angle={a.mean():.1f}°±{a.std():.1f}°  "
              f"bending={b.mean():.1f}°±{b.std():.1f}°  "
              f"(min={b.min():.1f}°, max={b.max():.1f}°)")

    ref_angle  = [(180, "gray", ":", "180° (straight)"),
                  (157, "red",  "--", "157° (~23° bend)")]
    ref_degree = [(0,   "gray", ":", "0° (straight)"),
                  (23,  "red",  "--", "23° (open state)")]

    # --- combined angle plots ---
    _plot_combined(df, subunit_labels,
                   f"M2 helix bending at G169 — {channel_type}  |  raw",
                   out_dir / "bending_angles.png",
                   "Bending angle (°)", ref_angle, None)
    _plot_combined(df, subunit_labels,
                   f"M2 helix bending at G169 — {channel_type}  |  smoothed (w={smoothing_window})",
                   out_dir / f"bending_angles_smooth{smoothing_window}.png",
                   "Bending angle (°)", ref_angle, smoothing_window)

    # --- combined bending degree plots ---
    _plot_combined(df_deg, subunit_labels,
                   f"M2 bending degree at G169 — {channel_type}  |  raw",
                   out_dir / "bending_degree.png",
                   "Bending degree (°)", ref_degree, None)
    _plot_combined(df_deg, subunit_labels,
                   f"M2 bending degree at G169 — {channel_type}  |  smoothed (w={smoothing_window})",
                   out_dir / f"bending_degree_smooth{smoothing_window}.png",
                   "Bending degree (°)", ref_degree, smoothing_window)

    # --- individual plots per subunit (2 panels each) ---
    print(f"\n  Saving individual subunit plots...")
    for col in subunit_labels:
        if col not in df.columns:
            continue
        safe_col = col.replace(".", "_")
        _plot_single(
            frames           = frames,
            vals_angle       = df[col].values,
            col              = col,
            channel_type     = channel_type,
            out_path         = out_dir / f"bending_{safe_col}.png",
            smoothing_window = smoothing_window,
        )

    print(f"\n✓ Helix bending analysis complete.")
    print(f"  Results in: {out_dir}")