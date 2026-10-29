"""
analysis/protein_rmsd_analysis.py
====================================
Computes the RMSD of the full protein (Cα atoms) over the trajectory,
using the first frame as reference. Each frame is superimposed on the
reference first (translation + rotation removed), so the RMSD measures
changes of the structure only, not the protein drifting or turning in the
box. Saves a CSV and plots.

Output (saved in results_dir/protein_rmsd/):
    protein_rmsd.csv              -- per-frame RMSD: rmsd_A (fitted),
                                     rmsd_nofit_A (without fitting, as before)
    protein_rmsd.png              -- raw plot
    protein_rmsd_smooth<W>.png    -- smoothed plot

Called from main.py:

    from analysis.protein_rmsd_analysis import run_protein_rmsd_analysis

    run_protein_rmsd_analysis(
        u                = u,
        results_dir      = results_dir,
        channel_type     = args.channel_type,
        start_frame      = start_frame,
        end_frame        = end_frame,
        smoothing_window = 100,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def fitted_ca_rmsd(u, start_frame=0, end_frame=None, progress=False):
    """
    Cα RMSD of every frame start_frame..end_frame against start_frame, after
    superimposing the frame on the reference (translation + rotation removed,
    so only changes of the structure count). Returns (frames, fitted RMSD,
    RMSD without fitting) in Å.
    """
    from MDAnalysis.analysis import rms

    end_frame = len(u.trajectory) - 1 if end_frame is None else end_frame
    ca = u.select_atoms("protein and name CA")
    u.trajectory[start_frame]
    ref = ca.positions.copy()
    n = end_frame - start_frame + 1
    fitted, nofit = np.zeros(n), np.zeros(n)
    frames_iter = u.trajectory[start_frame:end_frame + 1]
    if progress:
        frames_iter = tqdm(frames_iter, total=n, desc="  Protein RMSD", leave=False)
    for i, _ in enumerate(frames_iter):
        pos = ca.positions
        fitted[i] = rms.rmsd(pos, ref, center=True, superposition=True)
        nofit[i] = np.sqrt(((pos - ref) ** 2).sum(axis=1).mean())
    return np.arange(start_frame, end_frame + 1), fitted, nofit


def run_protein_rmsd_analysis(
    u,
    results_dir:      Path,
    channel_type:     str,
    start_frame:      int,
    end_frame:        int,
    smoothing_window: int = 100,
):
    results_dir = Path(results_dir)
    out_dir     = results_dir / "protein_rmsd"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PROTEIN RMSD ANALYSIS")
    print(f"  results_dir   : {results_dir}")
    print(f"  channel_type  : {channel_type}")
    print(f"  frames        : {start_frame} -> {end_frame}")
    print(f"  smoothing     : {smoothing_window} frames")
    print(f"  selection     : protein Cα atoms")
    print(f"  reference     : frame {start_frame}")
    print("=" * 70)

    print(f"\n  Cα atoms: {len(u.select_atoms('protein and name CA'))}")
    print(f"  Computing RMSD over {end_frame - start_frame + 1} frames...")
    frames, rmsd_values, rmsd_nofit = fitted_ca_rmsd(u, start_frame, end_frame, progress=True)

    # save CSV
    df = pd.DataFrame({"frame": frames, "rmsd_A": rmsd_values, "rmsd_nofit_A": rmsd_nofit})
    csv_path = out_dir / "protein_rmsd.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV -> {csv_path.name}")
    print(f"  Mean RMSD : {rmsd_values.mean():.2f} Å")
    print(f"  Max RMSD  : {rmsd_values.max():.2f} Å")
    print(f"  Final RMSD: {rmsd_values[-1]:.2f} Å")

    def _plot(vals, title, out_path):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(frames, vals, linewidth=0.8, alpha=0.8, color="steelblue")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("RMSD (Å)", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {out_path.name}")

    # raw plot
    _plot(
        rmsd_values,
        f"Protein Cα RMSD — {channel_type}  |  ref = frame {start_frame}",
        out_dir / "protein_rmsd.png",
    )

    # smoothed plot
    if smoothing_window and smoothing_window > 1:
        smooth_vals = (pd.Series(rmsd_values)
                       .rolling(window=smoothing_window, center=True, min_periods=1)
                       .mean().values)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(frames, rmsd_values, linewidth=0.5, alpha=0.3,
                color="steelblue", label="raw")
        ax.plot(frames, smooth_vals, linewidth=2.0, alpha=0.9,
                color="steelblue", label=f"smooth (w={smoothing_window})")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("RMSD (Å)", fontsize=13)
        ax.set_title(
            f"Protein Cα RMSD — {channel_type}  |  "
            f"smoothed (w={smoothing_window})  |  ref = frame {start_frame}",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / f"protein_rmsd_smooth{smoothing_window}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved -> protein_rmsd_smooth{smoothing_window}.png")

    print(f"\n✓ Protein RMSD analysis complete.")
    print(f"  Results in: {out_dir}")