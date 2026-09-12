"""
hbc_rmsf_analysis.py
--------------------
Calculates per-residue RMSF and RMSD of the HBC gate residues across the
trajectory, to detect whether only a subset of the four subunits moves.

If hbc_rmsf.csv AND hbc_rmsd.csv already exist, the trajectory is skipped
and only the plots are regenerated (labels read from CSV to avoid mismatches).

Usage:
    from analysis.hbc_rmsf_analysis import run_hbc_rmsf_analysis

    run_hbc_rmsf_analysis(
        u              = u,
        hbc_residues   = hbc_residues,
        results_dir    = results_dir,
        channel_type   = args.channel_type,
        start_frame    = start_frame,
        end_frame      = end_frame,
        frame_duration = 0.075,
        smoothing_window = 100,
    )

Outputs (results_dir/hbc_rmsf/):
    hbc_rmsf.csv                     - RMSF (A) per residue
    hbc_rmsd.csv                     - RMSD (A) per residue per frame
    hbc_rmsf_bar.png                 - RMSF bar chart
    hbc_rmsd_trajectories_raw.png    - RMSD vs frame (raw)
    hbc_rmsd_trajectories_smooth.png - RMSD vs frame (smoothed)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda

from analysis.converter import to_pdb

_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _make_plots(out_dir, labels, rmsf_values, frames, rmsd_traces,
                channel_type, start_frame, end_frame, smoothing_window):
    """Regenerate all plots from already-computed data."""

    # RMSF bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rmsf_values,
                  color=_COLOURS[:len(labels)], edgecolor="k", linewidth=0.7)
    for bar, val in zip(bars, rmsf_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSF (A)")
    ax.set_title(f"{channel_type} - HBC gate RMSF per residue\n"
                 f"(Ca, frames {start_frame}-{end_frame})\n"
                 "One number per residue - higher = more mobile overall")
    ax.set_ylim(0, max(rmsf_values) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "hbc_rmsf_bar.png", dpi=150)
    plt.close(fig)
    print(f"[HBC RMSF/RMSD] Saved: {out_dir / 'hbc_rmsf_bar.png'}")

    # RMSD trajectories (raw + smoothed)
    for suffix, smooth in [("raw", False), ("smooth", True)]:
        fig, ax = plt.subplots(figsize=(13, 4))
        for r_idx, lbl in enumerate(labels):
            y = rmsd_traces[r_idx]
            if smooth:
                y = (pd.Series(y)
                     .rolling(smoothing_window, center=True, min_periods=1)
                     .mean().values)
            ax.plot(frames, y,
                    color=_COLOURS[r_idx],
                    alpha=0.7 if not smooth else 1.0,
                    linewidth=0.8 if not smooth else 2.0,
                    label=lbl)
        ax.set_xlabel("Frame")
        ax.set_ylabel("RMSD from frame 0 (A)")
        title_extra = f" - rolling window {smoothing_window} frames" if smooth else ""
        ax.set_title(
            f"{channel_type} - HBC gate Ca RMSD vs frame{title_extra}\n"
            "Each line = one subunit's residue. Flat line = not moving."
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fname = f"hbc_rmsd_trajectories_{suffix}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"[HBC RMSF/RMSD] Saved: {out_dir / fname}")


def run_hbc_rmsf_analysis(
    u: mda.Universe,
    hbc_residues: list,
    results_dir: Path,
    channel_type: str,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_duration: float = 0.075,
    smoothing_window: int = 100,
    atom_name: str = "CA",
):
    results_dir = Path(results_dir)
    out_dir = results_dir / "hbc_rmsf"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(u.trajectory)
    if end_frame < 0 or end_frame >= n_frames:
        end_frame = n_frames - 1

    n_res = len(hbc_residues)

    rmsf_csv = out_dir / "hbc_rmsf.csv"
    rmsd_csv = out_dir / "hbc_rmsd.csv"

    # skip trajectory if CSVs already exist
    if rmsf_csv.exists() and rmsd_csv.exists():
        print("\n[HBC RMSF/RMSD] CSVs already exist - skipping trajectory, regenerating plots only.")
        rmsf_df     = pd.read_csv(rmsf_csv)
        rmsd_df     = pd.read_csv(rmsd_csv)
        rmsf_values = rmsf_df["RMSF_A"].tolist()
        # derive frame numbers: use "frame" column if present, else fall back from time_ns
        if "frame" in rmsd_df.columns:
            frames = rmsd_df["frame"].values
        else:
            frames = (rmsd_df["time_ns"].values / frame_duration).astype(int)
        # read labels from CSV columns directly - avoids any converter mismatch
        skip_cols   = {"time_ns", "frame"}
        labels      = [c for c in rmsd_df.columns if c not in skip_cols]
        rmsd_traces = [rmsd_df[lbl].values for lbl in labels]

        _make_plots(out_dir, labels, rmsf_values, frames, rmsd_traces,
                    channel_type, start_frame, end_frame, smoothing_window)
        return rmsf_df, rmsd_df

    # full trajectory calculation
    labels = [to_pdb(r, channel_type) for r in hbc_residues]

    selections = []
    for resid in hbc_residues:
        sel = u.select_atoms(f"resid {resid} and name {atom_name}")
        if len(sel) == 0:
            print(f"  WARNING: no {atom_name} for resid {resid}, using all heavy atoms")
            sel = u.select_atoms(f"resid {resid} and not type H")
        selections.append(sel)

    frame_indices = list(range(start_frame, end_frame + 1))
    n_time  = len(frame_indices)
    frames  = np.array(frame_indices)
    times   = frames * frame_duration

    print(f"\n[HBC RMSF/RMSD] Collecting positions for {n_res} residues "
          f"(frames {start_frame}-{end_frame}) ...")

    pos_data = [np.zeros((n_time, len(sel), 3)) for sel in selections]
    for t_idx, frame_idx in enumerate(frame_indices):
        u.trajectory[frame_idx]
        for r_idx, sel in enumerate(selections):
            pos_data[r_idx][t_idx] = sel.positions.copy()

    # RMSF
    rmsf_values = []
    for r_idx in range(n_res):
        mean_pos = pos_data[r_idx].mean(axis=0)
        diff     = pos_data[r_idx] - mean_pos[np.newaxis]
        per_atom = np.sqrt((diff**2).sum(axis=2)).mean(axis=0)
        rmsf_values.append(float(per_atom.mean()))

    # RMSD
    rmsd_traces = []
    for r_idx in range(n_res):
        ref  = pos_data[r_idx][0]
        diff = pos_data[r_idx] - ref[np.newaxis]
        rmsd = np.sqrt((diff**2).sum(axis=2).mean(axis=1))
        rmsd_traces.append(rmsd)

    # save CSVs
    rmsf_df = pd.DataFrame({
        "resid":     hbc_residues,
        "pdb_label": labels,
        "n_atoms":   [len(s) for s in selections],
        "RMSF_A":    rmsf_values,
    })
    rmsf_df.to_csv(rmsf_csv, index=False)
    print(f"[HBC RMSF/RMSD] Saved: {rmsf_csv}")

    rmsd_df = pd.DataFrame({"frame": frames, "time_ns": times})
    for r_idx, lbl in enumerate(labels):
        rmsd_df[lbl] = rmsd_traces[r_idx]
    rmsd_df.to_csv(rmsd_csv, index=False)
    print(f"[HBC RMSF/RMSD] Saved: {rmsd_csv}")

    _make_plots(out_dir, labels, rmsf_values, frames, rmsd_traces,
                channel_type, start_frame, end_frame, smoothing_window)

    # console summary
    print("\n[HBC RMSF/RMSD] Summary:")
    print(f"  {'PDB label':<16}  {'RMSF (A)':>10}  {'RMSD final (A)':>16}")
    print("  " + "-" * 46)
    for r_idx, (lbl, rmsf_val) in enumerate(zip(labels, rmsf_values)):
        print(f"  {lbl:<16}  {rmsf_val:>10.3f}  {rmsd_traces[r_idx][-1]:>16.3f}")

    mean_rmsf = np.mean(rmsf_values)
    std_rmsf  = np.std(rmsf_values)
    print(f"\n  RMSF mean +/- SD : {mean_rmsf:.3f} +/- {std_rmsf:.3f} A")
    outliers = [lbl for lbl, val in zip(labels, rmsf_values)
                if abs(val - mean_rmsf) > std_rmsf]
    if outliers:
        print(f"  Residues > 1 SD from mean: {', '.join(outliers)}")
    else:
        print("  All residues have similar RMSF (no clear outlier)")

    return rmsf_df, rmsd_df