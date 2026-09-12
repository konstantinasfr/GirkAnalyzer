import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from MDAnalysis.analysis import rms


PIP_HEADGROUP_ATOMS = (
    "P5 O53 O52 O51 O5P "
    "C1 C2 C3 C4 C5 C6 O4P P4 O41 O42 O43 O3 HO3 HO6 O6 O2 HO2 "
    "H2 H4 H6 H1 H3 H5 "
    "O1 P1 O11 O12 O13 C11"
)

PIP_RESIDS = [1301, 1302, 1303, 1304]


def run_pip_rmsd_analysis(
    u,
    results_dir,
    start_frame=0,
    end_frame=None,
    pip_resids=None,
    channel_type="G12",
    pip_mapping=None,
):
    """
    Calculate RMSD of PIP2 headgroup atoms for each PIP residue
    over the trajectory, using frame 0 as reference.

    Parameters
    ----------
    u            : MDAnalysis Universe
    results_dir  : Path or str — parent results directory (pip_rmsd/ will be created inside)
    start_frame  : int, first frame to analyse
    end_frame    : int or None, last frame (inclusive); None = last frame of trajectory
    pip_resids   : list of int, PIP residue IDs (default: [1301, 1302, 1303, 1304])
    channel_type : str, used in plot title only
    """

    if pip_resids is None:
        pip_resids = PIP_RESIDS

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    out_dir = Path(results_dir) / "pip_rmsd"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("RUNNING PIP HEADGROUP RMSD ANALYSIS")
    print("=" * 70)

    rmsd_data = {}
    frames = None

    for resid in pip_resids:
        sel_str = f"resname PIP and resid {resid} and name {PIP_HEADGROUP_ATOMS}"
        headgroup = u.select_atoms(sel_str)

        if len(headgroup) == 0:
            print(f"  [WARNING] No atoms found for PIP resid {resid} — skipping.")
            continue

        print(f"  PIP {resid}: {len(headgroup)} headgroup atoms selected")

        R = rms.RMSD(headgroup, ref_frame=0, select=f"resname PIP and resid {resid} and name {PIP_HEADGROUP_ATOMS}")
        R.run(start=start_frame, stop=end_frame + 1)

        if frames is None:
            frames = R.results.rmsd[:, 0].astype(int)

        rmsd_data[resid] = R.results.rmsd[:, 2]

    if not rmsd_data:
        print("  [ERROR] No RMSD data calculated — check residue names and IDs.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rmsd_data, index=frames)
    df.index.name = "frame"
    df.columns = [f"PIP_{r}" for r in df.columns]
    df = df.round(3)

    csv_path = out_dir / "pip_headgroup_rmsd.csv"
    df.to_csv(csv_path)
    print(f"\n  CSV saved → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, resid in enumerate(pip_resids):
        col = f"PIP_{resid}"
        if col not in df.columns:
            continue
        label = f"PIP {resid} ({pip_mapping[resid]})" if pip_mapping and resid in pip_mapping else f"PIP {resid}"
        ax.plot(df.index, df[col], label=label,
                color=colors[i % len(colors)], linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title(f"{channel_type} — PIP2 headgroup RMSD (ref = frame 0)", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    img_path = out_dir / "pip_headgroup_rmsd.png"
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {img_path}")

    # ── Smoothed plot ─────────────────────────────────────────────────────────
    window = 100  # rolling average window (frames)

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    for i, resid in enumerate(pip_resids):
        col = f"PIP_{resid}"
        if col not in df.columns:
            continue
        smoothed = df[col].rolling(window=window, center=True).mean()
        label = f"PIP {resid} ({pip_mapping[resid]})" if pip_mapping and resid in pip_mapping else f"PIP {resid}"
        ax2.plot(df.index, smoothed, label=label,
                 color=colors[i % len(colors)], linewidth=2.0)

    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("RMSD (Å)", fontsize=12)
    ax2.set_title(f"{channel_type} — PIP2 headgroup RMSD smoothed (window={window} frames)", fontsize=13)
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.25, linestyle="--")
    fig2.tight_layout()

    img_path2 = out_dir / "pip_headgroup_rmsd_smoothed.png"
    fig2.savefig(img_path2, dpi=150)
    plt.close(fig2)
    print(f"  Smoothed plot saved → {img_path2}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n  Summary (mean ± std RMSD over trajectory):")
    for col in df.columns:
        print(f"    {col}: {df[col].mean():.2f} ± {df[col].std():.2f} Å")

    print("\n  PIP RMSD analysis complete!")
    return df