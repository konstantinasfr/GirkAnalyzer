"""
analysis/hbond_pair_analysis.py
=================================
Detects hydrogen bonds between specific residue pairs across the MD trajectory.

H-bond criteria (same as neighbor_analysis.py / Simulaid):
    - H···A distance <= 3.5 Å
    - D-H···A angle  >= 120°
    where D = donor heavy atom (N, O, S)
          H = hydrogen
          A = acceptor heavy atom (N, O, S)

For each residue pair, per frame:
    - Detects all H-bonds between the two residues
    - Records donor/acceptor atoms, distance, angle
    - Binary yes/no H-bond state

Output (saved in results_dir/hbond_analysis/<resA_pdb>_<resB_pdb>/):
    hbond_per_frame.csv          -- per-frame: frame, n_hbonds, hbond (0/1), bond details
    hbond_atom_pair_summary.csv  -- which atom pairs form H-bonds and how often
    hbond_residue_summary.csv    -- overall H-bond frequency for this pair
    hbond_state.png              -- binary yes/no plot over trajectory
    hbond_state_smooth<W>.png    -- smoothed version
    hbond_count_summary.png      -- bar plot: frames with/without H-bond

Called from main.py:

    from analysis.hbond_pair_analysis import run_hbond_pair_analysis

    run_hbond_pair_analysis(
        u                = u,
        results_dir      = results_dir,
        channel_type     = args.channel_type,
        residue_pairs    = [(469, 1044)],   # raw resid pairs
        start_frame      = start_frame,
        end_frame        = end_frame,
        smoothing_window = 100,
        hbond_dist       = 3.5,
        hbond_angle      = 120.0,
    )
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# H-bond constants
# -----------------------------------------------------------------------------
HBOND_ELEMENTS  = {"N", "O", "S"}
HBOND_COVALENT  = 1.5   # max D-H covalent bond distance to find donor heavy atom


# -----------------------------------------------------------------------------
# H-bond detection (extracted from neighbor_analysis.py)
# -----------------------------------------------------------------------------

def _find_donor_heavy(h_atom, donor_heavy_list):
    """Find the heavy atom covalently bonded to a hydrogen."""
    try:
        bonded = [a for a in h_atom.bonded_atoms
                  if a.element.upper() in HBOND_ELEMENTS]
        if bonded:
            return bonded[0]
    except Exception:
        pass
    # fallback: find closest heavy atom within covalent bond distance
    H = h_atom.position
    best_dist, best_atom = HBOND_COVALENT, None
    for heavy in donor_heavy_list:
        if heavy.element.upper() not in HBOND_ELEMENTS:
            continue
        d = float(np.linalg.norm(heavy.position - H))
        if d < best_dist:
            best_dist, best_atom = d, heavy
    return best_atom


def _find_hbonds(atoms_a, label_a, atoms_b, label_b,
                 hbond_dist: float, hbond_angle: float) -> list:
    """
    Find all H-bonds between two atom groups.
    Checks both directions: A→B and B→A.

    Returns list of dicts with bond details.
    """
    bonds = []
    for donor_grp, donor_lbl, acceptor_grp, acceptor_lbl in [
        (atoms_a, label_a, atoms_b, label_b),
        (atoms_b, label_b, atoms_a, label_a),
    ]:
        donor_heavy = [a for a in donor_grp if not a.name.startswith("H")]
        for h in donor_grp:
            if not h.name.startswith("H"):
                continue
            D_atom = _find_donor_heavy(h, donor_heavy)
            if D_atom is None:
                continue
            D     = D_atom.position
            H_pos = h.position
            for acc in acceptor_grp:
                if acc.element.upper() not in HBOND_ELEMENTS:
                    continue
                if acc.name.startswith("H"):
                    continue
                A        = acc.position
                dist_HA  = float(np.linalg.norm(H_pos - A))
                if dist_HA > hbond_dist:
                    continue
                v1   = D     - H_pos
                v2   = A     - H_pos
                denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
                angle = float(np.degrees(
                    np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))
                ))
                if angle >= hbond_angle:
                    bonds.append({
                        "donor_res":      donor_lbl,
                        "donor_atom":     D_atom.name,
                        "hydrogen":       h.name,
                        "acceptor_res":   acceptor_lbl,
                        "acceptor_atom":  acc.name,
                        "dist_HA_A":      round(dist_HA, 3),
                        "angle_DHA_deg":  round(angle, 1),
                    })
    return bonds


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _plot_hbond_state(
    frames:           np.ndarray,
    hbond_binary:     np.ndarray,
    label_a:          str,
    label_b:          str,
    out_path:         Path,
    smoothing_window: int = None,
):
    """
    Binary H-bond state plot: y=1 (H-bond present) or y=0 (absent).
    Optionally overlays a smoothed line showing fraction of frames with H-bond
    in a rolling window.
    """
    fig, ax = plt.subplots(figsize=(14, 3))

    # draw binary state as filled area
    ax.fill_between(frames, hbond_binary, step="mid",
                    alpha=0.4, color="steelblue", label="H-bond present")
    ax.plot(frames, hbond_binary, drawstyle="steps-mid",
            linewidth=0.5, color="steelblue", alpha=0.6)

    if smoothing_window and smoothing_window > 1:
        smooth = (pd.Series(hbond_binary.astype(float))
                  .rolling(window=smoothing_window, center=True, min_periods=1)
                  .mean().values)
        ax.plot(frames, smooth, linewidth=2.0, color="navy",
                label=f"fraction (window={smoothing_window})", alpha=0.9)

    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel("H-bond", fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"], fontsize=12)
    ax.set_ylim(-0.1, 1.3)
    ax.set_title(f"H-bond: {label_a} ↔ {label_b}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.6)
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


def _plot_hbond_summary_bar(
    n_with:   int,
    n_without: int,
    label_a:  str,
    label_b:  str,
    out_path: Path,
):
    """Bar plot: frames with H-bond vs without."""
    total  = n_with + n_without
    pct_w  = 100.0 * n_with   / max(total, 1)
    pct_wo = 100.0 * n_without / max(total, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["H-bond\npresent", "H-bond\nabsent"],
        [n_with, n_without],
        color=["steelblue", "lightcoral"],
        edgecolor="white", alpha=0.85,
    )
    for bar, val, pct in zip(bars, [n_with, n_without], [pct_w, pct_wo]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Number of frames", fontsize=13)
    ax.set_title(f"H-bond frequency\n{label_a} ↔ {label_b}", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Main per-pair analysis
# -----------------------------------------------------------------------------

def _analyse_pair(
    u,
    resid_a:          int,
    resid_b:          int,
    label_a:          str,
    label_b:          str,
    start_frame:      int,
    end_frame:        int,
    out_dir:          Path,
    hbond_dist:       float,
    hbond_angle:      float,
    smoothing_window: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms_a = u.select_atoms(f"resid {resid_a}")
    atoms_b = u.select_atoms(f"resid {resid_b}")

    if len(atoms_a) == 0:
        print(f"  WARNING: resid {resid_a} ({label_a}) not found -- skipping.")
        return
    if len(atoms_b) == 0:
        print(f"  WARNING: resid {resid_b} ({label_b}) not found -- skipping.")
        return

    n_frames     = end_frame - start_frame + 1
    frames       = np.arange(start_frame, end_frame + 1)
    hbond_binary = np.zeros(n_frames, dtype=int)
    hbond_count  = np.zeros(n_frames, dtype=int)

    per_frame_rows = []
    atom_pair_counter = Counter()
    residue_pair_frames = set()

    print(f"\n  [{label_a} ↔ {label_b}] Scanning {n_frames} frames...")

    for fi, ts in enumerate(tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames,
        desc=f"  {label_a}↔{label_b}",
        leave=False,
    )):
        bonds = _find_hbonds(
            atoms_a, label_a, atoms_b, label_b,
            hbond_dist, hbond_angle,
        )
        n = len(bonds)
        hbond_count[fi]  = n
        hbond_binary[fi] = 1 if n > 0 else 0

        frame = int(frames[fi])
        if n > 0:
            residue_pair_frames.add(frame)

        for b in bonds:
            atom_pair_counter[(
                b["donor_res"], b["donor_atom"], b["hydrogen"],
                b["acceptor_res"], b["acceptor_atom"],
            )] += 1

        per_frame_rows.append({
            "frame":         frame,
            "n_hbonds":      n,
            "hbond_present": hbond_binary[fi],
            "bonds_detail":  str(bonds) if bonds else "",
        })

    # ------------------------------------------------------------------
    # Save per-frame CSV
    # ------------------------------------------------------------------
    df_per_frame = pd.DataFrame(per_frame_rows)
    df_per_frame.to_csv(out_dir / "hbond_per_frame.csv", index=False)
    print(f"    Saved -> hbond_per_frame.csv")

    # ------------------------------------------------------------------
    # Atom pair summary
    # ------------------------------------------------------------------
    if atom_pair_counter:
        atom_rows = []
        for (dr, da, hy, ar, aa), cnt in atom_pair_counter.most_common():
            atom_rows.append({
                "donor_res":    dr,
                "donor_atom":   da,
                "hydrogen":     hy,
                "acceptor_res": ar,
                "acceptor_atom":aa,
                "n_frames":     cnt,
                "pct_frames":   round(100.0 * cnt / n_frames, 2),
            })
        pd.DataFrame(atom_rows).to_csv(
            out_dir / "hbond_atom_pair_summary.csv", index=False)
        print(f"    Saved -> hbond_atom_pair_summary.csv")

    # ------------------------------------------------------------------
    # Residue pair summary
    # ------------------------------------------------------------------
    n_with    = int(hbond_binary.sum())
    n_without = n_frames - n_with
    pct_with  = round(100.0 * n_with / n_frames, 2)

    pd.DataFrame([{
        "residue_A":       label_a,
        "residue_B":       label_b,
        "n_frames_total":  n_frames,
        "n_frames_hbond":  n_with,
        "n_frames_no_hbond": n_without,
        "pct_frames_hbond": pct_with,
        "hbond_dist_threshold_A": hbond_dist,
        "hbond_angle_threshold_deg": hbond_angle,
    }]).to_csv(out_dir / "hbond_residue_summary.csv", index=False)
    print(f"    Saved -> hbond_residue_summary.csv")
    print(f"    H-bond present in {n_with}/{n_frames} frames ({pct_with}%)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # binary state plot
    _plot_hbond_state(
        frames           = frames,
        hbond_binary     = hbond_binary,
        label_a          = label_a,
        label_b          = label_b,
        out_path         = out_dir / "hbond_state.png",
        smoothing_window = None,
    )

    # smoothed state plot
    if smoothing_window and smoothing_window > 1:
        _plot_hbond_state(
            frames           = frames,
            hbond_binary     = hbond_binary,
            label_a          = label_a,
            label_b          = label_b,
            out_path         = out_dir / f"hbond_state_smooth{smoothing_window}.png",
            smoothing_window = smoothing_window,
        )

    # bar plot summary
    _plot_hbond_summary_bar(
        n_with    = n_with,
        n_without = n_without,
        label_a   = label_a,
        label_b   = label_b,
        out_path  = out_dir / "hbond_count_summary.png",
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_hbond_pair_analysis(
    u,
    results_dir:      Path,
    channel_type:     str,
    residue_pairs:    List[Tuple[int, int]],  # raw resid pairs
    start_frame:      int,
    end_frame:        int,
    smoothing_window: int   = 100,
    hbond_dist:       float = 3.5,    # H···A distance threshold (Å)
    hbond_angle:      float = 120.0,  # D-H···A angle threshold (degrees)
):
    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    results_dir = Path(results_dir)
    out_base    = results_dir / "hbond_analysis"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("H-BOND PAIR ANALYSIS")
    print(f"  results_dir   : {results_dir}")
    print(f"  channel_type  : {channel_type}")
    print(f"  pairs (raw)   : {residue_pairs}")
    print(f"  frames        : {start_frame} -> {end_frame}")
    print(f"  H-bond dist   : <= {hbond_dist} Å  (H···A)")
    print(f"  H-bond angle  : >= {hbond_angle}°  (D-H···A)")
    print(f"  smoothing     : {smoothing_window} frames")
    print("=" * 70)

    for resid_a, resid_b in residue_pairs:
        label_a = to_pdb(resid_a, channel_type)
        label_b = to_pdb(resid_b, channel_type)

        # get residue names for better labels
        sel_a = u.select_atoms(f"resid {resid_a}")
        sel_b = u.select_atoms(f"resid {resid_b}")
        if len(sel_a) > 0:
            label_a = f"{sel_a.residues[0].resname}{label_a}"
        if len(sel_b) > 0:
            label_b = f"{sel_b.residues[0].resname}{label_b}"

        # folder name e.g. "GLN197_2b_MET319_2b"
        folder_name = f"{label_a}_{label_b}".replace(".", "_")
        out_dir     = out_base / folder_name

        print(f"\n  Pair: {label_a} ↔ {label_b}  (raw: {resid_a}, {resid_b})")

        _analyse_pair(
            u                = u,
            resid_a          = resid_a,
            resid_b          = resid_b,
            label_a          = label_a,
            label_b          = label_b,
            start_frame      = start_frame,
            end_frame        = end_frame,
            out_dir          = out_dir,
            hbond_dist       = hbond_dist,
            hbond_angle      = hbond_angle,
            smoothing_window = smoothing_window,
        )

    print(f"\n✓ H-bond pair analysis complete.")
    print(f"  Results in: {out_base}")