"""
analysis/residue_neighbors_analysis.py
=========================================
Analyses proximity of protein residues to all other protein residues.
For each residue of interest, computes minimum distance to all neighboring
residues across the trajectory.

Strategy
--------
Phase 1 - Pre-screen (first `prescreen_frames` frames):
    Compute min distance from the residue of interest to every other protein
    residue. Keep only residues whose min distance < threshold * prescreen_factor.
    SKIPPED if distances_all_pdb.csv already exists (loaded instead).

Phase 2 - Full trajectory (start_frame -> end_frame):
    For each frame, compute distances only to the candidate set.
    SKIPPED if distances_all_pdb.csv already exists (loaded instead).

Output (per residue of interest):
    results_dir/residue_neighbors/<resid_pdb>/
        proximity_summary.csv         -- residues ever < threshold, sorted
        distances_all_pdb.csv         -- per-frame distances (pdb labels)
        distances_all_raw.csv         -- per-frame distances (raw labels)
        trajectory_plots/
            trajectory_close.png      -- residues ever < threshold
            trajectory_close_smooth<W>.png  -- smoothed version

Called from main.py:

    from analysis.residue_neighbors_analysis import run_residue_neighbors_analysis

    run_residue_neighbors_analysis(
        u                = u,
        results_dir      = results_dir,
        channel_type     = args.channel_type,
        resids           = [469],         # raw resids (will be converted to PDB labels)
        start_frame      = start_frame,
        end_frame        = end_frame,
        threshold        = 4.0,
        prescreen_factor = 5.0,
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
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Pre-screen
# -----------------------------------------------------------------------------

def _prescreen_candidates(
    u,
    resid:            int,
    protein_residues: list,
    threshold:        float,
    prescreen_frames: int,
    prescreen_factor: float,
    start_frame:      int,
) -> list:
    """
    Screen all protein residues in first prescreen_frames frames.
    Keep only those within threshold * prescreen_factor of the residue of interest.
    Excludes the residue itself.
    """
    cutoff    = threshold * prescreen_factor
    min_dists = np.full(len(protein_residues), np.inf)
    frame_end = min(start_frame + prescreen_frames, len(u.trajectory))

    target_atoms = u.select_atoms(f"resid {resid}")
    if len(target_atoms) == 0:
        print(f"  WARNING: resid {resid} not found in universe.")
        return []

    for ts in u.trajectory[start_frame:frame_end]:
        target_pos = target_atoms.positions
        for i, res in enumerate(protein_residues):
            if res.resid == resid:
                continue
            dists        = distance_array(target_pos, res.atoms.positions)
            min_dists[i] = min(min_dists[i], dists.min())

    candidates = [
        res for i, res in enumerate(protein_residues)
        if min_dists[i] < cutoff and res.resid != resid
    ]
    print(f"    Pre-screen: {len(candidates)}/{len(protein_residues)} residues "
          f"within {cutoff:.1f} A (threshold x {prescreen_factor})")
    return candidates


# -----------------------------------------------------------------------------
# Sidechain-only prescreen
# -----------------------------------------------------------------------------

def _prescreen_candidates_sidechain(
    u,
    resid:            int,
    protein_residues: list,
    threshold:        float,
    prescreen_frames: int,
    prescreen_factor: float,
    start_frame:      int,
) -> list:
    """
    Same as _prescreen_candidates but uses only sidechain atoms
    (excludes backbone N, CA, C, O and hydrogens).
    """
    cutoff    = threshold * prescreen_factor
    min_dists = np.full(len(protein_residues), np.inf)
    frame_end = min(start_frame + prescreen_frames, len(u.trajectory))

    # sidechain atoms of the target residue
    target_atoms = u.select_atoms(
        f"resid {resid} and not name N CA C O and not name H*"
    )
    if len(target_atoms) == 0:
        # fallback to all atoms if no sidechain (e.g. GLY)
        target_atoms = u.select_atoms(f"resid {resid}")
    if len(target_atoms) == 0:
        print(f"  WARNING: resid {resid} not found in universe.")
        return []

    # build sidechain atom cache ONCE before loop
    sc_cache = {}
    for res in protein_residues:
        sc = res.atoms.select_atoms("not name N CA C O and not name H*")
        sc_cache[res.resid] = sc if len(sc) > 0 else res.atoms

    for ts in u.trajectory[start_frame:frame_end]:
        target_pos = target_atoms.positions
        for i, res in enumerate(protein_residues):
            if res.resid == resid:
                continue
            dists        = distance_array(target_pos, sc_cache[res.resid].positions)
            min_dists[i] = min(min_dists[i], dists.min())

    candidates = [
        res for i, res in enumerate(protein_residues)
        if min_dists[i] < cutoff and res.resid != resid
    ]
    print(f"    Pre-screen (sidechain): {len(candidates)}/{len(protein_residues)} residues "
          f"within {cutoff:.1f} A (threshold x {prescreen_factor})")
    return candidates


# -----------------------------------------------------------------------------
# Trajectory plot
# -----------------------------------------------------------------------------

def _make_trajectory_plot(
    df_pdb:           pd.DataFrame,
    residue_cols:     List[str],
    title:            str,
    out_path:         Path,
    threshold:        float,
    smoothing_window: Optional[int] = None,
):
    """Plot distance vs frame for each neighboring residue."""
    if not residue_cols:
        print(f"    [trajectory plot] No residues to plot for: {out_path.name}")
        return

    n      = len(residue_cols)
    cmap   = plt.cm.get_cmap("tab20", max(n, 1))
    frames = df_pdb["frame"].values

    def _draw(ax, smooth=False):
        lw = 2.0 if smooth else 0.8
        for i, col in enumerate(residue_cols):
            if col not in df_pdb.columns:
                continue
            vals = df_pdb[col].values
            if smooth and smoothing_window and smoothing_window > 1:
                vals = (pd.Series(vals)
                        .rolling(window=smoothing_window, center=True, min_periods=1)
                        .mean().values)
            ax.plot(frames, vals, label=col, linewidth=lw,
                    alpha=0.9, color=cmap(i))
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                   label=f"threshold {threshold} Å")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("Min distance (Å)", fontsize=13)
        legend = ax.legend(fontsize=8, ncol=max(1, n // 10 + 1),
                           loc="upper right", framealpha=0.6)
        for legline in legend.get_lines():
            legline.set_linewidth(lw)
        ax.tick_params(labelsize=11)

    # raw plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(title, fontsize=14)
    _draw(ax, smooth=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    [trajectory plot] Saved -> {out_path.name}")

    # smoothed plot
    if smoothing_window and smoothing_window > 1:
        smooth_path = out_path.parent / (out_path.stem + f"_smooth{smoothing_window}.png")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title(f"{title}  [smoothed, window={smoothing_window}]", fontsize=14)
        _draw(ax, smooth=True)
        plt.tight_layout()
        fig.savefig(smooth_path, dpi=150)
        plt.close(fig)
        print(f"    [trajectory plot] Saved -> {smooth_path.name}")


# -----------------------------------------------------------------------------
# Main per-residue analysis
# -----------------------------------------------------------------------------

def _analyse_residue(
    u,
    resid:            int,
    pdb_label:        str,
    protein_residues: list,
    channel_type:     str,
    start_frame:      int,
    end_frame:        int,
    threshold:        float,
    prescreen_frames: int,
    prescreen_factor: float,
    out_dir:          Path,
    smoothing_window: Optional[int],
    min_frames:       int = 50,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    pdb_csv_path = out_dir / "distances_all_pdb.csv"
    raw_csv_path = out_dir / "distances_all_raw.csv"

    # ------------------------------------------------------------------
    # Either load existing distances or compute from trajectory
    # ------------------------------------------------------------------
    if pdb_csv_path.exists():
        print(f"\n  [{pdb_label}] distances_all_pdb.csv already exists -- "
              f"skipping trajectory. Loading for plots...")
        df_pdb     = pd.read_csv(pdb_csv_path)
        pdb_labels = [c for c in df_pdb.columns if c != "frame"]
        all_dists  = df_pdb[pdb_labels].values

        if raw_csv_path.exists():
            df_raw     = pd.read_csv(raw_csv_path)
            raw_labels = [c for c in df_raw.columns if c != "frame"]
        else:
            raw_labels = pdb_labels

        n_frames = len(df_pdb)

    else:
        print(f"\n  [{pdb_label}] Pre-screening candidates...")
        candidates = _prescreen_candidates(
            u, resid, protein_residues,
            threshold, prescreen_frames, prescreen_factor, start_frame,
        )
        if not candidates:
            print(f"  No candidates found within "
                  f"{threshold * prescreen_factor:.1f} A -- skipping.")
            return

        n_frames   = end_frame - start_frame + 1
        n_cands    = len(candidates)
        raw_labels = [f"{res.resname}{res.resid}" for res in candidates]
        pdb_labels = [f"{res.resname}{to_pdb(res.resid, channel_type)}"
                      for res in candidates]
        all_dists  = np.full((n_frames, n_cands), np.nan)

        target_atoms = u.select_atoms(f"resid {resid}")

        print(f"  [{pdb_label}] Full trajectory "
              f"({n_frames} frames, {n_cands} candidate residues)...")

        for fi, ts in enumerate(tqdm(
            u.trajectory[start_frame:end_frame + 1],
            total=n_frames,
            desc=f"  {pdb_label}",
            leave=False,
        )):
            target_pos = target_atoms.positions
            for ci, res in enumerate(candidates):
                dists             = distance_array(target_pos, res.atoms.positions)
                all_dists[fi, ci] = dists.min()

        # save distance matrices
        df_raw = pd.DataFrame(all_dists, columns=raw_labels)
        df_raw.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_raw.to_csv(raw_csv_path, index=False)

        df_pdb = pd.DataFrame(all_dists, columns=pdb_labels)
        df_pdb.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_pdb.to_csv(pdb_csv_path, index=False)

    # ------------------------------------------------------------------
    # Always regenerate plots
    # ------------------------------------------------------------------
    n_frames     = len(all_dists)
    frames_close = (all_dists < threshold).sum(axis=0)
    pct_close    = frames_close / n_frames * 100.0

    # proximity summary
    summary_rows = []
    for ci, (raw_lbl, pdb_lbl) in enumerate(zip(raw_labels, pdb_labels)):
        d       = all_dists[:, ci]
        d_valid = d[~np.isnan(d)]
        if frames_close[ci] == 0:
            continue
        summary_rows.append({
            "raw_label":      raw_lbl,
            "pdb_label":      pdb_lbl,
            "frames_close":   int(frames_close[ci]),
            "pct_close":      round(float(pct_close[ci]), 3),
            "n_total_frames": int(n_frames),
            "min_dist":       round(float(d_valid.min()),  3) if len(d_valid) else np.nan,
            "mean_dist":      round(float(d_valid.mean()), 3) if len(d_valid) else np.nan,
            "max_dist":       round(float(d_valid.max()),  3) if len(d_valid) else np.nan,
        })

    if summary_rows:
        (pd.DataFrame(summary_rows)
           .sort_values("frames_close", ascending=False)
           .to_csv(out_dir / "proximity_summary.csv", index=False))
        print(f"  [{pdb_label}] {len(summary_rows)} residues ever < {threshold} A")
    else:
        print(f"  [{pdb_label}] No residues ever within {threshold} A")

    # trajectory plots
    df_pdb_full = pd.read_csv(pdb_csv_path)
    ever_close  = [c for c in pdb_labels
                   if c in df_pdb_full.columns
                   and (df_pdb_full[c] < threshold).sum() >= min_frames]

    traj_dir = out_dir / "trajectory_plots"
    traj_dir.mkdir(exist_ok=True)

    _make_trajectory_plot(
        df_pdb           = df_pdb_full,
        residue_cols     = ever_close,
        title            = f"{pdb_label} — residues ever < {threshold} Å",
        out_path         = traj_dir / "trajectory_close.png",
        threshold        = threshold,
        smoothing_window = smoothing_window,
    )

    # same plot but with raw labels
    if raw_csv_path.exists():
        df_raw_full  = pd.read_csv(raw_csv_path)
        # map pdb_label -> raw_label
        pdb_to_raw   = {pdb: raw for pdb, raw in zip(pdb_labels, raw_labels)}
        ever_close_raw = [pdb_to_raw[c] for c in ever_close if c in pdb_to_raw]
        _make_trajectory_plot(
            df_pdb           = df_raw_full,
            residue_cols     = ever_close_raw,
            title            = f"{pdb_label} — residues ever < {threshold} Å (raw labels)",
            out_path         = traj_dir / "trajectory_close_raw.png",
            threshold        = threshold,
            smoothing_window = smoothing_window,
        )



# -----------------------------------------------------------------------------
# Sidechain-only analysis
# -----------------------------------------------------------------------------

def _analyse_residue_sidechain(
    u,
    resid:            int,
    pdb_label:        str,
    protein_residues: list,
    channel_type:     str,
    start_frame:      int,
    end_frame:        int,
    threshold:        float,
    prescreen_frames: int,
    prescreen_factor: float,
    out_dir:          Path,
    smoothing_window: Optional[int],
    min_frames:       int = 50,
):
    """
    Same as _analyse_residue but computes minimum distance between
    sidechain atoms only (excludes backbone N, CA, C, O and hydrogens).
    Results stored in out_dir/sidechain/.
    """
    sc_dir = out_dir / "sidechain"
    sc_dir.mkdir(parents=True, exist_ok=True)

    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    pdb_csv_path = sc_dir / "distances_all_pdb.csv"
    raw_csv_path = sc_dir / "distances_all_raw.csv"

    # sidechain atoms of target
    target_atoms = u.select_atoms(
        f"resid {resid} and not name N CA C O and not name H*"
    )
    if len(target_atoms) == 0:
        target_atoms = u.select_atoms(f"resid {resid}")

    if pdb_csv_path.exists():
        print(f"\n  [{pdb_label}][sidechain] distances_all_pdb.csv already exists -- "
              f"skipping trajectory. Loading for plots...")
        df_pdb     = pd.read_csv(pdb_csv_path)
        pdb_labels = [c for c in df_pdb.columns if c != "frame"]
        all_dists  = df_pdb[pdb_labels].values

        if raw_csv_path.exists():
            df_raw     = pd.read_csv(raw_csv_path)
            raw_labels = [c for c in df_raw.columns if c != "frame"]
        else:
            raw_labels = pdb_labels

        n_frames = len(df_pdb)

    else:
        print(f"\n  [{pdb_label}][sidechain] Pre-screening candidates...")
        candidates = _prescreen_candidates_sidechain(
            u, resid, protein_residues,
            threshold, prescreen_frames, prescreen_factor, start_frame,
        )
        if not candidates:
            print(f"  No sidechain candidates found within "
                  f"{threshold * prescreen_factor:.1f} A -- skipping.")
            return

        n_frames   = end_frame - start_frame + 1
        n_cands    = len(candidates)
        raw_labels = [f"{res.resname}{res.resid}" for res in candidates]
        pdb_labels = [f"{res.resname}{to_pdb(res.resid, channel_type)}"
                      for res in candidates]
        all_dists  = np.full((n_frames, n_cands), np.nan)

        print(f"  [{pdb_label}][sidechain] Full trajectory "
              f"({n_frames} frames, {n_cands} candidate residues)...")

        # build sidechain cache for candidates ONCE before loop
        sc_cache_cands = {}
        for res in candidates:
            sc = res.atoms.select_atoms("not name N CA C O and not name H*")
            sc_cache_cands[res.resid] = sc if len(sc) > 0 else res.atoms

        for fi, ts in enumerate(tqdm(
            u.trajectory[start_frame:end_frame + 1],
            total=n_frames,
            desc=f"  {pdb_label} SC",
            leave=False,
        )):
            target_pos = target_atoms.positions
            for ci, res in enumerate(candidates):
                dists             = distance_array(target_pos, sc_cache_cands[res.resid].positions)
                all_dists[fi, ci] = dists.min()

        df_raw = pd.DataFrame(all_dists, columns=raw_labels)
        df_raw.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_raw.to_csv(raw_csv_path, index=False)

        df_pdb = pd.DataFrame(all_dists, columns=pdb_labels)
        df_pdb.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_pdb.to_csv(pdb_csv_path, index=False)

    # plots
    n_frames     = len(all_dists)
    frames_close = (all_dists < threshold).sum(axis=0)
    pct_close    = frames_close / n_frames * 100.0

    summary_rows = []
    for ci, (raw_lbl, pdb_lbl) in enumerate(zip(raw_labels, pdb_labels)):
        d       = all_dists[:, ci]
        d_valid = d[~np.isnan(d)]
        if frames_close[ci] == 0:
            continue
        summary_rows.append({
            "raw_label":      raw_lbl,
            "pdb_label":      pdb_lbl,
            "frames_close":   int(frames_close[ci]),
            "pct_close":      round(float(pct_close[ci]), 3),
            "n_total_frames": int(n_frames),
            "min_dist":       round(float(d_valid.min()),  3) if len(d_valid) else np.nan,
            "mean_dist":      round(float(d_valid.mean()), 3) if len(d_valid) else np.nan,
            "max_dist":       round(float(d_valid.max()),  3) if len(d_valid) else np.nan,
        })

    if summary_rows:
        (pd.DataFrame(summary_rows)
           .sort_values("frames_close", ascending=False)
           .to_csv(sc_dir / "proximity_summary.csv", index=False))
        print(f"  [{pdb_label}][sidechain] {len(summary_rows)} residues ever < {threshold} A")
    else:
        print(f"  [{pdb_label}][sidechain] No residues ever within {threshold} A")

    df_pdb_full = pd.read_csv(pdb_csv_path)
    ever_close  = [c for c in pdb_labels
                   if c in df_pdb_full.columns
                   and (df_pdb_full[c] < threshold).sum() >= min_frames]

    traj_dir = sc_dir / "trajectory_plots"
    traj_dir.mkdir(exist_ok=True)

    _make_trajectory_plot(
        df_pdb           = df_pdb_full,
        residue_cols     = ever_close,
        title            = f"{pdb_label} — sidechain residues ever < {threshold} Å",
        out_path         = traj_dir / "trajectory_close.png",
        threshold        = threshold,
        smoothing_window = smoothing_window,
    )

    if raw_csv_path.exists():
        df_raw_full    = pd.read_csv(raw_csv_path)
        pdb_to_raw     = {pdb: raw for pdb, raw in zip(pdb_labels, raw_labels)}
        ever_close_raw = [pdb_to_raw[c] for c in ever_close if c in pdb_to_raw]
        _make_trajectory_plot(
            df_pdb           = df_raw_full,
            residue_cols     = ever_close_raw,
            title            = f"{pdb_label} — sidechain residues ever < {threshold} Å (raw labels)",
            out_path         = traj_dir / "trajectory_close_raw.png",
            threshold        = threshold,
            smoothing_window = smoothing_window,
        )


# -----------------------------------------------------------------------------
# H-bond analysis for one residue vs all neighbors
# -----------------------------------------------------------------------------

HBOND_ELEMENTS = {"N", "O", "S"}
HBOND_COVALENT = 1.5


def _find_donor_heavy_nb(h_atom, donor_heavy_list):
    """Find heavy atom covalently bonded to hydrogen."""
    try:
        bonded = [a for a in h_atom.bonded_atoms
                  if a.element.upper() in HBOND_ELEMENTS]
        if bonded:
            return bonded[0]
    except Exception:
        pass
    best_dist, best_atom = HBOND_COVALENT, None
    H = h_atom.position
    for heavy in donor_heavy_list:
        if heavy.element.upper() not in HBOND_ELEMENTS:
            continue
        d = float(np.linalg.norm(heavy.position - H))
        if d < best_dist:
            best_dist, best_atom = d, heavy
    return best_atom


def _find_hbonds_residue_vs_neighbor(
    target_atoms, target_label,
    neighbor_atoms, neighbor_label,
    hbond_dist, hbond_angle,
):
    """Find all H-bonds between target and one neighbor residue."""
    bonds = []
    for donor_grp, donor_lbl, acceptor_grp, acceptor_lbl in [
        (target_atoms,   target_label,   neighbor_atoms, neighbor_label),
        (neighbor_atoms, neighbor_label, target_atoms,   target_label),
    ]:
        donor_heavy = [a for a in donor_grp if not a.name.startswith("H")]
        for h in donor_grp:
            if not h.name.startswith("H"):
                continue
            D_atom = _find_donor_heavy_nb(h, donor_heavy)
            if D_atom is None:
                continue
            D     = D_atom.position
            H_pos = h.position
            for acc in acceptor_grp:
                if acc.element.upper() not in HBOND_ELEMENTS:
                    continue
                if acc.name.startswith("H"):
                    continue
                A       = acc.position
                dist_HA = float(np.linalg.norm(H_pos - A))
                if dist_HA > hbond_dist:
                    continue
                v1    = D - H_pos
                v2    = A - H_pos
                denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
                angle = float(np.degrees(
                    np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))
                ))
                if angle >= hbond_angle:
                    bonds.append({
                        "donor_res":     donor_lbl,
                        "donor_atom":    D_atom.name,
                        "hydrogen":      h.name,
                        "acceptor_res":  acceptor_lbl,
                        "acceptor_atom": acc.name,
                        "dist_HA_A":     round(dist_HA, 3),
                        "angle_DHA_deg": round(angle, 1),
                    })
    return bonds



def _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames, classify_fn=None):
    """
    Generate all H-bond bar plots with EXACT frame counts.

    For all-types: reads hbond_residue_summary.csv (exact, one count per partner)
    For bond-type breakdowns: reads hbond_type_per_frame.csv from each pair folder
        and counts frames where each type > 0 — no overcounting.
    """

    def _barplot(partner_counts, title, out_path, color):
        if not partner_counts:
            print(f"    No data for {out_path.name} -- skipping.")
            return
        partners = sorted(partner_counts.keys(), key=lambda p: -partner_counts[p])
        vals     = [partner_counts[p] for p in partners]
        fig_w    = max(8, len(partners) * 0.7)
        fig, ax  = plt.subplots(figsize=(fig_w, 5))
        x        = np.arange(len(partners))
        bars     = ax.bar(x, vals, color=color, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            pct = round(100.0 * val / n_frames, 1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val}\n({pct}%)", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Frames with H-bond", fontsize=13)
        ax.set_xlabel("Partner residue", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(top=max(vals) * 1.2)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path.name}")

    # -- all types: exact counts from residue_summary --
    counts_all = {}
    if residue_csv.exists():
        df_res = pd.read_csv(residue_csv)
        counts_all = {row["partner"]: int(row["n_frames_hbond"]) for _, row in df_res.iterrows()}

    # -- bond-type breakdowns: read hbond_type_per_frame.csv from pair folders --
    counts_bb_bb  = {}
    counts_sc_sc  = {}
    counts_bb_sc  = {}
    counts_sc_any = {}

    safe_label = pdb_label.replace(".", "_")
    for pair_dir in sorted(hb_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        # folder name: e.g. 197_2b_LYS200_2b
        # extract partner label: everything after safe_label + "_"
        folder = pair_dir.name
        prefix = safe_label + "_"
        if not folder.startswith(prefix):
            continue
        partner_raw = folder[len(prefix):]
        # convert LYS200_2b -> LYS200.2b (last _ becomes .)
        parts = partner_raw.rsplit("_", 1)
        partner_lbl = f"{parts[0]}.{parts[1]}" if len(parts) == 2 else partner_raw

        type_csv = pair_dir / "hbond_type_per_frame.csv"
        if not type_csv.exists():
            continue

        df_t = pd.read_csv(type_csv)
        n_bb  = int(df_t["bb_bb"].sum())
        n_sc  = int(df_t["sc_sc"].sum())
        n_bs  = int(df_t["bb_sc"].sum())
        sc_bb = df_t["sc_bb"].values if "sc_bb" in df_t.columns else np.zeros(len(df_t), dtype=int)
        n_sa  = int(np.clip(df_t["sc_sc"].values + df_t["bb_sc"].values + sc_bb, 0, 1).sum())

        if n_bb > 0:  counts_bb_bb[partner_lbl]  = n_bb
        if n_sc > 0:  counts_sc_sc[partner_lbl]  = n_sc
        if n_bs > 0:  counts_bb_sc[partner_lbl]  = n_bs
        if n_sa > 0:  counts_sc_any[partner_lbl] = n_sa

    _barplot(counts_all,   f"{pdb_label} — H-bond partner frequency (all types)",
             hb_dir / "hbond_partner_barplot.png",        "steelblue")
    _barplot(counts_bb_bb, f"{pdb_label} — backbone↔backbone H-bonds",
             hb_dir / "hbond_partner_barplot_bb_bb.png",  "steelblue")
    _barplot(counts_sc_sc, f"{pdb_label} — sidechain↔sidechain H-bonds",
             hb_dir / "hbond_partner_barplot_sc_sc.png",  "darkorange")
    _barplot(counts_bb_sc, f"{pdb_label} — backbone↔sidechain H-bonds",
             hb_dir / "hbond_partner_barplot_bb_sc.png",  "mediumseagreen")
    _barplot(counts_sc_any,f"{pdb_label} — sidechain involved H-bonds (sc-sc + bb-sc)",
             hb_dir / "hbond_partner_barplot_sc_any.png", "mediumpurple")




def _hbond_vs_hbc_threshold(
    hb_dir:      Path,
    pdb_label:   str,
    results_dir: Path,
    thresholds:  list = None,
):
    """
    For each HBC threshold, find frames where A_C >= threshold OR B_D >= threshold
    and compute H-bond partner frequencies for those frames.
    Shows how H-bonds change as gate opens more.

    Reads:
        hb_dir/hbond_per_frame.csv  (new format with bb_bb_partners etc.)
        results_dir/hbc_distances.csv

    Output: hb_dir/hbc_threshold_analysis/
        AC/  BD/
            plot_all.png
            plot_bb_bb.png
            plot_sc_sc.png
            plot_bb_sc.png
            plot_sc_any.png
            data_all.csv  (threshold x partner matrix)
            ...
    """
    if thresholds is None:
        thresholds = [5, 6, 7, 8, 9, 10, 11, 12]

    per_frame_csv = hb_dir / "hbond_per_frame.csv"
    hbc_csv       = results_dir / "hbc_distances.csv"

    if not per_frame_csv.exists():
        print(f"    [hbc_threshold] hbond_per_frame.csv not found -- skipping.")
        return
    if not hbc_csv.exists():
        print(f"    [hbc_threshold] hbc_distances.csv not found -- skipping.")
        return

    df_hb  = pd.read_csv(per_frame_csv)
    df_hbc = pd.read_csv(hbc_csv)

    # check new format
    if "bb_bb_partners" not in df_hb.columns:
        print(f"    [hbc_threshold] old format hbond_per_frame.csv -- skipping.")
        return

    # merge on frame
    merged = pd.merge(df_hb, df_hbc[["frame", "A_C", "B_D"]], on="frame", how="inner")
    if len(merged) < 10:
        print(f"    [hbc_threshold] too few overlapping frames -- skipping.")
        return

    out_dir = hb_dir / "hbc_threshold_analysis"
    out_dir.mkdir(exist_ok=True)

    bond_cols = {
        "all":    "partners",
        "bb_bb":  "bb_bb_partners",
        "sc_sc":  "sc_sc_partners",
        "bb_sc":  "bb_sc_partners",
        "sc_any": "sc_any_partners",
    }
    bond_titles = {
        "all":    "All H-bonds",
        "bb_bb":  "Backbone-Backbone",
        "sc_sc":  "Sidechain-Sidechain",
        "bb_sc":  "Backbone-Sidechain",
        "sc_any": "Sidechain Involved",
    }
    bond_colors = {
        "all":    "steelblue",
        "bb_bb":  "steelblue",
        "sc_sc":  "darkorange",
        "bb_sc":  "mediumseagreen",
        "sc_any": "mediumpurple",
    }

    n_total = len(merged)

    for gate_col, gate_name in [("A_C", "AC"), ("B_D", "BD")]:
        gate_dir = out_dir / gate_name
        gate_dir.mkdir(exist_ok=True)

        for bt, col in bond_cols.items():
            if col not in merged.columns:
                continue

            # build matrix: {partner: {threshold: pct}}
            partner_data = {}  # partner -> list of pcts per threshold

            for thr in thresholds:
                # frames where gate >= threshold
                sub = merged[merged[gate_col] >= thr]
                n_sub = len(sub)
                if n_sub == 0:
                    for p in partner_data:
                        partner_data[p].append(0.0)
                    continue

                # count partner frequencies
                partner_counts = {}
                for _, row in sub.iterrows():
                    partners_str = str(row[col])
                    if partners_str == "nan" or not partners_str.strip():
                        continue
                    for p in partners_str.split("|"):
                        p = p.strip()
                        if p:
                            partner_counts[p] = partner_counts.get(p, 0) + 1

                # update partner_data
                all_partners = set(list(partner_data.keys()) + list(partner_counts.keys()))
                for p in all_partners:
                    if p not in partner_data:
                        partner_data[p] = [0.0] * thresholds.index(thr)
                    pct = round(100.0 * partner_counts.get(p, 0) / n_sub, 2)
                    partner_data[p].append(pct)

            # pad any partners with missing values
            for p in partner_data:
                while len(partner_data[p]) < len(thresholds):
                    partner_data[p].append(0.0)

            if not partner_data:
                continue

            # save data CSV
            df_out = pd.DataFrame(partner_data, index=thresholds)
            df_out.index.name = "threshold_A"
            df_out.to_csv(gate_dir / f"data_{bt}.csv")

            # only plot partners that reach at least 5% at any threshold
            partners_to_plot = [
                p for p, vals in partner_data.items()
                if max(vals) >= 5.0
            ]
            if not partners_to_plot:
                partners_to_plot = list(partner_data.keys())[:10]

            # plot
            n = len(partners_to_plot)
            if n == 0:
                continue

            cmap = plt.cm.get_cmap("tab20", max(n, 1))
            fig, ax = plt.subplots(figsize=(10, 5))

            for i, p in enumerate(partners_to_plot):
                vals = partner_data[p]
                ax.plot(thresholds, vals, marker="o", linewidth=2.0,
                        label=p, color=cmap(i), alpha=0.9)

            ax.set_xlabel(f"HBC {gate_name} threshold (Å)", fontsize=13)
            ax.set_ylabel("% frames with H-bond", fontsize=13)
            ax.set_title(
                f"{pdb_label} — {bond_titles[bt]}\n"
                f"H-bonds when {gate_name} gate >= threshold",
                fontsize=13
            )
            ax.set_xticks(thresholds)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.6,
                      ncol=max(1, n // 8 + 1))
            ax.tick_params(labelsize=11)
            ax.set_ylim(0, 105)
            plt.tight_layout()
            fig.savefig(gate_dir / f"plot_{bt}.png", dpi=150)
            plt.close(fig)

        print(f"    [hbc_threshold] Saved {gate_name} plots -> {gate_dir}")

    print(f"    [hbc_threshold] Done -> {out_dir}")


def _regenerate_pair_plots(hb_dir, pdb_label, partners_with_hbonds, frames, n_frames, smoothing_window):
    """Regenerate all per-pair state plots from cached CSVs."""
    _BB_ATOMS = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3"}

    def _bond_type(da, aa):
        d_bb = da in _BB_ATOMS
        a_bb = aa in _BB_ATOMS
        if d_bb and a_bb:       return "bb_bb"
        elif d_bb and not a_bb: return "bb_sc"
        elif not d_bb and a_bb: return "sc_bb"
        else:                   return "sc_sc"

    def _make_state_plot(binary, frames, title, out_path, sw, color="steelblue"):
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.fill_between(frames, binary, step="mid", alpha=0.4, color=color, label="H-bond present")
        ax.plot(frames, binary, drawstyle="steps-mid", linewidth=0.5, color=color, alpha=0.6)
        if sw and sw > 1:
            smooth = (pd.Series(binary.astype(float))
                      .rolling(window=sw, center=True, min_periods=1).mean().values)
            ax.plot(frames, smooth, linewidth=2.0, color="navy", label=f"fraction (w={sw})")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("H-bond", fontsize=13)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No", "Yes"], fontsize=12)
        ax.set_ylim(-0.1, 1.3)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.6)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    safe_label = pdb_label.replace(".", "_")
    n_done = 0
    for lbl in partners_with_hbonds:
        safe_lbl = lbl.replace(".", "_")
        pair_dir = hb_dir / f"{safe_label}_{safe_lbl}"
        if not pair_dir.exists():
            continue

        # load binary from hbond_per_frame in pair folder
        pf_csv = pair_dir / "hbond_per_frame.csv"
        if not pf_csv.exists():
            continue
        df_pf  = pd.read_csv(pf_csv)
        binary = df_pf["hbond_present"].values.astype(int)

        base_title = f"H-bond: {pdb_label} ↔ {lbl}  ({int(binary.sum())} frames, {100*binary.mean():.1f}%)"
        _make_state_plot(binary, frames, base_title,
                         pair_dir / "hbond_state.png", smoothing_window)

        type_csv = pair_dir / "hbond_type_per_frame.csv"
        if type_csv.exists():
            df_t   = pd.read_csv(type_csv)
            bb_bb  = df_t["bb_bb"].values.astype(int)
            sc_sc  = df_t["sc_sc"].values.astype(int)
            bb_sc  = df_t["bb_sc"].values.astype(int)
            sc_bb  = df_t["sc_bb"].values.astype(int) if "sc_bb" in df_t.columns else np.zeros(n_frames, dtype=int)
            sc_any = np.clip(sc_sc + bb_sc + sc_bb, 0, 1)

            if bb_bb.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(bb_bb.sum())} frames, {100*bb_bb.mean():.1f}%) | backbone-backbone"
                _make_state_plot(bb_bb, frames, t,
                                 pair_dir / "hbond_state_bb_bb.png", smoothing_window, "steelblue")
            if sc_sc.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(sc_sc.sum())} frames, {100*sc_sc.mean():.1f}%) | sidechain-sidechain"
                _make_state_plot(sc_sc, frames, t,
                                 pair_dir / "hbond_state_sc_sc.png", smoothing_window, "darkorange")
            if bb_sc.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(bb_sc.sum())} frames, {100*bb_sc.mean():.1f}%) | backbone->sidechain"
                _make_state_plot(bb_sc, frames, t,
                                 pair_dir / "hbond_state_bb_sc.png", smoothing_window, "mediumseagreen")
            if sc_any.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(sc_any.sum())} frames, {100*sc_any.mean():.1f}%) | sidechain involved"
                _make_state_plot(sc_any, frames, t,
                                 pair_dir / "hbond_state_sc_any.png", smoothing_window, "mediumpurple")

        ap_csv = pair_dir / "hbond_atom_pair_per_frame.csv"
        if ap_csv.exists():
            df_ap = pd.read_csv(ap_csv)
            for col in [c for c in df_ap.columns if c != "frame"]:
                arr = df_ap[col].values.astype(int)
                if arr.sum() == 0:
                    continue
                da, aa = col.split("_", 1)
                bt     = _bond_type(da, aa)
                _make_state_plot(arr, frames, f"{base_title} | {da}->{aa} ({bt})",
                                 pair_dir / f"hbond_state_type_{col}.png",
                                 smoothing_window, "gray")
        n_done += 1

    print(f"    Regenerated plots for {n_done} pair folders")

def _analyse_residue_hbonds(
    u,
    resid,
    pdb_label,
    candidates,
    pdb_labels_cands,
    channel_type,
    start_frame,
    end_frame,
    out_dir,
    smoothing_window,
    hbond_dist=3.5,
    hbond_angle=120.0,
):
    from collections import Counter
    hb_dir = out_dir / "hbonds"
    hb_dir.mkdir(parents=True, exist_ok=True)

    atom_pair_csv = hb_dir / "hbond_atom_pair_summary.csv"
    residue_csv   = hb_dir / "hbond_residue_summary.csv"
    per_frame_csv = hb_dir / "hbond_per_frame.csv"

    n_frames = end_frame - start_frame + 1
    frames   = np.arange(start_frame, end_frame + 1)

    # if called with empty candidates and CSVs exist in NEW format — just make bar plots
    if len(candidates) == 0 and atom_pair_csv.exists() and residue_csv.exists():
        if per_frame_csv.exists():
            df_check = pd.read_csv(per_frame_csv)
            if "bb_bb_partners" in df_check.columns:
                # new format already exists — regenerate ALL plots
                print(f"\n  [{pdb_label}][hbonds] New-format CSV exists -- regenerating all plots...")
                _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames)
                # also regenerate per-pair state plots
                if residue_csv.exists():
                    df_res = pd.read_csv(residue_csv)
                    partners_with_hbonds = df_res["partner"].tolist()
                    _regenerate_pair_plots(hb_dir, pdb_label, partners_with_hbonds,
                                           np.arange(start_frame, end_frame+1), n_frames, smoothing_window)
                _hbond_vs_hbc_threshold(
                    hb_dir      = hb_dir,
                    pdb_label   = pdb_label,
                    results_dir = out_dir.parent.parent,
                )
                return
            else:
                # old format — need to rescan but we have no candidates!
                print(f"\n  [{pdb_label}][hbonds] Old-format CSV detected but no candidates available.")
                print(f"    Please run with full candidates (delete hbond_per_frame.csv to force rescan).")
                _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames)
                return
        else:
            print(f"\n  [{pdb_label}][hbonds] Regenerating bar plots from existing CSVs...")
            _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames)
            return

    _BB_ATOMS = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3"}

    def _bond_type(da, aa):
        d_bb = da in _BB_ATOMS
        a_bb = aa in _BB_ATOMS
        if d_bb and a_bb:       return "bb_bb"
        elif d_bb and not a_bb: return "bb_sc"
        elif not d_bb and a_bb: return "sc_bb"
        else:                   return "sc_sc"

    def _make_state_plot(binary, frames, title, out_path, sw, color="steelblue"):
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.fill_between(frames, binary, step="mid", alpha=0.4, color=color, label="H-bond present")
        ax.plot(frames, binary, drawstyle="steps-mid", linewidth=0.5, color=color, alpha=0.6)
        if sw and sw > 1:
            smooth = (pd.Series(binary.astype(float))
                      .rolling(window=sw, center=True, min_periods=1).mean().values)
            ax.plot(frames, smooth, linewidth=2.0, color="navy", label=f"fraction (w={sw})")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("H-bond", fontsize=13)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No", "Yes"], fontsize=12)
        ax.set_ylim(-0.1, 1.3)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.6)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Check if all pair CSVs exist — if so skip scan entirely
    # ------------------------------------------------------------------
    all_cached = False
    if residue_csv.exists() and atom_pair_csv.exists() and per_frame_csv.exists():
        df_pf_check = pd.read_csv(per_frame_csv)
        # new format has bb_bb_partners column — check for it
        if "bb_bb_partners" in df_pf_check.columns:
            all_cached = True
            df_res = pd.read_csv(residue_csv)
            partners_with_hbonds = df_res["partner"].tolist()
            print(f"\n  [{pdb_label}][hbonds] New-format CSVs exist -- skipping scan. Regenerating plots...")

    if all_cached:
        # load pair_binary from new-format hbond_per_frame.csv
        df_pf = pd.read_csv(per_frame_csv)
        pair_binary = {}
        for lbl in partners_with_hbonds:
            pair_binary[lbl] = np.zeros(n_frames, dtype=int)
        for fi, row in df_pf.iterrows():
            if row["partners"] and str(row["partners"]) != "nan":
                for p in str(row["partners"]).split("|"):
                    if p in pair_binary:
                        pair_binary[p][fi] = 1
        df_partners = pd.read_csv(residue_csv)

    else:
        # ------------------------------------------------------------------
        # Full scan
        # ------------------------------------------------------------------
        # atom_pair_csv alone exists? just make bar plots and return
        if atom_pair_csv.exists() and not residue_csv.exists():
            _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames)
            return

        if atom_pair_csv.exists() and residue_csv.exists() and per_frame_csv.exists():
            # main CSVs exist but type CSVs missing — need to scan for type info
            print(f"\n  [{pdb_label}][hbonds] Type CSVs missing -- scanning to build them...")
        else:
            print(f"\n  [{pdb_label}][hbonds] Scanning {n_frames} frames ({len(candidates)} candidates)...")

        target_atoms      = u.select_atoms(f"resid {resid}")
        per_frame_rows    = []
        pair_binary       = {lbl: np.zeros(n_frames, dtype=int) for lbl in pdb_labels_cands}
        atom_pair_counter = Counter()
        # per-pair type storage: lbl -> {bt -> array}
        pair_type_frames  = {lbl: {"bb_bb": np.zeros(n_frames, dtype=int),
                                    "sc_sc": np.zeros(n_frames, dtype=int),
                                    "bb_sc": np.zeros(n_frames, dtype=int),
                                    "sc_bb": np.zeros(n_frames, dtype=int)}
                             for lbl in pdb_labels_cands}
        # per-pair atom-pair storage: lbl -> {(da,aa) -> array}
        pair_atom_frames  = {lbl: {} for lbl in pdb_labels_cands}

        for fi, ts in enumerate(tqdm(
            u.trajectory[start_frame:end_frame + 1],
            total=n_frames, desc=f"  {pdb_label} HB", leave=False,
        )):
            frame          = int(frames[fi])
            any_hbond      = 0
            frame_partners = []
            target_pos     = target_atoms.positions

            for cand, cand_lbl in zip(candidates, pdb_labels_cands):
                cand_pos = cand.atoms.positions
                dists    = distance_array(target_pos, cand_pos)
                if dists.min() > hbond_dist:
                    continue
                bonds = _find_hbonds_residue_vs_neighbor(
                    target_atoms, pdb_label, cand.atoms, cand_lbl,
                    hbond_dist, hbond_angle,
                )
                if bonds:
                    pair_binary[cand_lbl][fi] = 1
                    any_hbond = 1
                    frame_partners.append(cand_lbl)
                    for b in bonds:
                        atom_pair_counter[(
                            b["donor_res"], b["donor_atom"], b["hydrogen"],
                            b["acceptor_res"], b["acceptor_atom"],
                        )] += 1
                        bt  = _bond_type(b["donor_atom"], b["acceptor_atom"])
                        pair_type_frames[cand_lbl][bt][fi] = 1
                        key = (b["donor_atom"], b["acceptor_atom"])
                        if key not in pair_atom_frames[cand_lbl]:
                            pair_atom_frames[cand_lbl][key] = np.zeros(n_frames, dtype=int)
                        pair_atom_frames[cand_lbl][key][fi] = 1

            # collect per-bond-type partner lists for this frame
            bb_bb_p = []
            sc_sc_p = []
            bb_sc_p = []
            sc_any_p = []
            for lbl in frame_partners:
                tf = pair_type_frames[lbl]
                if tf["bb_bb"][fi]: bb_bb_p.append(lbl)
                if tf["sc_sc"][fi]: sc_sc_p.append(lbl)
                if tf["bb_sc"][fi] or tf["sc_bb"][fi]: bb_sc_p.append(lbl)
                if tf["sc_sc"][fi] or tf["bb_sc"][fi] or tf["sc_bb"][fi]: sc_any_p.append(lbl)

            per_frame_rows.append({
                "frame":           frame,
                "any_hbond":       any_hbond,
                "n_partners":      len(frame_partners),
                "partners":        "|".join(frame_partners) if frame_partners else "",
                "bb_bb_partners":  "|".join(bb_bb_p)  if bb_bb_p  else "",
                "sc_sc_partners":  "|".join(sc_sc_p)  if sc_sc_p  else "",
                "bb_sc_partners":  "|".join(bb_sc_p)  if bb_sc_p  else "",
                "sc_any_partners": "|".join(sc_any_p) if sc_any_p else "",
            })

        # save main CSVs
        pd.DataFrame(per_frame_rows).to_csv(per_frame_csv, index=False)
        print(f"    Saved -> hbond_per_frame.csv")

        if atom_pair_counter:
            atom_rows = []
            for (dr, da, hy, ar, aa), cnt in atom_pair_counter.most_common():
                atom_rows.append({"donor_res": dr, "donor_atom": da, "hydrogen": hy,
                                   "acceptor_res": ar, "acceptor_atom": aa,
                                   "n_frames": cnt,
                                   "pct_frames": round(100.0 * cnt / n_frames, 2)})
            pd.DataFrame(atom_rows).to_csv(atom_pair_csv, index=False)
            print(f"    Saved -> hbond_atom_pair_summary.csv")

        partner_summary = []
        for lbl in pdb_labels_cands:
            n_with = int(pair_binary[lbl].sum())
            if n_with > 0:
                partner_summary.append({"partner": lbl,
                                         "n_frames_hbond": n_with,
                                         "pct_frames": round(100.0 * n_with / n_frames, 2)})
        partner_summary.sort(key=lambda x: -x["n_frames_hbond"])
        df_partners = pd.DataFrame(partner_summary)

        if df_partners.empty:
            print(f"    [{pdb_label}][hbonds] No H-bonds found.")
            return

        df_partners.to_csv(residue_csv, index=False)
        print(f"    Saved -> hbond_residue_summary.csv ({len(df_partners)} partners)")

        # save per-pair type and atom-pair CSVs
        for lbl in pdb_labels_cands:
            if pair_binary[lbl].sum() == 0:
                continue
            safe_lbl = lbl.replace(".", "_")
            pair_dir = hb_dir / f"{pdb_label.replace('.','_')}_{safe_lbl}"
            pair_dir.mkdir(exist_ok=True)

            # hbond_type_per_frame.csv
            tf = pair_type_frames[lbl]
            pd.DataFrame({"frame": frames,
                          "bb_bb": tf["bb_bb"], "sc_sc": tf["sc_sc"],
                          "bb_sc": tf["bb_sc"], "sc_bb": tf["sc_bb"],
                          }).to_csv(pair_dir / "hbond_type_per_frame.csv", index=False)

            # hbond_atom_pair_per_frame.csv
            ap = pair_atom_frames[lbl]
            if ap:
                df_ap = pd.DataFrame({"frame": frames})
                for (da, aa), arr in ap.items():
                    df_ap[f"{da}_{aa}"] = arr
                df_ap.to_csv(pair_dir / "hbond_atom_pair_per_frame.csv", index=False)

        print(f"    Saved per-pair type CSVs for {len(df_partners)} partners")
        partners_with_hbonds = df_partners["partner"].tolist()

    # ------------------------------------------------------------------
    # Bar plots (always regenerate)
    # ------------------------------------------------------------------
    _make_hbond_barplots(hb_dir, pdb_label, atom_pair_csv, residue_csv, n_frames)

    # ------------------------------------------------------------------
    # Per-pair state plots (always regenerate from cached CSVs)
    # ------------------------------------------------------------------
    n_pair_folders = 0
    for lbl in partners_with_hbonds:
        binary   = pair_binary.get(lbl, np.zeros(n_frames, dtype=int))
        safe_lbl = lbl.replace(".", "_")
        pair_dir = hb_dir / f"{pdb_label.replace('.','_')}_{safe_lbl}"
        pair_dir.mkdir(exist_ok=True)

        base_title = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(binary.sum())} frames, {100*binary.mean():.1f}%)"

        # Plot 1 -- all
        _make_state_plot(binary, frames, base_title,
                         pair_dir / "hbond_state.png", smoothing_window)

        # load type per frame
        type_csv = pair_dir / "hbond_type_per_frame.csv"
        if type_csv.exists():
            df_t  = pd.read_csv(type_csv)
            bb_bb = df_t["bb_bb"].values.astype(int)
            sc_sc = df_t["sc_sc"].values.astype(int)
            bb_sc = df_t["bb_sc"].values.astype(int)
            sc_bb = df_t.get("sc_bb", pd.Series(np.zeros(n_frames))).values.astype(int)
            sc_any = np.clip(sc_sc + bb_sc + sc_bb, 0, 1)

            if bb_bb.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(bb_bb.sum())} frames, {100*bb_bb.mean():.1f}%) | backbone-backbone"
                _make_state_plot(bb_bb, frames, t,
                                 pair_dir / "hbond_state_bb_bb.png", smoothing_window, "steelblue")
            if sc_sc.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(sc_sc.sum())} frames, {100*sc_sc.mean():.1f}%) | sidechain-sidechain"
                _make_state_plot(sc_sc, frames, t,
                                 pair_dir / "hbond_state_sc_sc.png", smoothing_window, "darkorange")
            if bb_sc.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(bb_sc.sum())} frames, {100*bb_sc.mean():.1f}%) | backbone->sidechain"
                _make_state_plot(bb_sc, frames, t,
                                 pair_dir / "hbond_state_bb_sc.png", smoothing_window, "mediumseagreen")
            if sc_any.sum() > 0:
                t = f"H-bond: {pdb_label} \u2194 {lbl}  ({int(sc_any.sum())} frames, {100*sc_any.mean():.1f}%) | sidechain involved"
                _make_state_plot(sc_any, frames, t,
                                 pair_dir / "hbond_state_sc_any.png", smoothing_window, "mediumpurple")

        # load atom-pair per frame
        ap_csv = pair_dir / "hbond_atom_pair_per_frame.csv"
        if ap_csv.exists():
            df_ap = pd.read_csv(ap_csv)
            for col in [c for c in df_ap.columns if c != "frame"]:
                arr = df_ap[col].values.astype(int)
                if arr.sum() == 0:
                    continue
                da, aa  = col.split("_", 1)
                bt      = _bond_type(da, aa)
                _make_state_plot(arr, frames, f"{base_title} | {da}->{aa} ({bt})",
                                 pair_dir / f"hbond_state_type_{col}.png",
                                 smoothing_window, "gray")
        n_pair_folders += 1

    print(f"    Saved per-pair plots for {n_pair_folders} partners")

    # H-bonds vs HBC threshold analysis
    _hbond_vs_hbc_threshold(
        hb_dir      = hb_dir,
        pdb_label   = pdb_label,
        results_dir = out_dir.parent.parent,  # residue_neighbors/../ = results_dir
    )



def _analyse_residue_neighbor_correlations(
    pdb_label:   str,
    out_dir:     Path,
    results_dir: Path,
    min_r:       float = 0.3,
):
    """
    Compute Spearman correlation between distance(197->neighbor) and HBC gate distances.

    For each neighbor residue:
        r_AC = Spearman(dist(197->neighbor), A_C gate distance)
        r_BD = Spearman(dist(197->neighbor), B_D gate distance)

    Reads:
        out_dir/distances_all_pdb.csv   -- neighbor distances
        results_dir/hbc_distances.csv   -- gate distances

    Output: out_dir/neighbor_correlations/
        neighbor_gate_correlation.csv
        neighbor_gate_correlation_filtered_r<min_r>.csv
    """
    from scipy.stats import spearmanr

    pdb_csv = out_dir / "distances_all_pdb.csv"
    if not pdb_csv.exists():
        print(f"  [{pdb_label}][neighbor corr] distances_all_pdb.csv not found -- skipping.")
        return

    hbc_csv = results_dir / "hbc_distances.csv"
    if not hbc_csv.exists():
        print(f"  [{pdb_label}][neighbor corr] hbc_distances.csv not found -- skipping.")
        return

    corr_dir = out_dir / "neighbor_correlations"
    corr_dir.mkdir(exist_ok=True)

    df_dist = pd.read_csv(pdb_csv)
    df_hbc  = pd.read_csv(hbc_csv)

    # merge on frame
    merged  = pd.merge(df_dist, df_hbc[["frame", "A_C", "B_D"]], on="frame", how="inner")
    if len(merged) < 10:
        print(f"  [{pdb_label}][neighbor corr] Too few overlapping frames -- skipping.")
        return

    neighbor_cols = [c for c in df_dist.columns if c != "frame"]
    print(f"\n  [{pdb_label}][neighbor corr] Computing Spearman vs HBC gates "
          f"for {len(neighbor_cols)} neighbors...")

    ac  = merged["A_C"].values
    bd  = merged["B_D"].values
    rows = []

    for col in neighbor_cols:
        vals = merged[col].values
        mask_ac = ~(np.isnan(vals) | np.isnan(ac))
        mask_bd = ~(np.isnan(vals) | np.isnan(bd))
        if mask_ac.sum() < 10 or mask_bd.sum() < 10:
            continue
        r_ac, p_ac = spearmanr(vals[mask_ac], ac[mask_ac])
        r_bd, p_bd = spearmanr(vals[mask_bd], bd[mask_bd])
        rows.append({
            "residue":   col,
            "r_AC":      round(float(r_ac), 4),
            "p_AC":      float(p_ac),
            "r_BD":      round(float(r_bd), 4),
            "p_BD":      float(p_bd),
            "abs_r_AC":  round(abs(float(r_ac)), 4),
            "abs_r_BD":  round(abs(float(r_bd)), 4),
            "max_abs_r": round(max(abs(float(r_ac)), abs(float(r_bd))), 4),
        })

    if not rows:
        print(f"  [{pdb_label}][neighbor corr] No correlations computed.")
        return

    df_corr    = (pd.DataFrame(rows)
                  .sort_values("max_abs_r", ascending=False)
                  .reset_index(drop=True))
    clean_cols = ["residue", "r_AC", "p_AC", "r_BD", "p_BD"]

    # save all
    all_path = corr_dir / "neighbor_gate_correlation.csv"
    df_corr[clean_cols].to_csv(all_path, index=False)
    print(f"    Saved -> neighbor_gate_correlation.csv  ({len(df_corr)} rows)")

    # save filtered by min_r
    df_filt   = df_corr[df_corr["max_abs_r"] >= min_r]
    filt_path = corr_dir / f"neighbor_gate_correlation_filtered_r{min_r}.csv"
    df_filt[clean_cols].to_csv(filt_path, index=False)
    print(f"    Saved -> {filt_path.name}  ({len(df_filt)} rows above |r|>={min_r})")

    # save filtered by proximity (residues that appear in trajectory plot)
    prox_csv = out_dir / "proximity_summary.csv"
    if prox_csv.exists():
        df_prox       = pd.read_csv(prox_csv)
        close_residues = set(df_prox["pdb_label"].astype(str).tolist())
        df_close      = df_corr[df_corr["residue"].isin(close_residues)].copy()
        close_path    = corr_dir / "neighbor_gate_correlation_close_residues.csv"
        df_close[clean_cols].to_csv(close_path, index=False)
        print(f"    Saved -> {close_path.name}  ({len(df_close)} close residues)")
    else:
        print(f"    proximity_summary.csv not found -- skipping close residues CSV.")

    # print top 10
    print(f"\n    Top 10 by max|r| for {pdb_label}:")
    print(f"    {'residue':<25} {'r_AC':>8} {'r_BD':>8}")
    print(f"    {'-'*45}")
    for _, row in df_corr.head(10).iterrows():
        print(f"    {row['residue']:<25} {row['r_AC']:>8.3f} {row['r_BD']:>8.3f}")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_residue_neighbors_analysis(
    u,
    results_dir:      Path,
    channel_type:     str,
    resids:           List[int],   # raw resids — will be converted to PDB labels
    start_frame:      int,
    end_frame:        int,
    threshold:        float = 4.0,
    prescreen_frames: int   = 100,
    prescreen_factor: float = 5.0,
    smoothing_window: int   = 100,
    min_frames:       int   = 50,   # min frames < threshold to include in plot
):
    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    results_dir = Path(results_dir)
    out_base    = results_dir / "residue_neighbors"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("RESIDUE NEIGHBORS ANALYSIS")
    print(f"  results_dir      : {results_dir}")
    print(f"  channel_type     : {channel_type}")
    print(f"  resids (raw)     : {resids}")
    print(f"  threshold        : {threshold} A")
    print(f"  prescreen_factor : {prescreen_factor}  "
          f"(cutoff = {threshold * prescreen_factor:.1f} A)")
    print(f"  frames           : {start_frame} -> {end_frame}")
    print(f"  smoothing        : {smoothing_window} frames")
    print(f"  min_frames       : {min_frames} frames < threshold to plot")
    print("=" * 70)

    protein_residues = list(u.select_atoms("protein").residues)
    print(f"  Total protein residues: {len(protein_residues)}")

    for resid in resids:
        pdb_label = to_pdb(resid, channel_type)
        # sanitize for folder name e.g. "197.2b" -> "197_2b"
        folder_name = pdb_label.replace(".", "_")
        out_dir     = out_base / folder_name

        print(f"\n  Processing resid {resid} -> {pdb_label}")

        _analyse_residue(
            u                = u,
            resid            = resid,
            pdb_label        = pdb_label,
            protein_residues = protein_residues,
            channel_type     = channel_type,
            start_frame      = start_frame,
            end_frame        = end_frame,
            threshold        = threshold,
            prescreen_frames = prescreen_frames,
            prescreen_factor = prescreen_factor,
            out_dir          = out_dir,
            smoothing_window = smoothing_window,
            min_frames       = min_frames,
        )

        # sidechain-only analysis
        _analyse_residue_sidechain(
            u                = u,
            resid            = resid,
            pdb_label        = pdb_label,
            protein_residues = protein_residues,
            channel_type     = channel_type,
            start_frame      = start_frame,
            end_frame        = end_frame,
            threshold        = threshold,
            prescreen_frames = prescreen_frames,
            prescreen_factor = prescreen_factor,
            out_dir          = out_dir,
            smoothing_window = smoothing_window,
            min_frames       = min_frames,
        )

        # H-bond analysis
        try:
            from analysis.converter import to_pdb as _to_pdb
        except ImportError:
            def _to_pdb(r, ct): return str(r)

        # check if new-format hbond_per_frame.csv exists — if so skip prescreen
        _hb_per_frame_csv = out_dir / "hbonds" / "hbond_per_frame.csv"
        _hb_atom_pair_csv = out_dir / "hbonds" / "hbond_atom_pair_summary.csv"

        _needs_scan = True
        if _hb_per_frame_csv.exists() and _hb_atom_pair_csv.exists():
            _df_check = pd.read_csv(_hb_per_frame_csv)
            if "bb_bb_partners" in _df_check.columns:
                _needs_scan = False  # new format — no scan needed

        if not _needs_scan:
            # new format exists — pass empty candidates, will just regenerate plots
            _analyse_residue_hbonds(
                u                = u,
                resid            = resid,
                pdb_label        = pdb_label,
                candidates       = [],
                pdb_labels_cands = [],
                channel_type     = channel_type,
                start_frame      = start_frame,
                end_frame        = end_frame,
                out_dir          = out_dir,
                smoothing_window = smoothing_window,
            )
        else:
            # need scan — run prescreen first
            hb_candidates = _prescreen_candidates(
                u, resid, protein_residues,
                threshold, prescreen_frames, prescreen_factor, start_frame,
            )
            hb_pdb_labels = [
                f"{res.resname}{_to_pdb(res.resid, channel_type)}"
                for res in hb_candidates
            ]
            _analyse_residue_hbonds(
                u                = u,
                resid            = resid,
                pdb_label        = pdb_label,
                candidates       = hb_candidates,
                pdb_labels_cands = hb_pdb_labels,
                channel_type     = channel_type,
                start_frame      = start_frame,
                end_frame        = end_frame,
                out_dir          = out_dir,
                smoothing_window = smoothing_window,
            )

        # neighbor-gate correlation analysis
        _analyse_residue_neighbor_correlations(
            pdb_label   = pdb_label,
            out_dir     = out_dir,
            results_dir = results_dir,
            min_r       = 0.3,
        )

    print(f"\n✓ Residue neighbors analysis complete.")
    print(f"  Results in: {out_base}")