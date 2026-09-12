"""
analysis/pip_proximity_analysis.py
====================================
Analyses proximity of PIP lipid headgroup atoms (P1, P4, P5) to all protein residues.

Strategy
--------
Phase 1 - Pre-screen (first `prescreen_frames` frames):
    Compute min distance from each P1/P4/P5 atom to every protein residue.
    Keep only residues whose min distance < threshold * prescreen_factor.
    SKIPPED if distances_all_pdb.csv already exists (loaded instead).

Phase 2 - Full trajectory (start_frame -> end_frame):
    For each frame, compute distances only to the candidate set.
    SKIPPED if distances_all_pdb.csv already exists (loaded instead).

Output (per PIP residue, per atom P1/P4/P5):
    results_dir/pip_proximity_<factor>/PIP_<resid>_<subunit>/<atom>/
        proximity_summary.csv
        distances_all_raw.csv / distances_all_pdb.csv
        barplot_frames_raw.png / barplot_frames_pdb.png
        barplot_percentage_raw.png / barplot_percentage_pdb.png
        boxplot_raw.png / boxplot_pdb.png
        histograms/
        combination_analysis/
        trajectory_plots/
            trajectory_close.png        -- residues ever < threshold
            trajectory_mandatory.png    -- mandatory residues only
            trajectory_combined.png     -- mandatory + ever < threshold
        asn_gate_frames/ hbc_gate_frames/ etc.

Called from main.py:

    from analysis.pip_proximity_analysis import run_pip_proximity_analysis

    run_pip_proximity_analysis(
        u                = u,
        results_dir      = results_dir,
        start_frame      = start_frame,
        end_frame        = end_frame,
        channel_type     = args.channel_type,
        pip_resids       = [1301, 1302, 1303, 1304],
        pip_mapping      = {1301: "G1", 1302: "2c", 1303: "2a", 1304: "2b"},
        threshold        = 4.0,
        prescreen_factor = 6.0,
        end2_all         = end2_all,
        end3_all         = end3_all,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Mandatory residues per atom (PDB numbering)
# -----------------------------------------------------------------------------
MANDATORY_RESIDUES = {
    "P1": [90, 91, 92],
    "P4": [60, 64, 199, 200, 194, 90],
    "P5": [60, 64, 199, 200, 194, 90],
}

# -----------------------------------------------------------------------------
# Atom colour mapping
# -----------------------------------------------------------------------------
ATOM_BAR_COLORS = {
    "P1": "mediumseagreen",
    "P4": "darkorange",
    "P5": "steelblue",
}

def _atom_color(atom_name: str) -> str:
    return ATOM_BAR_COLORS.get(atom_name, "steelblue")

def _pip_label(pip_resid: int, pip_mapping: Optional[Dict[int, str]]) -> str:
    """Return e.g. '1301_G1' if mapping exists, else just '1301'."""
    if pip_mapping and pip_resid in pip_mapping:
        return f"{pip_resid}_{pip_mapping[pip_resid]}"
    return str(pip_resid)

def _extract_subunit(pip_label: str) -> Optional[str]:
    """Extract subunit from pip_label e.g. '1301_G1' -> 'G1', '1302_2b' -> '2b'."""
    if "_" in pip_label:
        return pip_label.split("_", 1)[1]
    return None


# -----------------------------------------------------------------------------
# Trajectory plots
# -----------------------------------------------------------------------------

def _make_trajectory_plot(
    df_pdb: pd.DataFrame,
    residue_cols: List[str],
    title: str,
    out_path: Path,
    threshold: float,
    atom_name: str,
    missing_mandatory: List[str] = None,
    smoothing_window: int = None,
):
    """
    Plot distance vs frame for each residue in residue_cols.
    df_pdb must have a 'frame' column and residue columns named by pdb_label.
    If smoothing_window is set, saves both the original plot and a smoothed version.
    """
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
                if not smooth:  # only warn once
                    print(f"    [trajectory plot] WARNING: column '{col}' not in distances_all_pdb.csv -- skipping.")
                continue
            vals = df_pdb[col].values
            if smooth and smoothing_window and smoothing_window > 1:
                vals = pd.Series(vals).rolling(window=smoothing_window, center=True, min_periods=1).mean().values
            ax.plot(frames, vals, label=col, linewidth=lw, alpha=0.9, color=cmap(i))
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                   label=f"threshold {threshold} Å")
        ax.set_xlabel("Frame", fontsize=13)
        ax.set_ylabel("Min distance to PIP (Å)", fontsize=13)
        legend = ax.legend(fontsize=8, ncol=max(1, n // 10 + 1),
                           loc="upper right", framealpha=0.6)
        # make legend lines match the plot line thickness
        for legline in legend.get_lines():
            legline.set_linewidth(lw)
        ax.tick_params(labelsize=11)
        if missing_mandatory:
            note = "Not in data (never within prescreen cutoff): " + ", ".join(missing_mandatory)
            ax.get_figure().text(0.01, 0.01, note, fontsize=7, color="gray", va="bottom")

    # --- original plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(title, fontsize=14)
    _draw(ax, smooth=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    [trajectory plot] Saved -> {out_path.name}")

    # --- smoothed plot ---
    if smoothing_window and smoothing_window > 1:
        smooth_path = out_path.parent / (out_path.stem + f"_smooth{smoothing_window}.png")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title(f"{title}  [smoothed, window={smoothing_window}]", fontsize=14)
        _draw(ax, smooth=True)
        plt.tight_layout()
        fig.savefig(smooth_path, dpi=150)
        plt.close(fig)
        print(f"    [trajectory plot] Saved -> {smooth_path.name}")


def _make_all_trajectory_plots(
    df_pdb:           pd.DataFrame,
    atom_name:        str,
    pip_label:        str,
    threshold:        float,
    out_dir:          Path,
    smoothing_window: int = None,
):
    """Generate the three trajectory plots for a given atom (original + smoothed)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cols = [c for c in df_pdb.columns if c != "frame"]

    # extract subunit from pip_label e.g. "1301_G1" -> "G1", "1302_2b" -> "2b"
    subunit = _extract_subunit(pip_label)

    # --- residues ever < threshold ---
    ever_close = [c for c in all_cols if (df_pdb[c] < threshold).any()]

    # --- mandatory residues: match e.g. num=90, subunit="2b" -> "ARG90.2b" ---
    mandatory_pdb_nums = MANDATORY_RESIDUES.get(atom_name, [])
    mandatory_cols     = []
    missing_mandatory  = []
    for num in mandatory_pdb_nums:
        if subunit:
            matches = [c for c in all_cols if c.endswith(f"{num}.{subunit}")]
        else:
            matches = [c for c in all_cols if c.endswith(str(num))]
        if matches:
            mandatory_cols.extend(matches)
        else:
            missing_mandatory.append(str(num))
            print(f"    [trajectory plot] Mandatory residue {num} (PDB) not found in "
                  f"distances_all_pdb.csv for PIP{pip_label}:{atom_name} -- cannot plot.")

    # --- combined: union of ever_close and mandatory ---
    combined_cols = list(dict.fromkeys(ever_close + mandatory_cols))

    base_title = f"PIP{pip_label} — {atom_name} distances"

    # Plot 1 -- ever close
    _make_trajectory_plot(
        df_pdb            = df_pdb,
        residue_cols      = ever_close,
        title             = f"{base_title}  |  residues ever < {threshold} Å",
        out_path          = out_dir / "trajectory_close.png",
        threshold         = threshold,
        atom_name         = atom_name,
        smoothing_window  = smoothing_window,
    )

    # Plot 2 -- mandatory only
    _make_trajectory_plot(
        df_pdb            = df_pdb,
        residue_cols      = mandatory_cols,
        title             = f"{base_title}  |  mandatory residues",
        out_path          = out_dir / "trajectory_mandatory.png",
        threshold         = threshold,
        atom_name         = atom_name,
        missing_mandatory = missing_mandatory,
        smoothing_window  = smoothing_window,
    )

    # Plot 3 -- combined
    _make_trajectory_plot(
        df_pdb            = df_pdb,
        residue_cols      = combined_cols,
        title             = f"{base_title}  |  mandatory + ever < {threshold} Å",
        out_path          = out_dir / "trajectory_combined.png",
        threshold         = threshold,
        atom_name         = atom_name,
        missing_mandatory = missing_mandatory,
        smoothing_window  = smoothing_window,
    )


# -----------------------------------------------------------------------------
# Pre-screen
# -----------------------------------------------------------------------------

def _prescreen_candidates(
    u,
    pip_atom,
    protein_residues,
    threshold: float,
    prescreen_frames: int,
    prescreen_factor: float,
    start_frame: int,
) -> list:
    cutoff    = threshold * prescreen_factor
    min_dists = np.full(len(protein_residues), np.inf)
    frame_end = min(start_frame + prescreen_frames, len(u.trajectory))

    for ts in u.trajectory[start_frame:frame_end]:
        pip_pos = pip_atom.position.reshape(1, 3)
        for i, res in enumerate(protein_residues):
            dists        = distance_array(pip_pos, res.atoms.positions)[0]
            min_dists[i] = min(min_dists[i], dists.min())

    candidates = [res for i, res in enumerate(protein_residues) if min_dists[i] < cutoff]
    print(f"    Pre-screen: {len(candidates)}/{len(protein_residues)} residues "
          f"within {cutoff:.1f} A (threshold x {prescreen_factor})")
    return candidates


# -----------------------------------------------------------------------------
# Bar plot helper
# -----------------------------------------------------------------------------

def _make_barplot(labels, values, ylabel, title, out_path, color="steelblue"):
    if not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No residues within threshold",
                ha="center", va="center", fontsize=16,
                transform=ax.transAxes, color="gray")
        ax.set_xlabel(ylabel, fontsize=15)
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    pairs       = sorted(zip(values, labels), reverse=True)
    vals_sorted = [p[0] for p in pairs]
    labs_sorted = [p[1] for p in pairs]

    fig_w = max(10, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    x_pos = np.arange(len(labs_sorted))
    bars  = ax.bar(x_pos, vals_sorted, color=color, edgecolor="white", alpha=0.85)

    max_val = max(vals_sorted) if vals_sorted else 1
    for bar, val in zip(bars, vals_sorted):
        fmt = f"{val:.1f}" if isinstance(val, float) else str(val)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                fmt, va="bottom", ha="center", fontsize=13)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labs_sorted, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Combination analysis
# -----------------------------------------------------------------------------

def _combination_analysis(
    all_dists, frame_index, raw_labels, pdb_labels,
    threshold, pip_label, atom_name, out_dir,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames     = all_dists.shape[0]
    close_matrix = all_dists < threshold

    per_frame_raw     = []
    per_frame_pdb     = []
    combo_counter_raw = {}
    combo_counter_pdb = {}
    count_counter     = {}

    for fi in range(n_frames):
        frame     = int(frame_index[fi])
        close_idx = np.where(close_matrix[fi])[0]
        count     = len(close_idx)

        combo_raw = "_".join(sorted([raw_labels[i] for i in close_idx])) if count > 0 else "none"
        combo_pdb = "_".join(sorted([pdb_labels[i] for i in close_idx])) if count > 0 else "none"

        per_frame_raw.append({"frame": frame, "count": count, "combination": combo_raw})
        per_frame_pdb.append({"frame": frame, "count": count, "combination": combo_pdb})

        combo_counter_raw[combo_raw] = combo_counter_raw.get(combo_raw, 0) + 1
        combo_counter_pdb[combo_pdb] = combo_counter_pdb.get(combo_pdb, 0) + 1
        count_counter[count]         = count_counter.get(count, 0) + 1

    pd.DataFrame(per_frame_raw).to_csv(out_dir / "per_frame_raw.csv", index=False)
    pd.DataFrame(per_frame_pdb).to_csv(out_dir / "per_frame_pdb.csv", index=False)

    for counter, suffix in [(combo_counter_raw, "raw"), (combo_counter_pdb, "pdb")]:
        rows = []
        for combo, frames in sorted(counter.items(), key=lambda x: -x[1]):
            n_res = 0 if combo == "none" else len(combo.split("_"))
            rows.append({"combination": combo, "n_residues": n_res,
                         "frames": frames, "pct": round(frames / n_frames * 100, 3)})
        pd.DataFrame(rows).to_csv(out_dir / f"combination_counts_{suffix}.csv", index=False)

    dist_rows = []
    for n_res in sorted(count_counter.keys()):
        dist_rows.append({"n_residues_close": n_res, "frames": count_counter[n_res],
                          "pct": round(count_counter[n_res] / n_frames * 100, 3)})
    pd.DataFrame(dist_rows).to_csv(out_dir / "count_distribution.csv", index=False)

    counts_sorted = sorted(count_counter.keys())
    pcts          = [count_counter[k] / n_frames * 100 for k in counts_sorted]
    raw_counts    = [count_counter[k] for k in counts_sorted]
    x_labels      = [str(k) for k in counts_sorted]
    color         = _atom_color(atom_name)
    title_base    = f"PIP{pip_label}:{atom_name}\nCount distribution (threshold {threshold} A)"
    fig_w         = max(6, len(counts_sorted) * 0.8)

    for vals, ylabel, fname in [
        (pcts,       "% of frames",       "barplot_count_distribution_pct.png"),
        (raw_counts, "Number of frames",  "barplot_count_distribution_frames.png"),
    ]:
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        bars = ax.bar(x_labels, vals, color=color, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.1f}%" if isinstance(val, float) else str(val),
                    va="bottom", ha="center", fontsize=13)
        ax.set_xlabel("Number of residues within threshold", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title_base, fontsize=16)
        ax.tick_params(labelsize=13)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    top_n       = 15
    title_combo = f"PIP{pip_label}:{atom_name}\nTop combinations (threshold {threshold} A)"
    for counter, suffix in [(combo_counter_raw, "raw"), (combo_counter_pdb, "pdb")]:
        top    = sorted(counter.items(), key=lambda x: -x[1])[:top_n]
        labels = [t[0] for t in top]
        raw_c  = [t[1] for t in top]
        pcts_c = [t[1] / n_frames * 100 for t in top]
        if not labels:
            continue
        fig_w = max(8, len(labels) * 0.7)
        for vals, ylabel, fname in [
            (pcts_c, "% of frames",      f"barplot_top_combinations_{suffix}_pct.png"),
            (raw_c,  "Number of frames", f"barplot_top_combinations_{suffix}_frames.png"),
        ]:
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            bars = ax.bar(range(len(labels)), vals, color=color, edgecolor="white", alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{val:.1f}%" if isinstance(val, float) else str(val),
                        va="bottom", ha="center", fontsize=11)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            ax.tick_params(axis="y", labelsize=13)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_title(title_combo, fontsize=16)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
            plt.tight_layout()
            fig.savefig(out_dir / fname, dpi=150)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Subset analysis
# -----------------------------------------------------------------------------

def _subset_analysis(
    all_dists, frame_index, subset_frames, raw_labels, pdb_labels,
    threshold, pip_label, atom_name, out_dir, subset_name,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask  = np.array([int(f) in subset_frames for f in frame_index])
    n_sub = mask.sum()

    print(f"\n    [{subset_name}] {n_sub} frames")
    if n_sub == 0:
        print(f"    [{subset_name}] No frames found -- skipping.")
        return

    sub_dists    = all_dists[mask]
    frames_close = (sub_dists < threshold).sum(axis=0)
    pct_close    = frames_close / n_sub * 100.0

    rows = []
    for ci in range(len(raw_labels)):
        d       = sub_dists[:, ci]
        d_valid = d[~np.isnan(d)]
        if frames_close[ci] == 0:
            continue
        rows.append({
            "raw_label":      raw_labels[ci],
            "pdb_label":      pdb_labels[ci],
            "frames_close":   int(frames_close[ci]),
            "pct_close":      round(float(pct_close[ci]), 3),
            "n_total_frames": int(n_sub),
            "min_dist":       round(float(d_valid.min()),  3) if len(d_valid) else np.nan,
            "mean_dist":      round(float(d_valid.mean()), 3) if len(d_valid) else np.nan,
            "max_dist":       round(float(d_valid.max()),  3) if len(d_valid) else np.nan,
        })

    title_tag = f"PIP{pip_label}:{atom_name}  [{subset_name}, n={n_sub}]"
    color     = _atom_color(atom_name)

    if not rows:
        print(f"    [{subset_name}] No residues within threshold -- saving empty plots.")
        for suffix in ["raw", "pdb"]:
            _make_barplot([], [], "Number of frames",
                          f"{title_tag}  -- frames within {threshold} A",
                          out_dir / f"barplot_frames_{suffix}.png", color)
            _make_barplot([], [], "% of frames",
                          f"{title_tag}  -- % frames within {threshold} A",
                          out_dir / f"barplot_percentage_{suffix}.png", color)
    else:
        (pd.DataFrame(rows)
           .sort_values("frames_close", ascending=False)
           .to_csv(out_dir / "proximity_summary.csv", index=False))

        close_idx    = np.where(frames_close > 0)[0]
        close_raw    = [raw_labels[i]        for i in close_idx]
        close_pdb    = [pdb_labels[i]        for i in close_idx]
        close_frames = [int(frames_close[i]) for i in close_idx]
        close_pct    = [float(pct_close[i])  for i in close_idx]
        close_dists  = sub_dists[:, close_idx]

        for lbl_set, suffix in [(close_raw, "raw"), (close_pdb, "pdb")]:
            _make_barplot(lbl_set, close_frames, "Number of frames",
                          f"{title_tag}  -- frames within {threshold} A",
                          out_dir / f"barplot_frames_{suffix}.png", color)
            _make_barplot(lbl_set, close_pct, "% of frames",
                          f"{title_tag}  -- % frames within {threshold} A",
                          out_dir / f"barplot_percentage_{suffix}.png", color)

        medians  = np.nanmedian(close_dists, axis=0)
        sort_ord = np.argsort(medians)
        s_data   = [close_dists[:, i][~np.isnan(close_dists[:, i])] for i in sort_ord]
        for lbl_set, suffix in [
            ([close_raw[i] for i in sort_ord], "raw"),
            ([close_pdb[i] for i in sort_ord], "pdb"),
        ]:
            fig_w = max(10, len(close_idx) * 0.7)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            ax.boxplot(s_data, labels=lbl_set, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7),
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3))
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                       label=f"threshold {threshold} A")
            ax.set_xlabel("Protein residue", fontsize=15)
            ax.set_ylabel("Min distance to PIP (A)", fontsize=15)
            ax.set_title(f"{title_tag}  -- distance distribution", fontsize=16)
            ax.legend(fontsize=16)
            ax.tick_params(axis="y", labelsize=13)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            plt.tight_layout()
            fig.savefig(out_dir / f"boxplot_{suffix}.png", dpi=150)
            plt.close(fig)

        print(f"    [{subset_name}] Summary + plots saved -> {out_dir}")

    _combination_analysis(
        all_dists   = sub_dists,
        frame_index = frame_index[mask],
        raw_labels  = raw_labels,
        pdb_labels  = pdb_labels,
        threshold   = threshold,
        pip_label   = pip_label,
        atom_name   = atom_name,
        out_dir     = out_dir / "combination_analysis",
    )


# -----------------------------------------------------------------------------
# Main per-atom analysis
# -----------------------------------------------------------------------------

def _analyse_atom(
    u,
    pip_resid, pip_label, atom_name, protein_residues, channel_type,
    start_frame, end_frame, threshold, prescreen_frames, prescreen_factor,
    out_dir, gate_subsets, smoothing_window=None,
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
        print(f"\n  [PIP{pip_label}:{atom_name}] distances_all_pdb.csv already exists -- "
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
        sel = u.select_atoms(f"resid {pip_resid} and name {atom_name}")
        if len(sel) == 0:
            print(f"  WARNING: atom {atom_name} not found in resid {pip_resid}, skipping.")
            return
        pip_atom = sel[0]

        print(f"\n  [PIP{pip_label}:{atom_name}] Pre-screening candidates...")
        candidates = _prescreen_candidates(
            u, pip_atom, protein_residues,
            threshold, prescreen_frames, prescreen_factor, start_frame,
        )
        if not candidates:
            print(f"  No candidates found within {threshold * prescreen_factor:.1f} A -- skipping.")
            return

        n_frames   = end_frame - start_frame + 1
        n_cands    = len(candidates)
        raw_labels = [f"{res.resname}{res.resid}" for res in candidates]
        pdb_labels = [f"{res.resname}{to_pdb(res.resid, channel_type)}" for res in candidates]
        all_dists  = np.full((n_frames, n_cands), np.nan)

        print(f"  [PIP{pip_label}:{atom_name}] Full trajectory "
              f"({n_frames} frames, {n_cands} candidate residues)...")

        for fi, ts in enumerate(tqdm(
            u.trajectory[start_frame:end_frame + 1],
            total=n_frames,
            desc=f"  PIP{pip_label}:{atom_name}",
            leave=False,
        )):
            pip_pos = pip_atom.position.reshape(1, 3)
            for ci, res in enumerate(candidates):
                dists             = distance_array(pip_pos, res.atoms.positions)[0]
                all_dists[fi, ci] = dists.min()

        # save distance matrices
        df_raw = pd.DataFrame(all_dists, columns=raw_labels)
        df_raw.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_raw.to_csv(raw_csv_path, index=False)

        df_pdb = pd.DataFrame(all_dists, columns=pdb_labels)
        df_pdb.insert(0, "frame", np.arange(start_frame, end_frame + 1))
        df_pdb.to_csv(pdb_csv_path, index=False)

    # ------------------------------------------------------------------
    # All plots below always run (even if csv was pre-existing)
    # ------------------------------------------------------------------

    frames_close = (all_dists < threshold).sum(axis=0)
    pct_close    = frames_close / n_frames * 100.0
    color        = _atom_color(atom_name)

    # proximity summary
    summary_rows = []
    for ci, (raw_lbl, pdb_lbl) in enumerate(zip(raw_labels, pdb_labels)):
        d       = all_dists[:, ci]
        d_valid = d[~np.isnan(d)]
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
    (pd.DataFrame(summary_rows)
       .sort_values("frames_close", ascending=False)
       .query("frames_close > 0")
       .to_csv(out_dir / "proximity_summary.csv", index=False))

    close_idx = np.where(frames_close > 0)[0]

    if len(close_idx) == 0:
        print(f"    No residues ever within {threshold} A -- saving empty plots.")
        for suffix in ["raw", "pdb"]:
            _make_barplot([], [], "Number of frames",
                          f"PIP{pip_label}:{atom_name}  -- frames within {threshold} A",
                          out_dir / f"barplot_frames_{suffix}.png", color)
            _make_barplot([], [], "% of frames",
                          f"PIP{pip_label}:{atom_name}  -- % frames within {threshold} A",
                          out_dir / f"barplot_percentage_{suffix}.png", color)
    else:
        close_raw    = [raw_labels[i]        for i in close_idx]
        close_pdb    = [pdb_labels[i]        for i in close_idx]
        close_frames = [int(frames_close[i]) for i in close_idx]
        close_pct    = [float(pct_close[i])  for i in close_idx]
        close_dists  = all_dists[:, close_idx]

        for lbl_set, suffix in [(close_raw, "raw"), (close_pdb, "pdb")]:
            _make_barplot(lbl_set, close_frames, "Number of frames",
                          f"PIP{pip_label}:{atom_name}  -- frames within {threshold} A",
                          out_dir / f"barplot_frames_{suffix}.png", color)
            _make_barplot(lbl_set, close_pct, "% of frames",
                          f"PIP{pip_label}:{atom_name}  -- % frames within {threshold} A",
                          out_dir / f"barplot_percentage_{suffix}.png", color)

        medians  = np.nanmedian(close_dists, axis=0)
        sort_ord = np.argsort(medians)
        s_data   = [close_dists[:, i][~np.isnan(close_dists[:, i])] for i in sort_ord]
        for lbl_set, suffix in [
            ([close_raw[i] for i in sort_ord], "raw"),
            ([close_pdb[i] for i in sort_ord], "pdb"),
        ]:
            fig_w = max(10, len(close_idx) * 0.7)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            ax.boxplot(s_data, labels=lbl_set, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7),
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3))
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                       label=f"threshold {threshold} A")
            ax.set_xlabel("Protein residue", fontsize=15)
            ax.set_ylabel("Min distance to PIP (A)", fontsize=15)
            ax.set_title(f"PIP{pip_label}:{atom_name}  -- distance distribution", fontsize=16)
            ax.legend(fontsize=16)
            ax.tick_params(axis="y", labelsize=13)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            plt.tight_layout()
            fig.savefig(out_dir / f"boxplot_{suffix}.png", dpi=150)
            plt.close(fig)

        # histograms
        hist_dir = out_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)
        for raw_lbl, pdb_lbl, ci in zip(close_raw, close_pdb, close_idx):
            d = all_dists[:, ci]
            d = d[~np.isnan(d)]
            for lbl, suffix in [(raw_lbl, "raw"), (pdb_lbl, "pdb")]:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(d, bins=40, color=color, edgecolor="white", alpha=0.85)
                ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
                           label=f"threshold {threshold} A")
                ax.set_xlabel("Min distance to PIP (A)", fontsize=15)
                ax.set_ylabel("Frame count", fontsize=15)
                ax.set_title(f"{lbl}  -- PIP{pip_label}:{atom_name}", fontsize=16)
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=13)
                plt.tight_layout()
                fig.savefig(hist_dir / f"hist_{lbl}_{suffix}.png", dpi=150)
                plt.close(fig)

    # ------------------------------------------------------------------
    # Trajectory plots (always regenerated)
    # ------------------------------------------------------------------
    df_pdb_full = pd.read_csv(pdb_csv_path)
    _make_all_trajectory_plots(
        df_pdb           = df_pdb_full,
        atom_name        = atom_name,
        pip_label        = pip_label,
        threshold        = threshold,
        out_dir          = out_dir / "trajectory_plots",
        smoothing_window = smoothing_window,
    )

    # combination analysis -- full trajectory
    _combination_analysis(
        all_dists   = all_dists,
        frame_index = np.arange(len(all_dists)),
        raw_labels  = raw_labels,
        pdb_labels  = pdb_labels,
        threshold   = threshold,
        pip_label   = pip_label,
        atom_name   = atom_name,
        out_dir     = out_dir / "combination_analysis",
    )

    # gate subset analyses
    frame_index = np.arange(start_frame, start_frame + n_frames)
    for frame_set, folder_name, human_label in gate_subsets:
        if not frame_set:
            print(f"    [{human_label}] no frames provided -- skipping.")
            continue
        _subset_analysis(
            all_dists     = all_dists,
            frame_index   = frame_index,
            subset_frames = frame_set,
            raw_labels    = raw_labels,
            pdb_labels    = pdb_labels,
            threshold     = threshold,
            pip_label     = pip_label,
            atom_name     = atom_name,
            out_dir       = out_dir / folder_name,
            subset_name   = human_label,
        )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_pip_proximity_analysis(
    u,
    results_dir,
    start_frame:       int,
    end_frame:         int,
    channel_type:      str,
    pip_resids:        List[int]                = None,
    pip_mapping:       Optional[Dict[int, str]] = None,
    threshold:         float                    = 4.0,
    prescreen_frames:  int                      = 50,
    prescreen_factor:  float                    = 3.0,
    atoms:             List[str]                = None,
    end2_all:          dict                     = None,
    end3_all:          dict                     = None,
    smoothing_window:  int                      = 100,
):
    if pip_resids is None:
        pip_resids = [1301, 1302, 1303, 1304]
    if atoms is None:
        atoms = ["P1", "P4", "P5"]

    results_dir = Path(results_dir)

    # dynamic output folder based on prescreen_factor
    factor_str = str(int(prescreen_factor)) if prescreen_factor == int(prescreen_factor) else str(prescreen_factor)
    base_out = results_dir / f"pip_proximity_{factor_str}"
    base_out.mkdir(parents=True, exist_ok=True)

    # build gate frame sets
    def _flatten_with_prev(d):
        crossing, prev = set(), set()
        if d:
            for frames in d.values():
                for f in frames:
                    crossing.add(int(f))
                    if int(f) - 1 >= start_frame:
                        prev.add(int(f) - 1)
        return crossing, prev

    asn_gate_frames,  asn_gate_m1_frames = _flatten_with_prev(end2_all)
    hbc_gate_frames,  hbc_gate_m1_frames = _flatten_with_prev(end3_all)

    all_traj_frames = set(range(start_frame, end_frame + 1))

    gate_subsets = [
        (asn_gate_frames,                      "asn_gate_frames",        "ASN gate (end2)"),
        (all_traj_frames - asn_gate_frames,    "non_asn_gate_frames",    "non-ASN gate (end2)"),
        (asn_gate_m1_frames,                   "asn_gate_m1_frames",     "ASN gate -1 (pre-end2)"),
        (all_traj_frames - asn_gate_m1_frames, "non_asn_gate_m1_frames", "non-ASN gate -1 (pre-end2)"),
        (hbc_gate_frames,                      "hbc_gate_frames",        "HBC gate (end3)"),
        (all_traj_frames - hbc_gate_frames,    "non_hbc_gate_frames",    "non-HBC gate (end3)"),
        (hbc_gate_m1_frames,                   "hbc_gate_m1_frames",     "HBC gate -1 (pre-end3)"),
        (all_traj_frames - hbc_gate_m1_frames, "non_hbc_gate_m1_frames", "non-HBC gate -1 (pre-end3)"),
    ]

    print("\n" + "=" * 70)
    print("PIP PROXIMITY ANALYSIS")
    print(f"  PIP resids       : {pip_resids}")
    if pip_mapping:
        print(f"  PIP mapping      : {pip_mapping}")
    print(f"  Atoms            : {atoms}")
    print(f"  Threshold        : {threshold} A")
    print(f"  Pre-screen       : first {prescreen_frames} frames, "
          f"cutoff = {threshold * prescreen_factor:.1f} A")
    print(f"  Frames           : {start_frame} -> {end_frame}  ({len(all_traj_frames)} total)")
    print(f"  Channel type     : {channel_type}")
    print(f"  Output folder    : {base_out}")
    for frame_set, _, lbl in gate_subsets:
        if frame_set:
            print(f"  {lbl}: {len(frame_set)} frames")
    print("=" * 70)

    protein_residues = list(u.select_atoms("protein").residues)
    print(f"  Total protein residues: {len(protein_residues)}")

    for pip_resid in pip_resids:
        plabel = _pip_label(pip_resid, pip_mapping)
        for atom_name in atoms:
            out_dir = base_out / f"PIP_{plabel}" / atom_name
            _analyse_atom(
                u                = u,
                pip_resid        = pip_resid,
                pip_label        = plabel,
                atom_name        = atom_name,
                protein_residues = protein_residues,
                channel_type     = channel_type,
                start_frame      = start_frame,
                end_frame        = end_frame,
                threshold        = threshold,
                prescreen_frames = prescreen_frames,
                prescreen_factor = prescreen_factor,
                out_dir          = out_dir,
                gate_subsets     = gate_subsets,
                smoothing_window = smoothing_window,
            )

    print("\n✓ PIP proximity analysis complete.")
    print(f"  Results in: {base_out}")