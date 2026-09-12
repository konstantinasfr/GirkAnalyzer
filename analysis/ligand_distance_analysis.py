#!/usr/bin/env python3
"""
Ligand Distance Analysis — G12_ML and G12_GAT

Computes per-frame minimum distances between ligand hydrogen atoms and
key protein residue oxygen atoms.

G12_ML (ML297, resid 1305):
    H28, H29  →  GLU99  OE1/OE2  (saved in GLU_G1/)
    H28, H29  →  ASP131 OD1/OD2  (saved in ASP_G1/)

G12_GAT (GAT1508, resid 1305):
    H10, H11  →  GLU1073 OE1/OE2  (saved in GLU_G1/)
    H10, H11  →  ASP1105 OD1/OD2  (saved in ASP_G1/)

Each subfolder contains:
    ligand_distance_raw_data.pkl       — per-frame distance arrays
    ligand_distance_report.txt         — human-readable full frame table
    combined_distance_dihedrals.pkl    — distances + all dihedral angles
    combined_distance_dihedrals.txt    — human-readable combined table

Usage (from main.py):
    from ligand_distance_analysis import run_ligand_distance_analysis

    # G12_ML
    run_ligand_distance_analysis(u, results_dir, start_frame, end_frame,
                                 channel_type='G12_ML')

    # G12_GAT
    run_ligand_distance_analysis(u, results_dir, start_frame, end_frame,
                                 channel_type='G12_GAT')
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# DRUG CONFIGURATION
# =============================================================================

# Per channel type: which H atoms, and which protein residues/oxygens to measure
DRUG_CONFIG = {
    'G12_ML': {
        'lig_resid': 1305,
        'targets': [
            {
                'label':   'GLU141_G1',
                'resid':   99,
                'resname': 'GLU',
                'O_atoms': ['OE1', 'OE2'],
                'H_atoms': ['H28', 'H29'],
            },
            {
                'label':   'ASP173_G1',
                'resid':   131,
                'resname': 'ASP',
                'O_atoms': ['OD1', 'OD2'],
                'H_atoms': ['H28', 'H29'],
            },
            {
                'label':   'SER177_OG',
                'resid':   774,
                'resname': 'SER',
                'O_atoms': ['OG'],
                'H_atoms': ['N15'],
            },
            {
                'label':   'TRP95_NE1',
                'resid':   53,
                'resname': 'TRP',
                'O_atoms': ['NE1'],
                'H_atoms': ['O11'],
            },
        ],
    },
    'G12_GAT': {
        'lig_resid': 1305,
        'targets': [
            {
                'label':   'GLU141_G1',
                'resid':   1073,
                'resname': 'GLU',
                'O_atoms': ['OE1', 'OE2'],
                'H_atoms': ['H10', 'H11'],
            },
            {
                'label':   'ASP173_G1',
                'resid':   1105,
                'resname': 'ASP',
                'O_atoms': ['OD1', 'OD2'],
                'H_atoms': ['H10', 'H11'],
            },
            {
                'label':   'SER177_OG',
                'resid':   447,
                'resname': 'SER',
                'O_atoms': ['OG'],
                'H_atoms': ['N2'],
            },
            {
                'label':   'TRP95_NE1',
                'resid':   1027,
                'resname': 'TRP',
                'O_atoms': ['NE1'],
                'H_atoms': ['O1'],
            },
        ],
    },
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_ligand_distance_analysis(u, results_dir, start_frame=0, end_frame=None,
                                  channel_type='G12_ML', frame_duration=0.075,
                                  permeation_events=None):
    """
    Run distance analysis for both GLU and ASP targets and save to subfolders.

    Parameters
    ----------
    u            : MDAnalysis Universe
    results_dir  : Path — RUN-level results directory
    start_frame  : int
    end_frame    : int (default: last frame)
    channel_type : str — 'G12_ML' or 'G12_GAT'
    """
    if channel_type not in DRUG_CONFIG:
        print(f"  [ERROR] channel_type '{channel_type}' not in DRUG_CONFIG.")
        print(f"  Supported: {list(DRUG_CONFIG.keys())}")
        return None

    config      = DRUG_CONFIG[channel_type]
    results_dir = Path(results_dir) / 'ligand_distance_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    print("\n" + "=" * 70)
    print(f"LIGAND DISTANCE ANALYSIS  —  {channel_type}")
    print("=" * 70)
    print(f"  Ligand resid : {config['lig_resid']}")
    for t in config['targets']:
        print(f"  Target : {t['resname']}{t['resid']} "
              f"{'/'.join(t['O_atoms'])}  <- LIG {'/'.join(t['H_atoms'])}"
              f"  -> {t['label']}/")

    # ── Find ligand atoms ─────────────────────────────────────────────────────
    lig_resid    = config['lig_resid']
    dihedral_pkl = Path(results_dir).parent / 'dihedral_analysis' / 'dihedral_raw_data.pkl'
    lig_all      = u.select_atoms(f'resid {lig_resid}')
    if len(lig_all) == 0:
        print(f"  [ERROR] No atoms found for resid {lig_resid}")
        print(f"  Available resids (first 20): "
              f"{sorted(set(u.atoms.resids))[:20]}")
        return None

    print(f"\n  Ligand (resid {lig_resid}): {len(lig_all)} atoms total")

    # ── Build atom selections for ALL targets upfront ────────────────────────
    n_frames    = end_frame - start_frame + 1
    all_targets = []
    for target in config['targets']:
        t  = {**target, 'lig_resid': lig_resid}
        ok = True

        H_sels = {}
        for h in t['H_atoms']:
            sel = u.select_atoms(f"resid {lig_resid} and name {h}")
            if len(sel) == 0:
                print(f"  [ERROR] resid {lig_resid} atom {h} not found "
                      f"— skipping {t['label']}")
                ok = False
                break
            H_sels[h] = sel

        if not ok:
            continue

        O_sels = {}
        for o in t['O_atoms']:
            sel = u.select_atoms(f"resid {t['resid']} and name {o}")
            if len(sel) == 0:
                print(f"  [ERROR] {t['resname']}{t['resid']}:{o} not found "
                      f"— skipping {t['label']}")
                ok = False
                break
            O_sels[o] = sel

        if not ok:
            continue

        pair_keys = [(h, o) for h in t['H_atoms'] for o in t['O_atoms']]
        t['H_sels']      = H_sels
        t['O_sels']      = O_sels
        t['pair_keys']   = pair_keys
        t['pair_labels'] = [f'{h}-{o}' for h, o in pair_keys]
        t['pair_arrs']   = {k: np.zeros(n_frames) for k in pair_keys}
        t['dist_min']    = np.zeros(n_frames)
        all_targets.append(t)
        print(f"  {t['label']}: {len(pair_keys)} pairs ready")

    if not all_targets:
        print("  [ERROR] No valid targets found.")
        return None

    # ── Single trajectory scan — all targets computed simultaneously ──────────
    print(f"\n  Scanning trajectory: {n_frames} frames, {len(all_targets)} targets")
    for fi, ts in enumerate(tqdm(u.trajectory[start_frame: end_frame + 1],
                                  total=n_frames,
                                  desc="  computing distances")):
        for t in all_targets:
            vals = []
            for (h, o) in t['pair_keys']:
                d = float(np.linalg.norm(
                    t['H_sels'][h].positions[0] - t['O_sels'][o].positions[0]))
                t['pair_arrs'][(h, o)][fi] = d
                vals.append(d)
            t['dist_min'][fi] = min(vals)

    print(f"  Trajectory scan complete.")

    # ── Save each target's files ──────────────────────────────────────────────
    for t in all_targets:
        subfolder = results_dir / t['label']
        subfolder.mkdir(exist_ok=True)
        print(f"\n  Saving {t['label']}...")
        _save_target(t, subfolder, start_frame, end_frame, n_frames,
                     dihedral_pkl, channel_type)

    # ── Combined plots across all targets ─────────────────────────────────────
    _plot_combined_boxplot(all_targets, channel_type, results_dir)
    _plot_combined_timeseries(all_targets, channel_type, start_frame,
                              n_frames, frame_duration, results_dir,
                              permeation_events=permeation_events)

    print(f"\n{'='*70}")
    print(f"  Done. Results saved in subfolders:")
    for t in config['targets']:
        print(f"    {results_dir}/{t['label']}/")
    print(f"{'='*70}")


# =============================================================================
# SAVE ONE TARGET (called after the single shared trajectory scan)
# =============================================================================

def _save_target(target, output_dir, start_frame, end_frame, n_frames,
                 dihedral_pkl, channel_type):
    resid       = target['resid']
    resname     = target['resname']
    o_names     = target['O_atoms']
    h_names     = target['H_atoms']
    lig_resid   = target['lig_resid']
    pair_keys   = target['pair_keys']
    pair_arrs   = target['pair_arrs']
    dist_min    = target['dist_min']
    pair_labels = target['pair_labels']

    # ── Save pkl ──────────────────────────────────────────────────────────────
    results = {
        'label':        target['label'],
        'channel_type': channel_type,
        'protein_res':  f"{resname}{resid}",
        'H_atoms':      h_names,
        'O_atoms':      o_names,
        'lig_resid':    lig_resid,
        'start_frame':  start_frame,
        'end_frame':    end_frame,
        'n_frames':     n_frames,
        'min_distance': dist_min,
    }
    for (h, o), arr in pair_arrs.items():
        results[f'{h}_{o}'] = arr

    pkl_path = output_dir / 'ligand_distance_raw_data.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✅ Saved: {pkl_path}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = output_dir / 'ligand_distance_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"LIGAND DISTANCE ANALYSIS — {channel_type}\n")
        f.write(f"LIG (resid {lig_resid}) "
                f"{'/'.join(h_names)}  →  "
                f"{resname}{resid} {'/'.join(o_names)}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Frames: {start_frame} → {end_frame}  ({n_frames} total)\n\n")

        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Pair':<15} {'Mean':>8} {'Std':>8} {'Min':>8} "
                f"{'Max':>8} {'<3.5Å%':>8} {'<5.0Å%':>8}\n")
        f.write("-" * 70 + "\n")
        for lbl, (h, o) in zip(pair_labels, pair_keys):
            arr = pair_arrs[(h, o)]
            f.write(f"{lbl:<15} {np.mean(arr):>8.2f} {np.std(arr):>8.2f} "
                    f"{np.min(arr):>8.2f} {np.max(arr):>8.2f} "
                    f"{np.mean(arr<3.5)*100:>8.1f} {np.mean(arr<5.0)*100:>8.1f}\n")
        f.write(f"{'Min(all)':15} {np.mean(dist_min):>8.2f} {np.std(dist_min):>8.2f} "
                f"{np.min(dist_min):>8.2f} {np.max(dist_min):>8.2f} "
                f"{np.mean(dist_min<3.5)*100:>8.1f} "
                f"{np.mean(dist_min<5.0)*100:>8.1f}\n")

        f.write("\n\nFULL FRAME TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Frame':>8} {'Min':>8}  " +
                ''.join(f"{l:>10}" for l in pair_labels) + "  Closest\n")
        f.write("-" * 70 + "\n")
        for i in range(n_frames):
            vals    = [pair_arrs[(h, o)][i] for h, o in pair_keys]
            closest = pair_labels[int(np.argmin(vals))]
            f.write(f"{start_frame+i:>8} {dist_min[i]:>8.3f}  " +
                    ''.join(f"{v:>10.3f}" for v in vals) +
                    f"  {closest}\n")

    print(f"  ✅ Saved: {report_path}")

    # ── Histogram plot ────────────────────────────────────────────────────────
    # For GLU/ASP (multiple H and O atoms): plot dist_min — the shortest
    # distance across all H-O pairs at each frame. This is the physically
    # meaningful quantity (closest contact).
    # For SER/TRP (single pair): dist_min is identical to that one distance.
    _plot_histogram(dist_min, target['label'], resname, resid,
                    h_names, o_names, channel_type, output_dir)

    # ── Combined with dihedrals ───────────────────────────────────────────────
    if dihedral_pkl.exists():
        print(f"  Found dihedral_raw_data.pkl — creating combined file...")
        _create_combined_file(results, pair_arrs, pair_keys, pair_labels,
                              dist_min, dihedral_pkl, output_dir,
                              start_frame, n_frames, channel_type)
    else:
        print(f"  [INFO] dihedral_raw_data.pkl not found at {dihedral_pkl} "
              f"— skipping combined file")


# =============================================================================
# COMBINED BOXPLOT — all 4 targets in one figure
# =============================================================================

def _plot_combined_boxplot(all_targets, channel_type, output_dir):
    """
    One boxplot panel per target, all in a single figure.
    X axis: target label.  Y axis: min distance (Å).
    X tick label shows label + mean + median.
    """
    fig, ax = plt.subplots(figsize=(max(10, len(all_targets) * 3), 8))

    data   = [t['dist_min'] for t in all_targets]
    labels = []
    for t in all_targets:
        d   = t['dist_min']
        labels.append(
            f"{t['label']}\nmean={np.mean(d):.2f}\nmedian={np.median(d):.2f}"
        )

    bp = ax.boxplot(
        data,
        positions=range(len(all_targets)),
        widths=0.50,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.3,
                        markerfacecolor="#4878CF", markeredgecolor="none"),
        medianprops=dict(color="black", linewidth=2.5),
        boxprops=dict(facecolor="#4878CF", alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
    )

    ax.set_xticks(range(len(all_targets)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Min distance (Å)", fontsize=22)
    ax.set_title(f"{channel_type}  —  Ligand distances — all targets",
                 fontsize=22, pad=14)
    ax.tick_params(axis="y", labelsize=18)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = output_dir / "all_targets_boxplot.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Saved combined boxplot: {out}")


# =============================================================================
# COMBINED TIME SERIES — all 4 targets in one figure
# =============================================================================

def _plot_combined_timeseries(all_targets, channel_type, start_frame,
                               n_frames, frame_duration, output_dir,
                               permeation_events=None):
    """
    All 4 targets in ONE panel. Called twice:
      1. Without permeation events — plain distance traces
      2. With permeation events — adds vertical lines at end_2 frames
    """
    t_arr  = np.array([(start_frame + fi) * frame_duration
                       for fi in range(n_frames)])
    colors = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD"]

    def _make_plot(ax, with_permeation):
        from matplotlib.lines import Line2D
        perm_handle = None
        perm_label  = None

        for t, color in zip(all_targets, colors):
            d    = t['dist_min']
            mean = np.mean(d)
            ax.plot(t_arr, d, color=color, linewidth=0.8, alpha=0.75)
            ax.axhline(mean, color=color, linewidth=1.8,
                       linestyle="--", alpha=0.7)
            # thick colored line in legend (using axhline trick)
            ax.plot([], [], color=color, linewidth=4,
                    label=f"{t['label']}  "
                          f"mean={mean:.2f}  std={np.std(d):.2f}  "
                          f"median={np.median(d):.2f} Å")

        if with_permeation and permeation_events:
            for ev in permeation_events:
                t_perm = ev['end_2'] * frame_duration
                ax.axvline(t_perm, color="black", linewidth=1.0,
                           alpha=0.6, linestyle="--")
            # single legend entry for permeation lines
            from matplotlib.lines import Line2D
            perm_handle = Line2D([0], [0], color="black", linewidth=2.0,
                                 alpha=0.8, linestyle="--")
            perm_label  = f"permeation end_2  (n={len(permeation_events)})"

        ax.set_xlabel("Time (ns)", fontsize=22)
        ax.set_ylabel("Min distance (Å)", fontsize=22)
        suffix = " + permeation events" if with_permeation else ""
        ax.set_title(
            f"{channel_type}  —  Ligand distances over trajectory{suffix}",
            fontsize=22, pad=14)
        handles, labels = ax.get_legend_handles_labels()
        if with_permeation and permeation_events and perm_handle is not None:
            handles.append(perm_handle)
            labels.append(perm_label)
        ax.legend(handles=handles, labels=labels,
                  fontsize=15, loc="upper right", framealpha=0.9,
                  handlelength=2.5, handleheight=1.5)
        ax.tick_params(labelsize=18)
        ax.spines[["top", "right"]].set_visible(False)

    # ── plot 1: no permeation lines ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 8))
    _make_plot(ax, with_permeation=False)
    plt.tight_layout()
    out1 = output_dir / "all_targets_timeseries.png"
    plt.savefig(out1, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out1}")

    # ── plot 2: with permeation lines (only if events provided) ──────────
    if permeation_events:
        fig, ax = plt.subplots(figsize=(18, 8))
        _make_plot(ax, with_permeation=True)
        plt.tight_layout()
        out2 = output_dir / "all_targets_timeseries_with_permeation.png"
        plt.savefig(out2, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: {out2}")


# =============================================================================
# HISTOGRAM PLOT
# =============================================================================

def _plot_histogram(dist_min, label, resname, resid, h_names, o_names,
                    channel_type, output_dir):
    """
    Histogram of the minimum distance across all frames.
    For GLU/ASP: dist_min = min(all H-O pairs) per frame.
    For SER/TRP: dist_min = the single pair distance.
    """
    n        = len(dist_min)
    mean_d   = float(np.mean(dist_min))
    std_d    = float(np.std(dist_min))
    median_d = float(np.median(dist_min))
    min_d    = float(np.min(dist_min))
    max_d    = float(np.max(dist_min))
    pct_3_5  = float(np.mean(dist_min < 3.5) * 100)
    pct_5_0  = float(np.mean(dist_min < 5.0) * 100)

    if len(h_names) > 1 or len(o_names) > 1:
        dist_desc = (f"min distance  ({'/'.join(h_names)} → "
                     f"{resname}{resid} {'/'.join(o_names)})")
    else:
        dist_desc = f"{h_names[0]} → {resname}{resid} {o_names[0]}"

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.hist(dist_min, bins=60, color="#4878CF", alpha=0.70,
            edgecolor="white", linewidth=0.4, density=False)

    ax.axvline(mean_d,   color="red",    linewidth=2.5, linestyle="--",
               label=f"mean   = {mean_d:.2f} Å")
    ax.axvline(median_d, color="orange", linewidth=2.5, linestyle=":",
               label=f"median = {median_d:.2f} Å")
    ax.axvline(min_d,    color="green",  linewidth=1.8, linestyle="-.",
               label=f"min    = {min_d:.2f} Å")
    ax.axvline(max_d,    color="purple", linewidth=1.8, linestyle="-.",
               label=f"max    = {max_d:.2f} Å")
    ax.axvline(3.5, color="grey", linewidth=1.4, linestyle="--", alpha=0.7,
               label=f"3.5 Å  ({pct_3_5:.1f}% frames)")
    ax.axvline(5.0, color="grey", linewidth=1.4, linestyle=":",  alpha=0.7,
               label=f"5.0 Å  ({pct_5_0:.1f}% frames)")

    # single legend on the right with all stats
    handles, labels = ax.get_legend_handles_labels()
    # prepend an invisible entry for the sample count as a title-like line
    from matplotlib.patches import Patch
    count_handle = Patch(facecolor="none", edgecolor="none",
                         label=f"n = {n:,} frames   std = {std_d:.2f} Å")
    ax.legend(handles=[count_handle] + handles,
              labels=[count_handle.get_label()] + labels,
              fontsize=14, loc="upper right",
              framealpha=0.92, edgecolor="black",
              handlelength=2.0)

    ax.set_xlabel("Distance (Å)", fontsize=20)
    ax.set_ylabel("Number of frames", fontsize=20)
    ax.set_title(
        f"{channel_type}  —  {label}\n{dist_desc}",
        fontsize=20, pad=14
    )
    ax.tick_params(labelsize=17)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = output_dir / "distance_histogram.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")


# =============================================================================
# COMBINED DISTANCE + DIHEDRAL FILE
# =============================================================================

def _create_combined_file(dist_results, pair_arrs, pair_keys, pair_labels,
                           dist_min, dihedral_pkl, output_dir,
                           start_frame, n_frames, channel_type):
    with open(dihedral_pkl, 'rb') as f:
        dihedral_data = pickle.load(f)

    combined = {
        'channel_type': channel_type,
        'start_frame':  start_frame,
        'n_frames':     n_frames,
        'min_distance': dist_min,
        'dihedrals':    {}
    }
    for (h, o), arr in pair_arrs.items():
        combined[f'{h}_{o}'] = arr

    residue_list = []
    for res_name, angles in dihedral_data.items():
        if not isinstance(angles, dict):
            continue
        pdb_label = angles.get('pdb_label', res_name)
        chi1 = np.array(angles['chi1'])[start_frame: start_frame + n_frames] \
               if angles.get('chi1') is not None else None
        chi2 = np.array(angles['chi2'])[start_frame: start_frame + n_frames] \
               if angles.get('chi2') is not None else None
        combined['dihedrals'][res_name] = {'pdb_label': pdb_label,
                                            'chi1': chi1, 'chi2': chi2}
        residue_list.append((res_name, pdb_label))

    pkl_out = output_dir / 'combined_distance_dihedrals.pkl'
    with open(pkl_out, 'wb') as f:
        pickle.dump(combined, f)
    print(f"  ✅ Saved: {pkl_out}")

    txt_out = output_dir / 'combined_distance_dihedrals.txt'
    h_names = dist_results['H_atoms']
    with open(txt_out, 'w') as f:
        f.write(f"COMBINED DISTANCES + DIHEDRALS — {channel_type}\n")
        f.write(f"LIG (resid {dist_results['lig_resid']}) "
                f"{'/'.join(h_names)} → {dist_results['protein_res']}\n")
        f.write("=" * 120 + "\n")
        f.write(f"Frames: {start_frame} → {start_frame+n_frames-1} "
                f"({n_frames} total)\n\n")

        dist_hdr = ''.join(f"{l:>10}" for l in pair_labels)
        res_hdr  = ''.join(f"  {pdb:>7}_χ1  {pdb:>7}_χ2"
                           for _, pdb in residue_list)
        f.write(f"{'Frame':>8} {'MinDist':>9}{dist_hdr}  "
                f"{'Closest':>10}{res_hdr}\n")
        f.write("-" * 120 + "\n")

        for i in range(n_frames):
            vals    = [pair_arrs[(h, o)][i] for h, o in pair_keys]
            closest = pair_labels[int(np.argmin(vals))]
            drow    = ''.join(f"{v:>10.3f}" for v in vals)
            arow    = ''
            for res_name, _ in residue_list:
                chi1 = combined['dihedrals'][res_name]['chi1']
                chi2 = combined['dihedrals'][res_name]['chi2']
                c1 = f"{chi1[i]:>10.2f}" if chi1 is not None else f"{'N/A':>10}"
                c2 = f"{chi2[i]:>10.2f}" if chi2 is not None else f"{'N/A':>10}"
                arow += f"{c1}{c2}"
            f.write(f"{start_frame+i:>8} {dist_min[i]:>9.3f}"
                    f"{drow}  {closest:>10}{arow}\n")

    print(f"  ✅ Saved: {txt_out}")