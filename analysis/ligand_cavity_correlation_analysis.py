#!/usr/bin/env python3
"""
ligand_cavity_correlation_analysis.py
=======================================
Reads already-computed files and answers two questions:

  Q1: What are the ligand distances at permeation frames (end_2)?
      Are they different from the baseline (all frames)?

  Q2: Is there a correlation between ligand distance and
      the number of ions in the cavity?

Reads from disk (no trajectory scan):
    <run_dir>/ligand_distance_analysis/<label>/ligand_distance_raw_data.pkl
    <run_dir>/cavity_occupancy/trajectory_ion_counts.csv
    permeation_events  — passed in directly from analyzer

Outputs saved to <run_dir>/ligand_cavity_correlation/:
    permeation_distances.csv          one row per permeation event
    permeation_distances_vs_baseline.png  bar: mean distance at end_2 vs all frames
    ion_count_boxplot.png             distance distributions grouped by n_ions in cavity
    ion_count_correlation.csv         Pearson r + Spearman ρ per target
    ion_count_scatter.png             scatter: distance vs n_ions + trend line

Usage (from main.py):
    from analysis.ligand_cavity_correlation_analysis import (
        run_ligand_cavity_correlation_analysis
    )
    run_ligand_cavity_correlation_analysis(
        results_dir        = results_dir,
        permeation_events  = analyzer.permeation_events,
        channel_type       = args.channel_type,
    )
"""

import csv
import json
import pickle
from pathlib import Path
from scipy import stats as scipy_stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── target labels (same order as DRUG_CONFIG) ─────────────────────────────────
TARGET_LABELS = {
    'G12_ML':  ['GLU141_G1', 'ASP173_G1', 'SER177_OG', 'TRP95_NE1'],
    'G12_GAT': ['GLU141_G1', 'ASP173_G1', 'SER177_OG', 'TRP95_NE1'],
}
TARGET_COLORS = {
    'GLU141_G1': '#E74C3C',
    'ASP173_G1': '#2980B9',
    'SER177_OG': '#27AE60',
    'TRP95_NE1': '#8E44AD',
}

FS_TITLE  = 20
FS_LABEL  = 18
FS_TICK   = 16
FS_LEGEND = 14
FS_ANNOT  = 14


# =============================================================================
# LOADING HELPERS
# =============================================================================

def _load_distances(results_dir, label):
    """
    Load min_distance array for one target from ligand_distance_raw_data.pkl.
    Returns (dist_array, start_frame) or (None, None) if file missing.
    dist_array[i] corresponds to frame start_frame + i.
    """
    pkl = Path(results_dir) / 'ligand_distance_analysis' / label / \
          'ligand_distance_raw_data.pkl'
    if not pkl.exists():
        print(f"  [SKIP] {label}: {pkl} not found")
        return None, None
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    return data['min_distance'], int(data['start_frame'])


def _load_ion_counts(results_dir):
    """
    Load trajectory_ion_counts.csv.
    Returns dict {frame(int): n_total(int)} or empty dict if missing.
    """
    csv_path = Path(results_dir) / 'cavity_occupancy' / 'trajectory_ion_counts.csv'
    if not csv_path.exists():
        print(f"  [WARN] trajectory_ion_counts.csv not found at {csv_path}")
        return {}
    ion_counts = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            ion_counts[int(row['frame'])] = int(row['n_total'])
    return ion_counts


# =============================================================================
# Q1 — DISTANCES AT PERMEATION FRAMES
# =============================================================================

def _analyze_permeation_distances(targets_data, permeation_events, out_dir,
                                   channel_type):
    """
    For each permeation event, look up each target's distance at end_2.
    Save table and comparison plot (end_2 vs baseline).
    """
    if not permeation_events:
        print("  No permeation events — skipping permeation distance analysis.")
        return

    print(f"\n  Q1: Distances at permeation frames")
    print(f"  Events: {len(permeation_events)}")

    labels = list(targets_data.keys())

    # ── build per-event table ──────────────────────────────────────────────
    rows = []
    for ev in permeation_events:
        ion_id  = ev['ion_id']
        end2    = int(ev.get('end_2', ev.get('end_2_frame', -1)))
        if end2 < 0:
            continue
        row = {'ion_id': ion_id, 'end_2_frame': end2}
        for label, (dist_arr, start_fr) in targets_data.items():
            idx = end2 - start_fr
            if dist_arr is not None and 0 <= idx < len(dist_arr):
                row[f'{label}_dist'] = round(float(dist_arr[idx]), 3)
            else:
                row[f'{label}_dist'] = None
        rows.append(row)

    if not rows:
        print("  No valid events found.")
        return

    # ── save CSV ──────────────────────────────────────────────────────────
    fields = ['ion_id', 'end_2_frame'] + [f'{l}_dist' for l in labels]
    csv_path = out_dir / 'permeation_distances.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_path}")

    # ── comparison bar plot: end_2 vs baseline ────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.5), 7))
    x      = np.arange(len(labels))
    width  = 0.38
    rng    = np.random.default_rng(42)

    for xi, label in enumerate(labels):
        dist_arr, start_fr = targets_data[label]
        color = TARGET_COLORS.get(label, '#555555')

        # baseline: all frames
        baseline = dist_arr[dist_arr > 0] if dist_arr is not None else np.array([])

        # at end_2 frames
        at_perm = np.array([
            r[f'{label}_dist'] for r in rows
            if r.get(f'{label}_dist') is not None
        ])

        for vals, offset, face_alpha, hatch, tag in [
            (baseline, -width/2, 0.75, None,  'all frames'),
            (at_perm,  +width/2, 0.85, '//',  'at end_2'),
        ]:
            if len(vals) == 0:
                continue
            mean = np.mean(vals)
            std  = np.std(vals)
            bar  = ax.bar(xi + offset, mean, width, color=color,
                          alpha=face_alpha, edgecolor='black', linewidth=1.1,
                          hatch=hatch, zorder=2)
            ax.errorbar(xi + offset, mean, yerr=std, fmt='none',
                        color='black', capsize=6, elinewidth=1.8, zorder=3)
            ax.text(xi + offset, mean + std + 0.05,
                    f"{mean:.2f}\n±{std:.2f}",
                    ha='center', va='bottom', fontsize=FS_ANNOT - 1)

    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='grey', alpha=0.75, edgecolor='black', label='All frames'),
        Patch(facecolor='grey', alpha=0.85, edgecolor='black',
              hatch='//', label='At permeation (end_2)'),
    ], fontsize=FS_LEGEND, loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
    ax.set_title(
        f"{channel_type}  —  Ligand distances: permeation frames vs baseline\n"
        f"n_events = {len(rows)}",
        fontsize=FS_TITLE, pad=14)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = out_dir / 'permeation_distances_vs_baseline.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Q2 — DISTANCE vs ION COUNT IN CAVITY
# =============================================================================

def _analyze_ion_count_correlation(targets_data, ion_counts, out_dir,
                                    channel_type):
    """
    For every frame where ion count is known, pair it with each target's
    distance. Then:
      - Boxplot: distance distribution grouped by n_ions
      - Scatter + trend line per target
      - Pearson r + Spearman ρ correlation table
    """
    if not ion_counts:
        print("  No ion count data — skipping correlation analysis.")
        return

    print(f"\n  Q2: Distance vs cavity ion count")

    labels = list(targets_data.keys())

    # ── build aligned arrays ──────────────────────────────────────────────
    # frame → (n_ions, {label: distance})
    aligned = {}   # {frame: {'n_ions': int, label: float, ...}}

    for label, (dist_arr, start_fr) in targets_data.items():
        if dist_arr is None:
            continue
        for idx in range(len(dist_arr)):
            frame = start_fr + idx
            if frame not in ion_counts:
                continue
            if frame not in aligned:
                aligned[frame] = {'n_ions': ion_counts[frame]}
            aligned[frame][label] = float(dist_arr[idx])

    if not aligned:
        print("  No overlapping frames between distances and ion counts.")
        return

    frames_list = sorted(aligned.keys())
    n_ions_arr  = np.array([aligned[f]['n_ions'] for f in frames_list])
    unique_ns   = sorted(set(n_ions_arr))
    print(f"  Aligned frames: {len(frames_list)}  |  "
          f"Ion counts seen: {unique_ns}")

    # ── correlation CSV ───────────────────────────────────────────────────
    corr_rows = []
    for label in labels:
        d_arr = np.array([
            aligned[f].get(label, np.nan) for f in frames_list
        ])
        mask = ~np.isnan(d_arr)
        if mask.sum() < 10:
            continue
        r_p, p_p = scipy_stats.pearsonr(n_ions_arr[mask], d_arr[mask])
        r_s, p_s = scipy_stats.spearmanr(n_ions_arr[mask], d_arr[mask])
        corr_rows.append({
            'target':       label,
            'n_frames':     int(mask.sum()),
            'pearson_r':    round(r_p, 4),
            'pearson_p':    f"{p_p:.3e}",
            'spearman_rho': round(r_s, 4),
            'spearman_p':   f"{p_s:.3e}",
        })
        print(f"  {label:<20}  Pearson r={r_p:+.3f}  Spearman ρ={r_s:+.3f}")

    corr_csv = out_dir / 'ion_count_correlation.csv'
    if corr_rows:
        with open(corr_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            w.writeheader()
            w.writerows(corr_rows)
        print(f"  Saved: {corr_csv}")

    # print summary table clearly
    if corr_rows:
        print(f"\n  {'─'*65}")
        print(f"  {'TARGET':<22} {'Pearson r':>10} {'p':>12} {'Spearman ρ':>12} {'p':>12}")
        print(f"  {'─'*65}")
        for r in corr_rows:
            print(f"  {r['target']:<22} {r['pearson_r']:>+10.4f} {r['pearson_p']:>12} "
                  f"{r['spearman_rho']:>+12.4f} {r['spearman_p']:>12}")
        print(f"  {'─'*65}")

    # ── boxplot: distance grouped by n_ions ───────────────────────────────
    # only keep ion counts with enough frames (≥ 20)
    valid_ns = [n for n in unique_ns
                if np.sum(n_ions_arr == n) >= 20]
    if not valid_ns:
        print("  Not enough frames per ion count for boxplot — skipping.")
    else:
        n_targets = len(labels)
        fig, axes = plt.subplots(1, n_targets,
                                 figsize=(max(12, n_targets * 4), 7),
                                 sharey=False)
        if n_targets == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            color   = TARGET_COLORS.get(label, '#555555')
            d_arr   = np.array([
                aligned[f].get(label, np.nan) for f in frames_list
            ])
            box_data = [d_arr[n_ions_arr == n] for n in valid_ns]
            # remove NaNs per group
            box_data = [g[~np.isnan(g)] for g in box_data]

            bp = ax.boxplot(
                box_data,
                positions=range(len(valid_ns)),
                widths=0.55,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=3, alpha=0.25,
                                markerfacecolor=color, markeredgecolor='none'),
                medianprops=dict(color='black', linewidth=2.2),
                boxprops=dict(facecolor=color, alpha=0.65, linewidth=1.1),
                whiskerprops=dict(linewidth=1.3),
                capprops=dict(linewidth=1.3),
            )

            ax.set_xticks(range(len(valid_ns)))
            ax.set_xticklabels([str(n) for n in valid_ns], fontsize=FS_TICK)
            ax.set_xlabel('N ions in cavity', fontsize=FS_LABEL)
            ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
            ax.set_title(label, fontsize=FS_TITLE, fontweight='bold', pad=10)
            ax.tick_params(axis='y', labelsize=FS_TICK)
            ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle(
            f"{channel_type}  —  Ligand distances grouped by cavity ion count",
            fontsize=FS_TITLE, y=1.01)
        plt.tight_layout()
        out = out_dir / 'ion_count_boxplot.png'
        plt.savefig(out, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")

    # ── scatter: distance vs n_ions for each target ───────────────────────
    n_targets = len(labels)
    fig, axes = plt.subplots(1, n_targets,
                             figsize=(max(12, n_targets * 4), 6),
                             sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        color = TARGET_COLORS.get(label, '#555555')
        d_arr = np.array([
            aligned[f].get(label, np.nan) for f in frames_list
        ])
        mask = ~np.isnan(d_arr)
        x    = n_ions_arr[mask].astype(float)
        y    = d_arr[mask]

        # jitter x slightly so overlapping points are visible
        rng  = np.random.default_rng(42)
        x_j  = x + rng.uniform(-0.25, 0.25, size=len(x))

        ax.scatter(x_j, y, color=color, alpha=0.08, s=6, edgecolors='none')

        # trend line
        if len(x) >= 10:
            slope, intercept, r, p, _ = scipy_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color='black', linewidth=2.0,
                    label=f"r={r:+.3f}  p={p:.2e}")
            ax.legend(fontsize=FS_LEGEND)

        ax.set_xlabel('N ions in cavity', fontsize=FS_LABEL)
        ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE, fontweight='bold', pad=10)
        ax.set_xticks(sorted(set(x.astype(int))))
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f"{channel_type}  —  Ligand distance vs cavity ion count",
        fontsize=FS_TITLE, y=1.01)
    plt.tight_layout()
    out = out_dir / 'ion_count_scatter.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ── bar plot: % frames < threshold for each (n_ions × target) ────────
    _plot_ion_count_pct_bar(
        labels, frames_list, aligned, n_ions_arr, valid_ns,
        channel_type, out_dir,
        title_suffix="all frames",
        fname="ion_count_pct_under_bar_all.png")


# =============================================================================
# BAR PLOT: for each N ions group, % of frames where distance < 3.5 Å
# =============================================================================

def _plot_ion_count_pct_bar(labels, frames_list, aligned, n_ions_arr,
                             valid_ns, channel_type, out_dir,
                             title_suffix, fname, threshold=3.5):
    """
    Grouped bar chart:
        x groups = N ions in cavity (one group per ion count)
        bars within each group = one per target (4 bars)
        y = % of frames in that group where distance < threshold

    Also saves a CSV of the underlying numbers.
    """
    colors = [TARGET_COLORS.get(l, '#555555') for l in labels]
    n_groups = len(valid_ns)
    n_bars   = len(labels)
    width    = 0.80 / n_bars      # bars fit within width=0.8 per group
    x        = np.arange(n_groups)

    # build matrix: pct[i_group, j_label]
    pct_mat    = np.zeros((n_groups, n_bars))
    count_mat  = np.zeros((n_groups, n_bars), dtype=int)
    totals     = np.zeros(n_groups, dtype=int)

    for i, n in enumerate(valid_ns):
        mask_n   = (n_ions_arr == n)
        totals[i] = int(np.sum(mask_n))
        frames_in_group = [f for fi, f in enumerate(frames_list) if mask_n[fi]]
        for j, label in enumerate(labels):
            d_vals = np.array([
                aligned[f].get(label, np.nan) for f in frames_in_group
            ])
            valid = d_vals[~np.isnan(d_vals)]
            if len(valid) == 0:
                continue
            n_under = int(np.sum(valid < threshold))
            count_mat[i, j] = n_under
            pct_mat[i, j]   = 100. * n_under / len(valid)

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / fname.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_ions', 'n_frames'] + labels)
        for i, n in enumerate(valid_ns):
            w.writerow([n, totals[i]] +
                       [f"{pct_mat[i,j]:.1f}" for j in range(n_bars)])
    print(f"  Saved: {csv_path}")

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n  % frames with distance < {threshold} Å  ({title_suffix}):")
    print(f"  {'N ions':>8} {'n_frames':>10}" +
          ''.join(f"  {l:>16}" for l in labels))
    print('  ' + '─' * (10 + 10 + 18 * n_bars))
    for i, n in enumerate(valid_ns):
        row = f"  {n:>8} {totals[i]:>10}"
        for j in range(n_bars):
            row += f"  {pct_mat[i,j]:>15.1f}%"
        print(row)

    # ── bar plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n_groups * n_bars * 1.2), 8))

    for j, (label, color) in enumerate(zip(labels, colors)):
        offsets = x + (j - n_bars / 2 + 0.5) * width
        bars = ax.bar(offsets, pct_mat[:, j], width,
                      color=color, alpha=0.80, edgecolor='black',
                      linewidth=0.8, label=label, zorder=2)
        for bar, pct in zip(bars, pct_mat[:, j]):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + 0.8,
                        f"{pct:.0f}%",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold', rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n} ions\n(n={totals[i]:,})" for i, n in enumerate(valid_ns)],
        fontsize=FS_TICK + 1)
    ax.set_ylabel(f'% of frames with distance < {threshold} Å',
                  fontsize=FS_LABEL + 1)
    ax.set_title(
        f"{channel_type}  —  % frames with distance < {threshold} Å\n"
        f"grouped by N ions in cavity  ({title_suffix})",
        fontsize=FS_TITLE + 1, pad=14)
    ax.set_ylim(0, min(100, np.max(pct_mat) * 1.45) if np.max(pct_mat) > 0 else 10)
    ax.legend(fontsize=FS_LEGEND + 1, loc='upper right')
    ax.tick_params(axis='y', labelsize=FS_TICK + 1)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir / fname}")

    # ── second plot: top 3 most common n_ions groups only ─────────────────
    # sort by number of frames descending, take top 3
    top3_idx = sorted(range(n_groups), key=lambda i: totals[i], reverse=True)[:3]
    top3_idx = sorted(top3_idx)  # restore ascending order for x-axis
    top3_ns  = [valid_ns[i] for i in top3_idx]

    if len(top3_ns) >= 2:
        x3      = np.arange(len(top3_ns))
        fig, ax = plt.subplots(figsize=(max(8, len(top3_ns) * n_bars * 1.4), 8))
        for j, (label, color) in enumerate(zip(labels, colors)):
            offsets = x3 + (j - n_bars / 2 + 0.5) * width
            vals3   = [pct_mat[i, j] for i in top3_idx]
            bars    = ax.bar(offsets, vals3, width,
                             color=color, alpha=0.80, edgecolor='black',
                             linewidth=0.8, label=label, zorder=2)
            for bar, pct in zip(bars, vals3):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + 0.8,
                            f"{pct:.0f}%",
                            ha='center', va='bottom',
                            fontsize=13, fontweight='bold', rotation=45)

        ax.set_xticks(x3)
        ax.set_xticklabels(
            [f"{valid_ns[i]} ions\n(n={totals[i]:,})" for i in top3_idx],
            fontsize=FS_TICK + 2)
        ax.set_ylabel(f'% of frames with distance < {threshold} Å',
                      fontsize=FS_LABEL + 2)
        ax.set_title(
            f"{channel_type}  —  % frames with distance < {threshold} Å\n"
            f"top 3 most common ion counts  ({title_suffix})",
            fontsize=FS_TITLE + 1, pad=14)
        ax.set_ylim(0, min(100, max(pct_mat[top3_idx].max(), 1) * 1.45))
        ax.legend(fontsize=FS_LEGEND + 2, loc='upper right')
        ax.tick_params(axis='y', labelsize=FS_TICK + 2)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        top3_fname = fname.replace('.png', '_top3.png')
        plt.savefig(out_dir / top3_fname, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_dir / top3_fname}")


# =============================================================================
# PLOT: % of permeation events where distance < 3.5 Å  (4 bars)
# =============================================================================

def _plot_permeation_under_threshold(targets_data, permeation_events,
                                      out_dir, channel_type,
                                      threshold=3.5):
    """
    Bar chart: for each target, what % of permeation events had
    distance < threshold at end_2?
    """
    if not permeation_events:
        return

    labels = list(targets_data.keys())
    pcts   = []
    counts = []
    totals = []

    for label in labels:
        dist_arr, start_fr = targets_data[label]
        if dist_arr is None:
            pcts.append(0); counts.append(0); totals.append(0)
            continue
        vals = []
        for ev in permeation_events:
            end2 = int(ev.get('end_2', ev.get('end_2_frame', -1)))
            if end2 < 0:
                continue
            idx = end2 - start_fr
            if 0 <= idx < len(dist_arr):
                vals.append(float(dist_arr[idx]))
        n_total = len(vals)
        n_under = sum(1 for v in vals if v < threshold)
        pcts.append(100. * n_under / n_total if n_total > 0 else 0.)
        counts.append(n_under)
        totals.append(n_total)

    colors = [TARGET_COLORS.get(l, '#555555') for l in labels]
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 7))
    bars = ax.bar(x, pcts, color=colors, alpha=0.80,
                  edgecolor='black', linewidth=1.2, width=0.6)

    for xi, (bar, pct, cnt, tot) in enumerate(zip(bars, pcts, counts, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 1.5,
                f"{pct:.1f}%\n({cnt}/{tot})",
                ha='center', va='bottom', fontsize=FS_ANNOT, fontweight='bold')

    ax.axhline(50, color='grey', linewidth=1.2, linestyle='--', alpha=0.5,
               label='50%')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel(f'% of permeation events with distance < {threshold} Å',
                  fontsize=FS_LABEL)
    ax.set_title(
        f"{channel_type}  —  Ligand distances at permeation frames\n"
        f"% of events with distance < {threshold} Å  "
        f"(n_events = {totals[0] if totals else 0})",
        fontsize=FS_TITLE, pad=14)
    ax.set_ylim(0, max(pcts + [10]) * 1.30)
    ax.legend(fontsize=FS_LEGEND)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = out_dir / f'permeation_pct_under_{int(threshold*10)}A.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# PLOT: ion-count boxplot — PERMEATION FRAMES ONLY
# =============================================================================

def _plot_ioncount_boxplot_permeation(targets_data, permeation_events,
                                       ion_counts, out_dir, channel_type):
    """
    Same as ion_count_boxplot but using ONLY permeation frame (end_2) distances,
    grouped by n_ions in cavity at that frame.
    """
    if not permeation_events or not ion_counts:
        return

    labels = list(targets_data.keys())

    # build per-event: n_ions at end_2, and distance per target
    records = []
    for ev in permeation_events:
        end2 = int(ev.get('end_2', ev.get('end_2_frame', -1)))
        if end2 < 0 or end2 not in ion_counts:
            continue
        rec = {'n_ions': ion_counts[end2]}
        for label, (dist_arr, start_fr) in targets_data.items():
            if dist_arr is None:
                rec[label] = np.nan
                continue
            idx = end2 - start_fr
            rec[label] = float(dist_arr[idx]) if 0 <= idx < len(dist_arr) \
                         else np.nan
        records.append(rec)

    if not records:
        print("  No permeation events with ion count data — skipping.")
        return

    n_ions_perm = np.array([r['n_ions'] for r in records])
    unique_ns   = sorted(set(n_ions_perm))
    # keep groups with ≥ 3 events
    valid_ns    = [n for n in unique_ns if np.sum(n_ions_perm == n) >= 3]

    if not valid_ns:
        print("  Not enough permeation events per ion count — skipping.")
        return

    n_targets = len(labels)
    fig, axes = plt.subplots(1, n_targets,
                             figsize=(max(12, n_targets * 4), 7),
                             sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        color    = TARGET_COLORS.get(label, '#555555')
        d_arr    = np.array([r.get(label, np.nan) for r in records])
        box_data = [d_arr[n_ions_perm == n] for n in valid_ns]
        box_data = [g[~np.isnan(g)] for g in box_data]

        ax.boxplot(
            box_data,
            positions=range(len(valid_ns)),
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=5, alpha=0.5,
                            markerfacecolor=color, markeredgecolor='none'),
            medianprops=dict(color='black', linewidth=2.2),
            boxprops=dict(facecolor=color, alpha=0.65, linewidth=1.1),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
        )

        ax.set_xticks(range(len(valid_ns)))
        ax.set_xticklabels(
            [f"{n}\n(n={int(np.sum(n_ions_perm==n))})" for n in valid_ns],
            fontsize=FS_TICK - 1)
        ax.set_xlabel('N ions in cavity at end_2', fontsize=FS_LABEL)
        ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f"{channel_type}  —  Ligand distances at permeation frames\n"
        f"grouped by cavity ion count at end_2  "
        f"(n_events = {len(records)})",
        fontsize=FS_TITLE, y=1.02)
    plt.tight_layout()
    out = out_dir / 'ion_count_boxplot_permeation_only.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# RESIDUE COMBINATIONS CLOSE TO DRUG
# =============================================================================

# The 6 combinations of interest + 2 single-residue bars
COMBINATIONS = [
    ('GLU141_G1',),                              # GLU only
    ('GLU141_G1', 'TRP95_NE1', 'SER177_OG'),
    ('GLU141_G1', 'SER177_OG'),
    ('GLU141_G1', 'TRP95_NE1'),
    ('ASP173_G1',),                              # ASP only
    ('ASP173_G1', 'TRP95_NE1', 'SER177_OG'),
    ('ASP173_G1', 'SER177_OG'),
    ('ASP173_G1', 'TRP95_NE1'),
]

COMBO_LABELS = [
    'GLU141\nonly',
    'GLU141 + TRP95 + SER177',
    'GLU141 + SER177',
    'GLU141 + TRP95',
    'ASP173\nonly',
    'ASP173 + TRP95 + SER177',
    'ASP173 + SER177',
    'ASP173 + TRP95',
]

COMBO_COLORS = [
    '#922B21',             # GLU only — dark red
    '#C0392B', '#E74C3C',  # GLU-based — reds
    '#F39C12',             # GLU-based — orange
    '#1A5276',             # ASP only — dark blue
    '#2471A3', '#2980B9',  # ASP-based — blues
    '#1ABC9C',             # ASP-based — teal
]


def _get_dist_at_frame(targets_data, label, frame):
    """Return distance value for a given target at a given frame, or None."""
    if label not in targets_data:
        return None
    dist_arr, start_fr = targets_data[label]
    if dist_arr is None:
        return None
    idx = frame - start_fr
    if 0 <= idx < len(dist_arr):
        return float(dist_arr[idx])
    return None


def _check_combo(targets_data, combo, frame, threshold):
    """Return True if ALL residues in combo are within threshold at this frame."""
    for label in combo:
        d = _get_dist_at_frame(targets_data, label, frame)
        if d is None or d >= threshold:
            return False
    return True


def _analyze_combinations(targets_data, all_frames, permeation_events,
                           out_dir, channel_type, threshold=3.5,
                           frame_duration=0.075):
    """
    For each combination, count how many ALL frames and how many
    PERMEATION frames (end_2) have all residues within threshold.
    Saves CSV and two bar plots (all frames + permeation only).
    """
    print(f"\n  Residue combination analysis  (threshold = {threshold} Å)")

    n_all  = len(all_frames)
    perm_frames = []
    for ev in permeation_events:
        end2 = int(ev.get('end_2', ev.get('end_2_frame', -1)))
        if end2 >= 0:
            perm_frames.append(end2)
    n_perm = len(perm_frames)

    print(f"  All frames      : {n_all}")
    print(f"  Permeation end_2: {n_perm}")

    rows = []
    counts_all  = []
    counts_perm = []

    for combo, clabel in zip(COMBINATIONS, COMBO_LABELS):
        # skip if any required target is missing
        if any(c not in targets_data for c in combo):
            print(f"  [SKIP] {clabel} — missing target data")
            counts_all.append(0)
            counts_perm.append(0)
            rows.append({
                'combination':       clabel,
                'members':           ' + '.join(combo),
                'n_all_frames':      0,
                'pct_all_frames':    0.0,
                'n_perm_frames':     0,
                'pct_perm_frames':   0.0,
                'n_total_frames':    n_all,
                'n_total_perm':      n_perm,
                'threshold_A':       threshold,
            })
            continue

        # count in all frames
        n_c_all = sum(
            1 for fr in all_frames
            if _check_combo(targets_data, combo, fr, threshold)
        )

        # count in permeation frames
        n_c_perm = sum(
            1 for fr in perm_frames
            if _check_combo(targets_data, combo, fr, threshold)
        )

        pct_all  = 100. * n_c_all  / n_all  if n_all  > 0 else 0.
        pct_perm = 100. * n_c_perm / n_perm if n_perm > 0 else 0.

        counts_all.append(n_c_all)
        counts_perm.append(n_c_perm)

        rows.append({
            'combination':       clabel,
            'members':           ' + '.join(combo),
            'n_all_frames':      n_c_all,
            'pct_all_frames':    round(pct_all,  2),
            'n_perm_frames':     n_c_perm,
            'pct_perm_frames':   round(pct_perm, 2),
            'n_total_frames':    n_all,
            'n_total_perm':      n_perm,
            'threshold_A':       threshold,
        })

        print(f"  {clabel:<32}  "
              f"all={pct_all:5.1f}%  ({n_c_all:>5}/{n_all})  "
              f"perm={pct_perm:5.1f}%  ({n_c_perm}/{n_perm})")

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / 'combination_frequencies.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_path}")

    # ── helper for bar plot ───────────────────────────────────────────────
    def _combo_bar(counts, n_total, title, out_path):
        pcts = [100. * c / n_total if n_total > 0 else 0. for c in counts]

        fig, ax = plt.subplots(figsize=(max(14, len(COMBO_LABELS) * 2.0), 8))
        x    = np.arange(len(COMBO_LABELS))
        bars = ax.bar(x, pcts, color=COMBO_COLORS, alpha=0.82,
                      edgecolor='black', linewidth=1.2, width=0.65)

        # annotation above each bar
        max_pct = max(pcts + [1])
        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + max_pct * 0.02,
                    f"{pct:.1f}%\n(n={cnt:,})",
                    ha='center', va='bottom',
                    fontsize=FS_ANNOT, fontweight='bold')

        # vertical separator between GLU group (0-3) and ASP group (4-7)
        ax.axvline(3.5, color='black', linewidth=1.2, linestyle=':',
                   alpha=0.4)
        ax.text(3.5, max_pct * 1.22, '│', ha='center', va='top',
                fontsize=10, color='grey')

        ax.axhline(50, color='grey', linewidth=1.2, linestyle='--',
                   alpha=0.5, label='50%')
        ax.set_xticks(x)
        ax.set_xticklabels(COMBO_LABELS, fontsize=FS_TICK - 1,
                           rotation=25, ha='right')
        ax.set_ylabel(f'% of frames (total = {n_total:,})',
                      fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, pad=14)
        ax.set_ylim(0, max_pct * 1.35)
        ax.legend(fontsize=FS_LEGEND)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # ── plot 1: all frames ────────────────────────────────────────────────
    _combo_bar(
        counts_all, n_all,
        f"{channel_type}  —  Residue combinations within {threshold} Å of drug\n"
        f"ALL simulation frames  (n = {n_all:,})",
        out_dir / 'combinations_all_frames.png'
    )

    # ── plot 2: permeation frames only ────────────────────────────────────
    _combo_bar(
        counts_perm, n_perm,
        f"{channel_type}  —  Residue combinations within {threshold} Å of drug\n"
        f"PERMEATION frames only (end_2,  n_events = {n_perm})",
        out_dir / 'combinations_permeation_only.png'
    )


def _plot_ioncount_pct_bar_permeation(targets_data, permeation_events,
                                       ion_counts, out_dir, channel_type,
                                       threshold=3.5):
    """
    Same as the all-frames version but restricted to permeation end_2 frames.
    """
    if not permeation_events or not ion_counts:
        return

    labels = list(targets_data.keys())

    # build aligned dict for permeation frames only
    perm_frames_list = []
    perm_aligned     = {}
    perm_ions        = []

    for ev in permeation_events:
        end2 = int(ev.get('end_2', ev.get('end_2_frame', -1)))
        if end2 < 0 or end2 not in ion_counts:
            continue
        perm_frames_list.append(end2)
        perm_ions.append(ion_counts[end2])
        rec = {'n_ions': ion_counts[end2]}
        for label, (dist_arr, start_fr) in targets_data.items():
            if dist_arr is None:
                rec[label] = np.nan
                continue
            idx = end2 - start_fr
            rec[label] = float(dist_arr[idx]) if 0 <= idx < len(dist_arr) \
                         else np.nan
        perm_aligned[end2] = rec

    if not perm_frames_list:
        return

    n_ions_arr = np.array(perm_ions)
    unique_ns  = sorted(set(n_ions_arr))
    valid_ns   = [n for n in unique_ns if np.sum(n_ions_arr == n) >= 2]

    if not valid_ns:
        return

    # reuse _plot_ion_count_pct_bar with the permeation-frame data
    aligned_for_fn = {f: perm_aligned[f] for f in perm_frames_list}

    _plot_ion_count_pct_bar(
        labels, perm_frames_list, aligned_for_fn,
        n_ions_arr, valid_ns,
        channel_type, out_dir,
        title_suffix=f"permeation frames only  (n={len(perm_frames_list)})",
        fname="ion_count_pct_under_bar_permeation.png",
        threshold=threshold)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_ligand_cavity_correlation_analysis(
        results_dir,
        permeation_events,
        channel_type='G12_ML',
):
    """
    Parameters
    ----------
    results_dir        : Path — RUN-level results directory
    permeation_events  : list of dicts from analyzer.permeation_events
    channel_type       : str — 'G12_ML' or 'G12_GAT'
    """
    results_dir = Path(results_dir)
    out_dir     = results_dir / 'ligand_cavity_correlation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*70)
    print(f'LIGAND–CAVITY CORRELATION ANALYSIS  —  {channel_type}')
    print('='*70)

    labels = TARGET_LABELS.get(channel_type)
    if labels is None:
        print(f'  [ERROR] channel_type {channel_type!r} not recognised.')
        print(f'  Supported: {list(TARGET_LABELS.keys())}')
        return

    # ── load distances for all targets ────────────────────────────────────
    targets_data = {}   # {label: (dist_array, start_frame)}
    for label in labels:
        dist_arr, start_fr = _load_distances(results_dir, label)
        if dist_arr is not None:
            targets_data[label] = (dist_arr, start_fr)

    if not targets_data:
        print('  [ERROR] No distance data found. '
              'Run ligand_distance_analysis first.')
        return

    print(f'  Loaded {len(targets_data)} targets: {list(targets_data.keys())}')

    # ── load ion counts ────────────────────────────────────────────────────
    ion_counts = _load_ion_counts(results_dir)
    print(f'  Ion count frames loaded: {len(ion_counts)}')

    # ── Q1: permeation distances ───────────────────────────────────────────
    _analyze_permeation_distances(
        targets_data, permeation_events, out_dir, channel_type)

    # ── Q1b: % under threshold at permeation ──────────────────────────────
    _plot_permeation_under_threshold(
        targets_data, permeation_events, out_dir, channel_type, threshold=3.5)

    # ── Q2: distance vs ion count ──────────────────────────────────────────
    _analyze_ion_count_correlation(
        targets_data, ion_counts, out_dir, channel_type)

    # ── Q2b: ion count boxplot — permeation frames only ───────────────────
    _plot_ioncount_boxplot_permeation(
        targets_data, permeation_events, ion_counts, out_dir, channel_type)

    # ── Q2c: % under threshold per n_ions — permeation frames only ────────
    _plot_ioncount_pct_bar_permeation(
        targets_data, permeation_events, ion_counts, out_dir, channel_type)

    # ── Q3: residue combination frequencies ───────────────────────────────
    # build the full list of frames from any target's distance array
    all_frames = []
    for label, (dist_arr, start_fr) in targets_data.items():
        if dist_arr is not None:
            all_frames = list(range(start_fr, start_fr + len(dist_arr)))
            break

    if all_frames:
        _analyze_combinations(
            targets_data, all_frames, permeation_events,
            out_dir, channel_type, threshold=3.5)

    print(f'\n  Done. All outputs in: {out_dir}')
    print('='*70)