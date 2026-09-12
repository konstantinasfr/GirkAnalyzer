#!/usr/bin/env python3
"""
aggregate_ligand_analysis.py
==============================
Aggregates ligand distance and cavity correlation results across ALL runs
for a drug system (G12_ML or G12_GAT).

Reads from disk (no trajectory scan):
    <base_dir>/RUN*/ligand_distance_analysis/<label>/ligand_distance_raw_data.pkl
    <base_dir>/RUN*/cavity_occupancy/trajectory_ion_counts.csv
    <base_dir>/RUN*/ligand_cavity_correlation/combination_frequencies.csv

Reproduces ALL plots from ligand_distance_analysis + ligand_cavity_correlation:

    Distance plots (pooled across runs):
        histogram_<label>.png              — histogram per target
        all_targets_boxplot.png            — 4 boxplots in one figure
        all_targets_timeseries.png         — NOT produced (too many frames)

    Correlation plots (pooled across runs):
        permeation_distances_vs_baseline.png
        permeation_pct_under_35A.png
        ion_count_boxplot.png
        ion_count_scatter.png
        ion_count_boxplot_permeation_only.png
        combinations_all_frames.png
        combinations_permeation_only.png
        combination_frequencies.csv

Usage:
    python3 aggregate_ligand_analysis.py /path/to/G12_ML
    python3 aggregate_ligand_analysis.py /path/to/G12_ML --label G12_ML -o /path/to/out
"""

import argparse
import csv
import pickle
from pathlib import Path
from scipy import stats as scipy_stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── import constants from the per-run scripts ─────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from analysis.ligand_cavity_correlation_analysis import (
        TARGET_LABELS, TARGET_COLORS, COMBINATIONS, COMBO_LABELS, COMBO_COLORS,
        _check_combo, _get_dist_at_frame,
    )
except ImportError:
    # fallback: define inline if import fails
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
    COMBINATIONS = [
        ('GLU141_G1',),
        ('GLU141_G1', 'TRP95_NE1', 'SER177_OG'),
        ('GLU141_G1', 'SER177_OG'),
        ('GLU141_G1', 'TRP95_NE1'),
        ('ASP173_G1',),
        ('ASP173_G1', 'TRP95_NE1', 'SER177_OG'),
        ('ASP173_G1', 'SER177_OG'),
        ('ASP173_G1', 'TRP95_NE1'),
    ]
    COMBO_LABELS = [
        'GLU141\nonly', 'GLU141 + TRP95 + SER177',
        'GLU141 + SER177', 'GLU141 + TRP95',
        'ASP173\nonly', 'ASP173 + TRP95 + SER177',
        'ASP173 + SER177', 'ASP173 + TRP95',
    ]
    COMBO_COLORS = [
        '#922B21', '#C0392B', '#E74C3C', '#F39C12',
        '#1A5276', '#2471A3', '#2980B9', '#1ABC9C',
    ]

    def _get_dist_at_frame(targets_data, label, frame):
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
        for label in combo:
            d = _get_dist_at_frame(targets_data, label, frame)
            if d is None or d >= threshold:
                return False
        return True


FS_TITLE  = 20
FS_LABEL  = 18
FS_TICK   = 16
FS_LEGEND = 14
FS_ANNOT  = 14
THRESHOLD = 3.5


# =============================================================================
# LOADING
# =============================================================================

def load_all_runs(base_dir, channel_type):
    """
    Returns
    -------
    targets_data   : {label: (dist_array, frame_array)}
                     dist_array and frame_array are numpy arrays
                     pooled across ALL runs, aligned by position.
    ion_counts     : {frame_offset: n_total}  — NOT frame numbers, just offsets
                     For correlation we keep a flat list approach.
    perm_end2      : list of int — end_2 frame indices (global, not run-local)
    ion_counts_flat: list of int — n_total per frame, aligned with targets
    """
    labels    = TARGET_LABELS.get(channel_type, [])
    run_dirs  = sorted(Path(base_dir).glob('RUN*'))

    # accumulate per-label distance arrays
    dist_pool  = {l: [] for l in labels}   # list of np arrays per run
    frame_pool = {l: [] for l in labels}   # parallel frame index arrays

    ion_counts_by_frame = {}  # global_frame_id (run_idx*1e7+frame) → n_total
    perm_end2_global    = []  # (dist_idx_in_pool, )

    # for correlation: we need aligned (dist, n_ions) pairs
    aligned_rows = []   # list of {label: dist, 'n_ions': int}
    perm_rows    = []   # same but only at end_2 frames

    loaded_runs = 0
    for run_idx, run_dir in enumerate(run_dirs):
        run_name = run_dir.name

        # ── load distances ────────────────────────────────────────────────
        run_dists = {}
        ok = True
        for label in labels:
            pkl = (run_dir / 'ligand_distance_analysis' / label /
                   'ligand_distance_raw_data.pkl')
            if not pkl.exists():
                ok = False
                break
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            run_dists[label] = (data['min_distance'], int(data['start_frame']))

        if not ok:
            print(f"  [SKIP] {run_name}: missing distance data")
            continue

        # ── load ion counts ────────────────────────────────────────────────
        ion_csv = run_dir / 'cavity_occupancy' / 'trajectory_ion_counts.csv'
        run_ion = {}
        if ion_csv.exists():
            with open(ion_csv, newline='') as f:
                for row in csv.DictReader(f):
                    run_ion[int(row['frame'])] = int(row['n_total'])

        # ── load permeation events from ligand_cavity_correlation ──────────
        # We read the per_event CSV to get end_2 frames for this run
        perm_csv = (run_dir / 'ligand_cavity_correlation' /
                    'permeation_distances.csv')
        run_perm_frames = []
        if perm_csv.exists():
            with open(perm_csv, newline='') as f:
                for row in csv.DictReader(f):
                    run_perm_frames.append(int(row['end_2_frame']))

        # ── pool distances ─────────────────────────────────────────────────
        ref_label  = labels[0]
        ref_arr, ref_start = run_dists[ref_label]
        n_frames   = len(ref_arr)

        for label in labels:
            dist_arr, start_fr = run_dists[label]
            dist_pool[label].append(dist_arr)
            frames = np.arange(start_fr, start_fr + len(dist_arr))
            frame_pool[label].append(frames)

        # ── build aligned correlation rows (all frames) ───────────────────
        for fi in range(n_frames):
            frame = ref_start + fi
            n_ions = run_ion.get(frame)
            if n_ions is None:
                continue
            row = {'n_ions': n_ions}
            for label in labels:
                dist_arr, start_fr = run_dists[label]
                idx = frame - start_fr
                row[label] = float(dist_arr[idx]) if 0 <= idx < len(dist_arr) \
                              else np.nan
            aligned_rows.append(row)

        # ── build perm rows ────────────────────────────────────────────────
        for end2 in run_perm_frames:
            n_ions = run_ion.get(end2)
            row = {'n_ions': n_ions, 'end_2_frame': end2}
            for label in labels:
                dist_arr, start_fr = run_dists[label]
                idx = end2 - start_fr
                row[label] = float(dist_arr[idx]) if 0 <= idx < len(dist_arr) \
                              else np.nan
            perm_rows.append(row)

        loaded_runs += 1
        print(f"  {run_name}: {n_frames} frames, "
              f"{len(run_perm_frames)} perm events, "
              f"{len(run_ion)} ion-count frames")

    if loaded_runs == 0:
        return None, None, None

    # concatenate per-label
    targets_data = {}
    for label in labels:
        if dist_pool[label]:
            combined_dist  = np.concatenate(dist_pool[label])
            combined_frames = np.concatenate(frame_pool[label])
            targets_data[label] = (combined_dist, combined_frames)

    print(f"\n  Loaded {loaded_runs} runs")
    for label, (d, _) in targets_data.items():
        print(f"  {label}: {len(d):,} frames pooled")
    print(f"  Aligned corr rows : {len(aligned_rows):,}")
    print(f"  Permeation events : {len(perm_rows)}")

    return targets_data, aligned_rows, perm_rows


# =============================================================================
# DISTANCE PLOTS
# =============================================================================

def plot_histograms(targets_data, label_sys, out_dir):
    """One histogram per target."""
    for label, (dist_arr, _) in targets_data.items():
        color    = TARGET_COLORS.get(label, '#4878CF')
        n        = len(dist_arr)
        mean_d   = float(np.mean(dist_arr))
        std_d    = float(np.std(dist_arr))
        median_d = float(np.median(dist_arr))
        min_d    = float(np.min(dist_arr))
        max_d    = float(np.max(dist_arr))
        pct_35   = float(np.mean(dist_arr < 3.5) * 100)
        pct_50   = float(np.mean(dist_arr < 5.0) * 100)

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(dist_arr, bins=80, color=color, alpha=0.70,
                edgecolor='white', linewidth=0.4)

        ax.axvline(mean_d,   color='red',    linewidth=2.5, linestyle='--',
                   label=f'mean   = {mean_d:.2f} Å')
        ax.axvline(median_d, color='orange', linewidth=2.5, linestyle=':',
                   label=f'median = {median_d:.2f} Å')
        ax.axvline(min_d,    color='green',  linewidth=1.8, linestyle='-.',
                   label=f'min    = {min_d:.2f} Å')
        ax.axvline(max_d,    color='purple', linewidth=1.8, linestyle='-.',
                   label=f'max    = {max_d:.2f} Å')
        ax.axvline(3.5, color='grey', linewidth=1.4, linestyle='--',
                   alpha=0.7, label=f'3.5 Å  ({pct_35:.1f}%)')
        ax.axvline(5.0, color='grey', linewidth=1.4, linestyle=':',
                   alpha=0.7, label=f'5.0 Å  ({pct_50:.1f}%)')

        from matplotlib.patches import Patch
        count_h = Patch(facecolor='none', edgecolor='none',
                        label=f'n = {n:,} frames   std = {std_d:.2f} Å')
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles=[count_h] + handles,
                  labels=[count_h.get_label()] + lbls,
                  fontsize=FS_LEGEND, loc='upper right',
                  framealpha=0.92, edgecolor='black')

        ax.set_xlabel('Distance (Å)', fontsize=FS_LABEL)
        ax.set_ylabel('Number of frames', fontsize=FS_LABEL)
        ax.set_title(f'{label_sys}  —  {label}\n'
                     f'min distance  (pooled across all runs)',
                     fontsize=FS_TITLE, pad=12)
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        out = out_dir / f'histogram_{label}.png'
        plt.savefig(out, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {out}')


def plot_boxplot(targets_data, label_sys, out_dir):
    """All 4 targets in one boxplot figure."""
    labels    = list(targets_data.keys())
    all_vals  = [targets_data[l][0] for l in labels]
    colors    = [TARGET_COLORS.get(l, '#4878CF') for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 3), 8))
    bp = ax.boxplot(
        all_vals,
        positions=range(len(labels)),
        widths=0.50,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2, alpha=0.15,
                        markeredgecolor='none'),
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for flier, color in zip(bp['fliers'], colors):
        flier.set(markerfacecolor=color)

    tick_labels = []
    for l, vals in zip(labels, all_vals):
        tick_labels.append(
            f'{l}\nmean={np.mean(vals):.2f}\nmedian={np.median(vals):.2f}'
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK)
    ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
    ax.set_title(f'{label_sys}  —  Ligand distances — all targets\n'
                 f'(pooled across all runs)',
                 fontsize=FS_TITLE, pad=14)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = out_dir / 'all_targets_boxplot.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# =============================================================================
# CORRELATION PLOTS
# =============================================================================

def plot_permeation_vs_baseline(targets_data, perm_rows, label_sys, out_dir):
    """Bar: mean distance at end_2 vs all frames baseline."""
    labels = list(targets_data.keys())
    if not perm_rows:
        print('  No permeation rows — skipping baseline comparison.')
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.5), 7))
    x     = np.arange(len(labels))
    width = 0.38

    for xi, label in enumerate(labels):
        color    = TARGET_COLORS.get(label, '#555555')
        baseline = targets_data[label][0]
        at_perm  = np.array([r[label] for r in perm_rows
                              if r.get(label) is not None and
                              not np.isnan(r.get(label, np.nan))])

        for vals, offset, face_alpha, hatch in [
            (baseline, -width/2, 0.75, None),
            (at_perm,  +width/2, 0.85, '//'),
        ]:
            if len(vals) == 0:
                continue
            mean = np.mean(vals)
            std  = np.std(vals)
            ax.bar(xi + offset, mean, width, color=color,
                   alpha=face_alpha, edgecolor='black',
                   linewidth=1.1, hatch=hatch, zorder=2)
            ax.errorbar(xi + offset, mean, yerr=std, fmt='none',
                        color='black', capsize=6, elinewidth=1.8, zorder=3)
            ax.text(xi + offset, mean + std + 0.05,
                    f'{mean:.2f}\n±{std:.2f}',
                    ha='center', va='bottom', fontsize=FS_ANNOT - 1)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='grey', alpha=0.75, edgecolor='black',
              label='All frames'),
        Patch(facecolor='grey', alpha=0.85, edgecolor='black',
              hatch='//', label='At permeation (end_2)'),
    ], fontsize=FS_LEGEND, loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
    ax.set_title(f'{label_sys}  —  Ligand distances: permeation vs baseline\n'
                 f'n_events = {len(perm_rows)}  (pooled across all runs)',
                 fontsize=FS_TITLE, pad=14)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = out_dir / 'permeation_distances_vs_baseline.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def plot_permeation_pct_under(targets_data, perm_rows, label_sys, out_dir,
                               threshold=3.5):
    """% of permeation events with distance < threshold per target."""
    labels = list(targets_data.keys())
    if not perm_rows:
        return

    counts, totals, pcts = [], [], []
    for label in labels:
        vals    = [r[label] for r in perm_rows
                   if r.get(label) is not None and
                   not np.isnan(r.get(label, np.nan))]
        n_total = len(vals)
        n_under = sum(1 for v in vals if v < threshold)
        totals.append(n_total)
        counts.append(n_under)
        pcts.append(100. * n_under / n_total if n_total > 0 else 0.)

    colors = [TARGET_COLORS.get(l, '#555555') for l in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 7))
    bars = ax.bar(range(len(labels)), pcts, color=colors, alpha=0.80,
                  edgecolor='black', linewidth=1.2, width=0.6)
    for bar, pct, cnt, tot in zip(bars, pcts, counts, totals):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(pcts + [1]) * 0.02,
                f'{pct:.1f}%\n({cnt}/{tot})',
                ha='center', va='bottom', fontsize=FS_ANNOT, fontweight='bold')
    ax.axhline(50, color='grey', linewidth=1.2, linestyle='--',
               alpha=0.5, label='50%')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel(f'% of permeation events < {threshold} Å', fontsize=FS_LABEL)
    ax.set_title(f'{label_sys}  —  % permeation events with distance < {threshold} Å\n'
                 f'n_events = {totals[0] if totals else 0}  (pooled)',
                 fontsize=FS_TITLE, pad=14)
    ax.set_ylim(0, max(pcts + [5]) * 1.30)
    ax.legend(fontsize=FS_LEGEND)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = out_dir / f'permeation_pct_under_{int(threshold*10)}A.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def plot_ion_count_boxplot(aligned_rows, perm_rows, targets_data,
                           label_sys, out_dir, permeation_only=False):
    """Distance grouped by n_ions — all frames or permeation only."""
    rows   = perm_rows if permeation_only else aligned_rows
    labels = list(targets_data.keys())
    if not rows:
        return

    n_ions_arr = np.array([r['n_ions'] for r in rows
                            if r.get('n_ions') is not None])
    unique_ns  = sorted(set(n_ions_arr))
    min_n      = 3 if permeation_only else 20
    valid_ns   = [n for n in unique_ns
                  if np.sum(n_ions_arr == n) >= min_n]
    if not valid_ns:
        return

    fig, axes = plt.subplots(1, len(labels),
                             figsize=(max(12, len(labels) * 4), 7),
                             sharey=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        color  = TARGET_COLORS.get(label, '#555555')
        d_arr  = np.array([r.get(label, np.nan) for r in rows])
        n_arr  = np.array([r.get('n_ions', np.nan) for r in rows])
        bdata  = [d_arr[n_arr == n] for n in valid_ns]
        bdata  = [g[~np.isnan(g)] for g in bdata]

        ax.boxplot(
            bdata, positions=range(len(valid_ns)), widths=0.55,
            patch_artist=True, showfliers=True,
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

    suffix = 'permeation frames only' if permeation_only else 'all frames'
    n_shown = len(perm_rows) if permeation_only else len(aligned_rows)
    fig.suptitle(f'{label_sys}  —  Ligand distances grouped by cavity ion count\n'
                 f'{suffix}  (n = {n_shown:,}  pooled)',
                 fontsize=FS_TITLE, y=1.01)
    plt.tight_layout()
    fname = ('ion_count_boxplot_permeation_only.png' if permeation_only
             else 'ion_count_boxplot.png')
    out = out_dir / fname
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def plot_ion_count_scatter(aligned_rows, targets_data, label_sys, out_dir):
    """Scatter: distance vs n_ions with correlation + trend line."""
    labels = list(targets_data.keys())
    if not aligned_rows:
        return

    fig, axes = plt.subplots(1, len(labels),
                             figsize=(max(12, len(labels) * 4), 6),
                             sharey=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        color  = TARGET_COLORS.get(label, '#555555')
        x_all  = np.array([r['n_ions'] for r in aligned_rows
                            if r.get('n_ions') is not None and
                            r.get(label) is not None and
                            not np.isnan(r.get(label, np.nan))])
        y_all  = np.array([r[label] for r in aligned_rows
                            if r.get('n_ions') is not None and
                            r.get(label) is not None and
                            not np.isnan(r.get(label, np.nan))])

        rng  = np.random.default_rng(42)
        x_j  = x_all.astype(float) + rng.uniform(-0.25, 0.25, len(x_all))
        ax.scatter(x_j, y_all, color=color, alpha=0.06, s=5, edgecolors='none')

        if len(x_all) >= 10:
            slope, intercept, r, p, _ = scipy_stats.linregress(x_all, y_all)
            xl = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xl, slope * xl + intercept, color='black', linewidth=2.0,
                    label=f'r={r:+.3f}  p={p:.2e}')
            ax.legend(fontsize=FS_LEGEND)

        ax.set_xlabel('N ions in cavity', fontsize=FS_LABEL)
        ax.set_ylabel('Min distance (Å)', fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE, fontweight='bold', pad=10)
        ax.set_xticks(sorted(set(x_all.astype(int))))
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{label_sys}  —  Ligand distance vs cavity ion count  (pooled)',
                 fontsize=FS_TITLE, y=1.01)
    plt.tight_layout()
    out = out_dir / 'ion_count_scatter.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def save_correlation_csv(aligned_rows, targets_data, label_sys, out_dir):
    """Pearson + Spearman correlation table."""
    labels = list(targets_data.keys())
    if not aligned_rows:
        return

    n_ions = np.array([r['n_ions'] for r in aligned_rows
                       if r.get('n_ions') is not None])

    rows = []
    print(f'\n  {"─"*65}')
    print(f'  {"TARGET":<22} {"Pearson r":>10} {"p":>12} '
          f'{"Spearman ρ":>12} {"p":>12}')
    print(f'  {"─"*65}')
    for label in labels:
        d = np.array([r.get(label, np.nan) for r in aligned_rows])
        mask = ~np.isnan(d)
        if mask.sum() < 10:
            continue
        r_p, p_p = scipy_stats.pearsonr(n_ions[mask], d[mask])
        r_s, p_s = scipy_stats.spearmanr(n_ions[mask], d[mask])
        rows.append({
            'target': label, 'n_frames': int(mask.sum()),
            'pearson_r': round(r_p, 4), 'pearson_p': f'{p_p:.3e}',
            'spearman_rho': round(r_s, 4), 'spearman_p': f'{p_s:.3e}',
        })
        print(f'  {label:<22} {r_p:>+10.4f} {p_p:>12.3e} '
              f'{r_s:>+12.4f} {p_s:>12.3e}')
    print(f'  {"─"*65}')

    if rows:
        out = out_dir / 'ion_count_correlation.csv'
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'  Saved: {out}')


def plot_combinations(targets_data, aligned_rows, perm_rows,
                      label_sys, out_dir, threshold=3.5):
    """Combination bar plots — all frames and permeation only."""

    def _count_combo(rows_list, combo):
        count = 0
        for r in rows_list:
            if all(r.get(c) is not None and not np.isnan(r.get(c, np.nan))
                   and r[c] < threshold for c in combo):
                count += 1
        return count

    counts_all  = [_count_combo(aligned_rows, c) for c in COMBINATIONS]
    counts_perm = [_count_combo(perm_rows,    c) for c in COMBINATIONS]
    n_all  = len(aligned_rows)
    n_perm = len(perm_rows)

    # print table
    print(f'\n  Combination frequencies (threshold = {threshold} Å):')
    for clabel, ca, cp in zip(COMBO_LABELS, counts_all, counts_perm):
        name = clabel.replace('\n', ' ')
        pct_a = 100. * ca / n_all  if n_all  > 0 else 0.
        pct_p = 100. * cp / n_perm if n_perm > 0 else 0.
        print(f'  {name:<32}  all={pct_a:5.1f}% ({ca:>5}/{n_all})  '
              f'perm={pct_p:5.1f}% ({cp}/{n_perm})')

    # save CSV
    csv_rows = []
    for clabel, combo, ca, cp in zip(COMBO_LABELS, COMBINATIONS,
                                      counts_all, counts_perm):
        csv_rows.append({
            'combination':     clabel.replace('\n', ' '),
            'members':         ' + '.join(combo),
            'n_all_frames':    ca,
            'pct_all_frames':  round(100. * ca / n_all,  2) if n_all  > 0 else 0.,
            'n_perm_frames':   cp,
            'pct_perm_frames': round(100. * cp / n_perm, 2) if n_perm > 0 else 0.,
            'n_total_frames':  n_all,
            'n_total_perm':    n_perm,
            'threshold_A':     threshold,
        })
    with open(out_dir / 'combination_frequencies.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f'  Saved: {out_dir / "combination_frequencies.csv"}')

    def _bar(counts, n_total, title, out_path):
        pcts    = [100. * c / n_total if n_total > 0 else 0. for c in counts]
        max_pct = max(pcts + [1])
        fig, ax = plt.subplots(figsize=(max(14, len(COMBO_LABELS) * 2.0), 8))
        x    = np.arange(len(COMBO_LABELS))
        bars = ax.bar(x, pcts, color=COMBO_COLORS, alpha=0.82,
                      edgecolor='black', linewidth=1.2, width=0.65)
        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + max_pct * 0.02,
                    f'{pct:.1f}%\n(n={cnt:,})',
                    ha='center', va='bottom',
                    fontsize=FS_ANNOT, fontweight='bold')
        ax.axvline(3.5, color='black', linewidth=1.2, linestyle=':',
                   alpha=0.4)
        ax.axhline(50, color='grey', linewidth=1.2, linestyle='--',
                   alpha=0.5, label='50%')
        ax.set_xticks(x)
        ax.set_xticklabels(COMBO_LABELS, fontsize=FS_TICK - 1,
                           rotation=25, ha='right')
        ax.set_ylabel(f'% of frames (total = {n_total:,})', fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, pad=14)
        ax.set_ylim(0, max_pct * 1.35)
        ax.legend(fontsize=FS_LEGEND)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {out_path}')

    _bar(counts_all, n_all,
         f'{label_sys}  —  Residue combinations within {threshold} Å of drug\n'
         f'ALL simulation frames  (n = {n_all:,}  pooled)',
         out_dir / 'combinations_all_frames.png')

    _bar(counts_perm, n_perm,
         f'{label_sys}  —  Residue combinations within {threshold} Å of drug\n'
         f'PERMEATION frames only  (n_events = {n_perm}  pooled)',
         out_dir / 'combinations_permeation_only.png')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate ligand distance + correlation analysis across all runs.'
    )
    parser.add_argument('base_dir',
                        help='Directory containing RUN1, RUN2, ... subdirectories')
    parser.add_argument('--label', default=None,
                        help='System label for plots (default: folder name)')
    parser.add_argument('-c', '--channel', default='G12_ML',
                        choices=['G12_ML', 'G12_GAT'],
                        help='Channel/drug type (default: G12_ML)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: <base_dir>/ligand_aggregated)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    label_sys = args.label or base_dir.name
    out_dir   = Path(args.output) if args.output else \
                base_dir / 'ligand_aggregated'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'AGGREGATE LIGAND ANALYSIS  —  {label_sys}')
    print(f'{"="*70}')
    print(f'Base dir  : {base_dir}')
    print(f'Output    : {out_dir}')
    print(f'Channel   : {args.channel}\n')

    # ── load ──────────────────────────────────────────────────────────────
    targets_data, aligned_rows, perm_rows = load_all_runs(base_dir, args.channel)
    if targets_data is None:
        print('ERROR: No data loaded.')
        return 1

    # ── distance plots ─────────────────────────────────────────────────────
    print(f'\n  Distance plots...')
    plot_histograms(targets_data, label_sys, out_dir)
    plot_boxplot(targets_data, label_sys, out_dir)

    # ── correlation plots ──────────────────────────────────────────────────
    print(f'\n  Correlation plots...')
    plot_permeation_vs_baseline(targets_data, perm_rows, label_sys, out_dir)
    plot_permeation_pct_under(targets_data, perm_rows, label_sys, out_dir)
    plot_ion_count_boxplot(aligned_rows, perm_rows, targets_data,
                           label_sys, out_dir, permeation_only=False)
    plot_ion_count_boxplot(aligned_rows, perm_rows, targets_data,
                           label_sys, out_dir, permeation_only=True)
    plot_ion_count_scatter(aligned_rows, targets_data, label_sys, out_dir)
    save_correlation_csv(aligned_rows, targets_data, label_sys, out_dir)
    plot_combinations(targets_data, aligned_rows, perm_rows,
                      label_sys, out_dir)

    print(f'\n{"="*70}')
    print(f'DONE. All outputs in: {out_dir}')
    print(f'{"="*70}\n')
    return 0


if __name__ == '__main__':
    exit(main())


'''
python3 aggregate_ligand_analysis.py /path/to/G12_ML --channel G12_ML
python3 aggregate_ligand_analysis.py /path/to/G12_GAT --channel G12_GAT -o /path/to/output
'''