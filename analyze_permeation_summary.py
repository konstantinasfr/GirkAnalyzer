#!/usr/bin/env python3
"""
Permeation Event Summary Analyzer

Analyzes permeation events across multiple simulation runs.
Reports complete permeations (permeation_table.json) and a last-frame
snapshot of where ions were sitting in each biological region
(ion_region_tracking.json).

Usage:
    python analyze_permeation_summary.py /path/to/results/G12_GAT
    python analyze_permeation_summary.py /path/to/results/G12_GAT --runs 1 2 3 4 5

    # If your trajectories are not 0.075 ns per frame, say so:
    python analyze_permeation_summary.py /path/to/results/G12_GAT -ns_per_frame 0.1
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────
# Font sizes are deliberately large: these figures get shrunk down onto
# PowerPoint slides, and anything sized for a full-page figure becomes
# unreadable at slide scale. Bump FONT_SCALE to go larger still.

FONT_SCALE   = 1.0
FS_TITLE     = int(26 * FONT_SCALE)
FS_AXIS      = int(24 * FONT_SCALE)
FS_TICK      = int(21 * FONT_SCALE)
FS_LEGEND    = int(18 * FONT_SCALE)
FS_ANNOT     = int(15 * FONT_SCALE)
FS_BARVALUE  = int(17 * FONT_SCALE)


# ─────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────

def find_run_dirs(base_dir, run_numbers=None):
    base_dir = Path(base_dir)
    if run_numbers is None:
        run_dirs = sorted(base_dir.glob("RUN*"))
        print(f"Found {len(run_dirs)} RUN directories")
    else:
        run_dirs = [base_dir / f"RUN{n}" for n in run_numbers]
        print(f"Looking for {len(run_dirs)} specified RUN directories")
    return [d for d in run_dirs if d.is_dir()]


# ─────────────────────────────────────────────
# Frame count
# ─────────────────────────────────────────────

def get_trajectory_frames(run_dir):
    """Try to get total frame count from saved pickle state, then fallback."""
    run_dir = Path(run_dir)

    state_file = run_dir / "analyzer_state.pkl"
    if state_file.exists():
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            if 'end_frame' in state:
                return int(state['end_frame']) + 1
        except Exception:
            pass

    try:
        import MDAnalysis as mda
        for traj_name in ['protein.nc', 'trajectory.nc', 'traj.nc', 'prod.nc']:
            traj_file = run_dir / traj_name
            if not traj_file.exists():
                continue
            for top_name in ['com_4fs.prmtop', 'com.prmtop', 'topology.prmtop']:
                top_file = run_dir / top_name
                if top_file.exists():
                    u = mda.Universe(str(top_file), str(traj_file))
                    return len(u.trajectory)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────
# Stage counting from ion_region_tracking.json
# ─────────────────────────────────────────────

def count_stages_from_tracking(run_dir):
    """
    Read ion_region_tracking.json and report where each ion was sitting
    at the LAST frame it was observed inside the channel.

    Biological meaning of regions:
        1 = SF region        (R1: start1 -> start2)
        2 = GLU/ASN cavity   (R2: start2 -> end2)
        3 = GLU/ASN -> HBC   (R3: end2 -> end3)
        4 = After HBC        (R4: past end3) — completed permeation

    ion_region_tracking structure: { "ion_id": { "frame": region, ... } }

    Returns dict with counts, or None if file not found.
    """
    tracking_file = Path(run_dir) / "ion_region_tracking.json"
    if not tracking_file.exists():
        return None

    with open(tracking_file, 'r') as f:
        ion_region_tracking = json.load(f)

    # For each ion, get the region it was in at its last recorded frame
    last_frame_region = {}
    for ion_id_str, frame_dict in ion_region_tracking.items():
        if not frame_dict:
            continue
        last_frame = max(int(f) for f in frame_dict.keys())
        last_frame_region[int(ion_id_str)] = int(frame_dict[str(last_frame)])

    counts = defaultdict(int)
    for region in last_frame_region.values():
        counts[region] += 1

    return {
        'total_tracked_ions': len(last_frame_region),
        'In_SF':              counts.get(1, 0),   # R1: SF region
        'In_GLU_ASN':         counts.get(2, 0),   # R2: GLU/ASN cavity
        'In_ASN_HBC':         counts.get(3, 0),   # R3: GLU/ASN -> HBC
        'After_HBC':          counts.get(4, 0),   # R4: completed permeation
    }


# ─────────────────────────────────────────────
# Permeation table (complete events only)
# ─────────────────────────────────────────────

def load_permeation_table(run_dir):
    """
    Load permeation_table.json.
    Returns (total_entries, unique_ions) or (0, 0) if file missing.
    """
    perm_file = Path(run_dir) / "permeation_table.json"
    if not perm_file.exists():
        return 0, 0

    with open(perm_file, 'r') as f:
        events = json.load(f)

    total_entries = len(events)
    unique_ions   = len(set(e['ion_id'] for e in events))
    return total_entries, unique_ions




# ─────────────────────────────────────────────
# Per-run analysis
# ─────────────────────────────────────────────

def analyze_run(run_dir):
    run_dir  = Path(run_dir)
    run_name = run_dir.name
    print(f"\nProcessing {run_name}...")

    n_frames                       = get_trajectory_frames(run_dir)
    perm_entries, perm_unique_ions = load_permeation_table(run_dir)
    stage_counts                   = count_stages_from_tracking(run_dir)

    row = {
        'Run':                       run_name,
        'Total_Frames':              n_frames,
        'Full_Permeations_Unique':   perm_unique_ions,
        'Full_Permeations_Total':    perm_entries,
        'Entries_Per_Frame':         perm_entries / n_frames if n_frames else 0.0,
    }

    if stage_counts is not None:
        row.update({
            'Total_Tracked_Ions': stage_counts['total_tracked_ions'],
            'In_SF':              stage_counts['In_SF'],
            'In_GLU_ASN':         stage_counts['In_GLU_ASN'],
            'In_ASN_HBC':         stage_counts['In_ASN_HBC'],
            'After_HBC':          stage_counts['After_HBC'],
            'ion_region_tracking_found': True,
        })
        # HBC permeations = full permeations + ions still sitting after HBC
        after_hbc = stage_counts['After_HBC']
        row['HBC_Permeations'] = perm_entries + after_hbc
        print(f"  Frames:          {n_frames}")
        print(f"  HBC permeations (Full_Tot + AfterHBC): {row['HBC_Permeations']}")
        print(f"  Full permeations: {perm_entries} total  ({perm_unique_ions} unique ions)")
        print(f"  Last-frame snapshot (ion_region_tracking.json):")
        print(f"    Total ions tracked: {stage_counts['total_tracked_ions']}")
        print(f"    In SF region:       {stage_counts['In_SF']}")
        print(f"    In GLU/ASN cavity:  {stage_counts['In_GLU_ASN']}")
        print(f"    In ASN->HBC:    {stage_counts['In_ASN_HBC']}")
        print(f"    After HBC:          {stage_counts['After_HBC']}")
    else:
        row.update({
            'Total_Tracked_Ions':        None,
            'In_SF':                     None,
            'In_GLU_ASN':                None,
            'In_ASN_HBC':                None,
            'After_HBC':                 None,
            'ion_region_tracking_found': False,
        })
        row['HBC_Permeations'] = None
        print(f"  WARNING: ion_region_tracking.json not found")

    return row



# ─────────────────────────────────────────────
# Quartile classification of runs
# ─────────────────────────────────────────────

CATEGORY_ORDER  = ['HIGH', 'TYPICAL', 'LOW']
CATEGORY_COLORS = {'HIGH': '#C44E52', 'TYPICAL': '#4C72B0', 'LOW': '#8172B2'}


def classify_runs_by_quartiles(df, metric='Full_Permeations_Total'):
    """
    Splits runs into three categories using the quartiles of `metric`
    across the runs:

        HIGH    : value >  Q3
        TYPICAL : Q1 <= value <= Q3
        LOW     : value <  Q1

    Returns (categories, q1, q3) where `categories` is a pandas Series
    aligned to df's index (NaN for runs with no value for the metric).
    """
    if metric not in df.columns:
        return pd.Series([None] * len(df), index=df.index), None, None

    mask = df[metric].notna()
    if not mask.any():
        return pd.Series([None] * len(df), index=df.index), None, None

    vals = df.loc[mask, metric].astype(float)
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))

    def label(v):
        if v > q3:
            return 'HIGH'
        if v < q1:
            return 'LOW'
        return 'TYPICAL'

    categories = pd.Series([None] * len(df), index=df.index, dtype=object)
    categories.loc[mask] = [label(v) for v in vals]
    return categories, q1, q3


def write_category_outputs(df, output_dir, metric='Full_Permeations_Total',
                           cat_col='Category'):
    """
    Writes a per-category CSV and a plain-text list of run names per
    category, and prints the breakdown.
    """
    output_dir = Path(output_dir)
    if cat_col not in df.columns or df[cat_col].isna().all():
        return

    sub = df[df[cat_col].notna()].copy()
    sub = sub.sort_values(metric, ascending=False)
    q1 = float(sub['Q1'].iloc[0])
    q3 = float(sub['Q3'].iloc[0])
    med = float(sub[metric].astype(float).median())

    keep = ['Run', cat_col, metric, 'Q1', 'Q3']
    extra = [c for c in ['Total_Frames', 'Full_Permeations_Total',
                         'Full_Permeations_Unique', 'HBC_Permeations',
                         'Entries_Per_Frame', 'Total_Tracked_Ions',
                         'In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']
             if c in sub.columns and c not in keep]
    out_csv = output_dir / f'run_categories_{metric.lower()}.csv'
    sub[keep + extra].to_csv(out_csv, index=False)
    print(f"✅ run_categories CSV saved: {out_csv.name}")

    # plain run lists, with ready-made --runs strings for the other scripts
    lists_file = output_dir / f'run_categories_{metric.lower()}_lists.txt'
    with open(lists_file, 'w') as f:
        f.write(f"Metric: {metric}\n")
        f.write(f"Q1 = {q1:.3f}   Q3 = {q3:.3f}   median = {med:.3f}   n = {len(sub)} runs\n")
        f.write("HIGH = value > Q3,  TYPICAL = Q1 <= value <= Q3,  LOW = value < Q1\n\n")
        for cat in CATEGORY_ORDER:
            names = sub.loc[sub[cat_col] == cat, 'Run'].tolist()
            f.write(f"{cat} (n={len(names)}): {' '.join(names)}\n")
            nums = [n.replace('RUN', '') for n in names]
            f.write(f"  --runs {' '.join(nums)}\n\n")
    print(f"✅ run_categories lists saved: {lists_file.name}")

    print("\n" + "=" * 70)
    print(f"RUN CATEGORIES — {metric}")
    print("=" * 70)
    print(f"Q1 = {q1:.3f}    Q3 = {q3:.3f}    median = {med:.3f}    n = {len(sub)} runs")
    print("-" * 70)
    for cat in CATEGORY_ORDER:
        rows = sub[sub[cat_col] == cat]
        if rows.empty:
            print(f"{cat:<8} n=0")
            continue
        print(f"{cat:<8} n={len(rows):<3} mean={rows[metric].astype(float).mean():.2f}  "
              f"runs: {', '.join(rows['Run'].tolist())}")
    print("=" * 70)


# ─────────────────────────────────────────────
# Pooled permeation rates (all runs as one trajectory)
# ─────────────────────────────────────────────

def write_pooled_rate_csv(df, output_dir, ns_per_frame):
    """
    Writes permeation RATES rather than raw counts.

    Two things are reported, and they are not the same number:

      * per-run rates: count / frames for that run, converted to per ns
      * POOLED rate:   sum(counts) / sum(frames) across every run,
                       i.e. all runs treated as one long trajectory

    The pooled rate is frame-weighted, so a short run no longer carries
    the same weight as a long one. It is the right number to quote for
    "the permeation rate of this system"; the mean of per-run rates is
    not, unless every run happens to be the same length.

    Rates are also given per microsecond, since per-ns values for a
    channel like this tend to be small decimals that read badly on a
    slide.
    """
    output_dir = Path(output_dir)

    metrics = [m for m in ['Full_Permeations_Total',
                           'Full_Permeations_Unique',
                           'HBC_Permeations'] if m in df.columns]
    if not metrics:
        print("  No permeation metrics present — pooled rate CSV skipped.")
        return None

    usable = df[df['Total_Frames'].notna()].copy()
    if usable.empty:
        print("  No runs have a frame count — pooled rate CSV skipped.")
        return None

    n_missing = len(df) - len(usable)
    if n_missing:
        print(f"  NOTE: {n_missing} run(s) have no frame count and are excluded "
              f"from the pooled rate.")

    rows = []

    # ---- per-run rows ----
    for _, r in usable.iterrows():
        frames = float(r['Total_Frames'])
        ns = frames * ns_per_frame
        row = {
            'Run':       r['Run'],
            'Frames':    int(frames),
            'Time_ns':   round(ns, 3),
        }
        for m in metrics:
            val = r.get(m)
            if pd.isna(val):
                row[f'{m}'] = None
                row[f'{m}_per_frame'] = None
                row[f'{m}_per_ns'] = None
                row[f'{m}_per_us'] = None
            else:
                val = float(val)
                row[f'{m}'] = val
                row[f'{m}_per_frame'] = val / frames if frames else None
                row[f'{m}_per_ns'] = val / ns if ns else None
                row[f'{m}_per_us'] = (val / ns * 1000.0) if ns else None
        rows.append(row)

    # ---- pooled row ----
    total_frames = float(usable['Total_Frames'].sum())
    total_ns = total_frames * ns_per_frame
    pooled = {
        'Run':     f'POOLED ({len(usable)} runs)',
        'Frames':  int(total_frames),
        'Time_ns': round(total_ns, 3),
    }
    for m in metrics:
        total = float(usable[m].sum(skipna=True))
        pooled[f'{m}'] = total
        pooled[f'{m}_per_frame'] = total / total_frames if total_frames else None
        pooled[f'{m}_per_ns'] = total / total_ns if total_ns else None
        pooled[f'{m}_per_us'] = (total / total_ns * 1000.0) if total_ns else None
    rows.append(pooled)

    out = pd.DataFrame(rows)
    csv_file = output_dir / 'permeation_rates.csv'
    out.to_csv(csv_file, index=False)
    print(f"✅ permeation_rates.csv saved")

    # ---- printed summary ----
    print("\n" + "=" * 70)
    print("POOLED PERMEATION RATES (all runs as one trajectory)")
    print("=" * 70)
    print(f"ns per frame:   {ns_per_frame}")
    print(f"Runs pooled:    {len(usable)}")
    print(f"Total frames:   {int(total_frames):,}")
    print(f"Total time:     {total_ns:,.1f} ns  ({total_ns/1000.0:.3f} \u00b5s)")
    print("-" * 70)
    print(f"{'Metric':<28} {'Count':>8} {'/frame':>12} {'/ns':>12} {'/\u00b5s':>10}")
    print("-" * 70)
    for m in metrics:
        print(f"{m:<28} {pooled[m]:>8.0f} {pooled[f'{m}_per_frame']:>12.6f} "
              f"{pooled[f'{m}_per_ns']:>12.4f} {pooled[f'{m}_per_us']:>10.1f}")
    print("=" * 70)

    return pooled


# ─────────────────────────────────────────────
# Boxplot of permeation counts across runs
# ─────────────────────────────────────────────

def plot_permeation_boxplot(df, output_dir, metric='Full_Permeations_Total',
                            cat_col=None, q1=None, q3=None, channel=None):
    """
    Boxplot of per-run permeation counts, with every run shown as a labelled
    dot. Points sharing the same value are fanned out horizontally so the run
    labels don't sit on top of each other.

    If cat_col is given, dots are coloured by quartile category (HIGH /
    TYPICAL / LOW) and the Q1/Q3 split lines are drawn.

    Sized and lettered for slides: the figure is large and the fonts are
    large relative to it, so shrinking the PNG into a PowerPoint frame
    keeps the text readable.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    if metric not in df.columns:
        print(f"  Column {metric} not present — boxplot skipped.")
        return

    sub = df[df[metric].notna()].copy()
    if sub.empty:
        print(f"  No data for {metric} — boxplot skipped.")
        return

    vals = sub[metric].astype(float).to_numpy()
    runs = sub['Run'].tolist()
    cats = sub[cat_col].tolist() if (cat_col and cat_col in sub.columns) else None

    # beeswarm x-positions: fan ties out symmetrically around x=1
    xs = np.ones(len(vals))
    step = 0.17
    for v in np.unique(vals):
        idx = np.where(vals == v)[0]
        n = len(idx)
        xs[idx] = 1 + (np.arange(n) - (n - 1) / 2.0) * step

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.boxplot(vals, widths=0.45, patch_artist=True, showfliers=False,
               medianprops=dict(color='green', linewidth=2.6),
               boxprops=dict(facecolor='#EAF2FB', edgecolor='#2E4057', linewidth=1.8),
               whiskerprops=dict(color='#2E4057', linewidth=1.8),
               capprops=dict(color='#2E4057', linewidth=1.8))

    mean_val, med_val = float(np.mean(vals)), float(np.median(vals))
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2.0,
               label=f'mean = {mean_val:.1f}')
    ax.plot([], [], color='green', linewidth=2.6, label=f'median = {med_val:.1f}')

    if cats is not None:
        if q3 is not None:
            ax.axhline(q3, color=CATEGORY_COLORS['HIGH'], linestyle=':', linewidth=2.0,
                       label=f'Q3 = {q3:.1f}')
        if q1 is not None:
            ax.axhline(q1, color=CATEGORY_COLORS['LOW'], linestyle=':', linewidth=2.0,
                       label=f'Q1 = {q1:.1f}')
        for cat in CATEGORY_ORDER:
            m = np.array([c == cat for c in cats])
            if not m.any():
                continue
            ax.scatter(xs[m], vals[m], s=110, color=CATEGORY_COLORS[cat],
                       edgecolor='white', linewidth=1.2, zorder=3,
                       label=f'{cat} (n={int(m.sum())})')
    else:
        ax.scatter(xs, vals, s=95, color='#2E4057', edgecolor='white',
                   linewidth=1.2, zorder=3)

    for xi, yi, name in zip(xs, vals, runs):
        ax.annotate(name, (xi, yi), textcoords='offset points', xytext=(9, 0),
                    fontsize=FS_ANNOT, va='center', color='#2E4057')

    ax.set_xticks([1])
    ax.set_xticklabels([f'n = {len(vals)} runs'], fontsize=FS_AXIS)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=FS_AXIS)
    title = f'{metric.replace("_", " ")} per run'
    if channel:
        title = f'{channel}: {title}'
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='x', labelsize=FS_AXIS)
    ax.set_xlim(xs.min() - 0.35, xs.max() + 0.55)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)

    plt.tight_layout()
    out = output_dir / f'boxplot_{metric.lower()}.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {out}")


# ─────────────────────────────────────────────
# Region occupancy plot
# ─────────────────────────────────────────────

def plot_last_frame_regions(df, output_dir, channel=None):
    """
    Bar plot showing how many ions were in each biological region
    at their last observed frame, for each run.
    Also plots a stacked bar across all runs combined.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    region_cols   = ['In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']
    region_labels = ['SF (R1)', 'GLU/ASN cavity (R2)', 'ASN→HBC (R3)', 'After HBC (R4)']
    colors        = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    # Filter to runs that have tracking data
    plot_df = df[df['ion_region_tracking_found'] == True].copy()
    if plot_df.empty:
        print("  No ion_region_tracking data available for plotting.")
        return

    runs = plot_df['Run'].tolist()
    x    = range(len(runs))
    width = 0.18

    # ── Plot 1: grouped bar per run ──
    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 1.5), 7.5))
    for i, (col, label, color) in enumerate(zip(region_cols, region_labels, colors)):
        vals = plot_df[col].fillna(0).astype(int).tolist()
        offsets = [xi + (i - 1.5) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=label, color=color, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(v), ha='center', va='bottom', fontsize=FS_BARVALUE)

    ax.set_xticks(list(x))
    ax.set_xticklabels(runs, rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_ylabel('Number of ions', fontsize=FS_AXIS)
    t1 = 'Ion region occupancy at last observed frame (per run)'
    ax.set_title(f'{channel}: {t1}' if channel else t1,
                 fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = output_dir / 'last_frame_regions_per_run.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✅ Plot saved: {out}")

    # ── Plot 2: stacked bar — total across all runs ──
    totals = [plot_df[col].fillna(0).sum() for col in region_cols]
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(region_labels, totals, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(v)), ha='center', va='bottom', fontsize=FS_BARVALUE,
                fontweight='bold')
    ax.set_ylabel('Number of ions (all runs)', fontsize=FS_AXIS)
    t2 = 'Ion region occupancy at last observed frame (all runs)'
    ax.set_title(f'{channel}: {t2}' if channel else t2,
                 fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(bottom=0)
    plt.xticks(fontsize=FS_TICK, rotation=15, ha='right')
    plt.tight_layout()
    out = output_dir / 'last_frame_regions_total.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✅ Plot saved: {out}")

# ─────────────────────────────────────────────
# Summary & output
# ─────────────────────────────────────────────

def create_summary(df, output_dir, cat_metric='Full_Permeations_Total',
                   ns_per_frame=0.075, channel=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quartile categories (HIGH / TYPICAL / LOW) — added before any output
    # is written so they appear in the CSVs, the Excel file and the report.
    categories, q1, q3 = classify_runs_by_quartiles(df, cat_metric)
    df['Category'] = categories
    df['Q1'] = q1
    df['Q3'] = q3

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    n_sims   = len(df)
    valid    = df[df['Total_Frames'].notna()]
    n_frames = valid['Total_Frames'].mean() if len(valid) else 0

    total_full_unique  = df['Full_Permeations_Unique'].sum()
    total_full_total   = df['Full_Permeations_Total'].sum()
    hbc_u              = df['HBC_Permeations'].sum(skipna=True)
    mean_rate          = valid['Entries_Per_Frame'].mean() if len(valid) else 0
    has_tracking       = df['ion_region_tracking_found'].any()

    # Pooled (frame-weighted) rates — all runs treated as one trajectory
    pooled_frames = float(valid['Total_Frames'].sum()) if len(valid) else 0.0
    pooled_ns = pooled_frames * ns_per_frame
    pooled_per_frame = (total_full_total / pooled_frames) if pooled_frames else 0.0
    pooled_per_ns = (total_full_total / pooled_ns) if pooled_ns else 0.0

    summary_row = {
        'Number_of_Simulations':        n_sims,
        'Mean_Frames_per_Simulation':   round(n_frames, 0),
        'HBC_Permeations':              hbc_u,
        'Full_Permeations_Unique':      total_full_unique,
        'Full_Permeations_Total':       total_full_total,
        'Mean_Full_Permeations_Per_Frame': round(mean_rate, 6),
        'ns_per_frame':                 ns_per_frame,
        'Pooled_Total_Frames':          int(pooled_frames),
        'Pooled_Total_ns':              round(pooled_ns, 3),
        'Pooled_Full_Perm_Per_Frame':   round(pooled_per_frame, 6),
        'Pooled_Full_Perm_Per_ns':      round(pooled_per_ns, 4),
    }

    # Per-run distribution of each permeation metric: mean, median, std,
    # min and max ACROSS RUNS. These describe run-to-run variability and are
    # a different question from the pooled rate above -- the pooled number
    # weights every frame equally, these weight every run equally.
    for m in ['Full_Permeations_Total', 'Full_Permeations_Unique', 'HBC_Permeations']:
        if m not in df.columns:
            continue
        vals = df[m].dropna().astype(float)
        if vals.empty:
            continue
        summary_row[f'Mean_{m}_Per_Run']   = round(float(vals.mean()), 3)
        summary_row[f'Median_{m}_Per_Run'] = round(float(vals.median()), 3)
        summary_row[f'Std_{m}_Per_Run']    = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else 0.0
        summary_row[f'Min_{m}_Per_Run']    = float(vals.min())
        summary_row[f'Max_{m}_Per_Run']    = float(vals.max())

    if df['Category'].notna().any():
        summary_row[f'{cat_metric}_Q1'] = q1
        summary_row[f'{cat_metric}_Q3'] = q3
        for cat in CATEGORY_ORDER:
            summary_row[f'N_Runs_{cat}'] = int((df['Category'] == cat).sum())

    if has_tracking:
        for col in ['Total_Tracked_Ions', 'HBC_Permeations', 'In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']:
            summary_row[f'Total_{col}'] = df[col].sum(skipna=True)

    print(f"Simulations:                   {n_sims}")
    print(f"Mean frames/sim:               {n_frames:.0f}")
    print(f"HBC permeations (Full_Tot + AfterHBC): {hbc_u}")
    print(f"Full permeations:              {total_full_total} total  ({total_full_unique} unique ions)")
    print(f"Mean full permeations/frame:   {mean_rate:.6f}")
    print(f"Pooled full permeations/frame: {pooled_per_frame:.6f}")
    print(f"Pooled full permeations/ns:    {pooled_per_ns:.4f}")
    print("\nPer-run distribution (every run weighted equally):")
    for m in ['Full_Permeations_Total', 'Full_Permeations_Unique', 'HBC_Permeations']:
        if f'Mean_{m}_Per_Run' in summary_row:
            print(f"  {m:<26} mean={summary_row[f'Mean_{m}_Per_Run']:>8.2f}  "
                  f"median={summary_row[f'Median_{m}_Per_Run']:>8.2f}  "
                  f"std={summary_row[f'Std_{m}_Per_Run']:>8.2f}  "
                  f"range={summary_row[f'Min_{m}_Per_Run']:.0f}-{summary_row[f'Max_{m}_Per_Run']:.0f}")
    if has_tracking:
        print(f"\nLast-frame snapshot totals across all runs:")
        print(f"  In SF region:      {summary_row['Total_In_SF']}")
        print(f"  In GLU/ASN cavity: {summary_row['Total_In_GLU_ASN']}")
        print(f"  In ASN->HBC:   {summary_row['Total_In_ASN_HBC']}")
        print(f"  After HBC:         {summary_row['Total_After_HBC']}")

    # ── CSV ──
    pd.DataFrame([summary_row]).to_csv(output_dir / 'permeation_summary.csv', index=False)
    df.to_csv(output_dir / 'permeation_per_run.csv', index=False)
    print(f"\n✅ permeation_summary.csv saved")
    print(f"✅ permeation_per_run.csv saved")

    # ── Excel ──
    excel_file = output_dir / 'permeation_analysis.xlsx'
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = Workbook()

        header_font  = Font(name='Arial', bold=True, color='FFFFFF')
        header_fill  = PatternFill('solid', start_color='2E4057')
        center_align = Alignment(horizontal='center')
        alt_fill     = PatternFill('solid', start_color='EAF2FB')

        # Sheet 1: Summary
        ws = wb.active
        ws.title = "Summary"
        headers = list(summary_row.keys())
        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h.replace('_', ' '))
            cell.font = header_font; cell.fill = header_fill; cell.alignment = center_align
        for ci, h in enumerate(headers, 1):
            ws.cell(row=2, column=ci, value=summary_row[h])
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or '')) for c in col) + 4, 40)

        # Sheet 2: Per-run details
        ws2 = wb.create_sheet("Per_Run_Details")
        detail_cols = ['Run', 'Category', 'Total_Frames',
                       'HBC_Permeations',
                       'Full_Permeations_Unique', 'Full_Permeations_Total', 'Entries_Per_Frame',
                       'Total_Tracked_Ions', 'In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']
        detail_cols = [c for c in detail_cols if c in df.columns]
        for ci, h in enumerate(detail_cols, 1):
            cell = ws2.cell(row=1, column=ci, value=h.replace('_', ' '))
            cell.font = header_font; cell.fill = header_fill; cell.alignment = center_align
        for ri, (_, row) in enumerate(df[detail_cols].iterrows(), 2):
            for ci, val in enumerate(row, 1):
                cell = ws2.cell(row=ri, column=ci, value=val)
                cell.font = Font(name='Arial')
                if ri % 2 == 0:
                    cell.fill = alt_fill
        for col in ws2.columns:
            ws2.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or '')) for c in col) + 4, 30)

        wb.save(excel_file)
        print(f"✅ permeation_analysis.xlsx saved")
    except ImportError:
        print("⚠  openpyxl not installed — Excel file skipped")
    except Exception as e:
        print(f"⚠  Could not create Excel file: {e}")

    # ── Text report ──
    report_file = output_dir / 'permeation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERMEATION EVENT ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("OVERALL STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of simulations:                       {n_sims}\n")
        f.write(f"Mean frames per simulation:                  {n_frames:.0f}\n")
        f.write(f"HBC permeations (Full_Tot + AfterHBC): {hbc_u}\n")
        f.write(f"Full permeations (passed end4):  {total_full_total} total  ({total_full_unique} unique ions)\n")
        f.write(f"Mean full permeations per frame: {mean_rate:.6f}\n")
        f.write(f"\nPOOLED RATES (all runs as one trajectory, {ns_per_frame} ns/frame):\n")
        f.write(f"  Total frames:                  {int(pooled_frames):,}\n")
        f.write(f"  Total time:                    {pooled_ns:,.1f} ns\n")
        f.write(f"  Full permeations per frame:    {pooled_per_frame:.6f}\n")
        f.write(f"  Full permeations per ns:       {pooled_per_ns:.4f}\n")
        f.write(f"\nPER-RUN DISTRIBUTION (every run weighted equally):\n")
        for m in ['Full_Permeations_Total', 'Full_Permeations_Unique', 'HBC_Permeations']:
            if f'Mean_{m}_Per_Run' in summary_row:
                f.write(f"  {m:<26} mean={summary_row[f'Mean_{m}_Per_Run']:>8.2f}  "
                        f"median={summary_row[f'Median_{m}_Per_Run']:>8.2f}  "
                        f"std={summary_row[f'Std_{m}_Per_Run']:>8.2f}  "
                        f"range={summary_row[f'Min_{m}_Per_Run']:.0f}-"
                        f"{summary_row[f'Max_{m}_Per_Run']:.0f}\n")
        if has_tracking:
            f.write(f"\nLast-frame snapshot totals (ion_region_tracking.json):\n")
            f.write(f"  In SF region (R1):       {summary_row['Total_In_SF']}\n")
            f.write(f"  In GLU/ASN cavity (R2):  {summary_row['Total_In_GLU_ASN']}\n")
            f.write(f"  In ASN->HBC (R3):    {summary_row['Total_In_ASN_HBC']}\n")
            f.write(f"  After HBC (R4):          {summary_row['Total_After_HBC']}\n")
        if df['Category'].notna().any():
            f.write(f"\nRUN CATEGORIES (quartiles of {cat_metric}):\n")
            f.write(f"  Q1 = {q1:.3f}, Q3 = {q3:.3f}\n")
            f.write("  HIGH = value > Q3,  TYPICAL = Q1 <= value <= Q3,  LOW = value < Q1\n")
            for cat in CATEGORY_ORDER:
                names = df.loc[df['Category'] == cat, 'Run'].tolist()
                f.write(f"  {cat:<8} n={len(names):<3} {' '.join(names)}\n")
        f.write("\n" + "="*70 + "\n\n")
        def fmt(v): return f"{int(v)}" if pd.notna(v) else "N/A"

        f.write("PER-RUN DETAILS:\n")
        sep = "-" * 115
        f.write(sep + "\n")
        f.write(
            f"{'Run':<8} {'Category':<9} {'Frames':<8} "
            f"{'HBC_Perm':<10} "
            f"{'Full_Tot':<10} {'Full_Uniq':<11} "
            f"{'Tracked':<9} {'In_SF':<7} {'GLU_ASN':<9} {'ASN_HBC':<9} {'AfterHBC':<9}\n"
        )
        f.write(sep + "\n")
        for _, row in df.iterrows():
            fr = f"{int(row['Total_Frames'])}" if pd.notna(row['Total_Frames']) else "?"
            f.write(
                f"{row['Run']:<8} {str(row.get('Category') or '-'):<9} {fr:<8} "
                f"{fmt(row.get('HBC_Permeations')):<10} "
                f"{fmt(row.get('Full_Permeations_Total')):<10} "
                f"{fmt(row.get('Full_Permeations_Unique')):<11} "
                f"{fmt(row.get('Total_Tracked_Ions')):<9} "
                f"{fmt(row.get('In_SF')):<7} "
                f"{fmt(row.get('In_GLU_ASN')):<9} "
                f"{fmt(row.get('In_ASN_HBC')):<9} "
                f"{fmt(row.get('After_HBC')):<9}\n"
            )
        f.write("\n" + "="*70 + "\n")
        f.write("\nNOTES:\n")
        f.write("- Complete_Perm_Entries: rows in permeation_table.json (one per permeation\n")
        f.write("  event; same ion appears multiple times if it re-permeated).\n")
        f.write("- Complete_Perm_Unique_Ions: distinct ion IDs in permeation_table.json.\n")
        f.write("- Last-frame snapshot columns come from ion_region_tracking.json and show\n")
        f.write("  where each ion was sitting at the last frame it was observed in the channel.\n")
        f.write("  In_SF = SF region (R1), In_GLU_ASN = GLU/ASN cavity (R2),\n")
        f.write("  In_ASN_HBC = ASN->HBC (R3), After_HBC = completed permeation (R4).\n")
        f.write("- Pooled rates sum counts and frames across runs before dividing, so long\n")
        f.write("  runs carry proportionally more weight. This differs from the mean of\n")
        f.write("  per-run rates whenever the runs are not all the same length.\n")

    print(f"✅ permeation_report.txt saved")

    # Plot last-frame region occupancy
    plot_last_frame_regions(df, output_dir, channel=channel)

    # Boxplots of permeation counts across runs (each dot labelled with run name,
    # coloured by quartile category for the metric used to classify)
    plot_permeation_boxplot(df, output_dir, metric=cat_metric,
                            cat_col='Category', q1=q1, q3=q3, channel=channel)
    other = 'HBC_Permeations' if cat_metric != 'HBC_Permeations' else 'Full_Permeations_Total'
    plot_permeation_boxplot(df, output_dir, metric=other, channel=channel)

    # Category CSV + run lists
    write_category_outputs(df, output_dir, metric=cat_metric)

    # Pooled per-frame / per-ns rate table
    write_pooled_rate_csv(df, output_dir, ns_per_frame)

    return summary_row


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze permeation events across multiple simulation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/girk_analyser_results/G12_GAT
  %(prog)s /path/to/girk_analyser_results/G12_GAT --runs 1 2 3 4 5
  %(prog)s /path/to/girk_analyser_results/G12_GAT -o my_summary
  %(prog)s /path/to/girk_analyser_results/G2 --frame_duration 0.4
        """
    )
    parser.add_argument('base_dir', help='Base directory containing RUN1, RUN2, …')
    parser.add_argument('--runs', nargs='+', type=int,
                        help='Specific run numbers (e.g. --runs 1 2 3)')
    parser.add_argument('-o', '--output', default='permeation_summary',
                        help='Output subdirectory name inside base_dir (default: permeation_summary)')
    parser.add_argument('-cat_metric', default='Full_Permeations_Total',
                        help='Metric whose quartiles define the HIGH/TYPICAL/LOW run '
                             'categories (default: Full_Permeations_Total)')
    parser.add_argument('--frame_duration', '-ns_per_frame', type=float,
                        dest='frame_duration', default=0.075,
                        help='Simulation time per saved frame, in ns. This DIFFERS '
                             'between systems: 0.075 for G12, G12_ML, G12_GAT '
                             '(ntwx 25000 x dt 0.003 ps) but 0.4 for G2. Pass it '
                             'explicitly for every run -- the default of 0.075 is '
                             'silently wrong for G2 and would understate its per-ns '
                             'rate by more than 5x.')
    parser.add_argument('-channel', default=None,
                        help='Channel/system name shown in plot titles (e.g. G12, '
                             'G12_GAT). Defaults to the name of base_dir.')
    parser.add_argument('-font_scale', type=float, default=1.0,
                        help='Multiplier for all plot font sizes (default 1.0). Use '
                             '1.2-1.5 for figures that will be shrunk onto slides.')

    args       = parser.parse_args()
    base_dir   = Path(args.base_dir)
    output_dir = base_dir / args.output

    # Apply font scaling before any plotting happens
    global FONT_SCALE, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_ANNOT, FS_BARVALUE
    FONT_SCALE  = args.font_scale
    FS_TITLE    = int(26 * FONT_SCALE)
    FS_AXIS     = int(24 * FONT_SCALE)
    FS_TICK     = int(21 * FONT_SCALE)
    FS_LEGEND   = int(18 * FONT_SCALE)
    FS_ANNOT    = int(15 * FONT_SCALE)
    FS_BARVALUE = int(17 * FONT_SCALE)

    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return 1

    print("\n" + "="*70)
    print("PERMEATION EVENT SUMMARY ANALYSIS")
    print("="*70)
    print(f"Base directory:  {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame duration:   {args.frame_duration} ns/frame")
    print(f"Channel label:    {args.channel or base_dir.name}")

    run_dirs = find_run_dirs(base_dir, args.runs)
    if not run_dirs:
        print("ERROR: No RUN directories found.")
        return 1

    rows = [analyze_run(d) for d in run_dirs]
    df   = pd.DataFrame(rows)

    channel = args.channel or base_dir.name
    create_summary(df, output_dir, cat_metric=args.cat_metric,
                   ns_per_frame=args.frame_duration, channel=channel)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nAll results in: {output_dir}/")
    print("  • permeation_analysis.xlsx")
    print("  • permeation_summary.csv")
    print("  • permeation_per_run.csv")
    print("  • permeation_rates.csv")
    print("  • permeation_report.txt")
    print(f"  • run_categories_{args.cat_metric.lower()}.csv")
    print(f"  • run_categories_{args.cat_metric.lower()}_lists.txt\n")
    return 0


if __name__ == "__main__":
    exit(main())

'''
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G12/
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G12_GAT/
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G2/
'''

'''
RESULTS=/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results

# NOTE the frame_duration differs between systems -- G2 is 0.4 ns/frame,
# everything else is 0.075. Getting this wrong only corrupts the per-ns
# columns; the raw counts are unaffected.
python3 analyze_permeation_summary.py $RESULTS/G12/        --frame_duration 0.075
python3 analyze_permeation_summary.py $RESULTS/G12_ML/     --frame_duration 0.075
python3 analyze_permeation_summary.py $RESULTS/G12_GAT/    --frame_duration 0.075
python3 analyze_permeation_summary.py $RESULTS/G2/         --frame_duration 0.4
python3 analyze_permeation_summary.py $RESULTS/G12_D173N/  --frame_duration 0.075
python3 analyze_permeation_summary.py $RESULTS/G12_E141Q/  --frame_duration 0.075

# Even bigger lettering for slides:
python3 analyze_permeation_summary.py $RESULTS/G12/ -font_scale 1.3
'''

'''
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/permeation_summary  --frame_duration 0.075
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/permeation_summary --frame_duration 0.075
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/permeation_summary --frame_duration 0.075
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/permeation_summary --frame_duration 0.4
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/permeation_summary --frame_duration 0.075
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/permeation_summary --frame_duration 0.075
'''