"""
radial_force_correlation.py
============================
For each permeation event, computes the radial distance of the permeating ion
from the pore axis (using SF ions as per-event pore center) and correlates it
with the Z-component of the force on the ion.

The pore center XY is estimated per-event as the mean XY of the SF ions
already stored in force_correlation_data.json.

Usage:
    python3 radial_force_correlation.py <base_dir> <channel_type> [--output_dir <dir>]

Examples:
    python3 radial_force_correlation.py /path/to/G12     G12
    python3 radial_force_correlation.py /path/to/G12_GAT G12_GAT
    python3 radial_force_correlation.py /path/to/G12_ML  G12_ML
    python3 radial_force_correlation.py /path/to/G2      G2

Output (saved to <base_dir>/radial_force_correlation/ or --output_dir):
    - radial_force_scatter.png       : scatter plot radial distance vs force_z
    - radial_force_binned.png        : binned mean ± std (running mean)
    - radial_distribution.png        : histogram of radial distances
    - radial_force_data.csv          : per-event data table
    - correlation_summary.txt        : Pearson r, Spearman rho, statistics
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_runs(base_dir):
    """
    Load force_correlation_data.json from all RUN subdirectories.
    Returns a flat list of all events with run_name added.
    """
    base_dir = Path(base_dir)
    run_folders = sorted([d for d in base_dir.iterdir()
                          if d.is_dir() and d.name.startswith('RUN')])

    if not run_folders:
        raise FileNotFoundError(f"No RUN folders found in {base_dir}")

    all_events = []
    run_stats = []

    print(f"\nFound {len(run_folders)} RUN folders")

    for run_folder in run_folders:
        json_file = run_folder / "force_correlation_analysis" / "force_correlation_data.json"

        if not json_file.exists():
            print(f"  WARNING: {run_folder.name} — force_correlation_data.json not found, skipping")
            continue

        with open(json_file, 'r') as f:
            run_data = json.load(f)

        for event in run_data:
            event['run_name'] = run_folder.name

        all_events.extend(run_data)
        run_stats.append({'run_name': run_folder.name, 'n_events': len(run_data)})
        print(f"  ✓ {run_folder.name}: {len(run_data)} events")

    print(f"\nTotal events loaded: {len(all_events)}")
    return all_events, run_stats


# =============================================================================
# RADIAL DISTANCE CALCULATION
# =============================================================================

def compute_radial_distances(all_events):
    """
    For each event, compute the radial distance of the permeating ion
    from the pore axis, using the mean XY of SF ions as the pore center.

    Returns a DataFrame with columns:
        run_name, frame, ion_id, radial_distance, force_z,
        force_magnitude, ion_z, pore_center_x, pore_center_y,
        n_sf_ions
    """
    records = []

    for ev in all_events:
        sf_positions = ev.get('sf_ion_positions', [])

        if not sf_positions:
            # No SF ions recorded — skip this event
            continue

        # Per-event pore center = mean XY of SF ions
        sf_xy = np.mean([[p[0], p[1]] for p in sf_positions], axis=0)
        pore_cx, pore_cy = sf_xy[0], sf_xy[1]

        # Radial distance of permeating ion from pore center
        dx = ev['ion_position_x'] - pore_cx
        dy = ev['ion_position_y'] - pore_cy
        radial_dist = np.sqrt(dx**2 + dy**2)

        records.append({
            'run_name':       ev['run_name'],
            'frame':          ev['frame'],
            'ion_id':         ev['event_ion_id'],
            'radial_distance': radial_dist,
            'force_z':        ev['force_z_pN'],
            'force_magnitude': ev['force_magnitude_pN'],
            'ion_x':          ev['ion_position_x'],
            'ion_y':          ev['ion_position_y'],
            'ion_z':          ev['ion_position_z'],
            'pore_center_x':  pore_cx,
            'pore_center_y':  pore_cy,
            'n_sf_ions':      len(sf_positions),
        })

    df = pd.DataFrame(records)
    print(f"\nEvents with SF ions (usable): {len(df)}")
    print(f"Events skipped (no SF ions):  {len(all_events) - len(df)}")

    return df


# =============================================================================
# STATISTICS
# =============================================================================

def compute_correlations(df):
    """Compute Pearson and Spearman correlations between radial distance and force_z."""
    r_pearson, p_pearson   = stats.pearsonr(df['radial_distance'], df['force_z'])
    r_spearman, p_spearman = stats.spearmanr(df['radial_distance'], df['force_z'])

    return {
        'n':          len(df),
        'pearson_r':  r_pearson,
        'pearson_p':  p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'radial_mean':  df['radial_distance'].mean(),
        'radial_std':   df['radial_distance'].std(),
        'radial_median': df['radial_distance'].median(),
        'force_mean':   df['force_z'].mean(),
        'force_std':    df['force_z'].std(),
        'force_median': df['force_z'].median(),
        'pct_repulsive': (df['force_z'] > 0).mean() * 100,  # % pushing upward (+Z)
        'pct_attractive': (df['force_z'] < 0).mean() * 100, # % pulling downward (-Z)
    }


def save_summary(corr, channel_type, run_stats, output_dir):
    """Save plain text summary file."""
    lines = [
        "=" * 65,
        f"RADIAL DISTANCE vs Z-FORCE CORRELATION SUMMARY",
        f"Channel: {channel_type}",
        "=" * 65,
        "",
        f"Total events analysed : {corr['n']}",
        "",
        "Per-run breakdown:",
    ]
    for rs in run_stats:
        lines.append(f"  {rs['run_name']}: {rs['n_events']} events")

    lines += [
        "",
        "-" * 65,
        "RADIAL DISTANCE (Å):",
        f"  Mean   : {corr['radial_mean']:.3f}",
        f"  Std    : {corr['radial_std']:.3f}",
        f"  Median : {corr['radial_median']:.3f}",
        "",
        "Z-FORCE (pN):",
        f"  Mean   : {corr['force_mean']:.3f}",
        f"  Std    : {corr['force_std']:.3f}",
        f"  Median : {corr['force_median']:.3f}",
        f"  % events with force_z > 0 (repulsive/upward) : {corr['pct_repulsive']:.1f}%",
        f"  % events with force_z < 0 (attractive/downward): {corr['pct_attractive']:.1f}%",
        "",
        "-" * 65,
        "CORRELATIONS (radial_distance vs force_z):",
        f"  Pearson  r = {corr['pearson_r']:+.4f}  (p = {corr['pearson_p']:.3e})",
        f"  Spearman ρ = {corr['spearman_r']:+.4f}  (p = {corr['spearman_p']:.3e})",
        "",
        "Interpretation:",
        "  Positive correlation → ions further from pore axis experience larger upward force",
        "  Negative correlation → ions closer to axis experience larger upward force",
        "=" * 65,
    ]

    path = output_dir / "correlation_summary.txt"
    path.write_text("\n".join(lines))
    print(f"✓ Correlation summary saved: {path}")
    # Also print to terminal
    print("\n" + "\n".join(lines))


# =============================================================================
# PLOTS
# =============================================================================

def plot_scatter(df, corr, channel_type, output_dir):
    """Scatter plot: radial distance vs force_z, coloured by run."""
    fig, ax = plt.subplots(figsize=(10, 7))

    runs = df['run_name'].unique()
    cmap = plt.cm.get_cmap('tab10', len(runs))

    for i, run in enumerate(sorted(runs)):
        sub = df[df['run_name'] == run]
        ax.scatter(sub['radial_distance'], sub['force_z'],
                   s=80, alpha=0.6, color=cmap(i),
                   edgecolors='black', linewidth=0.8,
                   label=run, zorder=3)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Linear regression line
    slope, intercept, *_ = stats.linregress(df['radial_distance'], df['force_z'])
    x_line = np.linspace(df['radial_distance'].min(), df['radial_distance'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color='red', linewidth=2.5, linestyle='-', label='Linear fit', zorder=4)

    stats_text = (f"Pearson r = {corr['pearson_r']:+.3f}  (p={corr['pearson_p']:.2e})\n"
                  f"Spearman ρ = {corr['spearman_r']:+.3f}  (p={corr['spearman_p']:.2e})\n"
                  f"n = {corr['n']}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='black', linewidth=1.5))

    ax.set_xlabel('Radial Distance from Pore Axis (Å)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Force Z (pN)', fontsize=16, fontweight='bold')
    ax.set_title(f'{channel_type} — Permeating Ion: Radial Distance vs Z-Force',
                 fontsize=17, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    path = output_dir / "radial_force_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Scatter plot saved: {path}")


def plot_binned(df, channel_type, output_dir, n_bins=8):
    """
    Binned mean ± std plot: divide radial distance into bins,
    show mean force_z ± std per bin.
    """
    df_sorted = df.sort_values('radial_distance')

    # Equal-frequency bins (quantile-based) so each bin has similar n
    df['bin'] = pd.qcut(df['radial_distance'], q=n_bins, duplicates='drop')
    bin_stats = df.groupby('bin', observed=True)['force_z'].agg(
        mean='mean', std='std', n='count'
    ).reset_index()
    bin_centers = [interval.mid for interval in bin_stats['bin']]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(bin_centers)), bin_stats['mean'],
           color='steelblue', alpha=0.6, edgecolor='black', linewidth=1.5, width=0.7)
    ax.errorbar(range(len(bin_centers)), bin_stats['mean'],
                yerr=bin_stats['std'], fmt='none',
                color='black', capsize=10, capthick=2, linewidth=2)

    for i, (mean, n) in enumerate(zip(bin_stats['mean'], bin_stats['n'])):
        y_pos = mean + bin_stats['std'].iloc[i] + 5
        ax.text(i, y_pos, f'n={n}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=30, ha='right', fontsize=12)
    ax.set_xlabel('Radial Distance Bin Center (Å)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Force Z (pN)', fontsize=16, fontweight='bold')
    ax.set_title(f'{channel_type} — Mean Z-Force per Radial Distance Bin',
                 fontsize=17, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    path = output_dir / "radial_force_binned.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Binned plot saved: {path}")


def plot_radial_distribution(df, channel_type, output_dir):
    """Histogram of radial distances."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(df['radial_distance'], bins=25, color='coral', alpha=0.7,
            edgecolor='black', linewidth=1.2)

    mean_r = df['radial_distance'].mean()
    median_r = df['radial_distance'].median()
    ax.axvline(mean_r,   color='red',    linestyle='--', linewidth=2,
               label=f'Mean = {mean_r:.2f} Å')
    ax.axvline(median_r, color='purple', linestyle=':',  linewidth=2,
               label=f'Median = {median_r:.2f} Å')

    ax.set_xlabel('Radial Distance from Pore Axis (Å)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title(f'{channel_type} — Radial Distance Distribution (Permeating Ions)',
                 fontsize=17, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)

    plt.tight_layout()
    path = output_dir / "radial_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Radial distribution plot saved: {path}")


def plot_force_by_radial_quartile(df, channel_type, output_dir):
    """
    Split events into radial quartiles and show force_z distribution per quartile.
    Gives a clean categorical view of the correlation.
    """
    df = df.copy()
    df['quartile'] = pd.qcut(df['radial_distance'], q=4,
                              labels=['Q1\n(closest)', 'Q2', 'Q3', 'Q4\n(furthest)'])

    quartile_data = [df[df['quartile'] == q]['force_z'].values
                     for q in ['Q1\n(closest)', 'Q2', 'Q3', 'Q4\n(furthest)']]
    labels = ['Q1\n(closest)', 'Q2', 'Q3', 'Q4\n(furthest)']
    colors = ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7']

    fig, ax = plt.subplots(figsize=(10, 7))

    means = [np.mean(d) for d in quartile_data]
    stds  = [np.std(d)  for d in quartile_data]
    ns    = [len(d)     for d in quartile_data]

    bars = ax.bar(range(4), means, color=colors, alpha=0.65,
                  edgecolor='black', linewidth=2, width=0.6)
    ax.errorbar(range(4), means, yerr=stds, fmt='none',
                color='black', capsize=15, capthick=2, linewidth=2)

    # Jitter points
    for i, data in enumerate(quartile_data):
        jx = np.random.normal(i, 0.05, size=len(data))
        ax.scatter(jx, data, color='black', alpha=0.35, s=25, zorder=3)

    for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
        y_pos = mean + std + abs(ax.get_ylim()[1]) * 0.03
        ax.text(i, y_pos, f'{mean:.1f} ± {std:.1f}\n(n={n})',
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='black', alpha=0.9, linewidth=1.5))

    # Quartile radial ranges as subtitle
    quartile_ranges = df.groupby('quartile', observed=True)['radial_distance'].agg(['min', 'max'])
    range_labels = [f'{row["min"]:.2f}–{row["max"]:.2f} Å'
                    for _, row in quartile_ranges.iterrows()]
    for i, rl in enumerate(range_labels):
        ax.text(i, ax.get_ylim()[0] + 5, rl,
                ha='center', va='bottom', fontsize=10, color='dimgray', style='italic')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')
    ax.set_ylabel('Force Z (pN)', fontsize=17, fontweight='bold')
    ax.set_xlabel('Radial Distance Quartile', fontsize=17, fontweight='bold')
    ax.set_title(f'{channel_type} — Z-Force by Radial Distance Quartile',
                 fontsize=18, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    path = output_dir / "radial_quartile_force.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Quartile force plot saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlate permeating ion radial distance with Z-force"
    )
    parser.add_argument('base_dir',    type=str, help='Base directory with RUN folders')
    parser.add_argument('channel_type', type=str,
                        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'],
                        help='Channel type label')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <base_dir>/radial_force_correlation)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir \
                 else base_dir / "radial_force_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("RADIAL DISTANCE vs Z-FORCE CORRELATION")
    print("=" * 65)
    print(f"Channel    : {args.channel_type}")
    print(f"Base dir   : {base_dir}")
    print(f"Output dir : {output_dir}")

    # Load data
    all_events, run_stats = load_all_runs(base_dir)

    # Compute radial distances
    df = compute_radial_distances(all_events)

    if df.empty:
        print("ERROR: No usable events found. Exiting.")
        return

    # Save CSV
    csv_path = output_dir / "radial_force_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Per-event data saved: {csv_path}")

    # Correlations
    corr = compute_correlations(df)

    # Save summary
    save_summary(corr, args.channel_type, run_stats, output_dir)

    # Plots
    print("\nCreating plots...")
    plot_scatter(df, corr, args.channel_type, output_dir)
    plot_binned(df, args.channel_type, output_dir)
    plot_radial_distribution(df, args.channel_type, output_dir)
    plot_force_by_radial_quartile(df, args.channel_type, output_dir)

    print(f"\n✓ All outputs saved to: {output_dir}")
    print("  - radial_force_scatter.png")
    print("  - radial_force_binned.png")
    print("  - radial_distribution.png")
    print("  - radial_quartile_force.png")
    print("  - radial_force_data.csv")
    print("  - correlation_summary.txt")


if __name__ == "__main__":
    main()


# USAGE EXAMPLES:
# python3 radial_force_correlation.py /path/to/G12     G12
# python3 radial_force_correlation.py /path/to/G12_GAT G12_GAT
# python3 radial_force_correlation.py /path/to/G12_ML  G12_ML
# python3 radial_force_correlation.py /path/to/G2      G2
#
# With custom output:
"""
python3 radial_force_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
G2 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/radial_force_correlation

python3 radial_force_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
G12 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/radial_force_correlation

python3 radial_force_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
G12_ML --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/radial_force_correlation

python3 radial_force_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
G12_GAT --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/radial_force_correlation
"""