#!/usr/bin/env python3
"""
Combine Dihedral Analysis from Multiple Trajectories

CHANGES vs previous version:
    load_and_combine_dihedral_data() now stores chi angles PER RUN as a dict:

        combined_data['ASN_130']['chi1'] = {
            'RUN1': np.array([...]),
            'RUN2': np.array([...]),
            ...
        }

    instead of a single concatenated array. This makes per-run GMM label
    tracking and combination occupancy analysis straightforward and robust.

    All downstream functions (histograms, bucket analysis, plots) call
    get_all_frames(angles['chi1']) which concatenates on-the-fly — so
    everything else works exactly as before.

Usage:
    python combine_dihedral_runs.py /path/to/results/G12_GAT
    python combine_dihedral_runs.py /path/to/results/G12_GAT --runs 1 2 3 4 5
"""

import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# HELPER — get all frames from per-run dict OR plain array (backwards compat)
# =============================================================================

def get_all_frames(chi_data):
    """
    Return a single concatenated numpy array of all frames in sorted run order.
    Works whether chi_data is:
      - dict  {'RUN1': array, 'RUN2': array, ...}  (new format)
      - np.ndarray  (old format — backwards compatible)
    IMPORTANT: uses sorted key order to match get_run_names().
    """
    if isinstance(chi_data, dict):
        arrays = [chi_data[k] for k in sorted(chi_data.keys())
                  if chi_data[k] is not None and len(chi_data[k]) > 0]
        return np.concatenate(arrays) if arrays else np.array([])
    return np.asarray(chi_data)


def get_run_names(chi_data):
    """Return sorted run names if per-run dict, else ['ALL']."""
    if isinstance(chi_data, dict):
        return sorted(chi_data.keys())
    return ['ALL']


# =============================================================================
# LOAD AND COMBINE
# =============================================================================

def find_dihedral_files(base_dir, run_numbers=None):
    base_dir = Path(base_dir)
    dihedral_files = []

    if run_numbers is None:
        run_dirs = sorted(base_dir.glob("RUN*"))
        print("Found {} RUN directories".format(len(run_dirs)))
    else:
        run_dirs = [base_dir / "RUN{}".format(num) for num in run_numbers]
        print("Looking for {} specified RUN directories".format(len(run_dirs)))

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"  [SKIP] {run_dir.name} - directory not found")
            continue
        pkl_file = run_dir / "dihedral_analysis" / "dihedral_raw_data.pkl"
        if pkl_file.exists():
            dihedral_files.append(pkl_file)
            print(f"  [FOUND] {run_dir.name}/dihedral_raw_data.pkl")
        else:
            print(f"  [SKIP] {run_dir.name} - no dihedral_raw_data.pkl")

    return dihedral_files


def load_and_combine_dihedral_data(pkl_files):
    """
    Load dihedral data from multiple pickle files.

    CHANGED: chi angles are now stored as per-run dicts:
        combined_data[res_name]['chi1'] = {'RUN1': array, 'RUN2': array, ...}
        combined_data[res_name]['chi2'] = {'RUN1': array, 'RUN2': array, ...}

    This preserves run identity while still allowing easy concatenation via
    get_all_frames().
    """
    print("\n" + "="*70)
    print("LOADING AND COMBINING DIHEDRAL DATA")
    print("="*70)

    combined_data = {}
    run_info      = {}

    for pkl_file in pkl_files:
        run_name = pkl_file.parent.parent.name  # RUN1, RUN2 etc (not 'dihedral_analysis')
        print(f"\nProcessing {run_name}...")

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            run_frames = {}

            for res_name, angles in data.items():
                if res_name not in combined_data:
                    combined_data[res_name] = {
                        'chi1':         {},   # ← per-run dict
                        'chi2':         {},
                        'pdb_label':    angles.get('pdb_label', res_name),
                        'residue_type': angles.get('residue_type', 'UNK')
                    }
                    if 'chi3' in angles and angles['chi3'] is not None:
                        combined_data[res_name]['chi3'] = {}

                # store this run's arrays under run_name key
                if angles['chi1'] is not None:
                    combined_data[res_name]['chi1'][run_name] = np.asarray(angles['chi1'])
                    run_frames[res_name] = len(angles['chi1'])

                if angles['chi2'] is not None:
                    combined_data[res_name]['chi2'][run_name] = np.asarray(angles['chi2'])

                if ('chi3' in combined_data[res_name]
                        and 'chi3' in angles
                        and angles['chi3'] is not None):
                    combined_data[res_name]['chi3'][run_name] = np.asarray(angles['chi3'])

            run_info[run_name] = run_frames
            n = list(run_frames.values())[0] if run_frames else 0
            print(f"  ✓ Loaded {n} frames")

        except Exception as e:
            print(f"  ✗ Error loading {pkl_file}: {e}")
            continue

    # Summary
    print("\nCombining summary (per-run dicts — no concatenation):")
    for res_name, angles in combined_data.items():
        pdb_label  = angles['pdb_label']
        n_runs     = len(angles['chi1'])
        n_total    = sum(len(v) for v in angles['chi1'].values())
        print(f"  {res_name} (PDB: {pdb_label}): {n_runs} runs, {n_total} total frames")

    print("\n" + "="*70)
    total_frames = sum(sum(f.values()) for f in run_info.values())
    print(f"Total runs: {len(pkl_files)}")
    print(f"Total frames: {total_frames}")
    print(f"Residues: {len(combined_data)}")

    return combined_data, run_info


# =============================================================================
# PEAK DETECTION (unchanged)
# =============================================================================

def detect_and_merge_peaks(hist, min_prom=0.05, min_height=0.02, distance=8,
                           merge_bins=8, smooth_sigma=1, wrap_bins=8):
    num_bins = len(hist)
    smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
    padded   = np.concatenate([smoothed[-wrap_bins:], smoothed, smoothed[:wrap_bins]])

    raw_peaks, _ = find_peaks(
        padded,
        prominence=np.max(smoothed) * min_prom,
        height=np.max(smoothed) * min_height,
        distance=distance
    )

    shifted_peaks = raw_peaks - wrap_bins
    shifted_peaks = shifted_peaks[(shifted_peaks >= 0) & (shifted_peaks < num_bins)]

    if len(shifted_peaks) > 1:
        shifted_peaks = np.sort(shifted_peaks)
        merged = []
        i = 0
        while i < len(shifted_peaks):
            group = [shifted_peaks[i]]
            j = i + 1
            while j < len(shifted_peaks) and (shifted_peaks[j] - shifted_peaks[i]) <= merge_bins:
                group.append(shifted_peaks[j])
                j += 1
            merged.append(int(np.median(group)))
            i = j

        first = merged[0]
        last  = merged[-1]
        if (first + num_bins - last) <= merge_bins:
            merged    = merged[1:]
            merged[-1] = int(np.median([first, last]))

        shifted_peaks = np.array(merged)

    return np.array(shifted_peaks)


# =============================================================================
# COMBINED PEAK ANALYSIS
# =============================================================================

def analyze_combined_peaks(combined_data, output_dir, peak_tolerance=15):
    print("\n" + "="*70)
    print("PEAK ANALYSIS ON COMBINED DATA")
    print("="*70)

    peak_analysis = {}

    with open(output_dir / 'combined_peak_analysis.txt', 'w') as f:
        f.write("COMBINED PEAK ANALYSIS - ALL TRAJECTORIES\n")
        f.write("="*70 + "\n\n")

        for res_name, angles in combined_data.items():
            pdb_label = angles.get('pdb_label', res_name)
            res_type  = angles.get('residue_type', 'UNK')

            print(f"\n{res_name} (PDB: {pdb_label}):")
            f.write(f"{res_name} (PDB: {pdb_label}) - {res_type}\n")
            f.write("-"*60 + "\n")

            # use get_all_frames for histogram computation
            chi1 = get_all_frames(angles['chi1'])
            chi2 = get_all_frames(angles['chi2'])

            chi1_hist, chi1_bins = np.histogram(chi1, bins=72, range=(-180, 180))
            chi2_hist, chi2_bins = np.histogram(chi2, bins=72, range=(-180, 180))
            chi1_centers = (chi1_bins[:-1] + chi1_bins[1:]) / 2
            chi2_centers = (chi2_bins[:-1] + chi2_bins[1:]) / 2

            chi1_peaks = detect_and_merge_peaks(chi1_hist)
            chi2_peaks = detect_and_merge_peaks(chi2_hist, merge_bins=5)

            angles.update({
                'chi1_hist':         chi1_hist,
                'chi2_hist':         chi2_hist,
                'chi1_centers':      chi1_centers,
                'chi2_centers':      chi2_centers,
                'chi1_peak_indices': chi1_peaks,
                'chi2_peak_indices': chi2_peaks
            })

            chi1_peak_angles = chi1_centers[chi1_peaks]
            chi2_peak_angles = chi2_centers[chi2_peaks]

            print(f"  χ1 peaks: {chi1_peak_angles}")
            print(f"  χ2 peaks: {chi2_peak_angles}")
            f.write(f"χ1 peaks detected at: {chi1_peak_angles}\n")
            f.write(f"χ2 peaks detected at: {chi2_peak_angles}\n\n")

            f.write(f"χ1 statistics:\n")
            f.write(f"  Mean: {np.mean(chi1):.1f}°\n")
            f.write(f"  Std:  {np.std(chi1):.1f}°\n")
            f.write(f"  Range: [{np.min(chi1):.1f}, {np.max(chi1):.1f}]°\n\n")

            f.write(f"χ2 statistics:\n")
            f.write(f"  Mean: {np.mean(chi2):.1f}°\n")
            f.write(f"  Std:  {np.std(chi2):.1f}°\n")
            f.write(f"  Range: [{np.min(chi2):.1f}, {np.max(chi2):.1f}]°\n\n")

            if 'chi3' in angles and isinstance(angles['chi3'], dict):
                chi3 = get_all_frames(angles['chi3'])
                if len(chi3) > 0:
                    chi3_hist, chi3_bins = np.histogram(chi3, bins=72, range=(-180, 180))
                    chi3_centers = (chi3_bins[:-1] + chi3_bins[1:]) / 2
                    chi3_peaks   = detect_and_merge_peaks(chi3_hist)
                    angles.update({
                        'chi3_hist':         chi3_hist,
                        'chi3_centers':      chi3_centers,
                        'chi3_peak_indices': chi3_peaks
                    })
                    chi3_peak_angles = chi3_centers[chi3_peaks]
                    print(f"  χ3 peaks: {chi3_peak_angles}")
                    f.write(f"χ3 peaks detected at: {chi3_peak_angles}\n\n")

            f.write(f"Peak combinations (χ1 × χ2):\n")
            for chi1_peak in chi1_peak_angles:
                for chi2_peak in chi2_peak_angles:
                    chi1_cond = (chi1 >= chi1_peak - peak_tolerance) & (chi1 <= chi1_peak + peak_tolerance)
                    chi2_cond = (chi2 >= chi2_peak - peak_tolerance) & (chi2 <= chi2_peak + peak_tolerance)
                    count   = np.sum(chi1_cond & chi2_cond)
                    percent = (count / len(chi1)) * 100
                    f.write(f"  χ1={chi1_peak:6.1f}°, χ2={chi2_peak:6.1f}° → "
                            f"{count:6d} frames ({percent:5.1f}%)\n")

            f.write("\n" + "="*70 + "\n\n")

            peak_analysis[res_name] = {
                'pdb_label':  pdb_label,
                'chi1_peaks': chi1_peak_angles.tolist(),
                'chi2_peaks': chi2_peak_angles.tolist(),
            }

    with open(output_dir / 'combined_peak_analysis.json', 'w') as f:
        json.dump(peak_analysis, f, indent=2)

    print(f"\n✅ Peak analysis saved to: {output_dir}/combined_peak_analysis.txt")
    print(f"✅ Peak data saved to: {output_dir}/combined_peak_analysis.json")

    return peak_analysis


# =============================================================================
# BUCKET ANALYSIS
# =============================================================================

def analyze_chi1_buckets_combined(combined_data, output_dir):
    print("\n=== CHI1 BUCKET ANALYSIS (COMBINED) ===")
    bucket_summary = {}

    with open(output_dir / 'combined_chi1_bucket_analysis.txt', 'w') as f:
        f.write("CHI1 BUCKET ANALYSIS - COMBINED DATA\n")
        f.write("="*70 + "\n\n")
        f.write("  Bucket 1 (Up):    -125° to -25°\n")
        f.write("  Bucket 2 (Down):  -180° to -125° AND 125° to 180°\n")
        f.write("  Bucket 3 (Other): -25° to 125°\n\n")
        f.write("="*70 + "\n\n")

        for res_name, angles in combined_data.items():
            if 'chi1' not in angles:
                continue

            pdb_label    = angles.get('pdb_label', res_name)
            chi1         = get_all_frames(angles['chi1'])
            total_frames = len(chi1)

            b1 = np.sum((chi1 >= -125) & (chi1 <= -25))
            b2 = np.sum(((chi1 >= -180) & (chi1 <= -125)) | ((chi1 >= 125) & (chi1 <= 180)))
            b3 = np.sum((chi1 > -25) & (chi1 < 125))

            bucket_summary[res_name] = {
                'pdb_label':    pdb_label,
                'total_frames': total_frames,
                'bucket1': {'count': int(b1), 'percent': b1/total_frames*100},
                'bucket2': {'count': int(b2), 'percent': b2/total_frames*100},
                'bucket3': {'count': int(b3), 'percent': b3/total_frames*100}
            }

            f.write(f"{res_name} (PDB: {pdb_label})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total frames: {total_frames}\n\n")
            f.write(f"Bucket 1 (Up)    [-125° to -25°]:          {b1:6d} ({b1/total_frames*100:5.1f}%)\n")
            f.write(f"Bucket 2 (Down)  [-180° to -125° & 125°+]: {b2:6d} ({b2/total_frames*100:5.1f}%)\n")
            f.write(f"Bucket 3 (Other) [-25° to 125°]:           {b3:6d} ({b3/total_frames*100:5.1f}%)\n")
            f.write("\n" + "="*70 + "\n\n")

    print(f"✅ Bucket analysis saved to: {output_dir}/combined_chi1_bucket_analysis.txt")
    return bucket_summary


# =============================================================================
# PLOTTING HELPERS (all use get_all_frames — unchanged behaviour)
# =============================================================================

def plot_combined_distributions(combined_data, output_dir):
    print("\n" + "="*70)
    print("CREATING COMBINED DISTRIBUTION PLOTS")
    print("="*70)

    glu_results     = {k: v for k, v in combined_data.items() if v.get('residue_type') == 'GLU'}
    asn_asp_results = {k: v for k, v in combined_data.items() if v.get('residue_type') in ['ASN', 'ASP']}

    if glu_results:
        n_res = len(glu_results)
        fig, axes = plt.subplots(n_res, 3, figsize=(15, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)

        for i, (res_name, angles) in enumerate(glu_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            chi1      = get_all_frames(angles['chi1'])
            n_frames  = len(chi1)

            for j, chi_key in enumerate(['chi1', 'chi2', 'chi3']):
                hist    = angles.get(f'{chi_key}_hist')
                centers = angles.get(f'{chi_key}_centers')
                peaks   = angles.get(f'{chi_key}_peak_indices')
                colors  = ['skyblue', 'lightcoral', 'lightgreen']

                if hist is not None and len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if peaks is not None and len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks],
                                           color='red', s=100, marker='*',
                                           label='Peaks', zorder=5)

                axes[i, j].set_title(f'{pdb_label} - {chi_key.upper()}\n({n_frames:,} frames)',
                                     fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=11)
                axes[i, j].set_ylabel('Count', fontsize=11)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_distributions.png")

    if asn_asp_results:
        n_res = len(asn_asp_results)
        fig, axes = plt.subplots(n_res, 2, figsize=(12, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)

        for i, (res_name, angles) in enumerate(asn_asp_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            chi1      = get_all_frames(angles['chi1'])
            n_frames  = len(chi1)

            for j, chi_key in enumerate(['chi1', 'chi2']):
                hist    = angles.get(f'{chi_key}_hist')
                centers = angles.get(f'{chi_key}_centers')
                peaks   = angles.get(f'{chi_key}_peak_indices')
                colors  = ['lightsteelblue', 'salmon']

                if hist is not None and len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if peaks is not None and len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks],
                                           color='red', s=100, marker='*',
                                           label='Peaks', zorder=5)

                axes[i, j].set_title(f'{pdb_label} - {chi_key.upper()}\n({n_frames:,} frames)',
                                     fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=11)
                axes[i, j].set_ylabel('Count', fontsize=11)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_distributions.png")


def plot_combined_simple_histograms(combined_data, output_dir, n_runs, channel_type="G12"):
    print("\n=== CREATING COMBINED HISTOGRAM PLOTS ===")

    asn_asp_data = {k: v for k, v in combined_data.items()
                    if v.get('residue_type') in ['ASN', 'ASP']}
    glu_data     = {k: v for k, v in combined_data.items()
                    if v.get('residue_type') == 'GLU'}

    if channel_type == "G2":
        pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
    else:
        pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']

    def _ordered(data_dict, pdb_order):
        ordered = {}
        for pdb in pdb_order:
            for res_name, angles in data_dict.items():
                if angles.get('pdb_label') == pdb:
                    ordered[res_name] = angles
                    break
        return ordered

    def _grid(ordered, chi_key, chi_label, color, filename):
        if len(ordered) < 4:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        fig.suptitle(f'{chi_label} ({n_runs} runs)', fontsize=24, fontweight='bold', y=0.995)
        for idx, (res_name, angles) in enumerate(list(ordered.items())[:4]):
            ax        = axes[idx]
            pdb_label = angles.get('pdb_label', res_name)
            data      = get_all_frames(angles[chi_key])
            hist, bins = np.histogram(data, bins=72, range=(-180, 180))
            centers   = (bins[:-1] + bins[1:]) / 2
            ax.bar(centers, hist, width=5, alpha=0.85, color=color,
                   edgecolor='black', linewidth=0.8)
            ax.set_xlabel(f'{chi_label} Angle (degrees)', fontsize=26, fontweight='bold')
            ax.set_ylabel('Count', fontsize=26, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
            ax.set_xlim(-180, 180)
            ax.tick_params(axis='both', labelsize=22)
            ax.grid(True, alpha=0.3, linewidth=1)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/{filename}")

    ordered_asn = _ordered(asn_asp_data, pdb_order_asn)
    ordered_glu = _ordered(glu_data,     pdb_order_glu)

    _grid(ordered_asn, 'chi1', 'ASN/ASP χ1', 'steelblue', 'combined_ASN_ASP_chi1_grid.png')
    _grid(ordered_asn, 'chi2', 'ASN/ASP χ2', 'coral',     'combined_ASN_ASP_chi2_grid.png')
    _grid(ordered_glu, 'chi1', 'GLU χ1',     'steelblue', 'combined_GLU_chi1_grid.png')
    _grid(ordered_glu, 'chi2', 'GLU χ2',     'coral',     'combined_GLU_chi2_grid.png')


def plot_combined_chi1_buckets(bucket_summary, output_dir, n_runs, channel_type="G12"):
    print("\n=== CREATING COMBINED CHI1 BAR PLOTS ===")

    asn_asp_data = {k: v for k, v in bucket_summary.items()
                    if k.startswith('ASN') or k.startswith('ASP')}
    glu_data     = {k: v for k, v in bucket_summary.items()
                    if k.startswith('GLU')}

    if channel_type == "G2":
        pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
    else:
        pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']

    def _order(data_dict, pdb_order):
        ordered = {}
        for pdb in pdb_order:
            for res_name, data in data_dict.items():
                if data['pdb_label'] == pdb:
                    ordered[res_name] = data
                    break
        return ordered

    def _bar_grid(ordered, bucket_names, count_keys, colors, suptitle, filename):
        if len(ordered) < 4:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        for idx, (res_name, data) in enumerate(list(ordered.items())[:4]):
            ax     = axes[idx]
            counts = [data[k]['count'] for k in count_keys]
            bars   = ax.bar(range(len(bucket_names)), counts, color=colors,
                            alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(len(bucket_names)))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{data["pdb_label"]}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='both', labelsize=20)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count:,}', ha='center', va='bottom',
                        fontsize=18, fontweight='bold')
        plt.suptitle(suptitle, fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/{filename}")

    colors2 = ['#5DA5DA', '#FAA43A']
    colors3 = ['#5DA5DA', '#FAA43A', '#60BD68']

    for group_name, ordered in [('ASN_ASP', _order(asn_asp_data, pdb_order_asn)),
                                  ('GLU',     _order(glu_data, pdb_order_glu))]:
        t = f'{group_name} χ1 ({n_runs} runs)'
        _bar_grid(ordered, ['Up','Down'],       ['bucket1','bucket2'],         colors2, t,
                  f'combined_{group_name}_chi1_2buckets_counts_grid.png')
        _bar_grid(ordered, ['Up','Down','Other'],['bucket1','bucket2','bucket3'], colors3, t,
                  f'combined_{group_name}_chi1_3buckets_counts_grid.png')


# =============================================================================
# SAVE
# =============================================================================

def save_combined_data(combined_data, run_info, output_dir):
    """
    Save combined data.
    CHANGED: chi angles are now stored as per-run dicts inside combined_data.
    The structure is directly preserved — no extra transformation needed.
    """
    output_file = output_dir / 'combined_dihedral_data.pkl'
    save_dict   = {'combined_data': combined_data, 'run_info': run_info}
    with open(output_file, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"✅ Combined data (per-run format) saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combine dihedral analysis from multiple trajectory runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/girk_analyser_results/G12_GAT
  %(prog)s /path/to/girk_analyser_results/G12_GAT --runs 1 2 3 4 5
  %(prog)s /path/to/girk_analyser_results/G12_GAT -o combined_analysis -c G12_GAT
        """
    )
    parser.add_argument('base_dir')
    parser.add_argument('--runs', nargs='+', type=int)
    parser.add_argument('-c', '--channel', default='G12',
                        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'])
    parser.add_argument('-o', '--output', default='combined_dihedral_analysis')
    parser.add_argument('--peak-tolerance', type=float, default=15.0)

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return 1

    output_dir = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("COMBINING DIHEDRAL ANALYSIS FROM MULTIPLE RUNS")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    dihedral_files = find_dihedral_files(base_dir, args.runs)
    if not dihedral_files:
        print("\nERROR: No dihedral_raw_data.pkl files found!")
        return 1

    combined_data, run_info = load_and_combine_dihedral_data(dihedral_files)
    n_runs = len(dihedral_files)

    peak_analysis  = analyze_combined_peaks(combined_data, output_dir, args.peak_tolerance)
    bucket_summary = analyze_chi1_buckets_combined(combined_data, output_dir)

    plot_combined_chi1_buckets(bucket_summary, output_dir, n_runs, args.channel)
    plot_combined_simple_histograms(combined_data, output_dir, n_runs, args.channel)
    plot_combined_distributions(combined_data, output_dir)

    save_combined_data(combined_data, run_info, output_dir)

    print("\n" + "="*70)
    print("COMBINATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("  combined_peak_analysis.txt / .json")
    print("  combined_chi1_bucket_analysis.txt")
    print("  combined_dihedral_data.pkl  ← per-run format (new)")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())

# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis -c G12
# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis -c G2
# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis -c G12_GAT
# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis -c G12_ML
