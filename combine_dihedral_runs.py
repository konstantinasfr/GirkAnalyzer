#!/usr/bin/env python3
"""
Combine Dihedral Analysis from Multiple Trajectories

This script:
1. Finds all dihedral_raw_data.pkl files across multiple RUN directories
2. Combines chi angle data from all runs
3. Creates combined histograms and peak analysis
4. Generates comprehensive plots and statistics

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


def find_dihedral_files(base_dir, run_numbers=None):
    """
    Find all dihedral_raw_data.pkl files in RUN directories.
    
    Parameters
    ----------
    base_dir : Path or str
        Base directory containing RUN1, RUN2, etc.
    run_numbers : list of int, optional
        Specific run numbers to process. If None, finds all RUN* directories.
    
    Returns
    -------
    list of Path
        Paths to dihedral_raw_data.pkl files
    """
    base_dir = Path(base_dir)
    dihedral_files = []
    
    if run_numbers is None:
        # Find all RUN* directories
        run_dirs = sorted(base_dir.glob("RUN*"))
        print("Found {} RUN directories".format(len(run_dirs)))
    else:
        # Use specific run numbers
        run_dirs = [base_dir / "RUN{}".format(num) for num in run_numbers]
        print("Looking for {} specified RUN directories".format(len(run_dirs)))
    
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"  [SKIP] {run_dir.name} - directory not found")
            continue
        
        # Look for dihedral_raw_data.pkl
        pkl_file = run_dir / "dihedral_analysis" / "dihedral_raw_data.pkl"
        
        if pkl_file.exists():
            dihedral_files.append(pkl_file)
            print(f"  [FOUND] {run_dir.name}/dihedral_raw_data.pkl")
        else:
            print(f"  [SKIP] {run_dir.name} - no dihedral_raw_data.pkl")
    
    return dihedral_files


def load_and_combine_dihedral_data(pkl_files):
    """
    Load and combine dihedral data from multiple pickle files.
    
    Parameters
    ----------
    pkl_files : list of Path
        Paths to dihedral_raw_data.pkl files
    
    Returns
    -------
    combined_data : dict
        Combined dihedral data with concatenated chi angles
    run_info : dict
        Information about number of frames per run
    """
    print("\n" + "="*70)
    print("LOADING AND COMBINING DIHEDRAL DATA")
    print("="*70)
    
    combined_data = {}
    run_info = {}
    
    for pkl_file in pkl_files:
        run_name = pkl_file.parent.name
        print(f"\nProcessing {run_name}...")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            run_frames = {}
            
            # Combine data for each residue
            for res_name, angles in data.items():
                if res_name not in combined_data:
                    # Initialize with first run's data
                    combined_data[res_name] = {
                        'chi1': [],
                        'chi2': [],
                        'pdb_label': angles.get('pdb_label', res_name),
                        'residue_type': angles.get('residue_type', 'UNK')
                    }
                    if 'chi3' in angles and angles['chi3'] is not None:
                        combined_data[res_name]['chi3'] = []
                
                # Append chi angles from this run
                if angles['chi1'] is not None:
                    combined_data[res_name]['chi1'].append(angles['chi1'])
                    n_frames = len(angles['chi1'])
                    run_frames[res_name] = n_frames
                
                if angles['chi2'] is not None:
                    combined_data[res_name]['chi2'].append(angles['chi2'])
                
                if 'chi3' in combined_data[res_name] and 'chi3' in angles and angles['chi3'] is not None:
                    combined_data[res_name]['chi3'].append(angles['chi3'])
            
            run_info[run_name] = run_frames
            print(f"  ✓ Loaded {list(run_frames.values())[0] if run_frames else 0} frames")
            
        except Exception as e:
            print(f"  ✗ Error loading {pkl_file}: {e}")
            continue
    
    # Concatenate arrays for each residue
    print("\nCombining arrays...")
    for res_name in combined_data:
        combined_data[res_name]['chi1'] = np.concatenate(combined_data[res_name]['chi1'])
        combined_data[res_name]['chi2'] = np.concatenate(combined_data[res_name]['chi2'])
        
        if 'chi3' in combined_data[res_name] and combined_data[res_name]['chi3']:
            combined_data[res_name]['chi3'] = np.concatenate(combined_data[res_name]['chi3'])
        
        pdb_label = combined_data[res_name]['pdb_label']
        n_total = len(combined_data[res_name]['chi1'])
        print(f"  {res_name} (PDB: {pdb_label}): {n_total} total frames")
    
    # Summary
    print("\n" + "="*70)
    print("COMBINATION SUMMARY")
    print("="*70)
    total_frames = sum(sum(frames.values()) for frames in run_info.values())
    print(f"Total runs combined: {len(pkl_files)}")
    print(f"Total frames: {total_frames}")
    print(f"Residues analyzed: {len(combined_data)}")
    
    return combined_data, run_info


def detect_and_merge_peaks(hist, min_prom=0.05, min_height=0.02, distance=8, 
                           merge_bins=8, smooth_sigma=1, wrap_bins=8):
    """Detect peaks in circular histograms with smoothing and wraparound."""
    num_bins = len(hist)
    smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
    padded = np.concatenate([smoothed[-wrap_bins:], smoothed, smoothed[:wrap_bins]])
    
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
        last = merged[-1]
        if (first + num_bins - last) <= merge_bins:
            merged = merged[1:]
            merged[-1] = int(np.median([first, last]))
        
        shifted_peaks = np.array(merged)
    
    return np.array(shifted_peaks)


def analyze_combined_peaks(combined_data, output_dir, peak_tolerance=15):
    """
    Analyze peaks in combined dihedral distributions.
    """
    print("\n" + "="*70)
    print("PEAK ANALYSIS ON COMBINED DATA")
    print("="*70)
    
    peak_analysis = {}
    
    with open(output_dir / 'combined_peak_analysis.txt', 'w') as f:
        f.write("COMBINED PEAK ANALYSIS - ALL TRAJECTORIES\n")
        f.write("="*70 + "\n\n")
        
        for res_name, angles in combined_data.items():
            pdb_label = angles.get('pdb_label', res_name)
            res_type = angles.get('residue_type', 'UNK')
            
            print(f"\n{res_name} (PDB: {pdb_label}):")
            f.write(f"{res_name} (PDB: {pdb_label}) - {res_type}\n")
            f.write("-"*60 + "\n")
            
            chi1 = np.array(angles['chi1'])
            chi2 = np.array(angles['chi2'])
            
            # Create histograms
            chi1_hist, chi1_bins = np.histogram(chi1, bins=72, range=(-180, 180))
            chi2_hist, chi2_bins = np.histogram(chi2, bins=72, range=(-180, 180))
            chi1_centers = (chi1_bins[:-1] + chi1_bins[1:]) / 2
            chi2_centers = (chi2_bins[:-1] + chi2_bins[1:]) / 2
            
            # Detect peaks
            chi1_peaks = detect_and_merge_peaks(chi1_hist)
            chi2_peaks = detect_and_merge_peaks(chi2_hist, merge_bins=5)
            
            # Store histogram data
            angles.update({
                'chi1_hist': chi1_hist,
                'chi2_hist': chi2_hist,
                'chi1_centers': chi1_centers,
                'chi2_centers': chi2_centers,
                'chi1_peak_indices': chi1_peaks,
                'chi2_peak_indices': chi2_peaks
            })
            
            chi1_peak_angles = chi1_centers[chi1_peaks]
            chi2_peak_angles = chi2_centers[chi2_peaks]
            
            print(f"  χ1 peaks: {chi1_peak_angles}")
            print(f"  χ2 peaks: {chi2_peak_angles}")
            f.write(f"χ1 peaks detected at: {chi1_peak_angles}\n")
            f.write(f"χ2 peaks detected at: {chi2_peak_angles}\n\n")
            
            # Statistics
            f.write(f"χ1 statistics:\n")
            f.write(f"  Mean: {np.mean(chi1):.1f}°\n")
            f.write(f"  Std:  {np.std(chi1):.1f}°\n")
            f.write(f"  Range: [{np.min(chi1):.1f}, {np.max(chi1):.1f}]°\n\n")
            
            f.write(f"χ2 statistics:\n")
            f.write(f"  Mean: {np.mean(chi2):.1f}°\n")
            f.write(f"  Std:  {np.std(chi2):.1f}°\n")
            f.write(f"  Range: [{np.min(chi2):.1f}, {np.max(chi2):.1f}]°\n\n")
            
            # Handle chi3 for GLU
            if 'chi3' in angles and isinstance(angles['chi3'], np.ndarray):
                chi3 = angles['chi3']
                chi3_hist, chi3_bins = np.histogram(chi3, bins=72, range=(-180, 180))
                chi3_centers = (chi3_bins[:-1] + chi3_bins[1:]) / 2
                chi3_peaks = detect_and_merge_peaks(chi3_hist)
                
                angles.update({
                    'chi3_hist': chi3_hist,
                    'chi3_centers': chi3_centers,
                    'chi3_peak_indices': chi3_peaks
                })
                
                chi3_peak_angles = chi3_centers[chi3_peaks]
                print(f"  χ3 peaks: {chi3_peak_angles}")
                f.write(f"χ3 peaks detected at: {chi3_peak_angles}\n\n")
                
                f.write(f"χ3 statistics:\n")
                f.write(f"  Mean: {np.mean(chi3):.1f}°\n")
                f.write(f"  Std:  {np.std(chi3):.1f}°\n")
                f.write(f"  Range: [{np.min(chi3):.1f}, {np.max(chi3):.1f}]°\n\n")
            
            # Peak combinations
            f.write(f"Peak combinations (χ1 × χ2):\n")
            for chi1_peak in chi1_peak_angles:
                for chi2_peak in chi2_peak_angles:
                    chi1_cond = (chi1 >= chi1_peak - peak_tolerance) & (chi1 <= chi1_peak + peak_tolerance)
                    chi2_cond = (chi2 >= chi2_peak - peak_tolerance) & (chi2 <= chi2_peak + peak_tolerance)
                    count = np.sum(chi1_cond & chi2_cond)
                    percent = (count / len(chi1)) * 100
                    
                    f.write(f"  χ1={chi1_peak:6.1f}°, χ2={chi2_peak:6.1f}° → "
                           f"{count:6d} frames ({percent:5.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n\n")
            
            peak_analysis[res_name] = {
                'pdb_label': pdb_label,
                'chi1_peaks': chi1_peak_angles.tolist(),
                'chi2_peaks': chi2_peak_angles.tolist(),
            }
    
    # Save peak analysis to JSON
    with open(output_dir / 'combined_peak_analysis.json', 'w') as f:
        json.dump(peak_analysis, f, indent=2)
    
    print(f"\n✅ Peak analysis saved to: {output_dir}/combined_peak_analysis.txt")
    print(f"✅ Peak data saved to: {output_dir}/combined_peak_analysis.json")
    
    return peak_analysis


def plot_combined_distributions(combined_data, output_dir):
    """
    Plot combined dihedral distributions from all trajectories.
    """
    print("\n" + "="*70)
    print("CREATING COMBINED DISTRIBUTION PLOTS")
    print("="*70)
    
    # Separate by residue type
    glu_results = {k: v for k, v in combined_data.items() if v.get('residue_type') == 'GLU'}
    asn_asp_results = {k: v for k, v in combined_data.items() if v.get('residue_type') in ['ASN', 'ASP']}
    
    # Plot GLU residues
    if glu_results:
        n_res = len(glu_results)
        fig, axes = plt.subplots(n_res, 3, figsize=(15, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)
        
        for i, (res_name, angles) in enumerate(glu_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            n_frames = len(angles['chi1'])
            
            for j, chi in enumerate(['chi1', 'chi2', 'chi3']):
                hist = angles.get(f'{chi}_hist')
                centers = angles.get(f'{chi}_centers')
                peaks = angles.get(f'{chi}_peak_indices')
                colors = ['skyblue', 'lightcoral', 'lightgreen']
                
                if hist is not None and len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if peaks is not None and len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks], 
                                         color='red', s=100, marker='*', 
                                         label='Peaks', zorder=5, edgecolors='darkred', linewidths=1.5)
                
                axes[i, j].set_title(f'{pdb_label} - {chi.upper()}\n({n_frames:,} frames, all runs)', 
                                    fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=11)
                axes[i, j].set_ylabel('Count', fontsize=11)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)
                if peaks is not None and len(peaks) > 0:
                    axes[i, j].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_distributions.png")
    
    # Plot ASN/ASP residues
    if asn_asp_results:
        n_res = len(asn_asp_results)
        fig, axes = plt.subplots(n_res, 2, figsize=(12, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)
        
        for i, (res_name, angles) in enumerate(asn_asp_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            n_frames = len(angles['chi1'])
            
            for j, chi in enumerate(['chi1', 'chi2']):
                hist = angles.get(f'{chi}_hist')
                centers = angles.get(f'{chi}_centers')
                peaks = angles.get(f'{chi}_peak_indices')
                colors = ['lightsteelblue', 'salmon']
                
                if hist is not None and len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if peaks is not None and len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks], 
                                         color='red', s=100, marker='*', 
                                         label='Peaks', zorder=5, edgecolors='darkred', linewidths=1.5)
                
                axes[i, j].set_title(f'{pdb_label} - {chi.upper()}\n({n_frames:,} frames, all runs)', 
                                    fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=11)
                axes[i, j].set_ylabel('Count', fontsize=11)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)
                if peaks is not None and len(peaks) > 0:
                    axes[i, j].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_distributions.png")


def analyze_chi1_buckets_combined(combined_data, output_dir):
    """
    Analyze chi1 angles in specific buckets for combined data.
    
    Buckets:
    - Bucket 1 (Up): -125° to -25°
    - Bucket 2 (Down): -180° to -125° AND 125° to 180°
    - Bucket 3 (Other): -25° to 125°
    """
    print("\n=== CHI1 BUCKET ANALYSIS (COMBINED) ===")
    
    bucket_summary = {}
    
    with open(output_dir / 'combined_chi1_bucket_analysis.txt', 'w') as f:
        f.write("CHI1 BUCKET ANALYSIS - COMBINED DATA\n")
        f.write("="*70 + "\n\n")
        f.write("Bucket definitions:\n")
        f.write("  Bucket 1 (Up): -125° to -25°\n")
        f.write("  Bucket 2 (Down): -180° to -125° AND 125° to 180°\n")
        f.write("  Bucket 3 (Other): -25° to 125°\n\n")
        f.write("="*70 + "\n\n")
        
        for res_name, angles in combined_data.items():
            if 'chi1' not in angles or angles['chi1'] is None:
                continue
                
            pdb_label = angles.get('pdb_label', res_name)
            chi1 = np.array(angles['chi1'])
            total_frames = len(chi1)
            
            # Define buckets
            bucket1_mask = (chi1 >= -125) & (chi1 <= -25)
            bucket2_mask = ((chi1 >= -180) & (chi1 <= -125)) | ((chi1 >= 125) & (chi1 <= 180))
            bucket3_mask = (chi1 > -25) & (chi1 < 125)
            
            bucket1_count = np.sum(bucket1_mask)
            bucket2_count = np.sum(bucket2_mask)
            bucket3_count = np.sum(bucket3_mask)
            
            bucket1_pct = (bucket1_count / total_frames) * 100
            bucket2_pct = (bucket2_count / total_frames) * 100
            bucket3_pct = (bucket3_count / total_frames) * 100
            
            # Store results
            bucket_summary[res_name] = {
                'pdb_label': pdb_label,
                'total_frames': total_frames,
                'bucket1': {'count': int(bucket1_count), 'percent': bucket1_pct},
                'bucket2': {'count': int(bucket2_count), 'percent': bucket2_pct},
                'bucket3': {'count': int(bucket3_count), 'percent': bucket3_pct}
            }
            
            # Write to file
            f.write(f"{res_name} (PDB: {pdb_label})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total frames: {total_frames}\n\n")
            f.write(f"Bucket 1 (Up) [-125° to -25°]:  {bucket1_count:6d} frames ({bucket1_pct:5.1f}%)\n")
            f.write(f"Bucket 2 (Down) [-180° to -125° & 125° to 180°]: {bucket2_count:6d} frames ({bucket2_pct:5.1f}%)\n")
            f.write(f"Bucket 3 (Other) [-25° to 125°]:   {bucket3_count:6d} frames ({bucket3_pct:5.1f}%)\n")
            f.write("\n" + "="*70 + "\n\n")
            
            print(f"{res_name} (PDB: {pdb_label}):")
            print(f"  Bucket 1 (Up): {bucket1_count} frames ({bucket1_pct:.1f}%)")
            print(f"  Bucket 2 (Down): {bucket2_count} frames ({bucket2_pct:.1f}%)")
            print(f"  Bucket 3 (Other): {bucket3_count} frames ({bucket3_pct:.1f}%)")
    
    print(f"✅ Bucket analysis saved to: {output_dir}/combined_chi1_bucket_analysis.txt")
    
    return bucket_summary


def plot_combined_chi1_buckets(bucket_summary, output_dir, n_runs, channel_type="G12"):
    """Create 2x2 grid bar plots for chi1 bucket analysis on combined data."""
    print("\n=== CREATING COMBINED CHI1 BAR PLOTS ===")
    print(f"DEBUG: Received channel_type = '{channel_type}'")
    
    # Separate ASN/ASP and GLU data
    asn_asp_data = {k: v for k, v in bucket_summary.items() 
                    if k.startswith('ASN') or k.startswith('ASP')}
    glu_data = {k: v for k, v in bucket_summary.items() 
                if k.startswith('GLU')}
    
    # Determine PDB order based on channel type (in 2x2 grid order)
    if channel_type == "G2":
        pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
    else:  # G12, G12_GAT, G12_ML
        pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']
    
    print(f"DEBUG: Using pdb_order_asn = {pdb_order_asn}")
    
    # =========================================================================
    # ASN/ASP BAR PLOTS
    # =========================================================================
    if asn_asp_data:
        print(f"DEBUG: asn_asp_data keys and labels:")
        for res_name, data in asn_asp_data.items():
            print(f"  {res_name} -> {data.get('pdb_label', 'NO LABEL')}")
        
        ordered_data = {}
        for pdb in pdb_order_asn:
            found = False
            for res_name, data in asn_asp_data.items():
                if data['pdb_label'] == pdb:
                    ordered_data[res_name] = data
                    found = True
                    break
            if not found:
                print(f"WARNING: Could not find residue with PDB label {pdb}")
        
        print(f"DEBUG: Found {len(ordered_data)} ASN/ASP residues for plotting:")
        for res_name, data in ordered_data.items():
            print(f"  {res_name} -> {data['pdb_label']}")
        
        residues_list = list(ordered_data.items())
        
        # Plot 1: ASN/ASP 2 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down']
            counts = [data['bucket1']['count'], data['bucket2']['count']]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'ASN/ASP χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_chi1_2buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi1_2buckets_counts_grid.png")
        
        # Plot 2: ASN/ASP 2 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            total = data['total_frames']
            bucket_names = ['Up', 'Down']
            percents = [(data['bucket1']['count']/total)*100, (data['bucket2']['count']/total)*100]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'ASN/ASP χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_chi1_2buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi1_2buckets_percent_grid.png")
        
        # Plot 3: ASN/ASP 3 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down', 'Other']
            counts = [data['bucket1']['count'], data['bucket2']['count'], data['bucket3']['count']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'ASN/ASP χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_chi1_3buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi1_3buckets_counts_grid.png")
        
        # Plot 4: ASN/ASP 3 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down', 'Other']
            percents = [data['bucket1']['percent'], data['bucket2']['percent'], data['bucket3']['percent']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'ASN/ASP χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ASN_ASP_chi1_3buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi1_3buckets_percent_grid.png")
    
    # =========================================================================
    # GLU BAR PLOTS
    # =========================================================================
    if glu_data:
        ordered_glu = {}
        for pdb in pdb_order_glu:
            for res_name, data in glu_data.items():
                if data['pdb_label'] == pdb:
                    ordered_glu[res_name] = data
                    break
        
        residues_list_glu = list(ordered_glu.items())
        
        # Plot 1: GLU 2 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down']
            counts = [data['bucket1']['count'], data['bucket2']['count']]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'GLU χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_chi1_2buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_chi1_2buckets_counts_grid.png")
        
        # Plot 2: GLU 2 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            total = data['total_frames']
            bucket_names = ['Up', 'Down']
            percents = [(data['bucket1']['count']/total)*100, (data['bucket2']['count']/total)*100]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'GLU χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_chi1_2buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_chi1_2buckets_percent_grid.png")
        
        # Plot 3: GLU 3 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down', 'Other']
            counts = [data['bucket1']['count'], data['bucket2']['count'], data['bucket3']['count']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'GLU χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_chi1_3buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_chi1_3buckets_counts_grid.png")
        
        # Plot 4: GLU 3 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            bucket_names = ['Up', 'Down', 'Other']
            percents = [data['bucket1']['percent'], data['bucket2']['percent'], data['bucket3']['percent']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'GLU χ1 ({n_runs} runs)', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_GLU_chi1_3buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/combined_GLU_chi1_3buckets_percent_grid.png")


def plot_combined_simple_histograms(combined_data, output_dir, n_runs, channel_type="G12"):
    """Create 2x2 grid histograms for chi1 and chi2 separately."""
    print("\n=== CREATING COMBINED HISTOGRAM PLOTS ===")
    
    asn_asp_data = {k: v for k, v in combined_data.items() 
                    if v.get('residue_type') in ['ASN', 'ASP']}
    glu_data = {k: v for k, v in combined_data.items() 
                if v.get('residue_type') == 'GLU'}
    
    # Determine PDB order based on channel type (in 2x2 grid order)
    if channel_type == "G2":
        pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
    else:  # G12, G12_GAT, G12_ML
        pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']
    
    # ASN/ASP Chi1 Grid
    if asn_asp_data:
        ordered_asn = {}
        for pdb in pdb_order_asn:
            for res_name, angles in asn_asp_data.items():
                if angles.get('pdb_label') == pdb:
                    ordered_asn[res_name] = angles
                    break
        
        if len(ordered_asn) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            fig.suptitle(f'ASN/ASP χ1 ({n_runs} runs)', fontsize=24, fontweight='bold', y=0.995)
            
            for idx, (res_name, angles) in enumerate(list(ordered_asn.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi1 = np.array(angles['chi1'])
                
                hist, bins = np.histogram(chi1, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='steelblue', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ1 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'combined_ASN_ASP_chi1_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi1_grid.png")
        
        # ASN/ASP Chi2 Grid
        if len(ordered_asn) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            fig.suptitle(f'ASN/ASP χ2 ({n_runs} runs)', fontsize=24, fontweight='bold', y=0.995)
            
            for idx, (res_name, angles) in enumerate(list(ordered_asn.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi2 = np.array(angles['chi2'])
                
                hist, bins = np.histogram(chi2, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='coral', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ2 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'combined_ASN_ASP_chi2_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/combined_ASN_ASP_chi2_grid.png")
    
    # GLU Chi1 and Chi2 Grids (similar code)
    if glu_data:
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']
        ordered_glu = {}
        for pdb in pdb_order_glu:
            for res_name, angles in glu_data.items():
                if angles.get('pdb_label') == pdb:
                    ordered_glu[res_name] = angles
                    break
        
        if len(ordered_glu) >= 4:
            # GLU Chi1
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            fig.suptitle(f'GLU χ1 ({n_runs} runs)', fontsize=24, fontweight='bold', y=0.995)
            
            for idx, (res_name, angles) in enumerate(list(ordered_glu.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi1 = np.array(angles['chi1'])
                
                hist, bins = np.histogram(chi1, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='steelblue', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ1 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'combined_GLU_chi1_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/combined_GLU_chi1_grid.png")
            
            # GLU Chi2
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            fig.suptitle(f'GLU χ2 ({n_runs} runs)', fontsize=24, fontweight='bold', y=0.995)
            
            for idx, (res_name, angles) in enumerate(list(ordered_glu.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi2 = np.array(angles['chi2'])
                
                hist, bins = np.histogram(chi2, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='coral', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ2 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'combined_GLU_chi2_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/combined_GLU_chi2_grid.png")


def save_combined_data(combined_data, run_info, output_dir):
    """Save combined data for future use."""
    output_file = output_dir / 'combined_dihedral_data.pkl'
    
    save_dict = {
        'combined_data': combined_data,
        'run_info': run_info
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"✅ Combined data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine dihedral analysis from multiple trajectory runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all RUN* directories found in the base directory
  %(prog)s /path/to/girk_analyser_results/G12_GAT
  
  # Combine specific runs only
  %(prog)s /path/to/girk_analyser_results/G12_GAT --runs 1 2 3 4 5
  
  # Specify output directory
  %(prog)s /path/to/girk_analyser_results/G12_GAT -o combined_analysis
        """
    )
    
    parser.add_argument('base_dir', 
                       help='Base directory containing RUN1, RUN2, etc.')
    parser.add_argument('--runs', nargs='+', type=int,
                       help='Specific run numbers to combine (e.g., --runs 1 2 3 4 5)')
    parser.add_argument('-c', '--channel', default='G12',
                       choices=['G2', 'G12', 'G12_GAT', 'G12_ML'],
                       help='Channel type for PDB numbering (default: G12)')
    parser.add_argument('-o', '--output', default='combined_dihedral_analysis',
                       help='Output directory name (default: combined_dihedral_analysis)')
    parser.add_argument('--peak-tolerance', type=float, default=15.0,
                       help='Tolerance for peak matching in degrees (default: 15)')
    
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
    if args.runs:
        print(f"Specific runs: {args.runs}")
    else:
        print("Processing: All RUN* directories found")
    print("="*70)
    
    # Find dihedral files
    dihedral_files = find_dihedral_files(base_dir, args.runs)
    
    if not dihedral_files:
        print("\nERROR: No dihedral_raw_data.pkl files found!")
        print("\nMake sure:")
        print("  1. The base directory is correct")
        print("  2. RUN directories contain dihedral_raw_data.pkl")
        print("  3. You've run the dihedral analysis on each trajectory")
        return 1
    
    # Load and combine data
    combined_data, run_info = load_and_combine_dihedral_data(dihedral_files)
    
    n_runs = len(dihedral_files)
    
    # Analyze peaks
    peak_analysis = analyze_combined_peaks(combined_data, output_dir, args.peak_tolerance)
    
    # Analyze chi1 buckets
    bucket_summary = analyze_chi1_buckets_combined(combined_data, output_dir)
    
    # Create chi1 bucket bar plots
    plot_combined_chi1_buckets(bucket_summary, output_dir, n_runs, args.channel)
    
    # Create simple histogram grids
    plot_combined_simple_histograms(combined_data, output_dir, n_runs, args.channel)
    
    # Create original plots (with peaks)
    plot_combined_distributions(combined_data, output_dir)
    
    # Save combined data
    save_combined_data(combined_data, run_info, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("COMBINATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("  • combined_peak_analysis.txt - Peak summary")
    print("  • combined_peak_analysis.json - Peak data")
    print("  • combined_GLU_distributions.png - GLU plots")
    print("  • combined_ASN_ASP_distributions.png - ASN/ASP plots")
    print("  • combined_dihedral_data.pkl - Complete combined data")
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis
# python3 combine_dihedral_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis -c G2
