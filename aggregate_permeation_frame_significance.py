#!/usr/bin/env python3
"""
Script to read individual RUN analysis results and create ONE COMBINED analysis
by pooling all the distance vectors together.

This script:
1. Finds all RUN directories with significance_analysis folders
2. Reads the saved JSON files (which contain all the distance vectors)
3. Pools ALL vectors from ALL runs together
4. Runs ONE comprehensive significance analysis on the pooled data
5. Generates all plots and statistics for the combined analysis

Expected directory structure:
    base_path/
    ├── RUN1/
    │   └── significance_analysis/
    │       └── permeation_significance_analysis.json
    ├── RUN2/
    │   └── significance_analysis/
    │       └── permeation_significance_analysis.json
    └── ...

Usage:
    python pool_and_analyze_all_runs.py /path/to/base_directory
    
    Or with custom output:
    python pool_and_analyze_all_runs.py /path/to/base_directory --output combined_analysis
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from scipy import stats


def find_run_directories(base_path):
    """
    Find all RUN directories with significance_analysis folders.
    
    Returns:
        list of tuples: (run_name, analysis_path)
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    run_dirs = []
    
    # Look for directories with significance_analysis subfolder
    for item in sorted(base_path.iterdir()):
        if item.is_dir(): 
            # analysis_dir = item / "significance_analysis"
            analysis_dir = item / "significance_analysis_end1"
            if analysis_dir.exists() and analysis_dir.is_dir():
                json_file = analysis_dir / "permeation_significance_analysis.json"
                if json_file.exists():
                    run_dirs.append((item.name, analysis_dir))
                    print(f"  ✓ Found: {item.name}")
    
    return run_dirs


def load_vectors_from_json(json_file):
    """
    Load the distance vectors from a saved JSON file.
    
    The JSON doesn't directly store individual frame vectors, but we can 
    reconstruct information from the bootstrap distributions and summaries.
    
    Actually, we need to modify the main analysis code to SAVE the actual
    vectors in the JSON for this to work!
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Check if vectors are saved (new format)
    if 'raw_vectors' in data:
        state2_vectors = np.array(data['raw_vectors']['state2_vectors'])
        end2_vectors = np.array(data['raw_vectors']['end2_vectors'])
        return state2_vectors, end2_vectors
    else:
        # Old format - vectors not saved
        return None, None


def pool_and_analyze(base_path, output_dir="POOLED_ANALYSIS", channel_type="G12"):
    """
    Read all individual run JSONs, pool the vectors, and run combined analysis.
    
    Parameters:
    -----------
    base_path : str or Path
        Path to directory containing RUN1, RUN2, etc.
    output_dir : str or Path
        Where to save combined analysis results
    channel_type : str
        Channel type for residue naming
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("POOLING VECTORS FROM MULTIPLE RUNS")
    print("="*80)
    print(f"\nSearching for RUN directories in: {base_path}")
    
    # Find all run directories
    run_dirs = find_run_directories(base_path)
    
    if len(run_dirs) == 0:
        print("\n❌ ERROR: No RUN directories with significance_analysis found!")
        return None
    
    print(f"\n✓ Found {len(run_dirs)} RUN directories\n")
    
    # Pool all vectors
    all_pooled_state2 = []
    all_pooled_end2 = []
    run_stats = []
    all_residues = None
    residue_pdb_map = None
    
    print("="*80)
    print("LOADING AND POOLING VECTORS")
    print("="*80)
    
    for run_name, analysis_path in run_dirs:
        json_file = analysis_path / "permeation_significance_analysis.json"
        
        print(f"\nLoading {run_name}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check if raw vectors are saved
        if 'raw_vectors' not in data:
            print(f"  ❌ ERROR: {run_name} JSON doesn't contain raw vectors!")
            print(f"     You need to re-run the analysis with the updated code that saves vectors.")
            continue
        
        # Extract vectors
        state2_vecs = np.array(data['raw_vectors']['state2_vectors'])
        end2_vecs = np.array(data['raw_vectors']['end2_vectors'])
        
        all_pooled_state2.append(state2_vecs)
        all_pooled_end2.append(end2_vecs)
        
        # Get residue info (should be same for all runs)
        if all_residues is None:
            all_residues = data['analysis_info']['residue_ids']
            residue_pdb_map = {int(k): v for k, v in 
                             zip(data['analysis_info']['residue_ids'],
                                 data['analysis_info']['residue_pdb_names'])}
        
        run_stats.append({
            'run': run_name,
            'n_state2': len(state2_vecs),
            'n_end2': len(end2_vecs),
            'n_events': data['analysis_info']['n_events']
        })
        
        print(f"  → {len(state2_vecs):,} 'in cavity' vectors")
        print(f"  → {len(end2_vecs)} 'leaving cavity' vectors")
        print(f"  → {data['analysis_info']['n_events']} permeation events")
    
    if len(all_pooled_state2) == 0:
        print("\n❌ ERROR: No valid data loaded!")
        print("Make sure to re-run individual analyses with updated code.")
        return None
    
    # Concatenate all vectors
    all_pooled_state2 = np.vstack(all_pooled_state2)
    all_pooled_end2 = np.vstack(all_pooled_end2)
    
    print("\n" + "="*80)
    print("POOLED DATA SUMMARY")
    print("="*80)
    print(f"Total 'in cavity' frames (for Gaussian): {len(all_pooled_state2):,}")
    print(f"Total 'leaving cavity' frames (tested): {len(all_pooled_end2):,}")
    print(f"Total runs pooled: {len(run_stats)}")
    print(f"Total events: {sum(s['n_events'] for s in run_stats)}")
    
    print("\nPer-run contributions:")
    print(f"{'Run':<12} {'In Cavity':>12} {'Leaving':>10} {'Events':>8}")
    print("-"*50)
    for stat in run_stats:
        print(f"{stat['run']:<12} {stat['n_state2']:>12,} {stat['n_end2']:>10} {stat['n_events']:>8}")
    
    # Save pooling summary
    summary_file = output_dir / "POOLING_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("POOLED ANALYSIS FROM MULTIPLE RUNS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total runs pooled: {len(run_stats)}\n")
        f.write(f"Total 'in cavity' frames: {len(all_pooled_state2):,}\n")
        f.write(f"Total 'leaving cavity' frames: {len(all_pooled_end2):,}\n")
        f.write(f"Total permeation events: {sum(s['n_events'] for s in run_stats)}\n\n")
        f.write("Per-run contributions:\n")
        f.write("-"*80 + "\n")
        for stat in run_stats:
            f.write(f"{stat['run']}:\n")
            f.write(f"  Events: {stat['n_events']}\n")
            f.write(f"  'In cavity' frames: {stat['n_state2']:,}\n")
            f.write(f"  'Leaving cavity' frames: {stat['n_end2']}\n\n")
    
    print(f"\n✓ Pooling summary saved: {summary_file}")
    
    # Now run the combined analysis
    print("\n" + "="*80)
    print("RUNNING COMBINED SIGNIFICANCE ANALYSIS")
    print("="*80)
    print("This analysis treats ALL data as ONE large dataset\n")
    
    # Run both statistical methods on the pooled data
    results = run_combined_statistical_tests(
        all_pooled_state2,
        all_pooled_end2,
        all_residues,
        residue_pdb_map,
        output_dir,
        n_bootstrap=10000,
        sample_size=100
    )
    
    return results


def run_combined_statistical_tests(
    state2_vectors, end2_vectors, all_residues, residue_pdb_map,
    output_dir, n_bootstrap=1000, sample_size=50
):
    """
    Run the full statistical analysis on pooled vectors.
    """
    
    print(f"Sample sizes:")
    print(f"  'In cavity' vectors: {len(state2_vectors):,}")
    print(f"  'Leaving cavity' vectors: {len(end2_vectors):,}")
    
    # Create report file
    report_file = output_dir / "POOLED_ANALYSIS_REPORT.txt"
    report = open(report_file, 'w')
    
    def print_and_write(text):
        print(text)
        report.write(text + '\n')
    
    print_and_write("\n" + "="*80)
    print_and_write("POOLED PERMEATION SIGNIFICANCE ANALYSIS")
    print_and_write("="*80)
    print_and_write(f"'In cavity' vectors: {len(state2_vectors):,}")
    print_and_write(f"'Leaving cavity' vectors: {len(end2_vectors):,}")
    
    # Calculate means
    state2_mean = np.mean(state2_vectors, axis=0)
    end2_mean = np.mean(end2_vectors, axis=0)
    
    print_and_write(f"\n{'Residue (PDB)':<15} {'In cavity':<12} {'Leaving':<12} {'Difference':<12}")
    print_and_write("-"*55)
    for i, resid in enumerate(all_residues):
        pdb_name = residue_pdb_map[resid]
        diff = end2_mean[i] - state2_mean[i]
        print_and_write(f"{pdb_name:<15} {state2_mean[i]:>10.2f} Å  {end2_mean[i]:>10.2f} Å  {diff:>+10.2f} Å")
    
    # =========================================================================
    # METHOD 1: CLT BOOTSTRAP
    # =========================================================================
    print_and_write("\n" + "="*80)
    print_and_write("METHOD 1: CLT BOOTSTRAP ON POOLED DATA")
    print_and_write("="*80)
    
    print_and_write(f"\nBootstrap parameters: {n_bootstrap} samples of size {sample_size}")
    
    np.random.seed(42)
    bootstrap_means = {resid: [] for resid in all_residues}
    
    print("Running bootstrap...")
    for boot_idx in range(n_bootstrap):
        if (boot_idx + 1) % 200 == 0:
            print(f"  Bootstrap {boot_idx + 1}/{n_bootstrap}")
        
        sample_indices = np.random.choice(len(state2_vectors), size=sample_size, replace=False)
        sample_vectors = state2_vectors[sample_indices]
        sample_mean = np.mean(sample_vectors, axis=0)
        
        for i, resid in enumerate(all_residues):
            bootstrap_means[resid].append(sample_mean[i])
    
    print_and_write(f"\n✓ Completed {n_bootstrap} bootstrap samples")
    
    # Calculate z-scores
    print_and_write("\nPer-residue significance:")
    print_and_write("-"*80)
    
    results_per_residue = {}
    bootstrap_distributions = {}  # Store for plotting!
    
    for i, resid in enumerate(all_residues):
        boot_means = np.array(bootstrap_means[resid])
        mu_boot = np.mean(boot_means)
        sigma_boot = np.std(boot_means)
        end2_value = end2_mean[i]
        z_score = (end2_value - mu_boot) / sigma_boot
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        pdb_name = residue_pdb_map[resid]
        
        results_per_residue[int(resid)] = {
            'residue_pdb': pdb_name,
            'bootstrap_mean': float(mu_boot),
            'bootstrap_std': float(sigma_boot),
            'state2_mean': float(state2_mean[i]),
            'end2_mean': float(end2_value),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'is_significant': bool(p_value < 0.05)
        }
        
        # Store bootstrap distribution for plotting
        bootstrap_distributions[int(resid)] = boot_means
        
        sig = ""
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        
        print_and_write(f"\n{pdb_name}:")
        print_and_write(f"  Z-score: {z_score:+.2f}, p-value: {p_value:.4f} {sig}")
    
    # Multivariate test
    print_and_write("\n" + "-"*80)
    print_and_write("Multivariate test (all residues combined):")
    
    bootstrap_mean_vectors = np.array([[bootstrap_means[resid][boot_idx] 
                                       for resid in all_residues]
                                      for boot_idx in range(n_bootstrap)])
    
    cov_boot = np.cov(bootstrap_mean_vectors.T)
    
    try:
        inv_cov = np.linalg.inv(cov_boot)
        diff_vec = end2_mean - np.mean(bootstrap_mean_vectors, axis=0)
        mahal_dist = np.sqrt(diff_vec @ inv_cov @ diff_vec)
        p_value_mv = 1 - stats.chi2.cdf(mahal_dist**2, df=len(all_residues))
        
        print_and_write(f"\nMahalanobis distance: {mahal_dist:.2f}")
        print_and_write(f"p-value: {p_value_mv:.4e}")
        
        if p_value_mv < 0.001:
            print_and_write("Significance: *** HIGHLY SIGNIFICANT")
        elif p_value_mv < 0.01:
            print_and_write("Significance: ** VERY SIGNIFICANT")
        elif p_value_mv < 0.05:
            print_and_write("Significance: * SIGNIFICANT")
        else:
            print_and_write("Significance: NOT SIGNIFICANT")
        
        mv_result = {
            'mahalanobis_distance': float(mahal_dist),
            'p_value': float(p_value_mv),
            'is_significant': bool(p_value_mv < 0.05)
        }
    except:
        print_and_write("\nERROR: Singular covariance matrix")
        mv_result = None
    
    # =========================================================================
    # METHOD 2: HOTELLING'S T²
    # =========================================================================
    print_and_write("\n" + "="*80)
    print_and_write("METHOD 2: HOTELLING'S T² ON POOLED DATA")
    print_and_write("="*80)
    
    n1 = len(state2_vectors)
    n2 = len(end2_vectors)
    p = state2_vectors.shape[1]
    
    print_and_write(f"\nGroup sizes: n1={n1:,}, n2={n2}, p={p}")
    
    mean1 = np.mean(state2_vectors, axis=0)
    mean2 = np.mean(end2_vectors, axis=0)
    diff = mean2 - mean1
    
    cov1 = np.cov(state2_vectors.T)
    cov2 = np.cov(end2_vectors.T)
    pooled_cov = ((n1-1)*cov1 + (n2-1)*cov2) / (n1+n2-2)
    
    try:
        T2 = (n1*n2)/(n1+n2) * diff @ np.linalg.inv(pooled_cov) @ diff
        F = (n1+n2-p-1) / ((n1+n2-2)*p) * T2
        df1 = p
        df2 = n1+n2-p-1
        p_value_hotelling = 1 - f_dist.cdf(F, df1, df2)
        
        print_and_write(f"\nT² statistic: {T2:.4f}")
        print_and_write(f"F statistic: {F:.4f}")
        print_and_write(f"p-value: {p_value_hotelling:.4e}")
        
        if p_value_hotelling < 0.001:
            print_and_write("Significance: *** HIGHLY SIGNIFICANT")
        elif p_value_hotelling < 0.01:
            print_and_write("Significance: ** VERY SIGNIFICANT")
        elif p_value_hotelling < 0.05:
            print_and_write("Significance: * SIGNIFICANT")
        else:
            print_and_write("Significance: NOT SIGNIFICANT")
        
        hotelling_result = {
            'T2_statistic': float(T2),
            'F_statistic': float(F),
            'p_value': float(p_value_hotelling),
            'is_significant': bool(p_value_hotelling < 0.05)
        }
    except:
        print_and_write("\nERROR: Singular covariance matrix")
        hotelling_result = None
    
    # Summary
    print_and_write("\n" + "="*80)
    print_and_write("POOLED ANALYSIS CONCLUSION")
    print_and_write("="*80)
    
    n_sig = sum(1 for r in results_per_residue.values() if r['is_significant'])
    print_and_write(f"\nSignificant residues: {n_sig}/{len(all_residues)}")
    
    if mv_result and mv_result['is_significant']:
        print_and_write(f"Multivariate test: SIGNIFICANT (p={mv_result['p_value']:.2e})")
    
    if hotelling_result and hotelling_result['is_significant']:
        print_and_write(f"Hotelling's T²: SIGNIFICANT (p={hotelling_result['p_value']:.2e})")
    
    report.close()
    
    # Save JSON
    combined_results = {
        'pooled_analysis': True,
        'n_state2_frames': int(len(state2_vectors)),
        'n_end2_frames': int(len(end2_vectors)),
        'n_residues': len(all_residues),
        'method1_results': results_per_residue,
        'method1_multivariate': mv_result,
        'method2_hotelling': hotelling_result,
        'residue_pdb_map': residue_pdb_map,
        'bootstrap_distributions': {k: v.tolist() for k, v in bootstrap_distributions.items()}  # For plotting
    }
    
    json_file = output_dir / "pooled_analysis_results.json"
    with open(json_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ JSON saved: {json_file}")
    print(f"✓ Report saved: {report_file}")
    
    # Create plots
    print("\nCreating comprehensive visualizations...")
    create_pooled_plots(combined_results, all_residues, residue_pdb_map, output_dir)
    
    print("\n" + "="*80)
    print("✓ POOLED ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print("  1. POOLING_SUMMARY.txt - Contribution from each run")
    print("  2. POOLED_ANALYSIS_REPORT.txt - Detailed statistical results")
    print("  3. pooled_analysis_results.json - Machine-readable data")
    print("\nPlots created:")
    print("  4. pooled_bootstrap_distributions.png - All 8 residues (2×4 grid)")
    print("  5. pooled_bootstrap_GLU_only.png - GLU residues only (1×4)")
    print("  6. pooled_bootstrap_ASN_only.png - ASN residues only (1×4)")
    print("  7. pooled_zscores.png - Z-score significance bar chart")
    print("  8. pooled_mean_distances.png - In cavity vs Leaving comparison")
    print("  9. pooled_distance_changes.png - How much each residue changes")
    print("\n" + "="*80)
    
    return combined_results


def create_pooled_plots(results, all_residues, residue_pdb_map, output_dir):
    """Create comprehensive plots for pooled analysis with larger text."""
    
    per_residue = results['method1_results']
    bootstrap_dists = results['bootstrap_distributions']  # Get actual distributions!
    
    # PLOT 1: Bootstrap distributions with HISTOGRAMS (like individual analysis)
    print("  Creating bootstrap distribution plots...")
    n_residues = len(all_residues)
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))  # SMALLER figure!
    axes = axes.flatten()
    
    for i, resid in enumerate(all_residues):
        ax = axes[i]
        res = per_residue[int(resid)]
        pdb_name = residue_pdb_map[resid]
        
        # Get actual bootstrap distribution
        boot_dist = np.array(bootstrap_dists[int(resid)])
        
        # Plot HISTOGRAM (like individual analysis!)
        ax.hist(boot_dist, bins=50, alpha=0.6, color='steelblue', 
               edgecolor='black', density=True, label='Bootstrap (in cavity)', linewidth=2.5)
        
        # Fitted Gaussian on top
        mu = res['bootstrap_mean']
        sigma = res['bootstrap_std']
        x = np.linspace(min(boot_dist), max(boot_dist), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=6,  # Even thicker
               label='Gaussian (CLT)', alpha=0.9)
        
        # Mark the leaving cavity mean
        end2_mean = res['end2_mean']
        ax.axvline(end2_mean, color='green', linestyle='--', 
                  linewidth=6, label='Leaving cavity mean')  # Even thicker
        
        # Add text labels for the mean values
        y_max = ax.get_ylim()[1]
        
        # Label for bootstrap mean (in cavity)
        ax.text(mu, y_max * 0.45, f'{mu:.1f}Å', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='steelblue', alpha=0.85, edgecolor='black', linewidth=2.5))
        
        # Label for leaving cavity mean
        ax.text(end2_mean, y_max * 0.30, f'{end2_mean:.1f}Å', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.85, edgecolor='black', linewidth=2.5))
        
        # Significance marker
        sig_marker = ""
        if res['p_value'] < 0.001:
            sig_marker = "***"
        elif res['p_value'] < 0.01:
            sig_marker = "**"
        elif res['p_value'] < 0.05:
            sig_marker = "*"
        
        # Title
        ax.set_title(f"{pdb_name}\nz={res['z_score']:+.2f}, p={res['p_value']:.4f} {sig_marker}", 
                    fontsize=24, fontweight='bold', pad=15)
        ax.set_xlabel('Distance (Å)', fontsize=22, fontweight='bold', labelpad=10)
        ax.set_ylabel('Density', fontsize=22, fontweight='bold', labelpad=10)
        # NO individual legends
        ax.grid(True, alpha=0.3, linewidth=2)
        ax.tick_params(labelsize=20, width=2.5, length=8)
    
    # Add ONE legend for the entire figure - BACK AT TOP with COMPACT spacing
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', alpha=0.6, label='Bootstrap (in cavity)', linewidth=2.5),
        Line2D([0], [0], color='red', linewidth=6, label='Gaussian (CLT)'),
        Line2D([0], [0], color='green', linewidth=6, linestyle='--', label='Leaving cavity mean')
    ]
    # HORIZONTAL legend at TOP - COMPACT with minimal padding
    fig.legend(handles=legend_elements, loc='upper center', fontsize=20, framealpha=0.98,
              bbox_to_anchor=(0.5, 1.01), edgecolor='black', fancybox=False, shadow=False,  # Higher!
              ncol=3, borderpad=0.5, labelspacing=0.3, columnspacing=1.0,
              handlelength=2, handleheight=1.5, handletextpad=0.5, borderaxespad=0.5)
    
    # NO suptitle
    plt.tight_layout(rect=[0, 0, 1.0, 0.98])  # Leave more space at top
    plt.savefig(output_dir / "pooled_bootstrap_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Bootstrap distributions saved")
    
    # EXTRA PLOTS: Separate GLU and ASN residues
    print("  Creating separate GLU and ASN plots...")
    
    # DEBUG: Show all residues
    print(f"\n  All residues: {all_residues}")
    print(f"  Residue PDB map: {residue_pdb_map}")
    
    # Identify GLU and ASN residues - CORRECTED!
    glu_residues = [r for r in all_residues if residue_pdb_map[r].startswith(('152', '141'))]  # GLU: 152.x and 141.D
    asn_residues = [r for r in all_residues if residue_pdb_map[r].startswith(('184', '173'))]  # ASN: 184.x and 173.D (ASP treated as ASN)
    
    print(f"\n  GLU residues (internal IDs): {glu_residues}")
    print(f"  GLU residues (PDB names): {[residue_pdb_map[r] for r in glu_residues]}")
    print(f"  ASN residues (internal IDs): {asn_residues}")
    print(f"  ASN residues (PDB names): {[residue_pdb_map[r] for r in asn_residues]}")
    
    if len(glu_residues) == 0:
        print("\n  ERROR: No GLU residues found! Skipping GLU plot.")
    
    if len(asn_residues) == 0:
        print("\n  ERROR: No ASN residues found! Skipping ASN plot.")
    
    # PLOT: GLU residues only
    if len(glu_residues) == 4:
        print("\n  Creating GLU plot...")
        # Reorder: 152.B (top-left), 152.C (top-right), 152.A (bottom-left), 141.D (bottom-right)
        glu_order = []
        for name in ['152.B', '152.C', '152.A', '141.D']:
            for r in glu_residues:
                if residue_pdb_map[r] == name:
                    glu_order.append(r)
                    break
        
        print(f"  GLU plot order: {[residue_pdb_map[r] for r in glu_order]}")
        
        if len(glu_order) != 4:
            print(f"  WARNING: Could not order all GLU residues! Got {len(glu_order)}/4")
            glu_order = glu_residues  # Use original order as fallback
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))  # SMALLER figure!
        axes = axes.flatten()
        
        for i, resid in enumerate(glu_order):
            ax = axes[i]
            res = per_residue[int(resid)]
            pdb_name = residue_pdb_map[resid]
            
            boot_dist = np.array(bootstrap_dists[int(resid)])
            
            ax.hist(boot_dist, bins=50, alpha=0.6, color='steelblue', 
                   edgecolor='black', density=True, linewidth=3.5)
            
            mu = res['bootstrap_mean']
            sigma = res['bootstrap_std']
            x = np.linspace(min(boot_dist), max(boot_dist), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=8, alpha=0.9)
            
            end2_mean = res['end2_mean']
            ax.axvline(end2_mean, color='green', linestyle='--', linewidth=8)
            
            y_max = ax.get_ylim()[1]
            ax.text(mu, y_max * 0.45, f'{mu:.1f}Å', 
                   ha='center', va='center', fontsize=26, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='steelblue', alpha=0.85, edgecolor='black', linewidth=3))
            ax.text(end2_mean, y_max * 0.30, f'{end2_mean:.1f}Å', 
                   ha='center', va='center', fontsize=26, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.85, edgecolor='black', linewidth=3))
            
            sig_marker = ""
            if res['p_value'] < 0.001:
                sig_marker = "***"
            elif res['p_value'] < 0.01:
                sig_marker = "**"
            elif res['p_value'] < 0.05:
                sig_marker = "*"
            
            # Color-code z-score based on magnitude
            z_val = res['z_score']
            if abs(z_val) > 3.29:  # p < 0.001
                z_color = 'red'
            elif abs(z_val) > 1.96:  # p < 0.05
                z_color = 'orange'
            else:
                z_color = 'black'
            
            ax.set_title(f"{pdb_name} (GLU)\nz={res['z_score']:+.2f}, p={res['p_value']:.4f} {sig_marker}", 
                        fontsize=30, fontweight='bold', pad=20, color=z_color)
            ax.set_xlabel('Distance (Å)', fontsize=28, fontweight='bold', labelpad=12)
            ax.set_ylabel('Density', fontsize=28, fontweight='bold', labelpad=12)
            ax.grid(True, alpha=0.3, linewidth=2.5)
            ax.tick_params(labelsize=26, width=3, length=10)
        
        # Legend for GLU plot
        legend_elements = [
            Patch(facecolor='steelblue', edgecolor='black', alpha=0.6, label='Bootstrap (in cavity)', linewidth=3),
            Line2D([0], [0], color='red', linewidth=8, label='Gaussian (CLT)'),
            Line2D([0], [0], color='green', linewidth=8, linestyle='--', label='Leaving cavity mean')
        ]
        fig.legend(handles=legend_elements, loc='upper center', fontsize=22, framealpha=0.98,
                  bbox_to_anchor=(0.5, 1.02), edgecolor='black', fancybox=False, shadow=False,
                  ncol=3, borderpad=0.5, labelspacing=0.3, columnspacing=1.0,
                  handlelength=2, handleheight=1.5, handletextpad=0.5, borderaxespad=0.5)
        
        plt.tight_layout(rect=[0, 0, 1.0, 0.98])
        plt.savefig(output_dir / "pooled_bootstrap_GLU_only.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ GLU residues plot saved")
    else:
        print(f"    ✗ Skipping GLU plot (found {len(glu_residues)} residues, expected 4)")
    
    # PLOT: ASN residues only
    if len(asn_residues) == 4:
        print("\n  Creating ASN plot...")
        # Reorder: 184.B (top-left), 184.C (top-right), 184.A (bottom-left), 173.D (bottom-right)
        asn_order = []
        for name in ['184.B', '184.C', '184.A', '173.D']:
            for r in asn_residues:
                if residue_pdb_map[r] == name:
                    asn_order.append(r)
                    break
        
        print(f"  ASN plot order: {[residue_pdb_map[r] for r in asn_order]}")
        
        if len(asn_order) != 4:
            print(f"  WARNING: Could not order all ASN residues! Got {len(asn_order)}/4")
            asn_order = asn_residues  # Use original order as fallback
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))  # SMALLER figure!
        axes = axes.flatten()
        
        for i, resid in enumerate(asn_order):
            ax = axes[i]
            res = per_residue[int(resid)]
            pdb_name = residue_pdb_map[resid]
            
            boot_dist = np.array(bootstrap_dists[int(resid)])
            
            ax.hist(boot_dist, bins=50, alpha=0.6, color='steelblue', 
                   edgecolor='black', density=True, linewidth=3.5)
            
            mu = res['bootstrap_mean']
            sigma = res['bootstrap_std']
            x = np.linspace(min(boot_dist), max(boot_dist), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=8, alpha=0.9)
            
            end2_mean = res['end2_mean']
            ax.axvline(end2_mean, color='green', linestyle='--', linewidth=8)
            
            y_max = ax.get_ylim()[1]
            ax.text(mu, y_max * 0.45, f'{mu:.1f}Å', 
                   ha='center', va='center', fontsize=26, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='steelblue', alpha=0.85, edgecolor='black', linewidth=3))
            ax.text(end2_mean, y_max * 0.30, f'{end2_mean:.1f}Å', 
                   ha='center', va='center', fontsize=26, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.85, edgecolor='black', linewidth=3))
            
            sig_marker = ""
            if res['p_value'] < 0.001:
                sig_marker = "***"
            elif res['p_value'] < 0.01:
                sig_marker = "**"
            elif res['p_value'] < 0.05:
                sig_marker = "*"
            
            # Color-code z-score based on magnitude
            z_val = res['z_score']
            if abs(z_val) > 3.29:  # p < 0.001
                z_color = 'red'
            elif abs(z_val) > 1.96:  # p < 0.05
                z_color = 'orange'
            else:
                z_color = 'black'
            
            # Label 173.D as ASP, others as ASN
            residue_type = "ASP" if pdb_name.startswith('173') else "ASN"
            
            ax.set_title(f"{pdb_name} ({residue_type})\nz={res['z_score']:+.2f}, p={res['p_value']:.4f} {sig_marker}", 
                        fontsize=30, fontweight='bold', pad=20, color=z_color)
            ax.set_xlabel('Distance (Å)', fontsize=28, fontweight='bold', labelpad=12)
            ax.set_ylabel('Density', fontsize=28, fontweight='bold', labelpad=12)
            ax.grid(True, alpha=0.3, linewidth=2.5)
            ax.tick_params(labelsize=26, width=3, length=10)
        
        # Legend for ASN plot
        fig.legend(handles=legend_elements, loc='upper center', fontsize=22, framealpha=0.98,
                  bbox_to_anchor=(0.5, 1.02), edgecolor='black', fancybox=False, shadow=False,
                  ncol=3, borderpad=0.5, labelspacing=0.3, columnspacing=1.0,
                  handlelength=2, handleheight=1.5, handletextpad=0.5, borderaxespad=0.5)
        
        plt.tight_layout(rect=[0, 0, 1.0, 0.98])
        plt.savefig(output_dir / "pooled_bootstrap_ASN_only.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ ASN residues plot saved")
    else:
        print(f"    ✗ Skipping ASN plot (found {len(asn_residues)} residues, expected 4)")
    
    # PLOT 2: Z-scores bar plot (with larger text)
    print("  Creating z-scores bar plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pdb_names = [residue_pdb_map[r] for r in all_residues]
    z_scores = [per_residue[int(r)]['z_score'] for r in all_residues]
    p_values = [per_residue[int(r)]['p_value'] for r in all_residues]
    
    colors = ['red' if p < 0.001 else 'orange' if p < 0.05 else 'steelblue' 
             for p in p_values]
    
    bars = ax.bar(range(len(pdb_names)), z_scores, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=2)
    
    # Add value labels with larger font
    for i, (bar, z) in enumerate(zip(bars, z_scores)):
        height = bar.get_height()
        label_y = height + 0.3 if height > 0 else height - 0.3
        ax.text(i, label_y, f'{z:+.1f}', ha='center', 
               va='bottom' if height > 0 else 'top', fontsize=14, fontweight='bold')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.axhline(y=1.96, color='orange', linestyle='--', linewidth=2.5, label='p=0.05 (z=±1.96)')
    ax.axhline(y=-1.96, color='orange', linestyle='--', linewidth=2.5)
    ax.axhline(y=3.29, color='red', linestyle='--', linewidth=2.5, label='p=0.001 (z=±3.29)')
    ax.axhline(y=-3.29, color='red', linestyle='--', linewidth=2.5)
    
    ax.set_xticks(range(len(pdb_names)))
    ax.set_xticklabels(pdb_names, fontsize=16, fontweight='bold')
    ax.set_xlabel('Residue (PDB numbering)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Z-score', fontsize=18, fontweight='bold')
    ax.set_title('POOLED ANALYSIS: Z-scores for Each Residue\n(All runs combined)', 
                fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=15, loc='best', framealpha=0.9)
    
    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='p < 0.001 ***'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='p < 0.05 *'),
        Patch(facecolor='steelblue', edgecolor='black', alpha=0.7, label='p ≥ 0.05 ns')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, title='Significance', 
             title_fontsize=15, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_zscores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Z-scores plot saved")
    
    # PLOT 3: Mean distances comparison
    print("  Creating mean distances comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    state2_means = [per_residue[int(r)]['state2_mean'] for r in all_residues]
    end2_means = [per_residue[int(r)]['end2_mean'] for r in all_residues]
    
    x = np.arange(len(pdb_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, state2_means, width, label='In cavity', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, end2_means, width, label='Leaving cavity', 
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Residue (PDB numbering)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Distance (Å)', fontsize=18, fontweight='bold')
    ax.set_title('POOLED ANALYSIS: Mean Ion-Residue Distances\n(In cavity vs Leaving cavity)', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pdb_names, fontsize=16, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_mean_distances.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Mean distances comparison saved")
    
    # PLOT 4: Difference plot (how much each residue changes)
    print("  Creating difference plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    differences = [end2_means[i] - state2_means[i] for i in range(len(pdb_names))]
    colors_diff = ['red' if d > 0 else 'blue' for d in differences]
    
    bars = ax.bar(range(len(pdb_names)), differences, color=colors_diff, 
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        height = bar.get_height()
        label_y = height + 0.2 if height > 0 else height - 0.2
        ax.text(i, label_y, f'{diff:+.1f}', ha='center', 
               va='bottom' if height > 0 else 'top', fontsize=13, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xticks(range(len(pdb_names)))
    ax.set_xticklabels(pdb_names, fontsize=16, fontweight='bold')
    ax.set_xlabel('Residue (PDB numbering)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Distance Change (Å)', fontsize=18, fontweight='bold')
    ax.set_title('POOLED ANALYSIS: Change in Distance When Leaving Cavity\n(Positive = farther, Negative = closer)', 
                fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Farther when leaving'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.7, label='Closer when leaving')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=15, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_distance_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Distance changes plot saved")
    
    print("\n  ✓ All plots created successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pool vectors from multiple RUN directories and run combined analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pool_and_analyze_all_runs.py /path/to/base_directory
  python pool_and_analyze_all_runs.py /path/to/base_directory --output POOLED_ANALYSIS
  python pool_and_analyze_all_runs.py /path/to/base_directory -o my_combined_results
        """
    )
    
    parser.add_argument('base_path', type=str,
                       help='Path to directory containing RUN1, RUN2, etc.')
    parser.add_argument('-o', '--output', type=str, default='POOLED_ANALYSIS',
                       help='Output directory name (default: POOLED_ANALYSIS)')
    parser.add_argument('--channel-type', type=str, default='G12',
                       help='Channel type for residue naming (default: G12)')
    
    args = parser.parse_args()
    
    pool_and_analyze(args.base_path, args.output, args.channel_type)

# Or with custom output:
# python3 aggregate_permeation_frame_significance.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/permeation_frame_significance

# python3 aggregate_permeation_frame_significance.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ --output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/permeation_frame_significance

# python3 aggregate_permeation_frame_significance.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ --output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/permeation_frame_significance

# python3 aggregate_permeation_frame_significance.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ --output /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/permeation_frame_significance