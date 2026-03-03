import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from analysis.converter import convert_to_pdb_numbering


def combine_multiple_runs_force_correlation(
    base_dir,
    channel_type="G12",
    output_dir=None
):
    """
    Combine force correlation data from multiple RUN folders and create combined plots.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing RUN1, RUN2, RUN3, etc.
        Each RUN folder should have: force_correlation_analysis/force_correlation_data.json
    channel_type : str
        Channel type for PDB numbering (G2, G12, G12_ML, G12_GAT)
    output_dir : str or Path, optional
        Where to save combined plots. If None, saves in base_dir/combined_force_correlation
        
    Returns:
    --------
    dict : Combined data from all runs
    
    Example:
    --------
    combine_multiple_runs_force_correlation(
        base_dir="/path/to/results/G12",
        channel_type="G12"
    )
    """
    base_dir = Path(base_dir)
    
    if output_dir is None:
        output_dir = base_dir / "combined_force_correlation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMBINING FORCE CORRELATION DATA FROM MULTIPLE RUNS")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Channel type: {channel_type}")
    
    # Find all RUN folders
    run_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')])
    
    if len(run_folders) == 0:
        print(f"ERROR: No RUN folders found in {base_dir}")
        return None
    
    print(f"\nFound {len(run_folders)} RUN folders:")
    for folder in run_folders:
        print(f"  - {folder.name}")
    
    # Collect all force data from all runs
    all_force_data = []
    run_stats = []
    
    for run_folder in run_folders:
        json_file = run_folder / "force_correlation_analysis" / "force_correlation_data.json"
        
        if not json_file.exists():
            print(f"\nWARNING: {run_folder.name} missing force_correlation_data.json, skipping...")
            continue
        
        # Load data
        with open(json_file, 'r') as f:
            run_data = json.load(f)
        
        # Add run information to each event
        for event in run_data:
            event['run_name'] = run_folder.name
        
        all_force_data.extend(run_data)
        
        run_stats.append({
            'run_name': run_folder.name,
            'n_events': len(run_data)
        })
        
        print(f"\n{run_folder.name}: {len(run_data)} permeation events")
    
    if len(all_force_data) == 0:
        print("\nERROR: No data found in any RUN folders!")
        return None
    
    # Summary
    print("\n" + "="*80)
    print("COMBINED DATA SUMMARY")
    print("="*80)
    print(f"Total runs: {len(run_stats)}")
    print(f"Total permeation events: {len(all_force_data)}")
    print(f"\nPer-run breakdown:")
    for stat in run_stats:
        print(f"  {stat['run_name']}: {stat['n_events']} events")
    
    # Save combined data
    combined_json = output_dir / "combined_force_correlation_data.json"
    with open(combined_json, 'w') as f:
        json.dump({
            'run_stats': run_stats,
            'total_events': len(all_force_data),
            'all_events': all_force_data
        }, f, indent=2)
    print(f"\n✓ Combined data saved to: {combined_json}")
    
    # Extract residue lists from first event
    first_event = all_force_data[0]
    glu_residues = sorted([int(k) for k in first_event['glu_individual_distances'].keys()])
    asn_residues = sorted([int(k) for k in first_event['asn_individual_distances'].keys()])
    
    print(f"\nGLU residues: {glu_residues}")
    print(f"ASN residues: {asn_residues}")
    
    # Create combined plots
    print("\n" + "="*80)
    print("CREATING COMBINED PLOTS")
    print("="*80)
    
    create_combined_plots(all_force_data, glu_residues, asn_residues, 
                         channel_type, output_dir)
    
    print("\n" + "="*80)
    print("✓ COMBINED ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved in: {output_dir}")
    print(f"  - combined_force_correlation_data.json")
    print(f"  - force_corr_combined_individual_GLU.png")
    print(f"  - force_corr_combined_individual_ASN.png")
    print(f"  - force_corr_combined_mean_distances.png")
    
    return {
        'all_force_data': all_force_data,
        'run_stats': run_stats,
        'glu_residues': glu_residues,
        'asn_residues': asn_residues
    }


def create_combined_plots(all_force_data, glu_residues, asn_residues, 
                         channel_type, output_dir):
    """
    Create the 10 combined plots using all events from all runs.
    """
    
    # Extract force Z-component (along channel axis)
    forces_z = np.array([d['force_z_pN'] for d in all_force_data])
    
    print(f"\nCreating 10 combined plots using {len(all_force_data)} events...")
    print(f"Using Z-component of force")
    
    # =================================================================
    # PLOT 1-4: Individual GLU residues
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, glu_resid in enumerate(glu_residues):
        ax = axes[idx]
        
        # Extract distances to this specific GLU residue
        distances = []
        for d in all_force_data:
            if str(glu_resid) in d['glu_individual_distances']:
                distances.append(d['glu_individual_distances'][str(glu_resid)])
            elif glu_resid in d['glu_individual_distances']:
                distances.append(d['glu_individual_distances'][glu_resid])
            else:
                distances.append(np.nan)
        
        distances = np.array(distances)
        
        # Plot - NO labels on individual points
        ax.scatter(distances, forces_z, s=150, alpha=0.6, c='red', 
                  edgecolors='black', linewidth=1.5)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Convert to PDB numbering
        pdb_label = convert_to_pdb_numbering(glu_resid, channel_type)
        
        ax.set_xlabel(f'Distance to GLU {pdb_label} (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Force Z (pN)', fontsize=14, fontweight='bold')
        ax.set_title(f'GLU {pdb_label} (n={len(all_force_data)})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_file = output_dir / "force_corr_combined_individual_GLU.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Combined GLU individual plots saved: {output_file}")
    
    # =================================================================
    # PLOT 5-8: Individual ASN residues
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, asn_resid in enumerate(asn_residues):
        ax = axes[idx]
        
        # Extract distances to this specific ASN residue
        distances = []
        for d in all_force_data:
            if str(asn_resid) in d['asn_individual_distances']:
                distances.append(d['asn_individual_distances'][str(asn_resid)])
            elif asn_resid in d['asn_individual_distances']:
                distances.append(d['asn_individual_distances'][asn_resid])
            else:
                distances.append(np.nan)
        
        distances = np.array(distances)
        
        # Plot - NO labels on individual points
        ax.scatter(distances, forces_z, s=150, alpha=0.6, c='blue', 
                  edgecolors='black', linewidth=1.5)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Convert to PDB numbering
        pdb_label = convert_to_pdb_numbering(asn_resid, channel_type)
        
        ax.set_xlabel(f'Distance to ASN {pdb_label} (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Force Z (pN)', fontsize=14, fontweight='bold')
        ax.set_title(f'ASN {pdb_label} (n={len(all_force_data)})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_file = output_dir / "force_corr_combined_individual_ASN.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Combined ASN individual plots saved: {output_file}")
    
    # =================================================================
    # PLOT 9-10: Mean distances
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLOT 9: Mean GLU distance
    glu_mean_dist = np.array([d['glu_average_distance'] for d in all_force_data])
    
    ax1.scatter(glu_mean_dist, forces_z, s=180, alpha=0.6, c='darkred', 
               edgecolors='black', linewidth=2)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.set_xlabel('Mean Distance to All 4 GLU (Å)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Force Z (pN)', fontsize=15, fontweight='bold')
    ax1.set_title(f'Mean GLU Distance vs Force (n={len(all_force_data)})', 
                 fontsize=17, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=13)
    
    # PLOT 10: Mean ASN distance
    asn_mean_dist = np.array([d['asn_average_distance'] for d in all_force_data])
    
    ax2.scatter(asn_mean_dist, forces_z, s=180, alpha=0.6, c='darkblue', 
               edgecolors='black', linewidth=2)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Mean Distance to All 4 ASN (Å)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Force Z (pN)', fontsize=15, fontweight='bold')
    ax2.set_title(f'Mean ASN Distance vs Force (n={len(all_force_data)})', 
                 fontsize=17, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)
    
    plt.tight_layout()
    output_file = output_dir / "force_corr_combined_mean_distances.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Combined mean distance plots saved: {output_file}")
    
    print(f"\n✓ All 10 combined plots created successfully!")
    print(f"  - 4 individual GLU plots")
    print(f"  - 4 individual ASN plots") 
    print(f"  - 2 mean distance plots (GLU and ASN)")


# =============================================================================
# CONVENIENCE FUNCTION - USE THIS!
# =============================================================================

if __name__ == "__main__":
    """
    Example usage from command line or as a script.
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python combine_force_correlation_runs.py <base_dir> <channel_type> [output_dir]")
        print("\nArguments:")
        print("  base_dir      : Directory containing RUN1, RUN2, etc.")
        print("  channel_type  : G2, G12, G12_ML, or G12_GAT")
        print("  output_dir    : (Optional) Where to save results. Default: <base_dir>/combined_force_correlation")
        print("\nExample:")
        print("  python combine_force_correlation_runs.py /path/to/G12 G12")
        print("  python combine_force_correlation_runs.py /path/to/G12 G12 /path/to/output")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    channel_type = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    combine_multiple_runs_force_correlation(base_dir, channel_type, output_dir)


# python3 combine_force_correlation_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 G12 /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/force_correlation

# python3 combine_force_correlation_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML G12_ML /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/force_correlation

# python3 combine_force_correlation_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT G12_GAT /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/force_correlation

# python3 combine_force_correlation_runs.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 G2 /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/force_correlation