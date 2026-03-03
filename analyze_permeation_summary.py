#!/usr/bin/env python3
"""
Permeation Event Summary Analyzer

Analyzes permeation events across multiple simulation runs and creates summary statistics.

Usage:
    python analyze_permeation_summary.py /path/to/results/G12_GAT
    python analyze_permeation_summary.py /path/to/results/G12_GAT --runs 1 2 3 4 5
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path


def find_permeation_files(base_dir, run_numbers=None):
    """
    Find all permeation_table.json files in RUN directories.
    
    Parameters
    ----------
    base_dir : Path or str
        Base directory containing RUN1, RUN2, etc.
    run_numbers : list of int, optional
        Specific run numbers to process
    
    Returns
    -------
    dict : {run_name: permeation_file_path}
    """
    base_dir = Path(base_dir)
    permeation_files = {}
    
    if run_numbers is None:
        # Find all RUN* directories
        run_dirs = sorted(base_dir.glob("RUN*"))
        print(f"Found {len(run_dirs)} RUN directories")
    else:
        # Use specific run numbers
        run_dirs = [base_dir / f"RUN{num}" for num in run_numbers]
        print(f"Looking for {len(run_dirs)} specified RUN directories")
    
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"  [SKIP] {run_dir.name} - directory not found")
            continue
        
        # Look for permeation_table.json
        perm_file = run_dir / "permeation_table.json"
        
        if perm_file.exists():
            permeation_files[run_dir.name] = perm_file
            print(f"  [FOUND] {run_dir.name}/permeation_table.json")
        else:
            print(f"  [SKIP] {run_dir.name} - no permeation_table.json")
    
    return permeation_files


def get_trajectory_frames(base_dir, run_name):
    """
    Try to determine the number of frames in a trajectory.
    Looks for saved state or tries common trajectory file names.
    """
    run_dir = base_dir / run_name
    
    # Try to read from saved analyzer state
    try:
        import pickle
        state_file = run_dir / "analyzer_state.pkl"
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                if 'end_frame' in state:
                    return state['end_frame'] + 1  # end_frame is 0-indexed
    except:
        pass
    
    # Try to read trajectory file directly
    try:
        import MDAnalysis as mda
        # Try common trajectory file names
        for traj_name in ['protein.nc', 'trajectory.nc', 'traj.nc', 'prod.nc']:
            traj_file = run_dir / traj_name
            if traj_file.exists():
                # Need topology too
                for top_name in ['com.prmtop', 'topology.prmtop', 'system.psf']:
                    top_file = run_dir / top_name
                    if top_file.exists():
                        u = mda.Universe(str(top_file), str(traj_file))
                        return len(u.trajectory)
    except:
        pass
    
    return None


def analyze_all_runs(permeation_files, base_dir):
    """
    Analyze permeation events from all runs.
    
    Returns
    -------
    pd.DataFrame : Summary statistics
    dict : Detailed per-run data
    """
    print("\n" + "="*70)
    print("ANALYZING PERMEATION EVENTS")
    print("="*70)
    
    run_data = []
    
    for run_name, perm_file in sorted(permeation_files.items()):
        print(f"\nProcessing {run_name}...")
        
        # Load permeation events
        with open(perm_file, 'r') as f:
            events = json.load(f)
        
        # Get number of frames
        n_frames = get_trajectory_frames(base_dir, run_name)
        
        if not events:
            print(f"  No permeation events found")
            run_data.append({
                'Run': run_name,
                'Total_Frames': n_frames,
                'HBC_Permeations': 0,
                'Final_Permeations': 0,
                'HBC_Per_Frame': 0.0,
                'Final_Per_Frame': 0.0
            })
            continue
        
        # Count HBC permeations (end_3) and Final permeations (end_4)
        hbc_permeations = len(events)  # All events reach at least end_3
        final_permeations = len(events)  # All events reach end_4 in your analysis
        
        # Calculate per-frame rates
        hbc_per_frame = hbc_permeations / n_frames if n_frames else 0
        final_per_frame = final_permeations / n_frames if n_frames else 0
        
        run_data.append({
            'Run': run_name,
            'Total_Frames': n_frames,
            'HBC_Permeations': hbc_permeations,
            'Final_Permeations': final_permeations,
            'HBC_Per_Frame': hbc_per_frame,
            'Final_Per_Frame': final_per_frame
        })
        
        print(f"  Frames: {n_frames}")
        print(f"  HBC permeations: {hbc_permeations}")
        print(f"  Final permeations: {final_permeations}")
        if n_frames:
            print(f"  HBC per frame: {hbc_per_frame:.6f}")
            print(f"  Final per frame: {final_per_frame:.6f}")
    
    # Create DataFrame
    df = pd.DataFrame(run_data)
    
    return df


def create_summary_table(df, output_dir):
    """
    Create summary statistics table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Per-run data
    output_dir : Path
        Where to save output
    """
    print("\n" + "="*70)
    print("CREATING SUMMARY TABLE")
    print("="*70)
    
    # Calculate summary statistics
    n_simulations = len(df)
    
    # Handle cases where Total_Frames might be None
    valid_frames = df[df['Total_Frames'].notna()]['Total_Frames']
    mean_frames = valid_frames.mean() if len(valid_frames) > 0 else 0
    
    total_hbc = df['HBC_Permeations'].sum()
    total_final = df['Final_Permeations'].sum()
    
    # Calculate mean per-frame rates (only from runs with known frame counts)
    valid_runs = df[df['Total_Frames'].notna()]
    mean_hbc_per_frame = valid_runs['HBC_Per_Frame'].mean() if len(valid_runs) > 0 else 0
    mean_final_per_frame = valid_runs['Final_Per_Frame'].mean() if len(valid_runs) > 0 else 0
    
    # Create summary dictionary
    summary = {
        'Number_of_Simulations': n_simulations,
        'Mean_Frames_per_Simulation': mean_frames,
        'Total_HBC_Permeations': total_hbc,
        'Total_Final_Permeations': total_final,
        'Mean_HBC_Permeations_per_Frame': mean_hbc_per_frame,
        'Mean_Final_Permeations_per_Frame': mean_final_per_frame
    }
    
    # Print summary
    print("\nSUMMARY STATISTICS:")
    print("-"*70)
    print(f"Number of simulations: {n_simulations}")
    print(f"Mean frames per simulation: {mean_frames:.0f}")
    print(f"Total HBC permeations: {total_hbc}")
    print(f"Total final permeations: {total_final}")
    print(f"Mean HBC permeations per frame: {mean_hbc_per_frame:.6f}")
    print(f"Mean final permeations per frame: {mean_final_per_frame:.6f}")
    
    # Save summary table
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / 'permeation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Summary saved to: {summary_file}")
    
    # Save per-run details
    detail_file = output_dir / 'permeation_per_run.csv'
    df.to_csv(detail_file, index=False)
    print(f"✅ Per-run details saved to: {detail_file}")
    
    # Save to Excel file with multiple sheets
    excel_file = output_dir / 'permeation_analysis.xlsx'
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Summary statistics
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Per-run details
            df.to_excel(writer, sheet_name='Per_Run_Details', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Excel file saved to: {excel_file}")
    except ImportError:
        print(f"⚠ WARNING: openpyxl not installed, Excel file not created")
        print(f"  Install with: pip install openpyxl")
    except Exception as e:
        print(f"⚠ WARNING: Could not create Excel file: {e}")
    
    # Create formatted text report
    report_file = output_dir / 'permeation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERMEATION EVENT ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of simulations:              {n_simulations}\n")
        f.write(f"Mean frames per simulation:         {mean_frames:.0f}\n")
        f.write(f"Total HBC permeations:              {total_hbc}\n")
        f.write(f"Total final permeations:            {total_final}\n")
        f.write(f"Mean HBC permeations per frame:     {mean_hbc_per_frame:.6f}\n")
        f.write(f"Mean final permeations per frame:   {mean_final_per_frame:.6f}\n")
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("PER-RUN DETAILS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Run':<10} {'Frames':<10} {'HBC':<8} {'Final':<8} {'HBC/Frame':<12} {'Final/Frame':<12}\n")
        f.write("-"*70 + "\n")
        
        for _, row in df.iterrows():
            frames_str = f"{int(row['Total_Frames'])}" if pd.notna(row['Total_Frames']) else "Unknown"
            f.write(f"{row['Run']:<10} {frames_str:<10} "
                   f"{row['HBC_Permeations']:<8} {row['Final_Permeations']:<8} "
                   f"{row['HBC_Per_Frame']:<12.6f} {row['Final_Per_Frame']:<12.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("\nNOTE:\n")
        f.write("- HBC Permeations: Events that passed through HBC gate (end_3)\n")
        f.write("- Final Permeations: Events that completed full permeation (end_4)\n")
        f.write("- Per-frame rates calculated as: (permeations / total_frames)\n")
    
    print(f"✅ Text report saved to: {report_file}")
    
    return summary_df, df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze permeation events across multiple simulation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all RUN* directories
  %(prog)s /path/to/girk_analyser_results/G12_GAT
  
  # Analyze specific runs only
  %(prog)s /path/to/girk_analyser_results/G12_GAT --runs 1 2 3 4 5
  
  # Specify output directory
  %(prog)s /path/to/girk_analyser_results/G12_GAT -o permeation_summary
        """
    )
    
    parser.add_argument('base_dir', 
                       help='Base directory containing RUN1, RUN2, etc.')
    parser.add_argument('--runs', nargs='+', type=int,
                       help='Specific run numbers to analyze (e.g., --runs 1 2 3 4 5)')
    parser.add_argument('-o', '--output', default='permeation_summary',
                       help='Output directory name (default: permeation_summary)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return 1
    
    output_dir = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PERMEATION EVENT SUMMARY ANALYSIS")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    if args.runs:
        print(f"Specific runs: {args.runs}")
    else:
        print("Processing: All RUN* directories found")
    print("="*70)
    
    # Find permeation files
    permeation_files = find_permeation_files(base_dir, args.runs)
    
    if not permeation_files:
        print("\nERROR: No permeation_table.json files found!")
        print("\nMake sure:")
        print("  1. The base directory is correct")
        print("  2. RUN directories contain permeation_table.json")
        print("  3. You've run the ion permeation analysis on each trajectory")
        return 1
    
    # Analyze all runs
    df = analyze_all_runs(permeation_files, base_dir)
    
    # Create summary table
    summary_df, detail_df = create_summary_table(df, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("  • permeation_analysis.xlsx - Excel file with Summary and Per_Run_Details sheets")
    print("  • permeation_summary.csv - Summary statistics")
    print("  • permeation_per_run.csv - Detailed per-run data")
    print("  • permeation_report.txt - Formatted text report")
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())


# python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/permeation_summary 