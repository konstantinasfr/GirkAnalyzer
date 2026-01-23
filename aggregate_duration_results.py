import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


class DurationResultsAggregator:
    """
    Aggregates duration analysis results from multiple runs.
    Creates bar plots showing frequency of last/longest residues and subunits.
    """
    
    def __init__(self, base_directory, threshold, min_duration):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float
            Threshold value used in the analysis
        min_duration : int
            Minimum duration value used in the analysis
        """
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.min_duration = min_duration
        
        # Convert threshold to string (handle 3.0 -> "3")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)
        
        self.runs_data = {}
        
        # Data for plots: {residue_pdb/subunit: count}
        self.last_residue_counts = defaultdict(int)
        self.longest_residue_counts = defaultdict(int)
        self.last_subunit_counts = defaultdict(int)
        self.longest_subunit_counts = defaultdict(int)
        
    def load_all_runs(self):
        """Load duration analysis results from all subdirectories."""
        print(f"\nSearching for runs in: {self.base_dir}")
        print(f"Looking for threshold={self.threshold_str}, min_duration={self.min_duration}")
        
        # Find all RUN directories
        run_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')]
        
        if not run_dirs:
            print(f"No RUN directories found in {self.base_dir}")
            return False
        
        print(f"Found {len(run_dirs)} RUN directories")
        
        loaded_residue = 0
        loaded_subunit = 0
        
        # Load data from each run
        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            
            # Try both threshold formats: "3" and "3.0"
            threshold_variants = [
                f"{self.threshold_str}_{self.min_duration}",
                f"{self.threshold}_{self.min_duration}"
            ]
            
            # Try to load residue-level results
            residue_file = None
            for variant in threshold_variants:
                test_file = run_dir / "duration_residue" / variant / "last_duration_residue.json"
                if test_file.exists():
                    residue_file = test_file
                    break
            
            if residue_file and residue_file.exists():
                try:
                    with open(residue_file, 'r') as f:
                        residue_data = json.load(f)
                    
                    # Also need longest_duration_residue.json
                    longest_residue_file = residue_file.parent / "longest_duration_residue.json"
                    with open(longest_residue_file, 'r') as f:
                        longest_residue_data = json.load(f)
                    
                    # Aggregate LAST residue data
                    # Count the IDENTIFIER for each event (which residue/subunit won)
                    for event in residue_data['results']:
                        if event['identifier'] and event['residues']:
                            # In residue mode, identifier is the residue_id
                            # Get the PDB numbering from the residue list
                            res_pdb = event['residues'][0]['residue_pdb']
                            self.last_residue_counts[res_pdb] += 1
                    
                    # Aggregate LONGEST residue data
                    for event in longest_residue_data['results']:
                        if event['identifier'] and event['residues']:
                            res_pdb = event['residues'][0]['residue_pdb']
                            self.longest_residue_counts[res_pdb] += 1
                    
                    loaded_residue += 1
                    print(f"  ✓ {run_name}: Loaded residue-level ({len(residue_data['results'])} events)")
                    
                except Exception as e:
                    print(f"  ✗ {run_name}: Error loading residue-level: {e}")
            
            # Try to load subunit-level results
            subunit_file = None
            for variant in threshold_variants:
                test_file = run_dir / "duration_subunit" / variant / "last_duration_subunit.json"
                if test_file.exists():
                    subunit_file = test_file
                    break
            
            if subunit_file and subunit_file.exists():
                try:
                    with open(subunit_file, 'r') as f:
                        subunit_data = json.load(f)
                    
                    # Also need longest_duration_subunit.json
                    longest_subunit_file = subunit_file.parent / "longest_duration_subunit.json"
                    with open(longest_subunit_file, 'r') as f:
                        longest_subunit_data = json.load(f)
                    
                    # Aggregate LAST subunit data
                    # Count the IDENTIFIER for each event (which subunit won)
                    for event in subunit_data['results']:
                        if event['identifier']:
                            subunit = event['identifier']
                            self.last_subunit_counts[subunit] += 1
                    
                    # Aggregate LONGEST subunit data
                    for event in longest_subunit_data['results']:
                        if event['identifier']:
                            subunit = event['identifier']
                            self.longest_subunit_counts[subunit] += 1
                    
                    loaded_subunit += 1
                    print(f"  ✓ {run_name}: Loaded subunit-level ({len(subunit_data['results'])} events)")
                    
                except Exception as e:
                    print(f"  ✗ {run_name}: Error loading subunit-level: {e}")
        
        print(f"\nSuccessfully loaded:")
        print(f"  Residue-level: {loaded_residue} runs")
        print(f"  Subunit-level: {loaded_subunit} runs")
        
        return loaded_residue > 0 or loaded_subunit > 0
    
    def create_bar_plot(self, counts_dict, title, ylabel, output_file, is_subunit=False):
        """Create a simple bar plot for frequency counts."""
        if not counts_dict:
            print(f"No data for: {title}")
            return
        
        # Sort by count (descending)
        sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code
        if is_subunit:
            colors = ['steelblue', 'coral', 'lightgreen', 'plum'][:len(labels)]
        else:
            colors = []
            for label in labels:
                if '152' in label or '141' in label:  # GLU
                    colors.append('salmon')
                elif '184' in label:  # ASN
                    colors.append('lightblue')
                elif '173' in label:  # ASP
                    colors.append('lightgreen')
                else:
                    colors.append('gray')
        
        bars = ax.bar(range(len(labels)), counts, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        
        # Add count labels on top of bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xlabel('Subunit' if is_subunit else 'Residue (PDB)', fontsize=14, fontweight='bold')
        ax.set_title(f'{title}\n(Threshold={self.threshold}Å, Min Duration={self.min_duration} frames)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend for residue mode
        if not is_subunit:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='salmon', edgecolor='black', label='GLU'),
                Patch(facecolor='lightblue', edgecolor='black', label='ASN'),
                Patch(facecolor='lightgreen', edgecolor='black', label='ASP')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved: {output_file}")
    
    def create_all_plots(self, output_dir=None):
        """Create all 4 bar plots."""
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating bar plots...")
        
        # Plot 1: LAST - Residue level
        if self.last_residue_counts:
            self.create_bar_plot(
                self.last_residue_counts,
                "LAST Residue (Most Recent Before Permeation)",
                "Frequency (Number of Events)",
                output_dir / f"last_residue_frequency_t{self.threshold_str}_d{self.min_duration}.png",
                is_subunit=False
            )
        
        # Plot 2: LONGEST - Residue level
        if self.longest_residue_counts:
            self.create_bar_plot(
                self.longest_residue_counts,
                "LONGEST Residue (Longest Duration Before Permeation)",
                "Frequency (Number of Events)",
                output_dir / f"longest_residue_frequency_t{self.threshold_str}_d{self.min_duration}.png",
                is_subunit=False
            )
        
        # Plot 3: LAST - Subunit level
        if self.last_subunit_counts:
            self.create_bar_plot(
                self.last_subunit_counts,
                "LAST Subunit (Most Recent Before Permeation)",
                "Frequency (Number of Events)",
                output_dir / f"last_subunit_frequency_t{self.threshold_str}_d{self.min_duration}.png",
                is_subunit=True
            )
        
        # Plot 4: LONGEST - Subunit level
        if self.longest_subunit_counts:
            self.create_bar_plot(
                self.longest_subunit_counts,
                "LONGEST Subunit (Longest Duration Before Permeation)",
                "Frequency (Number of Events)",
                output_dir / f"longest_subunit_frequency_t{self.threshold_str}_d{self.min_duration}.png",
                is_subunit=True
            )
    
    def create_summary_table(self, output_dir=None):
        """Create summary tables for all 4 analyses."""
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Table 1: Residue-level summary
        if self.last_residue_counts or self.longest_residue_counts:
            all_residues = set(self.last_residue_counts.keys()) | set(self.longest_residue_counts.keys())
            
            residue_table = []
            for residue in sorted(all_residues):
                # Determine type
                if '152' in residue or '141' in residue:
                    res_type = 'GLU'
                elif '184' in residue:
                    res_type = 'ASN'
                elif '173' in residue:
                    res_type = 'ASP'
                else:
                    res_type = 'Unknown'
                
                residue_table.append({
                    'Residue_PDB': residue,
                    'Type': res_type,
                    'Last_Count': self.last_residue_counts.get(residue, 0),
                    'Longest_Count': self.longest_residue_counts.get(residue, 0)
                })
            
            df_residue = pd.DataFrame(residue_table)
            excel_file = output_dir / f"residue_summary_t{self.threshold_str}_d{self.min_duration}.xlsx"
            df_residue.to_excel(excel_file, index=False)
            print(f"\n✓ Residue summary table: {excel_file}")
        
        # Table 2: Subunit-level summary
        if self.last_subunit_counts or self.longest_subunit_counts:
            all_subunits = set(self.last_subunit_counts.keys()) | set(self.longest_subunit_counts.keys())
            
            subunit_table = []
            for subunit in sorted(all_subunits):
                subunit_table.append({
                    'Subunit': subunit,
                    'Last_Count': self.last_subunit_counts.get(subunit, 0),
                    'Longest_Count': self.longest_subunit_counts.get(subunit, 0)
                })
            
            df_subunit = pd.DataFrame(subunit_table)
            excel_file = output_dir / f"subunit_summary_t{self.threshold_str}_d{self.min_duration}.xlsx"
            df_subunit.to_excel(excel_file, index=False)
            print(f"✓ Subunit summary table: {excel_file}")
            
            # Print to console
            print("\n" + "="*60)
            print("RESIDUE-LEVEL SUMMARY")
            print("="*60)
            if self.last_residue_counts or self.longest_residue_counts:
                print(df_residue.to_string(index=False))
            
            print("\n" + "="*60)
            print("SUBUNIT-LEVEL SUMMARY")
            print("="*60)
            if self.last_subunit_counts or self.longest_subunit_counts:
                print(df_subunit.to_string(index=False))
            print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate duration analysis results from multiple runs"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing RUN subdirectories"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold value used in the analysis (e.g., 3.0)"
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        required=True,
        help="Minimum duration value used in the analysis (e.g., 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables (default: base_directory/aggregated_duration)"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = DurationResultsAggregator(args.base_directory, args.threshold, args.min_duration)
    
    if not aggregator.load_all_runs():
        print("No data found. Exiting.")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = aggregator.base_dir / f"aggregated_duration_t{aggregator.threshold_str}_d{args.min_duration}"
    
    # Create plots and tables
    print("\nGenerating outputs...")
    aggregator.create_all_plots(output_dir=output_dir)
    aggregator.create_summary_table(output_dir=output_dir)
    
    print(f"\n✓ Aggregation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


# USAGE EXAMPLE:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5
# 
# Or with custom output directory:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5 --output_dir /path/to/output

# USAGE EXAMPLE:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5
# 
# Or with custom output directory:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5 --output_dir /path/to/output


# USAGE EXAMPLE:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5
# 
# Or with custom output directory:
# python aggregate_duration_results.py /path/to/G12 --threshold 3.0 --min_duration 5 --output_dir /path/to/output

# python3 aggregate_duration_results.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12  --threshold 3.0 --min_duration 5 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_duration_results