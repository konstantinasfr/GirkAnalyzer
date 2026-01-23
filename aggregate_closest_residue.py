import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


class ClosestResidueAggregator:
    """
    Aggregates closest residue analysis results from multiple runs.
    Creates bar plots and summary tables.
    """
    
    def __init__(self, base_directory, threshold):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float or int
            Threshold value used in the analysis
        """
        self.base_dir = Path(base_directory)
        self.runs_data = {}
        self.threshold = threshold
        
        # Data structure: {case_name: {residue_pdb: {'count': N, 'distances': []}}}
        self.case_1_data = defaultdict(lambda: {'count': 0, 'distances': []})
        self.case_2_data = defaultdict(lambda: {'count': 0, 'distances': []})
        self.case_3_data = defaultdict(lambda: {'count': 0, 'distances': []})
        self.case_4_data = defaultdict(lambda: {'count': 0, 'distances': []})
        
    def load_all_runs(self):
        """
        Load closest residue analysis results from all subdirectories.
        """
        print(f"\nSearching for runs in: {self.base_dir}")
        print(f"Looking for threshold: {self.threshold}")
        
        # Convert threshold to string, removing unnecessary decimals
        # So 3.0 becomes "3" and 3.5 stays "3.5"
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            threshold_str = str(int(self.threshold))
        else:
            threshold_str = str(self.threshold)
        
        print(f"Looking for directory: closest_res/{threshold_str}/")
        
        # Find all subdirectories that contain closest_residue_analysis.json
        run_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                json_file = item / "closest_res" / threshold_str / "closest_residue_analysis.json"
                if json_file.exists():
                    run_dirs.append(item)
        
        if not run_dirs:
            print(f"No runs found with closest_residue_analysis.json in {self.base_dir}/*/closest_res/{threshold_str}/")
            return False
        
        print(f"Found {len(run_dirs)} runs:")
        
        # Load data from each run
        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            json_file = run_dir / "closest_res" / threshold_str / "closest_residue_analysis.json"
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.runs_data[run_name] = data
                print(f"  ✓ Loaded: {run_name} ({len(data['results'])} events)")
                
                # Aggregate data by case
                for event in data['results']:
                    # Case 1
                    if event['case_1_end_2_closest']:
                        c1 = event['case_1_end_2_closest']
                        res_pdb = c1['closest_residue_pdb']
                        self.case_1_data[res_pdb]['count'] += 1
                        self.case_1_data[res_pdb]['distances'].append(c1['distance'])
                    
                    # Case 2
                    if event['case_2_end_2_threshold']:
                        c2 = event['case_2_end_2_threshold']
                        res_pdb = c2['closest_residue_pdb']
                        self.case_2_data[res_pdb]['count'] += 1
                        self.case_2_data[res_pdb]['distances'].append(c2['distance'])
                    
                    # Case 3
                    if event['case_3_end_3_closest']:
                        c3 = event['case_3_end_3_closest']
                        res_pdb = c3['closest_residue_pdb']
                        self.case_3_data[res_pdb]['count'] += 1
                        self.case_3_data[res_pdb]['distances'].append(c3['distance'])
                    
                    # Case 4
                    if event['case_4_end_3_threshold']:
                        c4 = event['case_4_end_3_threshold']
                        res_pdb = c4['closest_residue_pdb']
                        self.case_4_data[res_pdb]['count'] += 1
                        self.case_4_data[res_pdb]['distances'].append(c4['distance'])
                
            except Exception as e:
                print(f"  ✗ Error loading {run_name}: {e}")
        
        return len(self.runs_data) > 0
    
    def create_bar_plot(self, case_data, case_name, output_file):
        """
        Create a bar plot for a single case showing winner counts.
        
        Parameters:
        -----------
        case_data : dict
            Dictionary with residue PDB as key and data as value
        case_name : str
            Name of the case (e.g., "Case 1")
        output_file : Path
            Output file path
        """
        if not case_data:
            print(f"No data for {case_name}")
            return
        
        # Sort residues by count (descending)
        sorted_residues = sorted(case_data.items(), key=lambda x: x[1]['count'], reverse=True)
        
        residues = [r[0] for r in sorted_residues]
        counts = [r[1]['count'] for r in sorted_residues]
        avg_distances = [np.mean(r[1]['distances']) for r in sorted_residues]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code by residue type
        colors = []
        for res in residues:
            if '152' in res or '141' in res:  # GLU residues
                colors.append('salmon')
            elif '184' in res:  # ASN residues
                colors.append('lightblue')
            elif '173' in res:  # ASP residue
                colors.append('lightgreen')
            else:
                colors.append('gray')
        
        bars = ax.bar(range(len(residues)), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add count labels on top of bars
        for i, (bar, count, avg_dist) in enumerate(zip(bars, counts, avg_distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({avg_dist:.2f} Å)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xticks(range(len(residues)))
        ax.set_xticklabels(residues, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('Number of Times Closest (Winner)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Residue (PDB Numbering)', fontsize=14, fontweight='bold')
        ax.set_title(f'{case_name} - Closest Residue Frequency (Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
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
        """
        Create bar plots for all 4 cases.
        """
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating bar plots...")
        
        self.create_bar_plot(self.case_1_data, "Case 1: Closest at end_2 (no threshold)", 
                            output_dir / f"case1_closest_residue_bar_threshold{self.threshold}.png")
        
        self.create_bar_plot(self.case_2_data, "Case 2: Threshold crossing from end_2", 
                            output_dir / f"case2_closest_residue_bar_threshold{self.threshold}.png")
        
        self.create_bar_plot(self.case_3_data, "Case 3: Closest at end_3 (no threshold)", 
                            output_dir / f"case3_closest_residue_bar_threshold{self.threshold}.png")
        
        self.create_bar_plot(self.case_4_data, "Case 4: Threshold crossing from end_3", 
                            output_dir / f"case4_closest_residue_bar_threshold{self.threshold}.png")
    
    def create_summary_table(self, output_dir=None):
        """
        Create a summary table showing counts and average distances for each residue.
        """
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all unique residues across all cases
        all_residues = set()
        all_residues.update(self.case_1_data.keys())
        all_residues.update(self.case_2_data.keys())
        all_residues.update(self.case_3_data.keys())
        all_residues.update(self.case_4_data.keys())
        
        # Sort residues
        sorted_residues = sorted(all_residues)
        
        # Build table data
        table_data = []
        for residue in sorted_residues:
            # Determine residue type
            if '152' in residue or '141' in residue:
                res_type = 'GLU'
            elif '184' in residue:
                res_type = 'ASN'
            elif '173' in residue:
                res_type = 'ASP'
            else:
                res_type = 'Unknown'
            
            row = {
                'Residue_PDB': residue,
                'Type': res_type,
                'Case1_Count': self.case_1_data[residue]['count'] if residue in self.case_1_data else 0,
                'Case1_Avg_Distance': np.mean(self.case_1_data[residue]['distances']) if residue in self.case_1_data and self.case_1_data[residue]['distances'] else np.nan,
                'Case2_Count': self.case_2_data[residue]['count'] if residue in self.case_2_data else 0,
                'Case2_Avg_Distance': np.mean(self.case_2_data[residue]['distances']) if residue in self.case_2_data and self.case_2_data[residue]['distances'] else np.nan,
                'Case3_Count': self.case_3_data[residue]['count'] if residue in self.case_3_data else 0,
                'Case3_Avg_Distance': np.mean(self.case_3_data[residue]['distances']) if residue in self.case_3_data and self.case_3_data[residue]['distances'] else np.nan,
                'Case4_Count': self.case_4_data[residue]['count'] if residue in self.case_4_data else 0,
                'Case4_Avg_Distance': np.mean(self.case_4_data[residue]['distances']) if residue in self.case_4_data and self.case_4_data[residue]['distances'] else np.nan,
            }
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save to Excel
        excel_file = output_dir / f"closest_residue_summary_table_threshold{self.threshold}.xlsx"
        df.to_excel(excel_file, index=False, float_format='%.2f')
        print(f"\n✓ Summary table saved: {excel_file}")
        
        # Save to CSV as well
        csv_file = output_dir / f"closest_residue_summary_table_threshold{self.threshold}.csv"
        df.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"✓ Summary table saved: {csv_file}")
        
        # Print table to console
        print("\n" + "="*100)
        print(f"SUMMARY TABLE (Threshold={self.threshold}Å)")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate closest residue analysis from multiple runs"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing run subdirectories (e.g., RUN1, RUN2, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables (default: same as base_directory)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="Threshold value used in the analysis (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Create aggregator and load data
    aggregator = ClosestResidueAggregator(args.base_directory, args.threshold)
    
    if not aggregator.load_all_runs():
        print("No runs found. Exiting.")
        return
    
    # Create plots and tables
    print("\nGenerating outputs...")
    aggregator.create_all_plots(output_dir=args.output_dir)
    aggregator.create_summary_table(output_dir=args.output_dir)
    
    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()


# python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/aggregate_closest_residue/3 --threshold 3