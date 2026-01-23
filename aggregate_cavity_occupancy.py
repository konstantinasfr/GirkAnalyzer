import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import Counter


class CavityOccupancyAggregator:
    """
    Aggregates cavity occupancy analysis results from multiple runs.
    Creates plots for subunit combinations, residue frequencies, and ion counts.
    """
    
    def __init__(self, base_directory, threshold=3.0):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float
            Threshold value used in the analysis
        """
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        
        # Convert threshold to string (handle 3.0 -> "3")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)
        
        self.all_results = []
        self.residue_combination_counts = Counter()
        self.subunit_combination_counts = Counter()
        self.residue_appearance_counts = Counter()  # How many times each residue appears
        self.subunit_appearance_counts = Counter()  # How many times each subunit appears
        self.total_combinations = 0
        self.cavity_ion_counts = []
        
    def load_all_runs(self):
        """Load cavity occupancy results from all subdirectories."""
        print(f"\nSearching for runs in: {self.base_dir}")
        
        # Find all RUN directories
        run_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')]
        
        if not run_dirs:
            print(f"No RUN directories found in {self.base_dir}")
            return False
        
        print(f"Found {len(run_dirs)} RUN directories")
        
        loaded_count = 0
        
        # Load data from each run
        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            
            # Look for cavity_occupancy folder
            json_file = run_dir / "cavity_occupancy" / "cavity_occupancy_at_end2.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    results = data['results']
                    self.all_results.extend(results)
                    
                    # Aggregate data
                    for event in results:
                        # Count residue combinations
                        if event['residue_combination']:
                            self.residue_combination_counts[event['residue_combination']] += 1
                            self.total_combinations += 1
                            
                            # Count individual residue appearances
                            for resid in event['residues_with_ions']:
                                self.residue_appearance_counts[resid] += 1
                        
                        # Count subunit combinations
                        if event['subunit_combination']:
                            self.subunit_combination_counts[event['subunit_combination']] += 1
                            
                            # Count individual subunit appearances
                            for subunit in event['subunits_with_ions']:
                                self.subunit_appearance_counts[subunit] += 1
                        
                        # Collect cavity ion counts
                        self.cavity_ion_counts.append(event['total_ions_in_cavity'])
                    
                    loaded_count += 1
                    print(f"  ✓ {run_name}: Loaded ({len(results)} events)")
                    
                except Exception as e:
                    print(f"  ✗ {run_name}: Error loading: {e}")
        
        print(f"\nSuccessfully loaded: {loaded_count} runs")
        print(f"Total events: {len(self.all_results)}")
        
        return loaded_count > 0
    
    def create_subunit_combination_plot(self, output_dir):
        """Create bar plot of subunit combination frequencies."""
        if not self.subunit_combination_counts:
            print("No subunit combination data")
            return
        
        # Get top 15 combinations
        top_combos = self.subunit_combination_counts.most_common(15)
        
        combos = [c[0] for c in top_combos]
        counts = [c[1] for c in top_combos]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Color by number of subunits involved
        colors = []
        for combo in combos:
            n_subunits = len(combo.split('_')) if combo else 0
            if n_subunits == 1:
                colors.append('lightblue')
            elif n_subunits == 2:
                colors.append('lightgreen')
            elif n_subunits == 3:
                colors.append('orange')
            else:
                colors.append('salmon')
        
        bars = ax.bar(range(len(combos)), counts, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('Frequency (Number of Events)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Subunit Combination', fontsize=14, fontweight='bold')
        ax.set_title(f'Subunit Combinations at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='1 subunit'),
            Patch(facecolor='lightgreen', edgecolor='black', label='2 subunits'),
            Patch(facecolor='orange', edgecolor='black', label='3 subunits'),
            Patch(facecolor='salmon', edgecolor='black', label='4 subunits')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        output_file = output_dir / f"subunit_combinations_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Subunit combinations plot saved: {output_file}")
    
    def create_residue_percentage_plot(self, output_dir):
        """Create bar plot showing percentage of times each residue appears."""
        if not self.residue_appearance_counts or self.total_combinations == 0:
            print("No residue appearance data")
            return
        
        # Calculate percentages
        residue_percentages = {}
        for residue, count in self.residue_appearance_counts.items():
            residue_percentages[residue] = (count / self.total_combinations) * 100
        
        # Sort by percentage (descending)
        sorted_residues = sorted(residue_percentages.items(), key=lambda x: x[1], reverse=True)
        
        residues = [r[0] for r in sorted_residues]
        percentages = [r[1] for r in sorted_residues]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Color by residue type
        colors = []
        for res in residues:
            if '152' in res or '141' in res:  # GLU
                colors.append('salmon')
            elif '184' in res:  # ASN
                colors.append('lightblue')
            elif '173' in res:  # ASP
                colors.append('lightgreen')
            else:
                colors.append('gray')
        
        bars = ax.bar(range(len(residues)), percentages, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Reference lines
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label='100% (all events)')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='50% (half of events)')
        
        ax.set_xticks(range(len(residues)))
        ax.set_xticklabels(residues, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('Percentage of Events (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Residue (PDB)', fontsize=14, fontweight='bold')
        ax.set_title(f'Residue Appearance Frequency at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='salmon', edgecolor='black', label='GLU'),
            Patch(facecolor='lightblue', edgecolor='black', label='ASN'),
            Patch(facecolor='lightgreen', edgecolor='black', label='ASP'),
            plt.Line2D([0], [0], color='green', linestyle='--', label='100%'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='50%')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        output_file = output_dir / f"residue_appearance_percentage_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Residue percentage plot saved: {output_file}")
    
    def create_subunit_percentage_plot(self, output_dir):
        """Create bar plot showing percentage of times each subunit appears."""
        if not self.subunit_appearance_counts or self.total_combinations == 0:
            print("No subunit appearance data")
            return
        
        # Calculate percentages
        subunit_percentages = {}
        for subunit, count in self.subunit_appearance_counts.items():
            subunit_percentages[subunit] = (count / self.total_combinations) * 100
        
        # Sort alphabetically
        sorted_subunits = sorted(subunit_percentages.items())
        
        subunits = [s[0] for s in sorted_subunits]
        percentages = [s[1] for s in sorted_subunits]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Different colors for each subunit
        colors = ['steelblue', 'coral', 'lightgreen', 'plum'][:len(subunits)]
        
        bars = ax.bar(range(len(subunits)), percentages, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.6)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Reference lines
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label='100% (all events)')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='50% (half of events)')
        
        ax.set_xticks(range(len(subunits)))
        ax.set_xticklabels(subunits, fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage of Events (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Subunit', fontsize=14, fontweight='bold')
        ax.set_title(f'Subunit Appearance Frequency at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        output_file = output_dir / f"subunit_appearance_percentage_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Subunit percentage plot saved: {output_file}")
    
    def create_cavity_ion_count_plot(self, output_dir):
        """Create bar plot showing distribution of ion counts in cavity."""
        if not self.cavity_ion_counts:
            print("No cavity ion count data")
            return
        
        # Count frequency of each ion count
        ion_count_freq = Counter(self.cavity_ion_counts)
        
        # Sort by ion count
        sorted_counts = sorted(ion_count_freq.items())
        
        ion_counts = [c[0] for c in sorted_counts]
        frequencies = [c[1] for c in sorted_counts]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        bars = ax.bar(ion_counts, frequencies, color='steelblue', alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.8)
        
        # Add frequency labels
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{freq}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add mean line
        mean_ions = np.mean(self.cavity_ion_counts)
        ax.axvline(x=mean_ions, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_ions:.1f}')
        
        ax.set_xlabel('Number of Ions in Cavity (Channel 2)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (Number of Events)', fontsize=14, fontweight='bold')
        ax.set_title(f'Ion Count Distribution in Cavity at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=12)
        
        # Add statistics text
        stats_text = f'Mean: {mean_ions:.1f}\nStd: {np.std(self.cavity_ion_counts):.1f}\nRange: {min(self.cavity_ion_counts)}-{max(self.cavity_ion_counts)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        output_file = output_dir / f"cavity_ion_counts_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cavity ion count plot saved: {output_file}")
    
    def create_all_plots(self, output_dir=None):
        """Create all plots."""
        if output_dir is None:
            output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating plots...")
        
        self.create_subunit_combination_plot(output_dir)
        self.create_residue_percentage_plot(output_dir)
        self.create_subunit_percentage_plot(output_dir)  # NEW: Subunit percentages
        self.create_cavity_ion_count_plot(output_dir)
    
    def create_summary_table(self, output_dir=None):
        """Create summary tables."""
        if output_dir is None:
            output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Table 1: Residue appearance percentages
        residue_table = []
        for residue, count in sorted(self.residue_appearance_counts.items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / self.total_combinations) * 100 if self.total_combinations > 0 else 0
            
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
                'Appearances': count,
                'Percentage': percentage
            })
        
        df_residue = pd.DataFrame(residue_table)
        excel_file = output_dir / f"residue_appearances_t{self.threshold_str}.xlsx"
        df_residue.to_excel(excel_file, index=False, float_format='%.1f')
        print(f"\n✓ Residue appearances table: {excel_file}")
        
        # Also save as CSV
        csv_file = output_dir / f"residue_appearances_t{self.threshold_str}.csv"
        df_residue.to_csv(csv_file, index=False, float_format='%.1f')
        print(f"✓ Residue appearances CSV: {csv_file}")
        
        # Table 2: Subunit combinations
        subunit_table = []
        for combo, count in self.subunit_combination_counts.most_common():
            subunit_table.append({
                'Subunit_Combination': combo,
                'Frequency': count,
                'Percentage': (count / len(self.all_results)) * 100 if self.all_results else 0
            })
        
        df_subunit = pd.DataFrame(subunit_table)
        excel_file = output_dir / f"subunit_combinations_t{self.threshold_str}.xlsx"
        df_subunit.to_excel(excel_file, index=False, float_format='%.1f')
        print(f"✓ Subunit combinations table: {excel_file}")
        
        # Also save as CSV
        csv_file = output_dir / f"subunit_combinations_t{self.threshold_str}.csv"
        df_subunit.to_csv(csv_file, index=False, float_format='%.1f')
        print(f"✓ Subunit combinations CSV: {csv_file}")
        
        # Also save as CSV
        csv_file = output_dir / f"subunit_combinations_t{self.threshold_str}.csv"
        df_subunit.to_csv(csv_file, index=False, float_format='%.1f')
        print(f"✓ Subunit combinations CSV: {csv_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("RESIDUE APPEARANCE PERCENTAGES")
        print("="*80)
        print(df_residue.to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP SUBUNIT COMBINATIONS")
        print("="*80)
        print(df_subunit.head(20).to_string(index=False))
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cavity occupancy analysis from multiple runs"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing RUN subdirectories"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Threshold value used in the analysis (default: 3.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables (default: base_directory/aggregated_cavity)"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = CavityOccupancyAggregator(args.base_directory, args.threshold)
    
    if not aggregator.load_all_runs():
        print("No data found. Exiting.")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None
    
    # Create plots and tables
    print("\nGenerating outputs...")
    aggregator.create_all_plots(output_dir=output_dir)
    aggregator.create_summary_table(output_dir=output_dir)
    
    print(f"\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()


# USAGE:
# python aggregate_cavity_occupancy.py /path/to/G12 --threshold 3.0
# 
# Or with custom output:
# python aggregate_cavity_occupancy.py /path/to/G12 --threshold 3.0 --output_dir /path/to/output

# USAGE:
# python aggregate_cavity_occupancy.py /path/to/G12 --threshold 3.0
# 
# Or with custom output:
# python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --threshold 3.0 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/cavity_occupancy