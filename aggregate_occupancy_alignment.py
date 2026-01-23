import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import Counter


class ComprehensiveEnd2Aggregator:
    """
    Aggregates comprehensive end2 analysis results from multiple runs.
    Creates plots for combinations, free ions, cavity occupancy, and SF alignment.
    """
    
    def __init__(self, base_directory, threshold=2.5):
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
        
        # Convert threshold to string (handle 2.5 -> "2.5")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)
        
        self.all_results = []
        self.residue_combination_counts = Counter()
        self.subunit_combination_counts = Counter()
        self.free_ion_counts = Counter()  # Count events by number of free ions
        self.cavity_ion_counts = []  # Including permeating ion
        
        # SF alignment data
        self.permeating_sf = []
        self.bound_sf = []
        self.free_sf = []
        
    def load_all_runs(self):
        """Load comprehensive end2 results from all subdirectories."""
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
            
            # Look for comprehensive_end2 results
            # Try both with and without threshold subfolder
            possible_paths = [
                run_dir / "comprehensive_end2" / "comprehensive_end2_analysis.json",
                run_dir / "occupancy_alignment" / self.threshold_str / "comprehensive_end2_analysis.json",
                run_dir / f"comprehensive_end2_{self.threshold_str}" / "comprehensive_end2_analysis.json"
            ]
            
            json_file = None
            for path in possible_paths:
                if path.exists():
                    json_file = path
                    break
            
            if json_file:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    results = data['results']
                    self.all_results.extend(results)
                    
                    # Aggregate data
                    for event in results:
                        # Count combinations
                        if event['residue_combination']:
                            self.residue_combination_counts[event['residue_combination']] += 1
                        if event['subunit_combination']:
                            self.subunit_combination_counts[event['subunit_combination']] += 1
                        
                        # Count free ions
                        n_free = event['n_free_ions']
                        self.free_ion_counts[n_free] += 1
                        
                        # Cavity ion count (add 1 for permeating ion)
                        total_with_permeating = event['total_ions_in_cavity'] + 1
                        self.cavity_ion_counts.append(total_with_permeating)
                        
                        # SF alignment data
                        if event['permeating_ion_sf_distance'] is not None:
                            self.permeating_sf.append(event['permeating_ion_sf_distance'])
                        
                        for bound_ion in event['bound_ions_sf_alignment']:
                            self.bound_sf.append(bound_ion['sf_distance'])
                        
                        for free_ion in event['free_ions_sf_alignment']:
                            self.free_sf.append(free_ion['sf_distance'])
                    
                    loaded_count += 1
                    print(f"  ✓ {run_name}: Loaded ({len(results)} events)")
                    
                except Exception as e:
                    print(f"  ✗ {run_name}: Error loading: {e}")
        
        print(f"\nSuccessfully loaded: {loaded_count} runs")
        print(f"Total events: {len(self.all_results)}")
        
        return loaded_count > 0
    
    def create_combination_csvs(self, output_dir):
        """Create CSV files with all combinations and frequencies."""
        # Residue combinations CSV
        residue_data = []
        for combo, count in self.residue_combination_counts.most_common():
            percentage = (count / len(self.all_results)) * 100 if self.all_results else 0
            residue_data.append({
                'Residue_Combination': combo,
                'Frequency': count,
                'Percentage': percentage
            })
        
        df_residue = pd.DataFrame(residue_data)
        csv_file = output_dir / f"residue_combinations_t{self.threshold_str}.csv"
        df_residue.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"✓ Residue combinations CSV: {csv_file}")
        
        # Subunit combinations CSV
        subunit_data = []
        for combo, count in self.subunit_combination_counts.most_common():
            percentage = (count / len(self.all_results)) * 100 if self.all_results else 0
            subunit_data.append({
                'Subunit_Combination': combo,
                'Frequency': count,
                'Percentage': percentage
            })
        
        df_subunit = pd.DataFrame(subunit_data)
        csv_file = output_dir / f"subunit_combinations_t{self.threshold_str}.csv"
        df_subunit.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"✓ Subunit combinations CSV: {csv_file}")
        
        return df_residue, df_subunit
    
    def create_top_combinations_plots(self, output_dir):
        """Create bar plots for top 10 residue and subunit combinations."""
        # Plot 1: Top 10 residue combinations
        if self.residue_combination_counts:
            top_res = self.residue_combination_counts.most_common(10)
            
            combos = [c[0] for c in top_res]
            counts = [c[1] for c in top_res]
            
            fig, ax = plt.subplots(figsize=(14, 7))
            bars = ax.bar(range(len(combos)), counts, color='steelblue', alpha=0.7,
                         edgecolor='black', linewidth=1.5)
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                pct = (count / len(self.all_results)) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{count}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(combos)))
            ax.set_xticklabels(combos, fontsize=9, fontweight='bold', rotation=45, ha='right')
            ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
            ax.set_xlabel('Residue Combination', fontsize=14, fontweight='bold')
            ax.set_title(f'Top 10 Residue Combinations at end_2\n(Threshold={self.threshold}Å)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"top10_residue_combinations_t{self.threshold_str}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Top 10 residue combinations plot saved")
        
        # Plot 2: Top 10 subunit combinations
        if self.subunit_combination_counts:
            top_sub = self.subunit_combination_counts.most_common(10)
            
            combos = [c[0] for c in top_sub]
            counts = [c[1] for c in top_sub]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Color by number of subunits
            colors = []
            for combo in combos:
                n_parts = len([p for p in combo.split('_') if not p.isdigit()])
                if 'FR' in combo:
                    n_parts -= 1
                if n_parts == 1:
                    colors.append('lightblue')
                elif n_parts == 2:
                    colors.append('lightgreen')
                elif n_parts == 3:
                    colors.append('orange')
                else:
                    colors.append('salmon')
            
            bars = ax.bar(range(len(combos)), counts, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1.5)
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                pct = (count / len(self.all_results)) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{count}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xticks(range(len(combos)))
            ax.set_xticklabels(combos, fontsize=11, fontweight='bold', rotation=45, ha='right')
            ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
            ax.set_xlabel('Subunit Combination', fontsize=14, fontweight='bold')
            ax.set_title(f'Top 10 Subunit Combinations at end_2\n(Threshold={self.threshold}Å)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"top10_subunit_combinations_t{self.threshold_str}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Top 10 subunit combinations plot saved")
    
    def create_free_ions_plot(self, output_dir):
        """Create bar plot showing percentage of events with 0, 1, 2, ... free ions."""
        if not self.free_ion_counts:
            print("No free ion data")
            return
        
        # Sort by number of free ions
        sorted_counts = sorted(self.free_ion_counts.items())
        
        n_free_ions = [c[0] for c in sorted_counts]
        frequencies = [c[1] for c in sorted_counts]
        percentages = [(f / len(self.all_results)) * 100 for f in frequencies]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        bars = ax.bar(n_free_ions, percentages, color='coral', alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.8)
        
        # Add labels
        for bar, freq, pct in zip(bars, frequencies, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{freq}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Number of Free Ions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage of Events (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Free Ion Distribution at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_free = np.mean([k for k, v in self.free_ion_counts.items() for _ in range(v)])
        stats_text = f'Mean: {mean_free:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"free_ions_distribution_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Free ions distribution plot saved")
    
    def create_cavity_ion_count_plot(self, output_dir):
        """Create bar plot of total ions in cavity (including permeating ion)."""
        if not self.cavity_ion_counts:
            print("No cavity ion data")
            return
        
        ion_count_freq = Counter(self.cavity_ion_counts)
        sorted_counts = sorted(ion_count_freq.items())
        
        n_ions = [c[0] for c in sorted_counts]
        frequencies = [c[1] for c in sorted_counts]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        bars = ax.bar(n_ions, frequencies, color='steelblue', alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.8)
        
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{freq}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        mean_ions = np.mean(self.cavity_ion_counts)
        ax.axvline(x=mean_ions, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_ions:.1f}')
        
        ax.set_xlabel('Total Ions in Cavity (including permeating)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Cavity Ion Count Distribution at end_2\n(Threshold={self.threshold}Å)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=12)
        
        stats_text = f'Mean: {mean_ions:.1f}\nStd: {np.std(self.cavity_ion_counts):.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"cavity_total_ions_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Cavity total ions plot saved")
    
    def create_sf_alignment_3category_plot(self, output_dir):
        """
        Create plot like the reference image but with 3 categories:
        Permeating Ions | Bound Ions | Free Ions
        """
        if not self.permeating_sf and not self.bound_sf and not self.free_sf:
            print("No SF alignment data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for 3 categories
        categories = []
        all_data = []
        colors = []
        
        if self.permeating_sf:
            categories.append('Permeating Ions')
            all_data.append(self.permeating_sf)
            colors.append('steelblue')
        
        if self.bound_sf:
            categories.append('Bound Ions')
            all_data.append(self.bound_sf)
            colors.append('lightcoral')
        
        if self.free_sf:
            categories.append('Free Ions')
            all_data.append(self.free_sf)
            colors.append('lightgreen')
        
        x_positions = range(len(categories))
        means = [np.mean(data) for data in all_data]
        stds = [np.std(data) for data in all_data]
        ns = [len(data) for data in all_data]
        
        # Draw bars
        bars = ax.bar(x_positions, means, color=colors, alpha=0.6, 
                     edgecolor='black', linewidth=2, width=0.6)
        
        # Add error bars
        ax.errorbar(x_positions, means, yerr=stds, fmt='none', 
                   color='black', capsize=15, capthick=2, linewidth=2)
        
        # Add individual data points
        for i, data in enumerate(all_data):
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, color='black', alpha=0.5, s=30, zorder=3)
        
        # Add statistics labels
        for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
            ax.text(i, mean + std + (ax.get_ylim()[1] * 0.05),
                   f'{mean:.2f} ± {std:.2f}\n(n={n})',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='black', alpha=0.9))
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
        ax.set_ylabel('Distance to SF Line (Å)', fontsize=14, fontweight='bold')
        ax.set_title('SF Line Alignment: Permeating vs Bound vs Free Ions', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sf_alignment_3categories_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SF alignment 3-category plot saved")
    
    def create_all_outputs(self, output_dir=None):
        """Create all plots and CSV files."""
        if output_dir is None:
            output_dir = self.base_dir / f"aggregated_comprehensive_end2_t{self.threshold_str}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating outputs...")
        
        # CSV files
        print("\nCreating CSV files...")
        self.create_combination_csvs(output_dir)
        
        # Plots
        print("\nCreating plots...")
        self.create_top_combinations_plots(output_dir)
        self.create_free_ions_plot(output_dir)
        self.create_cavity_ion_count_plot(output_dir)
        self.create_sf_alignment_3category_plot(output_dir)
        
        print(f"\n✓ All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate comprehensive end2 analysis from multiple runs"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing RUN subdirectories"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        help="Threshold value used in the analysis (default: 2.5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = ComprehensiveEnd2Aggregator(args.base_directory, args.threshold)
    
    if not aggregator.load_all_runs():
        print("No data found. Exiting.")
        return
    
    # Create all outputs
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None
    
    aggregator.create_all_outputs(output_dir=output_dir)
    
    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()


# USAGE:
# python aggregate_comprehensive_end2.py /path/to/G12 --threshold 2.5
# 
# Or with custom output:
# python3 aggregate_occupancy_alignment.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --threshold 3 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/occupancy_alignment