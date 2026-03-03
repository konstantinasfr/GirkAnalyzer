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
    
    def __init__(self, base_directory, threshold=2.5, channel_type="G12"):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float
            Threshold value used in the analysis
        channel_type : str
            Channel type ('G2', 'G12', 'G12_GAT', 'G12_ML')
        """
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.channel_type = channel_type
        
        # Convert threshold to string (handle 2.5 -> "2.5")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)
        
        self.all_results = []
        self.residue_combination_counts = Counter()
        self.subunit_combination_counts = Counter()
        self.free_ion_counts = Counter()
        self.cavity_ion_counts = []
        
        # SF alignment data
        self.permeating_sf = []
        self.bound_sf = []
        self.free_sf = []
        self.free_closest_sf = []
        self.other_free_sf = []
    
    def convert_D_to_G1(self, label):
        """Convert subunit D to G1 in labels, but ONLY for non-G2 channels"""
        if self.channel_type == "G2":
            return label  # Don't convert for G2!
        
        if label and isinstance(label, str):
            label = label.replace('_D', '_G1')
            label = label.replace('.D', '.G1')
            if label == 'D':
                label = 'G1'
            if label.startswith('D_'):
                label = 'G1_' + label[2:]
            return label
        return label
        
    def load_all_runs(self):
        """Load comprehensive end2 results from all subdirectories."""
        print(f"\nSearching for runs in: {self.base_dir}")
        
        run_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')]
        
        if not run_dirs:
            print(f"No RUN directories found in {self.base_dir}")
            return False
        
        print(f"Found {len(run_dirs)} RUN directories")
        
        loaded_count = 0
        
        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            
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
                    
                    for event in results:
                        if event['residue_combination']:
                            self.residue_combination_counts[event['residue_combination']] += 1
                        if event['subunit_combination']:
                            self.subunit_combination_counts[event['subunit_combination']] += 1
                        
                        n_free = event['n_free_ions']
                        self.free_ion_counts[n_free] += 1
                        
                        total_with_permeating = event['total_ions_in_cavity'] + 1
                        self.cavity_ion_counts.append(total_with_permeating)
                        
                        if event['permeating_ion_sf_distance'] is not None:
                            self.permeating_sf.append(event['permeating_ion_sf_distance'])
                        
                        for bound_ion in event['bound_ions_sf_alignment']:
                            self.bound_sf.append(bound_ion['sf_distance'])
                        
                        for free_ion in event['free_ions_sf_alignment']:
                            self.free_sf.append(free_ion['sf_distance'])
                        
                        if event.get('free_closest_to_sf'):
                            self.free_closest_sf.append(event['free_closest_to_sf']['sf_distance'])
                        
                        for other_free in event.get('other_free_ions_sf_alignment', []):
                            self.other_free_sf.append(other_free['sf_distance'])
                    
                    loaded_count += 1
                    print(f"  ✓ {run_name}: Loaded ({len(results)} events)")
                    
                except Exception as e:
                    print(f"  ✗ {run_name}: Error loading: {e}")
        
        print(f"\nSuccessfully loaded: {loaded_count} runs")
        print(f"Total events: {len(self.all_results)}")
        
        return loaded_count > 0
    
    def create_combination_csvs(self, output_dir):
        """Create CSV files with all combinations and frequencies."""
        residue_data = []
        for combo, count in self.residue_combination_counts.most_common():
            percentage = (count / len(self.all_results)) * 100 if self.all_results else 0
            residue_data.append({
                'Residue_Combination': self.convert_D_to_G1(combo),
                'Frequency': count,
                'Percentage': percentage
            })
        
        df_residue = pd.DataFrame(residue_data)
        csv_file = output_dir / f"residue_combinations_t{self.threshold_str}.csv"
        df_residue.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"✓ Residue combinations CSV: {csv_file}")
        
        subunit_data = []
        for combo, count in self.subunit_combination_counts.most_common():
            percentage = (count / len(self.all_results)) * 100 if self.all_results else 0
            subunit_data.append({
                'Subunit_Combination': self.convert_D_to_G1(combo),
                'Frequency': count,
                'Percentage': percentage
            })
        
        df_subunit = pd.DataFrame(subunit_data)
        csv_file = output_dir / f"subunit_combinations_t{self.threshold_str}.csv"
        df_subunit.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"✓ Subunit combinations CSV: {csv_file}")
        
        return df_residue, df_subunit
    
    def create_top_combinations_plots(self, output_dir):
        """Create bar plots for top combinations."""
        if self.subunit_combination_counts:
            top_sub = self.subunit_combination_counts.most_common(10)
            
            combos = [self.convert_D_to_G1(c[0]) for c in top_sub]
            counts = [c[1] for c in top_sub]
            
            # Narrower plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = []
            for combo in combos:
                n_parts = len([p for p in combo.split('_') if p in ['A', 'B', 'C', 'G1', 'D']])
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
                       ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            ax.set_xticks(range(len(combos)))
            ax.set_xticklabels(combos, fontsize=20, fontweight='bold', rotation=45, ha='right')
            ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
            ax.set_xlabel('Subunit Combination', fontsize=22, fontweight='bold')
            ax.set_title(f'Ion Leaves GLU/ASN Cavity - Subunit Combinations', 
                        fontsize=22, fontweight='bold', pad=18)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', labelsize=18)
            
            # Add gap at top - 15% extra space for labels
            max_count = max(counts)
            ax.set_ylim(0, max_count * 1.25)
            
            # Add gap at top
            max_count = max(counts)
            ax.set_ylim(0, max_count * 1.25)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"top10_subunit_combinations_t{self.threshold_str}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Top 10 subunit combinations plot saved")
    
    def create_free_ions_plot(self, output_dir):
        """Create bar plot showing percentage of events with free ions."""
        if not self.free_ion_counts:
            print("No free ion data")
            return
        
        sorted_counts = sorted(self.free_ion_counts.items())
        
        n_free_ions = [c[0] for c in sorted_counts]
        frequencies = [c[1] for c in sorted_counts]
        percentages = [(f / len(self.all_results)) * 100 for f in frequencies]
        
        # Narrower plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(n_free_ions, percentages, color='coral', alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.8)
        
        for bar, freq, pct in zip(bars, frequencies, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{freq}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('Number of Free Ions', fontsize=22, fontweight='bold')
        ax.set_ylabel('Percentage of Events (%)', fontsize=22, fontweight='bold')
        ax.set_title(f'Ion Leaves GLU/ASN Cavity - Free Ion Distribution', 
                    fontsize=22, fontweight='bold', pad=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=18)
        
        # Add gap at top
        max_pct = max(percentages)
        ax.set_ylim(0, max_pct * 1.1)
        
        mean_free = np.mean([k for k, v in self.free_ion_counts.items() for _ in range(v)])
        stats_text = f'Mean: {mean_free:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"free_ions_distribution_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Free ions distribution plot saved")
    
    def create_cavity_ion_count_plot(self, output_dir):
        """Create bar plot of total ions in cavity."""
        if not self.cavity_ion_counts:
            print("No cavity ion data")
            return
        
        ion_count_freq = Counter(self.cavity_ion_counts)
        sorted_counts = sorted(ion_count_freq.items())
        
        n_ions = [c[0] for c in sorted_counts]
        frequencies = [c[1] for c in sorted_counts]
        
        # Narrower plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(n_ions, frequencies, color='steelblue', alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.8)
        
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{freq}',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        mean_ions = np.mean(self.cavity_ion_counts)
        ax.axvline(x=mean_ions, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_ions:.1f}')
        
        ax.set_xlabel('Total Ions in Cavity', fontsize=22, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
        ax.set_title(f'Ion Leaves GLU/ASN Cavity - Ion Count Distribution', 
                    fontsize=22, fontweight='bold', pad=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=15)
        
        # Add gap at top
        max_freq = max(frequencies)
        ax.set_ylim(0, max_freq * 1.1)
        
        stats_text = f'Mean: {mean_ions:.1f}\nStd: {np.std(self.cavity_ion_counts):.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"cavity_total_ions_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Cavity total ions plot saved")
    
    def create_sf_alignment_3category_plot(self, output_dir):
        """Create 3-category SF alignment plot."""
        if not self.permeating_sf and not self.bound_sf and not self.free_sf:
            print("No SF alignment data")
            return
        
        # Narrower plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        bars = ax.bar(x_positions, means, color=colors, alpha=0.6, 
                     edgecolor='black', linewidth=2, width=0.6)
        
        ax.errorbar(x_positions, means, yerr=stds, fmt='none', 
                   color='black', capsize=15, capthick=2, linewidth=2)
        
        for i, data in enumerate(all_data):
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, color='black', alpha=0.5, s=30, zorder=3)
        
        for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
            label_y = mean + std + (ax.get_ylim()[1] * 0.05)
            ax.text(i, label_y,
                   f'{mean:.2f} ± {std:.2f}\n(n={n})',
                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='black', alpha=0.9, linewidth=1.5))
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, fontsize=20, fontweight='bold')
        ax.set_ylabel('Distance to SF Line (Å)', fontsize=22, fontweight='bold')
        ax.set_title('Ion Leaves GLU/ASN Cavity - SF Alignment', 
                    fontsize=22, fontweight='bold', pad=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=18)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sf_alignment_3categories_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SF alignment 3-category plot saved")
    
    def create_sf_alignment_4category_plot(self, output_dir):
        """Create 4-category SF alignment plot with bigger fonts."""
        if not self.permeating_sf and not self.bound_sf and not self.free_closest_sf and not self.other_free_sf:
            print("No SF alignment data for 4-category plot")
            return
        
        # Narrower plot for bigger fonts
        fig, ax = plt.subplots(figsize=(11, 6))
        
        categories = []
        all_data = []
        colors = []
        
        if self.permeating_sf:
            categories.append('Permeating\nIons')
            all_data.append(self.permeating_sf)
            colors.append('steelblue')
        
        if self.bound_sf:
            categories.append('Bound\nIons')
            all_data.append(self.bound_sf)
            colors.append('lightcoral')
        
        if self.free_closest_sf:
            categories.append('Free Closest\nto SF')
            all_data.append(self.free_closest_sf)
            colors.append('gold')
        
        if self.other_free_sf:
            categories.append('Other Free\nIons')
            all_data.append(self.other_free_sf)
            colors.append('lightgreen')
        
        x_positions = range(len(categories))
        means = [np.mean(data) for data in all_data]
        stds = [np.std(data) for data in all_data]
        ns = [len(data) for data in all_data]
        
        bars = ax.bar(x_positions, means, color=colors, alpha=0.6, 
                     edgecolor='black', linewidth=2, width=0.6)
        
        ax.errorbar(x_positions, means, yerr=stds, fmt='none', 
                   color='black', capsize=15, capthick=2, linewidth=2)
        
        for i, data in enumerate(all_data):
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, color='black', alpha=0.4, s=20, zorder=3)
        
        for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
            label_y = mean + std + (ax.get_ylim()[1] * 0.05)
            ax.text(i, label_y,
                   f'{mean:.2f} ± {std:.2f}\n(n={n})',
                   ha='center', va='bottom', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='black', alpha=0.9, linewidth=1.5))
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, fontsize=22, fontweight='bold')
        ax.set_ylabel('Distance to SF Line (Å)', fontsize=24, fontweight='bold')
        ax.set_title('Ion Leaves GLU/ASN Cavity - SF Alignment', 
                    fontsize=24, fontweight='bold', pad=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sf_alignment_4categories_t{self.threshold_str}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SF alignment 4-category plot saved")
    
    def create_all_outputs(self, output_dir=None):
        """Create all plots and CSV files."""
        if output_dir is None:
            output_dir = self.base_dir / f"aggregated_comprehensive_end2_t{self.threshold_str}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating outputs...")
        
        print("\nCreating CSV files...")
        self.create_combination_csvs(output_dir)
        
        print("\nCreating plots...")
        self.create_top_combinations_plots(output_dir)
        self.create_free_ions_plot(output_dir)
        self.create_cavity_ion_count_plot(output_dir)
        self.create_sf_alignment_3category_plot(output_dir)
        self.create_sf_alignment_4category_plot(output_dir)
        
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
        "-c", "--channel",
        type=str,
        default="G12",
        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'],
        help="Channel type (default: G12). Use G2 to keep D labels, G12/G12_GAT/G12_ML to convert D to G1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = ComprehensiveEnd2Aggregator(args.base_directory, args.threshold, args.channel)
    
    if not aggregator.load_all_runs():
        print("No data found. Exiting.")
        return
    
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