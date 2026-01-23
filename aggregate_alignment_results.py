import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class AlignmentResultsAggregator:
    """
    Aggregates alignment analysis results from multiple runs and creates summary plots.
    """
    
    def __init__(self, base_directory):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing subdirectories for each run
            Each run directory should contain 'sf_alignment_analysis.json'
        """
        self.base_dir = Path(base_directory)
        self.runs_data = {}
        self.aggregated_data = {
            'permeating_sf': [],
            'permeating_asn': [],
            'permeating_glu': [],
            'other_sf': [],
            'other_asn': [],
            'other_glu': []
        }
        
    def load_all_runs(self):
        """
        Load alignment analysis results from all subdirectories.
        """
        print(f"\nSearching for runs in: {self.base_dir}")
        
        # Find all subdirectories that contain sf_alignment_analysis.json
        run_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                json_file = item / "sf_alignment_analysis.json"
                if json_file.exists():
                    run_dirs.append(item)
        
        if not run_dirs:
            print(f"No runs found with sf_alignment_analysis.json in {self.base_dir}")
            return False
        
        print(f"Found {len(run_dirs)} runs:")
        
        # Load data from each run
        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            json_file = run_dir / "sf_alignment_analysis.json"
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.runs_data[run_name] = data
                print(f"  ✓ Loaded: {run_name} ({len(data)} permeation events)")
                
                # Extract data for aggregation
                for event in data:
                    if event['permeating_ion']:
                        self.aggregated_data['permeating_sf'].append({
                            'run': run_name,
                            'value': event['permeating_ion']['distance_to_sf_line']
                        })
                        self.aggregated_data['permeating_asn'].append({
                            'run': run_name,
                            'value': event['permeating_ion']['asn_alignment_percentage']
                        })
                        self.aggregated_data['permeating_glu'].append({
                            'run': run_name,
                            'value': event['permeating_ion']['glu_alignment_percentage']
                        })
                    
                    for other_ion in event['other_ions_in_region2']:
                        self.aggregated_data['other_sf'].append({
                            'run': run_name,
                            'value': other_ion['distance_to_sf_line']
                        })
                        self.aggregated_data['other_asn'].append({
                            'run': run_name,
                            'value': other_ion['asn_alignment_percentage']
                        })
                        self.aggregated_data['other_glu'].append({
                            'run': run_name,
                            'value': other_ion['glu_alignment_percentage']
                        })
                
            except Exception as e:
                print(f"  ✗ Error loading {run_name}: {e}")
        
        return len(self.runs_data) > 0
    
    def calculate_statistics(self, data_list):
        """
        Calculate mean, std, and individual values from data list.
        
        Parameters:
        -----------
        data_list : list of dict
            List of {'run': run_name, 'value': value} dicts
            
        Returns:
        --------
        dict : {'mean', 'std', 'values', 'n'}
        """
        if not data_list:
            return {'mean': 0, 'std': 0, 'values': [], 'n': 0}
        
        values = [d['value'] for d in data_list]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values,
            'n': len(values)
        }
    
    def create_summary_plots(self, output_dir=None):
        """
        Create 3 separate bar plots (SF, ASN, GLU), each comparing permeating vs other ions.
        """
        if not self.runs_data:
            print("No data loaded. Run load_all_runs() first.")
            return
        
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Calculate statistics
        stats = {
            'permeating_sf': self.calculate_statistics(self.aggregated_data['permeating_sf']),
            'permeating_asn': self.calculate_statistics(self.aggregated_data['permeating_asn']),
            'permeating_glu': self.calculate_statistics(self.aggregated_data['permeating_glu']),
            'other_sf': self.calculate_statistics(self.aggregated_data['other_sf']),
            'other_asn': self.calculate_statistics(self.aggregated_data['other_asn']),
            'other_glu': self.calculate_statistics(self.aggregated_data['other_glu'])
        }
        
        # Plot 1: SF Line Distance
        self._create_comparison_plot(
            permeating_stats=stats['permeating_sf'],
            other_stats=stats['other_sf'],
            ylabel='Distance to SF Line (Å)',
            title='SF Line Alignment: Permeating vs Other Ions',
            filename='sf_distance_comparison.png',
            output_dir=output_dir,
            permeating_color='steelblue',
            other_color='coral'
        )
        
        # Plot 2: ASN Alignment
        self._create_comparison_plot(
            permeating_stats=stats['permeating_asn'],
            other_stats=stats['other_asn'],
            ylabel='ASN Circle Alignment (%)',
            title='ASN Circle Alignment: Permeating vs Other Ions',
            filename='asn_alignment_comparison.png',
            output_dir=output_dir,
            permeating_color='green',
            other_color='lightgreen',
            add_reference_lines=True
        )
        
        # Plot 3: GLU Alignment
        self._create_comparison_plot(
            permeating_stats=stats['permeating_glu'],
            other_stats=stats['other_glu'],
            ylabel='GLU Circle Alignment (%)',
            title='GLU Circle Alignment: Permeating vs Other Ions',
            filename='glu_alignment_comparison.png',
            output_dir=output_dir,
            permeating_color='purple',
            other_color='plum',
            add_reference_lines=True
        )
        
        return stats
    
    def _create_comparison_plot(self, permeating_stats, other_stats, ylabel, title, 
                                filename, output_dir, permeating_color, other_color,
                                add_reference_lines=False):
        """
        Create a single comparison plot for permeating vs other ions.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_positions = [0, 1]
        means = [permeating_stats['mean'], other_stats['mean']]
        stds = [permeating_stats['std'], other_stats['std']]
        colors = [permeating_color, other_color]
        labels = ['Permeating Ions', 'Other Ions']
        n_values = [permeating_stats['n'], other_stats['n']]
        
        # Draw bars with error bars
        bars = ax.bar(x_positions, means, width=0.5, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
        ax.errorbar(x_positions, means, yerr=stds, fmt='none', 
                   color='black', capsize=15, capthick=2, linewidth=2)
        
        # Add individual data points
        for i, pos in enumerate(x_positions):
            if i == 0:
                values = permeating_stats['values']
            else:
                values = other_stats['values']
            
            if values:
                x_jitter = np.random.normal(pos, 0.04, size=len(values))
                ax.scatter(x_jitter, values, color='black', alpha=0.5, s=40, zorder=3)
        
        # Add reference lines for percentage plots
        if add_reference_lines:
            ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, 
                      alpha=0.4, label='100% (perfect center)', zorder=1)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.4, label='0% (at edge)', zorder=1)
        
        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        if add_reference_lines:
            ax.legend(fontsize=11, loc='upper right')
        
        # Add statistics annotations above bars
        for i, (pos, mean, std, n) in enumerate(zip(x_positions, means, stds, n_values)):
            # Get the maximum y-value to position text above
            y_max = mean + std
            y_text = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
            
            ax.text(pos, y_text, f'{mean:.2f} ± {std:.2f}\n(n={n})', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        output_file = output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved: {output_file}")
    
    def print_summary(self):
        """
        Print summary statistics to console.
        """
        if not self.runs_data:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("AGGREGATED ALIGNMENT SUMMARY")
        print("="*80)
        print(f"Total runs analyzed: {len(self.runs_data)}")
        print(f"Run names: {', '.join(self.runs_data.keys())}")
        
        print("\n--- PERMEATING IONS ---")
        stats = self.calculate_statistics(self.aggregated_data['permeating_sf'])
        print(f"SF Line Distance: {stats['mean']:.2f} ± {stats['std']:.2f} Å (n={stats['n']})")
        
        stats = self.calculate_statistics(self.aggregated_data['permeating_asn'])
        print(f"ASN Alignment:    {stats['mean']:.1f} ± {stats['std']:.1f} % (n={stats['n']})")
        
        stats = self.calculate_statistics(self.aggregated_data['permeating_glu'])
        print(f"GLU Alignment:    {stats['mean']:.1f} ± {stats['std']:.1f} % (n={stats['n']})")
        
        print("\n--- OTHER IONS ---")
        stats = self.calculate_statistics(self.aggregated_data['other_sf'])
        print(f"SF Line Distance: {stats['mean']:.2f} ± {stats['std']:.2f} Å (n={stats['n']})")
        
        stats = self.calculate_statistics(self.aggregated_data['other_asn'])
        print(f"ASN Alignment:    {stats['mean']:.1f} ± {stats['std']:.1f} % (n={stats['n']})")
        
        stats = self.calculate_statistics(self.aggregated_data['other_glu'])
        print(f"GLU Alignment:    {stats['mean']:.1f} ± {stats['std']:.1f} % (n={stats['n']})")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate alignment analysis results from multiple runs"
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
        help="Output directory for plots (default: same as base_directory)"
    )
    
    args = parser.parse_args()
    
    # Create aggregator and load data
    aggregator = AlignmentResultsAggregator(args.base_directory)
    
    if not aggregator.load_all_runs():
        print("No runs found. Exiting.")
        return
    
    # Print summary
    aggregator.print_summary()
    
    # Create plots
    print("\nCreating plots...")
    aggregator.create_summary_plots(output_dir=args.output_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()