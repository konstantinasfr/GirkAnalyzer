import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
import pandas as pd

class ProximitySummaryAcrossRuns:
    def __init__(self, base_results_dir, channel_type, output_dir=None):
        """
        Analyze proximity results across multiple runs.
        
        Parameters:
        -----------
        base_results_dir : str or Path
            Base directory containing RUN1, RUN2, RUN3, etc. folders
        channel_type : str
            Channel type (e.g., "G12", "G2")
        output_dir : str or Path, optional
            Where to save summary plots. If None, saves to base_results_dir
        """
        self.base_dir = Path(base_results_dir)
        self.channel_type = channel_type
        self.output_dir = Path(output_dir) if output_dir else self.base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import converter
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from analysis.converter import convert_to_pdb_numbering
        self.convert_to_pdb = convert_to_pdb_numbering
        
        # Storage for all runs
        self.run_data = {}
        self.glu_residues = None
        self.asn_residues = None
        
    def find_run_folders(self):
        """Find all RUN folders in the base directory."""
        run_folders = sorted([f for f in self.base_dir.iterdir() 
                            if f.is_dir() and f.name.startswith('RUN')])
        return run_folders
    
    def load_proximity_data(self, run_folder):
        """Load proximity summary from a single run folder."""
        summary_file = run_folder / "proximity_summary.json"
        
        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping...")
            return None
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def collect_all_runs(self):
        """Collect data from all run folders."""
        run_folders = self.find_run_folders()
        print(f"Found {len(run_folders)} run folders: {[f.name for f in run_folders]}")
        
        for run_folder in run_folders:
            run_name = run_folder.name
            data = self.load_proximity_data(run_folder)
            
            if data:
                self.run_data[run_name] = data
                # Store residue lists from first successful load
                if self.glu_residues is None:
                    # Sort by chain designation (extracted from PDB numbering)
                    self.glu_residues = sorted(data['glu_residues'], 
                                            key=lambda x: self.get_chain_sort_key(x))
                    self.asn_residues = sorted(data['asn_residues'],
                                            key=lambda x: self.get_chain_sort_key(x))
        
        print(f"Successfully loaded data from {len(self.run_data)} runs")
        return len(self.run_data) > 0

    def get_chain_sort_key(self, resid):
        """
        Get sorting key based on chain designation.
        Order: A, B, C, D, G1, G2, etc.
        """
        pdb_label = self.convert_to_pdb(resid, self.channel_type)
        
        # Extract chain (everything after the number)
        # e.g., "141.G1" -> "G1", "152.C" -> "C"
        parts = pdb_label.split('.')
        if len(parts) == 2:
            chain = parts[1]
        else:
            chain = ""
        
        # Define sort order: A, B, C, D, then G1, G2, etc.
        chain_order = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'G1': 4,
            'G2': 5,
            'G3': 6,
            'G4': 7
        }
        
        # Get the order value, default to 999 for unknown chains
        order = chain_order.get(chain, 999)
        
        # Return tuple: (order, residue_number) for secondary sorting
        residue_num = int(parts[0]) if len(parts) == 2 else resid
        return (order, residue_num)
        
    def get_residue_label(self, resid, is_glu_group):
        """Get the correct residue label including amino acid type."""
        pdb_num = self.convert_to_pdb(resid, self.channel_type)
        
        if is_glu_group:
            return f"GLU {pdb_num}"
        else:
            # Special case: residue 1105 in G12 is ASP, not ASN
            if self.channel_type == "G12" and resid == 1105:
                return f"ASP {pdb_num}"
            else:
                return f"ASN {pdb_num}"
    
    def calculate_statistics(self):
        """Calculate mean and std for each residue across runs (mean of per-run percentages)."""
        stats = {
            'glu': {},
            'asn': {}
        }
        
        # Calculate for GLU residues
        for resid in self.glu_residues:
            percentages = []
            for run_name, data in self.run_data.items():
                count = data['individual_glu_counts'][str(resid)]
                total = data['total_frames']
                percentage = (count / total) * 100
                percentages.append(percentage)
            
            stats['glu'][resid] = {
                'mean': np.mean(percentages),
                'std': np.std(percentages),
                'values': percentages
            }
        
        # Calculate for ASN residues
        for resid in self.asn_residues:
            percentages = []
            for run_name, data in self.run_data.items():
                count = data['individual_asn_counts'][str(resid)]
                total = data['total_frames']
                percentage = (count / total) * 100
                percentages.append(percentage)
            
            stats['asn'][resid] = {
                'mean': np.mean(percentages),
                'std': np.std(percentages),
                'values': percentages
            }
        
        # Calculate aggregate statistics
        stats['aggregate'] = {}
        for key in ['percentage_any_glu', 'percentage_any_asn', 'percentage_any_residue']:
            values = [data[key] for run_name, data in self.run_data.items()]
            stats['aggregate'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        return stats

    def calculate_pooled_statistics(self):
        """
        Calculate percentages by pooling raw frame counts across ALL runs,
        as if every run were one long concatenated trajectory. This is
        different from calculate_statistics(), which averages per-run
        percentages (and can be skewed by short runs counting as much as
        long ones). Here, every individual frame counts equally.
        """
        pooled = {'glu': {}, 'asn': {}, 'aggregate': {}}

        total_frames_all = sum(data['total_frames'] for data in self.run_data.values())
        pooled['total_frames_all'] = total_frames_all
        pooled['n_runs'] = len(self.run_data)

        # GLU: raw counts are stored per run, so sum them directly
        for resid in self.glu_residues:
            total_count = sum(data['individual_glu_counts'][str(resid)] for data in self.run_data.values())
            pooled['glu'][resid] = {
                'count': total_count,
                'percentage': 100.0 * total_count / total_frames_all if total_frames_all > 0 else np.nan
            }

        # ASN/ASP: same approach
        for resid in self.asn_residues:
            total_count = sum(data['individual_asn_counts'][str(resid)] for data in self.run_data.values())
            pooled['asn'][resid] = {
                'count': total_count,
                'percentage': 100.0 * total_count / total_frames_all if total_frames_all > 0 else np.nan
            }

        # Aggregate categories ("Any GLU" / "Any ASN" / "Any Residue") are only
        # stored as per-run percentages in the JSON, not raw counts. Pooling
        # those correctly means frame-weighting each run's percentage before
        # combining, which is mathematically equivalent to pooling the
        # underlying counts: sum(count_i) / sum(frames_i)
        #                   = sum(pct_i * frames_i / 100) / sum(frames_i)
        for key in ['percentage_any_glu', 'percentage_any_asn', 'percentage_any_residue']:
            weighted_sum = sum(data[key] * data['total_frames'] for data in self.run_data.values())
            pooled['aggregate'][key] = weighted_sum / total_frames_all if total_frames_all > 0 else np.nan

        return pooled
    
    def plot_summary_all_residues(self, stats):
        """Create summary bar plot for all 8 residues with error bars and individual run points."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Combine all residues
        all_residues = list(self.glu_residues) + list(self.asn_residues)
        means = ([stats['glu'][resid]['mean'] for resid in self.glu_residues] +
                [stats['asn'][resid]['mean'] for resid in self.asn_residues])
        stds = ([stats['glu'][resid]['std'] for resid in self.glu_residues] +
                [stats['asn'][resid]['std'] for resid in self.asn_residues])
        values_list = ([stats['glu'][resid]['values'] for resid in self.glu_residues] +
                    [stats['asn'][resid]['values'] for resid in self.asn_residues])
        
        # Labels with PDB numbering
        labels = ([self.get_residue_label(resid, True) for resid in self.glu_residues] +
                [self.get_residue_label(resid, False) for resid in self.asn_residues])
        
        colors = ['red']*len(self.glu_residues) + ['blue']*len(self.asn_residues)
        
        # Create bars with error bars
        x = range(len(all_residues))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.5, 
                    edgecolor='black', linewidth=1.5, capsize=5,
                    error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Add individual run data points
        for i, (x_pos, values) in enumerate(zip(x, values_list)):
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, color='black', s=50, alpha=0.7, 
                    zorder=5, edgecolors='white', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)  # Increased
        ax.set_ylabel('Percentage of Frames (%)', fontsize=20)  # Increased
        ax.set_xlabel('Residue', fontsize=20)  # Increased
        ax.set_title(f'{self.channel_type}: Ion Proximity Summary Across {len(self.run_data)} Runs (\u22643.0 \u00c5)', 
                    fontsize=22, fontweight='bold')  # Increased
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=16)  # Increased
        
        # Add value labels on bars - BIGGER
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.1f}%\n\u00b1{std:.1f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')  # Increased and bold
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, edgecolor='black', label='GLU'),
            Patch(facecolor='blue', alpha=0.5, edgecolor='black', label='ASN/ASP'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                markersize=8, label='Individual runs', markeredgecolor='white')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16)  # Increased
        
        # Add run count text
        ax.text(0.98, 0.98, f'n = {len(self.run_data)} runs',
                transform=ax.transAxes, fontsize=16, va='top', ha='right',  # Increased
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_file = self.output_dir / "proximity_summary_all_residues.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')  # Reduced to 150
        plt.close()
        print(f"Summary plot saved to: {plot_file}")

    def plot_summary_glu_asn_separate(self, stats):
        """Create separate bar plots for GLU and ASN with error bars and individual run points."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # GLU plot
        glu_means = [stats['glu'][resid]['mean'] for resid in self.glu_residues]
        glu_stds = [stats['glu'][resid]['std'] for resid in self.glu_residues]
        glu_values = [stats['glu'][resid]['values'] for resid in self.glu_residues]
        glu_labels = [self.get_residue_label(resid, True).replace(' ', '\n') 
                    for resid in self.glu_residues]
        
        x1 = range(len(self.glu_residues))
        bars1 = ax1.bar(x1, glu_means, yerr=glu_stds, color='red', alpha=0.5,
                        edgecolor='black', linewidth=1.5, capsize=5,
                        error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Add individual points for GLU
        for i, (x_pos, values) in enumerate(zip(x1, glu_values)):
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax1.scatter(x_jitter, values, color='black', s=60, alpha=0.7,
                    zorder=5, edgecolors='white', linewidth=0.5)
        
        ax1.set_xticks(x1)
        ax1.set_xticklabels(glu_labels, fontsize=18)  # Increased
        ax1.set_ylabel('Percentage of Frames (%)', fontsize=20)  # Increased
        ax1.set_title(f'{self.channel_type}: GLU Residues - Summary (n={len(self.run_data)} runs)', 
                    fontsize=20, fontweight='bold')  # Increased
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='y', labelsize=16)  # Increased
        
        for bar, mean, std in zip(bars1, glu_means, glu_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.1f}%\n\u00b1{std:.1f}',
                    ha='center', va='bottom', fontsize=15, fontweight='bold')  # Increased and bold
        
        # ASN plot
        asn_means = [stats['asn'][resid]['mean'] for resid in self.asn_residues]
        asn_stds = [stats['asn'][resid]['std'] for resid in self.asn_residues]
        asn_values = [stats['asn'][resid]['values'] for resid in self.asn_residues]
        asn_labels = [self.get_residue_label(resid, False).replace(' ', '\n') 
                    for resid in self.asn_residues]
        
        x2 = range(len(self.asn_residues))
        bars2 = ax2.bar(x2, asn_means, yerr=asn_stds, color='blue', alpha=0.5,
                        edgecolor='black', linewidth=1.5, capsize=5,
                        error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Add individual points for ASN
        for i, (x_pos, values) in enumerate(zip(x2, asn_values)):
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax2.scatter(x_jitter, values, color='black', s=60, alpha=0.7,
                    zorder=5, edgecolors='white', linewidth=0.5)
        
        ax2.set_xticks(x2)
        ax2.set_xticklabels(asn_labels, fontsize=18)  # Increased
        ax2.set_ylabel('Percentage of Frames (%)', fontsize=20)  # Increased
        ax2.set_title(f'{self.channel_type}: ASN/ASP Residues - Summary (n={len(self.run_data)} runs)', 
                    fontsize=20, fontweight='bold')  # Increased
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='y', labelsize=16)  # Increased
        
        for bar, mean, std in zip(bars2, asn_means, asn_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.1f}%\n\u00b1{std:.1f}',
                    ha='center', va='bottom', fontsize=15, fontweight='bold')  # Increased and bold
        
        # Add legend to first plot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                markersize=8, label='Individual runs', markeredgecolor='white')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=15)  # Increased
        
        plt.tight_layout()
        plot_file = self.output_dir / "proximity_summary_glu_asn_separate.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')  # Reduced to 150
        plt.close()
        print(f"Separate summary plot saved to: {plot_file}")

    def plot_aggregate_summary(self, stats):
        """Create bar plot for aggregate statistics with individual run points."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = ['Any GLU', 'Any ASN/ASP', 'Any Residue']
        means = [
            stats['aggregate']['percentage_any_glu']['mean'],
            stats['aggregate']['percentage_any_asn']['mean'],
            stats['aggregate']['percentage_any_residue']['mean']
        ]
        stds = [
            stats['aggregate']['percentage_any_glu']['std'],
            stats['aggregate']['percentage_any_asn']['std'],
            stats['aggregate']['percentage_any_residue']['std']
        ]
        values_list = [
            stats['aggregate']['percentage_any_glu']['values'],
            stats['aggregate']['percentage_any_asn']['values'],
            stats['aggregate']['percentage_any_residue']['values']
        ]
        
        colors = ['red', 'blue', 'green']
        x = range(len(categories))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.5,
                    edgecolor='black', linewidth=2, capsize=8,
                    error_kw={'linewidth': 2.5, 'ecolor': 'black'})
        
        # Add individual run data points
        for i, (x_pos, values) in enumerate(zip(x, values_list)):
            x_jitter = np.random.normal(x_pos, 0.03, size=len(values))
            ax.scatter(x_jitter, values, color='black', s=80, alpha=0.7,
                    zorder=5, edgecolors='white', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=20)  # Increased
        ax.set_ylabel('Percentage of Frames (%)', fontsize=22)  # Increased
        ax.set_title(f'{self.channel_type}: Aggregate Proximity Summary (n={len(self.run_data)} runs, \u22643.0 \u00c5)', 
                    fontsize=24, fontweight='bold')  # Increased
        ax.set_ylim(0, 120)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=18)  # Increased
        
        # Add value labels - MUCH BIGGER
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 3,
                    f'{mean:.1f}%\n\u00b1{std:.1f}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')  # Increased significantly
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                markersize=10, label='Individual runs', markeredgecolor='white')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16)  # Increased
        
        plt.tight_layout()
        plot_file = self.output_dir / "proximity_summary_aggregate.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')  # Reduced to 150
        plt.close()
        print(f"Aggregate summary plot saved to: {plot_file}")

    def plot_pooled_summary(self, pooled):
        """
        Create a bar plot treating all runs as one long concatenated trajectory:
        percentage = (total frames within threshold across ALL runs) /
                     (total frames across ALL runs) * 100.
        No error bars here since there's only one pooled value per residue,
        not a distribution across runs.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        all_residues = list(self.glu_residues) + list(self.asn_residues)
        percentages = ([pooled['glu'][resid]['percentage'] for resid in self.glu_residues] +
                       [pooled['asn'][resid]['percentage'] for resid in self.asn_residues])
        counts = ([pooled['glu'][resid]['count'] for resid in self.glu_residues] +
                  [pooled['asn'][resid]['count'] for resid in self.asn_residues])

        labels = ([self.get_residue_label(resid, True) for resid in self.glu_residues] +
                  [self.get_residue_label(resid, False) for resid in self.asn_residues])

        colors = ['red'] * len(self.glu_residues) + ['blue'] * len(self.asn_residues)

        x = range(len(all_residues))
        bars = ax.bar(x, percentages, color=colors, alpha=0.5,
                      edgecolor='black', linewidth=1.5, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
        ax.set_ylabel('Percentage of Pooled Frames (%)', fontsize=20)
        ax.set_xlabel('Residue', fontsize=20)
        ax.set_title(
            f'{self.channel_type}: Ion Proximity Summary - All {pooled["n_runs"]} Runs Pooled '
            f'({pooled["total_frames_all"]:,} total frames, \u22643.0 \u00c5)',
            fontsize=20, fontweight='bold'
        )
        ax.set_ylim(0, max(percentages) * 1.25 if percentages else 100)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=16)

        for bar, pct, count in zip(bars, percentages, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(percentages) * 0.02,
                    f'{pct:.1f}%\n(n={count:,})',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        legend_elements = [
            Patch(facecolor='red', alpha=0.5, edgecolor='black', label='GLU'),
            Patch(facecolor='blue', alpha=0.5, edgecolor='black', label='ASN/ASP'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16)

        plt.tight_layout()
        plot_file = self.output_dir / "proximity_summary_pooled_residues.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Pooled summary plot saved to: {plot_file}")

    def plot_pooled_aggregate(self, pooled):
        """
        Pooled version of the aggregate summary (Any GLU / Any ASN/ASP /
        Any Residue), computed as one frame-weighted percentage across all
        runs rather than a mean-of-run-percentages with error bars.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        categories = ['Any GLU', 'Any ASN/ASP', 'Any Residue']
        percentages = [
            pooled['aggregate']['percentage_any_glu'],
            pooled['aggregate']['percentage_any_asn'],
            pooled['aggregate']['percentage_any_residue'],
        ]

        colors = ['red', 'blue', 'green']
        x = range(len(categories))
        bars = ax.bar(x, percentages, color=colors, alpha=0.5,
                      edgecolor='black', linewidth=2, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=20)
        ax.set_ylabel('Percentage of Pooled Frames (%)', fontsize=22)
        ax.set_title(
            f'{self.channel_type}: Aggregate Proximity Summary - All {pooled["n_runs"]} Runs Pooled\n'
            f'({pooled["total_frames_all"]:,} total frames, \u22643.0 \u00c5)',
            fontsize=20, fontweight='bold'
        )
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=18)

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

        plt.tight_layout()
        plot_file = self.output_dir / "proximity_summary_pooled_aggregate.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Pooled aggregate summary plot saved to: {plot_file}")

    def save_pooled_statistics_table(self, pooled):
        """Save pooled statistics to CSV."""
        rows = []

        for resid in self.glu_residues:
            label = self.get_residue_label(resid, True)
            rows.append({
                'Residue': label,
                'Type': 'GLU',
                'Pooled_Count': pooled['glu'][resid]['count'],
                'Pooled_Percentage (%)': pooled['glu'][resid]['percentage'],
            })

        for resid in self.asn_residues:
            label = self.get_residue_label(resid, False)
            residue_type = 'ASP' if (self.channel_type == "G12" and resid == 1105) else 'ASN'
            rows.append({
                'Residue': label,
                'Type': residue_type,
                'Pooled_Count': pooled['asn'][resid]['count'],
                'Pooled_Percentage (%)': pooled['asn'][resid]['percentage'],
            })

        df = pd.DataFrame(rows)
        csv_file = self.output_dir / "proximity_summary_pooled_statistics.csv"
        df.to_csv(csv_file, index=False)
        print(f"Pooled statistics table saved to: {csv_file}")

        txt_file = self.output_dir / "proximity_summary_pooled_statistics.txt"
        with open(txt_file, 'w') as f:
            f.write(f"POOLED ION PROXIMITY STATISTICS ({pooled['n_runs']} RUNS, "
                     f"{pooled['total_frames_all']:,} TOTAL FRAMES)\n")
            f.write("=" * 80 + "\n\n")
            f.write("INDIVIDUAL RESIDUES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Residue':<15} {'Type':<8} {'Pooled Count':<15} {'Pooled %':<10}\n")
            f.write("-" * 80 + "\n")
            for _, row in df.iterrows():
                f.write(f"{row['Residue']:<15} {row['Type']:<8} "
                        f"{row['Pooled_Count']:<15} {row['Pooled_Percentage (%)']:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("AGGREGATE STATISTICS (frame-weighted pooling):\n")
            f.write("-" * 80 + "\n")
            for key, label in [('percentage_any_glu', 'Any GLU'),
                                ('percentage_any_asn', 'Any ASN/ASP'),
                                ('percentage_any_residue', 'Any Residue')]:
                f.write(f"{label:<20} Pooled %: {pooled['aggregate'][key]:.2f}%\n")

        print(f"Pooled statistics text file saved to: {txt_file}")
        
    def save_statistics_table(self, stats):
        """Save statistics to CSV and text files."""
        # Individual residues table
        rows = []
        
        for resid in self.glu_residues:
            label = self.get_residue_label(resid, True)
            rows.append({
                'Residue': label,
                'Type': 'GLU',
                'Mean (%)': stats['glu'][resid]['mean'],
                'Std (%)': stats['glu'][resid]['std'],
                'Values': stats['glu'][resid]['values']
            })
        
        for resid in self.asn_residues:
            label = self.get_residue_label(resid, False)
            residue_type = 'ASP' if (self.channel_type == "G12" and resid == 1105) else 'ASN'
            rows.append({
                'Residue': label,
                'Type': residue_type,
                'Mean (%)': stats['asn'][resid]['mean'],
                'Std (%)': stats['asn'][resid]['std'],
                'Values': stats['asn'][resid]['values']
            })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_file = self.output_dir / "proximity_summary_statistics.csv"
        df.to_csv(csv_file, index=False)
        print(f"Statistics table saved to: {csv_file}")
        
        # Save to text file with nice formatting
        txt_file = self.output_dir / "proximity_summary_statistics.txt"
        with open(txt_file, 'w') as f:
            f.write(f"ION PROXIMITY STATISTICS ACROSS {len(self.run_data)} RUNS\n")
            f.write("="*80 + "\n\n")
            
            f.write("INDIVIDUAL RESIDUES:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Residue':<15} {'Type':<8} {'Mean (%)':<12} {'Std (%)':<12} {'All Values'}\n")
            f.write("-"*80 + "\n")
            
            for _, row in df.iterrows():
                values_str = ", ".join([f"{v:.1f}" for v in row['Values']])
                f.write(f"{row['Residue']:<15} {row['Type']:<8} "
                       f"{row['Mean (%)']:>10.2f}  {row['Std (%)']:>10.2f}  "
                       f"{values_str}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("AGGREGATE STATISTICS:\n")
            f.write("-"*80 + "\n")
            
            for key, label in [('percentage_any_glu', 'Any GLU'),
                              ('percentage_any_asn', 'Any ASN/ASP'),
                              ('percentage_any_residue', 'Any Residue')]:
                mean = stats['aggregate'][key]['mean']
                std = stats['aggregate'][key]['std']
                values = stats['aggregate'][key]['values']
                values_str = ", ".join([f"{v:.1f}" for v in values])
                f.write(f"{label:<20} Mean: {mean:>6.2f}%  Std: {std:>6.2f}%  "
                       f"Values: {values_str}\n")
        
        print(f"Statistics text file saved to: {txt_file}")
    
    def run_summary_analysis(self):
        """Run complete summary analysis."""
        print("="*70)
        print("PROXIMITY SUMMARY ANALYSIS ACROSS RUNS")
        print("="*70)
        
        # Collect data from all runs
        if not self.collect_all_runs():
            print("Error: No valid run data found!")
            return
        
        # Calculate statistics (mean of per-run percentages, with error bars)
        print("\nCalculating statistics...")
        stats = self.calculate_statistics()
        
        # Generate plots
        print("\nGenerating summary plots...")
        self.plot_summary_all_residues(stats)
        self.plot_summary_glu_asn_separate(stats)
        self.plot_aggregate_summary(stats)
        
        # Save statistics
        print("\nSaving statistics tables...")
        self.save_statistics_table(stats)

        # Pooled statistics (all runs concatenated into one long trajectory)
        print("\nCalculating pooled statistics (all runs as one trajectory)...")
        pooled = self.calculate_pooled_statistics()

        print("\nGenerating pooled summary plots...")
        self.plot_pooled_summary(pooled)
        self.plot_pooled_aggregate(pooled)

        print("\nSaving pooled statistics tables...")
        self.save_pooled_statistics_table(pooled)
        
        print("\n" + "="*70)
        print("SUMMARY ANALYSIS COMPLETE!")
        print("="*70)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize proximity analysis across multiple runs")
    parser.add_argument("--base_dir", required=True, help="Base directory containing RUN folders")
    parser.add_argument("--channel_type", required=True, help="Channel type (G2, G12, etc.)")
    parser.add_argument("--output_dir", default=None, help="Output directory for summary plots")
    
    args = parser.parse_args()
    
    analyzer = ProximitySummaryAcrossRuns(
        base_results_dir=args.base_dir,
        channel_type=args.channel_type,
        output_dir=args.output_dir
    )
    
    analyzer.run_summary_analysis()


if __name__ == "__main__":
    main()


# python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
# python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
'''
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --channel_type G2
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --channel_type G12_ML
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --channel_type G12_GAT
'''
