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
    Creates bar plots (with counts + percentages) and summary tables.
    """

    def __init__(self, base_directory, threshold, channel_type=None):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float or int
            Threshold value used in the analysis
        channel_type : str or None
            Channel type label to show in plot titles (e.g. "G12", "G12_ML").
            If None, it is inferred from the base_directory folder name.
        """
        self.base_dir = Path(base_directory)
        self.runs_data = {}
        self.threshold = threshold
        self.channel_type = channel_type if channel_type is not None else self.base_dir.name

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
                print(f"  \u2713 Loaded: {run_name} ({len(data['results'])} events)")

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
                print(f"  \u2717 Error loading {run_name}: {e}")

        return len(self.runs_data) > 0

    def create_bar_plot(self, case_data, case_title, output_file):
        """
        Create a bar plot for a single case showing winner counts,
        with percentage-of-total and average distance annotated on each bar.

        Parameters:
        -----------
        case_data : dict
            Dictionary with residue PDB as key and data as value
        case_title : str
            Title for the plot
        output_file : Path
            Output file path
        """
        if not case_data:
            print(f"No data for {case_title}")
            return

        # Sort residues by count (descending)
        sorted_residues = sorted(case_data.items(), key=lambda x: x[1]['count'], reverse=True)

        residues = [r[0] for r in sorted_residues]
        counts = [r[1]['count'] for r in sorted_residues]
        avg_distances = [np.mean(r[1]['distances']) for r in sorted_residues]

        total = sum(counts)
        percentages = [100.0 * c / total if total > 0 else 0.0 for c in counts]

        # Color code by residue type (softer, more modern palette)
        color_map = {
            'GLU': '#E8746B',   # muted coral
            'ASN': '#5AA9C9',   # muted blue
            'ASP': '#6FBF73',   # muted green
            'Unknown': '#B0B0B0',
        }

        def res_type(res):
            if '152' in res or '141' in res:
                return 'GLU'
            elif '184' in res:
                return 'ASN'
            elif '173' in res:
                return 'ASP'
            return 'Unknown'

        colors = [color_map[res_type(r)] for r in residues]

        plt.rcParams['font.family'] = 'DejaVu Sans'

        fig, ax = plt.subplots(figsize=(10, 6.2))

        bars = ax.bar(
            range(len(residues)), counts,
            color=colors, alpha=0.85, edgecolor='black', linewidth=1.2,
            width=0.65, zorder=3
        )

        # Headroom above tallest bar so labels never collide with the title
        max_count = max(counts)
        ax.set_ylim(0, max_count * 1.35)

        # Label each bar: count, percentage, and average distance
        for bar, count, pct, avg_dist in zip(bars, counts, percentages, avg_distances):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + max_count * 0.03,
                f'{count} ({pct:.1f}%)\n{avg_dist:.2f} \u00c5',
                ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=4
            )

        # Formatting
        ax.set_xticks(range(len(residues)))
        ax.set_xticklabels(residues, fontsize=17, fontweight='bold', rotation=40, ha='right')
        ax.set_ylabel('Number of Times Closest', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_xlabel('Residue', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_title(case_title, fontsize=20, fontweight='bold', pad=22)
        ax.tick_params(axis='both', labelsize=15)

        # Cleaner grid / spines
        ax.grid(True, alpha=0.25, axis='y', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.1)

        # Legend (only for residue types actually present)
        from matplotlib.patches import Patch
        present_types = sorted(set(res_type(r) for r in residues), key=lambda t: ['GLU', 'ASN', 'ASP', 'Unknown'].index(t))
        legend_elements = [Patch(facecolor=color_map[t], edgecolor='black', label=t) for t in present_types]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=13, frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\u2713 Plot saved: {output_file}")

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

        # Case 1 & 3 are snapshot-at-frame results — the threshold value has
        # no effect on them, so it is intentionally left out of the title.
        self.create_bar_plot(self.case_1_data,
                              f"{self.channel_type}: Ion Leaves GLU/ASN Cavity - Closest Residue",
                              output_dir / f"ion_leaves_cavity_closest_threshold{self.threshold}.png")

        # Case 2 & 4 walk backwards until the threshold is crossed, so the
        # threshold value genuinely determines the result — kept in title.
        self.create_bar_plot(self.case_2_data,
                              f"{self.channel_type}: Ion Leaves GLU/ASN Cavity - Threshold={self.threshold}\u00c5",
                              output_dir / f"ion_leaves_cavity_threshold_crossing_threshold{self.threshold}.png")

        self.create_bar_plot(self.case_3_data,
                              f"{self.channel_type}: Ion Leaves HBC - Closest Residue",
                              output_dir / f"ion_leaves_hbc_closest_threshold{self.threshold}.png")

        self.create_bar_plot(self.case_4_data,
                              f"{self.channel_type}: Ion Leaves HBC - Threshold={self.threshold}\u00c5",
                              output_dir / f"ion_leaves_hbc_threshold_crossing_threshold{self.threshold}.png")

    def create_summary_table(self, output_dir=None):
        """
        Create a summary table showing counts, percentages, and average
        distances for each residue.
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

        sorted_residues = sorted(all_residues)

        total_1 = sum(v['count'] for v in self.case_1_data.values())
        total_2 = sum(v['count'] for v in self.case_2_data.values())
        total_3 = sum(v['count'] for v in self.case_3_data.values())
        total_4 = sum(v['count'] for v in self.case_4_data.values())

        def pct(count, total):
            return 100.0 * count / total if total > 0 else np.nan

        table_data = []
        for residue in sorted_residues:
            if '152' in residue or '141' in residue:
                res_type = 'GLU'
            elif '184' in residue:
                res_type = 'ASN'
            elif '173' in residue:
                res_type = 'ASP'
            else:
                res_type = 'Unknown'

            c1 = self.case_1_data[residue]['count'] if residue in self.case_1_data else 0
            c2 = self.case_2_data[residue]['count'] if residue in self.case_2_data else 0
            c3 = self.case_3_data[residue]['count'] if residue in self.case_3_data else 0
            c4 = self.case_4_data[residue]['count'] if residue in self.case_4_data else 0

            row = {
                'Residue_PDB': residue,
                'Type': res_type,
                'Cavity_Closest_Count': c1,
                'Cavity_Closest_Pct': pct(c1, total_1),
                'Cavity_Closest_Avg_Dist': np.mean(self.case_1_data[residue]['distances']) if residue in self.case_1_data and self.case_1_data[residue]['distances'] else np.nan,
                'Cavity_Threshold_Count': c2,
                'Cavity_Threshold_Pct': pct(c2, total_2),
                'Cavity_Threshold_Avg_Dist': np.mean(self.case_2_data[residue]['distances']) if residue in self.case_2_data and self.case_2_data[residue]['distances'] else np.nan,
                'HBC_Closest_Count': c3,
                'HBC_Closest_Pct': pct(c3, total_3),
                'HBC_Closest_Avg_Dist': np.mean(self.case_3_data[residue]['distances']) if residue in self.case_3_data and self.case_3_data[residue]['distances'] else np.nan,
                'HBC_Threshold_Count': c4,
                'HBC_Threshold_Pct': pct(c4, total_4),
                'HBC_Threshold_Avg_Dist': np.mean(self.case_4_data[residue]['distances']) if residue in self.case_4_data and self.case_4_data[residue]['distances'] else np.nan,
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)

        excel_file = output_dir / f"closest_residue_summary_threshold{self.threshold}.xlsx"
        df.to_excel(excel_file, index=False, float_format='%.2f')
        print(f"\n\u2713 Summary table saved: {excel_file}")

        csv_file = output_dir / f"closest_residue_summary_threshold{self.threshold}.csv"
        df.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"\u2713 Summary table saved: {csv_file}")

        print("\n" + "=" * 130)
        print(f"SUMMARY TABLE (Threshold={self.threshold}\u00c5)")
        print("=" * 130)
        print(df.to_string(index=False))
        print("=" * 130)

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
    parser.add_argument(
        "--channel_type",
        type=str,
        default=None,
        help="Channel type label for plot titles (e.g. G12, G12_ML). "
             "Defaults to the base_directory folder name if not given."
    )

    args = parser.parse_args()

    aggregator = ClosestResidueAggregator(args.base_directory, args.threshold, args.channel_type)

    if not aggregator.load_all_runs():
        print("No runs found. Exiting.")
        return

    print("\nGenerating outputs...")
    aggregator.create_all_plots(output_dir=args.output_dir)
    aggregator.create_summary_table(output_dir=args.output_dir)

    print("\n\u2713 Aggregation complete!")


if __name__ == "__main__":
    main()

# python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_closest_residue/3 --threshold 3
# python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/aggregate_closest_residue/3 --threshold 3

'''
python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregate_closest_residue/3 --threshold 3
python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/aggregate_closest_residue/3 --threshold 3
python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/aggregate_closest_residue/3 --threshold 3
python3 aggregate_closest_residue.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/aggregate_closest_residue/3 --threshold 3
'''