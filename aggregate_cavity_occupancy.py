import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from collections import Counter


class CavityOccupancyAggregator:
    """
    Aggregates cavity occupancy analysis results from multiple runs, for a
    single stage ("end_1", "end_2" or "end_3").

    stage="end_1" (new):
        reads cavity_occupancy_at_end_1.json + trajectory_ion_counts_end_1.csv

    stage="end_2" (default, original behaviour, unchanged filenames):
        reads cavity_occupancy_at_end2.json + trajectory_ion_counts.csv

    stage="end_3" (new):
        reads cavity_occupancy_at_end3.json + trajectory_ion_counts_end_3.csv

    Use aggregate_all_stages() at the bottom of this file to run all three
    automatically and produce three complete sets of outputs.
    """

    def __init__(self, base_directory, threshold=3.0, channel_type="G12", stage="end_2"):
        """
        Parameters:
        -----------
        base_directory : str or Path
            Base directory containing run subdirectories (RUN1, RUN2, etc.)
        threshold : float
            Threshold value used in the analysis
        channel_type : str
            Channel type ('G2', 'G12', 'G12_GAT', 'G12_ML','G12_D173N')
        stage : str
            "end_1" (SF), "end_2" (GLU/ASN cavity) or "end_3" (ASN->HBC region)
        """
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.channel_type = channel_type

        if stage not in ("end_1", "end_2", "end_3"):
            raise ValueError(f"Unknown stage '{stage}'. Use 'end_1', 'end_2' or 'end_3'.")
        self.stage = stage

        # File names + plot labels depend on the stage
        if stage == "end_1":
            self.results_json_name = "cavity_occupancy_at_end_1.json"
            self.traj_csv_name     = "trajectory_ion_counts_end_1.csv"
            self.filename_suffix   = "_end1"
            self.stage_label       = "ion leaves SF"
            self.region_label      = "selectivity filter (SF)"
        elif stage == "end_2":
            self.results_json_name = "cavity_occupancy_at_end_2.json"
            self.traj_csv_name     = "trajectory_ion_counts.csv"
            self.filename_suffix   = ""          # keep original filenames
            self.stage_label       = "ion leaves GLU/ASN cavity"
            self.region_label      = "GLU/ASN cavity"
        else:
            self.results_json_name = "cavity_occupancy_at_end_3.json"
            self.traj_csv_name     = "trajectory_ion_counts_end_3.csv"
            self.filename_suffix   = "_end3"
            self.stage_label       = "ion leaves HBC gate"
            self.region_label      = "ASN\u2192HBC region"

        # Convert threshold to string (handle 3.0 -> "3")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)

        self.all_results = []
        self.residue_combination_counts = Counter()
        self.subunit_combination_counts = Counter()
        self.residue_appearance_counts = Counter()
        self.subunit_appearance_counts = Counter()
        self.total_combinations = 0
        self.cavity_ion_counts = []          # counts at permeation frames — excludes permeating ion
        self.cavity_ion_counts_incl = []     # counts at permeation frames — includes permeating ion
        self.trajectory_ion_counts = []      # counts at every frame (all frames)
        self.combined_r2r3_traj_counts = []  # combined region 2+3 counts across all frames
        self.combined_r2r3_at_end3_excl = []  # combined region 2+3 counts at end_3 frames (excl. permeating ion)
        self.combined_r2r3_at_end3_incl = []  # combined region 2+3 counts at end_3 frames (incl. permeating ion)

    def convert_D_to_G1(self, label):
        """Convert subunit D to G1 in labels, but ONLY for non-G2 channels"""
        if self.channel_type == "G2":
            return label  # Don't convert for G2!

        if label and isinstance(label, str):
            # Replace D when it's at the end of subunit combinations
            label = label.replace('_D', '_G1')
            # Replace D when preceded by a dot (residue numbering)
            label = label.replace('.D', '.G1')
            # Replace standalone D
            if label == 'D':
                label = 'G1'
            # Replace D at the start of a combination
            if label.startswith('D_'):
                label = 'G1_' + label[2:]
            return label
        return label

    def _classify_residue(self, residue):
        """
        Return (type_label, color) for a residue, used for coloring the
        residue-percentage plot and for the 'Type' column in the summary
        table. The GLU/ASN/ASP PDB-number matching only applies to the
        cavity stages (end_2/end_3); for the SF stage (end_1) the residue
        set is different (selectivity-filter backbone residues), so every
        residue is labeled generically as 'SF' with a single color.
        """
        if self.stage == "end_1":
            return "SF", "mediumpurple"

        if '152' in residue or '141' in residue:
            return "GLU", "salmon"
        elif '184' in residue:
            return "ASN", "lightblue"
        elif '173' in residue:
            return "ASP", "lightgreen"
        else:
            return "Unknown", "gray"

    def load_all_runs(self):
        """Load cavity occupancy results from all subdirectories, for this stage."""
        print(f"\nSearching for runs in: {self.base_dir}  [stage={self.stage}]")

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

            # Look for cavity_occupancy folder, stage-specific results file
            json_file = run_dir / "cavity_occupancy" / self.results_json_name

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
                        # Including the permeating ion itself
                        incl = event.get('total_ions_including_permeating',
                                         event['total_ions_in_cavity'] + 1)
                        self.cavity_ion_counts_incl.append(incl)

                    loaded_count += 1
                    print(f"  \u2713 {run_name}: Loaded ({len(results)} events)")

                except Exception as e:
                    print(f"  \u2717 {run_name}: Error loading: {e}")
            else:
                print(f"  - {run_name}: {self.results_json_name} not found, skipping")

            # Load trajectory ion counts (every frame) — separate from permeation events
            traj_csv = run_dir / "cavity_occupancy" / self.traj_csv_name
            if traj_csv.exists():
                try:
                    import csv as _csv
                    with open(traj_csv, newline="") as f:
                        for row in _csv.DictReader(f):
                            self.trajectory_ion_counts.append(int(row["n_total"]))
                except Exception as e:
                    print(f"  \u2717 {run_name}: Error loading trajectory counts: {e}")

            # Load combined region 2+3 trajectory counts (only meaningful for end_2/end_3)
            if self.stage != "end_1":
                combined_traj_csv = run_dir / "cavity_occupancy" / "trajectory_ion_counts_combined_r2r3.csv"
                if combined_traj_csv.exists():
                    try:
                        import csv as _csv
                        with open(combined_traj_csv, newline="") as f:
                            for row in _csv.DictReader(f):
                                self.combined_r2r3_traj_counts.append(int(row["n_total_combined"]))
                    except Exception as e:
                        print(f"  \u2717 {run_name}: Error loading combined r2r3 trajectory: {e}")

                # Load combined region 2+3 counts at end_3 frames
                combined_end3_csv = run_dir / "cavity_occupancy" / "combined_r2r3_at_end3.csv"
                if combined_end3_csv.exists():
                    try:
                        import csv as _csv
                        with open(combined_end3_csv, newline="") as f:
                            for row in _csv.DictReader(f):
                                # support both old column name and new split columns
                                if "n_total_combined_excl" in row:
                                    excl = int(row["n_total_combined_excl"])
                                    incl = int(row["n_total_combined_incl"])
                                else:
                                    excl = int(row["n_total_combined"])
                                    incl = excl + 1
                                self.combined_r2r3_at_end3_excl.append(excl)
                                self.combined_r2r3_at_end3_incl.append(incl)
                    except Exception as e:
                        print(f"  \u2717 {run_name}: Error loading combined r2r3 at end_3: {e}")

        print(f"\nSuccessfully loaded: {loaded_count} runs")
        print(f"Total events: {len(self.all_results)}")

        return loaded_count > 0

    def create_subunit_combination_plot(self, output_dir, channel):
        """Create bar plot of subunit combination frequencies."""
        if not self.subunit_combination_counts:
            print("No subunit combination data")
            return

        # Get top 15 combinations
        top_combos = self.subunit_combination_counts.most_common(15)

        if channel == "G2":
            combos = [c[0] for c in top_combos]  # Convert D to G1
        else:
            combos = [self.convert_D_to_G1(c[0]) for c in top_combos]  # Convert D to G1
        counts = [c[1] for c in top_combos]

        # Width scales with number of combinations so labels never collide
        fig_width = max(6, len(combos) * 0.75)
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))

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
                     edgecolor='black', linewidth=1.5, width=0.7)

        # Add count + percentage labels — percentage is out of total events,
        # matching the same denominator used in the summary table
        total_events = len(self.all_results) if self.all_results else sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            pct = 100.0 * count / total_events if total_events > 0 else 0.0
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.02,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('Frequency', fontsize=15, fontweight='bold')
        ax.set_xlabel('Subunit Combination', fontsize=17, fontweight='bold')
        ax.set_title(f'{self.channel_type}: Subunit combinations when {self.stage_label}',
                    fontsize=15, fontweight='bold', pad=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)

        # Add gap at top — extra room for the two-line count+percentage labels
        max_count = max(counts)
        ax.set_ylim(0, max_count * 1.28)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='1 subunit'),
            Patch(facecolor='lightgreen', edgecolor='black', label='2 subunits'),
            Patch(facecolor='orange', edgecolor='black', label='3 subunits'),
            Patch(facecolor='salmon', edgecolor='black', label='4 subunits')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=15)

        plt.tight_layout()
        output_file = output_dir / f"subunit_combinations_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\u2713 Subunit combinations plot saved: {output_file}")

    def create_residue_percentage_plot(self, output_dir):
        """Create bar plot showing % of times each residue appears."""
        if not self.residue_appearance_counts or self.total_combinations == 0:
            print("No residue appearance data")
            return

        # Calculate percentages
        residue_percentages = {}
        for residue, count in self.residue_appearance_counts.items():
            residue_percentages[residue] = (count / self.total_combinations) * 100

        # Sort by percentage (descending)
        sorted_residues = sorted(residue_percentages.items(), key=lambda x: x[1], reverse=True)

        residues = [self.convert_D_to_G1(r[0]) for r in sorted_residues]  # Convert D to G1
        percentages = [r[1] for r in sorted_residues]

        # Width scales with number of residues so bars/labels have room to breathe
        fig_width = max(6, len(residues) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))

        # Color by residue type (SF residues are all one generic type/color)
        colors = [self._classify_residue(res)[1] for res in residues]

        bars = ax.bar(range(len(residues)), percentages, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.7)

        # Add percentage labels — sized to avoid collisions between neighboring bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Reference lines
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='100%')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='50%')

        ax.set_xticks(range(len(residues)))
        ax.set_xticklabels(residues, fontsize=13, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('% of events', fontsize=15, fontweight='bold')
        ax.set_xlabel('Residue', fontsize=17, fontweight='bold')
        ax.set_title(f'{self.channel_type}: Residue frequency when {self.stage_label}',
                    fontsize=15, fontweight='bold', pad=14)
        ax.set_ylim([0, 118])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=13)

        # Legend — SF stage has a single generic residue type, so use one
        # legend entry instead of the GLU/ASN/ASP breakdown used elsewhere.
        from matplotlib.patches import Patch
        if self.stage == "end_1":
            legend_elements = [
                Patch(facecolor='mediumpurple', edgecolor='black', label='SF residue'),
                plt.Line2D([0], [0], color='green', linestyle='--', label='100%'),
                plt.Line2D([0], [0], color='orange', linestyle='--', label='50%')
            ]
        else:
            legend_elements = [
                Patch(facecolor='salmon', edgecolor='black', label='GLU'),
                Patch(facecolor='lightblue', edgecolor='black', label='ASN'),
                Patch(facecolor='lightgreen', edgecolor='black', label='ASP'),
                plt.Line2D([0], [0], color='green', linestyle='--', label='100%'),
                plt.Line2D([0], [0], color='orange', linestyle='--', label='50%')
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=15)

        plt.tight_layout()
        output_file = output_dir / f"residue_appearance_percentage_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\u2713 Residue % plot saved: {output_file}")

    def create_subunit_percentage_plot(self, output_dir):
        """Create bar plot showing % of times each subunit appears."""
        if not self.subunit_appearance_counts or self.total_combinations == 0:
            print("No subunit appearance data")
            return

        # Calculate percentages
        subunit_percentages = {}
        for subunit, count in self.subunit_appearance_counts.items():
            subunit_percentages[subunit] = (count / self.total_combinations) * 100

        # Sort alphabetically
        sorted_subunits = sorted(subunit_percentages.items())

        subunits = [self.convert_D_to_G1(s[0]) for s in sorted_subunits]  # Convert D to G1
        percentages = [s[1] for s in sorted_subunits]

        # Narrower plot
        fig, ax = plt.subplots(figsize=(5, 4))

        # Different colors for each subunit
        colors = ['steelblue', 'coral', 'lightgreen', 'plum'][:len(subunits)]

        bars = ax.bar(range(len(subunits)), percentages, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5, width=0.6)

        # Add percentage labels - bigger font
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')

        # Reference lines
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='100%')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='50%')

        ax.set_xticks(range(len(subunits)))
        ax.set_xticklabels(subunits, fontsize=14, fontweight='bold')
        ax.set_ylabel('% of events', fontsize=15, fontweight='bold')
        ax.set_xlabel('Subunit', fontsize=17, fontweight='bold')
        ax.set_title(f'{self.channel_type}: Subunit frequency when {self.stage_label}',
                    fontsize=15, fontweight='bold', pad=10)
        ax.set_ylim([0, 110])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(fontsize=15, loc='upper right')

        plt.tight_layout()
        output_file = output_dir / f"subunit_appearance_percentage_t{self.threshold_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\u2713 Subunit % plot saved: {output_file}")

    def _ion_count_bar_plot(self, counts, title, ylabel, output_file,
                            annotation_fmt="{:.0f}", color="steelblue"):
        """Helper: single bar plot of ion count distribution.
        If annotation_fmt contains '%', y-axis shows % instead of raw counts.
        The channel type is automatically prepended to the title here, so
        every caller of this helper gets it for free.
        """
        ion_count_freq = Counter(counts)
        sorted_items   = sorted(ion_count_freq.items())
        ion_counts     = [c[0] for c in sorted_items]
        raw_freqs      = [c[1] for c in sorted_items]
        total          = len(counts)
        mean_ions      = np.mean(counts)

        # decide whether to plot raw counts or percentages
        use_pct = "%" in annotation_fmt
        values  = [100.0 * f / total for f in raw_freqs] if use_pct else raw_freqs

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(ion_counts, values, color=color, alpha=0.7,
                      edgecolor='black', linewidth=1.2, width=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    annotation_fmt.format(val),
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.axvline(x=mean_ions, color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {mean_ions:.2f}')

        ax.set_xlabel('Number of K\u207a ions in cavity', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(f'{self.channel_type}: {title}', fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(ion_counts)
        ax.set_xticklabels([str(n) for n in ion_counts], fontsize=14)
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.18)

        # Place legend and stats box on whichever side has the shorter bar
        tallest_idx = values.index(max(values))
        right_side = tallest_idx < len(values) / 2  # tallest bar is on left → boxes go right
        if right_side:
            legend_anchor = (0.98, 0.98)
            stats_x, stats_ha = 0.98, 'right'
            stats_y = 0.84
        else:
            legend_anchor = (0.02, 0.98)
            stats_x, stats_ha = 0.02, 'left'
            stats_y = 0.84

        ax.legend(fontsize=12, loc='upper left' if not right_side else 'upper right',
                  bbox_to_anchor=legend_anchor)



        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\u2713 Saved: {output_file}")

    def create_cavity_ion_count_plot(self, output_dir):
        """Four plots for permeation-frame ion counts:
           1. Raw counts — excluding permeating ion
           2. % — excluding permeating ion
           3. Raw counts — including permeating ion
           4. % — including permeating ion
        """
        if not self.cavity_ion_counts:
            print("No cavity ion count data")
            return

        n = len(self.cavity_ion_counts)

        # --- excluding permeating ion ---
        self._ion_count_bar_plot(
            self.cavity_ion_counts,
            title=f"K\u207a ions in {self.region_label}\nwhen {self.stage_label} (excl. permeating ion)",
            ylabel="Number of permeation events",
            output_file=output_dir / f"cavity_ion_counts_excl_t{self.threshold_str}.png",
            annotation_fmt="{:.0f}", color="steelblue")

        self._ion_count_bar_plot(
            self.cavity_ion_counts,
            title=f"K\u207a ions in {self.region_label}\nwhen {self.stage_label} \u2014 % (excl. permeating ion)",
            ylabel="% of permeation events",
            output_file=output_dir / f"cavity_ion_counts_excl_pct_t{self.threshold_str}.png",
            annotation_fmt="{:.1f}%", color="steelblue")

        # --- including permeating ion ---
        if self.cavity_ion_counts_incl:
            self._ion_count_bar_plot(
                self.cavity_ion_counts_incl,
                title=f"K\u207a ions in {self.region_label}\nwhen {self.stage_label}",
                ylabel="Number of permeation events",
                output_file=output_dir / f"cavity_ion_counts_incl_t{self.threshold_str}.png",
                annotation_fmt="{:.0f}", color="coral")

            self._ion_count_bar_plot(
                self.cavity_ion_counts_incl,
                title=f"K\u207a ions in {self.region_label}\nwhen {self.stage_label} \u2014 %",
                ylabel="% of permeation events",
                output_file=output_dir / f"cavity_ion_counts_incl_pct_t{self.threshold_str}.png",
                annotation_fmt="{:.1f}%", color="coral")

    def create_trajectory_ion_count_plots(self, output_dir):
        """Two separate plots for ALL-frames ion count distribution:
           1. Raw frame counts
           2. % of total frames
        """
        if not self.trajectory_ion_counts:
            print("No trajectory ion count data — skipping all-frames plot")
            return

        self._ion_count_bar_plot(
            self.trajectory_ion_counts,
            title=f"K\u207a occupancy in {self.region_label}\nacross all simulation frames",
            ylabel="Number of frames",
            output_file=output_dir / f"trajectory_ion_counts_all_frames.png",
            annotation_fmt="{:.0f}", color="#4878CF")

        self._ion_count_bar_plot(
            self.trajectory_ion_counts,
            title=f"K\u207a occupancy in {self.region_label}\nacross all simulation frames \u2014 %",
            ylabel="% of all frames",
            output_file=output_dir / f"trajectory_ion_counts_all_frames_pct.png",
            annotation_fmt="{:.1f}%", color="#4878CF")

    def create_combined_r2r3_trajectory_plot(self, output_dir):
        """Plots of combined region 2+3 ion occupancy across ALL frames.
        Not applicable to the SF stage (end_1) — only run for end_2/end_3."""
        if self.stage == "end_1":
            return
        if not self.combined_r2r3_traj_counts:
            print("No combined r2r3 trajectory data — skipping")
            return

        self._ion_count_bar_plot(
            self.combined_r2r3_traj_counts,
            title="K\u207a ions in GLU/HBC region\nacross all simulation frames",
            ylabel="Number of frames",
            output_file=output_dir / "trajectory_ion_counts_all_frames.png",
            annotation_fmt="{:.0f}", color="#8E44AD")

        self._ion_count_bar_plot(
            self.combined_r2r3_traj_counts,
            title="K\u207a ions in GLU/HBC region\nacross all simulation frames \u2014 %",
            ylabel="% of all frames",
            output_file=output_dir / "trajectory_ion_counts_all_frames_pct.png",
            annotation_fmt="{:.1f}%", color="#8E44AD")

    def create_combined_r2r3_at_end3_plot(self, output_dir):
        """Plots of combined region 2+3 ion counts at end_3 permeation frames.
        Not applicable to the SF stage (end_1) — only run for end_2/end_3."""
        if self.stage == "end_1":
            return
        if not self.combined_r2r3_at_end3_excl:
            print("No combined r2r3 at end_3 data — skipping")
            return

        # excl plots
        self._ion_count_bar_plot(
            self.combined_r2r3_at_end3_excl,
            title="K\u207a ions in GLU/HBC region\nwhen ion leaves HBC gate (excl. permeating ion)",
            ylabel="Number of permeation events",
            output_file=output_dir / "ion_count_at_end3_excl.png",
            annotation_fmt="{:.0f}", color="#E74C3C")

        self._ion_count_bar_plot(
            self.combined_r2r3_at_end3_excl,
            title="K\u207a ions in GLU/HBC region\nwhen ion leaves HBC gate \u2014 % (excl. permeating ion)",
            ylabel="% of permeation events",
            output_file=output_dir / "ion_count_at_end3_excl_pct.png",
            annotation_fmt="{:.1f}%", color="#E74C3C")

        # incl plots
        self._ion_count_bar_plot(
            self.combined_r2r3_at_end3_incl,
            title="K\u207a ions in GLU/HBC region\nwhen ion leaves HBC gate",
            ylabel="Number of permeation events",
            output_file=output_dir / "ion_count_at_end3_incl.png",
            annotation_fmt="{:.0f}", color="#C0392B")

        self._ion_count_bar_plot(
            self.combined_r2r3_at_end3_incl,
            title="K\u207a ions in GLU/HBC region\nwhen ion leaves HBC gate \u2014 %",
            ylabel="% of permeation events",
            output_file=output_dir / "ion_count_at_end3_incl_pct.png",
            annotation_fmt="{:.1f}%", color="#C0392B")


    def create_all_plots(self, output_dir=None, channel=None):
        """Create all plots."""
        if output_dir is None:
            output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage-specific subfolder
        if self.stage == "end_1":
            stage_dir = output_dir / "sf"
        elif self.stage == "end_2":
            stage_dir = output_dir / "glu_asn"
        else:
            stage_dir = output_dir / "asn_hbc"
        stage_dir.mkdir(parents=True, exist_ok=True)
        combined_dir = output_dir / "glu_hbc"
        combined_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating plots [stage={self.stage}] -> {stage_dir.name}/")

        self.create_subunit_combination_plot(stage_dir, channel)
        self.create_residue_percentage_plot(stage_dir)
        self.create_subunit_percentage_plot(stage_dir)
        self.create_cavity_ion_count_plot(stage_dir)
        self.create_trajectory_ion_count_plots(stage_dir)
        # combined region 2+3 plots only apply to end_2/end_3
        self.create_combined_r2r3_trajectory_plot(combined_dir)
        self.create_combined_r2r3_at_end3_plot(combined_dir)

    def create_summary_table(self, output_dir=None):
        """Create summary tables. output_dir should be the stage subfolder."""
        if output_dir is None:
            if self.stage == "end_1":
                output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}" / "sf"
            elif self.stage == "end_2":
                output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}" / "glu_asn"
            else:
                output_dir = self.base_dir / f"aggregated_cavity_t{self.threshold_str}" / "asn_hbc"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Table 1: Residue appearance percentages
        residue_table = []
        for residue, count in sorted(self.residue_appearance_counts.items(),
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / self.total_combinations) * 100 if self.total_combinations > 0 else 0

            # Determine type (generic 'SF' label for the selectivity-filter stage)
            res_type = self._classify_residue(residue)[0]

            residue_table.append({
                'Residue_PDB': self.convert_D_to_G1(residue),  # Convert D to G1
                'Type': res_type,
                'Appearances': count,
                '%': percentage
            })

        df_residue = pd.DataFrame(residue_table)
        excel_file = output_dir / f"residue_appearances_t{self.threshold_str}.xlsx"
        df_residue.to_excel(excel_file, index=False, float_format='%.1f')
        print(f"\n\u2713 Residue appearances table: {excel_file}")

        csv_file = output_dir / f"residue_appearances_t{self.threshold_str}.csv"
        df_residue.to_csv(csv_file, index=False, float_format='%.1f')
        print(f"\u2713 Residue appearances CSV: {csv_file}")

        # Table 2: Subunit combinations
        subunit_table = []
        for combo, count in self.subunit_combination_counts.most_common():
            subunit_table.append({
                'Subunit_Combination': self.convert_D_to_G1(combo),  # Convert D to G1
                'Frequency': count,
                '%': (count / len(self.all_results)) * 100 if self.all_results else 0
            })

        df_subunit = pd.DataFrame(subunit_table)
        excel_file = output_dir / f"subunit_combinations_t{self.threshold_str}.xlsx"
        df_subunit.to_excel(excel_file, index=False, float_format='%.1f')
        print(f"\u2713 Subunit combinations table: {excel_file}")

        csv_file = output_dir / f"subunit_combinations_t{self.threshold_str}.csv"
        df_subunit.to_csv(csv_file, index=False, float_format='%.1f')
        print(f"\u2713 Subunit combinations CSV: {csv_file}")

        # Print to console
        print("\n" + "="*80)
        print(f"RESIDUE APPEARANCE % [{self.stage}]")
        print("="*80)
        print(df_residue.to_string(index=False))

        print("\n" + "="*80)
        print(f"TOP SUBUNIT COMBINATIONS [{self.stage}]")
        print("="*80)
        print(df_subunit.head(20).to_string(index=False))
        print("="*80)


def aggregate_all_stages(base_directory, threshold=3.0, channel_type="G12", output_dir=None):
    """
    Run the full aggregation pipeline for ALL THREE stages (end_1, end_2,
    end_3) automatically, writing all sets of outputs into the same
    output_dir.

    end_1 outputs go into output_dir/sf/       (e.g. cavity_ion_counts_incl_t3.png)
    end_2 outputs go into output_dir/glu_asn/  (original filenames, unchanged)
    end_3 outputs go into output_dir/asn_hbc/  (e.g. cavity_ion_counts_incl_t3.png)

    Returns
    -------
    dict : {"end_1": CavityOccupancyAggregator_or_None,
            "end_2": CavityOccupancyAggregator_or_None,
            "end_3": CavityOccupancyAggregator_or_None}
    """
    base_directory = Path(base_directory)
    if output_dir is None:
        if isinstance(threshold, float) and threshold.is_integer():
            threshold_str = str(int(threshold))
        else:
            threshold_str = str(threshold)
        output_dir = base_directory / f"aggregated_cavity_t{threshold_str}"
    else:
        output_dir = Path(output_dir)

    results = {"end_1": None, "end_2": None, "end_3": None}

    print("\n" + "#"*80)
    print("# STAGE: end_1 (selectivity filter, SF)")
    print("#"*80)
    agg_end1 = CavityOccupancyAggregator(base_directory, threshold, channel_type, stage="end_1")
    if agg_end1.load_all_runs():
        agg_end1.create_all_plots(output_dir=output_dir, channel=channel_type)
        agg_end1.create_summary_table(output_dir=output_dir / "sf")
        results["end_1"] = agg_end1
    else:
        print("No end_1 data found — skipping end_1 (SF) aggregation.")

    print("\n" + "#"*80)
    print("# STAGE: end_2 (GLU/ASN cavity)")
    print("#"*80)
    agg_end2 = CavityOccupancyAggregator(base_directory, threshold, channel_type, stage="end_2")
    if agg_end2.load_all_runs():
        agg_end2.create_all_plots(output_dir=output_dir, channel=channel_type)
        agg_end2.create_summary_table(output_dir=output_dir / "glu_asn")
        results["end_2"] = agg_end2
    else:
        print("No end_2 data found — skipping end_2 aggregation.")

    print("\n" + "#"*80)
    print("# STAGE: end_3 (ASN\u2192HBC region)")
    print("#"*80)
    agg_end3 = CavityOccupancyAggregator(base_directory, threshold, channel_type, stage="end_3")
    if agg_end3.load_all_runs():
        agg_end3.create_all_plots(output_dir=output_dir, channel=channel_type)
        agg_end3.create_summary_table(output_dir=output_dir / "asn_hbc")
        results["end_3"] = agg_end3
    else:
        print("No end_3 data found — skipping end_3 aggregation.")

    return results


# Kept for backward compatibility with any existing scripts/imports.
def aggregate_both_stages(base_directory, threshold=3.0, channel_type="G12", output_dir=None):
    """
    Deprecated alias for aggregate_all_stages() — kept so existing calls
    (which expected only end_2/end_3) keep working. New code should call
    aggregate_all_stages() directly.
    """
    all_results = aggregate_all_stages(base_directory, threshold, channel_type, output_dir)
    return {"end_2": all_results["end_2"], "end_3": all_results["end_3"]}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cavity occupancy analysis from multiple runs (end_1, end_2 AND end_3 automatically)"
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
        "-c", "--channel",
        type=str,
        default="G12",
        choices=['G2', 'G12', 'G12_GAT', 'G12_ML', 'G12_D173N', 'G12_E141Q'],
        help="Channel type (default: G12). Use G2 to keep D labels, G12/G12_GAT/G12_ML to convert D to G1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and tables (default: base_directory/aggregated_cavity)"
    )

    args = parser.parse_args()

    results = aggregate_all_stages(
        base_directory=args.base_directory,
        threshold=args.threshold,
        channel_type=args.channel,
        output_dir=args.output_dir,
    )

    if results["end_1"] is None and results["end_2"] is None and results["end_3"] is None:
        print("\nNo data found for any stage. Exiting.")
        return

    print(f"\n\u2713 Aggregation complete!")


if __name__ == "__main__":
    main()

# USAGE (now aggregates end_1, end_2 AND end_3 automatically):
# python aggregate_cavity_occupancy.py /path/to/G12 --threshold 3.0
#
# Or with custom output:
# python aggregate_cavity_occupancy.py /path/to/G12 --threshold 3.0 --output_dir /path/to/output
'''
 python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/cavity_occupancy

  python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/cavity_occupancy

  python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/cavity_occupancy

   python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/cavity_occupancy --channel G2
'''

'''
 python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/cavity_occupancy

  python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/cavity_occupancy

  python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/cavity_occupancy

   python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/cavity_occupancy --channel G2

    python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/cavity_occupancy --channel G12_D173N

    python3 aggregate_cavity_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q --threshold 3.0 \
 --output_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/cavity_occupancy --channel G12_E141Q

 

'''