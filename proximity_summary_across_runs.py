import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
import pandas as pd

# ─────────────────────────────────────────────────────────────
#  Residue-name handling
# ─────────────────────────────────────────────────────────────
#
# Residue labels are now read from the AMBER topology instead of being
# inferred from which list a residue number happened to land in. This
# matters for mutant systems (E141Q, D173N, ...) where the residue in a
# given position is no longer the wild-type amino acid, and it removes
# the old hardcoded "resid 1105 is ASP" special case, which only fired
# when channel_type was exactly "G12" and therefore mislabelled that
# residue in every G12_* variant.

# Protonation / disulfide variants -> canonical display name
RESNAME_CANONICAL = {
    "GLH": "GLU", "ASH": "ASP",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "LYN": "LYS", "CYX": "CYS", "CYM": "CYS",
    "ARN": "ARG", "TYM": "TYR",
}


def read_residue_labels(prmtop_path):
    """
    Parse the RESIDUE_LABEL section of an AMBER prmtop.

    Returns a list of residue names where index 0 corresponds to AMBER
    resid 1, or None if the section could not be read.

    Plain text parsing is used so the script has no hard dependency on
    parmed; parmed is tried as a fallback for non-standard topologies.
    """
    prmtop_path = Path(prmtop_path)
    labels = []

    try:
        in_section = False
        with open(prmtop_path) as f:
            for line in f:
                if line.startswith("%FLAG"):
                    parts = line.split()
                    in_section = len(parts) > 1 and parts[1] == "RESIDUE_LABEL"
                    continue
                if line.startswith("%"):
                    continue
                if in_section:
                    stripped = line.rstrip("\n")
                    # FORTRAN format is 20a4: fixed 4-character fields
                    for i in range(0, len(stripped), 4):
                        name = stripped[i:i + 4].strip()
                        if name:
                            labels.append(name)
    except OSError as e:
        print(f"Warning: could not open topology {prmtop_path}: {e}")
        return None

    if labels:
        return labels

    # Fallback for topologies the simple parser can't handle
    try:
        import parmed
        top = parmed.load_file(str(prmtop_path))
        return [res.name for res in top.residues]
    except Exception as e:
        print(f"Warning: could not read residue names from {prmtop_path}: {e}")
        return None


def autodetect_prmtop(base_dir):
    """
    Look for a single topology file in base_dir or its parent.
    Returns a Path, or None if there isn't exactly one candidate.
    """
    base_dir = Path(base_dir)
    candidates = []
    for d in (base_dir, base_dir.parent):
        for pattern in ("*.prmtop", "*.parm7", "*.top"):
            candidates.extend(d.glob(pattern))
    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"Warning: found {len(candidates)} topology files near {base_dir}; "
              f"pass --prmtop explicitly to pick one:")
        for c in candidates:
            print(f"    {c}")
    return None


class ProximitySummaryAcrossRuns:
    def __init__(self, base_results_dir, channel_type, output_dir=None, prmtop=None):
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
        prmtop : str or Path, optional
            AMBER topology used to look up the ACTUAL residue names.
            If None, the script tries to auto-detect one near
            base_results_dir; if that fails it falls back to the old
            group-based naming and says so loudly.
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

        # ---- Residue names from the topology ----
        self.prmtop = Path(prmtop) if prmtop else autodetect_prmtop(self.base_dir)
        self.residue_labels = None
        if self.prmtop and self.prmtop.is_file():
            self.residue_labels = read_residue_labels(self.prmtop)
            if self.residue_labels:
                print(f"Residue names read from topology: {self.prmtop} "
                      f"({len(self.residue_labels)} residues)")
            else:
                print(f"Warning: topology {self.prmtop} gave no residue names.")
        else:
            if prmtop:
                print(f"Warning: topology not found: {prmtop}")

        if not self.residue_labels:
            print("*" * 70)
            print("WARNING: no topology available. Residue labels will fall back to")
            print("         group-based naming (GLU for the first list, ASN for the")
            print("         second). For mutant systems these labels will be WRONG.")
            print("         Pass --prmtop /path/to/system.prmtop to fix this.")
            print("*" * 70)

        # Storage for all runs
        self.run_data = {}
        self.glu_residues = None
        self.asn_residues = None

        # Display names for the two groups, filled in after loading
        self.glu_group_name = "GLU"
        self.asn_group_name = "ASN/ASP"

    # ─────────────────────────────────────────────────────────
    #  Residue naming
    # ─────────────────────────────────────────────────────────

    def get_resname(self, resid):
        """
        Actual residue name from the topology, canonicalised (GLH -> GLU etc.).
        Returns None if no topology is loaded or the resid is out of range.
        """
        if not self.residue_labels:
            return None
        if 1 <= resid <= len(self.residue_labels):
            raw = self.residue_labels[resid - 1].upper()
            return RESNAME_CANONICAL.get(raw, raw)
        print(f"Warning: resid {resid} is outside the topology "
              f"(1-{len(self.residue_labels)})")
        return None

    def get_residue_type(self, resid, is_glu_group):
        """Residue type string for the CSV 'Type' column."""
        name = self.get_resname(resid)
        if name is not None:
            return name
        # Legacy fallback, kept only for the no-topology case
        return "GLU" if is_glu_group else "ASN"

    def get_residue_label(self, resid, is_glu_group):
        """Get the residue label: ACTUAL residue name + PDB numbering."""
        pdb_num = self.convert_to_pdb(resid, self.channel_type)
        return f"{self.get_residue_type(resid, is_glu_group)} {pdb_num}"

    def get_group_name(self, residues, is_glu_group):
        """
        Display name for a whole group, built from the residue names actually
        present. A wild-type group gives "GLU"; a partially mutated one gives
        something like "GLN/GLU", which is a visible signal that the group is
        no longer homogeneous.
        """
        names = sorted({self.get_residue_type(r, is_glu_group) for r in residues})
        return "/".join(names)

    def validate_residue_groups(self):
        """
        Compare what the topology says against what the group membership
        implies, and report any disagreement. This is where mutations show
        up: a residue sitting in 'glu_residues' that the topology calls GLN
        means the upstream analysis may also be measuring the wrong thing,
        not just labelling it wrongly.
        """
        if not self.residue_labels:
            return

        mismatches = []
        for resid in self.glu_residues:
            name = self.get_resname(resid)
            if name is not None and name != "GLU":
                mismatches.append((resid, "glu_residues", "GLU", name))
        for resid in self.asn_residues:
            name = self.get_resname(resid)
            if name is not None and name not in ("ASN", "ASP"):
                mismatches.append((resid, "asn_residues", "ASN/ASP", name))

        if not mismatches:
            print("Residue-name check: all residues match their group "
                  "(GLU in the GLU group, ASN/ASP in the ASN group).")
            return

        print("\n" + "!" * 70)
        print("RESIDUE-NAME MISMATCH -- the topology disagrees with group membership:")
        for resid, group, expected, actual in mismatches:
            pdb_num = self.convert_to_pdb(resid, self.channel_type)
            print(f"  resid {resid:>5} ({pdb_num:<10}) is in {group:<14} "
                  f"expected {expected:<8} but topology says {actual}")
        print("")
        print("  Plots and tables will use the TOPOLOGY names, which is correct")
        print("  for labelling. But check that the upstream script writing")
        print("  proximity_summary.json picked these residue numbers from the")
        print("  structure and not from a hardcoded wild-type list -- if they")
        print("  are hardcoded, the counts themselves may refer to the wrong")
        print("  atoms, not merely the wrong name.")
        print("!" * 70 + "\n")

    # ─────────────────────────────────────────────────────────
    #  Data loading
    # ─────────────────────────────────────────────────────────

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

        if self.glu_residues is not None:
            self.glu_group_name = self.get_group_name(self.glu_residues, True)
            self.asn_group_name = self.get_group_name(self.asn_residues, False)
            self.validate_residue_groups()

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

    # ─────────────────────────────────────────────────────────
    #  Statistics
    # ─────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────
    #  Plots
    # ─────────────────────────────────────────────────────────

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
            Patch(facecolor='red', alpha=0.5, edgecolor='black', label=self.glu_group_name),
            Patch(facecolor='blue', alpha=0.5, edgecolor='black', label=self.asn_group_name),
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
        """Create separate bar plots for each group with error bars and individual run points."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # GLU-group plot
        glu_means = [stats['glu'][resid]['mean'] for resid in self.glu_residues]
        glu_stds = [stats['glu'][resid]['std'] for resid in self.glu_residues]
        glu_values = [stats['glu'][resid]['values'] for resid in self.glu_residues]
        glu_labels = [self.get_residue_label(resid, True).replace(' ', '\n')
                    for resid in self.glu_residues]

        x1 = range(len(self.glu_residues))
        bars1 = ax1.bar(x1, glu_means, yerr=glu_stds, color='red', alpha=0.5,
                        edgecolor='black', linewidth=1.5, capsize=5,
                        error_kw={'linewidth': 2, 'ecolor': 'black'})

        # Add individual points for the GLU group
        for i, (x_pos, values) in enumerate(zip(x1, glu_values)):
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax1.scatter(x_jitter, values, color='black', s=60, alpha=0.7,
                    zorder=5, edgecolors='white', linewidth=0.5)

        ax1.set_xticks(x1)
        ax1.set_xticklabels(glu_labels, fontsize=18)  # Increased
        ax1.set_ylabel('Percentage of Frames (%)', fontsize=20)  # Increased
        ax1.set_title(f'{self.channel_type}: {self.glu_group_name} Residues - Summary '
                      f'(n={len(self.run_data)} runs)',
                    fontsize=20, fontweight='bold')  # Increased
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='y', labelsize=16)  # Increased

        for bar, mean, std in zip(bars1, glu_means, glu_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.1f}%\n\u00b1{std:.1f}',
                    ha='center', va='bottom', fontsize=15, fontweight='bold')  # Increased and bold

        # ASN-group plot
        asn_means = [stats['asn'][resid]['mean'] for resid in self.asn_residues]
        asn_stds = [stats['asn'][resid]['std'] for resid in self.asn_residues]
        asn_values = [stats['asn'][resid]['values'] for resid in self.asn_residues]
        asn_labels = [self.get_residue_label(resid, False).replace(' ', '\n')
                    for resid in self.asn_residues]

        x2 = range(len(self.asn_residues))
        bars2 = ax2.bar(x2, asn_means, yerr=asn_stds, color='blue', alpha=0.5,
                        edgecolor='black', linewidth=1.5, capsize=5,
                        error_kw={'linewidth': 2, 'ecolor': 'black'})

        # Add individual points for the ASN group
        for i, (x_pos, values) in enumerate(zip(x2, asn_values)):
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax2.scatter(x_jitter, values, color='black', s=60, alpha=0.7,
                    zorder=5, edgecolors='white', linewidth=0.5)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(asn_labels, fontsize=18)  # Increased
        ax2.set_ylabel('Percentage of Frames (%)', fontsize=20)  # Increased
        ax2.set_title(f'{self.channel_type}: {self.asn_group_name} Residues - Summary '
                      f'(n={len(self.run_data)} runs)',
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

        categories = [f'Any {self.glu_group_name}',
                      f'Any {self.asn_group_name}',
                      'Any Residue']
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
            Patch(facecolor='red', alpha=0.5, edgecolor='black', label=self.glu_group_name),
            Patch(facecolor='blue', alpha=0.5, edgecolor='black', label=self.asn_group_name),
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

        categories = [f'Any {self.glu_group_name}',
                      f'Any {self.asn_group_name}',
                      'Any Residue']
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

    # ─────────────────────────────────────────────────────────
    #  Tables
    # ─────────────────────────────────────────────────────────

    def save_pooled_statistics_table(self, pooled):
        """Save pooled statistics to CSV."""
        rows = []

        for resid in self.glu_residues:
            rows.append({
                'Residue': self.get_residue_label(resid, True),
                'Type': self.get_residue_type(resid, True),
                'Pooled_Count': pooled['glu'][resid]['count'],
                'Pooled_Percentage (%)': pooled['glu'][resid]['percentage'],
            })

        for resid in self.asn_residues:
            rows.append({
                'Residue': self.get_residue_label(resid, False),
                'Type': self.get_residue_type(resid, False),
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
            f.write(f"Residue names from: "
                    f"{self.prmtop if self.residue_labels else 'NO TOPOLOGY (group-based fallback)'}\n")
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
            for key, label in [('percentage_any_glu', f'Any {self.glu_group_name}'),
                                ('percentage_any_asn', f'Any {self.asn_group_name}'),
                                ('percentage_any_residue', 'Any Residue')]:
                f.write(f"{label:<20} Pooled %: {pooled['aggregate'][key]:.2f}%\n")

        print(f"Pooled statistics text file saved to: {txt_file}")

    def save_statistics_table(self, stats):
        """Save statistics to CSV and text files."""
        # Individual residues table
        rows = []

        for resid in self.glu_residues:
            rows.append({
                'Residue': self.get_residue_label(resid, True),
                'Type': self.get_residue_type(resid, True),
                'Mean (%)': stats['glu'][resid]['mean'],
                'Std (%)': stats['glu'][resid]['std'],
                'Values': stats['glu'][resid]['values']
            })

        for resid in self.asn_residues:
            rows.append({
                'Residue': self.get_residue_label(resid, False),
                'Type': self.get_residue_type(resid, False),
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
            f.write(f"Residue names from: "
                    f"{self.prmtop if self.residue_labels else 'NO TOPOLOGY (group-based fallback)'}\n")
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

            for key, label in [('percentage_any_glu', f'Any {self.glu_group_name}'),
                              ('percentage_any_asn', f'Any {self.asn_group_name}'),
                              ('percentage_any_residue', 'Any Residue')]:
                mean = stats['aggregate'][key]['mean']
                std = stats['aggregate'][key]['std']
                values = stats['aggregate'][key]['values']
                values_str = ", ".join([f"{v:.1f}" for v in values])
                f.write(f"{label:<20} Mean: {mean:>6.2f}%  Std: {std:>6.2f}%  "
                       f"Values: {values_str}\n")

        print(f"Statistics text file saved to: {txt_file}")

    # ─────────────────────────────────────────────────────────
    #  Driver
    # ─────────────────────────────────────────────────────────

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
    parser.add_argument("--prmtop", default=None,
                        help="AMBER topology for this system, used to read the ACTUAL "
                             "residue names. Strongly recommended, and required for "
                             "mutants. If omitted, the script looks for a single "
                             "*.prmtop near --base_dir.")

    args = parser.parse_args()

    analyzer = ProximitySummaryAcrossRuns(
        base_results_dir=args.base_dir,
        channel_type=args.channel_type,
        output_dir=args.output_dir,
        prmtop=args.prmtop
    )

    analyzer.run_summary_analysis()


if __name__ == "__main__":
    main()


'''
RESULTS=/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results

python3 proximity_summary_across_runs.py --base_dir $RESULTS/G2 --channel_type G2 \
    --prmtop /path/to/G2/com_4fs.prmtop
python3 proximity_summary_across_runs.py --base_dir $RESULTS/G12 --channel_type G12 \
    --prmtop /path/to/G12/com_4fs.prmtop
python3 proximity_summary_across_runs.py --base_dir $RESULTS/G12_ML --channel_type G12_ML \
    --prmtop /path/to/G12_ML/com_4fs.prmtop
python3 proximity_summary_across_runs.py --base_dir $RESULTS/G12_GAT --channel_type G12_GAT \
    --prmtop /path/to/G12_GAT/com_4fs.prmtop

# Mutants: --prmtop matters most here. The plot will now say GLN/ASN where the
# mutation actually is, instead of blindly printing the wild-type name.
python3 proximity_summary_across_runs.py --base_dir $RESULTS/G12_D173N --channel_type G12_D173N \
    --prmtop /path/to/G12_D173N/com_4fs.prmtop
python3 proximity_summary_across_runs.py --base_dir $RESULTS/G12_E141Q --channel_type G12_E141Q \
    --prmtop /path/to/G12_E141Q/com_4fs.prmtop
'''

# python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
# python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
'''
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --channel_type G2 --prmtop /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2/RUN1/com.prmtop
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12 --prmtop /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12/RUN1/com.prmtop
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --channel_type G12_ML
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --channel_type G12_GAT
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N --channel_type G12_D173N --prmtop /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_D173N/RUN1/com.prmtop
python3 proximity_summary_across_runs.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q --channel_type G12_E141Q --prmtop /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN1/com.prmtop
'''
