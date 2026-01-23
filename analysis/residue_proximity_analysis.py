import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResidueProximityAnalysis:
    def __init__(self, universe, ion_selection, glu_residues, asn_residues, 
                 start_frame, end_frame, results_dir, channel_type, cutoff=3.0):
        """
        Analyzes how many frames have ions within cutoff distance of specific residues.
        
        Parameters:
        -----------
        universe : MDAnalysis Universe
        ion_selection : str
            Selection string for ions (e.g., "resname K+ K")
        glu_residues : list
            List of glutamate residue IDs (4 residues)
        asn_residues : list
            List of asparagine residue IDs (4 residues)
        start_frame : int
        end_frame : int
        results_dir : Path
            Directory to save results
        channel_type : str
            Channel type for PDB numbering conversion (e.g., "G4", "G2", "G12")
        cutoff : float
            Distance cutoff in Angstroms (default: 3.0)
        """
        self.u = universe
        self.ion_selection = ion_selection
        self.glu_residues = glu_residues
        self.asn_residues = asn_residues
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.results_dir = results_dir
        self.channel_type = channel_type
        self.cutoff = cutoff
        
        # Import converter function
        from .converter import convert_to_pdb_numbering
        self.convert_to_pdb = convert_to_pdb_numbering
        
        self.ions = self.u.select_atoms(self.ion_selection)
        
        # Create atom groups for each individual residue
        self.glu_atom_groups = {resid: self.u.select_atoms(f"resid {resid}") 
                                for resid in glu_residues}
        self.asn_atom_groups = {resid: self.u.select_atoms(f"resid {resid}") 
                                for resid in asn_residues}
        
        # Results storage for individual residues
        self.glu_close_frames = {resid: [] for resid in glu_residues}
        self.asn_close_frames = {resid: [] for resid in asn_residues}
        
        # Detailed tracking: which ions are close at each frame for each residue
        self.glu_close_details = {resid: {} for resid in glu_residues}
        self.asn_close_details = {resid: {} for resid in asn_residues}
        
        # Aggregate results
        self.any_glu_close_frames = []  # Frames where at least one ion is close to ANY GLU
        self.any_asn_close_frames = []  # Frames where at least one ion is close to ANY ASN
        self.any_residue_close_frames = []  # Frames where at least one ion is close to ANY of the 8 residues
        
    def is_close(self, ion_pos, residue_atoms):
        """
        Check if ion is within cutoff distance of any atom in residue_atoms.
        
        Parameters:
        -----------
        ion_pos : np.array
            Ion position (x, y, z)
        residue_atoms : AtomGroup
            Atoms to check distance against
            
        Returns:
        --------
        bool : True if ion is within cutoff of any atom
        """
        distances = np.linalg.norm(residue_atoms.positions - ion_pos, axis=1)
        return np.any(distances <= self.cutoff)
    
    def run_analysis(self):
        """Run the proximity analysis over all frames."""
        print(f"\nRunning residue proximity analysis (cutoff = {self.cutoff} Å)...")
        
        for ts in tqdm(self.u.trajectory[self.start_frame:self.end_frame+1],
                      total=(self.end_frame - self.start_frame + 1),
                      desc="Analyzing proximity", unit="frame"):
            
            frame = ts.frame
            
            # Track if ANY ion is close to ANY GLU/ASN this frame
            any_glu_close = False
            any_asn_close = False
            
            # Check each GLU residue individually
            for glu_resid, glu_atoms in self.glu_atom_groups.items():
                close_ions = []
                for ion in self.ions:
                    if self.is_close(ion.position, glu_atoms):
                        close_ions.append(ion.resid)
                
                if close_ions:
                    self.glu_close_frames[glu_resid].append(frame)
                    self.glu_close_details[glu_resid][frame] = close_ions
                    any_glu_close = True
            
            # Check each ASN residue individually
            for asn_resid, asn_atoms in self.asn_atom_groups.items():
                close_ions = []
                for ion in self.ions:
                    if self.is_close(ion.position, asn_atoms):
                        close_ions.append(ion.resid)
                
                if close_ions:
                    self.asn_close_frames[asn_resid].append(frame)
                    self.asn_close_details[asn_resid][frame] = close_ions
                    any_asn_close = True
            
            # Store aggregate results
            if any_glu_close:
                self.any_glu_close_frames.append(frame)
            if any_asn_close:
                self.any_asn_close_frames.append(frame)
            if any_glu_close or any_asn_close:
                self.any_residue_close_frames.append(frame)
    
    def print_results(self):
        """Print summary of proximity analysis."""
        total_frames = self.end_frame - self.start_frame + 1
        
        print(f"\n{'='*70}")
        print("RESIDUE PROXIMITY ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"Cutoff distance: {self.cutoff} Å")
        print(f"Total frames analyzed: {total_frames}")
        print(f"\n{'-'*70}")
        print("INDIVIDUAL GLU RESIDUES:")
        for resid in self.glu_residues:
            count = len(self.glu_close_frames[resid])
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            print(f"  GLU {pdb_label}: {count} frames ({count/total_frames*100:.2f}%)")
        
        print(f"\n{'-'*70}")
        print("INDIVIDUAL ASN RESIDUES:")
        for resid in self.asn_residues:
            count = len(self.asn_close_frames[resid])
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            print(f"  ASN {pdb_label}: {count} frames ({count/total_frames*100:.2f}%)")
        
        print(f"\n{'-'*70}")
        print("AGGREGATE RESULTS:")
        any_glu = len(self.any_glu_close_frames)
        any_asn = len(self.any_asn_close_frames)
        any_res = len(self.any_residue_close_frames)
        
        print(f"Frames with ion close to ANY GLU: {any_glu} ({any_glu/total_frames*100:.2f}%)")
        print(f"Frames with ion close to ANY ASN: {any_asn} ({any_asn/total_frames*100:.2f}%)")
        print(f"Frames with ion close to ANY residue: {any_res} ({any_res/total_frames*100:.2f}%)")
        print(f"{'-'*70}")
    
    def save_results(self):
        """Save proximity analysis results to JSON files."""
        
        # Summary statistics
        total_frames = self.end_frame - self.start_frame + 1
        summary = {
            "cutoff_angstroms": float(self.cutoff),
            "total_frames": int(total_frames),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "glu_residues": [int(r) for r in self.glu_residues],
            "asn_residues": [int(r) for r in self.asn_residues],
            "individual_glu_counts": {
                int(resid): int(len(self.glu_close_frames[resid])) 
                for resid in self.glu_residues
            },
            "individual_asn_counts": {
                int(resid): int(len(self.asn_close_frames[resid])) 
                for resid in self.asn_residues
            },
            "frames_close_to_any_glu": int(len(self.any_glu_close_frames)),
            "frames_close_to_any_asn": int(len(self.any_asn_close_frames)),
            "frames_close_to_any_residue": int(len(self.any_residue_close_frames)),
            "percentage_any_glu": float(len(self.any_glu_close_frames) / total_frames * 100),
            "percentage_any_asn": float(len(self.any_asn_close_frames) / total_frames * 100),
            "percentage_any_residue": float(len(self.any_residue_close_frames) / total_frames * 100),
        }
        
        summary_file = self.results_dir / "proximity_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
        # Detailed frame lists - convert all numpy types to Python types
        details = {
            "glu_close_frames": {
                str(k): [int(f) for f in v] 
                for k, v in self.glu_close_frames.items()
            },
            "asn_close_frames": {
                str(k): [int(f) for f in v] 
                for k, v in self.asn_close_frames.items()
            },
            "glu_close_details": {
                str(resid): {
                    str(frame): [int(ion) for ion in ions] 
                    for frame, ions in frames.items()
                }
                for resid, frames in self.glu_close_details.items()
            },
            "asn_close_details": {
                str(resid): {
                    str(frame): [int(ion) for ion in ions] 
                    for frame, ions in frames.items()
                }
                for resid, frames in self.asn_close_details.items()
            },
            "any_glu_close_frames": [int(f) for f in self.any_glu_close_frames],
            "any_asn_close_frames": [int(f) for f in self.any_asn_close_frames],
            "any_residue_close_frames": [int(f) for f in self.any_residue_close_frames],
        }
        
        details_file = self.results_dir / "proximity_details.json"
        with open(details_file, 'w') as f:
            json.dump(details, f, indent=2)
        print(f"Detailed results saved to: {details_file}")
    
    def plot_individual_glu_timelines(self):
        """Plot timeline for each of the 4 GLU residues."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        frames = range(self.start_frame, self.end_frame + 1)
        
        for idx, (ax, resid) in enumerate(zip(axes, self.glu_residues)):
            presence = [1 if f in self.glu_close_frames[resid] else 0 for f in frames]
            ax.fill_between(frames, 0, presence, alpha=0.6, color='red')
            
            # Use PDB numbering
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            ax.set_ylabel(f'GLU {pdb_label}', fontsize=10)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            # Add count to ylabel
            count = len(self.glu_close_frames[resid])
            total = self.end_frame - self.start_frame + 1
            ax.text(0.02, 0.5, f'{count} frames ({count/total*100:.1f}%)', 
                   transform=ax.transAxes, fontsize=9, va='center')
        
        axes[-1].set_xlabel('Frame', fontsize=12)
        fig.suptitle(f'Ion Proximity to Individual GLU Residues (≤{self.cutoff} Å)', 
                    fontsize=14, y=0.995)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_glu_individual.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GLU individual timeline saved to: {plot_file}")
    
    def plot_individual_asn_timelines(self):
        """Plot timeline for each of the 4 ASN residues."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        frames = range(self.start_frame, self.end_frame + 1)
        
        for idx, (ax, resid) in enumerate(zip(axes, self.asn_residues)):
            presence = [1 if f in self.asn_close_frames[resid] else 0 for f in frames]
            ax.fill_between(frames, 0, presence, alpha=0.6, color='blue')
            
            # Use PDB numbering
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            ax.set_ylabel(f'ASN {pdb_label}', fontsize=10)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            # Add count to ylabel
            count = len(self.asn_close_frames[resid])
            total = self.end_frame - self.start_frame + 1
            ax.text(0.02, 0.5, f'{count} frames ({count/total*100:.1f}%)', 
                   transform=ax.transAxes, fontsize=9, va='center')
        
        axes[-1].set_xlabel('Frame', fontsize=12)
        fig.suptitle(f'Ion Proximity to Individual ASN Residues (≤{self.cutoff} Å)', 
                    fontsize=14, y=0.995)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_asn_individual.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ASN individual timeline saved to: {plot_file}")
    
    def plot_all_residues_combined(self):
        """Plot all 8 residues together in one figure."""
        fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
        
        frames = range(self.start_frame, self.end_frame + 1)
        total = self.end_frame - self.start_frame + 1
        
        # Plot GLU residues (top 4)
        for idx, resid in enumerate(self.glu_residues):
            ax = axes[idx]
            presence = [1 if f in self.glu_close_frames[resid] else 0 for f in frames]
            ax.fill_between(frames, 0, presence, alpha=0.6, color='red')
            
            # Use PDB numbering
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            ax.set_ylabel(f'GLU {pdb_label}', fontsize=9)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            count = len(self.glu_close_frames[resid])
            ax.text(0.02, 0.5, f'{count} ({count/total*100:.1f}%)', 
                   transform=ax.transAxes, fontsize=8, va='center')
        
        # Plot ASN residues (bottom 4)
        for idx, resid in enumerate(self.asn_residues):
            ax = axes[4 + idx]
            presence = [1 if f in self.asn_close_frames[resid] else 0 for f in frames]
            ax.fill_between(frames, 0, presence, alpha=0.6, color='blue')
            
            # Use PDB numbering
            pdb_label = self.convert_to_pdb(resid, self.channel_type)
            ax.set_ylabel(f'ASN {pdb_label}', fontsize=9)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            count = len(self.asn_close_frames[resid])
            ax.text(0.02, 0.5, f'{count} ({count/total*100:.1f}%)', 
                   transform=ax.transAxes, fontsize=8, va='center')
        
        axes[-1].set_xlabel('Frame', fontsize=12)
        fig.suptitle(f'Ion Proximity to All Residues (≤{self.cutoff} Å)', 
                    fontsize=14, y=0.995)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_all_residues.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined timeline saved to: {plot_file}")

    def get_residue_label(self, resid, is_glu_group):
        """
        Get the correct residue label including amino acid type.
        
        Parameters:
        -----------
        resid : int
            Residue ID
        is_glu_group : bool
            True if from glu_residues, False if from asn_residues
        
        Returns:
        --------
        str : Label like "GLU 152.A", "ASN 184.B", or "ASP 173.D"
        """
        pdb_num = self.convert_to_pdb(resid, self.channel_type)
        
        if is_glu_group:
            return f"GLU {pdb_num}"
        else:
            # Special case: residue 1105 in G12 is ASP, not ASN
            if self.channel_type == "G12" and resid == 1105:
                return f"ASP {pdb_num}"
            else:
                return f"ASN {pdb_num}"
    
    def plot_aggregate_comparison(self):
        """Create a summary plot comparing GLU vs ASN vs ALL."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        frames = range(self.start_frame, self.end_frame + 1)
        
        # Create binary arrays
        any_glu = [1 if f in self.any_glu_close_frames else 0 for f in frames]
        any_asn = [1 if f in self.any_asn_close_frames else 0 for f in frames]
        any_res = [1 if f in self.any_residue_close_frames else 0 for f in frames]
        
        # Plot with offset for visibility
        ax.fill_between(frames, 0, any_glu, alpha=0.5, label='Any GLU', color='red')
        ax.fill_between(frames, -1, [-x for x in any_asn], alpha=0.5, label='Any ASN', color='blue')
        ax.fill_between(frames, 1.5, [x*0.5 + 1.5 for x in any_res], alpha=0.3, label='Any Residue', color='green')
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Ion Presence', fontsize=12)
        ax.set_title(f'Aggregate Ion Proximity (≤{self.cutoff} Å)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        total = self.end_frame - self.start_frame + 1
        stats_text = (f"Any GLU: {len(self.any_glu_close_frames)} frames ({len(self.any_glu_close_frames)/total*100:.1f}%)\n"
                     f"Any ASN: {len(self.any_asn_close_frames)} frames ({len(self.any_asn_close_frames)/total*100:.1f}%)\n"
                     f"Any Residue: {len(self.any_residue_close_frames)} frames ({len(self.any_residue_close_frames)/total*100:.1f}%)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_aggregate.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Aggregate comparison saved to: {plot_file}")

    def plot_individual_residue_bar_chart(self):
        """Bar chart showing frame counts for each individual residue."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        total_frames = self.end_frame - self.start_frame + 1
        
        # GLU bar chart
        glu_counts = [len(self.glu_close_frames[resid]) for resid in self.glu_residues]
        glu_percentages = [(count/total_frames)*100 for count in glu_counts]
        glu_labels = [self.get_residue_label(resid, True).replace(' ', '\n') for resid in self.glu_residues]
        
        bars1 = ax1.bar(range(len(self.glu_residues)), glu_counts, color='red', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(self.glu_residues)))
        ax1.set_xticklabels(glu_labels)
        ax1.set_ylabel('Number of Frames', fontsize=12)
        ax1.set_title(f'Ion Proximity to GLU Residues (≤{self.cutoff} Å)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars1, glu_counts, glu_percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        # ASN bar chart
        asn_counts = [len(self.asn_close_frames[resid]) for resid in self.asn_residues]
        asn_percentages = [(count/total_frames)*100 for count in asn_counts]
        asn_labels = [self.get_residue_label(resid, False).replace(' ', '\n') for resid in self.asn_residues]
        
        bars2 = ax2.bar(range(len(self.asn_residues)), asn_counts, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(self.asn_residues)))
        ax2.set_xticklabels(asn_labels)
        ax2.set_ylabel('Number of Frames', fontsize=12)
        ax2.set_title(f'Ion Proximity to ASN Residues (≤{self.cutoff} Å)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars2, asn_counts, asn_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_bar_individual.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Individual residue bar chart saved to: {plot_file}")
    
    def plot_all_residues_bar_chart(self):
        """Bar chart showing all 8 residues together."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        total_frames = self.end_frame - self.start_frame + 1
        
        # Combine GLU and ASN data
        all_residues = list(self.glu_residues) + list(self.asn_residues)
        all_counts = ([len(self.glu_close_frames[resid]) for resid in self.glu_residues] +
                     [len(self.asn_close_frames[resid]) for resid in self.asn_residues])
        all_percentages = [(count/total_frames)*100 for count in all_counts]
        
        # Labels with PDB numbering and colors
        labels = ([self.get_residue_label(resid, True) for resid in self.glu_residues] +
                [self.get_residue_label(resid, False) for resid in self.asn_residues])
        colors = ['red']*len(self.glu_residues) + ['blue']*len(self.asn_residues)
        
        # Create bars
        bars = ax.bar(range(len(all_residues)), all_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(all_residues)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Number of Frames', fontsize=12)
        ax.set_xlabel('Residue', fontsize=12)
        ax.set_title(f'Ion Proximity to All Residues (≤{self.cutoff} Å)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, all_counts, all_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, edgecolor='black', label='GLU'),
                          Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='ASN')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_bar_all_residues.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"All residues bar chart saved to: {plot_file}")
    
    def plot_aggregate_bar_chart(self):
        """Bar chart comparing aggregate statistics."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        total_frames = self.end_frame - self.start_frame + 1
        
        categories = ['Any GLU', 'Any ASN', 'Any Residue']
        counts = [
            len(self.any_glu_close_frames),
            len(self.any_asn_close_frames),
            len(self.any_residue_close_frames)
        ]
        percentages = [(count/total_frames)*100 for count in counts]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Number of Frames', fontsize=12)
        ax.set_title(f'Aggregate Ion Proximity Statistics (≤{self.cutoff} Å)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count} frames\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add total frames reference line
        ax.axhline(y=total_frames, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Total frames ({total_frames})')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_bar_aggregate.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Aggregate bar chart saved to: {plot_file}")
    
    def plot_percentage_comparison_bar_chart(self):
        """Bar chart showing percentages for easy comparison."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        total_frames = self.end_frame - self.start_frame + 1
        
        # All individual residues
        all_residues = list(self.glu_residues) + list(self.asn_residues)
        all_percentages = ([(len(self.glu_close_frames[resid])/total_frames)*100 for resid in self.glu_residues] +
                          [(len(self.asn_close_frames[resid])/total_frames)*100 for resid in self.asn_residues])
        
        labels = ([self.get_residue_label(resid, True) for resid in self.glu_residues] +
                [self.get_residue_label(resid, False) for resid in self.asn_residues])
        colors = ['red']*len(self.glu_residues) + ['blue']*len(self.asn_residues)
        
        bars = ax.bar(range(len(all_residues)), all_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(all_residues)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Frames (%)', fontsize=12)
        ax.set_xlabel('Residue', fontsize=12)
        ax.set_title(f'Percentage of Frames with Ion Proximity (≤{self.cutoff} Å)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, all_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, edgecolor='black', label='GLU'),
                          Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='ASN')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.results_dir / "proximity_bar_percentages.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Percentage comparison bar chart saved to: {plot_file}")