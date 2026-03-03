import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm


class DihedralAngleAnalyzer:
    def __init__(self, topology_file, trajectory_file, results_dir, 
                 start_frame, end_frame):
        """
        Initialize dihedral angle analyzer.
        
        Parameters:
        -----------
        topology_file : str or Path
            Path to topology file
        trajectory_file : str or Path
            Path to trajectory file
        results_dir : str or Path
            Directory to save results
        start_frame : int
            Starting frame
        end_frame : int
            Ending frame
        """
        self.results_dir = Path(results_dir)
        self.angles_dir = self.results_dir / "angles"
        self.angles_dir.mkdir(exist_ok=True)
        
        self.u = mda.Universe(topology_file, trajectory_file)
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Will store permeation event frames
        self.permeation_frames = []
        
    def load_permeation_events(self, stage='end_2'):
        """Load permeation events and extract frames for the specified stage"""
        events_file = self.results_dir / "permeation_table.json"
        
        if events_file.exists():
            with open(events_file, 'r') as f:
                data = json.load(f)
            
            # Map stage names to JSON keys
            stage_map = {
                'start_1': 'R1_start',
                'end_1': 'R1_end',
                'start_2': 'R2_start',
                'end_2': 'R2_end',
                'start_3': 'R3_start',
                'end_3': 'R3_end',
                'start_4': 'R4_start',
                'end_4': 'R4_end'
            }
            
            json_key = stage_map.get(stage, 'R2_end')
            self.permeation_frames = [entry[json_key] for entry in data]
            
            print(f"Loaded {len(self.permeation_frames)} permeation events at stage {stage}")
        else:
            print("No permeation events file found")
            self.permeation_frames = []
    
    def calculate_chi_angles(self, residue):
        """
        Calculate chi1 and chi2 dihedral angles using MDAnalysis.
        
        Returns:
        --------
        dict with 'chi1' and 'chi2' (in degrees), or None if atoms missing
        """
        resname = residue.resnames[0]
        
        try:
            # Chi1: N - CA - CB - CG
            n = residue.select_atoms("name N")
            ca = residue.select_atoms("name CA")
            cb = residue.select_atoms("name CB")
            cg = residue.select_atoms("name CG")
            
            if len(n) == 0 or len(ca) == 0 or len(cb) == 0 or len(cg) == 0:
                return None
            
            from MDAnalysis.lib.distances import calc_dihedrals
            
            chi1 = calc_dihedrals(
                n.positions[0],
                ca.positions[0],
                cb.positions[0],
                cg.positions[0]
            )
            chi1_deg = np.degrees(chi1)
            
            # Chi2 depends on residue type
            chi2_deg = None
            
            if resname == 'ASN':
                # Chi2: CA - CB - CG - OD1
                od1 = residue.select_atoms("name OD1")
                if len(od1) > 0:
                    chi2 = calc_dihedrals(
                        ca.positions[0],
                        cb.positions[0],
                        cg.positions[0],
                        od1.positions[0]
                    )
                    chi2_deg = np.degrees(chi2)
            
            elif resname == 'ASP':
                # Chi2: CA - CB - CG - OD1
                od1 = residue.select_atoms("name OD1")
                if len(od1) > 0:
                    chi2 = calc_dihedrals(
                        ca.positions[0],
                        cb.positions[0],
                        cg.positions[0],
                        od1.positions[0]
                    )
                    chi2_deg = np.degrees(chi2)
            
            elif resname == 'GLU':
                # Chi2: CA - CB - CG - CD
                cd = residue.select_atoms("name CD")
                if len(cd) > 0:
                    chi2 = calc_dihedrals(
                        ca.positions[0],
                        cb.positions[0],
                        cg.positions[0],
                        cd.positions[0]
                    )
                    chi2_deg = np.degrees(chi2)
            
            return {
                'chi1': float(chi1_deg),
                'chi2': float(chi2_deg) if chi2_deg is not None else None
            }
            
        except Exception as e:
            print(f"Error calculating chi angles for {resname}: {e}")
            return None
    
    def analyze_all_frames(self, residue_ids):
        """
        Calculate chi1 and chi2 for all residues across all frames.
        
        Parameters:
        -----------
        residue_ids : list
            List of residue IDs to analyze
        
        Returns:
        --------
        dict: {resid: {'frames': [], 'chi1': [], 'chi2': [], 'resname': str}}
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING CHI ANGLES FOR ALL FRAMES")
        print(f"{'='*80}")
        print(f"Residues: {residue_ids}")
        print(f"Frames: {self.start_frame} to {self.end_frame}")
        
        results = {}
        
        # Initialize storage for each residue
        for resid in residue_ids:
            residue = self.u.select_atoms(f"resid {resid}")
            if len(residue) > 0:
                resname = residue.resnames[0]
                results[resid] = {
                    'frames': [],
                    'chi1': [],
                    'chi2': [],
                    'resname': resname
                }
        
        # Loop through trajectory
        for ts in tqdm(self.u.trajectory[self.start_frame:self.end_frame+1],
                      total=(self.end_frame - self.start_frame + 1),
                      desc="Calculating angles", unit="frame"):
            
            frame = ts.frame
            
            for resid in residue_ids:
                if resid not in results:
                    continue
                
                residue = self.u.select_atoms(f"resid {resid}")
                
                if len(residue) == 0:
                    continue
                
                angles = self.calculate_chi_angles(residue)
                
                if angles is not None:
                    results[resid]['frames'].append(frame)
                    results[resid]['chi1'].append(angles['chi1'])
                    results[resid]['chi2'].append(angles['chi2'])
        
        print(f"\nAnalyzed {len(results)} residues")
        return results
    
    def save_angles_to_csv(self, results):
        """Save all angle data to CSV files"""
        print(f"\n{'='*80}")
        print(f"SAVING ANGLE DATA TO CSV")
        print(f"{'='*80}")
        
        for resid, data in results.items():
            resname = data['resname']
            
            df = pd.DataFrame({
                'frame': data['frames'],
                'chi1': data['chi1'],
                'chi2': data['chi2']
            })
            
            output_file = self.angles_dir / f"{resname}_{resid}_chi_angles.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
    
    def plot_time_series(self, results):
        """
        Create time series plots of chi angles (like your first image style).
        One subplot per residue showing chi1 and chi2 over time.
        """
        print(f"\n{'='*80}")
        print(f"CREATING TIME SERIES PLOTS")
        print(f"{'='*80}")
        
        for resid, data in results.items():
            resname = data['resname']
            frames = data['frames']
            chi1 = data['chi1']
            chi2 = data['chi2']
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'{resname} {resid} - Chi Angles Over Time', fontsize=14)
            
            # Plot chi1
            ax1.plot(frames, chi1, linewidth=0.5, color='blue')
            ax1.set_ylabel('Chi1 (degrees)')
            ax1.set_ylim(-180, 180)
            ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Mark permeation events
            for perm_frame in self.permeation_frames:
                ax1.axvline(perm_frame, color='red', linestyle='--', 
                           alpha=0.6, linewidth=1.5)
            
            # Plot chi2
            if any(x is not None for x in chi2):
                chi2_clean = [x if x is not None else np.nan for x in chi2]
                ax2.plot(frames, chi2_clean, linewidth=0.5, color='green')
            
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Chi2 (degrees)')
            ax2.set_ylim(-180, 180)
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Mark permeation events
            for perm_frame in self.permeation_frames:
                ax2.axvline(perm_frame, color='red', linestyle='--', 
                           alpha=0.6, linewidth=1.5)
            
            plt.tight_layout()
            
            output_file = self.angles_dir / f"{resname}_{resid}_timeseries.png"
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"Saved: {output_file}")
    
    def plot_polar_histograms(self, results):
        """
        Create polar histogram plots (like your second image style).
        Shows distribution of chi1 angles in polar coordinates.
        """
        print(f"\n{'='*80}")
        print(f"CREATING POLAR HISTOGRAM PLOTS")
        print(f"{'='*80}")
        
        # Determine grid size
        n_residues = len(results)
        n_cols = 2
        n_rows = int(np.ceil(n_residues / n_cols))
        
        fig = plt.figure(figsize=(12, 6 * n_rows))
        
        for idx, (resid, data) in enumerate(results.items(), 1):
            resname = data['resname']
            chi1 = np.array(data['chi1'])
            
            # Convert to radians for polar plot
            chi1_rad = np.deg2rad(chi1)
            
            # Create polar subplot
            ax = fig.add_subplot(n_rows, n_cols, idx, projection='polar')
            
            # Create histogram
            n_bins = 72  # 5-degree bins
            counts, bins = np.histogram(chi1_rad, bins=n_bins, 
                                       range=(-np.pi, np.pi))
            
            # Plot bars
            width = 2 * np.pi / n_bins
            bars = ax.bar(bins[:-1], counts, width=width, bottom=0.0)
            
            # Color bars
            for bar in bars:
                bar.set_facecolor('steelblue')
                bar.set_edgecolor('white')
                bar.set_alpha(0.8)
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title(f'{resname} {resid} χ1', va='bottom', fontsize=12)
            
            # Set radial ticks
            ax.set_ylim(0, max(counts) * 1.1)
        
        plt.tight_layout()
        
        output_file = self.angles_dir / "all_residues_polar_chi1.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Saved: {output_file}")
        
        # Also create individual polar plots for each residue
        for resid, data in results.items():
            resname = data['resname']
            chi1 = np.array(data['chi1'])
            chi2 = [x for x in data['chi2'] if x is not None]
            
            # Create figure with chi1 and chi2 if available
            if len(chi2) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                               subplot_kw=dict(projection='polar'))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), 
                                       subplot_kw=dict(projection='polar'))
            
            # Plot chi1
            chi1_rad = np.deg2rad(chi1)
            n_bins = 72
            counts1, bins1 = np.histogram(chi1_rad, bins=n_bins, 
                                         range=(-np.pi, np.pi))
            width = 2 * np.pi / n_bins
            bars1 = ax1.bar(bins1[:-1], counts1, width=width, bottom=0.0)
            
            for bar in bars1:
                bar.set_facecolor('steelblue')
                bar.set_edgecolor('white')
                bar.set_alpha(0.8)
            
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax1.set_title(f'{resname} {resid} χ1', va='bottom', fontsize=14)
            ax1.set_ylim(0, max(counts1) * 1.1)
            
            # Plot chi2 if available
            if len(chi2) > 0:
                chi2_rad = np.deg2rad(chi2)
                counts2, bins2 = np.histogram(chi2_rad, bins=n_bins, 
                                             range=(-np.pi, np.pi))
                bars2 = ax2.bar(bins2[:-1], counts2, width=width, bottom=0.0)
                
                for bar in bars2:
                    bar.set_facecolor('lightcoral')
                    bar.set_edgecolor('white')
                    bar.set_alpha(0.8)
                
                ax2.set_theta_zero_location('N')
                ax2.set_theta_direction(-1)
                ax2.set_title(f'{resname} {resid} χ2', va='bottom', fontsize=14)
                ax2.set_ylim(0, max(counts2) * 1.1)
            
            plt.tight_layout()
            
            output_file = self.angles_dir / f"{resname}_{resid}_polar.png"
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"Saved: {output_file}")
    
    def plot_combined_histogram(self, results):
        """
        Create combined linear histogram plots (like your second image).
        Shows chi1 distribution for all residues in a grid.
        """
        print(f"\n{'='*80}")
        print(f"CREATING COMBINED HISTOGRAM PLOTS")
        print(f"{'='*80}")
        
        # Determine grid size
        n_residues = len(results)
        n_cols = 2
        n_rows = int(np.ceil(n_residues / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_residues == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (resid, data) in enumerate(results.items()):
            resname = data['resname']
            chi1 = data['chi1']
            
            ax = axes[idx]
            ax.hist(chi1, bins=36, range=(-180, 180), 
                   color='steelblue', edgecolor='white', alpha=0.8)
            
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{resname}_{resid} χ1')
            ax.set_xlim(-180, 180)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_residues, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_file = self.angles_dir / "all_residues_histogram_chi1.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Saved: {output_file}")


def run_angle_analysis(topology_file, trajectory_file, results_dir, 
                       residue_ids, start_frame, end_frame, 
                       permeation_stage='end_2'):
    """
    Main function to run complete angle analysis.
    
    Parameters:
    -----------
    topology_file : str or Path
        Path to topology file
    trajectory_file : str or Path
        Path to trajectory file
    results_dir : str or Path
        Directory containing results and where to save angle analysis
    residue_ids : list
        List of residue IDs to analyze
    start_frame : int
        Starting frame
    end_frame : int
        Ending frame
    permeation_stage : str
        Which permeation stage to mark on plots (default: 'end_2')
    """
    # Initialize analyzer
    analyzer = DihedralAngleAnalyzer(
        topology_file=topology_file,
        trajectory_file=trajectory_file,
        results_dir=results_dir,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    # Load permeation events
    analyzer.load_permeation_events(stage=permeation_stage)
    
    # Analyze all frames
    results = analyzer.analyze_all_frames(residue_ids)
    
    # Save data
    analyzer.save_angles_to_csv(results)
    
    # Create plots
    analyzer.plot_time_series(results)
    analyzer.plot_polar_histograms(results)
    analyzer.plot_combined_histogram(results)
    
    print(f"\n{'='*80}")
    print(f"ANGLE ANALYSIS COMPLETE!")
    print(f"Results saved in: {analyzer.angles_dir}")
    print(f"{'='*80}")