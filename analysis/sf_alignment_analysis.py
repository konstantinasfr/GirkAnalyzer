import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


class SFAlignmentAnalysis:
    """
    Analyzes ion alignment with the selectivity filter (SF) line at the last frame 
    before ions pass through lower2 (end of region 2).
    """
    
    def __init__(self, universe, channel1, channel2, permeation_events, asn_residues, glu_residues, results_dir):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        channel1 : Channel
            Channel object for region 1 (defines SF line with upper/lower gates)
        channel2 : Channel
            Channel object for region 2 (where we look for ions)
        permeation_events : list
            List of permeation events from IonPermeationAnalysis
        asn_residues : list
            List of ASN residue IDs (4 residues forming a circle)
        glu_residues : list
            List of GLU residue IDs (4 residues forming a circle)
        results_dir : Path
            Directory to save results
        """
        self.u = universe
        self.channel1 = channel1
        self.channel2 = channel2
        self.permeation_events = permeation_events
        self.asn_residues = asn_residues
        self.glu_residues = glu_residues
        self.results_dir = Path(results_dir)
        self.alignment_results = []
        
    def calculate_sf_line(self):
        """
        Calculate the SF line. Priority:
        1. If 2+ ions in region 1: Use the 2 ions closest to lower1
        2. If 1 ion in region 1: Use that ion and lower1 COM
        3. If 0 ions in region 1: Use upper1 COM and lower1 COM (fallback)
        
        Returns the point and direction vector of the line.
        """
        # Get lower1 COM (always needed as reference)
        lower_residues = self.channel1.upper_gate_residues
        lower1_atoms = self.u.select_atoms(
            " or ".join([f"resid {rid}" for rid in lower_residues])
        )
        lower1_com = lower1_atoms.center_of_mass()
        
        # Find all ions in region 1 (channel1)
        all_ions = self.u.select_atoms("resname K+ K")
        ions_in_region1 = []
        
        for ion in all_ions:
            if self.channel1.is_within_cylinder(ion.position):
                ions_in_region1.append(ion)
        
        if len(ions_in_region1) >= 2:
            # Use 2 ions closest to lower1
            ion_distances = []
            for ion in ions_in_region1:
                dist = np.linalg.norm(ion.position - lower1_com)
                ion_distances.append((ion, dist))
            
            # Sort by distance and take 2 closest
            ion_distances.sort(key=lambda x: x[1])
            ion1_pos = ion_distances[0][0].position
            ion2_pos = ion_distances[1][0].position
            
            # Define line from lower ion to upper ion
            # Determine which is lower (closer to lower1)
            lower_ion_pos = ion1_pos  # Already sorted, so ion1 is closer to lower1
            upper_ion_pos = ion2_pos
            
            sf_direction = upper_ion_pos - lower_ion_pos
            sf_direction = sf_direction / np.linalg.norm(sf_direction)  # normalize
            
            return lower_ion_pos, sf_direction
            
        elif len(ions_in_region1) == 1:
            # Use the 1 ion and lower1 COM
            ion_pos = ions_in_region1[0].position
            
            sf_direction = ion_pos - lower1_com
            sf_direction = sf_direction / np.linalg.norm(sf_direction)  # normalize
            
            return lower1_com, sf_direction
            
        else:
            # Fallback: Use upper1 and lower1 COMs (original method)
            upper_residues = self.channel1.upper_gate_residues
            
            upper1_atoms = self.u.select_atoms(
                " or ".join([f"resid {rid}" for rid in upper_residues])
            )
            upper1_com = upper1_atoms.center_of_mass()
            
            sf_direction = upper1_com - lower1_com
            sf_direction = sf_direction / np.linalg.norm(sf_direction)  # normalize
            
            return lower1_com, sf_direction
    
    def calculate_circle_alignment(self, ion_position, residue_ids, residue_name="residues"):
        """
        Calculate how well the ion is aligned with the center of a circle formed by residues.
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion
        residue_ids : list
            List of residue IDs forming the circle
        residue_name : str
            Name for the residue type (for labeling)
            
        Returns:
        --------
        dict : {
            'center': center position of circle,
            'radius': average radius of circle,
            'distance_to_center': ion distance from center,
            'alignment_percentage': 100% = at center, 0% = at edge, <0% = outside
        }
        """
        # Get residue atoms
        res_atoms = self.u.select_atoms(
            " or ".join([f"resid {rid}" for rid in residue_ids])
        )
        
        # Calculate center of mass of all residues (center of circle)
        res_center = res_atoms.center_of_mass()
        
        # Calculate radius: average distance from center to each individual residue COM
        res_radii = []
        for resid in residue_ids:
            single_res_atoms = self.u.select_atoms(f"resid {resid}")
            res_com = single_res_atoms.center_of_mass()
            dist = np.linalg.norm(res_com - res_center)
            res_radii.append(dist)
        
        avg_radius = np.mean(res_radii)
        
        # Calculate ion distance from center
        ion_distance = np.linalg.norm(ion_position - res_center)
        
        # Calculate alignment percentage
        # 100% = at center (distance = 0)
        # 0% = at edge (distance = radius)
        # negative = outside circle
        alignment_percentage = 100.0 * (1.0 - ion_distance / avg_radius)
        
        return {
            'center': res_center.tolist(),
            'radius': float(avg_radius),
            'distance_to_center': float(ion_distance),
            'alignment_percentage': float(alignment_percentage),
            'individual_radii': [float(r) for r in res_radii]
        }

    def calculate_asn_circle_alignment(self, ion_position):
        """
        Calculate how well the ion is aligned with the center of the ASN residue circle.
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion
            
        Returns:
        --------
        dict : alignment data for ASN circle
        """
        return self.calculate_circle_alignment(ion_position, self.asn_residues, "ASN")
    
    def calculate_glu_circle_alignment(self, ion_position):
        """
        Calculate how well the ion is aligned with the center of the GLU residue circle.
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion
            
        Returns:
        --------
        dict : alignment data for GLU circle
        """
        return self.calculate_circle_alignment(ion_position, self.glu_residues, "GLU")
    
    def distance_point_to_line(self, point, line_point, line_direction):
        """
        Calculate the perpendicular distance from a point to a line.
        
        Parameters:
        -----------
        point : np.array
            The point coordinates
        line_point : np.array
            A point on the line (e.g., lower1 COM)
        line_direction : np.array
            Normalized direction vector of the line
        
        Returns:
        --------
        float : perpendicular distance from point to line
        """
        # Vector from line_point to the point
        point_vec = point - line_point
        
        # Project onto line direction
        projection_length = np.dot(point_vec, line_direction)
        projection = projection_length * line_direction
        
        # Perpendicular component
        perpendicular = point_vec - projection
        
        return np.linalg.norm(perpendicular)
    
    def find_ions_in_region2(self, frame, permeating_ion_id):
        """
        Find all ions currently in region 2 at the given frame.
        
        Parameters:
        -----------
        frame : int
            Frame number
        permeating_ion_id : int
            Ion ID of the permeating ion (to identify it separately)
            
        Returns:
        --------
        dict : {'permeating_ion': position, 'other_ions': [(id, position), ...]}
        """
        # Go to the frame
        self.u.trajectory[frame]
        
        # Update channel geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        
        # Get all potassium ions
        all_ions = self.u.select_atoms("resname K+ K")
        
        permeating_ion_pos = None
        other_ions = []
        
        for ion in all_ions:
            pos = ion.position
            
            # Check if in region 2 (use channel2, not channel1!)
            if self.channel2.is_within_cylinder(pos):
                if ion.resid == permeating_ion_id:
                    permeating_ion_pos = pos
                else:
                    other_ions.append((ion.resid, pos))
        
        return {
            'permeating_ion': permeating_ion_pos,
            'other_ions': other_ions
        }
    
    def analyze_alignment_for_event(self, event):
        """
        Analyze ion alignment with SF line for a single permeation event.
        
        Parameters:
        -----------
        event : dict
            Single permeation event dictionary
            
        Returns:
        --------
        dict : Analysis results for this event
        """
        ion_id = event['ion_id']
        last_frame_region2 = event['end_2']  # Last frame before passing lower2
        
        # Go to that frame
        self.u.trajectory[last_frame_region2]
        
        # Calculate SF line
        sf_point, sf_direction = self.calculate_sf_line()
        
        # Find ions in region 2
        ions_data = self.find_ions_in_region2(last_frame_region2, ion_id)
        
        # Calculate alignment for permeating ion
        permeating_ion_alignment = None
        if ions_data['permeating_ion'] is not None:
            # SF line alignment
            dist_sf = self.distance_point_to_line(
                ions_data['permeating_ion'], 
                sf_point, 
                sf_direction
            )
            
            # ASN circle alignment
            asn_alignment = self.calculate_asn_circle_alignment(ions_data['permeating_ion'])
            
            # GLU circle alignment
            glu_alignment = self.calculate_glu_circle_alignment(ions_data['permeating_ion'])
            
            permeating_ion_alignment = {
                'ion_id': ion_id,
                'distance_to_sf_line': float(dist_sf),
                'asn_circle_center': asn_alignment['center'],
                'asn_circle_radius': asn_alignment['radius'],
                'distance_to_asn_center': asn_alignment['distance_to_center'],
                'asn_alignment_percentage': asn_alignment['alignment_percentage'],
                'glu_circle_center': glu_alignment['center'],
                'glu_circle_radius': glu_alignment['radius'],
                'distance_to_glu_center': glu_alignment['distance_to_center'],
                'glu_alignment_percentage': glu_alignment['alignment_percentage']
            }
        
        # Calculate alignment for other ions
        other_ions_alignment = []
        for other_ion_id, other_pos in ions_data['other_ions']:
            # SF line alignment
            dist_sf = self.distance_point_to_line(other_pos, sf_point, sf_direction)
            
            # ASN circle alignment
            asn_alignment = self.calculate_asn_circle_alignment(other_pos)
            
            # GLU circle alignment
            glu_alignment = self.calculate_glu_circle_alignment(other_pos)
            
            other_ions_alignment.append({
                'ion_id': int(other_ion_id),
                'distance_to_sf_line': float(dist_sf),
                'distance_to_asn_center': asn_alignment['distance_to_center'],
                'asn_alignment_percentage': asn_alignment['alignment_percentage'],
                'distance_to_glu_center': glu_alignment['distance_to_center'],
                'glu_alignment_percentage': glu_alignment['alignment_percentage']
            })
        
        result = {
            'event_ion_id': ion_id,
            'frame': last_frame_region2,
            'start_frame_region1': event['start_1'],
            'sf_line_point': sf_point.tolist(),
            'sf_line_direction': sf_direction.tolist(),
            'permeating_ion': permeating_ion_alignment,
            'other_ions_in_region2': other_ions_alignment,
            'n_other_ions': len(other_ions_alignment)
        }
        
        return result
    
    def run_analysis(self):
        """
        Run alignment analysis for all permeation events.
        """
        print("\n" + "="*80)
        print("SF ALIGNMENT ANALYSIS")
        print("="*80)
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        
        for event in self.permeation_events:
            result = self.analyze_alignment_for_event(event)
            self.alignment_results.append(result)
            
            # Print summary
            print(f"\nEvent: Ion {result['event_ion_id']} at frame {result['frame']}")
            if result['permeating_ion']:
                print(f"  Permeating ion:")
                print(f"    Distance to SF line: {result['permeating_ion']['distance_to_sf_line']:.2f} Å")
                print(f"    ASN alignment: {result['permeating_ion']['asn_alignment_percentage']:.1f}%")
                print(f"    GLU alignment: {result['permeating_ion']['glu_alignment_percentage']:.1f}%")
            print(f"  Other ions in region 2: {result['n_other_ions']}")
            for other in result['other_ions_in_region2']:
                print(f"    Ion {other['ion_id']}: SF={other['distance_to_sf_line']:.2f} Å, ASN={other['asn_alignment_percentage']:.1f}%, GLU={other['glu_alignment_percentage']:.1f}%")
    
    def save_results(self):
        """
        Save alignment analysis results to JSON file.
        """
        output_file = self.results_dir / "sf_alignment_analysis.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.alignment_results, f, indent=2)
        
        print(f"\nAlignment analysis saved to: {output_file}")
    
    def plot_alignment_distances(self):
        """
        Create plots showing SF, ASN, and GLU alignment for all events.
        """
        if not self.alignment_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Create figure with 6 subplots (2x3)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        # ============ Plot 1: Permeating ion SF distances ============
        event_numbers_sf = []
        permeating_distances_sf = []
        
        for i, r in enumerate(self.alignment_results, 1):
            if r['permeating_ion'] and r['permeating_ion']['distance_to_sf_line'] is not None:
                event_numbers_sf.append(i)
                permeating_distances_sf.append(r['permeating_ion']['distance_to_sf_line'])
        
        if permeating_distances_sf:
            ax1.bar(event_numbers_sf, permeating_distances_sf, color='steelblue', alpha=0.7)
            ax1.set_xlabel('Permeation Event #')
            ax1.set_ylabel('Distance to SF Line (Å)')
            ax1.set_title('Permeating Ion - SF Line Alignment')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Permeating Ion - SF Line Alignment')
        
        # ============ Plot 2: Permeating ion ASN alignment % ============
        event_numbers_asn = []
        permeating_asn_percentages = []
        
        for i, r in enumerate(self.alignment_results, 1):
            if r['permeating_ion'] and r['permeating_ion']['asn_alignment_percentage'] is not None:
                event_numbers_asn.append(i)
                permeating_asn_percentages.append(r['permeating_ion']['asn_alignment_percentage'])
        
        if permeating_asn_percentages:
            colors = ['green' if p >= 80 else 'orange' if p >= 50 else 'red' 
                     for p in permeating_asn_percentages]
            ax2.bar(event_numbers_asn, permeating_asn_percentages, color=colors, alpha=0.7)
            ax2.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Permeation Event #')
            ax2.set_ylabel('ASN Alignment (%)')
            ax2.set_title('Permeating Ion - ASN Circle')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Permeating Ion - ASN Circle')
        
        # ============ Plot 3: Permeating ion GLU alignment % ============
        event_numbers_glu = []
        permeating_glu_percentages = []
        
        for i, r in enumerate(self.alignment_results, 1):
            if r['permeating_ion'] and r['permeating_ion']['glu_alignment_percentage'] is not None:
                event_numbers_glu.append(i)
                permeating_glu_percentages.append(r['permeating_ion']['glu_alignment_percentage'])
        
        if permeating_glu_percentages:
            colors = ['green' if p >= 80 else 'orange' if p >= 50 else 'red' 
                     for p in permeating_glu_percentages]
            ax3.bar(event_numbers_glu, permeating_glu_percentages, color=colors, alpha=0.7)
            ax3.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax3.set_xlabel('Permeation Event #')
            ax3.set_ylabel('GLU Alignment (%)')
            ax3.set_title('Permeating Ion - GLU Circle')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Permeating Ion - GLU Circle')
        
        # ============ Plot 4: Other ions SF distances ============
        ax4_data = []
        ax4_labels = []
        
        for i, result in enumerate(self.alignment_results):
            event_num = i + 1
            for other in result['other_ions_in_region2']:
                ax4_data.append(other['distance_to_sf_line'])
                ax4_labels.append(f"E{event_num}\nI{other['ion_id']}")
        
        if ax4_data:
            x_pos = range(len(ax4_data))
            ax4.bar(x_pos, ax4_data, color='coral', alpha=0.7)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(ax4_labels, rotation=45, ha='right', fontsize=7)
            ax4.set_ylabel('Distance to SF Line (Å)')
            ax4.set_title('Other Ions - SF Line')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No other ions', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Other Ions - SF Line')
        
        # ============ Plot 5: Other ions ASN alignment % ============
        ax5_data = []
        ax5_labels = []
        ax5_colors = []
        
        for i, result in enumerate(self.alignment_results):
            event_num = i + 1
            for other in result['other_ions_in_region2']:
                percentage = other['asn_alignment_percentage']
                ax5_data.append(percentage)
                ax5_labels.append(f"E{event_num}\nI{other['ion_id']}")
                ax5_colors.append('green' if percentage >= 80 else 'orange' if percentage >= 50 else 'red')
        
        if ax5_data:
            x_pos = range(len(ax5_data))
            ax5.bar(x_pos, ax5_data, color=ax5_colors, alpha=0.7)
            ax5.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(ax5_labels, rotation=45, ha='right', fontsize=7)
            ax5.set_ylabel('ASN Alignment (%)')
            ax5.set_title('Other Ions - ASN Circle')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No other ions', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Other Ions - ASN Circle')
        
        # ============ Plot 6: Other ions GLU alignment % ============
        ax6_data = []
        ax6_labels = []
        ax6_colors = []
        
        for i, result in enumerate(self.alignment_results):
            event_num = i + 1
            for other in result['other_ions_in_region2']:
                percentage = other['glu_alignment_percentage']
                ax6_data.append(percentage)
                ax6_labels.append(f"E{event_num}\nI{other['ion_id']}")
                ax6_colors.append('green' if percentage >= 80 else 'orange' if percentage >= 50 else 'red')
        
        if ax6_data:
            x_pos = range(len(ax6_data))
            ax6.bar(x_pos, ax6_data, color=ax6_colors, alpha=0.7)
            ax6.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(ax6_labels, rotation=45, ha='right', fontsize=7)
            ax6.set_ylabel('GLU Alignment (%)')
            ax6.set_title('Other Ions - GLU Circle')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No other ions', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Other Ions - GLU Circle')
        
        plt.tight_layout()
        
        output_file = self.results_dir / "sf_alignment_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Alignment plot saved to: {output_file}")
    
    def create_summary_statistics(self):
        """
        Create summary statistics of alignment analysis.
        """
        if not self.alignment_results:
            print("No results to summarize. Run analysis first.")
            return
        
        # SF line statistics
        permeating_distances_sf = [
            r['permeating_ion']['distance_to_sf_line'] 
            for r in self.alignment_results 
            if r['permeating_ion']
        ]
        
        # ASN alignment statistics
        permeating_asn_percentages = [
            r['permeating_ion']['asn_alignment_percentage']
            for r in self.alignment_results
            if r['permeating_ion']
        ]
        
        # GLU alignment statistics
        permeating_glu_percentages = [
            r['permeating_ion']['glu_alignment_percentage']
            for r in self.alignment_results
            if r['permeating_ion']
        ]
        
        all_other_distances_sf = []
        all_other_asn_percentages = []
        all_other_glu_percentages = []
        for result in self.alignment_results:
            for other in result['other_ions_in_region2']:
                all_other_distances_sf.append(other['distance_to_sf_line'])
                all_other_asn_percentages.append(other['asn_alignment_percentage'])
                all_other_glu_percentages.append(other['glu_alignment_percentage'])
        
        summary = {
            'total_permeation_events': len(self.alignment_results),
            'permeating_ion_sf_alignment': {
                'mean_distance': float(np.mean(permeating_distances_sf)) if permeating_distances_sf else None,
                'std_distance': float(np.std(permeating_distances_sf)) if permeating_distances_sf else None,
                'min_distance': float(np.min(permeating_distances_sf)) if permeating_distances_sf else None,
                'max_distance': float(np.max(permeating_distances_sf)) if permeating_distances_sf else None,
            },
            'permeating_ion_asn_alignment': {
                'mean_percentage': float(np.mean(permeating_asn_percentages)) if permeating_asn_percentages else None,
                'std_percentage': float(np.std(permeating_asn_percentages)) if permeating_asn_percentages else None,
                'min_percentage': float(np.min(permeating_asn_percentages)) if permeating_asn_percentages else None,
                'max_percentage': float(np.max(permeating_asn_percentages)) if permeating_asn_percentages else None,
            },
            'permeating_ion_glu_alignment': {
                'mean_percentage': float(np.mean(permeating_glu_percentages)) if permeating_glu_percentages else None,
                'std_percentage': float(np.std(permeating_glu_percentages)) if permeating_glu_percentages else None,
                'min_percentage': float(np.min(permeating_glu_percentages)) if permeating_glu_percentages else None,
                'max_percentage': float(np.max(permeating_glu_percentages)) if permeating_glu_percentages else None,
            },
            'other_ions_sf_alignment': {
                'total_observations': len(all_other_distances_sf),
                'mean_distance': float(np.mean(all_other_distances_sf)) if all_other_distances_sf else None,
                'std_distance': float(np.std(all_other_distances_sf)) if all_other_distances_sf else None,
                'min_distance': float(np.min(all_other_distances_sf)) if all_other_distances_sf else None,
                'max_distance': float(np.max(all_other_distances_sf)) if all_other_distances_sf else None,
            },
            'other_ions_asn_alignment': {
                'total_observations': len(all_other_asn_percentages),
                'mean_percentage': float(np.mean(all_other_asn_percentages)) if all_other_asn_percentages else None,
                'std_percentage': float(np.std(all_other_asn_percentages)) if all_other_asn_percentages else None,
                'min_percentage': float(np.min(all_other_asn_percentages)) if all_other_asn_percentages else None,
                'max_percentage': float(np.max(all_other_asn_percentages)) if all_other_asn_percentages else None,
            },
            'other_ions_glu_alignment': {
                'total_observations': len(all_other_glu_percentages),
                'mean_percentage': float(np.mean(all_other_glu_percentages)) if all_other_glu_percentages else None,
                'std_percentage': float(np.std(all_other_glu_percentages)) if all_other_glu_percentages else None,
                'min_percentage': float(np.min(all_other_glu_percentages)) if all_other_glu_percentages else None,
                'max_percentage': float(np.max(all_other_glu_percentages)) if all_other_glu_percentages else None,
            }
        }
        
        # Save summary
        output_file = self.results_dir / "sf_alignment_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("ALIGNMENT SUMMARY STATISTICS")
        print("="*80)
        print(f"\nTotal permeation events analyzed: {summary['total_permeation_events']}")
        
        print("\n--- Permeating Ions ---")
        print("SF Line Alignment:")
        print(f"  Mean distance: {summary['permeating_ion_sf_alignment']['mean_distance']:.2f} ± {summary['permeating_ion_sf_alignment']['std_distance']:.2f} Å")
        print(f"  Range: {summary['permeating_ion_sf_alignment']['min_distance']:.2f} - {summary['permeating_ion_sf_alignment']['max_distance']:.2f} Å")
        
        print("ASN Circle Alignment:")
        print(f"  Mean: {summary['permeating_ion_asn_alignment']['mean_percentage']:.1f} ± {summary['permeating_ion_asn_alignment']['std_percentage']:.1f} %")
        print(f"  Range: {summary['permeating_ion_asn_alignment']['min_percentage']:.1f} - {summary['permeating_ion_asn_alignment']['max_percentage']:.1f} %")
        
        print("GLU Circle Alignment:")
        print(f"  Mean: {summary['permeating_ion_glu_alignment']['mean_percentage']:.1f} ± {summary['permeating_ion_glu_alignment']['std_percentage']:.1f} %")
        print(f"  Range: {summary['permeating_ion_glu_alignment']['min_percentage']:.1f} - {summary['permeating_ion_glu_alignment']['max_percentage']:.1f} %")
        
        if summary['other_ions_sf_alignment']['total_observations'] > 0:
            print(f"\n--- Other Ions in Region 2 (n={summary['other_ions_sf_alignment']['total_observations']}) ---")
            print("SF Line Alignment:")
            print(f"  Mean distance: {summary['other_ions_sf_alignment']['mean_distance']:.2f} ± {summary['other_ions_sf_alignment']['std_distance']:.2f} Å")
            print(f"  Range: {summary['other_ions_sf_alignment']['min_distance']:.2f} - {summary['other_ions_sf_alignment']['max_distance']:.2f} Å")
            
            print("ASN Circle Alignment:")
            print(f"  Mean: {summary['other_ions_asn_alignment']['mean_percentage']:.1f} ± {summary['other_ions_asn_alignment']['std_percentage']:.1f} %")
            print(f"  Range: {summary['other_ions_asn_alignment']['min_percentage']:.1f} - {summary['other_ions_asn_alignment']['max_percentage']:.1f} %")
            
            print("GLU Circle Alignment:")
            print(f"  Mean: {summary['other_ions_glu_alignment']['mean_percentage']:.1f} ± {summary['other_ions_glu_alignment']['std_percentage']:.1f} %")
            print(f"  Range: {summary['other_ions_glu_alignment']['min_percentage']:.1f} - {summary['other_ions_glu_alignment']['max_percentage']:.1f} %")
        else:
            print("\nNo other ions found in Region 2 during permeation events")
        
        print(f"\nSummary saved to: {output_file}")
        
        return summary


def run_sf_alignment_analysis(universe, channel1, channel2, permeation_events, asn_residues, glu_residues, results_dir):
    """
    Convenience function to run the complete SF alignment analysis.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The MD universe object
    channel1 : Channel
        Channel object for region 1 (defines SF line)
    channel2 : Channel
        Channel object for region 2 (where we look for ions)
    permeation_events : list
        List of permeation events from IonPermeationAnalysis
    asn_residues : list
        List of ASN residue IDs (4 residues forming a circle)
    glu_residues : list
        List of GLU residue IDs (4 residues forming a circle)
    results_dir : Path or str
        Directory to save results
        
    Returns:
    --------
    SFAlignmentAnalysis : The analysis object with results
    """
    analyzer = SFAlignmentAnalysis(universe, channel1, channel2, permeation_events, asn_residues, glu_residues, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.plot_alignment_distances()
    analyzer.create_summary_statistics()
    
    return analyzer