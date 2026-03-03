import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


class IonForceCorrelationAnalysis:
    """
    Analyzes correlation between ion distance from residues and electrostatic force
    felt from other ions in the selectivity filter.
    
    Uses MINIMUM distance from ion to ANY ATOM in each residue (same as significance test).
    """
    
    def __init__(self, universe, channel1, channel2, permeation_events, 
                 asn_residues, glu_residues, results_dir):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        channel1 : Channel
            Channel object for region 1 (SF region)
        channel2 : Channel
            Channel object for region 2 (where permeating ion is)
        permeation_events : list
            List of permeation events from IonPermeationAnalysis
        asn_residues : list
            List of ASN residue IDs (4 residues)
        glu_residues : list
            List of GLU residue IDs (4 residues)
        results_dir : Path
            Directory to save results
        """
        self.u = universe
        self.channel1 = channel1
        self.channel2 = channel2
        self.permeation_events = permeation_events
        self.asn_residues = sorted(asn_residues)
        self.glu_residues = sorted(glu_residues)
        self.results_dir = Path(results_dir)
        
        # Physical constants
        self.k_coulomb = 8.9875517923e9  # N⋅m²/C² (Coulomb's constant)
        self.elem_charge = 1.602176634e-19  # C (elementary charge)
        self.angstrom_to_m = 1e-10  # Å to m conversion
        
        # Results storage
        self.force_data = []
    
    def calculate_min_distance_to_residue_atoms(self, ion_position, residue_id):
        """
        Calculate MINIMUM distance from ion to ANY ATOM in a single residue.
        This matches the approach in the significance test code.
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion
        residue_id : int
            Residue ID
            
        Returns:
        --------
        float : minimum distance in Angstroms
        """
        res_atoms = self.u.select_atoms(f"resid {residue_id}")
        if len(res_atoms) == 0:
            return None
        
        # Calculate distance to EACH atom in the residue, then take MINIMUM
        min_dist = np.min([np.linalg.norm(ion_position - atom_pos) 
                          for atom_pos in res_atoms.positions])
        return min_dist
    
    def calculate_min_distance_to_residue_group(self, ion_position, residue_ids):
        """
        Calculate minimum distance from ion to the CLOSEST residue in a group.
        Also returns distances to each individual residue.
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion
        residue_ids : list
            List of residue IDs (e.g., 4 GLU residues)
            
        Returns:
        --------
        dict : {
            'min_distance': closest distance to any residue in group,
            'individual_distances': list of distances to each residue,
            'closest_residue_id': which residue was closest
        }
        """
        individual_distances = []
        for resid in residue_ids:
            dist = self.calculate_min_distance_to_residue_atoms(ion_position, resid)
            if dist is not None:
                individual_distances.append((resid, dist))
        
        if not individual_distances:
            return None
        
        # Find closest
        closest_resid, min_dist = min(individual_distances, key=lambda x: x[1])
        
        return {
            'min_distance': float(min_dist),
            'individual_distances': {int(rid): float(d) for rid, d in individual_distances},
            'closest_residue_id': int(closest_resid)
        }
    
    def calculate_electrostatic_force(self, ion_position, other_ion_positions):
        """
        Calculate the total electrostatic force on an ion from other ions.
        Uses Coulomb's law: F = k * q1 * q2 / r²
        
        Parameters:
        -----------
        ion_position : np.array
            3D position of the ion experiencing force
        other_ion_positions : list of np.array
            List of 3D positions of other ions in SF
            
        Returns:
        --------
        dict : {
            'force_magnitude': total force magnitude in pN,
            'force_vector': 3D force vector in pN,
            'n_ions': number of ions contributing to force
        }
        """
        if len(other_ion_positions) == 0:
            return {
                'force_magnitude': 0.0,
                'force_vector': np.array([0.0, 0.0, 0.0]),
                'n_ions': 0
            }
        
        total_force_vector = np.array([0.0, 0.0, 0.0])
        
        # Calculate force from each ion in SF
        for other_pos in other_ion_positions:
            # Vector from other ion to our ion (direction of repulsion)
            r_vec = ion_position - other_pos  # Angstroms
            r_dist = np.linalg.norm(r_vec)  # Angstroms
            
            if r_dist < 0.1:  # Avoid division by zero (same position)
                continue
            
            # Convert to meters for calculation
            r_meters = r_dist * self.angstrom_to_m
            
            # Coulomb force: F = k * q1 * q2 / r²
            # For K+ ions: q1 = q2 = +e (both positive, so repulsive)
            force_magnitude = (self.k_coulomb * self.elem_charge * self.elem_charge) / (r_meters ** 2)
            
            # Convert to piconewtons (pN)
            force_pN = force_magnitude * 1e12
            
            # Force vector (pointing away from other ion)
            force_direction = r_vec / r_dist  # normalized
            force_vec = force_pN * force_direction
            
            total_force_vector += force_vec
        
        force_magnitude = np.linalg.norm(total_force_vector)
        
        return {
            'force_magnitude': float(force_magnitude),
            'force_vector': total_force_vector.tolist(),
            'n_ions': len(other_ion_positions)
        }
    
    def calculate_glu_center_of_mass(self):
        """
        Calculate the center of mass of all GLU residues (for reference).
        
        Returns:
        --------
        np.array : 3D position
        """
        glu_atoms = self.u.select_atoms(
            " or ".join([f"resid {rid}" for rid in self.glu_residues])
        )
        return glu_atoms.center_of_mass()
    
    def find_ions_in_sf(self, frame, permeating_ion_id):
        """
        Find the 2 ions in region 1 (SF) that are lower and closest to GLU residues.
        
        Parameters:
        -----------
        frame : int
            Frame number
        permeating_ion_id : int
            Ion ID of the permeating ion (to exclude it)
            
        Returns:
        --------
        list : List of ion positions (up to 2 ions)
        """
        # Go to the frame
        self.u.trajectory[frame]
        
        # Update channel geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        
        # Get GLU center of mass (reference point)
        glu_center = self.calculate_glu_center_of_mass()
        
        # Get all potassium ions
        all_ions = self.u.select_atoms("resname K+ K")
        
        # Find ions in region 1 (SF), excluding the permeating ion
        ions_in_sf = []
        for ion in all_ions:
            if ion.resid != permeating_ion_id and self.channel1.is_within_cylinder(ion.position):
                dist_to_glu = np.linalg.norm(ion.position - glu_center)
                ions_in_sf.append((ion.position.copy(), dist_to_glu))
        
        # Sort by distance to GLU center (closest first)
        ions_in_sf.sort(key=lambda x: x[1])
        
        # Return positions of the 2 closest ions to GLU
        selected_positions = [pos for pos, dist in ions_in_sf[:2]]
        
        return selected_positions
        
        # Calculate force from each ion in SF
        for other_pos in other_ion_positions:
            # Vector from other ion to our ion (direction of repulsion)
            r_vec = ion_position - other_pos  # Angstroms
            r_dist = np.linalg.norm(r_vec)  # Angstroms
            
            if r_dist < 0.1:  # Avoid division by zero (same position)
                continue
            
            # Convert to meters for calculation
            r_meters = r_dist * self.angstrom_to_m
            
            # Coulomb force: F = k * q1 * q2 / r²
            # For K+ ions: q1 = q2 = +e (both positive, so repulsive)
            force_magnitude = (self.k_coulomb * self.elem_charge * self.elem_charge) / (r_meters ** 2)
            
            # Convert to piconewtons (pN)
            force_pN = force_magnitude * 1e12
            
            # Force vector (pointing away from other ion)
            force_direction = r_vec / r_dist  # normalized
            force_vec = force_pN * force_direction
            
            total_force_vector += force_vec
        
        force_magnitude = np.linalg.norm(total_force_vector)
        
        return {
            'force_magnitude': float(force_magnitude),
            'force_vector': total_force_vector.tolist(),
            'n_ions': len(other_ion_positions)
        }
    
    def find_ions_in_sf(self, frame, permeating_ion_id):
        """
        Find the 2 ions in region 1 (SF) that are lower and closest to GLU residues.
        
        Parameters:
        -----------
        frame : int
            Frame number
        permeating_ion_id : int
            Ion ID of the permeating ion (to exclude it)
            
        Returns:
        --------
        list : List of ion positions (up to 2 ions)
        """
        # Go to the frame
        self.u.trajectory[frame]
        
        # Update channel geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        
        # Get GLU center of mass (reference point)
        glu_center = self.calculate_glu_center_of_mass()
        
        # Get all potassium ions
        all_ions = self.u.select_atoms("resname K+ K")
        
        # Find ions in region 1 (SF), excluding the permeating ion
        ions_in_sf = []
        for ion in all_ions:
            if ion.resid != permeating_ion_id and self.channel1.is_within_cylinder(ion.position):
                dist_to_glu = np.linalg.norm(ion.position - glu_center)
                ions_in_sf.append((ion.position.copy(), dist_to_glu))
        
        # Sort by distance to GLU center (closest first)
        ions_in_sf.sort(key=lambda x: x[1])
        
        # Return positions of the 2 closest ions to GLU
        selected_positions = [pos for pos, dist in ions_in_sf[:2]]
        
        return selected_positions
    
    def analyze_event(self, event):
        """
        Analyze force-distance correlation for a single permeation event.
        Calculates MINIMUM distances to residue atoms (matching significance test).
        
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
        
        # Update channel geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        
        # Get permeating ion position
        all_ions = self.u.select_atoms("resname K+ K")
        permeating_ion_pos = None
        
        for ion in all_ions:
            if ion.resid == ion_id and self.channel2.is_within_cylinder(ion.position):
                permeating_ion_pos = ion.position.copy()
                break
        
        if permeating_ion_pos is None:
            print(f"Warning: Permeating ion {ion_id} not found in region 2 at frame {last_frame_region2}")
            return None
        
        # Calculate distances to GLU residues (minimum distance to atoms)
        glu_distances = self.calculate_min_distance_to_residue_group(permeating_ion_pos, self.glu_residues)
        
        # Calculate distances to ASN residues (minimum distance to atoms)
        asn_distances = self.calculate_min_distance_to_residue_group(permeating_ion_pos, self.asn_residues)
        
        if glu_distances is None or asn_distances is None:
            print(f"Warning: Could not calculate distances for ion {ion_id} at frame {last_frame_region2}")
            return None
        
        # Find the 2 ions in SF closest to GLU
        sf_ion_positions = self.find_ions_in_sf(last_frame_region2, ion_id)
        
        # Calculate electrostatic force
        force_data = self.calculate_electrostatic_force(permeating_ion_pos, sf_ion_positions)
        
        result = {
            'event_ion_id': ion_id,
            'frame': last_frame_region2,
            'start_frame_region1': event['start_1'],
            
            # Ion position
            'ion_position_x': float(permeating_ion_pos[0]),
            'ion_position_y': float(permeating_ion_pos[1]),
            'ion_position_z': float(permeating_ion_pos[2]),
            
            # GLU distances
            'glu_min_distance': glu_distances['min_distance'],
            'glu_closest_residue_id': glu_distances['closest_residue_id'],
            'glu_individual_distances': glu_distances['individual_distances'],
            'glu_average_distance': float(np.mean(list(glu_distances['individual_distances'].values()))),
            
            # ASN distances
            'asn_min_distance': asn_distances['min_distance'],
            'asn_closest_residue_id': asn_distances['closest_residue_id'],
            'asn_individual_distances': asn_distances['individual_distances'],
            'asn_average_distance': float(np.mean(list(asn_distances['individual_distances'].values()))),
            
            # Force from SF ions (magnitude and components)
            'force_magnitude_pN': force_data['force_magnitude'],
            'force_x_pN': force_data['force_vector'][0],
            'force_y_pN': force_data['force_vector'][1],
            'force_z_pN': force_data['force_vector'][2],  # Z-component (along channel axis)
            'n_sf_ions': force_data['n_ions'],
            'sf_ion_positions': [pos.tolist() for pos in sf_ion_positions]
        }
        
        return result
    
    def run_analysis(self):
        """
        Run force-correlation analysis for all permeation events.
        """
        print("\n" + "="*80)
        print("ION FORCE-DISTANCE CORRELATION ANALYSIS")
        print("="*80)
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        print("Using 2 ions in SF closest to GLU residues for force calculation")
        print("Distance metric: MINIMUM distance from ion to ANY ATOM in residue")
        
        for event in self.permeation_events:
            result = self.analyze_event(event)
            
            if result is not None:
                self.force_data.append(result)
                
                # Print summary
                print(f"\nEvent: Ion {result['event_ion_id']} at frame {result['frame']}")
                print(f"  Ion position: ({result['ion_position_x']:.2f}, {result['ion_position_y']:.2f}, {result['ion_position_z']:.2f})")
                print(f"  GLU: min={result['glu_min_distance']:.2f} Å (closest: res {result['glu_closest_residue_id']}), avg={result['glu_average_distance']:.2f} Å")
                print(f"  ASN: min={result['asn_min_distance']:.2f} Å (closest: res {result['asn_closest_residue_id']}), avg={result['asn_average_distance']:.2f} Å")
                print(f"  Force from {result['n_sf_ions']} SF ions: Fz={result['force_z_pN']:.2f} pN (magnitude={result['force_magnitude_pN']:.2f} pN)")
        
        print(f"\nTotal events analyzed: {len(self.force_data)}")
    
    def save_results(self):
        """
        Save force-correlation analysis results to JSON file.
        """
        # Create directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.results_dir / "force_correlation_data.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.force_data, f, indent=2)
        
        print(f"\nForce correlation data saved to: {output_file}")
    
    def calculate_correlations(self):
        """
        Calculate correlation coefficients between distances and Z-component of force.
        
        Returns:
        --------
        dict : correlation statistics
        """
        if len(self.force_data) < 3:
            print("Warning: Too few data points for meaningful correlation analysis")
            return None
        
        # Extract data arrays - using Z-component of force
        glu_min_dist = np.array([d['glu_min_distance'] for d in self.force_data])
        glu_avg_dist = np.array([d['glu_average_distance'] for d in self.force_data])
        asn_min_dist = np.array([d['asn_min_distance'] for d in self.force_data])
        asn_avg_dist = np.array([d['asn_average_distance'] for d in self.force_data])
        forces_z = np.array([d['force_z_pN'] for d in self.force_data])  # Z-component!
        
        # Calculate Pearson correlation coefficients
        corr_glu_min, p_glu_min = stats.pearsonr(glu_min_dist, forces_z)
        corr_glu_avg, p_glu_avg = stats.pearsonr(glu_avg_dist, forces_z)
        corr_asn_min, p_asn_min = stats.pearsonr(asn_min_dist, forces_z)
        corr_asn_avg, p_asn_avg = stats.pearsonr(asn_avg_dist, forces_z)
        
        correlations = {
            'n_events': len(self.force_data),
            'force_component_used': 'z_component',
            'glu_min_distance_vs_force_z': {
                'correlation': float(corr_glu_min),
                'p_value': float(p_glu_min),
                'interpretation': self._interpret_correlation(corr_glu_min, p_glu_min)
            },
            'glu_avg_distance_vs_force_z': {
                'correlation': float(corr_glu_avg),
                'p_value': float(p_glu_avg),
                'interpretation': self._interpret_correlation(corr_glu_avg, p_glu_avg)
            },
            'asn_min_distance_vs_force_z': {
                'correlation': float(corr_asn_min),
                'p_value': float(p_asn_min),
                'interpretation': self._interpret_correlation(corr_asn_min, p_asn_min)
            },
            'asn_avg_distance_vs_force_z': {
                'correlation': float(corr_asn_avg),
                'p_value': float(p_asn_avg),
                'interpretation': self._interpret_correlation(corr_asn_avg, p_asn_avg)
            }
        }
        
        # Save correlations
        output_file = self.results_dir / "force_correlation_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(correlations, f, indent=2)
        
        print(f"Correlation statistics saved to: {output_file}")
        
        return correlations
    
    def _interpret_correlation(self, r, p):
        """
        Interpret correlation coefficient and p-value.
        """
        if p > 0.05:
            sig = "not significant (p > 0.05)"
        elif p > 0.01:
            sig = "significant (p < 0.05)"
        else:
            sig = "highly significant (p < 0.01)"
        
        if abs(r) < 0.3:
            strength = "weak"
        elif abs(r) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if r > 0 else "negative"
        
        return f"{strength} {direction} correlation, {sig}"
    
    def print_correlation_summary(self, correlations):
        """
        Print a summary of correlation results.
        """
        if correlations is None:
            return
        
        print("\n" + "="*80)
        print("CORRELATION SUMMARY (using Z-component of force)")
        print("="*80)
        print(f"Number of permeation events: {correlations['n_events']}")
        print()
        
        print("MINIMUM distance to closest GLU residue vs Force Z:")
        print(f"  r = {correlations['glu_min_distance_vs_force_z']['correlation']:.3f}, "
              f"p = {correlations['glu_min_distance_vs_force_z']['p_value']:.4f}")
        print(f"  {correlations['glu_min_distance_vs_force_z']['interpretation']}")
        print()
        
        print("AVERAGE distance to all 4 GLU residues vs Force Z:")
        print(f"  r = {correlations['glu_avg_distance_vs_force_z']['correlation']:.3f}, "
              f"p = {correlations['glu_avg_distance_vs_force_z']['p_value']:.4f}")
        print(f"  {correlations['glu_avg_distance_vs_force_z']['interpretation']}")
        print()
        
        print("MINIMUM distance to closest ASN residue vs Force Z:")
        print(f"  r = {correlations['asn_min_distance_vs_force_z']['correlation']:.3f}, "
              f"p = {correlations['asn_min_distance_vs_force_z']['p_value']:.4f}")
        print(f"  {correlations['asn_min_distance_vs_force_z']['interpretation']}")
        print()
        
        print("AVERAGE distance to all 4 ASN residues vs Force Z:")
        print(f"  r = {correlations['asn_avg_distance_vs_force_z']['correlation']:.3f}, "
              f"p = {correlations['asn_avg_distance_vs_force_z']['p_value']:.4f}")
        print(f"  {correlations['asn_avg_distance_vs_force_z']['interpretation']}")
        print("="*80)
    
    def plot_correlations(self):
        """
        Create 10 scatter plots showing distance vs Z-component of force:
        - 4 individual GLU residues
        - 4 individual ASN residues  
        - Mean of all GLUs
        - Mean of all ASNs
        """
        if len(self.force_data) < 2:
            print("Not enough data points to create plots")
            return
        
        # Import PDB conversion function
        from analysis.converter import convert_to_pdb_numbering
        
        # Get channel type from results_dir path (assuming it's in the path)
        # Try to infer from path, default to G12
        path_str = str(self.results_dir)
        if 'G2' in path_str:
            channel_type = 'G2'
        elif 'G12_ML' in path_str:
            channel_type = 'G12_ML'
        elif 'G12_GAT' in path_str or 'G12' in path_str:
            channel_type = 'G12'
        else:
            channel_type = 'G12'  # default
        
        # Extract force Z-component (along channel axis)
        forces_z = np.array([d['force_z_pN'] for d in self.force_data])
        
        print(f"\nCreating 10 correlation plots using Z-component of force...")
        
        # =================================================================
        # PLOT 1-4: Individual GLU residues
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, glu_resid in enumerate(self.glu_residues):
            ax = axes[idx]
            
            # Extract distances to this specific GLU residue
            distances = []
            for d in self.force_data:
                if glu_resid in d['glu_individual_distances']:
                    distances.append(d['glu_individual_distances'][glu_resid])
                else:
                    distances.append(np.nan)
            
            distances = np.array(distances)
            
            # Plot - NO labels on individual points
            ax.scatter(distances, forces_z, s=150, alpha=0.7, c='red', 
                      edgecolors='black', linewidth=2)
            
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            
            # Convert to PDB numbering
            pdb_label = convert_to_pdb_numbering(glu_resid, channel_type)
            
            ax.set_xlabel(f'Distance to GLU {pdb_label} (Å)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Force Z (pN)', fontsize=14, fontweight='bold')
            ax.set_title(f'GLU {pdb_label}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        output_file = self.results_dir / "force_corr_individual_GLU.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ GLU individual plots saved: {output_file}")
        
        # =================================================================
        # PLOT 5-8: Individual ASN residues
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, asn_resid in enumerate(self.asn_residues):
            ax = axes[idx]
            
            # Extract distances to this specific ASN residue
            distances = []
            for d in self.force_data:
                if asn_resid in d['asn_individual_distances']:
                    distances.append(d['asn_individual_distances'][asn_resid])
                else:
                    distances.append(np.nan)
            
            distances = np.array(distances)
            
            # Plot - NO labels on individual points
            ax.scatter(distances, forces_z, s=150, alpha=0.7, c='blue', 
                      edgecolors='black', linewidth=2)
            
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            
            # Convert to PDB numbering
            pdb_label = convert_to_pdb_numbering(asn_resid, channel_type)
            
            ax.set_xlabel(f'Distance to ASN {pdb_label} (Å)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Force Z (pN)', fontsize=14, fontweight='bold')
            ax.set_title(f'ASN {pdb_label}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        output_file = self.results_dir / "force_corr_individual_ASN.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ ASN individual plots saved: {output_file}")
        
        # =================================================================
        # PLOT 9-10: Mean distances
        # =================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PLOT 9: Mean GLU distance
        glu_mean_dist = np.array([d['glu_average_distance'] for d in self.force_data])
        
        ax1.scatter(glu_mean_dist, forces_z, s=180, alpha=0.7, c='darkred', 
                   edgecolors='black', linewidth=2.5)
        
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax1.set_xlabel('Mean Distance to All 4 GLU (Å)', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Force Z (pN)', fontsize=15, fontweight='bold')
        ax1.set_title('Mean GLU Distance vs Force', fontsize=17, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=13)
        
        # PLOT 10: Mean ASN distance
        asn_mean_dist = np.array([d['asn_average_distance'] for d in self.force_data])
        
        ax2.scatter(asn_mean_dist, forces_z, s=180, alpha=0.7, c='darkblue', 
                   edgecolors='black', linewidth=2.5)
        
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax2.set_xlabel('Mean Distance to All 4 ASN (Å)', fontsize=15, fontweight='bold')
        ax2.set_ylabel('Force Z (pN)', fontsize=15, fontweight='bold')
        ax2.set_title('Mean ASN Distance vs Force', fontsize=17, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=13)
        
        plt.tight_layout()
        output_file = self.results_dir / "force_corr_mean_distances.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Mean distance plots saved: {output_file}")
        
        print(f"\n✓ All 10 correlation plots created successfully!")
        print(f"  - 4 individual GLU plots")
        print(f"  - 4 individual ASN plots") 
        print(f"  - 2 mean distance plots (GLU and ASN)")


def run_force_correlation_analysis(universe, channel1, channel2, permeation_events, 
                                   asn_residues, glu_residues, results_dir):
    """
    Convenience function to run the complete force-correlation analysis.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The MD universe object
    channel1 : Channel
        Channel object for region 1 (SF region)
    channel2 : Channel
        Channel object for region 2 (where permeating ion is)
    permeation_events : list
        List of permeation events from IonPermeationAnalysis
    asn_residues : list
        List of ASN residue IDs
    glu_residues : list
        List of GLU residue IDs
    results_dir : Path or str
        Directory to save results
        
    Returns:
    --------
    IonForceCorrelationAnalysis : The analysis object with results
    """
    analyzer = IonForceCorrelationAnalysis(universe, channel1, channel2, 
                                          permeation_events, asn_residues, 
                                          glu_residues, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    
    correlations = analyzer.calculate_correlations()
    analyzer.print_correlation_summary(correlations)
    
    analyzer.plot_correlations()
    
    return analyzer