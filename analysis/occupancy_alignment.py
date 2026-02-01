import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
from analysis.converter import convert_to_pdb_numbering


class ComprehensiveEnd2Analysis:
    """
    Comprehensive analysis at end_2 frame combining:
    1. Ion-residue proximity (which ions are close to which residues)
    2. Free ions (ions in cavity but not close to any of the 8 residues)
       - Split into: closest to SF and other free ions
    3. SF/ASN/GLU alignment for permeating ion, bound ions, and free ions
    """
    
    def __init__(self, universe, channel1, channel2, permeation_events, 
                 asn_residues, glu_residues, subunit_groups, channel_type="G12",
                 threshold=2.5, results_dir="."):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        channel1 : Channel
            Channel for region 1 (defines SF line)
        channel2 : Channel
            Channel for region 2 (cavity)
        permeation_events : list
            List of permeation events
        asn_residues : list
            List of ASN residue IDs
        glu_residues : list
            List of GLU residue IDs
        subunit_groups : dict
            Subunit grouping
        channel_type : str
            Channel type for PDB conversion
        threshold : float
            Distance threshold for "close to residue" (default: 2.5 Å)
        results_dir : Path or str
            Directory to save results
        """
        self.u = universe
        self.channel1 = channel1
        self.channel2 = channel2
        self.permeation_events = permeation_events
        self.asn_residues = asn_residues
        self.glu_residues = glu_residues
        self.all_residues = asn_residues + glu_residues
        self.subunit_groups = subunit_groups
        self.channel_type = channel_type
        self.threshold = threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Build residue to subunit mapping
        self.residue_to_subunit = {}
        for subunit_name, residue_list in subunit_groups.items():
            for resid in residue_list:
                self.residue_to_subunit[resid] = subunit_name
    
    def calculate_sf_line(self):
        """Calculate SF line from ions in region 1 or fallback to residues."""
        # Get lower1 COM
        lower_residues = self.channel1.lower_gate_residues
        lower1_atoms = self.u.select_atoms(
            " or ".join([f"resid {rid}" for rid in lower_residues])
        )
        lower1_com = lower1_atoms.center_of_mass()
        
        # Find ions in region 1
        all_ions = self.u.select_atoms("resname K+ K")
        ions_in_region1 = [ion for ion in all_ions if self.channel1.is_within_cylinder(ion.position)]
        
        if len(ions_in_region1) >= 2:
            # Use 2 closest ions to lower1
            ion_distances = [(ion, np.linalg.norm(ion.position - lower1_com)) for ion in ions_in_region1]
            ion_distances.sort(key=lambda x: x[1])
            
            lower_ion_pos = ion_distances[0][0].position
            upper_ion_pos = ion_distances[1][0].position
            
            sf_direction = upper_ion_pos - lower_ion_pos
            sf_direction = sf_direction / np.linalg.norm(sf_direction)
            
            return lower_ion_pos, sf_direction, 'ions'
            
        elif len(ions_in_region1) == 1:
            ion_pos = ions_in_region1[0].position
            sf_direction = ion_pos - lower1_com
            sf_direction = sf_direction / np.linalg.norm(sf_direction)
            
            return lower1_com, sf_direction, 'ion_and_residue'
        else:
            # Fallback to residues
            upper_residues = self.channel1.upper_gate_residues
            upper1_atoms = self.u.select_atoms(" or ".join([f"resid {rid}" for rid in upper_residues]))
            upper1_com = upper1_atoms.center_of_mass()
            
            sf_direction = upper1_com - lower1_com
            sf_direction = sf_direction / np.linalg.norm(sf_direction)
            
            return lower1_com, sf_direction, 'residues'
    
    def distance_point_to_line(self, point, line_point, line_direction):
        """Calculate perpendicular distance from point to line."""
        point_vec = point - line_point
        projection_length = np.dot(point_vec, line_direction)
        projection = projection_length * line_direction
        perpendicular = point_vec - projection
        return np.linalg.norm(perpendicular)
    
    def calculate_circle_alignment(self, ion_position, residue_ids):
        """Calculate alignment with circle of residues."""
        res_atoms = self.u.select_atoms(" or ".join([f"resid {rid}" for rid in residue_ids]))
        res_center = res_atoms.center_of_mass()
        
        res_radii = []
        for resid in residue_ids:
            single_res = self.u.select_atoms(f"resid {resid}")
            res_com = single_res.center_of_mass()
            res_radii.append(np.linalg.norm(res_com - res_center))
        
        avg_radius = np.mean(res_radii)
        ion_distance = np.linalg.norm(ion_position - res_center)
        alignment_percentage = 100.0 * (1.0 - ion_distance / avg_radius)
        
        return {
            'distance_to_center': float(ion_distance),
            'alignment_percentage': float(alignment_percentage),
            'radius': float(avg_radius)
        }
    
    def analyze_ion_alignment(self, ion_pos, sf_point, sf_direction):
        """Calculate only SF alignment for an ion."""
        return float(self.distance_point_to_line(ion_pos, sf_point, sf_direction))
    
    def analyze_event(self, event):
        """Comprehensive analysis for one permeation event at end_2."""
        ion_id = event['ion_id']
        end_2_frame = event['end_2']
        
        # Go to frame
        self.u.trajectory[end_2_frame]
        
        # Update geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        
        # Calculate SF line
        sf_point, sf_direction, sf_source = self.calculate_sf_line()
        
        # Get all ions except permeating ion
        all_ions = self.u.select_atoms(f"resname K+ K and not resid {ion_id}")
        
        # Find ions in cavity (channel 2)
        ions_in_cavity = []
        for ion in all_ions:
            if self.channel2.is_within_cylinder(ion.position):
                ions_in_cavity.append(ion)
        
        # For each ion in cavity, find its CLOSEST residue (if any)
        ion_closest_residue = {}  # {ion_id: {'residue_pdb': ..., 'distance': ...}}
        
        for ion in ions_in_cavity:
            ion_id_int = int(ion.resid)
            min_dist = float('inf')
            closest_res_pdb = None
            closest_resid = None
            
            # Check all 8 residues
            for resid in self.all_residues:
                res_atoms = self.u.select_atoms(f"resid {resid}")
                if len(res_atoms) == 0:
                    continue
                
                distances = np.linalg.norm(res_atoms.positions - ion.position, axis=1)
                dist = np.min(distances)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_resid = resid
                    closest_res_pdb = convert_to_pdb_numbering(resid, self.channel_type)
            
            # Only count if within threshold
            if min_dist < self.threshold:
                ion_closest_residue[ion_id_int] = {
                    'residue_pdb': closest_res_pdb,
                    'residue_id': closest_resid,
                    'distance': float(min_dist)
                }
        
        # Build residue -> ion mapping (now each ion assigned to only ONE residue)
        residue_ion_mapping = {}
        for ion_id_int, data in ion_closest_residue.items():
            res_pdb = data['residue_pdb']
            if res_pdb not in residue_ion_mapping:
                residue_ion_mapping[res_pdb] = {'ion_ids': [], 'count': 0}
            residue_ion_mapping[res_pdb]['ion_ids'].append(ion_id_int)
            residue_ion_mapping[res_pdb]['count'] += 1
        
        # Identify bound and free ions
        bound_ion_ids = set(ion_closest_residue.keys())
        cavity_ion_ids = set(int(ion.resid) for ion in ions_in_cavity)
        free_ion_ids = cavity_ion_ids - bound_ion_ids
        
        # Map to subunits (each ion to ONE subunit via its closest residue)
        subunit_ion_mapping = {}
        for ion_id_int, data in ion_closest_residue.items():
            resid = data['residue_id']
            subunit = self.residue_to_subunit.get(resid)
            if subunit:
                if subunit not in subunit_ion_mapping:
                    subunit_ion_mapping[subunit] = {'ion_ids': [], 'count': 0}
                subunit_ion_mapping[subunit]['ion_ids'].append(ion_id_int)
                subunit_ion_mapping[subunit]['count'] += 1
        
        # Calculate SF distances for free ions to determine which is closest to SF
        free_ions_with_sf = []
        for free_id in free_ion_ids:
            ion_obj = self.u.select_atoms(f"resid {free_id}")
            if len(ion_obj) > 0:
                sf_dist = float(self.distance_point_to_line(ion_obj.positions[0], sf_point, sf_direction))
                free_ions_with_sf.append({'ion_id': free_id, 'sf_distance': sf_dist})
        
        # Sort free ions by SF distance and split
        free_ions_with_sf.sort(key=lambda x: x['sf_distance'])
        
        free_closest_to_sf_id = None
        free_closest_to_sf_distance = None
        other_free_ion_ids = []
        
        if len(free_ions_with_sf) > 0:
            free_closest_to_sf_id = free_ions_with_sf[0]['ion_id']
            free_closest_to_sf_distance = free_ions_with_sf[0]['sf_distance']
            other_free_ion_ids = [item['ion_id'] for item in free_ions_with_sf[1:]]
        
        # Create combination strings WITH COUNTS
        # Residue: "1_152.A_2_152.C_1_173.D_1_FR"
        residue_combo_parts = []
        for res_pdb in sorted(residue_ion_mapping.keys()):
            count = residue_ion_mapping[res_pdb]['count']
            residue_combo_parts.append(f"{count}_{res_pdb}")
        if free_ion_ids:
            residue_combo_parts.append(f"{len(free_ion_ids)}_FR")
        residue_combination = '_'.join(residue_combo_parts)
        
        # Subunit: "1_A_2_C_1_D_1_FR"
        subunit_combo_parts = []
        for subunit in sorted(subunit_ion_mapping.keys()):
            count = subunit_ion_mapping[subunit]['count']
            subunit_combo_parts.append(f"{count}_{subunit}")
        if free_ion_ids:
            subunit_combo_parts.append(f"{len(free_ion_ids)}_FR")
        subunit_combination = '_'.join(subunit_combo_parts)
        
        # Calculate SF alignments for permeating ion
        permeating_ion_obj = self.u.select_atoms(f"resid {ion_id}")
        permeating_sf = None
        if len(permeating_ion_obj) > 0:
            permeating_sf = float(self.distance_point_to_line(
                permeating_ion_obj.positions[0], sf_point, sf_direction
            ))
        
        # Bound ions SF alignment
        bound_ions_sf = []
        for bound_id in sorted(bound_ion_ids):
            ion_obj = self.u.select_atoms(f"resid {bound_id}")
            if len(ion_obj) > 0:
                sf_dist = float(self.distance_point_to_line(ion_obj.positions[0], sf_point, sf_direction))
                bound_ions_sf.append({
                    'ion_id': bound_id,
                    'sf_distance': sf_dist,
                    'close_to_residue': ion_closest_residue[bound_id]['residue_pdb']
                })
        
        # Free ions SF alignment (keep all for general stats)
        free_ions_sf = free_ions_with_sf  # Already calculated above
        
        # Split free ions for separate reporting
        free_closest_sf_info = None
        if free_closest_to_sf_id is not None:
            free_closest_sf_info = {
                'ion_id': free_closest_to_sf_id,
                'sf_distance': free_closest_to_sf_distance
            }
        
        other_free_ions_sf = []
        for item in free_ions_with_sf[1:]:
            other_free_ions_sf.append({
                'ion_id': item['ion_id'],
                'sf_distance': item['sf_distance']
            })
        
        return {
            'ion_id': int(ion_id),
            'end_2_frame': int(end_2_frame),
            'sf_line_source': sf_source,
            'total_ions_in_cavity': len(ions_in_cavity),
            'n_bound_ions': len(bound_ion_ids),
            'n_free_ions': len(free_ion_ids),
            'n_free_closest_to_sf': 1 if free_closest_to_sf_id is not None else 0,
            'n_other_free_ions': len(other_free_ion_ids),
            'residue_ion_mapping': residue_ion_mapping,
            'subunit_ion_mapping': subunit_ion_mapping,
            'bound_ion_ids': sorted(list(bound_ion_ids)),
            'free_ion_ids': sorted(list(free_ion_ids)),
            'free_closest_to_sf': free_closest_sf_info,
            'other_free_ion_ids': sorted(other_free_ion_ids),
            'residue_combination': residue_combination,
            'subunit_combination': subunit_combination,
            'permeating_ion_sf_distance': permeating_sf,
            'bound_ions_sf_alignment': bound_ions_sf,
            'free_ions_sf_alignment': free_ions_sf,  # All free ions (general stats)
            'free_closest_to_sf_alignment': free_closest_sf_info,  # Closest free ion
            'other_free_ions_sf_alignment': other_free_ions_sf  # Other free ions
        }
    
    def run_analysis(self):
        """Run comprehensive end_2 analysis for all events."""
        print("\n" + "="*80)
        print("COMPREHENSIVE END_2 ANALYSIS")
        print("="*80)
        print(f"Proximity threshold: {self.threshold} Å")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        
        for event in self.permeation_events:
            result = self.analyze_event(event)
            self.results.append(result)
            
            print(f"\n{'='*70}")
            print(f"Ion {result['ion_id']} at end_2 (frame {result['end_2_frame']}):")
            print(f"{'='*70}")
            print(f"SF line source: {result['sf_line_source']}")
            print(f"Total ions in cavity: {result['total_ions_in_cavity']}")
            print(f"  Bound ions (close to residues): {result['n_bound_ions']}")
            print(f"  Free ions (not close to residues): {result['n_free_ions']}")
            print(f"    - Free ion closest to SF: {result['n_free_closest_to_sf']}")
            print(f"    - Other free ions: {result['n_other_free_ions']}")
            
            if result['residue_ion_mapping']:
                print(f"\nResidues with ions (<{self.threshold}Å):")
                for res_pdb, data in sorted(result['residue_ion_mapping'].items()):
                    print(f"  {res_pdb}: {data['count']} ions - {data['ion_ids']}")
            print(f"Residue combination: {result['residue_combination']}")
            
            if result['subunit_ion_mapping']:
                print(f"\nSubunits with ions (<{self.threshold}Å):")
                for subunit, data in sorted(result['subunit_ion_mapping'].items()):
                    print(f"  Subunit {subunit}: {data['count']} ions - {data['ion_ids']}")
            print(f"Subunit combination: {result['subunit_combination']}")
            
            if result['free_ion_ids']:
                print(f"\nFree ions: {result['free_ion_ids']}")
                if result['free_closest_to_sf']:
                    print(f"  Closest to SF: Ion {result['free_closest_to_sf']['ion_id']} (SF={result['free_closest_to_sf']['sf_distance']:.2f} Å)")
                if result['other_free_ion_ids']:
                    print(f"  Other free ions: {result['other_free_ion_ids']}")
            
            # Print SF alignments
            if result['permeating_ion_sf_distance'] is not None:
                print(f"\nPermeating ion SF distance: {result['permeating_ion_sf_distance']:.2f} Å")
            
            if result['bound_ions_sf_alignment']:
                print(f"\nBound ions SF alignment:")
                for ba in result['bound_ions_sf_alignment']:
                    print(f"  Ion {ba['ion_id']} (near {ba['close_to_residue']}): SF={ba['sf_distance']:.2f} Å")
            
            if result['free_ions_sf_alignment']:
                print(f"\nFree ions SF alignment:")
                for fa in result['free_ions_sf_alignment']:
                    if result['free_closest_to_sf'] and fa['ion_id'] == result['free_closest_to_sf']['ion_id']:
                        print(f"  Ion {fa['ion_id']} (CLOSEST TO SF): SF={fa['sf_distance']:.2f} Å")
                    else:
                        print(f"  Ion {fa['ion_id']}: SF={fa['sf_distance']:.2f} Å")
    
    def save_results(self):
        """Save results to JSON."""
        output_file = self.results_dir / "comprehensive_end2_analysis.json"
        
        output_data = {
            'threshold': self.threshold,
            'channel_type': self.channel_type,
            'all_residues': self.all_residues,
            'subunit_groups': self.subunit_groups,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def create_summary(self):
        """Create summary statistics."""
        if not self.results:
            print("No results to summarize.")
            return
        
        from collections import Counter
        
        # Count combinations
        residue_combos = Counter(r['residue_combination'] for r in self.results if r['residue_combination'])
        subunit_combos = Counter(r['subunit_combination'] for r in self.results if r['subunit_combination'])
        
        # Ion count statistics
        bound_counts = [r['n_bound_ions'] for r in self.results]
        free_counts = [r['n_free_ions'] for r in self.results]
        free_closest_counts = [r['n_free_closest_to_sf'] for r in self.results]
        other_free_counts = [r['n_other_free_ions'] for r in self.results]
        total_counts = [r['total_ions_in_cavity'] for r in self.results]
        
        # Alignment statistics - permeating ions (SF only)
        perm_sf = [r['permeating_ion_sf_distance'] for r in self.results if r['permeating_ion_sf_distance'] is not None]
        
        # Bound ions SF
        all_bound_sf = []
        for r in self.results:
            for bound_ion in r['bound_ions_sf_alignment']:
                all_bound_sf.append(bound_ion['sf_distance'])
        
        # Free ions SF (all)
        all_free_sf = []
        for r in self.results:
            for free_ion in r['free_ions_sf_alignment']:
                all_free_sf.append(free_ion['sf_distance'])
        
        # Free ion closest to SF
        free_closest_sf = []
        for r in self.results:
            if r['free_closest_to_sf']:
                free_closest_sf.append(r['free_closest_to_sf']['sf_distance'])
        
        # Other free ions SF
        other_free_sf = []
        for r in self.results:
            for other_free in r['other_free_ions_sf_alignment']:
                other_free_sf.append(other_free['sf_distance'])
        
        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'top_residue_combinations': dict(residue_combos.most_common(10)),
            'top_subunit_combinations': dict(subunit_combos.most_common(10)),
            'ion_counts': {
                'bound_ions': {'mean': float(np.mean(bound_counts)), 'std': float(np.std(bound_counts))},
                'free_ions': {'mean': float(np.mean(free_counts)), 'std': float(np.std(free_counts))},
                'free_closest_to_sf': {'mean': float(np.mean(free_closest_counts)), 'std': float(np.std(free_closest_counts))},
                'other_free_ions': {'mean': float(np.mean(other_free_counts)), 'std': float(np.std(other_free_counts))},
                'total_ions': {'mean': float(np.mean(total_counts)), 'std': float(np.std(total_counts))}
            },
            'sf_alignment_stats': {
                'permeating_ions': {
                    'mean': float(np.mean(perm_sf)), 
                    'std': float(np.std(perm_sf)),
                    'n': len(perm_sf)
                } if perm_sf else None,
                'bound_ions': {
                    'mean': float(np.mean(all_bound_sf)),
                    'std': float(np.std(all_bound_sf)),
                    'n': len(all_bound_sf)
                } if all_bound_sf else None,
                'free_ions': {
                    'mean': float(np.mean(all_free_sf)),
                    'std': float(np.std(all_free_sf)),
                    'n': len(all_free_sf)
                } if all_free_sf else None,
                'free_closest_to_sf': {
                    'mean': float(np.mean(free_closest_sf)),
                    'std': float(np.std(free_closest_sf)),
                    'n': len(free_closest_sf)
                } if free_closest_sf else None,
                'other_free_ions': {
                    'mean': float(np.mean(other_free_sf)),
                    'std': float(np.std(other_free_sf)),
                    'n': len(other_free_sf)
                } if other_free_sf else None
            }
        }
        
        output_file = self.results_dir / "comprehensive_end2_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total events: {summary['total_events']}")
        print(f"\nIon counts in cavity:")
        print(f"  Bound ions: {summary['ion_counts']['bound_ions']['mean']:.1f} ± {summary['ion_counts']['bound_ions']['std']:.1f}")
        print(f"  Free ions (total): {summary['ion_counts']['free_ions']['mean']:.1f} ± {summary['ion_counts']['free_ions']['std']:.1f}")
        print(f"    - Free closest to SF: {summary['ion_counts']['free_closest_to_sf']['mean']:.1f} ± {summary['ion_counts']['free_closest_to_sf']['std']:.1f}")
        print(f"    - Other free ions: {summary['ion_counts']['other_free_ions']['mean']:.1f} ± {summary['ion_counts']['other_free_ions']['std']:.1f}")
        print(f"  Total: {summary['ion_counts']['total_ions']['mean']:.1f} ± {summary['ion_counts']['total_ions']['std']:.1f}")
        
        print(f"\nTop subunit combinations:")
        for combo, count in list(subunit_combos.most_common(5)):
            print(f"  {combo}: {count}")
        
        print(f"\n✓ Summary saved to: {output_file}")
        
        return summary


def run_comprehensive_end2_analysis(universe, channel1, channel2, permeation_events,
                                     asn_residues, glu_residues, subunit_groups,
                                     channel_type="G12", threshold=2.5, results_dir="."):
    """
    Run comprehensive end_2 analysis.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
    channel1 : Channel for region 1
    channel2 : Channel for region 2 (cavity)
    permeation_events : list
    asn_residues : list
    glu_residues : list
    subunit_groups : dict
    channel_type : str
    threshold : float (default: 2.5 Å)
    results_dir : Path or str
    
    Returns:
    --------
    ComprehensiveEnd2Analysis
    """
    analyzer = ComprehensiveEnd2Analysis(universe, channel1, channel2, permeation_events,
                                        asn_residues, glu_residues, subunit_groups,
                                        channel_type, threshold, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    
    return analyzer