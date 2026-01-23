import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
from collections import Counter
from analysis.converter import convert_to_pdb_numbering


class CavityOccupancyAnalysis:
    """
    Analyzes which residues/subunits have ions close to them at end_2 frames,
    and counts total ions in the cavity (channel 2).
    """
    
    def __init__(self, universe, channel2, permeation_events, asn_residues, glu_residues,
                 subunit_groups, channel_type="G12", threshold=3.0, results_dir="."):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        channel2 : Channel
            Channel object for region 2 (the cavity)
        permeation_events : list
            List of permeation events
        asn_residues : list
            List of ASN residue IDs
        glu_residues : list
            List of GLU residue IDs
        subunit_groups : dict
            Subunit grouping (e.g., {'A': [422, 454], 'B': [98, 130]})
        channel_type : str
            Channel type for PDB numbering conversion (e.g., "G2", "G12", "G12_ML")
        threshold : float
            Distance threshold in Angstroms (default: 3.0 Å)
        results_dir : Path or str
            Directory to save results
        """
        self.u = universe
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
    
    def find_ions_close_to_residues(self, frame, permeating_ion_id):
        """
        At a given frame, find which residues have ions within threshold distance.
        Excludes the permeating ion itself.
        
        Parameters:
        -----------
        frame : int
            Frame number
        permeating_ion_id : int
            Ion ID of the permeating ion (to exclude)
        
        Returns:
        --------
        dict : {
            'residues_with_ions': [residue_ids],
            'subunits_with_ions': [subunit_names],
            'residue_ion_mapping': {residue_id: [ion_ids]},
            'subunit_ion_mapping': {subunit: [ion_ids]},
            'total_ions_in_cavity': int
        }
        """
        self.u.trajectory[frame]
        
        # Update channel2 geometry
        self.channel2.compute_geometry(2)
        
        # Get all potassium ions EXCEPT the permeating one
        all_ions = self.u.select_atoms(f"resname K+ K and not resid {permeating_ion_id}")
        
        # Count ions in channel 2 (cavity) - excluding permeating ion
        ions_in_cavity = []
        for ion in all_ions:
            if self.channel2.is_within_cylinder(ion.position):
                ions_in_cavity.append(int(ion.resid))  # Convert to Python int
        
        # For each of the 8 residues, find ions within threshold
        residue_ion_mapping = {}
        residue_pdb_mapping = {}  # Maps residue_id -> PDB numbering
        
        for resid in self.all_residues:
            res_atoms = self.u.select_atoms(f"resid {resid}")
            if len(res_atoms) == 0:
                continue
            
            # Convert to PDB numbering
            pdb_num = convert_to_pdb_numbering(resid, self.channel_type)
            residue_pdb_mapping[resid] = pdb_num
            
            ions_close_to_this_residue = []
            
            for ion in all_ions:
                ion_pos = ion.position
                # Calculate minimum distance from ion to any atom in this residue
                distances = np.linalg.norm(res_atoms.positions - ion_pos, axis=1)
                min_dist = np.min(distances)
                
                if min_dist < self.threshold:
                    ions_close_to_this_residue.append(int(ion.resid))
            
            if ions_close_to_this_residue:
                residue_ion_mapping[pdb_num] = ions_close_to_this_residue
        
        # Get list of residues that have ions close (PDB numbering)
        residues_with_ions = list(residue_ion_mapping.keys())
        
        # Map to subunits using the original residue IDs
        subunit_ion_mapping = {}
        for resid, pdb_num in residue_pdb_mapping.items():
            if pdb_num in residue_ion_mapping:
                subunit = self.residue_to_subunit.get(resid)
                if subunit:
                    if subunit not in subunit_ion_mapping:
                        subunit_ion_mapping[subunit] = set()
                    subunit_ion_mapping[subunit].update(residue_ion_mapping[pdb_num])
        
        # Convert sets to sorted lists
        for subunit in subunit_ion_mapping:
            subunit_ion_mapping[subunit] = sorted(list(subunit_ion_mapping[subunit]))
        
        subunits_with_ions = list(subunit_ion_mapping.keys())
        
        return {
            'residues_with_ions': residues_with_ions,
            'subunits_with_ions': sorted(subunits_with_ions),
            'residue_ion_mapping': residue_ion_mapping,
            'subunit_ion_mapping': subunit_ion_mapping,
            'total_ions_in_cavity': len(ions_in_cavity),
            'ions_in_cavity': ions_in_cavity
        }
    
    def analyze_event(self, event):
        """Analyze cavity occupancy at end_2 for a single permeation event."""
        ion_id = event['ion_id']
        end_2_frame = event['end_2']
        
        occupancy = self.find_ions_close_to_residues(end_2_frame, ion_id)
        
        # Create combination strings
        residue_combination = '_'.join([str(r) for r in sorted(occupancy['residues_with_ions'])])
        subunit_combination = '_'.join(sorted(occupancy['subunits_with_ions']))
        
        return {
            'ion_id': int(ion_id),
            'end_2_frame': int(end_2_frame),
            'residues_with_ions': occupancy['residues_with_ions'],
            'residue_combination': residue_combination,
            'subunits_with_ions': occupancy['subunits_with_ions'],
            'subunit_combination': subunit_combination,
            'residue_ion_mapping': occupancy['residue_ion_mapping'],
            'subunit_ion_mapping': occupancy['subunit_ion_mapping'],
            'total_ions_in_cavity': occupancy['total_ions_in_cavity'],
            'ions_in_cavity': occupancy['ions_in_cavity']
        }
    
    def run_analysis(self):
        """Run cavity occupancy analysis for all permeation events."""
        print("\n" + "="*80)
        print("CAVITY OCCUPANCY ANALYSIS (at end_2)")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        
        for event in self.permeation_events:
            result = self.analyze_event(event)
            self.results.append(result)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Ion {result['ion_id']} at end_2 (frame {result['end_2_frame']}):")
            print(f"{'='*60}")
            print(f"Total ions in cavity (channel 2): {result['total_ions_in_cavity']}")
            print(f"  Ion IDs: {result['ions_in_cavity']}")
            
            print(f"\nResidues with ions close (<{self.threshold}Å): {len(result['residues_with_ions'])}")
            for resid, ions in result['residue_ion_mapping'].items():
                print(f"  Residue {resid}: {len(ions)} ions - {ions}")
            print(f"Residue combination: {result['residue_combination']}")
            
            print(f"\nSubunits with ions close (<{self.threshold}Å): {len(result['subunits_with_ions'])}")
            for subunit, ions in result['subunit_ion_mapping'].items():
                print(f"  Subunit {subunit}: {len(ions)} ions - {ions}")
            print(f"Subunit combination: {result['subunit_combination']}")
    
    def save_results(self):
        """Save cavity occupancy results to JSON file."""
        output_file = self.results_dir / "cavity_occupancy_at_end2.json"
        
        output_data = {
            'threshold': self.threshold,
            'all_residues_analyzed': self.all_residues,
            'subunit_groups': self.subunit_groups,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def create_summary(self):
        """Create summary statistics of residue and subunit combinations."""
        if not self.results:
            print("No results to summarize.")
            return
        
        # Count residue combinations
        residue_combo_counts = Counter()
        for r in self.results:
            if r['residue_combination']:
                residue_combo_counts[r['residue_combination']] += 1
        
        # Count subunit combinations
        subunit_combo_counts = Counter()
        for r in self.results:
            if r['subunit_combination']:
                subunit_combo_counts[r['subunit_combination']] += 1
        
        # Count ions in cavity
        cavity_ion_counts = [r['total_ions_in_cavity'] for r in self.results]
        
        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'residue_combination_counts': dict(residue_combo_counts.most_common()),
            'subunit_combination_counts': dict(subunit_combo_counts.most_common()),
            'cavity_ion_stats': {
                'mean': float(np.mean(cavity_ion_counts)),
                'std': float(np.std(cavity_ion_counts)),
                'min': int(np.min(cavity_ion_counts)),
                'max': int(np.max(cavity_ion_counts))
            }
        }
        
        # Save summary
        output_file = self.results_dir / "cavity_occupancy_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("CAVITY OCCUPANCY SUMMARY")
        print("="*80)
        print(f"Total events analyzed: {summary['total_events']}")
        
        print(f"\nMost common RESIDUE combinations:")
        for combo, count in list(residue_combo_counts.most_common(10)):
            print(f"  {combo}: {count} events")
        
        print(f"\nMost common SUBUNIT combinations:")
        for combo, count in list(subunit_combo_counts.most_common(10)):
            print(f"  {combo}: {count} events")
        
        print(f"\nIons in cavity statistics:")
        print(f"  Mean: {summary['cavity_ion_stats']['mean']:.1f} ± {summary['cavity_ion_stats']['std']:.1f}")
        print(f"  Range: {summary['cavity_ion_stats']['min']} - {summary['cavity_ion_stats']['max']}")
        
        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)
        
        return summary


def run_cavity_occupancy_analysis(universe, channel2, permeation_events, 
                                   asn_residues, glu_residues, subunit_groups,
                                   channel_type="G12", threshold=3.0, results_dir="."):
    """
    Convenience function to run cavity occupancy analysis.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The MD universe object
    channel2 : Channel
        Channel object for region 2 (cavity)
    permeation_events : list
        List of permeation events
    asn_residues : list
        List of ASN residue IDs
    glu_residues : list
        List of GLU residue IDs
    subunit_groups : dict
        Subunit grouping (e.g., {'A': [422, 454], 'B': [98, 130]})
    channel_type : str
        Channel type for PDB numbering conversion
    threshold : float
        Distance threshold in Angstroms (default: 3.0 Å)
    results_dir : Path or str
        Directory to save results
        
    Returns:
    --------
    CavityOccupancyAnalysis : The analysis object with results
    """
    analyzer = CavityOccupancyAnalysis(universe, channel2, permeation_events,
                                      asn_residues, glu_residues, subunit_groups,
                                      channel_type, threshold, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    
    return analyzer