import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
from analysis.converter import convert_to_pdb_numbering

class LastAndLongestDurationAnalysis:
    """
    Finds both:
    1. The LAST (most recent) residue/subunit within threshold for min_duration before end_2
    2. The LONGEST duration residue/subunit within threshold before end_2
    
    Can operate in two modes:
    - Residue mode (subunit_groups=None): Each residue treated individually  
    - Subunit mode (subunit_groups provided): Residues grouped by subunit
    """
    
    def __init__(self, universe, permeation_events, asn_residues, glu_residues, 
                 channel_type="G12", threshold=3.0, min_duration=10, 
                 analysis_start_frame=0, residue_types=['ASN', 'GLU'], 
                 subunit_groups=None, results_dir="."):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        permeation_events : list
            List of permeation events from IonPermeationAnalysis
        asn_residues : list
            List of ASN residue IDs
        glu_residues : list
            List of GLU residue IDs
        channel_type : str
            Channel type for PDB numbering conversion (e.g., "G2", "G12")
        threshold : float
            Distance threshold in Angstroms (default: 3.0 Å)
        min_duration : int
            Minimum number of consecutive frames to consider (default: 10)
        analysis_start_frame : int
            The starting frame of the overall analysis (default: 0)
        residue_types : list of str
            Which residue types to include: ['ASN', 'GLU'], ['ASN'], or ['GLU']
        subunit_groups : dict or None
            If provided, groups residues by subunit. Format:
            {'A': [422, 454], 'B': [98, 130], 'C': [747, 779], 'D': [1073, 1105]}
            Where each list contains residue IDs belonging to that subunit
            If None, operates in residue mode (default: None)
        results_dir : Path or str
            Directory to save results
        """
        self.u = universe
        self.permeation_events = permeation_events
        self.asn_residues = asn_residues
        self.glu_residues = glu_residues
        self.channel_type = channel_type
        self.threshold = threshold
        self.min_duration = min_duration
        self.analysis_start_frame = analysis_start_frame
        self.residue_types = residue_types
        self.subunit_groups = subunit_groups
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Special handling for ASP in G12
        if channel_type == "G12":
            self.asp_residue = 1105
        elif channel_type == "G12_ML":
            self.asp_residue = 131
        else:
            self.asp_residue = None
        
        # Build reverse mapping if in subunit mode (residue_id -> subunit_name)
        self.residue_to_subunit = {}
        if self.subunit_groups:
            for subunit_name, residue_list in self.subunit_groups.items():
                for resid in residue_list:
                    self.residue_to_subunit[resid] = subunit_name
    
    def find_closest_residue_at_frame(self, ion_id, frame):
        """Find the closest residue to an ion at a specific frame."""
        self.u.trajectory[frame]
        
        ion = self.u.select_atoms(f"resid {ion_id}")
        if len(ion) == 0:
            return None, float('inf')
        ion_pos = ion.positions[0]
        
        min_distance = float('inf')
        closest_resid = None
        closest_type = None
        
        # Check ASN residues
        if 'ASN' in self.residue_types:
            for resid in self.asn_residues:
                res_atoms = self.u.select_atoms(f"resid {resid}")
                if len(res_atoms) > 0:
                    distances = np.linalg.norm(res_atoms.positions - ion_pos, axis=1)
                    min_dist_to_res = np.min(distances)
                    
                    if min_dist_to_res < min_distance:
                        min_distance = min_dist_to_res
                        closest_resid = resid
                        if self.asp_residue is not None and resid == self.asp_residue:
                            closest_type = 'ASP'
                        else:
                            closest_type = 'ASN'
        
        # Check GLU residues
        if 'GLU' in self.residue_types:
            for resid in self.glu_residues:
                res_atoms = self.u.select_atoms(f"resid {resid}")
                if len(res_atoms) > 0:
                    distances = np.linalg.norm(res_atoms.positions - ion_pos, axis=1)
                    min_dist_to_res = np.min(distances)
                    
                    if min_dist_to_res < min_distance:
                        min_distance = min_dist_to_res
                        closest_resid = resid
                        closest_type = 'GLU'
        
        if closest_resid and min_distance < self.threshold:
            return {
                'residue_id': int(closest_resid),
                'residue_pdb': convert_to_pdb_numbering(closest_resid,self.channel_type),
                'residue_type': closest_type,
                'distance': float(min_distance),
                'subunit': self.residue_to_subunit.get(closest_resid) if self.subunit_groups else None
            }, min_distance
        
        return None, min_distance
    
    def find_last_and_longest_duration(self, ion_id, end_frame, ion_start_frame):
        """
        Find both last and longest duration.
        Works in both residue mode and subunit mode based on subunit_groups.
        
        Parameters:
        -----------
        ion_id : int
            Ion residue ID
        end_frame : int
            End frame to search from (typically end_2)
        ion_start_frame : int
            Start frame for this specific ion (typically end_1 - last frame in region 1)
        """
        is_subunit_mode = self.subunit_groups is not None
        
        current_identifier = None  # Will be residue_id or subunit name
        current_duration = 0
        current_start_frame = end_frame
        current_residues_list = []  # Track all residues involved in this duration
        
        all_durations = []
        
        # Last (most recent) duration
        last_identifier = None
        last_duration = 0
        last_start_frame = None
        last_end_frame = None
        last_residues_list = None
        
        # Longest duration
        longest_identifier = None
        longest_duration = 0
        longest_start_frame = None
        longest_end_frame = None
        longest_residues_list = None
        
        for frame in range(end_frame, ion_start_frame - 1, -1):
            residue_info, distance = self.find_closest_residue_at_frame(ion_id, frame)
            
            if residue_info:
                # Determine identifier based on mode
                if is_subunit_mode:
                    frame_identifier = residue_info['subunit']
                else:
                    frame_identifier = residue_info['residue_id']
                
                # Within threshold
                if current_identifier is None:
                    # Start new duration
                    current_identifier = frame_identifier
                    current_duration = 1
                    current_start_frame = frame
                    if is_subunit_mode:
                        residue_info_copy = residue_info.copy()
                        residue_info_copy['frame_count'] = 1
                        current_residues_list = [residue_info_copy]
                    else:
                        current_residues_list = [residue_info]
                elif frame_identifier == current_identifier:
                    # Same identifier (residue or subunit), extend duration
                    current_duration += 1
                    current_start_frame = frame
                    
                    # In subunit mode, track frame count per residue
                    if is_subunit_mode:
                        # Find if this residue is already in the list
                        found = False
                        for r in current_residues_list:
                            if r['residue_id'] == residue_info['residue_id']:
                                r['frame_count'] = r.get('frame_count', 0) + 1
                                found = True
                                break
                        
                        if not found:
                            # New residue in this subunit
                            residue_info_copy = residue_info.copy()
                            residue_info_copy['frame_count'] = 1
                            current_residues_list.append(residue_info_copy)
                    else:
                        # Residue mode - just add if not already tracked
                        if not any(r['residue_id'] == residue_info['residue_id'] for r in current_residues_list):
                            current_residues_list.append(residue_info)
                else:
                    # Different identifier - save previous duration if it meets minimum
                    if current_duration >= self.min_duration:
                        duration_entry = {
                            'identifier': current_identifier,
                            'residues': current_residues_list.copy(),
                            'duration': current_duration,
                            'start_frame': current_start_frame,
                            'end_frame': current_start_frame + current_duration - 1,
                            'mode': 'subunit' if is_subunit_mode else 'residue'
                        }
                        all_durations.append(duration_entry)
                        
                        # Check if this is the last (first one found = most recent)
                        if last_identifier is None:
                            last_identifier = current_identifier
                            last_duration = current_duration
                            last_start_frame = current_start_frame
                            last_end_frame = current_start_frame + current_duration - 1
                            last_residues_list = current_residues_list.copy()
                        
                        # Check if this is the longest
                        if current_duration > longest_duration:
                            longest_identifier = current_identifier
                            longest_duration = current_duration
                            longest_start_frame = current_start_frame
                            longest_end_frame = current_start_frame + current_duration - 1
                            longest_residues_list = current_residues_list.copy()
                    
                    # Start new duration with new identifier
                    current_identifier = frame_identifier
                    current_duration = 1
                    current_start_frame = frame
                    if is_subunit_mode:
                        residue_info['frame_count'] = 1
                    current_residues_list = [residue_info]
            else:
                # Not within threshold - save current duration if it exists
                if current_identifier and current_duration >= self.min_duration:
                    duration_entry = {
                        'identifier': current_identifier,
                        'residues': current_residues_list.copy(),
                        'duration': current_duration,
                        'start_frame': current_start_frame,
                        'end_frame': current_start_frame + current_duration - 1,
                        'mode': 'subunit' if is_subunit_mode else 'residue'
                    }
                    all_durations.append(duration_entry)
                    
                    # Check if this is the last
                    if last_identifier is None:
                        last_identifier = current_identifier
                        last_duration = current_duration
                        last_start_frame = current_start_frame
                        last_end_frame = current_start_frame + current_duration - 1
                        last_residues_list = current_residues_list.copy()
                    
                    # Check if this is the longest
                    if current_duration > longest_duration:
                        longest_identifier = current_identifier
                        longest_duration = current_duration
                        longest_start_frame = current_start_frame
                        longest_end_frame = current_start_frame + current_duration - 1
                        longest_residues_list = current_residues_list.copy()
                
                # Reset
                current_identifier = None
                current_duration = 0
                current_residues_list = []
        
        # Don't forget the last duration if we ended while still tracking one
        if current_identifier and current_duration >= self.min_duration:
            duration_entry = {
                'identifier': current_identifier,
                'residues': current_residues_list.copy(),
                'duration': current_duration,
                'start_frame': current_start_frame,
                'end_frame': current_start_frame + current_duration - 1,
                'mode': 'subunit' if is_subunit_mode else 'residue'
            }
            all_durations.append(duration_entry)
            
            # Check if this is the last
            if last_identifier is None:
                last_identifier = current_identifier
                last_duration = current_duration
                last_start_frame = current_start_frame
                last_end_frame = current_start_frame + current_duration - 1
                last_residues_list = current_residues_list.copy()
            
            # Check if this is the longest
            if current_duration > longest_duration:
                longest_identifier = current_identifier
                longest_duration = current_duration
                longest_start_frame = current_start_frame
                longest_end_frame = current_start_frame + current_duration - 1
                longest_residues_list = current_residues_list.copy()
        
        return {
            'last_identifier': last_identifier,
            'last_residues': last_residues_list,
            'last_duration': last_duration,
            'last_start_frame': last_start_frame,
            'last_end_frame': last_end_frame,
            'longest_identifier': longest_identifier,
            'longest_residues': longest_residues_list,
            'longest_duration': longest_duration,
            'longest_start_frame': longest_start_frame,
            'longest_end_frame': longest_end_frame,
            'all_durations': all_durations,
            'mode': 'subunit' if is_subunit_mode else 'residue'
        }
    
    def analyze_event(self, event):
        """Analyze a single permeation event."""
        ion_id = event['ion_id']
        end_2_frame = event['end_2']
        end_1_frame = event['end_1']  # Last frame in region 1
        
        result = self.find_last_and_longest_duration(ion_id, end_2_frame, end_1_frame)
        
        return {
            'ion_id': int(ion_id),
            'end_2_frame': int(end_2_frame),
            'mode': result['mode'],
            'last_identifier': result['last_identifier'],
            'last_residues': result['last_residues'],
            'last_duration': result['last_duration'],
            'last_start_frame': result['last_start_frame'],
            'last_end_frame': result['last_end_frame'],
            'longest_identifier': result['longest_identifier'],
            'longest_residues': result['longest_residues'],
            'longest_duration': result['longest_duration'],
            'longest_start_frame': result['longest_start_frame'],
            'longest_end_frame': result['longest_end_frame'],
            'n_durations_found': len(result['all_durations']),
            'all_durations': result['all_durations']
        }
    
    def run_analysis(self):
        """Run last and longest duration analysis for all permeation events."""
        mode_str = "SUBUNIT" if self.subunit_groups else "RESIDUE"
        
        print("\n" + "="*80)
        print(f"LAST & LONGEST DURATION ANALYSIS - {mode_str} MODE")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å")
        print(f"Minimum duration: {self.min_duration} frames")
        print(f"Analyzing residue types: {', '.join(self.residue_types)}")
        if self.subunit_groups:
            print(f"Subunit groups: {list(self.subunit_groups.keys())}")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        
        for event in self.permeation_events:
            result = self.analyze_event(event)
            self.results.append(result)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Ion {result['ion_id']} (end_2 at frame {result['end_2_frame']}):")
            print(f"{'='*60}")
            
            if result['last_identifier']:
                print(f"LAST ({result['mode']} mode): {result['last_identifier']}")
                if result['last_residues']:
                    if result['mode'] == 'subunit':
                        # Show frame count for each residue in subunit
                        res_details = []
                        for r in result['last_residues']:
                            frame_count = r.get('frame_count', '?')
                            res_details.append(f"{r['residue_type']} {r['residue_id']} ({r['residue_pdb']}): {frame_count} frames")
                        print(f"  Residues involved:")
                        for detail in res_details:
                            print(f"    - {detail}")
                    else:
                        # Residue mode - just list residues
                        res_str = ', '.join([f"{r['residue_type']} {r['residue_id']} ({r['residue_pdb']})" 
                                            for r in result['last_residues']])
                        print(f"  Residues involved: {res_str}")
                print(f"  Total duration: {result['last_duration']} frames")
                print(f"  Period: frames {result['last_start_frame']} to {result['last_end_frame']}")
            else:
                print(f"LAST: None found")
            
            if result['longest_identifier']:
                print(f"LONGEST ({result['mode']} mode): {result['longest_identifier']}")
                if result['longest_residues']:
                    if result['mode'] == 'subunit':
                        # Show frame count for each residue in subunit
                        res_details = []
                        for r in result['longest_residues']:
                            frame_count = r.get('frame_count', '?')
                            res_details.append(f"{r['residue_type']} {r['residue_id']} ({r['residue_pdb']}): {frame_count} frames")
                        print(f"  Residues involved:")
                        for detail in res_details:
                            print(f"    - {detail}")
                    else:
                        # Residue mode - just list residues
                        res_str = ', '.join([f"{r['residue_type']} {r['residue_id']} ({r['residue_pdb']})" 
                                            for r in result['longest_residues']])
                        print(f"  Residues involved: {res_str}")
                print(f"  Total duration: {result['longest_duration']} frames")
                print(f"  Period: frames {result['longest_start_frame']} to {result['longest_end_frame']}")
            else:
                print(f"LONGEST: None found")
            
            print(f"Total durations found (≥{self.min_duration} frames): {result['n_durations_found']}")
    
    def save_results(self):
        """Save analysis results to JSON files - separate files for last and longest."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        mode_str = "subunit" if self.subunit_groups else "residue"
        
        # Save complete results (includes both last and longest)
        complete_file = self.results_dir / f"duration_analysis_complete_{mode_str}.json"
        complete_data = {
            'threshold': self.threshold,
            'min_duration': self.min_duration,
            'mode': mode_str,
            'subunit_groups': self.subunit_groups,
            'asn_residues': self.asn_residues,
            'glu_residues': self.glu_residues,
            'residue_types': self.residue_types,
            'results': self.results
        }
        with open(complete_file, 'w') as f:
            json.dump(complete_data, f, indent=2)
        print(f"\n✓ Complete results saved to: {complete_file}")
        
        # Save LAST results only
        last_results = []
        for r in self.results:
            last_results.append({
                'ion_id': r['ion_id'],
                'end_2_frame': r['end_2_frame'],
                'identifier': r['last_identifier'],
                'residues': r['last_residues'],
                'duration': r['last_duration'],
                'start_frame': r['last_start_frame'],
                'end_frame': r['last_end_frame']
            })
        
        last_file = self.results_dir / f"last_duration_{mode_str}.json"
        with open(last_file, 'w') as f:
            json.dump({'threshold': self.threshold, 'min_duration': self.min_duration, 
                      'mode': mode_str, 'results': last_results}, f, indent=2)
        print(f"✓ LAST results saved to: {last_file}")
        
        # Save LONGEST results only
        longest_results = []
        for r in self.results:
            longest_results.append({
                'ion_id': r['ion_id'],
                'end_2_frame': r['end_2_frame'],
                'identifier': r['longest_identifier'],
                'residues': r['longest_residues'],
                'duration': r['longest_duration'],
                'start_frame': r['longest_start_frame'],
                'end_frame': r['longest_end_frame']
            })
        
        longest_file = self.results_dir / f"longest_duration_{mode_str}.json"
        with open(longest_file, 'w') as f:
            json.dump({'threshold': self.threshold, 'min_duration': self.min_duration, 
                      'mode': mode_str, 'results': longest_results}, f, indent=2)
        print(f"✓ LONGEST results saved to: {longest_file}")
    
    def create_summary(self):
        """Create summary statistics for both last and longest."""
        if not self.results:
            print("No results to summarize.")
            return
        
        is_subunit_mode = self.subunit_groups is not None
        
        if is_subunit_mode:
            # Count by subunit - LAST
            last_counts = {}
            for r in self.results:
                if r['last_identifier']:
                    subunit = r['last_identifier']
                    last_counts[subunit] = last_counts.get(subunit, 0) + 1
            
            # Count by subunit - LONGEST
            longest_counts = {}
            for r in self.results:
                if r['longest_identifier']:
                    subunit = r['longest_identifier']
                    longest_counts[subunit] = longest_counts.get(subunit, 0) + 1
            
            summary = {
                'total_events': len(self.results),
                'threshold': self.threshold,
                'min_duration': self.min_duration,
                'mode': 'subunit',
                'last_subunit_counts': last_counts,
                'longest_subunit_counts': longest_counts
            }
        else:
            # Count by residue type - LAST
            last_asn = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'ASN' for res in r['last_residues']))
            last_asp = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'ASP' for res in r['last_residues']))
            last_glu = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'GLU' for res in r['last_residues']))
            last_none = sum(1 for r in self.results if not r['last_identifier'])
            
            # Count by residue type - LONGEST
            longest_asn = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'ASN' for res in r['longest_residues']))
            longest_asp = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'ASP' for res in r['longest_residues']))
            longest_glu = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'GLU' for res in r['longest_residues']))
            longest_none = sum(1 for r in self.results if not r['longest_identifier'])
            
            summary = {
                'total_events': len(self.results),
                'threshold': self.threshold,
                'min_duration': self.min_duration,
                'mode': 'residue',
                'last_residue_stats': {
                    'asn_count': last_asn,
                    'asp_count': last_asp,
                    'glu_count': last_glu,
                    'none_count': last_none
                },
                'longest_residue_stats': {
                    'asn_count': longest_asn,
                    'asp_count': longest_asp,
                    'glu_count': longest_glu,
                    'none_count': longest_none
                }
            }
        
        # Average durations (same for both modes)
        last_durations = [r['last_duration'] for r in self.results if r['last_identifier']]
        longest_durations = [r['longest_duration'] for r in self.results if r['longest_identifier']]
        
        summary['last_duration_stats'] = {
            'avg': float(np.mean(last_durations)) if last_durations else None,
            'std': float(np.std(last_durations)) if last_durations else None,
            'min': int(np.min(last_durations)) if last_durations else None,
            'max': int(np.max(last_durations)) if last_durations else None
        }
        
        summary['longest_duration_stats'] = {
            'avg': float(np.mean(longest_durations)) if longest_durations else None,
            'std': float(np.std(longest_durations)) if longest_durations else None,
            'min': int(np.min(longest_durations)) if longest_durations else None,
            'max': int(np.max(longest_durations)) if longest_durations else None
        }
        
        # Save summary
        mode_str = "subunit" if is_subunit_mode else "residue"
        output_file = self.results_dir / f"duration_summary_{mode_str}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print(f"SUMMARY STATISTICS - {mode_str.upper()} MODE")
        print("="*80)
        print(f"Total events: {summary['total_events']}")
        print(f"Threshold: {self.threshold} Å, Min duration: {self.min_duration} frames")
        
        if is_subunit_mode:
            print(f"\n--- LAST (most recent) subunit ---")
            for subunit, count in sorted(summary['last_subunit_counts'].items()):
                print(f"  Subunit {subunit}: {count} events")
            
            print(f"\n--- LONGEST duration subunit ---")
            for subunit, count in sorted(summary['longest_subunit_counts'].items()):
                print(f"  Subunit {subunit}: {count} events")
        else:
            print(f"\n--- LAST residue (most recent before permeation) ---")
            print(f"  ASN: {summary['last_residue_stats']['asn_count']} events")
            if summary['last_residue_stats']['asp_count'] > 0:
                print(f"  ASP: {summary['last_residue_stats']['asp_count']} events")
            print(f"  GLU: {summary['last_residue_stats']['glu_count']} events")
            print(f"  None: {summary['last_residue_stats']['none_count']} events")
            
            print(f"\n--- LONGEST residue (longest duration) ---")
            print(f"  ASN: {summary['longest_residue_stats']['asn_count']} events")
            if summary['longest_residue_stats']['asp_count'] > 0:
                print(f"  ASP: {summary['longest_residue_stats']['asp_count']} events")
            print(f"  GLU: {summary['longest_residue_stats']['glu_count']} events")
            print(f"  None: {summary['longest_residue_stats']['none_count']} events")
        
        if last_durations:
            print(f"\nLAST Duration Statistics:")
            print(f"  Average: {summary['last_duration_stats']['avg']:.1f} ± {summary['last_duration_stats']['std']:.1f} frames")
            print(f"  Range: {summary['last_duration_stats']['min']} - {summary['last_duration_stats']['max']} frames")
        
        if longest_durations:
            print(f"\nLONGEST Duration Statistics:")
            print(f"  Average: {summary['longest_duration_stats']['avg']:.1f} ± {summary['longest_duration_stats']['std']:.1f} frames")
            print(f"  Range: {summary['longest_duration_stats']['min']} - {summary['longest_duration_stats']['max']} frames")
        
        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)
        
        return summary


def run_last_duration_analysis(universe, permeation_events, asn_residues, 
                                glu_residues, channel_type="G12", threshold=3.0, 
                                min_duration=10, analysis_start_frame=0, 
                                residue_types=['ASN', 'GLU'], subunit_groups=None, 
                                results_dir="."):
    """
    Convenience function to run the complete duration analysis.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The MD universe object
    permeation_events : list
        List of permeation events
    asn_residues : list
        List of ASN residue IDs
    glu_residues : list
        List of GLU residue IDs
    channel_type : str
        Channel type for PDB numbering conversion
    threshold : float
        Distance threshold in Angstroms (default: 3.0 Å)
    min_duration : int
        Minimum number of consecutive frames (default: 10)
    analysis_start_frame : int
        The starting frame of the overall analysis
    residue_types : list of str
        Which residue types to include
    subunit_groups : dict or None
        Subunit grouping (e.g., {'A': [422, 454], 'B': [98, 130]})
        If None, uses residue mode
    results_dir : Path or str
        Directory to save results
        
    Returns:
    --------
    LastAndLongestDurationAnalysis : The analysis object with results
    """
    analyzer = LastAndLongestDurationAnalysis(universe, permeation_events, asn_residues, 
                                              glu_residues, channel_type, threshold, 
                                              min_duration, analysis_start_frame, 
                                              residue_types, subunit_groups, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    
    return analyzer