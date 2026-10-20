import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from analysis.converter import convert_to_pdb_numbering, subunit_label
from matplotlib.ticker import MaxNLocator
from analysis.closest_residue_analysis import (ClosestResidueAnalysis, ResidueDistanceCalculator,
                                               residue_names_from_topology, ONE_LETTER)

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
        
        # Residue names come from the topology, so mutants are labelled
        # correctly (e.g. Q141 in E141Q, N173 in D173N) without hard-coding.
        self.resnames = residue_names_from_topology(self.u, asn_residues + glu_residues)

        # ASN list first, then GLU list: same order (and tie-breaking) as before
        competing = (list(asn_residues) if 'ASN' in residue_types else []) + \
                    (list(glu_residues) if 'GLU' in residue_types else [])
        self.distances = ResidueDistanceCalculator(self.u, competing)
        
        # Name each subunit group after its chain (G2.a, G2.b, G2.c, G1, ...)
        # instead of the A-D letters, which mean different chains per channel.
        # A group whose residues sit on different chains keeps its letter.
        if self.subunit_groups:
            renamed = {}
            for name, residue_list in self.subunit_groups.items():
                chains = {subunit_label(convert_to_pdb_numbering(r, channel_type)) for r in residue_list}
                renamed[chains.pop() if len(chains) == 1 else name] = residue_list
            if len(renamed) == len(self.subunit_groups):
                print(f"Subunit names: {dict(zip(self.subunit_groups, renamed))}")
                self.subunit_groups = renamed

        # Build reverse mapping if in subunit mode (residue_id -> subunit_name)
        self.residue_to_subunit = {}
        if self.subunit_groups:
            for subunit_name, residue_list in self.subunit_groups.items():
                for resid in residue_list:
                    self.residue_to_subunit[resid] = subunit_name
    
    def find_closest_residue_at_frame(self, ion_id, frame):
        """Find the closest residue to an ion at a specific frame."""
        frames, D = self.distances.min_distances(ion_id, frame, frame)
        if D is None:
            return None, float('inf')
        return self._closest_from_row(D[0])

    def _closest_from_row(self, row):
        """(info, distance) for one frame from its row of residue distances;
        info is None when the closest residue is not within the threshold."""
        j = int(np.argmin(row))
        resid = self.distances.resids[j]
        min_distance = float(row[j])
        if min_distance < self.threshold:
            return {
                'residue_id': int(resid),
                'residue_pdb': convert_to_pdb_numbering(resid, self.channel_type),
                'residue_type': self.resnames.get(resid, 'Unknown'),
                'distance': min_distance,
                'subunit': self.residue_to_subunit.get(resid) if self.subunit_groups else None
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
        
        # All frames of this event in one forward read, then walked backwards
        frames, D = self.distances.min_distances(ion_id, ion_start_frame, end_frame)
        rows = {} if D is None else dict(zip(frames.tolist(), D))

        for frame in range(end_frame, ion_start_frame - 1, -1):
            if frame in rows:
                residue_info, distance = self._closest_from_row(rows[frame])
            else:
                residue_info, distance = None, float('inf')
            
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
            'end_1_frame': int(end_1_frame),
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
            'channel_type': self.channel_type,
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
                'longest_subunit_counts': longest_counts,
                'last_none_count': sum(1 for r in self.results if not r['last_identifier']),
                'longest_none_count': sum(1 for r in self.results if not r['longest_identifier'])
            }
        else:
            # Count by residue type - LAST
            last_asn = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'ASN' for res in r['last_residues']))
            last_asp = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'ASP' for res in r['last_residues']))
            last_glu = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'GLU' for res in r['last_residues']))
            last_gln = sum(1 for r in self.results if r['last_residues'] and any(res['residue_type'] == 'GLN' for res in r['last_residues']))
            last_none = sum(1 for r in self.results if not r['last_identifier'])
            
            # Count by residue type - LONGEST
            longest_asn = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'ASN' for res in r['longest_residues']))
            longest_asp = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'ASP' for res in r['longest_residues']))
            longest_glu = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'GLU' for res in r['longest_residues']))
            longest_gln = sum(1 for r in self.results if r['longest_residues'] and any(res['residue_type'] == 'GLN' for res in r['longest_residues']))
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
                    'gln_count': last_gln,
                    'none_count': last_none
                },
                'longest_residue_stats': {
                    'asn_count': longest_asn,
                    'asp_count': longest_asp,
                    'glu_count': longest_glu,
                    'gln_count': longest_gln,
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
            print(f"  None: {summary['last_none_count']} events")
            
            print(f"\n--- LONGEST duration subunit ---")
            for subunit, count in sorted(summary['longest_subunit_counts'].items()):
                print(f"  Subunit {subunit}: {count} events")
            print(f"  None: {summary['longest_none_count']} events")
        else:
            print(f"\n--- LAST residue (most recent before permeation) ---")
            print(f"  ASN: {summary['last_residue_stats']['asn_count']} events")
            if summary['last_residue_stats']['asp_count'] > 0:
                print(f"  ASP: {summary['last_residue_stats']['asp_count']} events")
            print(f"  GLU: {summary['last_residue_stats']['glu_count']} events")
            if summary['last_residue_stats']['gln_count'] > 0:
                print(f"  GLN: {summary['last_residue_stats']['gln_count']} events")
            print(f"  None: {summary['last_residue_stats']['none_count']} events")
            
            print(f"\n--- LONGEST residue (longest duration) ---")
            print(f"  ASN: {summary['longest_residue_stats']['asn_count']} events")
            if summary['longest_residue_stats']['asp_count'] > 0:
                print(f"  ASP: {summary['longest_residue_stats']['asp_count']} events")
            print(f"  GLU: {summary['longest_residue_stats']['glu_count']} events")
            if summary['longest_residue_stats']['gln_count'] > 0:
                print(f"  GLN: {summary['longest_residue_stats']['gln_count']} events")
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


    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    # Same palette as closest_residue_analysis.py / aggregate_closest_residue.py
    COLOR_MAP = ClosestResidueAnalysis.COLOR_MAP
    TYPE_ORDER = ClosestResidueAnalysis.TYPE_ORDER
    SUBUNIT_COLOR = '#8E9AAF'     # neutral: subunits are not residue types
    NO_CONTACT_COLOR = '#D9D9D9'

    def _winner_counts(self, which):
        """
        Count, per residue (or subunit), how often it was the LAST / LONGEST
        contact. Returns ([(label, type, count, mean_duration)], n_no_contact).
        """
        is_subunit_mode = self.subunit_groups is not None
        per_id = {}
        n_no_contact = 0
        for r in self.results:
            ident = r[f'{which}_identifier']
            residues = r[f'{which}_residues'] or []
            if not ident or not residues:
                n_no_contact += 1
                continue
            if is_subunit_mode:
                # Chain name from the residues, so old JSONs (A-D) get it too
                label, rtype = subunit_label(residues[0]['residue_pdb']), None
            else:
                label, rtype = residues[0]['residue_pdb'], residues[0].get('residue_type') or 'Unknown'
            d = per_id.setdefault(label, {'type': rtype, 'durations': []})
            d['durations'].append(r[f'{which}_duration'])

        items = sorted(((k, v['type'], len(v['durations']), float(np.mean(v['durations'])))
                        for k, v in per_id.items()), key=lambda x: x[2], reverse=True)
        return items, n_no_contact

    def create_bar_plot(self, which, title, output_file):
        """
        Bar plot of how often each residue (or subunit) was the LAST / LONGEST
        contact before end_2, plus a "No contact" bar for events where the ion
        never stayed within the threshold for min_duration frames.
        Each bar: count (% of ALL events) and mean contact duration.
        """
        items, n_no_contact = self._winner_counts(which)
        n_events = len(self.results)
        if n_events == 0:
            print(f"No data for {title}")
            return
        is_subunit_mode = self.subunit_groups is not None

        labels, colors, counts, mean_durs = [], [], [], []
        for label, rtype, count, mean_dur in items:
            if is_subunit_mode:
                labels.append(label)
                colors.append(self.SUBUNIT_COLOR)
            else:
                labels.append(f"{ONE_LETTER.get(rtype, '')}{label}")
                colors.append(self.COLOR_MAP.get(rtype, self.COLOR_MAP['Unknown']))
            counts.append(count)
            mean_durs.append(mean_dur)
        # "No contact" always shown last, even when it is 0, so runs compare
        labels.append('No contact')
        colors.append(self.NO_CONTACT_COLOR)
        counts.append(n_no_contact)
        mean_durs.append(None)

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels) + 3), 6.2))
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=colors, alpha=0.85, edgecolor='black',
                      linewidth=1.2, width=0.65, zorder=3)
        bars[-1].set_hatch('//')

        max_count = max(max(counts), 1)
        ax.set_ylim(0, max_count * 1.35)
        for bar, count, mean_dur in zip(bars, counts, mean_durs):
            text = f'{count} ({100.0 * count / n_events:.1f}%)'
            if mean_dur is not None:
                text += f'\n{mean_dur:.1f} fr'
            ax.text(bar.get_x() + bar.get_width() / 2., count + max_count * 0.03, text,
                    ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=17, fontweight='bold', rotation=40, ha='right')
        ax.set_ylabel('Number of Events', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_xlabel('Subunit' if is_subunit_mode else 'Residue', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_title(f"{title}\n(n = {n_events} events, < {self.threshold} Å for ≥ {self.min_duration} frames)",
                     fontsize=19, fontweight='bold', pad=22)
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(True, alpha=0.25, axis='y', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.1)

        legend_elements = []
        if not is_subunit_mode:
            present = {rtype for _, rtype, _, _ in items}
            legend_elements = [Patch(facecolor=self.COLOR_MAP.get(t, self.COLOR_MAP['Unknown']),
                                     edgecolor='black', label=t)
                               for t in self.TYPE_ORDER if t in present]
        legend_elements.append(Patch(facecolor=self.NO_CONTACT_COLOR, edgecolor='black', hatch='//',
                                     label='No contact'))
        # Outside the axes: the "No contact" bar is last and would sit under it
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=13, frameon=True, framealpha=0.9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def create_duration_plot(self, output_file):
        """
        Three histograms (frames):
          - duration of the LAST contact
          - duration of the LONGEST contact
          - gap between the end of the LAST contact and end_2
            (0 = ion was still in contact when it left the cavity)
        Events with no contact are left out; their number is in the title.
        """
        last = [r for r in self.results if r['last_identifier']]
        longest = [r for r in self.results if r['longest_identifier']]
        n_no_contact = len(self.results) - len(last)
        panels = [
            ('Last contact duration', [r['last_duration'] for r in last]),
            ('Longest contact duration', [r['longest_duration'] for r in longest]),
            ('Last contact end → end_2', [r['end_2_frame'] - r['last_end_frame'] for r in last]),
        ]

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (name, values) in zip(axes, panels):
            ax.set_title(name, fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('Frames', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of events', fontsize=14, fontweight='bold')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(True, alpha=0.25, axis='y', zorder=0)
            ax.set_axisbelow(True)
            if not values:
                ax.text(0.5, 0.5, 'no contacts', transform=ax.transAxes, ha='center', va='center', fontsize=14)
                continue
            values = np.asarray(values)
            bins = np.histogram_bin_edges(values, bins=min(30, max(5, int(values.max() - values.min()) + 1)))
            ax.hist(values, bins=bins, color=self.SUBUNIT_COLOR, edgecolor='black',
                    linewidth=0.8, alpha=0.9, zorder=3)
            ax.axvline(np.median(values), color='black', linestyle='--', linewidth=1.2,
                       label=f'median = {np.median(values):g}')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
            ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')

        fig.suptitle(f"{self.channel_type}: contacts before leaving the cavity "
                     f"(< {self.threshold} Å, ≥ {self.min_duration} frames; "
                     f"{n_no_contact}/{len(self.results)} events with no contact)",
                     fontsize=17, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def create_plots(self, output_dir=None):
        """Create all per-run plots under results_dir/plots/."""
        if not self.results:
            print("No results to plot.")
            return
        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_str = "subunit" if self.subunit_groups else "residue"
        unit = "Subunit" if self.subunit_groups else "Residue"
        tag = f"t{self.threshold}_d{self.min_duration}"
        print("\nCreating duration plots...")

        self.create_bar_plot('last', f"{self.channel_type}: LAST {unit} Before Leaving Cavity",
                             output_dir / f"last_{mode_str}_frequency_{tag}.png")
        self.create_bar_plot('longest', f"{self.channel_type}: LONGEST {unit} Contact Before Leaving Cavity",
                             output_dir / f"longest_{mode_str}_frequency_{tag}.png")
        self.create_duration_plot(output_dir / f"contact_durations_{mode_str}_{tag}.png")


class FirstContactAnalysis(LastAndLongestDurationAnalysis):
    """
    The FIRST residue/subunit the ion contacts after it enters the GLU/ASN
    cavity: same contact rule as LastAndLongestDurationAnalysis (closest
    residue within `threshold` for >= `min_duration` consecutive frames),
    but searched forwards from start_2 (first frame in the cavity) to end_2.

    Results are saved next to the "last" ones but under their own names
    (duration_first_complete_*.json, first_duration_*.json), so the two
    analyses can share a results folder without overwriting each other.
    """

    def analyze_event(self, event):
        """Analyze a single permeation event: first contact in start_2..end_2."""
        ion_id = event['ion_id']
        start_2_frame = event['start_2']   # first frame in the GLU/ASN cavity
        end_2_frame = event['end_2']

        # Same contact segmentation as the "last" analysis, over this window;
        # the first contact is the qualifying contact that starts earliest.
        result = self.find_last_and_longest_duration(ion_id, end_2_frame, start_2_frame)
        durations = result['all_durations']
        first = min(durations, key=lambda d: d['start_frame']) if durations else None

        return {
            'ion_id': int(ion_id),
            'start_2_frame': int(start_2_frame),
            'end_2_frame': int(end_2_frame),
            'mode': result['mode'],
            'first_identifier': first['identifier'] if first else None,
            'first_residues': first['residues'] if first else None,
            'first_duration': first['duration'] if first else 0,
            'first_start_frame': first['start_frame'] if first else None,
            'first_end_frame': first['end_frame'] if first else None,
            'n_durations_found': len(durations),
            'all_durations': durations
        }

    def run_analysis(self):
        """Run first-contact analysis for all permeation events."""
        mode_str = "SUBUNIT" if self.subunit_groups else "RESIDUE"
        print("\n" + "="*80)
        print(f"FIRST CONTACT AFTER ENTERING THE CAVITY - {mode_str} MODE")
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

            print(f"\nIon {result['ion_id']} (start_2 {result['start_2_frame']} → end_2 {result['end_2_frame']}):")
            if result['first_identifier']:
                res_str = ', '.join(f"{r['residue_type']} {r['residue_id']} ({r['residue_pdb']})"
                                    for r in result['first_residues'])
                print(f"  FIRST ({result['mode']} mode): {result['first_identifier']}  [{res_str}]")
                print(f"  Duration: {result['first_duration']} frames "
                      f"(frames {result['first_start_frame']} to {result['first_end_frame']})")
            else:
                print("  FIRST: None found")

    def save_results(self):
        """Save first-contact results (complete + first_duration_*.json)."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        mode_str = "subunit" if self.subunit_groups else "residue"

        complete_file = self.results_dir / f"duration_first_complete_{mode_str}.json"
        with open(complete_file, 'w') as f:
            json.dump({
                'analysis': 'first',
                'threshold': self.threshold,
                'min_duration': self.min_duration,
                'mode': mode_str,
                'channel_type': self.channel_type,
                'subunit_groups': self.subunit_groups,
                'asn_residues': self.asn_residues,
                'glu_residues': self.glu_residues,
                'residue_types': self.residue_types,
                'results': self.results
            }, f, indent=2)
        print(f"\n✓ Complete results saved to: {complete_file}")

        first_file = self.results_dir / f"first_duration_{mode_str}.json"
        with open(first_file, 'w') as f:
            json.dump({'threshold': self.threshold, 'min_duration': self.min_duration, 'mode': mode_str,
                       'results': [{
                           'ion_id': r['ion_id'],
                           'start_2_frame': r['start_2_frame'],
                           'end_2_frame': r['end_2_frame'],
                           'identifier': r['first_identifier'],
                           'residues': r['first_residues'],
                           'duration': r['first_duration'],
                           'start_frame': r['first_start_frame'],
                           'end_frame': r['first_end_frame']
                       } for r in self.results]}, f, indent=2)
        print(f"✓ FIRST results saved to: {first_file}")

    def create_summary(self):
        """Counts per residue type (or subunit), no-contact count, duration stats."""
        if not self.results:
            print("No results to summarize.")
            return
        mode_str = "subunit" if self.subunit_groups else "residue"
        hits = [r for r in self.results if r['first_identifier']]

        counts = {}
        for r in hits:
            key = str(r['first_identifier']) if self.subunit_groups else r['first_residues'][0]['residue_type']
            counts[key] = counts.get(key, 0) + 1
        durations = [r['first_duration'] for r in hits]
        delays = [r['first_start_frame'] - r['start_2_frame'] for r in hits]

        def stats(v):
            return {'avg': float(np.mean(v)) if v else None, 'std': float(np.std(v)) if v else None,
                    'min': int(np.min(v)) if v else None, 'max': int(np.max(v)) if v else None}

        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'min_duration': self.min_duration,
            'mode': mode_str,
            'first_counts': counts,
            'none_count': len(self.results) - len(hits),
            'first_duration_stats': stats(durations),
            'start_2_to_first_contact_stats': stats(delays)
        }
        output_file = self.results_dir / f"duration_first_summary_{mode_str}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print(f"FIRST CONTACT SUMMARY - {mode_str.upper()} MODE")
        print("="*80)
        print(f"Total events: {summary['total_events']}")
        print(f"Threshold: {self.threshold} Å, Min duration: {self.min_duration} frames")
        for k, v in sorted(counts.items()):
            print(f"  {'Subunit ' if self.subunit_groups else ''}{k}: {v} events")
        print(f"  None: {summary['none_count']} events")
        if durations:
            s = summary['first_duration_stats']
            print(f"\nFIRST contact duration: {s['avg']:.1f} ± {s['std']:.1f} frames ({s['min']} - {s['max']})")
            s = summary['start_2_to_first_contact_stats']
            print(f"start_2 → first contact: {s['avg']:.1f} ± {s['std']:.1f} frames ({s['min']} - {s['max']})")
        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)
        return summary

    def create_duration_plot(self, output_file):
        """
        Two histograms (frames):
          - duration of the FIRST contact
          - delay from start_2 (entering the cavity) to the start of the first
            contact (0 = in contact from the first cavity frame)
        """
        hits = [r for r in self.results if r['first_identifier']]
        panels = [
            ('First contact duration', [r['first_duration'] for r in hits]),
            ('start_2 → first contact', [r['first_start_frame'] - r['start_2_frame'] for r in hits]),
        ]

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, (name, values) in zip(axes, panels):
            ax.set_title(name, fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('Frames', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of events', fontsize=14, fontweight='bold')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(True, alpha=0.25, axis='y', zorder=0)
            ax.set_axisbelow(True)
            if not values:
                ax.text(0.5, 0.5, 'no contacts', transform=ax.transAxes, ha='center', va='center', fontsize=14)
                continue
            values = np.asarray(values)
            bins = np.histogram_bin_edges(values, bins=min(30, max(5, int(values.max() - values.min()) + 1)))
            ax.hist(values, bins=bins, color=self.SUBUNIT_COLOR, edgecolor='black',
                    linewidth=0.8, alpha=0.9, zorder=3)
            ax.axvline(np.median(values), color='black', linestyle='--', linewidth=1.2,
                       label=f'median = {np.median(values):g}')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
            ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')

        fig.suptitle(f"{self.channel_type}: first contact after entering the cavity "
                     f"(< {self.threshold} Å, ≥ {self.min_duration} frames; "
                     f"{len(self.results) - len(hits)}/{len(self.results)} events with no contact)",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def create_plots(self, output_dir=None):
        """Create the first-contact plots under results_dir/plots/."""
        if not self.results:
            print("No results to plot.")
            return
        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_str = "subunit" if self.subunit_groups else "residue"
        unit = "Subunit" if self.subunit_groups else "Residue"
        tag = f"t{self.threshold}_d{self.min_duration}"
        print("\nCreating first-contact plots...")

        self.create_bar_plot('first', f"{self.channel_type}: FIRST {unit} After Entering Cavity",
                             output_dir / f"first_{mode_str}_frequency_{tag}.png")
        self.create_duration_plot(output_dir / f"first_contact_durations_{mode_str}_{tag}.png")


def plot_duration_results(json_file, output_dir=None, channel_type=None, topology=None):
    """
    Re-create the per-run plots from an existing duration_analysis_complete_*.json
    (or duration_first_complete_*.json) without touching the trajectory.

    If `topology` is given, residue names are re-read from it and written back
    into the JSON (and into last_/longest_duration_*.json next to it). Use this
    for JSONs made before names came from the topology.
    """
    json_file = Path(json_file)
    with open(json_file) as f:
        data = json.load(f)

    if topology is not None:
        u = mda.Universe(str(topology))
        names = residue_names_from_topology(u, data['asn_residues'] + data['glu_residues'])

        def relabel(results):
            for r in results:
                for key in ('last_residues', 'longest_residues', 'first_residues', 'residues'):
                    for res in r.get(key) or []:
                        res['residue_type'] = names.get(res['residue_id'], 'Unknown')
                for d in r.get('all_durations') or []:
                    for res in d.get('residues') or []:
                        res['residue_type'] = names.get(res['residue_id'], 'Unknown')

        relabel(data['results'])
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        for side in (('first',) if data.get('analysis') == 'first' else ('last', 'longest')):
            side_file = json_file.parent / f"{side}_duration_{data['mode']}.json"
            if side_file.exists():
                with open(side_file) as f:
                    side_data = json.load(f)
                relabel(side_data['results'])
                with open(side_file, 'w') as f:
                    json.dump(side_data, f, indent=2)
        print(f"✓ Residue names updated from {topology}: {names}")

    cls = FirstContactAnalysis if data.get('analysis') == 'first' else LastAndLongestDurationAnalysis
    analyzer = cls.__new__(cls)
    analyzer.results = data['results']
    analyzer.threshold = data['threshold']
    analyzer.min_duration = data['min_duration']
    analyzer.subunit_groups = data.get('subunit_groups')
    analyzer.channel_type = channel_type or data.get('channel_type', '')
    analyzer.results_dir = json_file.parent
    analyzer.create_plots(output_dir)
    return analyzer


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
    analyzer.create_plots()
    
    return analyzer


def run_first_duration_analysis(universe, permeation_events, asn_residues,
                                glu_residues, channel_type="G12", threshold=3.0,
                                min_duration=10, analysis_start_frame=0,
                                residue_types=['ASN', 'GLU'], subunit_groups=None,
                                results_dir="."):
    """
    Convenience function to run the FIRST-contact analysis (start_2 -> end_2).
    Same parameters as run_last_duration_analysis.

    Returns:
    --------
    FirstContactAnalysis : The analysis object with results
    """
    analyzer = FirstContactAnalysis(universe, permeation_events, asn_residues,
                                    glu_residues, channel_type, threshold,
                                    min_duration, analysis_start_frame,
                                    residue_types, subunit_groups, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    analyzer.create_plots()

    return analyzer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-plot duration_analysis_complete_*.json for one run")
    parser.add_argument("json_file", help="Path to duration_analysis_complete_residue.json "
                                          "or duration_first_complete_residue.json (or _subunit)")
    parser.add_argument("--output_dir", default=None, help="Default: <json dir>/plots")
    parser.add_argument("--channel_type", default=None, help="Label for plot titles (e.g. G12)")
    parser.add_argument("--topology", default=None,
                        help="Topology file to re-read residue names from (fixes old JSONs)")
    args = parser.parse_args()
    plot_duration_results(args.json_file, args.output_dir, args.channel_type, args.topology)
