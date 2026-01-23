import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
from analysis.converter import convert_to_pdb_numbering


class ClosestResidueAnalysis:
    """
    Finds the closest residue (ASN or GLU) to permeating ions at key frames.
    Walks backwards from end_2 and end_3 frames until distance threshold is met.
    """
    
    def __init__(self, universe, permeation_events, asn_residues, glu_residues, 
                 channel_type="G12", threshold=3.5, analysis_start_frame=0, 
                 residue_types=['ASN', 'GLU'], results_dir="."):
        """
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MD universe object
        permeation_events : list
            List of permeation events from IonPermeationAnalysis
        asn_residues : list
            List of ASN residue IDs (includes ASP for G12)
        glu_residues : list
            List of GLU residue IDs
        channel_type : str
            Channel type for PDB numbering conversion (e.g., "G2", "G12")
        threshold : float
            Distance threshold in Angstroms (default: 3.5 Å)
        analysis_start_frame : int
            The starting frame of the overall analysis (default: 0)
        residue_types : list of str
            Which residue types to include: ['ASN', 'GLU'], ['ASN'], or ['GLU'] (default: ['ASN', 'GLU'])
        results_dir : Path or str
            Directory to save results
        """
        self.u = universe
        self.permeation_events = permeation_events
        self.asn_residues = asn_residues
        self.glu_residues = glu_residues
        self.channel_type = channel_type
        self.threshold = threshold
        self.analysis_start_frame = analysis_start_frame
        self.residue_types = residue_types
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
        
    # def convert_to_pdb_numbering(self, residue_id):
    #     """
    #     Converts a residue ID to a PDB-style numbering.
        
    #     Parameters:
    #     -----------
    #     residue_id : int
    #         The residue ID to convert
        
    #     Returns:
    #     --------
    #     str : PDB numbering (e.g., "152.A", "184.B", "173.D")
    #     """
    #     if self.channel_type == "G2":
    #         # GLU residues mapping
    #         glu_mapping = {
    #             98: "152.A",
    #             426: "152.C",
    #             754: "152.B",
    #             1082: "152.D"
    #         }
    #         # ASN residues mapping
    #         asn_mapping = {
    #             130: "184.A",
    #             458: "184.C",
    #             786: "184.B",
    #             1114: "184.D"
    #         }
    #         # Check if it's a GLU residue
    #         if residue_id in glu_mapping:
    #             return glu_mapping[residue_id]
    #         # Check if it's an ASN residue
    #         if residue_id in asn_mapping:
    #             return asn_mapping[residue_id]
        
    #     elif self.channel_type == "G12":
    #         # GLU residues mapping
    #         glu_mapping = {
    #             422: "152.A",
    #             98: "152.B",
    #             747: "152.C",
    #             1073: "141.D"
    #         }
    #         # ASN and ASP residues mapping
    #         asn_asp_mapping = {
    #             454: "184.A",
    #             130: "184.B",
    #             779: "184.C",
    #             1105: "173.D"  # This is ASP, not ASN
    #         }
    #         # Check if it's a GLU residue
    #         if residue_id in glu_mapping:
    #             return glu_mapping[residue_id]
    #         # Check if it's an ASN/ASP residue
    #         if residue_id in asn_asp_mapping:
    #             return asn_asp_mapping[residue_id]
        
    #     # If not found in mappings, return the original residue_id as string
    #     return str(residue_id)
        
    def find_closest_residue_at_frame(self, ion_id, frame):
        """
        Find the closest residue (ASN/ASP or GLU) to an ion at a specific frame.
        
        Parameters:
        -----------
        ion_id : int
            Ion residue ID
        frame : int
            Frame number
            
        Returns:
        --------
        dict : {
            'closest_residue_id': int,
            'closest_residue_pdb': str (PDB numbering),
            'residue_type': 'ASN', 'ASP', or 'GLU',
            'distance': float,
            'frame': int
        }
        """
        # Go to frame
        self.u.trajectory[frame]
        
        # Get ion position
        ion = self.u.select_atoms(f"resid {ion_id}")
        if len(ion) == 0:
            return None
        ion_pos = ion.positions[0]
        
        # Check all ASN and GLU residues (based on residue_types filter)
        min_distance = float('inf')
        closest_resid = None
        closest_type = None
        
        # Check ASN residues (includes ASP for G12) - only if ASN is in residue_types
        if 'ASN' in self.residue_types:
            for resid in self.asn_residues:
                res_atoms = self.u.select_atoms(f"resid {resid}")
                if len(res_atoms) > 0:
                    # Calculate minimum distance to any atom in this residue
                    distances = np.linalg.norm(res_atoms.positions - ion_pos, axis=1)
                    min_dist_to_res = np.min(distances)
                    
                    if min_dist_to_res < min_distance:
                        min_distance = min_dist_to_res
                        closest_resid = resid
                        # Check if this is the ASP residue (1105 for G12)
                        if self.asp_residue is not None and resid == self.asp_residue:
                            closest_type = 'ASP'
                        else:
                            closest_type = 'ASN'
        
        # Check GLU residues - only if GLU is in residue_types
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
        
        return {
            'closest_residue_id': int(closest_resid) if closest_resid else None,
            'closest_residue_pdb': convert_to_pdb_numbering(closest_resid,self.channel_type) if closest_resid else None,
            'residue_type': closest_type,
            'distance': float(min_distance),
            'frame': int(frame)
        }
    
    def find_threshold_crossing(self, ion_id, end_frame, start_frame):
        """
        Walk backwards from end_frame to start_frame until distance < threshold.
        
        Parameters:
        -----------
        ion_id : int
            Ion residue ID
        end_frame : int
            Starting frame (walk backwards from here)
        start_frame : int
            Don't go before this frame
            
        Returns:
        --------
        dict : Closest residue info when threshold is met, or best result found
        """
        best_result = None
        min_distance_found = float('inf')
        
        for frame in range(end_frame, start_frame - 1, -1):
            result = self.find_closest_residue_at_frame(ion_id, frame)
            
            if result:
                # Track the overall minimum distance
                if result['distance'] < min_distance_found:
                    min_distance_found = result['distance']
                    best_result = result
                
                # If threshold is met, return immediately
                if result['distance'] < self.threshold:
                    return result
        
        # If no frame meets threshold, return the best (minimum distance) we found
        return best_result
    
    def analyze_event(self, event):
        """
        Analyze a single permeation event with 4 distinct cases.
        
        Parameters:
        -----------
        event : dict
            Permeation event dictionary
            
        Returns:
        --------
        dict : Analysis results for this event with 4 cases
        """
        ion_id = event['ion_id']
        
        # ===== CASE 1: Closest residue at end_2 (NO threshold) =====
        end_2_frame = event['end_2']
        case_1_result = self.find_closest_residue_at_frame(ion_id, end_2_frame)
        
        # ===== CASE 2: Walk backwards from end_2 until threshold is met =====
        # Walk back to the analysis start frame
        case_2_result = self.find_threshold_crossing(ion_id, end_2_frame, self.analysis_start_frame)
        
        # ===== CASE 3: Closest residue at end_3 (NO threshold) =====
        end_3_frame = event['end_3']
        case_3_result = self.find_closest_residue_at_frame(ion_id, end_3_frame)
        
        # ===== CASE 4: Walk backwards from end_3 until threshold is met =====
        # Walk back to the analysis start frame
        case_4_result = self.find_threshold_crossing(ion_id, end_3_frame, self.analysis_start_frame)
        
        return {
            'ion_id': int(ion_id),
            'event_start_frame': int(event['start_1']),
            'case_1_end_2_closest': case_1_result,  # Just closest at end_2
            'case_2_end_2_threshold': case_2_result,  # Walk backwards to threshold
            'case_3_end_3_closest': case_3_result,  # Just closest at end_3
            'case_4_end_3_threshold': case_4_result  # Walk backwards to threshold
        }
    
    def run_analysis(self):
        """
        Run closest residue analysis for all permeation events.
        """
        print("\n" + "="*80)
        print("CLOSEST RESIDUE ANALYSIS (4 CASES)")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å")
        print(f"Analyzing residue types: {', '.join(self.residue_types)}")
        
        # Count residues being analyzed
        n_asn = len(self.asn_residues) if 'ASN' in self.residue_types else 0
        n_glu = len(self.glu_residues) if 'GLU' in self.residue_types else 0
        total_residues = n_asn + n_glu
        
        print(f"Total residues in competition: {total_residues}", end="")
        if 'ASN' in self.residue_types and 'GLU' in self.residue_types:
            print(f" ({n_asn} ASN + {n_glu} GLU)")
        elif 'ASN' in self.residue_types:
            print(f" ({n_asn} ASN only)")
        elif 'GLU' in self.residue_types:
            print(f" ({n_glu} GLU only)")
        
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        
        for event in self.permeation_events:
            result = self.analyze_event(event)
            self.results.append(result)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Ion {result['ion_id']} (Event starting at frame {result['event_start_frame']}):")
            print(f"{'='*60}")
            
            # Case 1
            if result['case_1_end_2_closest']:
                c1 = result['case_1_end_2_closest']
                print(f"CASE 1 - Closest at end_2 (no threshold):")
                print(f"  Residue: {c1['residue_type']} {c1['closest_residue_id']} (PDB: {c1['closest_residue_pdb']})")
                print(f"  Distance: {c1['distance']:.2f} Å")
                print(f"  Frame: {c1['frame']}")
            
            # Case 2
            if result['case_2_end_2_threshold']:
                c2 = result['case_2_end_2_threshold']
                print(f"CASE 2 - Threshold crossing from end_2:")
                print(f"  Residue: {c2['residue_type']} {c2['closest_residue_id']} (PDB: {c2['closest_residue_pdb']})")
                print(f"  Distance: {c2['distance']:.2f} Å")
                print(f"  Frame: {c2['frame']}")
                if c2['distance'] >= self.threshold:
                    print(f"  WARNING: Never reached threshold (showing closest found)")
            
            # Case 3
            if result['case_3_end_3_closest']:
                c3 = result['case_3_end_3_closest']
                print(f"CASE 3 - Closest at end_3 (no threshold):")
                print(f"  Residue: {c3['residue_type']} {c3['closest_residue_id']} (PDB: {c3['closest_residue_pdb']})")
                print(f"  Distance: {c3['distance']:.2f} Å")
                print(f"  Frame: {c3['frame']}")
            
            # Case 4
            if result['case_4_end_3_threshold']:
                c4 = result['case_4_end_3_threshold']
                print(f"CASE 4 - Threshold crossing from end_3:")
                print(f"  Residue: {c4['residue_type']} {c4['closest_residue_id']} (PDB: {c4['closest_residue_pdb']})")
                print(f"  Distance: {c4['distance']:.2f} Å")
                print(f"  Frame: {c4['frame']}")
                if c4['distance'] >= self.threshold:
                    print(f"  WARNING: Never reached threshold (showing closest found)")
    
    def save_results(self):
        """
        Save analysis results to JSON file.
        """
        # Ensure directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.results_dir / "closest_residue_analysis.json"
        
        output_data = {
            'threshold': self.threshold,
            'asn_residues': self.asn_residues,
            'glu_residues': self.glu_residues,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def create_summary(self):
        """
        Create summary statistics for all 4 cases.
        """
        if not self.results:
            print("No results to summarize.")
            return
        
        # Case 1: Closest at end_2 (no threshold)
        case1_asn = sum(1 for r in self.results if r['case_1_end_2_closest'] and r['case_1_end_2_closest']['residue_type'] == 'ASN')
        case1_asp = sum(1 for r in self.results if r['case_1_end_2_closest'] and r['case_1_end_2_closest']['residue_type'] == 'ASP')
        case1_glu = sum(1 for r in self.results if r['case_1_end_2_closest'] and r['case_1_end_2_closest']['residue_type'] == 'GLU')
        case1_distances = [r['case_1_end_2_closest']['distance'] for r in self.results if r['case_1_end_2_closest']]
        
        # Case 2: Threshold crossing from end_2
        case2_asn = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'ASN')
        case2_asp = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'ASP')
        case2_glu = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'GLU')
        case2_distances = [r['case_2_end_2_threshold']['distance'] for r in self.results if r['case_2_end_2_threshold']]
        
        # Case 3: Closest at end_3 (no threshold)
        case3_asn = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'ASN')
        case3_asp = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'ASP')
        case3_glu = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'GLU')
        case3_distances = [r['case_3_end_3_closest']['distance'] for r in self.results if r['case_3_end_3_closest']]
        
        # Case 4: Threshold crossing from end_3
        case4_asn = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'ASN')
        case4_asp = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'ASP')
        case4_glu = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'GLU')
        case4_distances = [r['case_4_end_3_threshold']['distance'] for r in self.results if r['case_4_end_3_threshold']]
        
        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'case_1_end_2_closest': {
                'asn_count': case1_asn,
                'asp_count': case1_asp,
                'glu_count': case1_glu,
                'avg_distance': float(np.mean(case1_distances)) if case1_distances else None,
                'std_distance': float(np.std(case1_distances)) if case1_distances else None
            },
            'case_2_end_2_threshold': {
                'asn_count': case2_asn,
                'asp_count': case2_asp,
                'glu_count': case2_glu,
                'avg_distance': float(np.mean(case2_distances)) if case2_distances else None,
                'std_distance': float(np.std(case2_distances)) if case2_distances else None
            },
            'case_3_end_3_closest': {
                'asn_count': case3_asn,
                'asp_count': case3_asp,
                'glu_count': case3_glu,
                'avg_distance': float(np.mean(case3_distances)) if case3_distances else None,
                'std_distance': float(np.std(case3_distances)) if case3_distances else None
            },
            'case_4_end_3_threshold': {
                'asn_count': case4_asn,
                'asp_count': case4_asp,
                'glu_count': case4_glu,
                'avg_distance': float(np.mean(case4_distances)) if case4_distances else None,
                'std_distance': float(np.std(case4_distances)) if case4_distances else None
            }
        }
        
        # Save summary
        output_file = self.results_dir / "closest_residue_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS - 4 CASES")
        print("="*80)
        print(f"Total permeation events: {summary['total_events']}")
        print(f"Distance threshold: {self.threshold} Å (used in Cases 2 & 4)")
        
        print(f"\nCASE 1 - Closest at end_2 (no threshold):")
        print(f"  ASN closest: {case1_asn} events")
        if case1_asp > 0:
            print(f"  ASP closest: {case1_asp} events")
        print(f"  GLU closest: {case1_glu} events")
        if summary['case_1_end_2_closest']['avg_distance']:
            print(f"  Avg distance: {summary['case_1_end_2_closest']['avg_distance']:.2f} ± {summary['case_1_end_2_closest']['std_distance']:.2f} Å")
        
        print(f"\nCASE 2 - Threshold crossing from end_2 (threshold={self.threshold} Å):")
        print(f"  ASN closest: {case2_asn} events")
        if case2_asp > 0:
            print(f"  ASP closest: {case2_asp} events")
        print(f"  GLU closest: {case2_glu} events")
        if summary['case_2_end_2_threshold']['avg_distance']:
            print(f"  Avg distance: {summary['case_2_end_2_threshold']['avg_distance']:.2f} ± {summary['case_2_end_2_threshold']['std_distance']:.2f} Å")
        
        print(f"\nCASE 3 - Closest at end_3 (no threshold):")
        print(f"  ASN closest: {case3_asn} events")
        if case3_asp > 0:
            print(f"  ASP closest: {case3_asp} events")
        print(f"  GLU closest: {case3_glu} events")
        if summary['case_3_end_3_closest']['avg_distance']:
            print(f"  Avg distance: {summary['case_3_end_3_closest']['avg_distance']:.2f} ± {summary['case_3_end_3_closest']['std_distance']:.2f} Å")
        
        print(f"\nCASE 4 - Threshold crossing from end_3 (threshold={self.threshold} Å):")
        print(f"  ASN closest: {case4_asn} events")
        if case4_asp > 0:
            print(f"  ASP closest: {case4_asp} events")
        print(f"  GLU closest: {case4_glu} events")
        if summary['case_4_end_3_threshold']['avg_distance']:
            print(f"  Avg distance: {summary['case_4_end_3_threshold']['avg_distance']:.2f} ± {summary['case_4_end_3_threshold']['std_distance']:.2f} Å")
        
        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)
        
        return summary


def run_closest_residue_analysis(universe, permeation_events, asn_residues, 
                                  glu_residues, channel_type="G12", threshold=3.5, 
                                  analysis_start_frame=0, residue_types=['ASN', 'GLU'], 
                                  results_dir="."):
    """
    Convenience function to run the complete closest residue analysis.
    
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
        Channel type for PDB numbering conversion (e.g., "G2", "G12")
    threshold : float
        Distance threshold in Angstroms (default: 3.5 Å)
    analysis_start_frame : int
        The starting frame of the overall analysis (default: 0)
    residue_types : list of str
        Which residue types to include: ['ASN', 'GLU'], ['ASN'], or ['GLU'] (default: ['ASN', 'GLU'])
    results_dir : Path or str
        Directory to save results
        
    Returns:
    --------
    ClosestResidueAnalysis : The analysis object with results
    """
    analyzer = ClosestResidueAnalysis(universe, permeation_events, asn_residues, 
                                     glu_residues, channel_type, threshold, 
                                     analysis_start_frame, residue_types, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    
    return analyzer