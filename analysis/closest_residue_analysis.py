import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from analysis.converter import convert_to_pdb_numbering


# Protonated / alternate force-field names folded onto the standard residue name
RESNAME_ALIASES = {'ASH': 'ASP', 'ASPP': 'ASP', 'GLH': 'GLU', 'GLUP': 'GLU'}
ONE_LETTER = {'GLU': 'E', 'GLN': 'Q', 'ASP': 'D', 'ASN': 'N'}


def residue_names_from_topology(universe, resids):
    """{resid: residue name} read from the topology, e.g. {1105: 'ASP'}."""
    names = {}
    for resid in resids:
        res_atoms = universe.select_atoms(f"resid {resid}")
        if len(res_atoms) > 0:
            name = str(res_atoms.resnames[0]).upper()
            names[int(resid)] = RESNAME_ALIASES.get(name, name)
    return names


class ResidueDistanceCalculator:
    """
    Minimum ion-residue distances over a block of frames, computed fast:
    the atom selections are built once (not every frame), frames are read
    in order as one block, and all residues are done in one NumPy call.

    Residues are kept in the order given; ties go to the earlier one, the
    same as the old per-residue loops with a strict '<'.
    """

    def __init__(self, universe, resids):
        self.u = universe
        groups = [(int(r), universe.select_atoms(f"resid {r}")) for r in resids]
        groups = [(r, g) for r, g in groups if len(g) > 0]   # old loops skipped empty ones
        self.resids = [r for r, _ in groups]
        if groups:
            self.atoms = universe.atoms[np.concatenate([g.indices for _, g in groups])]
            self.starts = np.cumsum([0] + [len(g) for _, g in groups[:-1]])
        self._ions = {}

    def _ion(self, ion_id):
        if ion_id not in self._ions:
            self._ions[ion_id] = self.u.select_atoms(f"resid {ion_id}")
        return self._ions[ion_id]

    def min_distances(self, ion_id, start, stop):
        """
        Frames start..stop (inclusive). Returns (frames, D) with D[i, j] the
        minimum distance from the ion to residue self.resids[j] at frames[i],
        or (frames, None) when the ion or the residues are missing.
        """
        frames = np.arange(start, stop + 1)
        ion = self._ion(ion_id)
        if len(ion) == 0 or not self.resids or len(frames) == 0:
            return frames, None
        D = np.empty((len(frames), len(self.resids)), dtype=np.float32)
        for i, _ in enumerate(self.u.trajectory[start:stop + 1]):
            d = np.linalg.norm(self.atoms.positions - ion.positions[0], axis=1)
            D[i] = np.minimum.reduceat(d, self.starts)
        return frames, D


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
        
        # Residue names come from the topology, so mutants are labelled
        # correctly (e.g. Q141 in E141Q, N173 in D173N) without hard-coding.
        self.resnames = residue_names_from_topology(self.u, asn_residues + glu_residues)

        # Lower ring first, then upper ring: same order (and tie-breaking) as before
        competing = (list(asn_residues) if 'ASN' in residue_types else []) + \
                    (list(glu_residues) if 'GLU' in residue_types else [])
        self.distances = ResidueDistanceCalculator(self.u, competing)
        
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
            'residue_type': residue name from the topology (e.g. 'ASN', 'ASP', 'GLU', 'GLN'),
            'distance': float,
            'frame': int
        }
        """
        frames, D = self.distances.min_distances(ion_id, frame, frame)
        if D is None:
            return None
        return self._closest_result(D[0], frame)

    def _closest_result(self, row, frame):
        """Result dict for one frame, from its row of residue distances."""
        j = int(np.argmin(row))
        resid = self.distances.resids[j]
        return {
            'closest_residue_id': int(resid),
            'closest_residue_pdb': convert_to_pdb_numbering(resid, self.channel_type),
            'residue_type': self.resnames.get(resid, 'Unknown'),
            'distance': float(row[j]),
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

        # Read backwards in blocks (each block read forwards, which is fast).
        # Blocks start small, since the threshold is often met within a few
        # frames, and grow so a long walk back does not cost a read per frame.
        block = 50
        block_end = end_frame
        while block_end >= start_frame:
            block_start = max(start_frame, block_end - block + 1)
            frames, D = self.distances.min_distances(ion_id, block_start, block_end)
            if D is None:
                return None
            row_min = D.min(axis=1)
            for i in range(len(frames) - 1, -1, -1):
                # Track the overall minimum distance
                if row_min[i] < min_distance_found:
                    min_distance_found = row_min[i]
                    best_result = (D[i], frames[i])

                # If threshold is met, return immediately
                if row_min[i] < self.threshold:
                    return self._closest_result(D[i], frames[i])
            block_end = block_start - 1
            block = min(block * 2, 2000)

        # If no frame meets threshold, return the best (minimum distance) we found
        return self._closest_result(*best_result) if best_result else None
    
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
            'end_2_frame': int(end_2_frame),
            'end_3_frame': int(end_3_frame),
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
            'channel_type': self.channel_type,
            'residue_types': self.residue_types,
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
        case1_gln = sum(1 for r in self.results if r['case_1_end_2_closest'] and r['case_1_end_2_closest']['residue_type'] == 'GLN')
        case1_distances = [r['case_1_end_2_closest']['distance'] for r in self.results if r['case_1_end_2_closest']]
        
        # Case 2: Threshold crossing from end_2
        case2_asn = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'ASN')
        case2_asp = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'ASP')
        case2_glu = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'GLU')
        case2_gln = sum(1 for r in self.results if r['case_2_end_2_threshold'] and r['case_2_end_2_threshold']['residue_type'] == 'GLN')
        case2_distances = [r['case_2_end_2_threshold']['distance'] for r in self.results if r['case_2_end_2_threshold']]
        
        # Case 3: Closest at end_3 (no threshold)
        case3_asn = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'ASN')
        case3_asp = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'ASP')
        case3_glu = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'GLU')
        case3_gln = sum(1 for r in self.results if r['case_3_end_3_closest'] and r['case_3_end_3_closest']['residue_type'] == 'GLN')
        case3_distances = [r['case_3_end_3_closest']['distance'] for r in self.results if r['case_3_end_3_closest']]
        
        # Case 4: Threshold crossing from end_3
        case4_asn = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'ASN')
        case4_asp = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'ASP')
        case4_glu = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'GLU')
        case4_gln = sum(1 for r in self.results if r['case_4_end_3_threshold'] and r['case_4_end_3_threshold']['residue_type'] == 'GLN')
        case4_distances = [r['case_4_end_3_threshold']['distance'] for r in self.results if r['case_4_end_3_threshold']]
        
        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'case_1_end_2_closest': {
                'asn_count': case1_asn,
                'asp_count': case1_asp,
                'glu_count': case1_glu,
                'gln_count': case1_gln,
                'avg_distance': float(np.mean(case1_distances)) if case1_distances else None,
                'std_distance': float(np.std(case1_distances)) if case1_distances else None
            },
            'case_2_end_2_threshold': {
                'asn_count': case2_asn,
                'asp_count': case2_asp,
                'glu_count': case2_glu,
                'gln_count': case2_gln,
                'avg_distance': float(np.mean(case2_distances)) if case2_distances else None,
                'std_distance': float(np.std(case2_distances)) if case2_distances else None
            },
            'case_3_end_3_closest': {
                'asn_count': case3_asn,
                'asp_count': case3_asp,
                'glu_count': case3_glu,
                'gln_count': case3_gln,
                'avg_distance': float(np.mean(case3_distances)) if case3_distances else None,
                'std_distance': float(np.std(case3_distances)) if case3_distances else None
            },
            'case_4_end_3_threshold': {
                'asn_count': case4_asn,
                'asp_count': case4_asp,
                'glu_count': case4_glu,
                'gln_count': case4_gln,
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
        if case1_gln > 0:
            print(f"  GLN closest: {case1_gln} events")
        if summary['case_1_end_2_closest']['avg_distance']:
            print(f"  Avg distance: {summary['case_1_end_2_closest']['avg_distance']:.2f} ± {summary['case_1_end_2_closest']['std_distance']:.2f} Å")
        
        print(f"\nCASE 2 - Threshold crossing from end_2 (threshold={self.threshold} Å):")
        print(f"  ASN closest: {case2_asn} events")
        if case2_asp > 0:
            print(f"  ASP closest: {case2_asp} events")
        print(f"  GLU closest: {case2_glu} events")
        if case2_gln > 0:
            print(f"  GLN closest: {case2_gln} events")
        if summary['case_2_end_2_threshold']['avg_distance']:
            print(f"  Avg distance: {summary['case_2_end_2_threshold']['avg_distance']:.2f} ± {summary['case_2_end_2_threshold']['std_distance']:.2f} Å")
        
        print(f"\nCASE 3 - Closest at end_3 (no threshold):")
        print(f"  ASN closest: {case3_asn} events")
        if case3_asp > 0:
            print(f"  ASP closest: {case3_asp} events")
        print(f"  GLU closest: {case3_glu} events")
        if case3_gln > 0:
            print(f"  GLN closest: {case3_gln} events")
        if summary['case_3_end_3_closest']['avg_distance']:
            print(f"  Avg distance: {summary['case_3_end_3_closest']['avg_distance']:.2f} ± {summary['case_3_end_3_closest']['std_distance']:.2f} Å")
        
        print(f"\nCASE 4 - Threshold crossing from end_3 (threshold={self.threshold} Å):")
        print(f"  ASN closest: {case4_asn} events")
        if case4_asp > 0:
            print(f"  ASP closest: {case4_asp} events")
        print(f"  GLU closest: {case4_glu} events")
        if case4_gln > 0:
            print(f"  GLN closest: {case4_gln} events")
        if summary['case_4_end_3_threshold']['avg_distance']:
            print(f"  Avg distance: {summary['case_4_end_3_threshold']['avg_distance']:.2f} ± {summary['case_4_end_3_threshold']['std_distance']:.2f} Å")
        
        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)
        
        return summary


    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    # Same palette as aggregate_closest_residue.py so per-run and
    # all-runs figures read the same way.
    COLOR_MAP = {
        'GLU': '#E8746B',   # muted coral
        'GLN': '#E8A85B',   # muted amber (E141Q)
        'ASN': '#5AA9C9',   # muted blue
        'ASP': '#6FBF73',   # muted green
        'Unknown': '#B0B0B0',
    }
    TYPE_ORDER = ['GLU', 'GLN', 'ASN', 'ASP', 'Unknown']

    # (result key, stage label, file stem, uses threshold, end-frame key)
    PLOT_CASES = [
        ('case_1_end_2_closest',   'Ion Leaves GLU/ASN Cavity', 'ion_leaves_cavity_closest',             False, 'end_2_frame'),
        ('case_2_end_2_threshold', 'Ion Leaves GLU/ASN Cavity', 'ion_leaves_cavity_threshold_crossing',  True,  'end_2_frame'),
        ('case_3_end_3_closest',   'Ion Leaves HBC',            'ion_leaves_hbc_closest',                False, 'end_3_frame'),
        ('case_4_end_3_threshold', 'Ion Leaves HBC',            'ion_leaves_hbc_threshold_crossing',     True,  'end_3_frame'),
    ]

    def create_bar_plot(self, case_key, case_title, output_file, uses_threshold):
        """
        Bar plot for one case: how often each residue was the closest one,
        with count, percentage of events and average distance on each bar.

        For the threshold cases (2 & 4), events where the ion NEVER came
        within the threshold are drawn as a hatched segment on top of the
        bar, because for those events the residue shown is just the closest
        one found, not a real contact.
        """
        per_res = {}
        for r in self.results:
            c = r.get(case_key)
            if not c or c.get('closest_residue_pdb') is None:
                continue
            d = per_res.setdefault(c['closest_residue_pdb'], {
                'type': c.get('residue_type') or 'Unknown',
                'reached': 0, 'not_reached': 0, 'distances': []})
            if uses_threshold and c['distance'] >= self.threshold:
                d['not_reached'] += 1
            else:
                d['reached'] += 1
            d['distances'].append(c['distance'])

        if not per_res:
            print(f"No data for {case_title}")
            return

        sorted_res = sorted(per_res.items(),
                            key=lambda x: x[1]['reached'] + x[1]['not_reached'], reverse=True)
        residues = [k for k, _ in sorted_res]
        reached = [v['reached'] for _, v in sorted_res]
        not_reached = [v['not_reached'] for _, v in sorted_res]
        totals = [a + b for a, b in zip(reached, not_reached)]
        avg_distances = [np.mean(v['distances']) for _, v in sorted_res]
        colors = [self.COLOR_MAP.get(v['type'], self.COLOR_MAP['Unknown']) for _, v in sorted_res]

        n_events = sum(totals)
        percentages = [100.0 * t / n_events for t in totals]

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(10, 6.2))
        x = np.arange(len(residues))

        bars = ax.bar(x, reached, color=colors, alpha=0.85, edgecolor='black',
                      linewidth=1.2, width=0.65, zorder=3)
        if any(not_reached):
            ax.bar(x, not_reached, bottom=reached, color=colors, alpha=0.35,
                   edgecolor='black', linewidth=1.2, width=0.65, hatch='//', zorder=3)

        max_count = max(totals)
        ax.set_ylim(0, max_count * 1.35)

        for bar, total, pct, avg_dist in zip(bars, totals, percentages, avg_distances):
            ax.text(bar.get_x() + bar.get_width() / 2., total + max_count * 0.03,
                    f'{total} ({pct:.1f}%)\n{avg_dist:.2f} Å',
                    ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=4)

        # Tick labels carry the one-letter code from the topology, e.g. D173.G1
        tick_labels = [f"{ONE_LETTER.get(v['type'], '')}{k}" for k, v in sorted_res]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=17, fontweight='bold', rotation=40, ha='right')
        ax.set_ylabel('Number of Times Closest', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_xlabel('Residue', fontsize=19, fontweight='bold', labelpad=10)
        ax.set_title(f"{case_title}\n(n = {n_events} events)", fontsize=20, fontweight='bold', pad=22)
        ax.tick_params(axis='both', labelsize=15)

        ax.grid(True, alpha=0.25, axis='y', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.1)

        present_types = sorted({v['type'] for _, v in sorted_res},
                               key=lambda t: self.TYPE_ORDER.index(t) if t in self.TYPE_ORDER else len(self.TYPE_ORDER))
        legend_elements = [Patch(facecolor=self.COLOR_MAP.get(t, self.COLOR_MAP['Unknown']),
                                 edgecolor='black', label=t) for t in present_types]
        if any(not_reached):
            legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='//',
                                         label=f'never < {self.threshold} Å\n(closest shown)'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=13, frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def create_lag_plot(self, output_file):
        """
        For the threshold cases (2 & 4): how many frames before the exit
        frame (end_2 / end_3) the last contact happened. Lag 0 means the
        ion was still in contact at the exit frame. Events that never
        reached the threshold are left out.
        """
        threshold_cases = [c for c in self.PLOT_CASES if c[3]]
        if not any(r.get(c[4]) is not None for r in self.results for c in threshold_cases):
            print("Skipping lag plot: results have no end_2/end_3 frames (re-run the analysis to get them).")
            return

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(1, len(threshold_cases), figsize=(14, 5.5), sharey=False)

        for ax, (case_key, stage, _, _, end_key) in zip(np.atleast_1d(axes), threshold_cases):
            lags_by_type = {}
            for r in self.results:
                c = r.get(case_key)
                if not c or r.get(end_key) is None or c['distance'] >= self.threshold:
                    continue
                lags_by_type.setdefault(c.get('residue_type') or 'Unknown', []).append(r[end_key] - c['frame'])

            ax.set_title(f"{stage}", fontsize=17, fontweight='bold', pad=12)
            ax.set_xlabel('Frames before exit frame', fontsize=15, fontweight='bold')
            ax.set_ylabel('Number of events', fontsize=15, fontweight='bold')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(True, alpha=0.25, axis='y', zorder=0)
            ax.set_axisbelow(True)

            if not lags_by_type:
                ax.text(0.5, 0.5, 'no contacts', transform=ax.transAxes, ha='center', va='center', fontsize=14)
                continue

            types = [t for t in self.TYPE_ORDER if t in lags_by_type]
            all_lags = np.concatenate([lags_by_type[t] for t in types])
            bins = np.histogram_bin_edges(all_lags, bins=min(30, max(5, int(all_lags.max()) + 1)))
            ax.hist([lags_by_type[t] for t in types], bins=bins, stacked=True,
                    color=[self.COLOR_MAP[t] for t in types], edgecolor='black',
                    linewidth=0.8, alpha=0.85, label=types, zorder=3)
            ax.axvline(np.median(all_lags), color='black', linestyle='--', linewidth=1.2,
                       label=f'median = {np.median(all_lags):.0f}')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # Headroom so the one-row legend sits above the bars, never on them
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper center',
                      ncol=len(types) + 1, columnspacing=1.0, handlelength=1.5)

        fig.suptitle(f"{self.channel_type}: Last contact (< {self.threshold} Å) before exit",
                     fontsize=19, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def create_plots(self, output_dir=None):
        """
        Create all per-run plots under results_dir/plots/:
          - one bar plot per case (same layout as aggregate_closest_residue.py)
          - a lag histogram for the threshold cases
        """
        if not self.results:
            print("No results to plot.")
            return

        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\nCreating closest-residue plots...")

        for case_key, stage, stem, uses_threshold, _ in self.PLOT_CASES:
            # Cases 1 & 3 are snapshots at the exit frame, so the threshold
            # has no effect on them and is left out of the title.
            if uses_threshold:
                title = f"{self.channel_type}: {stage} - Threshold={self.threshold}Å"
            else:
                title = f"{self.channel_type}: {stage} - Closest Residue"
            self.create_bar_plot(case_key, title,
                                 output_dir / f"{stem}_threshold{self.threshold}.png",
                                 uses_threshold)

        self.create_lag_plot(output_dir / f"last_contact_lag_threshold{self.threshold}.png")


def plot_closest_residue_results(json_file, output_dir=None, channel_type=None, topology=None):
    """
    Re-create the per-run plots from an existing closest_residue_analysis.json
    without touching the trajectory.

    If `topology` is given (e.g. the run's .prmtop/.psf/.pdb), residue names are
    re-read from it and written back into the JSON. Use this for JSONs made
    before names came from the topology, where mutants (E141Q, D173N) and
    some ASP residues were mislabelled.
    """
    json_file = Path(json_file)
    with open(json_file) as f:
        data = json.load(f)

    if topology is not None:
        u = mda.Universe(str(topology))
        resids = {c['closest_residue_id'] for r in data['results'] for k, c in r.items()
                  if k.startswith('case_') and c and c.get('closest_residue_id') is not None}
        names = residue_names_from_topology(u, sorted(resids))
        for r in data['results']:
            for k, c in r.items():
                if k.startswith('case_') and c and c.get('closest_residue_id') is not None:
                    c['residue_type'] = names.get(c['closest_residue_id'], 'Unknown')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\u2713 Residue names updated from {topology}: {names}")

    analyzer = ClosestResidueAnalysis.__new__(ClosestResidueAnalysis)
    analyzer.results = data['results']
    analyzer.threshold = data['threshold']
    analyzer.channel_type = channel_type or data.get('channel_type', '')
    analyzer.results_dir = json_file.parent
    analyzer.create_plots(output_dir)
    return analyzer


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
    analyzer.create_plots()
    
    return analyzer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-plot closest_residue_analysis.json for one run")
    parser.add_argument("json_file", help="Path to closest_residue_analysis.json")
    parser.add_argument("--output_dir", default=None, help="Default: <json dir>/plots")
    parser.add_argument("--channel_type", default=None, help="Label for plot titles (e.g. G12)")
    parser.add_argument("--topology", default=None,
                        help="Topology file to re-read residue names from (fixes old JSONs)")
    args = parser.parse_args()
    plot_closest_residue_results(args.json_file, args.output_dir, args.channel_type, args.topology)
