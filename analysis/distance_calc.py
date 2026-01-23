import pandas as pd
import numpy as np
from analysis.converter import convert_to_pdb_numbering, get_first_ion_id_part

# total_distances_dict = {}

def get_overlapping_ions(ion_id, target_start, target_end, event_list):
    overlapping_ions = []
    for event in event_list:
        if event['ion_id'] == ion_id:
            continue
        start = event['start_frame']
        end = event['exit_frame']
        if not (target_end < start or end < target_start):
            overlapping_ions.append(event['ion_id'])
    return overlapping_ions

def calculate_distances(ion_permeated, analyzer, use_ca_only=True, use_min_distances=False, use_charges=False,
                        glu_residues=None, asn_residues=None, sf_residues=None):
    """
    Calculates the distances between a given ion and selected residues across relevant frames.
    Depending on the flags, uses CA atoms, all atoms (min), or functional atom-based charge centers.

    Args:
        ion_permeated: dictionary with 'ion_id'
        analyzer: object with trajectory and permeation event lists
        use_ca_only: if True, uses only CA atoms
        use_min_distances: if True, uses all atoms and takes min distance
        use_charges: if True, uses midpoint of charged/polar atoms (OE1/OE2 for Glu, OD1/ND2 for Asn)

    Returns:
        Dictionary of distances per frame for that ion
    """

    u = analyzer.u
    ch1, ch2, ch3 = analyzer.permeation_events1, analyzer.permeation_events2, analyzer.permeation_events3

    ion_id_str = str(ion_permeated['ion_id'])
    ion_id_int = int(get_first_ion_id_part(ion_id_str))

    # Get start/exit frame for this ion from Channel 2 event list
    start_frame, exit_frame = next(
        (event['start_frame'], event['exit_frame']) for event in ch2 if event['ion_id'] == ion_id_str
    )

    temp_distances_dict = {ion_id_str: []}
    ions_to_test = get_overlapping_ions(ion_id_str, start_frame, exit_frame, ch1) + \
                   get_overlapping_ions(ion_id_str, start_frame, exit_frame, ch2) + \
                   get_overlapping_ions(ion_id_str, start_frame, exit_frame, ch3)

    # Residue definitions
    all_residues = glu_residues + asn_residues + sf_residues

    if sum([use_ca_only, use_min_distances, use_charges]) != 1:
        raise ValueError("You must set exactly one of use_ca_only, use_min_distances, or use_charges to True.")

    for ts in u.trajectory[start_frame:exit_frame + 1]:
        frame_data = {'frame': ts.frame, 'residues': {}, 'ions': {}}

        ion = u.select_atoms(f"resname K+ K and resid {ion_id_int}")
        if len(ion) != 1:
            print(f"Warning: Ion resid {ion_id_int} not found uniquely.")
            continue

        ion_pos = ion.positions[0]

        # if ts.frame == 6727 and ion_id == 1341:
        #         atoms = u.select_atoms(f"resid 130 and name CG")
        #         print(atoms.positions)
        #         dist = float(np.linalg.norm(ion_pos - atoms.positions[0]))
        #         print(dist)
        
        for resid in glu_residues + asn_residues:
            if use_charges:
                if resid in glu_residues:
                    # Select CD, OE1, OE2 for GLU – key electrostatic atoms
                    # atoms = u.select_atoms(f"resid {resid} and name CD OE1 OE2")
                    atoms = u.select_atoms(f"resid {resid}")
                    if atoms.n_atoms >= 1:
                        # Calculate all distances and take the minimum
                        dists = np.linalg.norm(atoms.positions - ion_pos, axis=1)
                        dist = float(np.min(dists))
                    else:
                        print(f"Glu {resid} missing CD, OE1, or OE2 at frame {ts.frame}")
                        dist = float('nan')

                elif resid in asn_residues:
                    # Select CG, OD1, ND2, HD21, HD22 for ASN – full electrostatic group
                    # atoms = u.select_atoms(f"resid {resid} and name CG OD1 ND2 HD21 HD22")
                    atoms = u.select_atoms(f"resid {resid}")
                    if atoms.n_atoms >= 1:
                        dists = np.linalg.norm(atoms.positions - ion_pos, axis=1)
                        dist = float(np.min(dists))
                    else:
                        print(f"Asn {resid} missing sidechain atoms at frame {ts.frame}")
                        dist = float('nan')



            elif use_ca_only:
                atoms = u.select_atoms(f"resid {resid} and name CA")
                if atoms.n_atoms == 1:
                    dist = float(np.linalg.norm(ion_pos - atoms.positions[0]))
                else:
                    dist = float('nan')
            elif use_min_distances:
                atoms = u.select_atoms(f"resid {resid}")
                if atoms.n_atoms > 0:
                    dists = np.linalg.norm(atoms.positions - ion_pos, axis=1)
                    dist = float(np.min(dists))
                else:
                    dist = float('nan')

            frame_data['residues'][resid] = dist

        # Selectivity filter residues (always use atom-based min distance)
        sf_distances = []
        for sf_resid in sf_residues:
            atoms = u.select_atoms(f"resid {sf_resid}")
            if atoms.n_atoms > 0:
                dists = np.linalg.norm(atoms.positions - ion_pos, axis=1)
                dist = float(np.min(dists))
            else:
                dist = float('nan')
            sf_distances.append(dist)

#################### Change to consider SF only when ion goes back into the filter ####################
        # frame_data['residues']['SF'] = float(np.mean(sf_distances))
        atom_indices = []
        for resid in sf_residues:
            residue_atoms = u.select_atoms(f"resid {resid}")
            coords = residue_atoms.positions
            sorted_indices = coords[:, 2].argsort()  # sort Z-values
            upper_index = sorted_indices[0]  # second from the end
            atom_indices.append(residue_atoms[upper_index].index)
            

        upper_atoms = u.atoms[atom_indices]
        upper_center = upper_atoms.center_of_mass()

        if ion_pos[2] > upper_center[2]:
            frame_data['residues']['SF'] = float(np.mean(sf_distances))
        else:
            frame_data['residues']['SF'] = 100000 # Arbitrary large value to indicate not in SF
        ##############################################################################################


        # Distances to overlapping ions
        for ion_to_test in ions_to_test:
            other_ion = u.select_atoms(f"resname K+ K and resid {int(get_first_ion_id_part(ion_to_test))}")
            if len(other_ion) == 1:
                other_pos = other_ion.positions[0]
                dist = float(np.linalg.norm(ion_pos - other_pos))
                frame_data['ions'][ion_to_test] = dist

        temp_distances_dict[ion_id_str].append(frame_data)

    return temp_distances_dict

