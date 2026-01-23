import numpy as np
from tqdm import tqdm
import pandas as pd
from analysis.converter import convert_to_pdb_numbering, get_first_ion_id_part

# =========================
# Utility Functions
# =========================

def compute_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)

def compute_force(q1, q2, pos1, pos2, k=332):
    """
    Compute Coulomb force vector from ion2 to ion1.
    q1, q2: Charges
    pos1, pos2: Coordinates of ions
    k: Coulomb constant (kcal·Å/(mol·e²))
    https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/group__PhysConstants.html
    https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/group__PhysConstants.html#gabf54205837d054a965337502f46e1d65
    https://ambermd.org/FileFormats.php#:~:text=CHARGE%20%3A%20the%20atom%20charges,section%20of%20the%20prmtop%20file
    It is the Coulomb constant 1/4πε0 is expressed in MD-compatible units, i.e.:
    - Distance in Ångströms
    - Energy in kcal/mol
    - Charge in units of elementary charge (e)
    """
    r_vec = pos1 - pos2
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    return k * (q1 * q2) / (r ** 2) * (r_vec / r)

def get_motion_vector(ion_positions, frame):
    """Return movement vector from frame to frame+1."""
    if frame + 1 not in ion_positions or frame not in ion_positions:
        return None
    return ion_positions[frame + 1] - ion_positions[frame]

def unit_vector(v):
    """Return unit vector in direction of v."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else np.zeros_like(v)

# =========================
# Analysis Functions
# =========================

# def analyze_forces(positions, permeating_ion_id, frame, other_ions, charge_map,
#                   closest_residues_by_ion, cutoff=10.0,
#                   calculate_total_force=False, total_force_data=None):
#     """
#     Analyze one frame: compute ionic forces, motion, and optionally total force.
#     Also calculates cosine similarities between different vectors and force decomposition.
#     """
#     result = {
#         "frame": frame,
#         "ionic_force": [0.0, 0.0, 0.0],
#         "ionic_force_magnitude": None,
#         "motion_vector": None,
#         "cosine_ionic_motion": None,
#         "ionic_motion_component": None,
#         "ionic_force_x": None,
#         "ionic_force_y": None,
#         "ionic_force_z": None,
#         "radial_force": None,
#         "axial_force": None,
#         "before_closest_residue": None,
#         "closest_residue": None,
#         "next_closest_residue": None
#     }

#     permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
#     if permeating_pos is None:
#         return result

#     # Add closest residues for permeating ion
#     residue_track = closest_residues_by_ion.get(permeating_ion_id, [])
#     for r in residue_track:
#         if r["frame"] == frame - 1:
#             result["before_closest_residue"] = r["residue"]
#         elif r["frame"] == frame:
#             result["closest_residue"] = r["residue"]
#         elif r["frame"] == frame + 1:
#             result["next_closest_residue"] = r["residue"]

#     ionic_force = np.zeros(3)
#     contributions = []

#     for ion_id, pos in positions.get(frame, {}).items():
#         if ion_id == permeating_ion_id or ion_id not in other_ions:
#             continue
#         distance = compute_distance(permeating_pos, pos)
#         if distance <= cutoff:
#             force = compute_force(charge_map[permeating_ion_id], charge_map[ion_id], permeating_pos, pos)
#             ionic_force += force
#             magnitude = np.linalg.norm(force)

#             c = {
#                 "ion": int(ion_id),
#                 "force": [float(f) for f in force.tolist()],
#                 "magnitude": float(magnitude),
#                 "distance": float(distance),
#                 "before_closest_residue": None,
#                 "closest_residue": None,
#                 "next_closest_residue": None
#             }

#             contrib_track = closest_residues_by_ion.get(int(ion_id), [])
#             for r in contrib_track:
#                 if r["frame"] == frame - 1:
#                     c["before_closest_residue"] = r["residue"]
#                 elif r["frame"] == frame:
#                     c["closest_residue"] = r["residue"]
#                 elif r["frame"] == frame + 1:
#                     c["next_closest_residue"] = r["residue"]

#             contributions.append(c)

#     result["ionic_force"] = ionic_force.tolist()
#     result["ionic_force_magnitude"] = float(np.linalg.norm(ionic_force))

#     Fx, Fy, Fz = ionic_force
#     result.update({
#         "ionic_force_x": float(Fx),
#         "ionic_force_y": float(Fy),
#         "ionic_force_z": float(Fz),
#         "axial_force": float(Fz),  # assuming Z is the pore axis
#         "radial_force": float(np.sqrt(Fx**2 + Fy**2))
#     })

#     ion_positions_over_time = {
#         f: positions.get(f, {}).get(permeating_ion_id) for f in range(frame, frame + 2)
#     }

#     motion_vec = get_motion_vector(ion_positions_over_time, frame)
#     if motion_vec is not None:
#         unit_motion = unit_vector(motion_vec)
#         if np.linalg.norm(ionic_force) != 0 and np.linalg.norm(motion_vec) != 0:
#             cosine_ionic_motion = float(np.dot(ionic_force, motion_vec) / (np.linalg.norm(ionic_force) * np.linalg.norm(motion_vec)))
#             ionic_motion_component = cosine_ionic_motion * result["ionic_force_magnitude"]

#             result.update({
#                 "motion_vector": motion_vec.tolist(),
#                 "cosine_ionic_motion": cosine_ionic_motion,
#                 "ionic_motion_component": ionic_motion_component
#             })

#         for c in contributions:
#             force_vec = np.array(c["force"])
#             force_mag = np.linalg.norm(force_vec)
#             if force_mag != 0:
#                 cosine = float(np.dot(force_vec, unit_motion) / force_mag)
#                 projection = cosine * force_mag
#             else:
#                 cosine = 0.0
#                 projection = 0.0

#             c["cosine_with_motion"] = cosine
#             c["motion_component"] = projection

#     result["contributions"] = contributions

#     if calculate_total_force and total_force_data is not None:
#         tf = total_force_data[frame].get(permeating_ion_id)
#         if tf is not None:
#             total_force = np.array(tf)
#             total_mag = float(np.linalg.norm(total_force))
#             ionic_mag = result["ionic_force_magnitude"]
#             fraction = ionic_mag / total_mag if total_mag != 0 else 0.0

#             result.update({
#                 "total_force": total_force.tolist(),
#                 "total_force_magnitude": total_mag,
#                 "ionic_fraction_of_total": fraction
#             })

#             if np.linalg.norm(ionic_force) != 0 and np.linalg.norm(total_force) != 0:
#                 result["cosine_ionic_total"] = float(np.dot(ionic_force, total_force) / (np.linalg.norm(ionic_force) * np.linalg.norm(total_force)))

#             if motion_vec is not None and np.linalg.norm(total_force) != 0 and np.linalg.norm(motion_vec) != 0:
#                 result["cosine_total_motion"] = float(np.dot(total_force, motion_vec) / (np.linalg.norm(total_force) * np.linalg.norm(motion_vec)))

#     return result

def compute_alignment(force_vec, motion_vec):
    norm_force = np.linalg.norm(force_vec)
    norm_motion = np.linalg.norm(motion_vec)

    if norm_force == 0 or norm_motion == 0:
        return 0.0, 0.0, 0.0

    unit_motion = motion_vec / norm_motion
    component = np.dot(force_vec, unit_motion)
    percent = abs(component) / norm_force * 100
    cosine = component / norm_force
    return cosine, component, percent


def analyze_forces(u, positions, residue_positions, pip2_positions, pip2_resids, unique_pip2_atom_names, actual_pip2_names, permeating_ion_id, frame, other_ions, charge_map,
                  closest_residues_by_ion, glu_residues, asn_residues, total_sf_residues, cutoff=15.0,
                  calculate_total_force=False, total_force_data=None):
    """
    Analyze one frame: compute ionic, residue, and optionally total forces.
    Also calculates motion and cosine similarities.
    """
    result = {
        "frame": frame,
        "motion_vector": 0,
        "motion_vector_magnitude": 0,
        "field_force": [0.0, 0.0, 0.0],
        "field_force_magnitude": 0,
        "ionic_force": [0.0, 0.0, 0.0],
        "ionic_force_magnitude": 0,
        "ionic_force_up": [0.0, 0.0, 0.0],
        "ionic_force_up_magnitude": 0,
        "ionic_force_down": [0.0, 0.0, 0.0],
        "ionic_force_down_magnitude": 0,
        "radial_force": 0,
        "axial_force": 0,
        "glu_force": [0.0, 0.0, 0.0],
        "glu_force_magnitude": 0,
        "asn_force": [0.0, 0.0, 0.0],
        "asn_force_magnitude": 0,
        "sf_force": [0.0, 0.0, 0.0],
        "sf_force_magnitude": 0,
        "residue_force": [0.0, 0.0, 0.0],
        "residue_force_magnitude": 0,
        "pip2_force": [0.0, 0.0, 0.0],
        "pip2_force_magnitude": 0,
        "total_force": [0.0, 0.0, 0.0],
        "total_force_magnitude": 0,
        "cosine_total_motion": 0,
        "cosine_glu_motion": 0,
        "cosine_asn_motion": 0,
        "cosine_sf_motion": 0,
        "cosine_residue_motion": 0,
        "cosine_pip2_motion": 0,
        "cosine_ionic_motion": 0,
        "cosine_ionic_up_motion": 0,
        "cosine_ionic_down_motion": 0,
        "motion_component_total": 0,
        "motion_component_glu": 0,
        "motion_component_asn": 0,
        "motion_component_sf": 0,
        "motion_component_residue": 0,
        "motion_component_ionic": 0,
        "motion_component_ionic_up": 0,
        "motion_component_ionic_down": 0,
        "motion_component_pip2": 0,
        "motion_component_percent_total": 0,
        "motion_component_percent_glu": 0,
        "motion_component_percent_asn": 0,
        "motion_component_percent_sf": 0,
        "motion_component_percent_residue": 0,
        "motion_component_percent_ionic": 0,
        "motion_component_percent_ionic_up": 0,
        "motion_component_percent_ionic_down": 0,
        "motion_component_percent_pip2": 0,
        "ionic_contributions": [],
        "glu_contributions": [],
        "asn_contributions": [],
        "sf_contributions": []
    }

    permeating_ion_id = int(get_first_ion_id_part(permeating_ion_id))
    permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
    if permeating_pos is None:
        return result
    
    ion_positions_over_time = {
        f: positions.get(f, {}).get(permeating_ion_id) for f in range(frame, frame + 2)
    }
    motion_vec = get_motion_vector(ion_positions_over_time, frame)
    result["motion_vector"] = motion_vec.tolist() if motion_vec is not None else None
    result["motion_vector_magnitude"] = float(np.linalg.norm(motion_vec)) if motion_vec is not None else 0.0

    ionic_force = np.zeros(3)
    ionic_force_up = np.zeros(3)
    ionic_force_down = np.zeros(3)
    ionic_contributions = []
    for ion_id, pos in positions.get(frame, {}).items():
        if ion_id == permeating_ion_id or ion_id not in other_ions:
            continue
        distance = compute_distance(permeating_pos, pos)
        if distance <= cutoff:
            z_pos_perm = permeating_pos[2]
            z_pos_ion = pos[2]
            force = compute_force(charge_map[permeating_ion_id], charge_map[ion_id], permeating_pos, pos)
            ionic_force += force
            if z_pos_ion > z_pos_perm:
                ionic_force_up += force
            elif z_pos_ion < z_pos_perm:
                ionic_force_down += force
            cosine_ionic, component_ionic, percent_ionic = compute_alignment(force, motion_vec)
            ionic_contributions.append({
                "ion_id": int(ion_id),
                "distance": float(distance),
                "force": force.tolist(),
                "magnitude": float(np.linalg.norm(force)),
                # Alignment with motion vector
                "cosine_ionic_motion": float(cosine_ionic),
                "motion_component_ionic": float(component_ionic),
                "motion_component_percent_ionic": float(percent_ionic),
                "position" : "up" if z_pos_ion > z_pos_perm else "down" if z_pos_ion < z_pos_perm else "same"
            })

    result["ionic_force_up"] = ionic_force_up.tolist()
    result["ionic_force_up_magnitude"] = float(np.linalg.norm(ionic_force_up))
    result["ionic_force_down"] = ionic_force_down.tolist()
    result["ionic_force_down_magnitude"] = float(np.linalg.norm(ionic_force_down))
    
    external_field_strength = -0.06  # kcal/(mol·Å·e)
    external_field_direction = np.array([0.0, 0.0, 1.0])  # assuming field is along +Z
    q = charge_map[permeating_ion_id]
    field_force =  q * external_field_strength * external_field_direction
    result["field_force"] = field_force.tolist()
    result["field_force_magnitude"] = float(np.linalg.norm(result["field_force"]))

    result["ionic_force"] = ionic_force.tolist()
    result["ionic_force_magnitude"] = float(np.linalg.norm(ionic_force))
    result["ionic_contributions"] = ionic_contributions
    Fx, Fy, Fz = ionic_force
    result.update({
        "axial_force": float(Fz),
        "radial_force": float(np.sqrt(Fx**2 + Fy**2))
    })

    # Add residue forces (GLU + ASN)
    residue_result = analyze_residue_forces(
        u, positions, residue_positions, permeating_ion_id, frame, charge_map, motion_vec,
        glu_residues, asn_residues, total_sf_residues, cutoff=15
    )

    glu_force = np.array(residue_result["glu_force"])
    asn_force = np.array(residue_result["asn_force"])
    sf_force = np.array(residue_result["sf_force"])
    residue_force = np.array(residue_result["residue_force"])

    pip2_result = analyze_pip2_forces(
            u, positions, pip2_positions, permeating_ion_id, frame,
            charge_map, motion_vec, pip2_resids=pip2_resids, unique_pip2_atom_names=unique_pip2_atom_names, actual_pip2_name=actual_pip2_names,
            cutoff=50.0, headgroup_only=False
        )
    pip2_force = np.array(pip2_result["pip2_force"])

    total_force = ionic_force + residue_force + pip2_force +field_force

    result["glu_force"] = glu_force.tolist()
    result["glu_force_magnitude"] = float(np.linalg.norm(glu_force))
    result["asn_force"] = asn_force.tolist()
    result["asn_force_magnitude"] = float(np.linalg.norm(asn_force))
    result["sf_force"] = sf_force.tolist()
    result["sf_force_magnitude"] = float(np.linalg.norm(sf_force))
    result["residue_force"] = residue_force.tolist()
    result["residue_force_magnitude"] = float(np.linalg.norm(residue_force))
    result["pip2_force"] = pip2_force.tolist()
    result["pip2_force_magnitude"] = float(np.linalg.norm(pip2_force))
    result["total_force"] = total_force.tolist()
    result["total_force_magnitude"] = float(np.linalg.norm(total_force))
    result["glu_contributions"] = residue_result["glu_contributions"]
    result["asn_contributions"] = residue_result["asn_contributions"]
    result["sf_contributions"] = residue_result["sf_contributions"]
    result["pip2_contributions"] = pip2_result["pip2_contributions"]

    if motion_vec is not None and np.linalg.norm(motion_vec) > 0:
        unit_motion = unit_vector(motion_vec)
        for key, vec in zip(
            ["ionic", "ionic_up", "ionic_down", "glu", "asn", "sf", "residue", "pip2", "total"],
            [ionic_force, ionic_force_up, ionic_force_down, glu_force, asn_force, sf_force, residue_force, pip2_force, total_force]
        ):
            norm = np.linalg.norm(vec)
            if norm > 0:
                cosine = float(np.dot(vec, unit_motion) / norm)
                component = float(np.dot(vec, unit_motion))
                percent_aligned = (abs(component) / norm) * 100
                result[f"cosine_{key}_motion"] = cosine
                result[f"motion_component_{key}"] = component
                result[f"motion_component_percent_{key}"] = percent_aligned

    return result

def analyze_residue_forces(
    u,
    positions,
    residue_positions,
    permeating_ion_id,
    frame,
    charge_map,
    motion_vec,
    glu_residues,
    asn_residues,
    total_sf_residues,
    cutoff=6.0
):
    """
    Calculate electrostatic forces from GLU and ASN side chains on the ion using pre-extracted atom positions.

    Returns:
    - dict with GLU, ASN, and total residue force vectors, magnitudes, and per-atom contributions
    """
    import numpy as np

    ion_pos = positions[frame][permeating_ion_id]

    total_force = np.zeros(3)
    glu_force = np.zeros(3)
    asn_force = np.zeros(3)
    sf_force = np.zeros(3)
    glu_contributions = []
    asn_contributions = []
    sf_contributions = []

    for resid in glu_residues + asn_residues + total_sf_residues:
        residue = u.select_atoms(f"resid {resid}")
        if len(residue) == 0:
            print(f"Warning: Resid {resid} not found in topology.")
            continue

        resname = residue.residues[0].resname
        # if resname == "GLU":
        #     atom_names = ["CD", "OE1", "OE2"]
        # elif resname == "ASN":
        #     atom_names = ["CG", "OD1", "ND2", "HD21", "HD22"]
        # else:
        #     print(f"Warning: Resid {resid} is not GLU or ASN (found {resname}).")
        #     continue

        atom_names = []
        for atom in residue:
            atom_names.append(atom.name)

        for atom_name in atom_names:
            atom_pos = residue_positions.get(frame, {}).get((resid, atom_name))
            if atom_pos is None:
                continue  # Atom not found in this frame

            charge = charge_map[(resname, atom_name)]

            if charge is None:
                print(f"Warning: Charge not found for atom {atom_name} in resid {resid}.")
                continue

            r_vec = ion_pos - atom_pos
            r = np.linalg.norm(r_vec)
            if r > cutoff:
                continue

            force = compute_force(1.0, charge, ion_pos, atom_pos)
            total_force += force
            cosine_ionic, component_ionic, percent_ionic = compute_alignment(force, motion_vec)
            contribution = {
                "resid": int(resid),
                "resname": resname,
                "atom": atom_name,
                "charge": float(charge),
                "distance": float(r),
                "force": force.tolist(),
                "magnitude": float(np.linalg.norm(force)),
                "cosine_with_motion": float(cosine_ionic),
                "motion_component": float(component_ionic),
                "motion_component_percent": float(percent_ionic),
            }

            if resid  in glu_residues:
                glu_force += force
                glu_contributions.append(contribution)
            elif resid in asn_residues:
                asn_force += force
                asn_contributions.append(contribution)
            elif resid in total_sf_residues:
                sf_force += force
                sf_contributions.append(contribution)

    return {
        "residue_force": total_force.tolist(),
        "glu_force": glu_force.tolist(),
        "asn_force": asn_force.tolist(),
        "sf_force": sf_force.tolist(),
        "glu_contributions": glu_contributions,
        "asn_contributions": asn_contributions,
        "sf_contributions": sf_contributions
    }

def analyze_pip2_forces(
    u,
    positions,
    pip2_positions,
    permeating_ion_id,
    frame,
    charge_map,
    motion_vec,
    pip2_resids,
    unique_pip2_atom_names,
    actual_pip2_name,
    cutoff=50.0,
    headgroup_only=True
):
    """
    Calculate electrostatic forces from PIP2 atoms on the ion.
    
    Parameters:
    - pip2_resids: list of PIP2 residue IDs
    - headgroup_only: if True, only uses phosphate headgroup atoms

    Returns:
    - Dictionary with total force, magnitude, and per-atom contributions
    """
    important_atoms = {"P4", "P5", "O4P", "O5P", "O41", "O42", "O43", "O51", "O52", "O53"}
    ion_pos = positions[frame][permeating_ion_id]
    
    pip2_force = np.zeros(3)
    contributions = []

    for resid in pip2_resids:
        for atom_name in unique_pip2_atom_names:
            if headgroup_only and atom_name not in important_atoms:
                continue

            atom_pos = pip2_positions.get(frame, {}).get((resid, atom_name))
            if atom_pos is None:
                continue

            charge = charge_map[(actual_pip2_name,atom_name)]
            if charge is None:
                continue

            r_vec = ion_pos - atom_pos
            r = np.linalg.norm(r_vec)
            if r > cutoff:
                continue

            force = compute_force(1.0, charge, ion_pos, atom_pos)
            pip2_force += force

            cosine, component, percent = compute_alignment(force, motion_vec)

            contributions.append({
                "resid": int(resid),
                "atom": atom_name,
                "charge": float(charge),
                "distance": float(r),
                "force": force.tolist(),
                "magnitude": float(np.linalg.norm(force)),
                "cosine_with_motion": float(cosine),
                "motion_component": float(component),
                "motion_component_percent": float(percent)
            })

    return {
        "pip2_force": pip2_force.tolist(),
        "pip2_contributions": contributions
    }


def find_top_cosine_frames(event_data, top_n=5):
    """
    Find the top N frames with the highest cosine_ionic_motion for each ion.
    Also report if the frame is the permeation frame.

    Args:
        event_data: list of event dictionaries (each has 'permeated_ion', 'frame', 'analysis')
        top_n: number of top frames to return per ion

    Returns:
        results: dict {ion_id: list of dicts with frame, cosine_ionic_motion, is_permeation_frame}
    """
    ion_results = {}

    for event in event_data:
        permeated_ion = event["permeated_ion"]
        permeation_frame = event["frame"]
        analysis = event["analysis"]

        # Collect all frames and their cosine_ionic_motion
        frame_cosine_list = []
        for frame, frame_data in analysis.items():
            cosine = frame_data.get("cosine_ionic_motion")
            if cosine is not None:
                frame_cosine_list.append((frame, cosine))

        # Sort by descending cosine value
        sorted_frames = sorted(frame_cosine_list, key=lambda x: -x[1])

        # Pick top N
        top_frames = sorted_frames[:top_n]

        # Build the result for this ion
        top_info = []
        for frame, cosine in top_frames:
            top_info.append({
                "frame": frame,
                "cosine_ionic_motion": cosine,
                "is_permeation_frame": (frame == permeation_frame)
            })

        ion_results[permeated_ion] = top_info

    return ion_results


def collect_sorted_cosines_until_permeation(event_data):
    ion_results = {}

    for event in event_data:
        ion_id = str(event["permeated_ion"])  # ensure match with input keys
        permeation_frame = event["frame"]
        analysis = event["analysis"]

        frame_cosine_list = []
        for frame, frame_data in analysis.items():
            cosine = frame_data.get("cosine_ionic_motion")
            if cosine is not None:
                frame_cosine_list.append((frame, cosine))

        sorted_frames = sorted(frame_cosine_list, key=lambda x: -x[1])
        collected_frames = []

        for frame, cosine in sorted_frames:
            frame_data = analysis[frame].copy()
            frame_data["is_permeation_frame"] = (frame == permeation_frame)
            collected_frames.append(frame_data)

            if frame == permeation_frame:
                break

        ion_results[ion_id] = collected_frames

    return ion_results


def extract_permeation_frames(event_data, offset_from_end=1):
    """
    Extracts information from a selected frame before the permeation frame (default: last one).
    Returns both a contributor-expanded and a summary DataFrame.

    Parameters:
        event_data (list): List of permeation event dictionaries.
        offset_from_end (int): 1 = last frame, 2 = second-to-last, etc.

    Returns:
        tuple: (df_expanded, df_summary)
    """
    expanded_rows = []
    summary_rows = []

    for event in event_data:
        ion_id = str(event["permeated_ion"])
        analysis = event.get("analysis", {})
        # Convert frame keys to sorted list of integers
        frame_keys = sorted(int(k) for k in analysis.keys())
        if len(frame_keys) < offset_from_end:
            continue  # skip if not enough frames

        selected_frame_int = frame_keys[-offset_from_end]
        selected_frame = int(selected_frame_int)  # for lookup in original dict

        entry = analysis.get(selected_frame)
        if not entry:
            continue

        # --- Summary row ---
        summary_rows.append({
            "ion_id": ion_id,
            "frame": selected_frame_int,
            "ionic_force": entry.get("ionic_force"),
            "ionic_force_magnitude": entry.get("ionic_force_magnitude"),
            "motion_component_ionic": entry.get("motion_component_ionic"),
            "cosine_ionic_motion": entry.get("cosine_ionic_motion"),
            "radial_force": entry.get("radial_force"),
            "axial_force": entry.get("axial_force"),
            "motion_vector": entry.get("motion_vector"),
        })

        # --- Expanded contributor rows ---
        contributions = entry.get("contributions", [])
        contributions.sort(key=lambda c: c.get("cosine_with_motion", -1), reverse=True)

        for c in contributions:
            expanded_rows.append({
                "ion_id": ion_id,
                "frame": selected_frame_int,
                "ionic_force": entry.get("ionic_force"),
                "ionic_force_magnitude": entry.get("ionic_force_magnitude"),
                "motion_component_ionic": entry.get("motion_component_ionic"),
                "cosine_ionic_motion": entry.get("cosine_ionic_motion"),
                "radial_force": entry.get("radial_force"),
                "axial_force": entry.get("axial_force"),
                "motion_vector": entry.get("motion_vector"),
                # Contributor-specific
                "contributing_ion": c.get("ion"),
                "contrib_force": c.get("force"),
                "contrib_magnitude": c.get("magnitude"),
                "contrib_distance": c.get("distance"),
                "contrib_cosine_with_motion": c.get("cosine_with_motion"),
                "contrib_motion_component": c.get("motion_component"),
                "contrib_before_closest_residue": c.get("before_closest_residue"),
                "contrib_closest_residue": c.get("closest_residue"),
                "contrib_next_closest_residue": c.get("next_closest_residue")
            })

    df_expanded = pd.DataFrame(expanded_rows)
    df_summary = pd.DataFrame(summary_rows)

    return df_expanded, df_summary

   
def extract_last_frame_analysis(events):
    results = []
    for event in events:
        last_frame = event["frame"]
        last_data = event["analysis"].get(str(last_frame)) or event["analysis"].get(last_frame)
        if last_data:
            results.append({
                "start_frame": event["start_frame"],
                "frame": last_frame,
                "permeated_ion": event["permeated_ion"],
                "analysis": last_data
            })
    return results

import json
import pandas as pd

def extract_permeation_forces(data, output_dir, output_file="permeation_summary_forces.csv"):
  
    rows = []
    for event in data:
        analysis = event["analysis"]
        row = {
            "start_frame": event["start_frame"],
            "frame": event["frame"],
            "permeated_ion": event["permeated_ion"],
            "ionic_force_magnitude": analysis.get("ionic_force_magnitude"),
            "glu_force_magnitude": analysis.get("glu_force_magnitude"),
            "asn_force_magnitude": analysis.get("asn_force_magnitude"),
            "residue_force_magnitude": analysis.get("residue_force_magnitude"),
            "total_force_magnitude": analysis.get("total_force_magnitude"),
            "cosine_total_motion": analysis.get("cosine_total_motion"),
            "cosine_glu_motion": analysis.get("cosine_glu_motion"),
            "cosine_asn_motion": analysis.get("cosine_asn_motion"),
            "cosine_residue_motion": analysis.get("cosine_residue_motion"),
            "cosine_ionic_motion": analysis.get("cosine_ionic_motion"),
            "motion_component_total": analysis.get("motion_component_total"),
            "motion_component_glu": analysis.get("motion_component_glu"),
            "motion_component_asn": analysis.get("motion_component_asn"),
            "motion_component_residue": analysis.get("motion_component_residue"),
            "motion_component_ionic": analysis.get("motion_component_ionic"),
            "motion_component_percent_total": analysis.get("motion_component_percent_total"),
            "motion_component_percent_glu": analysis.get("motion_component_percent_glu"),
            "motion_component_percent_asn": analysis.get("motion_component_percent_asn"),
            "motion_component_percent_residue": analysis.get("motion_component_percent_residue"),
            "motion_component_percent_ionic": analysis.get("motion_component_percent_ionic"),
        }
        rows.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / output_file, index=False)

    df.to_csv(output_dir / output_file, index=False)
    print(f"✅ CSV file saved as {output_file}")
