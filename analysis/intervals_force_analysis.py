import numpy as np
from tqdm import tqdm
import pandas as pd
from analysis.force_analysis import compute_force, compute_alignment

def analyze_force_intervals(
    u,
    positions,
    residue_positions,
    permeating_ion_id,
    frame,
    charge_map,
    glu_residues,
    asn_residues,
    total_sf_residues,
    cutoff=15.0,
    n_steps=20,
    k=332.0
):
    """
    Analyze interpolated force components over N intervals within a frame to frame+1 range.
    Returns a list of dictionaries with the same structure as a single-frame analysis.
    """
    positions_n = positions.get(frame, {})
    positions_n1 = positions.get(frame + 1, {})
    if not positions_n or not positions_n1:
        return []

    motion_vec = None
    ion_pos_start = positions_n.get(permeating_ion_id)
    ion_pos_end = positions_n1.get(permeating_ion_id)
    if ion_pos_start is not None and ion_pos_end is not None:
        motion_vec = ion_pos_end - ion_pos_start

    results = []
    q1 = charge_map.get(permeating_ion_id)
    if q1 is None:
        return []

    for step in range(n_steps + 2):  # includes start and end
        alpha = step / (n_steps + 1)
        frame_result = {
            "frame": frame + alpha,
            "step": step,
            "ionic_force": [0.0, 0.0, 0.0],
            "ionic_force_magnitude": 0.0,
            "motion_vector": motion_vec.tolist() if motion_vec is not None else None,
            "radial_force": 0.0,
            "axial_force": 0.0,
            "glu_force": [0.0, 0.0, 0.0],
            "glu_force_magnitude": 0.0,
            "asn_force": [0.0, 0.0, 0.0],
            "asn_force_magnitude": 0.0,
            "sf_force": [0.0, 0.0, 0.0],
            "sf_force_magnitude": 0.0,
            "residue_force": [0.0, 0.0, 0.0],
            "residue_force_magnitude": 0.0,
            "total_force": [0.0, 0.0, 0.0],
            "total_force_magnitude": 0.0,
            "cosine_total_motion": 0.0,
            "cosine_glu_motion": 0.0,
            "cosine_asn_motion": 0.0,
            "cosine_sf_motion": 0.0,
            "cosine_residue_motion": 0.0,
            "cosine_ionic_motion": 0.0,
            "motion_component_total": 0.0,
            "motion_component_glu": 0.0,
            "motion_component_asn": 0.0,
            "motion_component_sf": 0.0,
            "motion_component_residue": 0.0,
            "motion_component_ionic": 0.0,
            "motion_component_percent_total": 0.0,
            "motion_component_percent_glu": 0.0,
            "motion_component_percent_asn": 0.0,
            "motion_component_percent_sf": 0.0,
            "motion_component_percent_residue": 0.0,
            "motion_component_percent_ionic": 0.0,
            "ionic_contributions": [],
            "glu_contributions": [],
            "asn_contributions": [],
            "sf_contributions": [],
        }

        ion_pos = (1 - alpha) * ion_pos_start + alpha * ion_pos_end
        ionic_force = np.zeros(3)
        ionic_contributions = []

        for ion_id in positions_n:
            if ion_id == permeating_ion_id or ion_id not in positions_n1:
                continue
            pos_other = (1 - alpha) * positions_n[ion_id] + alpha * positions_n1[ion_id]
            q2 = charge_map.get(ion_id)
            if q2 is None:
                continue
            dist = np.linalg.norm(ion_pos - pos_other)
            if dist > cutoff:
                continue
            force = compute_force(q1, q2, ion_pos, pos_other, k)
            ionic_force += force
            cosine_ionic, component_ionic, percent_ionic = compute_alignment(force, motion_vec)
            ionic_contributions.append({
                "ion_id": int(ion_id),
                "distance": float(dist),
                "force": force.tolist(),
                "magnitude": float(np.linalg.norm(force)),
                "cosine_motion": float(cosine_ionic),
                "motion_component": float(component_ionic),
                "motion_component_percent": float(percent_ionic),
            })

        frame_result["ionic_force"] = ionic_force.tolist()
        frame_result["ionic_force_magnitude"] = float(np.linalg.norm(ionic_force))
        Fx, Fy, Fz = ionic_force
        frame_result["axial_force"] = float(Fz)
        frame_result["radial_force"] = float(np.sqrt(Fx**2 + Fy**2))
        frame_result["ionic_contributions"] = ionic_contributions

        # Static residue forces
        residue_result = analyze_residue_forces(
            u, positions_n, positions_n1, residue_positions, frame, alpha, permeating_ion_id, charge_map, motion_vec,
            glu_residues, asn_residues, total_sf_residues, cutoff=6.0
        )
        glu_force = np.array(residue_result["glu_force"])
        asn_force = np.array(residue_result["asn_force"])
        sf_force = np.array(residue_result["sf_force"])
        residue_force = np.array(residue_result["residue_force"])
        total_force = ionic_force + residue_force

        frame_result["glu_force"] = glu_force.tolist()
        frame_result["glu_force_magnitude"] = float(np.linalg.norm(glu_force))
        frame_result["asn_force"] = asn_force.tolist()
        frame_result["asn_force_magnitude"] = float(np.linalg.norm(asn_force))
        frame_result["residue_force"] = residue_force.tolist()
        frame_result["residue_force_magnitude"] = float(np.linalg.norm(residue_force))
        frame_result["total_force"] = total_force.tolist()
        frame_result["total_force_magnitude"] = float(np.linalg.norm(total_force))
        frame_result["glu_contributions"] = residue_result["glu_contributions"]
        frame_result["asn_contributions"] = residue_result["asn_contributions"]

        if motion_vec is not None and np.linalg.norm(motion_vec) > 0:
            unit_motion = motion_vec / np.linalg.norm(motion_vec)
            for key, vec in zip(
                ["ionic", "glu", "asn", "residue", "total"],
                [ionic_force, glu_force, asn_force, residue_force, total_force]
            ):
                norm = np.linalg.norm(vec)
                if norm > 0:
                    cosine = float(np.dot(vec, unit_motion) / norm)
                    component = float(np.dot(vec, unit_motion))
                    percent_aligned = (abs(component) / norm) * 100
                    frame_result[f"cosine_{key}_motion"] = cosine
                    frame_result[f"motion_component_{key}"] = component
                    frame_result[f"motion_component_percent_{key}"] = percent_aligned

        results.append(frame_result)

    return results


def analyze_residue_forces(
    u,
    positions_n,
    positions_n1,
    residue_positions,
    frame,
    alpha,
    permeating_ion_id,
    charge_map,
    motion_vec,
    glu_residues,
    asn_residues,
    total_sf_residues,
    cutoff=6.0
):
    """
    Calculate electrostatic forces from GLU and ASN side chains on the ion using interpolated ion position.

    Returns:
    - dict with GLU, ASN, and total residue force vectors, magnitudes, and per-atom contributions
    """
    import numpy as np

    ion_pos = (1 - alpha) * positions_n[permeating_ion_id] + alpha * positions_n1[permeating_ion_id]

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
            pos_start = residue_positions.get(frame, {}).get((resid, atom_name))
            pos_end = residue_positions.get(frame + 1, {}).get((resid, atom_name))
            if pos_start is None or pos_end is None:
                continue

            atom_pos = (1 - alpha) * pos_start + alpha * pos_end

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
                "cosine_motion": float(cosine_ionic),
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
        "residue_force_magnitude": float(np.linalg.norm(total_force)),
        "glu_force": glu_force.tolist(),
        "glu_force_magnitude": float(np.linalg.norm(glu_force)),
        "asn_force": asn_force.tolist(),
        "asn_force_magnitude": float(np.linalg.norm(asn_force)),
        "sf_force": sf_force.tolist(),
        "sf_force_magnitude": float(np.linalg.norm(sf_force)),
        "glu_contributions": glu_contributions,
        "asn_contributions": asn_contributions,
        "sf_contributions": sf_contributions
    }


