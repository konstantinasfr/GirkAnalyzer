import json
import os
import numpy as np

backbone_atoms = {"N", "CA", "C", "O", "HA", "H"}

def analyze_frame_for_ion(json_folder, ion_id, frame, backbone_atoms_only=False):
    json_path = os.path.join(json_folder, f"{ion_id}.json")

    with open(json_path) as f:
        data = json.load(f)

    if str(frame) not in data["analysis"]:
        raise ValueError(f"Frame {frame} not found in JSON for ion {ion_id}.")

    frame_data = data["analysis"][str(frame)]

    results = {
        "glu_residues": {},
        "asn_residues": {},
        "pip2_residues": {},
        "ionic_up_forces": [],
        "ionic_down_forces": [],
        "ionic_up_total_force": None,
        "ionic_down_total_force": None
    }

    def accumulate_force(contributions, residue_dict, exclude_backbone=True):
        for contrib in contributions:
            atom = contrib["atom"]
            resid = contrib["resid"]
            if backbone_atoms_only and atom not in backbone_atoms:
                continue
            force_vec = np.array(contrib["force"])
            if resid not in residue_dict:
                residue_dict[resid] = force_vec
            else:
                residue_dict[resid] += force_vec

    # === GLU/ASN ===
    glu_forces = {}
    asn_forces = {}
    accumulate_force(frame_data.get("glu_contributions", []), glu_forces)
    accumulate_force(frame_data.get("asn_contributions", []), asn_forces)

    for resid, vec in glu_forces.items():
        results["glu_residues"][resid] = {
            "force": vec.tolist(),
            "magnitude": float(np.linalg.norm(vec))
        }
    for resid, vec in asn_forces.items():
        results["asn_residues"][resid] = {
            "force": vec.tolist(),
            "magnitude": float(np.linalg.norm(vec))
        }

    # === PIP2 ===
    pip2_forces = {}
    accumulate_force(frame_data.get("pip2_contributions", []), pip2_forces, exclude_backbone=False)

    for resid, vec in pip2_forces.items():
        results["pip2_residues"][resid] = {
            "force": vec.tolist(),
            "magnitude": float(np.linalg.norm(vec))
        }

    # === Ionic up/down ===
    up_vecs = []
    down_vecs = []

    for ion in frame_data.get("ionic_contributions", []):
        force_vec = np.array(ion["force"])
        force_entry = {
            "ion_id": ion["ion_id"],
            "distance": ion["distance"],
            "force": ion["force"],
            "magnitude": ion["magnitude"]
        }
        if ion["position"] == "up":
            results["ionic_up_forces"].append(force_entry)
            up_vecs.append(force_vec)
        elif ion["position"] == "down":
            results["ionic_down_forces"].append(force_entry)
            down_vecs.append(force_vec)

    if up_vecs:
        total_up = np.sum(up_vecs, axis=0)
        results["ionic_up_total_force"] = {
            "force": total_up.tolist(),
            "magnitude": float(np.linalg.norm(total_up))
        }

    if down_vecs:
        total_down = np.sum(down_vecs, axis=0)
        results["ionic_down_total_force"] = {
            "force": total_down.tolist(),
            "magnitude": float(np.linalg.norm(total_down))
        }

    return results
