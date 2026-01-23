import os
import json
import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
import re

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def strip_json_extension(filename):
    name, ext = os.path.splitext(filename)
    return name


def compute_sf_upper_center(u, sf_residues):
    atom_indices = []
    for resid in sf_residues:
        residue_atoms = u.select_atoms(f"resid {resid}")
        coords = residue_atoms.positions
        sorted_indices = coords[:, 2].argsort()
        upper_index = sorted_indices[0]
        residue_atoms = residue_atoms[[upper_index]]
        atom_indices.append(residue_atoms[0].index)
    upper_atoms = u.atoms[atom_indices]


    return upper_atoms.center_of_mass()

def ion_id_total_number_identifier(permeation_events2, ion_id, frame):
    
    for event in permeation_events2:
        if str(ion_id) in event["ion_id"]:
            if frame>= event["start_frame"] and frame <= event["exit_frame"]:
                return event["ion_id"]
    return 0
            

def find_clean_stuck_frames(permeation_events2, json_folder, output_path, u, sf_residues, z_margin=0.0, proximity_cutoff=2.0):
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    result = {}
    stuck_ions_during_simulation = {}

    for json_file in tqdm(json_files, desc="Checking JSONs for clean stuck frames"):
        ion_id = extract_number(json_file)
        ion_id_total_number = strip_json_extension(json_file)
        data = load_json(os.path.join(json_folder, json_file))

        try:
            start = sorted(map(int, data["analysis"].keys()))[0]
            end = data["start_frame"] - 1
        except:
            continue

        if end <= start:
            continue

        clean = []

        for ts in u.trajectory[start:end + 1]:
            # Compute SF center dynamically per frame
            sf_center = compute_sf_upper_center(u, sf_residues)
            sf_z_cutoff = sf_center[2] - z_margin

            # Get the ion's center of mass
            ion_sel = u.select_atoms(f"resname K+ K and resid {ion_id}")
            if len(ion_sel) == 0:
                continue
            ion_pos = ion_sel.center_of_mass()

            # Get all other K+ ions (not this ion)
            all_k_ions = u.select_atoms("resname K+ K")
            other_ions = all_k_ions[[atom.resid != ion_id for atom in all_k_ions]]
            # Check if any of them are above the SF
            # intruding = any(ion.position[2] > sf_z_cutoff for ion in other_ions)
            intruding = False
            for ion in other_ions:
                # if int(ion_id) == 2223 and int(ion.resid) == 2520:
                #     print(ts.frame, ion.position[2], sf_z_cutoff)
                if ion.position[2] > sf_z_cutoff and np.linalg.norm(ion.position - sf_center) < proximity_cutoff:
                    intruding = True
                    break

            # print(f"Checking ion {ion_id} at frame {ts.frame}, intruding: {intruding}")
            if not intruding:
                clean.append(ts.frame)
                stuck_ions_during_simulation[ts.frame] = ion_id_total_number
            else:
                print(ion.resid)
                intruding_ion = ion_id_total_number_identifier(permeation_events2, ion.resid, ts.frame)
                if intruding_ion!= 0:
                    stuck_ions_during_simulation[ts.frame] = intruding_ion
                print(f"Ion {ion_id} at frame {ts.frame} is intruding the SF, skipping.")

        result[ion_id_total_number] = clean

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "stuck_frames.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(output_path, "stuck_ions_during_simulation.json"), "w") as f:
        json.dump(stuck_ions_during_simulation, f, indent=2)
    print(f"✅ Clean stuck frame results saved to {output_path}/stuck_frames.json")

    return stuck_ions_during_simulation