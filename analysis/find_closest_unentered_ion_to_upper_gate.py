import numpy as np
import os
import json
from tqdm import tqdm

import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_closest_unentered_ion_to_upper_gate(u, upper_gate_residues, output_path=None, use_CA=False, use_single_min=True):
    """
    Finds the K+ ion closest to the upper gate center at each frame,
    but only among those that haven't crossed the gate (i.e., z >= gate center z).
    Also counts how many frames have any ion <3Å from the SF center from below.

    Parameters:
    - u: MDAnalysis Universe
    - upper_gate_residues: list of resid integers
    - output_path: folder to save JSON output and plot, if desired
    - use_CA: whether to use only CA atoms for gate center

    Returns:
    - result: dict of frame -> {"resid": ion_resid, "distance": dist}
    """
    result = {}
    frame_counts_by_cutoff = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}


    for ts in tqdm(u.trajectory, desc="Scanning for closest unentered ions"):
        # Compute upper gate center
        atom_indices = []
        min_z = float('inf')
        for resid in upper_gate_residues:
            if use_CA:
                atoms = u.select_atoms(f"resid {resid} and name CA")
            else:
                atoms = u.select_atoms(f"resid {resid}")
                coords = atoms.positions
                upper_index = coords[:, 2].argsort()[0]  # pick atom with lowest z (lowest index = highest atom)
                atoms = atoms[[upper_index]]
            atom_indices.append(atoms[0].index)
            if use_single_min:
                min_z = min(min_z, atoms[0].position[2])

        # Use (x, y) from center of mass
        com_xy = u.atoms[atom_indices].center_of_mass()[:2]

        # Combine with Z depending on mode
        if use_single_min:
            gate_z = min_z
        else:
            gate_z = u.atoms[atom_indices].center_of_mass()[2]

        upper_center = np.array([com_xy[0], com_xy[1], gate_z])

        # Get all K+ ions
        ions = u.select_atoms("resname K+ K")

        # Check for ions above the gate
        unentered_ions = [ion for ion in ions if ion.position[2] > gate_z]
        entered_ions = [ion for ion in ions if ion.position[2] <= gate_z]
        # if ts.frame<30:
        #     print(f"Frame {ts.frame}: {upper_center}")
        if unentered_ions:
            # Find the closest one
            distances = [np.linalg.norm(ion.position - upper_center) for ion in unentered_ions]
            min_index = np.argmin(distances)
            closest_ion = unentered_ions[min_index]

            result[ts.frame] = {
                "resid": int(closest_ion.resid),
                "distance": float(distances[min_index])
            }

        if entered_ions:
            # Count if any ion is within 3 Å of SF from below
            for cutoff in [1, 2, 3, 4, 5, 6]:
                for ion in ions:
                    if ion.position[2] < gate_z:
                        dist = np.linalg.norm(ion.position - upper_center)
                        if dist < cutoff:
                            frame_counts_by_cutoff[cutoff] += 1
                            break  # only once per frame per cutoff

    # --- Save results ---
    if output_path:
        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, "closest_sf_unentered_ions.json"), "w") as f:
            json.dump(result, f, indent=2)

        total_frames = u.trajectory.n_frames
        cutoffs = [1, 2, 3, 4, 5, 6]
        counts = [frame_counts_by_cutoff[c] for c in cutoffs]
        percentages = [(count / total_frames) * 100 for count in counts]
        labels = [f"<{c}Å" for c in cutoffs]

        fig, ax = plt.subplots(figsize=(6, 6))
        bar_positions = range(len(cutoffs))
        bar_width = 0.3

        ax.bar(bar_positions, counts, width=bar_width, color="lightblue", edgecolor="black")

        # Add text on top of bars
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax.text(i, count + 10, f"{count} ({pct:.1f}%)",
                    ha='center', va='bottom', fontsize=12)

        ax.set_xticks(bar_positions)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel("Frame Count", fontsize=16)
        ax.set_title("Ion Below Gate within Cutoff", fontsize=18)
        ax.tick_params(axis='y', labelsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "frames_with_ion_below_sf_cutoffs.png"))
        plt.close()



    return result

