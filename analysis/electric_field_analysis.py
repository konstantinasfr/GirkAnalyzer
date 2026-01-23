import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import json
from analysis.converter import convert_to_pdb_numbering

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm


def compute_electric_field_at_point(point, atom_positions, atom_charges, k=332):
    point = np.asarray(point, dtype=float)
    total_field = np.zeros(3)
    for pos, q in zip(atom_positions, atom_charges):
        r_vec = point - pos
        r = np.linalg.norm(r_vec)
        if r < 1e-6:
            continue
        total_field += k * q * r_vec / (r ** 3)
    return total_field


def add_totals_to_result(result, axis_unit_vector):
    def sum_fields(field_dict):
        return np.sum([np.array(entry["field"]) for entry in field_dict.values()], axis=0)

    glu_total = sum_fields(result["glu_residues"])
    asn_total = sum_fields(result["asn_residues"])
    pip2_total = sum_fields(result["pip2_residues"])
    ions_up = np.array(result["ionic_up_total_field"]["field"])
    ions_down = np.array(result["ionic_down_total_field"]["field"])

    total_no_ions = glu_total + asn_total + pip2_total
    total_asn_glu = glu_total + asn_total 
    total_all = total_no_ions + ions_up + ions_down

    result["total_glu"] = {
        "field": glu_total.tolist(),
        "magnitude": float(np.linalg.norm(glu_total)),
        "axial": float(np.dot(glu_total, axis_unit_vector))
    }
    result["total_asn"] = {
        "field": asn_total.tolist(),
        "magnitude": float(np.linalg.norm(asn_total)),
        "axial": float(np.dot(asn_total, axis_unit_vector))
    }
    result["total_no_ions"] = {
        "field": total_no_ions.tolist(),
        "magnitude": float(np.linalg.norm(total_no_ions)),
        "axial": float(np.dot(total_no_ions, axis_unit_vector))
    }

    result["total_asn_glu"] = {
        "field": total_asn_glu.tolist(),
        "magnitude": float(np.linalg.norm(total_asn_glu)),
        "axial": float(np.dot(total_asn_glu, axis_unit_vector))
    }

    result["total_pip"] = {
        "field": pip2_total.tolist(),
        "magnitude": float(np.linalg.norm(pip2_total)),
        "axial": float(np.dot(pip2_total, axis_unit_vector))
    }
    result["total"] = {
        "field": total_all.tolist(),
        "magnitude": float(np.linalg.norm(total_all)),
        "axial": float(np.dot(total_all, axis_unit_vector))
    }

    return result


def compute_frame_field(u, frame, point, glu_residues, asn_residues,
                        pip2_resids, exclude_backbone, sf_stuck_ions, headgroup_atoms,
                        axis_unit_vector, ion_selection="resname K+ K", k=332,
                        distance_cutoff=15.0):  # Å default threshold

    u.trajectory[frame]

    def collect_pa_ch(atom_group, exclude_backbone=False, allowed_names=None):
        pos, q = [], []
        for atom in atom_group:
            if exclude_backbone and atom.name in {"N", "CA", "C", "O", "HA", "H"}:
                continue
            if allowed_names and atom.name not in allowed_names:
                continue
            pos.append(atom.position)
            q.append(atom.charge)
        return pos, q

    result = {
        "glu_residues": {},
        "asn_residues": {},
        "pip2_residues": {},
        "ionic_up_fields": [],
        "ionic_down_fields": [],
        "ionic_up_total_field": None,
        "ionic_down_total_field": None
    }

    for resid_list, label in [(glu_residues, "glu_residues"), (asn_residues, "asn_residues")]:
        for resid in resid_list:
            atoms = u.select_atoms(f"resid {resid}")
            pos, q = collect_pa_ch(atoms, exclude_backbone)
            field = compute_electric_field_at_point(point, pos, q, k)
            result[label][int(resid)] = {
                "field": field.tolist(),
                "magnitude": float(np.linalg.norm(field)),
                "axial": float(np.dot(field, axis_unit_vector))
            }

    pip2_fields = {}
    for resid in pip2_resids:
        atoms = u.select_atoms(f"resid {resid}")
        pos, q = collect_pa_ch(atoms, exclude_backbone, allowed_names=headgroup_atoms)
        field = compute_electric_field_at_point(point, pos, q, k)
        pip2_fields[int(resid)] = {
            "field": field.tolist(),
            "magnitude": float(np.linalg.norm(field)),
            "axial": float(np.dot(field, axis_unit_vector))
        }
    result["pip2_residues"] = pip2_fields

    up_list, down_list = [], []
    up_total, down_total = np.zeros(3), np.zeros(3)

    stuck_resid = int(sf_stuck_ions[int(frame)]["resid"])

    for ion in u.select_atoms(ion_selection):

        # Skip the SF-stuck ion for this frame
        if stuck_resid is not None and ion.resid == stuck_resid:
            # print(ion.resid, frame)
            continue

        # Compute distance to point
        dist = np.linalg.norm(point - ion.position)
        if dist > distance_cutoff:
            continue  # ❌ skip distant ions

        # if frame == 3992:
        #     print(ion.resid, point, ion.position)

        # Compute electric field from this ion at the point
        field = compute_electric_field_at_point(point, [ion.position], [ion.charge], k)

        entry = {
            "ion_id": int(ion.resid),
            "field": field.tolist(),
            "magnitude": float(np.linalg.norm(field)),
            "axial": float(np.dot(field, axis_unit_vector)),
            "distance": float(dist)
        }

        # Classify based on Z-position relative to the point
        if ion.position[2] > point[2]:
            up_list.append(entry)
            up_total += field
        else:
            down_list.append(entry)
            down_total += field


    result["ionic_up_fields"] = up_list
    result["ionic_down_fields"] = down_list
    result["ionic_up_total_field"] = {
        "field": up_total.tolist(),
        "magnitude": float(np.linalg.norm(up_total)),
        "axial": float(np.dot(up_total, axis_unit_vector))
    }
    result["ionic_down_total_field"] = {
        "field": down_total.tolist(),
        "magnitude": float(np.linalg.norm(down_total)),
        "axial": float(np.dot(down_total, axis_unit_vector))
    }

    return result


def detect_pip2_resids(u, pip2_resname):
    pip2_atoms = u.select_atoms(f"resname {pip2_resname}")
    pip2_resids = sorted(list(set(pip2_atoms.resids)))
    if len(pip2_resids) != 4:
        raise ValueError(f"Expected 4 PIP2 residues, but found {len(pip2_resids)}: {pip2_resids}")
    return pip2_resids


def run_field_analysis(u, sf_residues, hbc_residues, glu_residues, asn_residues,
                       pip2_resname, headgroup_atoms, exclude_backbone, sf_stuck_ions,
                       output_path="sf_field_results.json",
                       point_strategy="sf_com", fixed_point=None,
                       ion_selection="resname K+ K"):

    pip2_resids = detect_pip2_resids(u, pip2_resname)
    field_by_frame = {}

    sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
    hbc_group = u.select_atoms("resid " + " ".join(map(str, hbc_residues)))

    axis_vector = hbc_group.center_of_mass() - sf_group.center_of_mass()
    # axis_unit_vector = axis_vector / np.linalg.norm(axis_vector)
    axis_unit_vector = np.array([0, 0, -1])  # προς τα πάνω


    for ts in tqdm(u.trajectory, desc="Frames"):
        if point_strategy == "sf_com":
            sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
            point = sf_group.center_of_mass()

        elif point_strategy == "sf_min_atoms":
            atom_indices = []
            for resid in sf_residues:
                residue_atoms = u.select_atoms(f"resid {resid}")
                if len(residue_atoms) == 0:
                    continue
                coords = residue_atoms.positions
                min_index = coords[:, 2].argmin()
                atom_indices.append(residue_atoms[min_index].index)
            sf_min_atoms = u.atoms[atom_indices]
            point = sf_min_atoms.center_of_mass()

        elif point_strategy == "sf_min_atoms_com_xy_min_z":
            atom_indices = []
            for resid in sf_residues:
                residue_atoms = u.select_atoms(f"resid {resid}")
                if len(residue_atoms) == 0:
                    continue
                coords = residue_atoms.positions
                min_index = coords[:, 2].argmin()
                atom_indices.append(residue_atoms[min_index].index)
            sf_min_atoms = u.atoms[atom_indices]
            com = sf_min_atoms.center_of_mass()
            min_z = sf_min_atoms.positions[:, 2].min()
            point = np.array([com[0], com[1], min_z])

        elif point_strategy == "fixed":
            if fixed_point is None:
                raise ValueError("fixed_point must be provided")
            point = np.asarray(fixed_point)

        else:
            raise ValueError("point_strategy must be one of 'sf_com', 'sf_min_atoms', or 'fixed'")

        # if ts.frame<30:
        #     print("field", ts.frame, point)
        frame_result = compute_frame_field(
            u, ts.frame, point, glu_residues, asn_residues,
            pip2_resids, exclude_backbone, sf_stuck_ions, headgroup_atoms,
            axis_unit_vector, ion_selection, k=332
        )
        field_by_frame[int(ts.frame)] = add_totals_to_result(frame_result, axis_unit_vector)

    with open(output_path, "w") as f:
        json.dump(field_by_frame, f, indent=2)

    print(f"✅ Saved electrostatic field analysis to {output_path}")

# Example usage:
# if __name__ == "__main__":
#     run_field_analysis(
#         topology="com_4fs.prmtop",
#         trajectory="protein.nc",
#         sf_residues=[100, 428, 756, 1084],
#         glu_residues=[98, 426, 754, 1082],
#         asn_residues=[130, 458, 786, 1114],
#         pip2_resname="PIP",
#         headgroup_atoms={"P4", "P5", "O4P", "O5P", "O41", "O42", "O43", "O51", "O52", "O53"},
#         output_path="sf_field_results.json",
#         point_strategy="sf_com"
#     )

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def smooth(values, window_size=20):
    values = np.array(values)
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='same')

def plot_field_magnitudes_from_json(field_json_path, ion_events, output_dir="field_magnitude_plots", channel_type="G2"):
    with open(field_json_path) as f:
        data = json.load(f)

    start_frames = [event["start_frame"] for event in ion_events]
    frames = sorted(data.keys(), key=lambda x: int(x))

    def extract_data(field_key):
        data_dict = {}
        for frame in frames:
            frame_data = data[frame].get(field_key, {})
            for resid, entry in frame_data.items():
                data_dict.setdefault(resid, {"magnitude": [], "axial": []})
                data_dict[resid]["magnitude"].append(entry.get("magnitude"))
                data_dict[resid]["axial"].append(entry.get("axial"))
        return data_dict

    def extract_total(key):
        mag, axial = [], []
        for frame in frames:
            entry = data[frame].get(key, {})
            mag.append(entry.get("magnitude"))
            axial.append(entry.get("axial"))
        return mag, axial

    glu_data = extract_data("glu_residues")
    asn_data = extract_data("asn_residues")
    pip_data = extract_data("pip2_residues")

    ion_up_data, ion_down_data = {}, {}
    ion_up_total_mag, ion_up_total_ax = [], []
    ion_down_total_mag, ion_down_total_ax = [], []

    for frame in frames:
        frame_data = data[frame]

        for ion in frame_data.get("ionic_up_fields", []):
            ion_id = ion["ion_id"]
            ion_up_data.setdefault(ion_id, {"magnitude": [], "axial": []})
            ion_up_data[ion_id]["magnitude"].append(ion.get("magnitude"))
            ion_up_data[ion_id]["axial"].append(ion.get("axial"))

        for ion in frame_data.get("ionic_down_fields", []):
            ion_id = ion["ion_id"]
            ion_down_data.setdefault(ion_id, {"magnitude": [], "axial": []})
            ion_down_data[ion_id]["magnitude"].append(ion.get("magnitude"))
            ion_down_data[ion_id]["axial"].append(ion.get("axial"))

        ion_up_total_mag.append(frame_data["ionic_up_total_field"]["magnitude"])
        ion_down_total_mag.append(frame_data["ionic_down_total_field"]["magnitude"])
        ion_up_total_ax.append(frame_data["ionic_up_total_field"]["axial"])
        ion_down_total_ax.append(frame_data["ionic_down_total_field"]["axial"])

    total_glu_mag, total_glu_ax = extract_total("total_glu")
    total_asn_mag, total_asn_ax = extract_total("total_asn")
    total_all_mag, total_all_ax = extract_total("total")
    total_no_ions_mag, total_no_ions_ax = extract_total("total_no_ions")
    total_pip_mag, total_pip_ax = extract_total("total_pip")
    total_asn_glu_mag, total_asn_glu_ax = extract_total("total_asn_glu")

    def save_plot(fig, base_dir, subfolder, filename):
        full_path = os.path.join(base_dir, subfolder)
        os.makedirs(full_path, exist_ok=True)
        fig.savefig(os.path.join(full_path, filename))
        

    def plot_two_subplots(data_dict, title, filename_base, subfolder, channel_type=None):
        for metric in ["magnitude", "axial"]:
            for with_lines in [False, True]:
                for smooth_flag in [False, True]:
                    fig = plt.figure(figsize=(12, 6))
                    for key, value in data_dict.items():
                        y = smooth(value[metric]) if smooth_flag else value[metric]
                        label = convert_to_pdb_numbering(key, channel_type) if channel_type else str(key)
                        plt.plot(y, label=label, alpha=0.8, linewidth=1.2)
                    if with_lines:
                        for i, x in enumerate(start_frames):
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8,
                                        label="Ion leaves SF" if i == 0 else None)
                    plt.xlabel("Frame", fontsize=16)
                    plt.ylabel("Field " + metric.capitalize(), fontsize=16)
                    plt.title(f"{title} ({metric.capitalize()}){' (Smoothed)' if smooth_flag else ''}", fontsize=18)
                    plt.legend(fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    suffix = f"_{metric}{'_smoothed' if smooth_flag else ''}{'_start_lines' if with_lines else ''}.png"
                    plt.tight_layout()
                    save_plot(fig, output_dir, subfolder, filename_base + suffix)
                    plt.close()

    def plot_combined(mag_lists, ax_lists, labels, title, filename_base, subfolder):
        for metric, sources in zip(["magnitude", "axial"], [mag_lists, ax_lists]):
            for with_lines in [False, True]:
                for smooth_flag in [False, True]:
                    fig = plt.figure(figsize=(12, 6))
                    for i, values in enumerate(sources):
                        y = smooth(values) if smooth_flag else values
                        plt.plot(y, label=labels[i], alpha=0.8, linewidth=1.2)
                    if with_lines:
                        for i, x in enumerate(start_frames):
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8,
                                        label="Ion leaves SF" if i == 0 else None)
                    plt.xlabel("Frame", fontsize=16)
                    plt.ylabel("Field " + metric.capitalize(), fontsize=16)
                    plt.title(f"{title} ({metric.capitalize()}){' (Smoothed)' if smooth_flag else ''}", fontsize=18)
                    plt.legend(fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    suffix = f"_{metric}{'_smoothed' if smooth_flag else ''}{'_start_lines' if with_lines else ''}.png"
                    plt.tight_layout()
                    save_plot(fig, output_dir, subfolder, filename_base + suffix)
                    plt.close()

    def plot_summary_combo(data_dicts, total_vals, labels, title, filename_base, subfolder,
                       output_dir, start_frames, channel_type=None):
        for metric in ["magnitude", "axial"]:
            for with_lines in [False, True]:
                for smooth_flag in [False, True]:
                    fig = plt.figure(figsize=(12, 6))
                    for i, data_dict in enumerate(data_dicts):
                        for resid, entry in data_dict.items():
                            y = smooth(entry[metric]) if smooth_flag else entry[metric]
                            label = convert_to_pdb_numbering(resid, channel_type) if int(resid)<1300 else str(resid)
                            plt.plot(y, alpha=0.5, linewidth=1.0, label=f"{labels[i]} {label}")

                    for i, total in enumerate(total_vals):
                        y = smooth(total[metric]) if smooth_flag else total[metric]
                        plt.plot(y, label=f"Total {labels[i]}", linewidth=2.0)

                    if with_lines:
                        for i, x in enumerate(start_frames):
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8,
                                        label="Ion leaves SF" if i == 0 else None)

                    plt.xlabel("Frame", fontsize=16)
                    plt.ylabel("Field " + metric.capitalize(), fontsize=16)
                    plt.title(f"{title} ({metric.capitalize()}){' (Smoothed)' if smooth_flag else ''}", fontsize=18)
                    plt.legend(fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    suffix = f"_{metric}{'_smoothed' if smooth_flag else ''}{'_start_lines' if with_lines else ''}.png"
                    full_path = os.path.join(output_dir, subfolder)
                    os.makedirs(full_path, exist_ok=True)
                    fig.savefig(os.path.join(full_path, filename_base + suffix))
                    plt.close()


    # === Generate plots in organized folders ===
    plot_two_subplots(glu_data, "GLU Residue Fields", "glu_field", "residues_glu", channel_type)
    plot_two_subplots(asn_data, "ASN Residue Fields", "asn_field", "residues_asn", channel_type)
    plot_two_subplots(pip_data, "PIP2 Residue Fields", "pip2_field", "residues_pip2")
    plot_two_subplots(ion_up_data, "Ionic Up Field (Individual)", "ion_up_field", "ions_up_individual")
    plot_two_subplots(ion_down_data, "Ionic Down Field (Individual)", "ion_down_field", "ions_down_individual")

    plot_combined([ion_up_total_mag, ion_down_total_mag], [ion_up_total_ax, ion_down_total_ax],
                  ["Up Total", "Down Total"], "Total Ion Field", "ion_total_field", "ions_total")
    plot_combined([total_glu_mag], [total_glu_ax], ["Total GLU"], "GLU Total Field", "total_glu", "total_glu")
    plot_combined([total_asn_mag], [total_asn_ax], ["Total ASN"], "ASN Total Field", "total_asn", "total_asn")
    plot_combined([total_all_mag], [total_all_ax], ["All Sources"], "Total Field from All Sources", "total_all_sources", "total_all")
    plot_combined([total_no_ions_mag], [total_no_ions_ax], ["ASN-GLU-PIP2"], "Total Field with ASN GLU PIP2", "total_no_ions", "total_no_ions")
    plot_combined([total_pip_mag], [total_pip_ax], ["Total PIP2"], "Total Field without Ions", "total_pip", "total_pip")
    plot_combined([total_asn_glu_mag], [total_asn_glu_ax], ["ASN-GLU-"], "ASN-GLU- Field", "total_asn_glu", "total_asn_glu")

    plot_summary_combo([asn_data], [{"magnitude": total_asn_mag, "axial": total_asn_ax}],
                       [""], "ASN + Total ASN", "combo_asn", "combo_asn",
                       output_dir, start_frames, channel_type)

    plot_summary_combo([glu_data], [{"magnitude": total_glu_mag, "axial": total_glu_ax}],
                       [""], "GLU + Total GLU", "combo_glu", "combo_glu",
                       output_dir, start_frames, channel_type)

    plot_summary_combo([pip_data], [{"magnitude": total_pip_mag, "axial": total_pip_ax}],
                       ["PIP2"], "PIP2 + Total PIP2", "combo_pip", "combo_pip",
                       output_dir, start_frames, channel_type)

    plot_summary_combo([asn_data, glu_data], [{"magnitude": total_asn_glu_mag, "axial": total_asn_glu_ax}],
                       ["", ""], "ASN + GLU + Total", "combo_asn_glu", "combo_asn_glu",
                       output_dir, start_frames, channel_type)

    plot_summary_combo([asn_data, glu_data, pip_data], [{"magnitude": total_no_ions_mag, "axial": total_no_ions_ax}],
                       ["", "", "PIP2"], "ASN + GLU + PIP2 + Total", "combo_all_no_ions", "combo_all_no_ions",
                       output_dir, start_frames, channel_type)


    print(f"✅ All field magnitude and axial direction plots saved in organized subfolders under: {output_dir}")



import json
import numpy as np
from scipy.stats import mannwhitneyu
import os


def significance_field_analysis(field_json_path, ion_events, output_folder):
    # === Create output folder ===
    os.makedirs(output_folder, exist_ok=True)

    # === Load field data from JSON ===
    with open(field_json_path) as f:
        field_data = json.load(f)

    # === Collect start frames from ion events ===
    start_frames = [event["start_frame"] - 1 for event in ion_events]

    # === Function to extract magnitude and axial lists for a given key ===
    def extract_lists(key):
        mag, axial = [], []
        for frame_str in field_data:
            entry = field_data[frame_str].get(key, {})
            mag.append(entry.get("magnitude", 0))
            axial.append(entry.get("axial", 0))
        return np.array(mag), np.array(axial)

    # === Define the total field categories to analyze ===
    fields = {
        "total_glu": "GLU",
        "total_asn": "ASN",
        "total_asn_glu": "ASN_GLU",
        "total_pip": "PIP2",
        "total_no_ions": "ASN_GLU_PIP2"
    }

    # === Open report file to write results ===
    report_path = os.path.join(output_folder, "significance_report.txt")
    with open(report_path, "w") as report_file:
        # === Loop over each field and perform analysis ===
        for json_key, label in fields.items():
            mag, axial = extract_lists(json_key)

            # Frame indices
            all_frames = np.arange(len(mag))
            is_start = np.isin(all_frames, start_frames)

            # Split into start frame and rest
            mag_start = mag[is_start]
            mag_rest = mag[~is_start]
            axial_start = axial[is_start]
            axial_rest = axial[~is_start]

            # Save frame-specific values
            np.savetxt(os.path.join(output_folder, f"{label}_magnitude_start_frames.txt"), mag_start)
            np.savetxt(os.path.join(output_folder, f"{label}_axial_start_frames.txt"), axial_start)

            # Mann-Whitney U test
            mag_stat, mag_p = mannwhitneyu(mag_start, mag_rest, alternative='two-sided')
            axial_stat, axial_p = mannwhitneyu(axial_start, axial_rest, alternative='two-sided')

            # Format result text
            result = (
                f"=== {label} ===\n"
                f"  Magnitude: U={mag_stat:.2f}, p={mag_p:.4f} -> {'Significant ✅' if mag_p < 0.05 else 'Not significant ❌'}\n"
                f"  Axial    : U={axial_stat:.2f}, p={axial_p:.4f} -> {'Significant ✅' if axial_p < 0.05 else 'Not significant ❌'}\n\n"
            )

            # Print and write to file
            print(result)
            report_file.write(result)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from tqdm import tqdm
import os
from analysis.converter import convert_to_pdb_numbering


def generate_electric_field_heatmap_along_axis(u, sf_residues, hbc_residues, glu_residues, asn_residues,
                                 pip2_resname, headgroup_atoms, exclude_backbone, sf_stuck_ions,
                                 ion_selection="resname K+ K", n_points=20,
                                 start=0, end=None, channel_type="G2", k=332,
                                 mode="axial",  # or "magnitude"
                                 output_dir="field_heatmaps",
                                 vmin=-200, vmax=200):

    import os
    os.makedirs(output_dir, exist_ok=True)

    pip2_resids = detect_pip2_resids(u, pip2_resname)
    sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
    hbc_group = u.select_atoms("resid " + " ".join(map(str, hbc_residues)))
    sf_com = sf_group.center_of_mass()
    hbc_com = hbc_group.center_of_mass()
    axis_vector = hbc_com - sf_com
    # axis_unit_vector = axis_vector / np.linalg.norm(axis_vector)
    axis_unit_vector = np.array([0, 0, -1])  # points down the channel

    points_along_axis = [sf_com + i * axis_vector / (n_points - 1) for i in range(n_points)]

    if end is None:
        end = len(u.trajectory)

    field_sources = {
        "total": "GLU + ASN + PIP2 + Ions",
        "total_no_ions": "GLU + ASN + PIP2",
        "total_asn_glu": "GLU + ASN",
        "total_glu": "GLU Only",
        "total_asn": "ASN Only"
    }

    # Compute heatmap for each source type
    for key, label in field_sources.items():
        print(f"📊 Computing heatmap: {label}")
        heatmap_matrix = []

        for ts in tqdm(u.trajectory[start:end], desc=label):
            frame_values = []
            for point in points_along_axis:
                result = compute_frame_field(
                    u, ts.frame, point, glu_residues, asn_residues,
                    pip2_resids, exclude_backbone, sf_stuck_ions, headgroup_atoms,
                    axis_unit_vector, ion_selection, k
                )
                result = add_totals_to_result(result, axis_unit_vector)
                value = result.get(key, {}).get(mode, 0)
                frame_values.append(value)
            heatmap_matrix.append(frame_values)

        heatmap_matrix = np.array(heatmap_matrix).T  # rows = axis points, cols = frames

        # === Map residue Z-positions to point index along axis ===
        def map_residues_to_axis_indices(residues):
            indices = []
            for resid in residues:
                atoms = u.select_atoms(f"resid {resid}")
                if len(atoms) == 0:
                    continue
                z = atoms.center_of_mass()[2]
                closest_idx = np.argmin([abs(point[2] - z) for point in points_along_axis])
                indices.append(closest_idx)
            return indices

        glu_indices = map_residues_to_axis_indices(glu_residues)
        asn_indices = map_residues_to_axis_indices(asn_residues)

        # === Plot ===
        # === Plot using real Z coordinates ===
        z_coords = [point[2] for point in points_along_axis]
        extent = [start, end, z_coords[0], z_coords[-1]]
        boundaries = max(abs(np.min(heatmap_matrix)), abs(np.max(heatmap_matrix)))
        vmin = -boundaries
        vmax = boundaries

        plt.figure(figsize=(12, 6))

        im = plt.imshow(heatmap_matrix, aspect='auto', origin='lower', cmap="coolwarm",
                        extent=extent, vmin=vmin, vmax=vmax)

        plt.colorbar(im, label=f"Electric Field {mode.capitalize()}")
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Z Position (Å)", fontsize=14)
        plt.title(f"{label} – Electric Field {mode.capitalize()} Along Channel Axis", fontsize=16)

        # === Compute mean Z positions for each GLU/ASN residue ===
        def compute_residue_mean_z(u, residue_ids, start=0, end=None):
            if end is None:
                end = len(u.trajectory)
            z_sums = {resid: 0.0 for resid in residue_ids}
            counts = {resid: 0 for resid in residue_ids}
            for ts in u.trajectory[start:end]:
                for resid in residue_ids:
                    atoms = u.select_atoms(f"resid {resid}")
                    if len(atoms) > 0:
                        z = atoms.center_of_mass()[2]
                        z_sums[resid] += z
                        counts[resid] += 1
            mean_z = {}
            for resid in residue_ids:
                if counts[resid] > 0:
                    mean_z[resid] = z_sums[resid] / counts[resid]
            return mean_z

        glu_z = compute_residue_mean_z(u, glu_residues, start, end)
        asn_z = compute_residue_mean_z(u, asn_residues, start, end)

        # === Unique colors for each residue ===
        glu_colors = cm.get_cmap("Greens")(np.linspace(0.4, 0.9, len(glu_z)))
        asn_colors = cm.get_cmap("Purples")(np.linspace(0.4, 0.9, len(asn_z)))

        # === Plot horizontal lines with labels ===
        for color, (resid, z) in zip(glu_colors, glu_z.items()):
            label = convert_to_pdb_numbering(resid, channel_type=channel_type)
            plt.axhline(y=z, linestyle="--", color=color, linewidth=1.0, label=f"{label}")

        for color, (resid, z) in zip(asn_colors, asn_z.items()):
            label = convert_to_pdb_numbering(resid, channel_type=channel_type)
            plt.axhline(y=z, linestyle="--", color=color, linewidth=1.0, label=f"{label}")

        # === Clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), fontsize=9, loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{key}_{mode}.png"))
        plt.close()
        print(f"✅ Saved heatmap to {output_dir}/heatmap_{key}_{mode}.png")