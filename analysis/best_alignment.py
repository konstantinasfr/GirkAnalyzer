import pandas as pd
import numpy as np
from itertools import combinations
from analysis.converter import convert_to_pdb_numbering

import pandas as pd
import numpy as np
from itertools import combinations

def convert_to_pdb_numbering(residue_id, channel_type):
    if channel_type == "G4":
        residues_per_chain = 325
        offset = 49
    elif channel_type == "G2":
        residues_per_chain = 328
        offset = 54
    elif channel_type == "G12":
        residues_per_chain = 325
        offset = 53
    else:
        raise ValueError("Unknown channel type")

    amino_acid_names = {
        152: "E", 184: "N", 141: "E", 173: "D"
    }

    if residue_id != "SF":
        chain_number = int(residue_id) // residues_per_chain
        chain_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
        pdb_number = residue_id - residues_per_chain * chain_number + offset
        if channel_type == "G12" and residue_id <= 325:
            pdb_number = residue_id + 42
        aa = amino_acid_names.get(pdb_number, "X")
        return f"{aa}{pdb_number}.{chain_dict.get(chain_number, '?')}"
    else:
        return "SF"

def get_best_aligned_residue_combo(df, frame_number, max_combo_size=None, source_filter=None, channel_type="G2"):
    def cosine_similarity(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    frame_df = df[df['frame'] == frame_number].copy()

    if source_filter:
        if isinstance(source_filter, list):
            frame_df = frame_df[frame_df['source_type'].isin(source_filter)]
        else:
            frame_df = frame_df[frame_df['source_type'] == source_filter]

    if frame_df.empty:
        return {"error": f"No data for frame {frame_number} with source_type {source_filter}."}

    # Motion vector and its unit vector
    motion_vec = np.array([
        frame_df["motion_vec_x"].iloc[0],
        frame_df["motion_vec_y"].iloc[0],
        frame_df["motion_vec_z"].iloc[0]
    ])
    motion_unit = motion_vec / np.linalg.norm(motion_vec)

    # Candidate force sources
    sources = list(frame_df[['resid', 'Fx', 'Fy', 'Fz', 'resname']].itertuples(index=False, name=None))
    max_size = max_combo_size if max_combo_size else len(sources)

    best_cosine = -1
    best_combo = None
    best_vector = None
    best_combo_numbers = None
    best_contributions = None

    for r in range(1, max_size + 1):
        for combo in combinations(sources, r):
            vector_sum = np.sum([[x[1], x[2], x[3]] for x in combo], axis=0)
            cos_sim = cosine_similarity(vector_sum, motion_vec)

            if cos_sim > best_cosine:
                best_cosine = cos_sim
                best_vector = vector_sum

                best_combo = []
                best_combo_numbers = []
                best_contributions = []

                for resid, fx, fy, fz, resname in combo:
                    force_vec = np.array([fx, fy, fz])
                    aligned_component = np.dot(force_vec, motion_unit)

                    try:
                        value = int(resid)
                        if value < 1300 and resname != "PIP":
                            label = convert_to_pdb_numbering(value, channel_type)
                        else:
                            label = f"{resname}_{value}"
                    except Exception:
                        label = str(resid)

                    best_combo.append(label)
                    best_combo_numbers.append(resid)
                    best_contributions.append((label, aligned_component))

                # Sort by aligned force contribution (descending)
                best_contributions.sort(key=lambda x: x[1], reverse=True)

    return {
        "frame": frame_number,
        "best_cosine_similarity": best_cosine,
        "residue_combination": best_combo,
        "residue_combination_numb": best_combo_numbers,
        "residue_contributions_sorted": best_contributions,
        "vector_sum": best_vector,
        "magnitude": np.linalg.norm(best_vector)
    }


import matplotlib.pyplot as plt
from collections import Counter
import ast

import matplotlib.pyplot as plt
from collections import Counter

def plot_residue_combination_frequency_from_df(df, output_path="residue_appearance_plot.png"):
    """
    Plots how many times each residue appears in the 'residue_combination' column of a DataFrame,
    with frequency and percentage shown on top of each bar.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'residue_combination' column (as actual lists).
        output_path (str): Path to save the output bar plot image.
    """
    all_residues = []

    for combo in df["residue_combination"]:
        if isinstance(combo, list):
            all_residues.extend(combo)
        else:
            print(f"[Warning] Skipping non-list entry: {combo}")

    # Count and sort
    residue_counts = Counter(all_residues)
    if not residue_counts:
        print("[Error] No valid residue combinations found.")
        return

    total = sum(residue_counts.values())
    sorted_items = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percent = (count / df.shape[0]) * 100
        label = f"{count} ({percent:.1f}%)"
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, label,
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.xlabel("Residue", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.yticks(fontsize=12)
    plt.title("Residue Appearance Frequency in Best Combinations", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_path}/residue_appearance_plot.png", dpi=300)
    plt.close()
    print(f"[Success] Plot saved to {output_path}")




import os
import pandas as pd
from collections import defaultdict

import os
import json
from tqdm import tqdm
import os


def run_best_combo_per_ion_from_json(csv_folder, frame_data, output_csv, source_filter="residue", channel_type="G2"):
    """
    Runs best-aligned force combo analysis for each ion using start_frame from a JSON list.

    Parameters:
    - csv_folder: folder with CSVs like 2223_1.csv
    - json_path: path to the JSON file with ion_id and start_frame
    - source_filter: 'residue', 'pip', etc.
    - channel_type: 'G2', 'G4', or 'G12'

    Returns:
    - List of result dictionaries
    """
    # Load JSON into a dict for quick lookup
    # with open(json_path, "r") as f:
    #     frame_data = json.load(f)
    frame_map = {entry["ion_id"]: entry["start_frame"]-1 for entry in frame_data}

    results = []
    for fname in tqdm(os.listdir(csv_folder), desc="Processing CSV files"):
        if not fname.endswith(".csv"):
            continue

        ion_id = fname[:-4]  # remove .csv
        if ion_id not in frame_map:
            print(f"⚠️ Skipping {ion_id}, no start_frame in JSON.")
            continue

        frame_number = frame_map[ion_id]
        filepath = os.path.join(csv_folder, fname)
        df = pd.read_csv(filepath)

        result = get_best_aligned_residue_combo(
            df,
            frame_number=frame_number,
            source_filter=source_filter,
            channel_type=channel_type
        )
        result["ion_id"] = ion_id
        result["start_frame"] = frame_number
        results.append(result)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_csv}/best_force_alignment_results.csv", index=False)
    print(f"✅ Saved results to {output_csv}")

    plot_residue_combination_frequency_from_df(results_df, output_csv)

    return results_df