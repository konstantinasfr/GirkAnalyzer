from analysis.converter import convert_to_pdb_numbering
import numpy as np

def compute_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)

def analyze_close_residues(positions, permeating_ion_id, frame, other_ions,
                  close_contacts_dict, cutoff=15.0):
    """
    Analyze one frame: compute ionic forces, motion, and optionally total force.
    Also calculates cosine similarities between different vectors and force decomposition.
    """
    result = {
        permeating_ion_id: None
    }

    permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
    if permeating_pos is None:
        return result

    for ion_id, pos in positions.get(frame, {}).items():
        # if ion_id == permeating_ion_id or ion_id not in other_ions:
        #     continue
        distance = compute_distance(permeating_pos, pos)


        if distance <= cutoff:
            if ion_id not in close_contacts_dict:
                result[int(ion_id)] = ["SF"]
            elif frame not in close_contacts_dict[ion_id]:
                result[int(ion_id)] = ["SF"]
            else:
                result[int(ion_id)] = close_contacts_dict[ion_id][frame]

    return result



###################################################################

import os
import json
import pandas as pd


def get_last_nth_frame_close_residues(event, n=-1, use_pdb_format=True, sort_residues=True, channel_type="G2"):
    """
    Extract close residues at a specific frame from a permeation event.

    Behavior:
    - If n < 0: counts from the end of the sorted frame list (e.g., -1 = last, -2 = second-last)
    - If n >= 0: directly uses frame number `n` as a key in event["analysis"]

    Parameters:
        event (dict): Contains 'analysis' with frame: {ion_id: residues}
        n (int): Frame position or frame number depending on sign
        use_pdb_format (bool): Whether to convert residues to PDB-style notation
        sort_residues (bool): Whether to sort residues alphabetically

    Returns:
        dict: {frame_number: {ion_id: "res1_res2_..."}}
    """
    frames = sorted(event["analysis"].keys(), key=lambda x: int(x))

    if n < 0:
        if abs(n) > len(frames):
            raise ValueError(f"Frame index {n} is out of range. Event has {len(frames)} frames.")
        selected_frame_key = frames[n]
    else:
        if int(n) not in event["analysis"]:
            raise ValueError(f"Frame {n} not found in event['analysis'].")
        selected_frame_key = int(n)

    original_data = event["analysis"][selected_frame_key]

    converted_data = {}
    for ion_id, residues in original_data.items():
        if sort_residues:
            if  any(isinstance(r, int) for r in residues):
            # Sort only if there are integers in the list
                residues = sorted(residues, key=lambda r: int(r))
            
        formatted_residues = [
            convert_to_pdb_numbering(res, channel_type) if use_pdb_format else str(res)
            for res in residues
        ]
        converted_data[ion_id] = "_".join(formatted_residues)

    return {selected_frame_key: converted_data}


import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_residue_counts(data, output_dir, filename="residue_counts.png", exclude=(), duplicates=True):
    """
    Plots and saves a bar chart of residue string occurrences, sorted by count.

    Parameters:
    - data (dict): {frame: {ion_id: residue_string}}
    - output_dir (str): Folder where the plot will be saved
    - filename (str): Name of output image file
    - exclude (tuple): Residue values to ignore (e.g. "SF", "no_close_residues")
    - duplicates (bool): 
        - If True (default), count all appearances
        - If False, count each unique residue string at most once per frame
    """
    # Step 1: Count residue strings
    residue_counter = Counter()
    os.makedirs(output_dir, exist_ok=True)

    for frame_dict in data.values():
        seen_in_frame = set()
        for residue_string in frame_dict.values():
            if residue_string in exclude:
                continue
            if not duplicates:
                if residue_string in seen_in_frame:
                    continue
                seen_in_frame.add(residue_string)
            residue_counter[residue_string] += 1

    if not residue_counter:
        print("⚠️ No residue strings to plot.")
        return

    # Step 2: Sort by frequency
    sorted_items = sorted(residue_counter.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Step 3: Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Plot
    plt.figure(figsize=(max(12, len(labels) * 0.5), 6))  # auto-width for many bars
    bars = plt.bar(labels, counts, color='steelblue')

    # Add count labels above each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count),
                 ha='center', va='bottom', fontsize=11)

    # Format y-axis: only integers and some top margin
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(False)

    # Labels and layout
    plt.title(f"Residue Combinations Occurrences in {len(data)} Events", fontsize=16)
    plt.xlabel("Residue Combinations", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Step 5: Save figure
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Clean plot saved to: {plot_path}")


import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

def analyze_residue_combinations(data, output_dir, top_n_plot=20):
    """
    Analyzes how often all possible residue string combinations appear across frames.

    Parameters:
    - data (dict): {frame: {ion_id: residue_string}}
    - output_dir (str): Folder to save CSV and plot
    - top_n_plot (int): Number of top combinations to show in the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build set of residues per frame (excluding unwanted ones)
    frame_residue_sets = {
        frame: set(val for val in ion_dict.values() if val not in ("SF"))
        for frame, ion_dict in data.items()
    }

    # Step 2: Determine max combination size automatically
    max_comb_size = max(len(s) for s in frame_residue_sets.values())

    # Step 3: Count how many frames each combination appears in
    combination_counter = defaultdict(int)

    for size in range(1, max_comb_size + 1):
        for residue_set in frame_residue_sets.values():
            for combo in combinations(sorted(residue_set), size):
                combination_counter[combo] += 1

    # Step 4: Save to CSV
    csv_path = os.path.join(output_dir, "residue_combination_frequencies.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Combination", "Count"])
        for combo, count in sorted(combination_counter.items(), key=lambda x: (-len(x[0]), -x[1])):
            writer.writerow(["+".join(combo), count])
    print(f"✅ CSV saved to: {csv_path}")

    # Step 5: Plot top N combinations
    sorted_combos = sorted(combination_counter.items(), key=lambda x: x[1], reverse=True)
    top_combos = sorted_combos[:top_n_plot]
    labels = ["+".join(combo) for combo, _ in top_combos]
    counts = [count for _, count in top_combos]

    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    bars = plt.bar(labels, counts, color="mediumseagreen")

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, count + 0.5, str(count),
                 ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Frames Appeared", fontsize=14)
    plt.title(f"Top {top_n_plot} Residue Combinations Across {len(data)} Frames",  fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "residue_combination_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved to: {plot_path}")


# def closest_residues_comb_before_permeation(close_residues_results, output_base_dir, n=-1, use_pdb_format=False, sort_residues=True):
#     """
#     Loop through all permeation events and apply get_last_nth_frame_close_residues.
#     Saves both JSON and CSV outputs.
#     """
#     output_dir = os.path.join(output_base_dir, "closest_residues_comb")
#     os.makedirs(output_dir, exist_ok=True)

#     summary = []
#     for i, event in enumerate(close_residues_results):
#         try:
#             frame_data = get_last_nth_frame_close_residues(
#                 event, n=n, use_pdb_format=use_pdb_format, sort_residues=sort_residues
#             )
#             summary.append(frame_data)
#         except Exception as e:
#             print(f"Skipping event {i} due to error: {e}")

#     # Save JSON
#     with open(os.path.join(output_dir, f"closest_residues_n_{n}.json"), "w") as f:
#         json.dump(summary, f, indent=2)

#     # Save CSV
#     flat_rows = []
#     for event_summary in summary:
#         for frame, ion_data in event_summary.items():
#             for ion_id, residue_str in ion_data.items():
#                 flat_rows.append({
#                     "frame": frame,
#                     "ion_id": ion_id,
#                     "residues": residue_str
#                 })

#     df = pd.DataFrame(flat_rows)
#     df.to_csv(os.path.join(output_dir, f"closest_residues_n_{n}.csv"), index=False)


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def find_closest_residues_percentage(data, result_folder, channel_type="G12"):
    all_rows = []

    for ion_id, frame_list in data.items():
        residue_counts = defaultdict(int)
        total_frames = len(frame_list)

        for frame_entry in frame_list:
            res_id = frame_entry["residue"]

            # ✅ Skip SF-labeled entries
            if isinstance(res_id, str) and res_id.upper() == "SF":
                continue

            # ✅ Apply PDB conversion
            pdb_res = str(convert_to_pdb_numbering(res_id, channel_type))
            residue_counts[pdb_res] += 1

        # === Percentage and counts
        percentage_dict = {f"{res}_pct": (count / total_frames) * 100 for res, count in residue_counts.items()}
        count_dict = {f"{res}_count": count for res, count in residue_counts.items()}
        row = {"ion_id": ion_id, "total_frames": total_frames}
        row.update(percentage_dict)
        row.update(count_dict)
        all_rows.append(row)

    # === Create and save DataFrame
    df = pd.DataFrame(all_rows).fillna(0)
    df.to_csv(f"{result_folder}/single_closest_residue_distribution.csv", index=False)

    # === Prepare for boxplot
    pct_cols = [col for col in df.columns if col.endswith("_pct")]
    df_pct = df[pct_cols].copy()
    df_pct_long = df_pct.melt(var_name="Residue", value_name="Percentage")
    df_pct_long["Residue"] = df_pct_long["Residue"].str.replace("_pct", "")

    # === Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_pct_long, x="Residue", y="Percentage")
    plt.ylabel("Percentage of Frames",  fontsize=14)
    plt.xlabel("Residue",  fontsize=14)
    plt.title("Residue Proximity Distribution Across Ions",  fontsize=16)
    plt.xticks(rotation=45,  fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/residue_proximity_boxplot.png", dpi=300)

    print("✅ CSV and boxplot saved (excluding 'SF').")



import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def count_frames_residue_closest(data, result_folder, total_frames, channel_type="G2"):
    residue_to_frames = defaultdict(set)

    for ion_id, frame_list in data.items():
        for entry in frame_list:
            frame = entry["frame"]
            residue = entry["residue"]

            # Skip SF if labeled as string
            if isinstance(residue, str) and residue.upper() == "SF":
                continue

            # Convert residue to PDB format
            pdb_res = str(convert_to_pdb_numbering(residue, channel_type))

            # Only count one appearance per frame
            residue_to_frames[pdb_res].add(frame)

    # === Count occurrences
    residue_frame_counts = {res: len(frames) for res, frames in residue_to_frames.items()}

    # === DataFrame of counts
    df_counts = pd.DataFrame({
        "Residue": list(residue_frame_counts.keys()),
        "FrameCount": list(residue_frame_counts.values())
    })

    df_counts = df_counts.sort_values(by="FrameCount", ascending=False)
    df_counts.to_csv(f"{result_folder}/closest_residue_counts.csv", index=False)

    # === Bar plot: frame counts
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_counts["Residue"], df_counts["FrameCount"])
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Residue", fontsize=14)
    plt.ylabel("Number of Frames Appearing as Closest", fontsize=14)
    plt.title("Residues Most Frequently Closest to Any Ion (Per Frame)", fontsize=16)

    for bar, count in zip(bars, df_counts["FrameCount"]):
        height = bar.get_height()
        percentage = (count / total_frames) * 100
        label = f"{int(count)} ({percentage:.1f}%)"
        plt.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                label, ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{result_folder}/closest_single_residue_counts_barplot.png", dpi=300)

    # === Percentages
    df_counts["Percentage"] = (df_counts["FrameCount"] / total_frames) * 100
    df_counts[["Residue", "Percentage"]].to_csv(f"{result_folder}/closest_residue_percentages.csv", index=False)

    # === Bar plot: percentages
    plt.figure(figsize=(10, 6))
    bars_pct = plt.bar(df_counts["Residue"], df_counts["Percentage"])
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Residue", fontsize=14)
    plt.ylabel("Percentage of Total Frames (%)", fontsize=14)
    plt.title("Residues Most Frequently Closest to Any Ion (as % of Frames)", fontsize=16)

    for bar in bars_pct:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{result_folder}/closest_single_residue_percentages_barplot.png", dpi=300)

    print("✅ All plots and CSVs saved successfully.")

def extract_min_mean_distance_pairs(data_by_ion):
    output = {}

    for ion_id, frame_list in data_by_ion.items():
        new_entries = []

        for frame_entry in frame_list:
            frame = frame_entry["frame"]
            residues = frame_entry["residues"]

            # Filter out "SF" and convert keys to int
            filtered = {int(k): v for k, v in residues.items() if k != "SF"}

            # Sort residues numerically
            sorted_residues = sorted(filtered.items())  # list of (resid, dist)

            # Break into consecutive pairs: (r1, r2), (r3, r4), ...
            pairs = [
                (sorted_residues[i], sorted_residues[i + 1])
                for i in range(0, len(sorted_residues) - 1, 2)
            ]

            # Compute mean distances
            pair_mean_distances = [
                {
                    "residue1": pair[0][0],
                    "residue2": pair[1][0],
                    "mean_distance": (pair[0][1] + pair[1][1]) / 2,
                    "frame": frame
                }
                for pair in pairs
            ]

            # Select pair with smallest mean distance
            if pair_mean_distances:
                best_pair = min(pair_mean_distances, key=lambda x: x["mean_distance"])
                new_entries.append(best_pair)

        output[ion_id] = new_entries

    return output



import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def count_frames_pair_closest(data, result_folder, total_frames, channel_type="G2"):
    pair_to_frames = defaultdict(set)

    for ion_id, frame_list in data.items():
        for entry in frame_list:
            frame = entry["frame"]
            res1 = entry["residue1"]
            res2 = entry["residue2"]

            # Convert both residues to PDB numbering
            pdb_res1 = convert_to_pdb_numbering(res1, channel_type)
            pdb_res2 = convert_to_pdb_numbering(res2, channel_type)

            # Create an ordered label for the pair
            pair_label = f"({min(pdb_res1, pdb_res2)}, {max(pdb_res1, pdb_res2)})"
            pair_to_frames[pair_label].add(frame)

    # === Count occurrences
    pair_frame_counts = {pair: len(frames) for pair, frames in pair_to_frames.items()}

    # === DataFrame of counts
    df_counts = pd.DataFrame({
        "ResiduePair": list(pair_frame_counts.keys()),
        "FrameCount": list(pair_frame_counts.values())
    })

    df_counts = df_counts.sort_values(by="FrameCount", ascending=False)
    df_counts.to_csv(f"{result_folder}/closest_residue_pair_counts.csv", index=False)

    # === Bar plot: frame counts
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_counts["ResiduePair"], df_counts["FrameCount"], width=0.5)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Residue Pair", fontsize=14)
    plt.ylabel("Number of Frames Appearing as Closest", fontsize=14)
    plt.title("Residue Pairs Most Frequently Closest to Any Ion (Per Frame)", fontsize=16)

    for bar, count in zip(bars, df_counts["FrameCount"]):
        height = bar.get_height()
        percentage = (count / total_frames) * 100
        label = f"{int(count)} ({percentage:.1f}%)"
        plt.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                label, ha='center', va='bottom', fontsize=11)


    plt.tight_layout()
    plt.savefig(f"{result_folder}/closest_residue_pair_counts_barplot.png", dpi=300)

    # === Percentages
    df_counts["Percentage"] = (df_counts["FrameCount"] / total_frames) * 100
    df_counts[["ResiduePair", "Percentage"]].to_csv(f"{result_folder}/closest_residue_pair_percentages.csv", index=False)

    # === Bar plot: percentages
    plt.figure(figsize=(10, 6))
    bars_pct = plt.bar(df_counts["ResiduePair"], df_counts["Percentage"], width=0.5)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Residue Pair", fontsize=14)
    plt.ylabel("Percentage of Total Frames (%)", fontsize=14)
    plt.title("Residue Pairs Most Frequently Closest to Any Ion (as % of Frames)", fontsize=16)

    for bar in bars_pct:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{result_folder}/closest_residue_pair_percentages_barplot.png", dpi=300)

    print("✅ Pair-based plots and CSVs saved successfully.")

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def plot_start_frame_residue_distribution(closest_dict, interval_list, result_folder, channel_type="G2"):
    residue_counts = defaultdict(int)

    for entry in interval_list:
        ion_id = entry["ion_id"]
        start_frame = int(entry["start_frame"])
        if ion_id == 2299:
            print(ion_id,start_frame)

        frames = closest_dict.get(ion_id, [])
        match = next((item for item in frames if int(item["frame"]) == int(start_frame)), None)

        if match is None:
            print(f"⚠️ No match for ion {ion_id} at or after frame {start_frame}")
            continue

        residue = match["residue"]

        if isinstance(residue, str) and residue.upper() == "SF":
            print(f"⚠️ Ion {ion_id} starts at residue 'SF' — skipping")
            continue

        pdb_res = str(convert_to_pdb_numbering(residue, channel_type))
        residue_counts[pdb_res] += 1

    total_ions = len(interval_list)
    df = pd.DataFrame({
        "Residue": list(residue_counts.keys()),
        "StartCount": list(residue_counts.values())
    })
    df["Percentage"] = (df["StartCount"] / total_ions) * 100
    df = df.sort_values(by="StartCount", ascending=False)

    df.to_csv(f"{result_folder}/start_frame_residue_counts.csv", index=False)

    # === Barplot 1: Raw Counts
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Residue"], df["StartCount"], width=0.6)
    plt.xticks(rotation=45)
    plt.xlabel("Residue")
    plt.ylabel("Number of Ions Starting Closest to Residue")
    plt.title("Start Frame Closest Residue per Ion (Raw Counts)")

    total_ions = df["StartCount"].sum()

    for bar, count in zip(bars, df["StartCount"]):
        height = bar.get_height()
        percentage = (count / total_ions) * 100
        label = f"{int(count)} ({percentage:.1f}%)"
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{result_folder}/start_frame_residue_counts_barplot.png", dpi=300)

    # === Barplot 2: Percentages
    plt.figure(figsize=(10, 6))
    bars_pct = plt.bar(df["Residue"], df["Percentage"], width=0.6)
    plt.xticks(rotation=45)
    plt.xlabel("Residue (PDB numbering)")
    plt.ylabel("Percentage of Ions Starting Closest (%)")
    plt.title("Start Frame Closest Residue per Ion (Percentage)")

    for bar, count, pct in zip(bars_pct, df["StartCount"], df["Percentage"]):
        label = f"{pct:.1f}% ({count})"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{result_folder}/start_frame_residue_percentage_barplot.png", dpi=300)

    print("✅ CSV, raw count plot, and percentage plot saved successfully.")
