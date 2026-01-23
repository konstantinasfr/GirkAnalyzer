import numpy as np
import json
from analysis.converter import convert_to_pdb_numbering

def cluster_frames_by_closest_residue(distance_data):
    clustered_results = {}
    min_results_per_frame = {}
    close_contacts_dict = {}

    for ion_id, frame_list in distance_data.items():
        clusters = []
        prev_residue = None
        start_frame = None
        distances = []
        close_contacts_dict[ion_id] = {}
        min_results_per_frame[ion_id] = []
        not_sf_starting = False

        SF_only = True
        for frame_data in frame_list[:-1]:
            frame = frame_data["frame"]
            residues = frame_data["residues"]
            closest_residue, closest_distance = min(residues.items(), key=lambda item: item[1])
            if closest_residue != "SF" :
                SF_only = False
                break

        for frame_data in frame_list[:-1]:
            frame = frame_data["frame"]
            residues = frame_data["residues"]

            # Find the closest residue (key with smallest value)
            closest_residue, closest_distance = min(residues.items(), key=lambda item: item[1])

            if closest_residue != "SF" or not_sf_starting or SF_only:
                min_results_per_frame[ion_id].append({
                    "frame":frame,
                    "residue":closest_residue,
                    "min_distance":closest_distance
                })
                not_sf_starting = True

                ######### find close contact residues, filter residues with distance < 6 #####################################3
                sorted_close_contacts = dict(
                                                sorted(
                                                    {resid: dist for resid, dist in residues.items() if dist < 6}.items(),
                                                    key=lambda item: item[1]
                                                )
                                            )
                close_residues = []
                no_close_contacts = True
                for resid, dist in sorted_close_contacts.items():
                    no_close_contacts = False
                    if resid == "SF":
                        # if closest is SF then we consider that ion is still in SF
                        if not close_residues:
                            close_residues.append(resid)
                            break
                        else:
                            continue
                    else:
                        close_residues.append(resid)

                if no_close_contacts:
                    close_residues.append("no_close_residues")

                close_contacts_dict[ion_id][frame] = close_residues

                ################################## Cluster creation per residue ########################
                if closest_residue != prev_residue:
                    # If ending a previous cluster, store it
                    if prev_residue is not None:
                        clusters.append({
                            "residue": prev_residue,
                            "start": start_frame,
                            "end": prev_frame,
                            "frames": prev_frame - start_frame + 1,
                            "mean_distance": sum(distances) / len(distances)
                        })
                    # Start a new cluster
                    start_frame = frame
                    prev_residue = closest_residue
                    distances = [closest_distance]
                else:
                    distances.append(closest_distance)

                prev_frame = frame

        # Add final cluster
        if prev_residue is not None:
            clusters.append({
                "residue": prev_residue,
                "start": start_frame,
                "end": prev_frame,
                "frames": prev_frame - start_frame + 1,
                "mean_distance": sum(distances) / len(distances)
            })

        clustered_results[ion_id] = clusters

    return clustered_results, min_results_per_frame, close_contacts_dict

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_and_plot_average_ions_per_residue(data, output_folder):
    # with open(json_path, "r") as f:
    #     data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    residue_frame_counts = defaultdict(lambda: defaultdict(int))
    all_frames = set()

    for ion, frame_dict in data.items():
        for frame_str, residues in frame_dict.items():
            frame = int(frame_str)
            all_frames.add(frame)
            for res in residues:
                if isinstance(res, int):  # skip "SF" etc
                    residue_frame_counts[res][frame] += 1

    total_frames = len(all_frames)
    avg_per_residue = {}
    median_per_residue = {}
    full_distributions = {}

    for res, frame_counts in residue_frame_counts.items():
        frame_vals = [frame_counts.get(f, 0) for f in all_frames]
        avg_per_residue[res] = np.mean(frame_vals)
        median_per_residue[res] = np.median(frame_vals)
        full_distributions[res] = frame_vals

    # Sort residues by ID for consistent plotting
    sorted_residues = sorted(avg_per_residue.keys())

    # --- BAR PLOT: AVERAGE AND MEDIAN ---
    x = np.arange(len(sorted_residues))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, [avg_per_residue[r] for r in sorted_residues], width, label='Average', color='skyblue')
    ax.bar(x + width/2, [median_per_residue[r] for r in sorted_residues], width, label='Median', color='lightgreen')

    ax.set_xlabel('Residue ID', fontsize=16)
    ax.set_ylabel('Number of Ions per Frame', fontsize=16)
    ax.set_title('Average and Median Ions per Frame per Residue', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_residues, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "avg_median_ions_per_residue.png"))
    plt.close()

    # --- BOXPLOT ---
    fig, ax = plt.subplots(figsize=(14, 6))
    data_to_plot = [full_distributions[r] for r in sorted_residues]
    x_positions = np.arange(len(sorted_residues))

    # Original boxplot with gray fill
    ax.boxplot(
        data_to_plot,
        positions=x_positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black"),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='red')
    )

    # # Overlay all values as blue dots (with jitter)
    # for i, y_vals in enumerate(data_to_plot):
    #     x_vals = np.random.normal(loc=x_positions[i], scale=0.08, size=len(y_vals))
    #     ax.scatter(x_vals, y_vals, color='blue', alpha=0.6, s=10, zorder=3)

    # Labels and formatting
    ax.set_xlabel('Residue ID', fontsize=16)
    ax.set_ylabel('Ion numbers', fontsize=16)
    ax.set_title('Ions close to residues per Frame', fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_residues, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "ion_count_boxplot_per_residue.png"))
    plt.close()


    return avg_per_residue, median_per_residue



import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def close_contact_residues_analysis(data, main_path, channel_type, max_bar_number=20):
    """
    For each ion, plots and saves a bar chart of residue combinations (unordered)
    that are close during trajectory frames, and writes full CSV summary.

    Parameters:
        data (dict): ion_id → frame_id → list of close residues
        results_dir (str): directory where two subfolders will be created:
                           - close_contact_residues/plots
                           - close_contact_residues/csv
        max_bar_number (int): max number of bars in each plot
    """

    main_path = os.path.join(main_path, "")
    plot_dir = os.path.join(main_path, "plots")
    csv_dir = os.path.join(main_path, "csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    total_residue_comb_over_all_frames = {}

    def normalize_combo(combo):
        return tuple(sorted(combo))

    total_combo_counts = Counter()

    for ion_id, frames in data.items():
        combo_counts = Counter()

        for frame, residues in frames.items():
            if frame not in total_residue_comb_over_all_frames:
                total_residue_comb_over_all_frames[frame] = {}
            if residues != ["SF"] and residues != ["no_close_residues"]:
                norm_combo = normalize_combo(residues)
                combo_counts[norm_combo] += 1
                total_combo_counts[norm_combo] += 1
                total_residue_comb_over_all_frames[frame][ion_id] = '_'.join(map(lambda r: convert_to_pdb_numbering(int(r), channel_type), norm_combo))

        if not combo_counts:
            continue

        combo_data = [{"residue_combination": '_'.join(map(lambda r: convert_to_pdb_numbering(int(r), channel_type), combo)), "count": count}
                      for combo, count in combo_counts.items()]
        df = pd.DataFrame(combo_data).sort_values(by="count", ascending=False)
        csv_path = os.path.join(csv_dir, f"{ion_id}.csv")
        df.to_csv(csv_path, index=False)

        top_combos = combo_counts.most_common(max_bar_number)
        labels = ['_'.join(map(lambda r: convert_to_pdb_numbering(r, channel_type), combo)) for combo, _ in top_combos]
        counts = [count for _, count in top_combos]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, counts)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.title(f"Ion {ion_id} — Top {max_bar_number} combos", fontsize=16)
        plt.xlabel("Residue combination", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"{ion_id}.png")
        plt.savefig(plot_path)
        plt.close()

        # print(f"✅ Ion {ion_id}: plot saved to {plot_path}, data to {csv_path}")

    # Global summary across all ions
    if total_combo_counts:
        total_combo_data = [{"residue_combination": '_'.join(map(lambda r: convert_to_pdb_numbering(r, channel_type), combo)), "count": count}
                            for combo, count in total_combo_counts.items()]
        df_total = pd.DataFrame(total_combo_data).sort_values(by="count", ascending=False)
        total_csv_path = os.path.join(main_path, "ALL_ions_combined.csv")
        df_total.to_csv(total_csv_path, index=False)

        top_total_combos = total_combo_counts.most_common(max_bar_number)
        labels = ['_'.join(map(lambda r: convert_to_pdb_numbering(r, channel_type), combo)) for combo, _ in top_total_combos]
        counts = [count for _, count in top_total_combos]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, counts)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.title(f"All Ions — Top {max_bar_number} Residue Combinations", fontsize=16)
        plt.xlabel("Residue combination", fontsize=14)
        plt.ylabel("Total Frequency", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.xticks(fontsize=12)
        plt.tight_layout()

        total_plot_path = os.path.join(main_path, "ALL_ions_combined.png")
        plt.savefig(total_plot_path)
        plt.close()

        print(f"📊 Combined plot saved to {total_plot_path}, data to {total_csv_path}")

    return total_residue_comb_over_all_frames
        



def get_ions_for_frame(data, target_frame):
    for frame_info in data:
        if frame_info["frame"] == target_frame:
            return frame_info["ions"]
    return None  # If not found

def tracking_ion_distances(
        permeation_events,        # list of dicts: {frame, ions, permeated}
        distances,               # list of dicts: {frame, ions: {ion_id: distance}, ...}
        ch2_entry_exit_list       # list of dicts: {ion_id, start_frame, exit_frame}
    ):
    from collections import defaultdict

    results = {}

    # Convert a flat list of CH2 entry/exit records into a dictionary grouped by ion_id
    ch2_entry_exit_dict = defaultdict(list)
    for entry in ch2_entry_exit_list:
        ion_id = (entry['ion_id'])
        ch2_entry_exit_dict[ion_id].append({
            'start_frame': entry['start_frame'],
            'exit_frame': entry['exit_frame']
        })

    # Keep only the latest event per ion
    latest_permeation_bounds = {
        ion_id: sorted(ranges, key=lambda x: x['exit_frame'], reverse=True)[0]
        for ion_id, ranges in ch2_entry_exit_dict.items()
    }

    for event in permeation_events:

        target_ion = (event['permeated'])
        results[target_ion] = []
        end_frame = event['frame']
        start_frame = latest_permeation_bounds[target_ion]['start_frame']

        # for ions_in_ch2 in event['ions']:
        #     for ion_ion_distance in distances[target_ion]["ions"]:
        #         if ion_ion_distance

        for f in range(start_frame, end_frame + 1):
            ion_ion_dist = get_ions_for_frame(distances[target_ion], f)
            proximate_ions = {}
            for ion_in_ch2 in event["ions"]:
                if ion_in_ch2 != target_ion:
                    try:
                        proximate_ions[(ion_in_ch2)] = ion_ion_dist[ion_in_ch2]
                    except KeyError:
                        continue
            results[target_ion].append({
                "frame": f,
                "distances": proximate_ions
            })

    return results


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ion_distance_traces(distance_data, results_dir):
    """
    Takes a dictionary of distance traces and saves two line plots per target ion:
    1. Full timeline of distances
    2. Last 15 frames before permeation (showing true frame numbers on x-axis)

    Parameters:
    - distance_data: dict
        {
            "2433": [
                {"frame": 3185, "distances": {"1313": 6.89, "1460": 9.15}},
                {"frame": 3186, "distances": {"1313": 8.06, "1460": 9.32}},
                ...
            ],
            ...
        }
    - results_dir: Path or str
        Base directory where plots should be saved.
    """
    # Ensure output folders exist
    results_dir = Path(results_dir)
    full_dir = results_dir / "ion_distances"
    last15_dir = results_dir / "ion_distances_last15"
    full_dir.mkdir(parents=True, exist_ok=True)
    last15_dir.mkdir(parents=True, exist_ok=True)

    # Flatten the nested dictionary into a long-format DataFrame
    records = []
    for target_ion, frames in distance_data.items():
        for entry in frames:
            frame = entry["frame"]
            for other_ion, distance in entry["distances"].items():
                records.append({
                    "target_ion": (target_ion),
                    "other_ion": (other_ion),
                    "frame": int(frame),
                    "distance": distance
                })

    df = pd.DataFrame(records)
    # Create and save one plot per target ion (full timeline + last 15 frames per other ion)
    for target_ion in df["target_ion"].unique():
        subset = df[df["target_ion"] == target_ion].sort_values("frame")

        # === Full plot ===
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="frame", y="distance", hue="other_ion", marker="o", palette="tab10")
        plt.title(f"Distances from Ion {target_ion} to Other Ions (Full Timeline)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Distance", fontsize=14)
        plt.legend(title="Other Ion", fontsize=12)
        plt.xticks(subset["frame"].unique(), fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(full_dir / f"{target_ion}.png")
        plt.close()

        # === Last 15 frame plot per other_ion, then merge ===
        last_15_frames = (
            subset.groupby("other_ion", group_keys=False)
            .apply(lambda x: x.sort_values("frame").tail(15))
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=last_15_frames, x="frame", y="distance", hue="other_ion", marker="o", palette="tab10")
        plt.title(f"Distances from Ion {target_ion} (Last 15 Frames per Ion Before Permeation)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Distance", fontsize=14)
        plt.legend(title="Other Ion", fontsize=12)
        plt.xticks(last_15_frames["frame"].unique(), fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(last15_dir / f"{target_ion}.png")
        plt.close()

import json
import matplotlib.pyplot as plt
from collections import Counter
from collections import Counter
import matplotlib.pyplot as plt

def plot_ion_channel_frequencies(data, folder="./"):
    """
    Plots a histogram of the number of ions detected near GLU/ASN gate residues during permeation.
    Each bar is annotated with frequency and percentage.
    """
    # Extract only valid 'number_of_ions_in_channel' entries
    ion_counts = [
        entry.get("number_of_ions_in_channel")
        for entry in data.values()
        if entry.get("number_of_ions_in_channel") is not None
    ]

    # Count frequencies
    count_freq = Counter(ion_counts)

    # Prepare data for plotting
    labels = sorted(count_freq.keys())
    frequencies = [count_freq[label] for label in labels]
    total = sum(frequencies)

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, frequencies)

    # Add annotated labels (value + percentage)
    for bar, freq in zip(bars, frequencies):
        percent = (freq / total) * 100
        label = f"{freq} ({percent:.1f}%)"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
                 ha='center', va='bottom', fontsize=14)

    # Axis formatting
    plt.xlabel("Number of ions within proximity to GLU/ASN", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Ion count near GLU/ASN during >6 Å displacement from ASN", fontsize=18)
    plt.xticks(labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    output_path = f"{folder}/number_of_ions_within_proximity_to_GLU_ASN.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()


import json
import matplotlib.pyplot as plt
from collections import Counter

def plot_ion_channel_percentages(data, folder="./"):
    """
    Reads a JSON file of ion permeation data and plots the percentage
    of 'number_of_ions_in_channel' values as a bar chart.

    Parameters:
    - json_path: str, path to the JSON file
    """

    # Extract values
    ion_counts = [entry["number_of_ions_in_channel"] for entry in data.values()]
    total = len(ion_counts)

    # Count frequencies
    count_freq = Counter(ion_counts)

    # Sort and convert to percentage
    labels = sorted(count_freq.keys())
    percentages = [(count_freq[label] / total) * 100 for label in labels]

    # Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, percentages)

    # Add text on top of each bar
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct:.0f}%",
                 ha='center', va='bottom', fontsize=12)

    
    plt.xlabel("Number of ions within proximity to GLU/ASN", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.title("Ion count near GLU/ASN during >6 Å displacement from ASN", fontsize=16)
    plt.xticks(labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, max(percentages) * 1.15)  # add some headroom
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = f"{folder}/percent_number_of_ions_within_proximity_to_GLU_ASN.png"
        # Save figure
    plt.savefig(output_path, dpi=300)


def summarize_coexistence_blocks(df):
    """
    Takes a DataFrame of coexistence blocks and returns:
    - ion_count
    - num_states
    - total_frames
    - percent_time
    - mean_frames
    """
    # Compute total simulation time from first start to last end
    total_simulation_frames = df["end"].max() - df["start"].min() + 1

    summary = (
        df.groupby("num_ions")
          .agg(
              num_states=("num_ions", "count"),
              total_frames=("duration", "sum"),
              mean_frames=("duration", "mean")
          )
          .reset_index()
          .rename(columns={"num_ions": "ion_count"})
    )

    summary["percent_time"] = (summary["total_frames"] / total_simulation_frames * 100).round(2)
    summary["mean_frames"] = summary["mean_frames"].round(2)
    return summary

import matplotlib.pyplot as plt

def plot_total_frames_by_ion_count(df, folder="./"):
    """
    Plots a bar chart of total_frames vs. ion_count with annotated values.
    Each bar shows the total number of frames and the % of total.
    """
    plt.figure(figsize=(10, 6))
    x = df["ion_count"].astype(int).tolist()
    y = df["total_frames"].tolist()

    total = sum(y)
    bars = plt.bar(x, y, width=0.6)

    # Add labels with count and frequency
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percent = (height / total) * 100
        label = f"{height} ({percent:.1f}%)"
        plt.text(bar.get_x() + bar.get_width() / 2, height, label,
                 ha='center', va='bottom', fontsize=14)

    # Axis and tick settings
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of ions near GLU/ASN residues", fontsize=16)
    plt.ylabel("Number of frames", fontsize=16)
    plt.title("Frames with N ions near GLU/ASN gate residues", fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = f"{folder}/total_frames_by_ion_count.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

import matplotlib.pyplot as plt

def plot_percent_time_by_ion_count(df, folder="./"):
    """
    Plots a bar chart of percent_time vs. ion_count with values annotated.
    """
    plt.figure(figsize=(8, 6))
    x = df["ion_count"].astype(int).tolist()
    y = df["percent_time"].tolist()  # or df["percent_time"].tolist()
    bars = plt.bar(x, y, width=0.6)


    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}%",
                 ha='center', va='bottom', fontsize=12)

    plt.xticks(x)
    plt.xlabel("Number of ions near GLU/ASN residues")
    plt.ylabel("Percentage of trajectory time (%)")
    plt.title("Proportion of trajectory spent with N ions near GLU/ASN")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = f"{folder}/percent_time_by_ion_count.png"
    plt.savefig(output_path, dpi=300)


# def parse_ion_string(ion_str):
#     """
#     Converts a comma-separated string of ion IDs into a list of integers.
#     Example: '2400, 2276, 2397, 2399' → [2400, 2276, 2397, 2399]
#     """
#     return [int(x.strip()) for x in ion_str.split(",") if x.strip().isdigit()]

def parse_ion_string(ion_str):
    """
    Converts a comma-separated string of ion IDs (e.g., '2400_1, 2276_1')
    into a list of strings like ['2400_1', '2276_1'].
    """
    return [x.strip() for x in ion_str.split(",") if x.strip()]


def get_clean_ion_coexistence_table(ion_events, end_frame, folder="./"):
    """
    Creates non-overlapping frame blocks where ions coexisted.
    Each block ends exactly where the next begins.
    """
    # Step 1: Build frame-by-frame ion presence
    timeline = {}
    permeation_frames_ion_coexistence = {}
    
    for ion in ion_events:
        ion_id = ion["ion_id"]
        exit_frame = ion["exit_frame"]
        # if ion_id == "2304_3":
        #     print("LALA")
        #     print(end_frame)
        #     print(exit_frame)
    
        # Build frame-by-frame presence
        for frame in range(ion["start_frame"], exit_frame + 1):
            timeline.setdefault(frame, set()).add(ion_id)
    
        # Handle permeation frame assignment
        if ion_id not in permeation_frames_ion_coexistence:
            permeation_frames_ion_coexistence[ion_id] = {
                "permeation_frame": exit_frame,
                "number_of_ions_in_channel": [],
                "ions_in_channel": 0
            }
        else:
            if permeation_frames_ion_coexistence[ion_id]["permeation_frame"] < exit_frame:
                permeation_frames_ion_coexistence[ion_id]["permeation_frame"] = exit_frame

    # Remove ions that permeated exactly at the end frame
    permeation_frames_ion_coexistence = {
        ion_id: data
        for ion_id, data in permeation_frames_ion_coexistence.items()
        if data["permeation_frame"] < end_frame-1  # Exclude ions that permeated at the last frame, -1 because in ch2_fixed we store the last frame in the region
        # not the exit frame from the region
}



    # Step 2: Sort frames and chunk by unique ion sets
    frames = sorted(timeline.keys())
    result = []
    prev_ions = None
    start = frames[0]
    for i, frame in enumerate(frames):
        current_ions = timeline[frame]
        if prev_ions is None:
            prev_ions = current_ions
            continue
        if current_ions != prev_ions:
            end = frames[i - 1]
            result.append({
                "start": start,
                "end": end,
                "duration": end - start + 1,
                "num_ions": len(prev_ions),
                "ions": (prev_ions)
            })
            start = frame
            prev_ions = current_ions
    # Handle final block
    end = frames[-1]
    result.append({
        "start": start,
        "end": end,
        "duration": end - start + 1,
        "num_ions": len(prev_ions),
        "ions": (prev_ions)
    })

    df = pd.DataFrame(result)
    df["ions"] = df["ions"].apply(lambda x: ", ".join(map(str, x)))  # Clean formatting

    for ion_id, ion_perm_event in permeation_frames_ion_coexistence.items():
        permation_frame = ion_perm_event['permeation_frame']
        # coexisting_ions = df[df["end"] == permation_frame]["ions"].values[0]
        match = df[df["end"] == permation_frame]

        if not match.empty:
            coexisting_ions = match["ions"].values[0]
            # proceed with analysis
        else:
            print(f"[Skip] No match for frame {permation_frame}, skipping.")
            continue  # or pass, depending on your loop logic

        coexisting_ions_list = parse_ion_string(coexisting_ions)
        permeation_frames_ion_coexistence[ion_id]["ions_in_channel"] = coexisting_ions_list
        permeation_frames_ion_coexistence[ion_id]["number_of_ions_in_channel"] = len(coexisting_ions_list)
    

    df.to_csv(f"{folder}/ion_coexistence.csv", index=False)
    summary = summarize_coexistence_blocks(df)
    summary.to_csv(f"{folder}/ion_coexistence_summary.csv", index=False)
    plot_total_frames_by_ion_count(summary, folder)
    plot_percent_time_by_ion_count(summary, folder)

    with open(f"{folder}/permeation_frames_ion_coexistence.json", "w") as f:
        json.dump(permeation_frames_ion_coexistence, f, indent=2)

    plot_ion_channel_frequencies(permeation_frames_ion_coexistence, folder)
    plot_ion_channel_percentages(permeation_frames_ion_coexistence, folder)
