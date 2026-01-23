import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def analyze_resid_changes_and_plot(data, output_folder, permeation_events2, filename="residues_permeating_sf.png"):
    # Sort keys as integers for sequential frame order
    sorted_keys = sorted(data.keys(), key=int)
    previous_resid = None
    resid_changes = 0
    resid_set = set()

    for key in sorted_keys:
        current_resid = data[key]["resid"]
        resid_set.add(current_resid)

        if previous_resid is not None and current_resid != previous_resid:
            resid_changes += 1

        previous_resid = current_resid

    unique_resid_count = len(permeation_events2)

    # Plotting
    labels = ['Residues Changes', 'Permeating Residues']
    values = [resid_changes, unique_resid_count]

    fig, ax = plt.subplots(figsize=(4, 6))  # Tall and narrow
    bar_width = 0.3  # Slim bars
    x = range(len(values))

    bars = ax.bar(x, values, width=bar_width, color=['skyblue', 'lightgreen'])

    # Add number labels on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, int(yval),
                ha='center', va='bottom', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('Residue Changes in SF end vs Residues leaving SF', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    plt.close()

    # return resid_changes, unique_resid_count, output_path

# import json
# import os
# import matplotlib.pyplot as plt

# def plot_field_histograms_from_json(json_path, output_folder):
#     target_keys = [
#         "total_no_ions", "total_asn_glu", "total_pip",
#         "total", "total_asn", "total_glu"
#     ]

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     magnitudes = {key: [] for key in target_keys}
#     axials = {key: [] for key in target_keys}

#     # Collect data
#     for frame_data in data.values():
#         for key in target_keys:
#             if key in frame_data:
#                 magnitudes[key].append(frame_data[key]["magnitude"])
#                 axials[key].append(frame_data[key]["axial"])

#     os.makedirs(output_folder, exist_ok=True)
#     plot_paths = []

#     for key in target_keys:
#         # Magnitude plot
#         plt.figure(figsize=(6, 4))
#         plt.hist(magnitudes[key], bins=100, color="skyblue", edgecolor='black')
#         plt.title(f"{key} - Magnitude Distribution")
#         plt.xlabel("Magnitude")
#         plt.ylabel("Frequency")
#         plt.tight_layout()
#         mag_path = os.path.join(output_folder, f"{key}_magnitude.png")
#         plt.savefig(mag_path)
#         plt.close()
#         plot_paths.append(mag_path)

#         # Axial plot
#         plt.figure(figsize=(6, 4))
#         plt.hist(axials[key], bins=100, color="salmon", edgecolor='black')
#         plt.title(f"{key} - Axial Field Distribution")
#         plt.xlabel("Axial (Z-component)")
#         plt.ylabel("Frequency")
#         plt.tight_layout()
#         axial_path = os.path.join(output_folder, f"{key}_axial.png")
#         plt.savefig(axial_path)
#         plt.close()
#         plot_paths.append(axial_path)

#     return plot_paths


import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_field_histograms_from_json(json_path, output_folder):
    import json
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    target_keys = [
        "total_no_ions", "total_asn_glu", "total_pip",
        "total", "total_asn", "total_glu"
    ]

    with open(json_path, "r") as f:
        data = json.load(f)

    magnitudes = {key: [] for key in target_keys}
    axials = {key: [] for key in target_keys}

    for frame_data in data.values():
        for key in target_keys:
            if key in frame_data:
                magnitudes[key].append(frame_data[key]["magnitude"])
                axials[key].append(frame_data[key]["axial"])

    os.makedirs(output_folder/"histograms", exist_ok=True)
    plot_paths = []

    for key in target_keys:
        # Magnitude plot
        plt.figure(figsize=(6, 4))
        plt.hist(magnitudes[key], bins=100, color="skyblue", edgecolor='black')
        plt.title(f"{key} - Magnitude Distribution", fontsize=16)
        plt.xlabel("Magnitude", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        mag_path = os.path.join(output_folder, f"{key}_magnitude.png")
        plt.savefig(mag_path)
        plt.close()
        plot_paths.append(mag_path)

        # Axial plot
        axial_data = np.array(axials[key])
        if key == "total":
            axial_data = axial_data[(axial_data >= -40) & (axial_data <= 40)]

        plt.figure(figsize=(6, 4))
        plt.hist(axial_data, bins=100, color="salmon", edgecolor='black')
        plt.title(f"{key} - Axial Field Distribution", fontsize=16)
        plt.xlabel("Axial (Z-component)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        axial_path = os.path.join(output_folder, f"{key}_axial.png")
        plt.savefig(axial_path)
        plt.close()
        plot_paths.append(axial_path)

    return plot_paths




##################################################################

def extract_permeation_frames_by_mode(data, mode="all"):
    """
    Detects permeation frames using different strategies:
    - "all": all resid changes are counted (frame-by-frame)
    - "last": only the last frame each ion is closest (i.e., just before it's replaced)
    - "first": only the first frame each ion appears (i.e., first entry)

    Parameters:
    - json_path: path to the closest_sf_unentered_ions.json
    - mode: "all", "last", or "first"

    Returns:
    - Set of frame indices (as integers)
    """
    import json

    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    sorted_frames = sorted(int(k) for k in data.keys())
    prev_resid = None

    permeation_frames = set()
    resid_to_frames = {}  # resid → list of frames

    for frame in sorted_frames:
        resid = data[int(frame)]["resid"]

        if mode == "all":
            if prev_resid is not None and resid != prev_resid:
                permeation_frames.add(frame - 1)  # frame where the previous ion exited
            prev_resid = resid

        elif mode in ("last", "first"):
            if resid not in resid_to_frames:
                resid_to_frames[resid] = []
            resid_to_frames[resid].append(frame)

    if mode == "last":
        for frames in resid_to_frames.values():
            permeation_frames.add(frames[-1])  # keep only the last frame for each ion

    elif mode == "first":
        for frames in resid_to_frames.values():
            permeation_frames.add(frames[0])  # keep only the first frame for each ion

    return permeation_frames

def analyze_field_at_permeation(json_path, closest_unentered_ion_to_upper_gate, permation_mode, output_path, key="total", field_component="axial"):
    import json, os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    with open(json_path, 'r') as f:
        data = json.load(f)

    permeation_frames = extract_permeation_frames_by_mode(closest_unentered_ion_to_upper_gate, permation_mode)
    with_permeation = []
    without_permeation = []

    for frame_str, frame_data in data.items():
        frame = int(frame_str)
        if key not in frame_data:
            continue
        value = frame_data[key][field_component]
        if frame in permeation_frames:
            with_permeation.append(value)
        else:
            without_permeation.append(value)

    # Create plot with more space for text
    os.makedirs(output_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))  # Increased figure size
    
    box_plot = ax.boxplot([with_permeation, without_permeation],
                         labels=["Permeation", "No Permeation"],
                         showfliers=False)

    ax.set_ylabel(f"{key} - {field_component.capitalize()} Field", fontsize=14)
    ax.set_title(f"{field_component.capitalize()} Field During vs Not During Permeation", fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    # Stats
    def get_stats(group):
        return {
            "mean": np.mean(group),
            "median": np.median(group),
            # "min": np.min(group),
            # "max": np.max(group),
            # "n": len(group)
        }

    stats_perm = get_stats(with_permeation)
    stats_no = get_stats(without_permeation)
    stat, p_value = mannwhitneyu(with_permeation, without_permeation, alternative='two-sided')

    def format_group_text(stats):
        return (
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}"
            # f"Min: {stats['min']:.2f}\n"
            # f"Max: {stats['max']:.2f}\n"
            # f"N: {stats['n']}"
        )

    # Get current axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Don't extend y-axis since we're using figure coordinates now
    # Keep original y-limits for clean plot area
    
    # Position text below the x-axis line
    # Get the actual bottom of the plot area (where x-axis line is)
    ax_bottom = ax.get_position().y0
    
    # Use figure coordinates to place text below the x-axis
    fig_height = fig.get_figheight()
    
    # Position the boxes very close to the x-axis line
    # Blue box under "Permeation", Green box under "No Permeation"
    stats_box_y = 0.47  # Position for stats boxes (very close to x-axis)
    pvalue_box_y = 0.47  # Position for p-value box (lower)
    
    # Position stats boxes directly under their respective boxplots
    fig.text(0.25, stats_box_y, format_group_text(stats_perm), 
            ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    fig.text(0.75, stats_box_y, format_group_text(stats_no), 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # P-value box centered below both, at a lower position
    fig.text(0.5, pvalue_box_y, f"Mann–Whitney U Test\np = {p_value:.3e}", 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.8))

    # Adjust layout to provide space for text boxes very close to x-axis
    plt.subplots_adjust(bottom=0.55)  # Even more space at bottom for very close positioning
    
    plot_path = os.path.join(output_path, f"{key}_{field_component}_permeation_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_all_field_permeation_analyses(json_path, closest_ion_dict, output_base_path):
    import os

    modes = ["all", "last", "first"]
    target_keys = [
        "total_no_ions", "total_asn_glu", "total_pip",
        "total", "total_asn", "total_glu"
    ]
    components = ["axial", "magnitude"]

    all_results = {}

    for mode in modes:
        mode_path = os.path.join(output_base_path, f"mode_{mode}")
        os.makedirs(mode_path, exist_ok=True)

        for key in target_keys:
            for component in components:
                result = analyze_field_at_permeation(
                    json_path=json_path,
                    closest_unentered_ion_to_upper_gate=closest_ion_dict,
                    permation_mode=mode,
                    output_path=mode_path,
                    key=key,
                    field_component=component
                )
                # Save result in a structured dict
                all_results[(mode, key, component)] = result

    return all_results
