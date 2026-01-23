import json
from typing import Dict, List, Any
from collections import Counter
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from typing import List, Dict, Any
from analysis.converter import convert_to_pdb_numbering

"""
analyze_ch2_permeation.py

This script identifies which ions are present and what residues they are closest to
at the exact frame where another ion permeates (i.e., exits the channel).
"""

def get_residues_at_frame(min_results_per_frame, target_frame, pdb_format, channel_type):
    """
    Returns a dictionary of ions and their closest residue at a specific frame.
    Only includes ions that have data at that frame.

    Example:
    {
        2433: 130,
        1313: 780,
        ...
    }
    """
    residues_at_frame = {}

    for ion_id, entries in min_results_per_frame.items():
        for entry in entries:
            if entry.get("frame") == target_frame:
                if pdb_format:
                    residues_at_frame[(ion_id)] = convert_to_pdb_numbering(entry["residue"], channel_type)
                else:
                    residues_at_frame[(ion_id)] = entry["residue"]
                break

    return residues_at_frame

def find_smallest_start_frame(events, target_ion_id):
    """
    Find the smallest start_frame for a given ion_id.

    Args:
        events (list of dict): List of events, each with 'ion_id', 'start_frame', etc.
        target_ion_id (int): The ion ID to search for.

    Returns:
        int or None: The smallest start_frame if found, otherwise None.
    """
    start_frames = [event["start_frame"] for event in events if event["ion_id"] == target_ion_id]
    if not start_frames:
        return None  # Ion ID not found
    return min(start_frames)


def analyze_ch2_permation_residues(min_results_per_frame, ch2_permeations, end_frame, channel_type):
    """
    For each ion, find the frame where it permeates (i.e., its last frame).
    Then collect the residues of all other ions present at that same frame.

    Returns a list of dictionaries like:
    {
        "frame": 3627,
        "ions": {1313: 780, ...},
        "permeated": 2433
    }
    """
    ch2_permation_residues = []
    ch2_permation_residues_pdb = []

    for ion_id, entries in min_results_per_frame.items():
        if not entries:
            continue

        # The last frame is considered the permeation frame
        ion_last_frame = max(entry["frame"] for entry in entries if "frame" in entry)

        # # Exclude ions that didn't permeate and remained in the channel 2 until the end frame
        # if ion_last_frame == end_frame-1:
        #     continue
        
        residues_at_frame = get_residues_at_frame(min_results_per_frame, ion_last_frame, False, channel_type)

        ch2_permation_residues.append({
            "start_frame": find_smallest_start_frame(ch2_permeations, ion_id),
            "frame": ion_last_frame,
            "ions": residues_at_frame,
            "permeated": (ion_id)
        })

        residues_at_frame = get_residues_at_frame(min_results_per_frame, ion_last_frame, True, channel_type)
        ch2_permation_residues_pdb.append({
            "start_frame": find_smallest_start_frame(ch2_permeations, ion_id),
            "frame": ion_last_frame,
            "ions": residues_at_frame,
            "permeated": (ion_id)
        })


    return ch2_permation_residues, ch2_permation_residues_pdb

from collections import defaultdict
from typing import List, Dict, Any, Union

def count_residue_combinations_with_duplicates(permeation_events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each permeation event:
    - Sort numeric residues (keep 'SF' at the end)
    - Create a residue key string
    - Track how often each combination appears
    - Record the permeated residues and frames
    """
    summary = defaultdict(lambda: {
        "count": 0,
        "permeated_residues": [],
        "frames": []
    })

    for event in permeation_events:
        residues_raw = list(event["ions"].values())

        numeric_residues = [int(r) for r in residues_raw if str(r).isdigit()]
        non_numeric_residues = [str(r) for r in residues_raw if not str(r).isdigit()]

        sorted_residues = sorted(numeric_residues) + non_numeric_residues
        residue_key = ", ".join(str(r) for r in sorted_residues)

        permeated_ion = event["permeated"]
        permeated_residue = event["ions"].get(permeated_ion)
        frame = event.get("frame")

        if permeated_residue is not None:
            summary[residue_key]["count"] += 1
            summary[residue_key]["permeated_residues"].append(permeated_residue)
            if frame is not None:
                summary[residue_key]["frames"].append(frame)

    return summary


import pandas as pd
from pathlib import Path
from typing import Dict, Any

def save_residue_combination_summary_to_excel(
    data: Dict[str, Dict[str, Any]],
    results_dir: Path,
    filename: str = "ch2_permation_residue_comb.xlsx"
):
    """
    Saves the residue combination summary to an Excel file.
    Each row includes:
    - Residue combination (as string)
    - Count
    - Permeated residues (comma-separated string)
    - Permeation frames (comma-separated string)
    """
    records = []

    for residue_key, info in data.items():
        record = {
            "Residue Combination": residue_key,
            "Count": info.get("count", 0),
            "Permeated Residues": ", ".join(str(r) for r in info.get("permeated_residues", [])),
            "Frames": ", ".join(str(f) for f in info.get("frames", []))
        }
        records.append(record)

    df = pd.DataFrame(records)
    output_path = results_dir / filename
    df.to_excel(output_path, index=False)
    print(f"✅ Excel file saved to: {output_path}")



from collections import Counter
from typing import List, Dict, Any

def count_last_residues(permeation_events: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Counts how many times each residue appears as the last residue 
    before an ion permeates.
    """
    counter = Counter()

    for event in permeation_events:
        permeated_ion = event["permeated"]
        last_residue = event["ions"].get(str(permeated_ion)) or event["ions"].get(int(permeated_ion))

        if last_residue is not None:
            counter[str(last_residue)] += 1  # convert to string for consistent plotting

    return dict(counter)

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def plot_last_residue_bar_chart(residue_counts: Dict[str, int], results_dir: Path, filename: str = "last_residues_barplot.png"):
    """
    Plots and saves a bar chart of how often each residue was the last before permeation.
    Bars are sorted by count (descending), with larger fonts for axes and labels.
    """
    # Sort residues by count descending
    sorted_items = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
    residues = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Plot setup
    plt.figure(figsize=(12, 7))
    bars = plt.bar(residues, counts, color="steelblue")

    # Set bigger fonts
    plt.xlabel("Residue", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.title("Last Residue Before Permeation", fontsize=20)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)

    # Annotate counts above bars with bigger font
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{int(height)}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=14)

    plt.tight_layout()

    # Save figure
    output_path = results_dir / filename
    plt.savefig(output_path)
    print(f"✅ Plot saved to: {output_path}")
    plt.close()

def find_all_pre_permeation_patterns(permeation_events, residue_track_dict):
    """
    For each permeation event, checks whether the full residue pattern at the permeation frame
    exists in any earlier frame (from start_frame to frame - 1).

    Args:
        permeation_events: list of dicts with keys 'start_frame', 'frame', 'ions', 'permeated'
        residue_track_dict: dict of ion_id -> list of dicts with 'frame', 'residue'

    Returns:
        List of results:
            [
                {
                    'permeated': ion_id,
                    'pattern': {ion_id: residue_id, ...},
                    'match_frames': [list of frames where pattern matched]
                },
                ...
            ]
    """
    results = []

    for event in permeation_events:
        start_frame = event["start_frame"]
        permeation_frame = event["frame"]
        ion_residue_pattern = event["ions"]  # e.g., {"2433": 130, "1313": 780}

        match_frames = []

        for frame in range(start_frame, permeation_frame):
            all_match = True
            for ion_id, expected_residue in ion_residue_pattern.items():
                # print(f"Checking ion {ion_id} at frame {frame}")
                ion_history = residue_track_dict.get(int(ion_id), [])
                # print(ion_history)
                this_frame_entry = next((d for d in ion_history if d["frame"] == frame), None)
                if this_frame_entry is None or this_frame_entry["residue"] != expected_residue:
                    all_match = False
                    break
            if all_match:
                match_frames.append(frame)

        results.append({
            "permeated": event["permeated"],
            "start_frame": start_frame,
            "permeation_frame": permeation_frame,
            "pattern": ion_residue_pattern,
            "match_frames": match_frames
        })

    return results
