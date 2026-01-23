import json
import numpy as np
import pandas as pd
from collections import defaultdict
from analysis.converter import convert_to_pdb_numbering
import os
import pandas as pd
import dataframe_image as dfi
from tqdm import tqdm

def strip_json_extension(filename):
    name, ext = os.path.splitext(filename)
    return name

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
        
def create_force_dataframe(json_data):
    rows = []

    for frame_str, frame_data in json_data["analysis"].items():
        frame = int(frame_str)

        motion_vec = np.array(frame_data["motion_vector"])
        motion_mag = np.linalg.norm(motion_vec)

        # === RESIDUES: GLU, ASN ===
        for key, default_resname in [
            ("glu_contributions", "GLU"),
            ("asn_contributions", "ASN"),
        ]:
            grouped = defaultdict(list)
            for contrib in frame_data.get(key, []):
                resid = contrib["resid"]
                resname = contrib.get("resname", default_resname)
                grouped[(resid, resname)].append(np.array(contrib["force"]))

            for (resid, resname), atom_forces in grouped.items():
                total = np.sum(atom_forces, axis=0)
                mag = np.linalg.norm(total)
                cosine = np.dot(total, motion_vec) / (mag * motion_mag) if mag and motion_mag else 0
                aligned = mag * cosine

                rows.append({
                    "frame": frame,
                    "resid": resid,
                    "resname": resname,
                    "Fx": total[0],
                    "Fy": total[1],
                    "Fz": total[2],
                    "magnitude": mag,
                    "cosine_with_motion": cosine,
                    "aligned_force_magnitude": aligned,
                    "source_type": "residue",
                    "motion_vec_x": motion_vec[0],
                    "motion_vec_y": motion_vec[1],
                    "motion_vec_z": motion_vec[2],
                    "motion_mag": motion_mag
                })

        # === PIP2 ===
        grouped_pips = defaultdict(list)
        for contrib in frame_data.get("pip2_contributions", []):
            resid = contrib["resid"]
            grouped_pips[resid].append(np.array(contrib["force"]))
        for resid, atom_forces in grouped_pips.items():
            total = np.sum(atom_forces, axis=0)
            mag = np.linalg.norm(total)
            cosine = np.dot(total, motion_vec) / (mag * motion_mag) if mag and motion_mag else 0
            aligned = mag * cosine

            rows.append({
                "frame": frame,
                "resid": resid,
                "resname": "PIP",
                "Fx": total[0],
                "Fy": total[1],
                "Fz": total[2],
                "magnitude": mag,
                "cosine_with_motion": cosine,
                "aligned_force_magnitude": aligned,
                "source_type": "pip",
                "motion_vec_x": motion_vec[0],
                "motion_vec_y": motion_vec[1],
                "motion_vec_z": motion_vec[2],
                "motion_mag": motion_mag
            })

        # === IONIC CONTRIBUTIONS ===
        ionic_up = []
        ionic_down = []
        for contrib in frame_data.get("ionic_contributions", []):
            force = np.array(contrib["force"])
            if contrib.get("position", "up") == "up":
                ionic_up.append(force)
            else:
                ionic_down.append(force)

        for label, group in [("ionic_up", ionic_up), ("ionic_down", ionic_down)]:
            total = np.sum(group, axis=0) if group else np.zeros(3)
            mag = np.linalg.norm(total)
            cosine = np.dot(total, motion_vec) / (mag * motion_mag) if mag and motion_mag else 0
            aligned = mag * cosine

            rows.append({
                "frame": frame,
                "resid": label,
                "resname": None,
                "Fx": total[0],
                "Fy": total[1],
                "Fz": total[2],
                "magnitude": mag,
                "cosine_with_motion": cosine,
                "aligned_force_magnitude": aligned,
                "source_type": label,
                "motion_vec_x": motion_vec[0],
                "motion_vec_y": motion_vec[1],
                "motion_vec_z": motion_vec[2],
                "motion_mag": motion_mag
            })

    return pd.DataFrame(rows)

import os
import matplotlib.pyplot as plt
import pandas as pd
import re

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None


def plot_force_magnitudes_for_one_ion(df, ion_id, stuck_frames, permeation_frame, channel_type = "G2",folder="./force_magnitude_plots"):
    """
    Generate histograms of force magnitudes during stuck frames for a single ion.
    Parameters:
        df (DataFrame): data for one ion
        ion_id (str or int): e.g., "2398"
        stuck_start_frame (int): e.g., 623
        permeation_frame (int): e.g., 706
        output_dir (str): root directory to save plots
    """
    output_dir = f"{folder}/force_magnitude_plots"
    stuck_start_frame = min(stuck_frames)

    df_stuck = df[df["frame"].isin(stuck_frames)]
    df_perm = df[df["frame"] == permeation_frame]

    ion_folder = os.path.join(output_dir, f"ion_{ion_id}")
    os.makedirs(ion_folder, exist_ok=True)

    for (resid, source_type), group in df_stuck.groupby(["resid", "source_type"]):
        magnitudes = group["magnitude"].dropna().values

        match = df_perm[(df_perm["resid"] == resid) & (df_perm["source_type"] == source_type)]
        if match.empty:
            continue

        mag_perm = match["magnitude"].values[0]
        resname = match["resname"].values[0] if "resname" in match.columns and pd.notnull(match["resname"].values[0]) else "None"
        filename = f"{resid}_{resname}.png"

        # Plot
        plt.figure(figsize=(8, 6))
        plt.hist(magnitudes, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mag_perm, color='red', linestyle='dashed', linewidth=2, label=f'Permeation = {mag_perm:.2f}')
        if source_type == "residue":
            title = f"{convert_to_pdb_numbering(resid,channel_type)} | Ion: {ion_id}"
        elif source_type == "pip":
            title = f"PIP {resid} | Ion: {ion_id}"
        elif source_type == "ionic_up":
            title = f"Ionic force from ions above ion {ion_id}"
        elif source_type == "ionic_down":
            title = f"Ionic force from ions below ion {ion_id}"
        plt.title(title)
        plt.xlabel('Force Magnitude (Stuck Frames)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()

        # Save
        save_path = os.path.join(ion_folder, filename)
        plt.savefig(save_path)
        plt.close()

    # print(f"✅ Histograms saved to: {ion_folder}")

import os
import matplotlib.pyplot as plt

def plot_force_magnitude_boxplots_for_one_ion(df, ion_id, stuck_frames, permeation_frame,channel_type = "G2",folder="./force_magnitude_boxplots"):
    """
    Generate boxplots of force magnitudes during stuck frames for a single ion.
    Marks the permeation frame's magnitude with a red line.
    
    Parameters:
        df (DataFrame): data for one ion
        ion_id (str or int): e.g., "2398"
        stuck_start_frame (int): e.g., 623
        permeation_frame (int): e.g., 706
        output_dir (str): root directory to save plots
    """
    stuck_start_frame = min(stuck_frames)
    df_stuck = df[df["frame"].isin(stuck_frames)]
    df_perm = df[df["frame"] == permeation_frame]

    output_dir = f"{folder}/force_magnitude_boxplots"
    ion_folder = os.path.join(output_dir, f"ion_{ion_id}")
    os.makedirs(ion_folder, exist_ok=True)

    for (resid, source_type), group in df_stuck.groupby(["resid", "source_type"]):
        magnitudes = group["magnitude"].dropna().values

        match = df_perm[(df_perm["resid"] == resid) & (df_perm["source_type"] == source_type)]
        if match.empty:
            continue

        mag_perm = match["magnitude"].values[0]
        resname = match["resname"].values[0] if "resname" in match.columns and pd.notnull(match["resname"].values[0]) else "None"
        filename = f"{resid}_{resname}.png"

        # Plot boxplot
        plt.figure(figsize=(8, 4))
        plt.boxplot(magnitudes, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='black'))
        plt.axvline(mag_perm, color='red', linestyle='--', linewidth=2, label=f'Permeation = {mag_perm:.2f}')
        
        if source_type == "residue":
            title = f"{convert_to_pdb_numbering(resid, channel_type)} | Ion: {ion_id}"
        elif source_type == "pip":
            title = f"PIP {resid} | Ion: {ion_id}"
        elif source_type == "ionic_up":
            title = f"Ionic force from ions above ion {ion_id}"
        elif source_type == "ionic_down":
            title = f"Ionic force from ions below ion {ion_id}"
        plt.title(title)
        plt.xlabel('Force Magnitude')
        plt.legend()
        plt.tight_layout()

        # Save
        save_path = os.path.join(ion_folder, filename)
        plt.savefig(save_path)
        plt.close()

    # print(f"✅ Boxplots saved to: {ion_folder}")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest

def sign_test(stuck_mags, mag_perm, alternative="two-sided"):
    stuck_mags = np.asarray(stuck_mags)
    n = len(stuck_mags)
    num_greater = np.sum(mag_perm > stuck_mags)
    num_less = np.sum(mag_perm < stuck_mags)

    if num_greater > num_less:
        direction = "greater"
    elif num_less > num_greater:
        direction = "less"
    else:
        direction = "equal"

    successes = max(num_greater, num_less)
    p_value = binomtest(successes, n=n, p=0.5, alternative=alternative).pvalue
    return p_value, direction

def boxplot_outlier_classification(stuck_mags, mag_perm):
    Q1 = np.percentile(stuck_mags, 25)
    Q3 = np.percentile(stuck_mags, 75)
    IQR = Q3 - Q1

    mild_upper = Q3 + 1.5 * IQR
    mild_lower = Q1 - 1.5 * IQR
    extreme_upper = Q3 + 3 * IQR
    extreme_lower = Q1 - 3 * IQR

    if mag_perm > extreme_upper or mag_perm < extreme_lower:
        return "extreme outlier"
    elif mag_perm > mild_upper or mag_perm < mild_lower:
        return "mild outlier"
    else:
        return "within range"
    
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu




def analyze_force_vectors(df, permeation_frame, stuck_frames):
    """
    Compares force vectors between stuck frames and a given permeation frame.
    Performs statistical tests and outlier analysis on force magnitudes.
    
    Parameters:
    - df: DataFrame with force vectors and a 'magnitude' column
    - permeation_frame: frame where ion permeation occurs
    - stuck_start_frame: first frame of the stuck period

    Returns:
    - DataFrame with comparison results and significance metrics
    """
    stuck_start_frame = min(stuck_frames)
    # Label groups
    df["group"] = df["frame"].apply(
        lambda f: "permeation" if f == permeation_frame else
                  "stuck" if f in stuck_frames else "other"
    )
    
    df_stuck = df[df["group"] == "stuck"]
    df_perm = df[df["group"] == "permeation"]
    total_frames = permeation_frame - stuck_start_frame + 1

    def vector_metrics(v1, v2):
        delta_vec = v2 - v1
        magnitude1 = np.linalg.norm(v1)
        magnitude2 = np.linalg.norm(v2)
        delta_magnitude = magnitude2 - magnitude1
        cosine = np.dot(v1, v2) / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
        return magnitude1, magnitude2, delta_magnitude, cosine

    results = []
    group_keys = ["resid", "resname", "source_type"]
    
    for key, df_group in df_stuck.groupby(group_keys):
        resid, resname, source_type = key
        vec_stuck = df_group[["Fx", "Fy", "Fz"]].mean().values

        match = df_perm[
            (df_perm["resid"] == resid) &
            (df_perm["source_type"] == source_type)
        ]
        if not match.empty:
            vec_perm = match.iloc[0][["Fx", "Fy", "Fz"]].values
            mag_stuck, mag_perm, delta_mag, cosine = vector_metrics(vec_stuck, vec_perm)

            stuck_mags = df_group["magnitude"].dropna().values
            mw_p = mannwhitneyu(stuck_mags, [mag_perm], alternative='two-sided').pvalue
            sign_p, sign_dir = sign_test(stuck_mags, mag_perm)
            box_class = boxplot_outlier_classification(stuck_mags, mag_perm)

            results.append({
                "stuck_frame": stuck_start_frame,
                "perm_frame": permeation_frame,
                "total_frames": total_frames,
                "resid": resid,
                "resname": resname,
                "source_type": source_type,
                "Fx_stuck": vec_stuck[0], "Fy_stuck": vec_stuck[1], "Fz_stuck": vec_stuck[2],
                "Fx_perm": vec_perm[0], "Fy_perm": vec_perm[1], "Fz_perm": vec_perm[2],
                "mag_stuck": mag_stuck,
                "mag_perm": mag_perm,
                "delta_magnitude": delta_mag,
                "cosine_angle": cosine,
                "cosine_with_motion": match.iloc[0]["cosine_with_motion"],
                "aligned_force_magnitude": match.iloc[0]["aligned_force_magnitude"],
                "mannwhitney_p": mw_p,
                "sign_test_p": sign_p,
                "sign_test_direction": sign_dir,
                "boxplot_outlier": box_class
            })

    # Analyze special cases for ion sources
    for source in ["ionic_up", "ionic_down"]:
        stuck_vectors = df_stuck[df_stuck["source_type"] == source].drop_duplicates(subset=["frame"])
        perm_vectors = df_perm[df_perm["source_type"] == source].drop_duplicates(subset=["frame"])

        if stuck_vectors.empty or perm_vectors.empty:
            continue

        vec_stuck = stuck_vectors[["Fx", "Fy", "Fz"]].mean().values
        vec_perm = perm_vectors[["Fx", "Fy", "Fz"]].mean().values

        mag_stuck, mag_perm, delta_mag, cosine = vector_metrics(vec_stuck, vec_perm)
        stuck_mags = stuck_vectors["magnitude"].dropna().values
        mw_p = mannwhitneyu(stuck_mags, [mag_perm], alternative='two-sided').pvalue
        sign_p, sign_dir = sign_test(stuck_mags, mag_perm)
        box_class = boxplot_outlier_classification(stuck_mags, mag_perm)

        results.append({
            "stuck_frame": stuck_start_frame,
            "perm_frame": permeation_frame,
            "total_frames": total_frames,
            "resid": source,
            "resname": None,
            "source_type": source,
            "Fx_stuck": vec_stuck[0], "Fy_stuck": vec_stuck[1], "Fz_stuck": vec_stuck[2],
            "Fx_perm": vec_perm[0], "Fy_perm": vec_perm[1], "Fz_perm": vec_perm[2],
            "mag_stuck": mag_stuck,
            "mag_perm": mag_perm,
            "delta_magnitude": delta_mag,
            "cosine_angle": cosine,
            "cosine_with_motion": perm_vectors.iloc[0]["cosine_with_motion"],
            "aligned_force_magnitude": perm_vectors.iloc[0]["aligned_force_magnitude"],
            "mannwhitney_p": mw_p,
            "sign_test_p": sign_p,
            "sign_test_direction": sign_dir,
            "boxplot_outlier": box_class
        })

    return pd.DataFrame(results)

from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def summarize_significant_forces(result_folder, output_plot_path_1, output_plot_path_2,
                                  significance_threshold=0.1, frame_threshold=10, channel_type="G2"):
    summary = defaultdict(lambda: {
        "count_significant": 0,
        "count_total": 0,
        "count_positive_delta": 0,
        "count_negative_delta": 0
    })

    for filename in os.listdir(result_folder):
        if filename.startswith("result_") and filename.endswith(".csv"):
            path = os.path.join(result_folder, filename)
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                if row["source_type"] == "residue":
                    key = f"{convert_to_pdb_numbering(int(row['resid']),channel_type)}"
                elif row["source_type"] == "pip":
                    key = f"PIP {row['resid']}"
                elif row["source_type"] == "ionic_up":
                    key = f"Ions up"
                elif row["source_type"] == "ionic_down":
                    key = f"Ions down"
                is_significant = False

                if row["total_frames"] < frame_threshold:
                    if row["sign_test_p"] < significance_threshold:
                        is_significant = True
                else:
                    if row["mannwhitney_p"] < significance_threshold:
                        is_significant = True

                if is_significant:
                    summary[key]["count_significant"] += 1

                
                    
                if row["delta_magnitude"] > 0:
                    summary[key]["count_positive_delta"] += 1
                elif row["delta_magnitude"] < 0:
                    summary[key]["count_negative_delta"] += 1

                summary[key]["count_total"] += 1

    # Convert to DataFrame
    summary_df = pd.DataFrame([
        {
            "resid_source": k,
            "significant": v["count_significant"],
            "total": v["count_total"],
            "positive_delta": v["count_positive_delta"],
            "negative_delta": v["count_negative_delta"]
        }
        for k, v in summary.items()
    ])

        # Custom sorting: PIPs → Residues → Ions
    def sort_key(label):
        if "PIP" in label:
            return (0, label)
        elif "Ions" in label or "ionic" in label.lower():
            return (2, label)
        else:
            return (1, label)
    
    summary_df = summary_df.copy()
    summary_df["sort_order"] = summary_df["resid_source"].apply(sort_key)
    summary_df = summary_df.sort_values("sort_order").drop(columns=["sort_order"])

    # summary_df = summary_df.sort_values("significant", ascending=False)

    # Barplot 1: Significant counts
    plt.figure(figsize=(14, 6))
    bars = plt.bar(summary_df["resid_source"], summary_df["significant"], color="teal")
    plt.xticks(rotation=45)
    plt.xlabel("Force source")
    plt.ylabel("Count of Significant Change in force magnitude")
    plt.title("Significant force magnitude change across ions")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_plot_path_1)
    plt.close()

    # Barplot 2: Positive vs Negative delta
    x = np.arange(len(summary_df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 6))
    pos_bars = ax.bar(x - width/2, summary_df["positive_delta"], width=width, label='Increase in magnitude', color='green')
    neg_bars = ax.bar(x + width/2, summary_df["negative_delta"], width=width, label='Decrease in magnitude', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["resid_source"], rotation=45)
    ax.set_ylabel("Count")
    ax.set_xlabel("Force source")
    ax.set_title("Change in Force Magnitude Across Permeation")
    ax.legend()

    for bar in pos_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{int(height)}', ha='center', va='bottom', fontsize=9)

    for bar in neg_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot_path_2)
    plt.close()

    return summary_df

def significant_forces(channel_type="G2", folder="./significant_forces"):
    
    # Define your data folders
    json_folder = f"{folder}/forces_per_ion/"
    csv_folder = f"{folder}/csv_per_ion/"
    result_folder = f"{folder}/comparison_results/"
    stuck_frames_json = f"{folder}/stuck_frames.json"

    stuck_frames_dict = load_json(stuck_frames_json)  # Load stuck frames data to ensure it exists
    # Create output folders if they don't exist
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # Loop over all JSON files
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    for filename in tqdm(json_files, desc="Processing JSON force files", total=len(json_files), unit="file"):
        # print(f"🔄 Processing {filename}")
        json_path = os.path.join(json_folder, filename)
        csv_filename = filename.replace(".json", ".csv")
        csv_path = os.path.join(csv_folder, csv_filename)
        result_csv_path = os.path.join(result_folder, f"result_{csv_filename}")
        result_image_path = os.path.join(result_folder, f"result_{csv_filename.replace('.csv', '.png')}")

        # print(f"🔄 Processing {filename} → {csv_filename}")
        json_data = load_json(json_path)
        try:
            df = create_force_dataframe(json_data)
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            continue

        # Extract permeation and stuck frame info
        try:
            stuck_frames_list = stuck_frames_dict[strip_json_extension(filename)]
            permeation_frame = json_data["start_frame"] - 1
        except Exception as e:
            print(f"❌ Failed to extract frame info from {filename}: {e}")
            continue
        
        # print(extract_number(filename),json_data["start_frame"], permeation_frame, start_stuck_frame)
        # Analyze
        
        df_result = analyze_force_vectors(df, permeation_frame, stuck_frames_list)
        # if filename == "2397_1.json":
        #     print(permeation_frame, start_stuck_frame)
        #     print(df)
        #     print(df_result)
        df_result = df_result.sort_values("delta_magnitude", ascending=False)
        plot_force_magnitudes_for_one_ion(df, extract_number(filename), stuck_frames_list, permeation_frame, channel_type, folder)
        plot_force_magnitude_boxplots_for_one_ion(df, extract_number(filename), stuck_frames_list, permeation_frame, channel_type,folder)

        # Save result as CSV
        df_result.to_csv(result_csv_path, index=False)

        # Save result as image
        try:
            # Drop specific columns before exporting
            excluded_columns = ["Fx_stuck", "Fy_stuck", "Fz_stuck", "Fx_perm", "Fy_perm", "Fz_perm"]
            df_filtered = df_result.drop(columns=excluded_columns)
            
            # Export the filtered DataFrame as an image
            dfi.export(df_filtered, result_image_path, table_conversion='matplotlib')
            # print(f"🖼️  Saved image to {result_image_path}")
        except Exception as e:
            print(f"❌ Failed to save image for {filename}: {e}")

    # === Call the aggregation function after processing all ions ===
    summary_df = summarize_significant_forces(
            result_folder=result_folder,
            output_plot_path_1=os.path.join(result_folder, "significant_force_summary.png"),
            output_plot_path_2=os.path.join(result_folder, "delta_direction_summary.png"),
            channel_type=channel_type
    )

