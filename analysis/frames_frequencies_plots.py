import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_intervals_by_frames(intervals, max_bar_number=20):
    """
    Plots and saves bar charts showing the top frame-duration intervals per ion,
    sorted by frame count.

    Parameters:
    - intervals: dict from residue_clusters.json
    - max_bar_number: how many top intervals to show (default: 20)
    """
    output_dir = os.path.join("results", "biggest_intervals")
    os.makedirs(output_dir, exist_ok=True)

    for ion_id, interval_list in intervals.items():
        df = pd.DataFrame(interval_list)
        df_sorted = df.sort_values(by="frames", ascending=False)

        if max_bar_number:
            df_plot = df_sorted[:max_bar_number]
        else:
            df_plot = df_sorted.copy()

        df_plot["interval"] = df_plot.apply(lambda row: f"({int(row['start'])},{int(row['end'])})", axis=1)

        # Plotting
        plt.figure(figsize=(14, 6), facecolor='white')
        bars = plt.bar(range(len(df_plot)), df_plot["frames"], color='skyblue')

        # Add residue labels on top of bars
        for i, (residue, frames) in enumerate(zip(df_plot["residue"], df_plot["frames"])):
            plt.text(i, frames + 0.5, str(residue), ha='center', va='bottom', fontsize=8)

        # Set axis labels
        plt.xticks(ticks=range(len(df_plot)), labels=df_plot["interval"], rotation=35)
        plt.xlabel("Interval (start, end)")
        plt.ylabel("Number of Frames")
        plt.title(f"Ion: {ion_id} – Interval Durations Sorted by Frame Count")
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(output_dir, f"{ion_id}.png")
        plt.savefig(plot_filename)
        print(f"Saved plot for Ion {ion_id} → {plot_filename}")

        # # Show the plot
        # plt.show()
