"""
combine_hbc_open_pip.py

Standalone script. Combines hbc_open_pip_analysis results across all RUN* folders.

For every PIP (e.g. PIP_1302_G1) and every phosphate (P1, P4, P5):
  - reads P*_contact_table.csv from each RUN
  - sums n_contact_frames and n_gate_open_frames across runs
  - recomputes contact_pct
  - saves a combined CSV + two plots (counts + percentage)

Usage:
    python combine_hbc_open_pip.py --base_dir /path/to/G12 --threshold 10.0

Output:
    base_dir/combined_hbc_open_pip_{threshold}/
        PIP_1302_G1/
            P1_combined.csv
            P1_combined_counts.png
            P1_combined_percentage.png
            ...
"""

"""
combine_hbc_open_pip.py

Standalone script. Combines hbc_open_pip_analysis results across all RUN* folders.

For every PIP (e.g. PIP_1302_G1) and every phosphate (P1, P4, P5):
  - reads P*_contact_table.csv from each RUN
  - sums n_contact_frames and n_gate_open_frames across runs
  - recomputes contact_pct
  - saves a combined CSV + two plots (counts + percentage)

Usage:
    python3 combine_hbc_open_pip.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --threshold 10.0

Output:
    base_dir/combined_hbc_open_pip_{threshold}/
        PIP_1302_G1/
            P1_combined.csv
            P1_combined_counts.png
            P1_combined_percentage.png
            ...
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── plot helpers ──────────────────────────────────────────────────────────────

def _plot_counts(table, title, out_path, color='steelblue', top_n=20):
    top = table.head(top_n)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(top))
    bars = ax.bar(x, top["n_contact_frames"], color=color, alpha=0.85)
    ax.axhline(top["n_gate_open_frames"].iloc[0], color="tomato",
               linestyle="--", linewidth=2.0, label="Total open frames")
    for bar, pct in zip(bars, top["contact_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=16)
    ax.set_ylabel("N contact frames (all runs)", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close()


def _plot_percentage(table, title, out_path, color='darkorange', top_n=20):
    top = table.head(top_n)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(top))
    bars = ax.bar(x, top["contact_pct"], color=color, alpha=0.85)
    ax.axhline(100, color="tomato", linestyle="--", linewidth=1.5, label="100%")
    for bar, pct in zip(bars, top["contact_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=16)
    ax.set_ylabel("% of open frames with contact", fontsize=16)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

PHOSPHATE_COLORS = {
    "P1": "#1a7a2e",   # dark green
    "P4": "#b85c00",   # dark orange
    "P5": "#1a3a7a",   # dark blue
}

def combine(base_dir: str, threshold: float, top_n: int = 20):

    analysis_folder = os.path.join("hbc_open_pip_analysis", str(threshold))
    out_dir = os.path.join(base_dir, "combined_hbc_open_pip", str(threshold))
    os.makedirs(out_dir, exist_ok=True)

    # find all RUN* folders
    run_dirs = sorted(glob.glob(os.path.join(base_dir, "RUN*")))
    if not run_dirs:
        print(f"No RUN* folders found in {base_dir}")
        return
    print(f"Found {len(run_dirs)} runs: {[os.path.basename(r) for r in run_dirs]}")

    # collect all PIP names from first available run
    pip_names = set()
    for run_dir in run_dirs:
        analysis_dir = os.path.join(run_dir, analysis_folder)
        if not os.path.isdir(analysis_dir):
            continue
        for pip_folder in glob.glob(os.path.join(analysis_dir, "PIP_*")):
            pip_names.add(os.path.basename(pip_folder))

    if not pip_names:
        print(f"No PIP_* folders found inside any {analysis_folder}")
        return

    pip_names = sorted(pip_names)
    print(f"PIPs found: {pip_names}")

    summary_rows = []

    for pip_name in pip_names:
        pip_out = os.path.join(out_dir, pip_name)
        os.makedirs(pip_out, exist_ok=True)

        for phosphate in ["P1", "P4", "P5"]:
            frames_per_residue = {}   # residue -> total n_contact_frames
            total_open_frames  = 0
            runs_found         = 0

            for run_dir in run_dirs:
                csv_path = os.path.join(run_dir, analysis_folder,
                                        pip_name, f"{phosphate}_contact_table.csv")
                if not os.path.isfile(csv_path):
                    continue

                df = pd.read_csv(csv_path, index_col="residue")
                runs_found += 1

                # n_gate_open_frames is the same for all residues in this file
                total_open_frames += int(df["n_gate_open_frames"].iloc[0])

                for residue, row in df.iterrows():
                    frames_per_residue[residue] = (
                        frames_per_residue.get(residue, 0) + int(row["n_contact_frames"])
                    )

            if runs_found == 0 or total_open_frames == 0:
                print(f"  {pip_name}/{phosphate}: no data found, skipping.")
                continue

            # build combined table
            combined = pd.DataFrame({
                "n_contact_frames":   pd.Series(frames_per_residue, dtype=int),
                "n_gate_open_frames": total_open_frames,
                "contact_pct":        pd.Series({
                    r: round(v / total_open_frames * 100, 1)
                    for r, v in frames_per_residue.items()
                }),
            })
            combined.index.name = "residue"
            combined = combined.sort_values("n_contact_frames", ascending=False)

            # save csv
            csv_out = os.path.join(pip_out, f"{phosphate}_combined.csv")
            combined.to_csv(csv_out)

            base_title = (f"{pip_name} | {phosphate} | combined {runs_found} runs"
                          f"  (total open frames={total_open_frames})")

            color = PHOSPHATE_COLORS.get(phosphate, "steelblue")
            _plot_counts(
                combined, title=base_title,
                out_path=os.path.join(pip_out, f"{phosphate}_combined_counts.png"),
                color=color, top_n=top_n,
            )
            _plot_percentage(
                combined, title=base_title,
                out_path=os.path.join(pip_out, f"{phosphate}_combined_percentage.png"),
                color=color, top_n=top_n,
            )

            print(f"  {pip_name}/{phosphate}: {len(combined)} residues, "
                  f"{runs_found} runs, {total_open_frames} total open frames")

            for residue, row in combined.head(5).iterrows():
                summary_rows.append({
                    "pip":                pip_name,
                    "phosphate":          phosphate,
                    "residue":            residue,
                    "n_contact_frames":   int(row["n_contact_frames"]),
                    "n_gate_open_frames": total_open_frames,
                    "contact_pct":        row["contact_pct"],
                    "n_runs":             runs_found,
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "summary_all_pips_combined.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary → {summary_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",  required=True,
                        help="Folder containing RUN1, RUN2, ... subfolders")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="HBC threshold used when running hbc_open_pip_analysis")
    parser.add_argument("--top_n",     type=int,   default=20,
                        help="Top N residues to show in plots")
    args = parser.parse_args()
    combine(args.base_dir, args.threshold, args.top_n)