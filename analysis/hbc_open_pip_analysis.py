"""
hbc_open_pip_analysis.py

For each PIP, uses the gate column that corresponds to the subunit:
  2a  →  A_C
  2c  →  A_C
  2b  →  B_D
  G1  →  B_D

Finds frames where that gate distance > threshold, then for each phosphate
(P1, P4, P5) reports:
  - every residue that was < 4 Å in at least one of those frames
  - number of frames that residue was < 4 Å
  - percentage of gate-open frames that residue was < 4 Å
  - total number of gate-open frames

Output per RUN under results_dir/hbc_open_pip_analysis_{threshold}/:
  open_frames_AC.csv / open_frames_BD.csv
  PIP_1301_2b/
    P1_contact_table.csv
    P1_contact_counts.png       ← raw frame counts
    P1_contact_percentage.png   ← percentage of open frames
    ...
  summary_all_pips.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHOSPHATE_COLORS = {
    "P1": "#1a7a2e",   # dark green
    "P4": "#b85c00",   # dark orange
    "P5": "#1a3a7a",   # dark blue
}

SUBUNIT_TO_GATE = {
    "2a": "A_C",
    "2c": "A_C",
    "2b": "B_D",
    "G1": "B_D",
}
# SUBUNIT_TO_GATE = {
#     "2a": "B_D",
#     "2c": "B_D",
#     "2b": "A_C",
#     "G1": "A_C",
# }

def _subunit_from_folder(folder_name: str):
    return folder_name.split("_")[-1]


def _get_open_frames(hbc_df: pd.DataFrame, gate_col: str, threshold: float):
    mask = hbc_df[gate_col] > threshold
    return hbc_df.loc[mask, "frame"]


def _contact_table(dist_csv: str, open_frames: pd.Series,
                   contact_cutoff: float, n_open_total: int):
    dist_df = pd.read_csv(dist_csv, index_col="frame")
    common = open_frames[open_frames.isin(dist_df.index)]
    if len(common) == 0:
        return None

    sub = dist_df.loc[common]
    n_contact = (sub < contact_cutoff).sum()
    n_contact = n_contact[n_contact >= 1]

    if n_contact.empty:
        return None

    table = pd.DataFrame({
        "n_contact_frames":    n_contact.astype(int),
        "n_gate_open_frames":  n_open_total,
        "contact_pct":         (n_contact / n_open_total * 100).round(1),
        "mean_dist_open":      sub[n_contact.index].mean().round(3),
        "min_dist_open":       sub[n_contact.index].min().round(3),
    })
    table.index.name = "residue"
    table = table.sort_values("n_contact_frames", ascending=False)
    return table


def _plot_counts(table: pd.DataFrame, title: str, out_path: str, color: str = 'steelblue', top_n: int = 20):
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
    ax.set_ylabel("N contact frames", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close()


def _plot_percentage(table: pd.DataFrame, title: str, out_path: str, color: str = 'darkorange', top_n: int = 20):
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
    ax.set_ylim(0, 110)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close()


# ── main entry point ──────────────────────────────────────────────────────────

def run_hbc_open_pip_analysis(
    results_dir: str,
    hbc_threshold: float = 10.0,
    contact_cutoff: float = 4.0,
    prescreen_factor: float = 5.0,
    top_n_plot: int = 20,
):
    # ── find pip_proximity folder ─────────────────────────────────────────────
    prox_name = f"pip_proximity_{int(prescreen_factor)}"
    pip_proximity_dir = os.path.join(results_dir, prox_name)
    if not os.path.isdir(pip_proximity_dir):
        candidates = sorted(glob.glob(os.path.join(results_dir, "pip_proximity_*")))
        if not candidates:
            print(f"[hbc_open_pip] No pip_proximity_* folder in {results_dir}")
            return
        pip_proximity_dir = candidates[0]
        print(f"[hbc_open_pip] Using: {pip_proximity_dir}")

    # ── load hbc distances ────────────────────────────────────────────────────
    hbc_path = os.path.join(results_dir, "hbc_distances.csv")
    if not os.path.isfile(hbc_path):
        print(f"[hbc_open_pip] hbc_distances.csv not found in {results_dir}")
        return

    hbc_df = pd.read_csv(hbc_path)

    open_frames_by_gate = {}
    for gate_col in ["A_C", "B_D"]:
        of = _get_open_frames(hbc_df, gate_col, hbc_threshold)
        open_frames_by_gate[gate_col] = of
        print(f"[hbc_open_pip] {gate_col} open frames: {len(of)}")

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = os.path.join(results_dir, "hbc_open_pip_analysis", str(hbc_threshold))
    os.makedirs(out_dir, exist_ok=True)

    for gate_col, of in open_frames_by_gate.items():
        hbc_df[hbc_df["frame"].isin(of)].to_csv(
            os.path.join(out_dir, f"open_frames_{gate_col}.csv"), index=False
        )

    # ── process each PIP folder ───────────────────────────────────────────────
    pip_folders = sorted(glob.glob(os.path.join(pip_proximity_dir, "PIP_*")))
    if not pip_folders:
        print(f"[hbc_open_pip] No PIP_* folders in {pip_proximity_dir}")
        return

    summary_rows = []

    for pip_folder in pip_folders:
        pip_name = os.path.basename(pip_folder)
        subunit  = _subunit_from_folder(pip_name)
        gate_col = SUBUNIT_TO_GATE.get(subunit)

        if gate_col is None:
            print(f"[hbc_open_pip] Unknown subunit '{subunit}' in {pip_name}, skipping.")
            continue

        open_frames = open_frames_by_gate[gate_col]
        n_open = len(open_frames)

        if n_open == 0:
            print(f"[hbc_open_pip] {pip_name}: no open frames for {gate_col}, skipping.")
            continue

        pip_out = os.path.join(out_dir, pip_name)
        os.makedirs(pip_out, exist_ok=True)

        print(f"[hbc_open_pip] {pip_name} → gate={gate_col}, n_open={n_open}")

        for phosphate in ["P1", "P4", "P5"]:
            dist_csv = os.path.join(pip_folder, phosphate, "distances_all_pdb.csv")
            if not os.path.isfile(dist_csv):
                continue

            table = _contact_table(dist_csv, open_frames, contact_cutoff, n_open)
            if table is None:
                print(f"  [{phosphate}] no contacts found")
                continue

            table.to_csv(os.path.join(pip_out, f"{phosphate}_contact_table.csv"))

            base_title = (f"{pip_name} | {phosphate} | gate={gate_col} > {hbc_threshold} Å"
                          f"  (n_open={n_open})")

            color = PHOSPHATE_COLORS.get(phosphate, "steelblue")
            _plot_counts(
                table, title=base_title,
                out_path=os.path.join(pip_out, f"{phosphate}_contact_counts.png"),
                color=color, top_n=top_n_plot,
            )
            _plot_percentage(
                table, title=base_title,
                out_path=os.path.join(pip_out, f"{phosphate}_contact_percentage.png"),
                color=color, top_n=top_n_plot,
            )

            print(f"  [{phosphate}] {len(table)} residues with ≥1 contact frame")

            for residue, row in table.head(5).iterrows():
                summary_rows.append({
                    "pip":                pip_name,
                    "subunit":            subunit,
                    "gate_col":           gate_col,
                    "phosphate":          phosphate,
                    "residue":            residue,
                    "n_contact_frames":   int(row["n_contact_frames"]),
                    "n_gate_open_frames": n_open,
                    "contact_pct":        row["contact_pct"],
                    "mean_dist_open":     row["mean_dist_open"],
                    "min_dist_open":      row["min_dist_open"],
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out_dir, "summary_all_pips.csv"), index=False)
        print(f"[hbc_open_pip] Summary → {os.path.join(out_dir, 'summary_all_pips.csv')}")

    print(f"[hbc_open_pip] Done. All outputs in: {out_dir}")