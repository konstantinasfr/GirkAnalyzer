"""
plot_run_variability.py
========================
Reads pip_pose/subunit_*/summary.csv from each RUN folder and plots
the run-to-run variability of PIP2 pose states.

For each subunit and each state, makes:
  1. Bar plot — one bar per run showing % of frames in that state
  2. Histogram — distribution of % across runs
  3. Summary CSV — per-run percentages for all states

Usage
-----
    python plot_run_variability.py \
        --runs_dir     /path/to/G12 \
        --channel_type G12 \
        --output_dir   /path/to/output   (default: runs_dir/run_variability)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SUBUNIT_ORDER = ["2a", "2b", "2c", "G1"]
ALL_STATES = [
    "p4_Nterm_p5_Nterm", "p4_Nterm_p5_Cterm",   "p4_Nterm_p5_none",
    "p4_Cterm_p5_Nterm",    "p4_Cterm_p5_Cterm",       "p4_Cterm_p5_none",
    "p4_none_p5_Nterm",    "p4_none_p5_Cterm",       "p4_none_p5_none",
]
STATE_COLORS = {
    "p4_Nterm_p5_Nterm": "#1565C0",
    "p4_Nterm_p5_Cterm":    "#4CAF50",
    "p4_Nterm_p5_none":    "#A5D6A7",
    "p4_Cterm_p5_Nterm":    "#AB47BC",
    "p4_Cterm_p5_Cterm":       "#B71C1C",
    "p4_Cterm_p5_none":       "#EF9A9A",
    "p4_none_p5_Nterm":    "#90CAF9",
    "p4_none_p5_Cterm":       "#FFCDD2",
    "p4_none_p5_none":       "#9E9E9E",
}
STATE_LABEL = lambda s: s.replace("p4_","").replace("_p5_","+")


def _sorted_run_dirs(runs_dir: Path) -> list:
    return sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.upper().startswith("RUN") and d.name[3:].isdigit()],
        key=lambda d: int(d.name[3:]) if d.name[3:].isdigit() else 999,
    )



def _translate_state(state: str) -> str:
    """Translate old shallow/deep state names to new N-term/C-term."""
    return (state
            .replace("shallow", "N-term")
            .replace("deep", "C-term"))



def _load_permeation_counts(runs_dir: Path) -> dict:
    """
    Read permeation counts from permeation_summary/permeation_report.txt
    in the runs_dir (not per-run, it's a combined report).
    Returns dict: {run_name: n_permeations}
    """
    report_path = runs_dir / "permeation_summary" / "permeation_report.txt"
    if not report_path.exists():
        print(f"  WARNING: permeation_report.txt not found at {report_path}")
        return {}

    counts = {}
    in_per_run = False
    for line in open(report_path):
        line = line.strip()
        if "Run" in line and "Frames" in line:
            in_per_run = True
            continue
        if in_per_run and line.startswith("==="):
            break
        if in_per_run and line.upper().startswith("RUN"):
            parts = line.split()
            if len(parts) >= 3:
                run_name = parts[0]
                try:
                    n_perm = int(parts[2])  # HBC column
                    counts[run_name] = n_perm
                except ValueError:
                    pass
    print(f"  Loaded permeation counts for {len(counts)} runs: {counts}")
    return counts


def _load_per_run_data(runs_dir: Path, subunit: str, subset: str = None) -> pd.DataFrame:
    """
    Load summary.csv from each RUN for a given subunit.
    Returns DataFrame with columns: run, state, pct
    """
    run_dirs = _sorted_run_dirs(runs_dir)
    base     = "pip_pose" + (f"/{subset}" if subset else "")
    rows     = []

    for run_dir in run_dirs:
        csv = run_dir / base / f"subunit_{subunit}" / "summary.csv"
        if not csv.exists():
            print(f"    MISSING: {csv}")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                "run":   run_dir.name,
                "state": _translate_state(row["state"]),
                "pct":   float(row["pct"]),
                "count": int(row["count"]),
            })
        # add zero rows for states not present in this run
        present = set(_translate_state(s) for s in df["state"].tolist())
        for state in ALL_STATES:
            if state not in present:
                rows.append({"run": run_dir.name, "state": state, "pct": 0.0, "count": 0})

    return pd.DataFrame(rows)


def _plot_per_run_bars(df: pd.DataFrame, state: str, channel_type: str,
                        subunit: str, label: str, out_path: Path,
                        perm_counts: dict = None):
    """One bar per run for a single state."""
    df_st = df[df["state"] == state].copy()
    df_st["_sort_key"] = df_st["run"].str.extract(r"(\d+)").astype(int)
    df_st = df_st.sort_values("_sort_key").drop(columns="_sort_key")
    if df_st.empty or df_st["pct"].sum() == 0:
        return

    runs = df_st["run"].tolist()
    pcts = df_st["pct"].tolist()
    color = STATE_COLORS.get(state, "#7E57C2")
    mean_pct = np.mean(pcts)
    std_pct  = np.std(pcts)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(runs))
    bars = ax.bar(x, pcts, color=color, edgecolor="white", alpha=0.85)
    ax.axhline(mean_pct, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_pct:.1f}% ± {std_pct:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("% of frames", fontsize=15)
    ax.set_title(f"{STATE_LABEL(state)} — {channel_type}\n"
                 f"Subunit {subunit} [{label}]", fontsize=15)
    ax.legend(fontsize=13, loc="upper left")
    ax.set_ylim(0, min(120, max(pcts) * 1.25) if pcts else 10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_histogram(df: pd.DataFrame, state: str, channel_type: str,
                     subunit: str, label: str, out_path: Path):
    """Histogram of % across runs for a single state."""
    df_st = df[df["state"] == state]
    pcts  = df_st["pct"].tolist()
    if not pcts or max(pcts) == 0:
        return

    color    = STATE_COLORS.get(state, "#7E57C2")
    mean_pct = np.mean(pcts)
    std_pct  = np.std(pcts)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pcts, bins=10, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(mean_pct, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_pct:.1f}% ± {std_pct:.1f}%")
    ax.set_xlabel("% of frames", fontsize=15)
    ax.set_ylabel("Number of runs", fontsize=15)
    ax.set_title(f"{STATE_LABEL(state)} — distribution across runs\n"
                 f"{channel_type} | Subunit {subunit} [{label}]", fontsize=14)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_all_states_overview(df: pd.DataFrame, channel_type: str,
                               subunit: str, label: str, out_path: Path):
    """
    Overview: for each state, show mean ± std across runs as a bar with error bar.
    Quick way to see which states are consistent vs variable across runs.
    """
    states_present = [s for s in ALL_STATES if s in df["state"].values
                      and df[df["state"]==s]["pct"].sum() > 0]
    if not states_present:
        return

    means = [df[df["state"]==s]["pct"].mean() for s in states_present]
    stds  = [df[df["state"]==s]["pct"].std()  for s in states_present]
    cols  = [STATE_COLORS.get(s, "#9E9E9E") for s in states_present]
    xlabs = [STATE_LABEL(s) for s in states_present]

    fig, ax = plt.subplots(figsize=(max(10, len(states_present)*1.2), 6))
    x = np.arange(len(states_present))
    bars = ax.bar(x, means, yerr=stds, color=cols, edgecolor="white",
                  alpha=0.85, capsize=5)
    max_val = max(m+s for m,s in zip(means,stds)) if means else 1
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + s + max_val*0.01,
                f"{m:.1f}%", va="bottom", ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("% of frames (mean ± std across runs)", fontsize=14)
    ax.set_title(f"PIP2 pose variability across runs\n"
                 f"{channel_type} | Subunit {subunit} [{label}]", fontsize=15)
    ax.set_ylim(0, min(115, max_val*1.3))
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)



def _compute_marginal_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-run state data, compute marginal P4 and P5 pose percentages.
    e.g. P4_shallow = sum of all states where p4 is shallow.
    Returns DataFrame with columns: run, atom, pose, pct
    """
    rows = []
    poses = ["N-term", "C-term", "none"]
    for run, grp in df.groupby("run"):
        total_pct = grp["pct"].sum()
        if total_pct == 0:
            continue
        for p_atom, prefix in [("P4", "p4_"), ("P5", "p5_")]:
            for pose in poses:
                if p_atom == "P4":
                    mask = grp["state"].str.startswith(f"p4_{pose}_")
                else:
                    mask = grp["state"].str.endswith(f"_p5_{pose}")
                pct = grp.loc[mask, "pct"].sum()
                rows.append({"run": run, "atom": p_atom, "pose": pose, "pct": pct})
    return pd.DataFrame(rows)


def _plot_marginal_histograms(df_marginal: pd.DataFrame, channel_type: str,
                               subunit: str, label: str, out_dir: Path,
                               perm_counts: dict = None):
    """
    For each P atom and pose, plot histogram of % across runs.
    Also plot overview bar with mean ± std.
    """
    poses  = ["N-term", "C-term", "none"]
    colors = {"N-term": "#1565C0", "C-term": "#B71C1C", "none": "#9E9E9E"}

    for p_atom in ["P4", "P5"]:
        df_p = df_marginal[df_marginal["atom"] == p_atom]

        # ── individual histograms per pose ────────────────────────────────────
        hist_dir = out_dir / f"marginal_{p_atom}"
        hist_dir.mkdir(exist_ok=True)

        for pose in poses:
            df_pose = df_p[df_p["pose"] == pose]
            pcts    = df_pose["pct"].tolist()
            if not pcts or max(pcts) == 0:
                continue
            mean_pct = np.mean(pcts)
            std_pct  = np.std(pcts)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(pcts, bins=10, color=colors[pose], edgecolor="white", alpha=0.85)
            ax.axvline(mean_pct, color="black", linestyle="--", linewidth=1.5,
                       label=f"Mean: {mean_pct:.1f}% ± {std_pct:.1f}%")
            ax.set_xlabel("% of frames", fontsize=15)
            ax.set_ylabel("Number of runs", fontsize=15)
            ax.set_title(f"{p_atom} {pose} — distribution across runs\n"
                         f"{channel_type} | Subunit {subunit} [{label}]", fontsize=14)
            ax.legend(fontsize=13)
            ax.tick_params(labelsize=14)
            plt.tight_layout()
            fig.savefig(hist_dir / f"hist_{pose}.png", dpi=120)
            plt.close(fig)

        # ── per-run bar per pose ───────────────────────────────────────────────
        for pose in poses:
            df_pose = df_p[df_p["pose"] == pose].copy()
            df_pose["_sk"] = df_pose["run"].str.extract(r"(\d+)").astype(int)
            df_pose = df_pose.sort_values("_sk").drop(columns="_sk")
            pcts    = df_pose["pct"].tolist()
            runs    = df_pose["run"].tolist()
            if not pcts or max(pcts) == 0:
                continue
            mean_pct = np.mean(pcts)
            std_pct  = np.std(pcts)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(np.arange(len(runs)), pcts, color=colors[pose],
                          edgecolor="white", alpha=0.85)
            ax.axhline(mean_pct, color="black", linestyle="--", linewidth=1.5,
                       label=f"Mean: {mean_pct:.1f}% ± {std_pct:.1f}%")
            if perm_counts:
                y_max = max(pcts) * 1.25 if pcts else 10
                for i, (run, pct) in enumerate(zip(runs, pcts)):
                    n_perm = perm_counts.get(run, None)
                    if n_perm is not None:
                        ax.text(i, pct + y_max * 0.02, str(n_perm),
                                ha="center", va="bottom", fontsize=14,
                                color="darkred", fontweight="bold", clip_on=False)
            ax.set_xticks(np.arange(len(runs)))
            ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=11)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_ylabel("% of frames", fontsize=15)
            ax.set_title(f"{p_atom} {pose} per run\n"
                         f"{channel_type} | Subunit {subunit} [{label}]", fontsize=15)
            ax.legend(fontsize=13)
            ax.set_ylim(0, min(120, max(pcts)*1.4) if pcts else 10)
            plt.tight_layout()
            fig.savefig(hist_dir / f"per_run_{pose}.png", dpi=120)
            plt.close(fig)

        # ── overview: mean ± std for shallow/deep/none ────────────────────────
        means = [df_p[df_p["pose"]==p]["pct"].mean() for p in poses]
        stds  = [df_p[df_p["pose"]==p]["pct"].std()  for p in poses]
        max_val = max(m+s for m,s in zip(means,stds)) if means else 1

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(np.arange(len(poses)), means, yerr=stds,
               color=[colors[p] for p in poses],
               edgecolor="white", alpha=0.85, capsize=5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m+s+max_val*0.01, f"{m:.1f}%",
                    ha="center", va="bottom", fontsize=13)
        ax.set_xticks(np.arange(len(poses)))
        ax.set_xticklabels(poses, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("% of frames (mean ± std)", fontsize=14)
        ax.set_title(f"{p_atom} pose — {channel_type}\n"
                     f"Subunit {subunit} [{label}]", fontsize=14)
        ax.set_ylim(0, min(115, max_val*1.3))
        plt.tight_layout()
        fig.savefig(out_dir / f"marginal_{p_atom}_overview.png", dpi=120)
        plt.close(fig)

def _run_for_subset(runs_dir: Path, out_dir: Path,
                    channel_type: str, subset: str = None,
                    perm_counts: dict = None):
    label    = subset or "all frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{label}]  →  {out_dir}")
    print(f"{'='*60}")

    for sub in SUBUNIT_ORDER:
        df = _load_per_run_data(runs_dir, sub, subset)
        if df.empty:
            print(f"  Subunit {sub}: no data found, skipping.")
            continue

        n_runs = df["run"].nunique()
        print(f"\n  Subunit {sub}: {n_runs} runs")

        sub_dir = out_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)

        # save per-run CSV
        pivot = (df.pivot_table(index="run", columns="state", values="pct", fill_value=0)
                   .reset_index())
        pivot.to_csv(sub_dir / "per_run_pct.csv", index=False)

        # overview plot: mean ± std per state
        _plot_all_states_overview(df, channel_type, sub, label,
                                  sub_dir / "overview_mean_std.png")

        # per-state plots
        states_dir = sub_dir / "per_state"
        states_dir.mkdir(exist_ok=True)
        hist_dir   = sub_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)

        for state in ALL_STATES:
            df_st = df[df["state"] == state]
            if df_st["pct"].sum() == 0:
                continue
            slug = STATE_LABEL(state).replace("+", "_")
            _plot_per_run_bars(df, state, channel_type, sub, label,
                               states_dir / f"per_run_{slug}.png",
                               perm_counts=perm_counts)
            _plot_histogram(df, state, channel_type, sub, label,
                            hist_dir / f"hist_{slug}.png")

        # marginal P4/P5 histograms
        df_marginal = _compute_marginal_per_run(df)
        df_marginal.to_csv(sub_dir / "marginal_per_run.csv", index=False)
        _plot_marginal_histograms(df_marginal, channel_type, sub, label, sub_dir,
                                  perm_counts=perm_counts)
        print(f"    Subunit {sub}: saved to {sub_dir}")


def run_plot_run_variability(
    runs_dir:     str,
    channel_type: str,
    output_dir:   str = None,
):
    runs_dir = Path(runs_dir)
    if output_dir is None:
        output_dir = runs_dir / "pip_variability"
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PIP VARIABILITY ANALYSIS")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60)

    perm_counts = _load_permeation_counts(runs_dir)

    # all frames
    _run_for_subset(runs_dir, output_dir, channel_type, subset=None, perm_counts=perm_counts)

    # check end2/end3
    run_dirs = _sorted_run_dirs(runs_dir)
    for subset in ["end2_frames", "end3_frames"]:
        has = any(
            (rd / "pip_pose" / subset / "subunit_2a" / "summary.csv").exists()
            for rd in run_dirs
        )
        if has:
            _run_for_subset(runs_dir, output_dir / subset,
                            channel_type, subset=subset, perm_counts=perm_counts)

    print(f"\n✓ Run variability analysis complete → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",     required=True)
    parser.add_argument("--channel_type", required=True)
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    run_plot_run_variability(
        runs_dir     = args.runs_dir,
        channel_type = args.channel_type,
        output_dir   = args.output_dir,
    )

'''
python3 plot_pip_variability.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12

python3 plot_pip_variability.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML

python3 plot_pip_variability.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT
'''