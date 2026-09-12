"""
hbond_causality_analysis.py

Detects temporal coupling between H-bond partner timeseries of two residues.
For each of the 4 column-pair cases, extracts per-partner binary timeseries,
computes pairwise cross-correlations, detects formation/breaking events,
and reports lag times.

Usage:
    python3 hbond_causality_analysis.py \
        --resA /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/197_2b/hbonds\
        --resB /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/199_2b/hbonds\
        --labelA GLN197.2b \
        --labelB LYS199.2b \
        --lag 500 \
        --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/hbond_causality_analysis/197_2b_199_2b
"""


import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

# ── Column pairs for the 4 cases ────────────────────────────────────────────
CASES = [
    ("sc_any_partners", "sc_any_partners"),
    ("sc_any_partners", "sc_sc_partners"),
    ("sc_sc_partners",  "sc_any_partners"),
    ("sc_sc_partners",  "sc_sc_partners"),
]

CASE_LABELS = [
    "scAny_vs_scAny",
    "scAny_vs_scSc",
    "scSc_vs_scAny",
    "scSc_vs_scSc",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_hbond(folder):
    path = os.path.join(folder, "hbond_per_frame.csv")
    df = pd.read_csv(path)
    # fill NaN partner columns with empty string
    for col in ["sc_any_partners", "sc_sc_partners", "bb_bb_partners", "bb_sc_partners", "partners"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def extract_partner_timeseries(df, col):
    """Return dict: partner_name -> binary np.array (1 = bonded that frame)."""
    n_frames = len(df)
    all_partners = set()
    for cell in df[col]:
        if cell:
            all_partners.update(cell.split("|"))
    all_partners.discard("")

    series = {}
    for partner in sorted(all_partners):
        ts = np.zeros(n_frames, dtype=np.int8)
        for i, cell in enumerate(df[col]):
            if partner in cell.split("|") if cell else []:
                ts[i] = 1
        series[partner] = ts
    return series



def bond_stats(ts):
    """
    Compute stability metrics for a binary timeseries:
    - total_on_frames: total frames bond is present
    - occupancy: fraction of simulation bond is present
    - n_transitions: total number of form+break events
    - mean_on_duration: mean continuous ON duration (frames)
    - max_on_duration: longest continuous ON stretch (frames)
    - mean_off_duration: mean continuous OFF duration (frames)
    - event_type: 'single_switch', 'stable_on', 'stable_off', 'flickering', 'oscillating'
    """
    n = len(ts)
    total_on = int(ts.sum())
    occupancy = round(total_on / n, 3)

    # find runs of ON and OFF
    def run_lengths(val):
        lengths = []
        in_run = False
        count = 0
        for x in ts:
            if x == val:
                in_run = True
                count += 1
            else:
                if in_run:
                    lengths.append(count)
                in_run = False
                count = 0
        if in_run:
            lengths.append(count)
        return lengths

    on_runs  = run_lengths(1)
    off_runs = run_lengths(0)

    n_on_runs  = len(on_runs)
    n_off_runs = len(off_runs)
    n_transitions = n_on_runs + n_off_runs - 1  # approximate

    mean_on  = round(np.mean(on_runs),  1) if on_runs  else 0
    max_on   = max(on_runs)  if on_runs  else 0
    mean_off = round(np.mean(off_runs), 1) if off_runs else 0

    # classify event type
    if n_on_runs <= 1 and occupancy > 0.1:
        etype = "single_switch_or_stable"
    elif occupancy > 0.95:
        etype = "stable_on"
    elif occupancy < 0.05:
        etype = "stable_off"
    elif n_on_runs > 20 and mean_on < 50:
        etype = "flickering"
    else:
        etype = "oscillating"

    return {
        "total_on_frames": total_on,
        "occupancy": occupancy,
        "n_on_runs": n_on_runs,
        "mean_on_duration": mean_on,
        "max_on_duration": max_on,
        "mean_off_duration": mean_off,
        "event_type": etype,
    }

def cross_correlation(a, b, max_lag):
    """
    Normalized cross-correlation of two binary timeseries at lags -max_lag..+max_lag.
    Positive lag τ means b follows a by τ frames (a leads b).
    """
    a = a - a.mean()
    b = b - b.mean()
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags))
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return lags, cc

    for i, lag in enumerate(lags):
        if lag >= 0:
            if n - lag > 0:
                cc[i] = np.dot(a[:n-lag], b[lag:]) / denom
        else:
            lag_abs = -lag
            if n - lag_abs > 0:
                cc[i] = np.dot(a[lag_abs:], b[:n-lag_abs]) / denom
    return lags, cc


def detect_transitions(ts):
    """Return (on_frames, off_frames) where bond forms and breaks."""
    on_frames  = np.where(np.diff(ts.astype(int)) == 1)[0] + 1   # 0→1
    off_frames = np.where(np.diff(ts.astype(int)) == -1)[0] + 1  # 1→0
    return on_frames, off_frames


def event_triggered_lag(ts_a, ts_b, event_a, event_b, max_lag):
    """
    For each event_a transition in ts_a, measure frames until event_b in ts_b.
    event_a / event_b: 'forms' (0->1) or 'breaks' (1->0)
    Returns list of lags (positive = b follows a).
    """
    sig_a = 1 if event_a == "forms" else -1
    sig_b = 1 if event_b == "forms" else -1
    events_a = np.where(np.diff(ts_a.astype(int)) == sig_a)[0] + 1
    events_b = np.where(np.diff(ts_b.astype(int)) == sig_b)[0] + 1

    lags = []
    for ea in events_a:
        candidates = events_b[(events_b > ea) & (events_b <= ea + max_lag)]
        if len(candidates) > 0:
            lags.append(candidates[0] - ea)
    return lags


# ── Per-case analysis ────────────────────────────────────────────────────────

def analyze_case(df_a, df_b, col_a, col_b, label_a, label_b, max_lag, out_dir, case_label, same_residue=False):
    ts_a = extract_partner_timeseries(df_a, col_a)
    ts_b = extract_partner_timeseries(df_b, col_b)

    if not ts_a or not ts_b:
        print(f"  [{case_label}] No partners found in one or both columns — skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for (pA, serA), (pB, serB) in product(ts_a.items(), ts_b.items()):
        if same_residue and pA == pB:
            continue
        lags, cc = cross_correlation(serA, serB, max_lag)
        peak_idx = np.argmax(np.abs(cc))
        peak_lag = lags[peak_idx]
        peak_r   = cc[peak_idx]

        # direction interpretation
        if peak_lag > 0:
            direction = f"{label_a}↔{pA}  LEADS  {label_b}↔{pB}  by {peak_lag} frames"
        elif peak_lag < 0:
            direction = f"{label_b}↔{pB}  LEADS  {label_a}↔{pA}  by {abs(peak_lag)} frames"
        else:
            direction = "simultaneous"

        # all 8 event combinations
        combos = [
            ("forms",  "forms"),
            ("forms",  "breaks"),
            ("breaks", "forms"),
            ("breaks", "breaks"),
        ]
        event_results = {}
        for (ea, eb) in combos:
            # A->B
            key_ab = f"A_{ea}_B_{eb}"
            l_ab = event_triggered_lag(serA, serB, ea, eb, max_lag)
            event_results[f"{key_ab}_mean_lag"] = round(np.mean(l_ab), 1) if l_ab else ""
            event_results[f"{key_ab}_n_events"]  = len(l_ab)
            # B->A
            key_ba = f"B_{ea}_A_{eb}"
            l_ba = event_triggered_lag(serB, serA, ea, eb, max_lag)
            event_results[f"{key_ba}_mean_lag"] = round(np.mean(l_ba), 1) if l_ba else ""
            event_results[f"{key_ba}_n_events"]  = len(l_ba)

        statsA = bond_stats(serA)
        statsB = bond_stats(serB)

        row = {
            "partnerA": pA,
            "partnerB": pB,
            "peak_lag_frames": peak_lag,
            "peak_r": round(peak_r, 4),
            "direction": direction,
            # bond A stability
            "A_occupancy": statsA["occupancy"],
            "A_n_on_runs": statsA["n_on_runs"],
            "A_mean_on_duration": statsA["mean_on_duration"],
            "A_max_on_duration": statsA["max_on_duration"],
            "A_mean_off_duration": statsA["mean_off_duration"],
            "A_event_type": statsA["event_type"],
            # bond B stability
            "B_occupancy": statsB["occupancy"],
            "B_n_on_runs": statsB["n_on_runs"],
            "B_mean_on_duration": statsB["mean_on_duration"],
            "B_max_on_duration": statsB["max_on_duration"],
            "B_mean_off_duration": statsB["mean_off_duration"],
            "B_event_type": statsB["event_type"],
        }
        row.update(event_results)
        row["cc_lags"]   = lags.tolist()
        row["cc_values"] = cc.tolist()
        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("peak_r", key=abs, ascending=False)

    # Save CSV (without the raw cc arrays)
    csv_cols = [c for c in df_out.columns if c not in ("cc_lags", "cc_values")]
    csv_path = os.path.join(out_dir, f"{case_label}_results.csv")
    df_out[csv_cols].to_csv(csv_path, index=False)
    print(f"  [{case_label}] Saved CSV: {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    # Top-10 pairs by |peak_r|
    top = df_out.head(min(10, len(df_out)))

    # 1. Cross-correlation curves for top pairs
    n_top = len(top)
    ncols = 2
    nrows = (n_top + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        lags_arr = np.array(row["cc_lags"])
        cc_arr   = np.array(row["cc_values"])
        ax.plot(lags_arr, cc_arr, color="#2c5f8a", lw=1.5)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(row["peak_lag_frames"], color="#c0392b", lw=1.2, ls="--",
                   label=f"peak lag={row['peak_lag_frames']} (r={row['peak_r']:.2f})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{label_a}↔{row['partnerA']}\nvs {label_b}↔{row['partnerB']}", fontsize=9)
        ax.set_xlabel("Lag (frames)", fontsize=8)
        ax.set_ylabel("Cross-corr", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Cross-correlation: {case_label}\n{label_a} vs {label_b}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{case_label}_crosscorr.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [{case_label}] Saved cross-corr plot: {plot_path}")

    # 2. Event lag distribution — A breaks → B forms
    pairs_with_events = top[top["A_breaks_B_forms_n_events"] > 0]
    if not pairs_with_events.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        pair_labels = [f"{r['partnerA']}→{r['partnerB']}" for _, r in pairs_with_events.iterrows()]
        mean_lags   = [r["A_breaks_B_forms_mean_lag"] for _, r in pairs_with_events.iterrows()]
        n_events    = [r["A_breaks_B_forms_n_events"] for _, r in pairs_with_events.iterrows()]
        bars = ax2.bar(pair_labels, mean_lags, color="#e67e22", edgecolor="black", linewidth=0.7)
        for bar, n in zip(bars, n_events):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"n={n}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("Mean lag (frames)", fontsize=10)
        ax2.set_title(f"{label_a} bond BREAKS → {label_b} bond FORMS\n{case_label}", fontsize=10)
        ax2.tick_params(axis="x", rotation=35, labelsize=8)
        plt.tight_layout()
        plot2_path = os.path.join(out_dir, f"{case_label}_event_lag_AbreaksBforms.png")
        plt.savefig(plot2_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{case_label}] Saved event lag plot: {plot2_path}")

    # 3. Binary timeseries for top-5 pairs
    top5 = top.head(5)
    fig3, axes3 = plt.subplots(len(top5), 1, figsize=(16, len(top5) * 2.2), sharex=True)
    if len(top5) == 1:
        axes3 = [axes3]
    for ax, (_, row) in zip(axes3, top5.iterrows()):
        serA = ts_a[row["partnerA"]].astype(float)
        serB = ts_b[row["partnerB"]].astype(float) * 0.8  # offset for visibility
        frames = np.arange(len(serA))
        ax.fill_between(frames, 0, serA, step="mid", alpha=0.6, color="#2c5f8a",
                        label=f"{label_a}↔{row['partnerA']}")
        ax.fill_between(frames, 0, -serB, step="mid", alpha=0.6, color="#c0392b",
                        label=f"{label_b}↔{row['partnerB']}")
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel("Bond state", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="black", lw=0.5)
        lag_txt = f"peak lag={row['peak_lag_frames']} frames, r={row['peak_r']:.2f}"
        ax.set_title(lag_txt, fontsize=8)

    axes3[-1].set_xlabel("Frame", fontsize=9)
    fig3.suptitle(f"Binary timeseries (top 5 pairs): {case_label}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plot3_path = os.path.join(out_dir, f"{case_label}_timeseries.png")
    plt.savefig(plot3_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [{case_label}] Saved timeseries plot: {plot3_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H-bond temporal coupling between two residues")
    parser.add_argument("--resA",   required=True, help="Path to hbonds folder of residue A (e.g. 197_2b/hbonds)")
    parser.add_argument("--resB",   required=True, help="Path to hbonds folder of residue B (e.g. 199_2b/hbonds)")
    parser.add_argument("--labelA", default="ResA", help="Label for residue A (e.g. GLN197.2b)")
    parser.add_argument("--labelB", default="ResB", help="Label for residue B (e.g. LYS199.2b)")
    parser.add_argument("--lag",    type=int, default=100, help="Max lag in frames (default 100)")
    parser.add_argument("--same_residue", action="store_true",
                        help="Set if resA and resB are the same residue — filters out self-pairs")
    parser.add_argument("--out",    default="./temporal_coupling_output", help="Output directory")
    args = parser.parse_args()

    print(f"\nLoading H-bond data...")
    df_a = load_hbond(args.resA)
    df_b = load_hbond(args.resB)
    print(f"  {args.labelA}: {len(df_a)} frames")
    print(f"  {args.labelB}: {len(df_b)} frames")

    if len(df_a) != len(df_b):
        print(f"  WARNING: frame counts differ ({len(df_a)} vs {len(df_b)}). Truncating to min.")
        n = min(len(df_a), len(df_b))
        df_a = df_a.iloc[:n].reset_index(drop=True)
        df_b = df_b.iloc[:n].reset_index(drop=True)

    print(f"\nRunning 4 cases with max_lag={args.lag} frames...\n")

    for (col_a, col_b), case_label in zip(CASES, CASE_LABELS):
        print(f"Case: {case_label}  ({col_a} vs {col_b})")
        case_out = os.path.join(args.out, case_label)
        analyze_case(df_a, df_b, col_a, col_b,
                     args.labelA, args.labelB,
                     args.lag, case_out, case_label,
                     same_residue=args.same_residue)
        print()

    print("Done.")


if __name__ == "__main__":
    main()