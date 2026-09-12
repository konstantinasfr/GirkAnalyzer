"""
switch_detector.py

Detects committed conformational switches in H-bond timeseries across multiple residues.
A "switch" is a permanent (or long-lasting) transition from one state to another,
ignoring short fluctuations below a minimum duration threshold.

Usage:
python3 switch_detector.py \
    --folders /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/197_2b/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/199_2b/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/315_2a/hbonds \
    --labels GLN197.2b LYS199.2b GLU315.2a \
    --occ_threshold 0.2 \
    --min_window 200 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/switch_output/197_2b_199_2b_315_2a

python3 switch_detector.py \
    --folders /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/197_2c/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/199_2c/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/315_2b/hbonds \
    --labels GLN197.2c LYS199.2c GLU315.2b\
    --occ_threshold 0.2 \
    --min_window 200 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/switch_output/197_2c_199_2c_315_2b


python3 switch_detector.py \
    --folders /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/197_2a/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/199_2a/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/304_G1/hbonds \
    --labels GLN197.2a LYS199.2a GLU304.G1\
    --occ_threshold 0.2 \
    --min_window 200 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/switch_output/197_2a_199_2a_304_G1

python3 switch_detector.py \
    --folders /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/186_G1/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/188_G1/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/315_2c/hbonds \
    --labels GLN186.G1 LYS188.G1 GLU315.2c\
    --occ_threshold 0.2 \
    --min_window 200 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/switch_output/186_G1_188_G1_315_2c

"""
"""
switch_detector.py

Detects committed conformational switches in H-bond timeseries across multiple residues.
A "switch" is a permanent (or long-lasting) transition from one state to another,
ignoring short fluctuations below a minimum duration threshold.

Usage:
python3 switch_detector.py \
    --folders /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/197_2b/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/199_2b/hbonds \
             /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/residue_neighbors/315_2a/hbonds \
    --labels GLN197.2b LYS199.2b GLU315.2a \
    --occ_threshold 0.5 \
    --min_window 200 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN19/switch_output/197_2b_199_2b_315_2a
"""

"""
hbond_existence_windows.py

For each H-bond partner of each residue, finds continuous windows
where occupancy >= occ_threshold. Outputs existence intervals and plots.
Runs 3 analyses automatically: all_interactions, sc_any_only, sc_sc_only.

Usage:
python3 hbond_existence_windows.py \
    --folders /path/197_2b/hbonds /path/199_2b/hbonds \
    --labels GLN197.2b LYS199.2b \
    --occ_threshold 0.2 \
    --min_window 200 \
    --out ./existence_output
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ANALYSES = {
    "all_interactions": ["sc_any_partners", "sc_sc_partners", "bb_sc_partners", "bb_bb_partners"],
    "sc_any_only":      ["sc_any_partners"],
    "sc_sc_only":       ["sc_sc_partners"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_hbond(folder):
    path = os.path.join(folder, "hbond_per_frame.csv")
    df = pd.read_csv(path)
    for c in ["sc_any_partners", "sc_sc_partners", "bb_sc_partners", "bb_bb_partners"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def extract_partner_timeseries(df, col):
    all_partners = set()
    for cell in df[col]:
        if cell:
            all_partners.update(cell.split("|"))
    all_partners.discard("")
    series = {}
    for partner in sorted(all_partners):
        ts = np.zeros(len(df), dtype=np.int8)
        for i, cell in enumerate(df[col]):
            if cell and partner in cell.split("|"):
                ts[i] = 1
        series[partner] = ts
    return series


def smooth_ts(ts, min_duration):
    """Fill short OFF gaps and remove short ON bursts."""
    ts = ts.copy().astype(int)
    i = 0
    while i < len(ts):
        if ts[i] == 0:
            j = i
            while j < len(ts) and ts[j] == 0:
                j += 1
            if (j - i) < min_duration and i > 0 and j < len(ts):
                ts[i:j] = 1
            i = j
        else:
            i += 1
    i = 0
    while i < len(ts):
        if ts[i] == 1:
            j = i
            while j < len(ts) and ts[j] == 1:
                j += 1
            if (j - i) < min_duration:
                ts[i:j] = 0
            i = j
        else:
            i += 1
    return ts


def find_existence_windows(ts, occ_threshold, min_window):
    ts_smooth = smooth_ts(ts, min_window)
    blocks = []
    i = 0
    while i < len(ts_smooth):
        if ts_smooth[i] == 1:
            j = i
            while j < len(ts_smooth) and ts_smooth[j] == 1:
                j += 1
            block_len = j - i
            if block_len >= min_window:
                occ = round(float(ts[i:j].mean()), 3)
                if occ >= occ_threshold:
                    blocks.append((i, j, occ, block_len))
            i = j
        else:
            i += 1
    return blocks


# ── Per-analysis run ──────────────────────────────────────────────────────────

def run_analysis(folders, labels, cols, occ_threshold, min_window, occ_change_threshold, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    all_ts = {}
    n_frames = None

    for folder, label in zip(folders, labels):
        print(f"  Processing {label}...")
        df = load_hbond(folder)
        if n_frames is None:
            n_frames = len(df)

        seen_partners = set()
        for col in cols:
            if col not in df.columns:
                continue
            ts_dict = extract_partner_timeseries(df, col)
            for partner, ts in ts_dict.items():
                if partner in seen_partners:
                    continue
                seen_partners.add(partner)

                bond_key = f"{label}↔{partner}"
                all_ts[bond_key] = ts
                blocks = find_existence_windows(ts, occ_threshold, min_window)

                if blocks:
                    for (start, end, occ, win_size) in blocks:
                        rows.append({
                            "bond": bond_key, "residue": label, "partner": partner, "col": col,
                            "exist_start": start, "exist_end": end, "exist_occ": occ,
                            "window_size": win_size, "global_occ": round(float(ts.mean()), 3),
                            "found": True,
                        })
                        print(f"    {bond_key}: frames {start}–{end} (occ={occ}, window={win_size})")
                else:
                    rows.append({
                        "bond": bond_key, "residue": label, "partner": partner, "col": col,
                        "exist_start": None, "exist_end": None, "exist_occ": None,
                        "window_size": None, "global_occ": round(float(ts.mean()), 3),
                        "found": False,
                    })

    df_out = pd.DataFrame(rows)
    df_found = df_out[df_out["found"]].copy()

    # Deduplicate: A↔B and B↔A are the same bond — keep first occurrence
    def canonical_bond(bond):
        parts = bond.split("↔")
        return "↔".join(sorted(parts))
    df_found["canonical"] = df_found["bond"].apply(canonical_bond)
    df_found = df_found.drop_duplicates(subset=["canonical", "exist_start", "exist_end"])
    df_found = df_found.drop(columns=["canonical"]).sort_values("exist_start")

    # Merge multiple windows of the same bond into one row with list of segments
    merged_rows = []
    for bond, group in df_found.groupby("bond", sort=False):
        group = group.sort_values("exist_start")
        segments = list(zip(group["exist_start"], group["exist_end"], group["exist_occ"]))
        row = group.iloc[0].to_dict()
        row["exist_start"] = group["exist_start"].min()
        row["exist_end"]   = group["exist_end"].max()
        row["segments"]    = segments
        row["n_segments"]  = len(segments)
        merged_rows.append(row)
    df_plot = pd.DataFrame(merged_rows).sort_values("exist_start")

    df_out.to_csv(os.path.join(out_dir, "existence_windows_all.csv"), index=False)
    df_found.to_csv(os.path.join(out_dir, "existence_windows_found.csv"), index=False)
    print(f"  Saved: {out_dir} ({len(df_found)} bonds with windows)")

    if df_found.empty:
        print("  No bonds found. Lower --occ_threshold or --min_window.")
        return

    # ── Character switch detection ────────────────────────────────────────────
    char_rows = {}
    for bond, group in df_found.groupby("bond"):
        ts = all_ts.get(bond)
        if ts is None:
            continue
        start = int(group["exist_start"].min())
        end   = int(group["exist_end"].max())
        mid   = (start + end) // 2
        occ_first  = round(float(ts[start:mid].mean()), 3) if mid > start else 0
        occ_second = round(float(ts[mid:end].mean()),  3) if end > mid   else 0
        delta = round(abs(occ_first - occ_second), 3)
        if delta >= occ_change_threshold:
            direction = "weakens" if occ_second < occ_first else "strengthens"
            char_rows[bond] = {
                "occ_first_half":   occ_first,
                "occ_second_half":  occ_second,
                "occ_change":       delta,
                "character_switch": f"{direction} ({occ_first}\u2192{occ_second})",
            }
        else:
            char_rows[bond] = {
                "occ_first_half":   occ_first,
                "occ_second_half":  occ_second,
                "occ_change":       delta,
                "character_switch": "",
            }

    for col_name in ["occ_first_half", "occ_second_half", "occ_change", "character_switch"]:
        df_found[col_name] = df_found["bond"].map(
            {b: v[col_name] for b, v in char_rows.items()}
        )
    # carry character switch into df_plot
    for col_name in ["occ_first_half", "occ_second_half", "occ_change", "character_switch"]:
        df_plot[col_name] = df_plot["bond"].map(
            {b: v[col_name] for b, v in char_rows.items()}
        ).fillna("")

    df_found.to_csv(os.path.join(out_dir, "existence_windows_found.csv"), index=False)

    flagged = [b for b, v in char_rows.items() if v["character_switch"]]
    if flagged:
        print(f"\n  CHARACTER SWITCHES DETECTED (delta >= {occ_change_threshold}):")
        for b in flagged:
            print(f"    {b}: {char_rows[b]['character_switch']}")

    residues = df_found["residue"].unique()
    color_map = {r: c for r, c in zip(residues,
                 ["#2c5f8a", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"])}

    # ── Gantt chart ───────────────────────────────────────────────────────────
    n_bonds = len(df_plot)
    fig, ax = plt.subplots(figsize=(16, max(4, n_bonds * 0.5 + 2)))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        color = color_map.get(row["residue"], "#555")
        segments = row["segments"]
        total_start = row["exist_start"]
        total_end   = row["exist_end"]
        mid = (total_start + total_end) / 2
        char_switch = str(row.get("character_switch", ""))

        # NO background span — gaps are white

        # draw each active segment
        for (seg_start, seg_end, seg_occ) in segments:
            if char_switch:
                # split at midpoint: pale+hatched for weaker half
                first_end    = min(seg_end, mid)
                second_start = max(seg_start, mid)
                if "weakens" in char_switch:
                    # first half strong, second half pale+hatched
                    if first_end > seg_start:
                        ax.barh(i, first_end - seg_start, left=seg_start, height=0.6,
                                color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
                    if second_start < seg_end:
                        ax.barh(i, seg_end - second_start, left=second_start, height=0.6,
                                color=color, alpha=0.3, edgecolor="black", linewidth=0.5,
                                hatch="///")
                else:
                    # strengthens: first half pale+hatched, second half strong
                    if first_end > seg_start:
                        ax.barh(i, first_end - seg_start, left=seg_start, height=0.6,
                                color=color, alpha=0.3, edgecolor="black", linewidth=0.5,
                                hatch="///")
                    if second_start < seg_end:
                        ax.barh(i, seg_end - second_start, left=second_start, height=0.6,
                                color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
                # dotted line + label at midpoint
                ax.axvline(mid, ymin=i/n_bonds, ymax=(i+0.8)/n_bonds,
                           color="black", lw=1.2, ls=":", alpha=0.7)
                occ1 = row.get("occ_first_half", "")
                occ2 = row.get("occ_second_half", "")
                arrow = "\u2193" if "weakens" in char_switch else "\u2191"
                ax.text(mid, i + 0.65, f"{arrow}{occ1}\u2192{occ2}",
                        ha="center", va="bottom", fontsize=6,
                        color="black", fontweight="bold")
            else:
                ax.barh(i, seg_end - seg_start, left=seg_start, height=0.6,
                        color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

        label = f"occ={row['exist_occ']} (win={row['window_size']})"
        if row["n_segments"] > 1:
            label += f" [{row['n_segments']} windows]"
        if char_switch:
            label += f"  {char_switch}"
        ax.text(total_end + 20, i, label, va="center", fontsize=7, color="#333")

    ax.set_yticks(range(n_bonds))
    ax.set_yticklabels(df_plot["bond"].tolist(), fontsize=8)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_xlim(0, n_frames + 200)
    ax.set_title(f"H-bond Existence Windows\nocc_threshold={occ_threshold}, min_window={min_window}",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=color_map[r], label=r) for r in residues],
              fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "existence_gantt.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ── Timeseries with highlighted windows ───────────────────────────────────
    frames = np.arange(n_frames)
    n_cols = 2
    n_rows = (n_bonds + 1) // 2
    fig2, axes = plt.subplots(n_rows, n_cols,
                              figsize=(16, max(4, n_rows * 2.2)), sharex=True)
    axes = np.array(axes).flatten()
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax = axes[i]
        ts = all_ts.get(row["bond"])
        if ts is None:
            ax.set_visible(False)
            continue
        color = color_map.get(row["residue"], "#555")
        ax.fill_between(frames, 0, ts, step="mid", alpha=0.4, color=color)
        for (seg_start, seg_end, seg_occ) in row["segments"]:
            ax.axvspan(seg_start, seg_end, alpha=0.25, color="gold")
            ax.axvline(seg_start, color="green", lw=1.2, ls="--")
            ax.axvline(seg_end,   color="red",   lw=1.2, ls="--")
        ax.set_ylim(-0.1, 1.3)
        seg_label = " | ".join([f"[{s}–{e}]" for s,e,_ in row["segments"]])
        ax.set_title(f"{row['bond']}  {seg_label}  occ={row['exist_occ']}", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    axes[-2].set_xlabel("Frame", fontsize=9)
    axes[-1].set_xlabel("Frame", fontsize=9)
    fig2.suptitle("H-bond timeseries with existence windows highlighted",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "existence_timeseries.png"), dpi=120, bbox_inches="tight")
    plt.close()


    # ── Switch mechanism summary ──────────────────────────────────────────────
    MAX_LAG = 800  # max frames between events to report as a pair

    # Build event list: each bond contributes up to 5 events
    # start, stop, and if character switch:
    #   weaken/strengthen at midpoint
    #   weaken also acts as stop (bond weakening = effective end of strong state)
    #   strengthen also acts as start (bond strengthening = effective start of strong state)
    events = []
    for _, row in df_plot.iterrows():
        bond = row["bond"]
        char = str(row.get("character_switch", ""))
        segs = row["segments"]
        start = int(segs[0][0])
        end   = int(segs[-1][1])
        mid   = (int(row["exist_start"]) + int(row["exist_end"])) // 2

        events.append({"bond": bond, "event_type": "start", "frame": start, "character_switch": char})
        events.append({"bond": bond, "event_type": "stop",  "frame": end,   "character_switch": char})
        if "weaken" in char:
            # weaken at midpoint — as itself
            events.append({"bond": bond, "event_type": "weaken", "frame": mid, "character_switch": char})
            # weaken also acts as stop (strong state ends at midpoint)
            events.append({"bond": bond, "event_type": "stop",   "frame": mid, "character_switch": char})
        elif "strengthen" in char:
            # strengthen at midpoint — as itself
            events.append({"bond": bond, "event_type": "strengthen", "frame": mid, "character_switch": char})
            # strengthen also acts as start (strong state begins at midpoint)
            events.append({"bond": bond, "event_type": "start", "frame": mid, "character_switch": char})

    df_events = pd.DataFrame(events).sort_values("frame").reset_index(drop=True)
    df_events.to_csv(os.path.join(out_dir, "switch_mechanism_transitions.csv"), index=False)

    # Build all pairwise combinations within MAX_LAG
    pair_rows = []
    event_type_map = {
        ("stop",       "start"):      "stop_start",
        ("start",      "start"):      "start_start",
        ("stop",       "stop"):       "stop_stop",
        ("start",      "stop"):       "start_stop",
        ("weaken",     "start"):      "weaken_start",
        ("weaken",     "stop"):       "weaken_stop",
        ("strengthen", "start"):      "strengthen_start",
        ("strengthen", "stop"):       "strengthen_stop",
        ("weaken",     "strengthen"): "weaken_strengthen",
        ("strengthen", "weaken"):     "strengthen_weaken",
        ("weaken",     "weaken"):     "weaken_weaken",
        ("strengthen", "strengthen"): "strengthen_strengthen",
        ("start",      "weaken"):     "start_weaken",
        ("start",      "strengthen"): "start_strengthen",
        ("stop",       "weaken"):     "stop_weaken",
        ("stop",       "strengthen"): "stop_strengthen",
    }

    for i, ea in df_events.iterrows():
        for j, eb in df_events.iterrows():
            if i >= j:
                continue
            if ea["bond"] == eb["bond"]:
                continue  # skip same bond pairs
            lag = eb["frame"] - ea["frame"]
            if abs(lag) > MAX_LAG:
                continue
            pair_type = event_type_map.get((ea["event_type"], eb["event_type"]), "other")
            pair_rows.append({
                "bond_A":      ea["bond"],
                "event_A":     ea["event_type"],
                "frame_A":     ea["frame"],
                "bond_B":      eb["bond"],
                "event_B":     eb["event_type"],
                "frame_B":     eb["frame"],
                "lag_frames":  lag,
                "pair_type":   pair_type,
                "char_A":      ea["character_switch"],
                "char_B":      eb["character_switch"],
            })

    df_pairs = pd.DataFrame(pair_rows).sort_values("frame_A")
    df_pairs.to_csv(os.path.join(out_dir, "switch_mechanism_pairs.csv"), index=False)

    # ── TXT summary ───────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("SWITCH MECHANISM SUMMARY")
    lines.append(f"occ_threshold={occ_threshold}  min_window={min_window}  max_lag={MAX_LAG}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("CHRONOLOGICAL EVENTS:")
    lines.append(f"  {'bond':<40} {'event':<12} {'frame':>6}")
    lines.append(f"  {'-'*60}")
    for _, row in df_events.iterrows():
        char = f"  [{row['character_switch']}]" if row["character_switch"] else ""
        lines.append(f"  {row['bond']:<40} {row['event_type']:<12} {row['frame']:>6}{char}")
    lines.append("")

    char_bonds = [(b, v) for b, v in char_rows.items() if v["character_switch"]]
    if char_bonds:
        lines.append("CHARACTER SWITCHES:")
        for bond, v in char_bonds:
            lines.append(f"  {bond}: {v['character_switch']}  (delta={v['occ_change']})")
        lines.append("")

    lines.append(f"SWITCH PAIRS (lag <= {MAX_LAG} frames):")
    lines.append(f"  {'bond_A':<35} {'evt_A':<12} {'frame_A':>7}  {'bond_B':<35} {'evt_B':<12} {'frame_B':>7}  {'lag':>6}  pair_type")
    lines.append(f"  {'-'*130}")
    for _, row in df_pairs.iterrows():
        lines.append(
            f"  {row['bond_A']:<35} {row['event_A']:<12} {row['frame_A']:>7}  "
            f"{row['bond_B']:<35} {row['event_B']:<12} {row['frame_B']:>7}  "
            f"{row['lag_frames']:>6}  {row['pair_type']}"
        )
    lines.append("")
    lines.append("=" * 70)

    txt_path = os.path.join(out_dir, "switch_mechanism_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: switch_mechanism_pairs.csv  ({len(df_pairs)} pairs)")
    print(f"  Saved: switch_mechanism_summary.txt")
    print(f"  Saved: switch_mechanism_transitions.csv")

    # ── Relationship summary ──────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print("  EXISTENCE WINDOW RELATIONSHIPS:")
    print(f"  {'='*65}")
    bonds = df_plot.to_dict("records")
    for i, b1 in enumerate(bonds):
        for b2 in bonds[i+1:]:
            s1, e1 = b1["exist_start"], b1["exist_end"]
            s2, e2 = b2["exist_start"], b2["exist_end"]
            overlap = max(0, min(e1, e2) - max(s1, s2))
            total   = max(e1, e2) - min(s1, s2)
            if overlap > 0:
                rel = f"OVERLAP {overlap} frames ({overlap/total*100:.0f}%)"
            elif e1 < s2:
                rel = f"{b1['bond']} ENDS {s2-e1} frames BEFORE {b2['bond']} STARTS"
            else:
                rel = f"{b2['bond']} ENDS {s1-e2} frames BEFORE {b1['bond']} STARTS"
            print(f"  {b1['bond']}  vs  {b2['bond']}")
            print(f"    → {rel}")
    print(f"  {'='*65}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find H-bond existence windows")
    parser.add_argument("--folders",       nargs="+", required=True)
    parser.add_argument("--labels",        nargs="+", required=True)
    parser.add_argument("--occ_threshold", type=float, default=0.6)
    parser.add_argument("--min_window",    type=int,   default=100)
    parser.add_argument("--occ_change_threshold", type=float, default=0.3,
                        help="Min occupancy change between halves to flag character switch (default: 0.3)")
    parser.add_argument("--out",           default="./existence_output")
    args = parser.parse_args()

    if len(args.folders) != len(args.labels):
        raise ValueError("--folders and --labels must match")

    print(f"\nocc_threshold={args.occ_threshold}  min_window={args.min_window}\n")

    for analysis_name, cols in ANALYSES.items():
        print(f"{'='*60}")
        print(f"Analysis: {analysis_name}")
        print(f"{'='*60}")
        run_analysis(
            args.folders, args.labels, cols,
            args.occ_threshold, args.min_window,
            args.occ_change_threshold,
            os.path.join(args.out, analysis_name)
        )

    print("All done.")


if __name__ == "__main__":
    main()