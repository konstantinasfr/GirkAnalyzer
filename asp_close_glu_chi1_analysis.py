#!/usr/bin/env python3
"""
asp_close_glu_chi1_analysis.py
================================
Answers the question:

    "When ASP173.D is close to the ligand (distance < threshold),
     what chi1 conformational state are the GLU residues in — and as a %?"

Data sources (already computed by your pipeline):
    combined_distance_dihedrals.pkl  →  ASP173.D ligand distances per frame
    gmm_results.pkl                  →  chi1 GMM state label per frame per run

Logic:
    For every frame where asp_min_distance < threshold:
        for each GLU residue (152.A, 152.B, 152.C, 141.D):
            record its chi1 GMM state at that frame

    Then: percentage of ASP-close frames where each GLU was in each chi1 state.

Also computes the baseline (all frames, not just ASP-close) so you can see
whether the GLU chi1 distribution shifts when ASP is engaged with the ligand.

Outputs (saved to output_dir/):
    asp_close_glu_chi1_results.csv      counts and percentages
    asp_close_glu_chi1_summary.txt      human-readable report
    asp_close_glu_chi1_{method}.png     2×2 bar chart, one panel per GLU residue
                                        showing ASP-close % vs baseline %

Usage:
    python3 asp_close_glu_chi1_analysis.py \\
        --gmm-pkl   .../G12_ML/dihedral_analysis/gmm_analysis/gmm_results.pkl \\
        --base-dir  .../G12_ML \\
        --subfolder ASP_G1 \\
        --threshold 4.0 \\
        -o          .../G12_ML/asp_close_glu_chi1 \\
        -c          G12_ML
"""

import argparse
import pickle
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GLU_PDB_LABELS = ['152.A', '152.B', '152.C', '141.D']

COMPONENT_COLORS = ['#E63946', '#2A9D8F', '#FF6B00', '#6A0572', '#457B9D']

FS_TITLE  = 20
FS_LABEL  = 18
FS_TICK   = 16
FS_ANNOT  = 15
FS_LEGEND = 14


# ── helpers ───────────────────────────────────────────────────────────────────

def find_res_by_label(gmm_results, pdb_label):
    """Find res_name and res dict by pdb_label."""
    for res_name, res in gmm_results.items():
        if res.get('pdb_label') == pdb_label:
            return res_name, res
    return None, None


def load_distances_per_run(base_dir, subfolder):
    """Load min_distance arrays per run from combined_distance_dihedrals.pkl."""
    dist_per_run = {}
    for run_dir in sorted(Path(base_dir).glob('RUN*')):
        pkl_path = (run_dir / 'ligand_distance_analysis' /
                    subfolder / 'combined_distance_dihedrals.pkl')
        if not pkl_path.exists():
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        dist_per_run[run_dir.name] = data['min_distance']
        print(f"  Loaded {run_dir.name}: {len(data['min_distance'])} frames")
    return dist_per_run


# ── core analysis ─────────────────────────────────────────────────────────────

def analyze(gmm_results, dist_per_run, lpr_key, threshold, channel_type):
    """
    For every frame where asp_distance < threshold, record chi1 state of
    each GLU. Also record baseline (all frames).

    Returns
    -------
    results : dict
        {glu_pdb_label: {
            'close_counter':    Counter {state: n_frames},
            'baseline_counter': Counter {state: n_frames},
            'n_asp_close':      int,
            'n_total':          int,
        }}
    """
    # build {glu_pdb_label: lpr dict}
    glu_lpr = {}
    for pdb in GLU_PDB_LABELS:
        res_name, res = find_res_by_label(gmm_results, pdb)
        if res_name is None or 'chi1' not in res:
            print(f"  [WARN] chi1 not found for {pdb} — skipping")
            continue
        glu_lpr[pdb] = res['chi1'].get(lpr_key, {})

    results = {pdb: {'close_counter':    Counter(),
                     'baseline_counter': Counter(),
                     'n_asp_close': 0,
                     'n_total':     0}
               for pdb in glu_lpr}

    for run_name, dist_arr in dist_per_run.items():
        n_frames = len(dist_arr)

        for pdb, lpr in glu_lpr.items():
            if run_name not in lpr:
                continue
            glu_labels = lpr[run_name]
            n          = min(n_frames, len(glu_labels))

            results[pdb]['n_total'] += n

            for fi in range(n):
                dist  = float(dist_arr[fi])
                state = round(float(glu_labels[fi]))

                # baseline — all frames
                results[pdb]['baseline_counter'][state] += 1

                # close — only when ASP is near ligand
                if dist < threshold:
                    results[pdb]['close_counter'][state] += 1
                    if pdb == list(glu_lpr.keys())[0]:   # count once
                        pass

            # count asp-close frames once (same for all GLUs)
            results[pdb]['n_asp_close'] += sum(
                1 for fi in range(n) if float(dist_arr[fi]) < threshold
            )

    return results


# ── save CSV ──────────────────────────────────────────────────────────────────

def save_csv(results, out_path, method_name, threshold):
    rows = []
    for pdb, data in results.items():
        n_close    = data['n_asp_close']
        n_total    = data['n_total']
        all_states = sorted(set(list(data['close_counter'].keys()) +
                                list(data['baseline_counter'].keys())))
        for state in all_states:
            c_count = data['close_counter'].get(state, 0)
            b_count = data['baseline_counter'].get(state, 0)
            rows.append({
                'glu_residue':       pdb,
                'method':            method_name,
                'asp_threshold_A':   threshold,
                'chi1_state':        state,
                'n_close_frames':    c_count,
                'pct_close':         round(100. * c_count / n_close, 2) if n_close > 0 else 0.,
                'n_baseline_frames': b_count,
                'pct_baseline':      round(100. * b_count / n_total, 2) if n_total > 0 else 0.,
                'n_asp_close_total': n_close,
                'n_total_frames':    n_total,
            })
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


# ── save summary text ─────────────────────────────────────────────────────────

def save_summary(results, out_path, method_name, threshold, channel_type):
    lines = [
        f"{'='*70}",
        f"ASP-CLOSE → GLU CHI1 STATE ANALYSIS",
        f"System: {channel_type}  |  Method: {method_name}  |  Threshold: {threshold} Å",
        f"{'='*70}",
        f"",
        f"Question: when ASP173.D is within {threshold} Å of the ligand,",
        f"          what chi1 state are the GLU residues in?",
        f"",
    ]
    for pdb, data in results.items():
        n_close = data['n_asp_close']
        n_total = data['n_total']
        pct_asp = 100. * n_close / n_total if n_total > 0 else 0.
        lines += [
            f"  {'─'*60}",
            f"  GLU {pdb}",
            f"  {'─'*60}",
            f"  ASP-close frames : {n_close:,} / {n_total:,}  ({pct_asp:.1f}% of all frames)",
            f"",
            f"  Chi1 state distribution WHEN ASP IS CLOSE:",
        ]
        all_states = sorted(data['close_counter'].keys())
        for state in all_states:
            c  = data['close_counter'][state]
            b  = data['baseline_counter'][state]
            cp = 100. * c / n_close  if n_close > 0  else 0.
            bp = 100. * b / n_total  if n_total > 0  else 0.
            lines.append(
                f"    chi1 = {state:>5}°   "
                f"ASP-close: {cp:5.1f}%  (n={c:,})   "
                f"baseline: {bp:5.1f}%  (n={b:,})"
            )
        lines.append("")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {out_path}")
    # also print to terminal
    print('\n'.join(lines))


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_2x2(results, method_name, threshold, dist_label,
             channel_type, out_path):
    """
    2×2 figure — one panel per GLU residue.
    For each chi1 state: two bars side by side — ASP-close % vs baseline %.
    """
    pdb_order = [p for p in GLU_PDB_LABELS if p in results]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    fig.suptitle(
        f"{channel_type}  —  GLU χ1 state when ASP173.D is close to ligand\n"
        f"Threshold = {threshold} Å   |   {dist_label}   |   {method_name}\n"
        f"Blue = when ASP close   Grey = all frames (baseline)",
        fontsize=FS_TITLE, fontweight='bold', y=1.02
    )

    for idx, pdb in enumerate(pdb_order):
        ax   = axes[idx]
        data = results[pdb]

        n_close = data['n_asp_close']
        n_total = data['n_total']

        all_states = sorted(set(list(data['close_counter'].keys()) +
                                list(data['baseline_counter'].keys())))

        if not all_states or n_close == 0:
            ax.text(0.5, 0.5, f"No frames with ASP distance < {threshold} Å",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=FS_LABEL)
            ax.set_title(f"GLU {pdb}", fontsize=FS_TITLE, fontweight='bold')
            continue

        x       = np.arange(len(all_states))
        width   = 0.38
        pcts_c  = [100. * data['close_counter'].get(s, 0) / n_close    for s in all_states]
        pcts_b  = [100. * data['baseline_counter'].get(s, 0) / n_total for s in all_states]

        bars_c = ax.bar(x - width/2, pcts_c, width, color='#2980B9',
                        alpha=0.85, edgecolor='black', linewidth=1.2,
                        label=f'ASP-close  (n={n_close:,})', zorder=2)
        bars_b = ax.bar(x + width/2, pcts_b, width, color='#95A5A6',
                        alpha=0.75, edgecolor='black', linewidth=1.2,
                        label=f'All frames  (n={n_total:,})', zorder=2)

        # annotate each bar
        for xi, (pc, pb) in enumerate(zip(pcts_c, pcts_b)):
            ax.text(xi - width/2, pc + 0.8, f"{pc:.1f}%",
                    ha='center', va='bottom', fontsize=FS_ANNOT,
                    fontweight='bold', color='#1A5276')
            ax.text(xi + width/2, pb + 0.8, f"{pb:.1f}%",
                    ha='center', va='bottom', fontsize=FS_ANNOT,
                    fontweight='bold', color='#5D6D7E')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}°" for s in all_states], fontsize=FS_TICK)
        ax.set_ylabel("% of frames", fontsize=FS_LABEL)
        ax.set_ylim(0, max(max(pcts_c), max(pcts_b)) * 1.35)
        ax.set_title(
            f"GLU {pdb} χ1  —  when ASP173.D close vs baseline\n"
            f"ASP-close frames: {n_close:,} / {n_total:,} "
            f"({100.*n_close/n_total:.1f}% of trajectory)",
            fontsize=FS_TITLE, fontweight='bold', pad=10
        )
        ax.legend(fontsize=FS_LEGEND, loc='upper right')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines[["top", "right"]].set_visible(False)

    # hide unused panels
    for idx in range(len(pdb_order), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='When ASP173.D is close to the ligand, what chi1 state are the GLUs in?'
    )
    parser.add_argument('--gmm-pkl',   required=True,
                        help='Path to gmm_results.pkl')
    parser.add_argument('--base-dir',  required=True,
                        help='Base dir containing RUN* folders (e.g. .../G12_ML)')
    parser.add_argument('--subfolder', default='ASP_G1',
                        help='Distance subfolder inside ligand_distance_analysis/ (default: ASP_G1)')
    parser.add_argument('--threshold', type=float, default=4.0,
                        help='Distance threshold in Å — ASP is "close" below this (default: 4.0)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory')
    parser.add_argument('-c', '--channel', default='G12_ML',
                        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'])
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output) if args.output \
                 else base_dir / 'asp_close_glu_chi1'
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_label = f"{args.subfolder} min distance < {args.threshold} Å"

    print(f"\n{'='*70}")
    print(f"ASP-CLOSE → GLU CHI1 ANALYSIS")
    print(f"{'='*70}")
    print(f"System    : {args.channel}")
    print(f"Subfolder : {args.subfolder}")
    print(f"Threshold : {args.threshold} Å")
    print(f"Output    : {output_dir}\n")

    # load
    with open(args.gmm_pkl, 'rb') as f:
        gmm_results = pickle.load(f)
    print(f"Loaded GMM results: {len(gmm_results)} residues")

    dist_per_run = load_distances_per_run(base_dir, args.subfolder)
    if not dist_per_run:
        print("ERROR: no distance files found — check --subfolder and --base-dir")
        return 1
    print(f"Loaded distances: {len(dist_per_run)} runs\n")

    methods = [
        ('probabilistic', 'labels_per_run'),
        ('nearest_mean',  'labels_per_run_near'),
    ]

    for method_name, lpr_key in methods:
        print(f"\n{'─'*70}")
        print(f"  Method: {method_name}")
        print(f"{'─'*70}")

        results = analyze(gmm_results, dist_per_run,
                          lpr_key, args.threshold, args.channel)

        # save outputs
        save_csv(results,
                 output_dir / f"asp_close_glu_chi1_{method_name}.csv",
                 method_name, args.threshold)

        save_summary(results,
                     output_dir / f"asp_close_glu_chi1_{method_name}.txt",
                     method_name, args.threshold, args.channel)

        plot_2x2(results, method_name, args.threshold, dist_label,
                 args.channel,
                 output_dir / f"asp_close_glu_chi1_{method_name}.png")

    print(f"\n{'='*70}")
    print(f"DONE. Results in: {output_dir}")
    print(f"{'='*70}\n")
    return 0


if __name__ == '__main__':
    exit(main())

'''
python3 asp_close_glu_chi1_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis_4gchi2/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/asp_close_glu_chi1_analysis \
        --subfolder ASP_G1 \
    --threshold 3.5 \
    -c G12_ML
'''