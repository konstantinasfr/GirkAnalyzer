#!/usr/bin/env python3
"""
Subunit GLU × ASN/ASP Combination Analysis

For each subunit pair (A, B, C, D), combines the chi1×chi2 state of the
GLU residue with the chi1×chi2 state of the paired ASN/ASP residue to find
the most frequent joint conformational states across all runs.

Subunit pairs:
    A: 152.A (GLU) × 184.A (ASN)
    B: 152.B (GLU) × 184.B (ASN)
    C: 152.C (GLU) × 184.C (ASN)
    D: 141.D (GLU) × 173.D (ASP)

For each frame: joint_state = "GLU(chi1°/chi2°) | ASN(chi1°/chi2°)"

Outputs (one set per assignment method):
    subunit_combinations_{method}.txt      — full counts, all combinations
    subunit_combinations_top8_{method}.png — 2×2 plot, top 8 per subunit
    subunit_combinations_all_{method}.png  — 2×2 plot, all combinations
    subunit_combinations_{method}.pkl      — raw counts dict

Usage:
    python subunit_combination_analysis.py \
        --gmm-pkl .../gmm_analysis/gmm_results.pkl \
        -o        .../subunit_combinations \
        -c        G2
"""

import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


COMPONENT_COLORS = [
    '#E63946', '#2A9D8F', '#FF6B00', '#6A0572',
    '#457B9D', '#E9C46A', '#F4A261', '#264653',
    '#8338EC', '#3A86FF', '#FB5607', '#FFBE0B',
    '#06D6A0', '#EF476F', '#118AB2', '#073B4C',
]

# subunit → (glu_pdb_label, asnasp_pdb_label) — defined per channel type
SUBUNIT_PAIRS_BY_CHANNEL = {
    'G2': {
        'A': ('152.A', '184.A'),
        'B': ('152.B', '184.B'),
        'C': ('152.C', '184.C'),
        'D': ('152.D', '184.D'),
    },
    'G12': {
        'A': ('152.A', '184.A'),
        'B': ('152.B', '184.B'),
        'C': ('152.C', '184.C'),
        'D': ('141.D', '173.D'),
    },
    'G12_GAT': {
        'A': ('152.A', '184.A'),
        'B': ('152.B', '184.B'),
        'C': ('152.C', '184.C'),
        'D': ('141.D', '173.D'),
    },
    'G12_ML': {
        'A': ('152.A', '184.A'),
        'B': ('152.B', '184.B'),
        'C': ('152.C', '184.C'),
        'D': ('141.D', '173.D'),
    },
}

SUBUNIT_ORDER = ['A', 'B', 'C', 'D']


# =============================================================================
# HELPERS
# =============================================================================

def find_res_by_label(gmm_results, pdb_label):
    for res_name, res in gmm_results.items():
        if res.get('pdb_label') == pdb_label:
            return res_name, res
    return None, None


def combo_str(l1, l2):
    return f'{round(float(l1))}°/{round(float(l2))}°'


def joint_str(glu_combo, asn_combo):
    return f'GLU({glu_combo})|ASN({asn_combo})'


def _sort_joint_key(s):
    """Sort by GLU chi1 abs, GLU chi2 abs, ASN chi1 abs, ASN chi2 abs."""
    try:
        glu_part = s.split('|')[0].replace('GLU(', '').replace(')', '')
        asn_part = s.split('|')[1].replace('ASN(', '').replace(')', '')
        g1, g2 = [abs(float(x.replace('°', ''))) for x in glu_part.split('/')]
        a1, a2 = [abs(float(x.replace('°', ''))) for x in asn_part.split('/')]
        return (g1, g2, a1, a2)
    except Exception:
        return (0, 0, 0, 0)


# =============================================================================
# COMPUTE JOINT STATES
# =============================================================================

def compute_joint_states(gmm_results, lpr_key, subunit_pairs):
    """
    For each subunit pair, compute joint GLU-combo × ASN-combo states
    across all runs using the specified label key.

    Returns
    -------
    joint_counts  : {subunit: {joint_state_str: int}}
    joint_n_total : {subunit: int}
    """
    joint_counts  = {s: {} for s in subunit_pairs}
    joint_n_total = {s: 0  for s in subunit_pairs}

    for subunit, (glu_lbl, asn_lbl) in subunit_pairs.items():
        _, glu_res = find_res_by_label(gmm_results, glu_lbl)
        _, asn_res = find_res_by_label(gmm_results, asn_lbl)

        if glu_res is None or asn_res is None:
            print(f"  [SKIP] subunit {subunit}: "
                  f"{glu_lbl}={'found' if glu_res else 'MISSING'}  "
                  f"{asn_lbl}={'found' if asn_res else 'MISSING'}")
            continue

        if 'chi1' not in glu_res or 'chi2' not in glu_res:
            print(f"  [SKIP] subunit {subunit}: GLU missing chi1 or chi2")
            continue
        if 'chi1' not in asn_res or 'chi2' not in asn_res:
            print(f"  [SKIP] subunit {subunit}: ASN/ASP missing chi1 or chi2")
            continue

        glu_lpr1 = glu_res['chi1'].get(lpr_key, {})
        glu_lpr2 = glu_res['chi2'].get(lpr_key, {})
        asn_lpr1 = asn_res['chi1'].get(lpr_key, {})
        asn_lpr2 = asn_res['chi2'].get(lpr_key, {})

        common_runs = sorted(
            set(glu_lpr1) & set(glu_lpr2) & set(asn_lpr1) & set(asn_lpr2)
        )
        print(f"  Subunit {subunit} ({glu_lbl} × {asn_lbl}): "
              f"{len(common_runs)} runs")

        # use correct residue type label: ASP for 173.x, ASN for 184.x
        partner_label = 'ASP' if asn_lbl.startswith('173') else 'ASN'

        counts  = {}
        n_total = 0

        for run_name in common_runs:
            gl1 = np.asarray(glu_lpr1[run_name])
            gl2 = np.asarray(glu_lpr2[run_name])
            al1 = np.asarray(asn_lpr1[run_name])
            al2 = np.asarray(asn_lpr2[run_name])
            n   = min(len(gl1), len(gl2), len(al1), len(al2))
            gl1 = gl1[:n]; gl2 = gl2[:n]
            al1 = al1[:n]; al2 = al2[:n]

            # build integer tuple keys, count with numpy, convert to strings
            # round to nearest int to match label display
            gl1r = np.round(gl1).astype(int)
            gl2r = np.round(gl2).astype(int)
            al1r = np.round(al1).astype(int)
            al2r = np.round(al2).astype(int)

            # stack into (n, 4) array and find unique rows
            combined = np.stack([gl1r, gl2r, al1r, al2r], axis=1)
            unique_rows, row_counts = np.unique(combined, axis=0,
                                                return_counts=True)
            for row, cnt in zip(unique_rows, row_counts):
                key = (f'GLU({row[0]}°/{row[1]}°)'
                       f'|{partner_label}({row[2]}°/{row[3]}°)')
                counts[key] = counts.get(key, 0) + int(cnt)
            n_total += n

        joint_counts[subunit]  = counts
        joint_n_total[subunit] = n_total
        print(f"    → {len(counts)} unique joint states, {n_total:,} frames")

    return joint_counts, joint_n_total


# =============================================================================
# PERMEATION FRAME LOADING
# =============================================================================

def load_permeation_frames(base_dir, run_names):
    """
    Load R2_end frame indices from permeation_table.json per run.
    Same logic as in gmm_dihedral_analysis.py.

    Returns
    -------
    perm_frames : {run_name: [frame_idx, ...]}
    """
    base_dir    = Path(base_dir)
    perm_frames = {}

    for run_name in run_names:
        json_path = base_dir / run_name / 'permeation_table.json'
        if not json_path.exists():
            print(f"  [SKIP] {run_name}: permeation_table.json not found")
            continue

        with open(json_path, 'r') as f:
            table = json.load(f)

        rows   = list(table.values()) if isinstance(table, dict) else table
        r2_ends = [int(row['R2_end']) for row in rows
                   if 'R2_end' in row and row['R2_end'] is not None]
        perm_frames[run_name] = r2_ends
        print(f"  {run_name}: {len(r2_ends)} exit events at frames {r2_ends[:5]}"
              f"{'...' if len(r2_ends) > 5 else ''}")

    return perm_frames


# =============================================================================
# COMPUTE JOINT STATES AT SPECIFIC FRAMES (permeation events)
# =============================================================================

def compute_joint_states_at_frames(gmm_results, lpr_key, subunit_pairs,
                                    perm_frames, offset=0):
    """
    For each subunit pair, compute joint GLU×ASN/ASP states only at
    permeation event frames (R2_end + offset).

    Parameters
    ----------
    perm_frames : {run_name: [frame_idx, ...]}
    offset      : 0=at exit, -1=before exit, +1=after exit

    Returns
    -------
    joint_counts  : {subunit: {joint_state_str: int}}
    joint_n_total : {subunit: int}
    """
    joint_counts  = {s: {} for s in subunit_pairs}
    joint_n_total = {s: 0  for s in subunit_pairs}

    for subunit, (glu_lbl, asn_lbl) in subunit_pairs.items():
        _, glu_res = find_res_by_label(gmm_results, glu_lbl)
        _, asn_res = find_res_by_label(gmm_results, asn_lbl)

        if glu_res is None or asn_res is None:
            print(f"  [SKIP] subunit {subunit}: residue missing")
            continue
        if 'chi1' not in glu_res or 'chi2' not in glu_res:
            print(f"  [SKIP] subunit {subunit}: GLU missing chi1/chi2")
            continue
        if 'chi1' not in asn_res or 'chi2' not in asn_res:
            print(f"  [SKIP] subunit {subunit}: ASN/ASP missing chi1/chi2")
            continue

        glu_lpr1 = glu_res['chi1'].get(lpr_key, {})
        glu_lpr2 = glu_res['chi2'].get(lpr_key, {})
        asn_lpr1 = asn_res['chi1'].get(lpr_key, {})
        asn_lpr2 = asn_res['chi2'].get(lpr_key, {})

        partner_label = 'ASP' if asn_lbl.startswith('173') else 'ASN'

        common_runs = sorted(
            set(glu_lpr1) & set(glu_lpr2) &
            set(asn_lpr1) & set(asn_lpr2) &
            set(perm_frames.keys())
        )

        counts  = {}
        n_total = 0

        for run_name in common_runs:
            gl1 = np.asarray(glu_lpr1[run_name])
            gl2 = np.asarray(glu_lpr2[run_name])
            al1 = np.asarray(asn_lpr1[run_name])
            al2 = np.asarray(asn_lpr2[run_name])
            n_run = min(len(gl1), len(gl2), len(al1), len(al2))

            for frame in perm_frames[run_name]:
                target = frame + offset
                if target < 0 or target >= n_run:
                    print(f"  [WARN] {run_name} frame {frame}+{offset}={target} "
                          f"out of bounds [0,{n_run}) — skipping")
                    continue
                g1 = int(round(float(gl1[target])))
                g2 = int(round(float(gl2[target])))
                a1 = int(round(float(al1[target])))
                a2 = int(round(float(al2[target])))
                key = (f'GLU({g1}°/{g2}°)'
                       f'|{partner_label}({a1}°/{a2}°)')
                counts[key] = counts.get(key, 0) + 1
                n_total += 1

        joint_counts[subunit]  = counts
        joint_n_total[subunit] = n_total
        print(f"    Subunit {subunit} ({glu_lbl} × {asn_lbl}): "
              f"{n_total} events, {len(counts)} unique states")

    return joint_counts, joint_n_total
# =============================================================================

def save_report(joint_counts, joint_n_total, output_dir, method_name,
                subunit_pairs):
    out = output_dir / f'subunit_combinations_{method_name}.txt'
    with open(out, 'w') as f:
        f.write("SUBUNIT GLU × ASN/ASP JOINT CONFORMATIONAL STATES\n")
        f.write(f"Method: {method_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Format: GLU(chi1/chi2) | ASN(chi1/chi2)\n\n")

        for subunit in SUBUNIT_ORDER:
            if subunit not in subunit_pairs:
                continue
            glu_lbl, asn_lbl = subunit_pairs[subunit]
            counts  = joint_counts.get(subunit, {})
            n_total = joint_n_total.get(subunit, 0)
            if not counts:
                continue

            f.write(f"SUBUNIT {subunit}: {glu_lbl} × {asn_lbl}  "
                    f"({n_total:,} total frames)\n")
            f.write("-" * 80 + "\n")
            for state, cnt in sorted(counts.items(),
                                     key=lambda x: -x[1]):
                pct = cnt / n_total * 100
                f.write(f"  {state:<50s}  {cnt:8,}  ({pct:5.1f}%)\n")
            f.write("\n" + "=" * 80 + "\n\n")

    pkl_out = output_dir / f'subunit_combinations_{method_name}.pkl'
    with open(pkl_out, 'wb') as f:
        pickle.dump({'joint_counts': joint_counts,
                     'joint_n_total': joint_n_total,
                     'subunit_pairs': subunit_pairs}, f)

    print(f"  ✅ Report: {out.name}")
    print(f"  ✅ PKL:    {pkl_out.name}")


# =============================================================================
# PLOTTING
# =============================================================================

def _draw_combo_panel(ax, counts, n_total, subunit, glu_lbl, asn_lbl,
                      channel_type, method_name, top_n=None):
    """Draw one bar panel for a subunit pair."""
    if not counts or n_total == 0:
        ax.set_visible(False)
        return

    sorted_states = sorted(counts.items(), key=lambda x: -x[1])
    if top_n is not None:
        sorted_states = sorted_states[:top_n]

    states  = [s for s, _ in sorted_states]
    values  = [c / n_total * 100 for _, c in sorted_states]
    raw_cnt = [c for _, c in sorted_states]

    x      = np.arange(len(states))
    colors = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
              for i in range(len(states))]

    bars = ax.bar(x, values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.5, zorder=2)

    any_large = any(c >= 1000 for c in raw_cnt)

    for xi, (bar, pct, cnt) in enumerate(zip(bars, values, raw_cnt)):
        if any_large:
            label = f'{pct:.1f}%\n({cnt:,}/\n{n_total:,})'
        else:
            label = f'{pct:.1f}%\n({cnt}/{n_total})'
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.3,
                label, ha='center', va='bottom',
                fontsize=17, fontweight='bold')

    tick_labels = []
    for s in states:
        parts = s.split('|')
        tick_labels.append(parts[0].strip() + '\n' + parts[1].strip())

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=15, fontweight='bold',
                       rotation=20, ha='right')
    ax.set_ylabel('Occupancy (%)', fontsize=21, fontweight='bold')
    ax.set_ylim(0, min(100, max(values) * 1.6 + 3))
    ax.set_title(f'Subunit {subunit}:  {glu_lbl} × {asn_lbl}',
                 fontsize=23, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='y', labelsize=19)


def plot_joint_combinations(joint_counts, joint_n_total,
                             output_dir, channel_type, method_name,
                             subunit_pairs, top_n=None):
    """2×2 grid — one panel per subunit pair."""
    suffix    = f'top{top_n}' if top_n else 'all'
    fig, axes = plt.subplots(2, 2, figsize=(26, 18))
    axes      = axes.flatten()
    title_sfx = f'top {top_n}' if top_n else 'all combinations'
    fig.suptitle(
        f'{channel_type}  —  GLU × ASN/ASP joint states  |  '
        f'{title_sfx}  |  {method_name}',
        fontsize=25, fontweight='bold', y=1.01
    )

    for idx, subunit in enumerate(SUBUNIT_ORDER):
        if subunit not in subunit_pairs:
            axes[idx].set_visible(False)
            continue
        glu_lbl, asn_lbl = subunit_pairs[subunit]
        counts  = joint_counts.get(subunit, {})
        n_total = joint_n_total.get(subunit, 0)
        _draw_combo_panel(axes[idx], counts, n_total,
                          subunit, glu_lbl, asn_lbl,
                          channel_type, method_name, top_n=top_n)

    plt.tight_layout()
    out = output_dir / f'subunit_combinations_{suffix}_{method_name}.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Plot: {out.name}')


def plot_focus_subunit_D(joint_counts, joint_n_total,
                          output_dir, channel_type, method_name,
                          subunit_pairs, top_n=8):
    """
    Focus plot for subunit D (141.D × 173.D) only — larger single panel.
    Only produced when subunit D has 141.D × 173.D (i.e. not G2).
    """
    if 'D' not in subunit_pairs:
        return
    glu_lbl, asn_lbl = subunit_pairs['D']
    if glu_lbl != '141.D' or asn_lbl != '173.D':
        return   # G2 uses 152.D×184.D — skip focus plot

    counts  = joint_counts.get('D', {})
    n_total = joint_n_total.get('D', 0)
    if not counts or n_total == 0:
        return

    sorted_states = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    states  = [s for s, _ in sorted_states]
    values  = [c / n_total * 100 for _, c in sorted_states]
    raw_cnt = [c for _, c in sorted_states]

    fig, ax = plt.subplots(figsize=(max(10, len(states) * 2.2), 8))
    fig.suptitle(
        f'{channel_type}  —  141.D (GLU) × 173.D (ASP)  |  '
        f'top {top_n}  |  {method_name}',
        fontsize=20, fontweight='bold'
    )

    x      = np.arange(len(states))
    colors = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
              for i in range(len(states))]
    bars   = ax.bar(x, values, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=1.5, zorder=2)

    any_large = any(c >= 1000 for c in raw_cnt)
    for bar, pct, cnt in zip(bars, values, raw_cnt):
        if any_large:
            label = f'{pct:.1f}%\n({cnt:,}/\n{n_total:,})'
        else:
            label = f'{pct:.1f}%\n({cnt}/{n_total})'
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.3,
                label, ha='center', va='bottom',
                fontsize=17, fontweight='bold')

    tick_labels = []
    for s in states:
        parts = s.split('|')
        tick_labels.append(parts[0].strip() + '\n' + parts[1].strip())

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=16, fontweight='bold',
                       rotation=20, ha='right')
    ax.set_ylabel('Occupancy (%)', fontsize=22, fontweight='bold')
    ax.set_ylim(0, min(100, max(values) * 1.6 + 3))
    ax.set_title('Subunit D:  141.D × 173.D',
                 fontsize=24, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    out = output_dir / f'focus_141D_173D_top{top_n}_{method_name}.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Focus plot: {out.name}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Subunit GLU × ASN/ASP joint combination analysis'
    )
    parser.add_argument('--gmm-pkl',  required=True,
                        help='Path to gmm_results.pkl')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory')
    parser.add_argument('-c', '--channel', default='G12',
                        help='Channel type (G2, G12, G12_GAT, G12_ML)')
    parser.add_argument('--top', type=int, default=8,
                        help='Number of top combinations to show (default: 8)')
    parser.add_argument('--base-dir', default=None,
                        help='Base dir with RUN* folders containing '
                             'permeation_table.json (enables ion exit analysis)')
    args = parser.parse_args()

    gmm_pkl    = Path(args.gmm_pkl)
    output_dir = Path(args.output) if args.output \
                 else gmm_pkl.parent / 'subunit_combinations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SUBUNIT GLU × ASN/ASP COMBINATION ANALYSIS")
    print("=" * 70)

    with open(gmm_pkl, 'rb') as f:
        gmm_results = pickle.load(f)
    print(f"Loaded GMM results: {len(gmm_results)} residues")
    print(f"Residues: {[(v['pdb_label'], k) for k, v in gmm_results.items()]}")

    # pick correct subunit pairs for this channel type
    subunit_pairs = SUBUNIT_PAIRS_BY_CHANNEL.get(
        args.channel,
        SUBUNIT_PAIRS_BY_CHANNEL['G12']  # default fallback
    )
    print(f"\nSubunit pairs for {args.channel}:")
    for s, (g, a) in subunit_pairs.items():
        print(f"  {s}: {g} × {a}")

    methods = [
        ('probabilistic', 'labels_per_run'),
        ('nearest_mean',  'labels_per_run_near'),
    ]

    for method_name, lpr_key in methods:
        print(f"\n{'='*70}")
        print(f"  METHOD: {method_name}")
        print(f"{'='*70}")

        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        joint_counts, joint_n_total = compute_joint_states(
            gmm_results, lpr_key, subunit_pairs)

        save_report(joint_counts, joint_n_total, method_dir,
                    method_name, subunit_pairs)

        plot_joint_combinations(
            joint_counts, joint_n_total,
            method_dir, args.channel, method_name,
            subunit_pairs, top_n=args.top)

        plot_joint_combinations(
            joint_counts, joint_n_total,
            method_dir, args.channel, method_name,
            subunit_pairs, top_n=None)

        # focus plot for 141.D × 173.D (G12/G12_ML/G12_GAT only)
        plot_focus_subunit_D(
            joint_counts, joint_n_total,
            method_dir, args.channel, method_name,
            subunit_pairs, top_n=args.top)

        # ── ion exit event analysis ───────────────────────────────────────
        if args.base_dir:
            base_dir = Path(args.base_dir)

            # get run names from any residue's labels_per_run
            sample_res = next(iter(gmm_results.values()))
            run_names  = sorted(sample_res['chi1'].get(lpr_key, {}).keys())

            print(f"\n  Loading permeation frames...")
            perm_frames = load_permeation_frames(base_dir, run_names)

            if perm_frames:
                for offset, frame_label in [(0,  'at_exit'),
                                             (-1, 'before_exit'),
                                             (1,  'after_exit')]:
                    perm_dir = method_dir / f'ion_exit_{frame_label}'
                    perm_dir.mkdir(exist_ok=True)

                    print(f"\n  --- {frame_label} (offset={offset}) ---")
                    jc, jn = compute_joint_states_at_frames(
                        gmm_results, lpr_key, subunit_pairs,
                        perm_frames, offset=offset)

                    save_report(jc, jn, perm_dir,
                                f'{method_name}_{frame_label}', subunit_pairs)

                    plot_joint_combinations(
                        jc, jn, perm_dir, args.channel,
                        f'{method_name} | {frame_label}',
                        subunit_pairs, top_n=args.top)

                    plot_joint_combinations(
                        jc, jn, perm_dir, args.channel,
                        f'{method_name} | {frame_label}',
                        subunit_pairs, top_n=None)

                    plot_focus_subunit_D(
                        jc, jn, perm_dir, args.channel,
                        f'{method_name} | {frame_label}',
                        subunit_pairs, top_n=args.top)
            else:
                print("  No permeation_table.json files found — skipping")

    print(f"\n{'='*70}")
    print(f"DONE. Results in: {output_dir}/")
    print(f"{'='*70}\n")
    return 0


if __name__ == '__main__':
    exit(main())

"""
python3 subunit_combination_analysis.py \
    --gmm-pkl .../G2/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o        .../G2/subunit_combinations \
    -c        G2 \
    --base-dir .../G2

python3 subunit_combination_analysis.py \
    --gmm-pkl .../G12/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o        .../G12/subunit_combinations \
    -c        G12 \
    --base-dir .../G12

python3 subunit_combination_analysis.py \
    --gmm-pkl .../G12_ML/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o        .../G12_ML/subunit_combinations \
    -c        G12_ML \
    --base-dir .../G12_ML

python3 subunit_combination_analysis.py \
    --gmm-pkl .../G12_GAT/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o        .../G12_GAT/subunit_combinations \
    -c        G12_GAT \
    --base-dir .../G12_GAT
"""






"""
python3 subunit_combination_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o       /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis/subunit_comb \
    -c        G2  \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2

python3 subunit_combination_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o       /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis/subunit_comb \
    -c        G12   \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12

python3 subunit_combination_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis_4gchi2/gmm_results.pkl \
    -o       /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/subunit_comb_4gchi2/ \
    -c        G12_ML  \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML

python3 subunit_combination_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis_4gchi2/gmm_results.pkl \
    -o       /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/subunit_comb_4gchi2/ \
    -c        G12_ML  \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML

python3 subunit_combination_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    -o       /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/subunit_comb \
    -c        G12_GAT  \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT
"""