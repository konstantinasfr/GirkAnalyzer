#!/usr/bin/env python3
"""
Ion Proximity per GMM State Analysis

For each residue, chi angle, and GMM state: what % of frames in that
state had an ion within threshold?

Key design: for each run, labels_per_run[run] and ion_per_run[run]
are loaded together — same run, same frame order, direct zip.
Only runs present in BOTH are used.

Usage:
    python ion_proximity_gmm_analysis.py \
        --gmm-pkl  .../gmm_analysis/gmm_results.pkl \
        --base-dir .../G2 \
        -o         .../G2/ion_proximity_gmm \
        -c         G2
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COMPONENT_COLORS = ['#E63946', '#2A9D8F', '#FF6B00', '#6A0572']


# =============================================================================
# LOAD ION DATA — keyed by run_name
# =============================================================================

def load_ion_per_run(base_dir):
    """
    Returns
    -------
    ion_per_run : {pdb_label: {run_name: bool array}}
    threshold   : float
    """
    base_dir    = Path(base_dir)
    ion_per_run = {}
    threshold   = None

    for run_dir in sorted(base_dir.glob('RUN*')):
        pkl_path = (run_dir / 'ion_proximity_analysis' /
                    'combined_ion_dihedrals.pkl')
        if not pkl_path.exists():
            print(f"  [SKIP] {run_dir.name}: no combined_ion_dihedrals.pkl")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        threshold = data.get('threshold', 3.5)
        print(f"  Loaded {run_dir.name}: {data['n_frames']} frames  "
              f"threshold={threshold} Å")
        for lbl in data['labels']:
            if lbl not in ion_per_run:
                ion_per_run[lbl] = {}
            ion_per_run[lbl][run_dir.name] = np.asarray(
                data['has_ion'][lbl], dtype=bool)

    return ion_per_run, threshold


# =============================================================================
# COMPUTE FRACTIONS — direct per-run pairing
# =============================================================================

def compute_chi_fractions(gmm_results, ion_per_run, res_name,
                           chi_key, lpr_key):
    """
    For each GMM state of (res_name, chi_key):
    count frames assigned to that state, count those with ion nearby.
    Only uses runs present in BOTH labels_per_run AND ion_per_run.
    """
    if res_name not in gmm_results or chi_key not in gmm_results[res_name]:
        return None

    pdb_label  = gmm_results[res_name]['pdb_label']
    components = gmm_results[res_name][chi_key]['components']
    lpr        = gmm_results[res_name][chi_key].get(lpr_key, {})

    if pdb_label not in ion_per_run:
        return None

    ion_runs    = ion_per_run[pdb_label]
    common_runs = sorted(set(lpr.keys()) & set(ion_runs.keys()))

    if not common_runs:
        print(f"  [WARN] {pdb_label} {chi_key}: no runs in common — "
              f"labels has {sorted(lpr.keys())}, "
              f"ion has {sorted(ion_runs.keys())}")
        return None

    print(f"  {pdb_label} {chi_key}: using {len(common_runs)} runs: "
          f"{common_runs}")

    state_n_total = {round(c['mean']): 0 for c in components}
    state_n_ion   = {round(c['mean']): 0 for c in components}

    for run_name in common_runs:
        labels = np.asarray(lpr[run_name])
        ion    = ion_runs[run_name]
        n      = min(len(labels), len(ion))
        labels = labels[:n]
        ion    = ion[:n]
        for state in state_n_total:
            mask = np.abs(labels - state) < 0.5
            state_n_total[state] += int(mask.sum())
            state_n_ion[state]   += int(ion[mask].sum())

    return {
        state: {
            'pct':     state_n_ion[state] / state_n_total[state] * 100
                       if state_n_total[state] > 0 else 0.0,
            'n_ion':   state_n_ion[state],
            'n_total': state_n_total[state]
        }
        for state in state_n_total
    }


def compute_combo_fractions(gmm_results, ion_per_run, res_name, lpr_key):
    """
    For each chi1×chi2 combination: count frames and those with ion.
    Only uses runs present in labels AND ion.
    """
    if res_name not in gmm_results:
        return None
    if ('chi1' not in gmm_results[res_name] or
            'chi2' not in gmm_results[res_name]):
        return None

    pdb_label = gmm_results[res_name]['pdb_label']
    lpr1      = gmm_results[res_name]['chi1'].get(lpr_key, {})
    lpr2      = gmm_results[res_name]['chi2'].get(lpr_key, {})

    if pdb_label not in ion_per_run:
        return None

    ion_runs    = ion_per_run[pdb_label]
    common_runs = sorted(
        set(lpr1.keys()) & set(lpr2.keys()) & set(ion_runs.keys()))

    if not common_runs:
        return None

    combo_n_total = {}
    combo_n_ion   = {}

    for run_name in common_runs:
        l1  = np.asarray(lpr1[run_name])
        l2  = np.asarray(lpr2[run_name])
        ion = ion_runs[run_name]
        n   = min(len(l1), len(l2), len(ion))

        for i in range(n):
            m1    = round(float(l1[i]))
            m2    = round(float(l2[i]))
            combo = f'{m1}°/{m2}°'
            combo_n_total[combo] = combo_n_total.get(combo, 0) + 1
            combo_n_ion[combo]   = combo_n_ion.get(combo, 0) + int(ion[i])

    return {
        combo: {
            'pct':     combo_n_ion[combo] / combo_n_total[combo] * 100,
            'n_ion':   combo_n_ion[combo],
            'n_total': combo_n_total[combo]
        }
        for combo in combo_n_total
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _sort_key(s):
    try:
        parts = str(s).replace('°', '').split('/')
        if len(parts) == 2:
            return (abs(float(parts[0])), abs(float(parts[1])))
        return (abs(float(parts[0])),)
    except Exception:
        return (0,)


def _draw_panel(ax, states, fractions, title, threshold, is_combo=False):
    max_pct = max((fractions[s]['pct'] for s in states), default=0)

    # check if any count exceeds 3 digits — if so stack vertically
    any_large = any(
        fractions[s]['n_ion'] >= 1000 or fractions[s]['n_total'] >= 1000
        for s in states
    )

    for xi, state in enumerate(states):
        info  = fractions[state]
        color = COMPONENT_COLORS[xi % len(COMPONENT_COLORS)]
        ax.bar(xi, info['pct'], width=0.5, color=color, alpha=0.85,
               edgecolor='black', linewidth=1.5, zorder=2)

        if any_large:
            label_txt = (f'{info["pct"]:.1f}%\n'
                         f'({info["n_ion"]:,}/\n'
                         f'{info["n_total"]:,})')
        else:
            label_txt = (f'{info["pct"]:.1f}%\n'
                         f'({info["n_ion"]:,}/{info["n_total"]:,})')

        ax.text(xi, info['pct'] + 0.5, label_txt,
                ha='center', va='bottom', fontsize=17, fontweight='bold')

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(
        [str(s) for s in states], fontsize=20, fontweight='bold',
        rotation=30 if is_combo else 0,
        ha='right' if is_combo else 'center')
    ax.set_ylabel(f'% frames with ion (< {threshold} Å)',
                  fontsize=20, fontweight='bold')
    ax.set_ylim(0, min(105, max_pct * 1.6 + 10))
    ax.set_title(title, fontsize=26, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='y', labelsize=18)


def plot_2x2(all_fractions, pdb_order, label, method_name,
             output_dir, channel_type, threshold, group, is_combo=False):
    fig, axes = plt.subplots(2, 2, figsize=(22 if is_combo else 20, 16))
    axes = axes.flatten()
    fig.suptitle(
        f'{channel_type}  —  Ion proximity (< {threshold} Å)  |  '
        f'{label}  |  {method_name}',
        fontsize=24, fontweight='bold', y=1.01)
    for idx, pdb in enumerate(pdb_order):
        ax = axes[idx]
        fr = all_fractions.get(pdb)
        if not fr:
            ax.set_visible(False)
            continue
        states = sorted(fr.keys(), key=_sort_key)
        _draw_panel(ax, states, fr, pdb, threshold, is_combo)
    plt.tight_layout()
    safe = (f'{label}_{method_name}_{group}'
            .replace(' ', '_').replace('/', '_')
            .replace('χ', 'chi').replace('×', 'x'))
    out = output_dir / f'2x2_ion_{safe}.png'
    plt.savefig(out, dpi=90, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {out.name}')


def plot_focus(fractions, pdb_label, label, method_name,
               output_dir, channel_type, threshold, is_combo=False):
    if not fractions:
        return
    states = sorted(fractions.keys(), key=_sort_key)
    fig, ax = plt.subplots(figsize=(max(8, len(states) * 2.2), 7))
    fig.suptitle(
        f'{channel_type}  —  Ion proximity (< {threshold} Å)  |  '
        f'{pdb_label}  {label}  |  {method_name}',
        fontsize=20, fontweight='bold')
    _draw_panel(ax, states, fractions, pdb_label, threshold, is_combo)
    plt.tight_layout()
    safe = (f'{pdb_label}_{label}_{method_name}'
            .replace(' ', '_').replace('/', '_')
            .replace('χ', 'chi').replace('×', 'x'))
    out = output_dir / f'focus_ion_{safe}.png'
    plt.savefig(out, dpi=90, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {out.name}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmm-pkl',  required=True)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-c', '--channel', default='G12')
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output) if args.output \
                 else base_dir / 'ion_proximity_gmm'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ION PROXIMITY PER GMM STATE ANALYSIS")
    print("=" * 70)

    with open(args.gmm_pkl, 'rb') as f:
        gmm_results = pickle.load(f)
    print(f"Loaded GMM results: {len(gmm_results)} residues")

    print("\nLoading ion proximity data...")
    ion_per_run, threshold = load_ion_per_run(base_dir)
    if not ion_per_run:
        print("[ERROR] No ion data found")
        return 1

    if args.channel == 'G2':
        pdb_order_glu    = ['152.B', '152.C', '152.A', '152.D']
        pdb_order_asnasp = ['184.B', '184.C', '184.A', '184.D']
    else:
        pdb_order_glu    = ['152.B', '152.C', '152.A', '141.D']
        pdb_order_asnasp = ['184.B', '184.C', '184.A', '173.D']

    focus_pdbs = ['141.D', '173.D']

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

        # ── chi1 and chi2 plots ───────────────────────────────────────────
        for chi_key in ['chi1', 'chi2']:
            chi_label = 'χ1' if chi_key == 'chi1' else 'χ2'
            glu_fr    = {}
            asnasp_fr = {}

            for res_name, res in gmm_results.items():
                pdb = res['pdb_label']
                fr  = compute_chi_fractions(
                    gmm_results, ion_per_run, res_name, chi_key, lpr_key)
                if fr is None:
                    continue
                prefix = pdb.split('.')[0]
                if prefix in ('152', '141'):
                    glu_fr[pdb] = fr
                elif prefix in ('184', '173'):
                    asnasp_fr[pdb] = fr

            if glu_fr:
                plot_2x2(glu_fr, pdb_order_glu, chi_label, method_name,
                         method_dir, args.channel, threshold, 'GLU')
            if asnasp_fr:
                plot_2x2(asnasp_fr, pdb_order_asnasp, chi_label,
                         method_name, method_dir, args.channel,
                         threshold, 'ASN_ASP')

            for pdb in focus_pdbs:
                fr = glu_fr.get(pdb) or asnasp_fr.get(pdb)
                if fr:
                    plot_focus(fr, pdb, chi_label, method_name,
                               method_dir, args.channel, threshold)

        # ── combination plots ─────────────────────────────────────────────
        print(f"\n  Combination plots...")
        glu_combo    = {}
        asnasp_combo = {}

        for res_name, res in gmm_results.items():
            pdb = res['pdb_label']
            cf  = compute_combo_fractions(
                gmm_results, ion_per_run, res_name, lpr_key)
            if cf is None:
                continue
            prefix = pdb.split('.')[0]
            if prefix in ('152', '141'):
                glu_combo[pdb] = cf
            elif prefix in ('184', '173'):
                asnasp_combo[pdb] = cf

        if glu_combo:
            plot_2x2(glu_combo, pdb_order_glu, 'χ1×χ2', method_name,
                     method_dir, args.channel, threshold, 'GLU',
                     is_combo=True)
        if asnasp_combo:
            plot_2x2(asnasp_combo, pdb_order_asnasp, 'χ1×χ2', method_name,
                     method_dir, args.channel, threshold, 'ASN_ASP',
                     is_combo=True)

        for pdb in focus_pdbs:
            cf = glu_combo.get(pdb) or asnasp_combo.get(pdb)
            if cf:
                plot_focus(cf, pdb, 'χ1×χ2', method_name,
                           method_dir, args.channel, threshold,
                           is_combo=True)

    print(f"\n{'='*70}")
    print(f"DONE. Results in: {output_dir}/")
    print(f"{'='*70}\n")
    return 0


if __name__ == '__main__':
    exit(main())
"""
python3 ion_proximity_gmm_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ion_proximity_gmm \
    -c G2

python3 ion_proximity_gmm_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ion_proximity_gmm \
    -c G12
    
python3 ion_proximity_gmm_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ion_proximity_gmm \
    -c G12_ML

python3 ion_proximity_gmm_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ion_proximity_gmm \
    -c G12_GAT
"""