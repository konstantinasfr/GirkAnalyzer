#!/usr/bin/env python3
"""
Ligand Distance per GMM State Analysis — G12_ML

For each residue and each chi angle, groups frames by their GMM-assigned
conformational state and shows the distribution of ligand distances per state.

Produces for each (distance_type × residue × chi):
    - Bar plot: mean ± std distance per GMM state
    - Overlaid semi-transparent scatter of individual frame distances
    - Done twice: probabilistic assignment and nearest-mean assignment

Inputs:
    - gmm_results.pkl          from gmm_analysis/ (contains labels_per_run
                               and labels_per_run_near for both methods)
    - combined_distance_dihedrals.pkl  from each RUN/ligand_distance_analysis/
                                       GLU_G1/ and ASP_G1/

Usage:
    python ligand_gmm_distance_analysis.py \
        --gmm-pkl  .../G12_ML/dihedral_analysis/gmm_analysis/gmm_results.pkl \
        --base-dir .../G12_ML \
        -o         .../G12_ML/ligand_gmm_distance
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from itertools import combinations
from pathlib import Path


COMPONENT_COLORS = ['#E63946', '#2A9D8F', '#FF6B00', '#6A0572']


# =============================================================================
# LOAD DISTANCES PER RUN
# =============================================================================

def load_distances_per_run(base_dir, subfolder):
    """
    Load min_distance arrays per run from combined_distance_dihedrals.pkl.

    Returns
    -------
    dist_per_run : dict  {run_name: np.ndarray of min distances}
    """
    base_dir     = Path(base_dir)
    dist_per_run = {}

    for run_dir in sorted(base_dir.glob('RUN*')):
        pkl_path = (run_dir / 'ligand_distance_analysis' /
                    subfolder / 'combined_distance_dihedrals.pkl')
        if not pkl_path.exists():
            print(f"  [SKIP] {run_dir.name}/{subfolder}: pkl not found")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        dist_per_run[run_dir.name] = data['min_distance']
        print(f"  Loaded {run_dir.name}/{subfolder}: "
              f"{len(data['min_distance'])} frames")

    return dist_per_run


# =============================================================================
# BUILD STATE→DISTANCES MAPPING
# =============================================================================

def build_state_distances(gmm_results, dist_per_run,
                          res_name, chi_key, label_key, lpr_key):
    """
    For each GMM state of (res_name, chi_key), collect all distances
    from frames assigned to that state.

    Parameters
    ----------
    gmm_results  : dict from gmm_results.pkl
    dist_per_run : dict {run_name: distance_array}
    res_name     : str  e.g. 'GLU_99'
    chi_key      : 'chi1' or 'chi2'
    label_key    : 'labels' or 'labels_nearest'
    lpr_key      : 'labels_per_run' or 'labels_per_run_near'

    Returns
    -------
    state_distances : dict {state_mean_angle: np.ndarray of distances}
    components      : list of component dicts (mean, std, weight)
    """
    if res_name not in gmm_results:
        return None, None
    if chi_key not in gmm_results[res_name]:
        return None, None

    chi_res    = gmm_results[res_name][chi_key]
    components = chi_res['components']
    lpr        = chi_res.get(lpr_key, {})

    # state_mean → list of distances
    state_distances = {round(c['mean']): [] for c in components}

    for run_name, dist_arr in dist_per_run.items():
        if run_name not in lpr:
            continue
        labels   = lpr[run_name]   # per-frame mean angle (rounded float)
        n_frames = min(len(labels), len(dist_arr))

        for fi in range(n_frames):
            state = round(float(labels[fi]))
            dist  = float(dist_arr[fi])
            if state in state_distances:
                state_distances[state].append(dist)

    # convert to arrays, sort by state value
    state_distances = {
        k: np.array(v)
        for k, v in sorted(state_distances.items())
    }
    return state_distances, components


# =============================================================================
# BUILD CHI1×CHI2 COMBINATION → DISTANCES MAPPING
# =============================================================================

def build_combo_distances(gmm_results, dist_per_run,
                          res_name, lpr_key):
    """
    For each chi1×chi2 combination state, collect all distances.

    Returns
    -------
    combo_distances : dict  {'chi1°/chi2°': np.ndarray of distances}
    """
    if res_name not in gmm_results:
        return None
    if 'chi1' not in gmm_results[res_name] or 'chi2' not in gmm_results[res_name]:
        return None

    lpr1 = gmm_results[res_name]['chi1'].get(lpr_key, {})
    lpr2 = gmm_results[res_name]['chi2'].get(lpr_key, {})

    combo_distances = {}

    for run_name, dist_arr in dist_per_run.items():
        if run_name not in lpr1 or run_name not in lpr2:
            continue
        l1       = lpr1[run_name]
        l2       = lpr2[run_name]
        n_frames = min(len(l1), len(l2), len(dist_arr))

        for fi in range(n_frames):
            m1    = round(float(l1[fi]))
            m2    = round(float(l2[fi]))
            combo = f'{m1}°/{m2}°'
            if combo not in combo_distances:
                combo_distances[combo] = []
            combo_distances[combo].append(float(dist_arr[fi]))

    # convert to arrays
    combo_distances = {k: np.array(v) for k, v in combo_distances.items()}
    return combo_distances


def _sort_combo_key(s):
    """Sort combo strings by abs(chi1) then abs(chi2)."""
    parts = s.replace('°', '').split('/')
    return (abs(float(parts[0])), abs(float(parts[1])))


# =============================================================================
# 2×2 COMBO BAR PLOT
# =============================================================================

def plot_2x2_combo_distances(all_combo_distances, pdb_order,
                              dist_label, method_name,
                              output_dir, channel_type, group=''):
    """
    2×2 bar+scatter plots where x = chi1×chi2 combination state,
    y = mean±std min distance. One panel per residue.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    axes      = axes.flatten()
    fig.suptitle(
        f'{channel_type}  —  {dist_label}  |  χ1×χ2 combinations  |  {method_name}',
        fontsize=18, fontweight='bold', y=1.01
    )

    for idx, pdb_label in enumerate(pdb_order):
        ax             = axes[idx]
        combo_dist     = all_combo_distances.get(pdb_label)
        if combo_dist is None or not combo_dist:
            ax.set_visible(False)
            continue

        # sort combos
        combos   = sorted(combo_dist.keys(), key=_sort_combo_key)
        x_pos    = np.arange(len(combos))
        bar_tops = []
        np.random.seed(42)

        for xi, combo in enumerate(combos):
            dists = combo_dist[combo]
            if len(dists) == 0:
                bar_tops.append(0)
                continue
            color = COMPONENT_COLORS[xi % len(COMPONENT_COLORS)]
            mean  = np.mean(dists)
            std   = np.std(dists)
            bar_tops.append(mean + std)

            n_s    = min(len(dists), 2000)
            sample = dists if len(dists) <= n_s else \
                     dists[np.random.choice(len(dists), n_s, replace=False)]
            jitter = np.random.uniform(-0.2, 0.2, size=len(sample))
            ax.scatter(xi + jitter, sample, color=color,
                       alpha=0.07, s=8, zorder=1)
            ax.bar(xi, mean, width=0.5, color=color, alpha=0.85,
                   edgecolor='black', linewidth=1.5, zorder=2,
                   label=f'{combo}  (n={len(dists):,})')
            ax.errorbar(xi, mean, yerr=std, fmt='none', color='black',
                        capsize=6, capthick=2, elinewidth=2, zorder=3)
            ax.text(xi, mean + std + 0.08, f'{mean:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    rotation=30)

        top = max(bar_tops) * 1.4 if bar_tops else 10
        ax.set_xticks(x_pos)
        ax.set_xticklabels(combos, fontsize=12, fontweight='bold',
                           rotation=30, ha='right')
        ax.set_ylabel('Min distance (Å)', fontsize=15, fontweight='bold')
        ax.set_ylim(0, top)
        ax.set_title(f'{pdb_label}', fontsize=20, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper right',
                  ncol=2 if len(combos) > 4 else 1)
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='y', labelsize=13)

    plt.tight_layout()
    safe = (f'combo_{dist_label}_{method_name}_{group}'
            .replace(' ', '_').replace('/', '_').replace('χ', 'chi'))
    out  = output_dir / f'2x2_combo_distances_{safe}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 2×2 combo plot: {out.name}')


# =============================================================================
# SINGLE SCATTER PLOT (one per angle)
# =============================================================================


# =============================================================================
# 2×2 BAR PLOT — NO SIGNIFICANCE BRACKETS
# =============================================================================

def plot_2x2_state_distances(all_state_distances, pdb_order,
                              chi_label, dist_label, method_name,
                              output_dir, channel_type, group=''):
    """2×2 bar+scatter plots, one panel per residue. No significance brackets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes      = axes.flatten()
    fig.suptitle(
        f'{channel_type}  —  {dist_label}  |  {chi_label}  |  {method_name}',
        fontsize=20, fontweight='bold', y=1.01
    )

    for idx, pdb_label in enumerate(pdb_order):
        ax              = axes[idx]
        state_distances = all_state_distances.get(pdb_label)
        if state_distances is None:
            ax.set_visible(False); continue
        states = sorted(state_distances.keys())
        if not states:
            ax.set_visible(False); continue

        x_pos    = np.arange(len(states))
        bar_tops = []
        np.random.seed(42)

        for xi, state in enumerate(states):
            dists = state_distances[state]
            if len(dists) == 0:
                bar_tops.append(0); continue
            color  = COMPONENT_COLORS[xi % len(COMPONENT_COLORS)]
            mean   = np.mean(dists)
            std    = np.std(dists)
            bar_tops.append(mean + std)

            n_s    = min(len(dists), 2000)
            sample = dists if len(dists) <= n_s else                      dists[np.random.choice(len(dists), n_s, replace=False)]
            jitter = np.random.uniform(-0.2, 0.2, size=len(sample))
            ax.scatter(xi + jitter, sample, color=color,
                       alpha=0.07, s=8, zorder=1)
            ax.bar(xi, mean, width=0.5, color=color, alpha=0.85,
                   edgecolor='black', linewidth=1.5, zorder=2,
                   label=f'{state}°  (n={len(dists):,})')
            ax.errorbar(xi, mean, yerr=std, fmt='none', color='black',
                        capsize=6, capthick=2, elinewidth=2, zorder=3)
            ax.text(xi, mean + std + 0.08, f'{mean:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        top = max(bar_tops) * 1.35 if bar_tops else 10
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s}°' for s in states],
                           fontsize=17, fontweight='bold')
        ax.set_ylabel('Min distance (Å)', fontsize=17, fontweight='bold')
        ax.set_ylim(0, top)
        ax.set_title(f'{pdb_label}', fontsize=22, fontweight='bold', pad=10)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='y', labelsize=15)

    plt.tight_layout()
    safe = (f'{chi_label}_{dist_label}_{method_name}_{group}'
            .replace(' ', '_').replace('/', '_').replace('χ', 'chi'))
    out  = output_dir / f'2x2_gmm_distances_{safe}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 2×2 bar plot: {out.name}')



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Distance per GMM state analysis'
    )
    parser.add_argument('--gmm-pkl',  required=True,
                        help='Path to gmm_results.pkl')
    parser.add_argument('--base-dir', required=True,
                        help='Base dir containing RUN* folders (e.g. .../G12_ML)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory')
    parser.add_argument('-c', '--channel', default='G12_ML')
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    gmm_pkl    = Path(args.gmm_pkl)
    output_dir = Path(args.output) if args.output \
                 else base_dir / 'ligand_gmm_distance'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LIGAND DISTANCE PER GMM STATE ANALYSIS")
    print("=" * 70)

    # load GMM results
    with open(gmm_pkl, 'rb') as f:
        gmm_results = pickle.load(f)
    print(f"Loaded GMM results: {len(gmm_results)} residues")

    # assignment methods
    methods = [
        ('probabilistic', 'labels',         'labels_per_run'),
        ('nearest_mean',  'labels_nearest',  'labels_per_run_near'),
    ]

    # distance types
    dist_types = [
        ('GLU_G1', 'GLU99 OE1/OE2'),
        ('ASP_G1', 'ASP131 OD1/OD2'),
    ]

    for subfolder, dist_label in dist_types:
        print(f"\n{'─'*70}")
        print(f"  Distance: {dist_label}")
        print(f"{'─'*70}")

        dist_per_run = load_distances_per_run(base_dir, subfolder)
        if not dist_per_run:
            continue

        # combine all distances + labels for scatter plots
        dist_all = np.concatenate(list(dist_per_run.values()))

        dist_out = output_dir / subfolder
        dist_out.mkdir(exist_ok=True)

        # ── scatter plots — 2x2 for GLU, 2x2 for ASN/ASP ────────────────
        print(f"\n  Creating scatter plots...")

        # load raw angles from combined_distance_dihedrals.pkl
        ang_per_res = {}
        pdb_labels  = {}
        for run_dir in sorted(base_dir.glob('RUN*')):
            pkl_path = (run_dir / 'ligand_distance_analysis' /
                        subfolder / 'combined_distance_dihedrals.pkl')
            if not pkl_path.exists():
                continue
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            for res_name, ang in data['dihedrals'].items():
                pdb_labels[res_name] = ang['pdb_label']
                if res_name not in ang_per_res:
                    ang_per_res[res_name] = {'chi1': [], 'chi2': []}
                for c in ['chi1', 'chi2']:
                    if ang[c] is not None:
                        ang_per_res[res_name][c].append(ang[c])

        # concatenate
        angles_raw = {}
        for res_name, arrs in ang_per_res.items():
            angles_raw[res_name] = {
                'pdb_label': pdb_labels[res_name],
                'chi1': np.concatenate(arrs['chi1']) if arrs['chi1'] else None,
                'chi2': np.concatenate(arrs['chi2']) if arrs['chi2'] else None,
            }

        # group by residue type
        pdb_order_glu    = ['152.B', '152.C', '152.A', '141.D']
        pdb_order_asnasp = ['184.B', '184.C', '184.A', '173.D']

        def _make_2x2_scatter(pdb_order, group_label):
            # build panels: (res_name, chi_key) in order chi1 then chi2
            # layout: row0=chi1, row1=chi2, cols=subunits → actually
            # 2x2 means 4 panels = 4 subunits, one chi at a time
            # so one 2x2 per chi angle
            for chi_key in ['chi1', 'chi2']:
                chi_label = 'χ1' if chi_key == 'chi1' else 'χ2'
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                fig.suptitle(
                    f'{args.channel}  —  {dist_label}  |  {group_label} {chi_label}\n'
                    f'y = min distance (Å)    N = {len(dist_all):,} frames',
                    fontsize=15, fontweight='bold'
                )
                for idx, pdb in enumerate(pdb_order):
                    ax = axes[idx]
                    # find res_name for this pdb_label
                    res_name = next((r for r, v in angles_raw.items()
                                     if v['pdb_label'] == pdb), None)
                    if res_name is None or angles_raw[res_name][chi_key] is None:
                        ax.set_visible(False)
                        continue

                    ang_arr = angles_raw[res_name][chi_key]
                    n_min   = min(len(ang_arr), len(dist_all))
                    ang_arr = ang_arr[:n_min]
                    d_arr   = dist_all[:n_min]

                    hb = ax.hexbin(ang_arr, d_arr, gridsize=80,
                                   cmap='YlOrRd', mincnt=1, bins='log')
                    plt.colorbar(hb, ax=ax, label='log(count)')

                    # running mean
                    bin_edges   = np.linspace(-180, 180, 91)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    run_means, run_x = [], []
                    for i in range(len(bin_edges) - 1):
                        mask = ((ang_arr >= bin_edges[i]) &
                                (ang_arr < bin_edges[i+1]))
                        if mask.sum() >= 5:
                            run_means.append(np.mean(d_arr[mask]))
                            run_x.append(bin_centers[i])
                    if run_x:
                        ax.plot(run_x, run_means, color='black',
                                lw=2.5, label='running mean', zorder=5)
                        ax.legend(fontsize=11)

                    pr, _ = scipy_stats.pearsonr(ang_arr,  d_arr)
                    sr, _ = scipy_stats.spearmanr(ang_arr, d_arr)
                    ax.set_xlabel(f'{chi_label} angle (°)',
                                  fontsize=13, fontweight='bold')
                    ax.set_ylabel('Min distance (Å)',
                                  fontsize=13, fontweight='bold')
                    ax.set_title(
                        f'{pdb} {chi_label}\n'
                        f'Pearson r = {pr:.3f}   Spearman ρ = {sr:.3f}',
                        fontsize=12, fontweight='bold'
                    )
                    ax.set_xlim(-180, 180)
                    ax.tick_params(labelsize=11)
                    ax.grid(True, alpha=0.2)

                plt.tight_layout()
                safe  = dist_label.replace(' ', '_').replace('/', '_')
                gsafe = group_label.replace('/', '_').replace(' ', '_')
                out   = dist_out / f'scatter_{safe}_{gsafe}_{chi_key}.png'
                plt.savefig(out, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✅ Scatter: {out.name}")

        _make_2x2_scatter(pdb_order_glu,    'GLU')
        _make_2x2_scatter(pdb_order_asnasp, 'ASN_ASP')

        # ── bar plots per GMM state, per method ───────────────────────────
        # group residues into GLU (152.x, 141.x) and ASN/ASP (184.x, 173.x)
        pdb_order_glu    = ['152.B', '152.C', '152.A', '141.D']
        pdb_order_asnasp = ['184.B', '184.C', '184.A', '173.D']

        for method_name, lbl_key, lpr_key in methods:
            print(f"\n  Method: {method_name}")
            method_dir = dist_out / method_name
            method_dir.mkdir(exist_ok=True)

            for chi_key in ['chi1', 'chi2']:
                chi_label = 'χ1' if chi_key == 'chi1' else 'χ2'

                # build {pdb_label: state_distances} for GLU and ASN/ASP
                glu_state_dist    = {}
                asnasp_state_dist = {}

                for res_name, res in gmm_results.items():
                    if chi_key not in res:
                        continue
                    pdb_label = res['pdb_label']
                    state_distances, components = build_state_distances(
                        gmm_results, dist_per_run,
                        res_name, chi_key, lbl_key, lpr_key
                    )
                    if state_distances is None:
                        continue
                    prefix = pdb_label.split('.')[0]
                    if prefix in ('152', '141'):
                        glu_state_dist[pdb_label] = state_distances
                    elif prefix in ('184', '173'):
                        asnasp_state_dist[pdb_label] = state_distances

                # GLU 2×2
                if glu_state_dist:
                    plot_2x2_state_distances(
                        glu_state_dist, pdb_order_glu,
                        chi_label, dist_label, method_name,
                        method_dir, args.channel, group='GLU'
                    )
                # ASN/ASP 2×2
                if asnasp_state_dist:
                    plot_2x2_state_distances(
                        asnasp_state_dist, pdb_order_asnasp,
                        chi_label, dist_label, method_name,
                        method_dir, args.channel, group='ASN_ASP'
                    )

            # ── chi1×chi2 combination bar plots (once per method, not per chi) ──
            print(f"\n  Combination distance plots...")
            glu_combo_dist    = {}
            asnasp_combo_dist = {}

            for res_name, res in gmm_results.items():
                pdb_label = res['pdb_label']
                combo_dist = build_combo_distances(
                    gmm_results, dist_per_run, res_name, lpr_key)
                if combo_dist is None:
                    continue
                prefix = pdb_label.split('.')[0]
                if prefix in ('152', '141'):
                    glu_combo_dist[pdb_label] = combo_dist
                elif prefix in ('184', '173'):
                    asnasp_combo_dist[pdb_label] = combo_dist

            if glu_combo_dist:
                plot_2x2_combo_distances(
                    glu_combo_dist, pdb_order_glu,
                    dist_label, method_name,
                    method_dir, args.channel, group='GLU'
                )
            if asnasp_combo_dist:
                plot_2x2_combo_distances(
                    asnasp_combo_dist, pdb_order_asnasp,
                    dist_label, method_name,
                    method_dir, args.channel, group='ASN_ASP'
                )

            # ── extra individual combo plots for 141.D and 173.D ──────────
            for focus_pdb, group_dist in [('141.D', glu_combo_dist),
                                           ('173.D', asnasp_combo_dist)]:
                combo_dist = group_dist.get(focus_pdb)
                if not combo_dist:
                    continue
                combos = sorted(combo_dist.keys(), key=_sort_combo_key)
                if not combos:
                    continue

                fig, ax = plt.subplots(figsize=(max(6, len(combos) * 1.8), 6))
                fig.suptitle(
                    f'{args.channel}  —  {dist_label}  |  '
                    f'{focus_pdb} χ1×χ2  |  {method_name}',
                    fontsize=16, fontweight='bold'
                )
                x_pos    = np.arange(len(combos))
                bar_tops = []
                np.random.seed(42)

                for xi, combo in enumerate(combos):
                    dists = combo_dist[combo]
                    if len(dists) == 0:
                        bar_tops.append(0); continue
                    color = COMPONENT_COLORS[xi % len(COMPONENT_COLORS)]
                    mean  = np.mean(dists)
                    std   = np.std(dists)
                    bar_tops.append(mean + std)

                    n_s    = min(len(dists), 2000)
                    sample = dists if len(dists) <= n_s else \
                             dists[np.random.choice(len(dists), n_s, replace=False)]
                    jitter = np.random.uniform(-0.2, 0.2, size=len(sample))
                    ax.scatter(xi + jitter, sample, color=color,
                               alpha=0.07, s=8, zorder=1)
                    ax.bar(xi, mean, width=0.5, color=color, alpha=0.85,
                           edgecolor='black', linewidth=1.5, zorder=2,
                           label=f'{combo}  (n={len(dists):,})')
                    ax.errorbar(xi, mean, yerr=std, fmt='none', color='black',
                                capsize=6, capthick=2, elinewidth=2, zorder=3)
                    ax.text(xi, mean + std + 0.08, f'{mean:.2f}±{std:.2f}',
                            ha='center', va='bottom', fontsize=13,
                            fontweight='bold', rotation=30)

                top = max(bar_tops) * 1.4 if bar_tops else 10
                ax.set_xticks(x_pos)
                ax.set_xticklabels(combos, fontsize=13, fontweight='bold',
                                   rotation=30, ha='right')
                ax.set_ylabel('Min distance (Å)', fontsize=15, fontweight='bold')
                ax.set_ylim(0, top)
                ax.set_title(f'{focus_pdb}', fontsize=20, fontweight='bold', pad=10)
                ax.legend(fontsize=10, loc='upper right',
                          ncol=2 if len(combos) > 4 else 1)
                ax.grid(axis='y', alpha=0.25)
                ax.tick_params(axis='y', labelsize=14)
                plt.tight_layout()

                safe = (f'{focus_pdb}_combo_{dist_label}_{method_name}'
                        .replace(' ', '_').replace('/', '_').replace('χ', 'chi'))
                out  = method_dir / f'focus_combo_{safe}.png'
                plt.savefig(out, dpi=130, bbox_inches='tight')
                plt.close()
                print(f'  ✅ Focus combo plot: {out.name}')

                # ── extra individual plots for 141.D and 173.D ────────────
                for focus_pdb, group_dist in [('141.D', glu_state_dist),
                                              ('173.D', asnasp_state_dist)]:
                    sd = group_dist.get(focus_pdb)
                    if sd is None:
                        continue
                    states = sorted(sd.keys())
                    if not states:
                        continue

                    fig, ax = plt.subplots(figsize=(max(5, len(states)*2.5), 6))
                    fig.suptitle(
                        f'{args.channel}  —  {dist_label}  |  '
                        f'{focus_pdb} {chi_label}  |  {method_name}',
                        fontsize=16, fontweight='bold'
                    )
                    x_pos    = np.arange(len(states))
                    bar_tops = []
                    np.random.seed(42)

                    for xi, state in enumerate(states):
                        dists  = sd[state]
                        if len(dists) == 0:
                            bar_tops.append(0); continue
                        color  = COMPONENT_COLORS[xi % len(COMPONENT_COLORS)]
                        mean   = np.mean(dists)
                        std    = np.std(dists)
                        bar_tops.append(mean + std)

                        n_s    = min(len(dists), 2000)
                        sample = dists if len(dists) <= n_s else \
                                 dists[np.random.choice(len(dists), n_s,
                                                        replace=False)]
                        jitter = np.random.uniform(-0.2, 0.2, size=len(sample))
                        ax.scatter(xi + jitter, sample, color=color,
                                   alpha=0.07, s=8, zorder=1)
                        ax.bar(xi, mean, width=0.5, color=color, alpha=0.85,
                               edgecolor='black', linewidth=1.5, zorder=2,
                               label=f'{state}°  (n={len(dists):,})')
                        ax.errorbar(xi, mean, yerr=std, fmt='none',
                                    color='black', capsize=6, capthick=2,
                                    elinewidth=2, zorder=3)
                        ax.text(xi, mean + std + 0.08,
                                f'{mean:.2f}±{std:.2f}',
                                ha='center', va='bottom',
                                fontsize=14, fontweight='bold')

                    top = max(bar_tops) * 1.35 if bar_tops else 10
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'{s}°' for s in states],
                                       fontsize=17, fontweight='bold')
                    ax.set_ylabel('Min distance (Å)', fontsize=17,
                                  fontweight='bold')
                    ax.set_ylim(0, top)
                    ax.set_title(f'{focus_pdb}', fontsize=20,
                                 fontweight='bold', pad=10)
                    ax.legend(fontsize=12, loc='upper right')
                    ax.grid(axis='y', alpha=0.25)
                    ax.tick_params(axis='y', labelsize=15)
                    plt.tight_layout()

                    safe = (f"{focus_pdb}_{chi_label}_{dist_label}_{method_name}"
                            .replace(' ', '_').replace('/', '_').replace('χ', 'chi'))
                    out  = method_dir / f'focus_{safe}.png'
                    plt.savefig(out, dpi=130, bbox_inches='tight')
                    plt.close()
                    print(f"  ✅ Focus plot: {out.name}")

    print(f"\n{'='*70}")
    print(f"DONE. Results in: {output_dir}/")
    print(f"  GLU_G1/scatter/          — scatter plots (hexbin)")
    print(f"  GLU_G1/probabilistic/    — bar plots, probabilistic assignment")
    print(f"  GLU_G1/nearest_mean/     — bar plots, nearest-mean assignment")
    print(f"  ASP_G1/...               — same for ASP131 distance")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
"""
python3 ligand_correlation_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ligand_correlation

python3 ligand_correlation_analysis.py \
    --gmm-pkl /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/gmm_analysis_4gchi2/gmm_results.pkl \
    --base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ligand_correlation_4gchi2

python3 ligand_correlation_analysis.py \
 /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ligand_correlation

"""