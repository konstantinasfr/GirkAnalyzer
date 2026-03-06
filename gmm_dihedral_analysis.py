#!/usr/bin/env python3
"""
GMM Dihedral Analysis - Peak-guided fitting with per-residue rules

Fitting rules:
    Residue type         | chi1 K | chi1 space | chi2 K | chi2 space
    ---------------------|--------|------------|--------|------------
    GLU (label=141.D)    |   2    | [0,360]    |   2    | [0,360]
    GLU (all others)     |   2    | [0,360]    |   3    | [0,360]
    ASN                  |   2    | [0,360]    |   3    | [-180,180]
    ASP                  |   2    | [0,360]    |   2    | [100,460]

    If 141.D or ASP residues do not exist in the data, they are skipped silently.

Peak-guided initialisation:
    - Detected peaks from combined_peak_analysis.json are used as means_init.
    - If detected peaks < required K, the remaining components are initialised
      at the midpoints of the largest gaps between known peaks.
    - All means are converted back to [-180, 180] for reporting and plotting.

Special overrides:
    G12_ML + ASN + chi2: always forces a third Gaussian init at -21.4°,
    even if the peak detector already found a peak there or nearby.

Usage:
    python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json
    python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json -c G12_ML
    python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json -o /out/
"""

import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture


# =============================================================================
# PER-RUN HELPER
# =============================================================================

def get_all_frames(chi_data):
    """
    Return concatenated numpy array of all frames in sorted run order.
    Handles both new per-run dict format and old concatenated array format.
    IMPORTANT: uses sorted key order to match get_run_names().
    """
    if isinstance(chi_data, dict):
        arrays = [chi_data[k] for k in sorted(chi_data.keys())
                  if chi_data[k] is not None and len(chi_data[k]) > 0]
        return np.concatenate(arrays) if arrays else np.array([])
    return np.asarray(chi_data)


def get_run_names(chi_data):
    """Return sorted run names if per-run dict, else ['ALL']."""
    if isinstance(chi_data, dict):
        return sorted(chi_data.keys())
    return ['ALL']


# =============================================================================
# RULES
# =============================================================================

def get_rules(res_type, pdb_label):
    """
    Return (chi1_k, chi1_space, chi2_k, chi2_space) for a residue.
    Returns None if the residue should be skipped.

    Rules are determined by PDB label PREFIX (reliable) not res_type
    (which can be wrong due to topology labelling):
        152.x  → GLU (standard):  chi1 K=2 [0,360]  chi2 K=3 [0,360]
        141.x  → GLU (special):   chi1 K=2 [0,360]  chi2 K=2 [0,360]
        184.x  → ASN:             chi1 K=2 [0,360]  chi2 K=3 [-180,180]
        173.x  → ASP:             chi1 K=2 [0,360]  chi2 K=2 [100,460]

    Spaces:
        '0_360'    : shift negatives +360  (cut at 0°)
        '100_460'  : shift values <100° by +360  (cut at 100°)
        '-180_180' : no conversion
    """
    # derive family from PDB label prefix (e.g. '184.A' → '184')
    prefix = pdb_label.split('.')[0] if '.' in pdb_label else ''

    if prefix == '152':       # GLU standard
        return 2, '0_360', 3, '0_360'
    elif prefix == '141':     # GLU special (e.g. 141.D)
        return 2, '0_360', 2, '0_360'
    elif prefix == '184':     # ASN
        return 2, '0_360', 3, '-180_180'
    elif prefix == '173':     # ASP
        return 2, '0_360', 2, '100_460'
    else:
        # unknown label — skip silently
        return None


# =============================================================================
# SPACE CONVERSION UTILITIES
# =============================================================================

def convert_angles(angles, space):
    """Convert raw angles array to the requested fitting space."""
    if space == '0_360':
        return np.where(angles < 0, angles + 360, angles)
    elif space == '100_460':
        return np.where(angles < 100, angles + 360, angles)
    else:  # '-180_180'
        return angles.copy()


def convert_peak(peak, space):
    """Convert a single peak position to the fitting space."""
    if space == '0_360':
        return peak + 360 if peak < 0 else peak
    elif space == '100_460':
        return peak + 360 if peak < 100 else peak
    else:
        return peak


def back_to_180(value):
    """Convert any fitting-space value back to [-180, 180]."""
    if value > 180:
        return value - 360
    return value


# =============================================================================
# INITIALISATION HELPERS
# =============================================================================

def build_means_init(known_peaks_in_space, k, data_in_space):
    """
    Build a list of K initial means for the GMM.

    - The first len(known_peaks) slots are filled with the detected peaks
      (already converted to the fitting space).
    - Any remaining slots (when k > len(known_peaks)) are filled with the
      midpoint of the largest gap between consecutive known peaks.
      If there are still not enough, additional midpoints are taken from
      the next largest gaps.

    Parameters
    ----------
    known_peaks_in_space : list of float   peaks already in fitting space
    k                    : int             required number of components
    data_in_space        : np.ndarray      angles in fitting space (for fallback)

    Returns
    -------
    list of float, length k
    """
    n_known = len(known_peaks_in_space)

    if n_known >= k:
        # more peaks than needed — use the k most prominent ones (keep order)
        return sorted(known_peaks_in_space)[:k]

    means = sorted(known_peaks_in_space)
    n_extra = k - n_known

    if n_extra > 0:
        # add data range boundaries to compute gaps at the edges too
        data_min = float(np.min(data_in_space))
        data_max = float(np.max(data_in_space))
        extended = [data_min] + means + [data_max]

        # gaps between consecutive points (including edges)
        gaps = []
        for i in range(len(extended) - 1):
            gap_size = extended[i + 1] - extended[i]
            midpoint = (extended[i] + extended[i + 1]) / 2.0
            gaps.append((gap_size, midpoint))

        # sort gaps largest first
        gaps.sort(key=lambda x: -x[0])

        for idx in range(n_extra):
            means.append(gaps[idx][1])

        means = sorted(means)

    return means


# =============================================================================
# GMM FITTING
# =============================================================================

def fit_gmm_with_init(data, means_init, n_init=20, random_state=42):
    """
    Fit a GMM with K = len(means_init), using means_init as starting point
    plus n_init additional random restarts. Returns the best fit.

    Parameters
    ----------
    data       : np.ndarray (N,)   angles in fitting space
    means_init : list of float     initial means in fitting space
    n_init     : int               extra random restarts

    Returns
    -------
    best_gmm : GaussianMixture
    """
    k       = len(means_init)
    data_2d = data.reshape(-1, 1)

    # guided initialisation
    gmm_guided = GaussianMixture(
        n_components=k,
        covariance_type='full',
        n_init=1,
        means_init=np.array(means_init).reshape(-1, 1),
        random_state=random_state
    )
    gmm_guided.fit(data_2d)
    best_score = gmm_guided.score(data_2d)
    best_gmm   = gmm_guided

    # additional random restarts
    if n_init > 1:
        gmm_random = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=n_init,
            random_state=random_state + 1
        )
        gmm_random.fit(data_2d)
        if gmm_random.score(data_2d) > best_score:
            best_gmm = gmm_random

    return best_gmm


def extract_components(gmm):
    """
    Extract GMM components sorted by mean (in [-180, 180]).

    Returns
    -------
    list of dicts: [{'mean', 'std', 'weight'}, ...]
    """
    means   = np.array([back_to_180(m) for m in gmm.means_[:, 0]])
    stds    = np.sqrt(gmm.covariances_[:, 0, 0])
    weights = gmm.weights_

    order = np.argsort(means)
    return [
        {
            'mean':   float(means[i]),
            'std':    float(stds[i]),
            'weight': float(weights[i])
        }
        for i in order
    ]


def assign_labels(gmm, data):
    """
    METHOD 1: Probabilistic assignment (GMM maximum likelihood).
    Assigns each frame to the component with highest probability — which
    depends on both distance AND Gaussian width (std). A wider Gaussian
    can claim frames that are geometrically closer to a narrower one.
    """
    raw_labels = gmm.predict(data.reshape(-1, 1))
    means_180  = np.array([back_to_180(m) for m in gmm.means_[:, 0]])
    return means_180[raw_labels]


def assign_labels_nearest(gmm, data_raw):
    """
    METHOD 2: Nearest-mean assignment (geometric / hard boundary).
    Assigns each frame to the component whose mean is geometrically
    closest, using CIRCULAR distance to handle ±180° wrap correctly.
    Ignores Gaussian width — purely based on angle proximity.

    Parameters
    ----------
    gmm      : fitted GaussianMixture
    data_raw : np.ndarray (N,)  raw angles in [-180, 180] (NOT fitting space)
    """
    means_180 = np.array([back_to_180(m) for m in gmm.means_[:, 0]])

    def circ_dist(a, b):
        d = np.abs(a - b) % 360
        return np.where(d > 180, 360 - d, d)

    # for each frame find the component with smallest circular distance
    # distances shape: (N, K)
    distances  = np.stack([circ_dist(data_raw, m) for m in means_180], axis=1)
    raw_labels = np.argmin(distances, axis=1)
    return means_180[raw_labels]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_gmm_analysis(combined_data, peak_analysis, output_dir, n_init=20, channel_type='G12'):
    """
    Fit GMMs to chi1 and chi2 for every residue using peak-guided initialisation.

    Parameters
    ----------
    combined_data : dict   from combined_dihedral_data.pkl['combined_data']
    peak_analysis : dict   from combined_peak_analysis.json
    output_dir    : Path
    n_init        : int    extra random restarts per fit
    channel_type  : str    e.g. 'G12_ML' triggers special overrides

    Returns
    -------
    gmm_results : dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GMM ANALYSIS  —  peak-guided initialisation")
    print("=" * 70)
    print("Rules (by PDB label prefix):")
    print("  152.x (GLU standard) chi1: K=2 [0,360]   chi2: K=3 [0,360]")
    print("  141.x (GLU special)  chi1: K=2 [0,360]   chi2: K=2 [0,360]")
    print("  184.x (ASN)          chi1: K=2 [0,360]   chi2: K=3 [-180,180]")
    print("  173.x (ASP)          chi1: K=2 [0,360]   chi2: K=2 [100,460]")
    print("  G12_ML overrides: 184.x chi2 forces -21.4°; 184.A forces K=4")
    print("=" * 70)

    gmm_results  = {}
    report_lines = []
    report_lines.append("GMM ANALYSIS REPORT\n" + "=" * 70 + "\n\n")
    report_lines.append(
        "Rules:\n"
        "  GLU (141.D)  chi1: K=2 [0,360]   chi2: K=2 [0,360]\n"
        "  GLU (others) chi1: K=2 [0,360]   chi2: K=3 [0,360]\n"
        "  ASN          chi1: K=2 [0,360]   chi2: K=3 [-180,180]\n"
        "  ASP          chi1: K=2 [0,360]   chi2: K=2 [100,460]\n\n"
    )
    report_lines.append("=" * 70 + "\n\n")

    for res_name, angles in combined_data.items():
        pdb_label = angles.get('pdb_label', res_name)
        res_type  = angles.get('residue_type', 'UNK')

        # get rules — returns None for residues to skip
        rules = get_rules(res_type, pdb_label)
        if rules is None:
            print(f"  [SKIP] {res_name} (PDB: {pdb_label}) [{res_type}] — no rules defined")
            continue

        chi1_k, chi1_space, chi2_k, chi2_space = rules

        # get detected peaks from JSON
        if res_name not in peak_analysis:
            print(f"  [SKIP] {res_name} not found in peak_analysis JSON")
            continue

        pa         = peak_analysis[res_name]
        chi1_peaks = pa.get('chi1_peaks', [])
        chi2_peaks = pa.get('chi2_peaks', [])

        print(f"\n{'─' * 60}")
        print(f"  {res_name}  (PDB: {pdb_label})  [{res_type}]")
        print(f"  chi1: K={chi1_k} space={chi1_space}  |  chi2: K={chi2_k} space={chi2_space}")
        print(f"  chi1 peaks from JSON: {chi1_peaks}")
        print(f"  chi2 peaks from JSON: {chi2_peaks}")
        print(f"{'─' * 60}")

        report_lines.append(f"{res_name}  (PDB: {pdb_label})  [{res_type}]\n")
        report_lines.append(
            f"  chi1: K={chi1_k} space={chi1_space}  |  "
            f"chi2: K={chi2_k} space={chi2_space}\n"
        )
        report_lines.append("─" * 60 + "\n")

        res_gmm = {'pdb_label': pdb_label, 'residue_type': res_type}

        # ------------------------------------------------------------------
        # CHI1
        # ------------------------------------------------------------------
        # concatenate all runs for fitting
        chi1_raw = get_all_frames(angles['chi1'])
        chi1_fit = convert_angles(chi1_raw, chi1_space)

        # convert known peaks to fitting space
        chi1_peaks_fit = [convert_peak(p, chi1_space) for p in chi1_peaks]

        # build init means (fill gaps if needed)
        chi1_init = build_means_init(chi1_peaks_fit, chi1_k, chi1_fit)

        print(f"  chi1 init means ({chi1_space}): {[f'{m:.1f}' for m in chi1_init]}")
        report_lines.append(
            f"\nchi1  (K={chi1_k}, space={chi1_space})\n"
            f"  init means: {[round(m,1) for m in chi1_init]}\n"
        )

        gmm1              = fit_gmm_with_init(chi1_fit, chi1_init, n_init=n_init)
        comps1            = extract_components(gmm1)
        labels1_all       = assign_labels(gmm1, chi1_fit)          # method 1: probabilistic
        labels1_all_near  = assign_labels_nearest(gmm1, chi1_raw)  # method 2: nearest mean

        # split labels back into per-run dicts
        labels1_per_run      = {}
        labels1_per_run_near = {}
        if isinstance(angles['chi1'], dict):
            idx = 0
            for run_name in get_run_names(angles['chi1']):
                n = len(angles['chi1'][run_name])
                labels1_per_run[run_name]      = labels1_all[idx: idx + n]
                labels1_per_run_near[run_name] = labels1_all_near[idx: idx + n]
                idx += n
        else:
            labels1_per_run['ALL']      = labels1_all
            labels1_per_run_near['ALL'] = labels1_all_near

        print(f"  chi1 result (K={chi1_k}):")
        for i, c in enumerate(comps1):
            pct = c['weight'] * 100
            print(f"    G{i+1}: mean={c['mean']:7.1f}°  std={c['std']:5.1f}°  occupancy={pct:5.1f}%")
            report_lines.append(
                f"  G{i+1}: mean = {c['mean']:7.1f}°   std = {c['std']:5.1f}°   "
                f"occupancy = {pct:5.1f}%\n"
            )

        res_gmm['chi1'] = {
            'k': chi1_k, 'space': chi1_space,
            'components':          comps1,
            'labels':              labels1_all,
            'labels_per_run':      labels1_per_run,
            'labels_nearest':      labels1_all_near,
            'labels_per_run_near': labels1_per_run_near
        }

        # ------------------------------------------------------------------
        # CHI2
        # ------------------------------------------------------------------
        chi2_raw = get_all_frames(angles['chi2'])
        chi2_fit = convert_angles(chi2_raw, chi2_space)

        chi2_peaks_fit = [convert_peak(p, chi2_space) for p in chi2_peaks]

        # ── G12_ML special overrides ────────────────────────────────────────
        # 184.x (ASN) chi2: take top-2 peaks by count + force -21.4° third
        # 184.A specifically: force K=4
        prefix = pdb_label.split('.')[0] if '.' in pdb_label else ''

        if channel_type == 'G12_ML' and prefix == '184':
            forced_peak     = -21.4
            forced_in_space = convert_peak(forced_peak, chi2_space)

            # override K for 184.A specifically
            if pdb_label == '184.A':
                chi2_k = 4
                print(f"  [G12_ML override] 184.A: forcing K=4 for chi2")
                report_lines.append("  [G12_ML override] 184.A: forcing K=4\n")

            # rank detected peaks by histogram count, keep top 2
            hist_counts, hist_bins = np.histogram(chi2_raw, bins=72, range=(-180, 180))
            hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

            def _peak_count(peak_orig):
                idx = np.argmin(np.abs(hist_centers - peak_orig))
                return hist_counts[idx]

            peaks_with_counts = sorted(
                chi2_peaks,
                key=lambda p: _peak_count(p),
                reverse=True
            )
            top2_peaks    = peaks_with_counts[:2]
            top2_in_space = [convert_peak(p, chi2_space) for p in top2_peaks]

            # remove any peak within 15° of forced peak to avoid duplicates
            top2_in_space = [p for p in top2_in_space
                             if abs(back_to_180(p) - forced_peak) > 15]

            # final: top-2 + forced -21.4°
            chi2_peaks_fit = top2_in_space + [forced_in_space]

            print(f"  [G12_ML override] 184.x chi2: top-2 peaks = "
                  f"{[f'{p:.1f}' for p in top2_peaks]}, "
                  f"forced third = {forced_peak}°")
            report_lines.append(
                f"  [G12_ML override] 184.x chi2: top-2 = {top2_peaks}, "
                f"forced third = {forced_peak}°\n"
            )

        chi2_init = build_means_init(chi2_peaks_fit, chi2_k, chi2_fit)

        print(f"  chi2 init means ({chi2_space}): {[f'{m:.1f}' for m in chi2_init]}")
        report_lines.append(
            f"\nchi2  (K={chi2_k}, space={chi2_space})\n"
            f"  init means: {[round(m,1) for m in chi2_init]}\n"
        )

        gmm2              = fit_gmm_with_init(chi2_fit, chi2_init, n_init=n_init)
        comps2            = extract_components(gmm2)
        labels2_all       = assign_labels(gmm2, chi2_fit)          # method 1: probabilistic
        labels2_all_near  = assign_labels_nearest(gmm2, chi2_raw)  # method 2: nearest mean

        # split labels back into per-run dicts
        labels2_per_run      = {}
        labels2_per_run_near = {}
        if isinstance(angles['chi2'], dict):
            idx = 0
            for run_name in get_run_names(angles['chi2']):
                n = len(angles['chi2'][run_name])
                labels2_per_run[run_name]      = labels2_all[idx: idx + n]
                labels2_per_run_near[run_name] = labels2_all_near[idx: idx + n]
                idx += n
        else:
            labels2_per_run['ALL']      = labels2_all
            labels2_per_run_near['ALL'] = labels2_all_near

        print(f"  chi2 result (K={chi2_k}):")
        for i, c in enumerate(comps2):
            pct = c['weight'] * 100
            print(f"    G{i+1}: mean={c['mean']:7.1f}°  std={c['std']:5.1f}°  occupancy={pct:5.1f}%")
            report_lines.append(
                f"  G{i+1}: mean = {c['mean']:7.1f}°   std = {c['std']:5.1f}°   "
                f"occupancy = {pct:5.1f}%\n"
            )

        res_gmm['chi2'] = {
            'k': chi2_k, 'space': chi2_space,
            'components':          comps2,
            'labels':              labels2_all,
            'labels_per_run':      labels2_per_run,
            'labels_nearest':      labels2_all_near,
            'labels_per_run_near': labels2_per_run_near
        }

        gmm_results[res_name] = res_gmm
        report_lines.append("\n" + "=" * 70 + "\n\n")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    report_path = output_dir / 'gmm_report.txt'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"\n✅ Report saved to: {report_path}")

    json_summary = {}
    for res_name, res in gmm_results.items():
        entry = {
            'pdb_label':    res['pdb_label'],
            'residue_type': res['residue_type'],
        }
        for chi in ['chi1', 'chi2']:
            if chi in res:
                entry[chi] = {
                    'k':          res[chi]['k'],
                    'space':      res[chi]['space'],
                    'components': res[chi]['components']
                }
        json_summary[res_name] = entry

    json_path = output_dir / 'gmm_summary.json'
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"✅ JSON summary saved to: {json_path}")

    pkl_path = output_dir / 'gmm_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(gmm_results, f)
    print(f"✅ Full results (with frame labels) saved to: {pkl_path}")

    return gmm_results


# =============================================================================
# PLOTTING
# =============================================================================

COMPONENT_COLORS = ['#E63946', '#2A9D8F', '#FF6B00', '#6A0572']
#                    G1: red     G2: teal    G3: intense orange  G4: purple


def _gaussian_curve(x, mean, std, weight, n_frames, bin_width):
    """Gaussian scaled to histogram counts."""
    amplitude = weight * n_frames * bin_width
    return amplitude / (std * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


def _plot_2x2_gmm(ordered_keys, combined_data, gmm_results,
                  chi_key, chi_label, bar_color, filename, output_dir,
                  channel_type=''):
    """
    2×2 grid: histogram in [-180,180] with GMM components overlaid.
    Periodic copies are added for shifted spaces to handle wraparound display.
    """
    if len(ordered_keys) < 4:
        print(f"  [SKIP] Not enough residues for {filename}")
        return

    BIN_WIDTH = 5
    N_BINS    = 72

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    if channel_type:
        fig.suptitle(f'{channel_type}  —  {chi_label} GMM fits',
                     fontsize=24, fontweight='bold', y=1.01)

    for idx, res_name in enumerate(ordered_keys[:4]):
        ax        = axes[idx]
        raw       = get_all_frames(combined_data[res_name][chi_key])
        pdb_label = combined_data[res_name].get('pdb_label', res_name)
        res_gmm   = gmm_results.get(res_name, {}).get(chi_key)

        # histogram always in [-180, 180]
        hist, bins = np.histogram(raw, bins=N_BINS, range=(-180, 180))
        centers    = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, hist, width=BIN_WIDTH, alpha=0.45,
               color=bar_color, edgecolor='none')

        x_plot = np.linspace(-180, 180, 1000)

        if res_gmm is not None:
            n_frames    = len(raw)
            components  = res_gmm['components']   # means in [-180, 180]
            space       = res_gmm['space']
            total_curve = np.zeros_like(x_plot)
            legend_patches = [mpatches.Patch(color=bar_color, alpha=0.5, label='Data')]

            for i, comp in enumerate(components):
                mean   = comp['mean']
                std    = comp['std']
                weight = comp['weight']
                color  = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]

                curve = _gaussian_curve(x_plot, mean, std, weight, n_frames, BIN_WIDTH)

                # periodic copies for shifted spaces
                if space in ('0_360', '100_460'):
                    curve += _gaussian_curve(x_plot, mean + 360, std, weight, n_frames, BIN_WIDTH)
                    curve += _gaussian_curve(x_plot, mean - 360, std, weight, n_frames, BIN_WIDTH)

                total_curve += curve
                ax.plot(x_plot, curve, color=color, lw=2.5, ls='--')
                ax.axvline(mean, color=color, lw=1.5, ls=':', alpha=0.8)
                legend_patches.append(mpatches.Patch(
                    color=color,
                    label=f'G{i+1}: {mean:.0f}° ± {std:.0f}°  ({weight*100:.1f}%)'
                ))

            ax.plot(x_plot, total_curve, color='black', lw=2.5)
            legend_patches.append(
                mpatches.Patch(color='black', label=f'Total (K={res_gmm["k"]})')
            )
            ax.legend(handles=legend_patches, fontsize=17,
                      loc='upper right', framealpha=0.85)

        ax.set_title(f'{pdb_label}', fontsize=30, fontweight='bold', pad=12)
        ax.set_xlabel(f'{chi_label} Angle (degrees)', fontsize=26, fontweight='bold')
        ax.set_ylabel('Count', fontsize=26, fontweight='bold')
        ax.set_xlim(-180, 180)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def _plot_2x2_occupancy(ordered_keys, combined_data, gmm_results,
                        chi_key, chi_label, filename, output_dir,
                        channel_type=''):
    """
    2×2 grid of bar plots: one bar per Gaussian component per residue.
    X axis: component label showing mean ± std
    Y axis: occupancy (%)
    Bars coloured consistently with the GMM fit plots.
    """
    if len(ordered_keys) < 4:
        print(f"  [SKIP] Not enough residues for {filename}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    if channel_type:
        fig.suptitle(f'{channel_type}  —  {chi_label} occupancy',
                     fontsize=24, fontweight='bold', y=1.01)

    for idx, res_name in enumerate(ordered_keys[:4]):
        ax        = axes[idx]
        pdb_label = combined_data[res_name].get('pdb_label', res_name)
        res_gmm   = gmm_results.get(res_name, {}).get(chi_key)

        if res_gmm is None:
            ax.set_title(f'{pdb_label}', fontsize=30, fontweight='bold')
            ax.set_visible(False)
            continue

        components = res_gmm['components']
        k          = res_gmm['k']
        labels_all = res_gmm['labels']   # per-frame mean angle values
        n_frames   = len(labels_all)

        x_labels  = [f'{c["mean"]:.0f}°\n±{c["std"]:.0f}°' for c in components]
        # count frames whose label == this component mean
        counts    = [int(np.sum(np.isclose(labels_all, c['mean'], atol=0.01)))
                     for c in components]
        occupancy = [cnt / n_frames * 100 for cnt in counts]
        colors    = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)] for i in range(k)]

        bars = ax.bar(range(k), occupancy, color=colors,
                      alpha=0.85, edgecolor='black', linewidth=2)

        # value labels on top of each bar — percentage + count
        for bar, occ, cnt in zip(bars, occupancy, counts):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{occ:.1f}%\n({cnt:,})',
                    ha='center', va='bottom', fontsize=17, fontweight='bold')

        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels, fontsize=22, fontweight='bold')
        ax.set_ylabel('Occupancy (%)', fontsize=26, fontweight='bold')
        ax.set_ylim(0, 125)
        ax.set_title(f'{pdb_label}  —  {chi_label}', fontsize=30,
                     fontweight='bold', pad=12)
        ax.tick_params(axis='y', labelsize=22)
        ax.grid(axis='y', alpha=0.3, linewidth=1)

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_gmm_fits(combined_data, gmm_results, output_dir, channel_type='G12'):
    """Create 2×2 GMM fit plots (histogram + Gaussian overlay). Same for both methods."""
    output_dir = Path(output_dir)

    fits_dir = output_dir / 'gmm_fits'
    fits_dir.mkdir(exist_ok=True)

    if channel_type == 'G2':
        pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
    else:
        pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']

    def _order(keys, pdb_order):
        ordered = []
        for pdb in pdb_order:
            for k in keys:
                if combined_data[k].get('pdb_label') == pdb:
                    ordered.append(k)
                    break
        return ordered

    analysed = set(gmm_results.keys())

    asn_asp_keys = [k for k, v in combined_data.items()
                    if v.get('pdb_label', '').split('.')[0] in ('184', '173')
                    and k in analysed]
    glu_keys     = [k for k, v in combined_data.items()
                    if v.get('pdb_label', '').split('.')[0] in ('152', '141')
                    and k in analysed]

    ordered_asn = _order(asn_asp_keys, pdb_order_asn)
    ordered_glu = _order(glu_keys,     pdb_order_glu)

    print("\n=== CREATING GMM FIT PLOTS → gmm_fits/ ===")
    _plot_2x2_gmm(ordered_asn, combined_data, gmm_results,
                  'chi1', 'χ1', 'steelblue',
                  'ASN_ASP_chi1_fits.png', fits_dir, channel_type)
    _plot_2x2_gmm(ordered_asn, combined_data, gmm_results,
                  'chi2', 'χ2', 'coral',
                  'ASN_ASP_chi2_fits.png', fits_dir, channel_type)
    _plot_2x2_gmm(ordered_glu, combined_data, gmm_results,
                  'chi1', 'χ1', 'steelblue',
                  'GLU_chi1_fits.png', fits_dir, channel_type)
    _plot_2x2_gmm(ordered_glu, combined_data, gmm_results,
                  'chi2', 'χ2', 'coral',
                  'GLU_chi2_fits.png', fits_dir, channel_type)

    return ordered_asn, ordered_glu


def plot_gmm_occupancy(combined_data, gmm_results, output_dir, channel_type='G12',
                       ordered_asn=None, ordered_glu=None):
    """
    Create 2×2 occupancy bar plots. Method-dependent — uses whichever
    labels are currently active in gmm_results['labels'].
    Must be called separately for each assignment method.
    """
    output_dir = Path(output_dir)

    occupancy_dir = output_dir / 'gmm_occupancy'
    occupancy_dir.mkdir(exist_ok=True)

    # if ordered keys not provided, recompute
    if ordered_asn is None or ordered_glu is None:
        if channel_type == 'G2':
            pdb_order_asn = ['184.B', '184.C', '184.A', '184.D']
            pdb_order_glu = ['152.B', '152.C', '152.A', '152.D']
        else:
            pdb_order_asn = ['184.B', '184.C', '184.A', '173.D']
            pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']

        def _order(keys, pdb_order):
            ordered = []
            for pdb in pdb_order:
                for k in keys:
                    if combined_data[k].get('pdb_label') == pdb:
                        ordered.append(k)
                        break
            return ordered

        analysed     = set(gmm_results.keys())
        asn_asp_keys = [k for k, v in combined_data.items()
                        if v.get('pdb_label', '').split('.')[0] in ('184', '173')
                        and k in analysed]
        glu_keys     = [k for k, v in combined_data.items()
                        if v.get('pdb_label', '').split('.')[0] in ('152', '141')
                        and k in analysed]
        ordered_asn  = _order(asn_asp_keys, pdb_order_asn)
        ordered_glu  = _order(glu_keys,     pdb_order_glu)

    print("\n=== CREATING OCCUPANCY BAR PLOTS → gmm_occupancy/ ===")
    _plot_2x2_occupancy(ordered_asn, combined_data, gmm_results,
                        'chi1', 'χ1',
                        'ASN_ASP_chi1_occupancy.png', occupancy_dir, channel_type)
    _plot_2x2_occupancy(ordered_asn, combined_data, gmm_results,
                        'chi2', 'χ2',
                        'ASN_ASP_chi2_occupancy.png', occupancy_dir, channel_type)
    _plot_2x2_occupancy(ordered_glu, combined_data, gmm_results,
                        'chi1', 'χ1',
                        'GLU_chi1_occupancy.png', occupancy_dir, channel_type)
    _plot_2x2_occupancy(ordered_glu, combined_data, gmm_results,
                        'chi2', 'χ2',
                        'GLU_chi2_occupancy.png', occupancy_dir, channel_type)


# =============================================================================
# CHI1 x CHI2 COMBINATION ANALYSIS
# =============================================================================

def compute_combinations(gmm_results, combined_data):
    """
    For every residue, combine per-run chi1 and chi2 labels to get
    chi1 x chi2 state combinations.

    Returns
    -------
    combo_results : dict
        {res_name: {
            'pdb_label': str,
            'combo_labels': {run_name: array of (chi1_label, chi2_label) tuples},
            'state_names':  list of str  e.g. ['-170°/-77°', '-170°/52°', ...]
            'overall':      {state_name: {'count': int, 'percent': float}},
            'per_run':      {run_name: {state_name: {'count': int, 'percent': float}}}
        }}
    """
    combo_results = {}

    for res_name, res in gmm_results.items():
        if 'chi1' not in res or 'chi2' not in res:
            continue

        pdb_label = res['pdb_label']
        comps1    = res['chi1']['components']
        comps2    = res['chi2']['components']
        lpr1      = res['chi1']['labels_per_run']
        lpr2      = res['chi2']['labels_per_run']

        # unique chi1 and chi2 mean values (the labels themselves)
        chi1_means = sorted(set(round(float(m), 1) for m in
                            np.concatenate(list(lpr1.values()))))
        chi2_means = sorted(set(round(float(m), 1) for m in
                            np.concatenate(list(lpr2.values()))))

        # state name = "chi1_mean°/chi2_mean°" — directly from label values
        state_names = {(m1, m2): f'{m1:.0f}°/{m2:.0f}°'
                       for m1 in chi1_means for m2 in chi2_means}

        # per-run combination counts
        per_run  = {}
        all_chi1 = []
        all_chi2 = []

        run_names = sorted(set(list(lpr1.keys()) + list(lpr2.keys())))

        for run_name in run_names:
            if run_name not in lpr1 or run_name not in lpr2:
                continue

            l1 = np.round(lpr1[run_name], 1)
            l2 = np.round(lpr2[run_name], 1)

            if len(l1) != len(l2):
                print(f"  [WARN] {res_name} {run_name}: chi1/chi2 label length mismatch "
                      f"({len(l1)} vs {len(l2)}) — skipping run")
                continue

            all_chi1.append(l1)
            all_chi2.append(l2)

            n      = len(l1)
            counts = {}
            for (m1, m2), name in state_names.items():
                mask = (l1 == m1) & (l2 == m2)
                counts[name] = {'count': int(np.sum(mask)),
                                'percent': float(np.sum(mask) / n * 100)}
            per_run[run_name] = counts

        # overall (all runs concatenated)
        if all_chi1:
            l1_all  = np.round(np.concatenate(all_chi1), 1)
            l2_all  = np.round(np.concatenate(all_chi2), 1)
            n_total = len(l1_all)
            overall = {}
            for (m1, m2), name in state_names.items():
                mask = (l1_all == m1) & (l2_all == m2)
                overall[name] = {'count': int(np.sum(mask)),
                                 'percent': float(np.sum(mask) / n_total * 100)}
        else:
            overall = {}

        combo_results[res_name] = {
            'pdb_label':   pdb_label,
            'state_names': list(state_names.values()),
            'overall':     overall,
            'per_run':     per_run
        }

    return combo_results


def plot_combination_bars(combo_results, gmm_results, output_dir,
                          channel_type='G12', group='ASN_ASP'):
    """
    2×2 grid of combination bar plots for one group (ASN_ASP or GLU).
    Each panel: overall occupancy bars + per-run mean ± std error bars.
    """
    output_dir = Path(output_dir)

    if channel_type == 'G2':
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '184.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '152.D']}
    else:
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '173.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '141.D']}

    order = pdb_order.get(group, [])

    # get keys in the right order
    ordered_keys = []
    for pdb in order:
        for res_name, res in combo_results.items():
            if res['pdb_label'] == pdb:
                ordered_keys.append(res_name)
                break

    if len(ordered_keys) < 4:
        print(f"  [SKIP] Not enough residues for combination plot ({group})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle(f'{channel_type}  —  χ1×χ2 combinations',
                 fontsize=24, fontweight='bold', y=1.01)

    for idx, res_name in enumerate(ordered_keys[:4]):
        ax       = axes[idx]
        res      = combo_results[res_name]
        pdb_lbl  = res['pdb_label']
        overall  = res['overall']
        per_run  = res['per_run']
        states   = res['state_names']

        if not overall:
            ax.set_visible(False)
            continue

        # sort states numerically: by chi1 value first, then chi2 value
        def state_sort_key(s):
            parts = s.replace('°', '').split('/')
            return (abs(float(parts[0])), abs(float(parts[1])))

        states  = sorted(overall.keys(), key=state_sort_key)
        x       = np.arange(len(states))
        occ_all = [overall[s]['percent'] for s in states]
        colors  = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)] for i in range(len(states))]

        bars = ax.bar(x, occ_all, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.5, zorder=3)

        # value labels on top of bars
        for bar, occ, state in zip(bars, occ_all, states):
            count = overall[state]['count']
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 1,
                    f'{occ:.1f}%\n({count:,})',
                    ha='center', va='bottom', fontsize=17, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=18, fontweight='bold', rotation=20, ha='right')
        ax.set_ylabel('Occupancy (%)', fontsize=26, fontweight='bold')
        ax.set_ylim(0, 125)
        ax.set_title(f'{pdb_lbl}', fontsize=30, fontweight='bold', pad=12)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3, linewidth=1)

    plt.tight_layout()
    fname     = f'gmm_{group}_chi1xchi2_combinations.png'
    save_path = output_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def save_combination_report(combo_results, output_dir):
    """Save combination occupancies to txt and json."""
    report_path = output_dir / 'gmm_combinations_report.txt'
    with open(report_path, 'w') as f:
        f.write("CHI1 x CHI2 COMBINATION OCCUPANCIES\n")
        f.write("=" * 70 + "\n\n")

        for res_name, res in combo_results.items():
            f.write(f"{res_name}  (PDB: {res['pdb_label']})\n")
            f.write("─" * 60 + "\n")
            f.write("Overall (all runs):\n")
            for state, vals in res['overall'].items():
                f.write(f"  {state:>25s}: {vals['count']:7d} frames  ({vals['percent']:5.1f}%)\n")
            f.write("\nPer run:\n")
            for run_name in sorted(res['per_run'].keys()):
                f.write(f"  {run_name}:\n")
                for state, vals in res['per_run'][run_name].items():
                    f.write(f"    {state:>25s}: {vals['count']:7d}  ({vals['percent']:5.1f}%)\n")
            f.write("\n" + "=" * 70 + "\n\n")

    # json (no numpy)
    json_path = output_dir / 'gmm_combinations_summary.json'
    with open(json_path, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'combo_labels'}
                   for k, v in combo_results.items()}, f, indent=2)

    print(f"  ✅ Saved: {report_path}")
    print(f"  ✅ Saved: {json_path}")


# =============================================================================
# PERMEATION EVENT ANALYSIS  (R2_end frames)
# =============================================================================

def load_permeation_frames(base_dir, run_names):
    """
    For each run, load permeation_table.json and extract R2_end frame indices.

    Parameters
    ----------
    base_dir  : Path   directory containing RUN1, RUN2, ...
    run_names : list   e.g. ['RUN1', 'RUN2', ...]  from gmm_results labels_per_run

    Returns
    -------
    perm_frames : dict  {run_name: [frame_idx, ...]}
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

        # table is a list of dicts (or dict-of-dicts with string keys)
        if isinstance(table, dict):
            rows = list(table.values())
        else:
            rows = table

        r2_ends = []
        for row in rows:
            if 'R2_end' in row and row['R2_end'] is not None:
                r2_ends.append(int(row['R2_end']))

        perm_frames[run_name] = r2_ends
        print(f"  {run_name}: {len(r2_ends)} ion exit events "
              f"at frames {r2_ends[:5]}{'...' if len(r2_ends) > 5 else ''}")

    return perm_frames


def compute_permeation_combinations(gmm_results, perm_frames, offset=0):
    """
    For each residue and each permeation event, look up the chi1 and chi2
    GMM label at frame (R2_end + offset) and record the combination state.

    Parameters
    ----------
    gmm_results  : dict  from run_gmm_analysis()
    perm_frames  : dict  {run_name: [frame_idx, ...]}  (R2_end frames)
    offset       : int   0 = R2_end, -1 = frame before, +1 = frame after

    Returns
    -------
    perm_combo : dict
        {res_name: {
            'pdb_label': str,
            'state_names': list of str,
            'overall': {state_name: {'count': int, 'percent': float}},
            'per_run': {run_name: {state_name: {'count': int, 'percent': float}}},
            'n_events_total': int
        }}
    """
    perm_combo = {}

    for res_name, res in gmm_results.items():
        if 'chi1' not in res or 'chi2' not in res:
            continue

        pdb_label = res['pdb_label']
        comps1    = res['chi1']['components']
        comps2    = res['chi2']['components']
        lpr1      = res['chi1']['labels_per_run']
        lpr2      = res['chi2']['labels_per_run']

        # state names come directly from label mean values
        # labels ARE the mean angles rounded to 1 decimal
        all_l1            = []
        all_l2            = []
        per_run           = {}
        verification_rows = []

        for run_name, frames in perm_frames.items():
            if run_name not in lpr1 or run_name not in lpr2:
                print(f"  [WARN] {res_name}: no labels for {run_name} — skipping")
                continue

            l1           = lpr1[run_name]
            l2           = lpr2[run_name]
            n_frames_run = len(l1)

            run_l1 = []
            run_l2 = []
            for frame in frames:
                target = frame + offset
                if target < 0 or target >= n_frames_run:
                    print(f"  [WARN] {res_name} {run_name}: "
                          f"frame {frame}+offset({offset})={target} out of bounds "
                          f"[0,{n_frames_run}) — skipping")
                    continue
                m1 = round(float(l1[target]), 1)
                m2 = round(float(l2[target]), 1)
                run_l1.append(m1)
                run_l2.append(m2)

                verification_rows.append({
                    'run':       run_name,
                    'R2_end':    frame,
                    'frame':     target,
                    'offset':    offset,
                    'chi1_mean': f'{m1:.1f}°',
                    'chi2_mean': f'{m2:.1f}°',
                    'combo':     f'{m1:.0f}°/{m2:.0f}°'
                })

            all_l1.extend(run_l1)
            all_l2.extend(run_l2)

            # per-run counts — group by combo string
            n_run      = len(run_l1)
            run_combos = [f'{m1:.0f}°/{m2:.0f}°' for m1, m2 in zip(run_l1, run_l2)]
            unique_combos = sorted(set(run_combos))
            run_counts = {name: {'count': run_combos.count(name),
                                  'percent': run_combos.count(name) / n_run * 100
                                  if n_run > 0 else 0.0}
                          for name in unique_combos}
            per_run[run_name] = run_counts

        # overall counts
        n_total    = len(all_l1)
        all_combos = [f'{m1:.0f}°/{m2:.0f}°' for m1, m2 in zip(all_l1, all_l2)]
        unique_all = sorted(set(all_combos))
        overall    = {name: {'count': all_combos.count(name),
                              'percent': all_combos.count(name) / n_total * 100
                              if n_total > 0 else 0.0}
                      for name in unique_all}

        # chi1 and chi2 separate counts
        chi1_vals    = [f'{m:.0f}°' for m in sorted(set(all_l1))]
        chi1_overall = {v: {'count': sum(1 for m in all_l1 if f'{m:.0f}°' == v),
                             'percent': sum(1 for m in all_l1 if f'{m:.0f}°' == v)
                             / n_total * 100 if n_total > 0 else 0.0}
                        for v in chi1_vals}

        chi2_vals    = [f'{m:.0f}°' for m in sorted(set(all_l2))]
        chi2_overall = {v: {'count': sum(1 for m in all_l2 if f'{m:.0f}°' == v),
                             'percent': sum(1 for m in all_l2 if f'{m:.0f}°' == v)
                             / n_total * 100 if n_total > 0 else 0.0}
                        for v in chi2_vals}

        state_names = unique_all  # keep for downstream compatibility

        perm_combo[res_name] = {
            'pdb_label':        pdb_label,
            'state_names':      state_names,          # list of combo strings
            'overall':          overall,
            'chi1_overall':     chi1_overall,
            'chi2_overall':     chi2_overall,
            'chi1_state_names': chi1_vals,
            'chi2_state_names': chi2_vals,
            'per_run':          per_run,
            'n_events_total':   n_total,
            'verification':     verification_rows
        }

        print(f"  {res_name} (PDB: {pdb_label}): {n_total} total ion exit events")

    return perm_combo


def plot_permeation_combination_bars(perm_combo, output_dir,
                                     channel_type='G12', group='ASN_ASP',
                                     frame_label='at_exit'):
    """
    2×2 bar plots of chi1×chi2 combination occupancy at permeation frames.
    frame_label: used in suptitle and filenames (e.g. 'R2_end', 'R2_end-1', 'R2_end+1')
    """
    output_dir = Path(output_dir)

    if channel_type == 'G2':
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '184.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '152.D']}
    else:
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '173.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '141.D']}

    order = pdb_order.get(group, [])
    ordered_keys = []
    for pdb in order:
        for res_name, res in perm_combo.items():
            if res['pdb_label'] == pdb:
                ordered_keys.append(res_name)
                break

    if len(ordered_keys) < 4:
        print(f"  [SKIP] Not enough residues for permeation combination plot ({group})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle(f'{channel_type}  —  Ion exits GLU/ASN cavity — {frame_label}  |  χ1×χ2',
                 fontsize=22, fontweight='bold', y=1.01)

    for idx, res_name in enumerate(ordered_keys[:4]):
        ax      = axes[idx]
        res     = perm_combo[res_name]
        pdb_lbl = res['pdb_label']
        overall = res['overall']
        states  = res['state_names']
        n_ev    = res['n_events_total']

        if not overall or n_ev == 0:
            ax.set_visible(False)
            continue

        def state_sort_key(s):
            parts = s.replace('°', '').split('/')
            return (abs(float(parts[0])), abs(float(parts[1])))

        states  = sorted(overall.keys(), key=state_sort_key)
        x       = np.arange(len(states))
        occ_all = [overall[s]['percent'] for s in states]
        colors  = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
                   for i in range(len(states))]

        bars = ax.bar(x, occ_all, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.5, zorder=3)

        for bar, occ, state in zip(bars, occ_all, states):
            count = overall[state]['count']
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 1,
                    f'{occ:.1f}%\n({count:,})',
                    ha='center', va='bottom', fontsize=17, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=18, fontweight='bold',
                           rotation=20, ha='right')
        ax.set_ylabel('Occupancy (%)', fontsize=24, fontweight='bold')
        ax.set_ylim(0, 125)
        ax.set_title(f'{pdb_lbl}  (n={n_ev} events)', fontsize=28,
                     fontweight='bold', pad=12)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3, linewidth=1)

    plt.tight_layout()
    safe_label = frame_label.replace('+', 'plus').replace('-', 'minus').replace(' ', '')
    fname      = f'permeation_{safe_label}_{group}_chi1xchi2.png'
    save_path  = output_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def plot_permeation_single_chi_bars(perm_combo, output_dir,
                                    chi_key, channel_type='G12', group='ASN_ASP',
                                    frame_label='at_exit'):
    """
    2×2 bar plots of chi1 OR chi2 occupancy separately at permeation frames.
    chi_key: 'chi1' or 'chi2'
    frame_label: used in suptitle and filename
    """
    output_dir = Path(output_dir)

    if channel_type == 'G2':
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '184.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '152.D']}
    else:
        pdb_order = {'ASN_ASP': ['184.B', '184.C', '184.A', '173.D'],
                     'GLU':     ['152.B', '152.C', '152.A', '141.D']}

    order = pdb_order.get(group, [])
    ordered_keys = []
    for pdb in order:
        for res_name, res in perm_combo.items():
            if res['pdb_label'] == pdb:
                ordered_keys.append(res_name)
                break

    if len(ordered_keys) < 4:
        print(f"  [SKIP] Not enough residues for permeation {chi_key} plot ({group})")
        return

    chi_label      = 'χ1' if chi_key == 'chi1' else 'χ2'
    overall_key    = f'{chi_key}_overall'
    state_name_key = f'{chi_key}_state_names'

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle(f'{channel_type}  —  Ion exits GLU/ASN cavity — {frame_label}  |  {chi_label}',
                 fontsize=22, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for idx, res_name in enumerate(ordered_keys[:4]):
        ax      = axes[idx]
        res     = perm_combo[res_name]
        pdb_lbl = res['pdb_label']
        overall = res[overall_key]
        states  = res[state_name_key]
        n_ev    = res['n_events_total']

        if not overall or n_ev == 0:
            ax.set_visible(False)
            continue

        # sort numerically by angle value
        states  = sorted(overall.keys(), key=lambda s: abs(float(s.replace('°', ''))))
        x       = np.arange(len(states))
        occ_all = [overall[s]['percent'] for s in states]
        colors  = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)] for i in range(len(states))]

        bars = ax.bar(x, occ_all, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.5)

        for bar, occ, state in zip(bars, occ_all, states):
            count = overall[state]['count']
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 1,
                    f'{occ:.1f}%\n({count:,})',
                    ha='center', va='bottom', fontsize=17, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=20, fontweight='bold')
        ax.set_ylabel(f'{chi_label} Occupancy (%)', fontsize=22, fontweight='bold')
        ax.set_ylim(0, 125)
        ax.set_title(f'{pdb_lbl}  (n={n_ev} events)', fontsize=28,
                     fontweight='bold', pad=12)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3, linewidth=1)

    plt.tight_layout()
    safe_label = frame_label.replace('+', 'plus').replace('-', 'minus').replace(' ', '')
    fname      = f'permeation_{safe_label}_{group}_{chi_key}.png'
    save_path  = output_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def save_permeation_report(perm_combo, output_dir, frame_label='at_exit'):
    """Save permeation combination counts to txt, verification table, and json."""
    safe_label  = frame_label.replace('+', 'plus').replace('-', 'minus').replace(' ', '')
    report_path = output_dir / f'permeation_{safe_label}_combinations.txt'
    with open(report_path, 'w') as f:
        f.write(f"CHI1 x CHI2 COMBINATIONS — {frame_label.replace('_', ' ').upper()}\n")
        f.write(f"(Ion exits GLU/ASN cavity)\n")
        f.write("=" * 70 + "\n\n")
        for res_name, res in perm_combo.items():
            f.write(f"{res_name}  (PDB: {res['pdb_label']})  "
                    f"— {res['n_events_total']} total events\n")
            f.write("─" * 60 + "\n")

            f.write("Chi1 x Chi2 combinations:\n")
            for state, vals in res['overall'].items():
                f.write(f"  {state:>25s}: {vals['count']:5d}  ({vals['percent']:5.1f}%)\n")

            f.write("\nChi1 separately:\n")
            for state, vals in res['chi1_overall'].items():
                f.write(f"  {state:>10s}: {vals['count']:5d}  ({vals['percent']:5.1f}%)\n")

            f.write("\nChi2 separately:\n")
            for state, vals in res['chi2_overall'].items():
                f.write(f"  {state:>10s}: {vals['count']:5d}  ({vals['percent']:5.1f}%)\n")

            f.write("\nPer run (combinations):\n")
            for run_name in sorted(res['per_run'].keys()):
                f.write(f"  {run_name}:\n")
                for state, vals in res['per_run'][run_name].items():
                    f.write(f"    {state:>25s}: {vals['count']:5d}  "
                            f"({vals['percent']:5.1f}%)\n")
            f.write("\n" + "=" * 70 + "\n\n")

    # verification table
    verif_path = output_dir / f'permeation_{safe_label}_verification.txt'
    with open(verif_path, 'w') as f:
        f.write(f"VERIFICATION TABLE — {frame_label.replace('_', ' ').upper()}\n")
        f.write("(first 20 events per residue for manual checking)\n")
        f.write("=" * 80 + "\n\n")
        for res_name, res in perm_combo.items():
            f.write(f"{res_name}  (PDB: {res['pdb_label']})\n")
            f.write(f"{'Run':<10} {'Ion exit':>10} {'Lookup':>10} "
                    f"{'χ1 mean':>10} {'χ2 mean':>10} {'Combo':>20}\n")
            f.write("─" * 80 + "\n")
            for row in res['verification'][:20]:
                f.write(f"{row['run']:<10} {row['R2_end']:>10} {row['frame']:>10} "
                        f"{row['chi1_mean']:>10} {row['chi2_mean']:>10} "
                        f"{row['combo']:>20}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    # json summary
    json_path    = output_dir / f'permeation_{safe_label}_combinations.json'
    serialisable = {res_name: {k: v for k, v in res.items() if k != 'verification'}
                    for res_name, res in perm_combo.items()}
    with open(json_path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"  ✅ Saved: {report_path}")
    print(f"  ✅ Saved: {verif_path}")
    print(f"  ✅ Saved: {json_path}")


def write_debug_report(combined_data, gmm_results, combo_results,
                       output_dir, channel_type):
    """
    Write a comprehensive debugging file covering every critical step
    of the analysis pipeline. Intended to catch any remaining logic bugs.

    Checks performed:
    1.  Data integrity       — frame counts per run, per residue
    2.  Run ordering         — sorted key order matches concatenation order
    3.  GMM fitting space    — confirms correct space per residue type
    4.  Component means      — raw fitting-space mean vs back-converted mean
    5.  Label integrity      — labels are float mean angles, not integers
    6.  Label↔component match — every unique label value matches a component mean
    7.  Label count vs weight — hard label counts vs GMM soft weights (should be close)
    8.  Per-run label split  — sum of per-run label arrays = total labels
    9.  Combination sanity   — sum of chi1×chi2 combos = total frames
    10. Combination marginals — summing combos over chi2 = chi1 occupancy, and vice versa
    11. Frame-level spot check — sample 5 random frames, show raw angle + assigned label
    """
    debug_path = output_dir / 'DEBUG_REPORT.txt'

    lines = []
    W = 80

    def h1(title):
        lines.append("\n" + "=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W + "\n")

    def h2(title):
        lines.append(f"\n{'─' * W}")
        lines.append(f"  {title}")
        lines.append(f"{'─' * W}")

    def ok(msg):  lines.append(f"  ✅  {msg}")
    def warn(msg): lines.append(f"  ⚠️   {msg}")
    def err(msg):  lines.append(f"  ❌  {msg}")
    def info(msg): lines.append(f"      {msg}")

    lines.append("GMM ANALYSIS DEBUG REPORT")
    lines.append(f"Channel: {channel_type}")
    lines.append(f"Output:  {output_dir}")

    # =========================================================================
    # 1. DATA INTEGRITY
    # =========================================================================
    h1("1. DATA INTEGRITY — frame counts per residue per run")
    for res_name, angles in combined_data.items():
        pdb_label = angles.get('pdb_label', res_name)
        chi1_data = angles.get('chi1', {})
        chi2_data = angles.get('chi2', {})

        if not isinstance(chi1_data, dict):
            warn(f"{res_name}: chi1 is not per-run dict (old format)")
            continue

        runs1 = sorted(chi1_data.keys())
        runs2 = sorted(chi2_data.keys()) if isinstance(chi2_data, dict) else []

        h2(f"{res_name}  (PDB: {pdb_label})")
        info(f"chi1 runs: {runs1}")
        info(f"chi2 runs: {runs2}")

        if runs1 != runs2:
            err(f"chi1 and chi2 have different run sets!")
        else:
            ok("chi1 and chi2 have same run sets")

        total1 = 0
        for run in runs1:
            n1 = len(chi1_data[run])
            n2 = len(chi2_data[run]) if run in chi2_data else -1
            total1 += n1
            if n1 != n2:
                err(f"{run}: chi1={n1} frames, chi2={n2} frames — MISMATCH")
            else:
                info(f"{run}: {n1} frames ✓")
        info(f"Total frames: {total1}")

    # =========================================================================
    # 2. RUN ORDERING
    # =========================================================================
    h1("2. RUN ORDERING — sorted key order vs insertion order")
    for res_name, angles in combined_data.items():
        chi1_data = angles.get('chi1', {})
        if not isinstance(chi1_data, dict):
            continue
        insertion_order = list(chi1_data.keys())
        sorted_order    = sorted(chi1_data.keys())
        if insertion_order == sorted_order:
            ok(f"{res_name}: insertion order = sorted order ✓")
        else:
            err(f"{res_name}: insertion order {insertion_order} ≠ sorted {sorted_order}")
            err(f"  This means get_all_frames() concatenation order was WRONG before fix!")
        break  # one residue is enough — all share same runs

    # =========================================================================
    # 3. GMM FITTING SPACE
    # =========================================================================
    h1("3. GMM FITTING SPACE — correct space per residue type")
    expected = {
        '152': ('0_360', '0_360'),
        '141': ('0_360', '0_360'),
        '184': ('0_360', '-180_180'),
        '173': ('0_360', '100_460'),
    }
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        prefix    = pdb_label.split('.')[0]
        exp       = expected.get(prefix, ('?', '?'))
        got_chi1  = res.get('chi1', {}).get('space', '?')
        got_chi2  = res.get('chi2', {}).get('space', '?')
        if (got_chi1, got_chi2) == exp:
            ok(f"{res_name} ({pdb_label}): chi1={got_chi1}, chi2={got_chi2} ✓")
        else:
            err(f"{res_name} ({pdb_label}): expected chi1={exp[0]},chi2={exp[1]} "
                f"but got chi1={got_chi1},chi2={got_chi2}")

    # =========================================================================
    # 4. COMPONENT MEANS — fitting space vs back-converted
    # =========================================================================
    h1("4. COMPONENT MEANS — fitting space vs [-180,180] display")
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        h2(f"{res_name}  (PDB: {pdb_label})")
        for chi_key in ['chi1', 'chi2']:
            if chi_key not in res:
                continue
            space = res[chi_key]['space']
            comps = res[chi_key]['components']
            info(f"{chi_key} (space={space}):")
            for i, c in enumerate(comps):
                info(f"  G{i+1}: mean={c['mean']:8.2f}°  std={c['std']:6.2f}°  "
                     f"weight={c['weight']:.4f}  ({c['weight']*100:.1f}%)")

    # =========================================================================
    # 5. LABEL INTEGRITY — labels must be floats (mean angles), not integers
    # =========================================================================
    h1("5. LABEL INTEGRITY — labels should be float mean angles, not int indices")
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        for chi_key in ['chi1', 'chi2']:
            if chi_key not in res:
                continue
            labels = res[chi_key]['labels']
            unique = np.unique(labels)
            # check they are floats (not 0,1,2...)
            all_small_ints = all(v in [0, 1, 2, 3, 4] for v in unique)
            if all_small_ints:
                err(f"{res_name} {chi_key}: labels look like integer indices! "
                    f"unique={unique} — assign_labels bug?")
            else:
                ok(f"{res_name} {chi_key}: labels are angle values: {unique.round(1)}")

    # =========================================================================
    # 6. LABEL↔COMPONENT MATCH
    # =========================================================================
    h1("6. LABEL↔COMPONENT MATCH — every unique label must match a component mean")
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        for chi_key in ['chi1', 'chi2']:
            if chi_key not in res:
                continue
            labels    = res[chi_key]['labels']
            comps     = res[chi_key]['components']
            comp_means = set(round(c['mean'], 1) for c in comps)
            unique_labels = set(round(float(v), 1) for v in np.unique(labels))

            unmatched = unique_labels - comp_means
            if unmatched:
                err(f"{res_name} {chi_key}: label values {unmatched} "
                    f"not found in component means {comp_means}!")
            else:
                ok(f"{res_name} {chi_key}: all label values match component means ✓")

    # =========================================================================
    # 7. LABEL COUNT VS WEIGHT
    # =========================================================================
    h1("7. LABEL COUNT VS WEIGHT — hard counts vs GMM soft weights")
    info("(Differences up to ~0.5% are normal due to hard vs soft assignment)")
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        h2(f"{res_name}  (PDB: {pdb_label})")
        for chi_key in ['chi1', 'chi2']:
            if chi_key not in res:
                continue
            labels   = res[chi_key]['labels']
            comps    = res[chi_key]['components']
            n_total  = len(labels)
            info(f"{chi_key}: total frames = {n_total}")
            for c in comps:
                mean       = round(c['mean'], 1)
                hard_count = int(np.sum(np.isclose(labels, mean, atol=0.05)))
                hard_pct   = hard_count / n_total * 100
                soft_pct   = c['weight'] * 100
                diff       = abs(hard_pct - soft_pct)
                msg = (f"  mean={mean:8.1f}°: hard={hard_count:7d} ({hard_pct:5.1f}%)  "
                       f"soft={soft_pct:5.1f}%  diff={diff:.2f}%")
                if diff > 2.0:
                    warn(msg + "  ← large discrepancy!")
                else:
                    info(msg)

    # =========================================================================
    # 8. PER-RUN LABEL SPLIT
    # =========================================================================
    h1("8. PER-RUN LABEL SPLIT — sum of per-run arrays = total labels")
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        for chi_key in ['chi1', 'chi2']:
            if chi_key not in res:
                continue
            labels_all = res[chi_key]['labels']
            lpr        = res[chi_key]['labels_per_run']
            n_total    = len(labels_all)
            n_sum      = sum(len(v) for v in lpr.values())
            if n_total != n_sum:
                err(f"{res_name} {chi_key}: total labels={n_total} but "
                    f"sum of per-run={n_sum} — split is WRONG!")
            else:
                ok(f"{res_name} {chi_key}: total={n_total} = sum of per-run ✓")

            # also check first/last frames of each run match the split
            chi_data = combined_data[res_name].get(chi_key, {})
            if isinstance(chi_data, dict):
                idx = 0
                for run_name in sorted(chi_data.keys()):
                    n_run = len(chi_data[run_name])
                    chunk = labels_all[idx: idx + n_run]
                    if len(chunk) != len(lpr.get(run_name, [])):
                        err(f"  {run_name}: chunk size {len(chunk)} ≠ "
                            f"stored {len(lpr.get(run_name,[]))}")
                    idx += n_run

    # =========================================================================
    # 9. COMBINATION SANITY
    # =========================================================================
    h1("9. COMBINATION SANITY — sum of all combo counts = total frames")
    for res_name, res in combo_results.items():
        pdb_label = res['pdb_label']
        overall   = res['overall']
        total_from_combos = sum(v['count'] for v in overall.values())

        # get total frames from gmm_results
        n_total = len(gmm_results[res_name]['chi1']['labels'])

        if total_from_combos != n_total:
            err(f"{res_name} ({pdb_label}): combo sum={total_from_combos} "
                f"≠ n_frames={n_total}")
        else:
            ok(f"{res_name} ({pdb_label}): combo sum = n_frames = {n_total} ✓")

    # =========================================================================
    # 10. COMBINATION MARGINALS
    # =========================================================================
    h1("10. COMBINATION MARGINALS — summing combos over chi2 should equal chi1 occupancy")
    for res_name, res in combo_results.items():
        pdb_label = res['pdb_label']
        overall   = res['overall']
        gmm_res   = gmm_results[res_name]
        labels1   = gmm_res['chi1']['labels']
        labels2   = gmm_res['chi2']['labels']

        h2(f"{res_name}  (PDB: {pdb_label})")

        # chi1 marginal: sum over all chi2 for each chi1 value
        chi1_from_combos = {}
        for state_name, vals in overall.items():
            chi1_val = state_name.split('/')[0]   # e.g. "-169°"
            chi1_from_combos[chi1_val] = (chi1_from_combos.get(chi1_val, 0)
                                           + vals['count'])

        # chi1 direct from labels
        chi1_direct = {}
        for lbl in np.unique(np.round(labels1, 1)):
            key = f'{lbl:.0f}°'
            chi1_direct[key] = int(np.sum(np.isclose(labels1, lbl, atol=0.05)))

        info("chi1 marginal check (combo sum vs direct label count):")
        all_ok = True
        for key in sorted(chi1_direct.keys()):
            combo_count  = chi1_from_combos.get(key, 0)
            direct_count = chi1_direct[key]
            if combo_count != direct_count:
                err(f"  chi1={key}: combo_sum={combo_count} ≠ direct={direct_count}")
                all_ok = False
            else:
                info(f"  chi1={key}: {direct_count} ✓")
        if all_ok:
            ok("chi1 marginals match ✓")

        # chi2 marginal: sum over all chi1 for each chi2 value
        chi2_from_combos = {}
        for state_name, vals in overall.items():
            chi2_val = state_name.split('/')[1]   # e.g. "-71°"
            chi2_from_combos[chi2_val] = (chi2_from_combos.get(chi2_val, 0)
                                           + vals['count'])

        chi2_direct = {}
        for lbl in np.unique(np.round(labels2, 1)):
            key = f'{lbl:.0f}°'
            chi2_direct[key] = int(np.sum(np.isclose(labels2, lbl, atol=0.05)))

        info("chi2 marginal check:")
        all_ok2 = True
        for key in sorted(chi2_direct.keys()):
            combo_count  = chi2_from_combos.get(key, 0)
            direct_count = chi2_direct[key]
            if combo_count != direct_count:
                err(f"  chi2={key}: combo_sum={combo_count} ≠ direct={direct_count}")
                all_ok2 = False
            else:
                info(f"  chi2={key}: {direct_count} ✓")
        if all_ok2:
            ok("chi2 marginals match ✓")

    # =========================================================================
    # 11. FRAME-LEVEL SPOT CHECK
    # =========================================================================
    h1("11. FRAME-LEVEL SPOT CHECK — 5 random frames per residue")
    info("For each frame: raw chi1 angle, assigned chi1 label, raw chi2 angle, "
         "assigned chi2 label, combo state")
    info("NOTE: GMM assigns by PROBABILITY (distance + std), not by nearest mean.")
    info("A frame may be assigned to a wider Gaussian even if its mean is farther away.")
    info("Mismatches here are expected in overlap regions and are NOT bugs.")
    info("Only flag if the assigned label doesn't match ANY component mean at all.\n")

    rng = np.random.default_rng(42)
    for res_name, res in gmm_results.items():
        pdb_label = res['pdb_label']
        h2(f"{res_name}  (PDB: {pdb_label})")

        labels1 = res['chi1']['labels']
        labels2 = res['chi2']['labels']
        comps1  = res['chi1']['components']
        comps2  = res['chi2']['components']
        space1  = res['chi1']['space']
        space2  = res['chi2']['space']

        raw1 = get_all_frames(combined_data[res_name]['chi1'])
        raw2 = get_all_frames(combined_data[res_name]['chi2'])

        n = min(len(raw1), len(raw2), len(labels1), len(labels2))
        sample_idx = rng.integers(0, n, size=min(5, n))

        means1 = sorted([c['mean'] for c in comps1])
        means2 = sorted([c['mean'] for c in comps2])

        info(f"chi1 component means (display): {[f'{m:.1f}°' for m in means1]}")
        info(f"chi2 component means (display): {[f'{m:.1f}°' for m in means2]}")
        info(f"chi1 fitting space: {space1},  chi2 fitting space: {space2}")
        info("")
        info(f"{'Frame':>8} {'Raw χ1':>10} {'Label χ1':>10} "
             f"{'Raw χ2':>10} {'Label χ2':>10}  Combo")
        info("─" * 65)

        for idx in sample_idx:
            r1 = float(raw1[idx])
            r2 = float(raw2[idx])
            l1 = float(labels1[idx])
            l2 = float(labels2[idx])

            # circular distance helper
            def circ_dist(a, b):
                d = abs(a - b) % 360
                return d if d <= 180 else 360 - d

            # real check: does the assigned label match a known component mean?
            l1_valid = any(circ_dist(l1, m) < 1.0 for m in means1)
            l2_valid = any(circ_dist(l2, m) < 1.0 for m in means2)

            # also show which component was nearest by circular distance
            nearest1 = min(means1, key=lambda m: circ_dist(m, r1))
            nearest2 = min(means2, key=lambda m: circ_dist(m, r2))
            note1 = f" (nearest by dist={nearest1:.0f}°)" if abs(l1 - nearest1) > 1 else ""
            note2 = f" (nearest by dist={nearest2:.0f}°)" if abs(l2 - nearest2) > 1 else ""

            flag = ""
            if not l1_valid:
                flag += f" ← χ1 INVALID LABEL {l1:.1f}° not in {means1}!"
            if not l2_valid:
                flag += f" ← χ2 INVALID LABEL {l2:.1f}° not in {means2}!"

            info(f"{idx:>8} {r1:>10.2f}° {l1:>10.1f}°{note1:<20} "
                 f"{r2:>10.2f}° {l2:>10.1f}°{note2:<20}  "
                 f"{l1:.0f}°/{l2:.0f}°{flag}")

    # =========================================================================
    # WRITE FILE
    # =========================================================================
    with open(debug_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Debug report saved to: {debug_path}")
    return debug_path




def main():
    parser = argparse.ArgumentParser(
        description='Peak-guided GMM dihedral analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rules:
  GLU (141.D)  chi1: K=2 [0,360]    chi2: K=2 [0,360]
  GLU (others) chi1: K=2 [0,360]    chi2: K=3 [0,360]
  ASN          chi1: K=2 [0,360]    chi2: K=3 [-180,180]
  ASP          chi1: K=2 [0,360]    chi2: K=2 [100,460]
  (141.D / ASP skipped silently if not present in the data)

Examples:
  python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json
  python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json -c G12_GAT
  python gmm_dihedral_analysis.py combined_dihedral_data.pkl combined_peak_analysis.json -o /out/
        """
    )
    parser.add_argument('pkl_file',  help='Path to combined_dihedral_data.pkl')
    parser.add_argument('peak_json', help='Path to combined_peak_analysis.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: gmm_analysis/ next to pkl)')
    parser.add_argument('-c', '--channel', default='G12',
                        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'],
                        help='Channel type for plot ordering (default: G12)')
    parser.add_argument('--n-init', type=int, default=20,
                        help='Extra random GMM restarts per fit (default: 20)')
    parser.add_argument('--base-dir', default=None,
                        help='Base directory containing RUN1, RUN2, ... with '
                             'permeation_table.json files. If provided, '
                             'ion exit event analysis is run.')

    args = parser.parse_args()

    pkl_path  = Path(args.pkl_file)
    json_path = Path(args.peak_json)

    for p in [pkl_path, json_path]:
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 1

    output_dir = Path(args.output) if args.output else pkl_path.parent / 'gmm_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    combined_data = saved['combined_data'] if 'combined_data' in saved else saved

    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        peak_analysis = json.load(f)

    gmm_results = run_gmm_analysis(
        combined_data, peak_analysis, output_dir,
        n_init=args.n_init,
        channel_type=args.channel
    )

    # GMM fit plots — same for both methods (fitting is identical)
    ordered_asn, ordered_glu = plot_gmm_fits(
        combined_data, gmm_results, output_dir, channel_type=args.channel)

    # ── run combinations + permeation for BOTH assignment methods ──────────
    methods = [
        ('probabilistic', 'labels',         'labels_per_run'),
        ('nearest_mean',  'labels_nearest',  'labels_per_run_near'),
    ]

    for method_name, lbl_key, lpr_key in methods:
        print(f"\n{'='*70}")
        print(f"  ASSIGNMENT METHOD: {method_name}")
        print(f"{'='*70}")

        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        # temporarily expose the right labels under the standard keys
        for res in gmm_results.values():
            for chi in ['chi1', 'chi2']:
                if chi in res:
                    res[chi]['_labels_bak']     = res[chi]['labels']
                    res[chi]['_lpr_bak']        = res[chi]['labels_per_run']
                    res[chi]['labels']          = res[chi][lbl_key]
                    res[chi]['labels_per_run']  = res[chi][lpr_key]

        # --- occupancy bar plots (method-dependent) ---
        plot_gmm_occupancy(combined_data, gmm_results, method_dir,
                           channel_type=args.channel,
                           ordered_asn=ordered_asn, ordered_glu=ordered_glu)
        combo_dir = method_dir / 'combinations'
        combo_dir.mkdir(exist_ok=True)
        print(f"\n=== CHI1 x CHI2 COMBINATIONS → {method_name}/combinations/ ===")
        combo_results = compute_combinations(gmm_results, combined_data)
        save_combination_report(combo_results, combo_dir)
        plot_combination_bars(combo_results, gmm_results, combo_dir,
                              channel_type=args.channel, group='ASN_ASP')
        plot_combination_bars(combo_results, gmm_results, combo_dir,
                              channel_type=args.channel, group='GLU')

        # --- debug report ---
        print(f"\n=== WRITING DEBUG REPORT → {method_name}/ ===")
        write_debug_report(combined_data, gmm_results, combo_results,
                           method_dir, channel_type=args.channel)

        # --- ion exit event analysis ---
        if args.base_dir:
            print(f"\n=== ION EXIT EVENT ANALYSIS → {method_name}/ ===")
            base_dir   = Path(args.base_dir)
            sample_res = next(iter(gmm_results.values()))
            run_names  = sorted(sample_res['chi1']['labels_per_run'].keys())
            perm_frames = load_permeation_frames(base_dir, run_names)

            if perm_frames:
                for offset, frame_label in [(0, 'at_exit'), (-1, 'before_exit'),
                                             (1, 'after_exit')]:
                    perm_dir = method_dir / f'ion_exit_{frame_label}'
                    perm_dir.mkdir(exist_ok=True)
                    print(f"  --- {frame_label} ---")
                    perm_combo = compute_permeation_combinations(
                        gmm_results, perm_frames, offset=offset
                    )
                    save_permeation_report(perm_combo, perm_dir,
                                           frame_label=frame_label)
                    for grp in ['ASN_ASP', 'GLU']:
                        plot_permeation_combination_bars(
                            perm_combo, perm_dir,
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
                        plot_permeation_single_chi_bars(
                            perm_combo, perm_dir, chi_key='chi1',
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
                        plot_permeation_single_chi_bars(
                            perm_combo, perm_dir, chi_key='chi2',
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
            else:
                print("  No permeation_table.json files found — skipping")

        # restore original labels
        for res in gmm_results.values():
            for chi in ['chi1', 'chi2']:
                if chi in res:
                    res[chi]['labels']         = res[chi]['_labels_bak']
                    res[chi]['labels_per_run'] = res[chi]['_lpr_bak']
                    del res[chi]['_labels_bak']
                    del res[chi]['_lpr_bak']

    print("\n" + "=" * 70)
    print("GMM ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results in: {output_dir}/")
    print("  gmm_report.txt / gmm_summary.json / gmm_results.pkl")
    print("  gmm_fits/           – histogram + Gaussian overlay plots")
    print("  gmm_occupancy/      – occupancy bar plots (mean±std vs %)")
    print("  combinations/       – chi1×chi2 combination plots + reports")
    if args.base_dir:
        print("  ion_exit_at_exit/     – conformation at ion exit frame")
        print("  ion_exit_before_exit/ – conformation 1 frame before exit")
        print("  ion_exit_after_exit/  – conformation 1 frame after exit")
    print("=" * 70 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())

    
"""
python3 gmm_dihedral_analysis.py \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis/combined_dihedral_data.pkl \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/dihedral_analysis/combined_peak_analysis.json \
-c G12 \
--base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12

python3 gmm_dihedral_analysis.py \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis/combined_dihedral_data.pkl \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/dihedral_analysis/combined_peak_analysis.json \
-c G2 \
--base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2

python3 gmm_dihedral_analysis.py \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/combined_dihedral_data.pkl \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/combined_peak_analysis.json \
-c G12_GAT \
--base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT

python3 gmm_dihedral_analysis.py \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/combined_dihedral_data.pkl \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/combined_peak_analysis.json \
-c G12_ML \
--base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML

python3 gmm_dihedral_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/combined_dihedral_data.pkl -c G12_GAT
"""