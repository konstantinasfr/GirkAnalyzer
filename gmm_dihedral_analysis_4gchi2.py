#!/usr/bin/env python3
"""
GMM Dihedral Analysis — variant with K=4 for 173.D chi2

Identical to gmm_dihedral_analysis.py in every way EXCEPT:
    173.D (ASP) chi2: K=4 instead of K=2

Output folder: gmm_analysis_4gchi2/ (next to the pkl file, or use -o)

Usage:
    python gmm_dihedral_analysis_4gchi2.py \
        combined_dihedral_data.pkl \
        combined_peak_analysis.json \
        -c G12_ML \
        --base-dir /path/to/G12_ML
"""

import sys
from pathlib import Path

# ── patch get_rules before importing the main module ─────────────────────────
# We import everything from gmm_dihedral_analysis and override get_rules only.
sys.path.insert(0, str(Path(__file__).parent))
import gmm_dihedral_analysis as _gda

# save original
_original_get_rules = _gda.get_rules

def _patched_get_rules(res_type, pdb_label):
    """
    Same as original get_rules but 173.x ASP chi2 uses K=4 instead of K=2.
    """
    result = _original_get_rules(res_type, pdb_label)
    if result is None:
        return None

    chi1_k, chi1_space, chi2_k, chi2_space = result
    prefix = pdb_label.split('.')[0] if '.' in pdb_label else ''

    if prefix == '173':   # ASP — override chi2 K from 2 to 4
        chi2_k = 4
        print(f"  [4gchi2 override] {pdb_label}: ASP chi2 K=2 → K=4")

    return chi1_k, chi1_space, chi2_k, chi2_space

# apply patch
_gda.get_rules = _patched_get_rules

# ── patch default output folder name ─────────────────────────────────────────
import argparse
import pickle
import json

def main():
    parser = argparse.ArgumentParser(
        description='GMM dihedral analysis — 173.D chi2 with K=4',
    )
    parser.add_argument('pkl_file',  help='Path to combined_dihedral_data.pkl')
    parser.add_argument('peak_json', help='Path to combined_peak_analysis.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: gmm_analysis_4gchi2/ '
                             'next to pkl)')
    parser.add_argument('-c', '--channel', default='G12',
                        choices=['G2', 'G12', 'G12_GAT', 'G12_ML'])
    parser.add_argument('--n-init', type=int, default=20)
    parser.add_argument('--base-dir', default=None,
                        help='Base directory with RUN* folders for permeation analysis')
    args = parser.parse_args()

    pkl_path  = Path(args.pkl_file)
    json_path = Path(args.peak_json)

    for p in [pkl_path, json_path]:
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 1

    # default output folder is gmm_analysis_4gchi2
    output_dir = Path(args.output) if args.output \
                 else pkl_path.parent / 'gmm_analysis_4gchi2'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GMM ANALYSIS VARIANT — 173.D chi2 K=4")
    print(f"Output: {output_dir}")
    print("=" * 70)

    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    combined_data = saved['combined_data'] if 'combined_data' in saved else saved

    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        peak_analysis = json.load(f)

    # run the full analysis using the patched get_rules
    gmm_results = _gda.run_gmm_analysis(
        combined_data, peak_analysis, output_dir,
        n_init=args.n_init,
        channel_type=args.channel
    )

    # all downstream plotting/analysis — identical to main script
    ordered_asn, ordered_glu = _gda.plot_gmm_fits(
        combined_data, gmm_results, output_dir, channel_type=args.channel)

    print("\n=== SAVING EXAMPLE FRAMES PER COMBINATION ===")
    _gda.save_example_frames(combined_data, gmm_results, output_dir)

    print("\n=== CREATING RUN COVERAGE HISTOGRAMS ===")
    _gda.plot_run_colored_histograms(combined_data, gmm_results, output_dir,
                                     channel_type=args.channel)

    methods = [
        ('probabilistic', 'labels',        'labels_per_run'),
        ('nearest_mean',  'labels_nearest', 'labels_per_run_near'),
    ]

    for method_name, lbl_key, lpr_key in methods:
        print(f"\n{'='*70}")
        print(f"  ASSIGNMENT METHOD: {method_name}")
        print(f"{'='*70}")

        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        # swap labels
        for res in gmm_results.values():
            for chi in ['chi1', 'chi2']:
                if chi in res:
                    res[chi]['_labels_bak']    = res[chi]['labels']
                    res[chi]['_lpr_bak']       = res[chi]['labels_per_run']
                    res[chi]['labels']         = res[chi][lbl_key]
                    res[chi]['labels_per_run'] = res[chi][lpr_key]

        _gda.plot_gmm_occupancy(combined_data, gmm_results, method_dir,
                                channel_type=args.channel,
                                ordered_asn=ordered_asn, ordered_glu=ordered_glu)

        combo_dir = method_dir / 'combinations'
        combo_dir.mkdir(exist_ok=True)
        print(f"\n=== CHI1 x CHI2 COMBINATIONS → {method_name}/combinations/ ===")
        combo_results = _gda.compute_combinations(gmm_results, combined_data)
        _gda.save_combination_report(combo_results, combo_dir)
        _gda.plot_combination_bars(combo_results, gmm_results, combo_dir,
                                   channel_type=args.channel, group='ASN_ASP')
        _gda.plot_combination_bars(combo_results, gmm_results, combo_dir,
                                   channel_type=args.channel, group='GLU')

        print(f"\n=== WRITING DEBUG REPORT → {method_name}/ ===")
        _gda.write_debug_report(combined_data, gmm_results, combo_results,
                                method_dir, channel_type=args.channel)

        if args.base_dir:
            print(f"\n=== ION EXIT EVENT ANALYSIS → {method_name}/ ===")
            base_dir    = Path(args.base_dir)
            sample_res  = next(iter(gmm_results.values()))
            run_names   = sorted(sample_res['chi1']['labels_per_run'].keys())
            perm_frames = _gda.load_permeation_frames(base_dir, run_names)

            if perm_frames:
                for offset, frame_label in [(0, 'at_exit'), (-1, 'before_exit'),
                                            (1, 'after_exit')]:
                    perm_dir = method_dir / f'ion_exit_{frame_label}'
                    perm_dir.mkdir(exist_ok=True)
                    print(f"  --- {frame_label} ---")
                    perm_combo = _gda.compute_permeation_combinations(
                        gmm_results, perm_frames, offset=offset)
                    _gda.save_permeation_report(perm_combo, perm_dir,
                                               frame_label=frame_label)
                    for grp in ['ASN_ASP', 'GLU']:
                        _gda.plot_permeation_combination_bars(
                            perm_combo, perm_dir,
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
                        _gda.plot_permeation_single_chi_bars(
                            perm_combo, perm_dir, chi_key='chi1',
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
                        _gda.plot_permeation_single_chi_bars(
                            perm_combo, perm_dir, chi_key='chi2',
                            channel_type=args.channel, group=grp,
                            frame_label=frame_label)
                    _gda.plot_permeation_histograms(
                        combined_data, gmm_results, perm_frames,
                        perm_dir, channel_type=args.channel,
                        offset=offset, frame_label=frame_label)
            else:
                print("  No permeation_table.json files found — skipping")

        # restore labels
        for res in gmm_results.values():
            for chi in ['chi1', 'chi2']:
                if chi in res:
                    res[chi]['labels']         = res[chi]['_labels_bak']
                    res[chi]['labels_per_run'] = res[chi]['_lpr_bak']
                    del res[chi]['_labels_bak']
                    del res[chi]['_lpr_bak']

    print("\n" + "=" * 70)
    print("GMM ANALYSIS (4gchi2 variant) COMPLETE")
    print(f"Results in: {output_dir}/")
    print("  173.D chi2 was fitted with K=4 instead of K=2")
    print("=" * 70 + "\n")
    return 0


if __name__ == '__main__':
    exit(main())


"""
python3 gmm_dihedral_analysis_4gchi2.py \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/combined_dihedral_data.pkl \
/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/dihedral_analysis/combined_peak_analysis.json \
-c G12_ML \
--base-dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML

python3 gmm_dihedral_analysis.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/dihedral_analysis/combined_dihedral_data.pkl -c G12_GAT
"""