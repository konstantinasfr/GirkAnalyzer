#!/usr/bin/env python3
"""
Ligand Category Classification — GLU (E) vs ASP (D), enriched with
cavity ion-occupancy mode columns, with an optional category-split
cavity aggregation pass.

For each RUN in a channel-type results directory:

1. Ligand category (from ligand_distance_analysis.py outputs)
   ---------------------------------------------------------------
   "Interacting" = min distance < INTERACT_THRESHOLD (default 3.5 A).
     pct_glu = % of frames where GLU141_G1 min_distance < threshold
     pct_asp = % of frames where ASP173_G1 min_distance < threshold
   Category (majority-vote, >50%):
     'E'       -> pct_glu > 50, pct_asp <= 50
     'D'       -> pct_asp > 50, pct_glu <= 50
     'E+D'     -> both > 50
     'Neither' -> neither > 50

2. Cavity occupancy mode columns (from cavity_occupancy_analysis.py outputs
   — NOT re-run here, just read from the files it already produced)
   ---------------------------------------------------------------
   For each of the three regions (glu_asn = region 2, asn_hbc = region 3,
   glu_hbc = combined region 2+3), report the ion count that occurs with
   the highest percentage of occupancy:
     *_all  -> mode of ion counts across ALL trajectory frames
     *_perm -> mode of ion counts across permeation events only
                (excl. the permeating ion itself)

   Source files (read as-is, no recomputation):
     glu_asn_all  <- trajectory_ion_counts.csv                  (n_total)
     glu_asn_perm <- cavity_occupancy_at_end_2.json             (total_ions_in_cavity)
     asn_hbc_all  <- trajectory_ion_counts_end_3.csv             (n_total)
     asn_hbc_perm <- cavity_occupancy_at_end_3.json             (total_ions_in_cavity)
     glu_hbc_all  <- trajectory_ion_counts_combined_r2r3.csv     (n_total_combined)
     glu_hbc_perm <- combined_r2r3_at_end3.csv                   (n_total_combined_excl)

3. Optional category-split aggregation (--split-by-category)
   ---------------------------------------------------------------
   Groups RUNs by their ligand category and re-runs BOTH:
     - the aggregate_cavity_occupancy.py plotting pipeline
     - the aggregate_cavity_refilling.py plotting pipeline (cavity_refilling/
       subfolder)
   separately for the E-category runs and the D-category runs. 'E+D' and
   'Neither' categories are excluded from BOTH groups. Neither
   aggregate_cavity_occupancy.py nor aggregate_cavity_refilling.py needs any
   modification — a temporary folder of symlinks (pointing only at the
   relevant RUN dirs) is built per category and passed in as the base
   directory, then removed again afterward. Requires both scripts to be
   importable (same directory or PYTHONPATH).

   Output layout (inside base_dir/ligand_inter_class/):
       ligand_categories.csv    <- same table as --output, saved here too
       E/                       <- E-category runs only
           glu_asn/  asn_hbc/  glu_hbc/          (cavity occupancy)
           cavity_refilling/                      (cavity refilling)
       D/                       <- D-category runs only
           glu_asn/  asn_hbc/  glu_hbc/
           cavity_refilling/

Usage:
    python ligand_category_classification.py /path/to/G12_ML
    python ligand_category_classification.py /path/to/G12_GAT \\
        --threshold 3.5 --cavity-subdir cavity_occupancy --output results.csv

    # also split cavity aggregation by category:
    python ligand_category_classification.py /path/to/G12_ML \\
        --split-by-category --channel G12_ML --cavity-threshold 3.0
"""

import argparse
import csv
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

# ── ligand distance config ──────────────────────────────────────────────────
GLU_LABEL = 'GLU141_G1'
ASP_LABEL = 'ASP173_G1'
LIG_SUBDIR = 'ligand_distance_analysis'
LIG_PKL_NAME = 'ligand_distance_raw_data.pkl'

# ── cavity occupancy config ─────────────────────────────────────────────────
CAVITY_SUBDIR_CANDIDATES = ['cavity_occupancy', 'cavity_occupancy_analysis']


# =============================================================================
# LIGAND CATEGORY
# =============================================================================

def _load_min_distance(run_dir: Path, label: str):
    pkl_path = run_dir / LIG_SUBDIR / label / LIG_PKL_NAME
    if not pkl_path.exists():
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return np.asarray(data['min_distance'])


def classify_ligand(run_dir: Path, threshold: float):
    glu_dist = _load_min_distance(run_dir, GLU_LABEL)
    asp_dist = _load_min_distance(run_dir, ASP_LABEL)

    if glu_dist is None or asp_dist is None:
        missing = []
        if glu_dist is None:
            missing.append(GLU_LABEL)
        if asp_dist is None:
            missing.append(ASP_LABEL)
        return {'n_frames': '', 'pct_glu(E)': '', 'pct_asp(D)': '',
                'category': '', 'ligand_error': f"missing: {', '.join(missing)}"}

    if len(glu_dist) != len(asp_dist):
        n = min(len(glu_dist), len(asp_dist))
        glu_dist = glu_dist[:n]
        asp_dist = asp_dist[:n]

    n_frames = len(glu_dist)
    pct_glu = float(np.sum(glu_dist < threshold)) / n_frames * 100
    pct_asp = float(np.sum(asp_dist < threshold)) / n_frames * 100

    if pct_glu > 50 and pct_asp <= 50:
        category = 'E'
    elif pct_asp > 50 and pct_glu <= 50:
        category = 'D'
    elif pct_glu > 50 and pct_asp > 50:
        category = 'E+D'
    else:
        category = 'Neither'

    return {
        'n_frames': n_frames,
        'pct_glu(E)': round(pct_glu, 1),
        'pct_asp(D)': round(pct_asp, 1),
        'category': category,
        'ligand_error': '',
    }


# =============================================================================
# CAVITY OCCUPANCY MODE COLUMNS
# =============================================================================

def _find_cavity_dir(run_dir: Path, override: str = None):
    candidates = [override] if override else CAVITY_SUBDIR_CANDIDATES
    for name in candidates:
        d = run_dir / name
        if d.exists():
            return d
    return None


def _mode_from_csv_column(csv_path: Path, column: str):
    """Return the most frequent (mode) value of an integer CSV column."""
    if not csv_path.exists():
        return None
    vals = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get(column, '') != '':
                vals.append(int(float(row[column])))
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _mode_from_json_results(json_path: Path, field: str):
    """Return the most frequent (mode) value of a field across
    the 'results' list of a cavity_occupancy_at_end_*.json file."""
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    vals = [r[field] for r in data.get('results', []) if field in r]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def get_cavity_columns(run_dir: Path, cavity_subdir_override: str = None,
                        include_permeating: bool = False):
    cavity_dir = _find_cavity_dir(run_dir, cavity_subdir_override)

    cols = {
        'glu_asn_all': '', 'glu_asn_perm': '',
        'asn_hbc_all': '', 'asn_hbc_perm': '',
        'glu_hbc_all': '', 'glu_hbc_perm': '',
        'cavity_error': '',
    }

    if cavity_dir is None:
        cols['cavity_error'] = (
            f"no cavity dir found (tried "
            f"{cavity_subdir_override or CAVITY_SUBDIR_CANDIDATES})"
        )
        return cols

    perm_field = 'total_ions_including_permeating' if include_permeating \
        else 'total_ions_in_cavity'
    combined_perm_col = 'n_total_combined_incl' if include_permeating \
        else 'n_total_combined_excl'

    glu_asn_all = _mode_from_csv_column(
        cavity_dir / 'trajectory_ion_counts.csv', 'n_total')
    glu_asn_perm = _mode_from_json_results(
        cavity_dir / 'cavity_occupancy_at_end_2.json', perm_field)

    asn_hbc_all = _mode_from_csv_column(
        cavity_dir / 'trajectory_ion_counts_end_3.csv', 'n_total')
    asn_hbc_perm = _mode_from_json_results(
        cavity_dir / 'cavity_occupancy_at_end_3.json', perm_field)

    glu_hbc_all = _mode_from_csv_column(
        cavity_dir / 'trajectory_ion_counts_combined_r2r3.csv', 'n_total_combined')
    glu_hbc_perm = _mode_from_csv_column(
        cavity_dir / 'combined_r2r3_at_end3.csv', combined_perm_col)

    cols.update({
        'glu_asn_all': glu_asn_all if glu_asn_all is not None else '',
        'glu_asn_perm': glu_asn_perm if glu_asn_perm is not None else '',
        'asn_hbc_all': asn_hbc_all if asn_hbc_all is not None else '',
        'asn_hbc_perm': asn_hbc_perm if asn_hbc_perm is not None else '',
        'glu_hbc_all': glu_hbc_all if glu_hbc_all is not None else '',
        'glu_hbc_perm': glu_hbc_perm if glu_hbc_perm is not None else '',
    })

    missing = [k for k in ['glu_asn_all', 'glu_asn_perm', 'asn_hbc_all',
                           'asn_hbc_perm', 'glu_hbc_all', 'glu_hbc_perm']
               if cols[k] == '']
    if missing:
        cols['cavity_error'] = f"missing source file(s) for: {', '.join(missing)}"

    return cols


# =============================================================================
# COMBINE PER RUN
# =============================================================================

def classify_run(run_dir: Path, threshold: float, cavity_subdir_override: str,
                  include_permeating: bool):
    row = {'run': run_dir.name}
    row.update(classify_ligand(run_dir, threshold))
    row.update(get_cavity_columns(run_dir, cavity_subdir_override, include_permeating))
    return row


def _find_run_dirs(base_dir: Path):
    return sorted(
        [p for p in base_dir.iterdir()
         if p.is_dir() and p.name.upper().startswith('RUN')],
        key=lambda p: p.name
    )


def _save_csv(rows, out_path: Path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    print(f"✅ Saved: {out_path}")


FIELDNAMES = ['run', 'n_frames', 'pct_glu(E)', 'pct_asp(D)', 'category',
              'glu_asn_all', 'glu_asn_perm', 'asn_hbc_all', 'asn_hbc_perm',
              'glu_hbc_all', 'glu_hbc_perm', 'ligand_error', 'cavity_error']


def run_classification(base_dir, threshold=3.5, output=None,
                        cavity_subdir=None, include_permeating=False):
    base_dir = Path(base_dir)
    run_dirs = _find_run_dirs(base_dir)

    if not run_dirs:
        print(f"[ERROR] No RUN* directories found in {base_dir}")
        return None

    rows = [classify_run(rd, threshold, cavity_subdir, include_permeating)
            for rd in run_dirs]

    # ── print table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"LIGAND CATEGORY + CAVITY OCCUPANCY — {base_dir.name}")
    print(f"Ligand interaction threshold: {threshold} Å   |   "
          f"permeation counts: {'incl.' if include_permeating else 'excl.'} permeating ion")
    print(f"{'=' * 130}")
    header = (f"{'Run':<8}{'Frames':>8}{'%E':>7}{'%D':>7}{'Cat':>8}"
              f"{'glu_asn_all':>13}{'glu_asn_perm':>14}"
              f"{'asn_hbc_all':>13}{'asn_hbc_perm':>14}"
              f"{'glu_hbc_all':>13}{'glu_hbc_perm':>14}")
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['run']:<8}{r['n_frames']!s:>8}{r['pct_glu(E)']!s:>7}"
              f"{r['pct_asp(D)']!s:>7}{r['category']:>8}"
              f"{r['glu_asn_all']!s:>13}{r['glu_asn_perm']!s:>14}"
              f"{r['asn_hbc_all']!s:>13}{r['asn_hbc_perm']!s:>14}"
              f"{r['glu_hbc_all']!s:>13}{r['glu_hbc_perm']!s:>14}")
        if r['ligand_error']:
            print(f"    [ligand warning] {r['ligand_error']}")
        if r['cavity_error']:
            print(f"    [cavity warning] {r['cavity_error']}")

    # ── category summary ─────────────────────────────────────────────────
    valid_cat = [r['category'] for r in rows if r['category']]
    print(f"\n{'-' * 130}")
    print("LIGAND CATEGORY SUMMARY")
    for cat in ['E', 'D', 'E+D', 'Neither']:
        n = valid_cat.count(cat)
        pct = n / len(valid_cat) * 100 if valid_cat else 0
        print(f"  {cat:<10} {n:>3} / {len(valid_cat)} runs ({pct:.1f}%)")
    print(f"{'=' * 130}\n")

    if output:
        _save_csv(rows, Path(output), FIELDNAMES)

    return rows


# =============================================================================
# OPTIONAL: category-split cavity aggregation
# =============================================================================

def _build_symlink_view(base_dir: Path, run_names, view_dir: Path):
    """
    Create a directory containing symlinks to only the given RUN dirs, so
    the UNMODIFIED aggregate_cavity_occupancy.py (which always scans its
    base_directory for every RUN* folder) only sees this subset.
    Nothing in aggregate_cavity_occupancy.py needs to change for this.
    """
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)
    for name in run_names:
        src = (base_dir / name).resolve()
        dst = view_dir / name
        dst.symlink_to(src, target_is_directory=True)
    return view_dir


def _run_cavity_refilling(view_dir: Path, output_dir: Path, frame_duration: float,
                           channel_name: str):
    """
    Run the aggregate_cavity_refilling.py pipeline against a symlink view
    (only the runs in one category), writing into output_dir.
    Imported lazily, same reasoning as aggregate_cavity_occupancy above.
    """
    from aggregate_cavity_refilling import (
        load_all_runs, save_aggregated_csv, save_flux_summary, plot_flux,
        print_summary, plot_all, plot_pct_bar_aggregated, plot_transit_histograms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, flux_rows = load_all_runs(view_dir)
    if not all_rows:
        print(f"  [cavity_refilling] No data found for {channel_name} — skipping.")
        return

    save_aggregated_csv(all_rows, output_dir)
    flux_result = save_flux_summary(flux_rows, output_dir)
    agg_flux = flux_result[1] if flux_result else None
    plot_flux(flux_rows, output_dir, aggregated_flux=agg_flux, channel_name=channel_name)
    print_summary(all_rows)
    plot_all(all_rows, output_dir, channel_name=channel_name)
    plot_pct_bar_aggregated(all_rows, output_dir, channel_name=channel_name)
    plot_transit_histograms(all_rows, output_dir, frame_duration=frame_duration,
                             channel_name=channel_name)


def split_by_category(base_dir, rows, channel_type, cavity_threshold, frame_duration=0.075):
    """
    Groups already-classified runs by category and re-runs BOTH the
    aggregate_cavity_occupancy.py AND aggregate_cavity_refilling.py plotting
    pipelines separately for E-category and D-category runs. 'E+D' and
    'Neither' are excluded from both groups.

    Requires NO changes to either aggregate_*.py file: instead of passing a
    run filter into them, this builds a temporary folder of symlinks to just
    the relevant RUN dirs and points the unmodified pipelines at that folder.
    The symlink folder is removed again afterward — only symlinks get
    unlinked, the real RUN directories are untouched.

    Both aggregate_cavity_occupancy.py and aggregate_cavity_refilling.py are
    imported lazily so this file still works standalone when
    --split-by-category isn't used.
    """
    from aggregate_cavity_occupancy import aggregate_both_stages

    base_dir = Path(base_dir)
    inter_class_dir = base_dir / 'ligand_inter_class'
    inter_class_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(rows, inter_class_dir / 'ligand_categories.csv', FIELDNAMES)

    e_runs = [r['run'] for r in rows if r['category'] == 'E']
    d_runs = [r['run'] for r in rows if r['category'] == 'D']
    excluded = [r['run'] for r in rows if r['category'] in ('E+D', 'Neither', '')]

    print(f"\n{'=' * 70}")
    print(f"CATEGORY SPLIT — {base_dir.name}")
    print(f"{'=' * 70}")
    print(f"  E runs      ({len(e_runs)}): {e_runs}")
    print(f"  D runs      ({len(d_runs)}): {d_runs}")
    print(f"  Excluded    ({len(excluded)}): {excluded}  "
          f"[E+D / Neither / errored runs — not aggregated]")
    print(f"{'=' * 70}\n")

    for label, run_list in [('E', e_runs), ('D', d_runs)]:
        out_dir = inter_class_dir / label
        if not run_list:
            print(f"[SKIP] No runs classified as '{label}' — skipping {out_dir}")
            continue

        print(f"\n{'#' * 80}")
        print(f"# AGGREGATING CATEGORY '{label}'  ({len(run_list)} runs)")
        print(f"{'#' * 80}")

        view_dir = inter_class_dir / f"_{label}_view"
        _build_symlink_view(base_dir, run_list, view_dir)
        try:
            aggregate_both_stages(
                base_directory=view_dir,
                threshold=cavity_threshold,
                channel_type=channel_type,
                output_dir=out_dir,
            )
            _run_cavity_refilling(
                view_dir=view_dir,
                output_dir=out_dir / "cavity_refilling",
                frame_duration=frame_duration,
                channel_name=f"{channel_type} ({label})",
            )
        finally:
            shutil.rmtree(view_dir, ignore_errors=True)

    print(f"\n✅ Category-split aggregation done. Results in: {inter_class_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('base_dir',
                         help='Channel-type results dir containing RUN* folders '
                              '(e.g. .../girk_analyser_results/G12_ML)')
    parser.add_argument('--threshold', type=float, default=3.5,
                         help='Ligand interaction distance threshold in Å (default: 3.5)')
    parser.add_argument('--cavity-subdir', default=None,
                         help='Name of the cavity_occupancy_analysis.py results '
                              'subfolder inside each RUN dir. If omitted, tries '
                              f'{CAVITY_SUBDIR_CANDIDATES} automatically.')
    parser.add_argument('--include-permeating', action='store_true',
                         help='Use ion counts INCLUDING the permeating ion itself '
                              'for the *_perm columns (default: excludes it).')
    parser.add_argument('--output', default=None,
                         help='CSV output path. Defaults to '
                              '<base_dir>/ligand_inter_class/ligand_categories.csv if omitted.')
    parser.add_argument('--no-split', dest='split_by_category', action='store_false',
                         default=True,
                         help="Skip the E/D category-split aggregation. By default this "
                              "runs automatically, re-running the aggregate_cavity_occupancy.py "
                              "and aggregate_cavity_refilling.py plotting pipelines separately "
                              "for E-category and D-category runs (no changes needed to either "
                              "script — see split_by_category()).")
    parser.add_argument('-c', '--channel', default=None,
                         choices=['G2', 'G12', 'G12_GAT', 'G12_ML'],
                         help='Channel type for the split-aggregation step. If omitted, '
                              "inferred from base_dir's folder name (e.g. '.../G12_ML' -> G12_ML), "
                              'matching the channel-inference convention used elsewhere in the pipeline.')
    parser.add_argument('--cavity-threshold', type=float, default=3.0,
                         help='Cavity occupancy distance threshold in Å, passed to '
                              'the aggregator when --split-by-category is used (default: 3.0)')
    parser.add_argument('--frame-duration', type=float, default=0.075,
                         help='Time (ns) per trajectory frame, passed to the cavity '
                              'refilling transit-time histograms (default: 0.075)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output = Path(args.output) if args.output else \
        base_dir / 'ligand_inter_class' / 'ligand_categories.csv'

    rows = run_classification(args.base_dir, threshold=args.threshold,
                               output=output, cavity_subdir=args.cavity_subdir,
                               include_permeating=args.include_permeating)

    if rows and args.split_by_category:
        channel = args.channel or base_dir.name
        split_by_category(args.base_dir, rows, channel, args.cavity_threshold,
                           frame_duration=args.frame_duration)


if __name__ == '__main__':
    main()

'''
python3 g12ml_ligand_category_classification.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML
'''