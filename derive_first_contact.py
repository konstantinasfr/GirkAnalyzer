"""
derive_first_contact.py
=======================
Makes the FIRST-contact results (residue and subunit mode) of every run from
its existing LAST-contact results, without touching the trajectories.

run_last_duration_analysis already stores, for every permeation event, all
the contacts the ion made between end_1 (leaving the SF) and end_2 (leaving
the GLU/ASN cavity) in duration_analysis_complete_<mode>.json. The first
contact is the earliest of them, so it can be read from there; the result is
the same as running run_first_duration_analysis on the trajectory.

For every channel/RUN*/duration_<mode>/<t>_<d>/duration_analysis_complete_<mode>.json
this writes channel/RUN*/duration_first_<mode>/<t>_<d>/ with
first_duration_<mode>.json, duration_first_complete_<mode>.json, the summary
and the per-run plots. Existing first-contact results there are replaced
(the old ones searched from start_2, which is wrong, see
analysis/longest_duration_analysis.py, FirstContactAnalysis).

USAGE
-----
    python3 derive_first_contact.py /path/to/girk_analyser_results --threshold 3.5 --min_duration 5
    python3 derive_first_contact.py /path/to/girk_analyser_results --threshold 3.5 --min_duration 5 --channels G12 G12_ML

Then aggregate / collect as usual:
    python3 aggregate_duration_results.py /path/to/G12 --threshold 3.5 --min_duration 5 --analysis first
    python3 collect_duration_results.py /path/to/girk_analyser_results --threshold 3.5 --min_duration 5
"""

import argparse
from pathlib import Path

from analysis.longest_duration_analysis import first_contact_from_last
from aggregate_duration_results import run_sort_key


def main():
    parser = argparse.ArgumentParser(description="First-contact results from existing last-contact results")
    parser.add_argument("results_root", help="Folder containing one folder per channel (G12, G12_ML, ...)")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold used (e.g. 3.5)")
    parser.add_argument("--min_duration", type=int, required=True, help="min_duration used (e.g. 5)")
    parser.add_argument("--channels", nargs='+', default=None,
                        help="Channels to process (default: every sub-folder with RUN* folders)")
    parser.add_argument("--no_plots", action="store_true", help="Only write the JSON files")
    args = parser.parse_args()

    root = Path(args.results_root)
    channels = args.channels or sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and any(c.is_dir() and c.name.startswith('RUN') for c in d.iterdir()))
    t = args.threshold
    variants = {f"{int(t) if float(t).is_integer() else t}_{args.min_duration}", f"{t}_{args.min_duration}"}

    for ch in channels:
        done, missing = [], []
        runs = sorted((d for d in (root / ch).iterdir() if d.is_dir() and d.name.startswith('RUN')),
                      key=lambda d: run_sort_key(d.name))
        for run_dir in runs:
            for mode in ('residue', 'subunit'):
                src = next((run_dir / f"duration_{mode}" / v / f"duration_analysis_complete_{mode}.json"
                            for v in variants
                            if (run_dir / f"duration_{mode}" / v / f"duration_analysis_complete_{mode}.json").exists()),
                           None)
                if src is None:
                    missing.append(f"{run_dir.name}/{mode}")
                    continue
                a = first_contact_from_last(src, make_plots=not args.no_plots)
                done.append((run_dir.name, mode, len(a.results), sum(1 for r in a.results if r['first_identifier'])))
        n_runs = len({r for r, *_ in done})
        print(f"✓ {ch}: {n_runs} runs")
        for run, mode, n, hit in done:
            if mode == 'residue':
                print(f"    {run:6s} {n:3d} events, {hit:3d} with a first contact")
        if missing:
            print(f"    no last-contact results for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
