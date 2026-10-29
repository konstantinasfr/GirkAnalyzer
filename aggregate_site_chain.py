"""
aggregate_site_chain.py
=======================
Pools analysis/site_chain.py over all runs of one or more systems: the frames,
stays and events CSVs of every RUN* are joined and the site-to-site chain is
analysed on the pooled ions, for both models (all residues, and grouped).

Input:  <base>/RUN*/site_chain/<settings>/{frames,stays,events,rmsd}_run*.csv
Output: <base>/aggregated_site_chain/<settings>/<model>_a<max_above>b<max_below>[_frame0][_preeq]/
        plus frames_all.csv / stays_all.csv / events_all.csv and equilibration.csv / .png
        (start frame of each run) in <settings>/
Only ions leaving the SF after their run has equilibrated are analysed (see
analysis/site_chain.py); --start_frames takes a CSV with columns run,start_frame
whose runs use that start instead of the one found from the RMSD.
<settings>: t<cutoff>_d<min_duration>, e.g. t3.5_d10

USAGE
-----
    python3 aggregate_site_chain.py /path/girk_analyser_results/G12 \\
        [/path/girk_analyser_results/G2 ...] [--models residue grouped]
        [--max_above 5] [--max_below 3] [--max_sf 5] [--window N] [--include_frame0]
        [--start_frames starts.csv] [--eq_tolerance 0.5] [--include_preequil]
Each system folder is aggregated on its own, for every <settings> folder found in its runs.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from aggregate_duration_results import run_sort_key
from analysis.site_chain import OUT_FOLDER, add_analysis_args, analyze_all, default_window

FRAME_DURATION = {"G2": 0.4}                 # ns per frame; everything else 0.075


def settings_folders(base):
    return sorted({d.name for run_dir in Path(base).glob("RUN*")
                   for d in (run_dir / OUT_FOLDER).glob("t*_d*") if d.is_dir()})


def load(base, settings):
    out = {"frames": [], "stays": [], "events": []}
    rmsd = []
    for run_dir in sorted(Path(base).glob("RUN*"), key=lambda p: run_sort_key(p.name)):
        d = run_dir / OUT_FOLDER / settings
        for f_csv in d.glob("frames_run*.csv"):
            parts = {k: d / f_csv.name.replace("frames_", f"{k}_") for k in out}
            if not all(p.exists() for p in parts.values()):
                print(f"  [SKIP] {f_csv}: stays / events CSV missing")
                continue
            out["frames"].append(pd.read_csv(parts["frames"], keep_default_na=False))
            out["stays"].append(pd.read_csv(parts["stays"]))
            out["events"].append(pd.read_csv(parts["events"]))
            r_csv = d / f_csv.name.replace("frames_", "rmsd_")
            if r_csv.exists():
                rmsd.append(pd.read_csv(r_csv))
            else:
                print(f"  [NOTE] {r_csv.name} missing: all ions of that run kept (rerun the run to add it)")
    if not out["events"]:
        return None
    data = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    data["rmsd"] = pd.concat(rmsd, ignore_index=True) if rmsd else None
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="+", help="system folders holding RUN*/")
    add_analysis_args(p)
    p.add_argument("--start_frames", help="CSV with columns run,start_frame: equilibration start of those runs")
    args = p.parse_args()
    starts = None
    if args.start_frames:
        t = pd.read_csv(args.start_frames)
        starts = dict(zip(t["run"].astype(int), t["start_frame"].astype(int)))

    for base in args.bases:
        base = Path(base)
        system = base.name
        dt = FRAME_DURATION.get(system.upper(), 0.075)
        window = default_window(system) if args.window is None else args.window
        for settings in settings_folders(base):
            data = load(base, settings)
            if data is None:
                continue
            out = base / f"aggregated_{OUT_FOLDER}" / settings
            out.mkdir(parents=True, exist_ok=True)
            for k in ("frames", "stays", "events"):
                data[k].to_csv(out / f"{k}_all.csv", index=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                analyze_all(data["frames"], data["stays"], data["events"], out,
                            f"{system} ({data['events']['run'].nunique()} runs)", dt, window,
                            models=args.models, max_above=args.max_above, max_below=args.max_below,
                            max_sf=args.max_sf, include_frame0=args.include_frame0, rmsd=data["rmsd"],
                            eq_tol=args.eq_tolerance, start_frames=starts,
                            include_preequil=args.include_preequil)


if __name__ == "__main__":
    main()
