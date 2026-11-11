#!/usr/bin/env python3
"""
swap_gate_columns.py
====================
Bring old gate-distance CSVs into the new pair order WITHOUT redoing the
trajectory pass.

Background: compute_constriction_point_diameters() writes the first
*_diagonal_pairs entry to column A_C and the second to B_D. The E141Q / D173N
/ GAT / ML configs used to list the B-D pair first, so in their old CSVs
"A_C" is subunits B-D (with GIRK1) and "B_D" is subunits A-C. The configs now
list A-C first (GIRK1 in B_D, as in G12 WT). Swapping A_C <-> B_D in the old
files gives exactly what the new order would compute: every distance is the
same, only the two columns trade places. `mean` does not change.

Files swapped in each RUN*/ folder:
    hbc_distances.csv  sf_distances.csv  upper_gl_distances.csv  lower_gl_distances.csv

SAFETY -- a run is swapped only if all of these hold:
  1. RUN*/subunit_check.txt exists and says PASSED, i.e. the CURRENT config of
     this system was checked against the topology (A_C = A-C, B_D = B-D,
     GIRK1 in B_D). Make it without the long analysis:
         python3 main.py --channel_type G12_E141Q --run_number 1 --subunit_check_only
  2. hbc_distances.csv is OLDER than the FIRST passing check
     (subunit_check_first_pass.txt, never rewritten; subunit_check.txt if that
     is missing), i.e. it was written before the new order was checked. A CSV newer than the check
     was written by a rerun with the new order and is left alone.
  3. there is no gate_columns_swapped.json marker (not swapped already).
Originals are copied to RUN*/gate_csv_backup_old_order/ first.

USAGE
    # see what would happen, change nothing
    python3 swap_gate_columns.py /media/ziyue/.../girk_analyser_results/G12_E141Q --dry_run

    python3 swap_gate_columns.py /media/ziyue/.../girk_analyser_results/G12_E141Q
    python3 swap_gate_columns.py /media/ziyue/.../girk_analyser_results/G12_GAT --runs 1-10

    # put the originals back
    python3 swap_gate_columns.py /media/ziyue/.../G12_E141Q --undo

AFTERWARDS rerun whatever was computed from these CSVs (gate_openness,
hbc_open_pip_analysis, pip_gate_correlation, ...). PNGs that main.py drew from
the old files (e.g. hbc_pairs_distances.png) still carry the old labels.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from gate_openness import expand_runs, natural_key

GATE_FILES = ["hbc_distances.csv", "sf_distances.csv",
              "upper_gl_distances.csv", "lower_gl_distances.csv"]
MARKER = "gate_columns_swapped.json"
BACKUP_DIR = "gate_csv_backup_old_order"
CHECK_FILE = "subunit_check.txt"
FIRST_PASS_FILE = "subunit_check_first_pass.txt"


def run_dirs(base, runs):
    names = runs if runs else sorted((d.name for d in base.glob("RUN*") if d.is_dir()),
                                     key=natural_key)
    return [base / n for n in names]


def check_status(rd):
    """(ok_to_swap, reason)."""
    if (rd / MARKER).exists():
        return False, "already swapped (marker present)"
    hbc = rd / "hbc_distances.csv"
    if not hbc.is_file():
        return False, "no hbc_distances.csv"
    chk = rd / CHECK_FILE
    if not chk.is_file():
        return False, (f"no {CHECK_FILE} -- run main.py ... --run_number "
                       f"{rd.name[3:]} --subunit_check_only first")
    if "SUBUNIT CHECK PASSED" not in chk.read_text():
        return False, f"{CHECK_FILE} does not say PASSED -- fix the config first"
    first = rd / FIRST_PASS_FILE      # time of the first passing check (never rewritten)
    ref = first if first.is_file() else chk
    if hbc.stat().st_mtime >= ref.stat().st_mtime:
        return False, ("hbc_distances.csv is newer than the subunit check -- "
                       "already written in the new order")
    return True, "old order"


def swap_run(rd, dry_run):
    done = []
    for name in GATE_FILES:
        f = rd / name
        if not f.is_file():
            print(f"      {name:<24} missing, skipped")
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)   # text: values copied exactly
        if not {"A_C", "B_D"} <= set(df.columns):
            print(f"      {name:<24} has no A_C/B_D columns, skipped")
            continue
        med = {c: pd.to_numeric(df[c], errors="coerce").median() for c in ("A_C", "B_D")}
        print(f"      {name:<24} A_C median {med['A_C']:5.2f} <-> B_D median {med['B_D']:5.2f}")
        if dry_run:
            continue
        (rd / BACKUP_DIR).mkdir(exist_ok=True)
        shutil.copy2(f, rd / BACKUP_DIR / name)
        df[["A_C", "B_D"]] = df[["B_D", "A_C"]].values
        tmp = f.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(f)
        done.append(name)
    if done and not dry_run:
        (rd / MARKER).write_text(json.dumps({
            "swapped": done,
            "when": datetime.now().isoformat(timespec="seconds"),
            "note": "A_C and B_D exchanged to match the new pair order "
                    "(A_C = subunits A-C, B_D = subunits B-D, GIRK1 in B_D); "
                    f"originals in {BACKUP_DIR}/",
        }, indent=2))
    return done


def undo_run(rd, dry_run):
    if not (rd / MARKER).exists():
        print("      not swapped, nothing to undo")
        return
    for name in json.loads((rd / MARKER).read_text())["swapped"]:
        print(f"      restore {name}")
        if not dry_run:
            shutil.copy2(rd / BACKUP_DIR / name, rd / name)
    if not dry_run:
        (rd / MARKER).unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_dir", help="e.g. .../girk_analyser_results/G12_E141Q")
    ap.add_argument("--runs", default=None, help='e.g. "1-6" (default: every RUN*)')
    ap.add_argument("--dry_run", action="store_true", help="report only, change nothing")
    ap.add_argument("--undo", action="store_true", help="restore the backed-up originals")
    args = ap.parse_args()

    base = Path(args.system_dir)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    tag = " (dry run)" if args.dry_run else ""
    print(f"{'UNDO' if args.undo else 'SWAP'} A_C <-> B_D in {base}{tag}")

    swapped, skipped = [], []
    for rd in run_dirs(base, expand_runs(args.runs) if args.runs else None):
        if not rd.is_dir():
            print(f"  {rd.name}: no such directory")
            continue
        if args.undo:
            print(f"  {rd.name}:")
            undo_run(rd, args.dry_run)
            continue
        ok, why = check_status(rd)
        print(f"  {rd.name}: {why}")
        if not ok:
            skipped.append(rd.name)
            continue
        if swap_run(rd, args.dry_run) or args.dry_run:
            swapped.append(rd.name)

    if not args.undo:
        verb = "would swap" if args.dry_run else "swapped"
        print(f"\n{verb}: {', '.join(swapped) or 'none'}")
        print(f"skipped: {', '.join(skipped) or 'none'}")


if __name__ == "__main__":
    main()
