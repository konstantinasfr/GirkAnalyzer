"""
analysis/relay_state_classification.py
========================================
For a SINGLE run, reads the functional_atoms (or other mode) distance
CSVs already written by network_199_315_324_233.py, classifies every
frame per subunit as open / closed / ambiguous using the relay logic,
and reports the per-subunit occupancy breakdown for that run.

Called directly from main.py after run_network_distance_analysis(),
using the same results_dir / channel_type / run_number -- it just reads
the CSVs that were already saved to disk, no need to touch the
trajectory again.

State definitions (threshold default 4.0 A):
    closed : 233-324 < thr  AND  324-315 > thr  AND  315-199 < thr
    open   : 233-324 > thr  AND  324-315 < thr  AND  315-199 > thr
    ambiguous : anything else

Output:
    results_dir/network_199_315_324_233/{channel_type}_RUN{run_number}_relay_state_summary.csv
    (one row per subunit, columns: pct_open, pct_closed, pct_ambiguous,
     n_transitions, is_switching)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

NETWORK_FOLDER = "network_199_315_324_233"
PAIR_PREFIXES = {
    "233_324": "dist_233_324_",
    "324_315": "dist_324_315_",
    "315_199": "dist_315_199_",
}
FILENAME_RE = re.compile(r"_subunit_([A-Za-z0-9]+)_raw_distances\.csv$")


def _find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 column starting with '{prefix}', found {matches}")
    return matches[0]


def classify_states(d233_324, d324_315, d315_199, threshold):
    closed = (d233_324 < threshold) & (d324_315 > threshold) & (d315_199 < threshold)
    openv = (d233_324 > threshold) & (d324_315 < threshold) & (d315_199 > threshold)
    state = np.full(len(d233_324), "ambiguous", dtype=object)
    state[closed] = "closed"
    state[openv] = "open"
    return state


def count_real_transitions(state):
    """Collapse out 'ambiguous' frames, then count closed<->open flips."""
    collapsed = state[state != "ambiguous"]
    if len(collapsed) < 2:
        return 0
    changes = collapsed[1:] != collapsed[:-1]
    return int(np.sum(changes))


def run_relay_state_classification(
    results_dir,
    channel_type,
    run_number,
    mode="functional_atoms",
    threshold=4.0,
    min_occupancy_pct=2.0,
):
    mode_dir = Path(results_dir) / NETWORK_FOLDER / mode
    if not mode_dir.exists():
        print(f"[relay_state_classification] {mode_dir} does not exist -- "
              f"run run_network_distance_analysis() first.")
        return None

    pattern = f"{channel_type}_RUN{run_number}_subunit_*_raw_distances.csv"
    csv_paths = sorted(mode_dir.glob(pattern))
    if not csv_paths:
        print(f"[relay_state_classification] No files matching '{pattern}' in {mode_dir}")
        return None

    rows = []
    print(f"\nRelay state breakdown -- {channel_type} RUN{run_number} ({mode}, threshold={threshold} Å):")

    for csv_path in csv_paths:
        m = FILENAME_RE.search(csv_path.name)
        if not m:
            continue
        subunit_label = m.group(1)

        df = pd.read_csv(csv_path)
        col_233_324 = _find_col(df, PAIR_PREFIXES["233_324"])
        col_324_315 = _find_col(df, PAIR_PREFIXES["324_315"])
        col_315_199 = _find_col(df, PAIR_PREFIXES["315_199"])

        d1 = df[col_233_324].to_numpy()
        d2 = df[col_324_315].to_numpy()
        d3 = df[col_315_199].to_numpy()

        state = classify_states(d1, d2, d3, threshold)
        n_total = len(state)
        n_open = int(np.sum(state == "open"))
        n_closed = int(np.sum(state == "closed"))
        n_ambig = int(np.sum(state == "ambiguous"))
        n_transitions = count_real_transitions(state)

        pct_open = 100.0 * n_open / n_total
        pct_closed = 100.0 * n_closed / n_total
        pct_ambiguous = 100.0 * n_ambig / n_total

        is_switching = (
            pct_open >= min_occupancy_pct
            and pct_closed >= min_occupancy_pct
            and n_transitions >= 1
        )

        print(f"  {subunit_label:>3}: open {pct_open:>5.1f}%   closed {pct_closed:>5.1f}%   "
              f"ambiguous {pct_ambiguous:>5.1f}%   transitions={n_transitions}   "
              f"switching={is_switching}")

        rows.append({
            "channel_type": channel_type,
            "run": run_number,
            "subunit": subunit_label,
            "n_frames": n_total,
            "pct_open": round(pct_open, 2),
            "pct_closed": round(pct_closed, 2),
            "pct_ambiguous": round(pct_ambiguous, 2),
            "n_transitions": n_transitions,
            "is_switching": is_switching,
        })

    result_df = pd.DataFrame(rows)
    out_csv = Path(results_dir) / NETWORK_FOLDER / f"{channel_type}_RUN{run_number}_relay_state_summary.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    return result_df