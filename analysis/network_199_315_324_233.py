"""
analysis/network_gating_distance_analysis.py
=============================================
Distance-only analysis for the H233 / R324 / E315 / K199 gating relay
positions (cf. Gazgalis et al. 2022, Fig 5F/G -- there called
H233/R313/E315/K188/K199; R324 is this system's actual resid for the
Arg position). H-bond geometry (angle criteria) is NOT handled here --
this is distances only, ahead of a proper H-bond/salt-bridge/cation-pi
geometric analysis.

Residue identities (confirmed):
    233 -> HIS (or variant)
    324 -> ARG
    315 -> GLU
    199 -> LYS

For each of the 4 subunit positions, computes ALL 6 pairwise minimum
distances per frame among {233, 324, 315, 199}:
    233-324, 233-315, 233-199, 324-315, 324-199, 315-199

Plus a second set of plots showing ONLY the 3 relay pairs the paper
actually proposes:
    233-324 (H233-R324), 324-315 (R324-E315), 315-199 (E315-K199)

Three atom-subset modes, computed generically from the actual residue
name at runtime (no hardcoded per-position atom names):
    full_residue     : all heavy atoms of each residue
    side_chain       : heavy atoms of the side chain only (no backbone)
    functional_atoms : the specific charge/interaction atoms
                        (Lys NZ; Arg NH1/NH2/NE; Glu OE1/OE2;
                         Asp OD1/OD2; His ND1/NE2)
                        falls back to side_chain (with a warning) if the
                        residue at that position isn't one of these.

Subunit groups are labeled by the chain of res_324[i] (e.g. "G1", "2a",
"2b", "2c"), matching to_pdb() conventions.

All three modes are computed in a SINGLE pass over the trajectory.

CACHING: if the raw distance CSVs already exist on disk for a given
subunit group (all three modes), the trajectory is NOT re-read for
that group -- existing CSVs are loaded and only plots/stats are
regenerated. Pass force_rerun=True to ignore the cache.

Outputs (all saved under results_dir/network_199_315_324_233/):
    full_residue/
        - raw distances CSV per subunit group (6 pair columns)
        - raw + smoothed plots (all 6 pairs) per subunit group
        - raw + smoothed relay-only plots (3 pairs) per subunit group
    side_chain/
        (same structure)
    functional_atoms/
        (same structure)
    network_distance_stats.csv   (combined, all 3 modes, top level)
"""

import os
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from MDAnalysis.analysis.distances import distance_array

from analysis.converter import to_pdb

OUTPUT_FOLDER_NAME = "network_199_315_324_233"

RESIDUE_KEYS = ("233", "324", "315", "199")
PAIRS = list(combinations(RESIDUE_KEYS, 2))  # all 6 pairs
RELAY_PAIRS = [("233", "324"), ("324", "315"), ("315", "199")]  # the 3 pairs the paper proposes

ATOM_MODES = ("full_residue", "side_chain", "functional_atoms")

# residue name -> functional (charge-bearing) heavy atom names
FUNCTIONAL_ATOMS_MAP = {
    "ARG": ["NH1", "NH2", "NE"],
    "LYS": ["NZ"],
    "LYN": ["NZ"],           # neutral lysine, still has NZ
    "GLU": ["OE1", "OE2"],
    "GLH": ["OE1", "OE2"],
    "ASP": ["OD1", "OD2"],
    "ASH": ["OD1", "OD2"],
    "HIS": ["ND1", "NE2"],
    "HID": ["ND1", "NE2"],
    "HIE": ["ND1", "NE2"],
    "HIP": ["ND1", "NE2"],
    "HSD": ["ND1", "NE2"],
    "HSE": ["ND1", "NE2"],
    "HSP": ["ND1", "NE2"],
}


def _resname(u, resid):
    ag = u.select_atoms(f"resid {resid}")
    if len(ag) == 0:
        return None
    return ag.residues.resnames[0]


def _full_residue(u, resid):
    return u.select_atoms(f"resid {resid} and not name H*")


def _side_chain(u, resid):
    return u.select_atoms(f"resid {resid} and not name H* and not backbone")


def _functional_atoms(u, resid, resname):
    names = FUNCTIONAL_ATOMS_MAP.get(resname)
    if names is None:
        warnings.warn(
            f"resid {resid} has resname '{resname}', not in FUNCTIONAL_ATOMS_MAP -- "
            f"falling back to side_chain atoms for this residue."
        )
        return _side_chain(u, resid)
    sel = " or ".join(f"name {n}" for n in names)
    ag = u.select_atoms(f"resid {resid} and ({sel})")
    if len(ag) == 0:
        warnings.warn(
            f"resid {resid} (resname {resname}) -- expected atoms {names} not found, "
            f"falling back to side_chain atoms."
        )
        return _side_chain(u, resid)
    return ag


def _build_atom_groups(u, resid_map):
    """resid_map: dict key -> resid. Returns {mode: {key: AtomGroup}}."""
    ags = {mode: {} for mode in ATOM_MODES}
    resnames = {key: _resname(u, resid) for key, resid in resid_map.items()}
    for key, resid in resid_map.items():
        ags["full_residue"][key] = _full_residue(u, resid)
        ags["side_chain"][key] = _side_chain(u, resid)
        ags["functional_atoms"][key] = _functional_atoms(u, resid, resnames[key])
        for mode in ATOM_MODES:
            if len(ags[mode][key]) == 0:
                raise ValueError(f"No atoms found for resid {resid} (key {key}, mode {mode})")
    return ags


def _chain_label(pdb_label):
    return pdb_label.split(".")[-1]


def _raw_csv_path(mode_dir, channel_type, run_number, subunit_label):
    return mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_raw_distances.csv"


def _pair_col(key1, key2, lbl1, lbl2):
    return f"dist_{key1}_{key2}_{lbl1}_{lbl2}"


def _plot(frames, pair_data, threshold, title, out_path, smooth_window=None):
    plt.figure(figsize=(13, 6))
    for label, series in pair_data.items():
        y = series if smooth_window is None else pd.Series(series).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        plt.plot(frames, y, label=label, linewidth=1 if smooth_window is None else 1.2)
    plt.axhline(threshold, color="red", linestyle="--", label=f"threshold {threshold} Å")
    plt.xlabel("Frame")
    plt.ylabel("Distance (Å)")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_network_distance_analysis(
    u,
    res_324,
    res_315,
    res_233,
    res_199,
    results_dir,
    channel_type,
    run_number,
    start_frame=0,
    end_frame=None,
    smooth_window=50,
    threshold=4.0,
    force_rerun=False,
):
    """
    Parameters
    ----------
    res_324, res_315, res_233, res_199 : list[int], length 4
        Residue ids (per subunit index) for the ARG, GLU, HIS, LYS
        positions respectively. Pairing is by index: subunit i uses
        (res_233[i], res_324[i], res_315[i], res_199[i]).
        Each group is labeled using the chain of res_324[i].
    force_rerun : bool
        If False (default), and raw distance CSVs already exist on disk
        for a subunit group (all 3 atom modes), skip re-reading the
        trajectory for that group. If True, always recompute.
    """
    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    out_dir = Path(results_dir) / OUTPUT_FOLDER_NAME
    mode_dirs = {}
    for mode in ATOM_MODES:
        d = out_dir / mode
        d.mkdir(parents=True, exist_ok=True)
        mode_dirs[mode] = d

    n_subunits = len(res_324)
    assert len(res_315) == n_subunits and len(res_233) == n_subunits and len(res_199) == n_subunits, \
        "res_324, res_315, res_233, res_199 must all be the same length"

    n_frames = end_frame - start_frame + 1
    frames = np.arange(start_frame, end_frame + 1)

    stats_rows = []

    for i in range(n_subunits):
        resid_map = {"233": res_233[i], "324": res_324[i], "315": res_315[i], "199": res_199[i]}

        lbl = {key: to_pdb(resid, channel_type) for key, resid in resid_map.items()}
        subunit_label = _chain_label(lbl["324"])

        pair_cols = {
            pair: _pair_col(pair[0], pair[1], lbl[pair[0]], lbl[pair[1]])
            for pair in PAIRS
        }

        existing_csvs = {
            mode: _raw_csv_path(mode_dirs[mode], channel_type, run_number, subunit_label)
            for mode in ATOM_MODES
        }
        all_cached = all(p.exists() for p in existing_csvs.values())

        dist_arrays = {mode: {} for mode in ATOM_MODES}

        if all_cached and not force_rerun:
            print(f"\n[subunit {subunit_label}] cached CSVs found, skipping trajectory read")
            for mode in ATOM_MODES:
                df_cached = pd.read_csv(existing_csvs[mode])
                for pair in PAIRS:
                    dist_arrays[mode][pair] = df_cached[pair_cols[pair]].to_numpy()
        else:
            print(f"\n[subunit {subunit_label}] "
                  f"{lbl['233']} / {lbl['324']} / {lbl['315']} / {lbl['199']} (reading trajectory)")

            ags = _build_atom_groups(u, resid_map)

            for mode in ATOM_MODES:
                for pair in PAIRS:
                    dist_arrays[mode][pair] = np.empty(n_frames, dtype=float)

            # ---- SINGLE pass over the trajectory, all modes + all pairs ----
            for idx, ts in enumerate(
                tqdm(u.trajectory[start_frame:end_frame + 1],
                     desc=f"subunit {subunit_label}", leave=False)
            ):
                for mode in ATOM_MODES:
                    a = ags[mode]
                    for pair in PAIRS:
                        k1, k2 = pair
                        dist_arrays[mode][pair][idx] = distance_array(
                            a[k1].positions, a[k2].positions
                        ).min()

            for mode in ATOM_MODES:
                df = pd.DataFrame({"frame": frames})
                for pair in PAIRS:
                    df[pair_cols[pair]] = dist_arrays[mode][pair]
                df.to_csv(existing_csvs[mode], index=False)

        # ---- per mode: plots + stats (always regenerated) ----
        for mode in ATOM_MODES:
            mode_dir = mode_dirs[mode]

            pair_label_map = {p: f"{lbl[p[0]]}-{lbl[p[1]]}" for p in PAIRS}
            pair_data_full = {pair_label_map[p]: dist_arrays[mode][p] for p in PAIRS}
            pair_data_relay = {pair_label_map[p]: dist_arrays[mode][p] for p in RELAY_PAIRS}

            # -- all 6 pairs --
            _plot(
                frames, pair_data_full, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode}, raw, all pairs)",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_raw.png",
            )
            _plot(
                frames, pair_data_full, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode}, smoothed, window={smooth_window}, all pairs)",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_smoothed.png",
                smooth_window=smooth_window,
            )

            # -- relay-only 3 pairs --
            _plot(
                frames, pair_data_relay, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode}, raw, relay only)",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_relay_raw.png",
            )
            _plot(
                frames, pair_data_relay, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode}, smoothed, window={smooth_window}, relay only)",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_relay_smoothed.png",
                smooth_window=smooth_window,
            )

            for pair in PAIRS:
                series = dist_arrays[mode][pair]
                n_below = int(np.sum(series < threshold))
                stats_rows.append({
                    "channel_type": channel_type,
                    "run": run_number,
                    "atom_mode": mode,
                    "subunit": subunit_label,
                    "pair": f"{pair[0]}-{pair[1]}",
                    "is_relay_pair": pair in RELAY_PAIRS,
                    "resid_1": resid_map[pair[0]],
                    "resid_2": resid_map[pair[1]],
                    "label_1": lbl[pair[0]],
                    "label_2": lbl[pair[1]],
                    "n_frames": n_frames,
                    "n_below_threshold": n_below,
                    "pct_below_threshold": 100.0 * n_below / n_frames,
                    "mean": np.mean(series),
                    "median": np.median(series),
                    "min": np.min(series),
                    "max": np.max(series),
                })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = out_dir / f"{channel_type}_RUN{run_number}_network_distance_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nSaved combined stats table -> {stats_csv}")

    return stats_df