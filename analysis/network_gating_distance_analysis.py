"""
analysis/network_gating_distance_analysis.py
=============================================
Distance analysis for the H233-R313-E315-K199 gating relay network
(cf. Gazgalis et al. 2022, Figure 5F/G: H233-R313 cation-pi, R313-E315
salt bridge, E315-K188/K199 salt bridge).

For each of the 4 subunit-index positions, computes 3 pairwise minimum
distances per frame, using BOTH heavy-atom-only and all-atom selections:
    pair "233-313" : res_233[i] <-> res_313[i]   (intra-chain)
    pair "313-315" : res_313[i] <-> res_315[i]   (intra-chain)
    pair "315-199" : res_315[i] <-> res_199[i]   (may cross to neighbor chain)

Subunit groups are labeled by the chain of res_313[i] (e.g. "G1", "2a",
"2b", "2c") rather than a numeric index, in both plot titles and
filenames. The res_199 partner may legitimately belong to a different
(neighboring) chain -- that's expected and shown in its own pair label.

Both atom-selection modes are computed in a single pass over the
trajectory (no re-reading frames twice).

CACHING: if the raw distance CSVs already exist on disk for a given
subunit group (both atom modes), the trajectory is NOT re-read for
that group -- the existing CSVs are loaded and only the plots/stats
are regenerated. Pass force_rerun=True to ignore existing CSVs and
recompute everything from the trajectory.

Outputs (all saved under results_dir/<script_name>/):
    heavyatom_heavyatom/
        - raw distances CSV per subunit group
        - raw + smoothed plots per subunit group
    all_atoms/
        - raw distances CSV per subunit group
        - raw + smoothed plots per subunit group
    network_distance_stats.csv   (combined, both atom modes, top level)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from MDAnalysis.analysis.distances import distance_array

from analysis.converter import to_pdb

ATOM_MODES = ("heavy", "all")
MODE_FOLDER = {"heavy": "heavyatom_heavyatom", "all": "all_atoms"}
PAIR_NAMES = ("233-313", "313-315", "315-199")


def _select(u, resid, mode):
    if mode == "heavy":
        return u.select_atoms(f"resid {resid} and not name H*")
    return u.select_atoms(f"resid {resid}")


def _chain_label(pdb_label):
    """Extract the chain suffix from a to_pdb() label, e.g. '313.2b' -> '2b'."""
    return pdb_label.split(".")[-1]


def _raw_csv_path(mode_dir, channel_type, run_number, subunit_label):
    return mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_raw_distances.csv"


def _plot(frames, pair_data, threshold, title, out_path, smooth_window=None):
    plt.figure(figsize=(12, 5))
    for label, series in pair_data.items():
        y = series if smooth_window is None else pd.Series(series).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        plt.plot(frames, y, label=label, linewidth=1 if smooth_window is None else 1.2)
    plt.axhline(threshold, color="red", linestyle="--", label=f"threshold {threshold} Å")
    plt.xlabel("Frame")
    plt.ylabel("Distance (Å)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_network_distance_analysis(
    u,
    res_313,
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
    res_313, res_315, res_233, res_199 : list[int], length 4
        Residue ids (per subunit index) for R313, E315, H233, K199.
        Pairing is by index: subunit i uses
        (res_233[i], res_313[i], res_315[i], res_199[i]).
        Each group is labeled using the chain of res_313[i]
        (e.g. "G1", "2a", "2b", "2c").
    force_rerun : bool
        If False (default), and raw distance CSVs already exist on disk
        for a subunit group (both atom modes), skip re-reading the
        trajectory for that group and just reload the CSVs to
        regenerate plots and stats. If True, always recompute from
        the trajectory.
    """
    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = Path(results_dir) / script_name
    mode_dirs = {}
    for mode in ATOM_MODES:
        d = out_dir / MODE_FOLDER[mode]
        d.mkdir(parents=True, exist_ok=True)
        mode_dirs[mode] = d

    n_subunits = len(res_313)
    assert len(res_315) == n_subunits and len(res_233) == n_subunits and len(res_199) == n_subunits, \
        "res_313, res_315, res_233, res_199 must all be the same length"

    n_frames = end_frame - start_frame + 1
    frames = np.arange(start_frame, end_frame + 1)

    stats_rows = []

    for i in range(n_subunits):
        r313, r315, r233, r199 = res_313[i], res_315[i], res_233[i], res_199[i]

        lbl313 = to_pdb(r313, channel_type)
        lbl315 = to_pdb(r315, channel_type)
        lbl233 = to_pdb(r233, channel_type)
        lbl199 = to_pdb(r199, channel_type)

        subunit_label = _chain_label(lbl313)  # e.g. "G1", "2a", "2b", "2c"

        col_233_313 = f"dist_233_313_{lbl233}_{lbl313}"
        col_313_315 = f"dist_313_315_{lbl313}_{lbl315}"
        col_315_199 = f"dist_315_199_{lbl315}_{lbl199}"

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
                dist_arrays[mode]["233-313"] = df_cached[col_233_313].to_numpy()
                dist_arrays[mode]["313-315"] = df_cached[col_313_315].to_numpy()
                dist_arrays[mode]["315-199"] = df_cached[col_315_199].to_numpy()
        else:
            # build atom groups for both modes up front (no rebuilding per frame)
            ags = {mode: {} for mode in ATOM_MODES}
            for mode in ATOM_MODES:
                for name, resid in [("313", r313), ("315", r315), ("233", r233), ("199", r199)]:
                    ag = _select(u, resid, mode)
                    if len(ag) == 0:
                        raise ValueError(
                            f"No atoms found for resid {resid} ({name}, mode={mode}, subunit {subunit_label})"
                        )
                    ags[mode][name] = ag

            print(f"\n[subunit {subunit_label}] {lbl233} - {lbl313} - {lbl315} - {lbl199} (reading trajectory)")

            for mode in ATOM_MODES:
                for pair in PAIR_NAMES:
                    dist_arrays[mode][pair] = np.empty(n_frames, dtype=float)

            # ---- SINGLE pass over the trajectory computing both atom modes ----
            for idx, ts in enumerate(
                tqdm(u.trajectory[start_frame:end_frame + 1],
                     desc=f"subunit {subunit_label}", leave=False)
            ):
                for mode in ATOM_MODES:
                    a = ags[mode]
                    dist_arrays[mode]["233-313"][idx] = distance_array(
                        a["233"].positions, a["313"].positions
                    ).min()
                    dist_arrays[mode]["313-315"][idx] = distance_array(
                        a["313"].positions, a["315"].positions
                    ).min()
                    dist_arrays[mode]["315-199"][idx] = distance_array(
                        a["315"].positions, a["199"].positions
                    ).min()

            # save raw CSVs now that we've (re)computed them
            for mode in ATOM_MODES:
                df = pd.DataFrame({
                    "frame": frames,
                    col_233_313: dist_arrays[mode]["233-313"],
                    col_313_315: dist_arrays[mode]["313-315"],
                    col_315_199: dist_arrays[mode]["315-199"],
                })
                df.to_csv(existing_csvs[mode], index=False)

        # ---- per mode: plots + stats (always regenerated) ----
        for mode in ATOM_MODES:
            mode_dir = mode_dirs[mode]
            d233_313 = dist_arrays[mode]["233-313"]
            d313_315 = dist_arrays[mode]["313-315"]
            d315_199 = dist_arrays[mode]["315-199"]

            pair_data = {
                f"{lbl233}-{lbl313} (H233-R313)": d233_313,
                f"{lbl313}-{lbl315} (R313-E315)": d313_315,
                f"{lbl315}-{lbl199} (E315-K199)": d315_199,
            }

            _plot(
                frames, pair_data, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode} atoms, raw)",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_raw.png",
            )
            _plot(
                frames, pair_data, threshold,
                title=f"{channel_type} RUN{run_number} - subunit {subunit_label} ({mode} atoms, smoothed, window={smooth_window})",
                out_path=mode_dir / f"{channel_type}_RUN{run_number}_subunit_{subunit_label}_smoothed.png",
                smooth_window=smooth_window,
            )

            for pair_name, series, r1, r2, l1, l2 in [
                ("233-313", d233_313, r233, r313, lbl233, lbl313),
                ("313-315", d313_315, r313, r315, lbl313, lbl315),
                ("315-199", d315_199, r315, r199, lbl315, lbl199),
            ]:
                n_below = int(np.sum(series < threshold))
                stats_rows.append({
                    "channel_type": channel_type,
                    "run": run_number,
                    "atom_mode": mode,
                    "subunit": subunit_label,
                    "pair": pair_name,
                    "resid_1": r1,
                    "resid_2": r2,
                    "label_1": l1,
                    "label_2": l2,
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