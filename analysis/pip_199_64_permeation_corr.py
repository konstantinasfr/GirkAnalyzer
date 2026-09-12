# analysis/pip_199_64_permeation_corr.py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from MDAnalysis.lib.distances import distance_array
from analysis.converter import to_pdb


def _min_dist_series(u, pip_resid, res_resid, phosphate, start_frame, end_frame, cache_path, force=False):
    if cache_path.exists() and not force:
        return np.load(cache_path)["dists"]

    pip_atom = u.select_atoms(f"resid {pip_resid} and name {phosphate}")
    res_atoms = u.select_atoms(f"protein and resid {res_resid}")

    n = end_frame - start_frame + 1
    dists = np.empty(n, dtype=np.float32)
    desc = f"{phosphate} PIP{pip_resid} vs resid{res_resid}"
    for i, ts in enumerate(tqdm(u.trajectory[start_frame:end_frame + 1], total=n, desc=desc)):
        dists[i] = distance_array(pip_atom.positions, res_atoms.positions).min()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, dists=dists)
    return dists


def build_run_permeation_correlation_table(
    u,
    triples,              # [(pip_resid, res199_resid, res64_resid), ...]
    pip_mapping,          # {pip_resid: subunit_label}
    results_dir,
    run_number,
    channel_type,
    start_frame,
    end_frame,
    n_permeations,
    phosphate="P5",
    cutoff=4.0,
    force=False,
):
    out_dir = Path(results_dir) / "pip_199_64_permeation_corr"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_distances"

    out_csv = out_dir / f"{phosphate}_199_64_permeation_table.csv"

    # skip whole run if final table already exists
    if out_csv.exists() and not force:
        print(f"Already exists, skipping -> {out_csv}")
        return pd.read_csv(out_csv)

    rows = []
    for pip_resid, res_199, res_64 in triples:
        subunit = pip_mapping.get(pip_resid, "?")

        d199 = _min_dist_series(
            u, pip_resid, res_199, phosphate, start_frame, end_frame,
            raw_dir / f"{phosphate}_{pip_resid}_199.npz", force=force,
        )
        d64 = _min_dist_series(
            u, pip_resid, res_64, phosphate, start_frame, end_frame,
            raw_dir / f"{phosphate}_{pip_resid}_64.npz", force=force,
        )

        n_frames = len(d199)
        pct_199 = 100.0 * (d199 <= cutoff).sum() / n_frames
        pct_64 = 100.0 * (d64 <= cutoff).sum() / n_frames

        rows.append({
            "run": run_number,
            "channel_type": channel_type,
            "subunit": subunit,
            "pip_resid": pip_resid,
            "phosphate": phosphate,
            "cutoff": cutoff,
            "n_frames": n_frames,
            "res_199_resid": res_199,
            "res_199_pdb": to_pdb(res_199, channel_type),
            "res_64_resid": res_64,
            "res_64_pdb": to_pdb(res_64, channel_type),
            "pct_close_199": pct_199,
            "pct_close_64": pct_64,
            "n_permeations": n_permeations,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    return df


def build_hbc_permeation_percentage_table(
    u,
    triples,              # [(pip_resid, res199_resid, res64_resid), ...]
    pip_mapping,          # {pip_resid: subunit_label}
    results_dir,
    run_number,
    channel_type,
    hbc_frames,           # list of frame numbers (ints), or list of dicts with a "frame" key
    start_frame,
    end_frame,
    phosphate="P5",
    cutoff=4.0,
    force=False,
):
    """
    Restricts to only the HBC permeation frames (instead of the whole
    trajectory) and computes, per subunit:
        pct_close_199 = % of permeation frames where P5 was within cutoff of 199
        pct_close_64  = % of permeation frames where P5 was within cutoff of 64
    Reuses the same cached per-frame distance arrays from
    build_run_permeation_correlation_table.
    """
    frames = []
    for e in hbc_frames:
        if isinstance(e, dict):
            frames.append(int(e["frame"]))
        else:
            frames.append(int(e))

    out_dir = Path(results_dir) / "pip_199_64_permeation_corr"
    raw_dir = out_dir / "raw_distances"
    out_csv = out_dir / f"{phosphate}_hbc_permeation_percentages.csv"

    if out_csv.exists() and not force:
        print(f"Already exists, skipping -> {out_csv}")
        return pd.read_csv(out_csv)

    idxs = np.array([f - start_frame for f in frames])
    valid = (idxs >= 0)  # upper bound checked per-subunit below (arrays are same length)

    rows = []
    for pip_resid, res_199, res_64 in triples:
        subunit = pip_mapping.get(pip_resid, "?")

        d199 = _min_dist_series(
            u, pip_resid, res_199, phosphate, start_frame, end_frame,
            raw_dir / f"{phosphate}_{pip_resid}_199.npz", force=force,
        )
        d64 = _min_dist_series(
            u, pip_resid, res_64, phosphate, start_frame, end_frame,
            raw_dir / f"{phosphate}_{pip_resid}_64.npz", force=force,
        )

        in_range = valid & (idxs < len(d199))
        sel_idxs = idxs[in_range]

        n_events = len(sel_idxs)
        if n_events == 0:
            pct_199 = np.nan
            pct_64 = np.nan
        else:
            pct_199 = 100.0 * (d199[sel_idxs] <= cutoff).sum() / n_events
            pct_64 = 100.0 * (d64[sel_idxs] <= cutoff).sum() / n_events

        rows.append({
            "run": run_number,
            "channel_type": channel_type,
            "subunit": subunit,
            "pip_resid": pip_resid,
            "phosphate": phosphate,
            "cutoff": cutoff,
            "n_hbc_permeation_events": n_events,
            "pct_close_199": pct_199,
            "pct_close_64": pct_64,
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    return df