"""
untracked_cavity_ions.py
========================
For every frame of the simulation, finds all K+ ions physically inside
channel2 (the GLU/ASN cavity) and checks whether each ion appears in the
existing permeation tracking records (ion_region_tracking.json).

An ion is "untracked" at a frame if:
  - it is physically inside channel2 at that frame, AND
  - it has NO entry in ion_region_tracking.json for that frame
    (meaning it was never recorded as passing through region 1 first)

This captures ions that entered the cavity from the intracellular side
or any other non-standard route.

Outputs (saved to results_dir/untracked_cavity_ions/):
  untracked_ions.csv      frame-by-frame record of every untracked ion
                          columns: frame, ion_resid, x, y, z
  summary.csv             per-ion summary: first_frame, last_frame,
                          n_frames_in_cavity, % of simulation
  untracked_count.png     plot: number of untracked ions in ch2 per frame
                          overlaid with total ions in ch2 per frame

Called from main.py:
    from analysis.untracked_cavity_ions import run_untracked_cavity_ions
    run_untracked_cavity_ions(
        u           = u,
        channel2    = ch2,
        results_dir = results_dir,
        start_frame = start_frame,
        end_frame   = end_frame,
    )
"""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_untracked_cavity_ions(
    u,
    channel2,
    results_dir,
    start_frame: int = 0,
    end_frame:   int = None,
    frame_duration: float = 0.075,
):
    """
    Main entry point — call this from main.py.

    Parameters
    ----------
    u              : MDAnalysis Universe
    channel2       : Channel object for region 2 (GLU/ASN cavity)
    results_dir    : Path to RUN-level results directory
    start_frame    : int, first frame to analyse (default 0)
    end_frame      : int, last frame inclusive (default: last frame)
    frame_duration : float, ns per frame (default 0.075)
    """
    results_dir = Path(results_dir)
    out_dir     = results_dir / "untracked_cavity_ions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    print(f"\n  Untracked cavity ion monitor")
    print(f"  Frames        : {start_frame} – {end_frame}")
    print(f"  Output dir    : {out_dir}")

    # ── skip if already computed ──────────────────────────────────────────
    untracked_csv_path = out_dir / "untracked_ions.csv"
    traj_counts_path   = out_dir / "trajectory_ion_counts.csv"

    if untracked_csv_path.exists() and traj_counts_path.exists():
        print(f"  untracked_ions.csv and trajectory_ion_counts.csv already exist — skipping.")
        print(f"  Delete files in {out_dir} to recompute.")
        return

    # ── if only trajectory_ion_counts.csv is missing, rebuild it from files ──
    if untracked_csv_path.exists() and not traj_counts_path.exists():
        print(f"  untracked_ions.csv exists but trajectory_ion_counts.csv is missing.")
        print(f"  Rebuilding trajectory_ion_counts.csv from JSON records (no trajectory scan).")

        tracking_path = results_dir / "ion_region_tracking.json"
        if not tracking_path.exists():
            print(f"  ERROR: ion_region_tracking.json not found — cannot rebuild.")
        else:
            with open(tracking_path) as f:
                raw = json.load(f)

            # build tracked-in-ch2 count per frame
            tracked_in_ch2_by_frame = {}
            for resid_str, frame_dict in raw.items():
                for fr_str, region in frame_dict.items():
                    if region == 2:
                        fr = int(fr_str)
                        if start_frame <= fr <= end_frame:
                            tracked_in_ch2_by_frame.setdefault(fr, set()).add(int(resid_str))

            # load untracked counts from existing CSV
            untracked_by_frame = {}
            with open(untracked_csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    fr = int(float(row["frame"]))
                    untracked_by_frame.setdefault(fr, set()).add(int(float(row["ion_resid"])))

            # write trajectory_ion_counts.csv
            with open(traj_counts_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "t_ns", "n_total", "n_untracked"])
                w.writeheader()
                for fr in range(start_frame, end_frame + 1):
                    n_tracked   = len(tracked_in_ch2_by_frame.get(fr, set()))
                    n_untracked = len(untracked_by_frame.get(fr, set()))
                    w.writerow({
                        "frame":       fr,
                        "t_ns":        round(fr * frame_duration, 4),
                        "n_total":     n_tracked + n_untracked,
                        "n_untracked": n_untracked,
                    })
            print(f"  Saved: {traj_counts_path}")
        return

    # ── load existing tracking records ────────────────────────────────────
    tracking_path = results_dir / "ion_region_tracking.json"
    if not tracking_path.exists():
        print(f"  WARNING: ion_region_tracking.json not found at {tracking_path}")
        print(f"  All cavity ions will be marked as untracked.")
        tracked_ions = {}   # empty — everything is untracked
    else:
        with open(tracking_path) as f:
            raw = json.load(f)
        # convert to { ion_resid(int): set_of_frames(int) }
        tracked_ions = {
            int(resid): set(int(fr) for fr in frames.keys())
            for resid, frames in raw.items()
        }
        print(f"  Loaded tracking records for {len(tracked_ions)} ions")

    # ── build per-frame tracked count from JSON (no trajectory scan needed) ─
    # tracked_in_ch2_by_frame[frame] = set of resids tracked as region 2
    tracked_in_ch2_by_frame = {}
    if tracking_path.exists():
        for resid, frame_dict in raw.items():
            resid_int = int(resid)
            for fr_str, region in frame_dict.items():
                if region == 2:
                    fr = int(fr_str)
                    if start_frame <= fr <= end_frame:
                        tracked_in_ch2_by_frame.setdefault(fr, set()).add(resid_int)
        print(f"  Built tracked-in-ch2 index for {len(tracked_in_ch2_by_frame)} frames")

    # ── frame-by-frame scan — ONLY to find untracked ions ─────────────────
    # We no longer use the cylinder to count n_total — that comes from the JSON.
    all_ions_sel = u.select_atoms("resname K+ K")

    frames_axis      = []
    total_in_ch2     = []   # tracked (from JSON) + untracked (from cylinder)
    untracked_in_ch2 = []   # untracked ions only

    untracked_records = []  # {frame, ion_resid, x, y, z}

    for ts in tqdm(u.trajectory[start_frame:end_frame + 1],
                   total=end_frame - start_frame + 1,
                   desc="  scanning for untracked ions"):
        frame = ts.frame
        channel2.compute_geometry(2)

        # tracked count for this frame — from JSON, no cylinder needed
        n_tracked = len(tracked_in_ch2_by_frame.get(frame, set()))

        # find ions physically in ch2 that are NOT in tracking records
        n_untracked = 0
        for ion in all_ions_sel:
            if not channel2.is_within_cylinder(ion.position):
                continue
            resid = int(ion.resid)
            if resid in tracked_ions and frame in tracked_ions[resid]:
                continue   # already tracked — skip
            # untracked ion found
            n_untracked += 1
            pos = ion.position.copy()
            untracked_records.append({
                "frame":     frame,
                "ion_resid": resid,
                "x":         round(float(pos[0]), 3),
                "y":         round(float(pos[1]), 3),
                "z":         round(float(pos[2]), 3),
            })

        frames_axis.append(frame)
        untracked_in_ch2.append(n_untracked)
        total_in_ch2.append(n_tracked + n_untracked)

    # ── per-ion summary ───────────────────────────────────────────────────
    ion_summary = {}   # resid -> {frames, first, last}
    for rec in untracked_records:
        r = rec["ion_resid"]
        f = rec["frame"]
        if r not in ion_summary:
            ion_summary[r] = {"frames": [], "first": f, "last": f}
        ion_summary[r]["frames"].append(f)
        ion_summary[r]["first"] = min(ion_summary[r]["first"], f)
        ion_summary[r]["last"]  = max(ion_summary[r]["last"],  f)

    total_frames = end_frame - start_frame + 1
    n_untracked_ions = len(ion_summary)
    total_untracked_observations = len(untracked_records)

    print(f"\n  Results:")
    print(f"    Unique untracked ions found in ch2 : {n_untracked_ions}")
    print(f"    Total frame-observations           : {total_untracked_observations}")
    if n_untracked_ions > 0:
        print(f"    Untracked ions (resid, n_frames):")
        for resid, info in sorted(ion_summary.items(),
                                  key=lambda x: len(x[1]["frames"]), reverse=True):
            n  = len(info["frames"])
            pct = 100.0 * n / total_frames
            ns  = n * frame_duration
            print(f"      resid {resid:>6}  {n:5d} frames  ({pct:.1f}%)  "
                  f"{ns:.1f} ns  frames {info['first']}–{info['last']}")

    # ── save untracked_ions.csv ───────────────────────────────────────────
    csv_path = out_dir / "untracked_ions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "ion_resid", "x", "y", "z"])
        w.writeheader()
        w.writerows(untracked_records)
    print(f"\n    Saved: {csv_path}")

    # ── save trajectory_ion_counts.csv ────────────────────────────────────
    # This file lets cavity_occupancy_analysis skip re-scanning the trajectory
    traj_csv_path = out_dir / "trajectory_ion_counts.csv"
    with open(traj_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_ns", "n_total", "n_untracked"])
        w.writeheader()
        for fr, n_tot, n_untr in zip(frames_axis, total_in_ch2, untracked_in_ch2):
            w.writerow({
                "frame":       fr,
                "t_ns":        round(fr * frame_duration, 4),
                "n_total":     n_tot,
                "n_untracked": n_untr,
            })
    print(f"    Saved: {traj_csv_path}")

    # ── save summary.csv ──────────────────────────────────────────────────
    sum_path = out_dir / "summary.csv"
    with open(sum_path, "w", newline="") as f:
        fields = ["ion_resid", "n_frames", "pct_simulation",
                  "time_ns", "first_frame", "last_frame"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for resid, info in sorted(ion_summary.items(),
                                  key=lambda x: len(x[1]["frames"]), reverse=True):
            n = len(info["frames"])
            w.writerow({
                "ion_resid":       resid,
                "n_frames":        n,
                "pct_simulation":  round(100.0 * n / total_frames, 3),
                "time_ns":         round(n * frame_duration, 3),
                "first_frame":     info["first"],
                "last_frame":      info["last"],
            })
    print(f"    Saved: {sum_path}")

    # ── plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    t = np.array(frames_axis) * frame_duration
    ax.fill_between(t, total_in_ch2, alpha=0.25, color="#4878CF",
                    label="Total ions in ch2")
    ax.plot(t, total_in_ch2, color="#4878CF", linewidth=0.8, alpha=0.6)
    ax.fill_between(t, untracked_in_ch2, alpha=0.45, color="#C0392B",
                    label=f"Untracked ions in ch2  (n unique = {n_untracked_ions})")
    ax.plot(t, untracked_in_ch2, color="#C0392B", linewidth=1.0)

    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Number of K⁺ ions", fontsize=14)
    ax.set_title(
        "K⁺ ions in GLU/ASN cavity (ch2) per frame\n"
        "Red = untracked ions (entered via non-standard route)",
        fontsize=14, pad=12)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plot_path = out_dir / "untracked_count.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {plot_path}")
    print(f"\n  Done. All outputs in: {out_dir}")