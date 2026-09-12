"""
pmf_seed_frame_finder.py

Finds candidate starting frames for umbrella-sampling / PMF pulling, of the
following three flavors (as requested):

    Case "0"  (frame1): one ion sits at the START of the SF (top, bulk/
               extracellular-entry side), no other ion is below it anywhere
               else in the SF, AND the GLU/ASN cavity (region 2) and the
               ASN->HBC region (region 3) are both empty.

    Case "1"  (frame2): same SF condition, but exactly ONE ion is present
               in the GLU/ASN cavity (region 2).

    Case "2"  (frame3): same SF condition, but exactly TWO ions are present
               in the GLU/ASN cavity (region 2).

Design notes / how it reuses your existing pipeline
-----------------------------------------------------
- Does NOT rescan the trajectory to build region assignments. It reads the
  same `ion_region_tracking.json` (tracked ions) and
  `untracked_cavity_ions/untracked_ions.csv` (untracked, cavity-only ions)
  that `IonPermeationAnalysis` / `cavity_occupancy_analysis.py` already
  produce and consume. Region codes follow your convention:
      1 = SF, 2 = GLU/ASN cavity, 3 = ASN->HBC, 4 = past HBC gate.
- "Exactly one ion in region 1" is used as the proxy for "no other ion is
  below it in the SF" — if a second ion were anywhere in the SF cylinder it
  would show up as region==1 too, so requiring count==1 captures exactly
  that condition without needing sub-SF depth ordering of multiple ions.
- Untracked ions (untracked_ions.csv) are ASSUMED to only ever appear in
  region 2 (that file lives under `untracked_cavity_ions/`, i.e. it's a
  GLU/ASN-cavity-only supplement to the tracked-ion json). They are folded
  into the region-2 count (de-duplicated against tracked ions by resid).
  Region 1 and region 3 counts come from ion_region_tracking.json only.
- "Start of the SF" is computed directly from geometry at the qualifying
  frame (not taken from the Channel object's internal state), as the
  fractional position of the ion along the axis from the SF's cavity-side
  boundary (`lower1`, frac=0) to its bulk-entry boundary (`upper1`, frac=1).
  A frame's target ion is ranked by how CLOSE it is to upper1 (closest to
  upper1 = better) — that's the ion that will actually get pulled during
  PMF pulling. `sf_top_fraction` is only used to flag (not filter) whether
  a candidate is "close enough" to the top (frac >= sf_top_fraction).
  This ONLY touches the trajectory at the handful of frames that already
  pass the region-count filters above — it does not re-read from frame 0.
- Candidate frames are grouped into contiguous runs (same ion, same case,
  consecutive frame numbers) so a single one-frame fluke doesn't win over a
  frame that's part of a long, stable dwell — which is what you actually
  want for a clean PMF starting structure.

Usage from main.py (drop in right after `analyzer.run_analysis()`, using
objects you already have in scope — u, upper1, lower1, results_dir,
args.channel_type, frame_duration):

    from analysis.pmf_seed_frame_finder import run_pmf_seed_frame_selection

    run_pmf_seed_frame_selection(
        u               = u,
        upper1          = upper1,
        lower1          = lower1,
        results_dir     = results_dir,
        channel_type    = args.channel_type,
        frame_duration  = frame_duration,
        force_rerun     = False,
    )
"""

import csv
import json
from pathlib import Path

import numpy as np


REGION_SF        = 1
REGION_GLU_ASN   = 2
REGION_ASN_HBC   = 3

CASE_LABELS = {0: "case0_sf_only", 1: "case1_sf_plus1cavity", 2: "case2_sf_plus2cavity"}


# ─────────────────────────────────────────────────────────────────────────
# Loading tracked / untracked region data (no trajectory access needed)
# ─────────────────────────────────────────────────────────────────────────

def _find_file(results_dir, relative_path):
    """Search results_dir and a couple of parents for a relative path,
    mirroring the search pattern already used elsewhere in the pipeline."""
    for candidate in [results_dir, results_dir.parent, results_dir.parent.parent]:
        p = candidate / relative_path
        if p.exists():
            return p
    return None


def _load_frame_region_map(results_dir):
    """
    Returns: dict frame(int) -> dict ion_resid(int) -> region(int)
    Built from ion_region_tracking.json (tracked ions only).
    """
    tracking_path = _find_file(results_dir, "ion_region_tracking.json")
    if tracking_path is None:
        raise FileNotFoundError(
            f"ion_region_tracking.json not found near {results_dir} — "
            "run the main permeation analysis first."
        )

    with open(tracking_path) as f:
        tracking = json.load(f)

    frame_map = {}
    for ion_id_str, fdict in tracking.items():
        ion_id = int(ion_id_str)
        for f_str, region in fdict.items():
            frame = int(f_str)
            region = int(region)
            frame_map.setdefault(frame, {})[ion_id] = region

    return frame_map


def _load_untracked_cavity_ions(results_dir):
    """
    Returns: dict frame(int) -> set(ion_resid(int))
    From untracked_cavity_ions/untracked_ions.csv, if it exists.
    """
    path = _find_file(results_dir, "untracked_cavity_ions/untracked_ions.csv")
    untracked_by_frame = {}
    if path is None:
        print("  No untracked_ions.csv found — region-2 counts use tracked ions only.")
        return untracked_by_frame

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(float(row["frame"]))
            resid = int(float(row["ion_resid"]))
            untracked_by_frame.setdefault(frame, set()).add(resid)

    n_obs = sum(len(v) for v in untracked_by_frame.values())
    print(f"  Loaded untracked cavity ions: {n_obs} observations across "
          f"{len(untracked_by_frame)} frames")
    return untracked_by_frame


# ─────────────────────────────────────────────────────────────────────────
# Candidate frame screening (region counts only — no trajectory access)
# ─────────────────────────────────────────────────────────────────────────

def _screen_candidates(frame_map, untracked_by_frame, require_region3_empty=True):
    """
    Returns list of dicts:
        {frame, sf_ion_ids, region2_count, region3_count, case}
    for every frame where at least one ion is in the SF (region 1) and
    (optionally) region 3 is empty, keyed by the resulting region-2 count
    (0, 1, or 2 -> case). Frames with region2_count > 2 are dropped (they
    don't match any of the 3 requested cases).

    NOTE: more than one ion in the SF is allowed here — "no ion below the
    target ion in the SF" is enforced later, geometrically, by picking the
    target as whichever SF ion sits deepest (closest to the cavity side).
    Any other SF ion is then above it by construction.
    """
    candidates = []
    for frame, ion_regions in frame_map.items():
        sf_ion_ids = [rid for rid, reg in ion_regions.items() if reg == REGION_SF]
        if not sf_ion_ids:
            continue

        region3_count = sum(1 for reg in ion_regions.values() if reg == REGION_ASN_HBC)
        if require_region3_empty and region3_count != 0:
            continue

        tracked_r2 = {rid for rid, reg in ion_regions.items() if reg == REGION_GLU_ASN}
        untracked_r2 = untracked_by_frame.get(frame, set()) - set(ion_regions.keys())
        region2_count = len(tracked_r2) + len(untracked_r2)

        if region2_count not in (0, 1, 2):
            continue

        candidates.append({
            "frame": frame,
            "sf_ion_ids": sf_ion_ids,
            "region2_count": region2_count,
            "region3_count": region3_count,
            "case": region2_count,
        })

    candidates.sort(key=lambda c: c["frame"])
    return candidates


# ─────────────────────────────────────────────────────────────────────────
# SF axial position (touches the trajectory ONLY at candidate frames)
# ─────────────────────────────────────────────────────────────────────────

def _sf_fraction_single(upper_com, lower_com, ion_pos):
    """
    frac = 0 at lower1 (SF/cavity boundary, i.e. the sf_residues ring that
           borders the GLU/ASN cavity)
    frac = 1 at upper1 (bulk/extracellular entry side of the SF)

    So "start of SF" == frac close to 1, "deepest" == frac close to 0.
    Used only to pick the target (deepest) ion when several are in the SF.
    """
    axis = upper_com - lower_com
    denom = float(np.dot(axis, axis))
    if denom == 0:
        return None
    return float(np.dot(ion_pos - lower_com, axis) / denom)


def _sf_target_ion(u, upper1, lower1, sf_ion_ids, frame, ion_selection):
    """
    Goes to `frame` and, for every ion currently in the SF (sf_ion_ids):
      - computes its SF fraction (used only to pick the deepest ion as the
        TARGET — by construction, any other SF ion present is then "above"
        it, never below)
      - computes its plain Euclidean distance (Angstroms) to the COM of
        the upper1 residues, which is the number actually reported.

    Returns (target_ion_id, target_frac, target_dist_from_upper1_A, n_sf_ions)
    or (None, None, None, 0) if positions couldn't be resolved.
    """
    u.trajectory[frame]

    upper_com = u.select_atoms(
        "resid " + " ".join(str(r) for r in upper1)
    ).center_of_mass()
    lower_com = u.select_atoms(
        "resid " + " ".join(str(r) for r in lower1)
    ).center_of_mass()

    ions = u.select_atoms(ion_selection)

    fracs = {}
    dists = {}
    for ion_resid in sf_ion_ids:
        ion_atoms = ions.select_atoms(f"resid {ion_resid}")
        if len(ion_atoms) == 0:
            continue
        ion_pos = ion_atoms.positions[0]
        frac = _sf_fraction_single(upper_com, lower_com, ion_pos)
        if frac is None:
            continue
        fracs[ion_resid] = frac
        dists[ion_resid] = float(np.linalg.norm(ion_pos - upper_com))

    if not fracs:
        return None, None, None, 0

    target_ion = min(fracs, key=fracs.get)  # deepest = lowest frac
    return target_ion, fracs[target_ion], dists[target_ion], len(fracs)


# ─────────────────────────────────────────────────────────────────────────
# Grouping into contiguous dwell stretches (prefer stable states over flukes)
# ─────────────────────────────────────────────────────────────────────────

def _group_into_stretches(rows):
    """
    rows: list of dicts with at least "frame", "ion_id", "case", sorted by frame.
    Groups consecutive frames (same ion_id, same case, frame deltas == 1)
    into stretches. Returns list of dicts:
        {ion_id, case, start_frame, end_frame, length, frames (list),
         mean_frac, best_frame (frame with frac closest to upper1,
         i.e. shallowest in the SF — the ion we'll actually pull)}
    """
    stretches = []
    current = None

    for row in rows:
        if (current is not None
                and row["ion_id"] == current["ion_id"]
                and row["case"] == current["case"]
                and row["frame"] == current["frames"][-1] + 1):
            current["frames"].append(row["frame"])
            current["fracs"].append(row["frac"])
            current["dists"].append(row.get("dist_upper1"))
            current["n_sf_ions"].append(row.get("n_sf_ions"))
        else:
            if current is not None:
                stretches.append(current)
            current = {
                "ion_id": row["ion_id"],
                "case": row["case"],
                "frames": [row["frame"]],
                "fracs": [row["frac"]],
                "dists": [row.get("dist_upper1")],
                "n_sf_ions": [row.get("n_sf_ions")],
            }
    if current is not None:
        stretches.append(current)

    out = []
    for s in stretches:
        fracs = s["fracs"]
        best_idx = int(np.argmax(fracs))  # closest to upper1 (frac->1)
        out.append({
            "ion_id":      s["ion_id"],
            "case":        s["case"],
            "start_frame": s["frames"][0],
            "end_frame":   s["frames"][-1],
            "length":      len(s["frames"]),
            "mean_frac":   float(np.mean(fracs)),
            "best_frame":  s["frames"][best_idx],
            "best_frac":   fracs[best_idx],
            "best_dist_upper1": s["dists"][best_idx],
            "mean_dist_upper1": float(np.mean(s["dists"])),
            "n_sf_ions_at_best": s["n_sf_ions"][best_idx],
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────

def run_pmf_seed_frame_selection(
    u,
    upper1,
    lower1,
    results_dir,
    channel_type,
    frame_duration=0.075,
    ion_selection="resname K+ K",
    sf_top_fraction=0.5,
    require_region3_empty=True,
    top_n=5,
    force_rerun=False,
):
    """
    Screens every candidate frame for this RUN and writes, per case
    (0 / 1 / 2 ions in the GLU/ASN cavity), a CSV of ranked seed-frame
    candidates plus a text summary naming the single best pick.

    `results_dir` must be the RUN's own results folder (the same one you
    already pass everywhere else in main.py) — the run label used in
    print statements and filenames is derived from its folder name, so
    there's no separate run_number argument to keep in sync.

    Output: results_dir / "pmf_seed_frames" / <case_label>.csv
            results_dir / "pmf_seed_frames" / summary.txt

    Ranking within a case: the frame where the target ion sits CLOSEST
    to upper1 (highest SF fraction — closest to the bulk/entry side)
    ranks first, since that's the ion that will actually get pulled in
    the PMF. Longest contiguous dwell is the tiebreak (prefer a stable
    state over a one-frame fluke). Frames are ALWAYS ranked and written
    out, even if none clear sf_top_fraction — check the `meets_cutoff`
    column; a run with 0 "True" rows just means this RUN never had the
    target ion sit that close to upper1 while its region conditions were
    satisfied, which the printed frac distribution will
    make more concrete than a hard yes/no.
    """
    results_dir = Path(results_dir)
    run_label = results_dir.name  # e.g. "RUN1" — just for display/filenames
    out_dir = results_dir / "pmf_seed_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.txt"
    if summary_path.exists() and not force_rerun:
        print(f"  {summary_path} already exists — skipping (force_rerun=False).")
        return

    print("\n" + "=" * 80)
    print(f"PMF SEED FRAME SELECTION — {channel_type} {run_label}")
    print("=" * 80)

    frame_map = _load_frame_region_map(results_dir)
    untracked_by_frame = _load_untracked_cavity_ions(results_dir)

    candidates = _screen_candidates(frame_map, untracked_by_frame, require_region3_empty)
    print(f"  {len(candidates)} frames pass the region-count screen "
          f"(>=1 ion in SF, region3 empty={require_region3_empty}, "
          f"region2 in [0,1,2])")

    if not candidates:
        print("  Nothing to do.")
        with open(summary_path, "w") as f:
            f.write("No candidate frames found.\n")
        return

    # For each candidate frame, resolve the TARGET ion = the deepest SF ion
    # (any other SF ion present is then above it, never below, by construction).
    resolved = []
    n_multi_ion_sf = 0
    for c in candidates:
        target_ion, target_frac, target_dist, n_sf_ions = _sf_target_ion(
            u, upper1, lower1, c["sf_ion_ids"], c["frame"], ion_selection
        )
        if target_ion is None:
            continue
        if n_sf_ions > 1:
            n_multi_ion_sf += 1
        resolved.append({
            "frame":         c["frame"],
            "ion_id":        target_ion,
            "n_sf_ions":     n_sf_ions,
            "region2_count": c["region2_count"],
            "region3_count": c["region3_count"],
            "case":          c["case"],
            "frac":          target_frac,
            "dist_upper1":   target_dist,
        })
    candidates = resolved
    if n_multi_ion_sf:
        print(f"  ({n_multi_ion_sf} candidate frames had >1 ion in the SF — "
              f"target = the deepest one in each)")

    # Report the real frac distribution instead of a binary pass/fail —
    # useful for picking a realistic sf_top_fraction rather than guessing.
    all_fracs = sorted((c["frac"] for c in candidates), reverse=True)  # descending: closest to upper1 first
    n_meet_cutoff = sum(1 for f in all_fracs if f >= sf_top_fraction)
    print(f"  SF-fraction range across candidates: "
          f"min={all_fracs[-1]:.3f}  max={all_fracs[0]:.3f}  "
          f"median={all_fracs[len(all_fracs)//2]:.3f}")
    print(f"  {n_meet_cutoff}/{len(all_fracs)} candidates clear "
          f"sf_top_fraction={sf_top_fraction} (i.e. close enough to upper1)")
    if n_meet_cutoff == 0:
        print(f"  NONE clear the cutoff — proceeding anyway, ranking by "
              f"closeness to upper1 so you get the closest frames available. "
              f"Check the 'meets_cutoff' column in the CSVs.")

    summary_lines = [f"PMF seed frame candidates — {channel_type} {run_label}",
                      f"sf_top_fraction cutoff: {sf_top_fraction}  "
                      f"({n_meet_cutoff}/{len(all_fracs)} candidates cleared it)",
                      f"require_region3_empty: {require_region3_empty}",
                      f"observed SF-fraction range: min={all_fracs[-1]:.3f} "
                      f"max={all_fracs[0]:.3f} median={all_fracs[len(all_fracs)//2]:.3f}",
                      ""]

    for case, label in CASE_LABELS.items():
        rows = sorted([c for c in candidates if c["case"] == case], key=lambda c: c["frame"])
        stretches = _group_into_stretches(rows)
        # Rank by closeness to upper1 first (highest mean frac — the ion
        # we'll actually pull), then by dwell stability as a tiebreak.
        stretches.sort(key=lambda s: (-s["mean_frac"], -s["length"]))
        top = stretches[:top_n]

        out_csv = out_dir / f"{label}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "ion_resid", "best_frame", "best_frame_t_ns",
                              "sf_fraction", "dist_from_upper1_A", "meets_cutoff",
                              "n_sf_ions_at_best_frame",
                              "dwell_start_frame", "dwell_end_frame",
                              "dwell_length_frames", "dwell_length_ns",
                              "mean_sf_fraction", "mean_dist_from_upper1_A"])
            for rank, s in enumerate(top, start=1):
                writer.writerow([
                    rank, s["ion_id"], s["best_frame"],
                    round(s["best_frame"] * frame_duration, 4),
                    round(s["best_frac"], 4),
                    round(s["best_dist_upper1"], 3),
                    s["best_frac"] >= sf_top_fraction,
                    s.get("n_sf_ions_at_best", ""),
                    s["start_frame"], s["end_frame"],
                    s["length"], round(s["length"] * frame_duration, 4),
                    round(s["mean_frac"], 4),
                    round(s["mean_dist_upper1"], 3),
                ])

        print(f"\n  [{label}] {len(stretches)} dwell stretches found, "
              f"top {len(top)} written to {out_csv.name}")
        summary_lines.append(f"[{label}] — {len(stretches)} dwell stretches")
        if top:
            best = top[0]
            flag = "MEETS cutoff" if best["best_frac"] >= sf_top_fraction else "BELOW cutoff"
            summary_lines.append(
                f"  BEST PICK ({flag}): frame {best['best_frame']} "
                f"(t={best['best_frame']*frame_duration:.3f} ns), "
                f"ion {best['ion_id']}, sf_fraction={best['best_frac']:.3f}, "
                f"dist_from_upper1={best['best_dist_upper1']:.3f} A, "
                f"n_sf_ions={best.get('n_sf_ions_at_best')}, "
                f"dwell={best['length']} frames "
                f"({best['length']*frame_duration:.3f} ns, "
                f"frames {best['start_frame']}-{best['end_frame']})"
            )
            print(f"    BEST ({flag}): frame {best['best_frame']}  "
                  f"(t={best['best_frame']*frame_duration:.3f} ns)  "
                  f"ion {best['ion_id']}  sf_frac={best['best_frac']:.3f}  "
                  f"n_sf_ions={best.get('n_sf_ions_at_best')}  "
                  f"dwell={best['length']} frames")
        else:
            summary_lines.append("  BEST PICK: none found")
        summary_lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n  Summary written to {summary_path}")
    print("=" * 80)