"""
residence_time_stratified_analysis.py

For each permeation event, computes the total residence time from SF exit
to HBC gate exit (end_1 -> end_3), and stratifies it by the number of ions
present anywhere between the SF and the HBC gate (region 2: GLU/ASN
cavity, and region 3: ASN->HBC region) at the moment the permeating ion
exits the HBC gate (end_3).

Definitions (matching cavity_refilling_analysis.py conventions):
  - end_1 = cavity entry frame (ion leaves SF / region 1)
            aliased as r2_start / R1_end in existing event dicts
  - end_3 = HBC gate exit frame (permeation complete)
  - residence_frames = end_3 - end_1
  - residence_ns      = residence_frames * frame_duration
  - N_ions_at_end3    = count of OTHER ions occupying region 2 OR region 3
                        (anywhere between SF and HBC gate) at frame end_3,
                        PLUS 1 for the permeating ion itself. This gives
                        the total number of ions that were part of the
                        SF-to-HBC population necessary for that permeation
                        event, even though the permeating ion has itself
                        already left regions 2/3 by end_3.

For each other ion, its region is taken as its last known region at or
before end_3 (same "step through sorted frames" approach used in
cavity_refilling_analysis.py).

Outputs (saved to results_dir/residence_time_stratified/):
  residence_time_stratified_results.csv     — per-event data
  group_summary.csv                         — mean/median/std/N per final-N group
  intermediate_occupancy_per_event.csv      — long-format: per-event, per-intermediate-N
                                               frame counts and % of transit time
  intermediate_occupancy_group_summary.csv  — per (final_N, intermediate_N): occurrence
                                               count/rate + mean/median/std % of transit
  plots:
    residence_time_boxplot.png                     — boxplot: N ions (x) vs residence time ns (y)
    intermediate_occupancy_finalN_<N>.png          — one per final N: occurrence count of
                                                      each intermediate N passed through,
                                                      annotated with mean/median % of transit

Call from main.py:
    from analysis.residence_time_stratified_analysis import run_residence_time_stratified_analysis
    run_residence_time_stratified_analysis(
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        frame_duration    = 0.075,
    )
"""

import json
import csv
import bisect
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def _load_tracking(results_dir):
    """Load ion_region_tracking.json from results_dir or parent dirs."""
    for candidate in [results_dir,
                      results_dir.parent,
                      results_dir.parent.parent]:
        p = candidate / "ion_region_tracking.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"ion_region_tracking.json not found near {results_dir}"
    )


def _get_end1(event):
    """end_1 with R1_end / r2_start fallback, matching existing conventions."""
    if "end_1" in event:
        return int(event["end_1"])
    if "R1_end" in event:
        return int(event["R1_end"])
    if "r2_start" in event:
        return int(event["r2_start"])
    raise KeyError("No end_1 / R1_end / r2_start field found in permeation event")


def _region_at_frame(frames, target_frame):
    """
    Given a sorted list of (frame, region) tuples for one ion, return the
    region at the last known frame <= target_frame. Returns None if the
    ion has no recorded region at or before target_frame.
    """
    region = None
    for fr, reg in frames:
        if fr <= target_frame:
            region = reg
        else:
            break
    return region


def _build_frame_arrays(ion_frames):
    """
    Convert {ion_id: [(frame, region), ...]} into {ion_id: (frame_array, region_array)}
    for fast bisect-based lookups (needed since the intermediate-occupancy
    analysis queries EVERY frame in a transit window, not just one).
    """
    arrays = {}
    for ion_id, frames in ion_frames.items():
        frs  = [f for f, r in frames]
        regs = [r for f, r in frames]
        arrays[ion_id] = (frs, regs)
    return arrays


def _region_at_frame_fast(frame_arrays_entry, target_frame):
    """Same semantics as _region_at_frame but O(log n) via bisect."""
    frs, regs = frame_arrays_entry
    idx = bisect.bisect_right(frs, target_frame) - 1
    if idx < 0:
        return None
    return regs[idx]


def analyze_residence_time_stratified(permeation_events, results_dir):
    """
    For each permeation event, compute:
      - residence_frames = end_3 - end_1
      - n_ions_at_end3   = number of OTHER ions in region 2 (cavity) at end_3

    Parameters
    ----------
    permeation_events : list of dicts with keys ion_id, end_1 (or R1_end), end_3
    results_dir       : Path — where ion_region_tracking.json lives

    Returns
    -------
    list of dicts, one per event
    """
    results_dir = Path(results_dir)
    tracking = _load_tracking(results_dir)

    min_frame = min(_get_end1(e) for e in permeation_events)
    max_frame = max(int(e["end_3"]) for e in permeation_events)

    # Per-ion sorted (frame, region) list, restricted to the relevant range
    ion_frames = {}
    for ion_id_str, frame_dict in tracking.items():
        ion_id = int(ion_id_str)
        frames = sorted(
            (int(f), int(r)) for f, r in frame_dict.items()
            if min_frame <= int(f) <= max_frame
        )
        if frames:
            ion_frames[ion_id] = frames

    results = []

    for event in permeation_events:
        perm_ion = int(event["ion_id"])
        end_1    = _get_end1(event)
        end_3    = int(event["end_3"])

        residence_frames = end_3 - end_1
        residence_ns     = residence_frames  # multiplied by frame_duration later

        n_ions_at_end3   = 0
        ion_ids_at_end3  = []
        for ion_id, frames in ion_frames.items():
            if ion_id == perm_ion:
                continue
            region = _region_at_frame(frames, end_3)
            if region in (2, 3):
                n_ions_at_end3 += 1
                ion_ids_at_end3.append(ion_id)
        # Include the permeating ion itself: it was necessarily part of the
        # cavity/HBC population that had to be present for this permeation
        # to happen, even though by end_3 it has already left regions 2/3.
        # Defensive check: it should be impossible for perm_ion to already
        # be in ion_ids_at_end3 (it's skipped in the loop above), but guard
        # against double-counting explicitly in case that logic ever changes.
        if perm_ion not in ion_ids_at_end3:
            n_ions_at_end3 += 1
            ion_ids_at_end3.append(perm_ion)

        # Sanity-check: what region is the permeating ion actually in at end_3?
        perm_ion_region_at_end3 = _region_at_frame(ion_frames.get(perm_ion, []), end_3)

        results.append({
            "ion_id":                  perm_ion,
            "end_1":                   end_1,
            "end_2":                   int(event["end_2"]) if "end_2" in event else "",
            "end_3":                   end_3,
            "residence_frames":        residence_frames,
            "n_ions_at_end3":          n_ions_at_end3,
            "ion_ids_at_end3":         ion_ids_at_end3,
            "perm_ion_region_at_end3": perm_ion_region_at_end3,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate occupancy analysis
#
# For each permeation event, walk EVERY frame from end_1 to end_3 (not just
# the end_3 snapshot) and count N ions in region 2/3 at each frame. This
# gives, per event, a distribution like "50% of frames were N=4, 20% N=5,
# 30% N=6" — i.e. what occupancy levels the transit actually passed through,
# and how much of the journey was spent at each. These per-event
# distributions are then grouped by the event's FINAL N (n_ions_at_end3,
# from analyze_residence_time_stratified) so you can ask, e.g.: "for the
# events that ended at N=5, how often did they pass through N=3, and what
# fraction of their transit time did N=3 typically cover?"
# ─────────────────────────────────────────────────────────────────────────────

def analyze_intermediate_occupancy(permeation_events, results_dir, final_n_results):
    """
    Parameters
    ----------
    permeation_events : list of dicts with keys ion_id, end_1 (or R1_end), end_3
    results_dir        : Path — where ion_region_tracking.json lives
    final_n_results     : list of dicts, the output of analyze_residence_time_stratified
                          (used to look up each event's final n_ions_at_end3
                          without recomputing it)

    Returns
    -------
    list of dicts, one per event:
        {
            "ion_id": ..., "end_1": ..., "end_3": ..., "final_N": ...,
            "residence_frames": ...,
            "frame_counts": {intermediate_N: n_frames, ...},
            "pct_dict":     {intermediate_N: pct_of_transit_time, ...},
        }
    """
    results_dir = Path(results_dir)
    tracking = _load_tracking(results_dir)

    min_frame = min(_get_end1(e) for e in permeation_events)
    max_frame = max(int(e["end_3"]) for e in permeation_events)

    ion_frames = {}
    for ion_id_str, frame_dict in tracking.items():
        ion_id = int(ion_id_str)
        frames = sorted(
            (int(f), int(r)) for f, r in frame_dict.items()
            if min_frame <= int(f) <= max_frame
        )
        if frames:
            ion_frames[ion_id] = frames

    frame_arrays = _build_frame_arrays(ion_frames)

    # Map (ion_id, end_3) -> final_N, from the already-computed results
    final_n_lookup = {
        (r["ion_id"], r["end_3"]): r["n_ions_at_end3"] for r in final_n_results
    }

    results = []

    for event in permeation_events:
        perm_ion = int(event["ion_id"])
        end_1    = _get_end1(event)
        end_3    = int(event["end_3"])
        residence_frames = end_3 - end_1
        # Total frames walked includes end_3 itself (the permeation frame),
        # so every event contributes at least 1 frame even when end_1==end_3.
        n_frames_walked = residence_frames + 1

        final_N = final_n_lookup.get((perm_ion, end_3))

        frame_counts = Counter()
        perm_arr = frame_arrays.get(perm_ion, ([], []))
        for f in range(end_1, end_3 + 1):
            n = 0
            for ion_id, arr in frame_arrays.items():
                if ion_id == perm_ion:
                    continue
                region = _region_at_frame_fast(arr, f)
                if region in (2, 3):
                    n += 1
            # The permeating ion itself: for every frame BEFORE end_3, count
            # it only if its real tracked region is actually 2 or 3 (it
            # should be, throughout its own transit, by construction). At
            # the exact exit frame end_3, force-include it regardless of
            # its real region — matching the same convention used for
            # final_N (which always adds +1 for the permeating ion there,
            # since by definition it has just left regions 2/3). Without
            # this, every event would be silently short by 1 ion exactly
            # at its final frame, making intermediate_N at end_3 disagree
            # with final_N by exactly 1 for every single event.
            if f < end_3:
                perm_region = _region_at_frame_fast(perm_arr, f)
                if perm_region in (2, 3):
                    n += 1
            else:  # f == end_3
                n += 1
            frame_counts[n] += 1

        pct_dict = {
            n: (count / n_frames_walked * 100.0)
            for n, count in frame_counts.items()
        }

        results.append({
            "ion_id":           perm_ion,
            "end_1":            end_1,
            "end_3":            end_3,
            "final_N":          final_N,
            "residence_frames": residence_frames,
            "frame_counts":     dict(frame_counts),
            "pct_dict":         pct_dict,
        })

    return results


def save_intermediate_occupancy_csv(results, output_dir, frame_duration=0.075):
    """Long-format CSV: one row per (event, intermediate_N) pair."""
    out = output_dir / "intermediate_occupancy_per_event.csv"
    fieldnames = [
        "ion_id", "end_1", "end_3", "final_N", "residence_frames", "residence_ns",
        "intermediate_N", "frames_at_N", "pct_of_transit",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            for n_inter in sorted(r["frame_counts"].keys()):
                w.writerow({
                    "ion_id":           r["ion_id"],
                    "end_1":            r["end_1"],
                    "end_3":            r["end_3"],
                    "final_N":          r["final_N"],
                    "residence_frames": r["residence_frames"],
                    "residence_ns":     round(r["residence_frames"] * frame_duration, 4),
                    "intermediate_N":   n_inter,
                    "frames_at_N":      r["frame_counts"][n_inter],
                    "pct_of_transit":   round(r["pct_dict"][n_inter], 3),
                })
    print(f"  Saved: {out}")
    return out


def aggregate_intermediate_by_final_group(results):
    """
    Groups per-event pct_dicts by final_N, and for every intermediate_N,
    collects the pct-of-transit values ONLY from events where that
    intermediate_N actually occurred (pct > 0) — since averaging in zeros
    from events that never passed through that N would understate how
    much of the journey it covers when it DOES occur.

    Returns
    -------
    {final_N: {intermediate_N: [pct_values, ...]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        final_N = r["final_N"]
        if final_N is None:
            continue
        for n_inter, pct in r["pct_dict"].items():
            if pct > 0:
                grouped[final_N][n_inter].append(pct)
    return grouped


def save_intermediate_group_summary_csv(grouped, results, output_dir):
    """
    One row per (final_N, intermediate_N) combination:
      occurrence_count = # events in this final_N group where intermediate_N occurred
      n_events_in_group = total # events with this final_N (denominator for occurrence rate)
      mean_pct / median_pct / std_pct = stats over pct-of-transit, among occurring events only
    """
    n_events_per_final_N = Counter(r["final_N"] for r in results if r["final_N"] is not None)

    out = output_dir / "intermediate_occupancy_group_summary.csv"
    fieldnames = [
        "final_N", "intermediate_N", "occurrence_count", "n_events_in_group",
        "occurrence_rate_pct", "mean_pct_of_transit", "median_pct_of_transit", "std_pct_of_transit",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for final_N in sorted(grouped.keys()):
            n_total = n_events_per_final_N[final_N]
            for n_inter in sorted(grouped[final_N].keys()):
                vals = np.array(grouped[final_N][n_inter])
                w.writerow({
                    "final_N":                final_N,
                    "intermediate_N":         n_inter,
                    "occurrence_count":       len(vals),
                    "n_events_in_group":      n_total,
                    "occurrence_rate_pct":    round(len(vals) / n_total * 100.0, 2) if n_total else "",
                    "mean_pct_of_transit":    round(np.mean(vals), 3),
                    "median_pct_of_transit":  round(np.median(vals), 3),
                    "std_pct_of_transit":     round(np.std(vals), 3),
                })
    print(f"  Saved: {out}")
    return out


def plot_intermediate_occupancy_barplots(grouped, output_dir, channel_type=""):
    """
    One barplot PER final_N group: x = intermediate N values (sorted),
    bar height = occurrence_count (how many events in this group passed
    through that N at all), annotated above each bar with mean/median %
    of transit time that N covered (among events where it occurred).
    """
    out_paths = []
    for final_N in sorted(grouped.keys()):
        inter_dict = grouped[final_N]
        if not inter_dict:
            continue

        n_inter_values = sorted(inter_dict.keys())
        occurrence_counts = [len(inter_dict[n]) for n in n_inter_values]

        fig, ax = plt.subplots(figsize=(6, 4.4))
        bars = ax.bar(
            [str(n) for n in n_inter_values], occurrence_counts,
            color="#8E44AD", alpha=0.75, edgecolor="black", linewidth=1.0, width=0.55,
        )

        for bar, n_inter in zip(bars, n_inter_values):
            vals = np.array(inter_dict[n_inter])
            mean_pct = np.mean(vals)
            median_pct = np.median(vals)
            ax.text(
                bar.get_x() + bar.get_width() / 2., bar.get_height() + max(occurrence_counts) * 0.02,
                f"n={len(vals)}\nmean={mean_pct:.0f}%\nmedian={median_pct:.0f}%",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Intermediate N (ions passed through during transit)", fontsize=13)
        ax.set_ylabel("Number of events that passed through this N", fontsize=12.5)
        ax.set_title(
            f"{channel_type} — Intermediate occupancy during transit\n"
            f"for events ending at N={final_N} (when ion leaves HBC gate)",
            fontsize=12.5, fontweight="bold", pad=10,
        )
        ax.set_ylim(0, max(occurrence_counts) * 1.35)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out = output_dir / f"intermediate_occupancy_finalN_{final_N}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
        out_paths.append(out)

    return out_paths




def save_results_csv(results, output_dir, frame_duration=0.075):
    out = output_dir / "residence_time_stratified_results.csv"
    fieldnames = [
        "ion_id", "end_1", "end_2", "end_3",
        "residence_frames", "residence_ns", "n_ions_at_end3",
        "ion_ids_at_end3", "perm_ion_region_at_end3",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "ion_id":                  r["ion_id"],
                "end_1":                   r["end_1"],
                "end_2":                   r["end_2"],
                "end_3":                   r["end_3"],
                "residence_frames":        r["residence_frames"],
                "residence_ns":            round(r["residence_frames"] * frame_duration, 4),
                "n_ions_at_end3":          r["n_ions_at_end3"],
                "ion_ids_at_end3":         ";".join(str(i) for i in r["ion_ids_at_end3"]),
                "perm_ion_region_at_end3": r["perm_ion_region_at_end3"],
            })
    print(f"  Saved: {out}")
    return out


def save_group_summary_csv(results, output_dir, frame_duration=0.075):
    groups = defaultdict(list)
    for r in results:
        groups[r["n_ions_at_end3"]].append(r["residence_frames"] * frame_duration)

    out = output_dir / "group_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_ions_at_end3", "N_events", "mean_ns", "median_ns", "std_ns", "min_ns", "max_ns"])
        w.writeheader()
        for n_ions in sorted(groups.keys()):
            vals = np.array(groups[n_ions])
            w.writerow({
                "n_ions_at_end3": n_ions,
                "N_events":       len(vals),
                "mean_ns":        round(np.mean(vals), 4),
                "median_ns":      round(np.median(vals), 4),
                "std_ns":         round(np.std(vals), 4),
                "min_ns":         round(np.min(vals), 4),
                "max_ns":         round(np.max(vals), 4),
            })
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot(results, output_dir, channel_type="", frame_duration=0.075):
    groups = defaultdict(list)
    for r in results:
        groups[r["n_ions_at_end3"]].append(r["residence_frames"] * frame_duration)

    n_ion_values = sorted(groups.keys())
    data = [groups[n] for n in n_ion_values]

    # x-tick labels: N ions on top line, mean/median stats below it
    labels = []
    for n in n_ion_values:
        vals = np.array(groups[n])
        labels.append(f"{n}\nmean={np.mean(vals):.1f} ns\nmedian={np.median(vals):.1f} ns")

    fig, ax = plt.subplots(figsize=(8, 6.6))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="green", linewidth=2),
        meanprops=dict(color="red", linestyle="--", linewidth=2),
        boxprops=dict(facecolor="#AED6F1", alpha=0.75, edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.5),
    )

    # annotate N (event count) above each box
    y_top = max(max(vals) for vals in data)
    for i, n in enumerate(n_ion_values):
        vals = np.array(groups[n])
        ax.text(
            i + 1, y_top * 1.05, f"N={len(vals)}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Number of ions between SF and HBC gate at permeation", fontsize=13, labelpad=45)
    ax.set_ylabel("Residence time (ns)\nSF exit → HBC gate exit", fontsize=13, labelpad=10)
    ax.set_title(f"{channel_type} — Residence time vs. cavity ion occupancy at permeation",
                 fontsize=14, fontweight="bold", pad=30)
    ax.set_ylim(0, y_top * 1.25)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color="green", linewidth=2, label="Median"),
        plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Mean"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=10, frameon=True)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.85, bottom=0.28)
    out = output_dir / "residence_time_boxplot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, frame_duration=0.075):
    groups = defaultdict(list)
    for r in results:
        groups[r["n_ions_at_end3"]].append(r["residence_frames"] * frame_duration)

    print("\n" + "="*70)
    print("RESIDENCE TIME STRATIFIED BY CAVITY ION OCCUPANCY AT end_3")
    print("="*70)
    print(f"Total permeation events: {len(results)}")
    for n_ions in sorted(groups.keys()):
        vals = np.array(groups[n_ions])
        print(f"  N_ions={n_ions}: n={len(vals)}  "
              f"mean={np.mean(vals):.3f} ns  median={np.median(vals):.3f} ns  "
              f"std={np.std(vals):.3f} ns  range=[{vals.min():.3f}, {vals.max():.3f}]")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_residence_time_stratified_analysis(permeation_events, results_dir,
                                           channel_type="", frame_duration=0.075,
                                           force_rerun=False):
    """
    Run the full residence-time-stratified-by-occupancy analysis and save
    all outputs.

    Parameters
    ----------
    permeation_events : list of dicts (from analyzer.permeation_events)
    results_dir       : Path — run results directory (contains ion_region_tracking.json)
    channel_type      : str
    frame_duration    : float, ns per frame (default 0.075)
    force_rerun       : bool — if True, overwrite existing results

    Call from main.py:
        from analysis.residence_time_stratified_analysis import run_residence_time_stratified_analysis
        run_residence_time_stratified_analysis(
            permeation_events = analyzer.permeation_events,
            results_dir       = results_dir,
            channel_type      = args.channel_type,
            frame_duration    = 0.075,
        )
    """
    results_dir = Path(results_dir)
    output_dir  = results_dir / "residence_time_stratified"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "residence_time_stratified_results.csv"
    if out_csv.exists() and not force_rerun:
        print(f"  residence_time_stratified_results.csv already exists — skipping "
              f"(use force_rerun=True to overwrite).")
        return

    print("\n" + "="*70)
    print("RESIDENCE TIME STRATIFIED ANALYSIS")
    print("="*70)
    print(f"Analysing {len(permeation_events)} permeation events...")

    results = analyze_residence_time_stratified(permeation_events, results_dir)
    save_results_csv(results, output_dir, frame_duration)
    save_group_summary_csv(results, output_dir, frame_duration)
    print_summary(results, frame_duration)
    plot_boxplot(results, output_dir, channel_type, frame_duration)

    # ── Intermediate occupancy: what N values did the transit pass through,
    #    and how much of the journey (%) did each one cover? ──
    print("\n" + "="*70)
    print("INTERMEDIATE OCCUPANCY ANALYSIS")
    print("="*70)
    intermediate_results = analyze_intermediate_occupancy(permeation_events, results_dir, results)
    save_intermediate_occupancy_csv(intermediate_results, output_dir, frame_duration)
    grouped = aggregate_intermediate_by_final_group(intermediate_results)
    save_intermediate_group_summary_csv(grouped, intermediate_results, output_dir)
    plot_intermediate_occupancy_barplots(grouped, output_dir, channel_type)

    print(f"\n✓ All outputs saved to: {output_dir}/")
    return results