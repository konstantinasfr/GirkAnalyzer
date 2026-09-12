"""
permeation_stage_times.py

For every permeation event, extracts the simulation frame/time at which the
permeating ion leaves each of the three gating stages:

    end_1 (or R1_end)  — ion leaves the selectivity filter (SF)
    end_2              — ion leaves the GLU/ASN cavity
    end_3              — ion leaves the HBC gate

Unlike cavity_refilling_analysis.py (which reports TRANSIT DURATIONS between
stages), this module reports the ABSOLUTE simulation time (ns) at which each
stage-exit happens for each ion — i.e. "when did ion X leave the SF", "when
did it leave the GLU/ASN cavity", "when did it leave HBC", across the whole
trajectory.

Outputs (saved to results_dir/permeation_stage_times/):
  stage_exit_times.csv        — per-ion frame + time (ns) for each stage
  sf_exit_time_ns.png         — histogram: SF-exit times across all events
  sf_exit_time_frames.png
  glu_asn_exit_time_ns.png    — histogram: GLU/ASN-exit times across all events
  glu_asn_exit_time_frames.png
  hbc_exit_time_ns.png        — histogram: HBC-exit times across all events
  hbc_exit_time_frames.png
  stage_exit_times_bar.png    — mean ± std exit time per stage, one bar chart

Call from main.py:
    from analysis.permeation_stage_times import run_permeation_stage_times_analysis
    run_permeation_stage_times_analysis(
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        frame_duration    = frame_duration,
    )
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_end1_frame(event):
    """end_1 is stored as 'end_1' in newer runs, 'R1_end' in older ones."""
    if "end_1" in event:
        return int(event["end_1"])
    if "R1_end" in event:
        return int(event["R1_end"])
    return None


def extract_stage_times(permeation_events, frame_duration=0.075):
    """
    Build one row per permeation event with the frame + time (ns) at which
    the ion leaves the SF, the GLU/ASN cavity, and the HBC gate.

    Parameters
    ----------
    permeation_events : list of dicts (from analyzer.permeation_events)
    frame_duration    : float, ns per frame

    Returns
    -------
    list of dicts
    """
    rows = []
    for event in permeation_events:
        ion_id     = int(event["ion_id"])
        end_1      = _get_end1_frame(event)
        end_2      = int(event["end_2"]) if "end_2" in event else None
        end_3      = int(event["end_3"]) if "end_3" in event else None

        rows.append({
            "ion_id":       ion_id,
            "end_1_frame":  end_1,
            "end_1_ns":     round(end_1 * frame_duration, 4) if end_1 is not None else "",
            "end_2_frame":  end_2,
            "end_2_ns":     round(end_2 * frame_duration, 4) if end_2 is not None else "",
            "end_3_frame":  end_3,
            "end_3_ns":     round(end_3 * frame_duration, 4) if end_3 is not None else "",
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_stage_times_csv(rows, output_dir):
    out = output_dir / "stage_exit_times.csv"
    fieldnames = ["ion_id",
                  "end_1_frame", "end_1_ns",
                  "end_2_frame", "end_2_ns",
                  "end_3_frame", "end_3_ns"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _get_vals(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "")
        if str(v).strip() != "":
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return vals


def _histogram(values, title, xlabel, out_path, color, bins=20):
    if not values:
        print(f"  No data for '{out_path.name}' — skipping.")
        return
    arr      = np.array(values)
    mean_v   = np.mean(arr)
    median_v = np.median(arr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=bins, color=color, alpha=0.75,
            edgecolor="black", linewidth=1.0)
    ax.axvline(mean_v,   color="red",    linewidth=2, linestyle="--",
               label=f"Mean = {mean_v:.2f}")
    ax.axvline(median_v, color="orange", linewidth=2, linestyle="-.",
               label=f"Median = {median_v:.2f}")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Number of permeation events", fontsize=14)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_stage_time_histograms(rows, output_dir, channel_type=""):
    """
    Histograms of the absolute simulation time (frames + ns) at which ions
    leave each of the three stages, across all permeation events.
    """
    stages = [
        ("end_1", "SF (selectivity filter)",  "#27AE60"),
        ("end_2", "GLU/ASN cavity",            "#E67E22"),
        ("end_3", "HBC gate",                  "#2980B9"),
    ]

    for key, label, color in stages:
        frames = _get_vals(rows, f"{key}_frame")
        ns_vals = _get_vals(rows, f"{key}_ns")

        fname_prefix = {"end_1": "sf", "end_2": "glu_asn", "end_3": "hbc"}[key]

        _histogram(
            frames,
            title=f"{channel_type} \u2014 Time ion leaves {label}\n(all permeation events)",
            xlabel="Frame",
            out_path=output_dir / f"{fname_prefix}_exit_time_frames.png",
            color=color,
        )
        _histogram(
            ns_vals,
            title=f"{channel_type} \u2014 Time ion leaves {label} (ns)\n(all permeation events)",
            xlabel="Time (ns)",
            out_path=output_dir / f"{fname_prefix}_exit_time_ns.png",
            color=color,
        )


def plot_stage_times_bar_summary(rows, output_dir, channel_type=""):
    """
    Bar chart: mean +/- std exit time (ns) for each of the 3 stages, with
    jittered individual event points overlaid — lets you see at a glance how
    the average SF / GLU-ASN / HBC exit times compare across the whole run.
    """
    stages = [
        ("end_1_ns", "SF"),
        ("end_2_ns", "GLU/ASN"),
        ("end_3_ns", "HBC"),
    ]
    colors = ["#27AE60", "#E67E22", "#2980B9"]

    all_vals = [_get_vals(rows, key) for key, _ in stages]
    labels   = [label for _, label in stages]

    if not any(all_vals):
        print("  No exit-time data for bar summary — skipping.")
        return

    means = [np.mean(v) if v else 0 for v in all_vals]
    stds  = [np.std(v)  if v else 0 for v in all_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=7, width=0.5,
           color=colors, alpha=0.80, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=2)

    rng = np.random.default_rng(42)
    for i, (vals, col) in enumerate(zip(all_vals, colors)):
        if not vals:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, alpha=0.25, s=14,
                   edgecolors="none", zorder=3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(means) * 0.03, f"{m:.2f}\n\u00b1{s:.2f}",
                ha="center", va="bottom", fontsize=11)

    n = len(rows)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Simulation time (ns)", fontsize=13)
    ax.set_title(f"{channel_type} \u2014 Mean time ion leaves each stage\n(n = {n} events)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25 if any(means) else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    out = output_dir / "stage_exit_times_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows):
    print("\n" + "="*70)
    print("PERMEATION STAGE EXIT TIMES SUMMARY")
    print("="*70)
    print(f"Total permeation events analysed: {len(rows)}")

    stages = [
        ("end_1_ns", "SF (selectivity filter)"),
        ("end_2_ns", "GLU/ASN cavity"),
        ("end_3_ns", "HBC gate"),
    ]
    for key, label in stages:
        vals = _get_vals(rows, key)
        if not vals:
            print(f"\n  {label}: no data")
            continue
        arr = np.array(vals)
        print(f"\n  Time ion leaves {label}:")
        print(f"    N      = {len(arr)}")
        print(f"    Mean   = {np.mean(arr):.3f} ns  (+/- {np.std(arr):.3f})")
        print(f"    Median = {np.median(arr):.3f} ns")
        print(f"    Range  = {arr.min():.3f} - {arr.max():.3f} ns")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_permeation_stage_times_analysis(permeation_events, results_dir,
                                         channel_type="", frame_duration=0.075,
                                         force_rerun=False):
    """
    Run the full stage-exit-times analysis and save all outputs.

    Parameters
    ----------
    permeation_events : list of dicts (from analyzer.permeation_events)
    results_dir       : Path — run results directory
    channel_type      : str
    frame_duration    : float, ns per frame (default 0.075)
    force_rerun       : bool — if True, overwrite existing results

    Call from main.py:
        from analysis.permeation_stage_times import run_permeation_stage_times_analysis
        run_permeation_stage_times_analysis(
            permeation_events = analyzer.permeation_events,
            results_dir       = results_dir,
            channel_type      = args.channel_type,
            frame_duration    = frame_duration,
        )
    """
    results_dir = Path(results_dir)
    output_dir  = results_dir / "permeation_stage_times"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "stage_exit_times.csv"
    if out_csv.exists() and not force_rerun:
        print(f"  stage_exit_times.csv already exists — skipping "
              f"(use force_rerun=True to overwrite).")
        return

    print("\n" + "="*70)
    print("PERMEATION STAGE EXIT TIMES ANALYSIS")
    print("="*70)
    print(f"Analysing {len(permeation_events)} permeation events...")

    rows = extract_stage_times(permeation_events, frame_duration)
    save_stage_times_csv(rows, output_dir)
    print_summary(rows)
    plot_stage_time_histograms(rows, output_dir, channel_type)
    plot_stage_times_bar_summary(rows, output_dir, channel_type)

    print(f"\n✓ All outputs saved to: {output_dir}/")
    return rows