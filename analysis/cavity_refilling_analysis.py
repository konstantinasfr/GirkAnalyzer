"""
cavity_refilling_analysis.py

Analyzes whether a new ion enters the GLU/ASN cavity (region 2) while the
permeating ion is traveling from end_2 to end_3 (GLU/ASN → HBC gate).

For each permeation event:
  - Window: end_2 → end_3
  - Checks if any OTHER ion transitions INTO region 2 during this window
  - Records the frame, how many frames after end_2 it happened, and what
    fraction of the permeation journey had elapsed (relative position %)

Outputs (saved to results_dir/cavity_refilling/):
  cavity_refilling_results.csv   — per-event data
  plots:
    yes_no_count.png             — bar: how many events had a new ion enter
    yes_no_pct.png               — same as %
    frames_after_end2.png        — histogram of raw frame delay (yes events only)
    relative_position_pct.png    — histogram of relative position % (yes events)

Call from main.py:
    from analysis.cavity_refilling_analysis import run_cavity_refilling_analysis
    run_cavity_refilling_analysis(
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        frame_duration    = 0.075,
    )
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


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


def analyze_cavity_refilling(permeation_events, results_dir):
    """
    For each permeation event, check whether a NEW ion enters region 2
    (GLU/ASN cavity) between end_2 and end_3.

    A "new ion" means an ion that was NOT in region 2 at frame end_2 but
    transitions TO region 2 at some frame f where end_2 < f <= end_3.
    The permeating ion itself is excluded.

    Parameters
    ----------
    permeation_events : list of dicts with keys ion_id, end_2, end_3
    results_dir       : Path — where ion_region_tracking.json lives

    Returns
    -------
    list of dicts, one per event
    """
    results_dir = Path(results_dir)
    tracking = _load_tracking(results_dir)

    # Pre-build per-frame region lookup: {ion_id: {frame: region}}
    # Only keep frames relevant to the windows we need
    min_frame = min(int(e["end_2"]) for e in permeation_events)
    max_frame = max(int(e["end_3"]) for e in permeation_events)

    ion_frames = {}  # {ion_id(int): sorted list of (frame, region)}
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
        perm_ion   = int(event["ion_id"])
        end_2      = int(event["end_2"])
        end_3      = int(event["end_3"])
        window_len = end_3 - end_2

        if window_len <= 0:
            # end_2 == end_3 — no window to search
            results.append({
                "ion_id":            perm_ion,
                "r2_start":          int(event["end_1"]) if "end_1" in event else (int(event["R1_end"]) if "R1_end" in event else ""),
                "end_2":             end_2,
                "end_3":             end_3,
                "window_length":     window_len,
                "new_ion_entered":   False,
                "entering_ion_id":   None,
                "entry_frame":       None,
                "frames_after_end2": None,
                "relative_position": None,
            })
            continue

        # For each other ion, check if it enters region 2 during (end_2, end_3]
        found = False
        earliest_entry_frame = None
        earliest_ion = None

        for ion_id, frames in ion_frames.items():
            if ion_id == perm_ion:
                continue

            # Get this ion's region at end_2 (or None if not tracked then)
            region_at_end2 = None
            for fr, reg in frames:
                if fr <= end_2:
                    region_at_end2 = reg
                elif fr > end_2:
                    break

            # Scan frames after end_2 up to end_3
            for fr, reg in frames:
                if fr <= end_2:
                    continue
                if fr > end_3:
                    break
                # Transition INTO region 2
                if reg == 2 and region_at_end2 != 2:
                    # New ion entered region 2
                    if earliest_entry_frame is None or fr < earliest_entry_frame:
                        earliest_entry_frame = fr
                        earliest_ion = ion_id
                    found = True
                    break  # earliest entry for this ion found
                region_at_end2 = reg  # update as we step through

        if found and earliest_entry_frame is not None:
            frames_after = earliest_entry_frame - end_2
            rel_pos = (frames_after / window_len) * 100.0
        else:
            frames_after = None
            rel_pos = None

        results.append({
            "ion_id":            perm_ion,
            "r2_start":          int(event["end_1"]) if "end_1" in event else (int(event["R1_end"]) if "R1_end" in event else ""),
            "end_2":             end_2,
            "end_3":             end_3,
            "window_length":     window_len,
            "new_ion_entered":   found,
            "entering_ion_id":   earliest_ion,
            "entry_frame":       earliest_entry_frame,
            "frames_after_end2": frames_after,
            "relative_position": rel_pos,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results, output_dir, frame_duration=0.075, end_frame=None):
    """Save per-event results to CSV, plus a flux summary CSV."""
    out = output_dir / "cavity_refilling_results.csv"
    fieldnames = [
        "ion_id", "r1_end", "end_2", "end_3", "window_length", "window_ns",
        "new_ion_entered", "entering_ion_id",
        "entry_frame", "frames_after_end2", "time_after_end2_ns",
        "relative_position",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "ion_id":            r["ion_id"],
                "r1_end":            r.get("r2_start", ""),
                "end_2":             r["end_2"],
                "end_3":             r["end_3"],
                "window_length":     r["window_length"],
                "window_ns":         round(r["window_length"] * frame_duration, 4),
                "new_ion_entered":   r["new_ion_entered"],
                "entering_ion_id":   r["entering_ion_id"] if r["entering_ion_id"] else "",
                "entry_frame":       r["entry_frame"] if r["entry_frame"] is not None else "",
                "frames_after_end2": r["frames_after_end2"] if r["frames_after_end2"] is not None else "",
                "time_after_end2_ns": round(r["frames_after_end2"] * frame_duration, 4)
                                      if r["frames_after_end2"] is not None else "",
                "relative_position": round(r["relative_position"], 2)
                                     if r["relative_position"] is not None else "",
            })
    print(f"  Saved: {out}")

    # ── Per-event transit times ──────────────────────────────────────────
    # Add derived columns to the CSV: r2_transit, r3_transit, total_transit
    transit_rows = []
    for r in results:
        row = dict(r)
        try:
            r2_start = int(r["r2_start"]) if str(r.get("r2_start","")).strip() != "" else (int(r["r1_end"]) if str(r.get("r1_end","")).strip() != "" else None)
            end_2    = int(r["end_2"])
            end_3    = int(r["end_3"])
            row["r2_transit_frames"] = end_2 - r2_start if r2_start is not None else ""
            row["r3_transit_frames"] = end_3 - end_2
            row["total_transit_frames"] = end_3 - r2_start if r2_start is not None else ""
            row["r2_transit_ns"]    = round((end_2 - r2_start) * frame_duration, 4) if r2_start is not None else ""
            row["r3_transit_ns"]    = round((end_3 - end_2) * frame_duration, 4)
            row["total_transit_ns"] = round((end_3 - r2_start) * frame_duration, 4) if r2_start is not None else ""
            # percentages of total transit time
            total_f = end_3 - r2_start if r2_start is not None else None
            row["r2_pct"] = round((end_2 - r2_start) / total_f * 100, 2) if total_f and total_f > 0 else ""
            row["r3_pct"] = round((end_3 - end_2)    / total_f * 100, 2) if total_f and total_f > 0 else ""
        except (ValueError, TypeError):
            row["r2_transit_frames"] = ""
            row["r3_transit_frames"] = ""
            row["total_transit_frames"] = ""
            row["r2_transit_ns"]    = ""
            row["r3_transit_ns"]    = ""
            row["total_transit_ns"] = ""
        transit_rows.append(row)

    # Rewrite CSV with transit columns
    # rename r2_start key to r1_end in transit_rows
    for row in transit_rows:
        if "r2_start" in row:
            row["r1_end"] = row.pop("r2_start")
    # Build fieldnames replacing r2_start with r1_end
    base_fields = [("r1_end" if k == "r2_start" else k) for k in results[0].keys()]
    # Ensure time_after_end2_ns is carried into transit_rows
    for row in transit_rows:
        if "frames_after_end2" in row and row["frames_after_end2"] != "" and row["frames_after_end2"] is not None:
            row["time_after_end2_ns"] = round(float(row["frames_after_end2"]) * frame_duration, 4)
        else:
            row["time_after_end2_ns"] = ""
    transit_fields = base_fields + [
        "time_after_end2_ns",
        "r2_transit_frames", "r3_transit_frames", "total_transit_frames",
        "r2_transit_ns",     "r3_transit_ns",     "total_transit_ns",
        "r2_pct",            "r3_pct",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=transit_fields)
        w.writeheader()
        w.writerows(transit_rows)

    # ── Flux summary CSV ─────────────────────────────────────────────────
    if end_frame is not None:
        total_frames   = end_frame + 1
        total_time_ns  = total_frames * frame_duration
        n_events       = len(results)
        flux           = n_events / total_time_ns  # ions/ns

        r2_transits    = [r["r2_transit_frames"]    for r in transit_rows
                          if str(r["r2_transit_frames"]).strip() != ""]
        r3_transits    = [r["r3_transit_frames"]    for r in transit_rows
                          if str(r["r3_transit_frames"]).strip() != ""]
        total_transits = [r["total_transit_frames"] for r in transit_rows
                          if str(r["total_transit_frames"]).strip() != ""]

        def _stats(vals):
            if not vals:
                return "", "", "", ""
            arr = [float(v) for v in vals]
            return (round(np.mean(arr), 2), round(np.std(arr), 2),
                    round(min(arr), 2),     round(max(arr), 2))

        flux_out = output_dir / "flux_summary.csv"
        with open(flux_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "metric", "value", "unit", "std", "min", "max"])
            w.writeheader()
            w.writerow({"metric": "total_frames",      "value": total_frames,
                        "unit": "frames", "std": "", "min": "", "max": ""})
            w.writerow({"metric": "total_time",        "value": round(total_time_ns, 3),
                        "unit": "ns",     "std": "", "min": "", "max": ""})
            w.writerow({"metric": "n_permeation_events","value": n_events,
                        "unit": "events", "std": "", "min": "", "max": ""})
            w.writerow({"metric": "flux",              "value": round(flux, 6),
                        "unit": "ions/ns","std": "", "min": "", "max": ""})
            w.writerow({"metric": "mean_time_per_event","value": round(total_time_ns / n_events, 3) if n_events else "",
                        "unit": "ns/ion", "std": "", "min": "", "max": ""})

            m, s, mn, mx = _stats(r2_transits)
            w.writerow({"metric": "r2_transit (R1_end → end_2)", "value": m,
                        "unit": "frames", "std": s, "min": mn, "max": mx})
            m, s, mn, mx = _stats([float(v)*frame_duration for v in r2_transits])
            w.writerow({"metric": "r2_transit (R1_end → end_2)", "value": m,
                        "unit": "ns",     "std": s, "min": mn, "max": mx})

            m, s, mn, mx = _stats(r3_transits)
            w.writerow({"metric": "r3_transit (ASN→HBC region)", "value": m,
                        "unit": "frames", "std": s, "min": mn, "max": mx})
            m, s, mn, mx = _stats([float(v)*frame_duration for v in r3_transits])
            w.writerow({"metric": "r3_transit (ASN→HBC region)", "value": m,
                        "unit": "ns",     "std": s, "min": mn, "max": mx})

            m, s, mn, mx = _stats(total_transits)
            w.writerow({"metric": "total_transit (R1_end → HBC gate)", "value": m,
                        "unit": "frames", "std": s, "min": mn, "max": mx})
            m, s, mn, mx = _stats([float(v)*frame_duration for v in total_transits])
            w.writerow({"metric": "total_transit (R1_end → HBC gate)", "value": m,
                        "unit": "ns",     "std": s, "min": mn, "max": mx})

        print(f"  Saved: {flux_out}")
        print(f"  Flux = {flux:.6f} ions/ns  ({n_events} events / {total_time_ns:.1f} ns)")

    return out, transit_rows


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _bar_plot_simple(labels, values, colors, title, ylabel, out_path,
                     annotate_fmt="{}", channel_type=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.75,
                  edgecolor="black", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(values) * 0.02,
                annotate_fmt.format(val),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _histogram(values, title, xlabel, out_path, color, bins=20, pct=False, total=None):
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if pct:
        weights = np.ones(len(values)) / (total or len(values)) * 100
        ax.hist(values, bins=bins, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0, weights=weights)
        ax.set_ylabel("% of permeation events", fontsize=14)
    else:
        ax.hist(values, bins=bins, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.set_ylabel("Number of permeation events", fontsize=14)

    mean_val = np.mean(values)
    ax.axvline(mean_val, color="red", linewidth=2, linestyle="--",
               label=f"Mean = {mean_val:.1f}")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



def _plot_transit_histograms(results, output_dir, frame_duration, channel_type=""):
    """
    Histograms of frames spent in each cavity region for ALL permeation events.
    Uses r2_transit_frames, r3_transit_frames, total_transit_frames.
    Also prints mean and median for each.
    """
    def _get(key):
        vals = []
        for r in results:
            v = r.get(key, "")
            if str(v).strip() != "":
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        return vals

    regions = [
        ("r2_transit_frames",    "GLU/ASN cavity",           "#E67E22"),
        ("r3_transit_frames",    "ASN\u2192HBC region",     "#2980B9"),
        ("total_transit_frames", "GLU/HBC region (combined)","#8E44AD"),
    ]
    pct_regions = [
        ("r2_pct", "% of transit time in GLU/ASN cavity",   "#E67E22"),
        ("r3_pct", "% of transit time in ASN\u2192HBC region","#2980B9"),
    ]

    transit_dir = output_dir / "transit_times"
    transit_dir.mkdir(parents=True, exist_ok=True)
    output_dir = transit_dir  # redirect all saves into subdir

    print("\n" + "="*60)
    print("TRANSIT TIME STATISTICS (all permeation events)")
    print("="*60)

    for key, label, color in regions:
        vals = _get(key)
        if not vals:
            continue

        arr      = np.array(vals)
        mean_f   = np.mean(arr)
        median_f = np.median(arr)
        mean_ns  = mean_f   * frame_duration
        median_ns= median_f * frame_duration

        print(f"\n  {label}:")
        print(f"    N      = {len(arr)}")
        print(f"    Mean   = {mean_f:.1f} frames  ({mean_ns:.3f} ns)")
        print(f"    Median = {median_f:.1f} frames ({median_ns:.3f} ns)")
        print(f"    Range  = {arr.min():.0f} – {arr.max():.0f} frames")

        # frames histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(arr, bins=30, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_f,   color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_f:.1f} f ({mean_ns:.3f} ns)")
        ax.axvline(median_f, color="orange", linewidth=2, linestyle="-.",
                   label=f"Median = {median_f:.1f} f ({median_ns:.3f} ns)")
        ax.set_xlabel("Frames", fontsize=14)
        ax.set_ylabel("Number of permeation events", fontsize=14)
        ax.set_title(f"Frames spent in {label}\n(all permeation events)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key.replace("_frames", "") + "_transit.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

        # ns histogram
        vals_ns = arr * frame_duration
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals_ns, bins=30, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_ns,   color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_ns:.3f} ns")
        ax.axvline(median_ns, color="orange", linewidth=2, linestyle="-.",
                   label=f"Median = {median_ns:.3f} ns")
        ax.set_xlabel("Time (ns)", fontsize=14)
        ax.set_ylabel("Number of permeation events", fontsize=14)
        ax.set_title(f"Time spent in {label} (ns)\n(all permeation events)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key.replace("_frames", "") + "_transit_ns.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

    # ── % of total transit time ──
    for key, label, color in pct_regions:
        vals = _get(key)
        if not vals:
            continue
        arr      = np.array(vals)
        mean_pct = np.mean(arr)
        med_pct  = np.median(arr)
        std_pct  = np.std(arr)
        print(f"\n  {label}:")
        print(f"    N      = {len(arr)}")
        print(f"    Mean   = {mean_pct:.1f} ± {std_pct:.1f} %")
        print(f"    Median = {med_pct:.1f} %")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(arr, bins=20, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_pct, color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_pct:.1f} ± {std_pct:.1f} %")
        ax.axvline(med_pct,  color="orange", linewidth=2, linestyle="-.",
                   label=f"Median = {med_pct:.1f} %")
        ax.set_xlabel("% of total transit time", fontsize=14)
        ax.set_ylabel("Number of permeation events", fontsize=14)
        ax.set_title(f"{label}\n(all permeation events)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key + "_histogram.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

    print("="*60)



def plot_pct_bar(results, output_dir, channel_type=""):
    """
    Bar plot of mean % time spent in GLU/ASN cavity vs ASN→HBC region,
    with std error bars and jittered individual points — one bar per region.
    """
    r2_pcts = []
    r3_pcts = []
    for r in results:
        try:
            r2_pcts.append(float(r["r2_pct"]))
            r3_pcts.append(float(r["r3_pct"]))
        except (ValueError, TypeError, KeyError):
            pass

    if not r2_pcts:
        print("  No pct data for bar plot — skipping.")
        return

    labels  = ["GLU/ASN cavity", "ASN\u2192HBC region"]
    means   = [np.mean(r2_pcts), np.mean(r3_pcts)]
    stds    = [np.std(r2_pcts),  np.std(r3_pcts)]
    colors  = ["#E67E22", "#2980B9"]
    all_vals= [r2_pcts, r3_pcts]
    n       = len(r2_pcts)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=7, width=0.5,
           color=colors, alpha=0.80, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=2)

    # jittered individual points
    rng = np.random.default_rng(42)
    for i, (vals, col) in enumerate(zip(all_vals, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, alpha=0.25, s=14,
                   edgecolors="none", zorder=3)

    # annotate mean ± std above error bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.1f}\n\u00b1{s:.1f}",
                ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("% of total transit time", fontsize=13)
    ax.set_title(f"{channel_type} \u2014 % transit time per cavity region\n(n = {n} events)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    out = output_dir / "pct_transit_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

def plot_results(results, output_dir, channel_type="", frame_duration=0.075):
    """Generate all plots from the refilling analysis results."""
    n_total = len(results)
    n_yes   = sum(1 for r in results if r["new_ion_entered"])
    n_no    = n_total - n_yes

    # ── Yes/No bar — counts ──
    _bar_plot_simple(
        labels=["New ion entered", "No new ion"],
        values=[n_yes, n_no],
        colors=["#E74C3C", "#95A5A6"],
        title=f"{channel_type} \u2014 New ion enters GLU/ASN cavity\nwhile permeating ion passes HBC gate",
        ylabel="Number of permeation events",
        out_path=output_dir / "yes_no_count.png",
        annotate_fmt="{}",
        channel_type=channel_type,
    )

    # ── Yes/No bar — % ──
    _bar_plot_simple(
        labels=["New ion entered", "No new ion"],
        values=[round(n_yes / n_total * 100, 1), round(n_no / n_total * 100, 1)],
        colors=["#E74C3C", "#95A5A6"],
        title=f"{channel_type} \u2014 New ion enters GLU/ASN cavity\nwhile permeating ion passes HBC gate \u2014 %",
        ylabel="% of permeation events",
        out_path=output_dir / "yes_no_pct.png",
        annotate_fmt="{:.1f}%",
        channel_type=channel_type,
    )

    # ── % transit time bar plot (all events) ──
    plot_pct_bar(results, output_dir, channel_type)

    # ── For YES events only ──
    yes_results = [r for r in results if r["new_ion_entered"]]
    if not yes_results:
        print("  No 'yes' events — skipping distribution plots.")
        return

    frames_after = [r["frames_after_end2"] for r in yes_results]
    time_after   = [f * frame_duration for f in frames_after]
    rel_pos      = [r["relative_position"] for r in yes_results]
    window_lens  = [r["window_length"] for r in yes_results]

    # ── Raw frames after end_2 ──
    _histogram(
        frames_after,
        title="Frames after end_2 when new ion\nenters GLU/ASN cavity",
        xlabel="Frames after ion leaves GLU/ASN cavity",
        out_path=output_dir / "frames_until_new_ion.png",
        color="#E74C3C", bins=20,
    )
    _histogram(
        frames_after,
        title="Frames after end_2 when new ion\nenters GLU/ASN cavity — %",
        xlabel="Frames after ion leaves GLU/ASN cavity",
        out_path=output_dir / "frames_until_new_ion_pct.png",
        color="#E74C3C", bins=20, pct=True, total=n_total,
    )

    # ── Time (ns) after end_2 ──
    _histogram(
        time_after,
        title="Time after end_2 when new ion\nenters GLU/ASN cavity (ns)",
        xlabel="Time after ion leaves GLU/ASN cavity (ns)",
        out_path=output_dir / "time_until_new_ion_ns.png",
        color="#E74C3C", bins=20,
    )

    # ── Relative position (%) ──
    _histogram(
        rel_pos,
        title=f"{channel_type} \u2014 Relative entry timing of new ion\n(0% = just left GLU/ASN, 100% = ion passes HBC gate)",
        xlabel="Relative position in journey (%)",
        out_path=output_dir / "relative_entry_timing.png",
        color="#8E44AD", bins=20,
    )
    _histogram(
        rel_pos,
        title=f"{channel_type} \u2014 Relative entry timing of new ion \u2014 %\n(0% = just left GLU/ASN, 100% = ion passes HBC gate)",
        xlabel="Relative position in journey (%)",
        out_path=output_dir / "relative_entry_timing_pct.png",
        color="#8E44AD", bins=20, pct=True, total=n_total,
    )

    # ── Window length distribution (for context) ──
    _histogram(
        window_lens,
        title=f"{channel_type} \u2014 Window length (GLU/ASN exit \u2192 HBC gate)\nfor events where new ion entered",
        xlabel="Frames (GLU/ASN exit → HBC gate)",
        out_path=output_dir / "window_length.png",
        color="#2980B9", bins=20,
    )

    # ── Transit time histograms (all permeation events) ──
    # These use all events, not just yes-events
    _plot_transit_histograms(results, output_dir, frame_duration, channel_type)



def plot_flux_single_run(results, output_dir, frame_duration, end_frame, channel_type=""):
    """Bar plot showing flux for this single run."""
    if end_frame is None:
        return
    n_events      = len(results)
    total_time_ns = (end_frame + 1) * frame_duration
    flux          = n_events / total_time_ns

    fig, ax = plt.subplots(figsize=(4, 4))
    bar = ax.bar(["Ion flux"], [flux], color="#E74C3C", alpha=0.75,
                 edgecolor="black", linewidth=1.2, width=0.4)
    ax.text(bar[0].get_x() + bar[0].get_width()/2.,
            flux + flux * 0.03,
            f"{flux:.5f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    ax.set_ylabel("Flux (ions/ns)", fontsize=14)
    ax.set_title(f"Ion permeation flux\n({n_events} events / {total_time_ns:.1f} ns)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, flux * 1.35)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = output_dir / "flux.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, frame_duration=0.075):
    n_total = len(results)
    n_yes   = sum(1 for r in results if r["new_ion_entered"])
    n_no    = n_total - n_yes

    print("\n" + "="*70)
    print("CAVITY REFILLING ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total permeation events analysed: {n_total}")
    print(f"  New ion entered GLU/ASN while permeating ion passes HBC gate: {n_yes} ({n_yes/n_total*100:.1f}%)")
    print(f"  No new ion entered:                         {n_no} ({n_no/n_total*100:.1f}%)")

    yes_r = [r for r in results if r["new_ion_entered"]]
    if yes_r:
        fa = [r["frames_after_end2"] for r in yes_r]
        rp = [r["relative_position"]  for r in yes_r]
        wl = [r["window_length"]       for r in yes_r]
        print(f"\nFor events WITH a new ion (N={len(yes_r)}):")
        print(f"  Frames after end_2 until new ion entered:")
        print(f"    Mean: {np.mean(fa):.1f} ± {np.std(fa):.1f}  "
              f"  Range: {min(fa)}–{max(fa)}")
        print(f"  Time after end_2 (ns): {np.mean(fa)*frame_duration:.3f} ± "
              f"{np.std(fa)*frame_duration:.3f}")
        print(f"  Relative position (%):")
        print(f"    Mean: {np.mean(rp):.1f} ± {np.std(rp):.1f}  "
              f"  Range: {min(rp):.1f}–{max(rp):.1f}")
        print(f"  Window length (GLU/ASN exit → HBC gate, frames):")
        print(f"    Mean: {np.mean(wl):.1f} ± {np.std(wl):.1f}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cavity_refilling_analysis(permeation_events, results_dir,
                                   channel_type="", frame_duration=0.075,
                                   end_frame=None, force_rerun=False):
    """
    Run the full cavity refilling analysis and save all outputs.

    Parameters
    ----------
    permeation_events : list of dicts (from analyzer.permeation_events)
    results_dir       : Path — run results directory (contains ion_region_tracking.json)
    channel_type      : str
    frame_duration    : float, ns per frame (default 0.075)
    end_frame         : int — last frame of trajectory (for flux calculation)
    force_rerun       : bool — if True, overwrite existing results

    Call from main.py:
        from analysis.cavity_refilling_analysis import run_cavity_refilling_analysis
        run_cavity_refilling_analysis(
            permeation_events = analyzer.permeation_events,
            results_dir       = results_dir,
            channel_type      = args.channel_type,
            frame_duration    = 0.075,
        )
    """
    results_dir = Path(results_dir)
    output_dir  = results_dir / "cavity_refilling"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "cavity_refilling_results.csv"
    if out_csv.exists() and not force_rerun:
        print(f"  cavity_refilling_results.csv already exists — skipping "
              f"(use force_rerun=True to overwrite).")
        return

    print("\n" + "="*70)
    print("CAVITY REFILLING ANALYSIS")
    print("="*70)
    print(f"Analysing {len(permeation_events)} permeation events...")

    results = analyze_cavity_refilling(permeation_events, results_dir)
    _, transit_rows = save_results_csv(results, output_dir, frame_duration, end_frame)
    print_summary(results, frame_duration)
    plot_results(transit_rows, output_dir, channel_type, frame_duration)
    plot_flux_single_run(results, output_dir, frame_duration, end_frame, channel_type)

    print(f"\n✓ All outputs saved to: {output_dir}/")
    return results