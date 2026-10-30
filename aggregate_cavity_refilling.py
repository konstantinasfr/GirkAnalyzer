"""
aggregate_cavity_refilling.py

Aggregates cavity_refilling_results.csv from all RUN directories and
produces combined plots and a merged CSV.

The runs are aggregated four times, each into its own folder of the output directory:
ALL/ (every run) and HIGH/, TYPICAL/, LOW/ (the runs of that permeation category, from
<base>/permeation_summary/run_categories_full_permeations_total.csv written by
analyze_permeation_summary.py; --categories for another file, --no_categories for ALL/ only).

In each folder the plots come twice: with all ions (old: transit_times_old/*_old.png and
old/) and without the ions present at frame 0 and those leaving the SF before their run's
equilibration start, taken from the site chain's equilibration.csv (transit_times/ and the
folder itself). The flux per run counts all permeations.

Usage:
    python aggregate_cavity_refilling.py /path/to/G12
    python aggregate_cavity_refilling.py /path/to/G12 --runs 1 2 3
    python aggregate_cavity_refilling.py /path/to/G12 --output_dir /path/to/output
    python aggregate_cavity_refilling.py /path/to/G12 --frame_duration 0.02
"""

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from aggregate_duration_results import CATEGORIES, load_categories, run_sort_key

plt.rcParams.update({"font.size": 15, "xtick.labelsize": 16, "ytick.labelsize": 16})   # readable when shrunk


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_all_runs(base_dir, run_numbers=None):
    base_dir = Path(base_dir)

    if run_numbers:
        run_dirs = [base_dir / f"RUN{n}" for n in run_numbers]
    else:
        run_dirs = sorted((d for d in base_dir.iterdir()
                           if d.is_dir() and d.name.startswith("RUN")), key=lambda d: run_sort_key(d.name))

    all_rows = []
    all_flux_rows = []
    loaded = 0

    for run_dir in run_dirs:
        csv_file = run_dir / "cavity_refilling" / "cavity_refilling_results.csv"
        if not csv_file.exists():
            print(f"  - {run_dir.name}: cavity_refilling_results.csv not found, skipping")
            continue
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                row["run"] = run_dir.name
                all_rows.append(row)
        loaded += 1

        # load flux summary if available
        flux_csv = run_dir / "cavity_refilling" / "flux_summary.csv"
        if flux_csv.exists():
            flux_row = {"run": run_dir.name}
            with open(flux_csv, newline="") as f:
                for row in csv.DictReader(f):
                    flux_row[row["metric"]] = row["value"]
            all_flux_rows.append(flux_row)
        print(f"  ✓ {run_dir.name}: loaded")

    print(f"\nLoaded {loaded} runs, {len(all_rows)} total events")
    return all_rows, all_flux_rows


# ─────────────────────────────────────────────────────────────────────────────
# Save aggregated CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_aggregated_csv(all_rows, output_dir, name="cavity_refilling_aggregated.csv"):
    out = output_dir / name
    if not all_rows:
        return
    fieldnames = ["run"] + [k for k in all_rows[0].keys() if k != "run"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"✓ Aggregated CSV saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_flux_summary(flux_rows, output_dir):
    """
    Save aggregated flux summary across all runs.

    Aggregated flux = total permeations across all runs
                      / (total_frames_per_run × n_runs)

    This treats all runs as one long combined simulation.
    """
    if not flux_rows:
        print("  No flux summaries found — skipping.")
        return

    out = output_dir / "flux_summary_aggregated.csv"

    # Collect per-run values
    n_runs         = len(flux_rows)
    total_events   = 0
    total_time_sum = 0.0
    per_run_fluxes = []

    for r in flux_rows:
        try:
            total_events   += int(float(r.get("n_permeation_events", 0)))
            total_time_sum += float(r.get("total_time", 0))
            per_run_fluxes.append(float(r.get("flux", 0)))
        except (ValueError, TypeError):
            pass

    # Aggregated flux: total events / total simulation time across all runs
    aggregated_flux = total_events / total_time_sum if total_time_sum > 0 else 0.0

    # Write per-run table + summary rows
    with open(out, "w", newline="") as f:
        fieldnames = ["run"] + [k for k in flux_rows[0].keys() if k != "run"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flux_rows)
        # append aggregated summary
        w.writerow({
            "run":                 "AGGREGATED",
            "n_permeation_events": total_events,
            "total_time":          round(total_time_sum, 3),
            "flux":                round(aggregated_flux, 6),
        })

    print(f"✓ Flux summary saved: {out}")
    print(f"  Runs: {n_runs}")
    print(f"  Total permeation events: {total_events}")
    print(f"  Total simulation time:   {total_time_sum:.1f} ns")
    print(f"  Aggregated flux:         {aggregated_flux:.6f} ions/ns")
    print(f"    = {total_events} events / {total_time_sum:.1f} ns")

    return per_run_fluxes, aggregated_flux


def plot_flux(flux_rows, output_dir, aggregated_flux=None, channel_name=""):
    """Bar plot of flux per run + aggregated flux as horizontal line."""
    if not flux_rows:
        return

    runs   = [r["run"] for r in flux_rows]
    fluxes = []
    for r in flux_rows:
        try:
            fluxes.append(float(r.get("flux", 0)))
        except (ValueError, TypeError):
            fluxes.append(0.0)

    if not fluxes:
        return

    mean_flux = aggregated_flux if aggregated_flux is not None else np.mean(fluxes)
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.7), 5.5))
    colors = ["#E74C3C" if f > mean_flux else "#4878CF" for f in fluxes]
    bars = ax.bar(runs, fluxes, color=colors, alpha=0.75,
                  edgecolor="black", linewidth=1.0)
    for bar, val in zip(bars, fluxes):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + max(fluxes)*0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.axhline(mean_flux, color="red", linewidth=2, linestyle="--",
               label=f"Aggregated flux = {mean_flux:.4f} ions/ns")
    ax.set_ylabel("Flux (ions/ns)", fontsize=17)
    ax.set_title(f"{channel_name} — Ion permeation flux per run", fontsize=17, fontweight="bold", pad=10)
    ax.set_xticks(range(len(runs)))
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=14)
    ax.legend(fontsize=15)
    ax.set_ylim(0, max(fluxes) * 1.25)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = output_dir / "flux_per_run.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out}")


def _bar_plot(labels, values, colors, title, ylabel, out_path, fmt="{}"):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.75,
                  edgecolor="black", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(values) * 0.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def _histogram(values, title, xlabel, out_path, color,
               bins=20, pct=False, total=None):
    if not values:
        print(f"  No data for {out_path.name} — skipping")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    n_total = total or len(values)
    if pct:
        weights = np.ones(len(values)) / n_total * 100
        ax.hist(values, bins=bins, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0, weights=weights)
        ax.set_ylabel("% of permeation events", fontsize=19)
    else:
        ax.hist(values, bins=bins, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.set_ylabel("Number of permeation events", fontsize=19)

    mean_val = np.mean(values)
    ax.axvline(mean_val, color="red", linewidth=2, linestyle="--",
               label=f"Mean = {mean_val:.1f}")
    ax.set_xlabel(xlabel, fontsize=19)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")



def plot_transit_histograms(all_rows, output_dir, frame_duration=0.075, channel_name="",
                            subdir="transit_times", suffix="", note="all runs combined"):
    """
    Aggregated histograms of frames/ns spent in each cavity region
    across all runs combined.

    Parameters
    ----------
    frame_duration : float
        Time (in ns) represented by a single trajectory frame. Used to
        convert frame counts into nanoseconds for the "_transit_ns.png"
        plots. Pass the value matching how your trajectories were saved
        (e.g. --frame_duration 0.075 for a frame saved every 0.075 ns).
    subdir, suffix, note : output subfolder, ending of the file names, and the
        second title line (which ions the histograms hold).
    """
    regions = [
        ("r2_transit_frames",    "GLU/ASN cavity (SF exit \u2192 GLU/ASN exit)",     "#E67E22"),
        ("r3_transit_frames",    "ASN\u2192HBC region (GLU/ASN exit \u2192 HBC gate)", "#2980B9"),
        ("total_transit_frames", "full cavity transit (SF exit \u2192 HBC gate)",      "#8E44AD"),
    ]
    pct_regions = [
        ("r2_pct", "% of transit time in GLU/ASN cavity",      "#E67E22"),
        ("r3_pct", "% of transit time in ASN\u2192HBC region","#2980B9"),
    ]

    transit_dir = output_dir / subdir
    transit_dir.mkdir(parents=True, exist_ok=True)
    output_dir = transit_dir  # redirect all saves into subdir

    print("\n" + "="*60)
    print("AGGREGATED TRANSIT TIME STATISTICS")
    print(f"(frame_duration = {frame_duration} ns/frame)")
    print("="*60)

    for key, label, color in regions:
        vals = []
        for r in all_rows:
            v = r.get(key, "")
            if str(v).strip() != "":
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if not vals:
            print(f"  No data for {label} — skipping")
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
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.hist(arr, bins=30, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_f,   color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_f:.1f} f ({mean_ns:.3f} ns)")
        ax.axvline(median_f, color="green", linewidth=2, linestyle="-.",
                   label=f"Median = {median_f:.1f} f ({median_ns:.3f} ns)")
        ax.set_xlabel("Frames", fontsize=20)
        ax.set_ylabel("Number of permeation events", fontsize=20)
        ax.set_title(f"{channel_name}: frames spent in {label}\n({note}, n = {len(arr)})",
                     fontsize=18, fontweight="bold", pad=10)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key.replace("_frames", "") + f"_transit{suffix}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

        # ns histogram
        vals_ns = arr * frame_duration
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.hist(vals_ns, bins=30, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_ns,   color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_ns:.3f} ns")
        ax.axvline(median_ns, color="green", linewidth=2, linestyle="-.",
                   label=f"Median = {median_ns:.3f} ns")
        ax.set_xlabel("Time (ns)", fontsize=20)
        ax.set_ylabel("Number of permeation events", fontsize=20)
        ax.set_title(f"{channel_name}: time spent in {label}\n({note}, n = {len(arr)})",
                     fontsize=18, fontweight="bold", pad=10)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key.replace("_frames", "") + f"_transit_ns{suffix}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

    # ── % of total transit time (all runs pooled) ──
    for key, label, color in pct_regions:
        vals = []
        for r in all_rows:
            v = r.get(key, "")
            if str(v).strip() != "":
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
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

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.hist(arr, bins=20, color=color, alpha=0.75,
                edgecolor="black", linewidth=1.0)
        ax.axvline(mean_pct, color="red",    linewidth=2, linestyle="--",
                   label=f"Mean = {mean_pct:.1f} ± {std_pct:.1f} %")
        ax.axvline(med_pct,  color="green", linewidth=2, linestyle="-.",
                   label=f"Median = {med_pct:.1f} %")
        ax.set_xlabel("% of total permeation time (SF exit → HBC gate)", fontsize=18)
        ax.set_ylabel("Number of permeation events", fontsize=20)
        ax.set_title(f"{channel_name}: {label}\n({note}, n = {len(arr)})",
                     fontsize=18, fontweight="bold", pad=10)
        ax.legend(fontsize=16)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = key + f"_histogram{suffix}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir / fname}")

    print("="*60)


def plot_pct_bar_aggregated(all_rows, output_dir, channel_name="", note="all runs pooled"):
    """
    Bar plot of mean % transit time in GLU/ASN vs ASN→HBC
    pooling ALL individual percentages from all runs.
    """
    r2_pcts, r3_pcts = [], []
    for r in all_rows:
        try:
            r2_pcts.append(float(r["r2_pct"]))
            r3_pcts.append(float(r["r3_pct"]))
        except (ValueError, TypeError, KeyError):
            pass

    if not r2_pcts:
        print("  No pct data for aggregated bar plot — skipping.")
        return

    labels  = ["GLU/ASN cavity", "ASN\u2192HBC region"]
    means   = [np.mean(r2_pcts), np.mean(r3_pcts)]
    stds    = [np.std(r2_pcts),  np.std(r3_pcts)]
    colors  = ["#E67E22", "#2980B9"]
    all_vals= [r2_pcts, r3_pcts]
    n       = len(r2_pcts)

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=7, width=0.5,
           color=colors, alpha=0.80, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=2)

    rng = np.random.default_rng(42)
    for i, (vals, col) in enumerate(zip(all_vals, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, alpha=0.25, s=14,
                   edgecolors="none", zorder=3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.1f}\n\u00b1{s:.1f}",
                ha="center", va="bottom", fontsize=15)

    ax.set_ylim(0, 135)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17)
    ax.set_ylabel("% of total transit time", fontsize=17)
    ax.set_title(f"{channel_name}: % transit time per cavity region\n(n = {n} events, {note})",
                 fontsize=17, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "pct_transit_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out}")

def plot_all(all_rows, output_dir, channel_name="", note=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(all_rows)
    tail = f"\n({note}, n = {n_total})" if note else f"\n(n = {n_total})"
    if n_total == 0:
        print("No data to plot.")
        return

    # Parse
    entered     = [r for r in all_rows if r["new_ion_entered"].strip().lower() == "true"]
    not_entered = [r for r in all_rows if r["new_ion_entered"].strip().lower() == "false"]
    n_yes = len(entered)
    n_no  = len(not_entered)

    # ── Yes/No bar — counts ──
    _bar_plot(
        labels=["New ion entered", "No new ion"],
        values=[n_yes, n_no],
        colors=["#E74C3C", "#95A5A6"],
        title=f"{channel_name}: " + "New ion enters GLU/ASN cavity\nwhile permeating ion passes HBC gate" + tail,
        ylabel="Number of permeation events",
        out_path=output_dir / "yes_no_count.png",
        fmt="{}",
    )

    # ── Yes/No bar — % ──
    _bar_plot(
        labels=["New ion entered", "No new ion"],
        values=[round(n_yes / n_total * 100, 1),
                round(n_no  / n_total * 100, 1)],
        colors=["#E74C3C", "#95A5A6"],
        title=f"{channel_name}: " + "New ion enters GLU/ASN cavity\nwhile permeating ion passes HBC gate — %" + tail,
        ylabel="% of permeation events",
        out_path=output_dir / "yes_no_pct.png",
        fmt="{:.1f}%",
    )

    if not entered:
        print("No 'yes' events across all runs — skipping distribution plots.")
        return

    # Parse numeric columns for yes-events
    def _floats(key):
        return [float(r[key]) for r in entered
                if key in r and str(r[key]).strip() != ""]

    frames_after = _floats("frames_after_end2")
    time_after   = _floats("time_after_end2_ns")
    rel_pos      = _floats("relative_position")
    window_lens  = _floats("window_length")

    # ── Raw frames after end_2 ──
    _histogram(frames_after,
        title=f"{channel_name}: " + "Frames after ion leaves GLU/ASN cavity\nuntil new ion enters" + tail,
        xlabel="Frames after ion leaves GLU/ASN cavity",
        out_path=output_dir / "frames_until_new_ion.png",
        color="#E74C3C", bins=20)

    _histogram(frames_after,
        title=f"{channel_name}: " + "Frames after ion leaves GLU/ASN cavity\nuntil new ion enters — %" + tail,
        xlabel="Frames after ion leaves GLU/ASN cavity",
        out_path=output_dir / "frames_until_new_ion_pct.png",
        color="#E74C3C", bins=20, pct=True, total=n_total)

    # ── Time (ns) after end_2 ──
    _histogram(time_after,
        title=f"{channel_name}: " + "Time after ion leaves GLU/ASN cavity\nuntil new ion enters (ns)" + tail,
        xlabel="Time after ion leaves GLU/ASN cavity (ns)",
        out_path=output_dir / "time_until_new_ion_ns.png",
        color="#E74C3C", bins=20)

    # ── Relative position ──
    _histogram(rel_pos,
        title=f"{channel_name}: " + "Relative entry timing of new ion\n(0% = just left GLU/ASN, 100% = ion passes HBC gate)" + tail,
        xlabel="Relative position in journey (%)",
        out_path=output_dir / "relative_entry_timing.png",
        color="#8E44AD", bins=20)

    _histogram(rel_pos,
        title=f"{channel_name}: " + "Relative entry timing of new ion — %\n(0% = just left GLU/ASN, 100% = ion passes HBC gate)" + tail,
        xlabel="Relative position in journey (%)",
        out_path=output_dir / "relative_entry_timing_pct.png",
        color="#8E44AD", bins=20, pct=True, total=n_total)

    # ── Window length for yes-events ──
    _histogram(window_lens,
        title=f"{channel_name}: " + "Window length (GLU/ASN exit → HBC gate)\nfor events where new ion entered" + tail,
        xlabel="Frames (GLU/ASN exit → HBC gate)",
        out_path=output_dir / "window_length.png",
        color="#2980B9", bins=20)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_rows):
    n_total = len(all_rows)
    if n_total == 0:
        print("No data.")
        return

    entered = [r for r in all_rows
               if r["new_ion_entered"].strip().lower() == "true"]
    n_yes = len(entered)
    n_no  = n_total - n_yes

    print("\n" + "="*70)
    print("CAVITY REFILLING — AGGREGATED SUMMARY")
    print("="*70)
    print(f"Total permeation events: {n_total}")
    print(f"  New ion entered GLU/ASN: {n_yes} ({n_yes/n_total*100:.1f}%)")
    print(f"  No new ion entered:      {n_no}  ({n_no/n_total*100:.1f}%)")

    if entered:
        def _floats(key):
            return [float(r[key]) for r in entered
                    if key in r and str(r[key]).strip() != ""]
        fa = _floats("frames_after_end2")
        rp = _floats("relative_position")
        ta = _floats("time_after_end2_ns")
        print(f"\nFor events WITH a new ion (N={len(entered)}):")
        if fa:
            print(f"  Frames after ion leaves GLU/ASN:  mean={np.mean(fa):.1f} ± {np.std(fa):.1f}  range=[{min(fa):.0f}–{max(fa):.0f}]")
        if ta:
            print(f"  Time after ion leaves GLU/ASN (ns): mean={np.mean(ta):.3f} ± {np.std(ta):.3f}")
        else:
            print(f"  Time after ion leaves GLU/ASN (ns): not available (re-run with force_rerun=True)")
        if rp:
            print(f"  Relative position (%): mean={np.mean(rp):.1f} ± {np.std(rp):.1f}  range=[{min(rp):.1f}–{max(rp):.1f}]")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Ions present at frame 0 / entering before equilibration
# ─────────────────────────────────────────────────────────────────────────────

def load_equilibration(base_dir, path=None):
    """{run number: equilibration start frame} from the site chain's equilibration.csv
    (default: the first <base>/aggregated_site_chain/*/equilibration.csv), or {} if none."""
    if path is None:
        found = sorted(Path(base_dir).glob("aggregated_site_chain/*/equilibration.csv"))
        if not found:
            return {}, None
        path = found[0]
    starts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("used_start_frame", "")).strip() != "":
                starts[int(float(row["run"]))] = int(float(row["used_start_frame"]))
    return starts, path


def drop_frame0_preequil(all_rows, eq_starts):
    """The rows without the ions already in the cavity at frame 0 (r1_end == 0) and
    without the ions leaving the SF (r1_end) before their run's equilibration start."""
    kept, n0, npre = [], 0, 0
    for r in all_rows:
        try:
            r1 = int(float(r["r1_end"]))
        except (ValueError, TypeError, KeyError):
            kept.append(r)
            continue
        run = int("".join(ch for ch in r["run"] if ch.isdigit()) or -1)
        if r1 == 0:
            n0 += 1
        elif r1 < eq_starts.get(run, 0):
            npre += 1
        else:
            kept.append(r)
    return kept, n0, npre


def write_group(name, rows, flux_rows, out_dir, channel_name, frame_duration, eq_starts, eq_path):
    """All outputs of one group of runs (ALL, HIGH, TYPICAL, LOW) in out_dir: the plots with
    all ions (old: transit_times_old/, old/) and without the frame-0 / pre-equilibration ions."""
    print("\n" + "#" * 70 + f"\n# {name}: {len({r['run'] for r in rows})} runs, {len(rows)} events\n" + "#" * 70)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_aggregated_csv(rows, out_dir)
    flux_result = save_flux_summary(flux_rows, out_dir)
    agg_flux = flux_result[1] if flux_result else None
    plot_flux(flux_rows, out_dir, aggregated_flux=agg_flux, channel_name=channel_name)
    print_summary(rows)
    kept, n0, npre = drop_frame0_preequil(rows, eq_starts)
    print(f"\nWithout {n0} ions present at frame 0 and {npre} entering before equilibration "
          f"({eq_path}): {len(kept)} of {len(rows)} ions")
    save_aggregated_csv(kept, out_dir, "cavity_refilling_aggregated_equilibrated.csv")
    # old: all ions
    plot_transit_histograms(rows, out_dir, frame_duration=frame_duration, channel_name=channel_name,
                            subdir="transit_times_old", suffix="_old", note="old: all ions")
    plot_all(rows, out_dir / "old", channel_name=channel_name, note="old: all ions")
    plot_pct_bar_aggregated(rows, out_dir / "old", channel_name=channel_name, note="old: all ions")
    # new: without the frame-0 / pre-equilibration ions
    new = "without frame-0 / pre-equilibration ions"
    plot_transit_histograms(kept, out_dir, frame_duration=frame_duration, channel_name=channel_name,
                            note=new)
    if kept:
        plot_all(kept, out_dir, channel_name=channel_name, note=new)
        plot_pct_bar_aggregated(kept, out_dir, channel_name=channel_name, note="without frame-0 / pre-eq. ions")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cavity refilling analysis across multiple runs"
    )
    parser.add_argument("base_dir",
                        help="Base directory containing RUN1, RUN2, ...")
    parser.add_argument("--runs", nargs="+", type=int,
                        help="Specific run numbers (e.g. --runs 1 2 3)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: base_dir/cavity_refilling_aggregated)")
    parser.add_argument("--frame_duration", type=float, default=None,
                        help="Time (ns) represented by one trajectory frame, "
                             "used to convert frame counts to ns in the "
                             "transit-time histograms (default: 0.4 for a G2 folder, else 0.075)")
    parser.add_argument("--equilibration", default=None,
                        help="equilibration.csv (run, used_start_frame) for leaving out the ions "
                             "entering before equilibration (default: the site chain's one in "
                             "base_dir/aggregated_site_chain/*/)")
    parser.add_argument("--categories", default=None,
                        help="run_categories CSV from analyze_permeation_summary.py (default: "
                             "base_dir/permeation_summary/run_categories_full_permeations_total.csv)")
    parser.add_argument("--no_categories", action="store_true", help="ALL/ only")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    if args.frame_duration is None:
        args.frame_duration = 0.4 if base_dir.name.upper() == "G2" else 0.075
    output_dir = Path(args.output_dir) if args.output_dir \
                 else base_dir / "cavity_refilling_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBase directory:  {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame duration:   {args.frame_duration} ns/frame")
    print("="*70)

    channel_name = base_dir.name

    all_rows, flux_rows = load_all_runs(base_dir, args.runs)
    if not all_rows:
        print("No data found. Exiting.")
        return

    eq_starts, eq_path = load_equilibration(base_dir, args.equilibration)
    if eq_path is None:
        print("\n  [WARN] no equilibration.csv found (run aggregate_site_chain.py first, or give "
              "--equilibration): only the ions present at frame 0 are left out")

    # ALL runs, then the HIGH / TYPICAL / LOW runs (permeation categories), each in its own folder
    groups = [("ALL", {r["run"] for r in all_rows})]
    if not args.no_categories:
        cat_file = Path(args.categories) if args.categories else \
            base_dir / "permeation_summary" / "run_categories_full_permeations_total.csv"
        if cat_file.exists():
            cats = load_categories(cat_file)
            print(f"\nCategories from: {cat_file}")
            groups += [(c, {r for r, v in cats.items() if v == c}) for c in CATEGORIES]
        else:
            print(f"\n⚠ Categories file not found: {cat_file}\n  No HIGH/TYPICAL/LOW folders "
                  f"(run analyze_permeation_summary.py first, or pass --categories).")
    for name, runs in groups:
        rows = [r for r in all_rows if r["run"] in runs]
        if not rows:
            print(f"\n--- {name}: no runs with cavity refilling results, folder skipped")
            continue
        write_group(name, rows, [r for r in flux_rows if r["run"] in runs], output_dir / name,
                    f"{channel_name} {name}" if name != "ALL" else channel_name,
                    args.frame_duration, eq_starts, eq_path)

    print(f"\n✓ Done. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()

# USAGE:
# python aggregate_cavity_refilling.py /path/to/G12
# python aggregate_cavity_refilling.py /path/to/G12_GAT --output_dir /path/to/output
# python aggregate_cavity_refilling.py /path/to/G12 --frame_duration 0.02
#
# python3 aggregate_cavity_refilling.py /media/ziyue/.../G12
# python3 aggregate_cavity_refilling.py /media/ziyue/.../G12_GAT
# python3 aggregate_cavity_refilling.py /media/ziyue/.../G12_ML
# python3 aggregate_cavity_refilling.py /media/ziyue/.../G2

'''
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N --frame_duration 0.075 
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q --frame_duration 0.075 
python3 aggregate_cavity_refilling.py /media/ziyue/.../G2
'''

'''
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --frame_duration 0.075 
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --frame_duration 0.4
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML --frame_duration 0.075
python3 aggregate_cavity_refilling.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT --frame_duration 0.075
python3 aggregate_cavity_refilling.py /media/ziyue/.../G2
'''