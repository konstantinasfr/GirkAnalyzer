#!/usr/bin/env python3
"""
permeation_correlation.py
=========================
The professor's first goal: for each permeating ion, correlate its residence
time in the cavity with the number of ions in the cavity — G12 versus G2.

Every permeating ion is ONE data point:

    y = cavity residence of that ion      (cavity_ns / cavity_frames)
    x = number of ions in the cavity      (see X-AXIS below)

X-AXIS CHOICES
--------------
Both are computed, because they answer slightly different questions:

  n_cavity_region        ions in region 2 or 3 at the moment of permeation
                         (end_3), the permeating ion included. This is the
                         column the professor's point (c) describes.

  n_other_ions_contact   distinct OTHER ions within the contact cutoff of any
                         of the 8 residues at end_3. Narrower, but immune to
                         the region-tracking caveat below.

Neither is a time-average over the ion's stay. A residence time is produced by
the crowding the ion experienced THROUGHOUT its visit, not at the single frame
it left, so both of these are proxies. Computing a true mean occupancy over
end_1..end_3 needs the trajectory and belongs in permeation_ion_table.py.

CAVEAT ON n_cavity_region
-------------------------
It is derived from ion_region_tracking.json by carrying each ion's last known
region forward. An ion whose record simply stops while it is in region 2 is
then counted as present indefinitely, which inflates the count. If this column
looks suspiciously flat across a run, prefer n_other_ions_contact.

RESIDUE OCCUPANCY AT PERMEATION  (new in 2026-10-06c)
-----------------------------------------------------
The per-ion table already carries one `nions_<LABEL>` column per residue — the
number of OTHER ions within the contact cutoff of that residue at frame end_3.
This script now collapses those columns to ONE number per residue, so the
question "which residues are occupied at the moment an ion permeates?" gets a
single table and a single bar plot, per run and pooled across runs.

  mean_ions       mean number of other ions on that residue at permeation
  frac_occupied   fraction of permeation events with >= 1 other ion there
  frac_2plus      fraction with >= 2 — the genuinely crowded sites

NOTE: `nions_` EXCLUDES the permeating ion itself (permeation_ion_table.py
subtracts ion_id). So these count the crowd the ion leaves behind, not the ion
under study. Residue labels are read from the CSV header, never hardcoded,
because they differ between G12 (E141.G1, D173.G1, ...) and G2 (E152.2a, ...).

MODES
-----
per-run (from main.py)   one channel, one run: scatter + Spearman + residue bars
pooled  (CLI --pool)     every run of one or more channels, pooled into the
                         G12-vs-G2 comparison figures

OUTPUTS — per run, in <results_dir>/permeation_ion_table/correlation/
    correlation_summary.csv              rho / r / n for each x-axis and unit
    residence_vs_occupancy_<x>.png       scatter + per-occupancy medians
    residue_occupancy_at_permeation.csv  one row per residue
    residue_occupancy_at_permeation.png  bar: mean ions + % occupied

OUTPUTS — pooled, in <out_dir> (default <first base_dir>/../permeation_correlation)
    pooled_ions.csv                  every ion from every run, both channels
    pooled_correlation_summary.csv   per channel: rho, r, n, per-occupancy stats
    residence_vs_occupancy_pooled_<x>.png    both channels on one figure
    residence_by_occupancy_box_<x>.png       boxplot per occupancy, side by side
    residue_occupancy_at_permeation_pooled.csv   one row per channel x residue
    residue_occupancy_at_permeation_<CH>.png     bar plot, one per channel

USAGE — from main.py
--------------------
    from analysis.permeation_correlation import run_permeation_correlation

    run_permeation_correlation(
        results_dir    = results_dir,
        channel_type   = channel_type,
        frame_duration = frame_duration,
        run            = run_number,
        force_rerun    = True,
    )

    # call AFTER run_permeation_ion_table()

USAGE — pooled comparison
-------------------------
    python3 permeation_correlation.py --pool \
        /media/ziyue/.../girk_analyser_results/G12 \
        /media/ziyue/.../girk_analyser_results/G2
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-06c  (+ per-residue occupancy at permeation)"

X_AXES = {
    "n_cavity_region": "Number of ions in the cavity (regions 2-3) at permeation",
    "n_other_ions_contact": "Number of other ions contacting the 8 residues at permeation",
}

CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22",
                 "G12_ML": "#27AE60", "G12_GAT": "#8E44AD"}

RUN_PATTERN = re.compile(r"^RUN(\d+)$", re.IGNORECASE)

# Prefix of the per-residue ion-count columns written by permeation_ion_table.py
RESIDUE_PREFIX = "nions_"


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics (numpy only; scipy used for p-values if available)
# ─────────────────────────────────────────────────────────────────────────────

def _rankdata(a):
    """Average ranks, ties shared."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    for v in np.unique(a):
        idx = np.where(a == v)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    p = float("nan")
    try:
        from scipy import stats
        rho, p = stats.spearmanr(x, y)
    except Exception:
        pass
    return float(rho), float(p)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    p = float("nan")
    try:
        from scipy import stats
        r, p = stats.pearsonr(x, y)
    except Exception:
        pass
    return float(r), float(p)


def _fmt(v, nd=3):
    return "" if v != v else round(float(v), nd)


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run_table(run_dir, unit="ns"):
    """Read one run's permeation_ion_table_<unit>.csv. Returns [] if missing."""
    path = run_dir / "permeation_ion_table" / f"permeation_ion_table_{unit}.csv"
    if not path.is_file():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_points(rows, xcol, ycol):
    """(x, y) arrays for rows where both columns are usable."""
    xs, ys = [], []
    for r in rows:
        xv, yv = r.get(xcol, ""), r.get(ycol, "")
        if str(xv).strip() == "" or str(yv).strip() == "":
            continue
        try:
            xs.append(float(xv))
            ys.append(float(yv))
        except ValueError:
            continue
    return np.array(xs), np.array(ys)


def per_occupancy_table(x, y):
    """mean / median / n at each distinct occupancy value."""
    out = []
    for v in sorted(np.unique(x)):
        sel = y[x == v]
        out.append(dict(occupancy=int(v), n_ions=int(sel.size),
                        mean=round(float(sel.mean()), 3),
                        median=round(float(np.median(sel)), 3),
                        std=round(float(sel.std(ddof=1)) if sel.size > 1 else 0.0, 3),
                        minimum=round(float(sel.min()), 3),
                        maximum=round(float(sel.max()), 3)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Residue occupancy AT THE MOMENT OF PERMEATION  (professor's point (c))
#
#  Collapses the per-ion nions_<LABEL> columns to ONE number per residue.
#  Reads only the existing permeation_ion_table CSVs — no trajectory pass.
# ─────────────────────────────────────────────────────────────────────────────

def residue_columns(rows):
    """
    nions_* columns, in CSV order. Discovered from the file rather than
    hardcoded, because G12 and G2 carry different residue labels.
    """
    if not rows:
        return []
    return [c for c in rows[0].keys() if c.startswith(RESIDUE_PREFIX)]


def _res_label(col):
    return col[len(RESIDUE_PREFIX):]


def _col_values(rows, col):
    """Numeric values of one column, skipping blanks and unparseable cells."""
    vals = []
    for r in rows:
        v = str(r.get(col, "")).strip()
        if v == "":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return np.array(vals, dtype=float)


def residue_occupancy_table(rows, channel_type="", run=""):
    """
    One row per residue, pooled over every permeation event in `rows`.

    mean_ions     : mean number of OTHER ions on that residue at end_3
    frac_occupied : fraction of permeation events with >= 1 other ion there
    frac_2plus    : fraction with >= 2
    """
    out = []
    for col in residue_columns(rows):
        a = _col_values(rows, col)
        if a.size == 0:
            continue
        out.append(dict(
            channel_type=channel_type,
            run=run,
            residue=_res_label(col),
            n_events=int(a.size),
            total_ions=int(a.sum()),
            mean_ions=round(float(a.mean()), 3),
            median_ions=round(float(np.median(a)), 3),
            max_ions=int(a.max()),
            frac_occupied=round(float((a >= 1).mean()), 4),
            frac_2plus=round(float((a >= 2).mean()), 4),
        ))
    return out


def write_rows_csv(rows, path):
    if not rows:
        print(f"    [warn] nothing to write to {Path(path).name}")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {Path(path).name}  ({len(rows)} rows)")


def plot_residue_occupancy(table, out_path, title, color="#2980B9"):
    """
    Left panel : mean number of other ions on each residue at permeation.
    Right panel: % of permeation events where that residue held >= 1 ion.
    """
    if not table:
        return
    labels = [t["residue"] for t in table]
    means = [t["mean_ions"] for t in table]
    fracs = [100 * t["frac_occupied"] for t in table]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.bar(x, means, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, means):
        ax1.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Mean number of other ions", fontsize=10)
    ax1.set_ylim(0, max(means) * 1.18 if max(means) > 0 else 1)
    ax1.set_title("Ions on each residue at permeation", fontsize=11)

    ax2.bar(x, fracs, color=color, alpha=0.55, edgecolor="black", linewidth=0.5,
            hatch="//")
    for xi, v in zip(x, fracs):
        ax2.text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("% of permeation events with ≥1 ion", fontsize=10)
    ax2.set_ylim(0, 108)
    ax2.set_title("Fraction of permeations with the residue occupied", fontsize=11)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def residue_occupancy_for_rows(rows, out_dir, title, channel_type="", run="",
                               stem="residue_occupancy_at_permeation"):
    """Build the table, write the CSV, draw the bar plot. Returns the table."""
    table = residue_occupancy_table(rows, channel_type, run)
    if not table:
        print("    [skip] no nions_* columns found — regenerate the per-ion "
              "table with permeation_ion_table.py 2026-09-26c or later.")
        return []
    write_rows_csv(table, Path(out_dir) / f"{stem}.csv")
    plot_residue_occupancy(table, Path(out_dir) / f"{stem}.png", title,
                           color=CHANNEL_COLOR.get(channel_type, "#2980B9"))
    for t in table:
        print(f"      {t['residue']:<12} mean={t['mean_ions']:<6} "
              f"occupied in {100 * t['frac_occupied']:.0f}% of permeations "
              f"(max {t['max_ions']})")
    return table


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def ylabel_for(unit):
    return "Cavity residence (ns)" if unit == "ns" else "Cavity residence (frames)"


def plot_scatter(x, y, out_path, title, xlabel, unit, color="#2980B9",
                 logy=False):
    rho, p_rho = spearman(x, y)
    r, p_r = pearson(x, y)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x + rng.uniform(-0.12, 0.12, x.size), y, s=26, alpha=0.55,
               color=color, edgecolors="black", linewidths=0.3, zorder=3)

    med = per_occupancy_table(x, y)
    ax.plot([m["occupancy"] for m in med], [m["median"] for m in med],
            "o-", color="green", linewidth=2, markersize=7,
            label="Median per occupancy", zorder=4)

    if np.unique(x).size > 1:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color="red", linewidth=1.5,
                label=f"Linear fit (slope {coef[0]:.2f})")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel_for(unit), fontsize=11)
    if logy and y.min() > 0:
        ax.set_yscale("log")
    ax.set_xticks(sorted(int(v) for v in np.unique(x)))
    sub = f"Spearman ρ = {rho:.3f}" + (f" (p = {p_rho:.3g})" if p_rho == p_rho else "")
    ax.set_title(f"{title}\n{sub},  n = {x.size} ions",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")
    return rho, p_rho, r, p_r


def plot_pooled_scatter(by_channel, out_path, xlabel, unit, logy=True):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(0)
    lines = []
    for i, (ch, (x, y)) in enumerate(by_channel.items()):
        if x.size == 0:
            continue
        color = CHANNEL_COLOR.get(ch, f"C{i}")
        off = -0.15 + 0.3 * i / max(1, len(by_channel) - 1) if len(by_channel) > 1 else 0
        ax.scatter(x + off + rng.uniform(-0.07, 0.07, x.size), y, s=22,
                   alpha=0.40, color=color, edgecolors="none", zorder=2)
        med = per_occupancy_table(x, y)
        rho, p = spearman(x, y)
        ax.plot([m["occupancy"] + off for m in med], [m["median"] for m in med],
                "o-", color=color, linewidth=2.5, markersize=8,
                markeredgecolor="black", markeredgewidth=0.5, zorder=4,
                label=f"{ch}  (n={x.size}, ρ={rho:.2f})")
        lines.append(ch)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel_for(unit), fontsize=11)
    if logy:
        ax.set_yscale("log")
    all_x = np.concatenate([x for x, _ in by_channel.values() if x.size])
    ax.set_xticks(sorted(int(v) for v in np.unique(all_x)))
    ax.set_title("Cavity residence vs cavity occupancy\n"
                 "(each dot is one permeating ion; lines are per-occupancy medians)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_pooled_box(by_channel, out_path, xlabel, unit):
    """Side-by-side boxplots of residence at each occupancy, per channel."""
    channels = [c for c, (x, _) in by_channel.items() if x.size]
    if not channels:
        return
    all_x = np.concatenate([by_channel[c][0] for c in channels])
    occs = sorted(int(v) for v in np.unique(all_x))
    n_ch = len(channels)
    width = 0.8 / n_ch

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, ch in enumerate(channels):
        x, y = by_channel[ch]
        data, pos = [], []
        for j, o in enumerate(occs):
            sel = y[x == o]
            if sel.size:
                data.append(sel)
                pos.append(j + i * width - 0.4 + width / 2)
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="green", linewidth=1.8))
        for box in bp["boxes"]:
            box.set_facecolor(CHANNEL_COLOR.get(ch, f"C{i}"))
            box.set_alpha(0.6)
            box.set_edgecolor("black")
        ax.plot([], [], color=CHANNEL_COLOR.get(ch, f"C{i}"), linewidth=8,
                alpha=0.6, label=f"{ch} (n={x.size})")

    ax.set_xticks(range(len(occs)))
    ax.set_xticklabels(occs)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel_for(unit), fontsize=11)
    ax.set_title("Cavity residence at each occupancy", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_permeation_correlation(results_dir, channel_type="", frame_duration=0.075,
                               run="", force_rerun=False, unit="ns"):
    """
    Per-run correlation of cavity residence against cavity occupancy, plus the
    per-residue occupancy table and bar plot at the moment of permeation.

    Note the correlation part is ONE run, so n is small and the occupancy range
    narrow — the real comparison is the pooled mode. This exists so every run
    carries its own figures alongside the rest of its outputs.
    """
    results_dir = Path(results_dir)
    table_dir = results_dir / "permeation_ion_table"
    out_dir = table_dir / "correlation"
    summary_path = out_dir / "correlation_summary.csv"

    rows = load_run_table(results_dir, unit)
    if not rows:
        print(f"  [skip] permeation_ion_table_{unit}.csv not found in {table_dir} "
              f"— run run_permeation_ion_table() first.")
        return None
    if summary_path.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {summary_path} exists. Pass force_rerun=True.")
        print("!" * 70)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    ycol = "cavity_ns" if unit == "ns" else "cavity_frames"

    print("\n" + "=" * 70)
    print("PERMEATION CORRELATION — residence vs cavity occupancy")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  {len(rows)} ions")

    summary = []
    for xcol, xlabel in X_AXES.items():
        x, y = extract_points(rows, xcol, ycol)
        if x.size < 3:
            print(f"  [skip] {xcol}: only {x.size} usable points")
            continue
        title = f"{channel_type}" + (f" RUN{run}" if str(run) != "" else "")
        rho, p_rho, r, p_r = plot_scatter(
            x, y, out_dir / f"residence_vs_occupancy_{xcol}.png",
            title, xlabel, unit,
            color=CHANNEL_COLOR.get(channel_type, "#2980B9"))
        print(f"  {xcol}: Spearman rho={rho:.3f}  Pearson r={r:.3f}  n={x.size}  "
              f"occupancy range {int(x.min())}-{int(x.max())}")
        summary.append(dict(channel_type=channel_type, run=run, unit=unit,
                            x_axis=xcol, n_ions=int(x.size),
                            spearman_rho=_fmt(rho), spearman_p=_fmt(p_rho, 6),
                            pearson_r=_fmt(r), pearson_p=_fmt(p_r, 6),
                            occupancy_min=int(x.min()), occupancy_max=int(x.max()),
                            residence_mean=round(float(y.mean()), 3),
                            residence_median=round(float(np.median(y)), 3)))
        for m in per_occupancy_table(x, y):
            print(f"      occupancy {m['occupancy']}: n={m['n_ions']:<3} "
                  f"mean={m['mean']:<9} median={m['median']}")

    if summary:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"    Saved: {summary_path.name}")

    # ---- residue occupancy at permeation ----
    print("\n  --- residue occupancy at the moment of permeation ---")
    run_tag = f" RUN{run}" if str(run) != "" else ""
    residue_occupancy_for_rows(
        rows, out_dir,
        title=f"{channel_type}{run_tag} — residue occupancy at permeation "
              f"(n = {len(rows)} ions)",
        channel_type=channel_type, run=run)

    print(f"\n✓ Done. Outputs in {out_dir}/")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Pooled mode — the actual G12 vs G2 comparison
# ─────────────────────────────────────────────────────────────────────────────

def pool_channels(base_dirs, unit="ns", out_dir=None):
    base_dirs = [Path(b) for b in base_dirs]
    for b in base_dirs:
        if not b.is_dir():
            sys.exit(f"ERROR: not a directory: {b}")
    out_dir = Path(out_dir) if out_dir else base_dirs[0].parent / "permeation_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ycol = "cavity_ns" if unit == "ns" else "cavity_frames"

    print("=" * 70)
    print("POOLED CORRELATION — residence vs cavity occupancy")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}  unit={unit}")

    all_rows = []
    for base in base_dirs:
        n_runs = 0
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or not RUN_PATTERN.match(sub.name):
                continue
            rows = load_run_table(sub, unit)
            if not rows:
                continue
            n_runs += 1
            for r in rows:
                r.setdefault("channel_type", base.name)
                if not str(r.get("run", "")).strip():
                    r["run"] = sub.name
                all_rows.append(r)
        print(f"  {base.name}: {n_runs} runs pooled")

    if not all_rows:
        sys.exit("ERROR: no permeation tables found under the given directories.")

    # pooled_ions.csv keeps the per-residue columns too, so the occupancy
    # numbers below can be re-derived from this one file if needed.
    res_cols = residue_columns(all_rows)
    keep = ["ion_rank", "ion_id", "run", "channel_type", "end_1", "end_3",
            ycol, "n_cavity_region", "n_other_ions_contact",
            "start_censored", "incomplete_range"] + res_cols
    with open(out_dir / "pooled_ions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"  Saved: pooled_ions.csv  ({len(all_rows)} ions)")

    channels = []
    for r in all_rows:
        ch = r.get("channel_type", "")
        if ch and ch not in channels:
            channels.append(ch)

    summary = []
    for xcol, xlabel in X_AXES.items():
        by_channel = {}
        for ch in channels:
            rows = [r for r in all_rows if r.get("channel_type") == ch]
            by_channel[ch] = extract_points(rows, xcol, ycol)

        print(f"\n  --- x = {xcol} ---")
        for ch, (x, y) in by_channel.items():
            if x.size < 3:
                print(f"    {ch}: only {x.size} points — skipped")
                continue
            rho, p_rho = spearman(x, y)
            r_, p_r = pearson(x, y)
            print(f"    {ch}: n={x.size}  Spearman rho={rho:.3f}"
                  + (f" (p={p_rho:.3g})" if p_rho == p_rho else "")
                  + f"  occupancy {int(x.min())}-{int(x.max())}")
            rec = dict(channel_type=ch, unit=unit, x_axis=xcol,
                       n_ions=int(x.size), spearman_rho=_fmt(rho),
                       spearman_p=_fmt(p_rho, 6), pearson_r=_fmt(r_),
                       pearson_p=_fmt(p_r, 6),
                       occupancy_min=int(x.min()), occupancy_max=int(x.max()),
                       residence_mean=round(float(y.mean()), 3),
                       residence_median=round(float(np.median(y)), 3))
            for m in per_occupancy_table(x, y):
                rec[f"median_at_{m['occupancy']}"] = m["median"]
                rec[f"n_at_{m['occupancy']}"] = m["n_ions"]
                print(f"        occupancy {m['occupancy']}: n={m['n_ions']:<4} "
                      f"median={m['median']:<9} mean={m['mean']}")
            summary.append(rec)

        usable = {c: v for c, v in by_channel.items() if v[0].size >= 3}
        if usable:
            plot_pooled_scatter(usable,
                                out_dir / f"residence_vs_occupancy_pooled_{xcol}.png",
                                xlabel, unit)
            plot_pooled_box(usable,
                            out_dir / f"residence_by_occupancy_box_{xcol}.png",
                            xlabel, unit)

    if summary:
        cols = []
        for rec in summary:
            for k in rec:
                if k not in cols:
                    cols.append(k)
        with open(out_dir / "pooled_correlation_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary)
        print(f"\n  Saved: pooled_correlation_summary.csv")

    # ---- residue occupancy at permeation, pooled per channel ----
    # One bar plot per channel rather than a shared grouped plot, because the
    # residue labels are not the same set in G12 and G2.
    print("\n  --- residue occupancy at the moment of permeation (pooled) ---")
    pooled_res = []
    for ch in channels:
        rows = [r for r in all_rows if r.get("channel_type") == ch]
        table = residue_occupancy_table(rows, channel_type=ch, run="pooled")
        if not table:
            print(f"    {ch}: no nions_* columns — skipped")
            continue
        n_runs = len({r.get("run", "") for r in rows})
        print(f"    {ch}  ({len(rows)} ions, {n_runs} runs):")
        for t in table:
            print(f"      {t['residue']:<12} mean={t['mean_ions']:<6} "
                  f"occupied in {100 * t['frac_occupied']:.0f}% of permeations "
                  f"(max {t['max_ions']})")
        plot_residue_occupancy(
            table,
            out_dir / f"residue_occupancy_at_permeation_{ch}.png",
            f"{ch} — residue occupancy at permeation "
            f"(pooled over {n_runs} runs, n = {len(rows)} ions)",
            color=CHANNEL_COLOR.get(ch, "#2980B9"))
        pooled_res.extend(table)

    if pooled_res:
        write_rows_csv(pooled_res,
                       out_dir / "residue_occupancy_at_permeation_pooled.csv")

    print(f"\n✓ Done. Outputs in {out_dir}/")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", metavar="BASE_DIR",
                    help="Channel folders to pool, e.g. .../G12 .../G2")
    ap.add_argument("--results_dir", help="Single RUN directory (per-run mode)")
    ap.add_argument("--channel_type", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--frame_duration", type=float, default=0.075)
    ap.add_argument("--unit", choices=["ns", "frames"], default="ns")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    if args.pool:
        pool_channels(args.pool, unit=args.unit, out_dir=args.out)
    elif args.results_dir:
        run_permeation_correlation(
            results_dir=args.results_dir, channel_type=args.channel_type,
            frame_duration=args.frame_duration, run=args.run,
            force_rerun=args.force_rerun, unit=args.unit)
    else:
        ap.error("give either --pool <dirs> or --results_dir <RUN dir>")


if __name__ == "__main__":
    main()