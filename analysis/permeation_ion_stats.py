#!/usr/bin/env python3
"""
permeation_ion_stats.py
=======================
Statistics on the per-ion permeation table produced by permeation_ion_table.py.

Two questions, in order:

  1. HOW LONG does each ion sit in the cavity?
  2. WHAT is it doing while it sits there?

GROUPS
------
Three duration groups, from the quartiles of cavity residence:

    below_Q1     fastest quarter
    Q1_Q3        the interquartile box (typical)
    above_Q3     slowest quarter

Plus two EXTREME sets, analysed separately alongside the three groups:

    low_extreme   the low outliers (< Q1 - 1.5*IQR) if any exist,
                  otherwise falls back to below_Q1
    high_extreme  the high outliers (> Q3 + 1.5*IQR) if any exist,
                  otherwise falls back to above_Q3

The fallback is recorded in every output ("source" column / plot subtitle), so
it is always visible whether a set is true outliers or a substituted quartile.
For a strictly positive, right-skewed variable the low fence is often negative,
in which case low outliers cannot exist and the fallback always fires.

UNITS
-----
The whole analysis is written twice, into two sibling folders:

    stats_frames/   durations in frames
    stats_ns/       the same durations in ns

The residue percentages are identical in both — a percentage of the cavity time
does not depend on the unit that time is measured in.

PERCENTAGES
-----------
Contacts are NOT exclusive: one frame can count toward several residues if the
ion bridges them, so the 8 percentages do not sum to 100. The sum is reported
per ion as `pct_sum` — above 100 means multi-residue coordination, below 100
means time in contact with nothing. The denominator is cavity_frames + 1,
because res_* counts run over end_1..end_3 inclusive.

CLUSTERING
----------
k-means over the 8 per-residue values, on two feature sets:

    raw   the res_* frame counts, used as they are
    pct   the pct_* percentages, used as they are

Neither set is z-scored: within each set the 8 columns already share a unit, so
the raw magnitudes carry the signal. Standardising would inflate residues that
are rarely contacted and flatten the difference between a heavily visited site
and a barely visited one.

k is chosen automatically by silhouette score over k = 2..K_MAX. The question
is whether ions that contact the same residues also take similar times to
permeate, so every cluster is reported with its cavity-time distribution and a
boxplot of cavity time per cluster. The clustering itself does not depend on
the unit of the cavity time, so the frames and ns runs give identical clusters;
only the reported times differ.

Note that the `raw` counts grow with residence time, so clustering them partly
rediscovers the duration groups. `pct` removes that and asks only WHERE the ion
sat, which is the sharper test.

OUTPUTS (in <results_dir>/permeation_ion_table/stats_frames|stats_ns/)
----------------------------------------------------------------------
    cavity_<unit>_summary.csv            overall + per-set statistics
    ion_groups.csv                       every ion, its group and percentages
    outliers.csv                         the true outliers, if any
    residue_percentages_by_group.csv     mean/median % per residue per set
    cavity_<unit>_boxplot.png
    cavity_<unit>_hist.png
    residue_pct_<set>.png                one barplot per group / extreme set
    residue_pct_all_groups.png
    clusters/
        cluster_assignments_<fs>.csv
        cluster_profiles_<fs>.csv
        cluster_selection_<fs>.csv       silhouette score per candidate k
        cluster_profiles_<fs>.png
        cluster_cavity_time_<fs>.png

USAGE — from main.py
--------------------
    from analysis.permeation_ion_stats import run_permeation_ion_stats

    run_permeation_ion_stats(
        results_dir    = results_dir,
        channel_type   = channel_type,
        frame_duration = frame_duration,
        run            = run_number,
        force_rerun    = True,
    )

    # call it AFTER run_permeation_ion_table(), which writes the CSV it reads

USAGE — standalone
------------------
    python3 permeation_ion_stats.py --results_dir /path/RUN3 --channel_type G12 \
        --run 3 --frame_duration 0.075 --force_rerun
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-09-26c  (3 groups + extreme sets, dual units, clustering, mobility)"

MAIN_GROUPS = ["below_Q1", "Q1_Q3", "above_Q3"]
EXTREME_SETS = ["low_extreme", "high_extreme"]
ALL_SETS = MAIN_GROUPS + EXTREME_SETS

SET_LABEL = {
    "below_Q1":     "Below Q1 (fast)",
    "Q1_Q3":        "Q1-Q3 (typical)",
    "above_Q3":     "Above Q3 (slow)",
    "low_extreme":  "Low extreme",
    "high_extreme": "High extreme",
}
SET_COLOR = {
    "below_Q1":     "#27AE60",
    "Q1_Q3":        "#2980B9",
    "above_Q3":     "#E67E22",
    "low_extreme":  "#8E44AD",
    "high_extreme": "#C0392B",
}

K_MAX = 6


# ─────────────────────────────────────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────────────────────────────────────

def load_table(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: {csv_path} has no rows.")
    return rows


def residue_columns(rows):
    return [c for c in rows[0].keys() if c.startswith("res_")]


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _is_true(v):
    return str(v).strip().lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
#  Grouping
# ─────────────────────────────────────────────────────────────────────────────

def compute_cuts(values):
    arr = np.asarray(values, dtype=float)
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    return dict(q1=float(q1), median=float(q2), q3=float(q3), iqr=float(iqr),
                lo_fence=float(q1 - 1.5 * iqr), hi_fence=float(q3 + 1.5 * iqr),
                minimum=float(arr.min()), maximum=float(arr.max()))


def assign_group(v, cuts):
    """Three mutually exclusive duration groups."""
    v = float(v)
    if v < cuts["q1"]:
        return "below_Q1"
    if v <= cuts["q3"]:
        return "Q1_Q3"
    return "above_Q3"


def build_sets(rows, values, cuts):
    """
    Returns ({set_name: (rows, source_description)}, low_outliers, high_outliers).

    The three main groups are the quartile bands. The two extreme sets are the
    true outliers when they exist, otherwise the adjacent quartile band, with
    the substitution recorded in the source string.
    """
    sets = {}
    for g in MAIN_GROUPS:
        sets[g] = ([r for r in rows if r["duration_group"] == g], "quartile band")

    low_out = [r for r, v in zip(rows, values) if v < cuts["lo_fence"]]
    high_out = [r for r, v in zip(rows, values) if v > cuts["hi_fence"]]

    if low_out:
        sets["low_extreme"] = (low_out, f"true outliers (< {cuts['lo_fence']:.1f})")
    else:
        sets["low_extreme"] = (sets["below_Q1"][0],
                               "no low outliers - fell back to below_Q1")

    if high_out:
        sets["high_extreme"] = (high_out, f"true outliers (> {cuts['hi_fence']:.1f})")
    else:
        sets["high_extreme"] = (sets["above_Q3"][0],
                                "no high outliers - fell back to above_Q3")

    return sets, low_out, high_out


def describe(values, frame_duration, unit):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return dict(n=0)
    scale = 1.0 if unit == "frames" else frame_duration
    a = arr * scale
    return dict(
        n=int(arr.size),
        mean=round(float(a.mean()), 3),
        median=round(float(np.median(a)), 3),
        std=round(float(a.std(ddof=1)) if arr.size > 1 else 0.0, 3),
        minimum=round(float(a.min()), 3),
        maximum=round(float(a.max()), 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Percentages
# ─────────────────────────────────────────────────────────────────────────────

def add_percentages(rows, res_cols):
    for r in rows:
        denom = _to_int(r.get("cavity_frames")) + 1
        total = 0.0
        for c in res_cols:
            n = _to_int(r.get(c))
            pct = 100.0 * n / denom if denom > 0 else 0.0
            r["pct_" + c[len("res_"):]] = round(pct, 3)
            total += pct
        r["pct_sum"] = round(total, 3)

        # Mobility: fraction of the cavity window spent touching at least one
        # residue vs touching nothing. Unlike pct_sum these two DO add to 100,
        # because they are computed per frame rather than per residue.
        window = _to_int(r.get("frames_window"), 0) or denom
        any_c = _to_int(r.get("frames_any_contact"), -1)
        if any_c < 0:
            # older table without the mobility columns
            r["pct_any_contact"] = ""
            r["pct_no_contact"] = ""
        else:
            no_c = _to_int(r.get("frames_no_contact"), window - any_c)
            r["pct_any_contact"] = round(100.0 * any_c / window, 3) if window else 0.0
            r["pct_no_contact"] = round(100.0 * no_c / window, 3) if window else 0.0
    return rows


def has_mobility(rows):
    return rows and str(rows[0].get("pct_any_contact", "")) != ""


def mobility_stats(sets, frame_duration, unit):
    """Per set: mean/median of the any-contact and no-contact time."""
    scale = 1.0 if unit == "frames" else frame_duration
    out = []
    for name in ALL_SETS:
        members, source = sets[name]
        if not members or str(members[0].get("pct_any_contact", "")) == "":
            continue
        any_f = np.array([_to_int(r.get("frames_any_contact")) for r in members],
                         dtype=float) * scale
        no_f = np.array([_to_int(r.get("frames_no_contact")) for r in members],
                        dtype=float) * scale
        any_p = np.array([float(r["pct_any_contact"]) for r in members])
        no_p = np.array([float(r["pct_no_contact"]) for r in members])
        out.append(dict(
            group=name, source=source, n_ions=len(members), unit=unit,
            mean_any=round(float(any_f.mean()), 3),
            median_any=round(float(np.median(any_f)), 3),
            mean_no=round(float(no_f.mean()), 3),
            median_no=round(float(np.median(no_f)), 3),
            mean_pct_any=round(float(any_p.mean()), 2),
            median_pct_any=round(float(np.median(any_p)), 2),
            mean_pct_no=round(float(no_p.mean()), 2),
            median_pct_no=round(float(np.median(no_p)), 2),
        ))
    return out


def set_percentage_stats(sets, res_cols):
    out = {}
    for name in ALL_SETS:
        members, source = sets[name]
        if not members:
            out[name] = dict(n=0, source=source, stats={})
            continue
        stats = {}
        for c in res_cols:
            label = c[len("res_"):]
            vals = np.array([float(r["pct_" + label]) for r in members])
            stats[label] = (round(float(vals.mean()), 2),
                            round(float(np.median(vals)), 2))
        out[name] = dict(n=len(members), source=source, stats=stats)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  k-means (numpy only — no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans(X, k, seed=0, n_init=10, max_iter=300):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels, best_inertia, best_centers = None, np.inf, None
    for _ in range(n_init):
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            c_arr = np.array(centers)
            d2 = np.min(((X[:, None, :] - c_arr[None, :, :]) ** 2).sum(-1), axis=1)
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            centers.append(X[rng.choice(n, p=probs)])
        centers = np.array(centers, dtype=float)

        labels = np.full(n, -1, dtype=int)
        for _ in range(max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                pts = X[labels == j]
                if pts.size:
                    centers[j] = pts.mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_labels, best_inertia, best_centers = labels.copy(), inertia, centers.copy()
    return best_labels, best_centers, best_inertia


def _silhouette(X, labels):
    """Mean silhouette score. O(n^2) — fine at these sizes."""
    n = X.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2 or n <= uniq.size:
        return float("nan")
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    scores = []
    for i in range(n):
        own = labels == labels[i]
        own_others = own.copy()
        own_others[i] = False
        if own_others.sum() == 0:
            scores.append(0.0)
            continue
        a = D[i, own_others].mean()
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def choose_k(X, k_max=K_MAX, seed=0):
    results = []
    upper = min(k_max, X.shape[0] - 1)
    for k in range(2, max(2, upper) + 1):
        labels, _, inertia = _kmeans(X, k, seed=seed)
        sil = _silhouette(X, labels)
        results.append(dict(k=k,
                            silhouette=round(sil, 4) if sil == sil else "",
                            inertia=round(inertia, 3),
                            sizes="|".join(str(int((labels == j).sum()))
                                           for j in range(k))))
    valid = [r for r in results if r["silhouette"] != ""]
    best_k = max(valid, key=lambda r: r["silhouette"])["k"] if valid else 2
    return best_k, results


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _title(channel_type, run, extra):
    prefix = f"{channel_type}" + (f" RUN{run}" if str(run) != "" else "")
    return f"{prefix} — {extra}" if prefix.strip() else extra


def unit_label(unit):
    return ("Frames in cavity (end_1 → end_3)" if unit == "frames"
            else "Time in cavity (ns)")


def plot_boxplot(values, out_path, channel_type, run, unit):
    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(arr, vert=True, widths=0.45, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="green", linewidth=2))
    bp["boxes"][0].set_facecolor("#D6EAF8")
    bp["boxes"][0].set_edgecolor("#2980B9")

    fmt = "{:.0f}" if unit == "frames" else "{:.2f}"
    ax.axhline(arr.mean(), color="red", linestyle="--", linewidth=1.5,
               label="Mean = " + fmt.format(arr.mean()))
    ax.plot([], [], color="green", linewidth=2,
            label="Median = " + fmt.format(np.median(arr)))

    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.uniform(-0.12, 0.12, arr.size), arr,
               s=14, alpha=0.35, color="#2980B9", edgecolors="none", zorder=3)

    ax.set_xticks([])
    ax.set_ylabel(unit_label(unit), fontsize=11)
    ax.set_title(_title(channel_type, run,
                        f"time each ion spends in the cavity\n(n = {arr.size} ions)"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_histogram(values, cuts_scaled, out_path, channel_type, run, unit):
    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=20, color="#5DADE2", edgecolor="black", alpha=0.8)
    fmt = "{:.0f}" if unit == "frames" else "{:.2f}"
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=2,
               label="Mean = " + fmt.format(arr.mean()))
    ax.axvline(np.median(arr), color="green", linewidth=2,
               label="Median = " + fmt.format(np.median(arr)))
    for key in ("q1", "q3"):
        ax.axvline(cuts_scaled[key], color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel(unit_label(unit), fontsize=11)
    ax.set_ylabel("Number of ions", fontsize=11)
    ax.set_title(_title(channel_type, run, "cavity residence distribution"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_set_bars(set_stats, res_cols, out_dir, channel_type, run):
    labels = [c[len("res_"):] for c in res_cols]
    x = np.arange(len(labels))
    for name in ALL_SETS:
        info = set_stats[name]
        if info["n"] == 0:
            continue
        means = [info["stats"][l][0] for l in labels]
        medians = [info["stats"][l][1] for l in labels]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.bar(x - 0.2, means, width=0.38, color=SET_COLOR[name], alpha=0.85,
               edgecolor="black", linewidth=0.6, label="Mean")
        ax.bar(x + 0.2, medians, width=0.38, color=SET_COLOR[name], alpha=0.45,
               edgecolor="black", linewidth=0.6, hatch="//", label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("% of cavity time in contact", fontsize=11)
        ax.set_title(_title(channel_type, run,
                            f"{SET_LABEL[name]}  (n = {info['n']} ions)"),
                     fontsize=11, fontweight="bold")
        ax.text(0.5, -0.34, info["source"], transform=ax.transAxes,
                ha="center", fontsize=8, color="gray")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = out_dir / f"residue_pct_{name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out.name}")


def plot_all_sets(set_stats, res_cols, out_path, channel_type, run):
    labels = [c[len("res_"):] for c in res_cols]
    present = [g for g in MAIN_GROUPS if set_stats[g]["n"] > 0]
    if not present:
        return
    x = np.arange(len(labels))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, g in enumerate(present):
        info = set_stats[g]
        means = [info["stats"][l][0] for l in labels]
        ax.bar(x + i * width - 0.4 + width / 2, means, width=width * 0.92,
               color=SET_COLOR[g], alpha=0.85, edgecolor="black", linewidth=0.5,
               label=f"{SET_LABEL[g]} (n={info['n']})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean % of cavity time in contact", fontsize=11)
    ax.set_title(_title(channel_type, run,
                        "residue contact profile by cavity-residence group"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_mobility(mob_rows, out_path, channel_type, run, unit):
    """Stacked % in contact vs touching nothing, per duration group."""
    names = [m["group"] for m in mob_rows if m["group"] in MAIN_GROUPS]
    if not names:
        return
    rows = [m for m in mob_rows if m["group"] in MAIN_GROUPS]
    x = np.arange(len(rows))
    any_p = [m["mean_pct_any"] for m in rows]
    no_p = [m["mean_pct_no"] for m in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.bar(x, any_p, width=0.6, color="#2980B9", alpha=0.85,
            edgecolor="black", linewidth=0.6, label="In contact")
    ax1.bar(x, no_p, width=0.6, bottom=any_p, color="#BDC3C7", alpha=0.9,
            edgecolor="black", linewidth=0.6, label="Touching nothing")
    for i, (a, n) in enumerate(zip(any_p, no_p)):
        ax1.text(i, a / 2, f"{a:.0f}%", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
        ax1.text(i, a + n / 2, f"{n:.0f}%", ha="center", va="center",
                 fontsize=9, color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{SET_LABEL[m['group']]}\n(n={m['n_ions']})"
                         for m in rows], fontsize=8)
    ax1.set_ylabel("Mean % of cavity time", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.set_title("Contact vs free", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    width = 0.38
    ax2.bar(x - width / 2, [m["mean_any"] for m in rows], width=width,
            color="#2980B9", alpha=0.85, edgecolor="black", linewidth=0.6,
            label="In contact")
    ax2.bar(x + width / 2, [m["mean_no"] for m in rows], width=width,
            color="#BDC3C7", alpha=0.9, edgecolor="black", linewidth=0.6,
            label="Touching nothing")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SET_LABEL[m["group"]] for m in rows], fontsize=8)
    ax2.set_ylabel(unit_label(unit).split(" (")[0] + f" ({unit})", fontsize=11)
    ax2.set_title("Absolute time", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(_title(channel_type, run, "ion mobility in the cavity"),
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_cluster_profiles(centers_display, labels_arr, res_labels, out_path,
                          channel_type, run, feature_set):
    k = centers_display.shape[0]
    x = np.arange(len(res_labels))
    width = 0.8 / k
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j in range(k):
        n_j = int((labels_arr == j).sum())
        ax.bar(x + j * width - 0.4 + width / 2, centers_display[j],
               width=width * 0.92, color=cmap[j % 10], alpha=0.85,
               edgecolor="black", linewidth=0.5, label=f"Cluster {j} (n={n_j})")
    ax.set_xticks(x)
    ax.set_xticklabels(res_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean % of cavity time in contact" if feature_set == "pct"
                  else "Mean frames in contact", fontsize=11)
    ax.set_title(_title(channel_type, run,
                        f"k-means cluster profiles ({feature_set} features)"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_cluster_times(labels_arr, times, out_path, channel_type, run,
                       feature_set, unit):
    k = int(labels_arr.max()) + 1
    data = [times[labels_arr == j] for j in range(k)]
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="green", linewidth=2))
    for j, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap[j % 10])
        box.set_alpha(0.55)
        box.set_edgecolor("black")
    rng = np.random.default_rng(1)
    for j, d in enumerate(data):
        if d.size:
            ax.scatter(j + 1 + rng.uniform(-0.12, 0.12, d.size), d,
                       s=16, alpha=0.5, color=cmap[j % 10],
                       edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([f"C{j}\n(n={d.size})" for j, d in enumerate(data)],
                       fontsize=9)
    ax.set_ylabel(unit_label(unit), fontsize=11)
    ax.set_title(_title(channel_type, run,
                        f"cavity time by contact cluster ({feature_set})"),
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_rows(rows, path, fieldnames=None):
    if not rows:
        print(f"    [info] nothing to write to {path.name}")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(rows, res_cols, values, out_dir, channel_type, run,
                   frame_duration, unit, k_fixed=None, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    res_labels = [c[len("res_"):] for c in res_cols]
    scale = 1.0 if unit == "frames" else frame_duration
    times = np.asarray(values, dtype=float) * scale

    feature_sets = {
        "raw": np.array([[_to_int(r[c]) for c in res_cols] for r in rows],
                        dtype=float),
        "pct": np.array([[float(r["pct_" + l]) for l in res_labels] for r in rows],
                        dtype=float),
    }

    for fs, X_disp in feature_sets.items():
        if X_disp.shape[0] < 4:
            print(f"    [skip] clustering '{fs}': only {X_disp.shape[0]} ions")
            continue
        # No z-scoring: both feature sets are already in consistent units across
        # the 8 columns (frames, or % of cavity time), so the raw magnitudes are
        # the signal. Standardising would inflate residues that are rarely
        # contacted and erase the difference between a heavily and a barely
        # visited site.
        X = X_disp.astype(float)

        if k_fixed:
            k, selection = int(k_fixed), []
        else:
            k, selection = choose_k(X, seed=seed)
        labels_arr, _, _ = _kmeans(X, k, seed=seed)

        if selection:
            write_rows(selection, out_dir / f"cluster_selection_{fs}.csv",
                       ["k", "silhouette", "inertia", "sizes"])
            best = [s for s in selection if s["k"] == k][0]
            print(f"    clustering '{fs}': k={k} "
                  f"(silhouette {best['silhouette']}), sizes {best['sizes']}")

        assign_rows = []
        for r, lab in zip(rows, labels_arr):
            row = dict(ion_rank=r["ion_rank"], ion_id=r["ion_id"],
                       run=run, channel_type=channel_type,
                       cluster=int(lab), duration_group=r["duration_group"],
                       cavity_frames=r["cavity_frames"],
                       cavity_ns=round(_to_int(r["cavity_frames"]) * frame_duration, 4))
            for l in res_labels:
                row[f"pct_{l}"] = r["pct_" + l]
            assign_rows.append(row)
        write_rows(assign_rows, out_dir / f"cluster_assignments_{fs}.csv")

        prof_rows = []
        centers_display = np.zeros((k, len(res_labels)))
        for j in range(k):
            mask = labels_arr == j
            if not mask.any():
                continue
            centers_display[j] = X_disp[mask].mean(axis=0)
            t = times[mask]
            row = dict(cluster=j, n_ions=int(mask.sum()),
                       mean_cavity=round(float(t.mean()), 3),
                       median_cavity=round(float(np.median(t)), 3),
                       min_cavity=round(float(t.min()), 3),
                       max_cavity=round(float(t.max()), 3),
                       unit=unit, run=run, channel_type=channel_type)
            for i, l in enumerate(res_labels):
                row[f"mean_{l}"] = round(float(centers_display[j][i]), 2)
            groups = [r["duration_group"] for r, m in zip(rows, mask) if m]
            row["group_mix"] = "|".join(f"{g}:{groups.count(g)}"
                                        for g in MAIN_GROUPS if groups.count(g))
            prof_rows.append(row)
        write_rows(prof_rows, out_dir / f"cluster_profiles_{fs}.csv")

        print(f"    cavity time by cluster ({fs}, {unit}):")
        for row in prof_rows:
            print(f"      C{row['cluster']} n={row['n_ions']:<3} "
                  f"mean={row['mean_cavity']:<10} "
                  f"median={row['median_cavity']:<10} [{row['group_mix']}]")

        plot_cluster_profiles(centers_display, labels_arr, res_labels,
                              out_dir / f"cluster_profiles_{fs}.png",
                              channel_type, run, fs)
        plot_cluster_times(labels_arr, times,
                           out_dir / f"cluster_cavity_time_{fs}.png",
                           channel_type, run, fs, unit)


# ─────────────────────────────────────────────────────────────────────────────
#  One unit's worth of analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_unit(rows, res_cols, values, cuts, sets, set_stats, out_dir,
                 channel_type, run, frame_duration, unit, low_out, high_out,
                 do_clustering=True, k_fixed=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 1.0 if unit == "frames" else frame_duration
    scaled = [v * scale for v in values]
    cuts_scaled = {key: val * scale for key, val in cuts.items()}

    print(f"\n  --- {unit.upper()} ---")

    summary_rows = []
    overall = describe(values, frame_duration, unit)
    overall.update(group="ALL", source="all ions", unit=unit, run=run,
                   channel_type=channel_type,
                   q1=round(cuts_scaled["q1"], 3),
                   q3=round(cuts_scaled["q3"], 3),
                   iqr=round(cuts_scaled["iqr"], 3),
                   low_fence=round(cuts_scaled["lo_fence"], 3),
                   high_fence=round(cuts_scaled["hi_fence"], 3))
    summary_rows.append(overall)

    print(f"    {'set':<14} {'n':>4} {'mean':>11} {'median':>11} "
          f"{'min':>9} {'max':>9}")
    print("    " + "-" * 64)
    print(f"    {'ALL':<14} {overall['n']:>4} {overall['mean']:>11} "
          f"{overall['median']:>11} {overall['minimum']:>9} "
          f"{overall['maximum']:>9}")

    for name in ALL_SETS:
        members, source = sets[name]
        vals = [_to_int(r["cavity_frames"]) for r in members]
        d = describe(vals, frame_duration, unit)
        d.update(group=name, source=source, unit=unit, run=run,
                 channel_type=channel_type)
        summary_rows.append(d)
        if d["n"]:
            print(f"    {name:<14} {d['n']:>4} {d['mean']:>11} "
                  f"{d['median']:>11} {d['minimum']:>9} {d['maximum']:>9}")

    write_rows(summary_rows, out_dir / f"cavity_{unit}_summary.csv",
               ["group", "source", "unit", "run", "channel_type", "n",
                "mean", "median", "std", "minimum", "maximum",
                "q1", "q3", "iqr", "low_fence", "high_fence"])

    write_rows(rows, out_dir / "ion_groups.csv")

    true_outliers = low_out + high_out
    if true_outliers:
        write_rows(true_outliers, out_dir / "outliers.csv")
    else:
        print("    [info] no true outliers — extreme sets fell back to quartiles")

    pct_rows = []
    for name in ALL_SETS:
        info = set_stats[name]
        if info["n"] == 0:
            continue
        for c in res_cols:
            label = c[len("res_"):]
            mean_pct, median_pct = info["stats"][label]
            pct_rows.append(dict(group=name, source=info["source"],
                                 n_ions=info["n"], residue=label,
                                 mean_pct=mean_pct, median_pct=median_pct,
                                 run=run, channel_type=channel_type))
    write_rows(pct_rows, out_dir / "residue_percentages_by_group.csv",
               ["group", "source", "n_ions", "residue", "mean_pct",
                "median_pct", "run", "channel_type"])

    # mobility: in contact with anything vs touching nothing
    mob_rows = mobility_stats(sets, frame_duration, unit)
    if mob_rows:
        write_rows(mob_rows, out_dir / f"mobility_{unit}.csv",
                   ["group", "source", "n_ions", "unit",
                    "mean_any", "median_any", "mean_no", "median_no",
                    "mean_pct_any", "median_pct_any",
                    "mean_pct_no", "median_pct_no"])
        print(f"    mobility ({unit}):")
        for m in mob_rows:
            print(f"      {m['group']:<14} n={m['n_ions']:<3} "
                  f"in contact {m['mean_pct_any']:>5.1f}% "
                  f"(median {m['median_pct_any']:>5.1f}%)   "
                  f"free {m['mean_pct_no']:>5.1f}%   "
                  f"[{m['mean_any']:.1f} vs {m['mean_no']:.1f} {unit}]")
        plot_mobility(mob_rows, out_dir / f"mobility_{unit}.png",
                      channel_type, run, unit)
    else:
        print("    [info] no mobility columns in the table — "
              "re-run permeation_ion_table.py (version 2026-09-26c or later)")

    plot_boxplot(scaled, out_dir / f"cavity_{unit}_boxplot.png",
                 channel_type, run, unit)
    plot_histogram(scaled, cuts_scaled, out_dir / f"cavity_{unit}_hist.png",
                   channel_type, run, unit)
    plot_set_bars(set_stats, res_cols, out_dir, channel_type, run)
    plot_all_sets(set_stats, res_cols, out_dir / "residue_pct_all_groups.png",
                  channel_type, run)

    if do_clustering:
        run_clustering(rows, res_cols, values, out_dir / "clusters",
                       channel_type, run, frame_duration, unit, k_fixed=k_fixed)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_permeation_ion_stats(results_dir, channel_type="", frame_duration=0.075,
                             run="", force_rerun=False, exclude_censored=False,
                             do_clustering=True, k_fixed=None):
    results_dir = Path(results_dir)
    table_dir = results_dir / "permeation_ion_table"
    csv_path = table_dir / "permeation_ion_table_frames.csv"
    if not csv_path.is_file():
        print(f"  [skip] {csv_path} not found — run run_permeation_ion_table() first.")
        return None

    marker = table_dir / "stats_frames" / "cavity_frames_summary.csv"
    if marker.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {marker} already exists. Nothing recomputed.")
        print(f"  Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None

    print("\n" + "=" * 70)
    print("PERMEATION ION STATISTICS")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  reading: {csv_path}")

    rows = load_table(csv_path)
    res_cols = residue_columns(rows)
    print(f"  {len(rows)} ions, {len(res_cols)} residues: "
          f"{[c[4:] for c in res_cols]}")

    n_cens = sum(1 for r in rows if _is_true(r.get("start_censored")))
    if n_cens:
        print(f"  [note] {n_cens} ion(s) start_censored (residence is a LOWER "
              f"BOUND){' — excluded' if exclude_censored else ''}")
        if exclude_censored:
            rows = [r for r in rows if not _is_true(r.get("start_censored"))]

    n_inc = sum(1 for r in rows if _is_true(r.get("incomplete_range")))
    if n_inc:
        print(f"  [warn] {n_inc} ion(s) have incomplete_range=True — contact "
              f"counts truncated, percentages understate.")

    values = [_to_int(r["cavity_frames"]) for r in rows]
    cuts = compute_cuts(values)
    print(f"\n  cavity_frames: min={cuts['minimum']:.0f}  Q1={cuts['q1']:.1f}  "
          f"median={cuts['median']:.1f}  Q3={cuts['q3']:.1f}  "
          f"max={cuts['maximum']:.0f}  IQR={cuts['iqr']:.1f}")
    print(f"  fences: low={cuts['lo_fence']:.1f}  high={cuts['hi_fence']:.1f}")
    if cuts["lo_fence"] < 0:
        print("  [note] low fence is negative — low outliers cannot exist; "
              "low_extreme falls back to below_Q1")

    for r, v in zip(rows, values):
        r["duration_group"] = assign_group(v, cuts)
        r["cavity_ns"] = round(v * frame_duration, 4)

    add_percentages(rows, res_cols)
    sets, low_out, high_out = build_sets(rows, values, cuts)
    set_stats = set_percentage_stats(sets, res_cols)

    print()
    for name in ALL_SETS:
        info = set_stats[name]
        if info["n"]:
            print(f"  {name:<14} n={info['n']:<4} ({info['source']})")

    for unit in ("frames", "ns"):
        analyse_unit(rows, res_cols, values, cuts, sets, set_stats,
                     table_dir / f"stats_{unit}", channel_type, run,
                     frame_duration, unit, low_out, high_out,
                     do_clustering=do_clustering, k_fixed=k_fixed)

    print(f"\n✓ Done. Outputs in {table_dir}/stats_frames/ and "
          f"{table_dir}/stats_ns/")
    return dict(rows=rows, cuts=cuts, sets=sets, set_stats=set_stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--channel_type", default="")
    ap.add_argument("--frame_duration", type=float, default=0.075)
    ap.add_argument("--run", default="")
    ap.add_argument("--force_rerun", action="store_true")
    ap.add_argument("--exclude_censored", action="store_true")
    ap.add_argument("--no_clustering", action="store_true")
    ap.add_argument("--k", type=int, default=None,
                    help="Fix the number of clusters instead of choosing by silhouette")
    args = ap.parse_args()

    run_permeation_ion_stats(
        results_dir=args.results_dir,
        channel_type=args.channel_type,
        frame_duration=args.frame_duration,
        run=args.run,
        force_rerun=args.force_rerun,
        exclude_censored=args.exclude_censored,
        do_clustering=not args.no_clustering,
        k_fixed=args.k,
    )


if __name__ == "__main__":
    main()