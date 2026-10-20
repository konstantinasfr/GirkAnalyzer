#!/usr/bin/env python3
"""
aggregate_cavity_water_regions.py
=================================
Pools every run's `cavity_water_regions_summary.csv` and asks one question:

    does a run with more water in the cavity permeate more ions?

One dot per RUN. Nine scatter plots — three groups x three regions:

    x = that run's mean water count for (group, region)
          group = all     : mean over every frame of the run
          group = end_2   : mean over the frames ions left the cavity
          group = end_3   : mean over the frames ions passed the gate
    y = that run's number of permeations
          = n_frames of the end_3 row (one end_3 frame per permeation event)

Spearman rho (rank-based, robust to one outlier run) and Pearson r are
printed and written for each panel, with a least-squares line drawn when
there are at least three runs.

FIGURE SIZE AND TEXT SIZE
-------------------------
What matters when a figure is dropped into a slide is the RATIO of text size
to figure size, since the whole image is then scaled to fit. Two flags
control it:

    --fontscale   multiplies every font size      (default 1.4)
    --figscale    multiplies every figure size    (default 0.85)

The defaults make the text about 65% larger relative to the canvas than
before, so a figure shrunk to half a slide still reads. Pass
`--fontscale 1 --figscale 1` for the old appearance, or push --fontscale to
1.8-2.0 for a figure that will be shown small.

WATCH THE Y-AXIS
----------------
The raw permeation count is not length-normalised: a longer run permeates
more ions for free. The CSV therefore carries both

    n_permeations        raw count
    permeation_rate      count / (n_frames_all * frame_duration)  [per ns]

and `--y rate` switches the plots to the rate. If your runs are all the same
length the two are proportional and it makes no difference; if they are not,
the rate is the honest axis.

OUTPUTS   <channel>/water_analysis/cavity_water_regions_across_runs/
--------------------------------------------------------------------
    water_vs_permeations_<CH>.csv        one row per run x group x region
    water_vs_permeations_runs_<CH>.csv   one row per run (wide, for a glance)
    water_vs_permeations_stats_<CH>.csv  rho / r / p / n for all nine panels
    water_vs_permeations_<CH>_<group>_<region>.png    the nine scatters
    water_vs_permeations_grid_<CH>.png                all nine on one sheet

USAGE
-----
    python3 aggregate_cavity_water_regions.py \
        /media/ziyue/.../girk_analyser_results/G12 --channel G12

    # bigger text still, for a figure that will be shown small
    python3 aggregate_cavity_water_regions.py .../G12 --channel G12 \
        --fontscale 1.8 --figscale 0.8

    # both channels at once (each writes under its own folder)
    python3 aggregate_cavity_water_regions.py \
        /media/ziyue/.../girk_analyser_results/G12 \
        /media/ziyue/.../girk_analyser_results/G2

    # per-ns rate instead of raw count
    python3 aggregate_cavity_water_regions.py .../G12 --channel G12 --y rate
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-14a  (+ fontscale / figscale)"

INPUT_NAME = "cavity_water_regions_summary.csv"
FRAME_NAME = "cavity_water_regions_per_frame.csv"
REGION_COL = {"GLU/ASN": "n_glu_asn", "ASN/HBC": "n_asn_hbc",
              "GLU/HBC": "n_glu_hbc"}
POOL_GROUPS = ["all", "baseline", "end_2", "end_3"]
POOL_LABEL = {"all": "all frames",
              "baseline": "baseline (time-matched)",
              "end_2": "end_2 (K⁺ leaves the cavity)",
              "end_3": "end_3 (K⁺ past the gate)"}
POOL_COLOR = {"all": "#90A4AE", "baseline": "#5D6D7E",
              "end_2": "#E67E22", "end_3": "#C0392B"}
WATER_ROOT = "water_analysis"
MODULE_DIR = "cavity_water_regions_across_runs"

GROUPS = ["all", "end_2", "end_3"]
REGIONS = ["GLU/ASN", "ASN/HBC", "GLU/HBC"]

GROUP_LABEL = {"all": "all frames",
               "end_2": "end_2 frames (K⁺ leaves the cavity)",
               "end_3": "end_3 frames (K⁺ past the gate)"}
GROUP_COLOR = {"all": "#90A4AE", "end_2": "#E67E22", "end_3": "#C0392B"}
CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22", "G12_E141Q": "#2980B9",
                 "G12_D173N": "#E67E22",
                 "G12_ML": "#27AE60", "G12_GAT": "#8E44AD"}


# ─────────────────────────────────────────────────────────────────────────────
#  Text and figure scaling
#
#  Every fontsize in this file goes through fs() and every figsize through
#  figsize(), so the two flags reach all plots without a per-plot edit. What
#  a slide actually shows is the ratio between the two: raising --fontscale
#  and lowering --figscale both make the text larger once the image has been
#  shrunk to fit.
# ─────────────────────────────────────────────────────────────────────────────

SCALE = dict(font=1.4, fig=0.85)


def fs(size):
    """Scaled font size."""
    return float(size) * SCALE["font"]


def figsize(w, h):
    """Scaled figure size, with a floor so a panel never collapses."""
    k = SCALE["fig"]
    return (max(3.0, w * k), max(2.4, h * k))


def apply_rc():
    """
    Defaults for anything not given an explicit fontsize (tick labels, the
    offset text on an axis, the legend). Set once, after the flags are read.
    """
    plt.rcParams.update({
        "font.size": fs(10),
        "axes.titlesize": fs(11),
        "axes.labelsize": fs(10),
        "xtick.labelsize": fs(9.5),
        "ytick.labelsize": fs(9.5),
        "legend.fontsize": fs(9),
        "figure.titlesize": fs(13),
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight",      # keeps larger text from being clipped
        "savefig.pad_inches": 0.08,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        def rank(a):
            o = a.argsort()
            r = np.empty(len(a), float)
            r[o] = np.arange(1, len(a) + 1)
            for v in np.unique(a):
                i = np.where(a == v)[0]
                if i.size > 1:
                    r[i] = r[i].mean()
            return r
        return float(np.corrcoef(rank(x), rank(y))[0, 1]), float("nan")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")


def _fmt(v, nd=4):
    return "" if v != v else round(float(v), nd)


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

# Where the summary CSV actually lives, relative to a channel folder or a RUN
# folder. These are checked FIRST and are bounded-depth, so the search never
# walks the trajectories. A full rglob is only used as a last resort, because
# on an external drive that can take minutes just to find a few small files.
# The layout is known: <channel>/RUN<number>/water_analysis/<module>/<csv>.
# Globbing RUN* rather than * means non-RUN siblings are never even listed.
RUN_GLOB = "RUN*"
FAST_PATTERNS = [
    f"{RUN_GLOB}/{WATER_ROOT}/*/{INPUT_NAME}",   # from a channel folder
    f"{WATER_ROOT}/*/{INPUT_NAME}",              # from one RUN folder
    f"*/{RUN_GLOB}/{WATER_ROOT}/*/{INPUT_NAME}", # from the results root
]


def expand_runs(spec):
    """
    "3", "3,5,14", "3 5 14", "RUN3", "1-5" -> ["RUN3", "RUN5", "RUN14"] etc.
    Used by --runs to build exact paths and skip directory listing entirely.
    """
    names = []
    for token in str(spec).replace(",", " ").split():
        t = token.strip()
        if not t:
            continue
        body = t[3:] if t.upper().startswith("RUN") else t
        if "-" in body and all(part.isdigit() for part in body.split("-", 1)):
            a, b = body.split("-", 1)
            names.extend(f"RUN{i}" for i in range(int(a), int(b) + 1))
        else:
            names.append(t if t.upper().startswith("RUN") else f"RUN{t}")
    return names


def find_csvs(inputs, deep=False, runs=None, pattern=None):
    """
    [(csv_path, search_root)].

    Fastest first:
      runs=[...]   exact paths, no directory listing at all
      pattern=...  one caller-supplied relative glob
      otherwise    the bounded FAST_PATTERNS above
      deep=True    full rglob, last resort and slow on large trees
    """
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() == ".csv":
            found.append((p, p.resolve().parent))
            continue
        if not p.is_dir():
            print(f"  [warn] not found, skipped: {p}")
            continue

        root = p.resolve()
        hits = []

        if runs:
            # No globbing: the path of every file is known up front.
            for name in runs:
                direct = p / name / WATER_ROOT / "cavity_water_regions" / INPUT_NAME
                if direct.is_file():
                    hits.append(direct)
                    continue
                # module subfolder renamed? one cheap glob inside that run only
                alt = sorted((p / name / WATER_ROOT).glob(f"*/{INPUT_NAME}"))
                if alt:
                    hits.extend(alt)
                else:
                    print(f"  [warn] no {INPUT_NAME} for {name} under {p}")
            found.extend((f, root) for f in sorted(hits))
            continue

        for pat in ([pattern] if pattern else FAST_PATTERNS):
            hits.extend(p.glob(pat))
        if not hits and deep:
            print(f"  [info] nothing at the expected depth under {p} — "
                  f"falling back to a full recursive search (slow)...")
            hits = list(p.rglob(INPUT_NAME))
        elif not hits:
            print(f"  [warn] no {INPUT_NAME} at the expected depth under {p}\n"
                  f"         expected <channel>/RUNnn/{WATER_ROOT}/"
                  f"cavity_water_regions/{INPUT_NAME}\n"
                  f"         re-run with --deep to search the whole tree")
        found.extend((f, root) for f in sorted(hits))

    seen, unique = set(), []
    for f, r in found:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append((f, r))
    return unique


def run_label(path, stated_run):
    """
    Prefer the run column; otherwise pull RUNnn out of the path, so dots can
    be identified even when the module was run without --run.
    """
    if str(stated_run).strip():
        s = str(stated_run).strip()
        return s if s.upper().startswith("RUN") else f"RUN{s}"
    for part in reversed(path.resolve().parts):
        if part.upper().startswith("RUN"):
            return part
    return path.parent.name


def load_run(path, channel_override=""):
    """
    One run -> dict with the nine means plus the permeation count.
    Returns None if the file is missing rows the plots need.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  [warn] empty: {path}")
        return None

    ch = channel_override or str(rows[0].get("channel_type", "")).strip()
    run = run_label(path, rows[0].get("run", ""))
    fd = float(rows[0].get("frame_duration_ns") or 0) or None

    means, nframes = {}, {}
    for r in rows:
        key = (str(r.get("group", "")).strip(), str(r.get("region", "")).strip())
        try:
            means[key] = float(r["mean"])
            nframes[key] = int(float(r["n_frames"]))
        except (KeyError, TypeError, ValueError):
            continue

    # number of permeations = how many end_3 frames this run had
    n_perm = None
    for region in REGIONS:
        if ("end_3", region) in nframes:
            n_perm = nframes[("end_3", region)]
            break
    if n_perm is None:
        print(f"  [warn] no end_3 rows (no permeations recorded?): {path}")
        return None

    n_all = next((nframes[("all", reg)] for reg in REGIONS
                  if ("all", reg) in nframes), None)
    rate = ""
    if n_all and fd:
        rate = round(n_perm / (n_all * fd), 5)

    return dict(channel_type=ch or "(unlabelled)", run=run, source=str(path),
                n_permeations=n_perm, n_frames_all=n_all or "",
                frame_duration_ns=fd or "", permeation_rate=rate,
                means=means, nframes=nframes)


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def long_rows(runs, ykey):
    out = []
    for d in runs:
        for group in GROUPS:
            for region in REGIONS:
                m = d["means"].get((group, region))
                if m is None:
                    continue
                out.append(dict(
                    channel_type=d["channel_type"], run=d["run"],
                    group=group, region=region, mean_waters=m,
                    n_frames_group=d["nframes"].get((group, region), ""),
                    n_permeations=d["n_permeations"],
                    n_frames_all=d["n_frames_all"],
                    frame_duration_ns=d["frame_duration_ns"],
                    permeation_rate=d["permeation_rate"],
                    y_plotted=d[ykey], source=d["source"]))
    return out


def wide_rows(runs):
    out = []
    for d in runs:
        row = dict(channel_type=d["channel_type"], run=d["run"],
                   n_permeations=d["n_permeations"],
                   n_frames_all=d["n_frames_all"],
                   frame_duration_ns=d["frame_duration_ns"],
                   permeation_rate=d["permeation_rate"])
        for group in GROUPS:
            for region in REGIONS:
                tag = f"{group}_{region.replace('/', '_')}"
                row[f"mean_{tag}"] = d["means"].get((group, region), "")
        out.append(row)
    return out


def write_csv(rows, path):
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


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _panel(ax, x, y, labels, color, ylabel, xlabel, annotate=True):
    ax.scatter(x, y, s=55, color=color, alpha=0.8, edgecolors="black",
               linewidths=0.6, zorder=3)
    if annotate:
        for xi, yi, lb in zip(x, y, labels):
            ax.annotate(lb, (xi, yi), fontsize=fs(7.5), alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
    rho, p_rho = spearman(x, y)
    r, p_r = pearson(x, y)
    if len(x) >= 3 and np.unique(x).size > 1:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color="black", linewidth=1.2,
                alpha=0.7, zorder=2)
    ax.set_xlabel(xlabel, fontsize=fs(10))
    ax.set_ylabel(ylabel, fontsize=fs(10))
    ax.tick_params(labelsize=fs(9.5))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return rho, p_rho, r, p_r


def plot_one(x, y, labels, group, region, ylabel, out_path, title, color):
    fig, ax = plt.subplots(figsize=figsize(6.2, 4.8))
    # The group goes in the title, not the x label: at a large --fontscale a
    # one-line label like "Mean waters in GLU/ASN — end_3 frames (K+ past the
    # gate)" is wider than the canvas and gets cropped.
    rho, p_rho, r, p_r = _panel(
        ax, x, y, labels, color, ylabel, f"Mean waters — {region}")
    sub = f"Spearman ρ = {rho:.3f}"
    if p_rho == p_rho:
        sub += f" (p = {p_rho:.3g})"
    sub += f",  n = {len(x)} runs"
    ax.set_title(f"{title}\n{GROUP_LABEL[group]}\n{sub}",
                 fontsize=fs(10.5), fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")
    return rho, p_rho, r, p_r


def plot_grid(runs, ykey, ylabel, out_path, title):
    fig, axes = plt.subplots(len(GROUPS), len(REGIONS),
                             figsize=figsize(15, 12))
    for i, group in enumerate(GROUPS):
        for j, region in enumerate(REGIONS):
            ax = axes[i][j]
            pts = [(d["means"][(group, region)], d[ykey], d["run"])
                   for d in runs if (group, region) in d["means"]]
            if len(pts) < 2:
                ax.set_axis_off()
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            lb = [p[2] for p in pts]
            rho, p_rho, _, _ = _panel(ax, x, y, lb, GROUP_COLOR[group],
                                      ylabel, f"Mean waters — {region}",
                                      annotate=False)
            ax.set_title(f"{group} / {region}   ρ={rho:.2f}"
                         + (f"  p={p_rho:.3g}" if p_rho == p_rho else ""),
                         fontsize=fs(10), fontweight="bold")
    fig.suptitle(title, fontsize=fs(14), fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.962])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  POOLED MODE — every run's frames treated as one long simulation
#
#  Reads cavity_water_regions_per_frame.csv rather than the summary, so the
#  group means can be recomputed over the concatenated frames. Two weightings
#  are produced, because they answer different questions:
#
#    pooled        every frame counts once. A run with more frames, or more
#                  permeations, contributes more. This is the right default
#                  for the event groups -- a run with 4 events should not
#                  carry the same weight as one with 37.
#    mean-of-means every RUN counts once. Useful as a check: if the two
#                  disagree, one run is driving the pooled answer.
#
#  THE BASELINE GROUP
#  Water keeps rising for the first ~1000 frames while the cavity fills, and
#  no ion permeates during that period, so comparing event frames against
#  ALL frames partly measures the fill-up. "baseline" restricts the
#  comparison frames to the window where permeation actually happened --
#  from the run's first event frame onward by default -- and excludes the
#  event frames themselves, so the two sides never overlap.
# ─────────────────────────────────────────────────────────────────────────────

def frame_path_for(summary_path):
    """The per-frame CSV sits next to the summary it was written with."""
    return Path(summary_path).parent / FRAME_NAME


def load_frames(path, baseline="after_first_event", skip=0):
    """
    One run's per-frame CSV -> dict of arrays plus the group masks.
    Returns None if the file is missing or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        return None
    frames, cols, e2, e3 = [], {r: [] for r in REGIONS}, [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fr = int(float(row["frame"]))
            except (KeyError, TypeError, ValueError):
                continue
            frames.append(fr)
            for region in REGIONS:
                cols[region].append(float(row[REGION_COL[region]]))
            e2.append(str(row.get("is_end_2", "")).strip().lower() == "true")
            e3.append(str(row.get("is_end_3", "")).strip().lower() == "true")
    if not frames:
        return None

    frames = np.array(frames)
    data = {r: np.array(v, dtype=float) for r, v in cols.items()}
    e2 = np.array(e2, dtype=bool)
    e3 = np.array(e3, dtype=bool)

    # baseline window
    event_frames = frames[e2 | e3]
    if baseline == "all":
        start = frames.min()
    elif baseline.startswith("skip"):
        start = frames.min() + int(skip)
    else:  # after_first_event
        start = int(event_frames.min()) if event_frames.size else frames.min()
    in_window = frames >= start
    base_mask = in_window & ~e2 & ~e3      # never overlap the event frames

    return dict(path=str(path), frames=frames, data=data,
                mask_all=np.ones(frames.shape, dtype=bool),
                mask_baseline=base_mask, mask_end_2=e2, mask_end_3=e3,
                baseline_start=int(start), n_frames=int(frames.size))


def describe_vals(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return dict(n_frames=0, mean="", std="", sem="", median="",
                    minimum="", maximum="")
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return dict(n_frames=int(v.size), mean=round(float(v.mean()), 3),
                std=round(sd, 3), sem=round(sd / np.sqrt(v.size), 3),
                median=round(float(np.median(v)), 3),
                minimum=round(float(v.min()), 1),
                maximum=round(float(v.max()), 1))


def pool_values(loaded, group, region):
    """Concatenated values for one group/region across every run."""
    key = f"mask_{group}"
    return np.concatenate([d["data"][region][d[key]] for d in loaded
                           if d["data"][region][d[key]].size]) \
        if any(d["data"][region][d[f"mask_{group}"]].size for d in loaded) \
        else np.array([])


def permutation_p(event_vals, baseline_vals, n_iter=10000, seed=0):
    """
    How often does a random draw of the same size from the baseline reach a
    mean as extreme as the events'? Two-sided, no normality assumption, and
    it respects the real n rather than the inflated frame count.
    """
    if event_vals.size == 0 or baseline_vals.size < event_vals.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    obs = float(event_vals.mean()) - float(baseline_vals.mean())
    draws = rng.choice(baseline_vals, size=(n_iter, event_vals.size),
                       replace=True).mean(axis=1) - baseline_vals.mean()
    p = float((np.abs(draws) >= abs(obs)).mean())
    return obs, max(p, 1.0 / n_iter)


def groups_for(baseline):
    """
    With --baseline all the baseline window IS the whole run, so a separate
    "baseline" bar would just duplicate "all frames" and is left out.
    """
    return (["all", "end_2", "end_3"] if baseline == "all" else POOL_GROUPS)


def pooled_tables(loaded, channel, n_iter=10000, baseline="all"):
    """(summary rows, mean-of-means rows, test rows)."""
    summary, per_run, tests = [], [], []
    groups = groups_for(baseline)

    for group in groups:
        for region in REGIONS:
            vals = pool_values(loaded, group, region)
            summary.append(dict(channel_type=channel, weighting="pooled_frames",
                                group=group, region=region,
                                n_runs=len(loaded), **describe_vals(vals)))

    # mean of means: one number per run, then averaged
    for group in groups:
        for region in REGIONS:
            per = []
            for d in loaded:
                v = d["data"][region][d[f"mask_{group}"]]
                if v.size:
                    per.append(float(v.mean()))
            if not per:
                continue
            a = np.array(per)
            per_run.append(dict(
                channel_type=channel, weighting="mean_of_run_means",
                group=group, region=region, n_runs=int(a.size),
                mean=round(float(a.mean()), 3),
                std=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
                sem=round(float(a.std(ddof=1) / np.sqrt(a.size))
                          if a.size > 1 else 0.0, 3),
                median=round(float(np.median(a)), 3),
                minimum=round(float(a.min()), 3),
                maximum=round(float(a.max()), 3)))

    # events vs the time-matched baseline
    for group in ("end_2", "end_3"):
        for region in REGIONS:
            ev = pool_values(loaded, group, region)
            # "baseline" is the whole run minus the event frames when
            # --baseline all, so the two sides never share data either way.
            base = pool_values(loaded, "baseline", region)
            diff, p = permutation_p(ev, base, n_iter=n_iter)
            allv = pool_values(loaded, "all", region)
            diff_all = (float(ev.mean()) - float(allv.mean())
                        if ev.size and allv.size else float("nan"))
            d_eff = (diff / float(base.std(ddof=1))
                     if base.size > 1 and diff == diff else float("nan"))
            tests.append(dict(
                channel_type=channel, group=group, region=region,
                n_event_frames=int(ev.size), n_baseline_frames=int(base.size),
                event_mean=_fmt(float(ev.mean()) if ev.size else float("nan"), 3),
                baseline_mean=_fmt(float(base.mean()) if base.size else float("nan"), 3),
                all_frames_mean=_fmt(float(allv.mean()) if allv.size else float("nan"), 3),
                baseline_window=baseline,
                diff_vs_baseline=_fmt(diff, 3),
                diff_vs_all_frames=_fmt(diff_all, 3),
                cohens_d=_fmt(d_eff, 3),
                permutation_p=_fmt(p, 6), n_iter=n_iter))
    return summary, per_run, tests


def plot_pooled_bars(rows, group, out_path, title):
    sel = [r for r in rows if r["group"] == group and r["mean"] != ""]
    if not sel:
        return
    sel = sorted(sel, key=lambda r: REGIONS.index(r["region"]))
    labels = [r["region"] for r in sel]
    means = [r["mean"] for r in sel]
    stds = [r["std"] for r in sel]
    ns = [r["n_frames"] for r in sel]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize(6.8, 5))
    ax.bar(x, means, yerr=stds, width=0.55, capsize=7,
           color=POOL_COLOR.get(group, "#2980B9"), alpha=0.85,
           edgecolor="black", linewidth=0.8,
           error_kw={"linewidth": 1.5, "ecolor": "black"})
    top = max(m + s for m, s in zip(means, stds))
    for xi, m, sd, n in zip(x, means, stds, ns):
        ax.text(xi, m + sd + top * 0.03,
                f"mean={m:.2f}\nstd={sd:.2f}\nn={n:,}",
                ha="center", va="bottom", fontsize=fs(9))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs(11))
    ax.tick_params(axis="y", labelsize=fs(10))
    ax.set_ylabel("# water molecules", fontsize=fs(11))
    ax.set_ylim(0, top * 1.38)
    ax.set_title(title, fontsize=fs(12), fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_pooled_comparison(rows, out_path, title, groups=None):
    groups = groups or POOL_GROUPS
    groups = [g for g in groups
              if any(r["group"] == g and r["mean"] != "" for r in rows)]
    if not groups:
        return
    x = np.arange(len(REGIONS))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=figsize(9.5, 5.2))
    for i, g in enumerate(groups):
        means, errs, ns = [], [], []
        for region in REGIONS:
            r = next((q for q in rows if q["group"] == g
                      and q["region"] == region and q["mean"] != ""), None)
            means.append(r["mean"] if r else 0.0)
            errs.append(r["sem"] if r else 0.0)
            ns.append(r["n_frames"] if r else 0)
        pos = x + i * width - 0.4 + width / 2
        ax.bar(pos, means, width * 0.9, yerr=errs, capsize=4,
               color=POOL_COLOR.get(g, f"C{i}"), alpha=0.88,
               edgecolor="black", linewidth=0.7,
               error_kw={"linewidth": 1.2, "ecolor": "black"},
               label=f"{POOL_LABEL.get(g, g)}  (n={ns[0]:,})")
        for pp, m in zip(pos, means):
            ax.text(pp, m, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=fs(8.5))

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, fontsize=fs(11))
    ax.tick_params(axis="y", labelsize=fs(10))
    ax.set_ylabel("Mean # water molecules (± SEM)", fontsize=fs(11))
    ax.set_title(title, fontsize=fs(12), fontweight="bold")
    ax.legend(fontsize=fs(9))
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_weighting_check(pooled_rows, perrun_rows, out_path, title):
    """pooled vs mean-of-run-means, side by side — do they agree?"""
    groups = [g for g in POOL_GROUPS
              if any(r["group"] == g and r["mean"] != "" for r in pooled_rows)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(REGIONS), figsize=figsize(14, 4.6),
                             sharey=False)
    x = np.arange(len(groups))
    for ax, region in zip(np.atleast_1d(axes), REGIONS):
        a = [next((r["mean"] for r in pooled_rows if r["group"] == g
                   and r["region"] == region and r["mean"] != ""), 0.0)
             for g in groups]
        b = [next((r["mean"] for r in perrun_rows if r["group"] == g
                   and r["region"] == region and r["mean"] != ""), 0.0)
             for g in groups]
        berr = [next((r["sem"] for r in perrun_rows if r["group"] == g
                      and r["region"] == region and r["mean"] != ""), 0.0)
                for g in groups]
        ax.bar(x - 0.2, a, 0.38, color="#2980B9", alpha=0.85,
               edgecolor="black", linewidth=0.6, label="pooled frames")
        ax.bar(x + 0.2, b, 0.38, yerr=berr, capsize=4, color="#F39C12",
               alpha=0.85, edgecolor="black", linewidth=0.6,
               error_kw={"linewidth": 1.0, "ecolor": "black"},
               label="mean of run means (± SEM)")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=fs(9.5))
        ax.tick_params(axis="y", labelsize=fs(9.5))
        ax.set_title(region, fontsize=fs(11), fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("# water molecules", fontsize=fs(10.5))
    np.atleast_1d(axes)[0].legend(fontsize=fs(8.5))
    fig.suptitle(title, fontsize=fs(13), fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.925])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def run_pooled(sel, ch, out_dir, baseline, skip, n_iter):
    """Load every run's per-frame CSV, pool, write tables and plots."""
    loaded, missing = [], []
    for d in sel:
        fd = load_frames(frame_path_for(d["source"]), baseline, skip)
        if fd is None:
            missing.append(d["run"])
            continue
        fd["run"] = d["run"]
        loaded.append(fd)
    if missing:
        print(f"  [warn] no {FRAME_NAME} for: {', '.join(missing)}")
    if not loaded:
        print(f"  [skip] pooled mode: no per-frame CSVs found for {ch}")
        return

    total = sum(d["n_frames"] for d in loaded)
    note = ("all frames" if baseline == "all"
            else f"comparison window: {baseline}")
    print(f"\n  --- pooled over {len(loaded)} runs, {total:,} frames "
          f"({note}) ---")
    for d in loaded:
        n_e3 = int(d["mask_end_3"].sum())
        extra = ("" if baseline == "all"
                 else f", comparison from frame {d['baseline_start']:,}")
        print(f"    {d['run']:<8} {d['n_frames']:>6,} frames, "
              f"{n_e3:>3} permeations{extra}")

    summary, per_run, tests = pooled_tables(loaded, ch, n_iter=n_iter,
                                            baseline=baseline)
    write_csv(summary + per_run, out_dir / f"pooled_frames_summary_{ch}.csv")
    write_csv(tests, out_dir / f"pooled_frames_tests_{ch}.csv")

    print(f"\n  {'group':<9} {'region':<10} {'mean':>8} {'vs base':>9} "
          f"{'d':>6} {'perm p':>9} {'n':>7}")
    for t in tests:
        print(f"  {t['group']:<9} {t['region']:<10} {t['event_mean']:>8} "
              f"{t['diff_vs_baseline']:>+9} {t['cohens_d']:>6} "
              f"{t['permutation_p']:>9} {t['n_event_frames']:>7,}")

    for group in groups_for(baseline):
        plot_pooled_bars(summary, group,
                         out_dir / f"pooled_frames_bars_{ch}_{group}.png",
                         f"{ch} — pooled waters per region, "
                         f"{POOL_LABEL.get(group, group)}")
    plot_pooled_comparison(summary,
                           out_dir / f"pooled_frames_comparison_{ch}.png",
                           f"{ch} — pooled over {len(loaded)} runs "
                           f"({total:,} frames)",
                           groups=groups_for(baseline))
    if baseline != "all":
        plot_pooled_comparison(
            summary, out_dir / f"pooled_frames_comparison_matched_{ch}.png",
            f"{ch} — events vs restricted baseline ({len(loaded)} runs)",
            groups=["baseline", "end_2", "end_3"])
    plot_weighting_check(summary, per_run,
                         out_dir / f"pooled_frames_weighting_check_{ch}.png",
                         f"{ch} — pooled frames vs mean of run means")


# ─────────────────────────────────────────────────────────────────────────────

def channel_output_dir(root, channel, subdir):
    root = Path(root)
    if root.name == channel:
        base = root
    elif (root / channel).is_dir():
        base = root / channel
    else:
        base = root
    return base / WATER_ROOT / subdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Channel folders, RUN folders, or direct CSV paths")
    ap.add_argument("-channel", "--channel", dest="channel", default="",
                    help="Channel type (G12, G2, ...). Forces the label, names "
                         "the files and title, and picks the output folder.")
    ap.add_argument("-subdir", "--subdir", default=MODULE_DIR,
                    help=f"Folder inside <channel>/water_analysis/ "
                         f"(default: {MODULE_DIR})")
    ap.add_argument("-out", "--out", default=None,
                    help="Write everything here instead of per channel")
    ap.add_argument("-y", "--y", choices=["count", "rate"], default="count",
                    help="y axis: raw permeation count (default) or per-ns rate")
    ap.add_argument("-fontscale", "--fontscale", type=float, default=1.4,
                    help="Multiply every font size (default 1.4; use 1 for the "
                         "old look, 1.8-2.0 for a figure shown small)")
    ap.add_argument("-figscale", "--figscale", type=float, default=0.85,
                    help="Multiply every figure size (default 0.85). Smaller "
                         "canvas with the same fonts = larger text once the "
                         "image is scaled into a slide.")
    ap.add_argument("-mode", "--mode",
                    choices=["scatter", "pooled", "both"], default="both",
                    help="scatter = per-run water vs permeations (9 panels); "
                         "pooled = all runs as one long simulation, rebuilding "
                         "the per-run bar plots on pooled frames; both (default)")
    ap.add_argument("-baseline", "--baseline", default="all",
                    help="Comparison window for the pooled test: all frames "
                         "(default), after_first_event to restrict each run to "
                         "its post-first-event window, or skip:N to drop the "
                         "first N frames of each run")
    ap.add_argument("-n_iter", "--n_iter", type=int, default=10000,
                    help="Permutation iterations for the pooled test")
    ap.add_argument("-runs", "--runs", default="",
                    help="Only these runs, e.g. --runs 3,5,14 or --runs 1-20 "
                         "or --runs RUN3. Builds exact paths, so nothing is "
                         "searched at all — the fastest option.")
    ap.add_argument("-glob", "--glob", dest="pattern", default="",
                    help="Custom relative glob instead of the built-in ones, "
                         f"e.g. 'RUN*/{WATER_ROOT}/*/{INPUT_NAME}'")
    ap.add_argument("-deep", "--deep", action="store_true",
                    help="Fall back to a full recursive search if nothing is "
                         "found at the expected depth (slow on big trees)")
    ap.add_argument("-no_labels", "--no_labels", action="store_true",
                    help="Do not annotate each dot with its run name")
    args = ap.parse_args()

    if args.fontscale <= 0 or args.figscale <= 0:
        sys.exit("ERROR: --fontscale and --figscale must be positive")
    SCALE["font"] = args.fontscale
    SCALE["fig"] = args.figscale
    apply_rc()

    print("=" * 70)
    print("AGGREGATE CAVITY WATER REGIONS — water vs permeations, per run")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  text: fontscale={args.fontscale:g}, figscale={args.figscale:g} "
          f"(effective text-to-figure ratio "
          f"{args.fontscale / args.figscale:.2f}x the original)")
    if args.channel:
        print(f"  channel: {args.channel}  (forced on every input row)")

    runs = expand_runs(args.runs) if args.runs else None
    if runs:
        print(f"  runs: {', '.join(runs)}  (exact paths, no search)", flush=True)
    elif args.pattern:
        print(f"  glob: {args.pattern}", flush=True)
    else:
        print("  searching for summary CSVs...", flush=True)
    csvs = find_csvs(args.inputs, deep=args.deep, runs=runs,
                     pattern=args.pattern or None)
    if not csvs:
        sys.exit(f"ERROR: no {INPUT_NAME} found under the given paths.")
    print(f"  Found {len(csvs)} input file(s)")

    runs = []
    for path, root in csvs:
        d = load_run(path, args.channel)
        if d:
            d["_root"] = root
            runs.append(d)
            print(f"    {d['channel_type']:<12} {d['run']:<8} "
                  f"{d['n_permeations']:>4} permeations   {path}")
    if not runs:
        sys.exit("ERROR: no usable runs loaded.")

    ykey = "n_permeations" if args.y == "count" else "permeation_rate"
    ylabel = ("Number of permeations" if args.y == "count"
              else "Permeation rate (ions / ns)")
    if args.y == "rate":
        bad = [d["run"] for d in runs if d["permeation_rate"] == ""]
        if bad:
            sys.exit(f"ERROR: --y rate needs n_frames_all and frame_duration_ns, "
                     f"missing for: {', '.join(bad)}")

    channels = []
    for d in runs:
        if d["channel_type"] not in channels:
            channels.append(d["channel_type"])

    baseline, skip = args.baseline, 0
    if baseline.startswith("skip"):
        parts = baseline.split(":", 1)
        skip = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        baseline = "skip"
    elif baseline not in ("all", "after_first_event"):
        sys.exit(f"ERROR: unknown --baseline '{args.baseline}' "
                 f"(use all, after_first_event, or skip:N)")

    forced = Path(args.out).resolve() if args.out else None
    if forced:
        forced.mkdir(parents=True, exist_ok=True)

    for ch in channels:
        sel = [d for d in runs if d["channel_type"] == ch]
        if forced:
            out_dir = forced
        else:
            roots = [str(d["_root"]) for d in sel]
            out_dir = channel_output_dir(max(set(roots), key=roots.count),
                                         ch, args.subdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ys = [d[ykey] for d in sel]
        print(f"\n  === {ch} — {len(sel)} runs, "
              f"{sum(d['n_permeations'] for d in sel)} permeations total ===")
        print(f"  y axis: {ylabel}  (range {min(ys)} – {max(ys)})")

        if args.mode == "pooled":
            run_pooled(sel, ch, out_dir, baseline, skip, args.n_iter)
            print(f"\n  -> {out_dir}/")
            continue

        write_csv(long_rows(sel, ykey), out_dir / f"water_vs_permeations_{ch}.csv")
        write_csv(wide_rows(sel), out_dir / f"water_vs_permeations_runs_{ch}.csv")

        stats_rows = []
        print(f"\n  {'group':<7} {'region':<10} {'rho':>7} {'p(rho)':>10} "
              f"{'r':>7} {'n':>4}")
        for group in GROUPS:
            for region in REGIONS:
                pts = [(d["means"][(group, region)], d[ykey], d["run"])
                       for d in sel if (group, region) in d["means"]]
                if len(pts) < 2:
                    print(f"  {group:<7} {region:<10}   only {len(pts)} run(s)"
                          f" — skipped")
                    continue
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                lb = [p[2] for p in pts]
                fname = (f"water_vs_permeations_{ch}_{group}_"
                         f"{region.replace('/', '_')}.png")
                rho, p_rho, r, p_r = plot_one(
                    x, y, [] if args.no_labels else lb, group, region, ylabel,
                    out_dir / fname,
                    f"{ch} — {ylabel.lower()} vs cavity water",
                    CHANNEL_COLOR.get(ch, GROUP_COLOR[group]))
                print(f"  {group:<7} {region:<10} {rho:>7.3f} "
                      f"{(f'{p_rho:.3g}' if p_rho == p_rho else '-'):>10} "
                      f"{r:>7.3f} {len(x):>4}")
                stats_rows.append(dict(
                    channel_type=ch, group=group, region=region,
                    y_axis=args.y, n_runs=len(x),
                    spearman_rho=_fmt(rho), spearman_p=_fmt(p_rho, 6),
                    pearson_r=_fmt(r), pearson_p=_fmt(p_r, 6),
                    x_min=round(min(x), 3), x_max=round(max(x), 3),
                    y_min=min(y), y_max=max(y)))

        write_csv(stats_rows, out_dir / f"water_vs_permeations_stats_{ch}.csv")
        plot_grid(sel, ykey, ylabel,
                  out_dir / f"water_vs_permeations_grid_{ch}.png",
                  f"{ch} — {ylabel.lower()} vs cavity water "
                  f"({len(sel)} runs, one dot per run)")

        if args.mode == "both":
            run_pooled(sel, ch, out_dir, baseline, skip, args.n_iter)

        print(f"\n  -> {out_dir}/")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()

'''
python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12

python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ --channel G2

python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ --channel G12_D173N

python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ --channel G12_E141Q

# bigger text still, for a figure that will be shown small in a slide
python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12 --fontscale 1.8 --figscale 0.8

# the old appearance
python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12 --fontscale 1 --figscale 1
'''

'''
python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12

python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2


python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--channel G12_D173N


python3 aggregate_cavity_water_regions.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--channel G12_E141Q
'''