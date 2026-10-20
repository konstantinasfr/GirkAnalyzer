#!/usr/bin/env python3
"""
aggregate_cavity_mobility.py
============================
Pools `cavity_mobility` output across runs: how far ions and waters move per
frame inside the GLU/HBC cavity.

FRAME SPACING AND NORMALISATION
-------------------------------
A step length is meaningless without the frame spacing it was measured at,
and the two channels differ (G12 0.075 ns, G2 0.4 ns). Three modes are
available via --normalise, and the chosen one is printed on every figure:

  none      raw Å per frame. Valid within one spacing only. This is the
            default, and the only mode that makes no assumption at all.

  per_ns    step / dt  ->  Å per ns. Assumes BALLISTIC motion (displacement
            linear in t), which a diffusing particle does not do. Included
            because it is the intuitive thing to ask for, but it will
            systematically understate the finer-spaced channel.

  sqrt_ns   step / sqrt(dt)  ->  Å per sqrt(ns). This is the diffusive
            scaling: for free 3D diffusion displacement grows as
            sqrt(6*D*t), so step/sqrt(dt) is spacing-independent and
            proportional to sqrt(D). Use this for the G12 vs G2 comparison.

  CAVEAT that applies to every mode: scaling only works while the particle is
  still diffusing freely over the interval. Free water travels ~10 A in 75 ps
  and ~24 A in 400 ps, both wider than the cavity, so water has SATURATED at
  both spacings and no scaling will make the two water numbers comparable.
  Ions are far more confined and are the safer comparison. The dimensionless
  ion/water step ratio within one run is spacing-robust by construction and
  is reported in per_run_mobility_<CH>.csv regardless of mode.

  The proper fix remains to stride the finer trajectory upstream in
  cavity_mobility.py so the spacings match (G12 every 5th frame -> 0.375 ns
  against G2's 0.4 ns).

--frame_duration overrides the per-run spacing read from
water_mobility_summary.csv, for runs whose summary is missing or wrong:

    --frame_duration 0.075    (G12 and its mutants)
    --frame_duration 0.4      (G2)

Without the override, spacing is read per run and runs at different spacings
are kept in separate groups rather than blended.

PERMEATION CORRELATIONS
-----------------------
Per-run permeation counts are read from <channel>/permeation_summary/, the
same file gate_openness.py uses. The permeation column is auto-detected,
since G12 and G2 label it differently (Full_Permeations_Unique vs
Final_Permeations). Each measure is drawn both in the three-panel figure and
as its own standalone PNG.

OUTPUTS   <channel>/water_analysis/cavity_mobility_across_runs/
---------------------------------------------------------------
    pooled_ions_<CH>.csv            every permeation event, all runs
    pooled_waters_<CH>.csv          every water visit, all runs
    pooled_summary_<CH>.csv         mean of means, pooled and per run
    per_run_mobility_<CH>.csv       one row per run per species

    pooled_box_<CH>.png             ions and waters together
    pooled_box_ions_<CH>.png        ions alone
    pooled_box_waters_<CH>.png      waters alone
    per_run_box_<CH>.png            one box per run, ions and waters
    pooled_hist_<CH>.png            distributions, both species
    water_residence_<CH>.png        pooled water residence times

    mobility_vs_permeations_<CH>.csv        per-run values + counts
    mobility_vs_permeations_<CH>.png        all three scatters together
    mean_ion_step_A_vs_permeations_<CH>.png          standalone
    mean_water_step_A_vs_permeations_<CH>.png        standalone
    mean_water_residence_ns_vs_permeations_<CH>.png  standalone
    mobility_vs_permeations_stats_<CH>.csv  rho and p per measure

USAGE
-----
    # within one channel, raw units
    python3 aggregate_cavity_mobility.py .../G12 --channel G12 \
        --frame_duration 0.075

    # for comparing G12 against G2, run both with the same normalisation
    python3 aggregate_cavity_mobility.py .../G12 --channel G12 \
        --frame_duration 0.075 --normalise sqrt_ns
    python3 aggregate_cavity_mobility.py .../G2  --channel G2 \
        --frame_duration 0.4   --normalise sqrt_ns
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-14a  (+ spacing override, normalisation, split plots)"

WATER_ROOT = "water_analysis"
MODULE_DIR = "cavity_mobility"
OUT_SUBDIR = "cavity_mobility_across_runs"

ION_NAME = "ion_mobility_per_ion.csv"
WATER_NAME = "water_mobility_per_visit.csv"

ION_COLOR = "#C0392B"
WATER_COLOR = "#2980B9"
RESIDENCE_COLOR = "#27AE60"
STEP_COL = "mean_step_A"

SPACING_MARKER = ["o", "s", "^", "D"]


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation
#
#  Held in one place so every table, axis label and caption reads the same
#  setting. Set once in main(), after the arguments are parsed.
# ─────────────────────────────────────────────────────────────────────────────

NORM = dict(mode="none", spacing={})

UNITS = {"none": "Å per frame",
         "per_ns": "Å per ns",
         "sqrt_ns": "Å per √ns"}

NORM_NOTE = {
    "none": "not normalised — do not compare across frame spacings",
    "per_ns": "scaled to Å per ns (step ÷ dt) — assumes ballistic motion",
    "sqrt_ns": "scaled to Å per √ns (step ÷ √dt) — diffusive scaling; "
               "invalid for saturated species",
}


def unit():
    return UNITS[NORM["mode"]]


def step_label():
    return f"Mean step ({unit()})"


def _scale(run):
    """Multiplier applied to one run's step lengths under the current mode."""
    dt = NORM["spacing"].get(run)
    if NORM["mode"] == "none" or not dt:
        return 1.0
    if NORM["mode"] == "per_ns":
        return 1.0 / float(dt)
    return 1.0 / np.sqrt(float(dt))


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def expand_runs(spec):
    names = []
    for token in str(spec).replace(",", " ").split():
        t = token.strip()
        if not t:
            continue
        body = t[3:] if t.upper().startswith("RUN") else t
        if "-" in body and all(p.isdigit() for p in body.split("-", 1)):
            a, b = body.split("-", 1)
            names.extend(f"RUN{i}" for i in range(int(a), int(b) + 1))
        else:
            names.append(t if t.upper().startswith("RUN") else f"RUN{t}")
    return names


def find_files(base, name, runs=None):
    base = Path(base)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    out = []
    names = runs if runs else sorted((d.name for d in base.glob("RUN*")
                                      if d.is_dir()), key=natural)
    for run in names:
        direct = base / run / WATER_ROOT / MODULE_DIR / name
        if direct.is_file():
            out.append((run, direct))
            continue
        alt = sorted((base / run / WATER_ROOT).glob(f"*/{name}"))
        if alt:
            out.append((run, alt[0]))
    return out


def natural(run):
    digits = "".join(c for c in run if c.isdigit())
    return (int(digits) if digits else 0, run)


def _f(row, key):
    v = str(row.get(key, "")).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load(base, name, runs, channel):
    rows = []
    for run, path in find_files(base, name, runs):
        with open(path, newline="") as f:
            got = list(csv.DictReader(f))
        for r in got:
            r["run_label"] = run
            r["channel_type"] = channel or r.get("channel_type", "")
        rows.extend(got)
        if got:
            print(f"    {run:<8} {len(got):>6} rows   {name}")
    return rows


def spacing_of(rows, base, runs, channel, override=None):
    """
    run -> frame spacing in ns, read from the per-run mobility summary.

    The spacing is a property of the trajectory output, not of the channel, so
    it is read per run rather than assumed -- a restrided run would otherwise
    be silently averaged in with the others. --frame_duration forces one value
    for every run, for sets whose summary files are missing or wrong.
    """
    labels = {r["run_label"] for r in rows}
    if override is not None:
        print(f"  frame spacing forced to {override:g} ns/frame for all "
              f"{len(labels)} run(s) (--frame_duration)")
        return {run: float(override) for run in labels}

    out = {}
    for run, path in find_files(base, "water_mobility_summary.csv", runs):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                fd = _f(r, "frame_duration_ns")
                if fd:
                    out[run] = fd
                    break
    missing = sorted(labels - set(out), key=natural)
    if missing:
        print(f"  [warn] no frame_duration_ns for: {', '.join(missing)} — "
              f"those runs are grouped as 'unknown' and left unscaled. "
              f"Pass --frame_duration to set it.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def describe(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return dict(n=0, mean="", median="", sd="", minimum="", maximum="")
    return dict(n=int(a.size),
                mean=round(float(a.mean()), 4),
                median=round(float(np.median(a)), 4),
                sd=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
                minimum=round(float(a.min()), 4),
                maximum=round(float(a.max()), 4))


def steps(rows):
    """Per-particle mean steps, scaled according to the current mode."""
    out = []
    for r in rows:
        v = _f(r, STEP_COL)
        if v is not None:
            out.append(v * _scale(r["run_label"]))
    return out


def per_run_table(irows, wrows, spacing, channel):
    out = []
    for species, rows in (("ions", irows), ("waters", wrows)):
        for run in sorted({r["run_label"] for r in rows}, key=natural):
            sel = [r for r in rows if r["run_label"] == run]
            d = describe(steps(sel))
            if d["n"] == 0:
                continue
            out.append(dict(channel_type=channel, run=run, species=species,
                            frame_duration_ns=spacing.get(run, ""),
                            normalisation=NORM["mode"], units=unit(),
                            n_particles=d.pop("n"), **d))
    # the dimensionless ratio: ion confinement against water in the same run,
    # at the same spacing -- the one number that survives a spacing change
    for run in sorted({r["run_label"] for r in irows}, key=natural):
        iv = describe(steps([r for r in irows if r["run_label"] == run]))
        wv = describe(steps([r for r in wrows if r["run_label"] == run]))
        if iv["n"] and wv["n"] and wv["mean"]:
            out.append(dict(channel_type=channel, run=run,
                            species="ion_to_water_ratio",
                            frame_duration_ns=spacing.get(run, ""),
                            normalisation=NORM["mode"],
                            units="dimensionless",
                            n_particles=iv["n"],
                            mean=round(iv["mean"] / wv["mean"], 4),
                            note="ion mean step / water mean step in the same "
                                 "run; unaffected by frame spacing"))
    return out


def pooled_table(irows, wrows, spacing, channel):
    out = []
    groups = {}
    for species, rows in (("ions", irows), ("waters", wrows)):
        for r in rows:
            fd = spacing.get(r["run_label"], "unknown")
            groups.setdefault((species, fd), []).append(r)
    for (species, fd), rows in groups.items():
        d = describe(steps(rows))
        runs = {r["run_label"] for r in rows}
        out.append(dict(channel_type=channel, species=species,
                        frame_duration_ns=fd, normalisation=NORM["mode"],
                        units=unit(), n_runs=len(runs),
                        n_particles=d.pop("n"), **d,
                        weighting="pooled_particles"))
        per_run = [describe(steps([r for r in rows
                                   if r["run_label"] == run]))["mean"]
                   for run in sorted(runs, key=natural)]
        per_run = [v for v in per_run if v != ""]
        if per_run:
            a = np.array(per_run, dtype=float)
            out.append(dict(channel_type=channel, species=species,
                            frame_duration_ns=fd, normalisation=NORM["mode"],
                            units=unit(), n_runs=len(runs),
                            n_particles=len(per_run),
                            mean=round(float(a.mean()), 4),
                            median=round(float(np.median(a)), 4),
                            sd=round(float(a.std(ddof=1))
                                     if a.size > 1 else 0.0, 4),
                            minimum=round(float(a.min()), 4),
                            maximum=round(float(a.max()), 4),
                            weighting="mean_of_run_means"))
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
#  Titles — the spacing and the normalisation go on every figure
# ─────────────────────────────────────────────────────────────────────────────

def spacing_tag(spacing, rows):
    """
    "0.075 ns/frame" when every run matches, or "mixed: 0.075, 0.4 ns/frame"
    when they do not — stated either way so a figure is never ambiguous.
    """
    vals = sorted({spacing.get(r["run_label"]) for r in rows
                   if spacing.get(r["run_label"])})
    if not vals:
        return "frame spacing unknown"
    if len(vals) == 1:
        return f"{vals[0]:g} ns/frame"
    return "mixed spacing: " + ", ".join(f"{v:g}" for v in vals) + " ns/frame"


def caption(spacing, rows):
    """Two short lines — a single long one gets clipped on narrow figures."""
    return (f"measured at {spacing_tag(spacing, rows)}\n"
            f"{NORM_NOTE[NORM['mode']]}")


def _titled(ax, title, sub, fontsize=11.5, sub_fontsize=8.5):
    """Bold title with a smaller, non-bold spacing note under it."""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=24)
    ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=sub_fontsize, color="#444444", linespacing=1.3)


# ─────────────────────────────────────────────────────────────────────────────
#  Box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pooled_box(groups, out_path, title, sub):
    """
    groups: list of (rows, label, colour). One entry gives a single box, two
    give the side-by-side comparison -- so the same function writes the
    combined figure and each species on its own.
    """
    data = []
    for rows, lab, c in groups:
        v = np.array(steps(rows), dtype=float)
        if v.size:
            data.append((v, f"{lab}\n(n={v.size:,})", c))
    if not data:
        print(f"    [skip] no values for {Path(out_path).name}")
        return

    fig, ax = plt.subplots(figsize=(4.9 if len(data) == 1 else 7, 5.4))
    bp = ax.boxplot([d for d, _, _ in data], widths=0.5, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for box, (_, _, c) in zip(bp["boxes"], data):
        box.set_facecolor(c)
        box.set_alpha(0.65)
    rng = np.random.default_rng(0)
    for i, (d, _, c) in enumerate(data, start=1):
        sample = d if d.size <= 400 else rng.choice(d, 400, replace=False)
        ax.scatter(i + rng.uniform(-0.11, 0.11, sample.size), sample, s=10,
                   color="#222222", alpha=0.3, zorder=3)
    top = max(d.max() for d, _, _ in data)
    for i, (d, _, _) in enumerate(data, start=1):
        ax.text(i, top * 1.02, f"mean {d.mean():.2f}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, top * 1.14)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([lab for _, lab, _ in data], fontsize=10)
    ax.set_ylabel(step_label(), fontsize=11)
    _titled(ax, title, sub)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_per_run_box(irows, wrows, spacing, out_path, title, sub):
    runs = sorted({r["run_label"] for r in irows} |
                  {r["run_label"] for r in wrows}, key=natural)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(runs) + 4), 5.6))
    width = 0.38
    for k, (rows, lab, c) in enumerate(((irows, "ions", ION_COLOR),
                                        (wrows, "waters", WATER_COLOR))):
        data, pos = [], []
        for i, run in enumerate(runs):
            v = np.array(steps([r for r in rows if r["run_label"] == run]),
                         dtype=float)
            if v.size:
                data.append(v)
                pos.append(i + (k - 0.5) * width)
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=width * 0.86,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.4))
        for box in bp["boxes"]:
            box.set_facecolor(c)
            box.set_alpha(0.65)
            box.set_edgecolor("black")
        ax.plot([], [], color=c, linewidth=8, alpha=0.65, label=lab)
    ax.set_xticks(range(len(runs)))
    labels = []
    for run in runs:
        fd = spacing.get(run)
        labels.append(f"{run}\n{fd:g} ns" if fd else run)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_ylabel(step_label(), fontsize=11)
    _titled(ax, title, sub)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_pooled_hist(irows, wrows, out_path, title, sub):
    iv = np.array(steps(irows), dtype=float)
    wv = np.array(steps(wrows), dtype=float)
    if iv.size == 0 and wv.size == 0:
        return
    allv = np.concatenate([v for v in (iv, wv) if v.size])
    bins = np.linspace(allv.min(), allv.max(), 50)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for v, lab, c in ((iv, "Ions", ION_COLOR), (wv, "Waters", WATER_COLOR)):
        if v.size:
            ax.hist(v, bins=bins, alpha=0.6, color=c, edgecolor="black",
                    linewidth=0.4, density=True,
                    label=f"{lab} (n={v.size:,}, mean {v.mean():.2f})")
    ax.set_xlabel(step_label(), fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    _titled(ax, title, sub)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_residence(wrows, out_path, title, sub):
    vals = np.array([_f(r, "residence_ns") for r in wrows
                     if _f(r, "residence_ns") is not None], dtype=float)
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(vals, bins=min(80, max(12, vals.size // 50)), color=WATER_COLOR,
            alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean {vals.mean():.3f} ns")
    ax.axvline(np.median(vals), color="black", linestyle=":", linewidth=1.5,
               label=f"median {np.median(vals):.3f} ns")
    ax.set_xlabel("Time a water stays in the cavity (ns)", fontsize=11)
    ax.set_ylabel("Visits (log)", fontsize=11)
    _titled(ax, title, sub)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Permeation counts and the correlations
# ─────────────────────────────────────────────────────────────────────────────

PERM_DIR = "permeation_summary"
PERM_COL_CANDIDATES = ["Full_Permeations_Unique", "Final_Permeations",
                       "HBC_Permeations", "Full_Permeations_Total",
                       "Permeations", "n_permeations"]


def load_permeations(base, perm_col=None):
    """
    {run: count} from <channel>/permeation_summary/*.csv.

    The column name differs between channels, so when the requested one is
    absent the first known alternative is used and reported.
    """
    folder = Path(base) / PERM_DIR
    if not folder.is_dir():
        print(f"  [warn] no {PERM_DIR}/ under {base} — "
              f"permeation correlations will be skipped")
        return {}, None
    for path in sorted(folder.glob("*.csv")):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "Run" not in rows[0]:
            continue
        chosen = perm_col if perm_col and perm_col in rows[0] else None
        if chosen is None:
            chosen = next((c for c in PERM_COL_CANDIDATES if c in rows[0]), None)
        if chosen is None:
            print(f"  [warn] no permeation column in {path.name}; available: "
                  f"{', '.join(k for k in rows[0] if k != 'Run')}")
            continue
        out = {}
        for r in rows:
            v = _f(r, chosen)
            if v is not None:
                out[str(r["Run"]).strip()] = v
        print(f"  Permeations from {path.name} "
              f"({len(out)} runs, column '{chosen}')")
        return out, chosen
    print(f"  [warn] no usable permeation summary in {folder}")
    return {}, None


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


def build_perm_table(irows, wrows, spacing, perms, channel):
    """One row per run: the three mobility means plus the permeation count."""
    runs = sorted({r["run_label"] for r in irows} |
                  {r["run_label"] for r in wrows}, key=natural)
    out = []
    for run in runs:
        isel = [r for r in irows if r["run_label"] == run]
        wsel = [r for r in wrows if r["run_label"] == run]
        res = [_f(r, "residence_ns") for r in wsel
               if _f(r, "residence_ns") is not None]
        di, dw = describe(steps(isel)), describe(steps(wsel))
        # standard error of each run's mean -- a run with 2 events has a far
        # noisier x than one with 58, and noise in x attenuates a correlation
        sem_i = (round(di["sd"] / np.sqrt(di["n"]), 4)
                 if di["n"] and di["sd"] != "" else "")
        sem_w = (round(dw["sd"] / np.sqrt(dw["n"]), 4)
                 if dw["n"] and dw["sd"] != "" else "")
        out.append(dict(
            channel_type=channel, run=run,
            frame_duration_ns=spacing.get(run, ""),
            normalisation=NORM["mode"], step_units=unit(),
            n_permeations=perms.get(run, ""),
            n_ion_events=di["n"], n_water_visits=dw["n"],
            mean_ion_step_A=di["mean"], sem_ion_step_A=sem_i,
            mean_water_step_A=dw["mean"], sem_water_step_A=sem_w,
            mean_water_residence_ns=(round(float(np.mean(res)), 4)
                                     if res else ""),
            median_water_residence_ns=(round(float(np.median(res)), 4)
                                       if res else ""),
            ion_to_water_ratio=(round(di["mean"] / dw["mean"], 4)
                                if di["mean"] != "" and dw["mean"] else "")))
    return out


def measures_list():
    """Built as a function, not a constant: the unit depends on --normalise."""
    u = unit()
    return [("mean_ion_step_A", f"Mean ion step ({u})",
             ION_COLOR, "sem_ion_step_A"),
            ("mean_water_step_A", f"Mean water step ({u})",
             WATER_COLOR, "sem_water_step_A"),
            ("mean_water_residence_ns", "Mean water residence (ns)",
             RESIDENCE_COLOR, None)]


def _scatter_panel(ax, usable, spacings, key, xlabel, color, sem_key,
                   show_err=False):
    """
    Draws one measure into one axis and returns its stats rows. Used for both
    the combined figure and the standalone ones, so the two can never drift
    apart.
    """
    stats_rows, any_pts = [], False
    for si, fd in enumerate(spacings or [""]):
        sel = [r for r in usable
               if (r["frame_duration_ns"] == fd or not spacings)
               and str(r.get(key, "")).strip() != ""]
        if not sel:
            continue
        x = np.array([float(r[key]) for r in sel])
        y = np.array([float(r["n_permeations"]) for r in sel])
        any_pts = True
        lbl = f"{fd:g} ns/frame" if fd != "" else "unknown spacing"

        if show_err and sem_key:
            xerr = np.array([float(r[sem_key])
                             if str(r.get(sem_key, "")).strip() != "" else 0.0
                             for r in sel])
            if np.any(xerr > 0):
                ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#777777",
                            elinewidth=0.9, capsize=2.5, zorder=2)

        ax.scatter(x, y, s=58, color=color, alpha=0.85,
                   marker=SPACING_MARKER[si % len(SPACING_MARKER)],
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=f"{lbl} (n={len(sel)})")
        for xi, yi, r in zip(x, y, sel):
            ax.annotate(r["run"], (xi, yi), fontsize=6.5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

        rho, p = spearman(x, y)
        if len(x) >= 3 and np.unique(x).size > 1:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "--", color="black",
                    linewidth=1.1, alpha=0.65, zorder=2)
        stats_rows.append(dict(channel_type=sel[0]["channel_type"],
                               measure=key, frame_duration_ns=fd,
                               normalisation=NORM["mode"], n_runs=len(sel),
                               spearman_rho=("" if rho != rho
                                             else round(rho, 4)),
                               spearman_p=("" if p != p else round(p, 6))))
        ax.set_title(f"ρ = {rho:.3f}" + (f"  p = {p:.3g}" if p == p else ""),
                     fontsize=10, fontweight="bold")

    if not any_pts:
        ax.set_axis_off()
        return stats_rows
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Number of permeations", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return stats_rows


def plot_vs_permeations(table, out_dir, ch, title, sub, show_err=False):
    """
    Writes the three-panel figure AND one standalone PNG per measure, so any
    single panel can go into a slide on its own. Runs at different frame
    spacings get different markers and separate correlations -- never one line
    through both, unless they have been put on a common scale with
    --normalise.
    """
    usable = [r for r in table if str(r.get("n_permeations", "")).strip() != ""]
    if len(usable) < 2:
        print("    [skip] fewer than 2 runs have a permeation count")
        return []

    spacings = sorted({r["frame_duration_ns"] for r in usable
                       if r["frame_duration_ns"] != ""})
    measures = measures_list()

    # ---- combined ----
    fig, axes = plt.subplots(1, len(measures), figsize=(5.2 * len(measures), 5))
    stats_rows = []
    for ax, (key, xlabel, color, sem_key) in zip(np.atleast_1d(axes), measures):
        stats_rows += _scatter_panel(ax, usable, spacings, key, xlabel, color,
                                     sem_key, show_err)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.938, sub, ha="center", va="top", fontsize=8.5,
             color="#444444", linespacing=1.3)
    plt.tight_layout(rect=[0, 0, 1, 0.862])
    combined = out_dir / f"mobility_vs_permeations_{ch}.png"
    plt.savefig(combined, dpi=150)
    plt.close()
    print(f"    Saved: {combined.name}")

    # ---- one PNG per measure ----
    for key, xlabel, color, sem_key in measures:
        fig, ax = plt.subplots(figsize=(6.4, 5.3))
        got = _scatter_panel(ax, usable, spacings, key, xlabel, color,
                             sem_key, show_err)
        if not got:
            plt.close()
            continue
        fig.text(0.5, 0.975, f"{ch} — {xlabel} vs permeations",
                 ha="center", va="top", fontsize=11, fontweight="bold")
        fig.text(0.5, 0.937, sub, ha="center", va="top", fontsize=8,
                 color="#444444", linespacing=1.3)
        plt.tight_layout(rect=[0, 0, 1, 0.855])
        single = out_dir / f"{key}_vs_permeations_{ch}.png"
        plt.savefig(single, dpi=150)
        plt.close()
        print(f"    Saved: {single.name}")

    return stats_rows


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="Channel folder holding RUN*/")
    ap.add_argument("-channel", "--channel", dest="channel", default="")
    ap.add_argument("-runs", "--runs", default="")
    ap.add_argument("-perm_col", "--perm_col", default=None,
                    help="Permeation column to use (auto-detected otherwise)")
    ap.add_argument("-frame_duration", "--frame_duration", type=float,
                    default=None,
                    help="ns per frame, forced for every run, overriding "
                         "water_mobility_summary.csv (G12 0.075, G2 0.4)")
    ap.add_argument("-normalise", "--normalise", default="none",
                    choices=["none", "per_ns", "sqrt_ns"],
                    help="none: raw A per frame, valid within one spacing. "
                         "per_ns: step/dt, assumes ballistic motion. "
                         "sqrt_ns: step/sqrt(dt), the diffusive scaling, use "
                         "this to compare G12 against G2")
    ap.add_argument("-errorbars", "--errorbars", action="store_true",
                    help="Draw the standard error of each run's mean as an x "
                         "error bar on the scatters")
    ap.add_argument("-out", "--out", default=None)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    ch = args.channel or base.name

    print("=" * 70)
    print("AGGREGATE CAVITY MOBILITY")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel: {ch}")

    runs = expand_runs(args.runs) if args.runs else None
    irows = load(base, ION_NAME, runs, ch)
    wrows = load(base, WATER_NAME, runs, ch)
    if not irows and not wrows:
        sys.exit(f"ERROR: no mobility output under {base}/RUN*/"
                 f"{WATER_ROOT}/{MODULE_DIR}/")

    spacing = spacing_of(irows + wrows, base, runs, ch, args.frame_duration)
    NORM["mode"] = args.normalise
    NORM["spacing"] = spacing

    unscaled = sorted({r["run_label"] for r in irows + wrows
                       if r["run_label"] not in spacing}, key=natural)
    if args.normalise != "none" and unscaled:
        print(f"  [warn] --normalise {args.normalise} cannot be applied to "
              f"{', '.join(unscaled)} (no spacing) — left raw and therefore "
              f"NOT comparable with the rest")

    n_runs = len({r["run_label"] for r in irows + wrows})
    sub = caption(spacing, irows + wrows)
    print(f"  {len(irows)} ion events, {len(wrows)} water visits, "
          f"{n_runs} run(s)")
    print(f"  normalisation: {args.normalise}  ->  units {unit()}")
    print(f"  {sub}")

    out_dir = Path(args.out) if args.out else base / WATER_ROOT / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    if irows:
        write_csv(irows, out_dir / f"pooled_ions_{ch}.csv")
    if wrows:
        write_csv(wrows, out_dir / f"pooled_waters_{ch}.csv")
    per_run = per_run_table(irows, wrows, spacing, ch)
    write_csv(per_run, out_dir / f"per_run_mobility_{ch}.csv")
    pooled = pooled_table(irows, wrows, spacing, ch)
    write_csv(pooled, out_dir / f"pooled_summary_{ch}.csv")

    print(f"\n  {'species':<20}{'dt (ns)':>9}{'runs':>6}{'n':>9}"
          f"{'mean':>11}{'median':>9}{'sd':>8}   [{unit()}]")
    for r in pooled:
        if r["weighting"] != "pooled_particles":
            continue
        fd = r["frame_duration_ns"]
        print(f"  {r['species']:<20}{(f'{fd:g}' if fd != 'unknown' else '?'):>9}"
              f"{r['n_runs']:>6}{r['n_particles']:>9,}{r['mean']:>11.3f}"
              f"{r['median']:>9.3f}{r['sd']:>8.3f}")
    ratios = [r for r in per_run if r["species"] == "ion_to_water_ratio"]
    if ratios:
        a = np.array([r["mean"] for r in ratios], dtype=float)
        print(f"\n  ion/water step ratio (dimensionless, spacing-robust): "
              f"mean {a.mean():.3f}, median {np.median(a):.3f}, "
              f"range {a.min():.3f}-{a.max():.3f} over {a.size} runs")

    perms, perm_col = load_permeations(base, args.perm_col)
    if perms:
        ptable = build_perm_table(irows, wrows, spacing, perms, ch)
        write_csv(ptable, out_dir / f"mobility_vs_permeations_{ch}.csv")
        srows = plot_vs_permeations(
            ptable, out_dir, ch,
            f"{ch} — cavity mobility vs permeation count (one dot per run)",
            sub, show_err=args.errorbars)
        if srows:
            write_csv(srows, out_dir / f"mobility_vs_permeations_stats_{ch}.csv")
            print(f"\n  {'measure':<28}{'dt':>7}{'runs':>6}{'rho':>8}{'p':>10}")
            for r in srows:
                fd = r["frame_duration_ns"]
                print(f"  {r['measure']:<28}"
                      f"{(f'{fd:g}' if fd != '' else '?'):>7}{r['n_runs']:>6}"
                      f"{r['spearman_rho']:>8}{r['spearman_p']:>10}")
        missing = [r["run"] for r in ptable
                   if str(r.get("n_permeations", "")).strip() == ""]
        if missing:
            print(f"  [warn] no permeation count for: {', '.join(missing)}")

    # ---- box plots: together, then each species on its own ----
    plot_pooled_box([(irows, "Ions", ION_COLOR),
                     (wrows, "Waters", WATER_COLOR)],
                    out_dir / f"pooled_box_{ch}.png",
                    f"{ch} — mobility in the GLU/HBC cavity ({n_runs} runs)",
                    sub)
    plot_pooled_box([(irows, "Ions", ION_COLOR)],
                    out_dir / f"pooled_box_ions_{ch}.png",
                    f"{ch} — ion mobility ({n_runs} runs)", sub)
    plot_pooled_box([(wrows, "Waters", WATER_COLOR)],
                    out_dir / f"pooled_box_waters_{ch}.png",
                    f"{ch} — water mobility ({n_runs} runs)", sub)

    plot_per_run_box(irows, wrows, spacing,
                     out_dir / f"per_run_box_{ch}.png",
                     f"{ch} — mobility per run", sub)
    plot_pooled_hist(irows, wrows, out_dir / f"pooled_hist_{ch}.png",
                     f"{ch} — step-length distributions", sub)
    if wrows:
        plot_residence(wrows, out_dir / f"water_residence_{ch}.png",
                       f"{ch} — water residence in the cavity "
                       f"({n_runs} runs)", sub)

    print(f"\n✓ Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()

'''
# within one channel, raw units (unchanged behaviour)
python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12 --frame_duration 0.075

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ --channel G2 --frame_duration 0.4

# for the G12 vs G2 comparison -- same normalisation on both
python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12 --frame_duration 0.075 --normalise sqrt_ns --out /tmp/cmp_G12

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ --channel G2 --frame_duration 0.4 --normalise sqrt_ns --out /tmp/cmp_G2

# mutants
python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ --channel G12_E141Q --frame_duration 0.075

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ --channel G12_D173N --frame_duration 0.075 --errorbars
'''

'''
python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12 --frame_duration 0.075

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2 --frame_duration 0.4

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--channel G12_E141Q --frame_duration 0.075

python3 aggregate_cavity_mobility.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--channel G12_D173N --frame_duration 0.075
'''