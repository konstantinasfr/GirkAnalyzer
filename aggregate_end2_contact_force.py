#!/usr/bin/env python3
"""
aggregate_end2_contact_force.py
===============================
Pools every run's `end2_contact_force.csv` and rebuilds the same figures on
all events at once, plus two that only make sense across runs.

WHY POOL
--------
One run answers "what happened here", not "what happens". RUN3 alone has 37
events but almost no spread in cavity occupancy (3-5 ions), so the
occupancy-dependence question cannot be asked inside it, and the G1 partial
states land at n = 6 and n = 3 where a single event moves a mean by several
kcal/mol/A. Pooling fixes both: the event count grows with the number of
runs, and occupancy finally varies.

Events are the unit, not runs. A run with 58 permeations contributes 58 rows
and a run with 2 contributes 2, which is correct here -- each row is one
independent exit, not one noisy run-level average. The per-run figures are
there to show whether any single run is driving the pooled answer.

MEDIAN FIRST, as in the per-run module: the force distributions are skewed,
so every panel shows the median as the solid line, the mean as a red dashed
marker, the individual events as dots, and "% >0" -- the share of events in
which that component pointed towards the gate. A median can be positive
while the sign flips from event to event; the percentage is what says
whether the direction is consistent.

OUTPUTS   <channel>/end2_contact_force_across_runs/
---------------------------------------------------
    pooled_events_<CH>.csv            every event, all runs, run column kept
    per_run_summary_<CH>.csv          one row per run
    force_stats_<CH>.csv              median/mean/%>0/sign-test per component
    g1_state_stats_<CH>.csv           the same, split by G1 occupancy state
    site_occupancy_stats_<CH>.csv     filled % per residue, pooled and per run

    force_balance_pooled_<CH>.png             the main figure
    force_balance_by_G1_state_<CH>.png        four states, shared y axis
    force_balance_<CH>_<state>.png            each state on its own
    site_occupancy_<CH>.png
    combinations_<CH>.png
    release_distance_<CH>.png
    net_force_by_run_<CH>.png         one box per run -- is one run driving it?
    occupancy_by_run_<CH>.png         runs x residues, filled %
    channel_comparison_net.png        when more than one channel is given

USAGE
-----
    python3 aggregate_end2_contact_force.py \
        /media/ziyue/.../girk_analyser_results/G12 --channel G12

    # both channels at once; each writes under its own folder, and a
    # side-by-side NET comparison is written beside the first
    python3 aggregate_end2_contact_force.py \
        /media/ziyue/.../girk_analyser_results/G12 \
        /media/ziyue/.../girk_analyser_results/G2

    # only some runs
    python3 aggregate_end2_contact_force.py .../G12 --channel G12 --runs 1-20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The plotting primitives live in the per-run module, so the pooled figures
# and the per-run ones can never drift apart.
try:
    from analysis.end2_contact_force import (
        FORCE_SPEC, YLABEL, MEAN_STYLE, ION_COLOR, NET_COLOR,
        _force_panel, _force_legend, _col, _fv, _save, g1_state,
        plot_site_occupancy, plot_combinations, plot_release_distance,
        plot_force_box, plot_force_box_by_state, plot_g1_states,
    )
except ImportError:
    try:
        from end2_contact_force import (
            FORCE_SPEC, YLABEL, MEAN_STYLE, ION_COLOR, NET_COLOR,
            _force_panel, _force_legend, _col, _fv, _save, g1_state,
            plot_site_occupancy, plot_combinations, plot_release_distance,
            plot_force_box, plot_force_box_by_state, plot_g1_states,
        )
    except ImportError:
        sys.exit("ERROR: end2_contact_force.py must sit next to this file "
                 "(or in analysis/) — the plots and the force definitions "
                 "are shared with it.")

SCRIPT_VERSION = "2026-10-15a"
MODULE_DIR = "end2_contact_force"
OUT_SUBDIR = "end2_contact_force_across_runs"
INPUT_NAME = "end2_contact_force.csv"      # overridden by --event
DETAIL_NAME = "end2_other_ions.csv"

CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22", "G12_E141Q": "#16A085",
                 "G12_D173N": "#8E44AD", "G12_ML": "#27AE60",
                 "G12_GAT": "#C0392B"}


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


def natural(run):
    digits = "".join(c for c in run if c.isdigit())
    return (int(digits) if digits else 0, run)


def find_csvs(base, runs=None, name=None):
    """[(run, path)] — exact paths when --runs is given, else one bounded glob."""
    name = name or INPUT_NAME
    base = Path(base)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    out = []
    names = runs if runs else sorted(
        (d.name for d in base.glob("RUN*") if d.is_dir()), key=natural)
    for run in names:
        direct = base / run / MODULE_DIR / name
        if direct.is_file():
            out.append((run, direct))
        elif runs:
            print(f"  [warn] no {name} for {run} under {base}")
    return out


def load_runs(base, runs, channel, name=None):
    rows = []
    found = find_csvs(base, runs, name)
    if not found:
        return rows
    for run, path in found:
        with open(path, newline="") as f:
            got = list(csv.DictReader(f))
        for r in got:
            # the run column is written by the per-run module, but a run
            # processed without --run would land empty; the folder name is
            # the reliable label
            r["run_label"] = run
            if channel:
                r["channel_type"] = channel
        rows.extend(got)
        print(f"    {run:<8} {len(got):>4} events   {path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Residue bookkeeping, read back out of the CSV
# ─────────────────────────────────────────────────────────────────────────────

def residue_labels(rows, n_glu=None):
    """
    The per-run module writes occ_<label> columns in ring order: every GLU
    first, then every ASN/ASP. Reading them back from the header keeps this
    script independent of CHANNEL_CONFIG and of the topology, so a mutant
    with a renamed residue still works.
    """
    # occ_combination, occ_free_residues and occ_free_*_ring also start with
    # "occ_", so a residue is recognised by having the matching contact_ and
    # dmin_ columns the per-run module writes alongside it
    header = list(rows[0])
    labels = [k[4:] for k in header
              if k.startswith("occ_") and not k.startswith("occ_by_")
              and f"contact_{k[4:]}" in header and f"dmin_{k[4:]}" in header]
    if not labels:
        sys.exit("ERROR: no occ_<residue> columns found — these CSVs predate "
                 "the occupancy columns. Re-run end2_contact_force.py with "
                 "force_rerun=True.")
    half = n_glu if n_glu else len(labels) // 2
    return labels, labels[:half], labels[half:]


def pick_pair(glu_labels, asn_labels, subunit=None):
    """
    Which (E, D/N) pair the state panels split on. Defaults to the first
    subunit in the list, which is GIRK1 for the G12 ordering. For a channel
    with no special subunit (G2) the default is simply its first subunit and
    the panel reads as "that subunit's pair", which is still a fair question
    to ask of it.
    """
    if subunit:
        for e, d in zip(glu_labels, asn_labels):
            if e.split(".")[-1].lower() == subunit.lower():
                return e, d
        print(f"  [warn] subunit '{subunit}' not found in "
              f"{[l.split('.')[-1] for l in glu_labels]} — using the first")
    return glu_labels[0], asn_labels[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def sign_test_p(values):
    """
    Two-sided sign test against a median of zero. Non-parametric, makes no
    distributional assumption, and with pooled events the n is finally large
    enough for it to mean something.
    """
    from math import comb
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v) & (v != 0)]
    n = v.size
    if n == 0:
        return float("nan")
    k = int((v > 0).sum())
    k = max(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return float(min(1.0, p))


def force_stats(rows, channel, group=""):
    out = []
    for key, lab, _ in FORCE_SPEC:
        v = _col(rows, key)
        if v.size == 0:
            continue
        out.append(dict(
            channel_type=channel, group=group or "all",
            component=key, label=lab.replace("\n", " "),
            n_events=int(v.size),
            median=round(float(np.median(v)), 3),
            mean=round(float(v.mean()), 3),
            sd=round(float(v.std(ddof=1)) if v.size > 1 else 0.0, 3),
            q1=round(float(np.percentile(v, 25)), 3),
            q3=round(float(np.percentile(v, 75)), 3),
            minimum=round(float(v.min()), 3),
            maximum=round(float(v.max()), 3),
            frac_positive=round(float((v > 0).mean()), 4),
            sign_test_p=round(sign_test_p(v), 8)))
    return out


def per_run_summary(rows, channel, labels):
    out = []
    for run in sorted({r["run_label"] for r in rows}, key=natural):
        sel = [r for r in rows if r["run_label"] == run]
        row = dict(channel_type=channel, run=run, n_events=len(sel))
        for key, lab, _ in FORCE_SPEC:
            v = _col(sel, key)
            if v.size:
                row[f"median_{key}"] = round(float(np.median(v)), 3)
                row[f"mean_{key}"] = round(float(v.mean()), 3)
                row[f"fracpos_{key}"] = round(float((v > 0).mean()), 3)
        for k in ("n_occupied", "n_other_ions_cavity", "n_contacts",
                  "n_other_ions_above", "n_other_ions_below"):
            v = _col(sel, k)
            if v.size:
                row[f"mean_{k}"] = round(float(v.mean()), 3)
        for lb in labels:
            row[f"filled_{lb}"] = round(
                float(np.mean([_fv(x, f"occ_{lb}") for x in sel])), 4)
        out.append(row)
    return out


def occupancy_stats(rows, channel, labels):
    out = []
    for lb in labels:
        occ = np.array([_fv(x, f"occ_{lb}") for x in rows])
        own = np.array([_fv(x, f"contact_{lb}") for x in rows])
        fz = _col(rows, f"fz_{lb}_out")
        out.append(dict(
            channel_type=channel, residue=lb, n_events=len(rows),
            filled_frac=round(float(np.nanmean(occ)), 4),
            free_frac=round(1 - float(np.nanmean(occ)), 4),
            held_by_permeating_frac=round(float(np.nanmean(own)), 4),
            median_fz_out=round(float(np.median(fz)), 3) if fz.size else "",
            mean_fz_out=round(float(fz.mean()), 3) if fz.size else ""))
    return out


def g1_state_stats(rows, e_label, d_label, channel):
    out = []
    for s in ["both", "E only", "D only", "neither"]:
        sel = [x for x in rows if g1_state(x, e_label, d_label) == s]
        if not sel:
            continue
        out.extend(force_stats(sel, channel, group=s))
    # a rank test across the four, which is the thing a reviewer asks for
    try:
        from scipy import stats
        for key, lab, _ in FORCE_SPEC:
            groups = [_col([x for x in rows
                            if g1_state(x, e_label, d_label) == s], key)
                      for s in ["both", "E only", "D only", "neither"]]
            groups = [g for g in groups if g.size > 1]
            if len(groups) >= 2:
                h, p = stats.kruskal(*groups)
                out.append(dict(channel_type=channel, group="kruskal_across_states",
                                component=key, label=lab.replace("\n", " "),
                                n_events=int(sum(g.size for g in groups)),
                                median="", mean="", sd="", q1="", q3="",
                                minimum="", maximum="", frac_positive="",
                                sign_test_p="", kruskal_H=round(float(h), 3),
                                kruskal_p=round(float(p), 6)))
    except ImportError:
        print("  [warn] scipy missing — Kruskal-Wallis across G1 states skipped")
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
#  Cross-run plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_net_by_run(rows, out_path, title, key="fz_total_out",
                    ylabel=None, color=NET_COLOR):
    """
    One box per run. The pooled median is drawn across it, so a run sitting
    entirely on one side of that line is visible immediately -- which is the
    check that the pooled answer is not one run's result wearing a bigger n.
    """
    runs = sorted({r["run_label"] for r in rows}, key=natural)
    data = [(run, _col([r for r in rows if r["run_label"] == run], key))
            for run in runs]
    data = [(run, v) for run, v in data if v.size]
    if not data:
        return
    pooled = _col(rows, key)

    fig, ax = plt.subplots(figsize=(max(7.5, 0.72 * len(data) + 3.2), 5.4))
    bp = ax.boxplot([v for _, v in data], widths=0.6, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, (_, v) in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=13,
                   color="#222222", alpha=0.45, zorder=3)
    ax.axhline(0, color="black", linewidth=1.1)
    ax.axhline(float(np.median(pooled)), color="green", linestyle="-.",
               linewidth=1.6,
               label=f"pooled median {np.median(pooled):+.1f} "
                     f"(n = {pooled.size})")
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([f"{run}\n(n={v.size})" for run, v in data],
                       fontsize=8.5)
    ax.set_ylabel(ylabel or YLABEL, fontsize=10.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


def plot_occupancy_by_run(rows, labels, n_glu, out_path, title):
    """Runs x residues, % of that run's events in which the site was filled."""
    runs = sorted({r["run_label"] for r in rows}, key=natural)
    mat = np.array([[100 * float(np.nanmean(
        [_fv(x, f"occ_{lb}") for x in rows if x["run_label"] == run]))
        for lb in labels] for run in runs])
    if mat.size == 0:
        return

    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3.4,
                                    0.42 * len(runs) + 2.8))
    im = ax.imshow(mat, cmap="BuPu", vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if mat[i, j] > 55 else "#222222")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(runs)))
    n_ev = {run: sum(1 for x in rows if x["run_label"] == run) for run in runs}
    ax.set_yticklabels([f"{r} (n={n_ev[r]})" for r in runs], fontsize=8.5)
    ax.axvline(n_glu - 0.5, color="black", linewidth=1.4)
    ax.set_title(f"{title}\n% of each run's events with the site filled",
                 fontsize=11.5, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% filled")
    _save(fig, out_path)


def plot_channel_comparison(by_channel, out_path, key="fz_total_out"):
    """Same component, one box per channel. Written only when >1 channel."""
    items = [(ch, _col(rows, key)) for ch, rows in by_channel.items()]
    items = [(ch, v) for ch, v in items if v.size]
    if len(items) < 2:
        return
    fig, ax = plt.subplots(figsize=(2.6 * len(items) + 2.6, 5.4))
    bp = ax.boxplot([v for _, v in items], widths=0.55, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2.2))
    for box, (ch, _) in zip(bp["boxes"], items):
        box.set_facecolor(CHANNEL_COLOR.get(ch, "#7F8C8D"))
        box.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, (_, v) in enumerate(items, start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=14,
                   color="#222222", alpha=0.4, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, **MEAN_STYLE, zorder=4)
    ax.axhline(0, color="black", linewidth=1.1)
    hi = max(v.max() for _, v in items)
    lo = min(v.min() for _, v in items)
    pad = 0.16 * (hi - lo)
    for i, (_, v) in enumerate(items, start=1):
        ax.text(i, hi + pad * 0.2,
                f"med {np.median(v):+.1f}\nmean {v.mean():+.1f}\n"
                f"{100 * (v > 0).mean():.0f}% >0",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(lo - pad * 0.3, hi + pad * 1.5)
    ax.set_xticks(range(1, len(items) + 1))
    ax.set_xticklabels([f"{ch}\n(n={v.size} events)" for ch, v in items],
                       fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10.5)
    ax.set_title("Net z-force on the permeating K⁺ at end_2, by channel",
                 fontsize=12, fontweight="bold")
    _force_legend(ax)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────

def process_channel(rows, ch, out_dir, subunit=None, n_glu=None):
    labels, glu_labels, asn_labels = residue_labels(rows, n_glu)
    e_label, d_label = pick_pair(glu_labels, asn_labels, subunit)
    runs = sorted({r["run_label"] for r in rows}, key=natural)
    tag = f"{ch}_pooled"

    print(f"\n  === {ch}: {len(rows)} events from {len(runs)} runs ===")
    print(f"  residues: {', '.join(labels)}")
    print(f"  state panels split on {e_label} / {d_label}")

    # ---- tables ----
    write_csv(rows, out_dir / f"pooled_events_{ch}.csv")
    write_csv(per_run_summary(rows, ch, labels),
              out_dir / f"per_run_summary_{ch}.csv")
    fstats = force_stats(rows, ch)
    write_csv(fstats, out_dir / f"force_stats_{ch}.csv")
    write_csv(g1_state_stats(rows, e_label, d_label, ch),
              out_dir / f"g1_state_stats_{ch}.csv")
    write_csv(occupancy_stats(rows, ch, labels),
              out_dir / f"site_occupancy_stats_{ch}.csv")

    # ---- the per-run figures, rebuilt on every event ----
    print()
    # plot_site_occupancy wants resid keys; the pooled rows carry labels
    # only, so a identity mapping over the label list is passed instead
    ident = {lb: lb for lb in labels}
    plot_site_occupancy(rows, ident, glu_labels, asn_labels,
                        out_dir / f"site_occupancy_{ch}.png",
                        f"{ch} — which sites are filled at end_2 "
                        f"({len(runs)} runs)")
    plot_force_box(rows, out_dir / f"force_balance_pooled_{ch}.png",
                   f"{ch} — force balance on the permeating K⁺ at end_2 "
                   f"({len(runs)} runs)")
    plot_force_box_by_state(rows, e_label, d_label, out_dir, tag,
                            f"{ch} — force balance by {e_label.split('.')[-1]} "
                            f"occupancy state")
    plot_g1_states(rows, e_label, d_label,
                   out_dir / f"state_medians_{ch}.png",
                   f"{ch} — does {e_label.split('.')[-1]} coverage change "
                   f"the force?")
    plot_combinations(rows, out_dir / f"combinations_{ch}.png",
                      f"{ch} — occupancy combinations and the net force")
    plot_release_distance(rows, ident, labels, 3.0,
                          out_dir / f"release_distance_{ch}.png",
                          f"{ch} — the exiting ion has already let go")

    # ---- the two that only exist across runs ----
    plot_net_by_run(rows, out_dir / f"net_force_by_run_{ch}.png",
                    f"{ch} — net z-force per run, is one run driving it?")
    plot_net_by_run(rows, out_dir / f"ion_force_by_run_{ch}.png",
                    f"{ch} — push from the cavity ions, per run",
                    key="fz_ions_cavity_out", color=ION_COLOR)
    plot_occupancy_by_run(rows, labels, len(glu_labels),
                          out_dir / f"occupancy_by_run_{ch}.png",
                          f"{ch} — site occupancy by run")

    # ---- console summary ----
    print(f"\n  {'component':<16}{'n':>6}{'median':>9}{'mean':>9}"
          f"{'% >0':>7}{'sign p':>12}")
    for s in fstats:
        print(f"  {s['label']:<16}{s['n_events']:>6}{s['median']:>9.2f}"
              f"{s['mean']:>9.2f}{100 * s['frac_positive']:>6.0f}%"
              f"{s['sign_test_p']:>12.3g}")

    occ = _col(rows, "n_other_ions_cavity")
    if occ.size:
        print(f"\n  cavity occupancy spread across the pool: "
              f"{occ.min():.0f}-{occ.max():.0f} other ions "
              f"(median {np.median(occ):.0f})")
        try:
            from scipy import stats
            net = _col(rows, "fz_ions_cavity_out")
            n = min(occ.size, net.size)
            if np.unique(occ[:n]).size > 1:
                rho, p = stats.spearmanr(occ[:n], net[:n])
                print(f"  occupancy vs push from the cavity ions: "
                      f"Spearman rho = {rho:+.3f}, p = {p:.3g}, n = {n}")
        except ImportError:
            pass
    print(f"\n  -> {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Channel folders holding RUN*/end2_contact_force/")
    ap.add_argument("-channel", "--channel", dest="channel", default="",
                    help="Force the channel label (otherwise the folder name)")
    ap.add_argument("-runs", "--runs", default="",
                    help="Only these runs, e.g. --runs 3,5,14 or --runs 1-20")
    ap.add_argument("-subunit", "--subunit", default=None,
                    help="Which subunit the state panels split on (default: "
                         "the first, i.e. G1 for the G12 ordering)")
    ap.add_argument("-n_glu", "--n_glu", type=int, default=None,
                    help="How many of the occ_ columns are the GLU ring "
                         "(default: half)")
    ap.add_argument("-event", "--event", default="end_2",
                    choices=["end_2", "end_3"],
                    help="Which per-run file to pool: end_2 (leaving the "
                         "cavity, the default) or end_3 (past the gate)")
    ap.add_argument("-out", "--out", default=None,
                    help="Write everything here instead of per channel")
    args = ap.parse_args()

    print("=" * 70)
    print("AGGREGATE END_2 CONTACTS AND FORCE")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")

    prefix = "end2" if args.event == "end_2" else "end3"
    input_name = f"{prefix}_contact_force.csv"
    print(f"  event: {args.event}  (pooling {input_name})")

    runs = expand_runs(args.runs) if args.runs else None
    if runs:
        print(f"  runs: {', '.join(runs)}")

    forced = Path(args.out).resolve() if args.out else None
    if forced:
        forced.mkdir(parents=True, exist_ok=True)

    by_channel = {}
    first_dir = None
    for item in args.inputs:
        base = Path(item).resolve()
        ch = (args.channel or base.name) + ("" if args.event == "end_2"
                                             else "_end3")
        print(f"\n  reading {ch} from {base}")
        rows = load_runs(base, runs, args.channel, input_name)
        if not rows:
            print(f"  [warn] no {input_name} found under {base}/RUN*/"
                  f"{MODULE_DIR}/ — skipped")
            continue
        out_dir = forced or (base / OUT_SUBDIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        first_dir = first_dir or out_dir
        by_channel[ch] = rows
        process_channel(rows, ch, out_dir, args.subunit, args.n_glu)

    if not by_channel:
        sys.exit(f"ERROR: no {input_name} found under any of the given paths.")

    if len(by_channel) > 1:
        print("\n  === channel comparison ===")
        plot_channel_comparison(by_channel,
                                first_dir / "channel_comparison_net.png")
        plot_channel_comparison(by_channel,
                                first_dir / "channel_comparison_ions.png",
                                key="fz_ions_cavity_out")
        rows = []
        for ch, rr in by_channel.items():
            rows.extend(force_stats(rr, ch))
        write_csv(rows, first_dir / "channel_comparison_stats.csv")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()

'''
python3 aggregate_end2_contact_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12 --event end_3

python3 aggregate_end2_contact_force.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2 --event end_3
'''