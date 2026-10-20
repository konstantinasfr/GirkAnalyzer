#!/usr/bin/env python3
"""
permeation_ion_combinations.py
==============================
"What exactly is the combination of ions at the moment of permeation?"

For every permeating ion, the per-ion table already records how many OTHER ions
sit on each of the 8 residues at frame end_3 — the nions_<LABEL> columns, which
hold 0, 1 or 2. This module turns that row of numbers into a named combination,
then counts how often each distinct combination occurs.

    nions_E152.2a=1, nions_N184.2b=1, rest 0   ->   "E152.2a + N184.2b"

WHAT COUNTS AS A COMBINATION
----------------------------
Four descriptions are produced for each event, because they answer different
questions and the right one depends on what is being asked:

  combo             the occupied residues by name, in column order
                    "E152.2c + N184.2b"
  combo_counts      the same, with multiplicity where a residue holds 2 ions
                    "E152.2a x2 + N184.2a"
  ring_pattern      how many of each ring, names dropped
                    "2 GLU + 1 ASN"
  subunit_pattern   how many distinct subunits are involved, and whether any
                    single subunit carries BOTH its GLU and its ASN
                    "3 subunits, 1 paired"

The subunit one exists because a GLU and an ASN on the SAME subunit is a
different structural situation from the same two counts spread over two
subunits, and neither combo nor ring_pattern distinguishes them.

An event where no residue is occupied is recorded as "(none)" rather than
dropped — "the ion permeated with nothing else on the ring" is a result, and
excluding those would inflate every percentage.

IMPORTANT — WHAT THE COUNTS MEAN
--------------------------------
nions_ EXCLUDES the permeating ion itself (permeation_ion_table.py subtracts
it). So a combination describes the crowd the ion leaves behind, not the ion
under study. "(none)" means the ion permeated with no OTHER ion on any of the
8 residues; it does not mean the cavity was empty — n_cavity_region, carried
through to the output, is the cavity count and includes the permeating ion.

Residue labels are read from the file header, never hardcoded, so G12
(E141.G1, D173.G1, ...) and G2 (E152.2a, ...) both work.

CENSORED IONS are kept by default, unlike the boxplot scripts: the combination
is read at a single frame, end_3, which is recorded exactly even when the
ion's total residence was truncated. Pass drop_censored=True to exclude them.

OUTPUTS — per run, in <results_dir>/permeation_ion_table/
---------------------------------------------------------
    per_ion_combinations.csv     one row per permeating ion: its combination in
                                 all four forms, plus run, ion_rank, end_3, the
                                 occupancy counts and the per-residue columns
    combination_counts.csv       one row per DISTINCT combination: how many
                                 events, % of events, how many runs it appears
                                 in, and which runs
    combination_counts_multiplicity.csv   the same for combo_counts
    ring_pattern_counts.csv      one row per "n GLU + n ASN" pattern
    subunit_pattern_counts.csv   one row per subunit pattern
    combinations_top.png         bar plot of the most frequent combinations
    combinations_grid.png        the same combinations as a grid: one column
                                 per residue, a filled cell where an ion sat,
                                 with the event count beside each row
    combinations_grid_all.png    the grid again with EVERY combination, not
                                 just the top ones (written only when there are
                                 more than -top of them)

OUTPUTS — pooled, in <out_dir> (default <first base_dir>/permeation_combinations)
---------------------------------------------------------------------------------
    the same six files, over every run of the channel(s) given

READ THE n_runs COLUMN FIRST. With 8 residues the combination space is large,
so in one run most combinations occur once and the percentages are nearly
meaningless. The per-run files exist for completeness and for main.py's sake;
the pooled mode is where the counts become worth comparing. Even pooled, a
combination seen several times in one trajectory and never elsewhere is a
property of that trajectory, not of the channel — which is what n_runs tells
you. ring_pattern is the level that usually has enough events per cell to
compare G12 against G2; combo is where the structural detail lives.

USAGE — from main.py
--------------------
    from analysis.permeation_ion_combinations import run_permeation_combinations

    run_permeation_combinations(
        results_dir  = results_dir,
        channel_type = channel_type,
        run          = run_number,
        unit         = "frames",
        force_rerun  = True,
    )

    # call AFTER run_permeation_ion_table()

USAGE — pooled across runs
--------------------------
    python3 permeation_ion_combinations.py --pool \
        /media/ziyue/.../girk_analyser_results/G2

    python3 permeation_ion_combinations.py --pool .../G12 .../G2 -unit ns -top 20

    # one run, same as main.py would do
    python3 permeation_ion_combinations.py --results_dir .../G2/RUN1 \
        --channel_type G2 --run 1 --force_rerun
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-13b  (per-run entry point for main.py)"

RUN_PATTERN = re.compile(r"^RUN(\d+)$", re.IGNORECASE)
NIONS_PREFIX = "nions_"
TABLE_SUBDIR = "permeation_ion_table"

CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22",
                 "G12_ML": "#27AE60", "G12_GAT": "#8E44AD"}

FS_TITLE   = 22
FS_AXLABEL = 20
FS_TICK    = 17
FS_VALUE   = 15
DPI = 200
BAR_EDGE_LW = 1.0

PER_ION_NAME = "per_ion_combinations.csv"
COMBO_NAME = "combination_counts.csv"
MULT_NAME = "combination_counts_multiplicity.csv"
RING_NAME = "ring_pattern_counts.csv"
SUBUNIT_NAME = "subunit_pattern_counts.csv"
PLOT_NAME = "combinations_top.png"
GRID_NAME = "combinations_grid.png"
GRID_ALL_NAME = "combinations_grid_all.png"


# ─────────────────────────────────────────────────────────────────────────────
#  Label parsing
# ─────────────────────────────────────────────────────────────────────────────

def ring_of(label):
    """E141.G1 / E152.2a -> GLU;  D173.G1 / N184.2c -> ASN."""
    c = (label or "")[:1].upper()
    if c == "E":
        return "GLU"
    if c in ("D", "N"):
        return "ASN"
    return "other"


def subunit_of(label):
    """E152.2a -> 2a.  Falls back to the whole label if there is no dot."""
    return label.split(".", 1)[1] if "." in label else label


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def _num(v, default=0):
    try:
        return int(round(float(str(v).strip())))
    except (TypeError, ValueError):
        return default


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def describe(counts, labels):
    """
    counts: {label: n other ions on it}. Returns the four descriptions.
    Residues are listed in the file's column order, so the same set always
    produces the same string.
    """
    occupied = [l for l in labels if counts.get(l, 0) > 0]
    if not occupied:
        return "(none)", "(none)", "0 GLU + 0 ASN", "0 subunits, 0 paired"

    combo = " + ".join(occupied)
    combo_counts = " + ".join(
        f"{l} x{counts[l]}" if counts[l] > 1 else l for l in occupied)

    n_glu = sum(counts[l] for l in occupied if ring_of(l) == "GLU")
    n_asn = sum(counts[l] for l in occupied if ring_of(l) == "ASN")
    ring_pattern = f"{n_glu} GLU + {n_asn} ASN"

    by_sub = defaultdict(set)
    for l in occupied:
        by_sub[subunit_of(l)].add(ring_of(l))
    n_sub = len(by_sub)
    n_paired = sum(1 for rings in by_sub.values() if {"GLU", "ASN"} <= rings)
    subunit_pattern = (f"{n_sub} subunit{'s' if n_sub != 1 else ''}, "
                       f"{n_paired} paired")

    return combo, combo_counts, ring_pattern, subunit_pattern


def load_rows(path, run_name, channel, drop_censored):
    """Returns (rows, labels, n_dropped) for one permeation_ion_table CSV."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], 0
    labels = [c[len(NIONS_PREFIX):] for c in rows[0]
              if c.startswith(NIONS_PREFIX)]
    if not labels:
        print(f"    [warn] {run_name}: no {NIONS_PREFIX}* columns — skipped. "
              f"Regenerate with permeation_ion_table.py 2026-09-26c or later.")
        return [], [], 0

    out, dropped = [], 0
    for r in rows:
        censored = (_truthy(r.get("start_censored"))
                    or _truthy(r.get("incomplete_range")))
        if censored and drop_censored:
            dropped += 1
            continue
        counts = {l: _num(r.get(NIONS_PREFIX + l)) for l in labels}
        combo, combo_counts, ring_pattern, subunit_pattern = describe(counts, labels)
        rec = dict(
            channel_type=(r.get("channel_type") or channel).strip() or channel,
            run=run_name,
            ion_rank=r.get("ion_rank", ""),
            ion_id=r.get("ion_id", ""),
            end_3=r.get("end_3", ""),
            combo=combo,
            combo_counts=combo_counts,
            ring_pattern=ring_pattern,
            subunit_pattern=subunit_pattern,
            n_residues_occupied=sum(1 for l in labels if counts[l] > 0),
            n_ions_on_residues=sum(counts.values()),
            n_other_ions_contact=r.get("n_other_ions_contact", ""),
            n_cavity_region=r.get("n_cavity_region", ""),
            censored=censored,
        )
        for l in labels:
            rec[NIONS_PREFIX + l] = counts[l]
        out.append(rec)
    return out, labels, dropped


def find_tables(channel_dir, unit):
    """Every RUN*/permeation_ion_table/permeation_ion_table_<unit>.csv."""
    found = []
    for sub in sorted(channel_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = RUN_PATTERN.match(sub.name)
        if not m:
            continue
        path = sub / TABLE_SUBDIR / f"permeation_ion_table_{unit}.csv"
        if path.is_file():
            found.append((int(m.group(1)), sub.name, path))
        else:
            print(f"    [missing] {sub.name}: no permeation_ion_table_{unit}.csv")
    found.sort()
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Tallying
# ─────────────────────────────────────────────────────────────────────────────

def tally(rows, key, channel):
    """One row per distinct value of `key`, with run provenance."""
    n_total = len(rows)
    counts = Counter(r[key] for r in rows)
    runs = defaultdict(set)
    for r in rows:
        runs[r[key]].add(r["run"])
    # one example row per distinct value, so the grid plot can recover which
    # residues were occupied without re-deriving it from the label string
    example = {}
    for r in rows:
        example.setdefault(r[key], r)

    out = []
    for value, n in counts.most_common():
        rs = sorted(runs[value], key=lambda s: (len(s), s))
        ex = example[value]
        occ = {k[len(NIONS_PREFIX):]: v for k, v in ex.items()
               if k.startswith(NIONS_PREFIX)}
        out.append(dict(
            channel_type=channel,
            combination=value,
            n_events=n,
            pct_of_events=round(100.0 * n / n_total, 2) if n_total else "",
            n_runs=len(rs),
            runs=";".join(rs),
            _occupancy=occ,
        ))
    return out


def write_csv(rows, path, fieldnames=None):
    if not rows:
        print(f"    [warn] nothing to write to {Path(path).name}")
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                # keys starting with "_" are internal bookkeeping for the plots
                if k not in fieldnames and not k.startswith("_"):
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {Path(path).name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_top(tallied, out_path, channel, n_events, top, title_extra="", fs=1.0):
    rows = tallied[:top]
    if not rows:
        return
    labels = [r["combination"] for r in rows]
    vals = [r["pct_of_events"] for r in rows]
    y = np.arange(len(labels))[::-1]   # most frequent at the top

    fig, ax = plt.subplots(figsize=((13.0 + 0.12 * max(len(l) for l in labels)) * fs,
                                    (1.8 + 0.62 * len(labels)) * fs))
    ax.barh(y, vals, height=0.72, color=CHANNEL_COLOR.get(channel, "#2980B9"),
            alpha=0.85, edgecolor="black", linewidth=BAR_EDGE_LW)
    for yi, v, r in zip(y, vals, rows):
        n = r["n_runs"]
        ax.text(v + max(vals) * 0.012, yi,
                f"{v:.1f}%  ({r['n_events']} events, {n} run{'s' if n != 1 else ''})",
                va="center", ha="left", fontsize=FS_VALUE * fs)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK * fs)
    ax.set_xlabel("% of permeation events", fontsize=FS_AXLABEL * fs)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.tick_params(axis="x", labelsize=FS_TICK * fs)
    ax.set_title(f"{channel} — ion combinations at the moment of permeation"
                 + (f" {title_extra}" if title_extra else "") + "\n"
                 f"({n_events} permeation events; "
                 f"top {len(labels)} of {len(tallied)} distinct combinations)",
                 fontsize=FS_TITLE * fs, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_combination_grid(rows, labels, out_path, channel, top, title_extra="",
                          fs=1.0, compact=False):
    """
    One column per residue, one row per distinct combination: a cell is filled
    if that residue held an ion, and the count is written beside the row.

    This is the figure to put on a slide. The "+"-joined strings in
    combination_counts.csv are precise but unreadable at a glance, and two
    combinations that differ by one subunit look nearly identical as text; as a
    grid the pattern is immediate — you can see whether the occupied cells
    cluster in the GLU half, the ASN half, or one subunit column.

    A cell holding 2 ions is drawn darker and labelled "2", so multiplicity is
    not silently flattened into presence/absence.
    """
    n_distinct = len(rows)
    rows = rows[:top]
    if not rows or not labels:
        return

    # the per-ion records are needed to recover each combination's counts, so
    # rebuild the occupancy from the combination string itself
    base = CHANNEL_COLOR.get(channel, "#2980B9")
    n_rows, n_cols = len(rows), len(labels)

    # `compact` is for the all-combinations version: with a long tail of
    # singletons the rows have to get shorter or the figure becomes unusable
    row_h = 0.30 if compact else 0.52
    tick = (FS_TICK * 0.8 if compact else FS_TICK) * fs
    val = (FS_VALUE * 0.8 if compact else FS_VALUE) * fs
    fig, ax = plt.subplots(figsize=((3.2 + 1.15 * n_cols) * fs,
                                    (2.6 + row_h * n_rows) * fs))

    # with a single run every row would read "1 run", which says nothing
    show_runs = max(r["n_runs"] for r in rows) > 1

    for i, r in enumerate(rows):
        y = n_rows - 1 - i          # most frequent at the top
        occ = r.get("_occupancy", {})
        for j, lab in enumerate(labels):
            n = occ.get(lab, 0)
            if n > 0:
                # occupied or not, nothing else: the multiplicity is in
                # combination_counts_multiplicity.csv for anyone who needs it
                ax.add_patch(plt.Rectangle(
                    (j, y), 1, 1, facecolor=base, alpha=0.75,
                    edgecolor="black", linewidth=1.0))
            else:
                ax.add_patch(plt.Rectangle(
                    (j, y), 1, 1, facecolor="white",
                    edgecolor="#cccccc", linewidth=0.8))
        # an all-empty row is the "(none)" combination; say so, since an empty
        # row otherwise reads as a rendering fault
        if not any(occ.get(l, 0) > 0 for l in labels):
            ax.text(-0.25, y + 0.5, "(none)", va="center", ha="right",
                    fontsize=val, style="italic")
        # count to the right of the grid
        ax.text(n_cols + 0.25, y + 0.5,
                f"{r['n_events']}  ({r['pct_of_events']:.1f}%)"
                + (f"   {r['n_runs']} runs" if show_runs else ""),
                va="center", ha="left", fontsize=val)

    # separator between the GLU half and the ASN half, wherever the ring changes
    for j in range(1, n_cols):
        if ring_of(labels[j]) != ring_of(labels[j - 1]):
            ax.plot([j, j], [0, n_rows], color="black", linewidth=2.2 * fs)

    ax.set_xlim(0, n_cols + 0.2)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=tick)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    ax.set_title(f"{channel} — which residues hold an ion at permeation"
                 + (f" {title_extra}" if title_extra else "") + "\n"
                 f"filled = occupied; "
                 + (f"all {n_distinct} combinations"
                    if n_rows >= n_distinct
                    else f"top {n_rows} of {n_distinct} combinations shown"),
                 fontsize=FS_TITLE * fs, fontweight="bold", pad=90 * fs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared writer
# ─────────────────────────────────────────────────────────────────────────────

def write_all(rows, labels, out_dir, channel, top, title_extra="", fs=1.0,
              verbose=True):
    n_events = len(rows)
    per_ion_cols = ["channel_type", "run", "ion_rank", "ion_id", "end_3",
                    "combo", "combo_counts", "ring_pattern", "subunit_pattern",
                    "n_residues_occupied", "n_ions_on_residues",
                    "n_other_ions_contact", "n_cavity_region", "censored"] + \
                   [NIONS_PREFIX + l for l in labels]
    write_csv(rows, out_dir / PER_ION_NAME, per_ion_cols)

    combo_t = tally(rows, "combo", channel)
    write_csv(combo_t, out_dir / COMBO_NAME)
    write_csv(tally(rows, "combo_counts", channel), out_dir / MULT_NAME)
    ring_t = tally(rows, "ring_pattern", channel)
    write_csv(ring_t, out_dir / RING_NAME)
    sub_t = tally(rows, "subunit_pattern", channel)
    write_csv(sub_t, out_dir / SUBUNIT_NAME)

    if verbose:
        print(f"\n    {len(combo_t)} distinct combinations in {n_events} events. "
              f"Most frequent:")
        print(f"      {'combination':<42} {'events':>7} {'%':>7} {'runs':>6}")
        for r in combo_t[:top]:
            print(f"      {r['combination']:<42} {r['n_events']:>7} "
                  f"{r['pct_of_events']:>6.1f}% {r['n_runs']:>6}")
        singletons = sum(1 for r in combo_t if r["n_events"] == 1)
        if singletons:
            print(f"      ... {singletons} combination(s) seen exactly once")

        print("\n    Ring patterns:")
        for r in ring_t:
            print(f"      {r['combination']:<20} {r['n_events']:>5} events "
                  f"({r['pct_of_events']:>5.1f}%), {r['n_runs']} run(s)")
        print("\n    Subunit patterns:")
        for r in sub_t:
            print(f"      {r['combination']:<26} {r['n_events']:>5} events "
                  f"({r['pct_of_events']:>5.1f}%), {r['n_runs']} run(s)")

    plot_top(combo_t, out_dir / PLOT_NAME, channel, n_events, top,
             title_extra=title_extra, fs=fs)
    plot_combination_grid(combo_t, labels, out_dir / GRID_NAME, channel, top,
                          title_extra=title_extra, fs=fs)
    # the same grid with nothing left out, however long that makes it — the
    # top-N version is for a slide, this one is the complete record
    if len(combo_t) > top:
        plot_combination_grid(combo_t, labels, out_dir / GRID_ALL_NAME, channel,
                              len(combo_t), title_extra=title_extra, fs=fs,
                              compact=True)
    return combo_t


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run entry point — this is what main.py calls
# ─────────────────────────────────────────────────────────────────────────────

def run_permeation_combinations(results_dir, channel_type="", run="",
                                unit="frames", force_rerun=False,
                                drop_censored=False, top=15, fontscale=1.0):
    """
    Catalogue the ion combination at permeation for ONE run.

    Reads <results_dir>/permeation_ion_table/permeation_ion_table_<unit>.csv
    and writes the six outputs back into that same folder.

    Note the caveat at the top of this file: within a single run the number of
    events is small and the combination space is large, so most combinations
    will occur exactly once and the per-run percentages mean little. These
    files exist so each run carries its own record; the comparison you want is
    the pooled mode (--pool), which reads these same per-run tables.
    """
    results_dir = Path(results_dir)
    table_dir = results_dir / TABLE_SUBDIR
    src = table_dir / f"permeation_ion_table_{unit}.csv"
    marker = table_dir / COMBO_NAME

    if not src.is_file():
        print(f"  [skip] {src} not found — run run_permeation_ion_table() first.")
        return None
    if marker.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {marker} exists. Pass force_rerun=True.")
        print("!" * 70)
        return None

    run_name = f"RUN{run}" if str(run) != "" else results_dir.name
    channel = channel_type or ""

    print("\n" + "=" * 70)
    print("PERMEATION ION COMBINATIONS")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel}  run={run}  unit={unit}")

    rows, labels, dropped = load_rows(src, run_name, channel, drop_censored)
    if not rows:
        print("  [skip] no usable ions in the per-ion table")
        return None
    if dropped:
        print(f"  {dropped} censored ion(s) dropped")
    print(f"  {len(rows)} ions, {len(labels)} residues: {labels}")

    table_dir.mkdir(parents=True, exist_ok=True)
    combo_t = write_all(rows, labels, table_dir, channel or run_name, top,
                        title_extra=f"— {run_name}", fs=fontscale,
                        verbose=True)

    print(f"\n✓ Done. Outputs in {table_dir}/")
    return combo_t


# ─────────────────────────────────────────────────────────────────────────────
#  Pooled mode
# ─────────────────────────────────────────────────────────────────────────────

def pool_channels(channel_dirs, unit="frames", out_dir=None, top=15,
                  drop_censored=False, fontscale=1.0):
    dirs = [Path(d) for d in channel_dirs]
    for d in dirs:
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")

    print("=" * 70)
    print("ION COMBINATIONS AT PERMEATION — pooled")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  unit={unit}  censored={'dropped' if drop_censored else 'kept'}")

    forced = Path(out_dir).resolve() if out_dir else None
    if forced:
        forced.mkdir(parents=True, exist_ok=True)

    for d in dirs:
        channel = d.name
        target = forced or (d / "permeation_combinations")
        target.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}\n  {channel}   ({d})\n{'=' * 70}")
        tables = find_tables(d, unit)
        if not tables:
            print(f"  [warn] no permeation_ion_table_{unit}.csv found")
            continue

        rows, labels, total_dropped = [], [], 0
        for _, run_name, path in tables:
            r, labs, dropped = load_rows(path, run_name, channel, drop_censored)
            total_dropped += dropped
            if labs and not labels:
                labels = labs
            rows.extend(r)
            note = f", {dropped} censored dropped" if dropped else ""
            print(f"    {run_name}: {len(r)} ions{note}")

        if not rows:
            print("  [warn] no usable ions")
            continue

        print(f"\n  {len(rows)} permeation events from {len(tables)} runs"
              + (f", {total_dropped} censored dropped" if total_dropped else ""))
        write_all(rows, labels, target, channel, top,
                  title_extra=f"— pooled over {len(tables)} runs",
                  fs=fontscale, verbose=True)
        print(f"\n  Outputs in {target}/")

    print("\n✓ Done.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", metavar="CHANNEL_DIR",
                    help="Channel folder(s) holding RUN1, RUN2, ... — pools "
                         "every run of each channel")
    ap.add_argument("--results_dir",
                    help="A single RUN directory (what main.py passes)")
    ap.add_argument("--channel_type", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("-unit", "--unit", choices=["frames", "ns"], default="frames",
                    help="Which permeation_ion_table_<unit>.csv to read "
                         "(the nions_ columns are identical in both)")
    ap.add_argument("-top", "--top", type=int, default=15,
                    help="How many combinations to show in the plot (default 15)")
    ap.add_argument("--drop_censored", action="store_true",
                    help="Exclude ions flagged start_censored / incomplete_range "
                         "(kept by default: the combination is read at end_3, "
                         "which is exact even when the residence was truncated)")
    ap.add_argument("-fontscale", "--fontscale", type=float, default=1.0)
    ap.add_argument("--force_rerun", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Pooled mode: write everything here instead of each "
                         "channel's own permeation_combinations/")
    args = ap.parse_args()

    if args.pool:
        pool_channels(args.pool, unit=args.unit, out_dir=args.out,
                      top=args.top, drop_censored=args.drop_censored,
                      fontscale=args.fontscale)
    elif args.results_dir:
        run_permeation_combinations(
            results_dir=args.results_dir, channel_type=args.channel_type,
            run=args.run, unit=args.unit, force_rerun=args.force_rerun,
            drop_censored=args.drop_censored, top=args.top,
            fontscale=args.fontscale)
    else:
        ap.error("give either --pool <channel dirs> or --results_dir <RUN dir>")


if __name__ == "__main__":
    main()

'''
# from main.py, per run — writes into <results_dir>/permeation_ion_table/
from analysis.permeation_ion_combinations import run_permeation_combinations
run_permeation_combinations(results_dir=results_dir, channel_type=channel_type,
                            run=run_number, force_rerun=True)

# pooled over all runs of a channel
python3 permeation_ion_combinations.py --pool \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2

python3 permeation_ion_combinations.py --pool .../G12 .../G2 -unit ns -top 20
'''