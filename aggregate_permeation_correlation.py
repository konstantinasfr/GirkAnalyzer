#!/usr/bin/env python3
"""
aggregate_residue_occupancy.py
==============================
Pools every per-run `residue_occupancy_at_permeation.csv` into one table per
channel: total permeation events, total ion counts per residue, and the
percentages recomputed from those totals.

POOL EVENTS, NOT MEANS
----------------------
Percentages are NOT averaged across runs. Each run's counts are summed first
and the fractions recomputed from the pooled totals, so a run with 37 ions
weighs more than a run with 2 — which is what you want, since a 2-ion run's
"50%" is one event.

    mean_ions      = sum(total_ions)      / sum(n_events)
    frac_occupied  = sum(n_occupied)      / sum(n_events)
    frac_2plus     = sum(n_2plus)         / sum(n_events)

The per-run CSV stores fractions, not the underlying event counts, so those
are reconstructed as round(frac * n_events). That is exact for any n_events
below ~5000 at the 4-decimal rounding the per-run file uses.

Residue labels are read from the files, never hardcoded — G12 and G2 carry
different sets and are pooled separately by channel_type.


STATISTICS  (new in 2026-10-12b)
--------------------------------
THE UNIT OF REPLICATION IS THE RUN, NOT THE PERMEATION EVENT.

Events inside one trajectory are not independent: the same four ions sit in
the same cavity for hundreds of ns, so twenty permeations from RUN3 are not
twenty independent observations of "how often is E141.G1 occupied". Treating
them as independent (a chi-square on ~700 pooled events) would return
p-values near zero for differences resting on two or three trajectories. That
is pseudoreplication, and it is the most likely statistical objection a
reviewer will raise on this figure. Everything below is clustered on the run.

  Error bars      CLUSTER BOOTSTRAP: runs are resampled with replacement
                  (B = 5000 by default), the pooled estimate recomputed inside
                  each resample, and the 2.5/97.5 percentiles taken. The point
                  estimate stays the event-weighted pooled value; only its
                  uncertainty comes from run-to-run spread. A run contributing
                  37 events still carries more weight than one contributing 2,
                  but a result driven by a single run now shows a wide bar.

  Channel test    G12 vs G2, Mann-Whitney U on PER-RUN values, n = number of
                  runs per channel. Performed at the RING level (GLU ring,
                  ASN/ASP ring), because the per-residue labels are not the
                  same set in the two channels — E141.G1 has no G2 counterpart
                  — so a label-matched residue-by-residue test is undefined.
                  The ring value is the sum of its four residues' mean_ions,
                  i.e. the mean number of ions on that ring at permeation,
                  which is additive and therefore recoverable from the
                  per-residue file. Fractions are NOT additive across residues
                  (two residues can be occupied in the same event), so no
                  ring-level percentage is computed.
                  If both channels happen to carry identical residue labels, a
                  per-residue Mann-Whitney is run as well.

  Asymmetry test  Within one channel and one ring, is occupancy spread evenly
                  over the four subunits? FRIEDMAN test across runs, with the
                  run as the block — the repeated-measures form, since the
                  four subunits are measured in the same trajectory. This is
                  the test behind the "contacts concentrate on the GIRK1 face
                  in G12 but spread across subunits in G2" claim.

  Multiplicity    Benjamini-Hochberg FDR across the residues/rings tested in
                  one family. Figures show stars from q, not raw p; the CSV
                  carries both.

  Stars           ns  q >= 0.05      *  q < 0.05
                  **  q < 0.01     ***  q < 0.001

WHAT TO WRITE IN A PAPER
------------------------
Report n at both levels — "n = 20 trajectories, 412 permeation events" — give
the pooled estimate with its cluster-bootstrap 95% CI, name the test and the
clustering unit, and state the multiplicity correction. If a reviewer wants
one model rather than a test per ring, the formally correct upgrade is a mixed
model: a binomial GLMM `occupied ~ channel * ring + (1 | run)` for the
fractions, or a Poisson / negative-binomial GLMM with the same random
intercept for the ion counts. That is beyond what this script does, and the
run-level nonparametric tests here are the conservative stand-in — they will
not manufacture significance that a GLMM would not also find.

Switch the statistics off entirely with -nostats.


INPUT DISCOVERY
---------------
Give it any mix of channel folders, RUN folders, or direct CSV paths; it
searches recursively for files named residue_occupancy_at_permeation.csv and
skips the pooled output of a previous invocation.

To get the G12-vs-G2 comparison you must pass BOTH channels in ONE call —
running the script twice, once per channel, can only produce each channel's
own figure.


PLOT SIZING
-----------
Every text element has an explicit font size, set in the FONT SIZES block
below, so nothing inherits matplotlib's default 10 pt and vanishes once the
figure is shrunk into a slide. Scale the lot from the command line with
-fontscale, or set the canvas with -figsize. Shrinking in PowerPoint scales
text along with everything else, so if the plots still read small, use a
SMALLER -figsize with the same fonts rather than a bigger one.


USAGE
-----
    # both channels in one call — needed for the comparison figure
    python3 aggregate_residue_occupancy.py \
        /media/ziyue/.../girk_analyser_results/G12 \
        /media/ziyue/.../girk_analyser_results/G2

    # one channel, custom output folder, no stats
    python3 aggregate_residue_occupancy.py .../G12 -out .../pooled_G12 -nostats

    # force a channel label when the CSVs have a blank channel_type column
    python3 aggregate_residue_occupancy.py .../G12 -channel G12

    # bigger text, more bootstrap resamples for a final figure
    python3 aggregate_residue_occupancy.py .../G12 .../G2 -fontscale 1.2 -nboot 20000


OUTPUTS
-------
Per channel, in <...>/girk_analyser_results/<CHANNEL>/permeation_correlations/

    residue_occupancy_pooled_<CH>.csv    one row per residue, + bootstrap CIs
    residue_occupancy_per_run_<CH>.csv   that channel's input rows, for audit
    residue_occupancy_pooled_<CH>.png    bar plot with CI error bars
    residue_occupancy_tests_<CH>.csv     within-channel ring asymmetry tests

When two or more channels are pooled in one call, additionally, in the first
channel's folder (or -out):

    channel_comparison_rings.csv         ring-level Mann-Whitney, per residue
                                         too if the label sets match
    channel_comparison_rings.png         grouped bars with significance brackets
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-12b  (cluster-aware statistics + larger fonts)"

INPUT_NAME = "residue_occupancy_at_permeation.csv"
POOLED_STEM = "residue_occupancy_pooled"
PER_RUN_STEM = "residue_occupancy_per_run"
TESTS_STEM = "residue_occupancy_tests"
COMPARE_STEM = "channel_comparison_rings"
# Output names carry the channel, so files from different channels never
# collide — including when everything is sent to one folder with --out.
POOLED_NAME = POOLED_STEM + ".csv"      # legacy name, still skipped on input
DEFAULT_SUBDIR = "permeation_correlations"

CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22",
                 "G12_ML": "#27AE60", "G12_GAT": "#8E44AD"}

# ─────────────────────────────────────────────────────────────────────────────
#  FONT SIZES — everything is multiplied by -fontscale at run time.
# ─────────────────────────────────────────────────────────────────────────────
FS_TITLE    = 25   # suptitle
FS_SUBTITLE = 21   # per-panel titles
FS_AXLABEL  = 21   # axis labels
FS_TICK     = 19   # residue names on x, numbers on y
FS_VALUE    = 16   # numbers printed on top of the bars
FS_LEGEND   = 18
FS_STAR     = 20   # significance stars

FIGSIZE = (16.0, 7.2)   # inches
DPI = 200
BAR_EDGE_LW = 1.0
ERR_LW = 1.6            # error-bar line width

ALPHA_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

def find_csvs(inputs):
    """
    Accept folders (searched recursively) or direct .csv paths.

    Returns [(csv_path, search_root)]. The search root is remembered because
    it is what the per-channel output folder is derived from: point the script
    at .../G12 and that run's results go back under .../G12/.
    """
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() == ".csv":
            found.append((p, p.resolve().parent))
        elif p.is_dir():
            root = p.resolve()
            found.extend((f, root) for f in sorted(p.rglob(INPUT_NAME)))
        else:
            print(f"  [warn] not found, skipped: {p}")
    # drop this script's own outputs if they sit inside a searched folder
    found = [(f, r) for f, r in found
             if not (f.stem.startswith(POOLED_STEM)
                     or f.stem.startswith(PER_RUN_STEM)
                     or f.stem.startswith(TESTS_STEM)
                     or f.stem.startswith(COMPARE_STEM))]
    # de-duplicate while keeping order
    seen, unique = set(), []
    for f, r in found:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append((f, r))
    return unique


def channel_output_dir(root, channel, subdir):
    """
    Where this channel's results belong.

      root is .../G12                  -> .../G12/<subdir>
      root is .../girk_analyser_results -> .../girk_analyser_results/G12/<subdir>
                                           (if that channel folder exists)
      anything else                    -> root/<subdir>

    So both `prog .../G12 .../G2` and `prog .../girk_analyser_results` put the
    output under the right channel.
    """
    root = Path(root)
    if root.name == channel:
        base = root
    elif (root / channel).is_dir():
        base = root / channel
    else:
        base = root
    return base / subdir


def _num(row, key, default=0.0):
    v = str(row.get(key, "")).strip()
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_csv(path, root, channel_override=""):
    """
    Returns a list of dicts with the EVENT COUNTS reconstructed, which is what
    pooling needs. The per-run fractions are kept alongside, because the
    run-level tests use them directly rather than the reconstruction.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  [warn] empty: {path}")
        return []
    if "residue" not in rows[0]:
        print(f"  [warn] no 'residue' column, not one of ours: {path}")
        return []

    out = []
    for r in rows:
        n_events = int(round(_num(r, "n_events")))
        if n_events <= 0:
            continue
        # counts: reconstructed from the stored fractions
        n_occ = int(round(_num(r, "frac_occupied") * n_events))
        n_2p = int(round(_num(r, "frac_2plus") * n_events))
        total_ions = int(round(_num(r, "total_ions")))
        out.append(dict(
            channel_type=channel_override or str(r.get("channel_type", "")).strip(),
            run=str(r.get("run", "")).strip(),
            residue=str(r.get("residue", "")).strip(),
            n_events=n_events,
            total_ions=total_ions,
            n_occupied=n_occ,
            n_2plus=n_2p,
            max_ions=int(round(_num(r, "max_ions"))),
            # per-run values, exact as stored — used by the run-level tests
            run_mean_ions=total_ions / n_events,
            run_frac_occupied=_num(r, "frac_occupied"),
            run_frac_2plus=_num(r, "frac_2plus"),
            source=str(path),
            _root=str(root),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Pooling
# ─────────────────────────────────────────────────────────────────────────────

def pool(all_rows):
    """
    Sum counts per (channel_type, residue), then recompute the percentages.

    One sanity point worth knowing: n_events is the number of PERMEATION
    EVENTS in a run and is identical across that run's 8 residue rows. So
    summing it per residue gives the right denominator, and all residues of a
    channel end up with the same total_events — if they don't, a run is
    missing a residue row and the mismatch is reported.
    """
    channels, residues = [], {}
    agg = {}
    for r in all_rows:
        ch, res = r["channel_type"] or "(unlabelled)", r["residue"]
        if ch not in channels:
            channels.append(ch)
            residues[ch] = []
        if res not in residues[ch]:
            residues[ch].append(res)
        key = (ch, res)
        a = agg.setdefault(key, dict(total_events=0, total_ions=0,
                                     n_occupied=0, n_2plus=0, max_ions=0,
                                     runs=set()))
        a["total_events"] += r["n_events"]
        a["total_ions"] += r["total_ions"]
        a["n_occupied"] += r["n_occupied"]
        a["n_2plus"] += r["n_2plus"]
        a["max_ions"] = max(a["max_ions"], r["max_ions"])
        a["runs"].add(r["run"])

    out = []
    for ch in channels:
        for res in residues[ch]:
            a = agg[(ch, res)]
            n = a["total_events"]
            out.append(dict(
                channel_type=ch,
                residue=res,
                n_runs=len(a["runs"]),
                total_events=n,
                total_ions=a["total_ions"],
                n_occupied=a["n_occupied"],
                n_2plus=a["n_2plus"],
                max_ions=a["max_ions"],
                mean_ions=round(a["total_ions"] / n, 4) if n else "",
                pct_occupied=round(100 * a["n_occupied"] / n, 2) if n else "",
                pct_2plus=round(100 * a["n_2plus"] / n, 2) if n else "",
            ))
    return out, channels


def check_denominators(pooled, channels):
    """Warn if a channel's residues don't share the same event total."""
    for ch in channels:
        rows = [r for r in pooled if r["channel_type"] == ch]
        totals = {r["total_events"] for r in rows}
        if len(totals) > 1:
            print(f"  [warn] {ch}: residues have different event totals "
                  f"{sorted(totals)} — some run is missing residue rows, so "
                  f"percentages are not on a common denominator.")


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def ring_of(label):
    """
    GLU ring vs ASN/ASP ring from the residue label's one-letter code
    (E141.G1 -> GLU, D173.G1 / N184.2c -> ASN/ASP). Unknown codes are kept in
    their own group rather than being forced into one of the two.
    """
    c = (label or "")[:1].upper()
    if c == "E":
        return "GLU"
    if c in ("D", "N"):
        return "ASN/ASP"
    return "other"


def per_run_index(all_rows):
    """(channel, residue) -> {run: row}, for the run-level tests."""
    idx = {}
    for r in all_rows:
        ch = r["channel_type"] or "(unlabelled)"
        idx.setdefault((ch, r["residue"]), {})[r["run"]] = r
    return idx


def cluster_bootstrap_ci(run_rows, numer_key, denom_key="n_events",
                         nboot=5000, seed=0):
    """
    Percentile CI for a pooled ratio, resampling RUNS with replacement.

    run_rows is one run's row per entry. The statistic recomputed inside each
    resample is exactly the pooled estimator — sum(numer)/sum(denom) — so the
    CI is for the number actually plotted, and run-to-run spread is what
    widens it. Returns (lo, hi), or (nan, nan) with fewer than 2 runs.
    """
    if len(run_rows) < 2:
        return float("nan"), float("nan")
    num = np.array([r[numer_key] for r in run_rows], dtype=float)
    den = np.array([r[denom_key] for r in run_rows], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(nboot, n))
    den_s = den[idx].sum(axis=1)
    num_s = num[idx].sum(axis=1)
    good = den_s > 0
    if not good.any():
        return float("nan"), float("nan")
    stat = num_s[good] / den_s[good]
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U. Returns (U, p, note) — p is nan without scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or b.size < 3:
        return float("nan"), float("nan"), f"too few runs (n={a.size}, {b.size})"
    try:
        from scipy import stats
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p), ""
    except ImportError:
        return float("nan"), float("nan"), "scipy not installed"
    except Exception as e:
        return float("nan"), float("nan"), str(e)


def friedman(columns):
    """
    Repeated-measures test across >=3 matched groups (here: the 4 subunits of
    a ring, blocked on run). columns is a list of equal-length sequences.
    Returns (chi2, p, note).
    """
    if len(columns) < 3:
        return float("nan"), float("nan"), "needs >=3 groups"
    lens = {len(c) for c in columns}
    if len(lens) != 1:
        return float("nan"), float("nan"), "unequal run coverage across subunits"
    n = lens.pop()
    if n < 3:
        return float("nan"), float("nan"), f"too few complete runs (n={n})"
    try:
        from scipy import stats
        chi2, p = stats.friedmanchisquare(*columns)
        return float(chi2), float(p), ""
    except ImportError:
        return float("nan"), float("nan"), "scipy not installed"
    except Exception as e:
        return float("nan"), float("nan"), str(e)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values. NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    sub = p[ok]
    m = sub.size
    order = np.argsort(sub)
    ranked = sub[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def stars(q):
    if q != q:
        return ""
    for thr, s in ALPHA_LEVELS:
        if q < thr:
            return s
    return "ns"


def _fmt_p(v):
    if v != v:
        return ""
    return f"{v:.2e}" if v < 1e-4 else round(float(v), 6)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def compute_channel_stats(pooled, all_rows, nboot, seed):
    """
    Adds cluster-bootstrap CIs to every pooled row, in place, and returns the
    per-run index for reuse by the tests.
    """
    idx = per_run_index(all_rows)
    for row in pooled:
        key = (row["channel_type"], row["residue"])
        run_rows = list(idx.get(key, {}).values())
        lo_m, hi_m = cluster_bootstrap_ci(run_rows, "total_ions",
                                          nboot=nboot, seed=seed)
        lo_o, hi_o = cluster_bootstrap_ci(run_rows, "n_occupied",
                                          nboot=nboot, seed=seed + 1)
        lo_2, hi_2 = cluster_bootstrap_ci(run_rows, "n_2plus",
                                          nboot=nboot, seed=seed + 2)
        row["mean_ions_ci_lo"] = round(lo_m, 4) if lo_m == lo_m else ""
        row["mean_ions_ci_hi"] = round(hi_m, 4) if hi_m == hi_m else ""
        row["pct_occupied_ci_lo"] = round(100 * lo_o, 2) if lo_o == lo_o else ""
        row["pct_occupied_ci_hi"] = round(100 * hi_o, 2) if hi_o == hi_o else ""
        row["pct_2plus_ci_lo"] = round(100 * lo_2, 2) if lo_2 == lo_2 else ""
        row["pct_2plus_ci_hi"] = round(100 * hi_2, 2) if hi_2 == hi_2 else ""
    return idx


def ring_values_per_run(idx, channel, residues):
    """
    {run: ring mean ions} — the sum of the ring's residues' per-run mean_ions.
    Only runs covering every residue of the ring are kept, so the sum is on a
    common denominator. Returns (dict, n_dropped).
    """
    per_res = [idx.get((channel, res), {}) for res in residues]
    if not per_res or any(not d for d in per_res):
        return {}, 0
    common = set(per_res[0])
    for d in per_res[1:]:
        common &= set(d)
    union = set().union(*[set(d) for d in per_res])
    dropped = len(union) - len(common)
    out = {run: sum(d[run]["run_mean_ions"] for d in per_res) for run in common}
    return out, dropped


def within_channel_tests(idx, channel, residues, metric="run_mean_ions"):
    """Friedman across the subunits of each ring, blocked on run."""
    rows = []
    rings = {}
    for res in residues:
        rings.setdefault(ring_of(res), []).append(res)

    for ring, members in rings.items():
        if len(members) < 3:
            continue
        per_res = [idx.get((channel, m), {}) for m in members]
        if any(not d for d in per_res):
            common = []
        else:
            common = sorted(set.intersection(*[set(d) for d in per_res]))
        if len(common) < 3:
            rows.append(dict(channel_type=channel,
                             test="Friedman (subunit asymmetry)",
                             ring=ring, group=" vs ".join(members),
                             n_runs=len(common), statistic="", p_value="",
                             q_value="", stars="",
                             note=f"too few runs covering all {len(members)} subunits"))
            continue
        columns = [[d[run][metric] for run in common] for d in per_res]
        chi2, p, note = friedman(columns)
        rows.append(dict(channel_type=channel,
                         test="Friedman (subunit asymmetry)",
                         ring=ring, group=" vs ".join(members),
                         n_runs=len(common),
                         statistic=round(chi2, 4) if chi2 == chi2 else "",
                         p_value=_fmt_p(p), q_value="", stars="", note=note))

    qs = bh_fdr([_f(r["p_value"]) if r["p_value"] != "" else np.nan
                 for r in rows])
    for r, q in zip(rows, qs):
        r["q_value"] = _fmt_p(q)
        r["stars"] = stars(q)
    return rows


def between_channel_tests(idx, pooled, channels):
    """
    Ring-level Mann-Whitney on per-run ring mean ions, plus a per-residue
    version when the channels share an identical label set.
    Returns (rows, ring_table) where ring_table feeds the comparison figure.
    """
    if len(channels) < 2:
        return [], {}

    res_by_ch = {ch: [r["residue"] for r in pooled if r["channel_type"] == ch]
                 for ch in channels}

    rings_by_ch = {}
    for ch, residues in res_by_ch.items():
        d = {}
        for res in residues:
            d.setdefault(ring_of(res), []).append(res)
        rings_by_ch[ch] = d

    rows = []
    ring_table = {}      # ring -> {channel: (values_per_run, pooled_point, members)}

    all_rings = []
    for ch in channels:
        for ring in rings_by_ch[ch]:
            if ring not in all_rings:
                all_rings.append(ring)

    for ring in all_rings:
        ring_table[ring] = {}
        for ch in channels:
            members = rings_by_ch[ch].get(ring, [])
            if not members:
                continue
            vals, dropped = ring_values_per_run(idx, ch, members)
            if dropped:
                print(f"  [warn] {ch} {ring}: {dropped} run(s) missing a "
                      f"subunit row, excluded from the ring value")
            point = sum(_f(r["mean_ions"]) for r in pooled
                        if r["channel_type"] == ch and r["residue"] in members
                        and r["mean_ions"] != "")
            ring_table[ring][ch] = (vals, point, members)

        present = [ch for ch in channels if ch in ring_table[ring]]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a_ch, b_ch = present[i], present[j]
                a = list(ring_table[ring][a_ch][0].values())
                b = list(ring_table[ring][b_ch][0].values())
                u, p, note = mannwhitney(a, b)
                rows.append(dict(
                    test="Mann-Whitney U (per-run ring mean ions)",
                    ring=ring, group_a=a_ch, group_b=b_ch,
                    n_runs_a=len(a), n_runs_b=len(b),
                    median_a=round(float(np.median(a)), 4) if a else "",
                    median_b=round(float(np.median(b)), 4) if b else "",
                    statistic=round(u, 3) if u == u else "",
                    p_value=_fmt_p(p), q_value="", stars="", note=note))

    # per-residue, only when the label sets match exactly
    sets = [tuple(sorted(set(res_by_ch[ch]))) for ch in channels]
    if len(set(sets)) == 1:
        for res in sorted(set(res_by_ch[channels[0]])):
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    a_ch, b_ch = channels[i], channels[j]
                    a = [r["run_mean_ions"]
                         for r in idx.get((a_ch, res), {}).values()]
                    b = [r["run_mean_ions"]
                         for r in idx.get((b_ch, res), {}).values()]
                    u, p, note = mannwhitney(a, b)
                    rows.append(dict(
                        test="Mann-Whitney U (per-run residue mean ions)",
                        ring=res, group_a=a_ch, group_b=b_ch,
                        n_runs_a=len(a), n_runs_b=len(b),
                        median_a=round(float(np.median(a)), 4) if a else "",
                        median_b=round(float(np.median(b)), 4) if b else "",
                        statistic=round(u, 3) if u == u else "",
                        p_value=_fmt_p(p), q_value="", stars="", note=note))
    else:
        print("  [note] residue labels differ between channels, so only the "
              "ring-level comparison is defined — a label-matched "
              "residue-by-residue test would be comparing different residues.")

    # FDR within each test family separately
    for family in {r["test"] for r in rows}:
        sel = [r for r in rows if r["test"] == family]
        qs = bh_fdr([_f(r["p_value"]) if r["p_value"] != "" else np.nan
                     for r in sel])
        for r, q in zip(sel, qs):
            r["q_value"] = _fmt_p(q)
            r["stars"] = stars(q)

    return rows, ring_table


# ─────────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, path):
    if not rows:
        print(f"  [warn] nothing to write to {Path(path).name}")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols and not k.startswith("_"):
                cols.append(k)   # keys starting with "_" are internal bookkeeping
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {Path(path).name}  ({len(rows)} rows)")


def _err(vals, los, his):
    """Asymmetric yerr array for plt.bar, NaN-safe."""
    lo, hi = [], []
    for v, l, h in zip(vals, los, his):
        lv, hv = _f(l), _f(h)
        lo.append(max(0.0, v - lv) if lv == lv else 0.0)
        hi.append(max(0.0, hv - v) if hv == hv else 0.0)
    return np.array([lo, hi])


def plot_pooled(rows, out_path, title, color="#2980B9", sort_by=None,
                fs=1.0, figsize=None, show_ci=True,
                title_left="Ions on each residue at permeation",
                title_right="Fraction of permeations with the residue occupied"):
    """
    Two-panel bar figure. `fs` scales every font size and, when figsize is not
    given explicitly, the figure size too, so proportions stay the same.
    Error bars are the cluster-bootstrap 95% CIs from compute_channel_stats.
    title_left / title_right let other aggregators (e.g. occupancy at SF exit)
    reuse the figure for another frame.
    """
    if not rows:
        return
    if sort_by == "mean":
        rows = sorted(rows, key=lambda r: -_f(r["mean_ions"]))
    labels = [r["residue"] for r in rows]
    means = [_f(r["mean_ions"]) for r in rows]
    pcts = [_f(r["pct_occupied"]) for r in rows]
    p2 = [_f(r["pct_2plus"]) for r in rows]
    x = np.arange(len(labels))

    has_ci = show_ci and "mean_ions_ci_lo" in rows[0]
    if has_ci:
        err_m = _err(means, [r.get("mean_ions_ci_lo", "") for r in rows],
                     [r.get("mean_ions_ci_hi", "") for r in rows])
        err_o = _err(pcts, [r.get("pct_occupied_ci_lo", "") for r in rows],
                     [r.get("pct_occupied_ci_hi", "") for r in rows])
    else:
        err_m = err_o = None

    if figsize is None:
        figsize = (FIGSIZE[0] * fs, FIGSIZE[1] * fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ekw = dict(ecolor="black", capsize=5 * fs,
               error_kw=dict(lw=ERR_LW * fs, capthick=ERR_LW * fs)) if has_ci else {}

    ax1.bar(x, means, color=color, alpha=0.85, edgecolor="black",
            linewidth=BAR_EDGE_LW, yerr=err_m, **ekw)
    up_m = err_m[1] if has_ci else np.zeros(len(means))
    top_m = max([m + e for m, e in zip(means, up_m)]) if means else 1
    for xi, v, e in zip(x, means, up_m):
        ax1.text(xi, v + e, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=FS_VALUE * fs)
    ax1.set_ylabel("Mean number of other ions", fontsize=FS_AXLABEL * fs)
    ax1.set_ylim(0, top_m * 1.25 if top_m > 0 else 1)
    ax1.set_title(title_left, fontsize=FS_SUBTITLE * fs)

    ax2.bar(x, pcts, color=color, alpha=0.55, edgecolor="black",
            linewidth=BAR_EDGE_LW, hatch="//", label="≥1 ion",
            yerr=err_o, **ekw)
    ax2.bar(x, p2, color=color, alpha=0.95, edgecolor="black",
            linewidth=BAR_EDGE_LW, label="≥2 ions")
    # The axis stops at 100 because that is the ceiling of a percentage. A bar
    # (or its upper CI) close to 100 would push its label off the top, so for
    # those the label goes inside the bar instead of above it.
    up_o = err_o[1] if has_ci else np.zeros(len(pcts))
    for xi, v, e in zip(x, pcts, up_o):
        if v + e > 90:
            ax2.text(xi, min(v, 98) - 2, f"{v:.0f}%", ha="center", va="top",
                     fontsize=FS_VALUE * fs, color="white", fontweight="bold")
        else:
            ax2.text(xi, v + e, f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=FS_VALUE * fs)
    ax2.set_ylabel("% of permeation events", fontsize=FS_AXLABEL * fs)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title(title_right, fontsize=FS_SUBTITLE * fs)
    ax2.legend(fontsize=FS_LEGEND * fs, loc="upper right", framealpha=0.9)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=FS_TICK * fs)
        ax.tick_params(axis="y", labelsize=FS_TICK * fs)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    sub = "error bars: 95% CI, bootstrap over runs" if has_ci else ""
    fig.suptitle(title + (f"\n{sub}" if sub else ""),
                 fontsize=FS_TITLE * fs, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    # bbox_inches="tight" keeps the larger rotated residue labels from being
    # clipped at the bottom edge.
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


def plot_channel_comparison(ring_table, test_rows, out_path, fs=1.0,
                            figsize=None):
    """
    Grouped bars: ring mean ions per channel, one dot per run overlaid, with a
    significance bracket carrying the FDR-corrected stars. The dots are there
    because with ~20 runs the reader should see the spread the test is on.
    """
    rings = [r for r in ring_table if ring_table[r]]
    if not rings:
        return
    channels = []
    for r in rings:
        for ch in ring_table[r]:
            if ch not in channels:
                channels.append(ch)
    if len(channels) < 2:
        return

    if figsize is None:
        figsize = (max(7.5, 3.8 * len(rings)) * fs, 7.0 * fs)
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / len(channels)
    rng = np.random.default_rng(0)
    top = 0.0
    for i, ch in enumerate(channels):
        xs, heights, n_runs_seen = [], [], 0
        for j, ring in enumerate(rings):
            if ch not in ring_table[ring]:
                continue
            vals, point, _m = ring_table[ring][ch]
            pos = j + i * width - 0.4 + width / 2
            xs.append(pos)
            heights.append(point)
            v = np.array(list(vals.values()), dtype=float)
            n_runs_seen = max(n_runs_seen, v.size)
            if v.size:
                ax.scatter(pos + rng.uniform(-width * 0.22, width * 0.22, v.size),
                           v, s=40 * fs, color="black", alpha=0.45,
                           zorder=4, edgecolors="none")
                top = max(top, float(v.max()))
            top = max(top, point)
        ax.bar(xs, heights, width=width * 0.9,
               color=CHANNEL_COLOR.get(ch, "#777777"), alpha=0.85,
               edgecolor="black", linewidth=BAR_EDGE_LW, zorder=2,
               label=f"{ch}  (n = {n_runs_seen} runs)")

    # significance brackets, ring-level tests only
    ring_tests = [t for t in test_rows
                  if t["test"].startswith("Mann-Whitney U (per-run ring")]
    for k, t in enumerate(ring_tests):
        if t["ring"] not in rings or not t["stars"]:
            continue
        j = rings.index(t["ring"])
        ia, ib = channels.index(t["group_a"]), channels.index(t["group_b"])
        xa = j + ia * width - 0.4 + width / 2
        xb = j + ib * width - 0.4 + width / 2
        y = top * 1.10
        h = top * 0.02
        ax.plot([xa, xa, xb, xb], [y, y + h, y + h, y], color="black",
                lw=1.5 * fs, zorder=5)
        ax.text((xa + xb) / 2, y + h, t["stars"], ha="center", va="bottom",
                fontsize=FS_STAR * fs,
                fontweight="bold" if t["stars"] != "ns" else "normal")

    ax.set_xticks(range(len(rings)))
    ax.set_xticklabels(rings, fontsize=FS_TICK * fs)
    ax.tick_params(axis="y", labelsize=FS_TICK * fs)
    ax.set_ylabel("Mean ions on the ring at permeation", fontsize=FS_AXLABEL * fs)
    ax.set_ylim(0, top * 1.22 if top > 0 else 1)
    ax.set_title("Ring occupancy at permeation, by channel\n"
                 "bars = event-weighted pooled value, dots = individual runs\n"
                 "Mann-Whitney U on per-run values, BH-FDR corrected",
                 fontsize=FS_SUBTITLE * fs, fontweight="bold")
    # legend outside the axes: inside it would sit on top of the brackets
    ax.legend(fontsize=FS_LEGEND * fs, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), framealpha=0.9, borderaxespad=0)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Channel folders, RUN folders, or direct CSV paths")
    ap.add_argument("-channel", "--channel", dest="channel", default="",
                    help="Channel type (G12, G2, G12_ML, G12_GAT). Forces this "
                         "label on every input row, is used in the plot title "
                         "and file names, and decides the output folder. "
                         "Needed when the CSVs' channel_type column is blank.")
    ap.add_argument("-subdir", "--subdir", default=DEFAULT_SUBDIR,
                    help=f"Folder name created inside each channel folder "
                         f"(default: {DEFAULT_SUBDIR})")
    ap.add_argument("-out", "--out", default=None,
                    help="Write EVERYTHING to this one directory instead of "
                         "splitting per channel")
    ap.add_argument("-sort", "--sort", choices=["file", "mean"], default="file",
                    help="Bar order: file = as written, mean = descending "
                         "(default: file, so G12 and G2 stay comparable)")
    ap.add_argument("-fontscale", "--fontscale", type=float, default=1.0,
                    help="Multiply every font size (and the figure size) by "
                         "this. Use >1 if the plots still read small in PPT.")
    ap.add_argument("-figsize", "--figsize", type=float, nargs=2,
                    metavar=("W", "H"), default=None,
                    help=f"Figure size in inches (default "
                         f"{FIGSIZE[0]} {FIGSIZE[1]}). Set explicitly to stop "
                         f"-fontscale from also scaling the canvas — a smaller "
                         f"canvas with the same fonts means larger text.")
    ap.add_argument("-nostats", "--nostats", action="store_true",
                    help="Skip all statistics: no bootstrap CIs, no tests")
    ap.add_argument("-nboot", "--nboot", type=int, default=5000,
                    help="Cluster-bootstrap resamples (default 5000; use "
                         "20000 for a final figure)")
    ap.add_argument("-seed", "--seed", type=int, default=0,
                    help="Bootstrap RNG seed, so figures are reproducible")
    args = ap.parse_args()

    print("=" * 70)
    print("AGGREGATE RESIDUE OCCUPANCY AT PERMEATION")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    if args.channel:
        print(f"  channel: {args.channel}  (forced on every input row)")
    figsize = tuple(args.figsize) if args.figsize else None
    if args.fontscale != 1.0 or figsize:
        print(f"  plots:   fontscale {args.fontscale}, figsize "
              f"{figsize or 'auto'}, dpi {DPI}")

    csvs = find_csvs(args.inputs)
    if not csvs:
        sys.exit(f"ERROR: no {INPUT_NAME} found under the given paths.")
    print(f"  Found {len(csvs)} input file(s)")

    all_rows = []
    for path, root in csvs:
        rows = load_csv(path, root, args.channel)
        if rows:
            ch = rows[0]["channel_type"] or "(unlabelled)"
            print(f"    {ch:<12} run {rows[0]['run'] or '?':<6} "
                  f"{rows[0]['n_events']:>5} events, {len(rows)} residues   {path}")
            all_rows.extend(rows)
    if not all_rows:
        sys.exit("ERROR: no usable rows loaded.")

    pooled, channels = pool(all_rows)
    check_denominators(pooled, channels)

    idx = {}
    if not args.nostats:
        print(f"\n  Statistics: cluster bootstrap over runs, B = {args.nboot}, "
              f"seed = {args.seed}")
        idx = compute_channel_stats(pooled, all_rows, args.nboot, args.seed)

    forced_out = Path(args.out).resolve() if args.out else None
    if forced_out:
        forced_out.mkdir(parents=True, exist_ok=True)
        write_csv(all_rows, forced_out / f"{PER_RUN_STEM}_all.csv")
        write_csv(pooled, forced_out / f"{POOLED_STEM}_all.csv")

    written = []
    first_out_dir = None
    for ch in channels:
        rows = [r for r in pooled if r["channel_type"] == ch]
        ch_inputs = [r for r in all_rows if (r["channel_type"] or
                                             "(unlabelled)") == ch]
        n_runs = max(r["n_runs"] for r in rows)
        n_ev = max(r["total_events"] for r in rows)

        if forced_out:
            out_dir = forced_out
        else:
            # Each channel's results go back under that channel's own folder.
            # Several roots can feed one channel (e.g. two RUN folders passed
            # separately); the most common one wins.
            roots = [r["_root"] for r in ch_inputs]
            root = max(set(roots), key=roots.count)
            out_dir = channel_output_dir(root, ch, args.subdir)
            out_dir.mkdir(parents=True, exist_ok=True)
        if first_out_dir is None:
            first_out_dir = out_dir

        print(f"\n  === {ch}  —  {n_runs} runs, {n_ev} permeation events ===")
        if args.nostats:
            print(f"  {'residue':<12} {'mean':>7} {'>=1 ion':>9} {'>=2 ions':>9} "
                  f"{'ions':>7}")
            for r in sorted(rows, key=lambda d: -_f(d["mean_ions"])):
                print(f"  {r['residue']:<12} {_f(r['mean_ions']):>7.3f} "
                      f"{_f(r['pct_occupied']):>8.1f}% "
                      f"{_f(r['pct_2plus']):>8.1f}% {r['total_ions']:>7}")
        else:
            print(f"  {'residue':<12} {'mean':>7} {'95% CI':>18} "
                  f"{'>=1 ion':>9} {'95% CI':>18}")
            for r in sorted(rows, key=lambda d: -_f(d["mean_ions"])):
                ci_m = (f"[{_f(r['mean_ions_ci_lo']):.2f}, "
                        f"{_f(r['mean_ions_ci_hi']):.2f}]"
                        if r["mean_ions_ci_lo"] != "" else "—")
                ci_o = (f"[{_f(r['pct_occupied_ci_lo']):.0f}, "
                        f"{_f(r['pct_occupied_ci_hi']):.0f}]%"
                        if r["pct_occupied_ci_lo"] != "" else "—")
                print(f"  {r['residue']:<12} {_f(r['mean_ions']):>7.3f} "
                      f"{ci_m:>18} {_f(r['pct_occupied']):>8.1f}% {ci_o:>18}")

        write_csv(ch_inputs, out_dir / f"{PER_RUN_STEM}_{ch}.csv")
        write_csv(rows, out_dir / f"{POOLED_STEM}_{ch}.csv")

        if not args.nostats:
            residues = [r["residue"] for r in rows]
            tests = within_channel_tests(idx, ch, residues)
            if tests:
                print(f"\n  Subunit asymmetry within {ch} "
                      f"(Friedman, blocked on run):")
                for t in tests:
                    detail = (f"chi2={t['statistic']}, p={t['p_value']}, "
                              f"q={t['q_value']} {t['stars']}"
                              if t["statistic"] != "" else f"[{t['note']}]")
                    print(f"    {t['ring']:<9} n={t['n_runs']:<3} {detail}")
                write_csv(tests, out_dir / f"{TESTS_STEM}_{ch}.csv")

        plot_pooled(rows, out_dir / f"{POOLED_STEM}_{ch}.png",
                    f"{ch} — residue occupancy at permeation "
                    f"({n_runs} runs, {n_ev} permeating ions)",
                    color=CHANNEL_COLOR.get(ch, "#2980B9"),
                    sort_by=None if args.sort == "file" else "mean",
                    fs=args.fontscale, figsize=figsize,
                    show_ci=not args.nostats)
        written.append((ch, out_dir))

    # ---- between-channel comparison ----
    if not args.nostats and len(channels) >= 2:
        print(f"\n  === channel comparison: {' vs '.join(channels)} ===")
        test_rows, ring_table = between_channel_tests(idx, pooled, channels)
        for t in test_rows:
            if not t["test"].startswith("Mann-Whitney U (per-run ring"):
                continue
            detail = (f"U={t['statistic']}, p={t['p_value']}, q={t['q_value']} "
                      f"{t['stars']}" if t["statistic"] != "" else f"[{t['note']}]")
            print(f"    {t['ring']:<9} {t['group_a']} (med {t['median_a']}, "
                  f"n={t['n_runs_a']}) vs {t['group_b']} (med {t['median_b']}, "
                  f"n={t['n_runs_b']}): {detail}")
        cmp_dir = forced_out or first_out_dir
        if test_rows:
            write_csv(test_rows, cmp_dir / f"{COMPARE_STEM}.csv")
            plot_channel_comparison(ring_table, test_rows,
                                    cmp_dir / f"{COMPARE_STEM}.png",
                                    fs=args.fontscale)
    elif not args.nostats:
        print("\n  [note] only one channel loaded — pass the G12 and G2 folders "
              "in the SAME call to get the between-channel comparison.")

    print("\n✓ Done.")
    for ch, d in written:
        print(f"    {ch:<12} -> {d}/")


if __name__ == "__main__":
    main()
'''
# both channels in one call — this is what produces the comparison figure
python3 aggregate_residue_occupancy.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/

# final figure: more resamples, bigger text
python3 aggregate_residue_occupancy.py .../G12/ .../G2/ -nboot 20000 -fontscale 1.15

# old behaviour, one channel, no statistics
python3 aggregate_residue_occupancy.py .../G12/ --channel G12 -nostats
'''
'''
python3 aggregate_residue_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12

python3 aggregate_residue_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ --channel G2

# even bigger text for a slide
python3 aggregate_residue_occupancy.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ --channel G12 -fontscale 1.3
'''
'''
python3 aggregate_permeation_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12

python3 aggregate_permeation_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2

python3 aggregate_permeation_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--channel G12_D173N

python3 aggregate_permeation_correlation.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--channel G12_E141Q
'''