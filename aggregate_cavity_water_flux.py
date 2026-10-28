#!/usr/bin/env python3
"""
Aggregate Cavity Water Flux & Residence
=======================================
Loads cavity_water_flux.pkl from every RUN* folder of one channel and
combines the water crossings and the water residence times.

Input per run
-------------
    <base_dir>/RUN<n>/water_analysis/cavity_water_flux/cavity_water_flux.pkl
(written by analysis/cavity_water_flux.py)

How runs are combined
---------------------
- Fluxes: each run's crossings are turned into a rate (per ns) first, so
  runs of different length weigh the same. Mean ± std is over runs.
- Residence: every complete visit from every run is pooled for the overall
  histogram / boxplot / mean / median. The per-run mean and median are also
  reported, and the per-run boxplot shows whether one run dominates the pool.
  Censored visits (still inside at a run's last frame) are left out, as in
  the per-run analysis.

Do not mix channels with different frame spacing (G12 0.075 ns vs G2
0.4 ns) in one call: the script warns if the runs disagree.

What it produces   <output_dir>  (default <base_dir>/water_analysis/cavity_water_flux_across_runs)
-------------------------------------------------------------------------------------------------
1. cavity_water_flux_per_run_<CH>.csv     one row per run: crossings, rates, residence stats
2. cavity_water_flux_across_runs_<CH>.txt across-run flux mean ± std, pooled residence
3. cavity_water_flux_across_runs_<CH>.pkl all of the above
4. flux_rates_<CH>.png                    forward / back / net per ns, bar = mean, dots = runs
5. flux_net_per_run_<CH>.png              net flux per run at each boundary
6. residence_hist_<CH>.png                pooled histogram per region
7. residence_box_<CH>.png                 pooled boxplot per region
8. residence_box_per_run_<CH>.png         one panel per region, one box per run
9. cavity_water_budget_<CH>.png           where the cavity's net water gain came from
10. cavity_water_overview_<CH>.png        the whole story on one sheet: a pore diagram
                                          with in / out / net waters at every boundary
                                          (mean per run), and the cavity's net gain over
                                          time split by source (mean over runs)
11. sf_transits_<CH>.png                  full SF transits per run, and net transits
                                          against K+ permeations
12. cavity_water_origin_<CH>.png          which way waters enter and leave the cavity:
                                          rates and shares, all runs and per run
13. cavity_water_composition_<CH>.png     share of the cavity's waters by the way they
                                          came in, over time (mean over runs)

Full SF transits
----------------
Waters that cross the whole SF, counted with two planes at the upper1 and
lower1 CA rings (see analysis/cavity_water_flux.py). Unlike sf_in / sf_back
they exclude waters that step in and out of the same end. Runs analysed
before this was added have no transit data and are left out of these numbers.

Where cavity water comes from
-----------------------------
Every water entering the cavity comes in from below the HBC (hbc_back), from
the sides (side_in) or from the SF (sf_in). The origin plot gives those
entry rates and their shares, from the crossing counts, so it works for
every run. The composition plot needs runs analysed with the current
cavity_water_flux.py: it shows, at each frame, what share of the waters in
the cavity came in through each route.

Cavity water budget
-------------------
Every water that enters GLU/HBC (regions 2 + 3) comes from the SF, from below
the HBC or through the sides, so per run

    net from below = hbc_back - hbc_out
    net from SF    = sf_in    - sf_back
    net from sides = side_in  - side_out

and the three add up exactly to the change in cavity water count between the
first and the last frame. A positive source fed the cavity, a negative one
drained it.

How to run
----------
    python aggregate_cavity_water_flux.py \\
        --base_dir /path/to/girk_analyser_results/G12 --channel_type G12

    # only some runs
    python aggregate_cavity_water_flux.py --base_dir .../G12 --channel_type G12 \\
        --runs 1-5,9
"""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.cavity_water_flux import (
    BOUNDARIES, CROSSINGS, RES_REGIONS, RES_COLORS, ROUTES, ROUTE_COLORS,
    SFTransitTracker, describe,
    _plot_residence_hist, _plot_residence_box,
)


PKL_REL = Path("water_analysis") / "cavity_water_flux" / "cavity_water_flux.pkl"

# route -> (crossing into the cavity, crossing out of it)
ENTRY_ROUTES = {"below": ("hbc_back", "hbc_out"), "sides": ("side_in", "side_out"),
                "SF": ("sf_in", "sf_back")}

# cavity budget source -> (crossing into the cavity, crossing out of it)
CAVITY_SOURCES = {
    "from below (HBC)": ("hbc_back", "hbc_out"),
    "from above (SF)":  ("sf_in", "sf_back"),
    "through the sides": ("side_in", "side_out"),
}
SOURCE_COLORS = {"from below (HBC)": "#43A047", "from above (SF)": "#1E88E5",
                 "through the sides": "#9E9E9E"}


def cavity_budget(row):
    """Net waters into GLU/HBC per source; their sum = cavity count change."""
    return {src: row[i] - row[o] for src, (i, o) in CAVITY_SOURCES.items()}


def _run_number(path):
    m = re.search(r"RUN(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _parse_runs(spec):
    """'1-5,9' or '3 5 14' -> {1,2,3,4,5,9}"""
    out = set()
    for tok in re.split(r"[,\s]+", spec.strip()):
        tok = re.sub(r"(?i)^run", "", tok)
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def load_runs(base_dir, runs=None):
    run_dirs = sorted((p for p in Path(base_dir).glob("RUN*") if p.is_dir()),
                      key=_run_number)
    if runs:
        run_dirs = [p for p in run_dirs if _run_number(p) in runs]

    loaded = {}
    for rd in run_dirs:
        pkl = rd / PKL_REL
        if not pkl.exists():
            print(f"  [SKIP] {rd.name}: {PKL_REL} not found")
            continue
        with open(pkl, "rb") as f:
            loaded[rd.name] = pickle.load(f)
        print(f"  [OK]   {rd.name}")
    return loaded


def run_row(name, d):
    fd = d["frame_duration"]
    sim_ns = (len(d["frame_indices"]) - 1) * fd
    row = {"run": name, "sim_ns": round(sim_ns, 3), "frame_duration": fd,
           "n_permeations": d["n_permeation_events"]}
    for k in CROSSINGS:
        row[k] = d["totals"][k]
        row[f"{k}_per_ns"] = d["totals"][k] / sim_ns if sim_ns > 0 else np.nan
    for b, (f, k) in BOUNDARIES.items():
        row[f"net_{_slug(b)}_per_ns"] = row[f"{f}_per_ns"] - row[f"{k}_per_ns"]
    for r in RES_REGIONS:
        st = d["residence_stats"].get(r)
        if st is None:  # run analysed before this region existed
            continue
        row[f"res_{_slug(r)}_n"] = st["n"]
        row[f"res_{_slug(r)}_censored"] = st["n_censored"]
        row[f"res_{_slug(r)}_mean_ns"] = st["mean"]
        row[f"res_{_slug(r)}_median_ns"] = st["median"]
    ent = sum(d["totals"][i] for i, _ in ENTRY_ROUTES.values())
    for r, (i, o) in ENTRY_ROUTES.items():
        row[f"entry_share_{r}"] = d["totals"][i] / ent if ent else np.nan
    if "cavity_by_route" in d:
        br = d["cavity_by_route"]
        tot = sum(br[r] for r in ROUTES).mean()
        for r in ROUTES:
            row[f"cavity_share_{r}"] = br[r].mean() / tot if tot else np.nan
    tr = d.get("sf_transits")
    if tr is not None and "net_jumps" in tr:
        for k in SFTransitTracker.COUNTS:
            row[f"sf_{k}"] = tr["totals"][k]
        row["sf_transit_net"] = tr["net"]
        row["sf_transit_net_per_ns"] = tr["net"] / sim_ns if sim_ns > 0 else np.nan
        row["sf_transit_net_per_ion"] = (tr["net"] / row["n_permeations"]
                                         if row["n_permeations"] else np.nan)
        row["sf_net_incl_jumps"] = tr["net"] + tr["net_jumps"]
    return row


def spearman(x, y):
    """Spearman rho (average ranks for ties); no p-value, to avoid scipy."""
    def rank(a):
        a = np.asarray(a, dtype=float)
        order = a.argsort()
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def complete_times(d, region):
    v = d["residence_visits"].get(region)
    if v is None:
        return np.zeros(0)
    return v["n_frames"][~v["censored"]] * d["frame_duration"]


def run_aggregate_flux(base_dir, output_dir=None, channel_type="G12", runs=None):
    base_dir = Path(base_dir)
    out_dir = (Path(output_dir) if output_dir
               else base_dir / "water_analysis" / "cavity_water_flux_across_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = channel_type

    print("\n" + "=" * 70)
    print(f"AGGREGATE CAVITY WATER FLUX & RESIDENCE — {ch}")
    print("=" * 70)

    data = load_runs(base_dir, runs)
    if not data:
        print("  [ERROR] no runs loaded")
        return None

    fds = sorted({d["frame_duration"] for d in data.values()})
    if len(fds) > 1:
        print(f"  [WARN] runs have different frame spacing {fds} ns — "
              f"residence times and rates are not comparable across them")
    fd = fds[0]

    rows = [run_row(n, d) for n, d in data.items()]
    names = [r["run"] for r in rows]

    # ── per-run csv ───────────────────────────────────────────────────────────
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    csv_path = out_dir / f"cavity_water_flux_per_run_{ch}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ── across-run flux ───────────────────────────────────────────────────────
    flux = {}
    for k in [f"{c}_per_ns" for c in CROSSINGS] + \
             [f"net_{_slug(b)}_per_ns" for b in BOUNDARIES]:
        vals = np.array([r[k] for r in rows], dtype=float)
        flux[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                   "median": float(np.nanmedian(vals)), "values": vals}

    # ── pooled residence ──────────────────────────────────────────────────────
    pooled = {r: np.concatenate([complete_times(d, r) for d in data.values()])
              for r in RES_REGIONS}
    pooled_stats = {r: describe(t) for r, t in pooled.items()}
    for r in RES_REGIONS:
        pooled_stats[r]["n_censored"] = int(sum(
            d["residence_visits"][r]["censored"].sum()
            for d in data.values() if r in d["residence_visits"]))
        run_means = np.array([describe(complete_times(d, r))["mean"] for d in data.values()])
        run_medians = np.array([describe(complete_times(d, r))["median"] for d in data.values()])
        pooled_stats[r]["mean_of_run_means"] = float(np.nanmean(run_means))
        pooled_stats[r]["std_of_run_means"] = float(np.nanstd(run_means))
        pooled_stats[r]["mean_of_run_medians"] = float(np.nanmean(run_medians))

    # ── summary txt ───────────────────────────────────────────────────────────
    lines = [
        f"CAVITY WATER FLUX & RESIDENCE ACROSS RUNS — {ch}",
        f"Runs ({len(rows)}): {', '.join(names)}",
        f"Frame spacing: {fds} ns",
        f"Total simulated: {sum(r['sim_ns'] for r in rows):.1f} ns, "
        f"K+ permeations: {sum(r['n_permeations'] for r in rows)}",
        "",
        "FLUX (per ns, statistics over runs)",
        "=" * 60,
        f"{'':<22} {'mean':>9} {'std':>9} {'median':>9}",
        "-" * 60,
    ]
    for k, st in flux.items():
        lines.append(f"{k:<22} {st['mean']:>9.3f} {st['std']:>9.3f} {st['median']:>9.3f}")
    lines += [
        "",
        "RESIDENCE (ns, all complete visits of all runs pooled)",
        "=" * 96,
        f"{'region':<9} {'n':>8} {'cens.':>7} {'mean':>8} {'median':>8} {'std':>8} "
        f"{'q1':>7} {'q3':>7} {'max':>8} {'run-mean ± std':>17}",
        "-" * 96,
    ]
    for r, st in pooled_stats.items():
        lines.append(
            f"{r:<9} {st['n']:>8} {st['n_censored']:>7} {st['mean']:>8.3f} "
            f"{st['median']:>8.3f} {st['std']:>8.3f} {st['q1']:>7.3f} {st['q3']:>7.3f} "
            f"{st['max']:>8.3f} {st['mean_of_run_means']:>9.3f} ± {st['std_of_run_means']:<6.3f}")
    budgets = [cavity_budget(r) for r in rows]
    gain = sum(sum(b.values()) for b in budgets)
    lines += [
        "",
        "CAVITY WATER BUDGET (net waters into GLU/HBC, all runs summed)",
        "=" * 60,
    ]
    for src in CAVITY_SOURCES:
        tot = sum(b[src] for b in budgets)
        share = f"{100 * tot / gain:>6.1f} %" if gain else ""
        lines.append(f"{src:<20} {tot:>8} {share}")
    lines += [
        "-" * 60,
        f"{'total cavity gain':<20} {gain:>8}   ({gain / len(rows):.1f} per run, "
        f"last frame - first frame)",
    ]
    tr_rows = [r for r in rows if "sf_transit_net" in r]
    if tr_rows:
        n_ion = sum(r["n_permeations"] for r in tr_rows)
        lines += [
            "",
            f"FULL SF TRANSITS (enter one end, leave by the other; {len(tr_rows)} runs)",
            "=" * 60,
        ]
        for k in SFTransitTracker.COUNTS:
            lines.append(f"{k:<16} {sum(r['sf_' + k] for r in tr_rows):>8}")
        net = sum(r["sf_transit_net"] for r in tr_rows)
        lines += [
            "-" * 60,
            f"net transits (down - up) : {net}"
            + (f"   ({net / n_ion:.2f} per K+ permeation, {n_ion} K+)" if n_ion else ""),
            f"net incl. jumps          : {sum(r['sf_net_incl_jumps'] for r in tr_rows)}",
            f"Spearman rho, net transits vs K+ per run : "
            f"{spearman([r['n_permeations'] for r in tr_rows], [r['sf_transit_net'] for r in tr_rows]):.2f}",
        ]
    lines += [
        "",
        "WHERE CAVITY (GLU/HBC) WATER COMES FROM (all runs)",
        "=" * 66,
        f"{'route':<8} {'in /ns':>8} {'share':>7} {'out /ns':>8} {'share':>7}"
        f" {'share of cavity':>16}",
        "-" * 66,
    ]
    rate = {r: np.mean([row[f"{i}_per_ns"] for row in rows]) for r, (i, _) in ENTRY_ROUTES.items()}
    rate_out = {r: np.mean([row[f"{o}_per_ns"] for row in rows]) for r, (_, o) in ENTRY_ROUTES.items()}
    comp_rows = [r for r in rows if "cavity_share_below" in r]
    for r in ENTRY_ROUTES:
        comp = (f"{100 * np.mean([x[f'cavity_share_{r}'] for x in comp_rows]):>15.1f}%"
                if comp_rows else f"{'n/a':>16}")
        lines.append(f"{r:<8} {rate[r]:>8.2f} {100 * rate[r] / sum(rate.values()):>6.1f}% "
                     f"{rate_out[r]:>8.2f} {100 * rate_out[r] / sum(rate_out.values()):>6.1f}%"
                     f" {comp}")
    if comp_rows:
        lines.append(f"{'initial':<8} {'':>8} {'':>7} {'':>8} {'':>7} "
                     f"{100 * np.mean([x['cavity_share_initial'] for x in comp_rows]):>15.1f}%")
    lines.append("(share of cavity = waters in the cavity by the way they came in, "
                 "mean over frames and runs)")
    print("\n".join("  " + s for s in lines))
    (out_dir / f"cavity_water_flux_across_runs_{ch}.txt").write_text("\n".join(lines) + "\n")

    result = {"channel_type": ch, "runs": names, "per_run": rows, "flux": flux,
              "residence_pooled": pooled, "residence_stats": pooled_stats,
              "frame_durations": fds}
    with open(out_dir / f"cavity_water_flux_across_runs_{ch}.pkl", "wb") as f:
        pickle.dump(result, f)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_rates_across(rows, out_dir / f"flux_rates_{ch}.png",
                       f"{ch}: water crossings per boundary ({len(rows)} runs)")
    _plot_net_per_run(rows, out_dir / f"flux_net_per_run_{ch}.png",
                      f"{ch}: net water flux per run (downward +)")
    _plot_residence_hist(pooled, pooled_stats, fd, out_dir / f"residence_hist_{ch}.png",
                         title=f"{ch}: water residence, all runs pooled")
    _plot_residence_box(pooled, pooled_stats, out_dir / f"residence_box_{ch}.png",
                        title=f"{ch}: water residence, all runs pooled (◇ = mean)")
    _plot_box_per_run(data, out_dir / f"residence_box_per_run_{ch}.png",
                      f"{ch}: water residence per run (◇ = mean)")
    _plot_cavity_budget(rows, out_dir / f"cavity_water_budget_{ch}.png",
                        f"{ch}: where the cavity's net water gain came from")
    _plot_cavity_overview(rows, data, fd, out_dir / f"cavity_water_overview_{ch}.png",
                          f"{ch}: water in the cavity ({len(rows)} runs)")
    if tr_rows:
        _plot_sf_transits(tr_rows, out_dir / f"sf_transits_{ch}.png",
                          f"{ch}: water crossing the whole SF")
    _plot_origin(rows, out_dir / f"cavity_water_origin_{ch}.png",
                 f"{ch}: which way water enters and leaves the cavity ({len(rows)} runs)")
    comp_data = {n: d for n, d in data.items() if "cavity_by_route" in d}
    if comp_data:
        _plot_composition_across(comp_data, fd, out_dir / f"cavity_water_composition_{ch}.png",
                                 f"{ch}: where the waters in the cavity came in from "
                                 f"({len(comp_data)} runs)")

    print(f"  ✅ All outputs in: {out_dir}")
    return result


# ── plotting helpers ───────────────────────────────────────────────────────────

def _plot_rates_across(rows, out_path, title):
    names = list(BOUNDARIES)
    x = np.arange(len(names))
    wdt = 0.27
    series = (
        ("forward (down)", "#1E88E5", [f"{f}_per_ns" for f, _ in BOUNDARIES.values()]),
        ("backward (up)", "#90CAF9", [f"{k}_per_ns" for _, k in BOUNDARIES.values()]),
        ("net", "#212121", [f"net_{_slug(b)}_per_ns" for b in BOUNDARIES]),
    )
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (label, color, keys), off in zip(series, (-wdt, 0, wdt)):
        vals = np.array([[r[k] for r in rows] for k in keys], dtype=float)
        mean, std = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
        bars = ax.bar(x + off, mean, wdt, yerr=std, capsize=3, color=color,
                      alpha=0.8, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        for i in range(len(keys)):
            jit = rng.uniform(-wdt / 4, wdt / 4, vals.shape[1])
            ax.scatter(x[i] + off + jit, vals[i], s=12, color="white",
                       edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, fontsize=11)
    ax.set_ylabel("waters per ns", fontsize=11)
    ax.set_title(f"{title}\nbar = mean ± std over runs, dots = runs",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_net_per_run(rows, out_path, title):
    x = np.arange(len(rows))
    wdt = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows) + 3), 5))
    for (b, color), off in zip(zip(BOUNDARIES, ("#1E88E5", "#FB8C00", "#43A047")),
                               (-wdt, 0, wdt)):
        ax.bar(x + off, [r[f"net_{_slug(b)}_per_ns"] for r in rows], wdt,
               color=color, label=b)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{r['run']}\n({r['n_permeations']} K⁺)" for r in rows],
                  fontsize=9)
    ax.set_ylabel("net waters per ns", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_origin(rows, out_path, title):
    """
    Left: waters entering and leaving the cavity per ns by route, mean over
    runs (dots = runs), labelled with the share. Right: per run, the share of
    entries by route as a 100% bar.
    """
    routes = list(ENTRY_ROUTES)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, max(5.5, 0.38 * len(rows) + 2)),
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    y = np.arange(len(routes))[::-1]
    h = 0.38
    rng = np.random.default_rng(0)
    xmax = 0.0
    for (word, idx, off, alpha) in (("in", 0, h / 2, 0.9), ("out", 1, -h / 2, 0.4)):
        vals = np.array([[row[f"{ENTRY_ROUTES[r][idx]}_per_ns"] for row in rows] for r in routes])
        mean = vals.mean(axis=1)
        total = mean.sum()
        ax.barh(y + off, mean, h, color=[ROUTE_COLORS[r] for r in routes],
                alpha=alpha, edgecolor="black", linewidth=0.5)
        for i in range(len(routes)):
            ax.scatter(vals[i], y[i] + off + rng.uniform(-h / 4, h / 4, vals.shape[1]),
                       s=9, color="white", edgecolor="black", linewidth=0.5, zorder=3)
            # label past the furthest run so the dots never cover it
            ax.text(vals[i].max() * 1.03 + 0.3, y[i] + off,
                    f"{word} {mean[i]:.2f} /ns ({100 * mean[i] / total:.1f}%)",
                    va="center", fontsize=9.5)
        xmax = max(xmax, vals.max())
    ax.set_yticks(y, routes, fontsize=11)
    ax.set_xlim(0, xmax * 1.55)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#757575", alpha=0.9, label="entering the cavity"),
                       Patch(facecolor="#757575", alpha=0.4, label="leaving the cavity")],
              fontsize=9, loc="lower right")
    ax.set_xlabel("waters per ns (bar = mean over runs, dots = runs)", fontsize=10)
    ax.set_title("Entering and leaving the cavity, by route", fontsize=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    yr = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for r in routes:
        share = np.array([100 * row[f"entry_share_{r}"] for row in rows])
        ax2.barh(yr, share, 0.7, left=left, color=ROUTE_COLORS[r], label=f"from {r}")
        left += share
    sf = [100 * row["entry_share_SF"] for row in rows]
    for yi, v in zip(yr, sf):
        ax2.text(101, yi, f"SF {v:.1f}%", va="center", fontsize=8, color="#1565C0")
    ax2.set_yticks(yr, [f"{row['run']} ({row['n_permeations']} K⁺)" for row in rows], fontsize=9)
    ax2.set_xlim(0, 112)
    ax2.set_xlabel("share of the waters entering the cavity (%)", fontsize=10)
    ax2.set_title("Per run: where the entering waters came from", fontsize=12)
    ax2.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.45, 1.06), ncol=3, frameon=False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_composition_across(data, frame_duration, out_path, title):
    """Share of the cavity's waters by entry route over time, mean over runs."""
    runs = list(data.values())
    n = min(len(d["frame_indices"]) for d in runs)
    t = np.arange(n) * frame_duration
    order = ("initial", "below", "sides", "SF")
    shares = {}
    for r in order:
        per_run = []
        for d in runs:
            br = d["cavity_by_route"]
            tot = sum(br[x][:n] for x in order).astype(float)
            per_run.append(np.divide(100 * br[r][:n], tot, out=np.zeros(n), where=tot > 0))
        shares[r] = np.mean(per_run, axis=0)
    late = slice(n // 2, None)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.stackplot(t, *(shares[r] for r in order), colors=[ROUTE_COLORS[r] for r in order],
                 labels=[f"{'already inside at frame 0' if r == 'initial' else 'came in from ' + r}"
                         f" — {shares[r][late].mean():.1f}% in the second half" for r in order],
                 alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time since the first frame (ns)", fontsize=11)
    ax.set_ylabel("share of the waters in the cavity (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="center right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_sf_transits(rows, out_path, title):
    """Left: transits down / up per run, dot = net. Right: net vs K+ per run."""
    x = np.arange(len(rows))
    down = np.array([r["sf_transit_down"] for r in rows])
    up = np.array([r["sf_transit_up"] for r in rows])
    jumps = np.array([r["sf_jump_down"] - r["sf_jump_up"] for r in rows])
    k = np.array([r["n_permeations"] for r in rows])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(max(11, 0.6 * len(rows) + 7), 5),
        gridspec_kw={"width_ratios": [max(2.5, len(rows) / 4), 1.3]})
    w = 0.38
    ax.bar(x - w / 2, down, w, color="#1E88E5", label="transits down (top → bottom)")
    ax.bar(x + w / 2, up, w, color="#90CAF9", label="transits up (bottom → top)")
    ax.scatter(x, down - up, color="black", zorder=3, s=25, label="net (down − up)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{r['run']}\n({r['n_permeations']} K⁺)" for r in rows], fontsize=9)
    ax.set_ylabel("waters per run", fontsize=11)
    ax.set_title("per run", fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

    net = down - up
    ax2.scatter(k, net, s=45, color="#1E88E5", edgecolor="black", linewidth=0.6, zorder=3)
    if np.any(jumps):
        ax2.scatter(k, net + jumps, s=30, marker="x", color="#757575", zorder=3,
                    label="net incl. unseen jumps")
        ax2.legend(fontsize=8, loc="upper left")
    for ki, ni, r in zip(k, net, rows):
        ax2.annotate(r["run"].replace("RUN", "R"), (ki, ni), xytext=(4, 3),
                     textcoords="offset points", fontsize=8, color="#555555")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("K⁺ permeations in the run", fontsize=11)
    ax2.set_ylabel("net water transits", fontsize=11)
    ax2.set_title(f"Spearman ρ = {spearman(k, net):.2f}", fontsize=11)
    ax2.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax2.margins(0.15)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_cavity_overview(rows, data, frame_duration, out_path, title):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1.6, 1])
    _draw_pore(fig.add_subplot(gs[:, 0]), rows)
    _draw_gain_over_time(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
                         data, frame_duration)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _draw_pore(ax, rows):
    """
    Pore diagram: SF, cavity (split by the ASN ring), below the HBC. Each
    boundary shows waters in / out / net, as the mean per run. Arrow width
    follows |net|, arrow direction is the direction of the net movement.
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    net_sf = mean("sf_in") - mean("sf_back")             # + = into the cavity
    net_hbc = mean("hbc_back") - mean("hbc_out")         # + = into the cavity
    net_side = mean("side_in") - mean("side_out")        # + = into the cavity
    net_asn = mean("asn_down") - mean("asn_up")          # + = downward
    gain = net_sf + net_hbc + net_side
    nmax = max(abs(net_sf), abs(net_hbc), abs(net_side), abs(net_asn), 1e-9)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.add_patch(Rectangle((4.3, 7.4), 1.4, 1.6, color=RES_COLORS["SF"], alpha=0.35))
    ax.add_patch(Rectangle((2.2, 3.4), 5.6, 3.6, color="#90CAF9", alpha=0.35))
    ax.add_patch(Rectangle((2.2, 0.4), 5.6, 2.6, color="#A5D6A7", alpha=0.35))
    ax.plot([2.2, 7.8], [5.2, 5.2], color="#FB8C00", linestyle="--", linewidth=1.5)
    ax.plot([2.2, 7.8], [3.2, 3.2], color="#2E7D32", linewidth=3)
    ax.text(5, 8.2, "SF", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5, 9.9, "above the SF", ha="center", fontsize=11, color="#555555")
    ax.text(2.35, 6.65, "GLU/ASN", fontsize=10, color="#1565C0")
    ax.text(2.35, 3.6, "ASN/HBC", fontsize=10, color="#1565C0")
    ax.text(7.75, 5.32, "ASN ring", fontsize=9, ha="right", color="#E65100")
    ax.text(2.35, 2.75, "HBC gate", fontsize=9, ha="left", va="top", color="#2E7D32")
    ax.text(5, 1.4, "below the HBC", ha="center", fontsize=11, color="#555555")
    ax.text(5, 6.2, f"cavity gains\n{gain:+.0f} waters per run",
            ha="center", va="center", fontsize=13, fontweight="bold")

    def arrow(x0, y0, x1, y1, net, color):
        width = 1.5 + 10 * abs(net) / nmax
        if net < 0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12 + width, linewidth=width,
                                     color=color, alpha=0.85))

    def label(x, y, inn, out, net, ha, what_in="in", what_out="out"):
        ax.text(x, y, f"{what_in} {inn:,.0f}\n{what_out} {out:,.0f}\nnet {net:+.1f}",
                ha=ha, va="center", fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#BDBDBD"})

    arrow(5, 7.45, 5, 6.55, net_sf, "#1E88E5")             # down = into the cavity
    label(5.9, 7.0, mean("sf_in"), mean("sf_back"), net_sf, "left")
    arrow(5, 2.55, 5, 3.85, net_hbc, "#43A047")            # up = into the cavity
    label(5.9, 2.2, mean("hbc_back"), mean("hbc_out"), net_hbc, "left")
    arrow(0.7, 5.2, 2.15, 5.2, net_side, "#757575")        # right = into the cavity
    label(0.1, 6.6, mean("side_in"), mean("side_out"), net_side, "left")
    ax.text(1.4, 4.75, "sides", ha="center", fontsize=9, color="#757575")
    arrow(8.4, 5.9, 8.4, 4.5, net_asn, "#FB8C00")           # down = downward
    label(8.65, 5.2, mean("asn_down"), mean("asn_up"), net_asn, "left",
          what_in="down", what_out="up")

    ax.text(5, 0.05, "in / out = waters entering / leaving the cavity across that "
            "boundary, mean per run\narrow = direction of the net movement, width = its size",
            ha="center", va="bottom", fontsize=9, color="#555555")


def _draw_gain_over_time(ax, ax_zoom, data, frame_duration):
    """
    Net waters gained by the cavity since the first frame, per source, mean
    over runs (band = ±1 std). Black = the cavity count itself; the three
    sources add up to it. Bottom: the first part of the runs, enlarged.
    """
    runs = list(data.values())
    n = min(len(d["frame_indices"]) for d in runs)
    t = np.arange(n) * frame_duration

    curves = {}
    for src, (i, o) in CAVITY_SOURCES.items():
        curves[src] = np.array([np.cumsum(d["counts"][i][:n] - d["counts"][o][:n])
                                for d in runs])
    cav = np.array([d["n_in_region"]["GLU/HBC"][:n] - d["n_in_region"]["GLU/HBC"][0]
                    for d in runs])

    zoom = max(2, min(n, int(round(min(25.0, 0.1 * t[-1]) / frame_duration))))
    for a, sl in ((ax, slice(None)), (ax_zoom, slice(0, zoom))):
        for src, c in curves.items():
            m, sd = c.mean(axis=0)[sl], c.std(axis=0)[sl]
            a.plot(t[sl], m, color=SOURCE_COLORS[src], linewidth=1.8, label=src)
            a.fill_between(t[sl], m - sd, m + sd, color=SOURCE_COLORS[src], alpha=0.15)
        a.plot(t[sl], cav.mean(axis=0)[sl], color="black", linewidth=2, linestyle="--",
               label="cavity water count (change since first frame) = sum of the three")
        a.axhline(0, color="black", linewidth=0.8)
        a.grid(linestyle=":", linewidth=0.5, alpha=0.6)
        a.set_ylabel("net waters gained", fontsize=11)
    ax.set_title("When does the cavity gain its water? (mean over runs, band = ±1 std)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax_zoom.set_title(f"first {t[zoom - 1]:.0f} ns, enlarged", fontsize=11)
    ax_zoom.set_xlabel("time since the first frame (ns)", fontsize=11)


def _plot_cavity_budget(rows, out_path, title):
    """
    Left: per run, net waters into the cavity from each source, stacked
    (positive sources above zero, negative below); black dot = their sum,
    the cavity's gain over the run. Right: the same summed over all runs.
    """
    budgets = [cavity_budget(r) for r in rows]
    sources = list(CAVITY_SOURCES)
    x = np.arange(len(rows))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(max(10, 0.6 * len(rows) + 6), 5.5),
        gridspec_kw={"width_ratios": [max(3, len(rows) / 3), 1.4]})

    pos = np.zeros(len(rows))
    neg = np.zeros(len(rows))
    for src in sources:
        v = np.array([b[src] for b in budgets], dtype=float)
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, 0.7, bottom=base, color=SOURCE_COLORS[src], label=src)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    gains = [sum(b.values()) for b in budgets]
    ax.scatter(x, gains, color="black", zorder=3, s=25,
               label="total cavity gain (last − first frame)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{r['run']}\n({r['n_permeations']} K⁺)" for r in rows], fontsize=9)
    ax.set_ylabel("net waters into the cavity", fontsize=11)
    ax.set_title("per run", fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

    totals = [sum(b[src] for b in budgets) for src in sources]
    gain = sum(totals)
    bars = ax2.bar(range(len(sources)), totals,
                   color=[SOURCE_COLORS[s] for s in sources])
    ax2.bar_label(bars, labels=[f"{t}\n({100 * t / gain:.0f}%)" if gain else str(t)
                                for t in totals], fontsize=9, padding=2)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(len(sources)), [s.replace(" (", "\n(") for s in sources],
                   fontsize=9)
    ax2.set_title(f"all runs: total gain {gain}", fontsize=11)
    ax2.margins(y=0.15)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_box_per_run(data, out_path, title):
    regions = list(RES_REGIONS)
    names = list(data)
    fig, axes = plt.subplots(len(regions), 1, sharex=True,
                             figsize=(max(8, 0.6 * len(names) + 3), 3.2 * len(regions)))
    for ax, r in zip(np.atleast_1d(axes), regions):
        times = [complete_times(d, r) for d in data.values()]
        keep = [i for i, t in enumerate(times) if t.size]
        if keep:
            bp = ax.boxplot([times[i] for i in keep], positions=[i + 1 for i in keep],
                            showmeans=True, patch_artist=True,
                            flierprops={"markersize": 2, "alpha": 0.3},
                            meanprops={"marker": "D", "markerfacecolor": "white",
                                       "markeredgecolor": "black"})
            for patch in bp["boxes"]:
                patch.set_facecolor(RES_COLORS[r])
                patch.set_alpha(0.7)
            ax.set_yscale("log")
        ax.set_ylabel(f"{r}\nresidence (ns)", fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axes[-1].set_xticks(range(1, len(names) + 1), names, fontsize=9, rotation=45)
    axes[0].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_dir", required=True,
                        help="channel folder holding RUN* subfolders")
    parser.add_argument("--output_dir", default=None,
                        help="default: <base_dir>/water_analysis/cavity_water_flux_across_runs")
    parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--runs", default=None,
                        help="subset of runs, e.g. '1-5,9' (default: all RUN* folders)")
    args = parser.parse_args()

    run_aggregate_flux(args.base_dir, args.output_dir, args.channel_type,
                       _parse_runs(args.runs) if args.runs else None)


'''
python3 aggregate_cavity_water_flux.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
python3 aggregate_cavity_water_flux.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --channel_type G2
'''
