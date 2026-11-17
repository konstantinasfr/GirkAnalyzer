#!/usr/bin/env python3
"""
aggregate_cavity_esp.py
=======================
Pools the classical cylinder ESP (analysis/cavity_esp_classical.py, RUN*/cavity_esp/) over the runs of one
or more systems, after the equilibration cut, and compares the systems.

USAGE
  python3 aggregate_cavity_esp.py \\
      --system G12=/path/girk_analyser_results/G12 \\
      --system G2=/path/girk_analyser_results/G2 \\
      --system D173N=/path/girk_analyser_results/G12_D173N,/path/girk_analyser_results/G12_D173N_4fs \\
      --system E141Q=/path/girk_analyser_results/G12_E141Q,/path/girk_analyser_results/G12_E141Q_4fs \\
      [--out cavity_esp_summary] [--no_eq] [--boot 5000]
The first --system is the reference for the differences. The equilibration start of each run is read from
<system>/aggregated_site_chain/<settings>/equilibration.csv (used_start_frame) when it exists; frames of
interest (e.g. the QM frames) are always reported, whether before or after the cut.

OUTPUTS (in --out)
  frames_of_interest.csv   the exact values at the frames of interest (each cylinder + four-subunit mean)
  per_run.csv              per run and cylinder: frames, mean, median, SD over frames (all frames and
                           frames with all four upper-ring side chains pointing in)
  summary.csv              per system and cylinder: runs, mean of the run means, median, SD between runs,
                           SEM, bootstrap 95% CI over runs; pooled over frames: mean, median, SD; the mean
                           spatial SD inside the cylinder; the same for inward frames
  differences.csv          four-subunit mean of each system minus the reference: estimate, bootstrap 95% CI
                           over runs (all frames and inward frames)
  contributions.csv        per system and cylinder: the mean contribution of each ring residue and the caps
  profiles.csv             axial profile (distance from the Glu CA): mean over runs, SD between runs
  cavity_esp.txt, per_run.png (each run), cylinders.png (each cylinder), profiles.png (axial profiles),
  frames_hist.png (distribution over frames), timeseries.png (each run over time),
  hist_<system>.png  histogram of the ESP over all cylinder points of all frames after equilibration, 2 x 2 like
                     esp_cylinders.py (5 kcal/mol bins, -300..-50)
  axial_<system>.png like esp_maps.py: median and IQR of all points in each 0.5 A slice, one line per cylinder
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.cavity_esp_classical import big_fonts
big_fonts()

FOUR = "four-subunit mean"
HIST = {}
GRID = "mesh"
MIN_T = 0.0          # --min_time_ns: also drop every frame before this time
COLORS = ["#1F4E79", "#D9784F", "#13896B", "#C0392B", "#6B6B6B", "#3B7DD8"]


def load_one(base, use_eq=True):
    base = Path(base)
    fr = [pd.read_csv(f) for f in sorted(base.glob("RUN*/cavity_esp/esp_frames_run*.csv"))]
    pr = [pd.read_csv(f) for f in sorted(base.glob("RUN*/cavity_esp/esp_profile_run*.csv.gz"))] if GRID == "mesh" else []
    if not fr:
        raise SystemExit(f"no RUN*/cavity_esp/esp_frames_run*.csv under {base}")
    frames = pd.concat(fr, ignore_index=True)
    prof = pd.concat(pr, ignore_index=True) if pr else pd.DataFrame()
    eq = {}
    if use_eq:
        for f in sorted(base.glob("aggregated_site_chain/*/equilibration.csv")):
            t = pd.read_csv(f)
            eq = dict(zip(t["run"].astype(int), t["used_start_frame"].fillna(0).astype(int)))
            print(f"  {base.name}: equilibration starts from {f}")
            break
        # runs missing from the pooled table: the per-run table written by run_site_chain (same RMSD rule)
        for run in sorted(frames["run"].unique()):
            if int(run) in eq:
                continue
            for f in sorted(base.glob(f"RUN{int(run)}/site_chain/*/equilibration.csv")):
                t = pd.read_csv(f)
                t = t[t["run"].astype(int) == int(run)]
                if len(t):
                    eq[int(run)] = int(t["used_start_frame"].fillna(0).iloc[0])
                    print(f"  {base.name}: RUN{int(run)} not in the pooled equilibration.csv; start frame {eq[int(run)]} "
                          f"from {f}")
                    break
        missing = [int(r) for r in sorted(frames["run"].unique()) if int(r) not in eq]
        if missing:
            print(f"  {base.name}: WARNING no equilibration start for RUN {missing}: all their frames are used")
    if MIN_T > 0:                                  # the later of the equilibration start and --min_time_ns
        dt = (frames["time_ns"] / frames["frame"].where(frames["frame"] > 0)).groupby(frames["run"]).median()
        for run, d in dt.items():
            if d == d and d > 0:
                eq[int(run)] = max(eq.get(int(run), 0), int(np.ceil(MIN_T / d)))
    frames["after_eq"] = frames["frame"] >= frames["run"].map(eq).fillna(0)
    # pooled histograms of the ESP over the cylinder points (after equilibration) and of the frames of interest
    hist, hist_foi = {}, {}
    if GRID == "mesh":
        foi_set = set(zip(frames.loc[frames["frame_of_interest"], "run"], frames.loc[frames["frame_of_interest"], "frame"]))
        for f in sorted(base.glob("RUN*/cavity_esp/esp_hist_run*.npz")):
            run = int(f.stem.replace("esp_hist_run", ""))
            z = np.load(f, allow_pickle=False)
            ok = z["frame"] >= eq.get(run, 0)
            isf = np.array([(run, int(fr)) in foi_set for fr in z["frame"]])
            for c in np.unique(z["cylinder"]):
                sel = z["cylinder"] == c
                hist[c] = hist.get(c, 0) + z["counts"][sel & ok].sum(axis=0).astype(np.int64)
                if isf.any():
                    hist_foi[c] = hist_foi.get(c, 0) + z["counts"][sel & isf].sum(axis=0).astype(np.int64)
            HIST["edges"] = z["edges"]
    # axial 2D histograms (blocks of frames after equilibration)
    axial, lengths = {}, {}
    if GRID == "mesh":
        for f in sorted(base.glob("RUN*/cavity_esp/esp_axial_run*.npz")):
            run = int(f.stem.replace("esp_axial_run", ""))
            z = np.load(f, allow_pickle=False)
            ok = z["block"] * int(z["block_size"]) >= eq.get(run, 0)
            for i, c in enumerate(z["cylinder"]):
                axial[c] = axial.get(c, 0) + z["counts"][ok, i].sum(axis=0).astype(np.int64)
                lengths.setdefault(c, []).append(float(z["length"][i]))
    HIST[str(base)] = (hist, hist_foi, axial, lengths)
    if GRID == "jaguar":
        if "jag_mean" not in frames:
            raise SystemExit(f"{base}: no Jaguar-like columns (run the newer cavity_esp_classical.py)")
        frames["mean"], frames["median"], frames["sd"], frames["n_points"] = (
            frames["jag_mean"], frames["jag_median"], frames["jag_sd"], frames["jag_n"])
        frames = frames[[c for c in frames.columns if not c.startswith("contrib_")]]
    if len(prof):
        prof = prof[prof["frame"] >= prof["run"].map(eq).fillna(0)]
    return frames, prof


def load_system(paths, use_eq=True):
    """One or more folders (comma-separated) pooled as one system; with several folders the runs are
    renamed <folder>:<run> so that RUN1 of one folder is not mixed with RUN1 of another."""
    paths = [p for p in paths.split(",") if p]
    F, P = [], []
    for p in paths:
        fr, pr = load_one(p, use_eq)
        if len(paths) > 1:
            tag = Path(p).name
            fr["run"] = tag + ":" + fr["run"].astype(str)
            if len(pr):
                pr["run"] = tag + ":" + pr["run"].astype(str)
        F.append(fr)
        if len(pr):
            P.append(pr)
    return pd.concat(F, ignore_index=True), (pd.concat(P, ignore_index=True) if P else pd.DataFrame()), None


def mark_inward(frames, label, max_r=None, tol=0.5):
    """all_inward per frame: all four upper-ring carboxylates at least as close to the pore axis as in the frames
    of interest (largest r_o there + tol), or closer than max_r if given. Returns (frames, threshold used)."""
    if "max_r_o_upper" not in frames:
        frames["all_inward"] = False
        return frames, np.nan
    thr = max_r
    if thr is None:
        ref = frames[frames["frame_of_interest"]]["max_r_o_upper"]
        thr = ref.max() + tol if len(ref) else np.nan
    frames["all_inward"] = frames["max_r_o_upper"] <= thr if thr == thr else False
    print(f"  {label}: 'all Glu pointing in' = all four carboxylates within {thr:.2f} A of the pore axis"
          if thr == thr else f"  {label}: no frame of interest and no --inward_max_r: inward frames not defined")
    return frames, thr


def add_four(frames):
    """Append, per frame, a 'four-subunit mean' row (mean of the four cylinder means)."""
    keys = ["run", "frame", "time_ns", "frame_of_interest", "all_inward", "after_eq"]
    agg = dict(mean=("mean", "mean"), median=("median", "mean"), sd=("sd", "mean"), n_points=("n_points", "sum"))
    if "jag_mean" in frames:
        agg["jag_mean"] = ("jag_mean", "mean")
    g = frames.groupby(keys, as_index=False).agg(**agg)
    g["cylinder"] = FOUR
    return pd.concat([frames, g], ignore_index=True)


def boot_ci(values, n=5000, seed=1):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n, len(v))).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff(a, b, n=5000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(axis=1) - rng.choice(b, (n, len(b))).mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="LABEL=PATH[,PATH2...] (folders holding RUN*/; several folders = one system, runs kept apart)")
    ap.add_argument("--out", default="cavity_esp_summary")
    ap.add_argument("--no_eq", action="store_true", help="use all frames (no equilibration cut)")
    ap.add_argument("--min_time_ns", type=float, default=0.0,
                    help="also leave out every frame before this time (ns), e.g. 250 = only the last part of each run; "
                         "a check that slow relaxation does not change the result")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--grid", choices=["mesh", "jaguar"], default="mesh",
                    help="mesh: the regular grid (default, the analysis); jaguar: the Jaguar-like grid, computed only for the "
                         "frames of interest (see frames_of_interest.csv)")
    ap.add_argument("--inward_max_r", type=float, default=None,
                    help="'all Glu pointing in' when all four carboxylates are within this distance (A) of the pore "
                         "axis; default: the largest value in that system's frames of interest (the QM frames) + 0.5 A")
    a = ap.parse_args()
    global GRID
    GRID = a.grid
    global MIN_T
    MIN_T = a.min_time_ns
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)

    systems, allf, allp, inward_thr = [], [], [], {}
    for s in a.system:
        lab, path = s.split("=", 1)
        fr, pr, eq = load_system(path, not a.no_eq)
        fr, thr = mark_inward(fr, lab, a.inward_max_r)
        inward_thr[lab] = thr
        fr = add_four(fr)
        fr.insert(0, "system", lab)
        allf.append(fr)
        if len(pr):
            pr.insert(0, "system", lab)
            allp.append(pr)
        systems.append(lab)
    F = pd.concat(allf, ignore_index=True)
    P = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()
    lines = ["CLASSICAL CAVITY ESP (kcal/mol per +1e), force-field charges of the QM cluster atoms, "
             f"{'regular mesh' if GRID == 'mesh' else 'Jaguar-like grid'} in the Glu -> Asn/Asp cylinders",
             "'all Glu pointing in' thresholds (largest carboxylate distance from the pore axis, A): "
             + ", ".join(f"{k} {v:.2f}" for k, v in inward_thr.items())]

    # 1. frames of interest
    foi = F[F["frame_of_interest"]].copy()
    foi[[c for c in ["system", "run", "frame", "cylinder", "mean", "median", "sd", "n_points", "jag_mean", "r_o_upper",
                     "all_inward", "after_eq"] if c in foi]].to_csv(
        od / "frames_of_interest.csv", index=False, float_format="%.2f")
    if len(foi):
        lines.append("\nFRAMES OF INTEREST (exact values on the mesh)")
        for (sy, run, fr_), g in foi.groupby(["system", "run", "frame"], sort=False):
            lines.append(f"  {sy} RUN{run} frame {fr_}  (all Glu pointing in: {bool(g['all_inward'].iloc[0])})")
            for r in g.to_dict("records"):
                lines.append(f"    {r['cylinder']:<20} mean {r['mean']:8.1f}  median {r['median']:8.1f}  "
                             f"SD in cylinder {r['sd']:6.1f}"
                             + (f"   Jaguar-like grid {r['jag_mean']:8.1f}" if r.get("jag_mean") == r.get("jag_mean") and "jag_mean" in r else ""))

    # 2. per run
    use = F[F["after_eq"]]
    rows = []
    for (sy, run, cyl), g in use.groupby(["system", "run", "cylinder"], sort=False):
        gi = g[g["all_inward"]]
        rows.append(dict(system=sy, run=run, cylinder=cyl, n_frames=len(g), mean=g["mean"].mean(),
                         median=g["mean"].median(), sd_frames=g["mean"].std(ddof=1), spatial_sd=g["sd"].mean(),
                         n_frames_inward=len(gi), mean_inward=gi["mean"].mean() if len(gi) else np.nan,
                         frac_inward=len(gi) / len(g)))
    R = pd.DataFrame(rows)
    R.to_csv(od / "per_run.csv", index=False, float_format="%.3f")

    # 3. summary per system and cylinder
    rows = []
    for (sy, cyl), g in R.groupby(["system", "cylinder"], sort=False):
        fr_ = use[(use["system"] == sy) & (use["cylinder"] == cyl)]
        fi = fr_[fr_["all_inward"]]
        lo, hi = boot_ci(g["mean"], a.boot)
        gi = g["mean_inward"].dropna()
        loi, hii = boot_ci(gi, a.boot)
        rows.append(dict(system=sy, cylinder=cyl, n_runs=len(g), n_frames=len(fr_),
                         mean_of_runs=g["mean"].mean(), median_of_runs=g["mean"].median(),
                         sd_between_runs=g["mean"].std(ddof=1), sem=g["mean"].std(ddof=1) / np.sqrt(len(g)),
                         ci95_lo=lo, ci95_hi=hi,
                         pooled_mean=fr_["mean"].mean(), pooled_median=fr_["mean"].median(),
                         pooled_sd_frames=fr_["mean"].std(ddof=1), mean_spatial_sd=fr_["sd"].mean(),
                         frac_frames_inward=len(fi) / len(fr_) if len(fr_) else np.nan,
                         inward_mean_of_runs=gi.mean() if len(gi) else np.nan, inward_ci95_lo=loi,
                         inward_ci95_hi=hii, inward_pooled_median=fi["mean"].median() if len(fi) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(od / "summary.csv", index=False, float_format="%.3f")
    lines.append("\nPER SYSTEM (after equilibration; runs are the unit)")
    for r in S.to_dict("records"):
        lines.append(f"  {r['system']:<7} {r['cylinder']:<20} runs {r['n_runs']:>2} frames {r['n_frames']:>6}  "
                     f"mean {r['mean_of_runs']:7.1f} [{r['ci95_lo']:.1f}, {r['ci95_hi']:.1f}]  median {r['median_of_runs']:7.1f}  "
                     f"SD runs {r['sd_between_runs']:5.1f}  SD frames {r['pooled_sd_frames']:5.1f}  "
                     f"inward frames {100 * r['frac_frames_inward']:4.0f}% -> {r['inward_mean_of_runs']:7.1f}")

    # 4. differences vs the reference (four-subunit mean)
    ref = systems[0]
    rows = []
    for sy in systems[1:]:
        for col, tag in (("mean", "all frames"), ("mean_inward", "inward frames")):
            a_ = R[(R["system"] == ref) & (R["cylinder"] == FOUR)][col].dropna()
            b_ = R[(R["system"] == sy) & (R["cylinder"] == FOUR)][col].dropna()
            lo, hi = boot_diff(b_, a_, a.boot)
            rows.append(dict(system=sy, reference=ref, frames=tag, difference=b_.mean() - a_.mean(),
                             ci95_lo=lo, ci95_hi=hi, n_runs=len(b_), n_runs_ref=len(a_)))
    D = pd.DataFrame(rows)
    D.to_csv(od / "differences.csv", index=False, float_format="%.3f")
    if len(D):
        lines.append(f"\nFOUR-SUBUNIT MEAN MINUS {ref} (bootstrap 95% CI over runs; positive = less negative than {ref})")
        for r in D.to_dict("records"):
            lines.append(f"  {r['system']:<7} {r['frames']:<14} {r['difference']:+7.1f} [{r['ci95_lo']:+.1f}, {r['ci95_hi']:+.1f}]")

    # 5. contributions
    cc = [c for c in use.columns if c.startswith("contrib_")]
    C = (use[use["cylinder"] != FOUR].groupby(["system", "run", "cylinder"])[cc].mean()
         .groupby(["system", "cylinder"]).mean().reset_index()) if cc else pd.DataFrame()
    if len(C):
        C.to_csv(od / "contributions.csv", index=False, float_format="%.2f")
        lines.append("\nWHERE EACH CYLINDER'S POTENTIAL COMES FROM (mean contribution, kcal/mol per +1e)")
    for r in C.to_dict("records"):
        parts = sorted(((k[8:], v) for k, v in r.items() if k.startswith("contrib_") and v == v), key=lambda x: x[1])
        lines.append(f"  {r['system']:<7} {r['cylinder']:<20} " + ", ".join(f"{k} {v:.1f}" for k, v in parts[:5]))

    # 6. profiles
    if len(P):
        pr = (P.groupby(["system", "run", "cylinder", "t_A"])["mean"].mean().reset_index()
              .groupby(["system", "cylinder", "t_A"])["mean"].agg(["mean", "std", "count"]).reset_index())
        pr.to_csv(od / "profiles.csv", index=False, float_format="%.3f")
        from analysis.cavity_esp_classical import cyl_color, cyl_order
        cyls = {sy: cyl_order(list(dict.fromkeys(pr[pr["system"] == sy]["cylinder"]))) for sy in systems}
        fig, axes = plt.subplots(1, len(systems), figsize=(5.2 * len(systems), 4.4), squeeze=False, sharey=True)
        for ax, sy in zip(axes[0], systems):
            for i, c in enumerate(cyls[sy]):
                col = cyl_color(c, i)
                g = pr[(pr["system"] == sy) & (pr["cylinder"] == c)]
                ax.plot(g["t_A"], g["mean"], color=col, lw=2, label=c)
                ax.fill_between(g["t_A"], g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.15, lw=0)
            ax.axvline(0, color="#1B2631", ls="--", lw=1)
            ax.set_title(sy)
            ax.set_xlabel("distance from the Glu CA along the cylinder (Å)")
            ax.legend(frameon=False)
        axes[0][0].set_ylabel("ESP (kcal/mol per +1e)")
        fig.tight_layout()
        fig.savefig(od / "profiles.png", dpi=180)
        plt.close(fig)

    # 7. per-run four-subunit means, and the frame distribution
    fig, ax = plt.subplots(figsize=(1.6 * len(systems) + 2.5, 4.6))
    for i, sy in enumerate(systems):
        g = R[(R["system"] == sy) & (R["cylinder"] == FOUR)]
        x = i + np.random.default_rng(i).uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g["mean"], s=28, color=COLORS[i % len(COLORS)], alpha=0.8, edgecolors="white", lw=0.8, zorder=3)
        s = S[(S["system"] == sy) & (S["cylinder"] == FOUR)].iloc[0]
        ax.errorbar(i + 0.3, s["mean_of_runs"], yerr=[[s["mean_of_runs"] - s["ci95_lo"]], [s["ci95_hi"] - s["mean_of_runs"]]],
                    fmt="o", color="#1B2631", capsize=4, zorder=4)
    for _, r in foi[foi["cylinder"] == FOUR].iterrows():
        i = systems.index(r["system"])
        ax.scatter(i - 0.3, r["mean"], marker="*", s=120, color="#F4D03F", edgecolors="#1B2631", zorder=5)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems)
    ax.set_ylabel("four-subunit mean ESP (kcal/mol per +1e)")
    ax.set_title("each dot = one run; black = mean ± 95% CI\nstar = frame of interest")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(od / "per_run.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, sy in enumerate(systems):
        v = use[(use["system"] == sy) & (use["cylinder"] == FOUR)]["mean"]
        ax.hist(v, bins=50, histtype="step", lw=2, color=COLORS[i % len(COLORS)], density=True, label=f"{sy} (n={len(v)})")
    ax.set_xlabel("four-subunit mean ESP per frame (kcal/mol per +1e)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(od / "frames_hist.png", dpi=180)
    plt.close(fig)

    # 8. each cylinder per system (distribution over frames, after equilibration), and the runs over time
    fig, axes = plt.subplots(1, len(systems), figsize=(4.6 * len(systems), 4.6), squeeze=False, sharey=True)
    for ax, sy in zip(axes[0], systems):
        g = use[use["system"] == sy]
        cy = [c for c in dict.fromkeys(g["cylinder"]) if c != FOUR] + [FOUR]
        data = [g[g["cylinder"] == c]["mean"].dropna().values for c in cy]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], cy):
            patch.set_facecolor("#C0392B" if c.startswith(("E141", "Q141")) else "#1B2631" if c == FOUR else "#D98A1C")
            patch.set_alpha(0.55)
        for m_ in bp["medians"]:
            m_.set_color("#1B2631")
        ax.set_xticks(range(1, len(cy) + 1))
        ax.set_xticklabels([c.split("/")[0] if c != FOUR else "mean" for c in cy], rotation=30, ha="right")
        ax.set_title(sy)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0][0].set_ylabel("cylinder ESP per frame (kcal/mol per +1e)")
    fig.suptitle("each cylinder, all frames after equilibration\n(box = quartiles, whiskers = 1.5 IQR)")
    fig.tight_layout()
    fig.savefig(od / "cylinders.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(systems), 1, figsize=(10, 2.6 * len(systems)), squeeze=False, sharex=True, sharey=True)
    for ax, sy, col in zip(axes[:, 0], systems, COLORS):
        g = F[(F["system"] == sy) & (F["cylinder"] == FOUR)]
        for run, gr in g.groupby("run"):
            gr = gr.sort_values("time_ns")
            w = max(1, len(gr) // 60)
            ax.plot(gr["time_ns"], gr["mean"].rolling(w, center=True, min_periods=1).mean(), color=col, lw=0.9, alpha=0.6)
            pre = gr[~gr["after_eq"]]
            if len(pre):
                ax.axvspan(pre["time_ns"].min(), pre["time_ns"].max(), color="#E6E6E6", alpha=0.25, lw=0)
        ax.set_ylabel(f"{sy}\nfour-subunit ESP")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[-1, 0].set_xlabel("time (ns); one line per run (running mean), grey = before equilibration")
    fig.tight_layout()
    fig.savefig(od / "timeseries.png", dpi=170)
    plt.close(fig)

    # 9. histograms of the ESP over the cylinder points, like esp_cylinders.py, pooled per system
    if "edges" in HIST:
        from analysis.cavity_esp_classical import plot_hist_panels, hist_stats
        for s_arg in a.system:
            lab, paths = s_arg.split("=", 1)
            hc, hf, ha, hl = {}, {}, {}, {}
            for pth in paths.split(","):
                h, f_, ax_, ln_ = HIST.get(str(Path(pth)), ({}, {}, {}, {}))
                for c, v in h.items():
                    hc[c] = hc.get(c, 0) + v
                for c, v in ax_.items():
                    ha[c] = ha.get(c, 0) + v
                for c, v in ln_.items():
                    hl.setdefault(c, []).extend(v)
            if hc:
                plot_hist_panels(hc, HIST["edges"], od / f"hist_{lab}.png",
                                 f"{lab}\nESP over the cylinder points, all runs, all frames after equilibration")
            if ha:
                from analysis.cavity_esp_classical import plot_axial
                plot_axial(ha, {c: float(np.mean(v)) for c, v in hl.items()}, od / f"axial_{lab}.png",
                           f"{lab}\naxial ESP profile, median and IQR, all runs, all frames after equilibration")
                lines.append(f"\nPOINT HISTOGRAM {lab} (all points, all frames after equilibration): " + ", ".join(
                    f"{c.split('/')[0]} mean {hist_stats(v, HIST['edges'])[0]:.1f} median {hist_stats(v, HIST['edges'])[1]:.1f}"
                    for c, v in hc.items()))

    (od / "cavity_esp.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\ncavity_esp: results in {od}")


if __name__ == "__main__":
    main()



'''
cd /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyzer
git pull origin main
R=/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results
python3 aggregate_cavity_esp.py \
  --system G12=$R/G12 --system G2=$R/G2 \
  --system D173N=$R/G12_D173N,$R/G12_D173N_4fs \
  --system E141Q=$R/G12_E141Q,$R/G12_E141Q_4fs \
  --out $R/cavity_esp_comparison
python3 aggregate_cavity_esp.py --system G2=$R/G2 --out $R/G2/cavity_esp_summary
'''
