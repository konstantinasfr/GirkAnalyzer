#!/usr/bin/env python3
"""
compare_cavity_environment.py
=============================

Standalone comparison of `cavity_environment/` outputs across any number of
runs.  Runs OUTSIDE main.py — you pass the run directories on the command line.

Each run directory must already contain a cavity_environment/ subfolder, i.e.
run_cavity_environment_analysis() must have been run for it first.

USAGE
-----
Two runs, labels auto-derived from the paths (G12_RUN16, G12_RUN3):

    python compare_cavity_environment.py \
        --runs /media/ziyue/.../girk_analyser_results/G12/RUN16 \
               /media/ziyue/.../girk_analyser_results/G12/RUN3 \
        --out  /media/ziyue/.../girk_analyser_results/G12/compare_RUN16_RUN3

Explicit labels, and dropping the first 1000 frames of equilibration:

    python compare_cavity_environment.py \
        --runs .../G12/RUN16 .../G12/RUN3 \
        --labels RUN16 RUN3 \
        --skip-frames 1000 \
        --out .../compare_RUN16_RUN3

More than two runs works too (delta columns are only added for exactly two):

    python compare_cavity_environment.py \
        --runs .../G12/RUN3 .../G12/RUN16 .../G12_ML/RUN3 \
        --out .../compare_three

WHY --skip-frames MATTERS
-------------------------
The first ~1000 frames are cavity filling from the starting configuration
(waters climbing from ~20 to ~90, ions from 2 to 4).  Including them biases
every baseline mean downward, and biases it differently per run.  Pass
--skip-frames to drop them from all per-frame statistics.  Permeation-event
frames before the cutoff are dropped too.

OUTPUTS (in --out)
------------------
    compare_summary.txt                 headline numbers, read this first
    compare_cavity_metrics.csv          one row per metric, one column per run
    compare_occupancy.csv               K+ occupancy distribution + conditional rate
    compare_gate_distances.csv          per gate/pair mean, std, frac_engaged
    compare_symmetry.csv                per gate asymmetry scalars
    compare_salt_bridges.csv            per pair occupancy, sorted by spread
    compare_radius_sensitivity.csv      is the water radius choice load-bearing?
    plots/occupancy_distribution.png
    plots/water_per_ion.png
    plots/ion_radial_distribution.png
    plots/hydration_distribution.png
    plots/gate_<gate>_tip_min.png       overlaid distributions, one panel per pair
    plots/symmetry_<gate>.png

Gate pairs are joined across runs on the converter-based PDB label
(e.g. 'D173.G1-N184.2a'), so runs of different channel types still align even
though their raw resids differ.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HBOND_CUT = 3.5


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
def default_label(run_dir):
    """.../girk_analyser_results/G12/RUN16 -> 'G12_RUN16'"""
    p = Path(run_dir).resolve()
    parts = [x for x in p.parts if x not in ("/", "")]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else str(p)


def load_run(run_dir, label, skip_frames=0):
    env = Path(run_dir) / "cavity_environment"
    if not env.is_dir():
        raise SystemExit(
            f"ERROR: {env} not found.\n"
            f"       Run run_cavity_environment_analysis() for {run_dir} first."
        )

    def _csv(name):
        p = env / name
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if skip_frames and "frame" in df.columns:
            df = df[df["frame"] >= skip_frames]
        return df

    meta = {}
    if (env / "run_meta.json").exists():
        meta = json.loads((env / "run_meta.json").read_text())

    perm = [int(f) for f in meta.get("permeation_frames", [])
            if int(f) >= skip_frames]

    return dict(
        label=label, dir=Path(run_dir), env=env, meta=meta,
        perm_frames=perm,
        geo=_csv("gate_distances.csv"),
        sym=_csv("gate_symmetry.csv"),
        wat=_csv("water_cavity.csv"),
        ions=_csv("ion_coordination.csv"),
        sb=_csv("salt_bridges.csv"),
        geo_sum=_csv("gate_distance_summary.csv"),
        sb_sum=_csv("salt_bridge_summary.csv"),
        rad=_csv("radius_sensitivity.csv"),
    )


# --------------------------------------------------------------------------- #
#  tables
# --------------------------------------------------------------------------- #
def add_delta(df, labels, cols=None):
    """For exactly two runs, add a `<col>_delta` = first - second."""
    if len(labels) != 2:
        return df
    a, b = labels
    for c in (cols or df.columns):
        ca, cb = f"{c}_{a}", f"{c}_{b}"
        if ca in df.columns and cb in df.columns:
            df[f"{c}_delta"] = df[ca] - df[cb]
    return df


def table_cavity_metrics(runs, out):
    rows = []
    for r in runs:
        rec = {"run": r["label"]}
        m, w, i = r["meta"], r["wat"], r["ions"]

        rec["channel_type"] = m.get("channel_type", "")
        rec["n_permeations"] = len(r["perm_frames"])
        rec["ion_count_mode"] = m.get("ion_count_mode", "")
        rec["ion_regions"] = str(m.get("ion_regions", ""))
        rec["cavity_radius"] = m.get("cavity_radius", np.nan)

        if w is not None and len(w):
            rec["n_frames_used"] = len(w)
            rec["n_k_mean"] = w["n_k_cavity"].mean()
            rec["n_k_mode"] = w["n_k_cavity"].mode().iloc[0]
            rec["n_k_std"] = w["n_k_cavity"].std()
            if "n_k_untracked" in w:
                rec["n_k_untracked_mean"] = w["n_k_untracked"].mean()
            rec["n_water_mean"] = w["n_water_cavity"].mean()
            rec["water_per_ion"] = w["water_per_ion"].mean()
            rec["cavity_height"] = w["cavity_height"].mean()
            # permeation rate per 1000 frames
            rec["perm_per_1000_frames"] = (
                1000.0 * len(r["perm_frames"]) / len(w))

        if i is not None and len(i):
            rec["hyd_mean"] = i["hyd_n"].mean()
            rec["d_carbox_min_mean"] = i["d_carbox_min"].mean()
            rec["r_axis_mean"] = i["r_axis"].mean()
            rec["frac_off_axis_gt6"] = (i["r_axis"] > 6.0).mean()
            for cls in ("contact", "solvent_sep", "free"):
                rec[f"frac_{cls}"] = (i["coord_class"] == cls).mean()
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("run").T
    if len(runs) == 2:
        a, b = [r["label"] for r in runs]
        num = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b],
                                                                    errors="coerce")
        df["delta"] = num
    df.to_csv(out / "compare_cavity_metrics.csv")
    return df


def table_occupancy(runs, out):
    """Occupancy distribution + conditional permeation rate per occupancy."""
    frames = []
    for r in runs:
        w = r["wat"]
        if w is None or not len(w):
            continue
        occ = w.set_index("frame")["n_k_cavity"]
        tab = occ.value_counts().sort_index().rename("n_frames").to_frame()
        tab["frac_frames"] = tab["n_frames"] / tab["n_frames"].sum()
        ev = pd.Series([occ.get(f, np.nan) for f in r["perm_frames"]]).dropna()
        tab["n_permeations"] = ev.value_counts().reindex(tab.index).fillna(0)
        # events per 1000 frames spent at this occupancy
        tab["rate_per_1000_frames"] = (
            1000.0 * tab["n_permeations"] / tab["n_frames"])
        tab.index.name = "n_k_cavity"
        tab = tab.reset_index()
        tab["run"] = r["label"]
        frames.append(tab)
    if not frames:
        return None
    df = pd.concat(frames)
    wide = df.pivot_table(index="n_k_cavity", columns="run",
                          values=["n_frames", "frac_frames",
                                  "n_permeations", "rate_per_1000_frames"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wide = add_delta(wide, [r["label"] for r in runs],
                     cols=["frac_frames", "rate_per_1000_frames"])
    wide.to_csv(out / "compare_occupancy.csv", index=False)
    return wide


def table_gate_distances(runs, out):
    parts = []
    for r in runs:
        g = r["geo_sum"]
        if g is None or not len(g):
            continue
        g = g.copy()
        g["run"] = r["label"]
        parts.append(g)
    if not parts:
        return None
    g = pd.concat(parts)
    wide = g.pivot_table(index=["gate", "pair", "kind"], columns="run",
                         values=["tip_min_mean", "tip_min_std",
                                 "frac_engaged", "ca_mean"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wide = add_delta(wide, [r["label"] for r in runs],
                     cols=["tip_min_mean", "frac_engaged", "ca_mean"])
    # biggest differences first
    sort_col = next((c for c in wide.columns if c == "frac_engaged_delta"), None)
    if sort_col:
        wide = wide.reindex(wide[sort_col].abs().sort_values(
            ascending=False).index)
    wide.to_csv(out / "compare_gate_distances.csv", index=False)
    return wide


def table_symmetry(runs, out):
    parts = []
    for r in runs:
        s = r["sym"]
        if s is None or not len(s):
            continue
        cols = [c for c in ("asym_adjacent", "diag_ratio", "adj_mean",
                            "diag_mean", "adj_min") if c in s.columns]
        agg = s.groupby("gate")[cols].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()
        agg["run"] = r["label"]
        parts.append(agg)
    if not parts:
        return None
    s = pd.concat(parts)
    value_cols = [c for c in s.columns if c not in ("gate", "run")]
    wide = s.pivot_table(index="gate", columns="run", values=value_cols)
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wide = add_delta(wide, [r["label"] for r in runs],
                     cols=[f"{c}" for c in value_cols])
    wide.to_csv(out / "compare_symmetry.csv", index=False)
    return wide


def table_salt_bridges(runs, out):
    parts = []
    for r in runs:
        s = r["sb_sum"]
        if s is None or not len(s):
            continue
        s = s.copy()
        s["run"] = r["label"]
        parts.append(s)
    if not parts:
        return None
    s = pd.concat(parts)
    wide = s.pivot_table(index=["kind", "partner_a", "partner_b"],
                         columns="run", values=["occupancy", "d_min_mean"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    occ_cols = [c for c in wide.columns if c.startswith("occupancy_")]
    for c in occ_cols:
        wide[c] = wide[c].fillna(0.0)
    if len(occ_cols) > 1:
        wide["occupancy_spread"] = (wide[occ_cols].max(axis=1)
                                    - wide[occ_cols].min(axis=1))
        wide = wide.sort_values("occupancy_spread", ascending=False)
    wide = add_delta(wide, [r["label"] for r in runs],
                     cols=["occupancy", "d_min_mean"])
    wide.to_csv(out / "compare_salt_bridges.csv", index=False)
    return wide


def table_radius(runs, out):
    parts = []
    for r in runs:
        rad = r["rad"]
        if rad is None or not len(rad):
            continue
        rad = rad.copy()
        rad["run"] = r["label"]
        parts.append(rad)
    if not parts:
        return None
    rad = pd.concat(parts)
    wide = rad.pivot_table(index="radius", columns="run",
                           values=["water_mean", "water_rel_to_used"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wcols = [c for c in wide.columns if c.startswith("water_mean_")]
    if len(wcols) == 2:
        # ratio between the two runs at each radius: flat = radius-robust
        wide["ratio_between_runs"] = wide[wcols[0]] / wide[wcols[1]]
    wide.to_csv(out / "compare_radius_sensitivity.csv", index=False)
    return wide


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #
def plot_overlay_hist(runs, key, col, xlabel, title, path, bins=50,
                      vline=None, integer=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for r in runs:
        df = r[key]
        if df is None or col not in getattr(df, "columns", []) or not len(df):
            continue
        v = df[col].dropna()
        if not len(v):
            continue
        any_data = True
        if integer:
            vals = sorted(v.unique())
            frac = [(v == x).mean() for x in vals]
            ax.plot(vals, frac, marker="o", lw=1.6, label=r["label"])
        else:
            ax.hist(v, bins=bins, histtype="step", density=True, lw=1.6,
                    label=r["label"])
    if not any_data:
        plt.close(fig)
        return
    if vline is not None:
        ax.axvline(vline, color="r", ls=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of frames" if integer else "density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gate_pairs(runs, plots):
    parts = []
    for r in runs:
        g = r["geo"]
        if g is None or not len(g):
            continue
        g = g[["gate", "pair", "kind", "tip_min", "ca"]].copy()
        g["run"] = r["label"]
        parts.append(g)
    if not parts:
        return
    allg = pd.concat(parts)
    for gate, sub in allg.groupby("gate"):
        pairs = sorted(sub["pair"].unique())
        n = len(pairs)
        ncol = min(3, max(1, n))
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4.3 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for ax, pair in zip(axes.ravel(), pairs):
            ps = sub[sub["pair"] == pair]
            for lab, pl in ps.groupby("run"):
                ax.hist(pl["tip_min"], bins=50, histtype="step",
                        density=True, lw=1.4, label=lab)
            ax.axvline(HBOND_CUT, color="r", ls=":", lw=1)
            kind = ps["kind"].iloc[0]
            ax.set_title(f"{pair}  [{kind}]", fontsize=8)
            ax.set_xlabel("tip_min (Å)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"{gate} gate — cross-subunit min distances", fontsize=11)
        fig.tight_layout()
        fig.savefig(plots / f"gate_{gate}_tip_min.png", dpi=160)
        plt.close(fig)


def plot_symmetry(runs, plots):
    parts = []
    for r in runs:
        s = r["sym"]
        if s is None or not len(s):
            continue
        s = s.copy()
        s["run"] = r["label"]
        parts.append(s)
    if not parts:
        return
    alls = pd.concat(parts)
    for gate, sub in alls.groupby("gate"):
        cols = [c for c in ("asym_adjacent", "diag_ratio") if c in sub.columns]
        if not cols:
            continue
        fig, axes = plt.subplots(1, len(cols),
                                 figsize=(5.2 * len(cols), 4), squeeze=False)
        for ax, c in zip(axes[0], cols):
            for lab, pl in sub.groupby("run"):
                ax.hist(pl[c].dropna(), bins=50, histtype="step",
                        density=True, lw=1.4, label=lab)
            ax.set_xlabel(c)
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle(f"{gate} gate symmetry", fontsize=11)
        fig.tight_layout()
        fig.savefig(plots / f"symmetry_{gate}.png", dpi=160)
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  headline text summary
# --------------------------------------------------------------------------- #
def write_summary(runs, metrics, occ, gates, sb, out, skip_frames):
    lines = []
    lines.append("=" * 78)
    lines.append("CAVITY ENVIRONMENT COMPARISON")
    lines.append("=" * 78)
    for r in runs:
        m = r["meta"]
        lines.append(
            f"  {r['label']:<16} {m.get('channel_type',''):<8} "
            f"frames {m.get('start_frame','?')}-{m.get('end_frame','?')}  "
            f"{len(r['perm_frames'])} permeations")
    if skip_frames:
        lines.append(f"  (first {skip_frames} frames excluded as equilibration)")
    lines.append("")

    lines.append("-" * 78)
    lines.append("CAVITY METRICS")
    lines.append("-" * 78)
    lines.append(metrics.to_string())
    lines.append("")

    if occ is not None:
        lines.append("-" * 78)
        lines.append("K+ OCCUPANCY DISTRIBUTION AND CONDITIONAL RATE")
        lines.append("  rate_per_1000_frames = events / time spent at that N.")
        lines.append("  This is the dwell-time-corrected number; raw event")
        lines.append("  counts are confounded by how long each run sits at N.")
        lines.append("-" * 78)
        lines.append(occ.to_string(index=False))
        lines.append("")

    if gates is not None and len(gates):
        lines.append("-" * 78)
        lines.append("GATE PAIRS — LARGEST DIFFERENCES IN frac_engaged (<3.5 Å)")
        lines.append("-" * 78)
        cols = ["gate", "pair", "kind"] + [
            c for c in gates.columns
            if c.startswith(("frac_engaged", "tip_min_mean"))]
        lines.append(gates[cols].head(12).to_string(index=False))
        lines.append("")

    if sb is not None and len(sb):
        lines.append("-" * 78)
        lines.append("SALT BRIDGES / H-BONDS — LARGEST OCCUPANCY DIFFERENCES")
        lines.append("-" * 78)
        cols = ["kind", "partner_a", "partner_b"] + [
            c for c in sb.columns if c.startswith("occupancy")]
        lines.append(sb[cols].head(12).to_string(index=False))
        lines.append("")

    lines.append("=" * 78)
    lines.append("HOW TO READ THIS")
    lines.append("=" * 78)
    lines.append(
        "  1. Salt bridge occupancy differences: is a specific interaction\n"
        "     present in one run and absent in the other?\n"
        "  2. frac_engaged on the asn gate: is it the ASN...ASP clamp?\n"
        "     Only the D173-containing pairs can form a charged H-bond.\n"
        "  3. frac_off_axis_gt6 and water_per_ion: does that clamp change\n"
        "     where ions sit and how hydrated they are?\n"
        "  4. If geometry differs but ion metrics do not, the clamp is\n"
        "     incidental — look at the hbc gate instead.\n"
        "  Small event counts make the conditional rates indicative only;\n"
        "  the frame-averaged metrics are the statistically solid ones.")

    text = "\n".join(lines)
    (out / "compare_summary.txt").write_text(text)
    print(text)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Compare cavity_environment outputs across runs.")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (each containing cavity_environment/)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="short labels, same order as --runs "
                         "(default: derived from the paths)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--skip-frames", type=int, default=0,
                    help="drop frames below this index from all per-frame "
                         "statistics (equilibration)")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        sys.exit("ERROR: --labels must have the same number of entries as --runs")
    labels = args.labels or [default_label(r) for r in args.runs]
    if len(set(labels)) != len(labels):
        sys.exit(f"ERROR: labels must be unique, got {labels}")

    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d, l, args.skip_frames) for d, l in zip(args.runs, labels)]

    metrics = table_cavity_metrics(runs, out)
    occ = table_occupancy(runs, out)
    gates = table_gate_distances(runs, out)
    table_symmetry(runs, out)
    sb = table_salt_bridges(runs, out)
    table_radius(runs, out)

    plot_overlay_hist(runs, "wat", "n_k_cavity",
                      "K+ in cavity", "K+ occupancy distribution",
                      plots / "occupancy_distribution.png", integer=True)
    plot_overlay_hist(runs, "wat", "water_per_ion",
                      "waters per K+", "Cavity hydration budget per ion",
                      plots / "water_per_ion.png")
    plot_overlay_hist(runs, "wat", "n_water_cavity",
                      "waters in cavity", "Cavity water count",
                      plots / "water_count.png")
    plot_overlay_hist(runs, "ions", "r_axis",
                      "radial distance from pore axis (Å)",
                      "On-axis vs off-axis K+ distribution",
                      plots / "ion_radial_distribution.png", vline=6.0)
    plot_overlay_hist(runs, "ions", "hyd_n",
                      "first-shell water count", "K+ hydration number",
                      plots / "hydration_distribution.png", integer=True)
    plot_overlay_hist(runs, "ions", "d_carbox_min",
                      "min K+–carboxylate O (Å)",
                      "Contact vs solvent-separated ion pairing",
                      plots / "ion_carboxylate_distance.png", vline=3.2)
    plot_gate_pairs(runs, plots)
    plot_symmetry(runs, plots)

    write_summary(runs, metrics, occ, gates, sb, out, args.skip_frames)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

'''
python3 compare_cavity_environment.py \
    --runs /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN16 \
           /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/RUN3 \
    --labels RUN16 RUN3 \
    --skip-frames 500 \
    --out /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/compare_RUN16_RUN3_skip500
'''