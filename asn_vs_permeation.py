#!/usr/bin/env python3
"""
asn_vs_permeation.py
====================

Tests one hypothesis across every G12 replica:

    Does a trajectory in which at least one pair of neighbouring ASN/ASP
    residues sits unusually close show reduced permeation?

For each RUN* directory it collects the four ADJACENT cross-subunit distances
of the ASN/ASP ring (for G12: D173.G1-N184.2a, D173.G1-N184.2c,
N184.2b-N184.2a, N184.2b-N184.2c) and pairs them with that run's permeation
count from permeation_per_run.csv.

Two ways to get the distances, tried in this order per run:

  1. <run>/cavity_environment/gate_distance_summary.csv
     Already there if run_cavity_environment_analysis() has been run.
     Free — no trajectory reading.

  2. Computed directly from the trajectory (needs --compute).
     Loops the frames once, measures the minimum heavy-atom distance between
     the sidechain functional groups (OD1/OD2 for ASP, OD1/ND2 for ASN), and
     caches the per-frame values to <run>/asn_adjacent_distances.csv so the
     second invocation is instant.

USAGE
-----
Read whatever already exists, report what is missing:

    python asn_vs_permeation.py \
        --base /media/ziyue/.../girk_analyser_results/G12 \
        --perm-csv /media/ziyue/.../girk_analyser_results/G12/permeation_summary/permeation_per_run.csv \
        --out /media/ziyue/.../girk_analyser_results/G12/permeation_summary/asn_vs_permeation

Fill in the missing runs from the trajectories (slower, once):

    python asn_vs_permeation.py ... --compute \
        --traj-root /media/ziyue/.../Konstantina/jaguar/GIRK12_WT

OUTPUTS
-------
    asn_vs_permeation_table.csv     one row per run: the 4 adjacent distances,
                                    the closest of them, the fraction of frames
                                    any pair is below --close-cut, and the
                                    permeation count
    heatmap_asn_vs_permeation.png   runs as rows sorted by permeation count,
                                    the 4 distances as coloured columns, and
                                    the permeation count as a separate column
                                    on the right
    scatter_min_asn_vs_perm.png     closest adjacent distance vs permeations,
                                    with Spearman and Pearson correlations
    scatter_fracclose_vs_perm.png   fraction of frames with a close pair vs
                                    permeations
    correlations.txt                every correlation computed, with p-values

WHAT TO LOOK FOR
----------------
If the hypothesis holds, the heatmap should show the low-permeation runs at the
top carrying at least one dark (short-distance) cell, and the high-permeation
runs at the bottom having none. The scatter should show a positive relationship
between "closest adjacent distance" and permeation count.

Correlation over 20 runs is suggestive, not causal: a clamped ring and a low
flux could both follow from something else. Treat a significant result as
grounds for the lag-correlation test within single runs, not as proof.
"""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  channel definition (G12).  Change here for other channel types.
# --------------------------------------------------------------------------- #
ASN_RESIDUES = [131, 456, 780, 1105]
ASN_DIAGONALS = [(780, 1105), (131, 456)]

FUNCTIONAL_ATOMS = {
    "ASP": ["OD1", "OD2"], "ASH": ["OD1", "OD2"],
    "ASN": ["OD1", "ND2"],
    "GLU": ["OE1", "OE2"], "GLH": ["OE1", "OE2"],
    "GLN": ["OE1", "NE2"],
}
ONE_LETTER = {"ASP": "D", "ASH": "D", "ASN": "N", "GLU": "E",
              "GLH": "E", "GLN": "Q"}


def derive_adjacent_pairs(residues, diagonals):
    diag = {frozenset(p) for p in diagonals}
    return [p for p in combinations(sorted(residues), 2)
            if frozenset(p) not in diag]


def try_to_pdb(resid, channel_type):
    """Use analysis.converter if importable, else return the raw resid."""
    try:
        from analysis.converter import to_pdb
        return to_pdb(int(resid), channel_type)
    except Exception:
        return str(resid)


# --------------------------------------------------------------------------- #
#  source 1 — existing cavity_environment summary
# --------------------------------------------------------------------------- #
def from_cavity_env(run_dir):
    """Return {pair_label: {'mean':, 'std':, 'min':}} or None."""
    p = Path(run_dir) / "cavity_environment" / "gate_distance_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["gate"] == "asn") & (df["kind"] == "adjacent")]
    if sub.empty:
        return None
    out = {}
    for _, r in sub.iterrows():
        out[r["pair"]] = dict(mean=float(r["tip_min_mean"]),
                              std=float(r["tip_min_std"]),
                              min=float(r["tip_min_min"]),
                              frac_engaged=float(r.get("frac_engaged", np.nan)))
    return out


def frac_close_from_cavity_env(run_dir, cut):
    """Fraction of frames in which ANY adjacent asn pair is below `cut`."""
    p = Path(run_dir) / "cavity_environment" / "gate_distances.csv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p, usecols=["frame", "gate", "kind", "tip_min"])
    sub = df[(df["gate"] == "asn") & (df["kind"] == "adjacent")]
    if sub.empty:
        return np.nan
    per_frame_min = sub.groupby("frame")["tip_min"].min()
    return float((per_frame_min < cut).mean())


# --------------------------------------------------------------------------- #
#  source 2 — compute from the trajectory
# --------------------------------------------------------------------------- #
def compute_from_trajectory(top_file, traj_file, adjacent, channel_type,
                            cache_csv, start_frame=0, end_frame=None,
                            frame_duration=0.075, force=False):
    """Write a per-frame CSV of the four adjacent distances; return the DataFrame."""
    cache_csv = Path(cache_csv)
    if cache_csv.exists() and not force:
        return pd.read_csv(cache_csv)

    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import distance_array
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(f"--compute needs MDAnalysis and tqdm: {e}")

    u = mda.Universe(str(top_file), str(traj_file))
    n = len(u.trajectory)
    end_frame = n - 1 if end_frame is None else min(end_frame, n - 1)

    def fg(resid):
        rn = u.select_atoms(f"resid {resid}").resnames[0]
        names = FUNCTIONAL_ATOMS.get(rn)
        ag = (u.select_atoms(f"resid {resid} and name {' '.join(names)}")
              if names else None)
        if ag is None or len(ag) == 0:
            ag = u.select_atoms(f"resid {resid} and not name N C O CA and not type H")
        return ag, rn

    groups, labels = {}, {}
    for r in ASN_RESIDUES:
        ag, rn = fg(r)
        groups[r] = ag
        one = ONE_LETTER.get(rn, rn)
        pdb = try_to_pdb(r, channel_type)
        labels[r] = f"{one}{pdb}" if pdb != str(r) else f"{rn}{r}"

    pair_labels = {(a, b): f"{labels[a]}-{labels[b]}" for a, b in adjacent}
    rows = []
    for fi in tqdm(range(start_frame, end_frame + 1),
                   desc=f"  {cache_csv.parent.name}"):
        u.trajectory[fi]
        row = {"frame": fi, "time_ns": fi * frame_duration}
        for (a, b), lab in pair_labels.items():
            row[lab] = float(distance_array(groups[a].positions,
                                            groups[b].positions).min())
        rows.append(row)

    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def summarise_per_frame(df, cut):
    pair_cols = [c for c in df.columns if c not in ("frame", "time_ns")]
    stats = {c: dict(mean=df[c].mean(), std=df[c].std(), min=df[c].min())
             for c in pair_cols}
    frac_close = float((df[pair_cols].min(axis=1) < cut).mean())
    return stats, frac_close


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #
def plot_heatmap(table, pair_cols, perm_col, out_png, close_cut):
    """Runs as rows sorted by permeation count; distances as a heatmap; the
    permeation count as a separate annotated column on the right."""
    t = table.sort_values(perm_col, ascending=True).reset_index(drop=True)
    M = t[pair_cols].to_numpy(dtype=float)
    nrow = len(t)

    fig = plt.figure(figsize=(1.55 * len(pair_cols) + 6.2, max(5.0, 0.34 * nrow + 2.2)))
    gs = fig.add_gridspec(1, 3, width_ratios=[len(pair_cols) * 1.25, 0.62, 1.9],
                          wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1], sharey=ax)
    axb = fig.add_subplot(gs[0, 2], sharey=ax)

    vmin, vmax = np.nanmin(M), np.nanmax(M)
    im = ax.imshow(M, aspect="auto", cmap="viridis",
                   norm=Normalize(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(pair_cols)))
    ax.set_xticklabels(pair_cols, rotation=32, ha="right", fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(t["Run"], fontsize=9)
    ax.set_title("Mean adjacent ASN/ASP cross-subunit distance (Å)\n"
                 "dark = closer;  runs sorted by permeation count",
                 fontsize=11, pad=10)
    for i in range(nrow):
        for j in range(len(pair_cols)):
            v = M[i, j]
            if np.isnan(v):
                continue
            close = v < close_cut
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v < (vmin + vmax) / 2 else "black",
                    fontweight="bold" if close else "normal")
            if close:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="red", lw=2.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("mean tip_min (Å)", fontsize=9)

    # closest-of-the-four column
    mn = t[pair_cols].min(axis=1).to_numpy(dtype=float)
    axp.imshow(mn.reshape(-1, 1), aspect="auto", cmap="viridis",
               norm=Normalize(vmin=vmin, vmax=vmax))
    axp.set_xticks([0]); axp.set_xticklabels(["closest\nof the 4"], fontsize=9)
    axp.tick_params(left=False, labelleft=False)
    for i, v in enumerate(mn):
        axp.text(0, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                 color="white" if v < (vmin + vmax) / 2 else "black",
                 fontweight="bold")

    # permeation bars
    axb.barh(range(nrow), t[perm_col].to_numpy(dtype=float),
             color="#C0392B", alpha=0.85, edgecolor="black", lw=0.6)
    axb.set_xlabel("permeations", fontsize=10)
    axb.tick_params(left=False, labelleft=False)
    axb.grid(axis="x", alpha=0.25)
    axb.set_title("flux", fontsize=10, pad=10)
    for i, v in enumerate(t[perm_col]):
        axb.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=8,
                 fontweight="bold")
    axb.set_xlim(0, max(t[perm_col]) * 1.18)
    axb.invert_yaxis(); ax.invert_yaxis()

    fig.suptitle("Is a clamped ASN/ASP pair associated with reduced permeation?",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_scatter(table, xcol, perm_col, out_png, xlabel, close_cut=None):
    t = table.dropna(subset=[xcol, perm_col])
    if len(t) < 3:
        return None
    x = t[xcol].to_numpy(dtype=float)
    y = t[perm_col].to_numpy(dtype=float)

    try:
        from scipy.stats import spearmanr, pearsonr
        rs, ps = spearmanr(x, y)
        rp, pp = pearsonr(x, y)
        stat = f"Spearman r = {rs:.2f} (p = {ps:.3g})\nPearson r = {rp:.2f} (p = {pp:.3g})"
    except Exception:
        rs = ps = rp = pp = np.nan
        stat = "scipy not available"

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(x, y, s=95, c="#1F77B4", edgecolor="black", lw=0.8, zorder=3)
    for xi, yi, lab in zip(x, y, t["Run"]):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color="#333333")
    if len(t) >= 3:
        z = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="#C0392B", lw=1.6,
                zorder=2, label="linear fit")
        ax.legend(fontsize=9)
    if close_cut is not None:
        ax.axvline(close_cut, color="grey", ls=":", lw=1.4)
        ax.text(close_cut, ax.get_ylim()[1], f" {close_cut} Å", va="top",
                fontsize=9, color="grey")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("permeation events", fontsize=12)
    ax.set_title("Ring geometry vs flux, one point per run", fontsize=12)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.97, stat, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                      edgecolor="black"))
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"  wrote {out_png}")
    return dict(spearman_r=rs, spearman_p=ps, pearson_r=rp, pearson_p=pp,
                n=len(t))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Relate ASN/ASP adjacent distances to permeation counts.")
    ap.add_argument("--base", required=True,
                    help="directory containing the RUN* folders")
    ap.add_argument("--perm-csv", required=True,
                    help="permeation_per_run.csv")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--perm-col", default="Full_Permeations_Unique",
                    help="which permeation column to use "
                         "(Full_Permeations_Unique | Full_Permeations_Total | "
                         "HBC_Permeations)")
    ap.add_argument("--channel-type", default="G12")
    ap.add_argument("--close-cut", type=float, default=6.0,
                    help="a pair below this counts as 'clamped' (Å). 6.0 catches "
                         "water-mediated bridges; 3.5 only direct H-bonds.")
    ap.add_argument("--compute", action="store_true",
                    help="compute distances from trajectories for runs with no "
                         "cavity_environment output")
    ap.add_argument("--traj-root", default=None,
                    help="root holding RUN*/com_4fs.prmtop and protein.nc "
                         "(needed with --compute)")
    ap.add_argument("--top-name", default="com_4fs.prmtop")
    ap.add_argument("--traj-name", default="protein.nc")
    ap.add_argument("--frame-duration", type=float, default=0.075)
    ap.add_argument("--force", action="store_true",
                    help="recompute cached per-frame CSVs")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    perm = pd.read_csv(args.perm_csv)
    if args.perm_col not in perm.columns:
        sys.exit(f"--perm-col '{args.perm_col}' not in {list(perm.columns)}")
    perm = perm[["Run", args.perm_col]].copy()

    adjacent = derive_adjacent_pairs(ASN_RESIDUES, ASN_DIAGONALS)
    print(f"ASN adjacent pairs (resids): {adjacent}")
    print(f"'clamped' threshold: {args.close_cut} Å\n")

    rows, missing = [], []
    for _, prow in perm.iterrows():
        run = str(prow["Run"])
        run_dir = base / run
        if not run_dir.is_dir():
            missing.append((run, "run directory not found"))
            continue

        stats = from_cavity_env(run_dir)
        source = "cavity_environment"
        frac_close = np.nan

        if stats is not None:
            frac_close = frac_close_from_cavity_env(run_dir, args.close_cut)

        if stats is None and args.compute:
            if not args.traj_root:
                missing.append((run, "no cavity_environment and --traj-root not given"))
                continue
            top = Path(args.traj_root) / run / args.top_name
            trj = Path(args.traj_root) / run / args.traj_name
            if not top.exists() or not trj.exists():
                missing.append((run, f"topology/trajectory not found ({top.name}/{trj.name})"))
                continue
            df = compute_from_trajectory(
                top, trj, adjacent, args.channel_type,
                cache_csv=run_dir / "asn_adjacent_distances.csv",
                frame_duration=args.frame_duration, force=args.force)
            st, frac_close = summarise_per_frame(df, args.close_cut)
            stats = st
            source = "trajectory"

        if stats is None:
            missing.append((run, "no cavity_environment output (use --compute)"))
            continue

        rec = {"Run": run, "permeations": float(prow[args.perm_col]),
               "source": source, "frac_frames_pair_close": frac_close}
        for lab, v in stats.items():
            rec[lab] = v["mean"]
            rec[lab + "__std"] = v["std"]
            rec[lab + "__min"] = v["min"]
        rows.append(rec)
        print(f"  {run:<7} {source:<19} permeations={int(prow[args.perm_col])}")

    if not rows:
        print("\nNo runs had usable ASN distances.")
        for r, why in missing:
            print(f"  {r}: {why}")
        sys.exit(1)

    table = pd.DataFrame(rows)
    pair_cols = sorted([c for c in table.columns
                        if "-" in c and not c.endswith(("__std", "__min"))])
    table["min_adjacent_mean"] = table[pair_cols].min(axis=1)
    table["max_adjacent_mean"] = table[pair_cols].max(axis=1)
    table["asym_adjacent"] = (table[pair_cols].std(axis=1)
                              / table[pair_cols].mean(axis=1))
    table["has_close_pair"] = table["min_adjacent_mean"] < args.close_cut

    ordered = (["Run", "permeations", "source"] + pair_cols
               + ["min_adjacent_mean", "max_adjacent_mean", "asym_adjacent",
                  "frac_frames_pair_close", "has_close_pair"]
               + [c for c in table.columns if c.endswith(("__std", "__min"))])
    table = table[[c for c in ordered if c in table.columns]]
    table.sort_values("permeations", inplace=True)
    table.to_csv(out / "asn_vs_permeation_table.csv", index=False)
    print(f"\n  wrote {out / 'asn_vs_permeation_table.csv'}")

    plot_heatmap(table, pair_cols, "permeations",
                 out / "heatmap_asn_vs_permeation.png", args.close_cut)
    c1 = plot_scatter(table, "min_adjacent_mean", "permeations",
                      out / "scatter_min_asn_vs_perm.png",
                      "closest adjacent ASN/ASP distance (Å)", args.close_cut)
    c2 = plot_scatter(table, "frac_frames_pair_close", "permeations",
                      out / "scatter_fracclose_vs_perm.png",
                      f"fraction of frames with a pair < {args.close_cut} Å")
    c3 = plot_scatter(table, "asym_adjacent", "permeations",
                      out / "scatter_asym_vs_perm.png",
                      "ring asymmetry (std/mean of the 4 adjacent distances)")

    lines = ["ASN/ASP adjacent distances vs permeation count",
             f"runs analysed: {len(table)}",
             f"permeation column: {args.perm_col}",
             f"clamped threshold: {args.close_cut} A", ""]
    for name, c in (("min_adjacent_mean", c1),
                    ("frac_frames_pair_close", c2),
                    ("asym_adjacent", c3)):
        if c:
            lines.append(f"{name} vs permeations   n={c['n']}  "
                         f"Spearman r={c['spearman_r']:.3f} p={c['spearman_p']:.4g}  "
                         f"Pearson r={c['pearson_r']:.3f} p={c['pearson_p']:.4g}")
    grp = table.groupby("has_close_pair")["permeations"].agg(["count", "mean", "median"])
    lines += ["", "split by whether any adjacent pair is below the threshold:",
              grp.to_string()]
    if missing:
        lines += ["", "runs skipped:"] + [f"  {r}: {w}" for r, w in missing]
    (out / "correlations.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

    if missing:
        print(f"\n{len(missing)} run(s) skipped — see correlations.txt")


if __name__ == "__main__":
    main()