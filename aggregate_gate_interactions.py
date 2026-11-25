#!/usr/bin/env python3
"""
aggregate_gate_interactions.py
==============================
Pools the gate Phe ring interactions of every run of one system, as if all
runs were one continuous trajectory, whatever the gate does (jump or not):

    % of frames with the interaction = frames with it (all runs) / all frames (all runs)

Input: <system>/RUN*/gate_states/frames.csv.gz and interaction_columns.csv,
written by  python3 main.py --channel_type <ct> --run_number <n> --gate_states_only
(one 0/1 column per gate Phe x partner residue x interaction kind: pi-pi,
S...pi, S-H...pi, CH-pi, cation-pi; see analysis/phe_interactions.py).

A run writes a column only when the interaction is present in >= 1 % of its
frames, so a missing column counts as 0 for that run: an interaction below
1 % in a run contributes nothing to the pooled %.

GROUPS
    ALL      every run
    HIGH / TYPICAL / LOW   the permeation categories of
             <system>/permeation_summary/run_categories_full_permeations_total.csv
             (analyze_permeation_summary.py); skipped if the file is missing

USAGE
    python3 aggregate_gate_interactions.py /path/girk_analyser_results/G12
        [--min_pct 5]        show interactions with >= this % in some group
        [--skip_frames 0]    leave out the first N frames of every run
        [--no_categories]    ALL only

OUTPUT  <system>/gate_interactions_pooled/
    runs.csv                          runs used, frames, permeation category
    interactions_<GROUP>.csv          one row per gate x partner x kind: pooled %,
                                      frames with it, per-run mean / SD / min / max
    interactions_<GROUP>.png          2 x 2 (one panel per gate Phe), one bar per
                                      interaction (pooled %, value on top); --run_sd adds the SD over runs
    interactions_by_category.csv/.png the same with HIGH / TYPICAL / LOW side by side
                                      (+ ALL as a grey outline)
Columns (partner position x kind) are the same in every panel, so the four
gates can be compared; the equivalent partner of a gate that never reaches
--min_pct is drawn in grey italics.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = "gate_interactions_pooled"
CATEGORY_FILE = "permeation_summary/run_categories_full_permeations_total.csv"
GROUPS = ("HIGH", "TYPICAL", "LOW")
GROUP_COLOURS = {"ALL": "#2a78d6", "HIGH": "#1A8F68", "TYPICAL": "#2a78d6", "LOW": "#C8651B"}
LABEL = {"pi-pi": "π–π", "S...pi": "S···π", "S-H...pi": "S–H···π", "CH-pi": "CH–π", "cation-pi": "cation–π"}
HBC_GIRK2 = 192
ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", ASH="D", CYS="C", CYX="C", CYM="C", GLN="Q", GLU="E",
           GLH="E", GLY="G", HIS="H", HID="H", HIE="H", HIP="H", HSD="H", HSE="H", HSP="H", ILE="I",
           LEU="L", LYS="K", LYN="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
KEY = ["gate", "gate_subunit", "partner", "resname", "subunit", "pos", "interaction"]


def natural_key(name):
    d = "".join(c for c in str(name) if c.isdigit())
    return (int(d) if d else 0, str(name))


def res_label(resname, pos, partner, subunit):
    """F83.2c for GIRK2 residues; GIRK1 residues with their own number and the
    GIRK2 position in brackets, F72.G1 (=F83) (as in analysis/bd_jump_residues.py)."""
    partner = str(partner)
    sub = partner.split(".")[-1] if "." in partner else str(subunit)
    g2 = f"{ONE.get(resname, 'X')}{HBC_GIRK2 + int(pos)}"
    if sub.upper().startswith("G") and "." in partner:
        return partner if partner.split(".")[0] == g2 else f"{partner}\n(={g2})"
    return f"{g2}.{sub}"


def load_categories(base):
    f = base / CATEGORY_FILE
    if not f.is_file():
        print(f"(no {CATEGORY_FILE}: ALL only)")
        return {}
    c = pd.read_csv(f)
    return {str(r).strip(): str(k).strip().upper() for r, k in zip(c["Run"], c["Category"]) if pd.notna(k)}


def load_runs(base, skip):
    """[(run, frames, counts)]: counts = Series 'column' -> frames with it; plus the column meta."""
    runs, meta = [], []
    for rd in sorted(base.glob("RUN*"), key=lambda d: natural_key(d.name)):
        d = rd / "gate_states"
        if not (d / "frames.csv.gz").is_file():
            continue
        fr = pd.read_csv(d / "frames.csv.gz")
        if skip:
            fr = fr.iloc[skip:]
        try:
            m = pd.read_csv(d / "interaction_columns.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            m = pd.DataFrame(columns=["column"] + KEY)
        m = m[m.column.isin(fr.columns)]
        counts = fr[m.column].sum() if len(m) else pd.Series(dtype=float)
        counts.index = [tuple(r) for r in m[KEY].itertuples(index=False)]
        runs.append((rd.name, len(fr), counts))
        meta.append(m[KEY])
        print(f"  {rd.name:>6}: {len(fr)} frames, {len(m)} interaction columns")
    meta = pd.concat(meta).drop_duplicates() if meta else pd.DataFrame(columns=KEY)
    return runs, meta


def pool(runs, meta):
    """Pooled % and per-run spread for every interaction in meta, over the given runs."""
    tot = sum(n for _, n, _ in runs)
    rows = []
    for k in meta.itertuples(index=False):
        key = tuple(k)
        hits = np.array([c.get(key, 0.0) for _, _, c in runs], float)
        per = 100 * hits / np.array([n for _, n, _ in runs], float)
        rows.append(dict(zip(KEY, key), pct=100 * hits.sum() / tot, frames_with=int(hits.sum()), frames_total=tot,
                         n_runs=len(runs), run_mean=per.mean(), run_sd=per.std(ddof=1) if len(per) > 1 else 0.0,
                         run_min=per.min(), run_max=per.max()))
    return pd.DataFrame(rows)


def plot(tabs, out_png, title, min_pct, groups, show_all_outline=False, sd=False):
    """tabs: {group: pooled table}; one bar per group in every column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    allt = pd.concat([t.assign(group=g) for g, t in tabs.items()], ignore_index=True)
    if allt.empty:
        return
    allt["pos"] = allt.pos.astype(int)
    best = allt.groupby(["gate", "partner", "interaction"]).pct.max().rename("best").reset_index()
    allt = allt.merge(best, on=["gate", "partner", "interaction"])
    slots = sorted(set(zip(allt.loc[allt.best >= min_pct, "pos"], allt.loc[allt.best >= min_pct, "interaction"])))
    if not slots:
        print(f"  nothing >= {min_pct} %: {out_png.name} not drawn")
        return
    gates = allt.drop_duplicates("gate").sort_values("gate_subunit")
    ng, fs = len(groups), 18
    lab_fs = fs - 2
    val_fs = fs - 5 if ng > 1 else fs - 3     # % on top of the bars
    # layout in inches: the residue labels under the bars sit outside the axes, so the gaps are set
    # here from their size (up to 4 lines) instead of tight_layout
    pt = 1 / 72
    lab_h = 4 * 1.18 * lab_fs * pt + 0.12            # residue labels under a panel
    title_h = 1.35 * (fs + 3) * pt                    # panel title
    sup_h = 2 * 1.3 * (fs - 1) * pt + 0.1             # 2-line figure title
    leg_h = 1.6 * (fs - 2) * pt
    W = max(16.0, 1.05 * len(slots) * 2 + 1.6)
    ax_h = 3.6
    H = sup_h + 2 * (title_h + ax_h + lab_h) + leg_h + 0.15
    fig, axes = plt.subplots(2, 2, figsize=(W, H), sharey=True)
    fig.subplots_adjust(left=0.95 / W, right=1 - 0.15 / W, top=1 - (sup_h + title_h) / H,
                        bottom=(lab_h + leg_h + 0.1) / H, wspace=0.04,
                        hspace=(lab_h + title_h) / ax_h)
    bw = 0.78 / ng
    for ax, (_, g) in zip(axes.flat, gates.iterrows()):
        s = allt[allt.gate == g.gate]
        for x, (p, kind) in enumerate(slots):
            c = s[(s.pos == p) & (s.interaction == kind)]
            if c.empty:
                ax.annotate(f"{HBC_GIRK2 + p}\n{LABEL.get(kind, kind)}\nnone", (x + 0.5, 0), xytext=(0, -4),
                            textcoords="offset points", ha="center", va="top", fontsize=lab_fs,
                            color="0.6", style="italic", annotation_clip=False)
                continue
            partner = c.sort_values("best", ascending=False).partner.iloc[0]
            c = c[c.partner == partner]
            ok = c.best.iloc[0] >= min_pct
            for i, grp in enumerate(groups):
                r = c[c.group == grp]
                if r.empty:
                    continue
                r = r.iloc[0]
                xb = x + 0.11 + (i + 0.5) * bw
                ax.bar(xb, r.pct, bw * 0.9, color=GROUP_COLOURS.get(grp, "#2a78d6"), zorder=2)
                if r.pct >= 0.5:
                    ax.text(xb, r.pct + 1.2, f"{r.pct:.0f}", ha="center", va="bottom", zorder=5,
                            fontsize=val_fs, color="0.15",
                            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
                if sd and r.n_runs > 1:
                    ax.errorbar(xb, r.pct, yerr=r.run_sd, color="0.25", lw=0.9, capsize=2, zorder=3)
            if show_all_outline and "ALL" in set(c.group):
                ra = c[c.group == "ALL"].iloc[0]
                ax.hlines(ra.pct, x + 0.08, x + 0.92, color="0.35", lw=1.2, ls="--", zorder=4)
            r0 = c.iloc[0]
            ax.annotate(f"{res_label(r0.resname, r0.pos, r0.partner, r0.subunit)}\n{LABEL.get(kind, kind)}",
                        (x + 0.5, 0), xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                        fontsize=lab_fs, color="0.1" if ok else "0.5", style="normal" if ok else "italic",
                        annotation_clip=False, linespacing=1.1)
        for x in range(1, len(slots)):
            ax.axvline(x, color="0.9", lw=0.8, zorder=0)
        ax.set_xlim(0, len(slots)); ax.set_xticks([]); ax.set_ylim(0, 110); ax.set_yticks(range(0, 101, 20))
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_title(f"gate {g.gate}  ({g.gate_subunit})", fontsize=fs + 3, fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    for ax in axes.flat[len(gates):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of all frames", fontsize=fs)
    leg = [Patch(color=GROUP_COLOURS.get(g, "#2a78d6"), label=f"{g} ({int(tabs[g].n_runs.iloc[0])} run{'s' if int(tabs[g].n_runs.iloc[0]) != 1 else ''}, "
                 f"{int(tabs[g].frames_total.iloc[0]):,} frames)") for g in groups if g in tabs and len(tabs[g])]
    if show_all_outline:
        from matplotlib.lines import Line2D
        leg.append(Line2D([], [], color="0.35", ls="--", lw=1.2, label="ALL runs"))
    if sd:
        leg.append(Patch(color="none", label="whisker: SD over runs"))
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), frameon=False, fontsize=fs - 2,
               bbox_to_anchor=(0.5, 0.05 / H), borderaxespad=0, handlelength=1.4, columnspacing=1.6)
    fig.suptitle(f"{title}: gate Phe ring interactions, % of all frames of all runs pooled (shown: ≥ {min_pct:g} % "
                 f"in some group)\nGIRK1 residues: own number, GIRK2 position in brackets; grey italics = this gate's "
                 f"equivalent partner below the threshold", fontsize=fs - 1, y=1 - 0.05 / H, va="top")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", help="folder holding RUN*/gate_states/")
    ap.add_argument("--min_pct", type=float, default=5.0)
    ap.add_argument("--skip_frames", type=int, default=0)
    ap.add_argument("--no_categories", action="store_true")
    ap.add_argument("--run_sd", action="store_true", help="also draw the SD over runs as whiskers")
    a = ap.parse_args()
    base = Path(a.system)
    print(f"[{base.name}] {base}")
    runs, meta = load_runs(base, a.skip_frames)
    if not runs:
        raise SystemExit(f"no RUN*/gate_states/frames.csv.gz in {base} (run main.py --gate_states_only first)")
    cats = {} if a.no_categories else load_categories(base)
    out = base / OUT_DIR; out.mkdir(exist_ok=True)
    pd.DataFrame([dict(run=r, frames=n, category=cats.get(r, "")) for r, n, _ in runs]).to_csv(out / "runs.csv", index=False)
    tabs = {"ALL": pool(runs, meta)}
    for g in GROUPS:
        sub = [x for x in runs if cats.get(x[0]) == g]
        if sub:
            tabs[g] = pool(sub, meta)
    missing = [r for r, _, _ in runs if cats and r not in cats]
    if missing:
        print(f"  no permeation category for {', '.join(missing)}: in ALL only")
    for g, t in tabs.items():
        t = t.sort_values(["gate_subunit", "pos", "interaction", "pct"], ascending=[True, True, True, False])
        t.to_csv(out / f"interactions_{g}.csv", index=False, float_format="%.3f")
        runs_g = [r for r, _, _ in runs] if g == "ALL" else [r for r, _, _ in runs if cats.get(r) == g]
        print(f"  {g}: {len(runs_g)} runs ({', '.join(runs_g)})")
        plot({g: t}, out / f"interactions_{g}.png", f"{base.name} · {g}", a.min_pct, [g], sd=a.run_sd)
    cat_groups = [g for g in GROUPS if g in tabs]
    if cat_groups:
        wide = pd.concat([tabs[g].assign(group=g) for g in ["ALL"] + cat_groups])
        wide.pivot_table(index=KEY, columns="group", values="pct").reset_index().to_csv(
            out / "interactions_by_category.csv", index=False, float_format="%.3f")
        plot({g: tabs[g] for g in ["ALL"] + cat_groups}, out / "interactions_by_category.png",
             f"{base.name} · by permeation category", a.min_pct, cat_groups, show_all_outline=True, sd=a.run_sd)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
