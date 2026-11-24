"""
analysis/gate_states.py
=======================
States of the HBC gate in ONE run, for later pooling over all runs:

    python3 main.py --channel_type G12 --run_number 12 --gate_states_only
    python3 main.py --channel_type G12 --run_number 12 --gate_states_only --gs_min_jump 0.8

Results:  <results_dir>/gate_states/

IDEA
----
The two HBC diagonals (A_C and B_D, CA-CA of the gate Phe of opposite
subunits) are split separately into flat stretches = states, with the same
method as --bd_jump_only (a cut where the average level before and after
differs by >= --gs_min_jump, default 1.0 A, stretches >= 150 frames; every
stretch is split down to 150 frames and only the merging of neighbours whose
levels differ by < min_jump decides which cuts stay, so a high-low-high trace
is not missed -- --bd_jump_only keeps its faster search). A
diagonal without a jump is one state: the whole run. The joint state is the
combination (A_C state, B_D state). For every state: the average CA-CA
distances, how often (% of its frames) every chemical interaction of every
gate Phe ring is present (pi-pi, S...pi, S-H...pi, CH-pi, cation-pi; see
analysis/phe_interactions.py) and the TM2 bending (kink at the lower and
upper glycine, see analysis/gate_timeline.py), the gate Phe chi1 and its
distance from the pore axis. Frames within 25 of a jump are left out of the
state averages (they are kept, flagged, in frames.csv.gz).

OUTPUT
------
frames.csv.gz              one row per frame: A_C, B_D, state_AC, state_BD,
                           state_joint, near_jump, per gate kink_G189 /
                           kink_G180 / bend_G189 / bend_G180 / r_gate / chi1 /
                           d_pocket / latch (F83) / latch186 (F186) /
                           in_pocket (F83 or F186) / d_gate / gate_stack
                           (suffix = subunit), and a 0/1
                           column per gate x partner x interaction present in
                           >= 1 % of the run ('gate|partner|kind'). This is
                           what the cross-run clustering reads.
interaction_columns.csv    what every 'gate|partner|kind' column is (gate,
                           partner, resid, position from its HBC Phe, kind)
states.csv                 one row per diagonal (A_C, B_D, joint) x state:
                           frames, CA-CA means, per gate latch % (F83) /
                           latch186 % (F186) / in_pocket % / gate-stack % /
                           kink / local bend / segment bend (+ outward, sideways parts) /
                           segment tilts / post-helix direction /
                           noise floor (mean, sd) /
                           helix intact / r_gate / chi1 / d_pocket
state_interactions.csv     long table: diagonal x state x gate x partner x
                           kind: % frames, median distance, pi-pi geometry
states_trace.png           the two diagonals with their states, in pocket, local
                           bend G189 / G180, outward part of the segment bend,
                           lean of the upper TM2 and helix check
states_bending_<A_C|B_D|joint>.png
                           rows = bending measures (local bend G189 / G180,
                           segment bend G189 + its outward part, lean of the
                           upper TM2, segment bend G180, noise floor at A185,
                           helix intact), columns = gates: distribution per
                           state (violin, mean +- sd). The old broad 8+8 CA
                           kink is still in the CSVs but not plotted (biased:
                           ~18 deg for a straight helix)
states_tilt_<A_C|B_D|joint>.png
                           tilt of the TM2 segments 182-188 and 189-194 (helix
                           end, with the gate) relative to the pore axis, and
                           the direction of the chain after the helix end (CA
                           195 -> 199) (lean out + / in -, and total)
bend_profile.csv           local bend at every TM2 position 176-198 per state and
                           gate (mean, sd)
states_bend_profile_<A_C|B_D|joint>.png
                           bend profile along TM2, one line per state
state_interactions_<A_C|B_D|joint>.png
                           2 x 2 grid (one panel per gate), for every
                           interaction one bar per state; same columns
                           (position x kind) in every panel, equivalent
                           partner in grey italics
run_info.json              run, channel type, frames, settings
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bd_jump_residues import (collect_aligned_positions, residue_table, pore_axis,
                                       flat_stretches, res_label)
from analysis.gate_timeline import gate_signals, SUB_COLOURS, _smooth
from analysis.phe_interactions import LABEL

OUT_DIRNAME = "gate_states"
DEFAULTS = dict(min_jump=1.0, min_stretch=150, buffer=25, pore_helix=12, plot_min_pct=5.0)
STATE_COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860",
                 "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]


def states_of(x, min_jump, min_stretch, buffer):
    """flat stretches of x -> (bounds, state number per frame (1..n), near-jump mask)."""
    b = flat_stretches(x, min_jump, min_stretch, exhaustive=True)
    st = np.empty(len(x), int)
    near = np.zeros(len(x), bool)
    for k in range(len(b) - 1):
        st[b[k]:b[k + 1]] = k + 1
    for j in b[1:-1]:
        near[max(0, j - buffer):j + buffer] = True
    return b, st, near


BENDS = (("bend_G189", "local bend G189\n(HELANAL)"),
         ("seg_G189", "segment bend G189\n182-188 vs 190-194"),
         ("seg_G189_rad", "segment bend G189:\noutward part (+ = away)"),
         ("tilt_up_rad", "upper TM2 182-188:\nlean out (+) / in (-)"),
         ("bend_G180", "local bend G180\n(HELANAL)"),
         ("seg_G180", "segment bend G180\n173-179 vs 181-187"),
         ("bend_ref", "noise floor: local\nbend A185 (no Gly)"),
         ("helix_G189", "helix intact 182-194\n(fraction of i→i+4 H-bonds)"))
TILTS = (("tilt_up_rad", "upper 182-188:\nlean out (+) / in (-)"),
         ("tilt_end_rad", "helix end 189-194:\nlean out (+) / in (-)"),
         ("post_rad", "after the helix end\n195→199: lean out (+) / in (-)"),
         ("tilt_up", "upper 182-188:\ntotal tilt"),
         ("tilt_end", "helix end 189-194:\ntotal tilt"),
         ("post", "after the helix end\n195→199: total"))


def _gate_means(sig, gates, idx):
    r = {}
    for e in gates:
        s = e["subunit"]
        for key, f in (("latch", 100), ("latch186", 100), ("in_pocket", 100),
                       ("gate_stack", 100), ("kink_G189", 1), ("kink_G180", 1),
                       ("bend_G189", 1), ("bend_G180", 1), ("bend_ref", 1),
                       ("seg_G189", 1), ("seg_G189_rad", 1), ("seg_G189_tan", 1),
                       ("seg_G180", 1), ("seg_G180_rad", 1),
                       ("helix_G189", 1), ("helix_G180", 1),
                       ("tilt_up", 1), ("tilt_up_rad", 1), ("tilt_up_tan", 1),
                       ("tilt_end", 1), ("tilt_end_rad", 1), ("tilt_end_tan", 1),
                       ("post", 1), ("post_rad", 1), ("post_tan", 1),
                       ("r_gate", 1), ("chi1", 1), ("d_pocket", 1), ("d_gate", 1)):
            c = f"{key}_{s}"
            if c in sig:
                x = sig[c].to_numpy()[idx]
                r[f"{key}{'_pct' if f == 100 else ''}_{e['gate']}"] = f * np.nanmean(x)
                if key.startswith(("kink", "bend", "seg", "tilt", "post")):
                    r[f"{key}_sd_{e['gate']}"] = np.nanstd(x)
    return r


def run_gate_states(u, hbc_diagonal_pairs, sf_lower, sf_upper, results_dir, subunit_groups=None,
                    asn_residues=None, channel_type=None, **settings):
    cfg = {**DEFAULTS, **settings}
    out = Path(results_dir) / OUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    run_name = Path(results_dir).name
    print(f"\n{'=' * 60}\nGATE STATES -> {out}\n{'=' * 60}")

    gate_resids = [r for pair in hbc_diagonal_pairs for r in pair]
    traj = collect_aligned_positions(u, sf_lower, sf_upper, cfg["pore_helix"], gate_resids,
                                     atom_sets=())
    info = residue_table(u, traj["residues"], hbc_diagonal_pairs, subunit_groups, asn_residues,
                         channel_type)
    ca = traj["ca"]
    idx = {r: i for i, r in enumerate(info.resid)}
    (a1, a2), (b1, b2) = hbc_diagonal_pairs
    diag = {"A_C": np.linalg.norm(ca[:, idx[a1]] - ca[:, idx[a2]], axis=1),
            "B_D": np.linalg.norm(ca[:, idx[b1]] - ca[:, idx[b2]], axis=1)}
    axis = pore_axis(ca, info, sf_lower, sf_upper)
    sig, gates = gate_signals(traj, info, axis)
    flags, meta = traj["rings"].flags(info, min_pct=1.0)

    # ---- states
    frames = traj["frames"]
    nf = len(frames)
    st, near, bounds = {}, np.zeros(nf, bool), {}
    for d, x in diag.items():
        bounds[d], st[d], nr = states_of(x, cfg["min_jump"], cfg["min_stretch"], cfg["buffer"])
        near |= nr
        jumps = [int(frames[j]) for j in bounds[d][1:-1]]
        print(f"  {d}: {len(bounds[d]) - 1} state(s)" + (f", jumps at frames {jumps}" if jumps else
                                                         " (no jump: the whole run is one state)"))
    pair = [f"{a}-{b}" for a, b in zip(st["A_C"], st["B_D"])]
    order = list(dict.fromkeys(pair))
    st["joint"] = np.array([order.index(p) + 1 for p in pair])

    # ---- per-frame table
    fr = pd.DataFrame(dict(frame=frames, A_C=diag["A_C"], B_D=diag["B_D"], state_AC=st["A_C"],
                           state_BD=st["B_D"], state_joint=st["joint"], near_jump=near.astype(int)))
    fr = pd.concat([fr, sig.drop(columns="frame"), flags], axis=1)
    fr.to_csv(out / "frames.csv.gz", index=False, float_format="%.3f")
    meta.to_csv(out / "interaction_columns.csv", index=False)

    # ---- per-state tables
    rows, inter, prof = [], [], []
    for d in ("A_C", "B_D", "joint"):
        for s in range(1, st[d].max() + 1):
            ins = st[d] == s
            use = np.flatnonzero(ins & ~near)
            if len(use) == 0:
                use = np.flatnonzero(ins)
            f_in = frames[ins]
            lab = (f"AC{st['A_C'][ins][0]}-BD{st['B_D'][ins][0]}" if d == "joint" else f"{d} {s}")
            row = dict(run=run_name, channel_type=channel_type, diagonal=d, state=s, label=lab,
                       first_frame=int(f_in.min()), last_frame=int(f_in.max()),
                       n_frames=int(ins.sum()), n_used=len(use),
                       A_C=diag["A_C"][use].mean(), A_C_sd=diag["A_C"][use].std(),
                       B_D=diag["B_D"][use].mean(), B_D_sd=diag["B_D"][use].std())
            row.update(_gate_means(sig, gates, use))
            rows.append(row)
            for e in gates:
                for c in sig.columns:
                    if c.startswith("bprof") and c.endswith("_" + e["subunit"]):
                        pp = int(c[5:].rsplit("_", 1)[0])
                        x = sig[c].to_numpy()[use]
                        prof.append(dict(run=run_name, diagonal=d, state=s, label=lab,
                                         gate=e["gate"], subunit=e["subunit"], pos=pp,
                                         position_girk2=192 + pp, mean=np.nanmean(x),
                                         sd=np.nanstd(x)))
            sm = traj["rings"].summary(use, use, info)
            if not sm.empty:
                sm = sm.rename(columns=dict(pct_before="pct", dist_before="median_dist",
                                            angle_before="median_angle",
                                            geometry_before="geometry"))
                sm = sm[[c for c in sm.columns if not c.endswith("_after") and c != "change_pct"]]
                sm.insert(0, "run", run_name)
                sm.insert(1, "diagonal", d)
                sm.insert(2, "state", s)
                sm.insert(3, "label", lab)
                inter.append(sm)
    states = pd.DataFrame(rows)
    states.to_csv(out / "states.csv", index=False, float_format="%.3f")
    bp = pd.DataFrame(prof)
    bp.to_csv(out / "bend_profile.csv", index=False, float_format="%.3f")
    si = pd.concat(inter, ignore_index=True) if inter else pd.DataFrame()
    si.to_csv(out / "state_interactions.csv", index=False, float_format="%.3f")
    json.dump(dict(run=run_name, channel_type=channel_type, n_frames=nf,
                   hbc_diagonal_pairs=[list(map(int, p)) for p in hbc_diagonal_pairs],
                   gates=[dict(subunit=e["subunit"], gate=e["gate"], pocket=e["pocket"])
                          for e in gates], settings=cfg), open(out / "run_info.json", "w"),
              indent=1)

    # ---- plots
    plot_trace(fr, diag, bounds, gates, out / "states_trace.png", run_name, cfg)
    for d in ("A_C", "B_D", "joint"):
        plot_bending(fr, gates, st[d], near, states[states.diagonal == d],
                     out / f"states_bending_{d}.png", f"{run_name}  TM2 bending per state of {d}")
        plot_bending(fr, gates, st[d], near, states[states.diagonal == d],
                     out / f"states_tilt_{d}.png", f"{run_name}  TM2 segment tilt per state of {d}",
                     rows_def=TILTS, note="tilt of each TM2 segment relative to the pore axis (axis "
                     "through the HELANAL axis points, pointing down);\nlean + = the lower end of the "
                     "segment is further from the pore axis, - = closer")
        if not bp.empty:
            plot_profile(bp[bp.diagonal == d], states[states.diagonal == d],
                         out / f"states_bend_profile_{d}.png",
                         f"{run_name}  bend profile along TM2 per state of {d}")
        if not si.empty:
            plot_state_interactions(si[si.diagonal == d], states[states.diagonal == d],
                                    out / f"state_interactions_{d}.png",
                                    f"{run_name}  states of {d}", cfg["plot_min_pct"])

    show = ["diagonal", "state", "label", "first_frame", "last_frame", "A_C", "B_D"] + \
        [c for c in states.columns if c.startswith(("latch_pct_", "in_pocket_pct_"))]
    print(states[show].round(2).to_string(index=False))
    print(f"  -> {out}")
    return out


def plot_trace(fr, diag, bounds, gates, out_png, run_name, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = fr.frame.to_numpy()
    fig, axes = plt.subplots(8, 1, figsize=(14, 20), sharex=True)
    for ax, d in zip(axes[:2], ("A_C", "B_D")):
        x = diag[d]
        ax.plot(f, x, color="0.7", lw=0.5)
        ax.plot(f, _smooth(x), color="black", lw=1.2)
        b = bounds[d]
        for k in range(len(b) - 1):
            lo, hi = b[k], b[k + 1]
            m = x[lo:hi].mean()
            c = STATE_COLOURS[k % len(STATE_COLOURS)]
            ax.hlines(m, f[lo], f[hi - 1], color=c, lw=3)
            ax.text((f[lo] + f[hi - 1]) / 2, np.nanmax(x[lo:hi]), f"{d} {k + 1}: {m:.2f} Å",
                    color=c, ha="center", va="bottom", fontsize=12, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        for j in b[1:-1]:
            ax.axvline(f[j], color="purple", ls="--", lw=1)
        ax.set_ylabel(f"{d} Cα-Cα (Å)", fontsize=13)
        ax.set_ylim(top=np.nanmax(x) + 0.12 * (np.nanmax(x) - np.nanmin(x)) + 0.4)
    for ax, key, lab in ((axes[2], "in_pocket", "in pocket: π-stack with\nF83 or F186 (%)"),
                         (axes[3], "bend_G189", "local bend G189\nHELANAL (deg)"),
                         (axes[4], "seg_G189_rad", "segment bend G189\noutward part (deg)"),
                         (axes[5], "tilt_up_rad", "upper TM2 182-188\nlean out (deg)"),
                         (axes[6], "bend_G180", "local bend G180\nHELANAL (deg)"),
                         (axes[7], "helix_G189", "helix intact\n182-194 (fraction)")):
        for e in gates:
            c = f"{key}_{e['subunit']}"
            if c not in fr:
                continue
            y = fr[c].to_numpy() * (100 if key in ("latch", "in_pocket") else 1)
            ax.plot(f, pd.Series(y).rolling(100, center=True, min_periods=1).mean(),  # noqa
                    color=SUB_COLOURS.get(e["subunit"], "0.3"), lw=1.6, label=e["gate"])
        ax.set_ylabel(lab, fontsize=13)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", ncol=4, fontsize=11, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.grid(color="0.92", lw=0.6)
    axes[-1].set_xlabel("frame", fontsize=13)
    fig.suptitle(f"{run_name}  states of the two HBC diagonals (jump ≥ {cfg['min_jump']:g} Å); "
                 f"latch / bend / kink = running mean of 100 frames\nHELANAL = standard local "
                 f"bending angle (186-192 for G189); segment = axes of 7 CA per side through the "
                 f"HELANAL axis points; the TM2 helix ends at ~195", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_bending(fr, gates, state, near, states, out_png, title, rows_def=None, note=None):
    """grid: rows = the four bending measures, columns = the four gates
    (subunits); per state the distribution of the per-frame values (violin,
    frames within 25 of a jump left out), mean +- sd written above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    nst = int(state.max())
    rows = [(k, lab) for k, lab in (rows_def or BENDS)
            if any(f"{k}_{e['subunit']}" in fr for e in gates)]
    if not rows or not gates:
        return
    fs = 13
    fig, axes = plt.subplots(len(rows), len(gates), figsize=(4.4 * len(gates), 3.6 * len(rows)),
                             squeeze=False, sharey="row")
    for i, (key, lab) in enumerate(rows):
        for jg, e in enumerate(gates):
            ax = axes[i, jg]
            c = f"{key}_{e['subunit']}"
            if c not in fr:
                ax.axis("off")
                continue
            x = fr[c].to_numpy()
            data, pos = [], []
            for s in range(1, nst + 1):
                v = x[(state == s) & ~near]
                v = v[np.isfinite(v)]
                if len(v) > 1:
                    data.append(v)
                    pos.append(s)
            if data:
                vp = ax.violinplot(data, positions=pos, widths=0.8, showmeans=True,
                                   showextrema=False)
                for b, s in zip(vp["bodies"], pos):
                    b.set_facecolor(STATE_COLOURS[(s - 1) % len(STATE_COLOURS)])
                    b.set_alpha(0.7)
                vp["cmeans"].set_color("black")
                top = max(np.percentile(v, 99) for v in data)
                for v, s in zip(data, pos):
                    ax.text(s, top, f"{v.mean():.1f}\n±{v.std():.1f}", ha="center", va="bottom",
                            fontsize=fs - 3)
            ax.set_xticks(range(1, nst + 1))
            ax.set_xticklabels([str(s) for s in range(1, nst + 1)], fontsize=fs - 2)
            ax.tick_params(axis="y", labelsize=fs - 2)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(e["gate"], fontsize=fs + 2, fontweight="bold")
            if jg == 0:
                ax.set_ylabel(lab if key.startswith("helix") else f"{lab}\n(deg)", fontsize=fs)
            if i == len(rows) - 1:
                ax.set_xlabel("state", fontsize=fs)
            ax.margins(y=0.25)
    leg = [Patch(color=STATE_COLOURS[(int(r.state) - 1) % len(STATE_COLOURS)],
                 label=f"{int(r.state)} = {r.label}: frames {r.first_frame}-{r.last_frame}, "
                       f"A_C {r.A_C:.1f} / B_D {r.B_D:.1f} Å") for _, r in states.iterrows()]
    fig.legend(handles=leg, loc="lower center", ncol=min(3, len(leg)), frameon=False,
               fontsize=fs - 1)
    expl = note or ("HELANAL = standard local bending angle at the glycine (h-3..h+3)\n"
                    "segment = angle between the axes of 182-188 and 190-194 (the TM2 helix ends at "
                    "~195), fitted through the HELANAL axis points\nnoise floor = the same local "
                    "bend at A185, which has no glycine")
    fig.suptitle(f"{title}\n{expl}\nnumbers = mean ± sd over the frames of the state",
                 fontsize=fs + 1)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_profile(bp, states, out_png, title):
    """one panel per gate: mean local bend (HELANAL) at every TM2 position, one
    line per state (+- sd as a band); the hinge glycines (180, 189) and the gate
    Phe (192) marked. A peak is spread over about +-2 residues (each value uses
    7 CA)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if bp.empty:
        return
    gates = bp.drop_duplicates("subunit")[["gate", "subunit"]].values.tolist()
    fs = 14
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True, sharex=True)
    for ax, (gname, sub) in zip(axes.flat, gates):
        d = bp[bp.subunit == sub]
        for s, g in d.groupby("state"):
            g = g.sort_values("position_girk2")
            c = STATE_COLOURS[(int(s) - 1) % len(STATE_COLOURS)]
            ax.plot(g.position_girk2, g["mean"], "-o", color=c, lw=2, ms=4)
            ax.fill_between(g.position_girk2, g["mean"] - g["sd"], g["mean"] + g["sd"], color=c,
                            alpha=0.12)
        for x, lab in ((180, "G180"), (189, "G189"), (192, "F192")):
            ax.axvline(x, color="0.4", ls="--" if lab != "F192" else ":", lw=1)
            ax.text(x, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=fs - 3, color="0.3")
        ax.axvspan(194.5, 198.5, color="0.85", alpha=0.5, zorder=0)
        ax.text(196.5, 0.97, "helix end /\nC-linker", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=fs - 4, color="0.35")
        ax.set_title(f"{gname}", fontsize=fs + 2, fontweight="bold", pad=18)
        ax.grid(color="0.92", lw=0.6)
        ax.tick_params(labelsize=fs - 2)
    for ax in axes[1]:
        ax.set_xlabel("TM2 position (GIRK2 numbering; GIRK1 = own number + 11)", fontsize=fs)
    for ax in axes[:, 0]:
        ax.set_ylabel("local bend (HELANAL, deg)", fontsize=fs)
    for ax in axes.flat[len(gates):]:
        ax.axis("off")
    leg = [Line2D([], [], color=STATE_COLOURS[(int(r.state) - 1) % len(STATE_COLOURS)], lw=3,
                  label=f"{r.label}: frames {r.first_frame}-{r.last_frame}, A_C {r.A_C:.1f} / "
                        f"B_D {r.B_D:.1f} Å") for _, r in states.iterrows()]
    fig.legend(handles=leg, loc="lower center", ncol=min(3, len(leg)), frameon=False, fontsize=fs - 2)
    fig.suptitle(f"{title}\nlocal bending angle at every TM2 residue (mean ± sd over the frames of "
                 f"the state); a peak is spread over about ±2 residues", fontsize=fs + 1)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_state_interactions(si, states, out_png, title, min_pct=5.0):
    """2 x 2 grid, one panel per gate; columns = union over the gates of
    (partner position, interaction kind) present >= min_pct in some state;
    per column the gate's partner (the most frequent one; grey italics if it
    is below min_pct everywhere) with one bar per state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if si.empty:
        return
    si = si.copy()
    si["pos"] = si.pos.astype(int)
    nst = int(states.state.max())
    best = si.groupby(["gate", "partner", "interaction"]).pct.max().rename("best").reset_index()
    si = si.merge(best, on=["gate", "partner", "interaction"])
    slots = sorted(set(zip(si.loc[si.best >= min_pct, "pos"], si.loc[si.best >= min_pct, "interaction"])))
    if not slots:
        return
    gates = si.drop_duplicates("gate").sort_values("gate_subunit")
    fs = 15
    width = max(16, 0.55 * nst * len(slots) * 2 + 3)
    fig, axes = plt.subplots(2, 2, figsize=(width, 13), sharey=True)
    bw = 0.8 / nst
    for ax, (_, g) in zip(axes.flat, gates.iterrows()):
        s = si[si.gate == g.gate]
        for x, (p, kind) in enumerate(slots):
            c = s[(s.pos == p) & (s.interaction == kind)]
            if c.empty:
                ax.annotate(f"{192 + p}\n{LABEL.get(kind, kind)}\nnone", (x + 0.5, 0), xytext=(0, -4),
                            textcoords="offset points", ha="center", va="top", fontsize=fs - 3,
                            color="0.6", style="italic", annotation_clip=False)
                continue
            partner = c.sort_values("best", ascending=False).partner.iloc[0]
            c = c[c.partner == partner].set_index("state")
            ok = c.best.iloc[0] >= min_pct
            for k in range(1, nst + 1):
                v = c.pct.get(k, 0.0)
                ax.bar(x + 0.1 + (k - 0.5) * bw, v, bw * 0.92,
                       color=STATE_COLOURS[(k - 1) % len(STATE_COLOURS)])
            r = c.iloc[0]
            name = res_label(r.resname, r.pos, r.partner, r.subunit, two_lines=True)
            ax.annotate(f"{name}\n{LABEL.get(kind, kind)}", (x + 0.5, 0), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top", fontsize=fs - 3,
                        color="0.1" if ok else "0.5", style="normal" if ok else "italic",
                        annotation_clip=False, linespacing=1.1)
        for x in range(1, len(slots)):
            ax.axvline(x, color="0.9", lw=0.8, zorder=0)
        ax.set_xlim(0, len(slots))
        ax.set_xticks([])
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_title(f"gate {g.gate}  ({g.gate_subunit})", fontsize=fs + 3, fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(gates):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% frames of the state", fontsize=fs)
    leg = [Patch(color=STATE_COLOURS[(int(r.state) - 1) % len(STATE_COLOURS)],
                 label=f"{r.label}: frames {r.first_frame}-{r.last_frame}, "
                       f"A_C {r.A_C:.1f} / B_D {r.B_D:.1f} Å") for _, r in states.iterrows()]
    fig.legend(handles=leg, loc="lower center", ncol=min(3, len(leg)), frameon=False, fontsize=fs - 2)
    fig.suptitle(f"{title}\ninteractions of the gate Phe rings per state (π–π, S···π, S–H···π, "
                 f"CH–π, cation–π); shown: ≥ {min_pct:g} % in some state\nGIRK1 residues: own "
                 f"number, GIRK2 position in brackets; grey italics = equivalent partner of this "
                 f"gate below the filter", fontsize=fs - 1)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88), h_pad=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
