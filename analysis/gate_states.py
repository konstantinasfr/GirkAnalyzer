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
differs by >= --gs_min_jump, default 1.0 A, stretches >= 150 frames). A
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
                           kink_G180 / r_gate / chi1 / d_pocket / latch /
                           d_gate / gate_stack (suffix = subunit), and a 0/1
                           column per gate x partner x interaction present in
                           >= 1 % of the run ('gate|partner|kind'). This is
                           what the cross-run clustering reads.
interaction_columns.csv    what every 'gate|partner|kind' column is (gate,
                           partner, resid, position from its HBC Phe, kind)
states.csv                 one row per diagonal (A_C, B_D, joint) x state:
                           frames, CA-CA means, per gate latch % / gate-stack
                           % / kink / r_gate / chi1 / d_pocket
state_interactions.csv     long table: diagonal x state x gate x partner x
                           kind: % frames, median distance, pi-pi geometry
states_trace.png           the two diagonals with their states, latch and kink
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
    b = flat_stretches(x, min_jump, min_stretch)
    st = np.empty(len(x), int)
    near = np.zeros(len(x), bool)
    for k in range(len(b) - 1):
        st[b[k]:b[k + 1]] = k + 1
    for j in b[1:-1]:
        near[max(0, j - buffer):j + buffer] = True
    return b, st, near


def _gate_means(sig, gates, idx):
    r = {}
    for e in gates:
        s = e["subunit"]
        for key, f in (("latch", 100), ("gate_stack", 100), ("kink_G189", 1), ("kink_G180", 1),
                       ("r_gate", 1), ("chi1", 1), ("d_pocket", 1), ("d_gate", 1)):
            c = f"{key}_{s}"
            if c in sig:
                r[f"{key}{'_pct' if f == 100 else ''}_{e['gate']}"] = f * np.nanmean(sig[c].to_numpy()[idx])
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
    rows, inter = [], []
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
        if not si.empty:
            plot_state_interactions(si[si.diagonal == d], states[states.diagonal == d],
                                    out / f"state_interactions_{d}.png",
                                    f"{run_name}  states of {d}", cfg["plot_min_pct"])

    show = ["diagonal", "state", "label", "first_frame", "last_frame", "A_C", "B_D"] + \
        [c for c in states.columns if c.startswith("latch_pct_")]
    print(states[show].round(2).to_string(index=False))
    print(f"  -> {out}")
    return out


def plot_trace(fr, diag, bounds, gates, out_png, run_name, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = fr.frame.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
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
            ax.text((f[lo] + f[hi - 1]) / 2, m + 0.15, f"{d} {k + 1}: {m:.2f} Å", color=c,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        for j in b[1:-1]:
            ax.axvline(f[j], color="purple", ls="--", lw=1)
        ax.set_ylabel(f"{d} Cα-Cα (Å)", fontsize=13)
    for ax, key, lab in ((axes[2], "latch", "latch: π-stack with\npocket F83 (%)"),
                         (axes[3], "kink_G189", "kink at G189\n(deg)")):
        for e in gates:
            c = f"{key}_{e['subunit']}"
            if c not in fr:
                continue
            y = fr[c].to_numpy() * (100 if key == "latch" else 1)
            ax.plot(f, pd.Series(y).rolling(100, center=True, min_periods=1).mean(),
                    color=SUB_COLOURS.get(e["subunit"], "0.3"), lw=1.6, label=e["gate"])
        ax.set_ylabel(lab, fontsize=13)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", ncol=4, fontsize=11, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.grid(color="0.92", lw=0.6)
    axes[-1].set_xlabel("frame", fontsize=13)
    fig.suptitle(f"{run_name}  states of the two HBC diagonals (jump ≥ {cfg['min_jump']:g} Å); "
                 f"latch / kink = running mean of 100 frames", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
