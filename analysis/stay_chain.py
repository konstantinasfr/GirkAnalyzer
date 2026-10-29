"""
stay_chain.py
=============
A stay-to-stay (jump) chain of permeation through the GLU/ASN cavity, and
the tests of whether it is Markovian.

Every permeation event (end_1 -> end_2) is turned into the list of its stays
on the cavity residues, in time order:

    Entry -> stay 1 -> stay 2 -> ... -> out

A STAY is what the existing analyses call one: >= min_duration consecutive
frames in which the same residue is the closest of the GLU / ASN-ASP rings
within `cutoff` of the ion (any atom of the residue),
LastAndLongestDurationAnalysis.find_last_and_longest_duration(...)
['all_durations'], as in touch_to_next_stay.py and the first-stay plots.
Shorter contacts are not stays, so flicker never makes a step.

STATE of a stay = where + how many K+ are above the ion when it LEAVES it:
    level='ring'     resname of the residue (GLU, ASN, ASP; GLN / ASN in the
                     E141Q / D173N mutants), e.g. 'ASN|3'
    level='residue'  the residue itself, e.g. '184.2a|3'
    K+ above = other K+ in the cavity (region 2, main.py's ch2 cylinder)
    further toward the SF along the ch2 axis than the permeating ion, at the
    last frame of the stay; the same count as n_above of touch_to_next_stay.py.
    Binned 0, 1, ..., max_bin-1, max_bin+ (default 0-3, 4+).
    Entry (the ion has left the SF, no stay yet) and out are single states.

THE CHAIN
---------
transitions     counts of state -> next state over every event (Entry -> first
                stay, stay -> stay, last stay -> out). The table shown is
                P(next ring or out | ring, K+ above when leaving).
control         the same table with the other ions taken from frames 1/4, 1/2,
                3/4 of the run later (the permeating ion at the real frame): it
                keeps what the ion's own position does to the count and removes
                any real effect of the other ions.

IS IT MARKOVIAN?
----------------
1 history       for each state with enough departures: is the next ring / out
                independent of the ring the ion came from (Entry, GLU, ASN,
                ASP)? Chi-square test of the contingency table (scipy), with the
                table itself. Markov: no dependence.
2 stay times    per ring, the distribution of stay lengths against an
                exponential with the same mean (survival plot).
3 prediction    the chain is run forward (semi-Markov: next state from the
                transition counts, stay lengths and the free times between
                stays drawn from the observed ones of that state / step) and
                compared with the real events: number of stays before exit,
                share of events with no stay, transit time end_1 -> end_2
                (median, KS gap).

OUTPUTS   <results_dir>/stay_chain/<settings>/   (settings e.g. t3.5_d10)
------------------------------------------------
    stays_run<run>.csv     one row per stay (residue, frames, K+ above / below /
                           in the SF when it leaves, the same for each control shift)
    events_run<run>.csv    one row per permeation event (end_1, end_2, number of stays)
    <level>/summary.txt    every number below; <level>/ = ring or residue
    <level>/transitions.csv, next_by_above.csv, history_test.csv
    <level>/next_by_above.png       P(next | ring, K+ above), one panel per ring
    <level>/out_vs_above.png        P(out next) against K+ above, real and control
    <level>/prediction.png          chain against real: stays per event, transit time
    <level>/stay_times.png          stay-length survival per ring
Over runs: aggregate_stay_chain.py (repo root) pools the stays CSVs of all
runs of a system and makes the same outputs, with per-run SEM in the tables.

USAGE - from main.py (after ch1, ch2 and analyzer exist)
--------------------------------------------------------
    from analysis.stay_chain import run_stay_chain
    run_stay_chain(u, analyzer.permeation_events, asn_residues, glu_residues,
                   channel1=ch1, channel2=ch2, results_dir=results_dir,
                   run=run_number, channel_type=args.channel_type,
                   frame_duration=frame_duration)

CLI (G12 runs 1-20 built in, like markov_check.py):
    python3 -m analysis.stay_chain --channel_type G12 --run 3
"""

import argparse
import contextlib
import io
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ION_SELECTION = "resname K+ K"
ENTRY, OUT = "Entry", "out"
LEVELS = ("ring", "residue")
CONTROL_SHIFTS = (0.25, 0.5, 0.75)
OUT_FOLDER = "stay_chain"
MIN_ROW = 20                      # departures needed for a row of the tables / tests
N_SIM = 20000
FROM_NEXT = "from \\ next"


def default_min_duration(channel_type):
    """Minimum stay: 2 frames for G2 (0.4 ns frames), 10 for the G12 systems."""
    return 2 if str(channel_type).upper().startswith("G2") else 10


def settings_folder(cutoff, min_duration):
    return f"t{cutoff:g}_d{min_duration}"


# ─────────────────────────────────────────────────────────────────────────────
# Collecting the stays (needs the trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def _inside(channel, pos):
    """Vectorised Channel.is_within_cylinder."""
    rel = pos - channel.channel_center
    axial = rel @ channel.channel_axis
    radial = np.linalg.norm(rel - axial[:, None] * channel.channel_axis, axis=1)
    return (radial <= channel.radius) & (np.abs(axial) <= channel.channel_length / 2)


class PoreFrames:
    """Per frame: position of every K+ along the pore (+ = toward the SF), in cavity, in SF."""

    def __init__(self, universe, channel1, channel2):
        self.u = universe
        self.ch1, self.ch2 = channel1, channel2
        self.ions = universe.select_atoms(ION_SELECTION)
        self.col = {int(r): i for i, r in enumerate(self.ions.resids)}
        self._cache = {}

    def get(self, frame):
        if frame not in self._cache:
            self.u.trajectory[int(frame)]
            self.ch1.compute_geometry(1)
            self.ch2.compute_geometry(2)
            axis = self.ch2.channel_axis
            to_sf = axis if np.dot(self.ch1.channel_center - self.ch2.channel_center, axis) > 0 else -axis
            pos = self.ions.positions
            in_sf = _inside(self.ch1, pos)
            in_cav = _inside(self.ch2, pos) & ~in_sf
            self._cache[frame] = ((pos - self.ch2.channel_center) @ to_sf, in_cav, in_sf)
        return self._cache[frame]

    def neighbours(self, ion_id, frame, other_frame=None):
        """(n_above, n_below, n_sf) of the other K+, taken at other_frame (default: frame)."""
        t, _, _ = self.get(frame)
        c = self.col[int(ion_id)]
        t2, cav2, sf2 = self.get(frame if other_frame is None else other_frame)
        others = np.ones(len(t2), dtype=bool)
        others[c] = False
        return (int((others & cav2 & (t2 > t[c])).sum()),
                int((others & cav2 & (t2 < t[c])).sum()),
                int((others & sf2).sum()))


def collect_stays(universe, permeation_events, asn_residues, glu_residues, channel1, channel2,
                  run, channel_type, cutoff, min_duration, analysis_start_frame=0):
    """(stays DataFrame, events DataFrame) of one run; see the module docstring."""
    from analysis.longest_duration_analysis import LastAndLongestDurationAnalysis

    with contextlib.redirect_stdout(io.StringIO()):            # its constructor prints
        durations = LastAndLongestDurationAnalysis(
            universe, permeation_events, list(asn_residues), list(glu_residues), channel_type,
            threshold=cutoff, min_duration=min_duration, analysis_start_frame=analysis_start_frame,
            residue_types=["ASN", "GLU"], subunit_groups=None, results_dir=Path("."))
    pore = PoreFrames(universe, channel1, channel2)
    n_frames = len(universe.trajectory)
    glu = {int(r) for r in glu_residues}

    stays, events = [], []
    for k, ev in enumerate(permeation_events):
        ion_id = int(ev["ion_id"])
        end_1 = int(ev["end_1"] if "end_1" in ev else ev["R1_end"])
        end_2 = int(ev["end_2"] if "end_2" in ev else ev["R2_end"])
        found = durations.find_last_and_longest_duration(ion_id, end_2, end_1)["all_durations"]
        found = sorted(found, key=lambda d: d["start_frame"])
        events.append(dict(run=run, event=k, ion_id=ion_id, end_1=end_1, end_2=end_2,
                           starts_at_frame0=end_1 == 0, n_stays=len(found)))
        for i, d in enumerate(found):
            res = d["residues"][0]
            leave = int(d["end_frame"])
            row = dict(run=run, event=k, ion_id=ion_id, end_1=end_1, end_2=end_2, stay=i,
                       residue_id=int(res["residue_id"]), residue_pdb=res["residue_pdb"],
                       resname=res["residue_type"],
                       ring="GLU" if int(res["residue_id"]) in glu else "LOWER",
                       start_frame=int(d["start_frame"]), end_frame=leave)
            row["n_above"], row["n_below"], row["n_sf"] = pore.neighbours(ion_id, leave)
            for j, frac in enumerate(CONTROL_SHIFTS, start=1):
                other = (leave + int(frac * n_frames)) % n_frames
                row[f"n_above_c{j}"], row[f"n_below_c{j}"], _ = pore.neighbours(ion_id, leave, other)
            stays.append(row)
    return pd.DataFrame(stays), pd.DataFrame(events)


# ─────────────────────────────────────────────────────────────────────────────
# The chain
# ─────────────────────────────────────────────────────────────────────────────

def bin_label(n, max_bin):
    return f"{max_bin}+" if n >= max_bin else str(int(n))


def where(row, level):
    return row["resname"] if level == "ring" else row["residue_pdb"]


def event_paths(stays, events, level, max_bin, above_col="n_above"):
    """
    One path per event: list of steps (from_state, from_where, prev_where, to_state,
    to_where, gap_frames, stay_frames_of_from). gap = free frames before the next stay
    (or before end_2 for out); stay frames = length of the from stay (None for Entry).
    """
    by_event = {key: g.sort_values("start_frame") for key, g in stays.groupby(["run", "event"])}
    paths = []
    for ev in events.itertuples(index=False):
        g = by_event.get((ev.run, ev.event))
        rows = [] if g is None else list(g.to_dict("records"))
        states = [f"{where(r, level)}|{bin_label(r[above_col], max_bin)}" for r in rows]
        wheres = [where(r, level) for r in rows]
        steps = []
        prev_end, prev_state, prev_where, before = ev.end_1, ENTRY, ENTRY, None
        prev_len = None
        for r, s, w in zip(rows, states, wheres):
            steps.append((prev_state, prev_where, before, s, w, r["start_frame"] - prev_end, prev_len))
            before = prev_where
            prev_state, prev_where = s, w
            prev_len = r["end_frame"] - r["start_frame"]
            prev_end = r["end_frame"]
        steps.append((prev_state, prev_where, before, OUT, OUT, ev.end_2 - prev_end, prev_len))
        paths.append(dict(run=ev.run, event=ev.event, transit=ev.end_2 - ev.end_1,
                          n_stays=len(rows), steps=steps))
    return paths


def transition_counts(paths):
    counts = {}
    for p in paths:
        for f, _, _, t, _, _, _ in p["steps"]:
            counts.setdefault(f, {}).setdefault(t, 0)
            counts[f][t] += 1
    return counts


def next_by_above(paths, wheres, bins):
    """
    DataFrame: one row per (where, bin) with the number of departures and the
    % going next to each where / out. Built from the steps leaving a stay.
    """
    targets = list(wheres) + [OUT]
    rows = []
    for w in wheres:
        for b in bins:
            state = f"{w}|{b}"
            nxt = [t for p in paths for f, _, _, _, t, _, _ in p["steps"] if f == state]
            if not nxt:
                continue
            row = dict(where=w, above=b, n=len(nxt))
            for t in targets:
                row[t] = 100.0 * sum(x == t for x in nxt) / len(nxt)
            rows.append(row)
    return pd.DataFrame(rows, columns=["where", "above", "n"] + targets)


def p_out_by_above(paths, w, bins):
    """(P(out next), n departures) per bin for one where."""
    out = []
    for b in bins:
        state = f"{w}|{b}"
        nxt = [t for p in paths for f, _, _, _, t, _, _ in p["steps"] if f == state]
        out.append((np.mean([t == OUT for t in nxt]) if nxt else np.nan, len(nxt)))
    return out


def bootstrap_p_out(paths, w, bins, n_boot, rng):
    """SD of P(out next) per bin over bootstrap resamples of the events."""
    reps = []
    for _ in range(n_boot):
        sample = [paths[i] for i in rng.integers(0, len(paths), len(paths))]
        reps.append([p for p, _ in p_out_by_above(sample, w, bins)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # bins empty in every resample
        return np.nanstd(np.array(reps, dtype=float), axis=0)


def history_test(paths, min_row=MIN_ROW):
    """
    For each stay state with >= min_row departures: contingency table previous
    where (Entry / ring) x next where (ring / out), chi-square p-value.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError:                                    # scipy optional
        chi2_contingency = None
    by_state = {}
    for p in paths:
        for f, _, before, _, t, _, _ in p["steps"]:
            if f == ENTRY:
                continue
            prev = before if before is not None else ENTRY
            by_state.setdefault(f, []).append((prev, t))
    results = []
    for state, pairs in sorted(by_state.items()):
        if len(pairs) < min_row:
            continue
        prevs = sorted({a for a, _ in pairs})
        nexts = sorted({b for _, b in pairs})
        table = np.array([[sum(1 for a, b in pairs if a == pv and b == nx) for nx in nexts]
                          for pv in prevs])
        keep = table.sum(axis=1) >= 5                       # previous wheres with >= 5 departures
        table_k = table[keep]
        p = np.nan
        if chi2_contingency is not None and table_k.shape[0] >= 2 and table_k.shape[1] >= 2:
            cols = table_k.sum(axis=0) > 0
            if cols.sum() >= 2:
                p = float(chi2_contingency(table_k[:, cols])[1])
        results.append(dict(state=state, n=len(pairs), prevs=prevs, nexts=nexts, table=table,
                            kept=[pv for pv, k in zip(prevs, keep) if k], p=p))
    return results


def simulate(paths, n_sim, rng):
    """
    Semi-Markov run of the chain: from Entry, the next state from the observed
    transition counts; the free time before it and the length of each stay drawn
    from the observed ones of that step / state (falling back to the where of the
    state, then to all). Returns (n_stays array, transit frames array).
    """
    counts = transition_counts(paths)
    nxt = {f: (list(d), np.array(list(d.values()), dtype=float) / sum(d.values()))
           for f, d in counts.items()}
    gaps, gaps_w, gaps_all = {}, {}, []
    stay_len, stay_len_w, stay_all = {}, {}, []
    for p in paths:
        for f, fw, _, t, tw, gap, flen in p["steps"]:
            gaps.setdefault((f, t), []).append(gap)
            gaps_w.setdefault((fw, tw), []).append(gap)
            gaps_all.append(gap)
            if f != ENTRY and flen is not None:
                stay_len.setdefault(f, []).append(flen)
                stay_len_w.setdefault(fw, []).append(flen)
                stay_all.append(flen)

    def draw(table, key, fallback, fkey, pooled):
        v = table.get(key)
        if not v or len(v) < 5:
            v = fallback.get(fkey) or pooled
        return v[rng.integers(len(v))]

    n_st = np.zeros(n_sim, dtype=int)
    total = np.zeros(n_sim, dtype=float)
    for k in range(n_sim):
        state, w, t_acc, n = ENTRY, ENTRY, 0.0, 0
        for _ in range(10000):
            if state != ENTRY:
                t_acc += draw(stay_len, state, stay_len_w, w, stay_all)
            if state not in nxt:
                break
            targets, probs = nxt[state]
            new = targets[rng.choice(len(targets), p=probs)]
            new_w = OUT if new == OUT else new.split("|")[0]
            t_acc += draw(gaps, (state, new), gaps_w, (w, new_w), gaps_all)
            if new == OUT:
                break
            state, w, n = new, new_w, n + 1
        n_st[k], total[k] = n, t_acc
    return n_st, total


def ks_gap(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a)
                               - np.searchsorted(b, x, side="right") / len(b))))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

WHERE_COLORS = {"GLU": "#e8766f", "GLN": "#f2a35e", "ASN": "#5aa5d6", "ASP": "#5cbf6a",
                OUT: "#bbbbbb"}


def _color(w, i):
    if w in WHERE_COLORS:
        return WHERE_COLORS[w]
    return plt.get_cmap("tab10")(i % 10)


def plot_next_by_above(table, wheres, bins, title, out_file):
    panels = [w for w in wheres if (table["where"] == w).any()]
    if not panels:
        return
    targets = list(wheres) + [OUT]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5), squeeze=False)
    for ax, w in zip(axes[0], panels):
        sub = table[table["where"] == w].set_index("above").reindex(bins)
        x = np.arange(len(bins))
        bottom = np.zeros(len(bins))
        for i, tgt in enumerate(targets):
            vals = sub[tgt].fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, color=_color(tgt, i), edgecolor="white",
                   hatch="//" if tgt == OUT else None, label=tgt)
            bottom += vals
        for xi, n in zip(x, sub["n"].to_numpy()):
            if np.isfinite(n):
                ax.text(xi, 101, f"n={int(n)}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_ylim(0, 110)
        ax.set_xlabel("K+ above the ion when it leaves")
        ax.set_title(f"leaving {w}")
    axes[0, 0].set_ylabel("% of departures")
    axes[0, -1].legend(title="next", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_out_vs_above(curves, bins, title, out_file):
    """curves: {where: (real p, real sd, control p, n)}."""
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(bins))
    for i, (w, (p, sd, pc, n)) in enumerate(curves.items()):
        c = _color(w, i)
        ok = np.asarray(n) >= 5
        ax.errorbar(x[ok], 100 * np.asarray(p)[ok], yerr=200 * np.asarray(sd)[ok], marker="o",
                    color=c, capsize=4, label=f"{w} real (+/- 2 SD)")
        ax.plot(x[ok], 100 * np.asarray(pc)[ok], linestyle="--", marker="x", color=c,
                label=f"{w} control")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("K+ above the ion when it leaves the stay")
    ax.set_ylabel("% going out next")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_prediction(obs_n, sim_n, obs_t, sim_t, dt, title, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    hi = int(max(obs_n.max(initial=0), np.percentile(sim_n, 99.5) if len(sim_n) else 0)) + 1
    bins = np.arange(0, hi + 1)
    ax1.bar(bins - 0.2, np.bincount(np.minimum(obs_n, hi), minlength=hi + 1) / max(len(obs_n), 1),
            width=0.4, color="#555555", label=f"real (n = {len(obs_n)})")
    ax1.bar(bins + 0.2, np.bincount(np.minimum(sim_n, hi), minlength=hi + 1) / max(len(sim_n), 1),
            width=0.4, color="#1c5cab", label="chain")
    ax1.set_xlabel("stays before exit")
    ax1.set_ylabel("fraction of events")
    ax1.legend()
    for arr, c, lab in ((obs_t, "black", "real"), (sim_t, "#1c5cab", "chain")):
        x = np.sort(np.maximum(arr, 1) * dt)
        ax2.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, label=lab)
    ax2.set_xscale("log")
    ax2.set_xlabel("transit time end_1 -> end_2 (ns)")
    ax2.set_ylabel("fraction of events done by t")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_stay_times(stay_frames, dt, title, out_file):
    keys = [k for k, v in stay_frames.items() if len(v)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        d = np.sort(np.asarray(stay_frames[k], dtype=float))
        surv = 1.0 - np.arange(len(d)) / len(d)
        ax.step(d * dt, surv, where="post")
        lam = 1.0 / d.mean()
        ax.plot(d * dt, np.exp(-lam * (d - d.min())), color="grey", linestyle="--",
                label="exponential, same mean")
        ax.set_yscale("log")
        ax.set_ylim(0.5 / len(d), 1.5)
        ax.set_xlabel("stay length (ns)")
        ax.set_title(f"{k} (n = {len(d)})", fontsize=9)
    axes[0, 0].set_ylabel("P(stay >= t)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis of a stays table (one run, or pooled runs)
# ─────────────────────────────────────────────────────────────────────────────

class _Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")


def analyze_chain(stays, events, out_dir, head, frame_duration, level="ring", max_bin=4,
                  n_boot=200, n_sim=N_SIM, seed=0):
    """Tables, tests and plots of one stays table; see the module docstring."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(frame_duration)
    rng = np.random.default_rng(seed)
    rep = _Report()
    bins = [str(k) for k in range(max_bin)] + [f"{max_bin}+"]
    where_col = "resname" if level == "ring" else "residue_pdb"
    if len(stays):
        order = stays.drop_duplicates(where_col)
        wheres = sorted(order[where_col].tolist(),
                        key=lambda w: (0 if str(w).startswith(("GLU", "GLN")) else 1, str(w)))
        if level == "residue":
            glu_res = set(stays.loc[stays["ring"] == "GLU", "residue_pdb"])
            wheres = sorted(wheres, key=lambda w: (0 if w in glu_res else 1, w))
    else:
        wheres = []
    runs = sorted(events["run"].astype(str).unique())

    paths = event_paths(stays, events, level, max_bin)
    rep("=" * 78)
    rep(f"STAY-TO-STAY CHAIN  {head}   (states: {level} | K+ above when leaving, bins {', '.join(bins)})")
    rep("=" * 78)
    rep(f"runs: {len(runs)}; permeation events: {len(events)}; stays: {len(stays)}; "
        f"events with no stay: {int((events['n_stays'] == 0).sum())}")
    if len(events):
        rep(f"stays per event: mean {events['n_stays'].mean():.2f}, median "
            f"{events['n_stays'].median():.0f}, max {events['n_stays'].max()}")

    # ── table: P(next | where, K+ above) ──
    table = next_by_above(paths, wheres, bins)
    table.to_csv(out_dir / "next_by_above.csv", index=False, float_format="%.2f")
    rep("")
    rep("P(NEXT | where, K+ above when leaving)   % of departures (n = departures)")
    targets = list(wheres) + [OUT]
    rep(f"    {'state':<16}{'n':>6}" + "".join(f"{t:>10}" for t in targets))
    for row in table.itertuples(index=False):
        d = row._asdict()
        rep(f"    {d['where'] + '|' + d['above']:<16}{d['n']:>6}"
            + "".join(f"{d[t]:>9.0f}%" for t in targets)
            + ("   *" if d["n"] < MIN_ROW else ""))
    entry_next = [t for p in paths for f, _, _, _, t, _, _ in p["steps"] if f == ENTRY]
    if entry_next:
        rep(f"    {'Entry':<16}{len(entry_next):>6}"
            + "".join(f"{100 * np.mean([x == t for x in entry_next]):>9.0f}%" for t in targets))
    rep(f"    (* fewer than {MIN_ROW} departures)")

    if len(runs) > 1:
        rep("")
        rep("P(OUT next) PER RUN: mean +/- SEM over runs (runs with >= 3 departures in the state)")
        for w in wheres:
            line = f"    {w:<14}"
            for b in bins:
                per = []
                for r in runs:
                    pr = [p for p in paths if str(p["run"]) == r]
                    po, n = p_out_by_above(pr, w, [b])[0]
                    if n >= 3:
                        per.append(po)
                line += (f"  {b}: {100 * np.mean(per):4.0f} +/- {100 * np.std(per, ddof=1) / np.sqrt(len(per)):3.0f}%"
                         f" ({len(per)} runs)" if len(per) >= 2 else f"  {b}:     -")
            rep(line)

    # ── control ──
    rep("")
    rep("P(OUT next) AGAINST K+ ABOVE: real (+/- SD, bootstrap over events) and control "
        "(other ions from frames 1/4, 1/2, 3/4 of the run later)")
    ctrl_paths = []
    for j in range(1, len(CONTROL_SHIFTS) + 1):
        col = f"n_above_c{j}"
        if col in stays:
            ctrl_paths += event_paths(stays, events, level, max_bin, above_col=col)
    curves = {}
    for w in wheres:
        real = p_out_by_above(paths, w, bins)
        if sum(n for _, n in real) < MIN_ROW:
            continue
        sd = bootstrap_p_out(paths, w, bins, n_boot, rng)
        ctrl = p_out_by_above(ctrl_paths, w, bins) if ctrl_paths else [(np.nan, 0)] * len(bins)
        curves[w] = ([p for p, _ in real], sd, [p for p, _ in ctrl], [n for _, n in real])
        rep(f"  {w}")
        for b, (p, n), s, (pc, nc) in zip(bins, real, sd, ctrl):
            if n == 0:
                continue
            rep(f"    {b:>3} above: real {100 * p:5.1f}% +/- {100 * s:4.1f} (n={n:>4})   "
                f"control {100 * pc:5.1f}% (n={nc:>4})" + ("   *" if n < MIN_ROW else ""))

    # ── 1 history ──
    rep("")
    rep("MARKOV TEST 1: DOES THE NEXT STEP DEPEND ON WHERE THE ION CAME FROM?")
    rep(f"  states with >= {MIN_ROW} departures; chi-square of previous where x next "
        "(previous wheres with >= 5 departures); p < 0.05 = memory")
    hist = history_test(paths)
    hist_rows = []
    for h in hist:
        rep(f"  {h['state']}  (n={h['n']}, p = {h['p']:.3g})" if np.isfinite(h["p"])
            else f"  {h['state']}  (n={h['n']}, not testable)")
        rep("      " + f"{FROM_NEXT:<14}" + "".join(f"{x:>9}" for x in h["nexts"]))
        for pv, row in zip(h["prevs"], h["table"]):
            tot = row.sum()
            rep(f"      {pv:<14}" + "".join(f"{100 * v / tot:>8.0f}%" for v in row) + f"   (n={tot})")
            hist_rows.append(dict(state=h["state"], previous=pv, n=int(tot), p_value=h["p"],
                                  **{f"next_{x}": int(v) for x, v in zip(h["nexts"], row)}))
    tested = [h for h in hist if np.isfinite(h["p"])]
    if tested:
        n_sig = sum(h["p"] < 0.05 for h in tested)
        rep(f"  {n_sig} of {len(tested)} testable states have p < 0.05 "
            f"(about {0.05 * len(tested):.1f} expected by chance)")
    pd.DataFrame(hist_rows).to_csv(out_dir / "history_test.csv", index=False)

    # ── 2 stay times ──
    rep("")
    rep("MARKOV TEST 2: STAY LENGTHS (exponential = memoryless; see stay_times.png)")
    stay_frames = {w: (stays.loc[stays[where_col] == w, "end_frame"]
                       - stays.loc[stays[where_col] == w, "start_frame"] + 1).to_numpy()
                   for w in wheres} if len(stays) else {}
    for w, d in stay_frames.items():
        if len(d):
            rep(f"    {w:<14} n={len(d):>5}  mean {d.mean() * dt:7.3f} ns  median "
                f"{np.median(d) * dt:7.3f} ns  SD/mean {d.std() / d.mean():.2f} "
                "(1 for an exponential)")
    plot_stay_times(stay_frames, dt, f"{head}: stay lengths", out_dir / "stay_times.png")

    # ── 3 prediction ──
    rep("")
    rep(f"MARKOV TEST 3: THE CHAIN RUN FORWARD ({n_sim} simulated ions) AGAINST THE REAL EVENTS")
    obs_n = events["n_stays"].to_numpy()
    obs_t = (events["end_2"] - events["end_1"]).to_numpy()
    sim_n, sim_t = simulate(paths, n_sim, rng)
    rep(f"    stays per event:    real mean {obs_n.mean():.2f} median {np.median(obs_n):.0f}   "
        f"chain mean {sim_n.mean():.2f} median {np.median(sim_n):.0f}   KS {ks_gap(obs_n, sim_n):.2f}")
    rep(f"    no stay at all:     real {100 * np.mean(obs_n == 0):.1f}%   chain {100 * np.mean(sim_n == 0):.1f}%")
    rep(f"    exactly one stay:   real {100 * np.mean(obs_n == 1):.1f}%   chain {100 * np.mean(sim_n == 1):.1f}%")
    rep(f"    transit time:       real median {np.median(obs_t) * dt:.2f} ns mean {obs_t.mean() * dt:.2f} ns   "
        f"chain median {np.median(sim_t) * dt:.2f} ns mean {sim_t.mean() * dt:.2f} ns   "
        f"KS {ks_gap(obs_t, sim_t):.2f} (n = {len(obs_t)}; KS ~ {1.36 / np.sqrt(max(len(obs_t), 1)):.2f} "
        "is the 5% level)")
    plot_prediction(obs_n, sim_n, obs_t, sim_t, dt, f"{head}: chain against real events",
                    out_dir / "prediction.png")

    # ── transitions CSV and plots ──
    counts = transition_counts(paths)
    pd.DataFrame([dict(from_state=f, to_state=t, count=c) for f, d in counts.items()
                  for t, c in d.items()]).to_csv(out_dir / "transitions.csv", index=False)
    plot_next_by_above(table, wheres, bins, f"{head}: next stay by K+ above when leaving",
                       out_dir / "next_by_above.png")
    plot_out_vs_above(curves, bins, f"{head}: P(out next) against K+ above",
                      out_dir / "out_vs_above.png")
    rep.write(out_dir / "summary.txt")
    print(f"\nstay_chain: results in {out_dir}")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# One run
# ─────────────────────────────────────────────────────────────────────────────

def run_stay_chain(universe, permeation_events, asn_residues, glu_residues, channel1, channel2,
                   results_dir, run, channel_type="G12", frame_duration=0.075, cutoff=3.5,
                   min_duration=None, levels=LEVELS, max_bin=4, analysis_start_frame=0,
                   force_rerun=False):
    """Collect the stays of one run (cached in CSVs) and analyse them; see the module docstring."""
    min_duration = default_min_duration(channel_type) if min_duration is None else int(min_duration)
    out_dir = Path(results_dir) / OUT_FOLDER / settings_folder(cutoff, min_duration)
    out_dir.mkdir(parents=True, exist_ok=True)
    stays_csv, events_csv = out_dir / f"stays_run{run}.csv", out_dir / f"events_run{run}.csv"
    if stays_csv.exists() and events_csv.exists() and not force_rerun:
        print(f"stay_chain: using {stays_csv}")
        stays, events = pd.read_csv(stays_csv), pd.read_csv(events_csv)
    else:
        stays, events = collect_stays(universe, permeation_events, asn_residues, glu_residues,
                                      channel1, channel2, run, channel_type, cutoff, min_duration,
                                      analysis_start_frame)
        stays.to_csv(stays_csv, index=False)
        events.to_csv(events_csv, index=False)
    for level in levels:
        analyze_chain(stays, events, out_dir / level, f"{channel_type} RUN{run}", frame_duration,
                      level=level, max_bin=max_bin)
    return stays, events


def main(argv=None):
    import MDAnalysis as mda
    from analysis.channels import Channel
    from analysis.markov_check import G12_CONFIG, _int_list

    p = argparse.ArgumentParser(description="Stay-to-stay chain of one run.")
    p.add_argument("--channel_type", default="G12")
    p.add_argument("--run", type=int, default=3)
    p.add_argument("--top")
    p.add_argument("--traj")
    p.add_argument("--results_dir", help="the RUN dir of main.py (holds analyzer_state.pkl)")
    for name in ("upper1", "lower1", "upper2", "lower2", "glu_residues", "asn_residues"):
        p.add_argument(f"--{name}", type=_int_list)
    p.add_argument("--frame_duration", type=float)
    p.add_argument("--cutoff", type=float, default=3.5)
    p.add_argument("--min_duration", type=int, help="frames (default 10 for G12, 2 for G2)")
    p.add_argument("--max_bin", type=int, default=4, help="K+ above binned 0 .. max_bin-1, max_bin+")
    p.add_argument("--force_rerun", action="store_true", help="recollect the stays")
    args = p.parse_args(argv)

    keys = ("top", "traj", "results_dir", "upper1", "lower1", "upper2", "lower2",
            "glu_residues", "asn_residues", "frame_duration")
    cfg = {}
    if args.channel_type == "G12" and args.run <= 20:
        cfg = {k: (v.format(run=args.run) if isinstance(v, str) else v) for k, v in G12_CONFIG.items()}
    cfg.update({k: getattr(args, k) for k in keys if getattr(args, k) is not None})
    missing = [k for k in keys if k not in cfg]
    if missing:
        sys.exit(f"stay_chain: no built-in settings for {args.channel_type} RUN{args.run}; "
                 f"pass --{' --'.join(missing)}")
    results_dir = Path(cfg["results_dir"])
    state_file = results_dir / "analyzer_state.pkl"
    if not state_file.exists():
        sys.exit(f"stay_chain: {state_file} not found; run main.py for this run first")
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    u = mda.Universe(cfg["top"], cfg["traj"])
    ch1 = Channel(u, cfg["upper1"], cfg["lower1"], num=1, radius=2.5)
    ch2 = Channel(u, cfg["upper2"], cfg["lower2"], num=2, radius=15.0)
    run_stay_chain(u, state["permeation_events"], cfg["asn_residues"], cfg["glu_residues"],
                   ch1, ch2, results_dir, args.run, channel_type=args.channel_type,
                   frame_duration=cfg["frame_duration"], cutoff=args.cutoff,
                   min_duration=args.min_duration, max_bin=args.max_bin,
                   force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
