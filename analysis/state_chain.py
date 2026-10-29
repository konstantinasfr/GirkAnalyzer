"""
state_chain.py
==============
Frame-by-frame state chain of permeation through the GLU/ASN cavity: every
permeating K+ is followed through every frame from the moment it leaves the SF
(end_1) to its exit (end_2), and each frame gets a STATE

    region | K+ above | K+ below          e.g.  ASN|2a1b

The ion stays in a state until any of the three changes - the ion moves to the
other region, or an ion arrives / leaves above or below it - and then jumps to
the next state; after end_2 it jumps to Exit. All ions' state sequences are
pooled into one continuous-time chain (counts of jumps i -> j, time spent in
each state), from which the kinetics follow.

REGION of a frame (from the stays of stay_chain.py: >= min_duration frames
closest to one GLU / ASN-ASP residue within `cutoff`)
    region_rule='last'    the ring of the last stay of this event (the ion keeps
                          its region until it stays somewhere else - core sets,
                          so a moment off the residue is not a move); before the
                          first stay: 'free'
    region_rule='strict'  the ring of the stay covering the frame, else 'free'
    level='region'        GLU (the GLU ring; GLN in E141Q) or ASN (the lower
                          ring: N184 x3 + D173 in G12)
    level='residue'       the residue itself (e.g. 173.G1)
K+ ABOVE / BELOW: the other K+ in the cavity (region 2, main.py's ch2 cylinder)
further toward / away from the SF along the ch2 axis than the ion (the n_above /
n_below of stay_chain.py and touch_to_next_stay.py); include_sf=True adds the
K+ in the SF to 'above'. Binned 0 .. max_above-1, max_above+ and 0 ..
max_below-1, max_below+ (default above 0,1,2,3+, below 0,1,2+). A count change
counts only once it has held for min_hold frames (shorter blips are given the
value before them), so an ion sitting level with ours does not make a jump
every frame.

WHAT IS COMPUTED (all ions of all events pooled)
------------------------------------------------
time in regions   per ion: time spent free / in GLU / in ASN, number of region
                  changes, the region path (e.g. free>ASN>GLU>ASN>Exit); the
                  commonest paths. No model needed.
states            per state: visits, total time, mean / median time per visit,
                  P(next is Exit / a move / a change of K+ above / below), and
                  the EXIT RATE k_exit = exits from the state / time spent in it
                  (per ns, bootstrap SD over events).
control           the same exit rates with the other ions taken from the frame
                  half a run later (the ion itself at the real frame): keeps what
                  the ion's own position does to the counts, removes any real
                  effect of the other ions. Real above control = the ions matter.
transitions       counts, P(j | i) and rates i -> j (per ns), with the kind of
                  jump (exit / move / above+ / above- / below+ / below- / counts).
mean time to exit from each state: measured (end_2 - start of the visit,
                  averaged over the visits of the state) and predicted by the
                  continuous-time Markov chain (rates above).

IS IT MARKOVIAN?
1 memory          for states with enough departures: does the kind of the next
                  jump depend on the kind of jump that led into the state
                  (start / move / above+ ...)? Chi-square; Markov: no.
2 dwell times     per state, SD / mean of the visit times (1 for an exponential).
3 prediction      the chain run forward from the observed first states, as a
                  continuous-time Markov chain (exponential visits) and as a
                  semi-Markov chain (visit times drawn from the observed ones of
                  the state), against the real events: transit time (KS gap),
                  time per region, region changes per ion. The mean-time-to-exit
                  comparison above is the same test per state.

OUTPUTS   <results_dir>/state_chain/<settings>/        (settings e.g. t3.5_d10)
    frames_run<run>.csv    one row per frame of every event: site of the stay
                           covering it (empty if none), ring, K+ above / below /
                           in the SF, and the same counts for the control
    events_run<run>.csv    one row per permeation event
    <analysis>/            analysis = <level>_<region_rule>_h<min_hold>[_sf]
        summary.txt, states.csv, transitions.csv, region_time.csv (per event),
        paths.csv, memory_test.csv, exit_rates.png, region_time.png, time_to_exit.png, prediction.png
Over runs: aggregate_state_chain.py (repo root).

USAGE - from main.py (after ch1, ch2 and analyzer exist)
    from analysis.state_chain import run_state_chain
    run_state_chain(u, analyzer.permeation_events, asn_residues, glu_residues,
                    channel1=ch1, channel2=ch2, results_dir=results_dir,
                    run=run_number, channel_type=args.channel_type,
                    frame_duration=frame_duration)
CLI (G12 runs 1-20 built in):
    python3 -m analysis.state_chain --channel_type G12 --run 3
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.stay_chain import (PoreFrames, collect_stays, default_min_duration, settings_folder,
                                 ks_gap, _Report, OUT_FOLDER as STAY_FOLDER)

OUT_FOLDER = "state_chain"
EXIT, FREE = "Exit", "free"
CONTROL_SHIFT = 0.5
MIN_ROW = 20
MIN_TIME_NS = 5.0                 # heat-map cells with less time in the state are greyed out
N_SIM = 20000
IN_OUT = "in \\ out"
REGION_COLORS = {FREE: "#bbbbbb", "GLU": "#e8766f", "ASN": "#5aa5d6"}
FRAME_COLUMNS = ["run", "event", "ion_id", "frame", "site", "ring", "n_above", "n_below", "n_sf",
                 "n_above_c", "n_below_c", "n_sf_c"]


def default_min_hold(channel_type):
    """Frames a K+ count must hold: 1 for G2 (0.4 ns frames), 4 for the G12 systems."""
    return 1 if str(channel_type).upper().startswith("G2") else 4


def analysis_folder(level, region_rule, min_hold, include_sf):
    return f"{level}_{region_rule}_h{min_hold}" + ("_sf" if include_sf else "")


# ─────────────────────────────────────────────────────────────────────────────
# Collecting the frames (needs the trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def collect_frames(universe, events, stays, channel1, channel2):
    """One row per frame end_1+1 .. end_2 of every event; see the module docstring."""
    pore = PoreFrames(universe, channel1, channel2)
    n_frames = len(universe.trajectory)
    shift = int(CONTROL_SHIFT * n_frames)
    wanted = set()
    for ev in events.itertuples(index=False):
        for f in range(ev.end_1 + 1, ev.end_2 + 1):
            wanted.update((f, (f + shift) % n_frames))
    for f in sorted(wanted):                 # read the trajectory once, in order
        pore.get(f)

    by_event = {k: g for k, g in stays.groupby(["run", "event"])} if len(stays) else {}
    rows = []
    for ev in events.itertuples(index=False):
        site = {}
        g = by_event.get((ev.run, ev.event))
        if g is not None:
            for s in g.to_dict("records"):
                for f in range(int(s["start_frame"]), int(s["end_frame"]) + 1):
                    site[f] = (s["residue_pdb"], s["ring"])
        for f in range(ev.end_1 + 1, ev.end_2 + 1):
            a, b, sf = pore.neighbours(ev.ion_id, f)
            ac, bc, sfc = pore.neighbours(ev.ion_id, f, (f + shift) % n_frames)
            res, ring = site.get(f, ("", ""))
            rows.append((ev.run, ev.event, ev.ion_id, f, res, ring, a, b, sf, ac, bc, sfc))
    return pd.DataFrame(rows, columns=FRAME_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Frames -> state sequences -> visits
# ─────────────────────────────────────────────────────────────────────────────

def _runs(values):
    """Run-length encoding: list of [value, length]."""
    out = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def debounce(values, min_hold):
    """Runs shorter than min_hold take the value before them (the first one: the value after it)."""
    if min_hold <= 1 or len(values) == 0:
        return list(values)
    out = []
    for v, n in _runs(values):
        if out and (n < min_hold or v == out[-1][0]):
            out[-1][1] += n
        else:
            out.append([v, n])
    if len(out) > 1 and out[0][1] < min_hold:
        out[1][1] += out[0][1]
        out.pop(0)
    return [v for v, n in out for _ in range(n)]


def _bin(n, top):
    return min(int(n), top)


def bin_text(n, top):
    return f"{top}+" if n >= top else str(n)


def state_name(region, a, b, max_above, max_below):
    return f"{region}|{bin_text(a, max_above)}a{bin_text(b, max_below)}b"


def regions_of(g, level, region_rule):
    """Region label per frame of one event (g sorted by frame)."""
    lab = []
    last = FREE
    for site, ring in zip(g["site"], g["ring"]):
        here = None
        if isinstance(site, str) and site:
            here = site if level == "residue" else ("GLU" if ring == "GLU" else "ASN")
        if region_rule == "last":
            last = here or last
            lab.append(last)
        else:
            lab.append(here or FREE)
    return lab


def build_visits(frames, events, level="region", region_rule="last", max_above=3, max_below=2,
                 min_hold=4, include_sf=False, control=False):
    """
    (visits DataFrame, region labels per event). One visit = consecutive frames in
    one state; columns run, event, state, region, above, below, start, length,
    next, kind (of the jump out), prev_kind (of the jump in), to_exit (frames from
    the start of the visit to the exit).
    """
    a_col, b_col, sf_col = ("n_above_c", "n_below_c", "n_sf_c") if control else ("n_above", "n_below", "n_sf")
    by_event = {k: g.sort_values("frame") for k, g in frames.groupby(["run", "event"])}
    rows, region_frames = [], {}
    for ev in events.itertuples(index=False):
        g = by_event.get((ev.run, ev.event))
        if g is None or not len(g):
            continue
        reg = regions_of(g, level, region_rule)
        above = g[a_col].to_numpy() + (g[sf_col].to_numpy() if include_sf else 0)
        above = debounce([_bin(x, max_above) for x in above], min_hold)
        below = debounce([_bin(x, max_below) for x in g[b_col].to_numpy()], min_hold)
        region_frames[(ev.run, ev.event)] = reg
        f0 = int(g["frame"].iloc[0])
        visits = []
        for (r, a, b), n in _runs(list(zip(reg, above, below))):
            start = f0 + sum(v["length"] for v in visits)
            visits.append(dict(run=ev.run, event=ev.event, region=r, above=a, below=b,
                               state=state_name(r, a, b, max_above, max_below), start=start,
                               length=n, to_exit=int(ev.end_2) - start + 1))
        prev_kind = "start"
        for i, v in enumerate(visits):
            nxt = visits[i + 1] if i + 1 < len(visits) else None
            v["next"] = nxt["state"] if nxt else EXIT
            v["kind"] = jump_kind(v, nxt)
            v["prev_kind"] = prev_kind
            prev_kind = v["kind"]
        rows += visits
    cols = ["run", "event", "state", "region", "above", "below", "start", "length", "to_exit",
            "next", "kind", "prev_kind"]
    return pd.DataFrame(rows, columns=cols), region_frames


def jump_kind(v, nxt):
    if nxt is None:
        return "exit"
    if nxt["region"] != v["region"]:
        return "move"
    da, db = nxt["above"] - v["above"], nxt["below"] - v["below"]
    if da and db:
        return "counts"
    if da:
        return "above+" if da > 0 else "above-"
    return "below+" if db > 0 else "below-"


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def state_table(visits, dt):
    rows = []
    for s, g in visits.groupby("state"):
        L = g["length"].to_numpy()
        T = L.sum() * dt
        kinds = g["kind"]
        n = len(g)
        rows.append(dict(state=s, region=g["region"].iloc[0], above=g["above"].iloc[0],
                         below=g["below"].iloc[0], visits=n, time_ns=T,
                         mean_visit_ns=L.mean() * dt, median_visit_ns=np.median(L) * dt,
                         sd_over_mean=L.std() / L.mean(), exits=int((kinds == "exit").sum()),
                         p_exit=float((kinds == "exit").mean()), p_move=float((kinds == "move").mean()),
                         p_above_change=float(kinds.isin(["above+", "above-"]).mean()),
                         p_below_change=float(kinds.isin(["below+", "below-"]).mean()),
                         k_exit_per_ns=(kinds == "exit").sum() / T,
                         mean_to_exit_ns=g["to_exit"].mean() * dt))
    return pd.DataFrame(rows)


def exit_rate_grid(visits, regions, max_above, max_below, dt, n_boot=0, rng=None):
    """
    {region: (rate[a, b] per ns, sd[a, b], exits[a, b], time_ns[a, b])}, the rate =
    exits / time spent with a K+ above and b K+ below; sd from resampling events.
    """
    key = visits["region"].astype(str) + "|" + visits["above"].astype(str) + "|" + visits["below"].astype(str)
    ev = visits["run"].astype(str) + ":" + visits["event"].astype(str)
    tab = pd.DataFrame(dict(ev=ev, key=key, t=visits["length"] * dt, x=(visits["kind"] == "exit").astype(float)))
    per = tab.groupby(["ev", "key"])[["t", "x"]].sum().unstack(fill_value=0.0)
    T, X = per["t"], per["x"]
    boot_T = boot_X = None
    if n_boot and rng is not None and len(T):
        w = rng.multinomial(len(T), np.full(len(T), 1.0 / len(T)), size=n_boot)   # resampled events
        boot_T, boot_X = w @ T.to_numpy(), w @ X.to_numpy()
    cols = list(T.columns)
    out = {}
    for r in regions:
        rate = np.full((max_above + 1, max_below + 1), np.nan)
        sd, exits, time = rate.copy(), np.zeros_like(rate), np.zeros_like(rate)
        for a in range(max_above + 1):
            for b in range(max_below + 1):
                k = f"{r}|{a}|{b}"
                if k not in cols:
                    continue
                j = cols.index(k)
                t, x = T[k].sum(), X[k].sum()
                exits[a, b], time[a, b] = x, t
                if t > 0:
                    rate[a, b] = x / t
                if boot_T is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        reps = boot_X[:, j] / boot_T[:, j]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        sd[a, b] = np.nanstd(reps)
        out[r] = (rate, sd, exits, time)
    return out


def ctmc_time_to_exit(visits, dt):
    """Mean time to Exit from every state of the continuous-time Markov chain (ns)."""
    states = sorted(visits["state"].unique())
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    T = visits.groupby("state")["length"].sum().reindex(states).to_numpy() * dt
    Q = np.zeros((n, n))
    for (s, t), c in visits.groupby(["state", "next"]).size().items():
        rate = c / T[idx[s]]
        if t != EXIT:
            Q[idx[s], idx[t]] += rate
        Q[idx[s], idx[s]] -= rate
    try:
        m = np.linalg.solve(Q, -np.ones(n))
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(Q, -np.ones(n), rcond=None)[0]
    return dict(zip(states, m))


def simulate(visits, n_sim, rng, exponential):
    """
    The chain run forward from the observed first states: next state from the
    observed jump counts, visit times exponential (exponential=True) or drawn
    from the observed visits of the state. Returns (transit frames, time per
    region dict of arrays (frames), region changes per ion).
    """
    nxt, lengths = {}, {}
    for s, g in visits.groupby("state"):
        c = g["next"].value_counts()
        nxt[s] = (list(c.index), c.to_numpy() / c.sum())
        lengths[s] = g["length"].to_numpy()
    region = dict(zip(visits["state"], visits["region"]))
    first = visits.groupby(["run", "event"])["state"].first().to_numpy()
    regions = sorted(set(region.values()))
    transit = np.zeros(n_sim)
    per_region = {r: np.zeros(n_sim) for r in regions}
    changes = np.zeros(n_sim, dtype=int)
    for k in range(n_sim):
        s = first[rng.integers(len(first))]
        for _ in range(100000):
            L = lengths[s]
            t = rng.exponential(L.mean()) if exponential else L[rng.integers(len(L))]
            transit[k] += t
            per_region[region[s]][k] += t
            targets, probs = nxt[s]
            new = targets[rng.choice(len(targets), p=probs)]
            if new == EXIT:
                break
            changes[k] += region[new] != region[s]
            s = new
    return transit, per_region, changes


def memory_test(visits, min_row=MIN_ROW):
    """Per state: chi-square of the kind of jump in x the kind of jump out."""
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        chi2_contingency = None
    results = []
    for s, g in visits.groupby("state"):
        if len(g) < min_row:
            continue
        table = pd.crosstab(g["prev_kind"], g["kind"])
        table_k = table[table.sum(axis=1) >= 5]
        table_k = table_k.loc[:, table_k.sum(axis=0) > 0]
        p = np.nan
        if chi2_contingency is not None and table_k.shape[0] >= 2 and table_k.shape[1] >= 2:
            p = float(chi2_contingency(table_k.to_numpy())[1])
        results.append(dict(state=s, n=len(g), p=p, table=table))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _rcolor(r, i):
    return REGION_COLORS.get(r, plt.get_cmap("tab10")(i % 10))


def plot_exit_rates(real, ctrl, mean_visit, regions, max_above, max_below, title, out_file):
    """Rows: real exit rate, control exit rate, mean visit time; one column per region."""
    if not regions:
        return
    fig, axes = plt.subplots(3, len(regions), figsize=(3.8 * len(regions) + 1, 11), squeeze=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN panels
        vmax = np.nanmax([np.nanmax(np.where(d[r][3] >= MIN_TIME_NS, d[r][0], np.nan))
                          for d in (real, ctrl) for r in regions] + [1e-3])
    a_lab = [bin_text(a, max_above) for a in range(max_above + 1)]
    b_lab = [bin_text(b, max_below) for b in range(max_below + 1)]
    rows = ((real, "exit rate (/ns)", "Reds"), (ctrl, "control exit rate (/ns)", "Reds"),
            (mean_visit, "mean visit (ns)", "Blues"))
    for i, (data, name, cmap) in enumerate(rows):
        for j, r in enumerate(regions):
            ax = axes[i, j]
            if i < 2:
                rate, sd, exits, time = data[r]
                val = np.where(time >= MIN_TIME_NS, rate, np.nan)
                ax.set_facecolor("#e6e6e6")
                im = ax.imshow(val, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
                for a in range(max_above + 1):
                    for b in range(max_below + 1):
                        if time[a, b] > 0:
                            ax.text(b, a, f"{rate[a, b]:.3f}\n{int(exits[a, b])} exits\n{time[a, b]:.0f} ns",
                                    ha="center", va="center", fontsize=7,
                                    color="black" if time[a, b] >= MIN_TIME_NS else "#888888")
            else:
                val = data[r]
                im = ax.imshow(val, origin="lower", cmap=cmap, aspect="auto")
                for a in range(max_above + 1):
                    for b in range(max_below + 1):
                        if np.isfinite(val[a, b]):
                            ax.text(b, a, f"{val[a, b]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(max_below + 1))
            ax.set_xticklabels(b_lab)
            ax.set_yticks(range(max_above + 1))
            ax.set_yticklabels(a_lab)
            ax.set_xlabel("K+ below")
            if j == 0:
                ax.set_ylabel(f"{name}\nK+ above")
            ax.set_title(f"{r}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{title}\n(grey: less than {MIN_TIME_NS:g} ns in the state)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_region_time(rt, regions, dt, title, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(regions):
        x = np.sort(rt[f"time_{r}_ns"].to_numpy())
        ax1.step(np.maximum(x, dt / 2), 1 - np.arange(len(x)) / len(x), where="post",
                 color=_rcolor(r, i), label=f"{r} (mean {x.mean():.1f} ns)")
    x = np.sort(rt["transit_ns"].to_numpy())
    ax1.step(x, 1 - np.arange(len(x)) / len(x), where="post", color="black", label="whole transit")
    ax1.set_xscale("log")
    ax1.set_xlabel("time per ion (ns)")
    ax1.set_ylabel("fraction of ions spending >= t")
    ax1.legend(fontsize=8)
    paths = rt["path"].value_counts().head(12)[::-1]
    ax2.barh(range(len(paths)), 100 * paths.to_numpy() / len(rt), color="#777777")
    ax2.set_yticks(range(len(paths)))
    ax2.set_yticklabels(paths.index, fontsize=8)
    ax2.set_xlabel("% of ions")
    ax2.set_title("commonest region paths")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_time_to_exit(st, title, out_file):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, (r, g) in enumerate(st.groupby("region")):
        ax.scatter(g["model_to_exit_ns"], g["mean_to_exit_ns"], s=10 + 2 * np.sqrt(g["visits"]) * 5,
                   color=_rcolor(r, i), edgecolor="black", alpha=0.8, label=r)
        for d in g[g["visits"] >= MIN_ROW].to_dict("records"):
            ax.annotate(d["state"].split("|")[1], (d["model_to_exit_ns"], d["mean_to_exit_ns"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(st["model_to_exit_ns"].max(), st["mean_to_exit_ns"].max()) * 1.05]
    ax.plot(lim, lim, color="grey", ls="--", label="equal")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean time to exit, Markov chain (ns)")
    ax.set_ylabel("mean time to exit, measured (ns)")
    ax.legend(fontsize=8, title="region (size: visits)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_prediction(obs_t, sims, obs_region, regions, title, out_file):
    """sims: {name: (transit ns, {region: ns}, changes)}."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"real": "black", "Markov (exponential visits)": "#1c5cab", "semi-Markov": "#d9822b"}
    for name, arr in [("real", obs_t)] + [(k, v[0]) for k, v in sims.items()]:
        x = np.sort(np.maximum(arr, 1e-3))
        ax1.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=colors.get(name), label=name)
    ax1.set_xscale("log")
    ax1.set_xlabel("transit time end_1 -> end_2 (ns)")
    ax1.set_ylabel("fraction of ions out by t")
    ax1.legend(fontsize=8)
    x = np.arange(len(regions))
    width = 0.8 / (1 + len(sims))
    ax2.bar(x, [obs_region[r].mean() for r in regions], width, color="black", label="real")
    for i, (name, (_, per, _)) in enumerate(sims.items(), start=1):
        ax2.bar(x + i * width, [per[r].mean() if r in per else 0 for r in regions], width,
                color=colors.get(name), label=name)
    ax2.set_xticks(x + width * len(sims) / 2)
    ax2.set_xticklabels(regions)
    ax2.set_ylabel("mean time per ion (ns)")
    ax2.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis of a frames table (one run, or pooled runs)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_states(frames, events, out_dir, head, frame_duration, level="region", region_rule="last",
                   max_above=3, max_below=2, min_hold=4, include_sf=False, n_boot=200, n_sim=N_SIM,
                   seed=0):
    """Tables, tests and plots of one frames table; see the module docstring."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(frame_duration)
    rng = np.random.default_rng(seed)
    rep = _Report()
    events = events[events["end_2"] > events["end_1"]]
    visits, region_frames = build_visits(frames, events, level, region_rule, max_above, max_below,
                                         min_hold, include_sf)
    if not len(visits):
        rep(f"state_chain: no frames for {head}")
        rep.write(out_dir / "summary.txt")
        return None
    ctrl_visits, _ = build_visits(frames, events, level, region_rule, max_above, max_below, min_hold,
                                  include_sf, control=True)
    regions = sorted(visits["region"].unique(), key=lambda r: (r != FREE, r != "GLU", r))
    above_txt = "K+ above (cavity" + (" + SF)" if include_sf else ")")

    rep("=" * 78)
    rep(f"STATE CHAIN  {head}   states: {level} | {above_txt} | K+ below, frame by frame")
    rep(f"  region rule '{region_rule}', K+ above binned 0..{max_above}+, below 0..{max_below}+, "
        f"count changes kept if they hold >= {min_hold} frames ({min_hold * dt:.2f} ns)")
    rep("=" * 78)
    rep(f"runs: {events['run'].nunique()}; permeation events: {len(events)}; frames: {len(frames)}; "
        f"visits: {len(visits)} in {visits['state'].nunique()} states")
    kinds = visits["kind"].value_counts()
    rep("jumps: " + ", ".join(f"{k} {int(kinds.get(k, 0))}" for k in
                               ("move", "above+", "above-", "below+", "below-", "counts", "exit")))

    # ── time in regions (no model) ──
    rt_rows = []
    for ev in events.itertuples(index=False):
        reg = region_frames.get((ev.run, ev.event))
        if reg is None:
            continue
        row = dict(run=ev.run, event=ev.event, ion_id=ev.ion_id, transit_ns=len(reg) * dt)
        for r in regions:
            row[f"time_{r}_ns"] = sum(x == r for x in reg) * dt
        path = [v for v, _ in _runs(reg)]
        row["region_changes"] = len(path) - 1
        row["path"] = ">".join(path + [EXIT])
        rt_rows.append(row)
    rt = pd.DataFrame(rt_rows)
    rt.to_csv(out_dir / "region_time.csv", index=False, float_format="%.3f")
    rep("")
    rep("TIME PER ION IN EACH REGION (ns)")
    rep(f"    {'':<10}{'mean':>9}{'median':>9}{'% of ions':>11}{'% of time':>11}")
    total = rt["transit_ns"].sum()
    for r in regions:
        x = rt[f"time_{r}_ns"]
        rep(f"    {r:<10}{x.mean():>9.2f}{x.median():>9.2f}{100 * (x > 0).mean():>10.0f}%"
            f"{100 * x.sum() / total:>10.0f}%")
    rep(f"    {'transit':<10}{rt['transit_ns'].mean():>9.2f}{rt['transit_ns'].median():>9.2f}")
    rep(f"    region changes per ion: mean {rt['region_changes'].mean():.2f}, median "
        f"{rt['region_changes'].median():.0f}")
    paths = rt["path"].value_counts()
    pd.DataFrame(dict(path=paths.index, n=paths.to_numpy(),
                      percent=100 * paths.to_numpy() / len(rt))).to_csv(out_dir / "paths.csv", index=False,
                                                                       float_format="%.1f")
    rep("  commonest region paths:")
    for p_, n in paths.head(10).items():
        rep(f"    {100 * n / len(rt):5.1f}%  ({n:>3})  {p_}")

    # ── states ──
    st = state_table(visits, dt)
    model = ctmc_time_to_exit(visits, dt)
    st["model_to_exit_ns"] = st["state"].map(model)
    st["order"] = st["region"].map({r: i for i, r in enumerate(regions)})
    st = st.sort_values(["order", "above", "below"]).drop(columns="order")
    st.to_csv(out_dir / "states.csv", index=False, float_format="%.4f")
    rep("")
    rep("STATES   (visit = consecutive frames in the state; P(next ...) over the visits; "
        "k_exit = exits / time in state)")
    rep(f"    {'state':<14}{'visits':>7}{'time ns':>9}{'mean ns':>9}{'P exit':>8}{'P move':>8}"
        f"{'P above':>9}{'P below':>9}{'k_exit/ns':>11}{'to exit':>9}{'model':>8}")
    for d in st.to_dict("records"):
        rep(f"    {d['state']:<14}{d['visits']:>7}{d['time_ns']:>9.1f}{d['mean_visit_ns']:>9.2f}"
            f"{100 * d['p_exit']:>7.0f}%{100 * d['p_move']:>7.0f}%{100 * d['p_above_change']:>8.0f}%"
            f"{100 * d['p_below_change']:>8.0f}%{d['k_exit_per_ns']:>11.4f}{d['mean_to_exit_ns']:>9.1f}"
            f"{d['model_to_exit_ns']:>8.1f}" + ("   *" if d["visits"] < MIN_ROW else ""))
    rep(f"    (P above / below: next jump is a change of the K+ above / below; 'to exit': mean time "
        f"from the start of a visit to the exit, measured and from the Markov chain; * fewer than "
        f"{MIN_ROW} visits)")

    # ── exit rates by K+ above x below, real and control ──
    real = exit_rate_grid(visits, regions, max_above, max_below, dt, n_boot, rng)
    ctrl = exit_rate_grid(ctrl_visits, regions, max_above, max_below, dt)
    rep("")
    rep("EXIT RATE (/ns) BY K+ ABOVE x K+ BELOW: real +/- SD (bootstrap over events) and control "
        "(other ions from the frame half a run later)")
    for r in regions:
        rep(f"  {r}")
        rate, sd, exits, time = real[r]
        crate, _, cexits, ctime = ctrl[r]
        for a in range(max_above + 1):
            for b in range(max_below + 1):
                if time[a, b] <= 0 and ctime[a, b] <= 0:
                    continue
                rep(f"    {bin_text(a, max_above):>2} above {bin_text(b, max_below):>2} below:  real "
                    f"{rate[a, b]:7.4f} +/- {sd[a, b]:6.4f} ({int(exits[a, b]):>3} exits / {time[a, b]:7.1f} ns)"
                    f"   control {crate[a, b]:7.4f} ({int(cexits[a, b]):>3} / {ctime[a, b]:7.1f} ns)")
        # marginal over below / above: does each matter on its own?
        for name, axis, top in (("above", 1, max_above), ("below", 0, max_below)):
            line = f"    by K+ {name}:"
            for k in range(top + 1):
                e = exits.sum(axis=axis)[k]
                t = time.sum(axis=axis)[k]
                ce, ct = cexits.sum(axis=axis)[k], ctime.sum(axis=axis)[k]
                if t > 0:
                    line += (f"  {bin_text(k, top)}: {e / t:.4f}"
                             + (f" (ctrl {ce / ct:.4f})" if ct > 0 else ""))
            rep(line)
    mean_visit = {}
    for r in regions:
        g = np.full((max_above + 1, max_below + 1), np.nan)
        sub = st[st["region"] == r]
        for d in sub.to_dict("records"):
            g[d["above"], d["below"]] = d["mean_visit_ns"]
        mean_visit[r] = g
    plot_exit_rates(real, ctrl, mean_visit, regions, max_above, max_below,
                    f"{head}: exit rate and visit time by K+ above x below", out_dir / "exit_rates.png")

    # ── transitions ──
    T = visits.groupby("state")["length"].sum() * dt
    n_vis = visits.groupby("state").size()
    tr = visits.groupby(["state", "next", "kind"]).size().reset_index(name="count")
    tr["p"] = tr["count"] / tr["state"].map(n_vis)
    tr["rate_per_ns"] = tr["count"] / tr["state"].map(T)
    tr.rename(columns={"state": "from_state", "next": "to_state"}).to_csv(
        out_dir / "transitions.csv", index=False, float_format="%.5f")

    # ── 1 memory ──
    rep("")
    rep("MARKOV TEST 1: DOES THE NEXT JUMP DEPEND ON THE JUMP THAT LED INTO THE STATE?")
    rep(f"  states with >= {MIN_ROW} visits; chi-square of kind in x kind out (rows with >= 5 visits); "
        "p < 0.05 = memory")
    rep("  (tables shown for the states with p < 0.05; all of them in memory_test.csv)")
    mem = memory_test(visits)
    mem_rows = []
    for m in mem:
        rep(f"  {m['state']}  (n={m['n']}, p = {m['p']:.3g})" if np.isfinite(m["p"])
            else f"  {m['state']}  (n={m['n']}, not testable)")
        t = m["table"]
        for idx, row in t.iterrows():
            mem_rows.append(dict(state=m["state"], p_value=m["p"], jump_in=idx, n=int(row.sum()),
                                 **{f"out_{c}": int(v) for c, v in row.items()}))
        if not (np.isfinite(m["p"]) and m["p"] < 0.05):
            continue
        rep("      " + f"{IN_OUT:<10}" + "".join(f"{c:>8}" for c in t.columns))
        for idx, row in t.iterrows():
            tot = row.sum()
            rep(f"      {idx:<10}" + "".join(f"{100 * v / tot:>7.0f}%" for v in row) + f"   (n={tot})")
    pd.DataFrame(mem_rows).to_csv(out_dir / "memory_test.csv", index=False)
    tested = [m for m in mem if np.isfinite(m["p"])]
    if tested:
        rep(f"  {sum(m['p'] < 0.05 for m in tested)} of {len(tested)} testable states have p < 0.05 "
            f"(about {0.05 * len(tested):.1f} expected by chance)")

    # ── 2 dwell times ──
    rep("")
    rep("MARKOV TEST 2: VISIT TIMES (SD/mean = 1 for an exponential, i.e. memoryless)")
    big = st[st["visits"] >= MIN_ROW]
    if len(big):
        rep(f"    states with >= {MIN_ROW} visits: SD/mean median {big['sd_over_mean'].median():.2f} "
            f"(range {big['sd_over_mean'].min():.2f} - {big['sd_over_mean'].max():.2f})")

    # ── 3 prediction ──
    rep("")
    rep(f"MARKOV TEST 3: THE CHAIN RUN FORWARD ({n_sim} simulated ions) AGAINST THE REAL EVENTS")
    obs_t = rt["transit_ns"].to_numpy()
    obs_region = {r: rt[f"time_{r}_ns"].to_numpy() for r in regions}
    sims = {}
    for name, expo in (("Markov (exponential visits)", True), ("semi-Markov", False)):
        tt, per, ch = simulate(visits, n_sim, rng, expo)
        sims[name] = (tt * dt, {r: v * dt for r, v in per.items()}, ch)
    rep(f"    {'':<30}{'transit median':>15}{'mean':>8}{'KS':>6}" + "".join(f"{r + ' mean':>{len(r) + 7}}" for r in regions)
        + f"{'changes':>9}")
    rep(f"    {'real':<30}{np.median(obs_t):>15.2f}{obs_t.mean():>8.2f}{'':>6}"
        + "".join(f"{obs_region[r].mean():>{len(r) + 7}.2f}" for r in regions) + f"{rt['region_changes'].mean():>9.2f}")
    for name, (tt, per, ch) in sims.items():
        rep(f"    {name:<30}{np.median(tt):>15.2f}{tt.mean():>8.2f}{ks_gap(obs_t, tt):>6.2f}"
            + "".join(f"{per.get(r, np.zeros(1)).mean():>{len(r) + 7}.2f}" for r in regions) + f"{ch.mean():>9.2f}")
    rep(f"    (times in ns; KS gap of the transit times, ~{1.36 / np.sqrt(len(obs_t)):.2f} is the 5% level "
        f"for {len(obs_t)} events)")
    plot_prediction(obs_t, sims, obs_region, regions, f"{head}: chain against real events",
                    out_dir / "prediction.png")
    plot_region_time(rt, regions, dt, f"{head}: time per ion in each region", out_dir / "region_time.png")
    plot_time_to_exit(st, f"{head}: mean time to exit per state", out_dir / "time_to_exit.png")
    rep.write(out_dir / "summary.txt")
    print(f"\nstate_chain: results in {out_dir}")
    return st


# ─────────────────────────────────────────────────────────────────────────────
# One run
# ─────────────────────────────────────────────────────────────────────────────

def run_state_chain(universe, permeation_events, asn_residues, glu_residues, channel1, channel2,
                    results_dir, run, channel_type="G12", frame_duration=0.075, cutoff=3.5,
                    min_duration=None, level="region", region_rule="last", max_above=3, max_below=2,
                    min_hold=None, include_sf=False, analysis_start_frame=0, force_rerun=False):
    """Collect the frames of one run (cached in CSVs) and analyse them; see the module docstring."""
    min_duration = default_min_duration(channel_type) if min_duration is None else int(min_duration)
    min_hold = default_min_hold(channel_type) if min_hold is None else int(min_hold)
    settings = settings_folder(cutoff, min_duration)
    out_dir = Path(results_dir) / OUT_FOLDER / settings
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_csv, events_csv = out_dir / f"frames_run{run}.csv", out_dir / f"events_run{run}.csv"
    if frames_csv.exists() and events_csv.exists() and not force_rerun:
        print(f"state_chain: using {frames_csv}")
        frames, events = pd.read_csv(frames_csv, keep_default_na=False), pd.read_csv(events_csv)
    else:
        stay_dir = Path(results_dir) / STAY_FOLDER / settings
        s_csv, e_csv = stay_dir / f"stays_run{run}.csv", stay_dir / f"events_run{run}.csv"
        if s_csv.exists() and e_csv.exists() and not force_rerun:
            print(f"state_chain: using the stays of {s_csv}")
            stays, events = pd.read_csv(s_csv), pd.read_csv(e_csv)
        else:
            stays, events = collect_stays(universe, permeation_events, asn_residues, glu_residues,
                                          channel1, channel2, run, channel_type, cutoff, min_duration,
                                          analysis_start_frame)
        frames = collect_frames(universe, events, stays, channel1, channel2)
        frames.to_csv(frames_csv, index=False)
        events.to_csv(events_csv, index=False)
    analyze_states(frames, events, out_dir / analysis_folder(level, region_rule, min_hold, include_sf),
                   f"{channel_type} RUN{run}", frame_duration, level=level, region_rule=region_rule,
                   max_above=max_above, max_below=max_below, min_hold=min_hold, include_sf=include_sf)
    return frames, events


def add_analysis_args(p):
    p.add_argument("--level", default="region", choices=["region", "residue"])
    p.add_argument("--region_rule", default="last", choices=["last", "strict"])
    p.add_argument("--max_above", type=int, default=3, help="K+ above binned 0 .. max_above-1, max_above+")
    p.add_argument("--max_below", type=int, default=2, help="K+ below binned 0 .. max_below-1, max_below+")
    p.add_argument("--min_hold", type=int, help="frames a count change must hold (default 4 for G12, 1 for G2)")
    p.add_argument("--include_sf", action="store_true", help="count the K+ in the SF as above")


def main(argv=None):
    import MDAnalysis as mda
    from analysis.channels import Channel
    from analysis.markov_check import G12_CONFIG, _int_list

    p = argparse.ArgumentParser(description="Frame-by-frame state chain of one run.")
    p.add_argument("--channel_type", default="G12")
    p.add_argument("--run", type=int, default=3)
    p.add_argument("--top")
    p.add_argument("--traj")
    p.add_argument("--results_dir", help="the RUN dir of main.py (holds analyzer_state.pkl)")
    for name in ("upper1", "lower1", "upper2", "lower2", "glu_residues", "asn_residues"):
        p.add_argument(f"--{name}", type=_int_list)
    p.add_argument("--frame_duration", type=float)
    p.add_argument("--cutoff", type=float, default=3.5)
    p.add_argument("--min_duration", type=int, help="frames of a stay (default 10 for G12, 2 for G2)")
    add_analysis_args(p)
    p.add_argument("--force_rerun", action="store_true", help="recollect the stays and frames")
    args = p.parse_args(argv)

    keys = ("top", "traj", "results_dir", "upper1", "lower1", "upper2", "lower2",
            "glu_residues", "asn_residues", "frame_duration")
    cfg = {}
    if args.channel_type == "G12" and args.run <= 20:
        cfg = {k: (v.format(run=args.run) if isinstance(v, str) else v) for k, v in G12_CONFIG.items()}
    cfg.update({k: getattr(args, k) for k in keys if getattr(args, k) is not None})
    missing = [k for k in keys if k not in cfg]
    if missing:
        sys.exit(f"state_chain: no built-in settings for {args.channel_type} RUN{args.run}; "
                 f"pass --{' --'.join(missing)}")
    results_dir = Path(cfg["results_dir"])
    state_file = results_dir / "analyzer_state.pkl"
    if not state_file.exists():
        sys.exit(f"state_chain: {state_file} not found; run main.py for this run first")
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    u = mda.Universe(cfg["top"], cfg["traj"])
    ch1 = Channel(u, cfg["upper1"], cfg["lower1"], num=1, radius=2.5)
    ch2 = Channel(u, cfg["upper2"], cfg["lower2"], num=2, radius=15.0)
    run_state_chain(u, state["permeation_events"], cfg["asn_residues"], cfg["glu_residues"],
                    ch1, ch2, results_dir, args.run, channel_type=args.channel_type,
                    frame_duration=cfg["frame_duration"], cutoff=args.cutoff,
                    min_duration=args.min_duration, level=args.level, region_rule=args.region_rule,
                    max_above=args.max_above, max_below=args.max_below, min_hold=args.min_hold,
                    include_sf=args.include_sf, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
