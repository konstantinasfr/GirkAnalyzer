#!/usr/bin/env python3
"""
hbc_state_windows.py
====================
Split a run into HBC-OPEN and HBC-CLOSED time windows, so that any later
analysis (Phe chi1/chi2, Calpha distances, PIP contacts, water, ...) can be
done separately for the two states.

Reads   <run_dir>/hbc_distances.csv      frame, mean, A_C, B_D
Writes  <run_dir>/hbc_states/
            hbc_states.csv           one row per frame: frame, distance,
                                     smoothed, state (open / closed / transition)
            hbc_state_segments.csv   one row per window: state, start_frame,
                                     end_frame (inclusive), n_frames, median and
                                     IQR of the raw distance, % frames > 5.69 A
            hbc_states.json          the settings used + the windows as
                                     {"open": [[start, end], ...], "closed": [...]}
            hbc_states.png           the trace with the windows shaded

Other scripts read the windows with
    from hbc_state_windows import load_state_frames
    open_frames   = load_state_frames(run_dir, "open")     # np.array of frames
    closed_frames = load_state_frames(run_dir, "closed")

WHICH DISTANCE
--------------
--column B_D (default), A_C, mean, or min (= min(A_C, B_D), the narrowest
diagonal). By the repo convention GIRK1 is in B_D (see swap_gate_columns.py /
main.py --subunit_check_only). The distance is the minimum atom-atom distance
between the two HBC Phe residues of the diagonal.

HOW THE WINDOWS ARE FOUND (automatic mode)
------------------------------------------
1. Smooth: centred rolling median over --smooth frames (default 50). The
   median ignores single-frame spikes and has no edge artefact at the start
   or end of the run.
2. Two thresholds (hysteresis): a closed stretch becomes open only when the
   smoothed distance rises ABOVE --open_above (default 4.5 A), and an open
   stretch becomes closed only when it falls BELOW --close_below (default
   4.0 A). Using two thresholds stops the state flickering when the distance
   hovers near one value.
3. Minimum dwell: any window shorter than --min_dwell frames (default 100)
   is absorbed into its neighbours (shortest first). A short excursion is not
   a state.
4. Transition buffer: --buffer frames (default 25) on each side of every
   switch are labelled "transition" and belong to neither state, so the open
   and closed sets are not polluted by half-open frames.

These thresholds are NOT the 5.69 A conduction criterion (Li et al. 2019).
They separate two conformational states: in E141Q the B_D diagonal sits at
~3 A when closed and ~5 A when "open", i.e. the open state is only partly
open by the 5.69 A criterion. The segment table reports % frames > 5.69 A per
window so both views are visible. Always check the plot.

MANUAL MODE
-----------
--manual "open:2780-3750;closed:0-2700,3800-6799"
defines the windows yourself (inclusive frame ranges). Frames not listed are
"excluded". Use it when the automatic split is not what you see, or to fix
exact windows for a paper. The settings, including the manual string, are
stored in hbc_states.json.

USAGE
-----
    # automatic, one or more runs
    python3 hbc_state_windows.py .../G12_E141Q/RUN3 .../G12_E141Q/RUN5

    # mark the permeation frame(s) on the plot
    python3 hbc_state_windows.py .../G12_E141Q/RUN3 --perm_frames 3180

    # the narrowest diagonal, stricter thresholds
    python3 hbc_state_windows.py .../RUN3 --column min --open_above 4.8 --close_below 4.2

    # manual windows (one run at a time)
    python3 hbc_state_windows.py .../RUN3 --manual "open:2780-3750;closed:0-2700,3800-6799"

A path to an hbc_distances.csv works too; the output folder is then made
next to the CSV.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDUCTION_THRESHOLD = 5.69   # A, Li et al. 2019 J Biol Chem, PMID 31659119
STATES = ("open", "closed")
OUT_DIRNAME = "hbc_states"


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def get_distance(df, column):
    if column == "min":
        return df[["A_C", "B_D"]].min(axis=1).to_numpy(float)
    return df[column].to_numpy(float)


def smooth(x, window):
    if window <= 1:
        return np.asarray(x, float).copy()
    return (pd.Series(x).rolling(window, center=True, min_periods=1)
            .median().to_numpy())


def hysteresis(s, open_above, close_below):
    """Boolean array, True = open."""
    if close_below > open_above:
        raise ValueError("--close_below must be <= --open_above")
    is_open = np.zeros(len(s), bool)
    state = s[0] >= open_above
    for i, v in enumerate(s):
        if state and v < close_below:
            state = False
        elif not state and v > open_above:
            state = True
        is_open[i] = state
    return is_open


def runs_of(labels):
    """[(label, start, end_inclusive), ...] for consecutive equal labels."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i - 1))
            start = i
    return out


def apply_min_dwell(is_open, min_dwell):
    """Absorb windows shorter than min_dwell into their neighbours, shortest
    first. The first and last window of the run are kept even if short only
    when they are the only window."""
    is_open = is_open.copy()
    while True:
        segs = runs_of(is_open)
        if len(segs) <= 1:
            return is_open
        lengths = [e - s + 1 for _, s, e in segs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_dwell:
            return is_open
        _, s, e = segs[k]
        is_open[s:e + 1] = not is_open[s]


def label_states(is_open, buffer):
    labels = np.where(is_open, "open", "closed").astype(object)
    if buffer > 0:
        switches = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
        for sw in switches:
            labels[max(0, sw - buffer):min(len(labels), sw + buffer)] = "transition"
    return labels


def parse_manual(text, n):
    """'open:2780-3750;closed:0-2700,3800-6799' -> labels array (index space)."""
    labels = np.full(n, "excluded", dtype=object)
    for block in text.split(";"):
        block = block.strip()
        if not block:
            continue
        state, ranges = block.split(":", 1)
        state = state.strip().lower()
        if state not in STATES:
            raise ValueError(f"manual state must be open or closed, got {state!r}")
        for r in ranges.split(","):
            a, b = (int(v) for v in r.strip().split("-"))
            if a > b or a < 0 or b >= n:
                raise ValueError(f"bad range {r!r} for a run of {n} frames")
            if np.any(labels[a:b + 1] != "excluded"):
                raise ValueError(f"range {r!r} overlaps another manual window")
            labels[a:b + 1] = state
    return labels


def segment_table(frames, dist, labels):
    rows = []
    for state, s, e in runs_of(list(labels)):
        if state not in STATES:
            continue
        d = dist[s:e + 1]
        rows.append(dict(state=state,
                         start_frame=int(frames[s]), end_frame=int(frames[e]),
                         n_frames=int(e - s + 1),
                         median_dist=round(float(np.median(d)), 3),
                         q25_dist=round(float(np.percentile(d, 25)), 3),
                         q75_dist=round(float(np.percentile(d, 75)), 3),
                         pct_above_5p69=round(100 * float(np.mean(d > CONDUCTION_THRESHOLD)), 1)))
    return pd.DataFrame(rows, columns=["state", "start_frame", "end_frame", "n_frames",
                                       "median_dist", "q25_dist", "q75_dist",
                                       "pct_above_5p69"])


def find_states(df, column="B_D", smooth_window=50, open_above=4.5, close_below=4.0,
                min_dwell=100, buffer=25, manual=None):
    """Returns (per-frame DataFrame, segment DataFrame)."""
    frames = df["frame"].to_numpy(int)
    dist = get_distance(df, column)
    s = smooth(dist, smooth_window)
    if manual:
        # manual ranges are frame numbers; map them to row indices
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError("manual mode needs consecutive frame numbers in the CSV")
        labels = parse_manual(manual, frames[0] + len(frames))[frames]
    else:
        is_open = apply_min_dwell(hysteresis(s, open_above, close_below), min_dwell)
        labels = label_states(is_open, buffer)
    per_frame = pd.DataFrame(dict(frame=frames, distance=dist, smoothed=s,
                                  state=labels))
    return per_frame, segment_table(frames, dist, labels)


# --------------------------------------------------------------------------
# reader for other scripts
# --------------------------------------------------------------------------
def load_state_frames(path, state):
    """Frames of `state` ('open' or 'closed') for a run.
    path: the run dir, its hbc_states/ folder, or hbc_states.csv."""
    p = Path(path)
    if p.is_dir():
        p = p / "hbc_states.csv" if (p / "hbc_states.csv").exists() \
            else p / OUT_DIRNAME / "hbc_states.csv"
    df = pd.read_csv(p)
    return df.loc[df["state"] == state, "frame"].to_numpy(int)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def plot(per_frame, segs, column, args, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"open": "#2a9d8f", "closed": "#e76f51"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for _, r in segs.iterrows():
        ax.axvspan(r.start_frame, r.end_frame + 1, color=colors[r.state], alpha=0.15, lw=0)
    ax.plot(per_frame.frame, per_frame.distance, color="0.6", lw=0.4, label=f"{column} (raw)")
    ax.plot(per_frame.frame, per_frame.smoothed, color="black", lw=1.4,
            label=f"rolling median ({args.smooth} frames)")
    if not args.manual:
        ax.axhline(args.open_above, color=colors["open"], ls="--", lw=1,
                   label=f"open above {args.open_above} Å")
        ax.axhline(args.close_below, color=colors["closed"], ls="--", lw=1,
                   label=f"close below {args.close_below} Å")
    ax.axhline(CONDUCTION_THRESHOLD, color="0.3", ls=":", lw=1, label="5.69 Å (Li 2019)")
    for i, f in enumerate(args.perm_frames or []):
        ax.axvline(f, color="purple", lw=1.2, label="permeation" if i == 0 else None)
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=colors[s], alpha=0.3) for s in STATES]
    labels += ["open window", "closed window"]
    ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"HBC {column} distance (Å)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def process(path, args):
    path = Path(path)
    csv = path / "hbc_distances.csv" if path.is_dir() else path
    out_dir = csv.parent / OUT_DIRNAME
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    per_frame, segs = find_states(df, args.column, args.smooth, args.open_above,
                                  args.close_below, args.min_dwell, args.buffer,
                                  args.manual)
    per_frame.to_csv(out_dir / "hbc_states.csv", index=False,
                     float_format="%.4f")
    segs.to_csv(out_dir / "hbc_state_segments.csv", index=False)
    settings = {k: getattr(args, k) for k in
                ("column", "smooth", "open_above", "close_below", "min_dwell",
                 "buffer", "manual", "perm_frames")}
    settings["mode"] = "manual" if args.manual else "automatic"
    settings["source_csv"] = str(csv.resolve())
    windows = {s: segs.loc[segs.state == s, ["start_frame", "end_frame"]].values.tolist()
               for s in STATES}
    (out_dir / "hbc_states.json").write_text(
        json.dumps(dict(settings=settings, windows=windows), indent=2))
    plot(per_frame, segs, args.column, args, out_dir / "hbc_states.png",
         f"HBC states — {csv.parent.name}")

    counts = per_frame.state.value_counts()
    print(f"\n{csv}")
    print("  frames: " + ", ".join(f"{s} {counts.get(s, 0)}" for s in
                                   ("open", "closed", "transition", "excluded")
                                   if counts.get(s, 0)))
    print(segs.to_string(index=False))
    for f in args.perm_frames or []:
        st = per_frame.loc[per_frame.frame == f, "state"]
        print(f"  permeation frame {f}: {st.iloc[0] if len(st) else 'not in CSV'}")
    print(f"  -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="RUN dirs or hbc_distances.csv files")
    ap.add_argument("--column", default="B_D", choices=["B_D", "A_C", "mean", "min"])
    ap.add_argument("--smooth", type=int, default=50, help="rolling-median window (frames)")
    ap.add_argument("--open_above", type=float, default=4.5)
    ap.add_argument("--close_below", type=float, default=4.0)
    ap.add_argument("--min_dwell", type=int, default=100, help="shortest window kept (frames)")
    ap.add_argument("--buffer", type=int, default=25,
                    help="frames each side of a switch left out of both states")
    ap.add_argument("--manual", default=None,
                    help='e.g. "open:2780-3750;closed:0-2700,3800-6799"')
    ap.add_argument("--perm_frames", type=int, nargs="*", default=None,
                    help="permeation frame(s) to mark on the plot")
    args = ap.parse_args()
    if args.manual and len(args.paths) > 1:
        ap.error("--manual windows are per run: give one run at a time")
    for p in args.paths:
        process(p, args)


if __name__ == "__main__":
    main()
