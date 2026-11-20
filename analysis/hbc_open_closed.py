"""
analysis/hbc_open_closed.py
===========================
HBC open vs closed analysis of ONE run, called from main.py:

    python3 main.py --channel_type G12_E141Q --run_number 3 --hbc_open_closed_only

main.py loads the trajectory and the residue numbers of the system as usual,
calls run_hbc_open_closed(...) and stops. Results go to
    <results_dir>/hbc_open_closed/

STEPS (each step writes its own subfolder)
------------------------------------------
step1_windows/   which frames is the HBC open, which closed      <- done
(step 2, Phe chi1/chi2 and Calpha-Calpha per state, comes next)


STEP 1 -- OPEN / CLOSED WINDOWS
===============================
Distance used: B_D column of hbc_distances.csv (the diagonal with GIRK1 by
the repo convention), i.e. the minimum atom-atom distance between the two
HBC Phe residues of that diagonal. If hbc_distances.csv is not in the
results folder, it is computed here from the trajectory with the same
definition (analysis/ion_analysis.py, compute_constriction_point_diameters).

How the frames are labelled:
1. Smooth: centred rolling median over 50 frames (ignores single-frame
   spikes, no artefact at the start/end of the run).
2. Two thresholds: a closed stretch becomes OPEN when the smoothed distance
   rises above 4.5 A; an open stretch becomes CLOSED when it falls below
   4.0 A. Two thresholds stop the label flickering near one value.
3. Minimum length: a window shorter than 100 frames is merged into its
   neighbours (a short excursion is not a state).
4. Transition: 25 frames on each side of every switch are labelled
   "transition" and belong to neither state.

4.5 / 4.0 A separate the two conformations seen in E141Q (closed ~3 A,
open ~5 A). They are NOT the 5.69 A conduction criterion (Li et al. 2019);
the window table reports % frames > 5.69 A for each window.

Manual windows instead of automatic (frames not listed = "excluded"):
    python3 main.py ... --hbc_open_closed_only \
        --hbc_windows "open:2780-3750;closed:0-2700,3800-6799"

Output, step1_windows/:
    hbc_states.csv           frame, distance, smoothed, state
                             (open / closed / transition / excluded)
    hbc_state_segments.csv   one row per window: state, start_frame,
                             end_frame (inclusive), n_frames, median and IQR
                             of the distance, % frames > 5.69 A
    hbc_states.json          settings used + {"open": [[start, end], ...],
                             "closed": [...]}
    hbc_states.png           distance vs frame, windows shaded, permeation
                             frames (end_3 of each event, from
                             analyzer_state.pkl if present) as purple lines

Later steps / other scripts get the frames with
    from analysis.hbc_open_closed import load_state_frames
    open_frames = load_state_frames(results_dir, "open")
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CONDUCTION_THRESHOLD = 5.69   # A, Li et al. 2019 J Biol Chem, PMID 31659119
STATES = ("open", "closed")
OUT_DIRNAME = "hbc_open_closed"
STEP1_DIRNAME = "step1_windows"

DEFAULTS = dict(column="B_D", smooth=50, open_above=4.5, close_below=4.0,
                min_dwell=100, buffer=25)


# --------------------------------------------------------------------------
# main entry point (called from main.py)
# --------------------------------------------------------------------------
def run_hbc_open_closed(u, hbc_residues, hbc_diagonal_pairs, results_dir,
                        manual_windows=None, **settings):
    """Run all steps for one run. `settings` may override DEFAULTS."""
    cfg = {**DEFAULTS, **settings}
    out_dir = Path(results_dir) / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nHBC OPEN / CLOSED ANALYSIS -> {out_dir}\n{'=' * 60}")

    df = load_or_compute_hbc_distances(u, hbc_residues, hbc_diagonal_pairs,
                                       results_dir, out_dir)
    perm_frames = load_permeation_frames(results_dir)
    step1_windows(df, out_dir / STEP1_DIRNAME, cfg, manual_windows, perm_frames,
                  title=f"HBC open / closed -- {Path(results_dir).name}")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_or_compute_hbc_distances(u, hbc_residues, hbc_diagonal_pairs,
                                  results_dir, out_dir):
    csv = Path(results_dir) / "hbc_distances.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if len(df) == len(u.trajectory):
            print(f"  using {csv}")
            return df
        print(f"  {csv} has {len(df)} frames, trajectory has "
              f"{len(u.trajectory)} -> recomputing")
    print("  computing HBC distances from the trajectory ...")
    atoms = {r: u.select_atoms(f"resid {r}") for r in hbc_residues}
    rows = []
    for ts in u.trajectory:
        d = []
        for r1, r2 in hbc_diagonal_pairs:
            p1, p2 = atoms[r1].positions, atoms[r2].positions
            d.append(float(np.min(np.linalg.norm(p1[:, None] - p2[None], axis=2))))
        rows.append(dict(frame=int(ts.frame), mean=float(np.mean(d)),
                         A_C=d[0], B_D=d[1]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hbc_distances_recomputed.csv", index=False)
    return df


def load_permeation_frames(results_dir):
    """end_3 of every permeation event (the frame the ion leaves the HBC
    region, as in hbc_pairs_distances.png), or [] if not available."""
    state_file = Path(results_dir) / "analyzer_state.pkl"
    if not state_file.is_file():
        print("  analyzer_state.pkl not found -> no permeation lines on the plot")
        return []
    try:
        with open(state_file, "rb") as f:
            events = pickle.load(f).get("permeation_events") or []
        return sorted(int(e["end_3"]) for e in events)
    except Exception as exc:   # plot without lines rather than fail
        print(f"  could not read permeation events ({exc}) -> no permeation lines")
        return []


# --------------------------------------------------------------------------
# step 1: windows
# --------------------------------------------------------------------------
def step1_windows(df, out_dir, cfg, manual_windows=None, perm_frames=(), title=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame, segs = find_states(df, manual=manual_windows, **cfg)

    per_frame.to_csv(out_dir / "hbc_states.csv", index=False, float_format="%.4f")
    segs.to_csv(out_dir / "hbc_state_segments.csv", index=False)
    windows = {s: segs.loc[segs.state == s, ["start_frame", "end_frame"]].values.tolist()
               for s in STATES}
    settings = dict(cfg, mode="manual" if manual_windows else "automatic",
                    manual_windows=manual_windows, permeation_frames=list(perm_frames))
    (out_dir / "hbc_states.json").write_text(
        json.dumps(dict(settings=settings, windows=windows), indent=2))
    plot_states(per_frame, segs, cfg, manual_windows, perm_frames,
                out_dir / "hbc_states.png", title)

    counts = per_frame.state.value_counts()
    print("  step 1 frames: " + ", ".join(
        f"{s} {counts[s]}" for s in ("open", "closed", "transition", "excluded")
        if s in counts))
    print(segs.to_string(index=False))
    for f in perm_frames:
        st = per_frame.loc[per_frame.frame == f, "state"]
        print(f"  permeation at frame {f}: {st.iloc[0] if len(st) else '?'}")
    print(f"  -> {out_dir}")
    return per_frame, segs


def find_states(df, column="B_D", smooth=50, open_above=4.5, close_below=4.0,
                min_dwell=100, buffer=25, manual=None):
    """Returns (per-frame DataFrame, window DataFrame)."""
    frames = df["frame"].to_numpy(int)
    dist = (df[["A_C", "B_D"]].min(axis=1) if column == "min" else df[column]).to_numpy(float)
    s = smooth_median(dist, smooth)
    if manual:
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError("manual windows need consecutive frame numbers")
        labels = parse_manual(manual, frames[0] + len(frames))[frames]
    else:
        is_open = apply_min_dwell(hysteresis(s, open_above, close_below), min_dwell)
        labels = label_states(is_open, buffer)
    per_frame = pd.DataFrame(dict(frame=frames, distance=dist, smoothed=s, state=labels))
    return per_frame, segment_table(frames, dist, labels)


def smooth_median(x, window):
    if window <= 1:
        return np.asarray(x, float).copy()
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()


def hysteresis(s, open_above, close_below):
    """Boolean array, True = open."""
    if close_below > open_above:
        raise ValueError("close_below must be <= open_above")
    is_open = np.zeros(len(s), bool)
    state = s[0] > open_above
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
    """Merge windows shorter than min_dwell into their neighbours, shortest first."""
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
    for sw in np.flatnonzero(is_open[1:] != is_open[:-1]) + 1:
        labels[max(0, sw - buffer):sw + buffer] = "transition"
    return labels


def parse_manual(text, n):
    """'open:2780-3750;closed:0-2700,3800-6799' -> label per frame 0..n-1."""
    labels = np.full(n, "excluded", dtype=object)
    for block in filter(None, (b.strip() for b in text.split(";"))):
        state, ranges = (t.strip() for t in block.split(":", 1))
        state = state.lower()
        if state not in STATES:
            raise ValueError(f"window state must be open or closed, got {state!r}")
        for r in ranges.split(","):
            a, b = (int(v) for v in r.strip().split("-"))
            if a > b or a < 0 or b >= n:
                raise ValueError(f"bad range {r!r} for a run of {n} frames")
            if np.any(labels[a:b + 1] != "excluded"):
                raise ValueError(f"range {r!r} overlaps another window")
            labels[a:b + 1] = state
    return labels


def segment_table(frames, dist, labels):
    rows = []
    for state, s, e in runs_of(list(labels)):
        if state not in STATES:
            continue
        d = dist[s:e + 1]
        rows.append(dict(state=state, start_frame=int(frames[s]), end_frame=int(frames[e]),
                         n_frames=int(e - s + 1),
                         median_dist=round(float(np.median(d)), 3),
                         q25_dist=round(float(np.percentile(d, 25)), 3),
                         q75_dist=round(float(np.percentile(d, 75)), 3),
                         pct_above_5p69=round(100 * float(np.mean(d > CONDUCTION_THRESHOLD)), 1)))
    return pd.DataFrame(rows, columns=["state", "start_frame", "end_frame", "n_frames",
                                       "median_dist", "q25_dist", "q75_dist",
                                       "pct_above_5p69"])


def plot_states(per_frame, segs, cfg, manual, perm_frames, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"open": "#2a9d8f", "closed": "#e76f51"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for _, r in segs.iterrows():
        ax.axvspan(r.start_frame, r.end_frame + 1, color=colors[r.state], alpha=0.15, lw=0)
    ax.plot(per_frame.frame, per_frame.distance, color="0.6", lw=0.4,
            label=f"{cfg['column']} (raw)")
    ax.plot(per_frame.frame, per_frame.smoothed, color="black", lw=1.4,
            label=f"rolling median ({cfg['smooth']} frames)")
    if not manual:
        ax.axhline(cfg["open_above"], color=colors["open"], ls="--", lw=1,
                   label=f"open above {cfg['open_above']} Å")
        ax.axhline(cfg["close_below"], color=colors["closed"], ls="--", lw=1,
                   label=f"close below {cfg['close_below']} Å")
    ax.axhline(CONDUCTION_THRESHOLD, color="0.3", ls=":", lw=1, label="5.69 Å (Li 2019)")
    for i, f in enumerate(perm_frames):
        ax.axvline(f, color="purple", lw=1.2, label="permeation" if i == 0 else None)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=colors[s], alpha=0.3) for s in STATES]
    labels += ["open window", "closed window"]
    ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"HBC {cfg['column']} distance (Å)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# reader for later steps / other scripts
# --------------------------------------------------------------------------
def load_state_frames(results_dir, state):
    """Frames labelled `state` ('open' or 'closed') in step 1 of this run."""
    p = Path(results_dir) / OUT_DIRNAME / STEP1_DIRNAME / "hbc_states.csv"
    df = pd.read_csv(p)
    return df.loc[df["state"] == state, "frame"].to_numpy(int)
