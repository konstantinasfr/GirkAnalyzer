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
step1_windows/        which frames is the HBC open, which closed
step2_gate_geometry/  what changes at the gate: Phe side chains (chi1,
                      chi2, ring position) and backbone (Calpha-Calpha),
                      open vs closed
step2b_backbone_vs_sidechain/  is the opening a whole-Phe (backbone)
                      motion or a side-chain reorientation? split per Phe,
                      per diagonal, per switch, and which moves first


STEP 1 -- OPEN / CLOSED WINDOWS
===============================
DEFAULT DEFINITION (Li, Jin, Gazgalis, Cui & Logothetis 2019, J Biol Chem,
PMID 31659119): the HBC is open when the gate distance is above 5.69 A.
Distance used: the `mean` column of hbc_distances.csv = the average of the
two diagonals (A_C + B_D) / 2. Each diagonal is the minimum atom-atom
distance between the two HBC Phe residues of that diagonal. If
hbc_distances.csv is not in the results folder, it is computed here from the
trajectory with the same definition (analysis/ion_analysis.py,
compute_constriction_point_diameters).

How the frames are labelled:
1. Smooth: centred rolling median over 50 frames. The raw distance jumps a
   lot from frame to frame (in the first E141Q run checked it never stays
   above 5.69 A for more than 33 frames in a row), so the threshold is
   applied to the smoothed value. No artefact at the start/end of the run.
2. Threshold: the smoothed distance above 5.69 A = OPEN, below = CLOSED.
   (Two different thresholds can be given, open_above > close_below; the
   state then only switches when the distance crosses the far one, which
   stops flickering near a single value. Default: both 5.69 A.)
3. Minimum length: a window shorter than 100 frames is merged into its
   neighbours (a short excursion is not a state).
4. Transition: 25 frames on each side of every switch are labelled
   "transition" and belong to neither state.

The window table also gives the median of each diagonal (A_C, B_D) per
window: the mean can hide one pinched diagonal (e.g. A_C ~7 A and B_D ~3 A
average to ~5 A), and the diagonals show which one actually closes.

Other definitions (main.py options):
    --hbc_column B_D|A_C|mean|min     min = narrowest diagonal in each frame
    --hbc_open_above X --hbc_close_below Y
e.g. the earlier E141Q split on the GIRK1 diagonal alone:
    python3 main.py ... --hbc_open_closed_only --hbc_column B_D \
        --hbc_open_above 4.5 --hbc_close_below 4.0
(in the first E141Q run both definitions give the same open window,
frames ~2810-3700).

Manual windows instead of automatic (frames not listed = "excluded"):
    python3 main.py ... --hbc_open_closed_only \
        --hbc_windows "open:2780-3750;closed:0-2700,3800-6799"

Output, step1_windows/:
    hbc_states.csv           frame, distance, smoothed, state
                             (open / closed / transition / excluded)
    hbc_state_segments.csv   one row per window: state, start_frame,
                             end_frame (inclusive), n_frames, median and IQR
                             of the distance used, % frames > 5.69 A, and
                             the median of A_C and of B_D
    hbc_states.json          settings used + {"open": [[start, end], ...],
                             "closed": [...]}
    hbc_states.png           distance vs frame, windows shaded, permeation
                             frames (end_3 of each event, from
                             analyzer_state.pkl if present) as purple lines

Later steps / other scripts get the frames with
    from analysis.hbc_open_closed import load_state_frames
    open_frames = load_state_frames(results_dir, "open")


STEP 2 -- GATE GEOMETRY, OPEN vs CLOSED
=======================================
Question: when the HBC opens, does a Phe side chain swing out of the pore
(rotamer flip), or do the helices move apart (backbone), or both? And which
subunit does it?

Per frame, for each of the 4 HBC Phe:
    chi1        N-CA-CB-CG dihedral (deg). Phe rotamers: g+ ~ +60,
                t ~ 180, g- ~ -60 (bins: 0..120 g+, 120..240 t, 240..360 g-,
                angles taken 0..360)
    chi2        CA-CB-CG-CD1 (deg), folded to 0..180 because the ring is
                symmetric (a 180 deg ring flip is the same conformation)
    ring_center distance (A) from the Phe ring centre (CG CD1 CD2 CE1 CE2
                CZ) to the pore centre at the gate (= mean position of the
                4 HBC Calpha). Smaller = ring further into the pore.
and per diagonal:
    ca_ca       Calpha-Calpha distance (A) across the pore. If it hardly
                changes between states while the minimum distance does, the
                opening is a side-chain motion; if it grows, the backbone
                (helices) moved apart.

Residues are labelled "<subunit>:F<resid>", e.g. "B:F462", subunit letters
from subunit_groups of the main.py config (as in the subunit check), and
"(GIRK1)" for the subunit whose cavity residue in asn_residues is ASP.
Without subunit_groups the label is the diagonal column (A_C / B_D).

Output, step2_gate_geometry/:
    gate_geometry_per_frame.csv   frame, state, every measure above
    gate_geometry_by_state.csv    per measure and state: n_frames, median,
                                  25-75 % range, mean, sd; and the
                                  open - closed difference of the medians
                                  (chi1/chi2: circular, see below)
    chi1_rotamers_by_state.csv    % of frames in g+ / t / g- per Phe and state
    gate_geometry_timeseries.png  the measures vs frame, windows shaded
    gate_geometry_distributions.png  histograms, open vs closed

No p-values: consecutive frames are strongly correlated and one run has
only one or a few open windows, so frame-level tests would be meaningless.
Compare runs (and WT) once more runs are done. For chi1/chi2 the "median"
is the circular mean direction and the spread the circular sd.


STEP 2b -- BACKBONE vs SIDE CHAIN
=================================
Question: does the gate open/close because the whole Phe moves (its helix
/ backbone moves in or out), or because the side chain only changes its
orientation (points more or less into the pore)?

Pore axis per frame: perpendicular to the best-fit plane of the 4 HBC
Calpha, through the centroid of ALL protein Calpha (a stable reference: the
mean of the 4 HBC Calpha would itself move towards a Phe that moves out).
For each Phe, per frame:
    ca_r      distance of its Calpha from the pore axis       = BACKBONE
    sc_min_r  distance from the axis of its side-chain atom (CB, CG, ring)
              closest to the axis = how far into the pore it reaches
    reach     ca_r - sc_min_r = how far the side chain sticks in beyond
              its own Calpha                                   = SIDE CHAIN
              (constant if the Phe moves as a rigid whole)
so exactly   sc_min_r = ca_r - reach
and for a change:   d(sc_min_r) = d(ca_r)      - d(reach)
                    pore wider  = backbone out + side chain reorients out
For a diagonal (two opposite Phe) the same with sums:
    radial_gap = sc_min_r(1) + sc_min_r(2) = backbone - reach_sum
radial_gap is an axis-based version of the gate distance; its correlation
with the real B_D / A_C distance is reported as a check.

Output, step2b_backbone_vs_sidechain/:
    backbone_sidechain_per_frame.csv  all the above, every frame
    decomposition_by_state.csv  per Phe and per diagonal, open vs closed
                                (means, so the parts add up exactly):
                                d_backbone, d_sidechain, d_total (A,
                                positive = wider when open) and the % of the
                                change from backbone / side chain
    decomposition_by_switch.csv the same for every open<->closed switch
                                separately (opening and closing), window
                                before vs window after, plus t_half_backbone
                                / t_half_sidechain: the frame where each part
                                (50-frame rolling median) has done half its
                                change -> which moves first
    correlation_with_gate.csv   frame by frame, how well the backbone part,
                                the side-chain part and both together follow
                                the real gate distance of each diagonal
                                (Pearson r, R^2 of a linear fit)
    backbone_sidechain_timeseries.png  gate distance, backbone, side-chain
                                reach vs frame per diagonal, windows shaded
    decomposition_by_state.png  bar chart: backbone / side chain / total

Reading it: pct_backbone ~100 -> the Phe move as a whole (helices move);
~0 -> only side-chain orientation; in between -> both. Percentages are
only shown when the total change is >= 0.05 A. The % can go below 0 or
above 100 when the two parts move in opposite directions.
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
STEP2_DIRNAME = "step2_gate_geometry"
STEP2B_DIRNAME = "step2b_backbone_vs_sidechain"
SIDECHAIN_ATOMS = ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")
STATE_COLORS = {"open": "#2a9d8f", "closed": "#e76f51"}
RING_ATOMS = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")

DEFAULTS = dict(column="mean", smooth=50, open_above=CONDUCTION_THRESHOLD,
                close_below=CONDUCTION_THRESHOLD,
                min_dwell=100, buffer=25)


# --------------------------------------------------------------------------
# main entry point (called from main.py)
# --------------------------------------------------------------------------
def run_hbc_open_closed(u, hbc_residues, hbc_diagonal_pairs, results_dir,
                        manual_windows=None, subunit_groups=None, asn_residues=None,
                        **settings):
    """Run all steps for one run. `settings` may override DEFAULTS."""
    cfg = {**DEFAULTS, **settings}
    out_dir = Path(results_dir) / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(results_dir).name
    print(f"\n{'=' * 60}\nHBC OPEN / CLOSED ANALYSIS -> {out_dir}\n{'=' * 60}")

    df = load_or_compute_hbc_distances(u, hbc_residues, hbc_diagonal_pairs,
                                       results_dir, out_dir)
    perm_frames = load_permeation_frames(results_dir)
    per_frame, segs = step1_windows(df, out_dir / STEP1_DIRNAME, cfg, manual_windows,
                                    perm_frames, title=f"HBC open / closed -- {run_name}")
    *_, pos, labels = step2_gate_geometry(
        u, hbc_diagonal_pairs, per_frame, segs, out_dir / STEP2_DIRNAME,
        subunit_groups, asn_residues, perm_frames,
        title=f"HBC gate geometry -- {run_name}")
    step2b_backbone_vs_sidechain(pos, hbc_diagonal_pairs, labels, df, per_frame, segs,
                                 out_dir / STEP2B_DIRNAME, perm_frames,
                                 title=f"HBC backbone vs side chain -- {run_name}")


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


def find_states(df, column="mean", smooth=50, open_above=CONDUCTION_THRESHOLD,
                close_below=CONDUCTION_THRESHOLD, min_dwell=100, buffer=25, manual=None):
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
    diagonals = {c: df[c].to_numpy(float) for c in ("A_C", "B_D") if c in df}
    return per_frame, segment_table(frames, dist, labels, diagonals)


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


def segment_table(frames, dist, labels, diagonals=None):
    """One row per window. `diagonals` {name: per-frame array}: their median
    per window is added as median_<name>."""
    diagonals = diagonals or {}
    rows = []
    for state, s, e in runs_of(list(labels)):
        if state not in STATES:
            continue
        d = dist[s:e + 1]
        row = dict(state=state, start_frame=int(frames[s]), end_frame=int(frames[e]),
                   n_frames=int(e - s + 1),
                   median_dist=round(float(np.median(d)), 3),
                   q25_dist=round(float(np.percentile(d, 25)), 3),
                   q75_dist=round(float(np.percentile(d, 75)), 3),
                   pct_above_5p69=round(100 * float(np.mean(d > CONDUCTION_THRESHOLD)), 1))
        for name, v in diagonals.items():
            row[f"median_{name}"] = round(float(np.median(v[s:e + 1])), 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=["state", "start_frame", "end_frame", "n_frames",
                                       "median_dist", "q25_dist", "q75_dist",
                                       "pct_above_5p69"]
                        + [f"median_{n}" for n in diagonals])


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
    one_threshold = cfg["open_above"] == cfg["close_below"]
    if not manual and one_threshold:
        ax.axhline(cfg["open_above"], color="0.2", ls="--", lw=1.2,
                   label=f"open / closed at {cfg['open_above']} Å")
    elif not manual:
        ax.axhline(cfg["open_above"], color=colors["open"], ls="--", lw=1,
                   label=f"open above {cfg['open_above']} Å")
        ax.axhline(cfg["close_below"], color=colors["closed"], ls="--", lw=1,
                   label=f"close below {cfg['close_below']} Å")
    if manual or not (one_threshold and cfg["open_above"] == CONDUCTION_THRESHOLD):
        ax.axhline(CONDUCTION_THRESHOLD, color="0.3", ls=":", lw=1, label="5.69 Å")
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
# step 2: gate geometry
# --------------------------------------------------------------------------
def dihedral(p0, p1, p2, p3):
    """Dihedral angles (deg, -180..180) for arrays of points, shape (n, 3)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=1, keepdims=True) * b1n
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1n, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def chi1_rotamer(chi1):
    a = np.mod(chi1, 360)
    return np.where(a < 120, "g+", np.where(a < 240, "t", "g-"))


def circ_stats(deg):
    """Circular mean (deg, -180..180) and circular sd (deg)."""
    r = np.radians(deg)
    C, S = np.mean(np.cos(r)), np.mean(np.sin(r))
    R = min(np.hypot(C, S), 1.0)
    sd = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else np.inf
    return float(np.degrees(np.arctan2(S, C))), float(sd)


def residue_labels(u, hbc_diagonal_pairs, subunit_groups=None, asn_residues=None):
    """{resid: label}, e.g. {462: 'B:F462 (GIRK1)'}."""
    def chain(resid):
        return int(u.select_atoms(f"resid {resid}").atoms[0].fragindex)

    try:
        chain(hbc_diagonal_pairs[0][0])
    except Exception:          # no bonds in the topology -> no chains
        print("  no chain information in the topology -> labels use the diagonal column")
        return {r: f"{col}:F{r}" for col, pair in zip(("A_C", "B_D"), hbc_diagonal_pairs)
                for r in pair}

    chain_to_sub = {}
    for letter, resids in (subunit_groups or {}).items():
        for r in resids:
            chain_to_sub[chain(r)] = letter
    girk1_chains = set()
    for r in asn_residues or []:
        if u.select_atoms(f"resid {r}").residues[0].resname in ("ASP", "ASH"):
            girk1_chains.add(chain(r))

    labels = {}
    for col, pair in zip(("A_C", "B_D"), hbc_diagonal_pairs):
        for r in pair:
            c = chain(r)
            lab = f"{chain_to_sub.get(c, col)}:F{r}"
            if c in girk1_chains:
                lab += " (GIRK1)"
            labels[r] = lab
    return labels


def compute_gate_geometry(u, hbc_diagonal_pairs, labels, keep_positions=False):
    """Per-frame chi1, chi2, ring-centre distance per Phe and Calpha-Calpha
    per diagonal. Returns a DataFrame indexed like the trajectory, and with
    keep_positions=True also the raw positions {resid: {"CA": (nf,3),
    "sc": (nf,7,3) side-chain heavy atoms, ...}} for step 2b."""
    resids = [r for pair in hbc_diagonal_pairs for r in pair]
    sel = {}
    for r in resids:
        res = u.select_atoms(f"resid {r}")
        sel[r] = {n: res.select_atoms(f"name {n}") for n in ("N", "CA", "CB", "CG", "CD1")}
        sel[r]["ring"] = res.select_atoms("name " + " ".join(RING_ATOMS))
        sel[r]["sc"] = res.select_atoms("name " + " ".join(SIDECHAIN_ATOMS))
        for n, ag in sel[r].items():
            if len(ag) == 0:
                raise ValueError(f"resid {r}: no atom(s) {n} -- is it a PHE?")
    nf = len(u.trajectory)
    pos = {r: {k: np.empty((nf, 3)) for k in ("N", "CA", "CB", "CG", "CD1", "ring")}
           for r in resids}
    for r in resids:
        pos[r]["sc"] = np.empty((nf, len(sel[r]["sc"]), 3))
    ref = u.select_atoms("protein and name CA")    # pore-centre reference for step 2b
    ref_center = np.empty((nf, 3)) if len(ref) else None
    frames = np.empty(nf, int)
    for i, ts in enumerate(u.trajectory):
        frames[i] = ts.frame
        if ref_center is not None:
            ref_center[i] = ref.positions.mean(axis=0)
        for r in resids:
            for k in ("N", "CA", "CB", "CG", "CD1"):
                pos[r][k][i] = sel[r][k].positions[0]
            pos[r]["ring"][i] = sel[r]["ring"].positions.mean(axis=0)
            pos[r]["sc"][i] = sel[r]["sc"].positions

    center = np.mean([pos[r]["CA"] for r in resids], axis=0)
    out = {"frame": frames}
    for r in resids:
        p, lab = pos[r], labels[r]
        out[f"chi1 {lab}"] = dihedral(p["N"], p["CA"], p["CB"], p["CG"])
        out[f"chi2 {lab}"] = np.mod(dihedral(p["CA"], p["CB"], p["CG"], p["CD1"]), 180)
        out[f"ring_center {lab}"] = np.linalg.norm(p["ring"] - center, axis=1)
    for col, (r1, r2) in zip(("A_C", "B_D"), hbc_diagonal_pairs):
        out[f"ca_ca {col}"] = np.linalg.norm(pos[r1]["CA"] - pos[r2]["CA"], axis=1)
    if keep_positions:
        pos["_ref_center"] = ref_center
        return pd.DataFrame(out), pos
    return pd.DataFrame(out)


def summarize_by_state(geo):
    measures = [c for c in geo.columns if c not in ("frame", "state")]
    rows = []
    for m in measures:
        is_angle = m.startswith(("chi1", "chi2"))
        stats = {}
        for st in STATES:
            v = geo.loc[geo.state == st, m].to_numpy(float)
            if len(v) == 0:
                continue
            if m.startswith("chi1"):
                c, sd = circ_stats(v)
                stats[st] = dict(n_frames=len(v), median=c, q25=np.nan, q75=np.nan,
                                 mean=c, sd=sd)
            elif m.startswith("chi2"):   # 0..180 period: double, average, halve
                c, sd = circ_stats(2 * v)
                stats[st] = dict(n_frames=len(v), median=np.mod(c / 2, 180), q25=np.nan,
                                 q75=np.nan, mean=np.mod(c / 2, 180), sd=sd / 2)
            else:
                stats[st] = dict(n_frames=len(v), median=np.median(v),
                                 q25=np.percentile(v, 25), q75=np.percentile(v, 75),
                                 mean=v.mean(), sd=v.std())
        for st, s in stats.items():
            rows.append(dict(measure=m, state=st, **{k: round(float(x), 3) if k != "n_frames"
                                                      else x for k, x in s.items()}))
        if len(stats) == 2:
            d = stats["open"]["median"] - stats["closed"]["median"]
            if is_angle:
                period = 360 if m.startswith("chi1") else 180
                d = (d + period / 2) % period - period / 2
            rows.append(dict(measure=m, state="open-closed", median=round(float(d), 3)))
    return pd.DataFrame(rows, columns=["measure", "state", "n_frames", "median",
                                       "q25", "q75", "mean", "sd"])


def rotamer_table(geo):
    rows = []
    for m in [c for c in geo.columns if c.startswith("chi1")]:
        for st in STATES:
            v = geo.loc[geo.state == st, m].to_numpy(float)
            if len(v) == 0:
                continue
            rot = chi1_rotamer(v)
            rows.append(dict(residue=m[len("chi1 "):], state=st, n_frames=len(v),
                             **{f"pct_{k}": round(100 * float(np.mean(rot == k)), 1)
                                for k in ("g+", "t", "g-")}))
    return pd.DataFrame(rows)


def step2_gate_geometry(u, hbc_diagonal_pairs, per_frame, segs, out_dir,
                        subunit_groups=None, asn_residues=None, perm_frames=(), title=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  step 2: gate geometry (chi1, chi2, ring position, Calpha-Calpha) ...")
    labels = residue_labels(u, hbc_diagonal_pairs, subunit_groups, asn_residues)
    geo, pos = compute_gate_geometry(u, hbc_diagonal_pairs, labels, keep_positions=True)
    geo = geo.merge(per_frame[["frame", "state"]], on="frame", how="left")
    geo.insert(1, "state", geo.pop("state"))

    geo.to_csv(out_dir / "gate_geometry_per_frame.csv", index=False, float_format="%.3f")
    summary = summarize_by_state(geo)
    summary.to_csv(out_dir / "gate_geometry_by_state.csv", index=False)
    rot = rotamer_table(geo)
    rot.to_csv(out_dir / "chi1_rotamers_by_state.csv", index=False)
    plot_geometry_timeseries(geo, segs, labels, perm_frames,
                             out_dir / "gate_geometry_timeseries.png", title)
    plot_geometry_distributions(geo, out_dir / "gate_geometry_distributions.png", title)

    diff = summary[summary.state == "open-closed"][["measure", "median"]]
    print("  open - closed (median; circular for chi):")
    print(diff.rename(columns={"median": "open-closed"}).to_string(index=False))
    print("  chi1 rotamers (% frames):")
    print(rot.to_string(index=False))
    print(f"  -> {out_dir}")
    return geo, summary, rot, pos, labels


def _shade(ax, segs):
    for _, r in segs.iterrows():
        ax.axvspan(r.start_frame, r.end_frame + 1, color=STATE_COLORS[r.state],
                   alpha=0.15, lw=0)


def plot_geometry_timeseries(geo, segs, labels, perm_frames, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("chi1", "χ1 (deg)"), ("chi2", "χ2 (deg, 0–180)"),
              ("ring_center", "ring – pore centre (Å)"), ("ca_ca", "Cα–Cα (Å)")]
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 10), sharex=True)
    for ax, (key, ylab) in zip(axes, groups):
        _shade(ax, segs)
        cols = [c for c in geo.columns if c.startswith(key + " ")]
        for c in cols:
            y = geo[c].to_numpy(float)
            if key == "chi1":
                y = np.mod(y, 360)   # 0..360 so g+ / t / g- are separate bands
                ax.plot(geo.frame, y, ".", ms=0.6, alpha=0.5, label=c[len(key) + 1:])
            else:
                sm = pd.Series(y).rolling(50, center=True, min_periods=1).median()
                ax.plot(geo.frame, y, lw=0.3, alpha=0.3)
                ax.plot(geo.frame, sm, lw=1.3, color=ax.lines[-1].get_color(),
                        label=c[len(key) + 1:])
        if key == "chi1":
            ax.set_yticks([0, 60, 120, 180, 240, 300, 360])
            for b in (120, 240):
                ax.axhline(b, color="0.5", lw=0.6, ls=":")
            ax.set_ylim(0, 360)
        for f in perm_frames:
            ax.axvline(f, color="purple", lw=1)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, loc="upper right", ncol=4, markerscale=10)
    axes[0].set_title(title + "   (χ1 bands: 0–120 g+, 120–240 t, 240–360 g−)")
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_geometry_distributions(geo, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = [c for c in geo.columns if c not in ("frame", "state")]
    ncol = 4
    nrow = int(np.ceil(len(measures) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    for ax, m in zip(axes.flat, measures):
        vals = {st: geo.loc[geo.state == st, m].to_numpy(float) for st in STATES}
        if m.startswith("chi1"):
            vals = {k: np.mod(v, 360) for k, v in vals.items()}
            bins = np.linspace(0, 360, 61)
        else:
            allv = np.concatenate([v for v in vals.values() if len(v)])
            bins = np.linspace(allv.min(), allv.max(), 50) if len(allv) else 10
        for st in STATES:
            if len(vals[st]):
                ax.hist(vals[st], bins=bins, density=True, alpha=0.5,
                        color=STATE_COLORS[st], label=f"{st} ({len(vals[st])})")
        ax.set_title(m, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(measures):]:
        ax.axis("off")
    fig.suptitle(title + "  — open vs closed (density)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# step 2b: backbone vs side chain
# --------------------------------------------------------------------------
def radial_frame(ca_stack, ref_center=None):
    """Pore axis per frame. Direction = normal of the best-fit plane of the 4
    HBC Calpha, shape (nf, 4, 3). Point on the axis = ref_center (centroid of
    all protein Calpha) if given, else the mean of the 4 HBC Calpha. The 4-Calpha
    mean moves towards a Phe that moves out (by 1/4 of its motion), which
    would split one subunit's motion over all four; the whole-protein
    centroid hardly moves. Diagonal sums are not affected either way."""
    hbc_center = ca_stack.mean(axis=1)
    _, _, vt = np.linalg.svd(ca_stack - hbc_center[:, None, :])
    return (hbc_center if ref_center is None else ref_center), vt[:, 2, :]


def radial_distance(p, center, axis):
    """Distance of points p (nf, 3) or (nf, k, 3) from the pore axis."""
    if p.ndim == 3:
        center, axis = center[:, None, :], axis[:, None, :]
    v = p - center
    v_perp = v - np.sum(v * axis, axis=-1, keepdims=True) * axis
    return np.linalg.norm(v_perp, axis=-1)


def compute_decomposition(pos, hbc_diagonal_pairs, labels):
    """Per frame, per Phe: ca_r, ring_r, sc_min_r, reach; per diagonal:
    backbone, reach_sum, radial_gap (= backbone - reach_sum)."""
    resids = [r for pair in hbc_diagonal_pairs for r in pair]
    center, axis = radial_frame(np.stack([pos[r]["CA"] for r in resids], axis=1),
                                pos.get("_ref_center"))
    out, per = {}, {}
    for r in resids:
        lab = labels[r]
        ca_r = radial_distance(pos[r]["CA"], center, axis)
        sc_min = radial_distance(pos[r]["sc"], center, axis).min(axis=1)
        per[r] = dict(ca_r=ca_r, sc_min_r=sc_min, reach=ca_r - sc_min)
        out[f"ca_r {lab}"] = ca_r
        out[f"ring_r {lab}"] = radial_distance(pos[r]["ring"], center, axis)
        out[f"sc_min_r {lab}"] = sc_min
        out[f"reach {lab}"] = ca_r - sc_min
    for col, (r1, r2) in zip(("A_C", "B_D"), hbc_diagonal_pairs):
        out[f"backbone {col}"] = per[r1]["ca_r"] + per[r2]["ca_r"]
        out[f"reach_sum {col}"] = per[r1]["reach"] + per[r2]["reach"]
        out[f"radial_gap {col}"] = per[r1]["sc_min_r"] + per[r2]["sc_min_r"]
    return pd.DataFrame(out)


def _split(before, after, bb_col, reach_col, total_col):
    """Mean change before -> after, split into backbone and side-chain parts."""
    d_bb = after[bb_col].mean() - before[bb_col].mean()
    d_sc = -(after[reach_col].mean() - before[reach_col].mean())
    d_tot = after[total_col].mean() - before[total_col].mean()
    pct = 100 * d_bb / d_tot if abs(d_tot) >= 0.05 else np.nan
    return dict(d_backbone=round(d_bb, 3), d_sidechain=round(d_sc, 3),
                d_total=round(d_tot, 3), pct_backbone=round(pct, 1),
                pct_sidechain=round(100 - pct, 1) if np.isfinite(pct) else np.nan)


def _units(hbc_diagonal_pairs, labels):
    """[(unit name, backbone col, reach col, total col), ...]: 4 Phe + 2 diagonals."""
    units = []
    for col, pair in zip(("A_C", "B_D"), hbc_diagonal_pairs):
        for r in pair:
            lab = labels[r]
            units.append((lab, f"ca_r {lab}", f"reach {lab}", f"sc_min_r {lab}"))
        units.append((f"diagonal {col}", f"backbone {col}", f"reach_sum {col}",
                      f"radial_gap {col}"))
    return units


def decomposition_by_state(dec, units):
    closed, opened = dec[dec.state == "closed"], dec[dec.state == "open"]
    rows = []
    if len(closed) and len(opened):
        for name, bb, rc, tot in units:
            rows.append(dict(unit=name,
                             closed_backbone=round(closed[bb].mean(), 3),
                             open_backbone=round(opened[bb].mean(), 3),
                             closed_reach=round(closed[rc].mean(), 3),
                             open_reach=round(opened[rc].mean(), 3),
                             **_split(closed, opened, bb, rc, tot)))
    return pd.DataFrame(rows)


def half_change_frame(s, frames, sw, a, b, lo, hi, search=300):
    """Frame where the smoothed series s passes half way from level a (window
    before) to level b (window after): the first frame on the "after" side
    that follows a frame on the "before" side, searched within sw +- search
    and inside [lo, hi] (start of the window before, end of the window after).
    NaN if |b - a| < 0.1 A (no clear change)."""
    if abs(b - a) < 0.1:
        return np.nan
    mid = (a + b) / 2
    idx = np.flatnonzero((frames >= max(sw - search, lo)) & (frames <= min(sw + search, hi)))
    past = (s[idx] - mid) * np.sign(b - a) > 0
    first_before = np.flatnonzero(~past)
    if not len(first_before):
        return np.nan
    hit = np.flatnonzero(past[first_before[0]:])
    return int(frames[idx[first_before[0] + hit[0]]]) if len(hit) else np.nan


def decomposition_by_switch(dec, segs, units, smooth=50):
    """Every open<->closed switch: the change from the window before to the
    window after, split into backbone / side chain, and when each crosses
    half way (which moves first)."""
    frames = dec.frame.to_numpy()
    sm = {c: smooth_median(dec[c].to_numpy(float), smooth) for c in dec.columns
          if c not in ("frame", "state")}
    rows = []
    segs = segs.reset_index(drop=True)
    for k in range(len(segs) - 1):
        s0, s1 = segs.iloc[k], segs.iloc[k + 1]
        if s0.state == s1.state:
            continue
        before = dec[(dec.frame >= s0.start_frame) & (dec.frame <= s0.end_frame)]
        after = dec[(dec.frame >= s1.start_frame) & (dec.frame <= s1.end_frame)]
        sw = (s0.end_frame + s1.start_frame) // 2
        for name, bb, rc, tot in units:
            lim = (s0.start_frame, s1.end_frame)
            t_bb = half_change_frame(sm[bb], frames, sw, before[bb].mean(), after[bb].mean(), *lim)
            t_sc = half_change_frame(-sm[rc], frames, sw, -before[rc].mean(), -after[rc].mean(),
                                     *lim)
            rows.append(dict(switch=f"{s0.state}->{s1.state}", switch_frame=int(sw),
                             unit=name, **_split(before, after, bb, rc, tot),
                             t_half_backbone=t_bb, t_half_sidechain=t_sc,
                             sidechain_minus_backbone=(t_sc - t_bb
                                                       if np.isfinite(t_bb) and np.isfinite(t_sc)
                                                       else np.nan)))
    return pd.DataFrame(rows)


def correlation_table(dec, hbc_df):
    """Which part follows the real gate distance frame by frame?"""
    rows = []
    d = dec.merge(hbc_df[["frame", "A_C", "B_D"]], on="frame", how="inner")
    for col in ("A_C", "B_D"):
        y = d[col].to_numpy(float)
        bb = d[f"backbone {col}"].to_numpy(float)
        sc = -d[f"reach_sum {col}"].to_numpy(float)
        if min(y.std(), bb.std(), sc.std()) == 0:      # a constant series: nothing to correlate
            rows.append(dict(diagonal=col))
            continue

        def r2(*xs):
            X = np.column_stack([np.ones_like(y), *xs])
            res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            return 1 - res.var() / y.var()

        rows.append(dict(diagonal=col,
                         r_gate_vs_backbone=round(np.corrcoef(y, bb)[0, 1], 3),
                         r_gate_vs_sidechain=round(np.corrcoef(y, sc)[0, 1], 3),
                         r_gate_vs_radial_gap=round(np.corrcoef(
                             y, d[f"radial_gap {col}"])[0, 1], 3),
                         R2_backbone_only=round(r2(bb), 3),
                         R2_sidechain_only=round(r2(sc), 3),
                         R2_both=round(r2(bb, sc), 3)))
    return pd.DataFrame(rows, columns=["diagonal", "r_gate_vs_backbone", "r_gate_vs_sidechain",
                                       "r_gate_vs_radial_gap", "R2_backbone_only",
                                       "R2_sidechain_only", "R2_both"])


def step2b_backbone_vs_sidechain(pos, hbc_diagonal_pairs, labels, hbc_df, per_frame, segs,
                                 out_dir, perm_frames=(), title=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  step 2b: backbone vs side chain ...")
    dec = compute_decomposition(pos, hbc_diagonal_pairs, labels)
    dec.insert(0, "frame", per_frame.frame.to_numpy())
    dec.insert(1, "state", per_frame.state.to_numpy())
    units = _units(hbc_diagonal_pairs, labels)

    by_state = decomposition_by_state(dec, units)
    by_switch = decomposition_by_switch(dec, segs, units)
    corr = correlation_table(dec, hbc_df)
    dec.to_csv(out_dir / "backbone_sidechain_per_frame.csv", index=False, float_format="%.3f")
    by_state.to_csv(out_dir / "decomposition_by_state.csv", index=False)
    by_switch.to_csv(out_dir / "decomposition_by_switch.csv", index=False)
    corr.to_csv(out_dir / "correlation_with_gate.csv", index=False)
    plot_decomposition_timeseries(dec, segs, perm_frames, hbc_df,
                                  out_dir / "backbone_sidechain_timeseries.png", title)
    plot_decomposition_bars(by_state, out_dir / "decomposition_by_state.png", title)

    show = ["unit", "d_backbone", "d_sidechain", "d_total", "pct_backbone", "pct_sidechain"]
    if len(by_state):
        print("  open - closed (mean, A; positive = wider):")
        print(by_state[show].to_string(index=False))
    if len(by_switch):
        print("  per switch (diagonals only):")
        print(by_switch[by_switch.unit.str.startswith("diagonal")][
            ["switch", "switch_frame", "unit", "d_backbone", "d_sidechain", "pct_backbone",
             "t_half_backbone", "t_half_sidechain"]].to_string(index=False))
    print("  frame-by-frame, which part follows the gate distance:")
    print(corr.to_string(index=False))
    print(f"  -> {out_dir}")
    return dec, by_state, by_switch, corr


def plot_decomposition_timeseries(dec, segs, perm_frames, hbc_df, out_png, title, smooth=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("gate distance (Å)", lambda c: hbc_df[c].to_numpy(float)),
            ("backbone: Cα–axis, sum of the 2 (Å)", lambda c: dec[f"backbone {c}"]),
            ("side chain: reach into pore, sum (Å)", lambda c: dec[f"reach_sum {c}"])]
    colors = {"A_C": "#1f77b4", "B_D": "#ff7f0e"}
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 8.5), sharex=True)
    for ax, (ylab, get) in zip(axes, rows):
        _shade(ax, segs)
        for c in ("A_C", "B_D"):
            y = np.asarray(get(c), float)
            ax.plot(dec.frame, y, color=colors[c], lw=0.3, alpha=0.3)
            ax.plot(dec.frame, smooth_median(y, smooth), color=colors[c], lw=1.4, label=c)
        for f in perm_frames:
            ax.axvline(f, color="purple", lw=1)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[2].text(0.01, 0.04, "larger reach = side chains point further into the pore",
                 transform=axes[2].transAxes, fontsize=7, color="0.3")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_decomposition_bars(by_state, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if by_state.empty:
        return
    x = np.arange(len(by_state))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(x)), 4.2))
    ax.bar(x - w, by_state.d_backbone, w, color="#4c72b0", label="backbone (Cα moves out)")
    ax.bar(x, by_state.d_sidechain, w, color="#dd8452", label="side chain (reorients outward)")
    ax.bar(x + w, by_state.d_total, w, color="0.35", label="total")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(by_state.unit, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("open − closed (Å)\npositive = gate wider")
    ax.set_title(title + "  — where the opening comes from", fontsize=10)
    ax.legend(fontsize=8)
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
