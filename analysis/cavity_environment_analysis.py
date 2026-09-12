"""
cavity_environment_analysis.py
==============================

Single-pass analysis of the GLU/ASN cavity environment, designed to explain why
some runs hold a static high ion occupancy with very few permeation events
(e.g. G12 RUN16) while others turn ions over (e.g. G12 RUN3).

Three blocks, ONE trajectory loop:

  BLOCK A - gate geometry / symmetry
      For each gate (glu, asn, hbc, sf, ...): the 2 diagonal pairs and the 4
      adjacent (nearest-neighbour) pairs.  Adjacent pairs are derived
      automatically as "all 6 pairs minus the 2 diagonals", so you only need
      the diagonal_pairs you already define in main.py.
      Per pair, three distance measures:
          tip_min : minimum heavy-atom distance between the two functional
                    groups (carboxylate O / amide O,N).  This is the
                    H-bond / salt-bridge relevant number.
          tip_com : COM-COM of the functional groups.
          ca      : CA-CA (backbone splay, independent of rotamer).
      Per gate, per frame, two symmetry scalars:
          asym_adjacent = std(4 adjacent tip_min) / mean(4 adjacent tip_min)
          diag_ratio    = max(d1,d2) / min(d1,d2) of the 2 diagonals
      Hypothesis tested: run16 is locked in a broken-C4 substate (one ASN
      clamped onto a neighbouring ASP) that creates an off-axis high-affinity
      trap instead of an on-axis conduction pathway.

  BLOCK B - cavity water + ion coordination
      Per frame:
          n_water_cavity   : waters inside the cavity cylinder
          n_k_cavity       : K+ inside the cavity cylinder
          water_per_ion    : n_water_cavity / max(n_k_cavity,1)
      Per cavity K+, per frame:
          hyd_n            : water O within HYD_CUT of the ion
          carbox_n         : GLU/ASP carboxylate O within CARBOX_CUT
          amide_n          : ASN/GLN amide O within CARBOX_CUT
          d_carbox_min     : min ion-carboxylate O distance
          r_axis           : radial distance from the pore axis  <-- on-axis
                             vs off-axis spectator ion
          coord_class      : 'contact' (d_carbox_min < 3.2), 'solvent_sep'
                             (3.2-5.0), 'free' (>5.0)
      Hypotheses tested: dehydrated / contact-ion-paired ions are trapped;
      off-axis ions inflate the occupancy count without contributing to flux.

  BLOCK C - salt bridges / H-bond network of the acidic ring
      Per frame, for every acidic gate residue (GLU/ASP at the glu_* and asn_*
      positions), min distance to every nearby basic N (ARG/LYS/HIS) and to
      every other gate functional group.  Reported as per-pair occupancy
      (fraction of frames below the cut) plus per-frame time series for the
      pairs that are ever engaged.
      Hypothesis tested: a carboxylate locked in an intra-protein salt bridge
      or an ASN-ASP H-bond has a reduced effective charge available for K+,
      which changes both the resident occupancy and the exit barrier.

Every metric is also split into "permeation window" vs "baseline" frames so you
get a direct delta per run, and `compare_runs()` overlays two runs and writes a
side-by-side delta table (that is the run16 vs run3 answer).

Outputs, all under  <results_dir>/cavity_environment/ :
    gate_distances.csv               per-frame, long format
    gate_distance_summary.csv        mean/std/min/max/occupancy, + perm-window delta
    gate_symmetry.csv                per-frame asymmetry scalars
    plots/gate_<name>_tip_min.png    4 adjacent + 2 diagonal traces, permeation vlines
    plots/gate_<name>_ca.png
    plots/symmetry_<name>.png
    water_cavity.csv / water_summary.csv / plots/water_*.png
    ion_coordination.csv / ion_coordination_summary.csv / plots/ion_*.png
    salt_bridges.csv / salt_bridge_summary.csv / plots/salt_bridges.png
"""

import csv
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

try:                                  # normal pipeline import
    from analysis.converter import to_pdb
except ImportError:                   # running the module standalone
    try:
        from .converter import to_pdb
    except ImportError:
        def to_pdb(resid, channel_type):
            return str(resid)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  tunables
# --------------------------------------------------------------------------- #
HBOND_CUT = 3.5      # A, tip_min below this = engaged H-bond / salt bridge
HYD_CUT = 3.4        # A, K+ - water O first shell
CARBOX_CUT = 3.2     # A, K+ - carboxylate O direct contact
SOLV_SEP_CUT = 5.0   # A, beyond CARBOX_CUT but within this = solvent separated
CAVITY_RADIUS = 12.0  # A, cylinder radius for the cavity census
PERM_WINDOW = 20     # frames either side of a permeation for the window split
BASIC_SEARCH_RADIUS = 18.0   # A, prescreen for basic residues near the gate

FUNCTIONAL_ATOMS = {
    "GLU": ["OE1", "OE2"], "GLH": ["OE1", "OE2"],
    "ASP": ["OD1", "OD2"], "ASH": ["OD1", "OD2"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
    "HIS": ["ND1", "NE2"], "HIE": ["ND1", "NE2"],
    "HID": ["ND1", "NE2"], "HIP": ["ND1", "NE2"],
    "LYS": ["NZ"], "LYN": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],
    "TRP": ["NE1"],
}
ACIDIC = {"GLU", "GLH", "ASP", "ASH"}
BASIC = {"LYS", "LYN", "ARG", "HIS", "HIE", "HID", "HIP"}

WATER_SEL = "resname WAT HOH SOL TIP3 T3P and name O OW OH2 O1"
ION_SEL = "resname K+ K"

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D", "CYS": "C",
    "CYX": "C", "GLN": "Q", "GLU": "E", "GLH": "E", "GLY": "G", "HIS": "H",
    "HIE": "H", "HID": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "LYN": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def derive_adjacent_pairs(residues, diagonal_pairs):
    """All 6 pairs of a 4-fold ring minus the 2 diagonals = the 4 adjacent pairs."""
    diag = {frozenset(p) for p in diagonal_pairs}
    adj = [p for p in combinations(sorted(residues), 2) if frozenset(p) not in diag]
    if len(adj) != 4:
        warnings.warn(
            f"derive_adjacent_pairs: expected 4 adjacent pairs, got {len(adj)} "
            f"for residues={residues} diagonals={diagonal_pairs}"
        )
    return adj


def _resname(u, resid):
    ag = u.select_atoms(f"resid {resid}")
    return ag.resnames[0] if len(ag) else "UNK"


def _functional_group(u, resid):
    """AtomGroup of the sidechain functional atoms; falls back to sidechain heavies, then CA."""
    rn = _resname(u, resid)
    names = FUNCTIONAL_ATOMS.get(rn)
    if names:
        ag = u.select_atoms(f"resid {resid} and name {' '.join(names)}")
        if len(ag):
            return ag
    ag = u.select_atoms(
        f"resid {resid} and not name N C O CA HA H* and not type H"
    )
    if len(ag):
        return ag
    return u.select_atoms(f"resid {resid} and name CA")


def _ca(u, resid):
    ag = u.select_atoms(f"resid {resid} and name CA")
    return ag if len(ag) else u.select_atoms(f"resid {resid}")


def load_region_tracking_occupancy(results_dir, regions=(2, 3),
                                   start_frame=0, end_frame=None):
    """
    Per-frame K+ occupancy using the SAME logic as
    CavityOccupancyAnalysis.save_combined_region_ion_counts():
    read ion_region_tracking.json and count ions whose assigned region is in
    `regions` (default 2+3 = GLU/ASN cavity + ASN->HBC = the 'glu_hbc' volume),
    then add untracked ions from untracked_cavity_ions/untracked_ions.csv.

    Both files are searched for by walking up from results_dir, exactly as the
    cavity_occupancy module does.

    Returns (per_frame, info) where
        per_frame : {frame(int): {"tracked": set(resid), "untracked": set(resid)}}
        info      : dict with the paths used and whether each was found
    """
    results_dir = Path(results_dir)
    candidates = [results_dir, results_dir.parent, results_dir.parent.parent]

    tracking_path = None
    for c in candidates:
        p = c / "ion_region_tracking.json"
        if p.exists():
            tracking_path = p
            break

    untracked_path = None
    for c in candidates:
        p = c / "untracked_cavity_ions" / "untracked_ions.csv"
        if p.exists():
            untracked_path = p
            break

    per_frame = {}

    if tracking_path is not None:
        with open(tracking_path) as f:
            raw = json.load(f)
        for resid_str, frame_dict in raw.items():
            resid = int(resid_str)
            for fr_str, region in frame_dict.items():
                if region in regions:
                    fr = int(fr_str)
                    if fr < start_frame:
                        continue
                    if end_frame is not None and fr > end_frame:
                        continue
                    per_frame.setdefault(
                        fr, {"tracked": set(), "untracked": set()}
                    )["tracked"].add(resid)

    if untracked_path is not None:
        with open(untracked_path, newline="") as f:
            for row in csv.DictReader(f):
                fr = int(float(row["frame"]))
                resid = int(float(row["ion_resid"]))
                if fr < start_frame:
                    continue
                if end_frame is not None and fr > end_frame:
                    continue
                d = per_frame.setdefault(
                    fr, {"tracked": set(), "untracked": set()})
                if resid not in d["tracked"]:
                    d["untracked"].add(resid)

    info = dict(
        tracking_path=str(tracking_path) if tracking_path else None,
        untracked_path=str(untracked_path) if untracked_path else None,
        regions=list(regions),
    )
    return per_frame, info


def make_labeller(u, channel_type, use_pdb_numbering=True):
    """
    Return label(resid) -> display string.

    With use_pdb_numbering=True the one-letter residue code from the topology is
    prefixed to analysis.converter.to_pdb(), giving labels like 'D173.G1',
    'E141.G1', 'N184.2a'.  Raw resids are always kept in the CSV columns
    resid_a / resid_b, so nothing is lost.
    """
    cache = {}

    def label(resid):
        resid = int(resid)
        if resid in cache:
            return cache[resid]
        rn = _resname(u, resid)
        one = ONE_LETTER.get(rn, rn)
        if use_pdb_numbering:
            pdb = to_pdb(resid, channel_type)
            out = f"{one}{pdb}" if pdb != str(resid) else f"{rn}{resid}"
        else:
            out = f"{rn}{resid}"
        cache[resid] = out
        return out

    return label


def coerce_permeation_frames(obj):
    """
    Accept whatever the pipeline hands over (list of ints, dict ion->frame,
    list of dicts with an 'end_3'/'frame'/'exit_frame' key, DataFrame) and
    return a sorted list of ints.
    """
    if obj is None:
        return []
    frames = []
    if isinstance(obj, pd.DataFrame):
        for key in ("end_3", "end3", "frame", "exit_frame", "permeation_frame"):
            if key in obj.columns:
                return sorted(int(v) for v in obj[key].dropna())
        return []
    if isinstance(obj, dict):
        for v in obj.values():
            frames.extend(coerce_permeation_frames(v))
        return sorted(set(frames))
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        for v in obj:
            if isinstance(v, (int, np.integer)):
                frames.append(int(v))
            elif isinstance(v, float) and not np.isnan(v):
                frames.append(int(v))
            elif isinstance(v, dict):
                for key in ("end_3", "end3", "frame", "exit_frame", "permeation_frame"):
                    if key in v and v[key] is not None:
                        frames.append(int(v[key]))
                        break
            elif hasattr(v, "end_3"):
                frames.append(int(getattr(v, "end_3")))
        return sorted(set(frames))
    if isinstance(obj, (int, np.integer)):
        return [int(obj)]
    return []


def _window_mask(frames, perm_frames, width=PERM_WINDOW):
    frames = np.asarray(frames)
    mask = np.zeros(len(frames), dtype=bool)
    for pf in perm_frames:
        mask |= (frames >= pf - width) & (frames <= pf + width)
    return mask


def _add_perm_vlines(ax, perm_frames, frame_duration, xmax=None):
    for i, pf in enumerate(perm_frames):
        x = pf * frame_duration
        if xmax is not None and x > xmax:
            continue
        ax.axvline(x, color="k", ls="--", lw=0.7, alpha=0.45,
                   label="permeation" if i == 0 else None)


# --------------------------------------------------------------------------- #
#  main entry point
# --------------------------------------------------------------------------- #
def run_cavity_environment_analysis(
    u,
    results_dir,
    gate_definitions,
    cavity_upper_residues,
    cavity_lower_residues,
    permeation_frames=None,
    permeation_events=None,
    permeation_stage="end_3",
    start_frame=0,
    end_frame=None,
    frame_duration=0.075,
    channel_type="",
    run_number="",
    do_geometry=True,
    do_water=True,
    do_salt_bridges=True,
    use_pdb_numbering=True,
    ion_count_mode="tracking",
    ion_regions=(2, 3),
    cavity_radius=CAVITY_RADIUS,
    radius_sweep=(8.0, 10.0, 12.0, 15.0, 18.0),
    force_rerun=False,
):
    """
    Parameters
    ----------
    gate_definitions : dict
        {gate_name: {"residues": [4 resids],
                     "diagonal_pairs": [(a,b),(c,d)],
                     "adjacent_pairs": optional explicit override}}
    cavity_upper_residues, cavity_lower_residues : list[int]
        Define the z-extent of the cavity census cylinder (e.g. sf_residues and
        hbc_residues).  Measured per frame from CA COM, so it follows the gates.
    permeation_frames : anything coerce_permeation_frames understands
    """
    out = Path(results_dir) / "cavity_environment"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    sentinel = out / "gate_distance_summary.csv"
    if sentinel.exists() and not force_rerun:
        print(f"[cavity_env] outputs already exist in {out} - skipping "
              f"(force_rerun=True to redo)")
        return out

    n_frames_traj = len(u.trajectory)
    if end_frame is None:
        end_frame = n_frames_traj - 1
    end_frame = min(int(end_frame), n_frames_traj - 1)
    start_frame = max(int(start_frame), 0)

    if permeation_events is not None:
        perm_frames = sorted(int(e[permeation_stage]) for e in permeation_events
                             if permeation_stage in e
                             and e[permeation_stage] is not None)
        src = f"permeation_events['{permeation_stage}']"
    else:
        perm_frames = coerce_permeation_frames(permeation_frames)
        src = "permeation_frames"
    perm_frames = [f for f in perm_frames if start_frame <= f <= end_frame]
    print(f"[cavity_env] {channel_type} RUN{run_number}: frames "
          f"{start_frame}-{end_frame}, {len(perm_frames)} permeation events "
          f"from {src} -> {perm_frames}")

    # ---------------- labelling ---------------- #
    lab = make_labeller(u, channel_type, use_pdb_numbering=use_pdb_numbering)

    # ---------------- static selections ---------------- #
    # gate geometry
    gate_meta = {}
    for gname, spec in gate_definitions.items():
        residues = list(spec["residues"])
        diagonals = [tuple(p) for p in spec.get("diagonal_pairs", [])]
        adjacent = spec.get("adjacent_pairs")
        adjacent = ([tuple(p) for p in adjacent] if adjacent
                    else derive_adjacent_pairs(residues, diagonals))
        pairs = ([(a, b, "diagonal") for a, b in diagonals] +
                 [(a, b, "adjacent") for a, b in adjacent])
        gate_meta[gname] = {
            "residues": residues,
            "pairs": pairs,
            "fg": {r: _functional_group(u, r) for r in residues},
            "ca": {r: _ca(u, r) for r in residues},
            "resname": {r: _resname(u, r) for r in residues},
        }
        print(f"  gate '{gname}': " +
              ", ".join(f"{lab(r)} (resid {r})" for r in residues))

    # cavity cylinder anchors
    cav_up_ca = u.select_atoms(
        "name CA and resid " + " ".join(str(r) for r in cavity_upper_residues))
    cav_lo_ca = u.select_atoms(
        "name CA and resid " + " ".join(str(r) for r in cavity_lower_residues))

    # all gate residues pooled -> pore axis centre + acidic/amide oxygens
    all_gate_res = sorted({r for g in gate_meta.values() for r in g["residues"]})
    axis_ca = u.select_atoms(
        "name CA and resid " + " ".join(str(r) for r in all_gate_res))

    water_O = u.select_atoms(WATER_SEL)
    ions = u.select_atoms(ION_SEL)
    ion_index_by_resid = {int(r): i for i, r in enumerate(ions.resids)}

    # ---- K+ occupancy source: same logic as CavityOccupancyAnalysis ---- #
    occ_by_frame, occ_info = {}, {}
    if do_water and ion_count_mode == "tracking":
        occ_by_frame, occ_info = load_region_tracking_occupancy(
            results_dir, regions=tuple(ion_regions),
            start_frame=start_frame, end_frame=end_frame)
        if occ_info.get("tracking_path"):
            print(f"  K+ occupancy from ion_region_tracking.json "
                  f"regions {tuple(ion_regions)} "
                  f"({occ_info['tracking_path']})")
            print(f"  untracked ions: "
                  f"{occ_info.get('untracked_path') or 'NOT FOUND - counts may be low'}")
        else:
            print("  WARNING: ion_region_tracking.json not found - "
                  "falling back to geometric cylinder counting")
            ion_count_mode = "geometric"
    elif do_water:
        print(f"  K+ occupancy from geometric cylinder (r={cavity_radius} A)")

    # carboxylate / amide oxygens of the cavity-lining ring, for ion coordination
    ring_res = sorted({r for gname in ("glu", "asn") if gname in gate_meta
                       for r in gate_meta[gname]["residues"]})
    if not ring_res:
        ring_res = all_gate_res
    carbox_atoms = u.select_atoms(
        "resid " + " ".join(str(r) for r in ring_res) +
        " and name OE1 OE2 OD1 OD2") if ring_res else u.select_atoms("index -1")
    amide_atoms = u.select_atoms(
        "resid " + " ".join(str(r) for r in ring_res) +
        " and name ND2 NE2 OD1 OE1") if ring_res else u.select_atoms("index -1")

    # salt-bridge partners: acidic gate residues vs nearby basic N
    u.trajectory[start_frame]
    acidic_res = [r for r in ring_res if gate_res_name(gate_meta, r) in ACIDIC]
    gate_centre0 = axis_ca.positions.mean(axis=0) if len(axis_ca) else np.zeros(3)
    basic_candidates = []
    if do_salt_bridges and acidic_res:
        basic_ag = u.select_atoms(
            "protein and name NZ NE NH1 NH2 ND1 NE2 and around "
            f"{BASIC_SEARCH_RADIUS} (resid "
            + " ".join(str(r) for r in acidic_res) + ")")
        for res in set(basic_ag.resids):
            rn = _resname(u, res)
            if rn in BASIC:
                basic_candidates.append(int(res))
        basic_candidates = sorted(set(basic_candidates))
        print(f"  salt-bridge basic candidates: "
              + (", ".join(lab(r) for r in basic_candidates) or "none"))
    basic_fg = {r: _functional_group(u, r) for r in basic_candidates}
    acidic_fg = {r: _functional_group(u, r) for r in acidic_res}

    # gate-gate functional pairs across the whole acidic/amide ring (for BLOCK C)
    ring_fg = {r: _functional_group(u, r) for r in ring_res}
    ring_pairs = list(combinations(ring_res, 2))

    # ---------------- accumulators ---------------- #
    geo_rows, sym_rows, water_rows, ion_rows, sb_rows = [], [], [], [], []

    frames_iter = range(start_frame, end_frame + 1)
    for fi in tqdm(frames_iter, desc="[cavity_env]"):
        u.trajectory[fi]
        t_ns = fi * frame_duration

        # ---------- BLOCK A ---------- #
        if do_geometry:
            for gname, meta in gate_meta.items():
                tipmins = {}
                for a, b, kind in meta["pairs"]:
                    fa, fb = meta["fg"][a], meta["fg"][b]
                    d = distance_array(fa.positions, fb.positions)
                    tip_min = float(d.min())
                    tip_com = float(np.linalg.norm(
                        fa.positions.mean(axis=0) - fb.positions.mean(axis=0)))
                    ca_d = float(np.linalg.norm(
                        meta["ca"][a].positions.mean(axis=0)
                        - meta["ca"][b].positions.mean(axis=0)))
                    pdb_a, pdb_b = lab(a), lab(b)
                    pair_label = f"{pdb_a}-{pdb_b}"
                    geo_rows.append(dict(
                        frame=fi, time_ns=t_ns, gate=gname, pair=pair_label,
                        resid_a=a, resid_b=b, pdb_a=pdb_a, pdb_b=pdb_b,
                        kind=kind,
                        tip_min=tip_min, tip_com=tip_com, ca=ca_d))
                    tipmins[(a, b, kind)] = tip_min

                adj = [v for k, v in tipmins.items() if k[2] == "adjacent"]
                dia = [v for k, v in tipmins.items() if k[2] == "diagonal"]
                row = dict(frame=fi, time_ns=t_ns, gate=gname)
                if adj:
                    row["adj_mean"] = float(np.mean(adj))
                    row["adj_std"] = float(np.std(adj))
                    row["asym_adjacent"] = float(np.std(adj) / np.mean(adj))
                    row["adj_min"] = float(np.min(adj))
                if len(dia) == 2:
                    row["diag_1"] = dia[0]
                    row["diag_2"] = dia[1]
                    row["diag_mean"] = float(np.mean(dia))
                    row["diag_ratio"] = float(max(dia) / max(min(dia), 1e-6))
                sym_rows.append(row)

        # ---------- BLOCK B ---------- #
        if do_water or do_salt_bridges:
            centre_xy = (axis_ca.positions.mean(axis=0)[:2]
                         if len(axis_ca) else np.zeros(2))
        if do_water:
            z_hi = cav_up_ca.positions[:, 2].mean() if len(cav_up_ca) else 0.0
            z_lo = cav_lo_ca.positions[:, 2].mean() if len(cav_lo_ca) else 0.0
            if z_lo > z_hi:
                z_lo, z_hi = z_hi, z_lo

            def _in_cavity(pos, radius=cavity_radius):
                r = np.linalg.norm(pos[:, :2] - centre_xy, axis=1)
                return (r <= radius) & (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)

            wpos = water_O.positions
            wmask = _in_cavity(wpos) if len(wpos) else np.zeros(0, dtype=bool)
            n_water = int(wmask.sum())

            # ---- K+ occupancy ---- #
            if ion_count_mode == "tracking":
                d = occ_by_frame.get(fi, {"tracked": set(), "untracked": set()})
                cav_resids = sorted(d["tracked"] | d["untracked"])
                n_k_tracked = len(d["tracked"])
                n_k_untracked = len(d["untracked"])
                idx = [ion_index_by_resid[r] for r in cav_resids
                       if r in ion_index_by_resid]
                cav_ion_pos = ions.positions[idx] if idx else np.zeros((0, 3))
                cav_ion_ids = np.array([r for r in cav_resids
                                        if r in ion_index_by_resid], dtype=int)
            else:
                ipos = ions.positions
                imask = _in_cavity(ipos) if len(ipos) else np.zeros(0, dtype=bool)
                cav_ion_pos = ipos[imask]
                cav_ion_ids = ions.resids[imask]
                n_k_tracked, n_k_untracked = int(imask.sum()), 0
            n_k = len(cav_ion_ids)

            wrow = dict(
                frame=fi, time_ns=t_ns, z_low=float(z_lo), z_high=float(z_hi),
                cavity_height=float(z_hi - z_lo),
                n_water_cavity=n_water, n_k_cavity=n_k,
                n_k_tracked=n_k_tracked, n_k_untracked=n_k_untracked,
                water_per_ion=n_water / max(n_k, 1))

            # ---- radius sensitivity: same z-window, different radii ---- #
            if radius_sweep and len(wpos):
                rw = np.linalg.norm(wpos[:, :2] - centre_xy, axis=1)
                zin = (wpos[:, 2] >= z_lo) & (wpos[:, 2] <= z_hi)
                for rad in radius_sweep:
                    wrow[f"n_water_r{rad:g}"] = int(((rw <= rad) & zin).sum())
                if ion_count_mode == "geometric" and len(ions):
                    ri = np.linalg.norm(ions.positions[:, :2] - centre_xy, axis=1)
                    zi = ((ions.positions[:, 2] >= z_lo)
                          & (ions.positions[:, 2] <= z_hi))
                    for rad in radius_sweep:
                        wrow[f"n_k_r{rad:g}"] = int(((ri <= rad) & zi).sum())
            water_rows.append(wrow)

            if n_k:
                dw = (distance_array(cav_ion_pos, wpos) if len(wpos)
                      else np.full((n_k, 1), 99.0))
                dc = (distance_array(cav_ion_pos, carbox_atoms.positions)
                      if len(carbox_atoms) else np.full((n_k, 1), 99.0))
                da = (distance_array(cav_ion_pos, amide_atoms.positions)
                      if len(amide_atoms) else np.full((n_k, 1), 99.0))
                for j in range(n_k):
                    d_cb = float(dc[j].min())
                    if d_cb < CARBOX_CUT:
                        cclass = "contact"
                    elif d_cb < SOLV_SEP_CUT:
                        cclass = "solvent_sep"
                    else:
                        cclass = "free"
                    r_axis = float(np.linalg.norm(cav_ion_pos[j][:2] - centre_xy))
                    ion_rows.append(dict(
                        frame=fi, time_ns=t_ns, ion_resid=int(cav_ion_ids[j]),
                        z=float(cav_ion_pos[j][2]), r_axis=r_axis,
                        hyd_n=int((dw[j] < HYD_CUT).sum()),
                        carbox_n=int((dc[j] < CARBOX_CUT).sum()),
                        amide_n=int((da[j] < CARBOX_CUT).sum()),
                        d_carbox_min=d_cb, coord_class=cclass,
                        n_k_cavity=n_k))

        # ---------- BLOCK C ---------- #
        if do_salt_bridges:
            for a in acidic_res:
                pa = acidic_fg[a].positions
                for b in basic_candidates:
                    d = float(distance_array(pa, basic_fg[b].positions).min())
                    if d < 8.0:   # only store plausibly-engaged pairs
                        sb_rows.append(dict(
                            frame=fi, time_ns=t_ns, kind="acid-base",
                            partner_a=lab(a), partner_b=lab(b),
                            resid_a=a, resid_b=b,
                            d_min=d, engaged=int(d < HBOND_CUT)))
            for a, b in ring_pairs:
                d = float(distance_array(ring_fg[a].positions,
                                         ring_fg[b].positions).min())
                if d < 8.0:
                    sb_rows.append(dict(
                        frame=fi, time_ns=t_ns, kind="ring-ring",
                        partner_a=lab(a), partner_b=lab(b),
                        resid_a=a, resid_b=b,
                        d_min=d, engaged=int(d < HBOND_CUT)))

    # ---------------- write + plot ---------------- #
    resid_labels = {str(r): lab(r) for r in all_gate_res}
    resid_labels.update({str(r): lab(r) for r in basic_candidates})

    meta_json = dict(channel_type=channel_type, run_number=str(run_number),
                     start_frame=start_frame, end_frame=end_frame,
                     frame_duration=frame_duration,
                     use_pdb_numbering=use_pdb_numbering,
                     ion_count_mode=ion_count_mode,
                     ion_regions=list(ion_regions),
                     cavity_radius=cavity_radius,
                     occupancy_source=occ_info,
                     resid_labels=resid_labels,
                     n_permeations=len(perm_frames),
                     permeation_frames=perm_frames,
                     cuts=dict(HBOND_CUT=HBOND_CUT, HYD_CUT=HYD_CUT,
                               CARBOX_CUT=CARBOX_CUT,
                               SOLV_SEP_CUT=SOLV_SEP_CUT,
                               CAVITY_RADIUS=CAVITY_RADIUS,
                               PERM_WINDOW=PERM_WINDOW))
    (out / "run_meta.json").write_text(json.dumps(meta_json, indent=2))

    if geo_rows:
        geo = pd.DataFrame(geo_rows)
        geo.to_csv(out / "gate_distances.csv", index=False)
        _summarise_geometry(geo, perm_frames, out)
        _plot_gates(geo, perm_frames, frame_duration, plots,
                    channel_type, run_number)
    if sym_rows:
        sym = pd.DataFrame(sym_rows)
        sym.to_csv(out / "gate_symmetry.csv", index=False)
        _plot_symmetry(sym, perm_frames, frame_duration, plots,
                       channel_type, run_number)
    if water_rows:
        wat = pd.DataFrame(water_rows)
        wat.to_csv(out / "water_cavity.csv", index=False)
        _summarise_generic(wat, ["n_water_cavity", "n_k_cavity",
                                 "n_k_tracked", "n_k_untracked",
                                 "water_per_ion", "cavity_height"],
                           perm_frames, out / "water_summary.csv")
        _summarise_radius_sweep(wat, cavity_radius, out)
        _plot_water(wat, perm_frames, frame_duration, plots,
                    channel_type, run_number)
    if ion_rows:
        ionsdf = pd.DataFrame(ion_rows)
        ionsdf.to_csv(out / "ion_coordination.csv", index=False)
        _summarise_ions(ionsdf, perm_frames, out)
        _plot_ions(ionsdf, perm_frames, frame_duration, plots,
                   channel_type, run_number)
    if sb_rows:
        sb = pd.DataFrame(sb_rows)
        sb.to_csv(out / "salt_bridges.csv", index=False)
        _summarise_salt_bridges(sb, perm_frames, out)
        _plot_salt_bridges(sb, perm_frames, frame_duration, plots,
                           channel_type, run_number)

    print(f"[cavity_env] done -> {out}")
    return out


def gate_res_name(gate_meta, resid):
    for meta in gate_meta.values():
        if resid in meta["resname"]:
            return meta["resname"][resid]
    return "UNK"


# --------------------------------------------------------------------------- #
#  summaries
# --------------------------------------------------------------------------- #
def _perm_split(df, perm_frames, value_cols):
    """Return dict of col -> (mean_all, mean_window, mean_baseline, delta)."""
    if not perm_frames:
        return {c: (df[c].mean(), np.nan, df[c].mean(), np.nan)
                for c in value_cols if c in df}
    frames = df["frame"].to_numpy()
    mask = _window_mask(frames, perm_frames)
    res = {}
    for c in value_cols:
        if c not in df:
            continue
        v = df[c].to_numpy(dtype=float)
        mw = np.nanmean(v[mask]) if mask.any() else np.nan
        mb = np.nanmean(v[~mask]) if (~mask).any() else np.nan
        res[c] = (float(np.nanmean(v)), float(mw), float(mb), float(mw - mb))
    return res


def _summarise_geometry(geo, perm_frames, out):
    rows = []
    for (gate, pair), sub in geo.groupby(["gate", "pair"]):
        kind = sub["kind"].iloc[0]
        split = _perm_split(sub, perm_frames, ["tip_min", "ca"])
        rows.append(dict(
            gate=gate, pair=pair, kind=kind, n_frames=len(sub),
            tip_min_mean=sub["tip_min"].mean(),
            tip_min_std=sub["tip_min"].std(),
            tip_min_min=sub["tip_min"].min(),
            tip_min_max=sub["tip_min"].max(),
            tip_com_mean=sub["tip_com"].mean(),
            tip_com_std=sub["tip_com"].std(),
            ca_mean=sub["ca"].mean(), ca_std=sub["ca"].std(),
            frac_engaged=float((sub["tip_min"] < HBOND_CUT).mean()),
            tip_min_permwin=split.get("tip_min", (np.nan,)*4)[1],
            tip_min_baseline=split.get("tip_min", (np.nan,)*4)[2],
            tip_min_delta=split.get("tip_min", (np.nan,)*4)[3],
        ))
    df = pd.DataFrame(rows).sort_values(["gate", "kind", "pair"])
    df.to_csv(out / "gate_distance_summary.csv", index=False)
    return df


def _summarise_generic(df, cols, perm_frames, path):
    split = _perm_split(df, perm_frames, cols)
    rows = []
    for c in cols:
        if c not in df:
            continue
        rows.append(dict(metric=c, mean=df[c].mean(), std=df[c].std(),
                         min=df[c].min(), max=df[c].max(),
                         mean_permwin=split[c][1],
                         mean_baseline=split[c][2],
                         delta=split[c][3]))
    pd.DataFrame(rows).to_csv(path, index=False)


def _summarise_radius_sweep(wat, cavity_radius, out):
    """
    How sensitive are the counts to CAVITY_RADIUS?  One row per swept radius,
    with the count relative to the radius actually used, so you can see
    whether the choice changes conclusions or only absolute numbers.
    """
    wcols = sorted([c for c in wat.columns if c.startswith("n_water_r")],
                   key=lambda c: float(c.replace("n_water_r", "")))
    if not wcols:
        return
    kcols = {float(c.replace("n_k_r", "")): c
             for c in wat.columns if c.startswith("n_k_r")}
    rows = []
    ref = None
    for c in wcols:
        rad = float(c.replace("n_water_r", ""))
        m = wat[c].mean()
        if ref is None or abs(rad - cavity_radius) < 1e-9:
            if abs(rad - cavity_radius) < 1e-9:
                ref = m
        rows.append(dict(radius=rad, water_mean=m, water_std=wat[c].std(),
                         k_mean=(wat[kcols[rad]].mean()
                                 if rad in kcols else np.nan)))
    ref = ref if ref else rows[0]["water_mean"]
    for r in rows:
        r["water_rel_to_used"] = r["water_mean"] / ref if ref else np.nan
        r["is_radius_used"] = abs(r["radius"] - cavity_radius) < 1e-9
    pd.DataFrame(rows).to_csv(out / "radius_sensitivity.csv", index=False)


def _summarise_ions(ionsdf, perm_frames, out):
    rows = []
    for cls, sub in ionsdf.groupby("coord_class"):
        rows.append(dict(scope=f"class:{cls}", n_obs=len(sub),
                         frac_of_obs=len(sub) / len(ionsdf),
                         hyd_n_mean=sub["hyd_n"].mean(),
                         r_axis_mean=sub["r_axis"].mean(),
                         d_carbox_mean=sub["d_carbox_min"].mean()))
    # on-axis vs off-axis split, with the median radius as the divider
    r_med = ionsdf["r_axis"].median()
    for name, sub in (("on_axis", ionsdf[ionsdf.r_axis <= r_med]),
                      ("off_axis", ionsdf[ionsdf.r_axis > r_med])):
        rows.append(dict(scope=name, n_obs=len(sub),
                         frac_of_obs=len(sub) / len(ionsdf),
                         hyd_n_mean=sub["hyd_n"].mean(),
                         r_axis_mean=sub["r_axis"].mean(),
                         d_carbox_mean=sub["d_carbox_min"].mean()))
    per_frame = ionsdf.groupby("frame")[["hyd_n", "d_carbox_min", "r_axis"]].mean()
    per_frame = per_frame.reset_index()
    split = _perm_split(per_frame, perm_frames,
                        ["hyd_n", "d_carbox_min", "r_axis"])
    for c, (m, mw, mb, d) in split.items():
        rows.append(dict(scope=f"permsplit:{c}", n_obs=len(per_frame),
                         frac_of_obs=np.nan, hyd_n_mean=m,
                         r_axis_mean=mw, d_carbox_mean=mb, delta=d))
    pd.DataFrame(rows).to_csv(out / "ion_coordination_summary.csv", index=False)

    # conditional occupancy table: how much time at each N, and events there
    occ = ionsdf.groupby("frame")["n_k_cavity"].first()
    tab = occ.value_counts().sort_index().rename("n_frames").to_frame()
    tab["frac_frames"] = tab["n_frames"] / tab["n_frames"].sum()
    if perm_frames:
        ev = pd.Series([occ.get(f, np.nan) for f in perm_frames]).dropna()
        tab["n_permeations"] = ev.value_counts().reindex(tab.index).fillna(0)
        tab["rate_per_frame"] = tab["n_permeations"] / tab["n_frames"]
    tab.index.name = "n_k_cavity"
    tab.to_csv(out / "occupancy_conditional_rate.csv")


def _summarise_salt_bridges(sb, perm_frames, out):
    rows = []
    n_frames = sb["frame"].nunique()
    for (kind, a, b), sub in sb.groupby(["kind", "partner_a", "partner_b"]):
        split = _perm_split(sub, perm_frames, ["d_min"])
        rows.append(dict(
            kind=kind, partner_a=a, partner_b=b,
            n_frames_observed=len(sub),
            occupancy=float(sub["engaged"].sum()) / max(n_frames, 1),
            d_min_mean=sub["d_min"].mean(), d_min_std=sub["d_min"].std(),
            d_min_min=sub["d_min"].min(),
            d_min_permwin=split.get("d_min", (np.nan,)*4)[1],
            d_min_baseline=split.get("d_min", (np.nan,)*4)[2],
            d_min_delta=split.get("d_min", (np.nan,)*4)[3]))
    df = pd.DataFrame(rows).sort_values("occupancy", ascending=False)
    df.to_csv(out / "salt_bridge_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #
def _title(base, channel_type, run_number):
    tag = f"{channel_type} RUN{run_number}".strip()
    return f"{base} - {tag}" if tag else base


def _plot_gates(geo, perm_frames, fd, plots, ct, rn):
    for measure in ("tip_min", "ca"):
        for gate, sub in geo.groupby("gate"):
            fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            for ax, kind in zip(axes, ("adjacent", "diagonal")):
                s = sub[sub["kind"] == kind]
                for pair, ps in s.groupby("pair"):
                    ps = ps.sort_values("frame")
                    ax.plot(ps["time_ns"], ps[measure], lw=0.8, label=pair)
                _add_perm_vlines(ax, perm_frames, fd)
                if measure == "tip_min":
                    ax.axhline(HBOND_CUT, color="r", ls=":", lw=1,
                               label=f"{HBOND_CUT} A")
                ax.set_ylabel(f"{kind}\n{measure} (A)")
                ax.legend(fontsize=7, ncol=3, loc="upper right")
                ax.grid(alpha=0.25)
            axes[0].set_title(_title(f"{gate} gate cross-subunit distances "
                                     f"({measure})", ct, rn))
            axes[-1].set_xlabel("time (ns)")
            fig.tight_layout()
            fig.savefig(plots / f"gate_{gate}_{measure}.png", dpi=160)
            plt.close(fig)

            # distribution panel - bimodality is the diagnostic
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for pair, ps in sub.groupby("pair"):
                ax.hist(ps[measure], bins=60, histtype="step", lw=1.2,
                        density=True, label=f"{pair} ({ps['kind'].iloc[0][:3]})")
            if measure == "tip_min":
                ax.axvline(HBOND_CUT, color="r", ls=":", lw=1)
            ax.set_xlabel(f"{measure} (A)")
            ax.set_ylabel("density")
            ax.set_title(_title(f"{gate} gate {measure} distributions", ct, rn))
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / f"gate_{gate}_{measure}_hist.png", dpi=160)
            plt.close(fig)


def _plot_symmetry(sym, perm_frames, fd, plots, ct, rn):
    for gate, sub in sym.groupby("gate"):
        sub = sub.sort_values("frame")
        cols = [c for c in ("asym_adjacent", "diag_ratio") if c in sub]
        if not cols:
            continue
        fig, axes = plt.subplots(len(cols), 1, figsize=(13, 3.2 * len(cols)),
                                 sharex=True, squeeze=False)
        for ax, c in zip(axes[:, 0], cols):
            ax.plot(sub["time_ns"], sub[c], lw=0.8, color="tab:purple")
            _add_perm_vlines(ax, perm_frames, fd)
            ax.set_ylabel(c)
            ax.grid(alpha=0.25)
        axes[0, 0].set_title(_title(f"{gate} gate symmetry", ct, rn))
        axes[-1, 0].set_xlabel("time (ns)")
        fig.tight_layout()
        fig.savefig(plots / f"symmetry_{gate}.png", dpi=160)
        plt.close(fig)


def _plot_water(wat, perm_frames, fd, plots, ct, rn):
    wat = wat.sort_values("frame")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, c, lab in zip(axes,
                          ["n_water_cavity", "n_k_cavity", "water_per_ion"],
                          ["waters in cavity", "K+ in cavity",
                           "waters per K+"]):
        ax.plot(wat["time_ns"], wat[c], lw=0.8)
        _add_perm_vlines(ax, perm_frames, fd)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    axes[0].set_title(_title("cavity water / ion census", ct, rn))
    axes[-1].set_xlabel("time (ns)")
    fig.tight_layout()
    fig.savefig(plots / "water_cavity_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(wat["n_water_cavity"], bins=40, alpha=0.8)
    ax.set_xlabel("waters in cavity")
    ax.set_ylabel("frames")
    ax.set_title(_title("cavity water distribution", ct, rn))
    fig.tight_layout()
    fig.savefig(plots / "water_cavity_hist.png", dpi=160)
    plt.close(fig)


def _plot_ions(ionsdf, perm_frames, fd, plots, ct, rn):
    per_frame = ionsdf.groupby("frame").agg(
        time_ns=("time_ns", "first"),
        hyd_mean=("hyd_n", "mean"),
        d_carbox_min=("d_carbox_min", "min"),
        r_axis_max=("r_axis", "max")).reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, c, lab in zip(axes,
                          ["hyd_mean", "d_carbox_min", "r_axis_max"],
                          ["mean K+ hydration no.",
                           "min K+-carboxylate O (A)",
                           "max off-axis radius (A)"]):
        ax.plot(per_frame["time_ns"], per_frame[c], lw=0.8)
        _add_perm_vlines(ax, perm_frames, fd)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    axes[1].axhline(CARBOX_CUT, color="r", ls=":", lw=1)
    axes[0].set_title(_title("cavity K+ coordination", ct, rn))
    axes[-1].set_xlabel("time (ns)")
    fig.tight_layout()
    fig.savefig(plots / "ion_coordination_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(ionsdf["r_axis"], ionsdf["z"], c=ionsdf["hyd_n"],
                    s=4, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="hydration no.")
    ax.set_xlabel("radial distance from pore axis (A)")
    ax.set_ylabel("z (A)")
    ax.set_title(_title("cavity K+ positions (on- vs off-axis)", ct, rn))
    fig.tight_layout()
    fig.savefig(plots / "ion_radial_z_map.png", dpi=160)
    plt.close(fig)


def _plot_salt_bridges(sb, perm_frames, fd, plots, ct, rn, top_n=8):
    key = sb.groupby(["kind", "partner_a", "partner_b"])["engaged"].sum()
    key = key.sort_values(ascending=False).head(top_n)
    if key.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for (kind, a, b) in key.index:
        s = sb[(sb.kind == kind) & (sb.partner_a == a)
               & (sb.partner_b == b)].sort_values("frame")
        ax.plot(s["time_ns"], s["d_min"], lw=0.8, label=f"{a}-{b} [{kind}]")
    ax.axhline(HBOND_CUT, color="r", ls=":", lw=1, label=f"{HBOND_CUT} A")
    _add_perm_vlines(ax, perm_frames, fd)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("min heavy-atom distance (A)")
    ax.set_title(_title("most-engaged salt bridges / H-bonds", ct, rn))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "salt_bridges.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  cross-run comparison  (run16 vs run3)
# --------------------------------------------------------------------------- #
def compare_runs(run_dirs, labels, out_dir, frame_duration=0.075):
    """
    run_dirs : list of results_dir paths (each must already contain
               cavity_environment/ from run_cavity_environment_analysis)
    labels   : list of short names, e.g. ["RUN16", "RUN3"]
    """
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    envs = [Path(d) / "cavity_environment" for d in run_dirs]

    # ---- geometry delta table ---- #
    geos = []
    for env, lab in zip(envs, labels):
        p = env / "gate_distance_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["run"] = lab
            geos.append(df)
    if geos:
        g = pd.concat(geos)
        wide = g.pivot_table(index=["gate", "pair", "kind"], columns="run",
                             values=["tip_min_mean", "tip_min_std",
                                     "frac_engaged"])
        wide.columns = ["_".join(c) for c in wide.columns]
        if len(labels) == 2:
            a, b = labels
            for m in ("tip_min_mean", "frac_engaged"):
                ca_, cb_ = f"{m}_{a}", f"{m}_{b}"
                if ca_ in wide and cb_ in wide:
                    wide[f"{m}_delta"] = wide[ca_] - wide[cb_]
        wide = wide.reset_index()
        wide.to_csv(out / "compare_gate_distances.csv", index=False)

        # overlay distributions per gate/pair
        raws = []
        for env, lab in zip(envs, labels):
            p = env / "gate_distances.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["run"] = lab
                raws.append(df)
        if raws:
            r = pd.concat(raws)
            for gate, sub in r.groupby("gate"):
                pairs = sorted(sub["pair"].unique())
                n = len(pairs)
                ncol = 3
                nrow = int(np.ceil(n / ncol))
                fig, axes = plt.subplots(nrow, ncol,
                                         figsize=(4.2 * ncol, 3.0 * nrow),
                                         squeeze=False)
                for ax, pair in zip(axes.ravel(), pairs):
                    ps = sub[sub["pair"] == pair]
                    for lab, pl in ps.groupby("run"):
                        ax.hist(pl["tip_min"], bins=50, histtype="step",
                                density=True, lw=1.3, label=lab)
                    ax.axvline(HBOND_CUT, color="r", ls=":", lw=1)
                    ax.set_title(pair, fontsize=8)
                    ax.set_xlabel("tip_min (A)")
                    ax.legend(fontsize=7)
                for ax in axes.ravel()[n:]:
                    ax.axis("off")
                fig.suptitle(f"{gate} gate: tip_min distributions, "
                             + " vs ".join(labels))
                fig.tight_layout()
                fig.savefig(out / "plots" / f"compare_{gate}_tip_min.png",
                            dpi=160)
                plt.close(fig)

    # ---- water / occupancy / coordination deltas ---- #
    rows = []
    for env, lab in zip(envs, labels):
        rec = {"run": lab}
        p = env / "water_cavity.csv"
        if p.exists():
            w = pd.read_csv(p)
            rec.update(n_water_mean=w["n_water_cavity"].mean(),
                       n_water_std=w["n_water_cavity"].std(),
                       n_k_mean=w["n_k_cavity"].mean(),
                       n_k_mode=w["n_k_cavity"].mode().iloc[0],
                       water_per_ion=w["water_per_ion"].mean(),
                       cavity_height=w["cavity_height"].mean())
        p = env / "ion_coordination.csv"
        if p.exists():
            i = pd.read_csv(p)
            rec.update(hyd_mean=i["hyd_n"].mean(),
                       frac_contact=(i["coord_class"] == "contact").mean(),
                       frac_solvent_sep=(i["coord_class"] == "solvent_sep").mean(),
                       r_axis_mean=i["r_axis"].mean(),
                       frac_off_axis_gt6=(i["r_axis"] > 6.0).mean())
        p = env / "run_meta.json"
        if p.exists():
            m = json.loads(p.read_text())
            rec.update(n_permeations=m.get("n_permeations"),
                       n_frames=m.get("end_frame", 0) - m.get("start_frame", 0) + 1)
        rows.append(rec)
    comp = pd.DataFrame(rows).set_index("run").T
    if len(labels) == 2:
        comp["delta"] = comp[labels[0]] - comp[labels[1]]
    comp.to_csv(out / "compare_cavity_metrics.csv")

    # ---- salt bridge occupancy side by side ---- #
    sbs = []
    for env, lab in zip(envs, labels):
        p = env / "salt_bridge_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["run"] = lab
            sbs.append(df)
    if sbs:
        s = pd.concat(sbs)
        w = s.pivot_table(index=["kind", "partner_a", "partner_b"],
                          columns="run", values=["occupancy", "d_min_mean"])
        w.columns = ["_".join(c) for c in w.columns]
        if len(labels) == 2:
            ca_, cb_ = f"occupancy_{labels[0]}", f"occupancy_{labels[1]}"
            if ca_ in w and cb_ in w:
                w["occupancy_delta"] = w[ca_].fillna(0) - w[cb_].fillna(0)
                w = w.sort_values("occupancy_delta", key=abs, ascending=False)
        w.reset_index().to_csv(out / "compare_salt_bridges.csv", index=False)

    print(f"[compare_runs] -> {out}")
    return out