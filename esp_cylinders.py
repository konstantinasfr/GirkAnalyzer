#!/usr/bin/env python3
"""
esp_cylinders.py
================
Reads a Jaguar RESP grid file (.resp, written by iplotesp=1 + icfit=1) and
measures the QM electrostatic potential inside cylindrical regions defined by
Glu-CA / Asn-CA (or Glu-CA / Asp-CA) residue pairs -- one cylinder per subunit.

Outputs:
    - a stats table (n, mean, median, std, min, max, Q1, Q3) as CSV + printed
    - a 2x2 panel of ESP histograms, one per cylinder (auto-scaled)
    - a SECOND 2x2 panel on FIXED axes (--xlim / --binwidth / --ylim),
      so histograms from different clusters can be laid side by side and a real
      shift is visible instead of being hidden by per-figure autoscaling
    - a boxplot figure comparing the cylinders
    - optionally the raw per-cylinder ESP values, for re-plotting / pooling

UNITS: the .resp file stores potential in hartree/e and coordinates in bohr.
Everything here is converted to kcal/mol per +1e and Angstroms.
A value of -150 means a +1 point charge placed there would be stabilised by
150 kcal/mol (gas-phase, unscreened, excised-fragment -- see caveats below).

-------------------------------------------------------------------------------
STEP 1 -- find your CA atom indices:

    python3 esp_cylinders.py --resp frame5606.resp --out frame5606.out --list-ca

  This prints every detected backbone CA with its coordinates and what kind of
  sidechain hangs off it (CARBOXYLATE = Glu/Asp, AMIDE = Asn/Gln). Read off the
  indices and build your pairs.

STEP 2 -- run the analysis:

    python3 esp_cylinders.py --resp frame5606.resp --out frame5606.out \
        --pairs 7:97 30:142 52:187 75:201 \
        --radius 3.0 --extend 3.0 --label G12_frame5606

  --pairs takes GLU_CA_INDEX:ASN_CA_INDEX, 1-based, matching Jaguar's atom
  numbering (C1, H2, C3 ... -> 1, 2, 3 ...). Give as many pairs as you have
  subunits; 4 is the normal case.

  Use --names to label them, e.g. --names E141.G1/D173.G1 E152.2a/N184.2a ...

STEP 3 -- same command on the G2 cluster, then compare.

-------------------------------------------------------------------------------
GEOMETRY MODES:

  --axis ca-ca  (default)  Cylinder axis is the line from the Glu CA to the
                           Asn CA. Extended by --extend Angstrom beyond each
                           CA ALONG THAT AXIS. Radius measured perpendicular
                           to it. This is the per-subunit inter-ring region.

  --axis z                 Cylinder axis is vertical (parallel to z), passing
                           through the xy midpoint of the two CAs. Axial range
                           is [min(z_CA) - extend, max(z_CA) + extend].
                           This is the literal "z(CA_glu)+3 to z(CA_asn)-3".

-------------------------------------------------------------------------------
CAVEATS worth repeating in a methods section:

  * The Jaguar ESP grid is built on vdW shells, so the lumen is sampled
    incidentally and sparsely. The script reports n per cylinder -- if n is
    small (< ~200) the mean is dominated by where points happened to land, and
    two structures will not be sampled equivalently. Check n before trusting a
    between-structure difference. --min-points warns about this.

  * The cluster is an excised fragment in vacuum. Part of any G12-vs-G2
    difference is the unscreened monopole (~dQ/r) of a -5 vs -4 fragment, which
    does not exist in the solvated channel. Keep the analysis close to the
    groups and do not extrapolate outward.

  * For a strictly paired comparison between two structures, prefer
    --mesh N, which evaluates on a regular cylindrical mesh (interpolated from
    the QM grid) so both structures are sampled at corresponding points.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

BOHR = 0.52917721092          # Angstrom per bohr
HARTREE_KCAL = 627.509474     # kcal/mol per hartree

# ──────────────────────────────────────────────────────────────────────
#  Figure styling
# ──────────────────────────────────────────────────────────────────────

_FONT_SCALE = 1.0


def FS(size):
    """Scale a font size by the current --fontscale."""
    return size * _FONT_SCALE


def set_font_scale(scale):
    """
    Scales every text element in the figures. Base sizes are tuned for a
    full-width figure; raise this when the figures will be shrunk into a
    manuscript column or a slide, so the labels stay readable.
    """
    global _FONT_SCALE
    _FONT_SCALE = float(scale)
    import matplotlib
    matplotlib.rcParams.update({
        "font.size":         FS(10),
        "axes.titlesize":    FS(12),
        "axes.labelsize":    FS(11),
        "xtick.labelsize":   FS(10),
        "ytick.labelsize":   FS(10),
        "legend.fontsize":   FS(10),
        "figure.titlesize":  FS(13),
        "axes.linewidth":    1.0 + 0.3 * (_FONT_SCALE - 1),
        "xtick.major.width": 1.0 + 0.3 * (_FONT_SCALE - 1),
        "ytick.major.width": 1.0 + 0.3 * (_FONT_SCALE - 1),
        "savefig.bbox":      "tight",
    })


# ──────────────────────────────────────────────────────────────────────
#  Cylinder names, order and colours (same scheme as the ensemble plots)
# ──────────────────────────────────────────────────────────────────────

SUBUNIT_COLORS = {"a": "#3b6ea5", "b": "#D98A1C", "c": "#2E8B57", "d": "#8E6BB0"}
GIRK1_COLOR = "#C0392B"


def _head(name):
    return str(name).split("/")[0].strip()


def is_girk1(name):
    """True for the GIRK1-associated cylinder: a name like G1 or E141.G1/D173.G1."""
    return _head(name).upper().endswith("G1")


def cyl_color(name, i=0):
    """GIRK1 cylinder red, GIRK2 cylinders by subunit letter (a, b, c, d), as in the ensemble plots."""
    if is_girk1(name):
        return GIRK1_COLOR
    return SUBUNIT_COLORS.get(_head(name)[-1:].lower(), ["#555555", "#999999"][i % 2])


def cyl_order(names):
    """GIRK1 cylinder first, then the GIRK2 cylinders by subunit letter."""
    return sorted(names, key=lambda n: (not is_girk1(n), _head(n)[-1:].lower()))


# ──────────────────────────────────────────────────────────────────────
#  Parsing
# ──────────────────────────────────────────────────────────────────────


def read_resp(path: Path):
    """
    Jaguar RESP-format file:
        line 1 : natoms  npoints
        next natoms lines      : x y z            (bohr)
        next npoints lines     : potential x y z  (hartree/e, bohr)

    Jaguar warns that with many grid points the first line is not in the
    standard 2i5 format; whitespace splitting handles both cases.

    Returns (atom_xyz_A, grid_xyz_A, esp_kcal).
    """
    with open(path) as fh:
        first = fh.readline().split()
        if len(first) < 2:
            raise SystemExit(f"ERROR: unexpected first line in {path}: {first!r}")
        natoms, npoints = int(first[0]), int(first[1])

        atoms = np.empty((natoms, 3))
        for i in range(natoms):
            p = fh.readline().split()
            if len(p) < 3:
                raise SystemExit(f"ERROR: truncated atom block at atom {i+1} in {path}")
            atoms[i] = [float(p[-3]), float(p[-2]), float(p[-1])]

        grid = np.loadtxt(fh)

    if grid.ndim == 1:
        grid = grid[None, :]
    if grid.shape[1] != 4:
        raise SystemExit(f"ERROR: expected 4 columns in the grid block, got {grid.shape[1]}")
    if grid.shape[0] != npoints:
        print(f"  NOTE: header says {npoints} grid points, found {grid.shape[0]} -- using what was found")

    return atoms * BOHR, grid[:, 1:4] * BOHR, grid[:, 0] * HARTREE_KCAL


def read_out_labels(path: Path):
    """
    Pulls the atom labels (C1, H2, C3, ...) and the input geometry from a
    Jaguar .out file. Needed only for --list-ca, which has to know elements.
    Returns (labels, elements, xyz_A).
    """
    lines = Path(path).read_text().split("\n")
    starts = [n for n, l in enumerate(lines) if l.strip() == "Input geometry:"]
    if not starts:
        raise SystemExit(f"ERROR: no 'Input geometry:' block found in {path}")
    labels, xyz = [], []
    for l in lines[starts[0] + 3:]:
        p = l.split()
        if len(p) != 4:
            break
        try:
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            break
        labels.append(p[0])
    labels = np.array(labels)
    elements = np.array([re.match(r"[A-Za-z]+", s).group() for s in labels])
    return labels, elements, np.array(xyz)


# ──────────────────────────────────────────────────────────────────────
#  CA detection
# ──────────────────────────────────────────────────────────────────────

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "K": 2.03}


def neighbours(xyz, elements, i, slack=0.45):
    """Indices bonded to atom i, by covalent-radius sum + slack."""
    ri = COV.get(elements[i], 0.77)
    d = np.linalg.norm(xyz - xyz[i], axis=1)
    cut = np.array([ri + COV.get(e, 0.77) + slack for e in elements])
    out = np.where((d > 0.1) & (d < cut))[0]
    return out


def classify_sidechain(xyz, elements, ca, backbone_exclude):
    """
    Walk outward from CA, skipping backbone N and backbone carbonyl C, and
    report what terminal functional group is reached.
      CARBOXYLATE : C bonded to 2 O, no O-H     (Glu / Asp)
      CARBOXYLIC  : C bonded to 2 O, one O-H    (protonated Glu / Asp)
      AMIDE       : C bonded to 1 O and 1 N(H2) (Asn / Gln)
    """
    seen = set([ca]) | set(backbone_exclude)
    frontier = [n for n in neighbours(xyz, elements, ca)
                if n not in seen and elements[n] == "C"]
    seen |= set(frontier)
    group = "—"
    while frontier:
        nxt = []
        for a in frontier:
            nb = neighbours(xyz, elements, a)
            if elements[a] == "C":
                os_ = [j for j in nb if elements[j] == "O"]
                ns_ = [j for j in nb if elements[j] == "N"]
                if len(os_) == 2:
                    has_oh = any(
                        any(elements[k] == "H" for k in neighbours(xyz, elements, o))
                        for o in os_
                    )
                    return "CARBOXYLIC" if has_oh else "CARBOXYLATE"
                if len(os_) == 1 and len(ns_) == 1:
                    nh = sum(1 for k in neighbours(xyz, elements, ns_[0]) if elements[k] == "H")
                    if nh >= 2:
                        return "AMIDE"
            for j in nb:
                if j not in seen and elements[j] in ("C", "N", "O", "S"):
                    seen.add(j)
                    nxt.append(j)
        frontier = nxt
    return group


def detect_ca(xyz, elements, labels):
    """
    Backbone CA signature: a carbon bonded to at least one N and at least one
    carbonyl carbon (a C that carries an O at bonding distance).
    """
    found = []
    for i, e in enumerate(elements):
        if e != "C":
            continue
        nb = neighbours(xyz, elements, i)
        ns_ = [j for j in nb if elements[j] == "N"]
        if not ns_:
            continue
        carbonyls = []
        for j in nb:
            if elements[j] != "C":
                continue
            if any(elements[k] == "O" and
                   np.linalg.norm(xyz[k] - xyz[j]) < 1.35
                   for k in neighbours(xyz, elements, j)):
                carbonyls.append(j)
        if not carbonyls:
            continue
        side = classify_sidechain(xyz, elements, i, backbone_exclude=ns_ + carbonyls)
        found.append(dict(index=i + 1, label=labels[i], xyz=xyz[i], sidechain=side))
    return found


PDB_RECORDS = (b"ATOM", b"HETATM", b"MODEL", b"CRYST1", b"HEADER", b"TITLE",
               b"REMARK", b"SEQRES", b"EXPDTA", b"COMPND")


def looks_like_pdb(path):
    """
    True if the file is readable, uncompressed, and carries PDB records.
    Used to bypass MDAnalysis for plain PDBs -- its opener attempts bz2 and
    gzip decompression on every file before falling back to plain text, and a
    truncated-stream EOFError there escapes its own error handling.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if head[:3] == b"BZh" or head[:2] == b"\x1f\x8b" or head[:4] == b"PK\x03\x04":
        return False                      # genuinely compressed
    for line in head.split(b"\n"):
        if line[:6].strip() in PDB_RECORDS:
            return True
    return False


def read_pdb_ca(struct, resids):
    """Plain PDB CA reader. Returns {resid: [xyz, ...]} in Angstrom."""
    want = set(resids)
    out = {r: [] for r in resids}
    nlines = 0
    with open(struct, "r", errors="replace") as fh:
        for line in fh:
            nlines += 1
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):          # skip altloc B, C, ...
                continue
            try:
                resid = int(line[22:26])
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            if resid in want:
                out[resid].append(np.array(xyz))
    if nlines == 0:
        raise SystemExit(f"ERROR: {struct} is empty.")
    missing = [r for r in resids if not out[r]]
    if missing:
        raise SystemExit(
            f"ERROR: no CA found for residue(s) {sorted(missing)} in {struct}.\n"
            f"  Read {nlines} lines. Check that the numbering in --groups matches this\n"
            f"  file, and that it is the frame the QM cluster was carved from.")
    return out


def load_reference_ca(struct, coords, resids):
    """
    Gets the CA coordinate(s) of each residue id in `resids` from a reference
    structure (the full MD system that the QM cluster was carved out of).

    --struct may be:
      * a PDB on its own  -> read directly, no MDAnalysis needed, or
      * a topology (prmtop etc.) with --coords (rst / ncrst / nc) -> MDAnalysis.

    A residue id can legitimately match several CA atoms when the PDB numbers
    residues PER CHAIN rather than continuously. All candidates are returned
    and map_resid_to_cluster() picks the one that actually sits in the QM
    cluster, so per-chain numbering needs no extra flags.

    Returns {resid: [xyz, ...]} in Angstrom.
    """
    sp = Path(struct)
    if not sp.is_file():
        raise SystemExit(f"ERROR: --struct file not found: {struct}")
    if sp.stat().st_size == 0:
        raise SystemExit(f"ERROR: --struct file is empty (0 bytes): {struct}")

    # Plain PDB and no separate coordinate file -> read it ourselves.
    if coords is None and looks_like_pdb(sp):
        return read_pdb_ca(sp, resids)

    try:
        import MDAnalysis as mda
    except ImportError:
        raise SystemExit(
            f"ERROR: {struct} does not read as a plain PDB, and MDAnalysis is not\n"
            f"  installed to try other formats. Either the file is a topology that\n"
            f"  needs --coords, or it is compressed, truncated or corrupt.\n"
            f"  Check it with:  head -3 '{struct}'  and  file '{struct}'")

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(str(sp), coords) if coords else mda.Universe(str(sp))
    except Exception as e:
        if looks_like_pdb(sp):
            print(f"  NOTE: MDAnalysis failed on {sp.name} ({type(e).__name__}); "
                  f"falling back to the built-in PDB reader.")
            return read_pdb_ca(sp, resids)
        raise SystemExit(
            f"ERROR: MDAnalysis could not open --struct {struct}"
            f"{' with --coords ' + str(coords) if coords else ''}:\n"
            f"  {type(e).__name__}: {e}\n"
            f"  If the file is a PDB, check it is not truncated or compressed.")

    out = {}
    for r in resids:
        sel = u.select_atoms(f"resid {r} and name CA")
        if len(sel) == 0:
            raise SystemExit(
                f"ERROR: 'resid {r} and name CA' matched nothing in {struct}. "
                f"Check that your residue numbering matches this structure.")
        out[r] = [np.array(q, dtype=float) for q in sel.positions]
    return out


def map_resid_to_cluster(ref_ca, cluster_xyz, tol=0.6):
    """
    Matches each reference CA to the nearest atom of the QM cluster by position.
    The cluster was carved from the same frame, so the match should be near-exact
    (a fraction of an Angstrom).

    When a residue id has several candidate CAs (per-chain PDB numbering), the
    candidate closest to any cluster atom wins -- the cluster coordinates do the
    disambiguation, so no chain flag is needed.

    Returns {resid: (cluster_index_1based, distance, n_candidates)}.
    """
    out = {}
    for resid, candidates in ref_ca.items():
        best = None
        for p in candidates:
            d = np.linalg.norm(cluster_xyz - p, axis=1)
            j = int(np.argmin(d))
            if best is None or d[j] < best[1]:
                best = (j + 1, float(d[j]))
        out[resid] = (best[0], best[1], len(candidates))

    multi = {r: v for r, v in out.items() if v[2] > 1}
    if multi:
        print(f"\n  NOTE: {len(multi)} residue id(s) matched more than one CA "
              f"(per-chain numbering). Resolved by proximity to the QM cluster:")
        for r, (j, d, n) in sorted(multi.items()):
            print(f"    resid {r}: {n} candidates -> cluster atom {j} at {d:.3f} A")

    bad = {r: v for r, v in out.items() if v[1] > tol}
    if bad:
        print(f"\n  WARNING: these residues matched no cluster atom within {tol} A:")
        for r, (j, d, n) in sorted(bad.items()):
            print(f"    resid {r}: nearest cluster atom {j} at {d:.3f} A")
        print("  Likely causes: the reference structure is not the frame the cluster")
        print("  was carved from, different residue numbering, or the residue is not")
        print("  part of the cluster. Raise --match-tol only if you know why.")
    return out


def parse_group_members(group_args):
    """
    Parses --groups 'A:748,780' into [(name, [res1, res2]), ...] without
    deciding yet which member is the Glu end and which is the Asn end.
    """
    out = []
    for g in group_args:
        if ":" not in g:
            raise SystemExit(f"ERROR: bad group {g!r}, expected NAME:RES1,RES2 e.g. A:748,780")
        name, rest = g.split(":", 1)
        try:
            members = [int(x) for x in rest.replace(",", " ").split()]
        except ValueError:
            raise SystemExit(f"ERROR: non-integer residue id in group {g!r}")
        if len(members) != 2:
            raise SystemExit(f"ERROR: group {name} has {len(members)} residues, expected 2")
        out.append((name.strip(), members))
    return out


def classify_groups(raw_groups, glu_list, asn_list, mapping, cluster_xyz):
    """
    Decides, for every group, which residue is the UPPER-ring (Glu) end and
    which is the LOWER-ring (Asn/Asp) end.

    If --glu / --asn were given, membership in those lists decides, and a group
    that does not contain exactly one of each is an error -- the user stated an
    intent and it was not met.

    If neither was given, the assignment is made geometrically: within each
    group the residue whose CA sits at higher z is the upper-ring end. This
    works for any channel and any numbering, and correctly handles a lower-ring
    Asp (which a residue-name rule would misclassify as a Glu).

    Returns [(name, glu_resid, asn_resid), ...].
    """
    out = []
    if glu_list or asn_list:
        glu_set, asn_set = set(glu_list or []), set(asn_list or [])
        for name, members in raw_groups:
            g_res = [r for r in members if r in glu_set]
            a_res = [r for r in members if r in asn_set]
            if len(g_res) != 1 or len(a_res) != 1:
                raise SystemExit(
                    f"ERROR: group {name} ({members}) does not contain exactly one "
                    f"residue from --glu and one from --asn.\n"
                    f"  --glu {sorted(glu_set)}\n  --asn {sorted(asn_set)}\n"
                    f"  Either correct those lists for this system, or drop both "
                    f"flags entirely and the upper/lower end of each pair will be "
                    f"assigned from the CA z-coordinates.")
            out.append((name, g_res[0], a_res[0]))
        return out

    print("\n  --glu / --asn not given: assigning each pair by CA z-coordinate "
          "(higher z = upper ring).")
    print(f"  {'subunit':<10}{'upper (Glu) end':>17}{'z':>9}   {'lower (Asn) end':>17}{'z':>9}")
    print("  " + "-" * 66)
    for name, members in raw_groups:
        zs = {r: float(cluster_xyz[mapping[r][0] - 1][2]) for r in members}
        hi, lo = sorted(members, key=lambda r: -zs[r])
        if abs(zs[hi] - zs[lo]) < 1.0:
            print(f"  WARNING: group {name} members differ by only "
                  f"{abs(zs[hi]-zs[lo]):.2f} A in z -- check this assignment.")
        print(f"  {name:<10}{hi:>17}{zs[hi]:>9.2f}   {lo:>17}{zs[lo]:>9.2f}")
        out.append((name, hi, lo))
    return out


def parse_groups(group_args, glu_list, asn_list):
    """
    Parses --groups 'A:748,780' 'B:424,456' ... into
    [(name, glu_resid, asn_resid), ...].

    Order inside each group does not matter; membership of --glu / --asn decides
    which residue is the Glu end and which is the Asn/Asp end.
    """
    glu_set, asn_set = set(glu_list), set(asn_list)
    out = []
    for g in group_args:
        if ":" not in g:
            raise SystemExit(f"ERROR: bad group {g!r}, expected NAME:RES1,RES2 e.g. A:748,780")
        name, rest = g.split(":", 1)
        try:
            members = [int(x) for x in rest.replace(",", " ").split()]
        except ValueError:
            raise SystemExit(f"ERROR: non-integer residue id in group {g!r}")
        if len(members) != 2:
            raise SystemExit(f"ERROR: group {name} has {len(members)} residues, expected 2")
        g_res = [r for r in members if r in glu_set]
        a_res = [r for r in members if r in asn_set]
        if len(g_res) != 1 or len(a_res) != 1:
            raise SystemExit(
                f"ERROR: group {name} ({members}) does not contain exactly one residue "
                f"from --glu and one from --asn.\n  --glu {sorted(glu_set)}\n  --asn {sorted(asn_set)}")
        out.append((name.strip(), g_res[0], a_res[0]))
    return out


def suggest_pairs(cas):
    """
    Guesses the per-subunit pairing by splitting the detected CA atoms into an
    upper and a lower ring on z, then matching each upper CA to the nearest
    lower CA in the xy plane (greedy, one-to-one). This is a convenience only --
    always eyeball the printed distances before using the suggestion.
    """
    if len(cas) < 2:
        return
    zs = np.array([c["xyz"][2] for c in cas])
    split = 0.5 * (zs.max() + zs.min())
    upper = sorted([c for c in cas if c["xyz"][2] >= split], key=lambda d: d["index"])
    lower = [c for c in cas if c["xyz"][2] < split]
    if not upper or not lower:
        print("\n(Could not split the CA atoms into two rings on z -- pair them manually.)")
        return

    taken, out = set(), []
    for u in upper:
        best, bestd = None, np.inf
        for l in lower:
            if l["index"] in taken:
                continue
            d = float(np.linalg.norm(u["xyz"][:2] - l["xyz"][:2]))
            if d < bestd:
                best, bestd = l, d
        if best is None:
            continue
        taken.add(best["index"])
        out.append((u, best, bestd))

    print(f"\nSuggested pairing (upper ring z >= {split:.2f}, matched on xy distance):\n")
    print(f"  {'upper CA':<10}{'sidechain':<14}{'lower CA':<10}{'sidechain':<14}{'xy dist':>9}")
    print("  " + "-" * 58)
    for u, l, d in out:
        flag = "   <-- check" if d > 4.0 else ""
        print(f"  {u['label']:<10}{u['sidechain']:<14}{l['label']:<10}"
              f"{l['sidechain']:<14}{d:>9.2f}{flag}")
    print("\n  --pairs " + " ".join(f"{u['index']}:{l['index']}" for u, l, _ in out))


# ──────────────────────────────────────────────────────────────────────
#  Cylinder selection
# ──────────────────────────────────────────────────────────────────────


def cylinder_mask(points, p_start, p_end, radius, extend, mode="ca-ca"):
    """
    Boolean mask of grid points inside the cylinder.

    mode 'ca-ca' : axis along (p_end - p_start), extended by `extend` beyond
                   each endpoint along that same axis.
    mode 'z'     : axis parallel to +z through the xy midpoint of the two
                   endpoints; axial span [min(z)-extend, max(z)+extend].
    """
    if mode == "z":
        centre_xy = 0.5 * (p_start[:2] + p_end[:2])
        zlo = min(p_start[2], p_end[2]) - extend
        zhi = max(p_start[2], p_end[2]) + extend
        radial = np.linalg.norm(points[:, :2] - centre_xy, axis=1)
        axial = points[:, 2]
        return (radial <= radius) & (axial >= zlo) & (axial <= zhi), (zlo, zhi), zhi - zlo

    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        raise SystemExit("ERROR: the two CA atoms of a pair are at the same position")
    u = vec / length
    rel = points - p_start
    t = rel @ u
    radial = np.linalg.norm(rel - np.outer(t, u), axis=1)
    return (radial <= radius) & (t >= -extend) & (t <= length + extend), \
           (-extend, length + extend), length + 2 * extend


def describe(values):
    if values.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, q1=np.nan, q3=np.nan)
    return dict(
        n=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        min=float(np.min(values)),
        max=float(np.max(values)),
        q1=float(np.percentile(values, 25)),
        q3=float(np.percentile(values, 75)),
    )


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resp", required=True, help="Jaguar .resp grid file")
    ap.add_argument("--out", default=None,
                    help="Matching Jaguar .out (needed for --list-ca; also used "
                         "to cross-check that atom order matches the .resp)")
    ap.add_argument("--list-ca", action="store_true",
                    help="Detect and print candidate CA atoms, then exit")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="GLU_CA:ASN_CA atom index pairs, 1-based, e.g. 7:97 30:142")
    ap.add_argument("--names", nargs="+", default=None,
                    help="Optional label per pair, same order as --pairs")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="Subunit definitions by RESIDUE NUMBER, e.g. "
                         "A:748,780 B:424,456 C:1073,1105 G1:99,131 . "
                         "Order within a group does not matter. Needs --struct.")
    ap.add_argument("--struct", default=None,
                    help="Reference structure the QM cluster was carved from. "
                         "A PDB of the frame works on its own; a prmtop needs --coords. "
                         "Required with --groups.")
    ap.add_argument("--coords", default=None,
                    help="Coordinate file for --struct (rst/ncrst/nc frame). "
                         "OMIT THIS if --struct is a PDB -- it already has coordinates.")
    ap.add_argument("--glu", nargs="+", type=int, default=None,
                    help="Upper-ring (Glu) residue ids. Optional: if omitted, the "
                         "upper/lower end of each --groups pair is assigned from the "
                         "CA z-coordinates, which works for any system.")
    ap.add_argument("--asn", nargs="+", type=int, default=None,
                    help="Lower-ring (Asn/Asp) residue ids. Optional, see --glu.")
    ap.add_argument("--match-tol", type=float, default=0.6,
                    help="Max CA-to-cluster-atom distance accepted when mapping "
                         "residues onto the QM cluster (Angstrom, default 0.6)")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="Cylinder radius in Angstrom (default 3.0)")
    ap.add_argument("--extend", type=float, default=3.0,
                    help="Extension beyond each CA along the axis, Angstrom (default 3.0)")
    ap.add_argument("--axis", choices=["ca-ca", "z"], default="ca-ca",
                    help="Cylinder axis definition (default ca-ca)")
    ap.add_argument("--label", default="esp_cylinders",
                    help="Prefix for all output file NAMES")
    ap.add_argument("--title", default=None,
                    help="Short title printed on the figures, e.g. 'GIRK1/2, frame 5606' "
                         "(default: --label). Settings go in the figure legend, not the title.")
    ap.add_argument("--counts", action="store_true",
                    help="Histogram y-axis as raw counts instead of %% of points "
                         "(default %% of points, as in the ensemble histograms)")
    ap.add_argument("--fontscale", type=float, default=1.5,
                    help="Multiplier for every font size in the figures "
                         "(default 1.5). Raise it if you shrink the figures into "
                         "a manuscript column or a slide.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure resolution (default 200)")
    ap.add_argument("--outdir", "--folder", "-o", dest="outdir", default=None,
                    help="Folder to save everything into. Created if it does not "
                         "exist. Defaults to a folder named after --label. "
                         "Pass '.' to write into the current directory instead.")
    ap.add_argument("--min-points", type=int, default=200,
                    help="Warn if a cylinder holds fewer grid points than this")
    ap.add_argument("--save-raw", action="store_true",
                    help="Also write every in-cylinder ESP value to CSV")
    ap.add_argument("--bins", type=int, default=40,
                    help="Bins for the auto-scaled histogram figure (default 40)")
    ap.add_argument("--xlim", nargs=2, type=float, default=[-300.0, -50.0],
                    metavar=("LO", "HI"),
                    help="ESP range for the SECOND, fixed-axis histogram figure "
                         "(default -300 -50). Use the same value for every cluster "
                         "you intend to compare.")
    ap.add_argument("--binwidth", type=float, default=5.0,
                    help="Bin width in kcal/mol for the fixed-axis figure (default 5). "
                         "Fixed width means identical bin edges across clusters.")
    ap.add_argument("--ylim", type=float, default=None,
                    help="Common y ceiling for the fixed-axis panels, in %% of points "
                         "(or counts with --counts), e.g. 10. Set the same "
                         "value for every cluster so bar heights are comparable too; "
                         "omit to scale to the tallest bar in each figure.")
    ap.add_argument("--no-fixed", action="store_true",
                    help="Skip the fixed-axis histogram figure")
    ap.add_argument("--no-plots", action="store_true", help="Skip the figures")
    args = ap.parse_args()
    set_font_scale(args.fontscale)

    resp_path = Path(args.resp)
    if not resp_path.is_file():
        raise SystemExit(f"ERROR: .resp file not found: {resp_path}")
    outdir = Path(args.outdir) if args.outdir is not None else Path(args.label)
    if not args.list_ca:                       # --list-ca writes nothing
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {outdir.resolve()}")

    atoms, grid_xyz, esp = read_resp(resp_path)
    print(f"Read {resp_path.name}")
    print(f"  atoms:       {len(atoms)}")
    print(f"  grid points: {len(grid_xyz)}")
    print(f"  ESP range:   {esp.min():.1f} to {esp.max():.1f} kcal/mol per +1e")
    print(f"  grid z span: {grid_xyz[:,2].min():.1f} to {grid_xyz[:,2].max():.1f} A")

    # ---- --list-ca ----
    if args.list_ca:
        if not args.out:
            raise SystemExit("ERROR: --list-ca needs --out (the .out file carries the element labels)")
        labels, elements, xyz_out = read_out_labels(Path(args.out))
        if len(xyz_out) != len(atoms):
            print(f"  WARNING: .out has {len(xyz_out)} atoms, .resp has {len(atoms)}")
        shift = np.abs(xyz_out[:len(atoms)] - atoms).max() if len(xyz_out) >= len(atoms) else np.nan
        print(f"  max coord mismatch .out vs .resp: {shift:.4f} A "
              f"({'OK, same atom order' if shift < 0.01 else 'CHECK -- order may differ'})")

        cas = detect_ca(xyz_out, elements, labels)
        print(f"\nDetected {len(cas)} backbone CA atoms:\n")
        print(f"  {'idx':>5}  {'label':<7} {'x':>9} {'y':>9} {'z':>9}   sidechain")
        print("  " + "-" * 60)
        for c in sorted(cas, key=lambda d: -d["xyz"][2]):
            x, y, z = c["xyz"]
            print(f"  {c['index']:>5}  {c['label']:<7} {x:9.3f} {y:9.3f} {z:9.3f}   {c['sidechain']}")
        print("\nPair a CARBOXYLATE CA (Glu/Asp) with an AMIDE CA (Asn) from the same")
        print("subunit and pass them as --pairs GLU_IDX:ASN_IDX ...")
        suggest_pairs(cas)
        return

    if args.groups and args.pairs:
        raise SystemExit("ERROR: use --groups or --pairs, not both")

    if args.groups:
        if not args.struct:
            raise SystemExit("ERROR: --groups needs --struct (the structure the cluster came from)")
        raw_groups = parse_group_members(args.groups)
        resids = sorted({r for _, members in raw_groups for r in members})
        ref_ca = load_reference_ca(args.struct, args.coords, resids)
        mapping = map_resid_to_cluster(ref_ca, atoms, tol=args.match_tol)
        groups = classify_groups(raw_groups, args.glu, args.asn, mapping, atoms)

        print("\nResidue -> QM cluster atom mapping:")
        print(f"  {'subunit':<8}{'Glu res':>9}{'-> atom':>9}{'dist':>8}   "
              f"{'Asn res':>9}{'-> atom':>9}{'dist':>8}")
        print("  " + "-" * 64)
        pairs, names = [], []
        for name, g_res, a_res in groups:
            gi, gd = mapping[g_res][0], mapping[g_res][1]
            ai, ad = mapping[a_res][0], mapping[a_res][1]
            print(f"  {name:<8}{g_res:>9}{gi:>9}{gd:>8.3f}   {a_res:>9}{ai:>9}{ad:>8.3f}")
            pairs.append((gi, ai))
            names.append(name)
    else:
        if not args.pairs:
            raise SystemExit("ERROR: pass --groups (with --struct), or --pairs, "
                             "or --list-ca to find the indices first")
        pairs = []
        for s in args.pairs:
            if ":" not in s:
                raise SystemExit(f"ERROR: bad pair {s!r}, expected GLU_CA:ASN_CA e.g. 7:97")
            a, b = s.split(":")
            pairs.append((int(a), int(b)))
        names = args.names if args.names else [f"cyl{i+1}" for i in range(len(pairs))]
        if len(names) != len(pairs):
            raise SystemExit(f"ERROR: {len(names)} --names for {len(pairs)} --pairs")

    for a, b in pairs:
        for idx in (a, b):
            if not (1 <= idx <= len(atoms)):
                raise SystemExit(f"ERROR: atom index {idx} outside 1..{len(atoms)}")

    print(f"\nCylinders: radius {args.radius} A, extend {args.extend} A beyond each CA, "
          f"axis mode '{args.axis}'\n")

    results, raw = [], {}
    for (ia, ib), name in zip(pairs, names):
        p_glu, p_asn = atoms[ia - 1], atoms[ib - 1]
        mask, span, length = cylinder_mask(grid_xyz, p_glu, p_asn,
                                           args.radius, args.extend, args.axis)
        vals = esp[mask]
        raw[name] = vals
        stats = describe(vals)
        volume = np.pi * args.radius ** 2 * length
        ca_sep = float(np.linalg.norm(p_asn - p_glu))
        stats.update(name=name, glu_ca=ia, asn_ca=ib,
                     ca_separation_A=round(ca_sep, 3),
                     cyl_length_A=round(float(length), 3),
                     cyl_volume_A3=round(float(volume), 1),
                     pts_per_A3=round(stats["n"] / volume, 3) if volume else np.nan)
        results.append(stats)

        flag = "  <-- SPARSE" if stats["n"] < args.min_points else ""
        print(f"  {name:<22} CA{ia}->CA{ib}  sep {ca_sep:5.2f} A  "
              f"len {length:5.2f} A  n={stats['n']:>6}  "
              f"mean={stats['mean']:8.2f}  median={stats['median']:8.2f}{flag}")

    sparse = [r["name"] for r in results if r["n"] < args.min_points]
    if sparse:
        print(f"\n  WARNING: sparse sampling in {', '.join(sparse)} (< {args.min_points} points).")
        print("  The Jaguar grid sits on vdW shells, so lumen coverage is incidental.")
        print("  Means from sparse cylinders are not safely comparable between structures.")

    # ---- table ----
    cols = ["name", "glu_ca", "asn_ca", "ca_separation_A", "cyl_length_A",
            "cyl_volume_A3", "n", "pts_per_A3", "mean", "median", "std",
            "min", "max", "q1", "q3"]
    table_path = outdir / f"{args.label}_esp_stats.csv"
    with open(table_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in results:
            fh.write(",".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    print("\n" + "=" * 94)
    print("ESP in cylinders (kcal/mol per +1e)")
    print("=" * 94)
    hdr = f"{'cylinder':<22}{'n':>8}{'mean':>10}{'median':>10}{'std':>9}{'min':>10}{'max':>10}{'Q1':>10}{'Q3':>10}"
    print(hdr)
    print("-" * 94)
    for r in results:
        print(f"{r['name']:<22}{r['n']:>8}{r['mean']:>10.2f}{r['median']:>10.2f}"
              f"{r['std']:>9.2f}{r['min']:>10.2f}{r['max']:>10.2f}{r['q1']:>10.2f}{r['q3']:>10.2f}")
    pooled = np.concatenate([v for v in raw.values() if v.size]) if any(
        v.size for v in raw.values()) else np.array([])
    if pooled.size:
        p = describe(pooled)
        print("-" * 94)
        print(f"{'POOLED (4 cylinders)':<22}{p['n']:>8}{p['mean']:>10.2f}{p['median']:>10.2f}"
              f"{p['std']:>9.2f}{p['min']:>10.2f}{p['max']:>10.2f}{p['q1']:>10.2f}{p['q3']:>10.2f}")
    print("=" * 94)
    print(f"\nWrote {table_path}")

    if args.save_raw:
        raw_path = outdir / f"{args.label}_esp_raw.csv"
        with open(raw_path, "w") as fh:
            fh.write("cylinder,esp_kcal_per_e\n")
            for name, vals in raw.items():
                for v in vals:
                    fh.write(f"{name},{v:.4f}\n")
        print(f"Wrote {raw_path}")

    if args.no_plots:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [(n, v) for n, v in raw.items() if v.size]
    if not present:
        print("\nNo cylinder contained any grid points -- nothing to plot.")
        return

    lo = min(v.min() for _, v in present)
    hi = max(v.max() for _, v in present)
    bin_edges = np.linspace(lo, hi, args.bins + 1)

    # ---- histograms ----
    title = args.title or args.label
    order = cyl_order(list(raw))

    def draw_histograms(edges, path, subtitle, xlim=None, ylim=None):
        """
        2x2 panel of per-cylinder histograms, GIRK1 cylinder first.

        edges  : bin edges, shared by all panels
        xlim   : (lo, hi) applied to every panel, or None to let data decide
        ylim   : common y ceiling for every panel (%% of points, or counts with
                 --counts), or None to use the tallest bar in THIS figure
        """
        ncol = 2
        nrow = int(np.ceil(len(raw) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(11 * (0.9 + 0.1 * _FONT_SCALE), 4.0 * nrow),
                                 squeeze=False, layout="constrained")
        ymax = 0
        for ax, name in zip(axes.ravel(), order):
            vals = raw[name]
            if vals.size:
                w = None if args.counts else np.full(vals.size, 100.0 / vals.size)
                counts, _, _ = ax.hist(vals, bins=edges, weights=w, color="#3b6ea5", alpha=0.85,
                                       edgecolor="white", linewidth=0.3)
                ymax = max(ymax, counts.max())
                ax.axvline(np.mean(vals), color="crimson", linestyle="--", linewidth=1.5,
                           label=f"mean {np.mean(vals):.1f}")
                ax.axvline(np.median(vals), color="darkorange", linestyle="-.", linewidth=1.5,
                           label=f"median {np.median(vals):.1f}")
                ax.legend(fontsize=FS(8), loc="best")
            else:
                ax.text(0.5, 0.5, "no grid points", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(name, fontsize=FS(11))
            ax.set_xlabel("ESP (kcal/mol per +1e)", fontsize=FS(9))
            ax.set_ylabel("count" if args.counts else "% of points", fontsize=FS(9))
            ax.grid(alpha=0.25)
            if xlim is not None:
                ax.set_xlim(*xlim)
        top = ylim if ylim is not None else (ymax * 1.18 if ymax else None)
        if top:
            for ax in axes.ravel()[:len(raw)]:
                ax.set_ylim(0, top)
        for ax in axes.ravel()[len(raw):]:
            ax.axis("off")
        fig.suptitle(subtitle, fontsize=FS(12))
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {path}")

    # (a) auto-scaled to this cluster's own data
    draw_histograms(
        bin_edges,
        outdir / f"{args.label}_esp_histograms.png",
        f"{title}\nQM ESP over the cylinder points")

    # (b) fixed axes, so figures from DIFFERENT clusters can be compared directly.
    #     Identical bin edges and identical limits everywhere, which is what makes
    #     a shift between two systems visible rather than hidden by autoscaling.
    if not args.no_fixed:
        flo, fhi = args.xlim
        fixed_edges = np.arange(flo, fhi + args.binwidth, args.binwidth)
        draw_histograms(
            fixed_edges,
            outdir / f"{args.label}_esp_histograms_fixed.png",
            f"{title}\nQM ESP over the cylinder points",
            xlim=(flo, fhi), ylim=args.ylim)

    # ---- boxplots ----
    fig, ax = plt.subplots(figsize=((2.2 * len(raw) + 3) * (0.9 + 0.1 * _FONT_SCALE), 6.6))
    data = [raw[n] if raw[n].size else np.array([np.nan]) for n in order]
    ticks = []
    for name in order:
        v = raw[name]
        ticks.append(f"{name}\nmean {np.mean(v):.1f}" if v.size else f"{name}\n(no points)")
    kw = dict(showmeans=True, patch_artist=True,
              medianprops=dict(color="black", linewidth=1.6),
              meanprops=dict(marker="D", markerfacecolor="crimson",
                             markeredgecolor="crimson", markersize=6 * (0.6 + 0.4 * _FONT_SCALE)))
    try:                                   # matplotlib >= 3.9
        bp = ax.boxplot(data, tick_labels=ticks, **kw)
    except TypeError:                      # matplotlib < 3.9
        bp = ax.boxplot(data, labels=ticks, **kw)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(GIRK1_COLOR if is_girk1(name) else "#D98A1C")
        patch.set_alpha(0.55)
    ax.set_ylabel("ESP (kcal/mol per +1e)", fontsize=FS(12))
    ax.set_title(f"{title}\nQM ESP per cylinder", fontsize=FS(12))
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=FS(9))
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="black", linewidth=1.6, label="median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="crimson",
               markeredgecolor="crimson", markersize=6 * (0.6 + 0.4 * _FONT_SCALE), label="mean"),
    ], fontsize=FS(9), loc="best")
    fig.tight_layout()
    box_path = outdir / f"{args.label}_esp_boxplot.png"
    fig.savefig(box_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {box_path}")


if __name__ == "__main__":
    main()

'''

python3 esp_cylinders.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
        --xlim -300 -50 --binwidth 5 --ylim 1300 \
    --label G12_frame5606 --folder ./results/G12/RUN3/frame5606 --save-raw

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --label G12_frame5606 --folder ./results/G12/RUN3/frame5606

    python3 esp_cylinders.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
        --xlim -300 -50 --binwidth 5 --ylim 1300 \
    --label G12_frame5606_D173N --folder ./results/G12/RUN3/frame5606_mutated --save-raw

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --label G12_frame5606_D173N --folder ./results/G12/RUN3/frame5606_mutated

python3 esp_cylinders.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
       --xlim -300 -50 --binwidth 5 --ylim 1300 \
    --label G12_frame5606_E141Q --folder ./results/G12/RUN3/frame5606_E141Q --save-raw



python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --label G12_frame5606_E141Q --folder ./results/G12/RUN3/frame5606_E141Q

python3 esp_cylinders.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/frame814_3ions.prj/jobs/jag_frame814_spe_B3LYP-D3_LACVPs/jag_frame814_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/pdbs/frame814_K.pdb \
    --groups A:98,130 B:754,786 C:426,458 D:1082,1114 \
        --xlim -300 -50 --binwidth 5 --ylim 1300 \
    --label G2_frame814 --folder ./results/G2/RUN14/frame814 --save-raw


python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/frame814_3ions.prj/jobs/jag_frame814_spe_B3LYP-D3_LACVPs/jag_frame814_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/pdbs/frame814_K.pdb \
    --groups A:98,130 B:754,786 C:426,458 D:1082,1114 \
    --label G2_frame814 --folder ./results/G2/RUN14/frame814



        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
python3 esp_cylinders.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/frame6220.prj/jobs/jag_frame6220_spe_B3LYP-D3_LACVPs/jag_frame6220_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/pdbs/frame6220_K.pdb \
    --groups A:422,454 B:98,130 C:747,779 G1:1073,1105 \
       --xlim -300 -50 --binwidth 5 --ylim 1300 \
    --label G12_E141Q_frame6220 --folder ./results/G12_E141Q/RUN5/frame6220 --save-raw
'''
