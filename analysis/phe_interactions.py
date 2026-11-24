"""
analysis/phe_interactions.py
============================
Chemical interactions of the gate Phe rings, frame by frame, used by
analysis/bd_jump_residues.py (folder interactions/ of every mode).

A Phe side chain has no charge and no N-H / O-H, so it forms no salt bridge
and no classical hydrogen bond. What it can do with its ring:

  pi-pi      ring of another PHE / TYR / TRP / HIS
             centroid-centroid distance d and angle between the ring planes:
               parallel / offset stacked  d <= 5.5 A and angle < 30 deg
               tilted                     d <= 6.0 A and 30-60 deg
               T-shaped (edge-to-face)    d <= 6.5 A and angle > 60 deg
  S...pi     sulfur of CYS (SG) or MET (SD) within 5.5 A of the ring centroid
             (sulfur-aromatic interaction; theta = angle between the ring
             normal and the centroid->S line, small = S over the ring face)
  S-H...pi   CYS thiol H pointing at the ring: H...centroid <= 3.2 A and
             angle S-H...centroid >= 120 deg (needs the HG / HG1 atom)
  CH-pi      a methyl carbon (ALA CB, VAL CG1/CG2, LEU CD1/CD2, ILE CG2/CD1,
             THR CG2, MET CE) over the ring face: C...centroid <= 4.5 A and
             theta <= 40 deg
  cation-pi  LYS NZ or ARG CZ within 6.0 A of the ring centroid

Partners are taken within 15 A of a gate ring in the first frame (ring
centroid for aromatics, the atom itself otherwise); the gate rings move by
at most a few A, so nothing within reach is missed.
"""
import numpy as np
import pandas as pd

RINGS = {  # ring atoms in cyclic order
    "PHE": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TYR": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TRP": ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
}
for _h in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
    RINGS[_h] = ("CG", "ND1", "CE1", "NE2", "CD2")
SULFUR = {"CYS": "SG", "CYX": "SG", "CYM": "SG", "MET": "SD"}
THIOL_H = ("HG", "HG1")
METHYL = {"ALA": ("CB",), "VAL": ("CG1", "CG2"), "LEU": ("CD1", "CD2"),
          "ILE": ("CG2", "CD1"), "THR": ("CG2",), "MET": ("CE",)}
CATION = {"LYS": "NZ", "ARG": "CZ"}

PI_PARALLEL, PI_TILTED, PI_T = 5.5, 6.0, 6.5
S_CUT, SH_CUT, SH_ANGLE = 5.5, 3.2, 120.0
CH_CUT, CH_THETA = 4.5, 40.0
CAT_CUT = 6.0
KINDS = ("pi-pi", "S...pi", "S-H...pi", "CH-pi", "cation-pi")


def _ring_ag(res):
    names = RINGS.get(res.resname)
    if not names:
        return None
    ag = [res.atoms.select_atoms(f"name {n}") for n in names]
    if any(len(a) != 1 for a in ag):
        return None
    out = ag[0]
    for a in ag[1:]:
        out = out + a
    return out


def _centre_normal(P):
    """P (n, k, 3) ring atoms in cyclic order -> centroids (n, 3), unit normals (n, 3)."""
    c = P.mean(1)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, -1] - P[:, 0])
    return c, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _centres(ags):
    """centroids and unit normals of rings with different atom counts (His has 5)."""
    c = np.empty((len(ags), 3))
    n = np.empty((len(ags), 3))
    by_size = {}
    for i, ag in enumerate(ags):
        by_size.setdefault(len(ag), []).append(i)
    for idx in by_size.values():
        c[idx], n[idx] = _centre_normal(np.array([ags[i].positions for i in idx]))
    return c, n


def dihedral(p):
    """dihedral angle (deg, -180..180) of 4 points p (4, 3)."""
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def _theta(v, n):
    """angle (deg, 0-90) between vectors v (..., 3) and the ring normal n (broadcast)."""
    d = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(np.abs((v * n).sum(-1)) / d, 0, 1)))


class RingTracker:
    """Collects, frame by frame, the geometry of every gate Phe ring with its
    possible partners. Call update(i) once per frame (any coordinate frame:
    only distances and angles are kept), then summary(b, a, info) per jump."""

    def __init__(self, residues, resmap, gate_resids, nf, cutoff=15.0):
        self.residues, self.resmap, self.nf, self.cutoff = residues, resmap, nf, cutoff
        self.gates = []                    # (gate number in gate_resids order, residue index, ring ag)
        self.chi_ags = []                  # N, CA, CB, CG of each gate Phe (chi1)
        k = 0
        for gr in gate_resids:
            sel = residues.atoms.select_atoms(f"resid {gr}")
            if not len(sel):
                continue
            ri = resmap[sel[0].resindex]
            ag = _ring_ag(residues[ri])
            if ag is not None:
                self.gates.append((k, ri, ag))
                chi = [residues[ri].atoms.select_atoms(f"name {n}") for n in ("N", "CA", "CB", "CG")]
                self.chi_ags.append(sum(chi[1:], chi[0]) if all(len(c) == 1 for c in chi)
                                    else None)
            k += 1
        self.ready = False
        self.chi1 = np.full((nf, len(self.gates)), np.nan, np.float32)

    # ---- first frame: choose the partners
    def _setup(self):
        G = len(self.gates)
        gpos = np.array([ag.positions for _, _, ag in self.gates])
        gc, _ = _centre_normal(gpos)
        near = lambda X: np.min(np.linalg.norm(X[:, None] - gc[None], axis=2), 1) <= self.cutoff
        arom, sulf, meth, cat = [], [], [], []
        for ri, res in enumerate(self.residues):
            rn = res.resname
            if rn in RINGS:                    # other gate rings too (gate-gate pi-pi)
                ag = _ring_ag(res)
                if ag is not None:
                    arom.append((ri, ag))
            if rn in SULFUR:
                s = res.atoms.select_atoms(f"name {SULFUR[rn]}")
                if len(s) == 1:
                    h = res.atoms.select_atoms("name " + " ".join(THIOL_H)) if rn != "MET" else []
                    sulf.append((ri, s, h[:1] if len(h) else None))
            for n in METHYL.get(rn, ()):
                c = res.atoms.select_atoms(f"name {n}")
                if len(c) == 1:
                    meth.append((ri, c))
            if rn in CATION:
                c = res.atoms.select_atoms(f"name {CATION[rn]}")
                if len(c) == 1:
                    cat.append((ri, c))
        if arom:
            ac, _ = _centres([ag for _, ag in arom])
            arom = [x for x, keep in zip(arom, near(ac)) if keep]
        sulf = [x for x in sulf if near(x[1].positions)[0]]
        meth = [x for x in meth if near(x[1].positions)[0]]
        cat = [x for x in cat if near(x[1].positions)[0]]
        nf = self.nf
        z = lambda n: np.full((nf, G, n), np.nan, np.float32)
        self.arom = dict(res=np.array([r for r, _ in arom], int), ags=[a for _, a in arom],
                         d=z(len(arom)), ang=z(len(arom)))
        self.sulf = dict(res=np.array([r for r, _, _ in sulf], int),
                         s=[s for _, s, _ in sulf], h=[h for _, _, h in sulf],
                         d=z(len(sulf)), theta=z(len(sulf)), hd=z(len(sulf)), hang=z(len(sulf)))
        self.meth = dict(res=np.array([r for r, _ in meth], int), ags=[c for _, c in meth],
                         d=z(len(meth)), theta=z(len(meth)))
        self.cat = dict(res=np.array([r for r, _ in cat], int), ags=[c for _, c in cat],
                        d=z(len(cat)))
        self.ready = True

    def update(self, i):
        if not self.gates:
            return
        if not self.ready:
            self._setup()
        for g, ag in enumerate(self.chi_ags):
            if ag is not None:
                self.chi1[i, g] = dihedral(ag.positions)
        gc, gn = _centre_normal(np.array([ag.positions for _, _, ag in self.gates]))
        A = self.arom
        if len(A["res"]):
            ac, an = _centres(A["ags"])
            A["d"][i] = np.linalg.norm(ac[None] - gc[:, None], axis=2)
            A["ang"][i] = np.degrees(np.arccos(np.clip(np.abs(gn @ an.T), 0, 1)))
            for g, (_, ri, _) in enumerate(self.gates):      # a ring with itself
                A["d"][i, g, A["res"] == ri] = np.nan
        S = self.sulf
        if len(S["res"]):
            sp = np.array([s.positions[0] for s in S["s"]])
            v = sp[None] - gc[:, None]
            S["d"][i] = np.linalg.norm(v, axis=2)
            S["theta"][i] = _theta(v, gn[:, None])
            for j, h in enumerate(S["h"]):
                if h is None:
                    continue
                hp = h.positions[0]
                hx = gc - hp                                  # H -> centroid
                hs = sp[j] - hp                               # H -> S
                S["hd"][i, :, j] = np.linalg.norm(hx, axis=1)
                cosang = (hx @ hs) / (np.linalg.norm(hx, axis=1) * np.linalg.norm(hs))
                S["hang"][i, :, j] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        M = self.meth
        if len(M["res"]):
            v = np.array([c.positions[0] for c in M["ags"]])[None] - gc[:, None]
            M["d"][i] = np.linalg.norm(v, axis=2)
            M["theta"][i] = _theta(v, gn[:, None])
        C = self.cat
        if len(C["res"]):
            C["d"][i] = np.linalg.norm(
                np.array([c.positions[0] for c in C["ags"]])[None] - gc[:, None], axis=2)

    # ---- per jump
    def summary(self, b, a, info):
        """One row per gate Phe x partner residue x interaction kind that is
        present in >= 1 % of the before or after frames, or whose partner is
        within 12 A (median, before or after) -- kept so that every gate has a
        value for the equivalent partner of the other gates."""
        if not self.gates or not self.ready:
            return pd.DataFrame()
        rows = []

        def add(g, ri, kind, okb, oka, db, da, extra=None):
            pb, pa = 100 * np.nanmean(okb), 100 * np.nanmean(oka)
            if max(pb, pa) < 1 and not (np.nanmedian(db) <= 12 or np.nanmedian(da) <= 12):
                return
            gk, gri, _ = self.gates[g]
            gi, n = info.iloc[gri], info.iloc[ri]
            rows.append(dict(gate=gi.pdb, gate_resid=int(gi.resid), gate_subunit=gi.subunit,
                             partner=n.pdb, resid=int(n.resid), resname=n.resname,
                             subunit=n.subunit, pos=n.pos, same_subunit=bool(n.chain == gi.chain),
                             interaction=kind, pct_before=pb, pct_after=pa, change_pct=pa - pb,
                             dist_before=np.nanmedian(db), dist_after=np.nanmedian(da),
                             **(extra or {})))

        def geom(ok, ang):
            if not ok.any():
                return ""
            x = ang[ok]
            f = [np.mean(x < 30), np.mean((x >= 30) & (x <= 60)), np.mean(x > 60)]
            return ["parallel", "tilted", "T-shaped"][int(np.argmax(f))] + f" ({100 * max(f):.0f} %)"

        with np.errstate(invalid="ignore"):
            for g in range(len(self.gates)):
                A = self.arom
                for j, ri in enumerate(A["res"]):
                    d, ang = A["d"][:, g, j], A["ang"][:, g, j]
                    if np.all(np.isnan(d)):
                        continue
                    ok = (((d <= PI_PARALLEL) & (ang < 30)) | ((d <= PI_TILTED) & (ang >= 30)
                          & (ang <= 60)) | ((d <= PI_T) & (ang > 60)))
                    add(g, ri, "pi-pi", ok[b], ok[a], d[b], d[a],
                        dict(angle_before=np.nanmedian(ang[b]), angle_after=np.nanmedian(ang[a]),
                             geometry_before=geom(ok[b], ang[b]),
                             geometry_after=geom(ok[a], ang[a])))
                S = self.sulf
                for j, ri in enumerate(S["res"]):
                    d, th = S["d"][:, g, j], S["theta"][:, g, j]
                    ok = d <= S_CUT
                    add(g, ri, "S...pi", ok[b], ok[a], d[b], d[a],
                        dict(angle_before=np.nanmedian(th[b]), angle_after=np.nanmedian(th[a])))
                    if S["h"][j] is not None:
                        hd, hang = S["hd"][:, g, j], S["hang"][:, g, j]
                        ok = (hd <= SH_CUT) & (hang >= SH_ANGLE)
                        add(g, ri, "S-H...pi", ok[b], ok[a], hd[b], hd[a],
                            dict(angle_before=np.nanmedian(hang[b]),
                                 angle_after=np.nanmedian(hang[a])))
                M = self.meth
                for ri in np.unique(M["res"]):
                    cols = np.flatnonzero(M["res"] == ri)
                    d, th = M["d"][:, g, cols], M["theta"][:, g, cols]
                    ok = ((d <= CH_CUT) & (th <= CH_THETA)).any(1)
                    dm = np.nanmin(d, 1)
                    add(g, ri, "CH-pi", ok[b], ok[a], dm[b], dm[a])
                C = self.cat
                for j, ri in enumerate(C["res"]):
                    d = C["d"][:, g, j]
                    ok = d <= CAT_CUT
                    add(g, ri, "cation-pi", ok[b], ok[a], d[b], d[a])
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in ("angle_before", "angle_after", "geometry_before", "geometry_after"):
            if c not in df:
                df[c] = np.nan
        df["abs_chg"] = df.change_pct.abs()
        df = df.sort_values(["gate_subunit", "abs_chg"], ascending=[True, False])
        return df.drop(columns="abs_chg").reset_index(drop=True)


def interaction_name(r, two_lines=False):
    from analysis.bd_jump_residues import res_label
    return res_label(r.resname, r.pos, r.partner, r.subunit, two_lines)


HBC_GIRK2 = 192
LABEL = {"pi-pi": "π–π", "S...pi": "S···π", "S-H...pi": "S–H···π", "CH-pi": "CH–π",
         "cation-pi": "cation–π"}


def plot_interactions(tab, out_png, title, min_pct=5.0, min_change=0.0):
    """2 x 2 grid, one panel per gate Phe: for every partner x interaction with
    >= min_pct before or after and a change of >= min_change points, % frames
    with the interaction before (grey) and after (green gained / red lost);
    under it the residue (GIRK2 numbering + subunit), the interaction type
    (+ geometry for pi-pi) and the change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if tab is None or tab.empty:
        return
    tab = tab.copy()
    tab["pos"] = tab.pos.astype(int)
    keep = ((tab[["pct_before", "pct_after"]].max(axis=1) >= min_pct)
            & (tab.change_pct.abs() >= min_change))                 # min_change 0 = all
    d = tab[keep].copy()
    if d.empty:
        return
    d["passes_filter"] = True
    # same columns in every panel: the union of (position, interaction); a gate
    # without a selected partner there gets its equivalent partner (same position,
    # same interaction, most frames, then closest), drawn in grey italics
    slots = sorted(set(zip(d.pos, d.interaction)), key=lambda s_: (s_[0], KINDS.index(s_[1])
                                                                    if s_[1] in KINDS else 9))
    rest = tab[~keep].copy()
    rest["key"] = rest[["pct_before", "pct_after"]].max(axis=1)
    rest["dmin"] = rest[["dist_before", "dist_after"]].min(axis=1)
    have = set(zip(d.gate, d.pos, d.interaction))
    fill = []
    for gate in tab.gate.unique():
        for p, kind in slots:
            if (gate, p, kind) in have:
                continue
            c = rest[(rest.gate == gate) & (rest.pos == p) & (rest.interaction == kind)]
            if len(c):
                fill.append(c.sort_values(["key", "dmin"], ascending=[False, True]).iloc[[0]])
    if fill:
        f = pd.concat(fill).drop(columns=["key", "dmin"])
        f["passes_filter"] = False
        d = pd.concat([d, f], ignore_index=True)
    nk = {s_: int(d[(d.pos == s_[0]) & (d.interaction == s_[1])].groupby("gate").size().max())
          for s_ in slots}
    x0, x = {}, 0.0
    for s_ in slots:
        x0[s_] = x
        x += nk[s_]
    xmax = max(x, 1)
    gates = tab.drop_duplicates("gate").sort_values("gate_subunit")
    fs = 16
    fig, axes = plt.subplots(2, 2, figsize=(max(16, 1.6 * xmax * 2 + 3), 13), sharey=True)
    for ax, (_, g) in zip(axes.flat, gates.iterrows()):
        s = d[d.gate == g.gate]
        for s_ in slots:
            grp = s[(s.pos == s_[0]) & (s.interaction == s_[1])].sort_values("subunit")
            if grp.empty:                                 # no equivalent residue at all
                ax.annotate(f"{HBC_GIRK2 + s_[0]}\n{LABEL.get(s_[1], s_[1])}\nnone",
                            (x0[s_] + nk[s_] / 2, 0), xytext=(0, -4),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=fs - 2, annotation_clip=False, color="0.6",
                            style="italic", linespacing=1.1)
            off = (nk[s_] - len(grp)) / 2
            for jj, (_, r) in enumerate(grp.iterrows()):
                xc, bw = x0[s_] + off + jj + 0.5, 0.4
                ax.bar(xc - bw / 2, r.pct_before, bw, color="#9aa5b1")
                ax.bar(xc + bw / 2, r.pct_after, bw,
                       color="#2a9d8f" if r.change_pct >= 0 else "#e76f51")
                kind = LABEL.get(r.interaction, r.interaction)
                if r.interaction == "pi-pi":
                    geo = str(r.geometry_after if r.change_pct >= 0 else r.geometry_before)
                    kind += f"\n{geo.split(' (')[0]}" if geo and geo != "nan" else "\n–"
                text = f"{interaction_name(r, two_lines=True)}\n{kind}"
                nl = text.count("\n") + 1
                ax.annotate(text, (xc, 0), xytext=(0, -4), textcoords="offset points",
                            ha="center", va="top", fontsize=fs - 2, annotation_clip=False,
                            linespacing=1.1, color="0.1" if r.passes_filter else "0.5",
                            style="normal" if r.passes_filter else "italic")
                c = round(float(r.change_pct)) + 0
                ax.annotate(f"{c:+d} %", (xc, 0), xytext=(0, -8 - nl * (fs - 2) * 1.25),
                            textcoords="offset points", ha="center", va="top", fontsize=fs,
                            annotation_clip=False, fontweight="bold",
                            color="#1f7a6f" if c > 0 else "#c0503a" if c < 0 else "0.45")
        for s_ in slots[1:]:
            ax.axvline(x0[s_], color="0.9", lw=0.8, zorder=0)
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_title(f"gate {g.gate}  ({g.gate_subunit})", fontsize=fs + 3, fontweight="bold")
        if s.empty:
            ax.text(0.5, 0.5, "no interactions", transform=ax.transAxes, ha="center",
                    fontsize=fs, color="0.5")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(gates):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% frames with the interaction", fontsize=fs)
    fig.legend(handles=[Patch(color="#9aa5b1", label="before the jump"),
                        Patch(color="#2a9d8f", label="after: gained"),
                        Patch(color="#e76f51", label="after: lost")],
               loc="lower center", ncol=3, frameon=False, fontsize=fs)
    fig.suptitle(f"{title}\ninteractions of the gate Phe ring: π–π (ring-ring, ≤ 5.5-6.5 Å), "
                 f"S···π (Cys/Met S ≤ 5.5 Å), S–H···π (H ≤ 3.2 Å, ≥ 120°),\nCH–π (methyl C ≤ 4.5 Å over "
                 f"the ring face), cation–π (Lys NZ / Arg CZ ≤ 6 Å)\nshown: ≥ {min_pct:g} % before "
                 f"or after" + (f" and a change of ≥ {min_change:g} points" if min_change else "")
                 + "\nGIRK1 residues: own number, GIRK2 position in brackets, e.g. F72.G1 (=F83)\n"
                 "grey italic = the equivalent partner of this gate (same position and type), "
                 "below the filter",
                 fontsize=fs - 2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.85), h_pad=9)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
