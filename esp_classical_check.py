#!/usr/bin/env python3
"""
esp_classical_check.py
======================
Step A of the ESP robustness test: does a CLASSICAL electrostatic potential (fixed point charges)
reproduce your QM ESP on the same frame, at the same points, in the same cylinders?

For one Jaguar single point (the .resp grid + the .out) it computes, at EVERY Jaguar grid point,
    V = 332.0637 * sum_i q_i / r_i          (kcal/mol per +1e, the same units as esp_cylinders.py)
with three sets of point charges on the QM cluster atoms (210 atoms: 8 ring residues + ACE/NME caps):
    FF       force-field charges from the prmtop for every cluster atom that maps onto the MD frame;
             each cap (the neighbouring-residue backbone pieces) is shifted to net 0, as the neutral
             ACE / NME caps of the QM cluster are; added cap hydrogens start at 0. Cluster total = QM
             total (-5 for G12, -4 for G2).
    FF_core  the same, but only the 8 ring residues (caps set to 0): how much the neighbouring
             backbone contributes.
    ESPfit   Jaguar's own charges fitted to the QM potential (from the .out, "Atomic charges from
             electrostatic potential"): separates "electrons vs point charges" from "force field vs QM".
It then selects the points in the same four cylinders as esp_cylinders.py (Glu CA -> Asn/Asp CA of each
subunit, radius 3 A, extended 3 A, ca-ca axis; upper/lower end by CA z) and compares QM with each model:
per-cylinder mean / median, the four-subunit mean, and the point-by-point correlation.

INPUTS
  --resp    Jaguar .resp (QM ESP on the grid; same file as for esp_cylinders.py)
  --out     the matching Jaguar .out (atom labels + ESP-fitted charges)
  --struct  the PDB of the MD frame the cluster was cut from (frame5606_K.pdb), for residue numbers
            and atom names (atoms are matched to the cluster by position, < --tol A)
  --prmtop  the prmtop of that simulation (charges, read with MDAnalysis), OR a CSV with columns
            resid,name,charge
  --groups  as for esp_cylinders.py, e.g. A:780,748 B:456,424 C:1073,1105 G1:99,131

USAGE (G12 frame 5606)
  python3 esp_classical_check.py \\
    --resp .../GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.resp \\
    --out  .../GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.out \\
    --struct .../GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \\
    --prmtop .../GIRK12_WT/RUN3/com_4fs.prmtop \\
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \\
    --label G12_frame5606 --folder ./results_classical/G12_frame5606

OUTPUTS (in --folder)
  <label>_classical_vs_qm.csv   per cylinder and pooled: n, mean / median of QM, FF, FF_core, ESPfit,
                                Pearson r, slope and RMS difference of each model vs QM
  <label>_mapping.txt           which cluster atoms were matched to the MD frame, cap charges, totals
  <label>_scatter.png           QM vs FF and QM vs ESPfit at the cylinder points
  <label>_points.csv            (with --save-raw) every cylinder point: cylinder, QM, FF, FF_core, ESPfit

NOTE: an in-silico mutant cluster (e.g. D173N built on the WT frame) has atoms that are not in the WT
prmtop; the script reports them and the FF columns are then not a fair comparison (ESPfit still is).
"""

import argparse
import re
from pathlib import Path

import numpy as np

BOHR = 0.52917721092
HARTREE_KCAL = 627.509474
COULOMB = 332.0637            # kcal/mol * A / e^2


# ── readers ──────────────────────────────────────────────────────────────────

def read_resp(path):
    """Jaguar RESP file -> (atom_xyz_A, grid_xyz_A, esp_kcal), as in esp_cylinders.py."""
    with open(path) as fh:
        natoms, npoints = (int(x) for x in fh.readline().split()[:2])
        atoms = np.array([[float(v) for v in fh.readline().split()[-3:]] for _ in range(natoms)])
        grid = np.loadtxt(fh)
    grid = np.atleast_2d(grid)
    if grid.shape[0] != npoints:
        print(f"  NOTE: header says {npoints} grid points, found {grid.shape[0]}")
    return atoms * BOHR, grid[:, 1:4] * BOHR, grid[:, 0] * HARTREE_KCAL


def read_out(path):
    """Jaguar .out -> (labels, elements, xyz_A, espfit_charges or None, net_charge)."""
    lines = Path(path).read_text().split("\n")
    start = next(n for n, l in enumerate(lines) if l.strip() == "Input geometry:")
    labels, xyz = [], []
    for l in lines[start + 3:]:
        p = l.split()
        if len(p) != 4:
            break
        try:
            xyz.append([float(v) for v in p[1:]])
        except ValueError:
            break
        labels.append(p[0])
    elements = [re.match(r"[A-Za-z]+", s).group().capitalize() for s in labels]
    net = None
    for l in lines:
        if "net molecular charge" in l:
            net = int(float(l.split()[-1]))
            break
    q = {}
    try:
        k = next(n for n, l in enumerate(lines) if "Atomic charges from electrostatic potential" in l)
        atoms_row = None
        for l in lines[k + 1:]:
            s = l.split()
            if not s:
                continue
            if s[0] == "Atom":
                atoms_row = s[1:]
            elif s[0] == "Charge" and atoms_row:
                for a, v in zip(atoms_row, s[1:]):
                    q[a] = float(v)
                atoms_row = None
            elif "sum of atomic charges" in l:
                break
    except StopIteration:
        pass
    espfit = np.array([q[a] for a in labels]) if q and all(a in q for a in labels) else None
    return labels, elements, np.array(xyz), espfit, net


def read_pdb(path):
    """Plain PDB -> list of (resid, resname, name, xyz)."""
    out = []
    with open(path, errors="replace") as fh:
        for l in fh:
            if l.startswith(("ATOM", "HETATM")) and l[16] in (" ", "A"):
                try:
                    out.append((int(l[22:26]), l[17:20].strip(), l[12:16].strip(),
                                np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
                except ValueError:
                    continue
    if not out:
        raise SystemExit(f"ERROR: no ATOM records in {path}")
    return out


def read_charges(path):
    """{(resid, atom name): charge} from a prmtop (MDAnalysis) or a CSV resid,name,charge."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv
        with open(p) as fh:
            return {(int(r["resid"]), r["name"].strip()): float(r["charge"]) for r in csv.DictReader(fh)}
    import MDAnalysis as mda
    u = mda.Universe(str(p), topology_format="PRMTOP")
    return {(int(a.resid), a.name): float(a.charge) for a in u.atoms}


def element_of(name):
    m = re.match(r"[A-Za-z]", name)
    return m.group().upper() if m else "?"


# ── cylinders (same definition as esp_cylinders.py, --axis ca-ca) ─────────────

def cylinder_mask(points, p_start, p_end, radius, extend):
    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    u = vec / length
    rel = points - p_start
    t = rel @ u
    radial = np.linalg.norm(rel - np.outer(t, u), axis=1)
    return (radial <= radius) & (t >= -extend) & (t <= length + extend)


def parse_groups(args):
    out = []
    for g in args:
        name, rest = g.split(":", 1)
        a, b = (int(x) for x in rest.replace(",", " ").split())
        out.append((name.strip(), a, b))
    return out


# ── potential ────────────────────────────────────────────────────────────────

def potential(points, xyz, q, chunk=20000):
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        d = np.linalg.norm(points[s:s + chunk, None, :] - xyz[None, :, :], axis=2)
        out[s:s + chunk] = COULOMB * (q[None, :] / np.maximum(d, 1e-6)).sum(axis=1)
    return out


def compare(qm, model):
    ok = np.isfinite(model)
    if ok.sum() < 3:
        return dict(r=np.nan, slope=np.nan, rms=np.nan)
    a, b = qm[ok], model[ok]
    r = float(np.corrcoef(a, b)[0, 1])
    slope = float(np.polyfit(a, b, 1)[0])
    return dict(r=r, slope=slope, rms=float(np.sqrt(np.mean((b - a) ** 2))))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resp", required=True)
    ap.add_argument("--out", required=True, help="Jaguar .out of the same job")
    ap.add_argument("--struct", required=True, help="PDB of the MD frame the cluster was cut from")
    ap.add_argument("--prmtop", required=True, help="prmtop of that simulation, or CSV resid,name,charge")
    ap.add_argument("--groups", nargs="+", required=True, help="e.g. A:780,748 B:456,424 C:1073,1105 G1:99,131")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--extend", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=0.05, help="max distance (A) to match a cluster atom to the PDB")
    ap.add_argument("--label", default="esp_classical")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--save-raw", action="store_true")
    a = ap.parse_args()
    od = Path(a.folder) if a.folder else Path(a.label)
    od.mkdir(parents=True, exist_ok=True)

    atoms_resp, grid, esp = read_resp(a.resp)
    labels, elements, xyz_out, espfit, net = read_out(a.out)
    if len(labels) != len(atoms_resp):
        raise SystemExit(f"ERROR: .out has {len(labels)} atoms, .resp has {len(atoms_resp)}")
    shift = np.abs(xyz_out - atoms_resp).max()
    if shift > 0.01:
        print(f"  WARNING: .out and .resp coordinates differ by up to {shift:.3f} A (atom order?)")
    xyz = atoms_resp
    print(f"cluster: {len(xyz)} atoms, net charge {net}; grid: {len(grid)} points")

    # match cluster atoms to the MD frame
    pdb = read_pdb(a.struct)
    pxyz = np.array([p[3] for p in pdb])
    charges = read_charges(a.prmtop)
    groups = parse_groups(a.groups)
    ring = {r for _, g, s in groups for r in (g, s)}
    resid = np.full(len(xyz), -1)
    name = [""] * len(xyz)
    q_ff = np.zeros(len(xyz))
    mapped = np.zeros(len(xyz), bool)
    bad_heavy = []
    for i, p in enumerate(xyz):
        d = np.linalg.norm(pxyz - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= a.tol and element_of(pdb[j][2]) == elements[i][0].upper():
            key = (pdb[j][0], pdb[j][2])
            if key in charges:
                resid[i], name[i], q_ff[i], mapped[i] = pdb[j][0], pdb[j][2], charges[key], True
                continue
        if elements[i] != "H":
            bad_heavy.append(i)
    # unmapped hydrogens (added cap H) -> residue of the nearest heavy cluster atom
    heavy = np.array([e != "H" for e in elements])
    for i in np.where(~mapped)[0]:
        if elements[i] == "H":
            d = np.linalg.norm(xyz[heavy] - xyz[i], axis=1)
            k = np.where(heavy)[0][int(np.argmin(d))]
            resid[i] = resid[k]
    # charge models
    is_core = np.array([r in ring for r in resid])
    caps = sorted({int(r) for r in resid[~is_core] if r >= 0})
    q_cap = q_ff.copy()
    cap_report = []
    for r in caps:
        sel = resid == r
        before = q_cap[sel].sum()
        q_cap[sel] -= before / sel.sum()
        cap_report.append((r, int(sel.sum()), before))
    q_core = np.where(is_core, q_ff, 0.0)
    ring_sum = {r: q_ff[resid == r].sum() for r in sorted(ring)}

    with open(od / f"{a.label}_mapping.txt", "w") as fh:
        fh.write(f"cluster atoms {len(xyz)}, mapped to the MD frame {mapped.sum()}, unmapped H {int((~mapped & ~heavy).sum())}, "
                 f"unmapped heavy {len(bad_heavy)}\n")
        fh.write(f"QM net charge {net}; FF total before cap neutralisation {q_ff.sum():.3f}, after {q_cap.sum():.3f}, "
                 f"ring residues only {q_core.sum():.3f}\n\nring residues (FF charge):\n")
        for r, s in ring_sum.items():
            fh.write(f"  resid {r}: {s:+.3f}\n")
        fh.write("\ncaps (neighbouring residues): resid, atoms, FF charge before -> set to 0\n")
        for r, n_, b in cap_report:
            fh.write(f"  resid {r}: {n_} atoms, {b:+.3f}\n")
        if bad_heavy:
            fh.write("\nUNMAPPED HEAVY ATOMS (not in the MD frame / prmtop, e.g. an in-silico mutation): "
                     + ", ".join(labels[i] for i in bad_heavy) + "\n")
        fh.write("\natom-by-atom: label, element, resid, name, FF charge, FF (caps neutral), ESP-fit\n")
        for i in range(len(xyz)):
            fh.write(f"  {labels[i]:<6} {elements[i]:<2} {resid[i]:>6} {name[i]:<5} {q_ff[i]:+.4f} {q_cap[i]:+.4f} "
                     f"{(espfit[i] if espfit is not None else np.nan):+.4f}\n")
    print(f"mapped {mapped.sum()} / {len(xyz)} cluster atoms; FF total {q_cap.sum():+.3f} (QM {net}); caps {len(caps)}")
    if bad_heavy:
        print(f"  WARNING: {len(bad_heavy)} heavy atoms not in the MD frame ({', '.join(labels[i] for i in bad_heavy[:8])}"
              f"{'...' if len(bad_heavy) > 8 else ''}): FF columns are not a fair comparison for this cluster")
    if abs(q_cap.sum() - (net or 0)) > 0.05:
        print("  WARNING: FF cluster charge differs from the QM net charge - check the mapping file")

    # cylinders: CA of each group residue = the cluster atom at the PDB CA position
    ca = {}
    for r in ring:
        cand = [p[3] for p in pdb if p[0] == r and p[2] == "CA"]
        if not cand:
            raise SystemExit(f"ERROR: no CA for resid {r} in {a.struct}")
        best = min(cand, key=lambda c: np.linalg.norm(xyz - c, axis=1).min())
        ca[r] = xyz[int(np.argmin(np.linalg.norm(xyz - best, axis=1)))]
    models = {"FF": potential(grid, xyz, q_cap), "FF_core": potential(grid, xyz, q_core)}
    if espfit is not None:
        models["ESPfit"] = potential(grid, xyz, espfit)
    rows, raw, masks = [], [], []
    for gname, r1, r2 in groups:
        hi, lo = (r1, r2) if ca[r1][2] >= ca[r2][2] else (r2, r1)     # higher CA = Glu end
        m = cylinder_mask(grid, ca[hi], ca[lo], a.radius, a.extend)
        masks.append((gname, m))
        row = dict(cylinder=gname, glu=hi, lower=lo, n=int(m.sum()), QM_mean=esp[m].mean(), QM_median=np.median(esp[m]))
        for k, v in models.items():
            row[f"{k}_mean"], row[f"{k}_median"] = v[m].mean(), np.median(v[m])
            c = compare(esp[m], v[m])
            row.update({f"{k}_r": c["r"], f"{k}_slope": c["slope"], f"{k}_rms": c["rms"]})
        rows.append(row)
        if a.save_raw:
            for i in np.where(m)[0]:
                raw.append([gname, esp[i]] + [models[k][i] for k in models])
    allm = np.any([m for _, m in masks], axis=0)
    pooled = dict(cylinder="four-subunit mean", glu="", lower="", n=int(sum(r["n"] for r in rows)),
                  QM_mean=np.mean([r["QM_mean"] for r in rows]), QM_median=np.nan)
    for k, v in models.items():
        pooled[f"{k}_mean"] = np.mean([r[f"{k}_mean"] for r in rows])
        pooled[f"{k}_median"] = np.nan
        c = compare(esp[allm], v[allm])
        pooled.update({f"{k}_r": c["r"], f"{k}_slope": c["slope"], f"{k}_rms": c["rms"]})
    rows.append(pooled)

    import csv
    cols = list(rows[0].keys())
    with open(od / f"{a.label}_classical_vs_qm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    if a.save_raw:
        with open(od / f"{a.label}_points.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cylinder", "QM"] + list(models))
            w.writerows([[x[0]] + [f"{v:.3f}" for v in x[1:]] for x in raw])

    print(f"\n{'cylinder':<18}{'n':>6}{'QM':>9}" + "".join(f"{k:>9}" for k in models) +
          "".join(f"{'r(' + k + ')':>12}" for k in models))
    for r in rows:
        print(f"{r['cylinder']:<18}{r['n']:>6}{r['QM_mean']:>9.1f}" + "".join(f"{r[k + '_mean']:>9.1f}" for k in models) +
              "".join(f"{r[k + '_r']:>12.2f}" for k in models))
    print("(means in kcal/mol per +1e; r = point-by-point Pearson correlation with QM inside the cylinder)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [k for k in ("FF", "ESPfit") if k in models]
    fig, axes = plt.subplots(1, len(show), figsize=(6.2 * len(show), 5.6), squeeze=False)
    cols_ = ["#C0392B", "#D98A1C", "#13896B", "#3B7DD8", "#6B6B6B"]
    for ax, k in zip(axes[0], show):
        for (gname, m), c in zip(masks, cols_):
            ax.scatter(esp[m], models[k][m], s=6, alpha=0.45, color=c, label=gname, edgecolors="none")
        lo_, hi_ = np.percentile(np.r_[esp[allm], models[k][allm]], [0.5, 99.5])
        ax.plot([lo_, hi_], [lo_, hi_], color="#1B2631", lw=1)
        cc = compare(esp[allm], models[k][allm])
        ax.set_title(f"{k} vs QM   r = {cc['r']:.2f}, slope {cc['slope']:.2f}, RMS {cc['rms']:.0f}", fontsize=11)
        ax.set_xlabel("QM ESP (kcal/mol per +1e)")
        ax.set_ylabel(f"{k} ESP (kcal/mol per +1e)")
        ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
        ax.legend(markerscale=3, frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"{a.label}: classical vs QM at the Jaguar grid points in the cylinders", fontsize=12)
    fig.tight_layout()
    fig.savefig(od / f"{a.label}_scatter.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {od}/")


if __name__ == "__main__":
    main()
