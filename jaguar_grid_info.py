#!/usr/bin/env python3
"""
jaguar_grid_info.py
===================
How did Jaguar place its ESP grid points? Reads a .resp (points) and the matching .out (elements) and reports,
for every grid point, its distance to the nearest atom, absolute and divided by that atom's van der Waals
radius, so the shells (e.g. 1.4 / 1.6 / 1.8 / 2.0 x vdW) and their density show up; and whether the points
sit on a regular lattice instead. Used to build a Jaguar-like grid for the ensemble ESP.

USAGE
  python3 jaguar_grid_info.py --resp <job>.resp --out <job>.out
Prints the summary and writes jaguar_grid_info.png next to where you run it.
"""

import argparse
import re
from pathlib import Path

import numpy as np

BOHR = 0.52917721092
BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def read_resp(path):
    with open(path) as fh:
        natoms, npoints = (int(x) for x in fh.readline().split()[:2])
        atoms = np.array([[float(v) for v in fh.readline().split()[-3:]] for _ in range(natoms)])
        grid = np.atleast_2d(np.loadtxt(fh))
    return atoms * BOHR, grid[:, 1:4] * BOHR


def read_elements(path):
    lines = Path(path).read_text().split("\n")
    start = next(n for n, l in enumerate(lines) if l.strip() == "Input geometry:")
    out = []
    for l in lines[start + 3:]:
        p = l.split()
        if len(p) != 4:
            break
        try:
            [float(v) for v in p[1:]]
        except ValueError:
            break
        out.append(re.match(r"[A-Za-z]+", p[0]).group().capitalize())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    atoms, grid = read_resp(a.resp)
    el = read_elements(a.out)
    rv = np.array([BONDI.get(e, 1.7) for e in el])
    dmin = np.empty(len(grid))
    near = np.empty(len(grid), int)
    ratio_min = np.empty(len(grid))
    for s in range(0, len(grid), 20000):
        d = np.linalg.norm(grid[s:s + 20000, None, :] - atoms[None], axis=2)
        near[s:s + 20000] = d.argmin(axis=1)
        dmin[s:s + 20000] = d.min(axis=1)
        ratio_min[s:s + 20000] = (d / rv[None, :]).min(axis=1)
    print(f"{len(atoms)} atoms, {len(grid)} grid points ({len(grid) / len(atoms):.0f} per atom)")
    print(f"distance to the nearest atom (A): min {dmin.min():.2f}  5% {np.percentile(dmin, 5):.2f}  median "
          f"{np.median(dmin):.2f}  95% {np.percentile(dmin, 95):.2f}  max {dmin.max():.2f}")
    print(f"min over atoms of distance / Bondi vdW radius: min {ratio_min.min():.3f}  5% {np.percentile(ratio_min, 5):.3f}  "
          f"median {np.median(ratio_min):.3f}  95% {np.percentile(ratio_min, 95):.3f}  max {ratio_min.max():.3f}")
    h, e = np.histogram(ratio_min, bins=np.arange(0.8, 3.01, 0.02))
    peaks = [(round(0.5 * (e[i] + e[i + 1]), 2), int(h[i])) for i in range(1, len(h) - 1)
             if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.02 * h.max()]
    print("peaks of distance / vdW radius (shell factors), with point counts:", peaks)
    for E in sorted(set(el)):
        sel = np.array(el)[near] == E
        if sel.sum():
            print(f"  nearest atom {E}: {sel.sum():>7} points, distance to it: median {np.median(dmin[sel]):.2f} A, "
                  f"5-95% {np.percentile(dmin[sel], 5):.2f}-{np.percentile(dmin[sel], 95):.2f} A")
    # lattice test: are the coordinates on a regular spacing?
    for k, axn in enumerate("xyz"):
        u = np.unique(np.round(grid[:, k], 4))
        dif = np.diff(u)
        dif = dif[dif > 1e-3]
        mode = np.round(np.median(dif), 4) if len(dif) else np.nan
        print(f"  {axn}: {len(u)} distinct values; median gap {mode} A (few distinct values + constant gap = lattice)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dmin, bins=200, color="#1F4E79")
    ax[0].set_xlabel("distance to the nearest atom (Å)")
    ax[1].hist(ratio_min, bins=200, color="#D9784F")
    ax[1].set_xlabel("min distance / Bondi vdW radius")
    for x in ax:
        x.set_ylabel("grid points")
    fig.tight_layout()
    fig.savefig("jaguar_grid_info.png", dpi=150)
    print("wrote jaguar_grid_info.png")


if __name__ == "__main__":
    main()
