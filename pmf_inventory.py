"""
pmf_inventory.py

Finds every umbrella-sampling / PMF calculation under one or more folders and packs the small files
(restraint inputs, per-window coordinate time series, WHAM metadata and outputs, force CSVs, plots,
scripts) into one tarball that can be analysed elsewhere. Trajectories and restart files are listed
but never copied.

A folder is taken as a PMF calculation when it, or its subfolders, hold any of:
  - Amber umbrella inputs (mdin with nmropt=1 and a DISANG restraint file) or their DUMPAVE output
  - Grossfield WHAM metadata (lines "<file> <centre> <k>") or WHAM output ("#Coor ... Free")
  - files or folders whose name contains pmf, umbrella, wham, window, win, us_, smd or pull

Outputs (in --out, default ./pmf_inventory):
  inventory.txt         one block per PMF calculation: windows, ns per window, restraint centres and
                        force constants, WHAM files, trajectories found (not copied)
  windows.csv           one row per window: calculation, window, centre, k, n samples, time span (ns)
  pmf_small_files.tgz   the small files, with their paths relative to the search root

Usage:
  python3 pmf_inventory.py --root /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina
  python3 pmf_inventory.py --root /path/A --root /path/B --max_mb 50
"""

import argparse
import csv
import os
import re
import tarfile
from collections import defaultdict
from pathlib import Path

KEYWORDS = re.compile(r"(pmf|umbrella|wham|window|win\d|win_|us_|_us|smd|pull|steer)", re.I)
BIG_EXT = {".nc", ".ncdf", ".dcd", ".xtc", ".trr", ".ncrst", ".cpt", ".tpr", ".edr", ".h5", ".dtr", ".npy"}
TEXT_COORD_EXT = {".rst7", ".rst", ".mdcrd", ".crd", ".inpcrd"}   # big only when > 1 MB (Amber .RST restraints are tiny)


def is_big(path, max_mb):
    size = path.stat().st_size
    ext = path.suffix.lower()
    return ext in BIG_EXT or (ext in TEXT_COORD_EXT and size > 1e6) or size > max_mb * 1e6
SMALL_EXT = {".in", ".mdin", ".disang", ".dat", ".txt", ".csv", ".xvg", ".out", ".mdout", ".log",
             ".py", ".sh", ".slurm", ".pbs", ".json", ".png", ".pdf", ".svg", ".meta", ".dist",
             ".tsv", ".yaml", ".yml", ".ipynb", ".RST", ".plumed"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def read_head(path, n=4000):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read(n)
    except OSError:
        return ""


def classify(path):
    """Return a short tag for a file that marks a PMF calculation, or None."""
    name = path.name
    head = read_head(path) if not is_big(path, 200) else ""
    if re.search(r"nmropt\s*=\s*1", head, re.I) and re.search(r"DISANG", head, re.I):
        return "amber_mdin"
    if re.search(r"&rst\b", head, re.I) and re.search(r"r2\s*=", head, re.I):
        return "amber_restraint"
    if re.search(r"DUMPAVE", name, re.I) or re.search(r"dumpave", head, re.I):
        return "dumpave"
    if re.search(r"#\s*Coor\s+Free", head) or re.search(r"#Coor\s+Free", head):
        return "wham_out"
    lines = [l.split() for l in head.splitlines() if l.strip() and not l.startswith("#")]
    if len(lines) >= 3 and all(len(l) >= 3 for l in lines[:3]):
        try:
            [float(l[1]) + float(l[2]) for l in lines[:3]]
            if any("/" in l[0] or re.search(r"[A-Za-z]", l[0]) for l in lines[:3]):
                return "wham_meta"
        except ValueError:
            pass
    if re.search(r"pull[xf]", name, re.I) and path.suffix == ".xvg":
        return "gromacs_pull"
    if KEYWORDS.search(name):
        return "named"
    return None


def amber_restraint(path):
    """Centre (r2) and force constant (rk2) of an Amber &rst file."""
    txt = read_head(path, 20000)
    r2 = re.search(r"r2\s*=\s*([-\d.eE+]+)", txt)
    rk2 = re.search(r"rk2\s*=\s*([-\d.eE+]+)", txt)
    iat = re.search(r"iat\s*=\s*([-\d,\s]+)", txt)
    return (float(r2.group(1)) if r2 else None, float(rk2.group(1)) if rk2 else None,
            iat.group(1).strip().rstrip(",") if iat else "")


def amber_mdin_info(path):
    txt = read_head(path, 20000)
    out = {}
    for k in ("nstlim", "dt", "ntwx", "ntpr", "irest", "temp0"):
        m = re.search(rf"\b{k}\s*=\s*([-\d.eE+]+)", txt)
        if m:
            out[k] = float(m.group(1))
    d = re.search(r"DISANG\s*=\s*(\S+)", txt)
    dump = re.search(r"DUMPAVE\s*=\s*(\S+)", txt)
    out["disang"] = d.group(1) if d else ""
    out["dumpave"] = dump.group(1) if dump else ""
    return out


def series_info(path):
    """Number of numeric rows and the first/last value of column 0 of a time series."""
    n, first, last = 0, None, None
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                s = line.split()
                if not s or line[0] in "#@;":
                    continue
                try:
                    t = float(s[0])
                except ValueError:
                    continue
                n += 1
                first = t if first is None else first
                last = t
    except OSError:
        pass
    return n, first, last


def calc_root(path, root):
    """Group files by the nearest ancestor whose name looks like a PMF calculation, else parent."""
    rel = path.relative_to(root).parts[:-1]
    for i in range(len(rel), 0, -1):
        if KEYWORDS.search(rel[i - 1]) and not re.fullmatch(r"(win|window)[_-]?\d+", rel[i - 1], re.I):
            return root.joinpath(*rel[:i])
    parent = path.parent
    if re.fullmatch(r"(win|window|w)[_-]?\d+", parent.name, re.I):
        return parent.parent
    return parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default="pmf_inventory")
    ap.add_argument("--max_mb", type=float, default=50.0, help="largest file copied into the tarball")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    calcs = defaultdict(lambda: {"tags": defaultdict(list), "big": [], "small": []})
    for root in map(Path, a.root):
        root = root.resolve()
        print(f"scanning {root}")
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    if not p.is_file():
                        continue
                    tag = classify(p)
                except OSError:
                    continue
                in_named = any(KEYWORDS.search(x) for x in p.relative_to(root).parts[:-1])
                if tag is None and not in_named:
                    continue
                c = calcs[(root, calc_root(p, root))]
                if tag:
                    c["tags"][tag].append(p)
                if is_big(p, a.max_mb):
                    c["big"].append(p)
                elif p.suffix in SMALL_EXT or p.suffix.lower() in SMALL_EXT or tag or re.search(r"(dumpave|disang|rst|meta)", fn, re.I):
                    c["small"].append(p)

    # keep groups that really look like PMF work
    keep = {k: v for k, v in calcs.items()
            if v["tags"].keys() & {"amber_mdin", "amber_restraint", "dumpave", "wham_out", "wham_meta", "gromacs_pull"}
            or KEYWORDS.search(k[1].name)}

    rows, lines = [], []
    for (root, cdir), c in sorted(keep.items(), key=lambda kv: str(kv[0][1])):
        rel = cdir.relative_to(root)
        lines.append(f"=== {root.name}/{rel}")
        for tag, files in sorted(c["tags"].items()):
            lines.append(f"  {tag:<16} {len(files):>4} files   e.g. {files[0].relative_to(root)}")
        # windows from Amber restraint files
        rst = sorted(c["tags"].get("amber_restraint", []))
        mdins = c["tags"].get("amber_mdin", [])
        if mdins:
            info = amber_mdin_info(mdins[0])
            if "nstlim" in info and "dt" in info:
                lines.append(f"  mdin: nstlim {int(info['nstlim'])}, dt {info['dt']} ps -> "
                             f"{info['nstlim'] * info['dt'] / 1000:.2f} ns per segment, temp0 {info.get('temp0')}")
        for f in rst:
            r2, k, iat = amber_restraint(f)
            rows.append({"calculation": f"{root.name}/{rel}", "kind": "restraint", "file": str(f.relative_to(root)),
                         "centre": r2, "k": k, "atoms": iat, "n": "", "t_first": "", "t_last": ""})
        if rst:
            cs = [amber_restraint(f)[0] for f in rst]
            ks = {amber_restraint(f)[1] for f in rst}
            cs = sorted(x for x in cs if x is not None)
            if cs:
                lines.append(f"  windows: {len(cs)}, centres {cs[0]:.2f} .. {cs[-1]:.2f} A, k = {sorted(k for k in ks if k is not None)}")
        series = set(c["tags"].get("dumpave", []) + c["tags"].get("gromacs_pull", []))
        for m in mdins:                     # the DUMPAVE file each umbrella mdin writes
            d = amber_mdin_info(m).get("dumpave")
            if d and (m.parent / d).is_file():
                series.add(m.parent / d)
        for f in sorted(series):
            n, t0, t1 = series_info(f)
            rows.append({"calculation": f"{root.name}/{rel}", "kind": "series", "file": str(f.relative_to(root)),
                         "centre": "", "k": "", "atoms": "", "n": n, "t_first": t0, "t_last": t1})
        for tag in ("wham_meta", "wham_out"):
            for f in c["tags"].get(tag, []):
                lines.append(f"  {tag}: {f.relative_to(root)}  ({f.stat().st_mtime_ns // 10**9:.0f} mtime)")
        if c["big"]:
            lines.append(f"  trajectories / large files (not copied): {len(c['big'])}, "
                         f"{sum(p.stat().st_size for p in c['big']) / 1e9:.1f} GB, e.g. {c['big'][0].relative_to(root)}")
        lines.append(f"  small files copied: {len(c['small'])}")
        lines.append("")

    (out / "inventory.txt").write_text("\n".join(lines))
    with open(out / "windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["calculation", "kind", "file", "centre", "k", "atoms", "n", "t_first", "t_last"])
        w.writeheader()
        w.writerows(rows)

    # all file listing (including trajectories) so window lengths can be checked from sizes
    with open(out / "all_files.tsv", "w") as fh:
        fh.write("root\tpath\tbytes\tmtime\n")
        for (root, _), c in keep.items():
            for p in sorted(set(c["small"] + c["big"])):
                st = p.stat()
                fh.write(f"{root}\t{p.relative_to(root)}\t{st.st_size}\t{int(st.st_mtime)}\n")

    tgz = out / "pmf_small_files.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        for f in ("inventory.txt", "windows.csv", "all_files.tsv"):
            tar.add(out / f, arcname=f"pmf_inventory/{f}")
        seen = set()
        for (root, _), c in keep.items():
            for p in c["small"]:
                if p in seen:
                    continue
                seen.add(p)
                tar.add(p, arcname=f"pmf_inventory/files/{root.name}/{p.relative_to(root)}")
    print(f"{len(keep)} PMF folders, {len(seen)} small files -> {tgz} ({tgz.stat().st_size / 1e6:.1f} MB)")
    print(f"read {out / 'inventory.txt'} for the summary")


if __name__ == "__main__":
    main()
