"""
validate_subunits.py
====================
Check, before any distance is computed, that the residue lists and diagonal
pairs of a channel config really point at the subunits we think they do.

Why: compute_constriction_point_diameters() writes the FIRST pair of a
*_diagonal_pairs list to the column "A_C" and the SECOND to "B_D", whatever
subunits they are. If the pair order differs between systems, the same column
name means different subunits (this happened: in G12 WT GIRK1 was in B_D, in
E141Q / D173N / GAT it was in A_C).

Convention enforced here (letters from the config's subunit_groups):
    first pair  -> column A_C -> subunits A and C
    second pair -> column B_D -> subunits B and D
    GIRK1 (the chain with ASP at the asn_residues position, GIRK1 D173 vs
    GIRK2 N184) is subunit D, so it is always in B_D.

What is checked, per gate:
    - every resid exists and is a single residue
    - expected residue names (HBC = PHE, lower filter = THR)
    - the four residues of the gate are in four different chains
    - each pair is a real diagonal: its CA-CA distance (current frame) is
      larger than every adjacent CA-CA distance of that gate
    - first pair = subunits {A, C}, second pair = {B, D}
    - for a 1 GIRK1 : 3 GIRK2 channel, GIRK1 is subunit D and in B_D

Chains are taken from the bond graph (MDAnalysis fragments), so it works with
Amber prmtops that have no chain IDs.

Usage (main.py does this right after loading the Universe):
    from analysis.validate_subunits import validate_subunits
    validate_subunits(u, channel_type, gates, subunit_groups,
                      asn_residues, glu_residues, results_dir=results_dir)
A failed check raises SubunitCheckError; main.py turns that into an exit
unless --skip_subunit_check is given.
"""

import os
from itertools import combinations
from pathlib import Path

import numpy as np

COLUMN_NAMES = ("A_C", "B_D")
EXPECTED_SUBUNITS = ({"A", "C"}, {"B", "D"})
# gate name -> residue names it may have
EXPECTED_RESNAMES = {
    "hbc": {"PHE"},
    "sf_low_res": {"THR"},
    "glu": {"GLU", "GLH", "GLN"},
    "asn": {"ASN", "ASP", "ASH"},
}
GIRK1_MARKERS = {"ASP", "ASH"}   # GIRK1 D173 (GIRK2 has N184)


class SubunitCheckError(RuntimeError):
    pass


def _residue(u, resid):
    res = u.select_atoms(f"resid {resid}").residues
    if len(res) != 1:
        raise SubunitCheckError(f"resid {resid}: found {len(res)} residues (expected 1)")
    return res[0]


def _chain_of(res):
    """Chain index of a residue = index of its bonded fragment."""
    return int(res.atoms[0].fragindex)


def _ca(res):
    ca = res.atoms.select_atoms("name CA")
    return ca.positions[0] if len(ca) else res.atoms.center_of_geometry()


FIRST_PASS_FILE = "subunit_check_first_pass.txt"
PIP_CUTOFF = 6.0      # A, protein heavy atoms around the PIP2 phosphorus atoms
PIP_MARGIN = 1.5      # top subunit needs >= this many times the contacts of the next


def _check_pips(u, pip_mapping, chain_to_sub, girk1_chains, out, errors, warnings):
    """
    Which subunit is each PIP2 bound to? Counts protein heavy atoms within
    PIP_CUTOFF of the PIP's phosphorus atoms (P1/P4/P5) on the current frame,
    per chain. Labels must follow the subunit letters: G1 = the GIRK1 subunit
    (D), 2a/2b/2c/2d = GIRK2 subunits A/B/C/D.
    """
    out("\nPIP2 sites (protein heavy atoms within "
        f"{PIP_CUTOFF:.0f} A of the PIP phosphorus atoms, per subunit):")
    for resid, label in sorted(pip_mapping.items()):
        pip = u.select_atoms(f"resid {resid}")
        if not len(pip):
            errors.append(f"pip_mapping: PIP resid {resid} not found")
            continue
        phos = pip.select_atoms("name P1 P4 P5") or pip.select_atoms("name P*")
        if not len(phos):
            errors.append(f"PIP {resid}: no phosphorus atoms (P1/P4/P5) found")
            continue
        near = u.select_atoms(f"protein and not name H* and around {PIP_CUTOFF} group phos",
                              phos=phos)
        counts = {}
        for a in near:
            sub = chain_to_sub.get(int(a.fragindex), "?")
            counts[sub] = counts.get(sub, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        label = str(label)
        if label.upper() == "G1":
            expected = sorted({chain_to_sub[c] for c in girk1_chains if c in chain_to_sub}) or ["D"]
        elif len(label) == 2 and label[0] == "2" and label[1].upper() in "ABCD":
            expected = [label[1].upper()]
        else:
            expected = []
        top = ranked[0][0] if ranked else "?"
        second = ranked[1][1] if len(ranked) > 1 else 0
        clear = bool(ranked) and ranked[0][1] >= PIP_MARGIN * max(second, 1)
        txt = ", ".join(f"{s} {n}" for s, n in ranked[:3]) or "no protein contact"
        verdict = ("ok" if top in expected else
                   "MISMATCH" if clear and expected else "unclear")
        out(f"   PIP {resid} labelled {label:<3} -> closest subunit {top}  ({txt})   {verdict}")
        if not expected:
            warnings.append(f"PIP {resid}: label '{label}' not understood (expected G1, 2a, 2b, 2c, 2d)")
        elif top not in expected:
            msg = (f"PIP {resid} is labelled {label} (subunit {'/'.join(expected)}) but sits "
                   f"closest to subunit {top} ({txt})")
            (errors if clear else warnings).append(msg + ("" if clear else " -- too close to call"))


def validate_subunits(u, channel_type, gates, subunit_groups=None,
                      asn_residues=None, glu_residues=None,
                      results_dir=None, verbose=True, pip_mapping=None):
    """
    gates: {name: (residues, diagonal_pairs)}, e.g.
        {"hbc": (hbc_residues, hbc_diagonal_pairs), ...}
    Entries whose residues or pairs are None are skipped.
    Returns the report text. Raises SubunitCheckError on any failed check.
    """
    lines, errors, warnings = [], [], []
    out = lines.append

    try:
        u.atoms[0].fragindex
    except Exception as exc:
        raise SubunitCheckError(
            f"cannot get chains from the topology (no bonds?): {exc}")

    out("=" * 78)
    out(f"SUBUNIT CHECK   channel_type = {channel_type}")
    out("=" * 78)

    # ── subunit letters -> chains ───────────────────────────────────────────
    chain_to_sub = {}
    if subunit_groups:
        for letter, resids in subunit_groups.items():
            chains = set()
            for r in resids:
                try:
                    chains.add(_chain_of(_residue(u, r)))
                except SubunitCheckError as e:
                    errors.append(f"subunit_groups[{letter}]: {e}")
            if len(chains) > 1:
                errors.append(f"subunit_groups['{letter}'] = {resids} spans chains {sorted(chains)}")
            for c in chains:
                if c in chain_to_sub and chain_to_sub[c] != letter:
                    errors.append(f"chain {c} is both subunit {chain_to_sub[c]} and {letter}")
                chain_to_sub[c] = letter
    else:
        warnings.append("no subunit_groups in this config: A_C/B_D subunit letters not checked")

    # ── GIRK1 / mutation markers ────────────────────────────────────────────
    girk1_chains = set()
    if asn_residues:
        out("\nCavity Asn/Asp (GIRK1 D173 = ASP, GIRK2 N184 = ASN)   and pore Glu:")
        glu_by_chain = {}
        for r in (glu_residues or []):
            try:
                res = _residue(u, r)
                glu_by_chain[_chain_of(res)] = f"{res.resname}{r}"
            except SubunitCheckError as e:
                errors.append(f"glu_residues: {e}")
        for r in asn_residues:
            try:
                res = _residue(u, r)
            except SubunitCheckError as e:
                errors.append(f"asn_residues: {e}")
                continue
            c = _chain_of(res)
            is_g1 = res.resname in GIRK1_MARKERS
            if is_g1:
                girk1_chains.add(c)
            out(f"   chain {c}  subunit {chain_to_sub.get(c, '?')}   {res.resname}{r:<6} "
                f"{glu_by_chain.get(c, ''):<10} -> {'GIRK1' if is_g1 else 'GIRK2'}")
    heteromer = str(channel_type).upper().startswith("G12")
    if not girk1_chains and heteromer and "D" in chain_to_sub.values():
        # e.g. G12_D173N: the GIRK1 Asp is mutated, so the Asp test cannot see it
        girk1_chains = {c for c, s in chain_to_sub.items() if s == "D"}
        warnings.append("no ASP at asn_residues (GIRK1 D173 mutated?): assuming GIRK1 = "
                        "subunit D, as in the other G12 configs -- NOT verified from the sequence")
    g1_letters = sorted(chain_to_sub.get(c, "?") for c in girk1_chains)
    out(f"   GIRK1 subunit(s): {', '.join(g1_letters) if g1_letters else 'none (homotetramer?)'}")
    if heteromer and len(girk1_chains) != 1:
        warnings.append(f"G12 config but {len(girk1_chains)} GIRK1 chain(s) found (expected 1)")

    if len(girk1_chains) == 1 and chain_to_sub:
        if g1_letters != ["D"]:
            errors.append(f"GIRK1 is subunit {g1_letters[0]}, expected D "
                          f"(relabel subunit_groups so GIRK1 is D)")

    # ── each gate ───────────────────────────────────────────────────────────
    for gate, (residues, pairs) in gates.items():
        if residues is None or pairs is None:
            continue
        out(f"\n[{gate}]  residues {list(residues)}   pairs {[tuple(p) for p in pairs]}")
        try:
            res = {r: _residue(u, r) for r in set(residues) | {x for p in pairs for x in p}}
        except SubunitCheckError as e:
            errors.append(f"{gate}: {e}")
            continue
        chain = {r: _chain_of(x) for r, x in res.items()}

        if gate in EXPECTED_RESNAMES:
            for r in residues:
                if res[r].resname not in EXPECTED_RESNAMES[gate]:
                    errors.append(f"{gate}: resid {r} is {res[r].resname}, expected "
                                  f"{'/'.join(sorted(EXPECTED_RESNAMES[gate]))}")
        if len({chain[r] for r in residues}) != len(residues):
            errors.append(f"{gate}: residues {list(residues)} are not in {len(residues)} different chains "
                          f"(chains {[chain[r] for r in residues]})")

        if len(pairs) != 2:
            errors.append(f"{gate}: expected 2 diagonal pairs, got {len(pairs)}")
            continue

        # geometry: diagonals must be the two longest CA-CA distances
        pos = {r: _ca(res[r]) for r in residues}
        d = {frozenset(p): float(np.linalg.norm(pos[p[0]] - pos[p[1]]))
             for p in combinations(residues, 2)}
        diag = [frozenset(p) for p in pairs]
        adjacent = [v for k, v in d.items() if k not in diag]
        shortest_diag = min(d.get(k, np.nan) for k in diag)
        geom_ok = bool(adjacent) and shortest_diag > max(adjacent)

        for i, (a, b) in enumerate(pairs):
            subs = {chain_to_sub.get(chain[a], "?"), chain_to_sub.get(chain[b], "?")}
            g1 = any(chain[x] in girk1_chains for x in (a, b))
            out(f"   column {COLUMN_NAMES[i]}: {res[a].resname}{a} (chain {chain[a]}, "
                f"subunit {chain_to_sub.get(chain[a], '?')})  -  {res[b].resname}{b} "
                f"(chain {chain[b]}, subunit {chain_to_sub.get(chain[b], '?')})   "
                f"CA-CA {d[frozenset((a, b))]:5.1f} A   {'<- GIRK1' if g1 else ''}")
            if chain[a] == chain[b]:
                errors.append(f"{gate} {COLUMN_NAMES[i]}: {a} and {b} are in the same chain")
            if chain_to_sub and subs != EXPECTED_SUBUNITS[i]:
                hint = (f" (swap the order of the {gate} pairs)" if subs == EXPECTED_SUBUNITS[1 - i]
                        else "")
                errors.append(f"{gate}: column {COLUMN_NAMES[i]} is subunits {'-'.join(sorted(subs))}, "
                              f"expected {'-'.join(sorted(EXPECTED_SUBUNITS[i]))}{hint}")
            if g1 and i == 0 and len(girk1_chains) == 1:
                errors.append(f"{gate}: GIRK1 is in column A_C, expected B_D")
        out(f"   adjacent CA-CA: {', '.join(f'{v:.1f}' for v in sorted(adjacent))} A   "
            f"-> pairs are {'diagonal' if geom_ok else 'NOT diagonal'}")
        if not geom_ok:
            errors.append(f"{gate}: pairs {[tuple(p) for p in pairs]} are not the diagonals "
                          f"(a pair CA-CA {shortest_diag:.1f} A <= longest adjacent {max(adjacent):.1f} A)")

    # ── PIP2 -> subunit (pip_mapping) ──────────────────────────────────────
    if pip_mapping and chain_to_sub:
        _check_pips(u, pip_mapping, chain_to_sub, girk1_chains, out, errors, warnings)

    # ── verdict ─────────────────────────────────────────────────────────────
    out("")
    for w in warnings:
        out(f"WARNING: {w}")
    if errors:
        out(f"SUBUNIT CHECK FAILED ({len(errors)} problem(s)):")
        for e in errors:
            out(f"   - {e}")
    else:
        out("SUBUNIT CHECK PASSED: A_C = subunits A-C, B_D = subunits B-D"
            + (", GIRK1 = D (in B_D)" if girk1_chains else ""))
    out("=" * 78)

    report = "\n".join(lines)
    if verbose:
        print(report)
    if results_dir is not None:
        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            chk = Path(results_dir) / "subunit_check.txt"
            first = Path(results_dir) / FIRST_PASS_FILE
            # swap_gate_columns.py tells old-order gate CSVs (written BEFORE the
            # config first passed this check) from new-order ones (written after)
            # by the time of the FIRST passing check. That time lives in its own
            # file, never overwritten; if only an older passing report exists,
            # the marker gets that report's time, not "now".
            if not first.is_file():
                old_ok = chk.is_file() and "SUBUNIT CHECK PASSED" in chk.read_text()
                if old_ok or not errors:
                    t = chk.stat().st_mtime if old_ok else None
                    first.write_text("first passing subunit check (time = this file's "
                                     "modification time); used by swap_gate_columns.py\n")
                    if t is not None:
                        os.utime(first, (t, t))
            if not chk.is_file() or chk.read_text() != report + "\n":
                chk.write_text(report + "\n")
        except OSError as exc:
            print(f"  [warn] could not write subunit_check.txt: {exc}")
    if errors:
        raise SubunitCheckError("; ".join(errors))
    return report
