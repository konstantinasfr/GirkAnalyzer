"""
glu_occupancy_at_exit.py
========================
Are the glutamates (and the ASN/ASP) already occupied by other K+ ions when a
permeating ion leaves the selectivity filter?

Idea being tested: the first touch after the SF is ASN/ASP only because the
GLU ring is already taken by other ions; when the GLU ring is free, the first
touch is a glutamate.

For every permeation event this runs the FIRST-TOUCH analysis
(FirstTouchAnalysis in longest_duration_analysis.py, residue mode: first
ASN/ASP/GLU residue within `threshold` after end_1), and reads the occupancy of every GLU and every ASN/ASP residue,
each one separately:

    end_1        the last frame of the ion in the SF (the moment it leaves)
    touch_frame  the frame of the first touch (None if there was no touch)
    time course  every frame from end_1 - timecourse_frames to end_1 + timecourse_frames

Where the first touch is searched:
    window=10    the first 10 frames after end_1 (never past end_2)
    window=None  no window: every frame from end_1 to end_2, the whole stay in
                 the cavity. "No contact" then means the ion crossed the cavity
                 without coming within `threshold` of any GLU or ASN/ASP.

A residue is "occupied" when another K+ / K ion (not the permeating ion) is
at <= `threshold` from any of its atoms. Stored per event and residue: the
ions on it and the closest other ion; per ring: how many residues are
occupied and how many distinct ions sit on them.

Regions: a second count that also catches ions that are near a ring but not
within `threshold` of it (e.g. one water layer away). Recomputed every frame:
    c_glu  = mean position of the 4 GLU (GLN) CA atoms
    c_low  = mean position of the 4 ASN/ASP CA atoms
    axis   = c_low -> c_glu; heights are measured along it, from c_low
    mid    = midpoint of c_glu and c_low (the cut)
    top    = SF bottom: mean of the SF residues' CA - `margin` (the lower edge
             of channel 1 in channels.py); without sf_residues: c_glu + `margin`

    GLU region  mid < height <= top          within `region_radius` of the axis
    ASN region  -margin <= height <= mid     within `region_radius` of the axis

Counted: every other K+ (not the permeating ion) in each region, at end_1, at
the touch frame and around end_1.

Results (results_dir, suggested f"{results_dir}/glu_occupancy_exit/{settings_folder(threshold, window)}",
i.e. glu_occupancy_exit/3.5_w10 or, with window=None, glu_occupancy_exit/3.5_no_w):
    glu_occupancy_exit_complete.json   every event, all fields
    glu_occupancy_exit_summary.json    cross-tables (counts)
    plots/                             the plots below, for this run
Aggregate over runs with aggregate_glu_occupancy_at_exit.py (same plots, pooled).

Plots (shared with the aggregator):
    plot_occupancy_vs_touch       occupied GLUs (or ASN/ASP) at end_1 per first-touch ring, and the reverse
    plot_first_touch_by_occupancy number of occupied GLUs (or ASN/ASP) on x, one bar per first touch;
                                  raw counts and % of all events in two files
    plot_occupied_combinations    which GLUs (or ASN/ASP) are occupied together: a panel per number occupied,
                                  a bar per combination (top 10), stacked by first touch; all panels in one
                                  file plus one file per panel
    plot_residue_occupancy        % of events each residue is occupied at end_1, per first-touch ring
    plot_touched_residue_heatmap  first-touched residue (rows) x occupied residue at end_1 (columns)
    plot_ring_timecourse          mean occupied GLUs / ASN/ASP vs frames from end_1, per first-touch ring
    plot_residue_timecourse       % occupied vs frames from end_1, one panel per residue
  region/   plot_occupancy_vs_touch, plot_first_touch_by_occupancy and the ring time
            course again, with the number of other K+ in the GLU / ASN region instead
            of the number of occupied residues
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from MDAnalysis.lib.distances import distance_array

from analysis.converter import convert_to_pdb_numbering
from analysis.closest_residue_analysis import ONE_LETTER
from analysis.longest_duration_analysis import FirstTouchAnalysis


ION_SELECTION = "resname K+ K"
NO_CONTACT = 'No contact'
RINGS = ['GLU', 'ASN']                      # upper ring, lower ring (ASN list holds the ASP too)
# n occupied residues: white (free ring) -> dark red (all four taken)
OCC_COLORS = ['#F7F7F7', '#FDD0A2', '#FC8D59', '#D7301F', '#7F0000']
RING_COLORS = {'GLU': '#E8746B', 'ASN': '#5AA9C9', NO_CONTACT: '#A0A0A0'}


def ring_name(resnames, resids):
    """'ASN/ASP', 'GLU', 'GLN' (E141Q), ...: the residue names of one ring."""
    return '/'.join(sorted({resnames.get(int(r), 'Unknown') for r in resids}))


def window_tag(window):
    """'w10' for a 10-frame first-touch window, 'no_w' for no window (window=None)."""
    return 'no_w' if window is None else f"w{window}"


def settings_folder(threshold, window):
    """Results sub-folder of one setting: '3.5_w10', or '3.5_no_w' without a window."""
    return f"{threshold}_{window_tag(window)}"


def touch_rule(window):
    """How the first touch was searched, for plot titles."""
    return ("first touch anywhere in the cavity (end_1 to end_2)" if window is None
            else f"first touch within {window} frames")


def residue_tick(res):
    """'E152.2a' from {'pdb': '152.2a', 'residue_type': 'GLU'}."""
    return f"{ONE_LETTER.get(res['residue_type'], '')}{res['pdb']}"


class GluOccupancyAtExitAnalysis(FirstTouchAnalysis):
    """
    First touch after the SF (residue mode) + occupancy of every GLU and
    ASN/ASP residue by other K+ ions. See the module docstring.
    """

    ANALYSIS = 'glu_occupancy_exit'
    NAME = 'GLU / ASN OCCUPANCY AT SF EXIT'
    FILE_PREFIX = 'glu_occupancy_exit'

    def __init__(self, universe, permeation_events, asn_residues, glu_residues,
                 channel_type="G12", threshold=3.5, window=10, analysis_start_frame=0,
                 ion_selection=ION_SELECTION, results_dir=".", timecourse_frames=10,
                 sf_residues=None, region_radius=12.0, margin=3.0):
        # Residue mode, both rings compete for the first touch
        super().__init__(universe, permeation_events, asn_residues, glu_residues,
                         channel_type, threshold, window if window is not None else 1,
                         analysis_start_frame, ['ASN', 'GLU'], None, results_dir)
        self.window = window                       # None: no window, search end_1 ... end_2
        self.timecourse_frames = int(timecourse_frames)
        self.ions = self.u.select_atoms(ion_selection)
        self.ion_selection = ion_selection

        # Every residue of both rings, GLUs first; checked one by one
        self.ring_of = {int(r): 'GLU' for r in glu_residues}
        self.ring_of.update({int(r): 'ASN' for r in asn_residues})
        groups = [(int(r), self.u.select_atoms(f"resid {r}")) for r in list(glu_residues) + list(asn_residues)]
        groups = [(r, g) for r, g in groups if len(g) > 0]
        self.residues = [{'pdb': convert_to_pdb_numbering(r, channel_type), 'resid': r,
                          'ring': self.ring_of[r], 'residue_type': self.resnames.get(r, 'Unknown')}
                         for r, _ in groups]
        self.res_atoms = self.u.atoms[np.concatenate([g.indices for _, g in groups])]
        self.res_starts = np.cumsum([0] + [len(g) for _, g in groups[:-1]])

        # Ring labels from the topology, so mutants are named correctly
        self.ring_labels = {'GLU': ring_name(self.resnames, glu_residues),
                            'ASN': ring_name(self.resnames, asn_residues)}

        # GLU / ASN regions: CA atoms of both rings and of the SF bottom
        self.region_radius = float(region_radius)
        self.margin = float(margin)
        self.sf_residues = list(sf_residues) if sf_residues is not None else None
        self.glu_ca = self._ca(glu_residues, 'GLU')
        self.low_ca = self._ca(asn_residues, 'ASN/ASP')
        self.sf_ca = self._ca(sf_residues, 'SF') if sf_residues is not None else None

    def _ca(self, resids, name):
        """CA atoms of `resids`; stops if one is missing."""
        ca = self.u.select_atoms(f"name CA and resid {' '.join(str(int(r)) for r in resids)}")
        if len(ca) != len(resids):
            raise ValueError(f"{name} ring: {len(ca)} CA atoms found for residues {list(resids)}")
        return ca

    def _region(self, ion_id):
        """
        Other K+ in the GLU and ASN/ASP regions at the current frame (see the
        module docstring): {'GLU': [ion ids], 'ASN': [ion ids], 'geometry': {...}}.
        """
        c_glu = self.glu_ca.positions.mean(axis=0)
        c_low = self.low_ca.positions.mean(axis=0)
        axis = c_glu - c_low
        length = float(np.linalg.norm(axis))
        axis /= length
        mid = length / 2
        top = (float(np.dot(self.sf_ca.positions.mean(axis=0) - c_low, axis)) - self.margin
               if self.sf_ca is not None else length + self.margin)
        others = self.ions[self.ions.resids != int(ion_id)]
        rel = others.positions - c_low
        height = rel @ axis
        inside = np.linalg.norm(rel - np.outer(height, axis), axis=1) <= self.region_radius
        glu = inside & (height > mid) & (height <= top)
        low = inside & (height >= -self.margin) & (height <= mid)
        return {'GLU': [int(i) for i in others.resids[glu]],
                'ASN': [int(i) for i in others.resids[low]],
                'geometry': {'glu_low_ca_distance': length, 'top_above_glu_ca': top - length}}

    def rule_text(self):
        return (f"first touch < {self.threshold} Å, {touch_rule(self.window)}; "
                f"occupied = other K+ <= {self.threshold} Å")

    def _window(self, end_1_frame, end_2_frame):
        """Frames searched for the first touch: end_1+1 ... end_2 without a window."""
        if self.window is None:
            return end_1_frame + 1, end_2_frame
        return super()._window(end_1_frame, end_2_frame)

    def _distances(self, ion_id):
        """(other ions, D[residue, ion]) at the current frame; the permeating ion is left out."""
        others = self.ions[self.ions.resids != int(ion_id)]
        if not len(others):
            return others, None
        d = distance_array(self.res_atoms.positions, others.positions, box=self.u.dimensions)
        return others, np.minimum.reduceat(d, self.res_starts, axis=0)

    def occupancy(self, frame, ion_id):
        """
        Every GLU and ASN/ASP at one frame: {pdb: {'ions': [...], 'distance':
        closest other K+ in Å, ...}}, plus per ring the number of occupied
        residues and of distinct ions on them.
        """
        self.u.trajectory[frame]
        others, D = self._distances(ion_id)
        per_res, on_ring = {}, {g: set() for g in RINGS}
        for j, res in enumerate(self.residues):
            ions = []
            if D is not None:
                close = np.where(D[j] <= self.threshold)[0]
                ions = [int(others.resids[k]) for k in close[np.argsort(D[j][close])]]
            on_ring[res['ring']].update(ions)
            per_res[res['pdb']] = {'resid': res['resid'], 'ring': res['ring'],
                                   'residue_type': res['residue_type'], 'ions': ions,
                                   'distance': float(D[j].min()) if D is not None else None}
        out = {'frame': int(frame), 'residues': per_res}
        region = self._region(ion_id)
        for g, key in (('GLU', 'glu'), ('ASN', 'asn')):
            occupied = [p for p, v in per_res.items() if v['ring'] == g and v['ions']]
            out[f'n_{key}_occupied'] = len(occupied)
            out[f'n_ions_on_{key}'] = len(on_ring[g])
            out[f'occupied_{key}'] = occupied
            out[f'{key}_region_ions'] = region[g]
            out[f'n_{key}_region'] = len(region[g])
        out['region_geometry'] = region['geometry']
        return out

    def timecourse(self, end_1_frame, ion_id):
        """
        At end_1 -/+ timecourse_frames (None outside the trajectory): occupied (1)
        or not (0) of every residue, and the number of other K+ in each region.
        """
        n = self.timecourse_frames
        offsets = list(range(-n, n + 1))
        occupied = {res['pdb']: [None] * len(offsets) for res in self.residues}
        region = {g: [None] * len(offsets) for g in RINGS}
        first = max(0, end_1_frame - n)
        last = min(len(self.u.trajectory) - 1, end_1_frame + n)
        for ts in self.u.trajectory[first:last + 1]:
            i = ts.frame - end_1_frame + n
            _, D = self._distances(ion_id)
            for j, res in enumerate(self.residues):
                occupied[res['pdb']][i] = int(D is not None and bool((D[j] <= self.threshold).any()))
            in_region = self._region(ion_id)
            for g in RINGS:
                region[g][i] = len(in_region[g])
        return {'offsets': offsets, 'occupied': occupied, 'region': region}

    def analyze_event(self, event):
        r = super().analyze_event(event)
        touch = r['touch_residues'][0] if r['touch_residues'] else None
        ring = self.ring_of.get(touch['residue_id']) if touch else None
        r['touch_ring'] = ring or NO_CONTACT
        r['touch_ring_label'] = self.ring_labels[ring] if ring else NO_CONTACT
        r['touch_residue_type'] = touch['residue_type'] if touch else None
        r['touch_residue_pdb'] = touch['residue_pdb'] if touch else None
        r['occupancy_at_end_1'] = self.occupancy(r['end_1_frame'], r['ion_id'])
        r['occupancy_at_touch'] = (self.occupancy(r['touch_frame'], r['ion_id'])
                                   if r['touch_frame'] is not None else None)
        r['occupancy_timecourse'] = self.timecourse(r['end_1_frame'], r['ion_id'])
        return r

    def run_analysis(self):
        print("\n" + "="*80)
        print(f"{self.NAME}")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å, {touch_rule(self.window)}")
        print(f"Residues: {', '.join(residue_tick(r) for r in self.residues)}; "
              f"ions: '{self.ion_selection}' ({len(self.ions)} atoms)")
        print(f"Regions: split at the GLU / ASN CA midpoint, radius {self.region_radius} Å, margin {self.margin} Å, "
              f"top = {'SF bottom (CA of ' + str(self.sf_residues) + ')' if self.sf_residues else 'GLU CA + margin'}")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        for event in self.permeation_events:
            r = self.analyze_event(event)
            self.results.append(r)
            occ = r['occupancy_at_end_1']
            where = (f"{r['touch_ring_label']} {r['touch_residue_pdb']} at frame {r['touch_frame']}"
                     if r['touch_frame'] is not None else NO_CONTACT)
            print(f"Ion {r['ion_id']}: end_1 {r['end_1_frame']}: "
                  f"{occ['n_glu_occupied']} GLU occupied [{', '.join(occ['occupied_glu']) or '-'}], "
                  f"{occ['n_asn_occupied']} ASN/ASP occupied [{', '.join(occ['occupied_asn']) or '-'}]; "
                  f"K+ in region: GLU {occ['n_glu_region']}, ASN/ASP {occ['n_asn_region']}; "
                  f"first touch: {where}")
            if occ['region_geometry']['top_above_glu_ca'] < -occ['region_geometry']['glu_low_ca_distance'] / 2:
                print(f"  ⚠ SF bottom is below the GLU / ASN midpoint at frame {r['end_1_frame']}: "
                      f"the GLU region is empty")

    def save_results(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        complete_file = self.results_dir / f"{self.ANALYSIS}_complete.json"
        with open(complete_file, 'w') as f:
            json.dump({
                'analysis': self.ANALYSIS,
                'threshold': self.threshold,
                'window': self.window,
                'timecourse_frames': self.timecourse_frames,
                'channel_type': self.channel_type,
                'ion_selection': self.ion_selection,
                'asn_residues': self.asn_residues,
                'glu_residues': self.glu_residues,
                'sf_residues': self.sf_residues,
                'region_radius': self.region_radius,
                'region_margin': self.margin,
                'residues': self.residues,
                'ring_labels': self.ring_labels,
                'results': self.results
            }, f, indent=2)
        print(f"\n✓ Complete results saved to: {complete_file}")

    def _crosstab(self, at, key):
        """{touch ring: {n occupied: count}} for occupancy at 'end_1' or 'touch'."""
        table = {}
        for r in self.results:
            occ = r[f'occupancy_at_{at}']
            if occ is None:
                continue
            row = table.setdefault(r['touch_ring_label'], {})
            row[occ[key]] = row.get(occ[key], 0) + 1
        return table

    def create_summary(self):
        if not self.results:
            print("No results to summarize.")
            return
        summary = {
            'total_events': len(self.results),
            'threshold': self.threshold,
            'window': self.window,
            'ring_labels': self.ring_labels,
            'touch_counts': {},
            'residue_occupied_at_end_1': {},
        }
        for at in ('end_1', 'touch'):
            for key in ('n_glu_occupied', 'n_asn_occupied', 'n_glu_region', 'n_asn_region'):
                summary[f'{key}_at_{at}'] = self._crosstab(at, key)
        for r in self.results:
            g = r['touch_ring_label']
            summary['touch_counts'][g] = summary['touch_counts'].get(g, 0) + 1
            row = summary['residue_occupied_at_end_1'].setdefault(g, {})
            for p, v in r['occupancy_at_end_1']['residues'].items():
                row[p] = row.get(p, 0) + int(bool(v['ions']))
        output_file = self.results_dir / f"{self.ANALYSIS}_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nOccupied GLUs at end_1, by first-touch ring:")
        for ring, row in summary['n_glu_occupied_at_end_1'].items():
            print(f"  {ring:10s} " + ", ".join(f"{n} occ: {c}" for n, c in sorted(row.items())))
        print(f"Other K+ in the GLU region at end_1, by first-touch ring:")
        for ring, row in summary['n_glu_region_at_end_1'].items():
            print(f"  {ring:10s} " + ", ".join(f"{n} K+: {c}" for n, c in sorted(row.items())))
        print(f"✓ Summary saved to: {output_file}")
        return summary

    def create_plots(self, output_dir=None):
        if not self.results:
            print("No results to plot.")
            return
        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_all_plots([event_row(r) for r in self.results], self.residues, self.ring_labels,
                        self.channel_type, self.threshold, self.window, output_dir,
                        f"t{self.threshold}_{window_tag(self.window)}",
                        touch_rows=[event_row(r, 'touch') for r in self.results])


# ----------------------------------------------------------------------
# Plots (shared with aggregate_glu_occupancy_at_exit.py)
# ----------------------------------------------------------------------

def event_row(r, at='end_1'):
    """One event, flattened for the plots: first-touch ring + occupancy at `at` + time course."""
    occ = r.get(f'occupancy_at_{at}')
    tc = r.get('occupancy_timecourse')
    return {
        'ring': r['touch_ring_label'],
        'touch_residue': r['touch_residue_pdb'],
        'n_glu': occ['n_glu_occupied'] if occ else None,
        'n_asn': occ['n_asn_occupied'] if occ else None,
        'occ': {p: int(bool(v['ions'])) for p, v in occ['residues'].items()} if occ else None,
        'tc': tc['occupied'] if tc else None,
        'offsets': tc['offsets'] if tc else None,
        # other K+ in the GLU / ASN regions (None in result files from before the regions)
        'n_glu_region': occ.get('n_glu_region') if occ else None,
        'n_asn_region': occ.get('n_asn_region') if occ else None,
        'tc_region': tc.get('region') if tc else None,
    }


def _rings_present(rows, ring_labels):
    order = [ring_labels['ASN'], ring_labels['GLU'], NO_CONTACT]
    return [g for g in order if any(r['ring'] == g for r in rows)]


def _ring_colors(ring_labels):
    return {ring_labels['ASN']: RING_COLORS['ASN'], ring_labels['GLU']: RING_COLORS['GLU'],
            NO_CONTACT: RING_COLORS[NO_CONTACT]}


def _occ_colors(n_max):
    """One colour per count 0..n_max: OCC_COLORS up to 4, white -> dark red beyond."""
    if n_max < len(OCC_COLORS):
        return OCC_COLORS
    return [matplotlib.colors.to_hex(c) for c in plt.cm.OrRd(np.linspace(0.02, 1, n_max + 1))]


def _count_names(occ_name, frame_desc, region):
    """(short name, axis label) of what is counted: occupied residues, or other K+ in the ring's region."""
    if region:
        return f'K+ in {occ_name} region', f'Other K+ in the {occ_name} region at {frame_desc}'
    return f'{occ_name} occupied', f'{occ_name} occupied by another K+ at {frame_desc}'


def _style(ax, xlabel, ylabel, ylim=(0, 108)):
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, axis='y', zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=12)


def _save(fig, output_file):
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {output_file}")


def plot_occupancy_vs_touch(rows, key, n_max, ring_labels, occ_name, frame_desc, title, output_file,
                            region=False):
    """
    Two panels from rows (event_row), key 'n_glu' or 'n_asn' ('n_glu_region' or
    'n_asn_region' with region=True: other K+ in the region):
      left   for each first-touch ring, % of its events with 0..n_max residues occupied
      right  for each number of occupied residues, % of events whose first touch was each ring
    Event counts are written over the bars.
    """
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    occ_colors = _occ_colors(n_max)
    short, xlabel = _count_names(occ_name, frame_desc, region)
    occs = list(range(n_max + 1))
    counts = np.array([[sum(1 for r in rows if r['ring'] == g and r[key] == n) for n in occs]
                       for g in rings], dtype=float)          # (ring, n_occ)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: occupancy distribution within each first-touch ring
    ax = axes[0]
    totals = counts.sum(axis=1)
    bottom = np.zeros(len(rings))
    for j, n in enumerate(occs):
        pct = np.divide(counts[:, j], totals, out=np.zeros_like(totals), where=totals > 0) * 100
        ax.bar(range(len(rings)), pct, bottom=bottom, color=occ_colors[n],
               edgecolor='black', linewidth=0.8, zorder=3, label=f"{n}")
        bottom += pct
    for i, t in enumerate(totals):
        ax.text(i, 101, f"n={int(t)}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(rings)))
    ax.set_xticklabels(rings)
    _style(ax, 'First touch after the SF', '% of events')
    ax.set_title(f'{short} at {frame_desc}, per first touch', fontsize=15, fontweight='bold', pad=12)
    ax.legend(title=short, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=11, title_fontsize=11)

    # Right: first-touch ring within each occupancy level
    ax = axes[1]
    totals = counts.sum(axis=0)
    bottom = np.zeros(len(occs))
    for i, g in enumerate(rings):
        pct = np.divide(counts[i], totals, out=np.zeros_like(totals), where=totals > 0) * 100
        ax.bar(occs, pct, bottom=bottom, color=colors[g], edgecolor='black', linewidth=0.8, zorder=3,
               hatch='//' if g == NO_CONTACT else None)
        bottom += pct
    for n, t in zip(occs, totals):
        ax.text(n, 101, f"n={int(t)}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(occs)
    _style(ax, xlabel, '% of events')
    ax.set_title(f'First touch, per number of {short}', fontsize=15, fontweight='bold', pad=12)
    ax.legend(handles=[Patch(facecolor=colors[g], edgecolor='black', hatch='//' if g == NO_CONTACT else None,
                             label=g) for g in rings],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, output_file)


def plot_first_touch_by_occupancy(rows, key, n_max, ring_labels, occ_name, frame_desc, title, output_file,
                                  percent=False, region=False):
    """
    x: number of occupied residues (key 'n_glu' or 'n_asn', 0..n_max; with
    region=True 'n_glu_region' or 'n_asn_region', other K+ in the region); for
    each, one bar per first touch (ASN/ASP, GLU, No contact). y: number of events,
    or with percent=True the % of ALL events (every bar of the plot adds up to 100%).
    """
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    occs = np.arange(n_max + 1)
    width = 0.8 / len(rings)
    total = len(rows)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(max(9, 2 * len(occs)), 6))
    top = 0
    for i, g in enumerate(rings):
        counts = np.array([sum(1 for r in rows if r['ring'] == g and r[key] == n) for n in occs], dtype=float)
        values = 100 * counts / total if percent else counts
        xs = occs + (i - (len(rings) - 1) / 2) * width
        ax.bar(xs, values, width, color=colors[g], edgecolor='black', linewidth=0.8, zorder=3,
               hatch='//' if g == NO_CONTACT else None, label=g)
        for xi, v in zip(xs, values):
            ax.text(xi, v, f"{v:.1f}%" if percent else f"{int(v)}", ha='center', va='bottom',
                    fontsize=9 if percent else 10)
        top = max(top, values.max())
    ax.set_xticks(occs)
    _style(ax, _count_names(occ_name, frame_desc, region)[1],
           f'% of all events (n={total})' if percent else 'Number of events', ylim=(0, top * 1.18 + 1))
    if not percent:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title='First touch', loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11, title_fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    _save(fig, output_file)


def _combination_counts(rows, pdbs, k, top):
    """Events with exactly k of `pdbs` occupied, and their `top` most common combinations [(combo, n)]."""
    sub = [r for r in rows if sum(r['occ'].get(p, 0) for p in pdbs) == k]
    counts = {}
    for r in sub:
        combo = tuple(p for p in pdbs if r['occ'].get(p, 0))
        counts[combo] = counts.get(combo, 0) + 1
    return sub, sorted(counts.items(), key=lambda c: -c[1])[:top]


def _combination_panel(ax, sub, combos, k, ticks, ring_name, rings, colors):
    """
    One panel: a bar per combination of k occupied residues, stacked by first
    touch. Above the bar: events (% of the events with k occupied); inside each
    colour: % of that bar (left out where the colour is too thin to fit it).
    """
    ax.set_title(f"{k} {ring_name} occupied (n={len(sub)})", fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if not combos:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes, ha='center', va='center', fontsize=13)
        ax.set_xticks([])
        return
    ymax = max(n for _, n in combos) * 1.25 + 1
    x = np.arange(len(combos))
    bottom = np.zeros(len(combos))
    for g in rings:
        v = np.array([sum(1 for r in sub if r['ring'] == g and tuple(p for p in c if r['occ'].get(p, 0)) == c)
                      for c, _ in combos], dtype=float)
        ax.bar(x, v, 0.7, bottom=bottom, color=colors[g], edgecolor='black', linewidth=0.8, zorder=3,
               hatch='//' if g == NO_CONTACT else None, label=g)
        for xi, vi, b, (_, n) in zip(x, v, bottom, combos):
            if vi > 0 and vi / ymax >= 0.045:            # only where the label fits
                ax.text(xi, b + vi / 2, f"{100 * vi / n:.0f}%", ha='center', va='center', fontsize=10,
                        fontweight='bold', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8)
                        if g == NO_CONTACT else None)
        bottom += v
    for xi, (_, n) in zip(x, combos):
        ax.text(xi, n, f"{n}\n({100 * n / len(sub):.0f}%)", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([' +\n'.join(ticks[p] for p in c) for c, _ in combos], fontsize=10)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_occupied_combinations(rows, residues, ring, ring_labels, frame_desc, title, output_file,
                               top=10, separate_files=True):
    """
    Which residues of one ring ('GLU' or 'ASN') are occupied together at
    `frame_desc`: one panel per number occupied (1, 2, 3, 4), a bar per
    combination (most common first, at most `top`), stacked by first touch.
    Writes the figure with all panels to output_file and, with
    separate_files, every panel on its own as <output_file>_<k>occupied.png.
    """
    rows = [r for r in rows if r['occ'] is not None]
    if not rows:
        return
    pdbs = [res['pdb'] for res in residues if res['ring'] == ring]
    ticks = {res['pdb']: residue_tick(res) for res in residues}
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    ring_name = ring_labels[ring]
    panels = [(k,) + _combination_counts(rows, pdbs, k, top) for k in range(1, len(pdbs) + 1)]
    n_free = sum(1 for r in rows if not any(r['occ'].get(p, 0) for p in pdbs))
    subtitle = (f"above bar: events (% of the events with that many occupied); inside: % of the bar per first touch; "
                f"0 {ring_name} occupied: n={n_free} of {len(rows)}")

    def legend(ax):
        ax.legend(title='First touch', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, title_fontsize=11)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    widths = [max(1.6, len(combos)) for _, _, combos in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(12, 1.3 * sum(widths) + 3), 7),
                             gridspec_kw={'width_ratios': widths}, squeeze=False)
    for ax, (k, sub, combos) in zip(axes[0], panels):
        _combination_panel(ax, sub, combos, k, ticks, ring_name, rings, colors)
    axes[0][0].set_ylabel('Number of events', fontsize=14, fontweight='bold')
    legend(axes[0][-1])
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight='bold')
    _save(fig, output_file)

    if separate_files:
        output_file = Path(output_file)
        for k, sub, combos in panels:
            fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(combos) + 3), 6.5))
            _combination_panel(ax, sub, combos, k, ticks, ring_name, rings, colors)
            ax.set_ylabel('Number of events', fontsize=14, fontweight='bold')
            legend(ax)
            fig.suptitle(f"{title}\n{subtitle}", fontsize=12, fontweight='bold')
            _save(fig, output_file.with_name(f"{output_file.stem}_{k}occupied{output_file.suffix}"))


def plot_residue_occupancy(rows, residues, ring_labels, frame_desc, title, output_file):
    """Each GLU and ASN/ASP on the x axis: % of events it is occupied, one bar per first-touch ring."""
    rows = [r for r in rows if r['occ'] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    x = np.arange(len(residues))
    width = 0.8 / len(rings)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(residues)), 6))
    n_glu = sum(1 for res in residues if res['ring'] == 'GLU')
    ax.axvspan(-0.5, n_glu - 0.5, color=RING_COLORS['GLU'], alpha=0.08, zorder=0)
    ax.axvspan(n_glu - 0.5, len(residues) - 0.5, color=RING_COLORS['ASN'], alpha=0.08, zorder=0)
    for i, g in enumerate(rings):
        sub = [r for r in rows if r['ring'] == g]
        pct = [100 * np.mean([r['occ'].get(res['pdb'], 0) for r in sub]) for res in residues]
        xs = x + (i - (len(rings) - 1) / 2) * width
        ax.bar(xs, pct, width, color=colors[g], edgecolor='black', linewidth=0.8, zorder=3,
               hatch='//' if g == NO_CONTACT else None, label=f"first touch {g} (n={len(sub)})")
        for xi, p in zip(xs, pct):
            ax.text(xi, p + 1, f"{p:.0f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([residue_tick(res) for res in residues], rotation=30, ha='right')
    ax.set_xlim(-0.5, len(residues) - 0.5)
    _style(ax, 'Residue', f'% of events occupied at {frame_desc}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    _save(fig, output_file)


def plot_touched_residue_heatmap(rows, residues, frame_desc, title, output_file):
    """
    Rows: the residue touched first (each one separately, plus No contact).
    Columns: every GLU and ASN/ASP. Cell: % of those events in which the
    column residue was occupied by another K+ at `frame_desc`.
    """
    rows = [r for r in rows if r['occ'] is not None]
    if not rows:
        return
    ticks = {res['pdb']: residue_tick(res) for res in residues}
    touched = [res['pdb'] for res in residues if any(r['touch_residue'] == res['pdb'] for r in rows)]
    groups = [(ticks[p], [r for r in rows if r['touch_residue'] == p]) for p in touched]
    nc = [r for r in rows if r['touch_residue'] is None]
    if nc:
        groups.append((NO_CONTACT, nc))
    M = np.array([[100 * np.mean([r['occ'].get(res['pdb'], 0) for r in sub]) for res in residues]
                  for _, sub in groups])

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(residues) + 3), max(4, 0.65 * len(groups) + 2.5)))
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha='center', va='center', fontsize=10,
                    color='white' if M[i, j] > 60 else 'black')
    n_glu = sum(1 for res in residues if res['ring'] == 'GLU')
    ax.axvline(n_glu - 0.5, color='black', linewidth=1.5)
    ax.set_xticks(range(len(residues)))
    ax.set_xticklabels([residue_tick(res) for res in residues], rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{name} (n={len(sub)})" for name, sub in groups], fontsize=11)
    ax.set_xlabel(f'Residue occupied by another K+ at {frame_desc}', fontsize=13, fontweight='bold')
    ax.set_ylabel('First residue touched', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% of events occupied', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    _save(fig, output_file)


def _mean_sem(values):
    """Mean and SEM per column of a 2D list with None (outside the trajectory) left out."""
    A = np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)
    n = np.sum(~np.isnan(A), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # frames outside the trajectory in every event
        mean = np.nanmean(A, axis=0) if len(A) else np.full(A.shape[1:], np.nan)
        sem = np.where(n > 1, np.nanstd(A, axis=0, ddof=1) / np.sqrt(n), 0)
    return mean, sem


def plot_ring_timecourse(rows, residues, ring_labels, title, output_file):
    """Mean number of occupied GLUs (left) and ASN/ASP (right) vs frames from end_1, per first-touch ring (± SEM)."""
    rows = [r for r in rows if r['tc'] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    offsets = np.array(rows[0]['offsets'])

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, ring in zip(axes, RINGS):
        pdbs = [res['pdb'] for res in residues if res['ring'] == ring]
        for g in rings:
            sub = [r for r in rows if r['ring'] == g]
            # per event, per frame: number of occupied residues of this ring (None outside the trajectory)
            counts = [[None if any(r['tc'][p][i] is None for p in pdbs) else sum(r['tc'][p][i] for p in pdbs)
                       for i in range(len(offsets))] for r in sub]
            mean, sem = _mean_sem(counts)
            ax.plot(offsets, mean, color=colors[g], linewidth=2.2, marker='o', markersize=4,
                    label=f"first touch {g} (n={len(sub)})", linestyle='--' if g == NO_CONTACT else '-')
            ax.fill_between(offsets, mean - sem, mean + sem, color=colors[g], alpha=0.2)
        ax.axvline(0, color='black', linestyle=':', linewidth=1.2)
        _style(ax, 'Frames from end_1 (leaving the SF)', f'Mean {ring_labels[ring]} occupied', ylim=None)
        ax.set_ylim(0, len(pdbs))
        ax.set_title(f'{ring_labels[ring]} ring', fontsize=15, fontweight='bold', pad=12)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, output_file)


def plot_region_timecourse(rows, ring_labels, title, output_file):
    """Mean number of other K+ in the GLU region (left) and ASN/ASP region (right) vs frames from end_1 (± SEM)."""
    rows = [r for r in rows if r['tc_region'] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    offsets = np.array(rows[0]['offsets'])

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, ring in zip(axes, RINGS):
        for g in rings:
            sub = [r for r in rows if r['ring'] == g]
            mean, sem = _mean_sem([r['tc_region'][ring] for r in sub])
            ax.plot(offsets, mean, color=colors[g], linewidth=2.2, marker='o', markersize=4,
                    label=f"first touch {g} (n={len(sub)})", linestyle='--' if g == NO_CONTACT else '-')
            ax.fill_between(offsets, mean - sem, mean + sem, color=colors[g], alpha=0.2)
        ax.axvline(0, color='black', linestyle=':', linewidth=1.2)
        _style(ax, 'Frames from end_1 (leaving the SF)', f'Mean other K+ in {ring_labels[ring]} region', ylim=None)
        ax.set_ylim(bottom=0)
        ax.set_title(f'{ring_labels[ring]} region', fontsize=15, fontweight='bold', pad=12)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    _save(fig, output_file)


def plot_residue_timecourse(rows, residues, ring_labels, title, output_file):
    """One panel per residue: % of events it is occupied vs frames from end_1, one line per first-touch ring."""
    rows = [r for r in rows if r['tc'] is not None]
    if not rows:
        return
    rings = _rings_present(rows, ring_labels)
    colors = _ring_colors(ring_labels)
    offsets = np.array(rows[0]['offsets'])
    ncols = max(sum(1 for res in residues if res['ring'] == g) for g in RINGS)
    by_ring = [[res for res in residues if res['ring'] == g] for g in RINGS]
    nrows = sum(1 for b in by_ring if b)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for i, ring_res in enumerate(b for b in by_ring if b):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(ring_res):
                ax.axis('off')
                continue
            res = ring_res[j]
            for g in rings:
                sub = [r for r in rows if r['ring'] == g]
                mean, _ = _mean_sem([r['tc'][res['pdb']] for r in sub])
                ax.plot(offsets, 100 * mean, color=colors[g], linewidth=2, marker='o', markersize=3,
                        linestyle='--' if g == NO_CONTACT else '-', label=f"first touch {g} (n={len(sub)})")
            ax.axvline(0, color='black', linestyle=':', linewidth=1)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.25, zorder=0)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_title(residue_tick(res), fontsize=13, fontweight='bold',
                         color=RING_COLORS[res['ring']])
            if j == 0:
                ax.set_ylabel('% occupied', fontsize=12, fontweight='bold')
            if i == nrows - 1:
                ax.set_xlabel('Frames from end_1', fontsize=12, fontweight='bold')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(rings), fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    _save(fig, output_file)


def write_all_plots(rows, residues, ring_labels, channel_type, threshold, window, output_dir, tag,
                    touch_rows=None, label=''):
    """Every plot of this analysis for one set of events (one run, or a group of runs)."""
    output_dir = Path(output_dir)
    head = f"{channel_type}{' ' + label if label else ''}"
    rule = f"(K+ <= {threshold} Å, {touch_rule(window)})"
    n_ring = {g: sum(1 for res in residues if res['ring'] == g) for g in RINGS}
    frames = [('end_1', 'end_1', rows)] + ([('touch', 'first touch', touch_rows)] if touch_rows else [])
    for at, desc, rr in frames:
        for key, ring in (('n_glu', 'GLU'), ('n_asn', 'ASN')):
            plot_occupancy_vs_touch(rr, key, n_ring[ring], ring_labels, ring_labels[ring], desc,
                                    f"{head}: {ring_labels[ring]} occupied at {desc} vs first touch {rule}",
                                    output_dir / f"{ring.lower()}_occupancy_{at}_{tag}.png")
            for percent, kind in ((False, 'counts'), (True, 'percent')):
                plot_first_touch_by_occupancy(
                    rr, key, n_ring[ring], ring_labels, ring_labels[ring], desc,
                    f"{head}: first touch per number of {ring_labels[ring]} occupied at {desc}\n{rule}",
                    output_dir / f"first_touch_by_{ring.lower()}_occupied_{at}_{kind}_{tag}.png", percent)
            plot_occupied_combinations(
                rr, residues, ring, ring_labels, desc,
                f"{head}: which {ring_labels[ring]} are occupied together at {desc} {rule}",
                output_dir / f"{ring.lower()}_combinations_{at}_{tag}.png")
        plot_residue_occupancy(rr, residues, ring_labels, desc,
                               f"{head}: each residue occupied at {desc}, per first touch {rule}",
                               output_dir / f"residue_occupancy_{at}_{tag}.png")
        plot_touched_residue_heatmap(rr, residues, desc,
                                     f"{head}: first residue touched vs residues occupied at {desc}\n{rule}",
                                     output_dir / f"touched_residue_heatmap_{at}_{tag}.png")
    plot_ring_timecourse(rows, residues, ring_labels,
                         f"{head}: occupied residues around SF exit, per first touch {rule}",
                         output_dir / f"ring_occupancy_timecourse_{tag}.png")
    plot_residue_timecourse(rows, residues, ring_labels,
                            f"{head}: each residue occupied around SF exit, per first touch {rule}",
                            output_dir / f"residue_occupancy_timecourse_{tag}.png")

    # The same ring plots, counting the other K+ in the GLU / ASN region (region/)
    region_rule = f"({touch_rule(window)}; regions split at the {ring_labels['GLU']}/{ring_labels['ASN']} CA midpoint)"
    region_dir = output_dir / 'region'
    has_region = False
    for key, ring in (('n_glu_region', 'GLU'), ('n_asn_region', 'ASN')):
        values = [r[key] for _, _, rr in frames for r in rr if r[key] is not None]
        if not values:
            continue                       # result files from before the regions
        region_dir.mkdir(parents=True, exist_ok=True)
        has_region = True
        n_max = max(1, max(values))        # same x range at end_1 and at the touch
        for at, desc, rr in frames:
            plot_occupancy_vs_touch(rr, key, n_max, ring_labels, ring_labels[ring], desc,
                                    f"{head}: K+ in the {ring_labels[ring]} region at {desc} vs first touch\n"
                                    f"{region_rule}",
                                    region_dir / f"{ring.lower()}_region_occupancy_{at}_{tag}.png", region=True)
            for percent, kind in ((False, 'counts'), (True, 'percent')):
                plot_first_touch_by_occupancy(
                    rr, key, n_max, ring_labels, ring_labels[ring], desc,
                    f"{head}: first touch per number of K+ in the {ring_labels[ring]} region at {desc}\n"
                    f"{region_rule}",
                    region_dir / f"first_touch_by_{ring.lower()}_region_ions_{at}_{kind}_{tag}.png",
                    percent, region=True)
    if has_region:
        plot_region_timecourse(rows, ring_labels,
                               f"{head}: K+ in the GLU / ASN regions around SF exit, per first touch\n{region_rule}",
                               region_dir / f"region_ion_timecourse_{tag}.png")


def run_glu_occupancy_at_exit_analysis(universe, permeation_events, asn_residues, glu_residues,
                                       channel_type="G12", threshold=3.5, window=10,
                                       analysis_start_frame=0, ion_selection=ION_SELECTION,
                                       results_dir=".", timecourse_frames=10,
                                       sf_residues=None, region_radius=12.0, margin=3.0):
    """
    Convenience function: first touch after the SF + occupancy of every GLU
    and ASN/ASP by other K+ (<= threshold) at end_1, at the touch frame and
    around end_1, and the number of other K+ in the GLU and ASN/ASP regions
    (sf_residues: the SF residues whose CA, minus `margin`, is the top of the
    GLU region; lower1 in main.py). window=None: no window, the first touch is searched over
    the whole stay in the cavity (end_1 to end_2). Suggested results_dir:
    f"{results_dir}/glu_occupancy_exit/{settings_folder(threshold, window)}"
    (3.5_w10, or 3.5_no_w without a window)

    Returns:
    --------
    GluOccupancyAtExitAnalysis : The analysis object with results
    """
    analyzer = GluOccupancyAtExitAnalysis(universe, permeation_events, asn_residues, glu_residues,
                                          channel_type, threshold, window, analysis_start_frame,
                                          ion_selection, results_dir, timecourse_frames,
                                          sf_residues, region_radius, margin)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    analyzer.create_plots()
    return analyzer
