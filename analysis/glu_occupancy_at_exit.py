"""
glu_occupancy_at_exit.py
========================
Are the glutamates already occupied by other K+ ions when a permeating ion
leaves the selectivity filter?

Idea being tested: the first touch after the SF is ASN/ASP only because the
GLU ring is already taken by other ions; when the GLU ring is free, the first
touch is a glutamate.

For every permeation event this runs the FIRST-TOUCH analysis
(FirstTouchAnalysis in longest_duration_analysis.py, residue mode: first
ASN/ASP/GLU residue within `threshold` in the first `window` frames after
end_1), and also reads the GLU ring at two frames:

    end_1        the last frame of the ion in the SF (the moment it leaves)
    touch_frame  the frame of the first touch (None if there was no touch)

A glutamate is "occupied" when another K+ (not the permeating ion) is within
`threshold` of any of its atoms, the same distance rule as the contacts.
Stored per event: which GLUs are occupied, by which ions, how many GLUs are
occupied (0-4) and how many distinct ions sit on the GLU ring.

The summary cross-tabulates the first-touch ring (ASN/ASP, GLU, No contact)
against the number of occupied GLUs at end_1 (and at the touch frame).

Results (results_dir, suggested f"{results_dir}/glu_occupancy_exit/{threshold}_w{window}"):
    glu_occupancy_exit_complete.json   every event, all fields
    glu_occupancy_exit_summary.json    cross-tables (counts)
    plots/glu_occupancy_exit_t*_w*.png
Aggregate over runs with aggregate_glu_occupancy_at_exit.py.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from MDAnalysis.lib.distances import distance_array

from analysis.converter import convert_to_pdb_numbering
from analysis.longest_duration_analysis import FirstTouchAnalysis


ION_SELECTION = "resname K+ K"
NO_CONTACT = 'No contact'
# n occupied GLUs: white (free ring) -> dark red (all four taken)
OCC_COLORS = ['#F7F7F7', '#FDD0A2', '#FC8D59', '#D7301F', '#7F0000']
RING_COLORS = {'GLU': '#E8746B', 'ASN': '#5AA9C9', NO_CONTACT: '#D9D9D9'}


def ring_name(resnames, resids):
    """'ASN/ASP', 'GLU', 'GLN' (E141Q), ...: the residue names of one ring."""
    return '/'.join(sorted({resnames.get(int(r), 'Unknown') for r in resids}))


class GluOccupancyAtExitAnalysis(FirstTouchAnalysis):
    """
    First touch after the SF (residue mode) + GLU-ring occupancy by other
    K+ ions at end_1 and at the touch frame. See the module docstring.
    """

    ANALYSIS = 'glu_occupancy_exit'
    NAME = 'GLU OCCUPANCY AT SF EXIT'
    FILE_PREFIX = 'glu_occupancy_exit'

    def __init__(self, universe, permeation_events, asn_residues, glu_residues,
                 channel_type="G12", threshold=3.5, window=10, analysis_start_frame=0,
                 ion_selection=ION_SELECTION, results_dir="."):
        # Residue mode, both rings compete for the first touch
        super().__init__(universe, permeation_events, asn_residues, glu_residues,
                         channel_type, threshold, window, analysis_start_frame,
                         ['ASN', 'GLU'], None, results_dir)
        self.ions = self.u.select_atoms(ion_selection)
        self.ion_selection = ion_selection

        groups = [(int(r), self.u.select_atoms(f"resid {r}")) for r in glu_residues]
        groups = [(r, g) for r, g in groups if len(g) > 0]
        self.glu_ids = [r for r, _ in groups]
        self.glu_pdb = {r: convert_to_pdb_numbering(r, channel_type) for r in self.glu_ids}
        self.glu_atoms = self.u.atoms[np.concatenate([g.indices for _, g in groups])]
        self.glu_starts = np.cumsum([0] + [len(g) for _, g in groups[:-1]])

        # Ring labels from the topology, so mutants are named correctly
        self.ring_labels = {'GLU': ring_name(self.resnames, glu_residues),
                            'ASN': ring_name(self.resnames, asn_residues)}
        self.ring_of = {int(r): 'GLU' for r in glu_residues}
        self.ring_of.update({int(r): 'ASN' for r in asn_residues})

    def rule_text(self):
        return (f"first touch < {self.threshold} Å within {self.window} frames after the SF; "
                f"GLU occupied = other K+ < {self.threshold} Å")

    def glu_occupancy(self, frame, ion_id):
        """
        GLU ring at one frame: {glu_pdb: {'ions': [...], 'distance': closest
        other K+ in Å}} plus the number of occupied GLUs and of distinct ions
        on them. The permeating ion itself is not counted.
        """
        self.u.trajectory[frame]
        others = self.ions[self.ions.resids != int(ion_id)]
        per_glu, on_ring = {}, set()
        if len(others):
            d = distance_array(self.glu_atoms.positions, others.positions, box=self.u.dimensions)
            D = np.minimum.reduceat(d, self.glu_starts, axis=0)      # (n_glu, n_ions)
            for j, resid in enumerate(self.glu_ids):
                close = np.where(D[j] < self.threshold)[0]
                ions = [int(others.resids[k]) for k in close[np.argsort(D[j][close])]]
                on_ring.update(ions)
                per_glu[self.glu_pdb[resid]] = {'resid': resid, 'ions': ions,
                                                'distance': float(D[j].min())}
        return {
            'frame': int(frame),
            'n_glu_occupied': sum(1 for g in per_glu.values() if g['ions']),
            'n_ions_on_glu': len(on_ring),
            'occupied_glu': [p for p, g in per_glu.items() if g['ions']],
            'glu': per_glu,
        }

    def analyze_event(self, event):
        r = super().analyze_event(event)
        touch = r['touch_residues'][0] if r['touch_residues'] else None
        ring = self.ring_of.get(touch['residue_id']) if touch else None
        r['touch_ring'] = ring or NO_CONTACT
        r['touch_ring_label'] = self.ring_labels[ring] if ring else NO_CONTACT
        r['touch_residue_type'] = touch['residue_type'] if touch else None
        r['touch_residue_pdb'] = touch['residue_pdb'] if touch else None
        r['occupancy_at_end_1'] = self.glu_occupancy(r['end_1_frame'], r['ion_id'])
        r['occupancy_at_touch'] = (self.glu_occupancy(r['touch_frame'], r['ion_id'])
                                   if r['touch_frame'] is not None else None)
        return r

    def run_analysis(self):
        print("\n" + "="*80)
        print(f"{self.NAME}")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å, window: {self.window} frames")
        print(f"GLU ring: {', '.join(self.glu_pdb.values())}; ions: '{self.ion_selection}' ({len(self.ions)} atoms)")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")
        for event in self.permeation_events:
            r = self.analyze_event(event)
            self.results.append(r)
            occ = r['occupancy_at_end_1']
            where = (f"{r['touch_ring_label']} {r['touch_residue_pdb']} at frame {r['touch_frame']}"
                     if r['touch_frame'] is not None else NO_CONTACT)
            print(f"Ion {r['ion_id']}: end_1 {r['end_1_frame']}: {occ['n_glu_occupied']} GLU occupied "
                  f"[{', '.join(occ['occupied_glu']) or '-'}]; first touch: {where}")

    def save_results(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        complete_file = self.results_dir / f"{self.ANALYSIS}_complete.json"
        with open(complete_file, 'w') as f:
            json.dump({
                'analysis': self.ANALYSIS,
                'threshold': self.threshold,
                'window': self.window,
                'channel_type': self.channel_type,
                'ion_selection': self.ion_selection,
                'asn_residues': self.asn_residues,
                'glu_residues': self.glu_residues,
                'glu_pdb': list(self.glu_pdb.values()),
                'ring_labels': self.ring_labels,
                'results': self.results
            }, f, indent=2)
        print(f"\n✓ Complete results saved to: {complete_file}")

    def _crosstab(self, at):
        """{touch ring: {n occupied: count}} for occupancy at 'end_1' or 'touch'."""
        table = {}
        for r in self.results:
            occ = r[f'occupancy_at_{at}']
            if occ is None:
                continue
            row = table.setdefault(r['touch_ring_label'], {})
            row[occ['n_glu_occupied']] = row.get(occ['n_glu_occupied'], 0) + 1
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
            'n_glu_occupied_at_end_1': self._crosstab('end_1'),
            'n_glu_occupied_at_touch': self._crosstab('touch'),
        }
        for r in self.results:
            summary['touch_counts'][r['touch_ring_label']] = summary['touch_counts'].get(r['touch_ring_label'], 0) + 1
        output_file = self.results_dir / f"{self.ANALYSIS}_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nOccupied GLUs at end_1, by first-touch ring:")
        for ring, row in summary['n_glu_occupied_at_end_1'].items():
            print(f"  {ring:10s} " + ", ".join(f"{n} occ: {c}" for n, c in sorted(row.items())))
        print(f"✓ Summary saved to: {output_file}")
        return summary

    def create_plots(self, output_dir=None):
        if not self.results:
            print("No results to plot.")
            return
        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [{'ring': r['touch_ring_label'], 'ring_key': r['touch_ring'],
                 'n_occ': r['occupancy_at_end_1']['n_glu_occupied']} for r in self.results]
        output_file = output_dir / f"{self.FILE_PREFIX}_t{self.threshold}_w{self.window}.png"
        plot_occupancy_vs_touch(rows, len(self.glu_ids), self.ring_labels,
                                f"{self.channel_type}: GLU occupancy at SF exit (end_1) vs first touch",
                                output_file)


def plot_occupancy_vs_touch(rows, n_glu, ring_labels, title, output_file):
    """
    Two panels from rows of {'ring', 'ring_key', 'n_occ'} (one per event):
      left   for each first-touch ring, % of its events with 0..n_glu GLUs occupied
      right  for each number of occupied GLUs, % of events whose first touch was each ring
    Event counts are written over the bars. Shared with the aggregator.
    """
    rings = [ring_labels['ASN'], ring_labels['GLU'], NO_CONTACT]
    ring_color = {ring_labels['ASN']: RING_COLORS['ASN'], ring_labels['GLU']: RING_COLORS['GLU'],
                  NO_CONTACT: RING_COLORS[NO_CONTACT]}
    rings = [g for g in rings if any(r['ring'] == g for r in rows)]
    occs = list(range(n_glu + 1))
    counts = np.array([[sum(1 for r in rows if r['ring'] == g and r['n_occ'] == n) for n in occs]
                       for g in rings], dtype=float)          # (ring, n_occ)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def style(ax, xlabel):
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel('% of events', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 108)
        ax.grid(True, alpha=0.25, axis='y', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=12)

    # Left: occupancy distribution within each first-touch ring
    ax = axes[0]
    totals = counts.sum(axis=1)
    bottom = np.zeros(len(rings))
    for j, n in enumerate(occs):
        pct = np.divide(counts[:, j], totals, out=np.zeros_like(totals), where=totals > 0) * 100
        ax.bar(range(len(rings)), pct, bottom=bottom, color=OCC_COLORS[min(n, len(OCC_COLORS) - 1)],
               edgecolor='black', linewidth=0.8, zorder=3, label=f"{n}")
        bottom += pct
    for i, t in enumerate(totals):
        ax.text(i, 101, f"n={int(t)}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(rings)))
    ax.set_xticklabels(rings)
    style(ax, 'First touch after the SF')
    ax.set_title('Occupied GLUs at end_1, per first touch', fontsize=15, fontweight='bold', pad=12)
    ax.legend(title='GLUs occupied', loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11, title_fontsize=11)

    # Right: first-touch ring within each occupancy level
    ax = axes[1]
    totals = counts.sum(axis=0)
    bottom = np.zeros(len(occs))
    for i, g in enumerate(rings):
        pct = np.divide(counts[i], totals, out=np.zeros_like(totals), where=totals > 0) * 100
        ax.bar(occs, pct, bottom=bottom, color=ring_color[g], edgecolor='black', linewidth=0.8, zorder=3,
               hatch='//' if g == NO_CONTACT else None)
        bottom += pct
    for n, t in zip(occs, totals):
        ax.text(n, 101, f"n={int(t)}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(occs)
    style(ax, 'GLUs occupied by another K+ at end_1')
    ax.set_title('First touch, per number of occupied GLUs', fontsize=15, fontweight='bold', pad=12)
    ax.legend(handles=[Patch(facecolor=ring_color[g], edgecolor='black', hatch='//' if g == NO_CONTACT else None,
                             label=g) for g in rings],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {output_file}")


def run_glu_occupancy_at_exit_analysis(universe, permeation_events, asn_residues, glu_residues,
                                       channel_type="G12", threshold=3.5, window=10,
                                       analysis_start_frame=0, ion_selection=ION_SELECTION,
                                       results_dir="."):
    """
    Convenience function: first touch after the SF + GLU occupancy by other
    K+ at end_1 and at the touch frame. Suggested results_dir:
    f"{results_dir}/glu_occupancy_exit/{threshold}_w{window}"

    Returns:
    --------
    GluOccupancyAtExitAnalysis : The analysis object with results
    """
    analyzer = GluOccupancyAtExitAnalysis(universe, permeation_events, asn_residues, glu_residues,
                                          channel_type, threshold, window, analysis_start_frame,
                                          ion_selection, results_dir)
    analyzer.run_analysis()
    analyzer.save_results()
    analyzer.create_summary()
    analyzer.create_plots()
    return analyzer
