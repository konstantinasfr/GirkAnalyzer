"""
aggregate_glu_occupancy_at_exit.py
==================================
Aggregates analysis/glu_occupancy_at_exit.py over all runs of one channel:
for every permeation event, the first residue touched after the SF (ASN/ASP
ring, GLU ring, or No contact) against which GLU and ASN/ASP residues were
occupied by another K+ (<= threshold from any atom of the residue) when the
ion left the SF (end_1), at the first-touch frame, and around end_1.

Question: is the first touch ASN/ASP only when the GLU ring is already
occupied (and GLU when it is free)?

The runs are aggregated into ALL/, HIGH/, TYPICAL/, LOW/ (and D/, E/ for
G12_ML), with the same category files as aggregate_duration_results.py.
Events of all runs in a folder are pooled.

Runs analysed without a window (window=None in main.py, results in
RUN*/glu_occupancy_exit/<t>_no_w/) are aggregated with --no_window.

Outputs (per folder; t = threshold, w = window or no_w; <at> = end_1 or touch):
    glu_occupancy_<at>_t*_w*.png            left: occupied GLUs per first-touch ring
                                            right: first-touch ring per number of occupied GLUs
    asn_occupancy_<at>_t*_w*.png            the same for the ASN/ASP ring
    residue_occupancy_<at>_t*_w*.png        % occupied of each residue separately, per first-touch ring
    touched_residue_heatmap_<at>_t*_w*.png  first-touched residue x occupied residue (% of events)
    ring_occupancy_timecourse_t*_w*.png     mean occupied GLUs / ASN/ASP vs frames from end_1
    residue_occupancy_timecourse_t*_w*.png  % occupied vs frames from end_1, one panel per residue
    summary_t*_w*.xlsx                      sheets:
        <at>_glu_counts / _glu_pct_touch / _glu_pct_occ   GLU cross-tables (counts, row %, column %)
        <at>_asn_counts / ...                              the same for ASN/ASP
        <at>_residue_pct            % occupied of each residue, per first-touch ring
        <at>_heatmap                % occupied of each residue, per first-touched residue
        <at>_residue_fisher         per residue: occupied when the first touch was ASN/ASP
                                    vs GLU, Fisher exact test
        stats                       Fisher (first touch GLU vs ASN/ASP x GLU ring free vs >= 1
                                    occupied), Mann-Whitney on occupied GLUs, mean +/- SEM
        timecourse                  mean occupied GLUs / ASN/ASP per frame from end_1
    per_run_t*_w*.xlsx                      one row per run: counts per ring, mean occupied
    events_t*_w*.xlsx                       one row per event, one 0/1 column per residue

Default output folder: <base>/aggregated_glu_occupancy_exit/t*_w*/

USAGE
-----
    python3 aggregate_glu_occupancy_at_exit.py /path/to/G12 --threshold 3.5 --window 10
    python3 aggregate_glu_occupancy_at_exit.py /path/to/G12 --threshold 3.5 --no_window
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from aggregate_duration_results import (CATEGORIES, LIGAND_CATEGORIES, LIGAND_CHANNELS, run_sort_key,
                                        load_categories, load_ligand_categories, report_coverage)
from analysis.glu_occupancy_at_exit import (NO_CONTACT, RINGS, event_row, residue_tick, window_tag,
                                            write_all_plots)

FRAMES = {'end_1': 'end_1 (leaving the SF)', 'touch': 'the first-touch frame'}


def _fisher(table):
    """(odds ratio, p) of a 2x2 table, or (nan, nan) when a row or column is empty."""
    t = np.array(table)
    if (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
        return np.nan, np.nan
    return fisher_exact(t)


class GluOccupancyAggregator:
    """Loads the per-run glu_occupancy_exit_complete.json files, then writes each folder."""

    def __init__(self, base_directory, threshold, window, channel_type=None):
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.window = window
        self.channel_type = channel_type or self.base_dir.name
        self.threshold_str = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        self.tag = f"t{self.threshold_str}_{window_tag(window)}"
        self.events = []          # one dict per event: run + event_row at end_1 and at the touch
        self.runs_loaded = []
        self.ring_labels = None
        self.residues = None

    def _find_file(self, run_dir):
        for t in (self.threshold_str, str(self.threshold)):
            f = run_dir / "glu_occupancy_exit" / f"{t}_{window_tag(self.window)}" / "glu_occupancy_exit_complete.json"
            if f.exists():
                return f
        return None

    def load_all_runs(self):
        print(f"\nSearching for runs in: {self.base_dir}")
        run_dirs = sorted((d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')),
                          key=lambda d: run_sort_key(d.name))
        for run_dir in run_dirs:
            f = self._find_file(run_dir)
            if f is None:
                continue
            with open(f) as fh:
                data = json.load(fh)
            if 'residues' not in data:
                print(f"  ✗ {run_dir.name}: old result file (GLUs only), re-run the analysis")
                continue
            self.ring_labels = self.ring_labels or data['ring_labels']
            self.residues = self.residues or data['residues']
            for r in data['results']:
                self.events.append({'run': run_dir.name, 'raw': r,
                                    'end_1': event_row(r, 'end_1'), 'touch': event_row(r, 'touch')})
            self.runs_loaded.append(run_dir.name)
            print(f"  ✓ {run_dir.name}: {len(data['results'])} events")
        print(f"Loaded {len(self.runs_loaded)} runs, {len(self.events)} events")
        return bool(self.runs_loaded)

    # ------------------------------------------------------------------

    def _rings(self, rows):
        order = [self.ring_labels['ASN'], self.ring_labels['GLU'], NO_CONTACT]
        return [g for g in order if any(r['ring'] == g for r in rows)]

    def _pdbs(self, ring):
        return [res['pdb'] for res in self.residues if res['ring'] == ring]

    def crosstabs(self, rows, key, ring):
        """Counts, row % (per first-touch ring) and column % (per occupancy level)."""
        rows = [r for r in rows if r[key] is not None]
        counts = pd.DataFrame([[sum(1 for r in rows if r['ring'] == g and r[key] == n)
                                for n in range(len(self._pdbs(ring)) + 1)] for g in self._rings(rows)],
                              index=pd.Index(self._rings(rows), name='first touch'),
                              columns=pd.Index(range(len(self._pdbs(ring)) + 1),
                                               name=f'{self.ring_labels[ring]} occupied'))
        row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        col_pct = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 100
        return counts, row_pct, col_pct

    def residue_tables(self, rows):
        """% occupied per residue (columns) per first-touch ring, and per first-touched residue."""
        rows = [r for r in rows if r['occ'] is not None]
        cols = [residue_tick(res) for res in self.residues]

        def pct(sub):
            return [100 * np.mean([r['occ'].get(res['pdb'], 0) for r in sub]) if sub else np.nan
                    for res in self.residues]

        by_ring = pd.DataFrame([[g, len(sub)] + pct(sub) for g in self._rings(rows)
                                for sub in [[r for r in rows if r['ring'] == g]]],
                               columns=['first touch', 'n'] + cols)
        groups = [(residue_tick(res), [r for r in rows if r['touch_residue'] == res['pdb']]) for res in self.residues]
        groups += [(NO_CONTACT, [r for r in rows if r['touch_residue'] is None])]
        by_res = pd.DataFrame([[name, len(sub)] + pct(sub) for name, sub in groups if sub],
                              columns=['first residue touched', 'n'] + cols)

        # Per residue: is it occupied more often when the first touch was ASN/ASP than GLU?
        asn, glu = self.ring_labels['ASN'], self.ring_labels['GLU']
        fisher = []
        for res in self.residues:
            a = [r['occ'].get(res['pdb'], 0) for r in rows if r['ring'] == asn]
            b = [r['occ'].get(res['pdb'], 0) for r in rows if r['ring'] == glu]
            odds, p = _fisher([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]])
            fisher.append({'residue': residue_tick(res), 'ring': self.ring_labels[res['ring']],
                           f'{asn} touch: occupied': sum(a), f'{asn} touch: n': len(a),
                           f'{asn} touch: % occupied': 100 * np.mean(a) if a else np.nan,
                           f'{glu} touch: occupied': sum(b), f'{glu} touch: n': len(b),
                           f'{glu} touch: % occupied': 100 * np.mean(b) if b else np.nan,
                           'Fisher odds ratio': odds, 'Fisher p': p})
        return by_ring, by_res, pd.DataFrame(fisher)

    def stats(self, events):
        """Fisher, Mann-Whitney and mean +/- SEM of occupied GLUs, for both frames."""
        asn, glu = self.ring_labels['ASN'], self.ring_labels['GLU']
        out = []
        for at, desc in FRAMES.items():
            rows = [e[at] for e in events if e[at]['n_glu'] is not None]
            a = np.array([r['n_glu'] for r in rows if r['ring'] == asn])
            b = np.array([r['n_glu'] for r in rows if r['ring'] == glu])
            nc = np.array([r['n_glu'] for r in rows if r['ring'] == NO_CONTACT])
            #                 free ring           >= 1 occupied
            table = [[int((a == 0).sum()), int((a > 0).sum())],
                     [int((b == 0).sum()), int((b > 0).sum())]]
            odds, p = _fisher(table)
            row = {'occupancy at': desc,
                   f'{asn} touch, 0 GLU occupied': table[0][0], f'{asn} touch, >=1 occupied': table[0][1],
                   f'{glu} touch, 0 GLU occupied': table[1][0], f'{glu} touch, >=1 occupied': table[1][1],
                   'Fisher odds ratio': odds, 'Fisher p': p,
                   'Mann-Whitney p (n GLU occupied)': (mannwhitneyu(a, b, alternative='two-sided').pvalue
                                                       if len(a) and len(b) else np.nan)}
            for name, v in ((asn, a), (glu, b), (NO_CONTACT, nc)):
                row[f'{name}: n'] = len(v)
                row[f'{name}: mean GLU occupied'] = v.mean() if len(v) else np.nan
                row[f'{name}: SEM'] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            out.append(row)
        return pd.DataFrame(out)

    def timecourse_table(self, rows):
        """Mean occupied residues of each ring per frame from end_1, per first-touch ring."""
        rows = [r for r in rows if r['tc'] is not None]
        if not rows:
            return pd.DataFrame()
        offsets = rows[0]['offsets']
        table = {'frames from end_1': offsets}
        for ring in RINGS:
            pdbs = self._pdbs(ring)
            for g in self._rings(rows):
                vals = [[sum(r['tc'][p][i] for p in pdbs) for r in rows if r['ring'] == g
                         and all(r['tc'][p][i] is not None for p in pdbs)] for i in range(len(offsets))]
                table[f'{self.ring_labels[ring]} occupied | first touch {g}'] = \
                    [np.mean(v) if v else np.nan for v in vals]
        return pd.DataFrame(table)

    def per_run_table(self, events, run_info):
        rows = []
        for run in sorted({e['run'] for e in events}, key=run_sort_key):
            d = [e['end_1'] for e in events if e['run'] == run]
            row = {'run': run, **{c.lower(): v.get(run, '') for c, v in run_info.items()}, 'events': len(d)}
            for g in self._rings(d):
                row[f'first touch {g}'] = sum(1 for r in d if r['ring'] == g)
            for key, ring in (('n_glu', 'GLU'), ('n_asn', 'ASN')):
                for g in self._rings(d):
                    v = [r[key] for r in d if r['ring'] == g]
                    row[f'mean {self.ring_labels[ring]} occupied at end_1 ({g})'] = np.mean(v) if v else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def events_table(self, events, run_info):
        rows = []
        for e in sorted(events, key=lambda e: (run_sort_key(e['run'])[0], e['raw']['end_1_frame'])):
            r = e['raw']
            row = {'run': e['run'], **{c.lower(): v.get(e['run'], '') for c, v in run_info.items()},
                   'ion_id': r['ion_id'], 'end_1_frame': r['end_1_frame'], 'end_2_frame': r['end_2_frame'],
                   'first_touch': r['touch_ring_label'], 'touch_residue': r['touch_residue_pdb'],
                   'touch_type': r['touch_residue_type'], 'touch_frame': r['touch_frame'],
                   'touch_delay': r['touch_delay'],
                   'frames_in_cavity': r['end_2_frame'] - r['end_1_frame'],
                   'closest_residue_distance': r.get('min_distance_in_window')}
            for at in FRAMES:
                occ = r[f'occupancy_at_{at}']
                for key, name in (('glu', 'GLU'), ('asn', 'ASN')):
                    row[f'n_{key}_occupied_{at}'] = occ[f'n_{key}_occupied'] if occ else None
                    row[f'n_ions_on_{key}_{at}'] = occ[f'n_ions_on_{key}'] if occ else None
                for res in self.residues:
                    v = occ['residues'].get(res['pdb']) if occ else None
                    row[f'{residue_tick(res)} occupied {at}'] = int(bool(v['ions'])) if v else None
                    if at == 'end_1':
                        row[f'{residue_tick(res)} ions end_1'] = ', '.join(map(str, v['ions'])) if v else None
            rows.append(row)
        return pd.DataFrame(rows)

    def write_group(self, group, runs, output_dir, run_info):
        output_dir = Path(output_dir)
        events = [e for e in self.events if e['run'] in runs]
        print(f"\n--- {group}: {len(runs)} runs, {len(events)} events -> {output_dir}")
        if not events:
            print("    no permeation events in these runs, folder skipped")
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        write_all_plots([e['end_1'] for e in events], self.residues, self.ring_labels, self.channel_type,
                        self.threshold, self.window, output_dir, self.tag,
                        touch_rows=[e['touch'] for e in events], label=group)

        stats = self.stats(events)
        with pd.ExcelWriter(output_dir / f"summary_{self.tag}.xlsx") as xl:
            for at in FRAMES:
                rows = [e[at] for e in events]
                for key, ring in (('n_glu', 'GLU'), ('n_asn', 'ASN')):
                    counts, row_pct, col_pct = self.crosstabs(rows, key, ring)
                    counts.to_excel(xl, sheet_name=f"{at}_{ring.lower()}_counts")
                    row_pct.round(1).to_excel(xl, sheet_name=f"{at}_{ring.lower()}_pct_touch")
                    col_pct.round(1).to_excel(xl, sheet_name=f"{at}_{ring.lower()}_pct_occ")
                    if key == 'n_glu':
                        print(f"\nGLUs occupied at {FRAMES[at]} ({group}, counts):")
                        print(counts.to_string())
                by_ring, by_res, fisher = self.residue_tables(rows)
                by_ring.round(1).to_excel(xl, sheet_name=f"{at}_residue_pct", index=False)
                by_res.round(1).to_excel(xl, sheet_name=f"{at}_heatmap", index=False)
                fisher.to_excel(xl, sheet_name=f"{at}_residue_fisher", index=False)
                if at == 'end_1':
                    print(f"\n% of events each residue is occupied at end_1 ({group}):")
                    print(by_ring.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
            stats.to_excel(xl, sheet_name="stats", index=False)
            self.timecourse_table([e['end_1'] for e in events]).to_excel(xl, sheet_name="timecourse", index=False)
        for _, s in stats.iterrows():
            p = s['Fisher p']
            print(f"  {s['occupancy at']}: Fisher p = {p:.3g}" if pd.notna(p)
                  else f"  {s['occupancy at']}: Fisher test not possible (empty row/column)")

        self.per_run_table(events, run_info).to_excel(output_dir / f"per_run_{self.tag}.xlsx", index=False)
        self.events_table(events, run_info).to_excel(output_dir / f"events_{self.tag}.xlsx", index=False)
        print(f"✓ Tables saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GLU / ASN occupancy at SF exit vs first touch over runs (ALL, HIGH, TYPICAL, LOW; plus D and E for G12_ML)")
    parser.add_argument("base_directory", help="Base directory containing RUN subdirectories")
    parser.add_argument("--threshold", type=float, default=3.5, help="Threshold used in the analysis (default: 3.5)")
    parser.add_argument("--window", type=int, default=10, help="First-touch window used in the analysis (default: 10)")
    parser.add_argument("--no_window", action="store_true",
                        help="Aggregate the runs analysed without a window (window=None, <t>_no_w folders)")
    parser.add_argument("--output_dir", default=None,
                        help="Default: base_directory/aggregated_glu_occupancy_exit/t*_w*")
    parser.add_argument("--categories", default=None,
                        help="run_categories CSV from analyze_permeation_summary.py (default: "
                             "base_directory/permeation_summary/run_categories_full_permeations_total.csv)")
    parser.add_argument("--no_categories", action="store_true", help="Only aggregate all runs")
    parser.add_argument("--ligand_categories", default=None,
                        help="G12_ML only: ligand_categories.csv (default: "
                             "base_directory/ligand_inter_class/ligand_categories.csv)")
    parser.add_argument("--channel_type", default=None, help="Label for plot titles (default: folder name)")
    args = parser.parse_args()

    window = None if args.no_window else args.window
    agg = GluOccupancyAggregator(args.base_directory, args.threshold, window, args.channel_type)
    if not agg.load_all_runs():
        print("No data found. Exiting.")
        return
    output_dir = Path(args.output_dir) if args.output_dir else \
        agg.base_dir / "aggregated_glu_occupancy_exit" / agg.tag
    all_runs = set(agg.runs_loaded)

    categories, ligand = {}, {}
    if not args.no_categories:
        cat_file = Path(args.categories) if args.categories else \
            agg.base_dir / "permeation_summary" / "run_categories_full_permeations_total.csv"
        if cat_file.exists():
            categories = load_categories(cat_file)
            report_coverage('permeation', categories, all_runs)
        else:
            print(f"\n⚠ Categories file not found: {cat_file}\n  No HIGH/TYPICAL/LOW folders.")
        if agg.channel_type.upper() in LIGAND_CHANNELS:
            lig_file = Path(args.ligand_categories) if args.ligand_categories else \
                agg.base_dir / "ligand_inter_class" / "ligand_categories.csv"
            if lig_file.exists():
                ligand, _ = load_ligand_categories(lig_file)
                report_coverage('ligand', ligand, all_runs)
            else:
                print(f"\n⚠ Ligand categories file not found: {lig_file}\n  No D/E folders.")

    run_info = {}
    if categories:
        run_info['Category'] = categories
    if ligand:
        run_info['Ligand'] = ligand

    agg.write_group('ALL', all_runs, output_dir / 'ALL', run_info)
    for mapping, cats, label in ((categories, CATEGORIES, '{}'), (ligand, LIGAND_CATEGORIES, 'ligand {}')):
        for cat in cats:
            runs = {r for r, c in mapping.items() if c == cat and r in all_runs}
            if runs:
                agg.write_group(label.format(cat), runs, output_dir / cat, run_info)

    print(f"\n✓ Aggregation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


'''
python3 aggregate_glu_occupancy_at_exit.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--threshold 3.5 --window 10

python3 aggregate_glu_occupancy_at_exit.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--threshold 3.5 --window 10

python3 aggregate_glu_occupancy_at_exit.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--threshold 3.5 --no_window
'''
