"""
aggregate_glu_occupancy_at_exit.py
==================================
Aggregates analysis/glu_occupancy_at_exit.py over all runs of one channel:
for every permeation event, the first residue touched after the SF (ASN/ASP
ring, GLU ring, or No contact) against how many GLUs were occupied by
another K+ when the ion left the SF (end_1), and at the first-touch frame.

Question: is the first touch ASN/ASP only when the GLU ring is already
occupied (and GLU when it is free)?

The runs are aggregated into ALL/, HIGH/, TYPICAL/, LOW/ (and D/, E/ for
G12_ML), with the same category files as aggregate_duration_results.py.
Events of all runs in a folder are pooled.

Outputs (per folder; t = threshold, w = window):
    glu_occupancy_end_1_t*_w*.png     left: occupied GLUs at end_1 per first-touch ring
                                      right: first-touch ring per number of occupied GLUs
    glu_occupancy_touch_t*_w*.png     the same with occupancy at the first-touch frame
    summary_t*_w*.xlsx                cross-tables (counts, row %, column %) for both
                                      frames, and a 'stats' sheet:
                                      - Fisher exact test, first touch GLU vs ASN/ASP ring
                                        x GLU ring free (0 occupied) vs >= 1 occupied
                                      - Mann-Whitney U on the number of occupied GLUs,
                                        ASN/ASP first touch vs GLU first touch
                                      - mean +/- SEM occupied GLUs per first-touch ring
    per_run_t*_w*.xlsx                one row per run: counts per ring, mean occupied GLUs
    events_t*_w*.xlsx                 one row per event

Default output folder: <base>/aggregated_glu_occupancy_exit/t*_w*/

USAGE
-----
    python3 aggregate_glu_occupancy_at_exit.py /path/to/G12 --threshold 3.5 --window 10
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from aggregate_duration_results import (CATEGORIES, LIGAND_CATEGORIES, LIGAND_CHANNELS, run_sort_key,
                                        load_categories, load_ligand_categories, report_coverage)
from analysis.glu_occupancy_at_exit import NO_CONTACT, plot_occupancy_vs_touch

FRAMES = {'end_1': 'end_1 (leaving the SF)', 'touch': 'the first-touch frame'}


class GluOccupancyAggregator:
    """Loads the per-run glu_occupancy_exit_complete.json files, then writes each folder."""

    def __init__(self, base_directory, threshold, window, channel_type=None):
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.window = window
        self.channel_type = channel_type or self.base_dir.name
        self.threshold_str = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        self.tag = f"t{self.threshold_str}_w{window}"
        self.rows = []
        self.runs_loaded = []
        self.ring_labels = None
        self.n_glu = 0

    def _find_file(self, run_dir):
        for t in (self.threshold_str, str(self.threshold)):
            f = run_dir / "glu_occupancy_exit" / f"{t}_w{self.window}" / "glu_occupancy_exit_complete.json"
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
            self.ring_labels = self.ring_labels or data['ring_labels']
            self.n_glu = max(self.n_glu, len(data['glu_pdb']))
            for r in data['results']:
                row = {
                    'run': run_dir.name,
                    'ion_id': r['ion_id'],
                    'end_1_frame': r['end_1_frame'],
                    'end_2_frame': r['end_2_frame'],
                    'ring': r['touch_ring_label'],
                    'ring_key': r['touch_ring'],
                    'touch_residue': r['touch_residue_pdb'],
                    'touch_type': r['touch_residue_type'],
                    'touch_frame': r['touch_frame'],
                    'touch_delay': r['touch_delay'],
                }
                for at in FRAMES:
                    occ = r[f'occupancy_at_{at}']
                    row[f'n_occ_{at}'] = occ['n_glu_occupied'] if occ else None
                    row[f'n_ions_on_glu_{at}'] = occ['n_ions_on_glu'] if occ else None
                    row[f'occupied_glu_{at}'] = ', '.join(occ['occupied_glu']) if occ else None
                self.rows.append(row)
            self.runs_loaded.append(run_dir.name)
            print(f"  ✓ {run_dir.name}: {len(data['results'])} events")
        print(f"Loaded {len(self.runs_loaded)} runs, {len(self.rows)} events")
        return bool(self.runs_loaded)

    # ------------------------------------------------------------------

    def _rings(self, df):
        order = [self.ring_labels['ASN'], self.ring_labels['GLU'], NO_CONTACT]
        return [g for g in order if g in set(df['ring'])]

    def crosstabs(self, df, at):
        """Counts, row % (per first-touch ring) and column % (per occupancy level)."""
        col = f'n_occ_{at}'
        d = df[df[col].notna()]
        counts = pd.crosstab(d['ring'], d[col].astype(int)).reindex(
            index=self._rings(d), columns=range(self.n_glu + 1), fill_value=0)
        counts.index.name = 'first touch'
        counts.columns.name = 'GLUs occupied'
        row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        col_pct = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 100
        return counts, row_pct, col_pct

    def stats(self, df):
        """Fisher, Mann-Whitney and mean +/- SEM of occupied GLUs, for both frames."""
        asn, glu = self.ring_labels['ASN'], self.ring_labels['GLU']
        out = []
        for at, desc in FRAMES.items():
            col = f'n_occ_{at}'
            d = df[df[col].notna() & df['ring'].isin([asn, glu])]
            a, b = d[d['ring'] == asn][col], d[d['ring'] == glu][col]
            #                 free ring    >= 1 occupied
            table = [[int((a == 0).sum()), int((a > 0).sum())],
                     [int((b == 0).sum()), int((b > 0).sum())]]
            row = {'occupancy at': desc,
                   f'{asn} touch, 0 GLU occupied': table[0][0], f'{asn} touch, >=1 occupied': table[0][1],
                   f'{glu} touch, 0 GLU occupied': table[1][0], f'{glu} touch, >=1 occupied': table[1][1]}
            if all(sum(r) > 0 for r in table) and all(table[0][j] + table[1][j] > 0 for j in range(2)):
                odds, p = fisher_exact(table)
                row.update({'Fisher odds ratio': odds, 'Fisher p': p})
            if len(a) and len(b):
                row['Mann-Whitney p (n occupied)'] = mannwhitneyu(a, b, alternative='two-sided').pvalue
            for name, v in ((asn, a), (glu, b), (NO_CONTACT, df[df['ring'] == NO_CONTACT][col].dropna())):
                row[f'{name}: n'] = len(v)
                row[f'{name}: mean occupied'] = v.mean() if len(v) else np.nan
                row[f'{name}: SEM'] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            out.append(row)
        return pd.DataFrame(out)

    def per_run_table(self, df, run_info):
        rows = []
        for run, d in df.groupby('run'):
            row = {'run': run, **{c.lower(): v.get(run, '') for c, v in run_info.items()},
                   'events': len(d)}
            for g in self._rings(df):
                row[f'first touch {g}'] = int((d['ring'] == g).sum())
            for g in self._rings(df):
                row[f'mean occupied at end_1 ({g})'] = d[d['ring'] == g]['n_occ_end_1'].mean()
            row['mean occupied at end_1 (all)'] = d['n_occ_end_1'].mean()
            rows.append(row)
        return pd.DataFrame(rows).sort_values('run', key=lambda s: s.map(lambda r: run_sort_key(r)[0]))

    def write_group(self, group, runs, output_dir, run_info):
        output_dir = Path(output_dir)
        df = pd.DataFrame([r for r in self.rows if r['run'] in runs])
        print(f"\n--- {group}: {len(runs)} runs, {len(df)} events -> {output_dir}")
        if df.empty:
            print("    no permeation events in these runs, folder skipped")
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        for at, desc in FRAMES.items():
            col = f'n_occ_{at}'
            rows = [{'ring': r['ring'], 'ring_key': r['ring_key'], 'n_occ': int(r[col])}
                    for _, r in df.iterrows() if pd.notna(r[col])]
            plot_occupancy_vs_touch(rows, self.n_glu, self.ring_labels,
                                    f"{self.channel_type} {group}: GLUs occupied at {desc} vs first touch "
                                    f"(< {self.threshold} Å, {self.window}-frame window)",
                                    output_dir / f"glu_occupancy_{at}_{self.tag}.png")

        stats = self.stats(df)
        with pd.ExcelWriter(output_dir / f"summary_{self.tag}.xlsx") as xl:
            for at in FRAMES:
                counts, row_pct, col_pct = self.crosstabs(df, at)
                counts.to_excel(xl, sheet_name=f"{at}_counts")
                row_pct.round(1).to_excel(xl, sheet_name=f"{at}_pct_per_touch")
                col_pct.round(1).to_excel(xl, sheet_name=f"{at}_pct_per_occupancy")
                print(f"\nGLUs occupied at {FRAMES[at]} ({group}, counts):")
                print(counts.to_string())
            stats.to_excel(xl, sheet_name="stats", index=False)
        for _, s in stats.iterrows():
            p = s.get('Fisher p', np.nan)
            print(f"  {s['occupancy at']}: Fisher p = {p:.3g}" if pd.notna(p)
                  else f"  {s['occupancy at']}: Fisher test not possible (empty row/column)")

        self.per_run_table(df, run_info).to_excel(output_dir / f"per_run_{self.tag}.xlsx", index=False)
        ev = df.drop(columns='ring_key').rename(columns={'ring': 'first_touch'})
        for i, (col, values) in enumerate(run_info.items()):
            ev.insert(1 + i, col.lower(), [values.get(r, '') for r in ev['run']])
        ev['_run_order'] = ev['run'].map(lambda r: run_sort_key(r)[0])
        ev.sort_values(['_run_order', 'end_1_frame']).drop(columns='_run_order').to_excel(
            output_dir / f"events_{self.tag}.xlsx", index=False)
        print(f"✓ Tables saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GLU occupancy at SF exit vs first touch over runs (ALL, HIGH, TYPICAL, LOW; plus D and E for G12_ML)")
    parser.add_argument("base_directory", help="Base directory containing RUN subdirectories")
    parser.add_argument("--threshold", type=float, default=3.5, help="Threshold used in the analysis (default: 3.5)")
    parser.add_argument("--window", type=int, default=10, help="First-touch window used in the analysis (default: 10)")
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

    agg = GluOccupancyAggregator(args.base_directory, args.threshold, args.window, args.channel_type)
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
'''
