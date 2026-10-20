"""
aggregate_duration_results.py
=============================
Aggregates run_last_duration_analysis (analysis/longest_duration_analysis.py)
over all runs of one channel: which residue / subunit was the LAST and the
LONGEST contact before the ion left the GLU/ASN cavity (end_2).

The runs are aggregated four times, each into its own folder inside the
output directory:

    ALL/       every run
    HIGH/      runs in the HIGH category      } categories read from the
    TYPICAL/   runs in the TYPICAL category   } run_categories_*.csv written
    LOW/       runs in the LOW category       } by analyze_permeation_summary.py

so runs with many permeations can be compared with runs with few.

Two ways of combining runs are reported in every folder:

  POOLED   every event of every run counted once, as if all runs were one
           long run. Runs with many events weigh more.
  PER RUN  the fraction is computed in each run first, then averaged over
           runs (mean +/- SEM, one dot per run). Every run weighs the same,
           so a few busy runs cannot dominate.

Events where the ion never stayed within the threshold for min_duration
frames are counted as "No contact" everywhere (bars, tables, fractions).

Outputs (per folder, t = threshold, d = min_duration):
    last_residue_frequency_t*_d*.png          pooled bar plots
    longest_residue_frequency_t*_d*.png
    last_subunit_frequency_t*_d*.png
    longest_subunit_frequency_t*_d*.png
    last_residue_per_run_t*_d*.png            per-run mean +/- SEM + dots
    longest_residue_per_run_t*_d*.png
    last_subunit_per_run_t*_d*.png
    longest_subunit_per_run_t*_d*.png
    contact_durations_residue_t*_d*.png       pooled duration histograms
    residue_summary_t*_d*.xlsx                pooled counts, %, durations
    subunit_summary_t*_d*.xlsx
    per_run_t*_d*.xlsx                        one row per run (counts + fractions)
    events_t*_d*.xlsx                         one row per event

USAGE
-----
    python3 aggregate_duration_results.py /path/to/G12 --threshold 2.5 --min_duration 3

The categories CSV is found automatically at
<base>/permeation_summary/run_categories_full_permeations_total.csv;
pass --categories to use another file, or --no_categories for ALL only.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from analysis.converter import subunit_label


NO_CONTACT = 'No contact'
CATEGORIES = ['HIGH', 'TYPICAL', 'LOW']
ONE_LETTER = {'GLU': 'E', 'GLN': 'Q', 'ASP': 'D', 'ASN': 'N'}

# Same palette as the per-run plots (closest_residue_analysis.py)
COLOR_MAP = {
    'GLU': '#E8746B',   # muted coral
    'GLN': '#E8A85B',   # muted amber (E141Q)
    'ASN': '#5AA9C9',   # muted blue
    'ASP': '#6FBF73',   # muted green
    'Unknown': '#B0B0B0',
}
TYPE_ORDER = ['GLU', 'GLN', 'ASN', 'ASP', 'Unknown']
SUBUNIT_COLOR = '#8E9AAF'
NO_CONTACT_COLOR = '#D9D9D9'
MODES = ['residue', 'subunit']
SIDES = ['last', 'longest']


def run_sort_key(name):
    """RUN2 before RUN10."""
    digits = ''.join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


def guess_type(label):
    """WT numbering guess, only for JSONs without a stored residue_type."""
    if '152' in label or '141' in label:
        return 'GLU'
    if '184' in label:
        return 'ASN'
    if '173' in label:
        return 'ASP'
    return 'Unknown'


class DurationResultsAggregator:
    """
    Loads the per-run duration JSONs once, then writes the aggregated plots
    and tables for any subset of runs (ALL / HIGH / TYPICAL / LOW).
    """

    def __init__(self, base_directory, threshold, min_duration, channel_type=None):
        self.base_dir = Path(base_directory)
        self.threshold = threshold
        self.min_duration = min_duration
        self.channel_type = channel_type or self.base_dir.name

        # Convert threshold to string (handle 3.0 -> "3")
        if isinstance(self.threshold, float) and self.threshold.is_integer():
            self.threshold_str = str(int(self.threshold))
        else:
            self.threshold_str = str(self.threshold)
        self.tag = f"t{self.threshold_str}_d{self.min_duration}"

        # events[mode][side] = list of dicts, one per permeation event
        self.events = {m: {s: [] for s in SIDES} for m in MODES}
        self.runs_loaded = {m: [] for m in MODES}
        # {residue_pdb: residue name}, from the JSONs (read from the topology)
        self.res_types = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _find_file(self, run_dir, mode, side):
        for variant in (f"{self.threshold_str}_{self.min_duration}",
                        f"{self.threshold}_{self.min_duration}"):
            f = run_dir / f"duration_{mode}" / variant / f"{side}_duration_{mode}.json"
            if f.exists():
                return f
        return None

    def _event_row(self, run, mode, ev):
        """One normalised row per event: which residue/subunit won, and how long."""
        residues = ev.get('residues') or []
        if ev.get('identifier') and residues:
            pdb = residues[0]['residue_pdb']
            if mode == 'subunit':
                # Chain name (G2.a, G1, ...) from the residues; old JSONs store A-D
                label, rtype = subunit_label(pdb), None
            else:
                label = pdb
                rtype = residues[0].get('residue_type') or guess_type(pdb)
                self.res_types[pdb] = rtype
            duration = ev.get('duration')
            gap = (ev['end_2_frame'] - ev['end_frame']
                   if ev.get('end_frame') is not None and ev.get('end_2_frame') is not None else None)
        else:
            # Ion never stayed within the threshold for min_duration frames
            label, rtype, duration, gap = NO_CONTACT, None, None, None
        return {
            'run': run,
            'ion_id': ev.get('ion_id'),
            'end_2_frame': ev.get('end_2_frame'),
            'label': label,
            'type': rtype,
            'duration': duration,
            'start_frame': ev.get('start_frame'),
            'end_frame': ev.get('end_frame'),
            'gap_to_end_2': gap,
        }

    def load_all_runs(self):
        """Load the duration results of every RUN* directory."""
        print(f"\nSearching for runs in: {self.base_dir}")
        print(f"Looking for threshold={self.threshold_str}, min_duration={self.min_duration}")

        run_dirs = sorted((d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('RUN')),
                          key=lambda d: run_sort_key(d.name))
        if not run_dirs:
            print(f"No RUN directories found in {self.base_dir}")
            return False
        print(f"Found {len(run_dirs)} RUN directories")

        for run_dir in run_dirs:
            for mode in MODES:
                files = {side: self._find_file(run_dir, mode, side) for side in SIDES}
                if not all(files.values()):
                    continue
                try:
                    loaded = {}
                    for side, f in files.items():
                        with open(f) as fh:
                            loaded[side] = json.load(fh)['results']
                except Exception as e:
                    print(f"  ✗ {run_dir.name}: error loading {mode}-level: {e}")
                    continue
                for side in SIDES:
                    self.events[mode][side] += [self._event_row(run_dir.name, mode, ev) for ev in loaded[side]]
                self.runs_loaded[mode].append(run_dir.name)
                print(f"  ✓ {run_dir.name}: {mode}-level ({len(loaded['last'])} events)")

        for mode in MODES:
            print(f"  {mode}-level: {len(self.runs_loaded[mode])} runs")
        return any(self.runs_loaded.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rtype(self, label):
        return self.res_types.get(label) or guess_type(label)

    def _tick(self, mode, label):
        if mode == 'residue' and label != NO_CONTACT:
            return f"{ONE_LETTER.get(self._rtype(label), '')}{label}"
        return label

    def _color(self, mode, label):
        if label == NO_CONTACT:
            return NO_CONTACT_COLOR
        if mode == 'subunit':
            return SUBUNIT_COLOR
        return COLOR_MAP.get(self._rtype(label), COLOR_MAP['Unknown'])

    def _legend(self, ax, mode, labels):
        handles = []
        if mode == 'residue':
            present = {self._rtype(l) for l in labels if l != NO_CONTACT}
            handles = [Patch(facecolor=COLOR_MAP[t], edgecolor='black', label=t)
                       for t in TYPE_ORDER if t in present]
        handles.append(Patch(facecolor=NO_CONTACT_COLOR, edgecolor='black', hatch='//', label=NO_CONTACT))
        # Outside the axes: the "No contact" bar is last and would sit under it
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=13, frameon=True, framealpha=0.9)

    @staticmethod
    def _style(ax):
        ax.grid(True, alpha=0.25, axis='y', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.1)

    def _label_order(self, df):
        """Labels by pooled count (descending), "No contact" always last."""
        counts = df['label'].value_counts()
        labels = [l for l in counts.index if l != NO_CONTACT]
        return labels + [NO_CONTACT]

    def _per_run_fractions(self, df, labels):
        """DataFrame run x label of fractions (runs with 0 events left out)."""
        counts = df.groupby(['run', 'label']).size().unstack(fill_value=0)
        counts = counts.reindex(columns=labels, fill_value=0)
        totals = counts.sum(axis=1)
        counts = counts[totals > 0]
        return counts.div(counts.sum(axis=1), axis=0), counts

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def pooled_bar_plot(self, df, mode, side, title, output_file):
        """POOLED: every event counted once. Count, % of all events, mean duration."""
        labels = self._label_order(df)
        counts = [int((df['label'] == l).sum()) for l in labels]
        total = sum(counts)
        mean_dur = [df.loc[df['label'] == l, 'duration'].mean() if l != NO_CONTACT else None for l in labels]

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels) + 3), 6.2))
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=[self._color(mode, l) for l in labels], alpha=0.85,
                      edgecolor='black', linewidth=1.2, width=0.65, zorder=3)
        bars[-1].set_hatch('//')

        top = max(max(counts), 1)
        ax.set_ylim(0, top * 1.35)
        for bar, c, m in zip(bars, counts, mean_dur):
            text = f"{c} ({100.0 * c / total:.1f}%)" if total else f"{c}"
            if m is not None and not np.isnan(m):
                text += f"\n{m:.1f} fr"
            ax.text(bar.get_x() + bar.get_width() / 2., c + top * 0.03, text,
                    ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([self._tick(mode, l) for l in labels], fontsize=16, fontweight='bold',
                           rotation=40, ha='right')
        ax.set_ylabel('Number of Events', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_xlabel('Subunit' if mode == 'subunit' else 'Residue', fontsize=18, fontweight='bold', labelpad=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=14)
        n_runs = df['run'].nunique()
        ax.set_title(f"{title}\n(pooled: n = {total} events from {n_runs} runs; "
                     f"< {self.threshold} Å for ≥ {self.min_duration} frames)",
                     fontsize=17, fontweight='bold', pad=20)
        self._style(ax)
        self._legend(ax, mode, labels)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def per_run_plot(self, df, mode, side, title, output_file):
        """PER RUN: fraction in each run, then mean +/- SEM over runs, one dot per run."""
        labels = self._label_order(df)
        frac, _ = self._per_run_fractions(df, labels)
        if frac.empty:
            return
        pct = 100.0 * frac
        means = pct.mean(axis=0)
        sems = pct.std(axis=0, ddof=1) / np.sqrt(len(pct)) if len(pct) > 1 else pct.mean(axis=0) * 0

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels) + 3), 6.2))
        x = np.arange(len(labels))
        bars = ax.bar(x, means.values, yerr=sems.values, capsize=6,
                      color=[self._color(mode, l) for l in labels], alpha=0.85,
                      edgecolor='black', linewidth=1.2, width=0.65, zorder=3,
                      error_kw={'elinewidth': 1.4, 'zorder': 4})
        bars[-1].set_hatch('//')

        # One dot per run, spread a little sideways so they do not overlap
        rng = np.random.default_rng(0)
        for i, l in enumerate(labels):
            jitter = rng.uniform(-0.18, 0.18, len(pct))
            ax.scatter(i + jitter, pct[l].values, s=26, color='black', alpha=0.65, zorder=5, linewidths=0)

        top = max(float(pct.values.max()), 1.0)
        ax.set_ylim(0, top * 1.2)
        for bar, m, e in zip(bars, means.values, sems.values):
            ax.text(bar.get_x() + bar.get_width() / 2., top * 1.12, f"{m:.1f}±{e:.1f}%",
                    ha='center', va='top', fontsize=12, fontweight='bold', zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels([self._tick(mode, l) for l in labels], fontsize=16, fontweight='bold',
                           rotation=40, ha='right')
        ax.set_ylabel('% of events in a run', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_xlabel('Subunit' if mode == 'subunit' else 'Residue', fontsize=18, fontweight='bold', labelpad=10)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f"{title}\n(per run: mean ± SEM over {len(pct)} runs, dots = runs; "
                     f"< {self.threshold} Å for ≥ {self.min_duration} frames)",
                     fontsize=17, fontweight='bold', pad=20)
        self._style(ax)
        self._legend(ax, mode, labels)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    def duration_plot(self, last_df, longest_df, group, output_file):
        """Pooled histograms: last / longest contact duration, gap from last contact to end_2."""
        last = last_df[last_df['label'] != NO_CONTACT]
        longest = longest_df[longest_df['label'] != NO_CONTACT]
        panels = [
            ('Last contact duration', last['duration'].dropna().values),
            ('Longest contact duration', longest['duration'].dropna().values),
            ('Last contact end → end_2', last['gap_to_end_2'].dropna().values),
        ]

        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (name, values) in zip(axes, panels):
            ax.set_title(name, fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('Frames', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of events', fontsize=14, fontweight='bold')
            self._style(ax)
            if len(values) == 0:
                ax.text(0.5, 0.5, 'no contacts', transform=ax.transAxes, ha='center', va='center', fontsize=14)
                continue
            values = np.asarray(values, dtype=float)
            bins = np.histogram_bin_edges(values, bins=min(30, max(5, int(values.max() - values.min()) + 1)))
            ax.hist(values, bins=bins, color=SUBUNIT_COLOR, edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3)
            ax.axvline(np.median(values), color='black', linestyle='--', linewidth=1.2,
                       label=f'median = {np.median(values):g}')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
            ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')

        n_no = int((last_df['label'] == NO_CONTACT).sum())
        fig.suptitle(f"{self.channel_type} {group}: contacts before leaving the cavity "
                     f"(< {self.threshold} Å, ≥ {self.min_duration} frames; pooled over "
                     f"{last_df['run'].nunique()} runs; {n_no}/{len(last_df)} events with no contact)",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {output_file}")

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def summary_table(self, mode, frames):
        """Pooled counts, % of all events and contact durations, last and longest side by side."""
        labels = []
        for side in SIDES:
            labels += [l for l in self._label_order(frames[side]) if l not in labels and l != NO_CONTACT]
        labels = sorted(labels) + [NO_CONTACT]

        rows = []
        for l in labels:
            row = {'Subunit' if mode == 'subunit' else 'Residue_PDB': l}
            if mode == 'residue':
                row['Type'] = '' if l == NO_CONTACT else self._rtype(l)
            for side in SIDES:
                df = frames[side]
                sel = df[df['label'] == l]
                cap = side.capitalize()
                row[f'{cap}_Count'] = len(sel)
                row[f'{cap}_Pct'] = 100.0 * len(sel) / len(df) if len(df) else np.nan
                row[f'{cap}_Mean_Duration_fr'] = sel['duration'].mean() if l != NO_CONTACT else np.nan
                row[f'{cap}_Median_Duration_fr'] = sel['duration'].median() if l != NO_CONTACT else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def per_run_table(self, mode, side, df, categories):
        """One row per run: n events, count and fraction for every label."""
        # Fixed column order (sorted, "No contact" last) so sheets line up across groups
        labels = sorted(l for l in self._label_order(df) if l != NO_CONTACT) + [NO_CONTACT]
        frac, counts = self._per_run_fractions(df, labels)
        out = pd.DataFrame(index=counts.index)
        out.insert(0, 'Category', [categories.get(r, '') for r in out.index])
        out['N_Events'] = counts.sum(axis=1)
        for l in labels:
            out[f'{l}_Count'] = counts[l]
        for l in labels:
            out[f'{l}_Frac'] = frac[l]
        out = out.loc[sorted(out.index, key=run_sort_key)]
        out.index.name = 'Run'
        return out.reset_index()

    # ------------------------------------------------------------------
    # One folder
    # ------------------------------------------------------------------

    def write_group(self, group, runs, output_dir, categories):
        """All plots and tables for one set of runs, in output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {group}: {len(runs)} runs -> {output_dir}")
        print(f"    {', '.join(sorted(runs, key=run_sort_key))}")

        with pd.ExcelWriter(output_dir / f"per_run_{self.tag}.xlsx") as per_run_xl, \
             pd.ExcelWriter(output_dir / f"events_{self.tag}.xlsx") as events_xl:
            for mode in MODES:
                frames = {side: pd.DataFrame([e for e in self.events[mode][side] if e['run'] in runs])
                          for side in SIDES}
                if frames['last'].empty:
                    continue
                unit = 'Subunit' if mode == 'subunit' else 'Residue'

                for side in SIDES:
                    what = 'LAST' if side == 'last' else 'LONGEST'
                    title = f"{self.channel_type} {group}: {what} {unit} Before Leaving Cavity"
                    self.pooled_bar_plot(frames[side], mode, side, title,
                                         output_dir / f"{side}_{mode}_frequency_{self.tag}.png")
                    self.per_run_plot(frames[side], mode, side, title,
                                      output_dir / f"{side}_{mode}_per_run_{self.tag}.png")
                    self.per_run_table(mode, side, frames[side], categories).to_excel(
                        per_run_xl, sheet_name=f"{side}_{mode}", index=False)

                summary = self.summary_table(mode, frames)
                summary.to_excel(output_dir / f"{mode}_summary_{self.tag}.xlsx", index=False, float_format='%.2f')
                print(f"\n{unit.upper()}-LEVEL SUMMARY ({group}, pooled)")
                print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

                # One row per event, last and longest side by side
                ev = frames['last'].rename(columns={c: f"last_{c}" for c in
                                                    ['label', 'type', 'duration', 'start_frame', 'end_frame']})
                lg = frames['longest'][['label', 'duration', 'start_frame', 'end_frame']].add_prefix('longest_')
                ev = pd.concat([ev.reset_index(drop=True), lg.reset_index(drop=True)], axis=1)
                ev.insert(1, 'category', [categories.get(r, '') for r in ev['run']])
                ev['_run_order'] = ev['run'].map(lambda r: run_sort_key(r)[0])
                ev = ev.sort_values(['_run_order', 'end_2_frame']).drop(columns='_run_order')
                ev.to_excel(events_xl, sheet_name=mode, index=False)

                if mode == 'residue':
                    self.duration_plot(frames['last'], frames['longest'], group,
                                       output_dir / f"contact_durations_residue_{self.tag}.png")

        print(f"✓ Tables saved in {output_dir}")


def load_categories(path):
    """{run: category} from analyze_permeation_summary.py's run_categories CSV."""
    df = pd.read_csv(path)
    return {str(r).strip(): str(c).strip().upper() for r, c in zip(df['Run'], df['Category'])
            if pd.notna(c)}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate duration analysis results from multiple runs (ALL, HIGH, TYPICAL, LOW)"
    )
    parser.add_argument("base_directory", type=str,
                        help="Base directory containing RUN subdirectories")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Threshold value used in the analysis (e.g., 3.0)")
    parser.add_argument("--min_duration", type=int, required=True,
                        help="Minimum duration value used in the analysis (e.g., 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: base_directory/aggregated_duration_t*_d*). "
                             "ALL/, HIGH/, TYPICAL/ and LOW/ are created inside it.")
    parser.add_argument("--categories", type=str, default=None,
                        help="run_categories CSV from analyze_permeation_summary.py (default: "
                             "base_directory/permeation_summary/run_categories_full_permeations_total.csv)")
    parser.add_argument("--no_categories", action="store_true",
                        help="Only aggregate all runs (no HIGH/TYPICAL/LOW folders)")
    parser.add_argument("--channel_type", type=str, default=None,
                        help="Label for plot titles (default: base_directory folder name)")
    args = parser.parse_args()

    aggregator = DurationResultsAggregator(args.base_directory, args.threshold, args.min_duration,
                                           args.channel_type)
    if not aggregator.load_all_runs():
        print("No data found. Exiting.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else \
        aggregator.base_dir / f"aggregated_duration_{aggregator.tag}"

    all_runs = set(aggregator.runs_loaded['residue']) | set(aggregator.runs_loaded['subunit'])

    categories = {}
    if not args.no_categories:
        cat_file = Path(args.categories) if args.categories else \
            aggregator.base_dir / "permeation_summary" / "run_categories_full_permeations_total.csv"
        if cat_file.exists():
            categories = load_categories(cat_file)
            print(f"\nCategories from: {cat_file}")
            missing = sorted(set(categories) - all_runs, key=run_sort_key)
            unlisted = sorted(all_runs - set(categories), key=run_sort_key)
            if missing:
                print(f"  ⚠ In the categories file but no duration results: {', '.join(missing)}")
            if unlisted:
                print(f"  ⚠ Have duration results but no category (only in ALL): {', '.join(unlisted)}")
        else:
            print(f"\n⚠ Categories file not found: {cat_file}\n  Only ALL will be written "
                  f"(run analyze_permeation_summary.py first, or pass --categories).")

    print("\nGenerating outputs...")
    aggregator.write_group('ALL', all_runs, output_dir / 'ALL', categories)
    for cat in CATEGORIES:
        runs = {r for r, c in categories.items() if c == cat and r in all_runs}
        if runs:
            aggregator.write_group(cat, runs, output_dir / cat, categories)
        elif categories:
            print(f"\n--- {cat}: no runs with duration results, folder skipped")

    print(f"\n✓ Aggregation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


# USAGE EXAMPLE:
# python3 aggregate_duration_results.py /path/to/G12 --threshold 2.5 --min_duration 3
#
# Custom categories file / output directory:
# python3 aggregate_duration_results.py /path/to/G12 --threshold 2.5 --min_duration 3 \
#     --categories /path/to/G12/permeation_summary/run_categories_full_permeations_total.csv \
#     --output_dir /path/to/output
