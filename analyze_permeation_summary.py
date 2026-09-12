#!/usr/bin/env python3
"""
Permeation Event Summary Analyzer

Analyzes permeation events across multiple simulation runs.
Reports complete permeations (permeation_table.json) and a last-frame
snapshot of where ions were sitting in each biological region
(ion_region_tracking.json).

Usage:
    python analyze_permeation_summary.py /path/to/results/G12_GAT
    python analyze_permeation_summary.py /path/to/results/G12_GAT --runs 1 2 3 4 5
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────

def find_run_dirs(base_dir, run_numbers=None):
    base_dir = Path(base_dir)
    if run_numbers is None:
        run_dirs = sorted(base_dir.glob("RUN*"))
        print(f"Found {len(run_dirs)} RUN directories")
    else:
        run_dirs = [base_dir / f"RUN{n}" for n in run_numbers]
        print(f"Looking for {len(run_dirs)} specified RUN directories")
    return [d for d in run_dirs if d.is_dir()]


# ─────────────────────────────────────────────
# Frame count
# ─────────────────────────────────────────────

def get_trajectory_frames(run_dir):
    """Try to get total frame count from saved pickle state, then fallback."""
    run_dir = Path(run_dir)

    state_file = run_dir / "analyzer_state.pkl"
    if state_file.exists():
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            if 'end_frame' in state:
                return int(state['end_frame']) + 1
        except Exception:
            pass

    try:
        import MDAnalysis as mda
        for traj_name in ['protein.nc', 'trajectory.nc', 'traj.nc', 'prod.nc']:
            traj_file = run_dir / traj_name
            if not traj_file.exists():
                continue
            for top_name in ['com_4fs.prmtop', 'com.prmtop', 'topology.prmtop']:
                top_file = run_dir / top_name
                if top_file.exists():
                    u = mda.Universe(str(top_file), str(traj_file))
                    return len(u.trajectory)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────
# Stage counting from ion_region_tracking.json
# ─────────────────────────────────────────────

def count_stages_from_tracking(run_dir):
    """
    Read ion_region_tracking.json and report where each ion was sitting
    at the LAST frame it was observed inside the channel.

    Biological meaning of regions:
        1 = SF region        (R1: start1 -> start2)
        2 = GLU/ASN cavity   (R2: start2 -> end2)
        3 = GLU/ASN -> HBC   (R3: end2 -> end3)
        4 = After HBC        (R4: past end3) — completed permeation

    ion_region_tracking structure: { "ion_id": { "frame": region, ... } }

    Returns dict with counts, or None if file not found.
    """
    tracking_file = Path(run_dir) / "ion_region_tracking.json"
    if not tracking_file.exists():
        return None

    with open(tracking_file, 'r') as f:
        ion_region_tracking = json.load(f)

    # For each ion, get the region it was in at its last recorded frame
    last_frame_region = {}
    for ion_id_str, frame_dict in ion_region_tracking.items():
        if not frame_dict:
            continue
        last_frame = max(int(f) for f in frame_dict.keys())
        last_frame_region[int(ion_id_str)] = int(frame_dict[str(last_frame)])

    counts = defaultdict(int)
    for region in last_frame_region.values():
        counts[region] += 1

    return {
        'total_tracked_ions': len(last_frame_region),
        'In_SF':              counts.get(1, 0),   # R1: SF region
        'In_GLU_ASN':         counts.get(2, 0),   # R2: GLU/ASN cavity
        'In_ASN_HBC':         counts.get(3, 0),   # R3: GLU/ASN -> HBC
        'After_HBC':          counts.get(4, 0),   # R4: completed permeation
    }


# ─────────────────────────────────────────────
# Permeation table (complete events only)
# ─────────────────────────────────────────────

def load_permeation_table(run_dir):
    """
    Load permeation_table.json.
    Returns (total_entries, unique_ions) or (0, 0) if file missing.
    """
    perm_file = Path(run_dir) / "permeation_table.json"
    if not perm_file.exists():
        return 0, 0

    with open(perm_file, 'r') as f:
        events = json.load(f)

    total_entries = len(events)
    unique_ions   = len(set(e['ion_id'] for e in events))
    return total_entries, unique_ions




# ─────────────────────────────────────────────
# Per-run analysis
# ─────────────────────────────────────────────

def analyze_run(run_dir):
    run_dir  = Path(run_dir)
    run_name = run_dir.name
    print(f"\nProcessing {run_name}...")

    n_frames                       = get_trajectory_frames(run_dir)
    perm_entries, perm_unique_ions = load_permeation_table(run_dir)
    stage_counts                   = count_stages_from_tracking(run_dir)

    row = {
        'Run':                       run_name,
        'Total_Frames':              n_frames,
        'Full_Permeations_Unique':   perm_unique_ions,
        'Full_Permeations_Total':    perm_entries,
        'Entries_Per_Frame':         perm_entries / n_frames if n_frames else 0.0,
    }

    if stage_counts is not None:
        row.update({
            'Total_Tracked_Ions': stage_counts['total_tracked_ions'],
            'In_SF':              stage_counts['In_SF'],
            'In_GLU_ASN':         stage_counts['In_GLU_ASN'],
            'In_ASN_HBC':         stage_counts['In_ASN_HBC'],
            'After_HBC':          stage_counts['After_HBC'],
            'ion_region_tracking_found': True,
        })
        # HBC permeations = full permeations + ions still sitting after HBC
        after_hbc = stage_counts['After_HBC']
        row['HBC_Permeations'] = perm_entries + after_hbc
        print(f"  Frames:          {n_frames}")
        print(f"  HBC permeations (Full_Tot + AfterHBC): {row['HBC_Permeations']}")
        print(f"  Full permeations: {perm_entries} total  ({perm_unique_ions} unique ions)")
        print(f"  Last-frame snapshot (ion_region_tracking.json):")
        print(f"    Total ions tracked: {stage_counts['total_tracked_ions']}")
        print(f"    In SF region:       {stage_counts['In_SF']}")
        print(f"    In GLU/ASN cavity:  {stage_counts['In_GLU_ASN']}")
        print(f"    In ASN->HBC:    {stage_counts['In_ASN_HBC']}")
        print(f"    After HBC:          {stage_counts['After_HBC']}")
    else:
        row.update({
            'Total_Tracked_Ions':        None,
            'In_SF':                     None,
            'In_GLU_ASN':                None,
            'In_ASN_HBC':                None,
            'After_HBC':                 None,
            'ion_region_tracking_found': False,
        })
        row['HBC_Permeations'] = None
        print(f"  WARNING: ion_region_tracking.json not found")

    return row



# ─────────────────────────────────────────────
# Region occupancy plot
# ─────────────────────────────────────────────

def plot_last_frame_regions(df, output_dir):
    """
    Bar plot showing how many ions were in each biological region
    at their last observed frame, for each run.
    Also plots a stacked bar across all runs combined.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    output_dir = Path(output_dir)
    region_cols   = ['In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']
    region_labels = ['SF (R1)', 'GLU/ASN cavity (R2)', 'ASN→HBC (R3)', 'After HBC (R4)']
    colors        = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    # Filter to runs that have tracking data
    plot_df = df[df['ion_region_tracking_found'] == True].copy()
    if plot_df.empty:
        print("  No ion_region_tracking data available for plotting.")
        return

    runs = plot_df['Run'].tolist()
    x    = range(len(runs))
    width = 0.18

    # ── Plot 1: grouped bar per run ──
    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 1.2), 6))
    for i, (col, label, color) in enumerate(zip(region_cols, region_labels, colors)):
        vals = plot_df[col].fillna(0).astype(int).tolist()
        offsets = [xi + (i - 1.5) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=label, color=color, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(v), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(runs, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of ions', fontsize=12)
    ax.set_title('Ion region occupancy at last observed frame (per run)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = output_dir / 'last_frame_regions_per_run.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Plot saved: {out}")

    # ── Plot 2: stacked bar — total across all runs ──
    totals = [plot_df[col].fillna(0).sum() for col in region_cols]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(region_labels, totals, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(v)), ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Number of ions (all runs)', fontsize=12)
    ax.set_title('Ion region occupancy at last observed frame (all runs)', fontsize=13)
    ax.set_ylim(bottom=0)
    plt.xticks(fontsize=10, rotation=15, ha='right')
    plt.tight_layout()
    out = output_dir / 'last_frame_regions_total.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Plot saved: {out}")

# ─────────────────────────────────────────────
# Summary & output
# ─────────────────────────────────────────────

def create_summary(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    n_sims   = len(df)
    valid    = df[df['Total_Frames'].notna()]
    n_frames = valid['Total_Frames'].mean() if len(valid) else 0

    total_full_unique  = df['Full_Permeations_Unique'].sum()
    total_full_total   = df['Full_Permeations_Total'].sum()
    hbc_u              = df['HBC_Permeations'].sum(skipna=True)
    mean_rate          = valid['Entries_Per_Frame'].mean() if len(valid) else 0
    has_tracking       = df['ion_region_tracking_found'].any()

    summary_row = {
        'Number_of_Simulations':        n_sims,
        'Mean_Frames_per_Simulation':   round(n_frames, 0),
        'HBC_Permeations':              hbc_u,
        'Full_Permeations_Unique':      total_full_unique,
        'Full_Permeations_Total':       total_full_total,
        'Mean_Full_Permeations_Per_Frame': round(mean_rate, 6),
    }

    if has_tracking:
        for col in ['Total_Tracked_Ions', 'HBC_Permeations', 'In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']:
            summary_row[f'Total_{col}'] = df[col].sum(skipna=True)

    print(f"Simulations:                   {n_sims}")
    print(f"Mean frames/sim:               {n_frames:.0f}")
    print(f"HBC permeations (Full_Tot + AfterHBC): {hbc_u}")
    print(f"Full permeations:              {total_full_total} total  ({total_full_unique} unique ions)")
    print(f"Mean full permeations/frame:   {mean_rate:.6f}")
    if has_tracking:
        print(f"\nLast-frame snapshot totals across all runs:")
        print(f"  In SF region:      {summary_row['Total_In_SF']}")
        print(f"  In GLU/ASN cavity: {summary_row['Total_In_GLU_ASN']}")
        print(f"  In ASN->HBC:   {summary_row['Total_In_ASN_HBC']}")
        print(f"  After HBC:         {summary_row['Total_After_HBC']}")

    # ── CSV ──
    pd.DataFrame([summary_row]).to_csv(output_dir / 'permeation_summary.csv', index=False)
    df.to_csv(output_dir / 'permeation_per_run.csv', index=False)
    print(f"\n✅ permeation_summary.csv saved")
    print(f"✅ permeation_per_run.csv saved")

    # ── Excel ──
    excel_file = output_dir / 'permeation_analysis.xlsx'
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = Workbook()

        header_font  = Font(name='Arial', bold=True, color='FFFFFF')
        header_fill  = PatternFill('solid', start_color='2E4057')
        center_align = Alignment(horizontal='center')
        alt_fill     = PatternFill('solid', start_color='EAF2FB')

        # Sheet 1: Summary
        ws = wb.active
        ws.title = "Summary"
        headers = list(summary_row.keys())
        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h.replace('_', ' '))
            cell.font = header_font; cell.fill = header_fill; cell.alignment = center_align
        for ci, h in enumerate(headers, 1):
            ws.cell(row=2, column=ci, value=summary_row[h])
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or '')) for c in col) + 4, 40)

        # Sheet 2: Per-run details
        ws2 = wb.create_sheet("Per_Run_Details")
        detail_cols = ['Run', 'Total_Frames',
                       'HBC_Permeations',
                       'Full_Permeations_Unique', 'Full_Permeations_Total', 'Entries_Per_Frame',
                       'Total_Tracked_Ions', 'In_SF', 'In_GLU_ASN', 'In_ASN_HBC', 'After_HBC']
        detail_cols = [c for c in detail_cols if c in df.columns]
        for ci, h in enumerate(detail_cols, 1):
            cell = ws2.cell(row=1, column=ci, value=h.replace('_', ' '))
            cell.font = header_font; cell.fill = header_fill; cell.alignment = center_align
        for ri, (_, row) in enumerate(df[detail_cols].iterrows(), 2):
            for ci, val in enumerate(row, 1):
                cell = ws2.cell(row=ri, column=ci, value=val)
                cell.font = Font(name='Arial')
                if ri % 2 == 0:
                    cell.fill = alt_fill
        for col in ws2.columns:
            ws2.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or '')) for c in col) + 4, 30)

        wb.save(excel_file)
        print(f"✅ permeation_analysis.xlsx saved")
    except ImportError:
        print("⚠  openpyxl not installed — Excel file skipped")
    except Exception as e:
        print(f"⚠  Could not create Excel file: {e}")

    # ── Text report ──
    report_file = output_dir / 'permeation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERMEATION EVENT ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("OVERALL STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of simulations:                       {n_sims}\n")
        f.write(f"Mean frames per simulation:                  {n_frames:.0f}\n")
        f.write(f"HBC permeations (Full_Tot + AfterHBC): {hbc_u}\n")
        f.write(f"Full permeations (passed end4):  {total_full_total} total  ({total_full_unique} unique ions)\n")
        f.write(f"Mean full permeations per frame: {mean_rate:.6f}\n")
        if has_tracking:
            f.write(f"\nLast-frame snapshot totals (ion_region_tracking.json):\n")
            f.write(f"  In SF region (R1):       {summary_row['Total_In_SF']}\n")
            f.write(f"  In GLU/ASN cavity (R2):  {summary_row['Total_In_GLU_ASN']}\n")
            f.write(f"  In ASN->HBC (R3):    {summary_row['Total_In_ASN_HBC']}\n")
            f.write(f"  After HBC (R4):          {summary_row['Total_After_HBC']}\n")
        f.write("\n" + "="*70 + "\n\n")
        def fmt(v): return f"{int(v)}" if pd.notna(v) else "N/A"

        f.write("PER-RUN DETAILS:\n")
        sep = "-" * 105
        f.write(sep + "\n")
        f.write(
            f"{'Run':<8} {'Frames':<8} "
            f"{'HBC_Perm':<10} "
            f"{'Full_Tot':<10} {'Full_Uniq':<11} "
            f"{'Tracked':<9} {'In_SF':<7} {'GLU_ASN':<9} {'ASN_HBC':<9} {'AfterHBC':<9}\n"
        )
        f.write(sep + "\n")
        for _, row in df.iterrows():
            fr = f"{int(row['Total_Frames'])}" if pd.notna(row['Total_Frames']) else "?"
            f.write(
                f"{row['Run']:<8} {fr:<8} "
                f"{fmt(row.get('HBC_Permeations')):<10} "
                f"{fmt(row.get('Full_Permeations_Total')):<10} "
                f"{fmt(row.get('Full_Permeations_Unique')):<11} "
                f"{fmt(row.get('Total_Tracked_Ions')):<9} "
                f"{fmt(row.get('In_SF')):<7} "
                f"{fmt(row.get('In_GLU_ASN')):<9} "
                f"{fmt(row.get('In_ASN_HBC')):<9} "
                f"{fmt(row.get('After_HBC')):<9}\n"
            )
        f.write("\n" + "="*70 + "\n")
        f.write("\nNOTES:\n")
        f.write("- Complete_Perm_Entries: rows in permeation_table.json (one per permeation\n")
        f.write("  event; same ion appears multiple times if it re-permeated).\n")
        f.write("- Complete_Perm_Unique_Ions: distinct ion IDs in permeation_table.json.\n")
        f.write("- Last-frame snapshot columns come from ion_region_tracking.json and show\n")
        f.write("  where each ion was sitting at the last frame it was observed in the channel.\n")
        f.write("  In_SF = SF region (R1), In_GLU_ASN = GLU/ASN cavity (R2),\n")
        f.write("  In_ASN_HBC = ASN->HBC (R3), After_HBC = completed permeation (R4).\n")

    print(f"✅ permeation_report.txt saved")

    # Plot last-frame region occupancy
    plot_last_frame_regions(df, output_dir)

    return summary_row


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze permeation events across multiple simulation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/girk_analyser_results/G12_GAT
  %(prog)s /path/to/girk_analyser_results/G12_GAT --runs 1 2 3 4 5
  %(prog)s /path/to/girk_analyser_results/G12_GAT -o my_summary
        """
    )
    parser.add_argument('base_dir', help='Base directory containing RUN1, RUN2, …')
    parser.add_argument('--runs', nargs='+', type=int,
                        help='Specific run numbers (e.g. --runs 1 2 3)')
    parser.add_argument('-o', '--output', default='permeation_summary',
                        help='Output subdirectory name inside base_dir (default: permeation_summary)')

    args       = parser.parse_args()
    base_dir   = Path(args.base_dir)
    output_dir = base_dir / args.output

    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return 1

    print("\n" + "="*70)
    print("PERMEATION EVENT SUMMARY ANALYSIS")
    print("="*70)
    print(f"Base directory:  {base_dir}")
    print(f"Output directory: {output_dir}")

    run_dirs = find_run_dirs(base_dir, args.runs)
    if not run_dirs:
        print("ERROR: No RUN directories found.")
        return 1

    rows = [analyze_run(d) for d in run_dirs]
    df   = pd.DataFrame(rows)

    create_summary(df, output_dir)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nAll results in: {output_dir}/")
    print("  • permeation_analysis.xlsx")
    print("  • permeation_summary.csv")
    print("  • permeation_per_run.csv")
    print("  • permeation_report.txt\n")
    return 0


if __name__ == "__main__":
    exit(main())

'''
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G12/
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G12_GAT/
python3 analyze_permeation_summary.py /media/ziyue/.../girk_analyser_results/G2/
'''

'''
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/permeation_summary 
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/permeation_summary 
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/permeation_summary 
python3 analyze_permeation_summary.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ -o /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/permeation_summary 
'''