"""
combine_pip_199_64_permeation.py
=================================
Standalone script — run this separately from main.py after all RUNs
are done.

Reads every RUN's P5_199_64_permeation_table.csv under:
    <base_dir>/RUN{n}/pip_199_64_permeation_corr/{phosphate}_199_64_permeation_table.csv

and combines into one row per run:
    run | pip_2a_199 | pip_2a_64 | pip_2b_199 | pip_2b_64 | ... | n_permeations

Usage:
    python3 combine_pip_199_64_permeation.py --base_dir /path/to/G12 --phosphate P5 --n_runs 20
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def combine_permeation_correlation_tables(
    base_dir,
    channel_type="G12",
    phosphate="P5",
    subunit_order=("2a", "2b", "2c", "G1"),
    out_path=None,
):
    base_dir = Path(base_dir)
    run_dirs = sorted(
        base_dir.glob("RUN*"),
        key=lambda p: int(p.name.replace("RUN", ""))
    )

    rows = []

    for run_dir in run_dirs:
        run = int(run_dir.name.replace("RUN", ""))
        csv_path = run_dir / "pip_199_64_permeation_corr" / f"{phosphate}_199_64_permeation_table.csv"
        if not csv_path.exists():
            print(f"Missing, skipping -> {csv_path}")
            continue

        df_run = pd.read_csv(csv_path)
        row = {"run": run}
        for subunit in subunit_order:
            sub_df = df_run[df_run["subunit"] == subunit]
            if sub_df.empty:
                row[f"pip_{subunit}_199"] = np.nan
                row[f"pip_{subunit}_64"] = np.nan
            else:
                row[f"pip_{subunit}_199"] = sub_df["pct_close_199"].iloc[0]
                row[f"pip_{subunit}_64"] = sub_df["pct_close_64"].iloc[0]

        row["n_permeations"] = df_run["n_permeations"].iloc[0]
        rows.append(row)

    combined = pd.DataFrame(rows)
    pct_cols = [c for c in combined.columns if c.startswith("pip_")]
    combined[pct_cols] = combined[pct_cols].round(2)

    if out_path is None:
        out_path = base_dir / f"combined_{channel_type}_{phosphate}_199_64_permeation_table.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Parent folder containing RUN1...RUNn")
    parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--phosphate", default="P5")
    args = parser.parse_args()

    combine_permeation_correlation_tables(
        base_dir=args.base_dir,
        channel_type=args.channel_type,
        phosphate=args.phosphate,
    )

'''
python3 combine_pip_199_64_permeation.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12
'''