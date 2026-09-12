"""
Creates a combined CSV comparing specific PIP2 combinations across G12, G12_ML, G12_GAT.
Extracts pct values for selected combinations from each system's global_combination_counts_pdb.csv.
"""

import pandas as pd
from pathlib import Path

# ── file paths and system labels ─────────────────────────────────────────────
## 2b
files = {
    "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1301/P4/combination_analysis/global_combination_counts_pdb.csv",
    "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1302/P4/combination_analysis/global_combination_counts_pdb.csv",
    "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1304/P4/combination_analysis/global_combination_counts_pdb.csv",
}

# ## 2c
# files = {
#     "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1303/P4/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1304/P4/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1302/P4/combination_analysis/global_combination_counts_pdb.csv",
# }

# ## 2a
# files = {
#     "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1304/P4/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1303/P4/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1303/P4/combination_analysis/global_combination_counts_pdb.csv",
# }


# ## 2b
# files = {
#     "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1301/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1302/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1304/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
# }

# ## 2c
# files = {
#     "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1303/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1304/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1302/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
# }

# ## 2a
# files = {
#     "G12":     "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/combined_pip_proximity/PIP_1304/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_ML":  "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML/combined_pip_proximity/PIP_1303/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
#     "G12_GAT": "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT/combined_pip_proximity/PIP_1303/P4/end2_frames/combination_analysis/global_combination_counts_pdb.csv",
# }

# ── combinations to extract ───────────────────────────────────────────────────
# These are partial strings — we match any combination that contains ALL the
# residue names listed and has exactly the right number of residues
TARGET_COMBINATIONS = [
    "ARG60+LYS64",
    "LYS64",
    "LYS199+LYS200",
    "LYS199+LYS64",
    "LYS200+LYS64",
    "LYS199",
    "LYS200",
    "LYS199+LYS200+LYS64",
    "none",
]
 
def find_combination(df: pd.DataFrame, target: str) -> float:
    """
    Find the pct value for a target combination.
    Matches rows where combination contains all the residue parts
    and has exactly the right count of residues.
    
    For single residues (e.g. 'LYS199'), matches rows where
    the combination IS exactly that residue (n_residues == 1).
    """
    if target == "none":
        match = df[df["combination"] == "none"]
        if match.empty:
            return 0.0
        return round(float(match.iloc[0]["pct"]), 1)
 
    parts = [p.strip() for p in target.split("+")]
    n = len(parts)
 
    # filter by residue count first
    candidates = df[df["n_residues"] == n].copy()
 
    # then check all parts are present in the combination string
    for part in parts:
        candidates = candidates[candidates["combination"].str.contains(part, regex=False)]
 
    if candidates.empty:
        return 0.0
    # return the highest pct if multiple matches (shouldn't happen but just in case)
    return round(float(candidates["pct"].max()), 1)
 
 
# ── build the combined table ──────────────────────────────────────────────────
rows = []
for combo in TARGET_COMBINATIONS:
    row = {"combination": combo}
    for system, path in files.items():
        p = Path(path)
        if not p.exists():
            print(f"WARNING: file not found: {p}")
            row[system] = None
            continue
        df = pd.read_csv(p)
        row[system] = find_combination(df, combo)
    rows.append(row)
 
df_out = pd.DataFrame(rows)
for col in ["G12", "G12_ML", "G12_GAT"]:
    df_out[col] = df_out[col].round(1)
print(df_out.to_string(index=False))



# ── save ──────────────────────────────────────────────────────────────────────
out_path = Path("/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/P4/end2_frames/2c.csv")
out_path = Path("/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/P4/2b.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(out_path, index=False)
print(f"\n✓ Saved to: {out_path}")


# python3 make_combination_comparison.py