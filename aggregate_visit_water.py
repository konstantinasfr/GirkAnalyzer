"""
aggregate_visit_water.py
========================
Water at the site-chain stops, pooled over the runs of one system (see analysis/visit_water.py).

Needs:
  <base>/RUN*/visit_water/water_ions_run*.csv, water_sites_run*.csv   (run_visit_water from main.py)
  <site_chain folder>/visits.csv, events.csv, handover_stays/handover_visits.csv  (aggregate_site_chain.py)
  optional --pairs pairs.csv (site,hbond_dir) to split the residues' water by their H-bond to the
  residue above (RUN*/hbond_analysis/<hbond_dir>/hbond_per_frame.csv, as aggregate_visit_hbond.py)

USAGE
    python3 aggregate_visit_water.py /path/girk_analyser_results/G12 \\
        --site_chain /path/girk_analyser_results/G12/aggregated_site_chain/t3.5_d10/residue_a5b3 [--pairs pairs.csv]
"""

import argparse

import pandas as pd

from analysis.visit_water import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", help="system folder holding RUN*/")
    p.add_argument("--site_chain", required=True, help="site-chain residue-model folder (visits.csv, events.csv)")
    p.add_argument("--pairs", help="optional CSV with columns site,hbond_dir")
    a = p.parse_args()
    pairs = None
    if a.pairs:
        t = pd.read_csv(a.pairs)
        pairs = dict(zip(t["site"], t["hbond_dir"]))
    analyze(a.base, a.site_chain, pairs)


if __name__ == "__main__":
    main()
