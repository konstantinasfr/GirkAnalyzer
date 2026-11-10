"""
aggregate_visit_hbond.py
========================
Splits the visit-1 results of the site chain by the H-bond state of each glutamate to the residue
above it (see analysis/visit_hbond.py).

Needs, for the system:
  <base>/RUN*/hbond_analysis/<pair>/hbond_per_frame.csv   (analysis/hbond_pair_analysis.py, one per pair)
  <site_chain folder>/visits.csv, events.csv, handover_stays/handover_visits.csv
      (aggregate_site_chain.py, residue model, e.g. <base>/aggregated_site_chain/t3.5_d10/residue_a5b3)

USAGE
    python3 aggregate_visit_hbond.py /path/girk_analyser_results/G12 \\
        --site_chain /path/girk_analyser_results/G12/aggregated_site_chain/t3.5_d10/residue_a5b3 \\
        [--pairs auto | pairs.csv] [--max_k 5]
pairs.csv: columns site,hbond_dir, e.g.
    site,hbond_dir
    E152.2a,SER148_2a_GLU152_2a
    E152.2b,SER148_2b_GLU152_2b
    E152.2c,SER148_2c_GLU152_2c
(site = the glutamate label used by the site chain; hbond_dir = the folder name under RUN*/hbond_analysis/)
"""

import argparse

import pandas as pd

from analysis.visit_hbond import analyze, read_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", help="system folder holding RUN*/")
    p.add_argument("--site_chain", required=True, help="site-chain residue-model folder (visits.csv, events.csv)")
    p.add_argument("--pairs", default="auto", help="auto (default: from the RUN*/hbond_analysis folder names) or a CSV with columns site,hbond_dir")
    p.add_argument("--max_k", type=int, default=5, help="top bin of the other K+ in the cavity (default 5 -> 5+)")
    a = p.parse_args()
    analyze(a.base, a.site_chain, read_pairs(a.base, a.pairs), a.max_k)


if __name__ == "__main__":
    main()

'''
python3 aggregate_visit_hbond.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --site_chain /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/aggregated_site_chain/t3.5_d10/residue_a5b3
python3 aggregate_visit_hbond.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --site_chain /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/aggregated_site_chain/t3.5_d2/residue_a5b3
'''
