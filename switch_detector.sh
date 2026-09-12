#!/bin/bash

BASE=/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12
SCRIPT=/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyzer/switch_detector.py  # <-- update this to where your script is

for RUN in $(seq 1 20); do
    echo "========================================"
    echo "Processing RUN${RUN}"
    echo "========================================"

    R=${BASE}/RUN${RUN}

    # ── 2b subunit ───────────────────────────────────────────────────────────
    python3 $SCRIPT \
        --folders ${R}/residue_neighbors/197_2b/hbonds \
                  ${R}/residue_neighbors/199_2b/hbonds \
                  ${R}/residue_neighbors/315_2a/hbonds \
        --labels GLN197.2b LYS199.2b GLU315.2a \
        --occ_threshold 0.2 \
        --min_window 200 \
        --out ${R}/switch_output/197_2b_199_2b_315_2a

    # ── 2c subunit ───────────────────────────────────────────────────────────
    python3 $SCRIPT \
        --folders ${R}/residue_neighbors/197_2c/hbonds \
                  ${R}/residue_neighbors/199_2c/hbonds \
                  ${R}/residue_neighbors/315_2b/hbonds \
        --labels GLN197.2c LYS199.2c GLU315.2b \
        --occ_threshold 0.2 \
        --min_window 200 \
        --out ${R}/switch_output/197_2c_199_2c_315_2b

    # ── 2a subunit ───────────────────────────────────────────────────────────
    python3 $SCRIPT \
        --folders ${R}/residue_neighbors/197_2a/hbonds \
                  ${R}/residue_neighbors/199_2a/hbonds \
                  ${R}/residue_neighbors/304_G1/hbonds \
        --labels GLN197.2a LYS199.2a GLU304.G1 \
        --occ_threshold 0.2 \
        --min_window 200 \
        --out ${R}/switch_output/197_2a_199_2a_304_G1

    # ── G1 subunit ───────────────────────────────────────────────────────────
    python3 $SCRIPT \
        --folders ${R}/residue_neighbors/186_G1/hbonds \
                  ${R}/residue_neighbors/188_G1/hbonds \
                  ${R}/residue_neighbors/315_2c/hbonds \
        --labels GLN186.G1 LYS188.G1 GLU315.2c \
        --occ_threshold 0.2 \
        --min_window 200 \
        --out ${R}/switch_output/186_G1_188_G1_315_2c

    echo "Done RUN${RUN}"
    echo ""
done

echo "All RUNs completed."