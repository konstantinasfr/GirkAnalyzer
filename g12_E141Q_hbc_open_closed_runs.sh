# HBC open vs closed analysis (analysis/hbc_open_closed.py), only for the runs listed.
# Put the run numbers in RUNS. For the 4 fs runs use CHANNEL=G12_E141Q_4fs.
CHANNEL=G12_E141Q_4fs
CHANNEL=G12
RUNS="2 4"          # e.g. RUNS="3 5"
# RUNS="1 6 9"          # e.g. RUNS="3 5"

if [ -z "$RUNS" ]; then echo "Edit RUNS in $0 first"; exit 1; fi
for r in $RUNS; do
    python3 main.py --channel_type $CHANNEL --run_number $r --hbc_open_closed_only
done
