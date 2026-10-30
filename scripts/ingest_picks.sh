#!/usr/bin/env bash
# 22 picked episodes (2 per cluster). Run AFTER the Colab captions cell. Venv must be active.
#   bash scripts/ingest_picks.sh            # all 22
#   bash scripts/ingest_picks.sh ID1 ID2    # only these
# profile -> process -> publish for each. One episode failing does not stop the others.
set -uo pipefail
cd "$(dirname "$0")/.."
IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then IDS=(sX6gEwH-SA4 YVlWj1LCv54 74dqM_xtvAA sGpc8-f2e8U MdeQMVBuGgY GhZgfEfz8og x63HCoDfAhQ 1S73MBFacFE v9AHBtbk-E0 jr0k_fOR1b0 b9-Mxcoxx-Q zN_lhtJFsgg Y566_T-YlNQ gfCg5srOjN0 E3oG313_kps K-Q4r37XgVw 4Vz6L8B73i4 kKNoBH0iE1k p4qPp_3_f2Y YMTJw1G3yOM ftVNoXn-uAo q1hvfs-VL5U); fi
failed=()
for id in "${IDS[@]}"; do
  echo "=== $id"
  if python -m fo.ingest.cli profile --video "$id" \
     && python -m fo.ingest.cli process --video "$id" \
     && python -m fo.ingest.cli publish --video "$id"; then :; else failed+=("$id"); echo "FAILED: $id"; fi
done
python -m fo.ingest.cli status
if [ ${#failed[@]} -gt 0 ]; then echo "failed episodes: ${failed[*]}"; exit 1; fi
