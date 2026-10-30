#!/usr/bin/env bash
# Pilot: three real episodes (English, Hindi, short) through the whole pipeline.
#   bash scripts/pilot.sh                 # default three
#   bash scripts/pilot.sh ID1 ID2         # your own episode ids
# Run scripts/doctor.py first. Stops at the first failing command.
set -euo pipefail
cd "$(dirname "$0")/.."
IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then IDS=(Y566_T-YlNQ E3oG313_kps 9QXCkMTbrSk); fi

python -m fo.ingest.cli discover
for id in "${IDS[@]}"; do
  echo "=== $id ==="
  python -m fo.ingest.cli captions --video "$id"
  python -m fo.ingest.cli profile  --video "$id"
  # Scribe may fail for some videos; the episode then continues on its caption transcript
  python -m fo.ingest.cli scribe   --video "$id" || echo "scribe failed for $id: continuing with captions"
  python -m fo.ingest.cli process  --video "$id"
  python -m fo.ingest.cli publish  --video "$id"
done
python -m fo.ingest.cli status
echo "Done. Start the app:  uvicorn fo.app.main:app --reload   then ask about these episodes."
