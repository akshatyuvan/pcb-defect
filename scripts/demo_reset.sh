#!/usr/bin/env bash
# Reset demo state so scripts/demo.sh produces fresh alerts.
#
# Why this is needed: the alerts consumer dedupes by board_id and rehydrates
# that state from artifacts/alerts/alerts.jsonl, so it survives a container
# restart. The producer is also deterministic across runs, so every demo draws
# the same boards - which means the second run onwards suppresses everything.
#
# The committed alerts.jsonl is EVIDENCE (Day 8: 107 alerts, 214 duplicates
# suppressed) and is not deleted. It is moved aside and restored by
# scripts/demo_restore.sh.
set -euo pipefail

LOG=artifacts/alerts/alerts.jsonl

if [ -f "${LOG}.bak" ]; then
  # Refuse to overwrite an existing backup. Running reset twice would move a
  # throwaway demo log over the real one and silently destroy the Day 8
  # evidence - which is exactly what happened once. Fail loudly instead.
  echo "ERROR: ${LOG}.bak already exists."
  echo "Run ./scripts/demo_restore.sh first, or delete the stale backup."
  exit 1
fi

if [ -f "$LOG" ]; then
  mv "$LOG" "${LOG}.bak"
  echo "moved $LOG -> ${LOG}.bak"
else
  echo "no existing log to move"
fi

docker compose down
docker compose up -d --build

echo
echo "waiting for kafka and api to report healthy..."
for _ in $(seq 1 40); do
  if docker compose ps | grep -q "kafka.*healthy" && docker compose ps | grep -q "pcb-api.*healthy"; then
    echo "stack healthy - run ./scripts/demo.sh now"
    exit 0
  fi
  sleep 3
done

echo "timed out waiting for healthy; check: docker compose ps"
exit 1