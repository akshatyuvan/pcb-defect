#!/usr/bin/env bash
# Day 8. Poll consumer-group lag while a benchmark runs.
#
# Lag is the honest measure of keeping up. Throughput alone cannot tell you: a
# pipeline running 3x too slow still reports a throughput -- it reports the
# producer's rate with an ever-growing queue behind it. Watch the LAG column's
# TREND, not its instantaneous value.
#
#   bash scripts/lag_watch.sh
#   bash scripts/lag_watch.sh pcb-alerts 5
set -euo pipefail

GROUP="${1:-pcb-inference}"
INTERVAL="${2:-2}"

while true; do
  echo "===== $(date +%H:%M:%S)  group=${GROUP} ====="
  docker compose exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 --describe --group "${GROUP}" \
    || echo "(group not active yet)"
  sleep "${INTERVAL}"
done