#!/usr/bin/env bash
# A scripted end-to-end demo, paced for screen recording.
#
# Runs on the Mac with (pcb) active and the compose stack already up and HEALTHY.
# Every sleep is there so a viewer can read a line before the next one lands.
#
# Note on the alerts consumer: it dedupes by board_id in an in-memory bounded
# dict. If it has already seen these boards in THIS process lifetime, step 5
# prints nothing. Record on the first run after a `docker compose down`.
set -euo pipefail

GROUP="${PCB_GROUP:-pcb-inference}"

hdr() { printf '\n\033[1;36m== %s\033[0m\n' "$1"; sleep 1.5; }

hdr "1. The stack"
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
sleep 3

hdr "2. The consumer holds no model - this is supposed to fail"
docker compose exec -T inference python -c "import torch" || true
sleep 3

hdr "3. The API knows exactly which model version it is serving"
curl -s localhost:8000/model | python -c "
import json, sys
m = json.load(sys.stdin)
print('registry   :', m['registry_name'], 'v' + str(m['registry_version']))
print('run_id     :', m['run_id'])
print('sha256     :', m['staged_model_sha256'][:32] + '...')
print('classes    :', ', '.join(m['classes']))
print('board rule :', m['board_operating_point']['statistic'],
      'pass <', round(m['board_operating_point']['pass_threshold'], 6),
      '| fail >=', round(m['board_operating_point']['fail_threshold'], 6))
"
sleep 4

hdr "4. Stream 12 boards: real defective boards plus clean templates"
python -m src.streaming.producer --limit 12 --rate 3 --shuffle --include-templates
sleep 2

hdr "5. Alerts from THIS run. Note the _temp board: a clean template routed to review"
sleep 8
python -c "
import json, time
# The alert log accumulates across runs and the consumer dedupes by board_id,
# rehydrating that state from the file itself - so a bare tail would show a
# viewer alerts that have nothing to do with the boards just streamed. Filter
# to this run's window instead.
cutoff = time.time() * 1000 - 180_000
rows = [json.loads(l) for l in open('artifacts/alerts/alerts.jsonl')]
rows = [r for r in rows if r['received_ms'] >= cutoff]
if not rows:
    print('  no NEW alerts - every board here was already in the log')
    print('  (dedupe working; run scripts/demo_reset.sh for a clean demo)')
for r in rows[-8:]:
    print(f\"{r['board_id']:>14}  {r['verdict']:>7}  {r['severity']:>8}  score={r['score']:.6f}\")
"
sleep 4

hdr "6. Consumer lag is zero - every board committed AFTER inference"
docker compose exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --describe --group "$GROUP" \
  | awk 'NR==1 || $0 ~ /pcb.boards/ {print $2, $3, $6}' | column -t
sleep 4