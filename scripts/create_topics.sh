#!/usr/bin/env bash
# Create the four topics the QC line needs. Idempotent: --if-not-exists.
# Mac. Requires `docker compose up -d kafka` first.
#
# Partition counts are a design decision, not a default:
#   pcb.boards   3  -> the parallelism ceiling for inference consumers. Day 8
#                      measures 1-vs-3 consumers against exactly this number;
#                      a 4th consumer in the group would sit idle.
#   pcb.results  3  -> mirrors boards so a result keeps its board's ordering
#   pcb.alerts   1  -> alerting is low-volume and total ordering is easier to
#                      reason about than throughput we do not need
#   pcb.dlq      1  -> poison messages should be rare; if they are not, that is
#                      the finding, and one partition makes them easy to read
#
# Retention is 1h: this is a demo line replaying a fixed 1500-board dataset,
# and the 8GB machine should not accumulate a week of base64 PNGs.
set -euo pipefail

K="docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092"

create () {
  local name=$1 parts=$2
  echo "==> $name (partitions=$parts)"
  $K --create --if-not-exists --topic "$name" \
     --partitions "$parts" --replication-factor 1 \
     --config retention.ms=3600000 --config segment.bytes=134217728
}

create pcb.boards  3
create pcb.results 3
create pcb.alerts  1
create pcb.dlq     1

echo
echo "==> topics now present:"
$K --list
