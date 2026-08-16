"""Take ONE message off pcb.boards, verify its hash, POST it to the API.

This is not Day 7's consumer - no manual commit, no DLQ, no results topic.
It exists only to prove today that the wire format the producer writes is
exactly what the API can eat, before Day 7 builds real routing on top of it.
Mac.
"""
import base64
import json
import sys
import uuid

import requests
from confluent_kafka import Consumer, KafkaError

from src.streaming.schema import BoardMessage

c = Consumer({
    "bootstrap.servers": "localhost:29092",
    "group.id": f"verify-{uuid.uuid4()}",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
})
c.subscribe(["pcb.boards"])

m = None
for _ in range(20):
    m = c.poll(5.0)
    if m is not None and not m.error():
        break
    if m is not None and m.error() and m.error().code() != KafkaError._PARTITION_EOF:
        sys.exit(str(m.error()))
if m is None:
    sys.exit("no message on pcb.boards - run the producer first")

msg = BoardMessage.from_json(m.value())
raw = msg.image_bytes()          # raises if sha256 disagrees
print(f"board {msg.board_id} (group {msg.group}) "
      f"partition={m.partition()} offset={m.offset()} image={len(raw):,}B sha OK")

r = requests.post(
    "http://127.0.0.1:8000/predict/board",
    json={
        "image_b64": base64.b64encode(raw).decode(),
        "board_id": msg.board_id,
        "trace_id": str(uuid.uuid4()),
    },
    timeout=30,
)
r.raise_for_status()
b = r.json()
keys = ("board_id", "verdict", "n_flagged", "max_defect_score",
        "class_counts", "latency_ms")
print(json.dumps({k: b[k] for k in keys}, indent=2))
c.close()
print("")
print("END-TO-END OK: producer -> kafka -> (manual hop) -> api")