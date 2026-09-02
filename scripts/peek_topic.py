"""Read N messages off a topic and print the envelope WITHOUT the base64 blob.

Mac. Uses a throwaway consumer group with auto-commit so it never disturbs the
real inference group's offsets (Day 7).
"""
import argparse
import json
import sys
import uuid

from confluent_kafka import Consumer, KafkaError

ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="pcb.boards")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--bootstrap", default="localhost:29092")
a = ap.parse_args()

c = Consumer({
    "bootstrap.servers": a.bootstrap,
    # Throwaway group id: never collides with the pcb-inference group whose
    # committed offsets Day 7 depends on. Reading a topic must not mutate it.
    "group.id": f"peek-{uuid.uuid4()}",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
})
c.subscribe([a.topic])

seen = 0
try:
    while seen < a.n:
        m = c.poll(10.0)
        if m is None:
            print("no more messages within 10s")
            break
        if m.error():
            if m.error().code() == KafkaError._PARTITION_EOF:
                continue
            sys.exit(str(m.error()))
        d = json.loads(m.value())
        # Pop the blob before printing: a 23,000-char base64 string makes the
        # envelope unreadable, and the envelope is the thing being verified.
        # Board envelopes carry an image block; result, alert and DLQ records
        # deliberately do not (locked decision 11 -- results are a few hundred
        # bytes, not 44 KiB). Strip the blob when present, otherwise print as-is.
        img = d.get("image")
        b64len = len(img.pop("b64")) if isinstance(img, dict) and "b64" in img else None
        print("")
        print(f"partition={m.partition()} offset={m.offset()} "
              f"key={m.key().decode()} size={len(m.value()):,}B")
        hdrs = {k: v.decode() for k, v in (m.headers() or [])}
        print(f"  headers: {hdrs}")
        print(f"  envelope: {json.dumps(d, indent=4)}")
        if b64len is not None:
            print(f"  b64 length: {b64len:,} chars  "
                  f"(~{b64len * 3 // 4 / 1024:.1f} KiB of image)")
        seen += 1
finally:
    c.close()