"""
pcb.alerts -> durable alert log + operator signal.

Runs in Docker (compose service `alerts`). Single instance: pcb.alerts has one
partition on purpose, because alert ORDER matters to a human reading a log, and
three partitions would interleave them.

Three things it does beyond printing:

1. Deduplicates by board_id. The inference consumer is at-least-once, so a
   crash between produce and commit replays a board and re-alerts on it. The
   fix belongs at the SINK, not in the pipeline: making the pipeline
   exactly-once would cost Kafka transactions and throughput to solve a problem
   a bounded dedupe window solves for free.

2. Tracks a rolling verdict mix over a sliding window. With --include-templates
   the stream carries both pass and fail, so a shifting mix is meaningful: a
   rising `review` share means the model is increasingly landing between the
   calibrated thresholds, i.e. seeing inputs outside its calibration range.
   That is the distribution-shift signal. A fail-rate monitor was rejected --
   on an all-defective stream it reads 100% forever and detects nothing.

3. Writes the alert log line-buffered and fsynced BEFORE committing the offset,
   the same ordering argument as the inference consumer: a crash between write
   and commit replays the alert (dedupe absorbs it); a crash between commit and
   write would lose it silently.

No torch, no model. Same 167MB image as the inference consumer.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict

from confluent_kafka import Consumer

BOOTSTRAP = os.environ.get("PCB_BOOTSTRAP", "kafka:9092")
GROUP = os.environ.get("PCB_ALERTS_GROUP", "pcb-alerts")
T_ALERTS = os.environ.get("PCB_TOPIC_ALERTS", "pcb.alerts")
OUT = Path(os.environ.get("PCB_ALERT_LOG", "artifacts/alerts/alerts.jsonl"))
DEDUPE_MAX = int(os.environ.get("PCB_ALERT_DEDUPE_MAX", "5000"))
WINDOW = int(os.environ.get("PCB_ALERT_WINDOW", "50"))
REVIEW_RATE_ALARM = float(os.environ.get("PCB_REVIEW_RATE_ALARM", "0.20"))
ALARM_COOLDOWN_S = float(os.environ.get("PCB_ALARM_COOLDOWN_S", "60"))
CONSUMER_ID = os.environ.get("PCB_CONSUMER_ID", socket.gethostname())

RUNNING = True


def log(event: str, **kw: Any) -> None:
    rec = {"ts": time.time(), "logger": "consumer_alerts",
           "consumer_id": CONSUMER_ID, "event": event}
    rec.update(kw)
    print(json.dumps(rec, default=str), flush=True)


def _stop(signum, frame):  # noqa: ARG001
    global RUNNING
    RUNNING = False
    log("shutdown_signal", signal=signum)


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a", buffering=1)      # line-buffered: a crash keeps whole lines

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP,
        "enable.auto.commit": False,     # same discipline as the inference consumer
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 45000,
    })
    consumer.subscribe([T_ALERTS])
    log("subscribed", topic=T_ALERTS, group=GROUP, out=str(OUT),
        window=WINDOW, review_rate_alarm=REVIEW_RATE_ALARM)

    seen: "OrderedDict[str, float]" = OrderedDict()
    window: Deque[str] = deque(maxlen=WINDOW)
    last_alarm = 0.0
    n_written = n_dupe = 0

    try:
        while RUNNING:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log("kafka_error", error=str(msg.error()))
                continue
            try:
                doc: Dict[str, Any] = json.loads(msg.value())
            except Exception as e:
                # Undecodable alert: log and commit. There is no second DLQ for
                # the alert stream -- an alert we cannot parse is already a
                # symptom of a bug upstream, and blocking on it would stop every
                # subsequent alert reaching the operator.
                log("undecodable_alert", error=str(e),
                    partition=msg.partition(), offset=msg.offset())
                consumer.commit(message=msg, asynchronous=False)
                continue

            bid = str(doc.get("board_id", f"?p{msg.partition()}o{msg.offset()}"))
            if bid in seen:
                n_dupe += 1
                log("duplicate_suppressed", board_id=bid)
                consumer.commit(message=msg, asynchronous=False)
                continue
            seen[bid] = time.time()
            while len(seen) > DEDUPE_MAX:
                seen.popitem(last=False)     # FIFO eviction, bounded memory

            verdict = doc.get("verdict", "?")
            window.append(verdict)

            fh.write(json.dumps({
                "received_ms": time.time() * 1000.0,
                "severity": doc.get("severity"),
                "board_id": bid,
                "verdict": verdict,
                "score": doc.get("score"),
                "statistic": doc.get("statistic"),
                "n_flagged": doc.get("n_flagged"),
                "class_counts": doc.get("class_counts"),
                "e2e_ms": (doc.get("timings_ms") or {}).get("e2e"),
            }) + "\n")
            os.fsync(fh.fileno())
            consumer.commit(message=msg, asynchronous=False)
            n_written += 1

            log("alert", board_id=bid, verdict=verdict,
                severity=doc.get("severity"), score=doc.get("score"))

            if len(window) == WINDOW:
                rate = sum(1 for v in window if v == "review") / WINDOW
                if rate >= REVIEW_RATE_ALARM and (time.time() - last_alarm) > ALARM_COOLDOWN_S:
                    last_alarm = time.time()
                    log("ALARM_review_rate", window=WINDOW,
                        review_rate=round(rate, 3), threshold=REVIEW_RATE_ALARM,
                        message="sustained uncertain boards -- model is seeing "
                                "inputs outside its calibration range")
    finally:
        log("closing", written=n_written, duplicates_suppressed=n_dupe)
        fh.close()
        consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())