"""
pcb.boards -> FastAPI -> pcb.results (+ pcb.alerts, + pcb.dlq)

Runs in Docker (compose service `inference`). Can also run on the Mac host
against localhost for debugging.

THE THREE THINGS THIS FILE EXISTS TO GET RIGHT
----------------------------------------------
1. Manual offset commit, AFTER the result is durably produced.
   enable.auto.commit=False. The sequence is: infer -> produce result ->
   flush and confirm delivery -> commit. A crash anywhere before the commit
   means the board is redelivered and reprocessed. That is at-least-once:
   duplicates possible, loss impossible. Committing after inference returns but
   BEFORE producing would open a window where a crash loses the board silently,
   which is exactly the failure a QC line cannot tolerate.

2. A 4xx/5xx distinction mapping to two different beliefs.
   400 (bad base64) and 422 (wrong dimensions) mean THIS MESSAGE is broken.
   Retrying is guaranteed to fail forever and would block the partition head,
   so it goes to the DLQ and we commit. 5xx / refused / timeout means WE are
   broken; retrying is the only correct action and we must NOT commit. After N
   attempts we exit non-zero, compose restarts us, and the uncommitted offset
   means the board is picked up again.

3. Three-outcome routing from a MEASURED board threshold (Day 7 calibration),
   not the Day 2 patch operating point applied as a max over 100 tiles.

Locked decision 10: no torch anywhere in this file's import graph.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from confluent_kafka import Consumer, KafkaException, Producer

from src.streaming import schema as msgschema
from src.streaming.api_client import ApiClientRejection, ApiServerError, PcbApiClient
from src.streaming.policy import FAIL, PASS, REVIEW, BoardCalibration, route
from src.streaming.results import build_dlq_record, build_result, now_ms

# ------------------------------------------------------------------ config ---
# Defaults are the DOCKER values. On the host, override PCB_BOOTSTRAP to
# localhost:29092 -- 9092 is the internal listener and is not reachable from
# the Mac. Mixing them up gives a connect timeout that looks like a dead broker.
BOOTSTRAP = os.environ.get("PCB_BOOTSTRAP", "kafka:9092")
GROUP = os.environ.get("PCB_CONSUMER_GROUP", "pcb-inference")
T_BOARDS = os.environ.get("PCB_TOPIC_BOARDS", "pcb.boards")
T_RESULTS = os.environ.get("PCB_TOPIC_RESULTS", "pcb.results")
T_ALERTS = os.environ.get("PCB_TOPIC_ALERTS", "pcb.alerts")
T_DLQ = os.environ.get("PCB_TOPIC_DLQ", "pcb.dlq")
API_URL = os.environ.get("PCB_API_URL", "http://api:8000")
CALIB_PATH = Path(os.environ.get("PCB_CALIBRATION",
                                 "artifacts/serving/board_calibration.json"))
# Cumulative attempts per (partition, offset) BEFORE we give up and DLQ.
# MAX_5XX_ATTEMPTS is per process life; this one survives restarts within a
# process and, combined with restart:on-failure, bounds the poison-pill loop.
#
# The tradeoff, stated plainly: capping risks discarding a board during an
# outage that outlasts the cap, while not capping means one message that
# reliably 500s blocks its partition head forever and the consumer
# restart-loops. We measured the second failure on Day 7 (a corrupt image made
# the API 500, and the consumer looped indefinitely), so we bound it. 15
# attempts spans roughly 2-3 minutes of a real outage -- long enough that a
# deploy blip does not trigger it, short enough that a poison pill clears fast.
MAX_5XX_ATTEMPTS = int(os.environ.get("PCB_MAX_5XX_ATTEMPTS", "5"))
MAX_TOTAL_ATTEMPTS = int(os.environ.get("PCB_MAX_TOTAL_ATTEMPTS", "15"))
BACKOFF_BASE_S = float(os.environ.get("PCB_BACKOFF_BASE_S", "0.5"))
LIMIT = int(os.environ.get("PCB_CONSUMER_LIMIT", "0"))   # 0 = run forever
CONSUMER_ID = os.environ.get("PCB_CONSUMER_ID", socket.gethostname())

RUNNING = True


def log(event: str, **kw: Any) -> None:
    """Structured JSON to stdout, same shape as the Day 5 service, so
    `docker compose logs` is greppable across the whole stack."""
    rec = {"ts": time.time(), "logger": "consumer_inference",
           "consumer_id": CONSUMER_ID, "event": event}
    rec.update(kw)
    print(json.dumps(rec, default=str), flush=True)


def _stop(signum, frame):  # noqa: ARG001
    global RUNNING
    RUNNING = False
    log("shutdown_signal", signal=signum)


def extract_board_id(env: Any) -> str:
    for name in ("board_id", "boardId", "id"):
        v = getattr(env, name, None)
        if isinstance(v, str) and v:
            return v
    raise AttributeError(f"no board id on envelope; attrs={sorted(vars(env))}")


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    calib = BoardCalibration.from_file(CALIB_PATH)
    log("calibration_loaded", statistic=calib.statistic,
        pass_threshold=calib.pass_threshold, fail_threshold=calib.fail_threshold,
        source=calib.source,
        mode=("three_outcome" if calib.pass_threshold is not None else "two_outcome"))

    api = PcbApiClient(API_URL, timeout=60.0)
    health = api.wait_for_health(180.0)
    log("api_ready", url=API_URL, status=health.get("status"),
        image_field=api.image_field())

    delivery_errors: List[str] = []

    def on_delivery(err, msg):  # noqa: ANN001, ARG001
        if err is not None:
            delivery_errors.append(str(err))

    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "acks": "all",                # a result we cannot afford to lose
        "enable.idempotence": True,
        "compression.type": "gzip",
        "linger.ms": 20,
    })

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP,
        "enable.auto.commit": False,  # THE point of this consumer
        "auto.offset.reset": "earliest",
        # cooperative-sticky gives incremental rebalances: when Day 8 scales
        # 1 -> 3, an existing consumer keeps the partitions it is not giving
        # away, instead of every consumer stopping the world.
        "partition.assignment.strategy": "cooperative-sticky",
        "session.timeout.ms": 45000,
        "max.poll.interval.ms": 300000,
    })
    consumer.subscribe([T_BOARDS])
    log("subscribed", topic=T_BOARDS, group=GROUP, bootstrap=BOOTSTRAP)

    def produce_and_flush(topic: str, key: str, doc: Dict[str, Any]) -> None:
        # flush() per message is deliberate: it makes "the broker has it" true
        # before we commit. It costs a round trip per board, which is nothing at
        # ~1 board/s and would dominate at 100/s. Day 8 measures the cost.
        delivery_errors.clear()
        producer.produce(topic, key=key.encode(), value=json.dumps(doc).encode(),
                         on_delivery=on_delivery)
        remaining = producer.flush(30.0)
        if remaining or delivery_errors:
            raise KafkaException(
                f"delivery to {topic} failed: remaining={remaining} "
                f"errors={delivery_errors}")

    processed = 0
    counts = {PASS: 0, REVIEW: 0, FAIL: 0, "dlq": 0}
    # (partition, offset) -> cumulative 5xx attempts. In-memory only: a
    # container restart resets it, which is why MAX_TOTAL_ATTEMPTS is compared
    # against a running total that the restart loop keeps incrementing within
    # each life. Bounded by the number of in-flight uncommitted offsets, so it
    # cannot grow without limit.
    attempt_ledger: Dict[tuple, int] = {}

    try:
        while RUNNING:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log("kafka_error", error=str(msg.error()))
                continue

            t_consume = now_ms()
            part, off = msg.partition(), msg.offset()
            ts_type, ts_val = msg.timestamp()
            t_produced = float(ts_val) if ts_type != 0 else None  # 0 = NO_TIMESTAMP

            # ------- decode + integrity. Failures here are poison, not ours ---
            try:
                env = msgschema.BoardMessage.from_json(msg.value())
                raw = env.image_bytes()        # verifies sha256 internally
                board_id = extract_board_id(env)
            except Exception as e:
                produce_and_flush(T_DLQ, f"p{part}o{off}", build_dlq_record(
                    reason="malformed", detail=f"{type(e).__name__}: {e}",
                    raw_value=msg.value(), partition=part, offset=off,
                    consumer_id=CONSUMER_ID))
                consumer.commit(message=msg, asynchronous=False)
                counts["dlq"] += 1
                log("dlq", reason="malformed", partition=part, offset=off,
                    detail=f"{type(e).__name__}: {e}")
                continue

            # Re-encode the bytes we just sha256-verified rather than forwarding
            # the envelope's base64 string. Identical bytes either way, but this
            # way what we send is provably what we checked.
            image_b64 = base64.b64encode(raw).decode("ascii")

            # --------------------------------------- inference, with retry ----
            payload: Optional[Dict[str, Any]] = None
            api_ms = 0.0
            attempts = 0
            poisoned: Optional[ApiClientRejection] = None
            while attempts < MAX_5XX_ATTEMPTS and RUNNING:
                attempts += 1
                t0 = time.perf_counter()
                try:
                    payload = api.predict_board(image_b64,
                                                extra={"board_id": board_id})
                    api_ms = (time.perf_counter() - t0) * 1000.0
                    break
                except ApiClientRejection as e:
                    poisoned = e
                    break
                except ApiServerError as e:
                    wait = BACKOFF_BASE_S * (2 ** (attempts - 1))
                    log("api_5xx_retry", board_id=board_id, attempt=attempts,
                        wait_s=wait, error=str(e))
                    time.sleep(wait)

            if poisoned is not None:
                produce_and_flush(T_DLQ, board_id, build_dlq_record(
                    reason=f"rejected_{poisoned.status}", detail=str(poisoned),
                    raw_value=msg.value(), partition=part, offset=off,
                    consumer_id=CONSUMER_ID, board_id=board_id))
                consumer.commit(message=msg, asynchronous=False)
                counts["dlq"] += 1
                log("dlq", reason=f"rejected_{poisoned.status}", board_id=board_id,
                    partition=part, offset=off)
                continue

            if payload is None:
                key = (part, off)
                attempt_ledger[key] = attempt_ledger.get(key, 0) + attempts
                total = attempt_ledger[key]

                if total >= MAX_TOTAL_ATTEMPTS:
                    # This specific message has failed persistently across many
                    # attempts. Treating it as an outage any longer would block
                    # the partition head indefinitely, so we reclassify it as
                    # poison, DLQ it and move on. The DLQ record keeps the raw
                    # payload, so it is replayable once the cause is fixed.
                    produce_and_flush(T_DLQ, board_id, build_dlq_record(
                        reason="retry_exhausted",
                        detail=f"{total} cumulative attempts against 5xx; "
                               f"reclassified as poison to unblock the partition",
                        raw_value=msg.value(), partition=part, offset=off,
                        consumer_id=CONSUMER_ID, board_id=board_id))
                    consumer.commit(message=msg, asynchronous=False)
                    counts["dlq"] += 1
                    log("dlq", reason="retry_exhausted", board_id=board_id,
                        partition=part, offset=off, total_attempts=total)
                    continue

                # Not yet exhausted: assume a real outage. Do NOT commit. Exit
                # non-zero so the restart policy takes over and replays this
                # board.
                log("fatal_api_unavailable", board_id=board_id,
                    attempts=attempts, total_attempts=total,
                    partition=part, offset=off, offset_committed=False)
                return 1

            # --------------------------------------------- route + emit -------
            r = route(payload, calib)
            result = build_result(
                board_id=board_id, verdict=r.verdict, score=r.score,
                statistic=r.statistic, reason=r.reason,
                fail_threshold=calib.fail_threshold,
                pass_threshold=calib.pass_threshold,
                api_response=payload, image_bytes_len=len(raw),
                t_produced_ms=t_produced, t_consume_ms=t_consume, t_api_ms=api_ms,
                consumer_id=CONSUMER_ID, partition=part, offset=off,
                attempts=attempts,
            )

            # results first (every board), then alerts (only what a human must
            # see), and only then the commit.
            produce_and_flush(T_RESULTS, board_id, result)
            if r.verdict in (FAIL, REVIEW):
                alert = dict(result)
                alert["severity"] = "critical" if r.verdict == FAIL else "review"
                produce_and_flush(T_ALERTS, board_id, alert)

            consumer.commit(message=msg, asynchronous=False)

            counts[r.verdict] += 1
            processed += 1
            log("routed", board_id=board_id, verdict=r.verdict,
                score=round(r.score, 6), statistic=r.statistic,
                api_ms=round(api_ms, 1), e2e_ms=result["timings_ms"]["e2e"],
                partition=part, offset=off)

            if LIMIT and processed >= LIMIT:
                log("limit_reached", limit=LIMIT)
                break
    finally:
        log("closing", processed=processed, counts=counts)
        try:
            producer.flush(10.0)
        finally:
            consumer.close()   # clean rebalance instead of a session timeout
    return 0


if __name__ == "__main__":
    sys.exit(main())