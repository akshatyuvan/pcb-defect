"""
The pcb.results and pcb.dlq wire contracts.

Design rule: results carry NO image bytes. A board envelope on pcb.boards is
~34 KiB and essentially all of that is the base64 JPEG; a result is a few
hundred bytes. Keeping them separate means the results topic can be retained
far longer than boards for the same disk, and Day 8's benchmark consumer can
drain results at negligible cost while measuring the pipeline it is measuring.

Timings are recorded here rather than computed later because the only place
that knows when the board was produced, when we picked it up and when we
finished is this consumer. t_produced_ms comes from the Kafka record timestamp,
which the broker takes from the producer -- so end-to-end latency spans the
actual queue, not just from when we happened to poll.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

RESULT_SCHEMA_VERSION = 1
DLQ_SCHEMA_VERSION = 1


def now_ms() -> float:
    return time.time() * 1000.0


def build_result(
    *,
    board_id: str,
    verdict: str,
    score: float,
    statistic: str,
    reason: str,
    fail_threshold: float,
    pass_threshold: Optional[float],
    api_response: Dict[str, Any],
    image_bytes_len: int,
    t_produced_ms: Optional[float],
    t_consume_ms: float,
    t_api_ms: float,
    consumer_id: str,
    partition: int,
    offset: int,
    attempts: int,
) -> Dict[str, Any]:
    t_result_ms = now_ms()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "board_id": board_id,
        "verdict": verdict,
        "score": score,
        "statistic": statistic,
        "reason": reason,
        "thresholds": {"pass": pass_threshold, "fail": fail_threshold},
        # a compact echo of the model's own view, for error analysis later.
        # api_verdict is the SERVICE's patch-threshold verdict; `verdict` above
        # is ours from the board calibration. They will usually disagree, and
        # that disagreement is the whole point of Day 7.
        "n_flagged": api_response.get("n_flagged"),
        "class_counts": api_response.get("class_counts"),
        "api_verdict": api_response.get("verdict"),
        "image_bytes": image_bytes_len,
        "timings_ms": {
            "produced": t_produced_ms,
            "consume_start": t_consume_ms,
            "api": round(t_api_ms, 2),
            "result": t_result_ms,
            "e2e": (None if t_produced_ms is None
                    else round(t_result_ms - t_produced_ms, 2)),
            "queue_wait": (None if t_produced_ms is None
                           else round(t_consume_ms - t_produced_ms, 2)),
        },
        "source": {"partition": partition, "offset": offset, "attempts": attempts},
        "consumer_id": consumer_id,
    }


def build_dlq_record(
    *,
    reason: str,
    detail: str,
    raw_value: bytes,
    partition: int,
    offset: int,
    consumer_id: str,
    board_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The DLQ keeps the ORIGINAL payload verbatim, base64 blob and all.

    That was the point of locked decision 11: a dead-lettered board must be
    hand-inspectable and, once the cause is fixed, replayable byte-for-byte. A
    DLQ storing only an error string tells you a board died and nothing about
    which one or why."""
    return {
        "schema_version": DLQ_SCHEMA_VERSION,
        "reason": reason,             # malformed | rejected_400 | rejected_422
        "detail": detail[:2000],
        "board_id": board_id,
        "source": {"partition": partition, "offset": offset},
        "consumer_id": consumer_id,
        "failed_at_ms": now_ms(),
        "raw": raw_value.decode("utf-8", "replace")[:400_000],
    }