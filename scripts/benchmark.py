"""
Day 8. Measure the shipped system, not a component of it.

Consumes pcb.results in a throwaway group (so it never disturbs the real
consumer's offsets) and computes:

  e2e         result written - board produced. t_produced comes from the Kafka
              record timestamp, set by the producer, so this spans the queue.
  queue_wait  pickup - produced. This is what grows under load and what extra
              consumers reduce. API latency does not.
  api         the model's own share, measured inside the consumer.
  throughput  boards/s AND KiB/s, reported separately, because DeepPCB message
              sizes vary with copper density (measured 27-45 KiB on the test
              split) and the two do not track each other.

Run this BEFORE the producer, in its own terminal. It exits when --expect
results have arrived or --timeout elapses.

Runs on the MAC, so the bootstrap default is the EXTERNAL listener 29092.
    python scripts/benchmark.py --expect 200 --label c1 --consumers 1
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from confluent_kafka import Consumer

ROOT = Path(__file__).resolve().parents[1]


def pct(values: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile. With n>=100 the interpolation choice is noise,
    and nearest-rank is the one you can explain without a reference."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def summarise(name: str, xs: List[float]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"metric": name, "n": 0}
    return {"metric": name, "n": len(xs),
            "min": round(min(xs), 2), "p50": round(pct(xs, 50), 2),
            "p95": round(pct(xs, 95), 2), "p99": round(pct(xs, 99), 2),
            "max": round(max(xs), 2), "mean": round(mean(xs), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--topic", default="pcb.results")
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--label", required=True, help="e.g. c1, c3, c4")
    ap.add_argument("--consumers", type=int, required=True)
    args = ap.parse_args()

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        # fresh group per run: reads only what arrives from now on, and commits
        # nothing the real pipeline cares about
        "group.id": f"bench-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([args.topic])
    print(f"listening on {args.topic} @ {args.bootstrap}, "
          f"waiting for {args.expect} results.")
    print("Start the producer in another terminal now.")

    docs: List[Dict[str, Any]] = []
    first_ms = last_ms = None
    deadline = time.monotonic() + args.timeout
    while len(docs) < args.expect and time.monotonic() < deadline:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        d = json.loads(msg.value())
        docs.append(d)
        t = d["timings_ms"]["result"]
        first_ms = t if first_ms is None else min(first_ms, t)
        last_ms = t if last_ms is None else max(last_ms, t)
        if len(docs) % 25 == 0:
            print(f"  {len(docs)}/{args.expect}")
    consumer.close()

    if not docs:
        print("no results received -- is the stack up and the producer running?")
        return 1
    if len(docs) < args.expect:
        print(f"WARNING: timed out with {len(docs)}/{args.expect}. "
              "Numbers below describe what arrived, not the full run.")

    e2e = [d["timings_ms"]["e2e"] for d in docs]
    qw = [d["timings_ms"]["queue_wait"] for d in docs]
    api = [d["timings_ms"]["api"] for d in docs]
    total_bytes = sum(d.get("image_bytes", 0) for d in docs)
    # span from first to last RESULT: the pipeline's own output rate, not the
    # producer's input rate. Under saturation these differ, which is the point.
    span_s = max((last_ms - first_ms) / 1000.0, 1e-9)

    verdicts: Dict[str, int] = {}
    workers: Dict[str, int] = {}
    parts: Dict[str, int] = {}
    for d in docs:
        verdicts[d["verdict"]] = verdicts.get(d["verdict"], 0) + 1
        workers[d["consumer_id"]] = workers.get(d["consumer_id"], 0) + 1
        p = str(d["source"]["partition"])
        parts[p] = parts.get(p, 0) + 1

    report = {
        "label": args.label,
        "consumers_configured": args.consumers,
        "n_results": len(docs),
        "wall_s": round(span_s, 3),
        "throughput_boards_per_s": round(len(docs) / span_s, 3),
        "throughput_KiB_per_s": round(total_bytes / 1024.0 / span_s, 1),
        "mean_message_KiB": round(total_bytes / 1024.0 / len(docs), 1),
        "latency": [summarise("e2e_ms", e2e), summarise("queue_wait_ms", qw),
                    summarise("api_ms", api)],
        "verdicts": verdicts,
        "work_per_consumer": workers,
        "results_per_partition": parts,
    }
    print("\n" + json.dumps(report, indent=2))

    out = ROOT / "artifacts" / f"day8_bench_{args.label}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())