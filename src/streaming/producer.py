"""Simulated QC camera: streams DeepPCB boards onto pcb.boards.

Runs on: Mac, in the `pcb` conda env, against localhost:29092.
Deliberately NOT containerised - it stands in for a physical camera, which
would be outside the stack too. It also keeps a container's worth of RAM free.

    python -m src.streaming.producer --limit 50 --rate 5

Key flags:
    --rate N     boards/sec (0 = as fast as possible; use this on Day 8)
    --limit N    stop after N boards
    --loop       cycle the dataset forever (sustained-throughput runs)
    --boards-file path/to/test_board_ids.json  restrict to the test split
    --include-templates   also stream the paired _temp.jpg files

On --include-templates: every DeepPCB _test.jpg carries 3-12 defects, so a
stream of them alone routes to `fail` 100% of the time. The paired _temp.jpg
templates are defect-free by construction (they are what Day 3's classical
baseline differenced against, and what Day 7 calibrated the board thresholds
on), so mixing them in produces a stream with both verdicts. That matters for
Day 8: a benchmark where every board takes the same branch measures one code
path, and an alerting consumer watching a metric pinned at 100% is watching
nothing. Template board ids keep their `_temp` suffix so the two streams stay
distinguishable downstream and do not collide on the Kafka key.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

from confluent_kafka import Producer

from src.streaming.schema import TOPIC_BOARDS, build_message

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "PCBData"


def discover_boards(boards_file, include_templates=False):
    """Board image paths to stream, optionally including defect-free templates.

    Returns _test.jpg paths first, then _temp.jpg paths if requested. The
    caller shuffles, so ordering here only matters for reproducibility.
    """
    paths = sorted(RAW.rglob("*_test.jpg"))
    if not paths:
        sys.exit(f"no *_test.jpg under {RAW} - run scripts/get_data.sh (Day 5 step 5.3)")
    if boards_file:
        # Explicit type check. set(json.loads(...)) on a JSON OBJECT iterates
        # over its KEYS, which is silently wrong: {"boards": [...]} yields
        # {"boards"} and {"00041000": ...} yields a plausible-but-arbitrary
        # board set. The first case happens to die below at "matched no boards
        # on disk"; the second would not. Reject the shape here so the failure
        # names the actual problem instead of pointing at the dataset.
        loaded = json.loads(Path(boards_file).read_text())
        if not isinstance(loaded, list):
            sys.exit(
                f"{boards_file} must be a bare JSON array of board ids, "
                f"got {type(loaded).__name__}"
            )
        wanted = set(loaded)
        paths = [p for p in paths if p.stem.replace("_test", "") in wanted]
        if not paths:
            sys.exit(f"{boards_file} matched no boards on disk")

    if include_templates:
        # Only templates PAIRED with a board already in `paths`. Globbing
        # *_temp.jpg independently would pull in the known orphan template
        # (one _temp.jpg has no matching _test.jpg) and, when --boards-file is
        # set, would ignore the split restriction entirely -- silently mixing
        # training-split templates into a test-split benchmark.
        temps = [p.with_name(p.stem.replace("_test", "_temp") + p.suffix)
                 for p in paths]
        temps = [t for t in temps if t.exists()]
        paths = paths + temps

    return paths


def make_producer(bootstrap):
    return Producer({
        "bootstrap.servers": bootstrap,
        "client.id": "qc-camera-sim-01",
        # acks=all + idempotence: the broker confirms the write is durable
        # before the delivery callback fires. Without it a broker restart can
        # lose boards that the camera believes it sent, and a QC line that
        # silently drops boards is worse than one that stops.
        "acks": "all",
        "enable.idempotence": True,
        # gzip: the payload is base64 of already-compressed JPEG. gzip recovers
        # most of base64's 33% inflation (base64 uses 6 bits per byte, so the
        # redundancy is real even though the underlying JPEG is incompressible).
        # Day 8 measures the actual saving.
        "compression.type": "gzip",
        # 50ms linger lets small messages batch. At --rate 5 this is invisible;
        # at --rate 0 it is a meaningful throughput win.
        "linger.ms": 50,
        "batch.size": 262144,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--topic", default=TOPIC_BOARDS)
    ap.add_argument("--rate", type=float, default=2.0, help="boards/sec; 0 = unthrottled")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--boards-file", default=None)
    ap.add_argument("--include-templates", action="store_true",
                    help="also stream paired _temp.jpg (defect-free) boards, "
                         "so the stream contains both pass and fail verdicts")
    args = ap.parse_args()

    paths = discover_boards(args.boards_file, args.include_templates)
    if args.shuffle:
        random.seed(42)
        random.shuffle(paths)
    print(f"{len(paths)} boards available -> topic {args.topic} @ {args.bootstrap}")

    p = make_producer(args.bootstrap)
    stats = {"ok": 0, "err": 0, "bytes": 0}

    def on_delivery(err, msg):
        # Fired from poll(); this is where "the broker actually took it" is
        # confirmed. produce() returning is NOT delivery - it only enqueues.
        if err is not None:
            stats["err"] += 1
            print(f"  DELIVERY FAILED {err}", file=sys.stderr)
        else:
            stats["ok"] += 1
            stats["bytes"] += len(msg.value())

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    t_start = time.perf_counter()
    seq = 0
    remaining = 0
    try:
        while True:
            for path in paths:
                if args.limit and seq >= args.limit:
                    raise StopIteration
                board_id = path.stem.replace("_test", "")
                group = path.parent.parent.name
                raw = path.read_bytes()
                encoding = path.suffix.lstrip(".").replace("jpg", "jpeg")
                msg = build_message(board_id, group, raw, encoding, seq)
                p.produce(
                    args.topic,
                    key=board_id.encode(),
                    value=msg.to_json(),
                    headers=[
                        ("schema_version", b"1"),
                        ("content_type", b"application/json"),
                        ("image_encoding", msg.image.encoding.encode()),
                        ("board_id", board_id.encode()),
                        ("trace_id", str(uuid.uuid4()).encode()),
                    ],
                    on_delivery=on_delivery,
                )
                # poll(0) services delivery callbacks without blocking. Skip it
                # and the internal queue fills and produce() starts raising
                # BufferError - the classic first confluent-kafka bug.
                p.poll(0)
                seq += 1
                if seq % 10 == 0:
                    print(f"  queued {seq} (delivered {stats['ok']}, failed {stats['err']})")
                if interval:
                    time.sleep(interval)
            if not args.loop:
                break
    except (StopIteration, KeyboardInterrupt):
        pass
    finally:
        remaining = p.flush(30)
        dt = time.perf_counter() - t_start

    print("")
    print("--- producer summary ---")
    print(f"  queued      : {seq}")
    print(f"  delivered   : {stats['ok']}")
    print(f"  failed      : {stats['err']}")
    print(f"  undelivered : {remaining}")
    print(f"  wall time   : {dt:.2f}s")
    if stats["ok"]:
        mean_kib = stats["bytes"] / stats["ok"] / 1024
        print(f"  mean msg    : {mean_kib:.1f} KiB (pre-compression)")
        print(f"  rate        : {stats['ok'] / dt:.2f} boards/s")
    if stats["err"] or remaining:
        sys.exit(1)


if __name__ == "__main__":
    main()