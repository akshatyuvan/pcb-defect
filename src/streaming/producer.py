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


def discover_boards(boards_file):
    paths = sorted(RAW.rglob("*_test.jpg"))
    if not paths:
        sys.exit(f"no *_test.jpg under {RAW} - run scripts/get_data.sh (Day 5 step 5.3)")
    if boards_file:
        wanted = set(json.loads(Path(boards_file).read_text()))
        paths = [p for p in paths if p.stem.replace("_test", "") in wanted]
        if not paths:
            sys.exit(f"{boards_file} matched no boards on disk")
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
    args = ap.parse_args()

    paths = discover_boards(args.boards_file)
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