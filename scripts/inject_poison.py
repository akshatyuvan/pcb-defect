"""
Day 7. Deliberately produce bad messages to prove the DLQ path works.

Three failure modes, because they take three different code paths and only two
of them are interesting:

  --kind malformed   unknown schema_version / unparseable envelope
                     -> BoardMessage.from_json raises  -> DLQ "malformed"
  --kind corrupt     valid envelope, image bytes that are not a decodable
                     image. sha256 still matches, so this gets PAST the
                     integrity check and reaches the API
                     -> API 400                        -> DLQ "rejected_400"
  --kind wrongsize   valid envelope, a real decodable JPEG at the wrong
                     dimensions
                     -> API 422                        -> DLQ "rejected_422"

corrupt and wrongsize are the ones that matter: they prove the consumer
distinguishes "this message is broken" (commit + DLQ) from "we are broken"
(retry, never commit). A message failing sha256 never reaches that decision.

Runs on the MAC, so the bootstrap default is the EXTERNAL listener 29092.
    python scripts/inject_poison.py --kind corrupt
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from confluent_kafka import Producer
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.streaming.schema import TOPIC_BOARDS, build_message  # noqa: E402


def make_wrong_size() -> bytes:
    """A perfectly valid JPEG at the wrong dimensions. Decodes fine, then fails
    the shape check -- exactly the 422 case."""
    buf = io.BytesIO()
    Image.new("L", (320, 320), color=200).save(buf, format="JPEG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--topic", default=TOPIC_BOARDS)
    ap.add_argument("--kind", required=True,
                    choices=["malformed", "corrupt", "wrongsize"])
    args = ap.parse_args()

    p = Producer({"bootstrap.servers": args.bootstrap, "acks": "all"})
    board_id = f"POISON-{args.kind.upper()}"

    if args.kind == "malformed":
        # a schema_version the consumer must refuse rather than guess at
        value = json.dumps({"schema_version": 99, "board_id": board_id}).encode()
    else:
        raw = (b"\xff\xd8\xff" + b"this is not a pcb board" * 40
               if args.kind == "corrupt" else make_wrong_size())
        # build_message computes sha256 itself, so integrity PASSES and the
        # bytes reach the API. That is the whole point of these two cases.
        msg = build_message(board_id=board_id, group="poison", raw=raw,
                            encoding="jpeg", seq=0)
        value = msg.to_json()

    p.produce(args.topic, key=board_id.encode(), value=value)
    remaining = p.flush(20)
    print(f"produced kind={args.kind} board_id={board_id} "
          f"bytes={len(value)} undelivered={remaining}")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())