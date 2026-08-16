"""Single definition of the pcb.boards envelope. Producer (Day 6) and consumer
(Day 7) both import from here so the contract cannot drift between them.

Full rationale in docs/message_format.md.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1
TOPIC_BOARDS = "pcb.boards"
TOPIC_RESULTS = "pcb.results"
TOPIC_ALERTS = "pcb.alerts"
TOPIC_DLQ = "pcb.dlq"


@dataclass
class ImagePayload:
    encoding: str          # "jpeg" | "png"
    colour: str            # always "grayscale" for DeepPCB
    height: int
    width: int
    sha256: str
    b64: str


@dataclass
class BoardMessage:
    schema_version: int
    board_id: str
    group: str
    captured_at: str
    source: str
    seq: int
    image: ImagePayload

    def to_json(self) -> bytes:
        # separators=(",",":") — no cosmetic whitespace on the wire. Sll, but
        # it is free, and at 1500 messages it is a real number in the benchmark.
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_json(raw: bytes) -> "BoardMessage":
        d: dict[str, Any] = json.loads(raw.decode("utf-8"))
        if d.get("schema_version") != SCHEMA_VERSION:
            # Explicit, so Day 7's consumer can catch this and DLQ it rather
            # than half-parsing a future format.
            raise ValueError(f"unsupported schema_version {d.get('schema_version')!r}")
        d["image"] = ImagePayload(**d["image"])
        return BoardMessage(**d)

    def image_bytes(self) -> bytes:
        raw = base64.b64decode(self.image.b64, validate=True)
        got = hashlib.sha256(raw).hexdigest()
        if got != self.image.sha256:
            raise ValueError(f"image sha256 mismatch: sent {self.image.sha256}, got {got}")
        return raw


def build_message(board_id: str, group: str, raw: bytes, encoding: str,
                  seq: int, source: str = "qc-camera-sim-01",
                  height: int = 640, width: int = 640) -> BoardMessage:
    return BoardMessage(
        schema_version=SCHEMA_VERSION,
        board_id=board_id,
        group=group,
        captured_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source=source,
        seq=seq,
        image=ImagePayload(
            encoding=encoding, colour="grayscale", height=height, width=width,
            sha256=hashlib.sha256(raw).hexdigest(),
            b64=base64.b64encode(raw).decode("ascii"),
        ),
    )
