"""Tests for the Kafka message contract.

The sha256 in the envelope is the ONLY integrity check between producer and
consumer. If it stops firing, a truncated payload reaches the model and comes
back with a confident, meaningless verdict.

NOTE ON pytest.raises: these deliberately name specific exception types rather
than bare Exception. An earlier draft of this file passed str where from_json
wants bytes; pytest.raises(Exception) caught the resulting AttributeError and
the test went GREEN while testing nothing at all. A test that passes for the
wrong reason is worse than no test.
"""

import base64
import json

import pytest

from src.streaming import schema

RAW = b"\xff\xd8\xff\xe0" + b"pretend jpeg payload" * 8


def _msg():
    return schema.build_message(
        board_id="00041000",
        group="00041",
        raw=RAW,
        encoding="jpeg",
        seq=0,
        source="pytest",
    )


def _as_wire(envelope: dict) -> bytes:
    """from_json takes BYTES, matching what confluent-kafka hands a consumer
    (msg.value()). Encoding here rather than accepting str keeps the test on
    the same code path production uses."""
    return json.dumps(envelope).encode("utf-8")


def _swap_longest_string(obj, new_value):
    """Find and replace the longest string in a nested dict - that is the b64
    payload. Located structurally rather than by key name so this test does not
    break when the envelope gains a field."""
    best = {"holder": None, "key": None, "len": -1}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and len(v) > best["len"]:
                    best.update(holder=node, key=k, len=len(v))
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    assert best["holder"] is not None, "no string payload found in envelope"
    best["holder"][best["key"]] = new_value
    return obj


def test_to_json_produces_bytes():
    """Pinned because the whole file depends on it: the producer hands to_json's
    output straight to confluent-kafka, which requires bytes."""
    assert isinstance(_msg().to_json(), bytes)


def test_round_trip_preserves_bytes_exactly():
    back = schema.BoardMessage.from_json(_msg().to_json())
    assert back.board_id == "00041000"
    # Locked decision: original JPEG bytes are forwarded, never re-encoded.
    # A re-encode would change the pixels the model sees vs the ones on disk.
    assert back.image_bytes() == RAW


def test_image_bytes_rejects_a_tampered_payload():
    """The sha256 must be checked at image_bytes(), not at parse time - a
    tampered envelope is still structurally valid JSON. This is the guard that
    stops a corrupted board reaching the model and scoring confidently."""
    envelope = json.loads(_msg().to_json().decode("utf-8"))
    tampered = base64.b64encode(RAW + b"extra").decode()
    envelope = _swap_longest_string(envelope, tampered)

    msg = schema.BoardMessage.from_json(_as_wire(envelope))
    with pytest.raises(ValueError):
        msg.image_bytes()


def test_from_json_rejects_an_unknown_schema_version():
    """This is what the malformed poison message exercised on Day 7: an
    envelope with schema_version 99 must be rejected at parse time and DLQ'd,
    not half-processed against fields that may have moved."""
    envelope = json.loads(_msg().to_json().decode("utf-8"))
    envelope["schema_version"] = 99
    with pytest.raises(ValueError):
        schema.BoardMessage.from_json(_as_wire(envelope))


def test_schema_version_is_pinned():
    """If this bumps, docs/message_format.md and every consumer's version check
    must move together. Failing here is the reminder."""
    assert schema.SCHEMA_VERSION == 1


def test_topic_names_are_stable():
    """Topic names appear in scripts/create_topics.sh, the compose file and
    three consumers. Pinning them here means a rename breaks the test suite
    rather than producing a consumer that silently subscribes to nothing."""
    assert schema.TOPIC_BOARDS == "pcb.boards"
    assert schema.TOPIC_RESULTS == "pcb.results"
    assert schema.TOPIC_ALERTS == "pcb.alerts"
    assert schema.TOPIC_DLQ == "pcb.dlq"