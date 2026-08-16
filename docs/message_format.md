# Kafka message contract — `pcb.boards`

Version 1. Frozen on Day 6. Any breaking change bumps `schema_version` and the
consumer routes unknown versions to `pcb.dlq` rather than guessing.

## Envelope

`value` is UTF-8 JSON:

```json
{
  "schema_version": 1,
  "board_id": "90100009",
  "group": "group90100",
  "captured_at": "2026-08-15T09:41:02.481390Z",
  "source": "qc-camera-sim-01",
  "seq": 17,
  "image": {
    "encoding": "jpeg",
    "colour": "grayscale",
    "height": 640,
    "width": 640,
    "sha256": "9f2c...",
    "b64": "<base64 of the raw image file bytes>"
  }
}
```

`key` is the UTF-8 `board_id`.

## Kafka headers

Duplicated deliberately so a consumer, a DLQ inspector, or a monitoring tool can
filter and trace **without deserialising a ~60KB base64 payload**:

| header | value |
|---|---|
| `schema_version` | `"1"` |
| `content_type` | `"application/json"` |
| `image_encoding` | `"jpeg"` |
| `board_id` | e.g. `"90100009"` |
| `tra_id` | uuid4, one per message, propagated into the API call and every log line |

## Decisions and their costs

**Key = `board_id`.** Kafka guarantees ordering within a partition, and the
default partitioner hashes the key — so every message for a given board always
lands on the same partition and is processed in order. With 1500 distinct boards
over 3 partitions the load spreads evenly. Cost: a single very hot board_id
would create a hot partition. Not a risk here; would be in a real line where one
panel is re-imaged repeatedly.

**Image is base64 inside the JSON, not raw bytes in `value` with metadata in
headers.** Base64 inflates the payload by ~33%, and raw-bytes-plus-headers is
the more efficient design. Chosen against anyway, because:
  * `kafka-console-consumer` output stays human-readable, which matters a lot
    when debugging Day 7's dead-letter routing
  * a DLQ message can be inspected, fixed, and replayed by hand
  * `gzip` compression on the producer recovers most of the overhead, because
  base64 of already-compressed JPEG data still has only 6 bits of entropy per
    byte — Day 8 measures the actual on-wire size
This is a debuggability-for-bandwidth trade, made knowingly at demo scale. A
production line at thousands of boards/sec should invert it, or move to a schema
registry with Avro/Protobuf so the contract is enforced rather than documented.

**Original file bytes, not re-encoded.** DeepPCB ships boards as JPEG. The
producer sends those exact bytes rather than decoding and re-encoding to PNG, so
the pixels the model sees at serve time are bit-identical to the pixels it was
trained on. `scripts/verify_decode_parity.py` (Day 5) confirmed Pillow and
OpenCV decode them identically, which is what makes that guarantee real rather
than assumed. `encoding: "png"` is also accepted by the API.

**`sha256` of the image bytes.** Lets a consumer prove it processed the exact
board the producer sent, and makes duplicate delivery visible. Kafka is
at-least-once by default; with manual commit after infence (Day 7), a crash
between inference and commit causes a redelivery. The hash makes that
observable instead of invisible.

**Max message size.** A 640x640 grayscale DeepPCB JPEG is ~30-60KB; base64 takes
it to ~40-80KB. Kafka's default `max.message.bytes` is 1MB, so no broker or
producer tuning is needed. Recorded here so nobody "fixes" a limit that was
never hit.

**Measured on Day 6.** Mean message 34.0 KiB pre-compression across 20 boards,
against source JPEGs of 13-26 KiB. Sizes vary widely because JPEG compression
depends on copper density, so bytes/sec and boards/sec will not track linearly
in Day 8's benchmark. Partition spread over 20 messages was 7/6/7 - the default
murmur2 hash on `board_id` distributes evenly, which is what makes the 1-vs-3
consumer scaling test meaningful.

**Kafka 4.0 tooling note.** The old `kafka.tools.*` classes (`GetOffsetShell`,
`DefaultMessageFormatter`) were removed. Use the `bin/*.sh` wrappers
(`kafka-get-offsets.sh`) rather than `kafka-run-class.sh`.