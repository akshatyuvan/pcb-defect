"""
Day 7, step 2. Score every val board through the real HTTP service and dump the
raw responses, so calibration is a pure offline analysis over a file we can
re-analyse without re-running inference.

Two classes of board:
  defective : <id>_test.jpg   -- DeepPCB guarantees 3-12 defects, never clean
  clean     : <id>_temp.jpg   -- the paired reference, defect-free by construction

The templates are the ONLY defect-free boards in this dataset. Without them a
board-level threshold sweep has no negatives and is meaningless.

Known caveat, recorded in the output file and carried into the calibration:
templates are reference captures. They may be systematically cleaner than a
real clean board off the line, so measured specificity is an upper bound.

Runs on the MAC, against a running API (host uvicorn or the container).
    python scripts/collect_board_scores.py --split val --limit 5
    python scripts/collect_board_scores.py --split val
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.board_index import index_boards  # noqa: E402
from src.streaming.api_client import ApiClientRejection, PcbApiClient  # noqa: E402

RAW = ROOT / "data" / "raw" / "PCBData"
SPLITS = ROOT / "artifacts" / "serving" / "board_splits.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all boards in the split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or ROOT / "artifacts" / f"day7_board_scores_{args.split}.jsonl"
    ids = sorted(json.loads(SPLITS.read_text())[args.split])
    if args.limit:
        ids = ids[: args.limit]

    boards = index_boards(RAW)
    client = PcbApiClient(args.api, timeout=60.0)
    health = client.wait_for_health(60.0)
    print(f"api healthy: status={health.get('status')} device={health.get('device')} "
          f"fail_threshold={health.get('fail_threshold')}")
    print(f"image field resolved to: {client.image_field()!r}")

    jobs = []
    for bid in ids:
        b = boards[bid]
        jobs.append((bid, "defective", b.test_path))
        if b.temp_path is not None:
            jobs.append((bid, "clean", b.temp_path))

    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_rej = 0
    t0 = time.time()
    with out.open("w") as fh:
        for i, (bid, label, path) in enumerate(jobs, 1):
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            t = time.perf_counter()
            try:
                resp = client.predict_board(b64, extra={"board_id": bid})
            except ApiClientRejection as e:
                n_rej += 1
                print(f"  REJECTED {path.name}: {e}")
                continue
            dt = (time.perf_counter() - t) * 1000.0
            fh.write(json.dumps({
                "board_id": bid,
                "label": label,               # ground truth at BOARD level
                "file": path.name,
                "bytes": len(raw),
                "latency_ms": round(dt, 2),
                "response": resp,
            }) + "\n")
            n_ok += 1
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ok={n_ok} rejected={n_rej}  "
                      f"{(time.time()-t0):.1f}s elapsed")

    print(f"\nwrote {out.relative_to(ROOT)}  ({n_ok} rows, {n_rej} rejected)")
    return 0 if n_rej == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())