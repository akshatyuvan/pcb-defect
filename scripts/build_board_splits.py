"""
Day 7, step 1. Reproduce the Day 1 board split exactly and freeze it to
artifacts/serving/board_splits.json.

This is a REPLAY, not a re-derivation. src/data/build_patches.py does:

    trainval    = read_index(raw / "trainval.txt", raw)   # 1000 boards, file order
    test_boards = read_index(raw / "test.txt",     raw)   #  500 boards, file order
    order = list(range(len(trainval)))
    random.Random(42).shuffle(order)
    n_val = int(len(order) * 0.15)                        # 150
    val   = [trainval[i] for i in order[:n_val]]
    train = [trainval[i] for i in order[n_val:]]

Every input is deterministic: DeepPCB's own index files supply the starting
order, and random.Random is the stdlib Mersenne Twister, which is reproducible
across machines and Python versions. We import the same read_index and run the
same four lines, so membership is identical by construction rather than
recovered and hoped about.

Why that distinction is worth the paragraph: a re-derivation with the WRONG rng
(np.random.default_rng, or sklearn's random_state) would still produce
850/150/500. The sizes would look right, the membership would be wrong, and
calibration would silently run on boards the model trained on. Nothing
downstream would ever raise.

Why it matters at all:
  - Calibration (step 7.2) must use VAL boards. Calibrating on training boards
    gives an optimistically low threshold you would not discover in testing.
  - Day 8's headline throughput must be measured on TEST boards, or the number
    describes a mixture you cannot name in an interview.

Runs on the MAC.
    python scripts/build_board_splits.py
    python scripts/build_board_splits.py --write
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.board_index import index_boards, orphan_templates  # noqa: E402
from src.data.deeppcb import read_index  # noqa: E402

# same defaults as build_patches.py's argparse
SEED = 42
VAL_FRAC = 0.15

RAW = ROOT / "data" / "raw" / "PCBData"
OUT_SPLITS = ROOT / "artifacts" / "serving" / "board_splits.json"
OUT_TEST_IDS = ROOT / "artifacts" / "serving" / "test_board_ids.json"
OUT_TEST_PATHS = ROOT / "artifacts" / "serving" / "test_board_paths.txt"

EXPECTED = {"train": 850, "val": 150, "test": 500}


def ids_from(index_path: Path) -> List[str]:
    """Board ids in file order, via Day 1's own reader.

    We import read_index rather than parsing the index file ourselves: if Day 1
    skipped blank lines, stripped a prefix, or reordered anything, parsing it
    again here would diverge in exactly the way this script exists to prevent.
    """
    return [b.board_id for b in read_index(index_path, RAW)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    boards = index_boards(RAW)
    print(f"boards on disk: {len(boards)}   orphan templates: {len(orphan_templates(RAW))}")

    trainval = ids_from(RAW / "trainval.txt")
    test = ids_from(RAW / "test.txt")
    print(f"trainval.txt: {len(trainval)} boards   test.txt: {len(test)} boards")

    # ---- the four lines from build_patches.py, verbatim -------------------
    order = list(range(len(trainval)))
    random.Random(SEED).shuffle(order)
    n_val = int(len(order) * VAL_FRAC)
    val = [trainval[i] for i in order[:n_val]]
    train = [trainval[i] for i in order[n_val:]]
    # -----------------------------------------------------------------------

    split = {"train": train, "val": val, "test": test}

    ok = True
    for k, n in EXPECTED.items():
        got = len(split[k])
        flag = "" if got == n else "   <-- MISMATCH"
        ok &= got == n
        print(f"  {k:<6s} {got:>4d}  (expected {n}){flag}")

    all_ids = train + val + test
    if len(set(all_ids)) != len(all_ids):
        print("FAIL: splits overlap")
        ok = False
    missing = [i for i in all_ids if i not in boards]
    if missing:
        print(f"FAIL: {len(missing)} split ids not on disk, e.g. {missing[:5]}")
        ok = False

    # Templates are the ONLY defect-free boards in DeepPCB, so they are the
    # negatives for step 7.2's board-level calibration. A val board without one
    # is a val board we cannot calibrate against.
    no_temp = [i for i in val if boards[i].temp_path is None]
    print(f"val boards missing a paired template: {len(no_temp)}  (must be 0)")
    ok &= not no_temp

    print(f"\nval sample (first 5): {val[:5]}")
    print(f"test sample (first 5): {test[:5]}")

    if not ok:
        print("\nchecks failed -- not writing")
        return 1
    if not args.write:
        print("\nDry run. Re-run with --write to save.")
        return 0

    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPLITS.write_text(json.dumps({
        "schema_version": 1,
        "method": "replay of build_patches.py: random.Random(42).shuffle over "
                  "trainval.txt index order, val_frac=0.15; test.txt used verbatim",
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "counts": {k: len(v) for k, v in split.items()},
        **{k: sorted(v) for k, v in split.items()},
    }, indent=2))

    # BARE JSON ARRAY, not an object. src/streaming/producer.py does
    #   wanted = set(json.loads(Path(boards_file).read_text()))
    # so an object would give a set of its KEYS, match zero boards, and the
    # producer would sys.exit -- looking like a dataset problem, not a format
    # problem. Match the consumer of this file, not our own taste.
    OUT_TEST_IDS.write_text(json.dumps(sorted(test), indent=0))

    OUT_TEST_PATHS.write_text(
        "\n".join(str(boards[i].test_path) for i in sorted(test)) + "\n")

    print(f"\nwrote {OUT_SPLITS.relative_to(ROOT)}")
    print(f"wrote {OUT_TEST_IDS.relative_to(ROOT)}")
    print(f"wrote {OUT_TEST_PATHS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())