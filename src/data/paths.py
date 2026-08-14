"""
Resolve a board id (as stored in the patch .npz files) to the actual image
paths on disk.

WHY THIS EXISTS AS A SEPARATE FILE
  Day 3 (template differencing) and Day 4 (board-level stitched map) both need
  the raw 640x640 board image, and Day 3 also needs the paired defect-free
  template. Both start from `board_ids` inside a .npz rather than from
  deeppcb.py's Board records, because the .npz is the exact set the CNN was
  scored on. This module is the bridge between the two.

  It deliberately does not care how deeppcb.py formats an id (bare stem,
  group/stem, with or without the _test suffix). It indexes the filesystem and
  matches on the filename stem, which is unique across DeepPCB.
"""
from __future__ import annotations

from pathlib import Path


def board_key(board_id) -> str:
    """Normalise anything that identifies a board down to its bare stem.

    '00041000', 'group00041/00041/00041000_test.jpg' and '00041000_test' all
    collapse to '00041000'. numpy string scalars are handled by the str() call.
    """
    s = str(board_id).strip()
    s = s.replace("\\", "/").split("/")[-1]          # drop any directory prefix
    for suffix in ("_test.jpg", "_temp.jpg", "_test", "_temp", ".jpg", ".txt"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def index_boards(raw_root: str | Path) -> dict[str, tuple[Path, Path]]:
    """Walk data/raw/PCBData once and build {stem: (test_path, temp_path)}.

    ~1500 entries, takes well under a second even on a Drive FUSE mount because
    it is a directory walk and not 1500 file opens. Boards whose template is
    missing are skipped rather than silently paired with the wrong file, which
    is why the return is a tuple and not two separate dicts.
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")

    index: dict[str, tuple[Path, Path]] = {}
    for test_path in raw_root.rglob("*_test.jpg"):
        stem = test_path.name[: -len("_test.jpg")]
        temp_path = test_path.with_name(stem + "_temp.jpg")
        if temp_path.exists():
            index[stem] = (test_path, temp_path)

    if not index:
        raise RuntimeError(
            f"found no *_test.jpg under {raw_root}. Is data/ symlinked to Drive? "
            "Remember: `ls -la data` describes the symlink, `ls -la data/` follows it."
        )
    return index


def resolve(board_ids, index: dict[str, tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """Map an array of npz board_ids to (test, temp) path pairs, in order.

    Fails loudly with sample keys from both sides, because a silent partial
    match here would quietly change the evaluation set and void the whole
    CNN-vs-classical comparison.
    """
    out, missing = [], []
    for bid in board_ids:
        key = board_key(bid)
        if key not in index:
            missing.append(key)
        else:
            out.append(index[key])
    if missing:
        sample_have = sorted(index)[:5]
        raise KeyError(
            f"{len(missing)} board ids did not resolve to files. "
            f"first few unresolved: {missing[:5]}. "
            f"first few keys available on disk: {sample_have}"
        )
    return out
