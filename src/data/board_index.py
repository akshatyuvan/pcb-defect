"""
Canonical board enumeration for DeepPCB.

One place that answers "what boards exist and where are their files", used by
the split builder, the calibration collector and the Day 8 boards-file writer.

It also guards the duplicate-dataset hazard: a nested PCBData/PCBData/ (which
happened once already) makes every board id appear twice under different
paths. Silently taking the last one would double Day 8's throughput
denominator, so we raise instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

TEST_SUFFIX = "_test.jpg"
TEMP_SUFFIX = "_temp.jpg"


@dataclass(frozen=True)
class Board:
    board_id: str
    test_path: Path                 # imaged board: 3-12 annotated defects, never clean
    temp_path: Optional[Path]       # paired reference: defect-free by construction


def index_boards(raw_root: Path) -> Dict[str, Board]:
    """Map board_id -> Board. Raises on duplicate ids (doubled dataset)."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"dataset root not found: {raw_root}")

    seen: Dict[str, List[Path]] = {}
    for tp in sorted(raw_root.rglob("*" + TEST_SUFFIX)):
        bid = tp.name[: -len(TEST_SUFFIX)]
        seen.setdefault(bid, []).append(tp)

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        sample = list(dupes.items())[:3]
        raise RuntimeError(
            f"{len(dupes)} board ids appear more than once under {raw_root}. "
            f"This is the nested-duplicate-dataset bug. Sample: {sample}"
        )

    boards: Dict[str, Board] = {}
    for bid, paths in seen.items():
        tp = paths[0]
        temp = tp.with_name(bid + TEMP_SUFFIX)
        boards[bid] = Board(bid, tp, temp if temp.exists() else None)
    return boards


def orphan_templates(raw_root: Path) -> List[Path]:
    """_temp.jpg files with no matching _test.jpg. DeepPCB ships exactly one."""
    raw_root = Path(raw_root)
    out: List[Path] = []
    for tmp in sorted(raw_root.rglob("*" + TEMP_SUFFIX)):
        bid = tmp.name[: -len(TEMP_SUFFIX)]
        if not tmp.with_name(bid + TEST_SUFFIX).exists():
            out.append(tmp)
    return out