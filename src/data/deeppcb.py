"""
DeepPCB index + annotation parsing.

Layout shipped by the dataset:
  PCBData/group00041/00041/00041000_test.jpg   <- defective board
  PCBData/group00041/00041/00041000_temp.jpg   <- paired defect-free template (Day 3)
  PCBData/group00041/00041_not/00041000.txt    <- boxes for the _test image
  PCBData/trainval.txt   (1000 lines)
  PCBData/test.txt       (500 lines)

Each index line is:  <test_image_relpath> <annotation_relpath>
Each annotation line is:  x1 y1 x2 y2 class_id
Coordinates are absolute pixels in the 640x640 test image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Locked decision 3. Index 0 is reserved for "good", which is NOT an annotation
# id: it is the *absence* of any overlapping box. Annotation ids run 1..6 in this
# order. This ordering is the single most load-bearing constant in the project.
CLASSES = ["good", "open", "short", "mousebite", "spur", "copper", "pinhole"]
ID_TO_INDEX = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}  # identity, but explicit beats implicit


@dataclass(frozen=True)
class Board:
    board_id: str          # e.g. "00041000"
    test_path: Path        # defective image
    temp_path: Path        # paired defect-free template
    ann_path: Path         # box annotations for test_path


def read_index(list_file: Path, root: Path) -> list[Board]:
    """Parse trainval.txt / test.txt into Board records.

    root is the PCBData directory. We resolve the template path by string
    substitution on the test path rather than globbing, because the pairing is
    guaranteed by the dataset's naming convention and globbing 1500 directories
    on a Drive mount is slow.
    """
    boards: list[Board] = []
    for raw in Path(list_file).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed index line in {list_file}: {raw!r}")
        img_rel, ann_rel = parts[0], parts[1]
        test_path = root / img_rel
        temp_path = Path(str(test_path).replace("_test.jpg", "_temp.jpg"))
        ann_path = root / ann_rel
        boards.append(
            Board(
                board_id=test_path.stem.replace("_test", ""),
                test_path=test_path,
                temp_path=temp_path,
                ann_path=ann_path,
            )
        )
    return boards


def read_boxes(ann_path: Path) -> list[tuple[int, int, int, int, int]]:
    """Return [(x1, y1, x2, y2, class_index), ...] with class_index in 1..6.

    The class id is validated loudly. A silent off-by-one here would poison every
    downstream number in the project and would not show up as a crash, only as a
    confusion matrix that looks slightly wrong eight days from now.
    """
    boxes = []
    for raw in Path(ann_path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        v = line.split()
        x1, y1, x2, y2 = (int(float(t)) for t in v[:4])
        cid = int(float(v[4]))
        if cid not in ID_TO_INDEX:
            raise ValueError(f"Unexpected class id {cid} in {ann_path}")
        # Normalise ordering defensively; some public annotation sets ship x2<x1.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        boxes.append((x1, y1, x2, y2, ID_TO_INDEX[cid]))
    return boxes


def verify_dataset(root: Path) -> dict:
    """Cheap integrity check, run before spending ten minutes tiling."""
    trainval = read_index(root / "trainval.txt", root)
    test = read_index(root / "test.txt", root)
    missing = [b.board_id for b in trainval + test
               if not (b.test_path.exists() and b.temp_path.exists() and b.ann_path.exists())]
    n_boxes = [len(read_boxes(b.ann_path)) for b in trainval[:50]]
    return {
        "trainval_boards": len(trainval),
        "test_boards": len(test),
        "missing_files": missing[:10],
        "n_missing": len(missing),
        "boxes_per_board_sample_min": min(n_boxes),
        "boxes_per_board_sample_max": max(n_boxes),
    }