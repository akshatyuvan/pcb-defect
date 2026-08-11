"""
DeepPCB index + annotation parsing.

Layout shipped by the dataset:
  PCBData/group20085/20085/20085000_test.jpg   <- defective board
  PCBData/group20085/20085/20085000_temp.jpg   <- paired defect-free template (Day 3)
  PCBData/group20085/20085_not/20085000.txt    <- boxes for the _test image
  PCBData/trainval.txt   (1000 lines)
  PCBData/test.txt       (500 lines)

IMPORTANT, and not obvious from the dataset's own README:
Each index line is  <board_relpath> <annotation_relpath>  where board_relpath
carries NO _test suffix, e.g.

    group20085/20085/20085000.jpg group20085/20085_not/20085000.txt

That path does not exist on disk. It is an identifier, not a file. The two real
images are formed by appending _test / _temp to its stem. Verified empirically
against the shipped tree, because taking the line literally silently fails for
all 1500 boards while annotations continue to resolve, which makes the failure
look like a path-root problem when it is not.

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
    board_id: str          # e.g. "20085000"
    test_path: Path        # defective image
    temp_path: Path        # paired defect-free template
    ann_path: Path         # box annotations for test_path


def _image_pair(root: Path, rel: str) -> tuple[Path, Path]:
    """Turn an index board path into its (test, temp) image paths.

    Handles the suffix-less form the dataset actually ships, and tolerates a
    path that already carries _test in case a future copy of the index differs.
    Both images are built from one stem so they can never disagree about which
    board they belong to.
    """
    rel_path = Path(rel)
    stem = rel_path.stem
    for suffix in ("_test", "_temp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    d = root / rel_path.parent
    return d / f"{stem}_test.jpg", d / f"{stem}_temp.jpg"


def read_index(list_file: Path, root: Path) -> list[Board]:
    """Parse trainval.txt / test.txt into Board records.

    root is the PCBData directory.
    """
    boards: list[Board] = []
    for raw in Path(list_file).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed index line in {list_file}: {raw!r}")
        test_path, temp_path = _image_pair(root, parts[0])
        boards.append(
            Board(
                board_id=test_path.stem.replace("_test", ""),
                test_path=test_path,
                temp_path=temp_path,
                ann_path=root / parts[1],
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
    """Integrity check, run before spending ten minutes tiling.

    Reports each path kind separately. A single missing-count tells you something
    broke; knowing it was the image and not the annotation tells you where to
    look, which is the difference between a two-minute fix and an hour.
    """
    trainval = read_index(root / "trainval.txt", root)
    test = read_index(root / "test.txt", root)
    all_boards = trainval + test

    miss_test = [b.board_id for b in all_boards if not b.test_path.exists()]
    miss_temp = [b.board_id for b in all_boards if not b.temp_path.exists()]
    miss_ann = [b.board_id for b in all_boards if not b.ann_path.exists()]
    n_boxes = [len(read_boxes(b.ann_path)) for b in trainval[:50]]

    return {
        "trainval_boards": len(trainval),
        "test_boards": len(test),
        "n_missing_test_images": len(miss_test),
        "n_missing_templates": len(miss_temp),
        "n_missing_annotations": len(miss_ann),
        "example_missing_test": miss_test[:3],
        "example_missing_template": miss_temp[:3],
        "example_missing_annotation": miss_ann[:3],
        "example_resolved_test_path": str(all_boards[0].test_path),
        "example_resolved_temp_path": str(all_boards[0].temp_path),
        "boxes_per_board_sample_min": min(n_boxes),
        "boxes_per_board_sample_max": max(n_boxes),
    }