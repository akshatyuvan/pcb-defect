"""
Derive a patch-level classification task from DeepPCB's detection annotations.

WHY THIS EXISTS (locked decision 2)
DeepPCB has no defect-free *test* images. Every tested board carries 3-12
annotated defects. A whole-image "defective vs clean" classifier would therefore
be a perfectly balanced, perfectly trivial task and would prove nothing. Tiling
each 640x640 board into a 10x10 grid of 64x64 patches produces a genuinely
imbalanced 7-class problem where the imbalance comes from the physics of the data
(defects are small and sparse), not from a sampling choice I made.

LABELLING RULE (--assign)
  frac = area(patch ∩ box) / area(box)

  --assign frac (default)
    frac >= --min-frac for some box -> patch takes the class of the box with the
      LARGEST frac. Largest, not first, so a patch straddling two defects gets
      the one it actually contains most of.
    0 < frac < --min-frac for every overlapping box -> AMBIGUOUS, patch DROPPED.
      Critically it is NOT relabelled 'good'. With exhaustive non-overlapping
      tiling a defect near a boundary leaves a sliver in the neighbouring tile;
      filing that sliver under 'good' would teach the model that a partial
      mousebite is clean, which is the worst possible direction for a labelling
      error on a QC line.
    no overlap at all -> 'good'.

  --assign centre
    The patch containing the box centre takes the class. Every other patch the
    box touches is marked AMBIGUOUS and dropped. Exactly one positive per box.

  Trade-off, worth stating in the README: 'frac' can emit two positives for one
  boundary-straddling box (both halves are real defect, both labels honest, but
  they are near-duplicates within one board). 'centre' emits exactly one and
  discards the other genuine half. Both defensible. The choice is logged to
  MLflow via patch_stats.json so every run is traceable to its label definition.

SPLITTING (locked decision 8) happens at BOARD level. Patches from one board
share lighting, registration error and trace layout, so splitting after tiling
would leak between train and test.

NPZ ARRAY CONTRACT
  Per-patch (length N, aligned index-for-index):
    X          uint8   (N, 64, 64)  the patch pixels
    y          int64   (N,)         0=good, 1..6 defect class
    board_idx  int32   (N,)         index into board_ids
    grid_y     int16   (N,)         tile row on the 10x10 board grid
    grid_x     int16   (N,)         tile col
    off_y      int16   (N,)         patch top edge in BOARD pixels (grid_y * 64)
    off_x      int16   (N,)         patch left edge in BOARD pixels
  Per-box side table (length M, ragged relationship to patches):
    box_patch_idx int32   (M,)      which patch this box row belongs to
    box_local     int16   (M, 4)    x1,y1,x2,y2 CLIPPED to the patch, in
                                    PATCH-LOCAL pixels 0..63. This is what Day 4's
                                    pointing game compares the Grad-CAM argmax
                                    against.
    box_global    int16   (M, 4)    same box, unclipped, in board pixels
    box_class     int8    (M,)      1..6
    box_frac      float32 (M,)      intersection / box area for this patch
    box_primary   bool    (M,)      True for the box that determined y[patch].
                                    Slivers from other boxes are recorded but
                                    flagged False so Day 4 can exclude them.
  Per-board (length B):
    board_ids     str  (B,)
    board_labels  int8 (B, 10, 10)  full label grid INCLUDING dropped tiles
    board_dropped bool (B, 10, 10)  True where a tile was dropped as ambiguous.
                                    Day 4's stitched board map renders these as
                                    'abstained' rather than as holes.

OUTPUTS under --out (default data/patches):
  {train,val,test}.npz
  sample/{split}/{class}/*.png    capped visual sample, eyeball material only
  patch_stats.json                class counts, labelling rule, norm stats
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.deeppcb import CLASSES, Board, read_boxes, read_index

PATCH = 64
BOARD = 640
GRID = BOARD // PATCH


def label_grid(boxes, min_frac: float, assign: str):
    """Return (labels, ambiguous, cell_boxes, winner) for one board.

    labels     (G, G) int64, 0=good, 1..6
    ambiguous  (G, G) bool
    cell_boxes dict[(iy, ix)] -> list of (x1,y1,x2,y2,cls,frac,box_index).
               EVERY box with frac > 0 is recorded, including slivers, so Day 4
               can see near-misses. Which one is primary is resolved separately.
    winner     dict[(iy, ix)] -> index of the box that set the cell's label
    """
    labels = np.zeros((GRID, GRID), dtype=np.int64)
    ambiguous = np.zeros((GRID, GRID), dtype=bool)
    best_frac = np.zeros((GRID, GRID), dtype=np.float32)
    winner: dict[tuple[int, int], int] = {}
    cell_boxes: dict[tuple[int, int], list] = {}

    for bi, (x1, y1, x2, y2, cls) in enumerate(boxes):
        box_area = max(1, (x2 - x1) * (y2 - y1))
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        centre_cell = (min(GRID - 1, max(0, cy // PATCH)),
                       min(GRID - 1, max(0, cx // PATCH)))

        # Only visit grid cells the box can possibly touch. With ~10 boxes over
        # 1500 boards this keeps the whole tiling well under a minute.
        iy0, iy1 = max(0, y1 // PATCH), min(GRID - 1, y2 // PATCH)
        ix0, ix1 = max(0, x1 // PATCH), min(GRID - 1, x2 // PATCH)

        for iy in range(iy0, iy1 + 1):
            py1, py2 = iy * PATCH, (iy + 1) * PATCH
            ih = min(py2, y2) - max(py1, y1)
            if ih <= 0:
                continue
            for ix in range(ix0, ix1 + 1):
                px1, px2 = ix * PATCH, (ix + 1) * PATCH
                iw = min(px2, x2) - max(px1, x1)
                if iw <= 0:
                    continue
                frac = float(iw * ih) / box_area
                cell_boxes.setdefault((iy, ix), []).append(
                    (x1, y1, x2, y2, cls, frac, bi)
                )

                takes = ((iy, ix) == centre_cell) if assign == "centre" else (frac >= min_frac)
                if takes:
                    if frac > best_frac[iy, ix]:
                        best_frac[iy, ix] = frac
                        labels[iy, ix] = cls
                        winner[(iy, ix)] = bi
                else:
                    ambiguous[iy, ix] = True

    # A cell confidently labelled by one box is not ambiguous merely because a
    # second, unrelated box clipped its corner.
    ambiguous &= labels == 0
    return labels, ambiguous, cell_boxes, winner


def tile_board(board: Board, min_frac: float, assign: str):
    """Tile one board into per-patch arrays plus a per-box list."""
    # L mode = single channel. DeepPCB images are already binarised, so RGB would
    # triple memory for zero information (locked decision 4).
    arr = np.asarray(Image.open(board.test_path).convert("L"), dtype=np.uint8)
    if arr.shape != (BOARD, BOARD):
        raise ValueError(f"{board.board_id}: expected {BOARD}x{BOARD}, got {arr.shape}")

    labels, ambiguous, cell_boxes, winner = label_grid(
        read_boxes(board.ann_path), min_frac, assign
    )

    px, pl, pgy, pgx = [], [], [], []
    boxes_out = []          # (local_patch_idx, x1,y1,x2,y2, cls, frac, is_primary)
    local_idx = 0

    for iy in range(GRID):
        for ix in range(GRID):
            if ambiguous[iy, ix]:
                continue    # dropped from the dataset, but recorded in board_dropped
            px.append(arr[iy * PATCH:(iy + 1) * PATCH, ix * PATCH:(ix + 1) * PATCH])
            pl.append(labels[iy, ix])
            pgy.append(iy)
            pgx.append(ix)

            for (x1, y1, x2, y2, cls, frac, bi) in cell_boxes.get((iy, ix), []):
                boxes_out.append((local_idx, x1, y1, x2, y2, cls, frac,
                                  winner.get((iy, ix)) == bi))
            local_idx += 1

    return {
        "X": np.stack(px).astype(np.uint8),
        "y": np.asarray(pl, dtype=np.int64),
        "grid_y": np.asarray(pgy, dtype=np.int16),
        "grid_x": np.asarray(pgx, dtype=np.int16),
    }, boxes_out, labels.astype(np.int8), ambiguous


def build_split(boards: list[Board], min_frac: float, assign: str,
                max_good_per_board: int | None, rng):
    """Tile a list of boards into flat arrays with globally-consistent indices."""
    X, Y, GY, GX, BIDX = [], [], [], [], []
    BP, BLOC, BGLOB, BCLS, BFRAC, BPRIM = [], [], [], [], [], []
    board_labels, board_dropped = [], []
    offset = 0                       # running global patch index

    for bi, b in enumerate(boards):
        parts, boxes_out, lab_grid, amb_grid = tile_board(b, min_frac, assign)
        keep = np.ones(len(parts["y"]), dtype=bool)

        if max_good_per_board is not None:
            good_idx = np.flatnonzero(parts["y"] == 0)
            if len(good_idx) > max_good_per_board:
                drop = rng.choice(good_idx,
                                  size=len(good_idx) - max_good_per_board,
                                  replace=False)
                keep[drop] = False

        # Remap local patch indices after the optional 'good' subsample. Without
        # this the box side table would point at whatever slid into the vacated
        # positions. That bug does NOT crash: it produces a plausible-looking but
        # meaningless Day 4 pointing-game score. Unconditional on purpose, even
        # though max_good_per_board is off by default.
        remap = np.full(len(keep), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.sum())

        for k in parts:
            parts[k] = parts[k][keep]

        for (li, x1, y1, x2, y2, cls, frac, prim) in boxes_out:
            ni = remap[li]
            if ni < 0:
                continue             # its patch was subsampled away
            ox = int(parts["grid_x"][ni]) * PATCH
            oy = int(parts["grid_y"][ni]) * PATCH
            # Clip to the patch, then shift into patch-local coordinates.
            lx1, ly1 = max(x1, ox) - ox, max(y1, oy) - oy
            lx2, ly2 = min(x2, ox + PATCH) - ox, min(y2, oy + PATCH) - oy
            BP.append(offset + ni)
            BLOC.append((lx1, ly1, lx2, ly2))
            BGLOB.append((x1, y1, x2, y2))
            BCLS.append(cls)
            BFRAC.append(frac)
            BPRIM.append(prim)

        n = len(parts["y"])
        X.append(parts["X"]); Y.append(parts["y"])
        GY.append(parts["grid_y"]); GX.append(parts["grid_x"])
        BIDX.append(np.full(n, bi, dtype=np.int32))
        board_labels.append(lab_grid)
        board_dropped.append(amb_grid)
        offset += n

    gy = np.concatenate(GY)
    gx = np.concatenate(GX)
    return {
        "X": np.concatenate(X),
        "y": np.concatenate(Y),
        "board_idx": np.concatenate(BIDX),
        "grid_y": gy,
        "grid_x": gx,
        # Redundant with grid_* but stored anyway: Day 4's board stitching reads
        # offsets in board pixels, and a *64 scattered through plotting code is a
        # bug farm.
        "off_y": (gy.astype(np.int32) * PATCH).astype(np.int16),
        "off_x": (gx.astype(np.int32) * PATCH).astype(np.int16),
        "box_patch_idx": np.asarray(BP, dtype=np.int32),
        "box_local": np.asarray(BLOC, dtype=np.int16).reshape(-1, 4),
        "box_global": np.asarray(BGLOB, dtype=np.int16).reshape(-1, 4),
        "box_class": np.asarray(BCLS, dtype=np.int8),
        "box_frac": np.asarray(BFRAC, dtype=np.float32),
        "box_primary": np.asarray(BPRIM, dtype=bool),
        "board_ids": np.array([b.board_id for b in boards]),
        "board_labels": np.stack(board_labels),
        "board_dropped": np.stack(board_dropped),
    }


def dump_samples(X, Y, out_dir: Path, split: str, per_class: int, rng):
    """Capped PNG sample per class. Eyeball material only, never read for training.

    The full ~150k patches are deliberately NOT written as individual PNGs:
    small-file writes over a Drive FUSE mount are pathologically slow and
    training loads from the npz anyway.
    """
    for ci, cname in enumerate(CLASSES):
        idx = np.flatnonzero(Y == ci)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        d = out_dir / "sample" / split / cname
        d.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(pick):
            Image.fromarray(X[i]).save(d / f"{k:03d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/PCBData"))
    ap.add_argument("--out", type=Path, default=Path("data/patches"))
    ap.add_argument("--assign", choices=["frac", "centre"], default="frac")
    ap.add_argument("--min-frac", type=float, default=0.25,
                    help="min intersection/box-area to take a defect label "
                         "(--assign frac only)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-good-per-board", type=int, default=None,
                    help="cap good patches per board. Leave unset to preserve the "
                         "true imbalance. If you set it, say so in the README.")
    ap.add_argument("--sample-per-class", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    trainval = read_index(args.raw / "trainval.txt", args.raw)
    test_boards = read_index(args.raw / "test.txt", args.raw)

    # Fixed seed matters more here than anywhere else in the project: Day 3's
    # classical baseline must be measured on the exact same test boards or the
    # CNN-vs-classical comparison is void.
    order = list(range(len(trainval)))
    random.Random(args.seed).shuffle(order)
    n_val = int(len(order) * args.val_frac)
    val_boards = [trainval[i] for i in order[:n_val]]
    train_boards = [trainval[i] for i in order[n_val:]]

    stats = {"assign": args.assign, "min_frac": args.min_frac, "seed": args.seed,
             "classes": CLASSES, "splits": {}}

    for split, boards in [("train", train_boards), ("val", val_boards),
                          ("test", test_boards)]:
        out = build_split(boards, args.min_frac, args.assign,
                          args.max_good_per_board, rng)
        # savez_compressed, not savez: binarised images compress roughly 10x,
        # turning a 500MB Drive write into ~50MB. Decompression costs ~2s.
        np.savez_compressed(args.out / f"{split}.npz", **out)
        dump_samples(out["X"], out["y"], args.out, split, args.sample_per_class, rng)

        counts = Counter(out["y"].tolist())
        stats["splits"][split] = {
            "boards": len(boards),
            "patches": int(len(out["y"])),
            "dropped_ambiguous": int(out["board_dropped"].sum()),
            "boxes_recorded": int(len(out["box_class"])),
            "boxes_primary": int(out["box_primary"].sum()),
            "per_class": {CLASSES[c]: int(counts.get(c, 0)) for c in range(len(CLASSES))},
            "defect_patches": int((out["y"] != 0).sum()),
        }
        print(f"[{split}] {len(boards)} boards -> {len(out['y'])} patches, "
              f"{int(out['board_dropped'].sum())} dropped, "
              f"{len(out['box_class'])} box records")
        for c in range(len(CLASSES)):
            print(f"    {CLASSES[c]:>10}: {counts.get(c, 0)}")

        if split == "train":
            # TRAIN ONLY. Computing normalisation stats on val/test is leakage.
            f = out["X"].astype(np.float32) / 255.0
            stats["norm"] = {"mean": float(f.mean()), "std": float(f.std() + 1e-8)}

    (args.out / "patch_stats.json").write_text(json.dumps(stats, indent=2))
    art = Path("artifacts"); art.mkdir(exist_ok=True)
    (art / "classes.json").write_text(json.dumps(CLASSES, indent=2))
    (art / "norm.json").write_text(json.dumps(stats["norm"], indent=2))
    print("\nwrote", args.out / "patch_stats.json")


if __name__ == "__main__":
    main()