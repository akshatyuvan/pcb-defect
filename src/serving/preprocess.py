"""Decode + normalise, in the EXACT order Day 1's to_tensor() used.

Order is load-bearing and was verified twice on Colab:
    uint8 -> float32/255 -> subtract mean -> divide std -> unsqueeze
Getting the mean/std subtraction before the /255 (or vice versa) produces a
model that still runs, still outputs plausible probabilities, and is quietly
wrong. There is no runtime error to catch it. Hence: one function, one place.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from src.serving.config import BOARD_SIZE, GRID, PATCH_SIZE


def decode_gray(raw: bytes) -> np.ndarray:
    """Bytes (PNG or JPEG) -> uint8 (H,W) single channel.

    Pillow rather than OpenCV so the Docker image stays OpenCV-free.
    Verified bit-identical to cv2.IMREAD_GRAYSCALE on DeepPCB by
    scripts/verify_decode_parity.py (25 boards, max diff 0) - do not change
    decoders without re-running it.
    """
    img = Image.open(io.BytesIO(raw))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def to_tensor(arr_u8: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """uint8 (N,64,64) or (64,64) -> float32 (N,1,64,64), normalised.

    Cast to float happens here, not at storage time: Day 1 keeps patches as
    uint8 (~579MB) precisely so the float32 blow-up (4x) never has to exist
    for the whole dataset at once.
    """
    if arr_u8.ndim == 2:
        arr_u8 = arr_u8[None, ...]
    if arr_u8.ndim != 3:
        raise ValueError(f"expected (N,H,W) or (H,W) uint8, got {arr_u8.shape}")
    if arr_u8.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"expected {PATCH_SIZE}x{PATCH_SIZE} patches, got {arr_u8.shape[-2:]}")
    x = torch.from_numpy(np.ascontiguousarray(arr_u8)).float()
    x = x / 255.0
    x = (x - mean) / std
    return x.unsqueeze(1)               # (N,1,64,64)


def tile_board(board_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """640x640 uint8 -> (100,64,64) patches + (100,2) [grid_y, grid_x].

    Row-major, same order as Day 1's builder, so grid coordinates returned by
    the API line up with board_labels[b, gy, gx] in the npz files. Day 1 DROPPED
    ~5.8% of patches for labelling reasons; inference drops nothing - every tile
    gets a prediction, because at serve time there is no label to be ambiguous.
    """
    h, w = board_u8.shape
    if (h, w) != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"expected {BOARD_SIZE}x{BOARD_SIZE} board, got {(h, w)}")
    patches = np.empty((GRID * GRID, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    coords = np.empty((GRID * GRID, 2), dtype=np.int16)
    k = 0
    for gy in range(GRID):
        for gx in range(GRID):
            patches[k] = board_u8[gy * PATCH_SIZE:(gy + 1) * PATCH_SIZE,
                                  gx * PATCH_SIZE:(gx + 1) * PATCH_SIZE]
            coords[k] = (gy, gx)
            k += 1
    return patches, coords