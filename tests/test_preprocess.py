"""Tests for the train/serve preprocessing boundary.

This is the highest-value test file in the project, because every failure mode
here is SILENT. A tiling bug does not raise, it just scores the wrong 64x64
region. A normalisation bug does not raise, it just shifts every prediction.
The one exception is the decode path, and that one already bit us on Day 7.
"""

import io

import numpy as np
import pytest
from PIL import Image

from src.serving import preprocess


def _fake_board(size=640, value=200):
    """A synthetic 640x640 grayscale JPEG. We do not touch data/ in tests:
    the dataset is gitignored, so a test that reads it cannot run in CI."""
    arr = np.full((size, size), value, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_decode_gray_returns_single_channel_640():
    # np.asarray() so this passes whether decode_gray hands back a PIL image
    # or an ndarray - the contract we care about is shape and dtype.
    arr = np.asarray(preprocess.decode_gray(_fake_board()))
    assert arr.ndim == 2, "serving must produce single-channel; model input is (B,1,64,64)"
    assert arr.shape == (640, 640)
    assert arr.dtype == np.uint8


def test_decode_gray_rejects_undecodable_bytes():
    """REGRESSION, Day 7.

    Pillow raises UnidentifiedImageError, which subclasses OSError, not
    ValueError. app.py caught only ValueError, so /predict/board returned 500
    instead of 422. The wrong status code is what actually hurt: the consumer
    retries 5xx and never commits, so one bad image blocked a partition head
    and crash-looped the consumer through three full restart cycles.
    """
    with pytest.raises(ValueError):
        preprocess.decode_gray(b"this is definitely not an image")


def test_decode_gray_rejects_truncated_jpeg():
    """Pillow decodes lazily. A truncated file passes Image.open() and only
    fails at pixel access, which is why decode_gray calls img.load() eagerly.
    This test is what stops someone deleting that line as redundant."""
    raw = _fake_board()
    with pytest.raises(ValueError):
        preprocess.decode_gray(raw[: len(raw) // 3])


def test_tile_board_is_100_patches_in_row_major_order():
    """Row-major order is load-bearing: the 10x10 grid_defect_score returned by
    the API, the CAM stitching, and board_stats.find_grid all assume flat index
    i maps to row i//10, col i%10. Column-major tiling would silently TRANSPOSE
    every board heatmap and nothing would raise - the shapes stay identical.

    tile_board returns (patches, coords). Checking both is deliberate: coords
    is what tile_board CLAIMS the order is, and the pixel values are what it
    actually did. A bug that got both wrong in the same direction is possible
    but far less likely than one that desyncs them.
    """
    # Paint tile (r, c) with the constant value r*10 + c, so a patch's pixel
    # value IS its expected row-major index.
    board = np.zeros((640, 640), dtype=np.uint8)
    for r in range(10):
        for c in range(10):
            board[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = r * 10 + c

    patches, coords = preprocess.tile_board(board)

    assert len(patches) == 100
    assert len(coords) == 100

    for i in range(100):
        patch = np.asarray(patches[i])
        assert patch.shape[-2:] == (64, 64), f"patch {i} is not 64x64"

        # 1. the declared coordinate is row-major
        row, col = int(coords[i][0]), int(coords[i][1])
        assert (row, col) == (i // 10, i % 10), f"coords[{i}] = {(row, col)}, expected {(i // 10, i % 10)}"

        # 2. the pixels actually came from that tile
        assert int(patch.flat[0]) == i, f"patch {i} holds tile {int(patch.flat[0])}"


def test_tile_board_coords_are_consistent_with_patch_contents():
    """Same invariant, checked the other way round: index the board directly
    using each declared coordinate and require a byte-for-byte match. This is
    what would catch an off-by-one in the slice bounds - a one-pixel shift
    keeps the shape, keeps the order, and quietly changes every input."""
    rng = np.random.default_rng(0)
    board = rng.integers(0, 256, size=(640, 640), dtype=np.uint8)

    patches, coords = preprocess.tile_board(board)

    for i in (0, 7, 42, 99):
        r, c = int(coords[i][0]), int(coords[i][1])
        expected = board[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]
        np.testing.assert_array_equal(np.asarray(patches[i]).reshape(64, 64), expected)


def test_to_tensor_normalisation_matches_training():
    """The exact arithmetic from Day 1: uint8 -> /255 -> -mean -> /std.
    Any drift here is a train/serve skew that produces no error, only worse
    numbers - the same failure class as a decoder mismatch."""
    mean, std = 0.6472, 0.4760
    patch = np.full((64, 64), 255, dtype=np.uint8)

    t = preprocess.to_tensor(patch, mean, std)

    assert tuple(t.shape) == (1, 1, 64, 64), "needs a batch dim AND a channel dim"
    expected = (1.0 - mean) / std
    assert float(t.max()) == pytest.approx(expected, abs=1e-5)
    assert float(t.min()) == pytest.approx(expected, abs=1e-5)