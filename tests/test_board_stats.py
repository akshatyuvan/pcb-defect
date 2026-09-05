"""Tests for the single shared definition of every board statistic.

Why this module gets its own tests: src/streaming/board_stats.py is imported by
BOTH the calibrator (src/mlops/calibrate_board.py) and the inference consumer.
A drift between them cannot raise - it would just route every board wrongly,
forever, while the pipeline reported lag 0 and looked perfectly healthy. That
is the worst failure mode in the system, so it gets pinned arithmetic rather
than smoke tests.
"""

import numpy as np
import pytest

from src.streaming.board_stats import COUNT_THRESHOLDS, compute_stats, find_grid, get_stat


def _response(scores):
    """A minimal /predict/board response. grid_defect_score is the 10x10 nested
    list the API actually returns (confirmed by scripts/probe_contracts.py);
    find_grid flattens it."""
    grid = np.asarray(scores, dtype=float).reshape(10, 10).tolist()
    return {
        "board_id": "test0000",
        "grid_defect_score": grid,
        "grid_pred": [[0] * 10 for _ in range(10)],
        "n_flagged": int(sum(1 for s in np.ravel(grid) if s > 0.000416)),
    }


def test_find_grid_flattens_to_100_floats():
    grid = find_grid(_response([0.25] * 100))
    assert grid is not None
    assert len(grid) == 100
    assert all(isinstance(v, float) for v in grid)


def test_find_grid_prefers_defect_score_over_class_predictions():
    """grid_pred holds argmax class ids 0-6; grid_defect_score holds the
    1 - P(good) score in [0, 1]. Reading the wrong one puts integers where
    floats belong and every threshold comparison silently becomes nonsense -
    a class id of 6 would clear every count threshold on every board."""
    resp = _response([0.4] * 100)
    resp["grid_pred"] = [[6] * 10 for _ in range(10)]

    grid = find_grid(resp)
    assert max(grid) <= 1.0
    assert grid[0] == pytest.approx(0.4)


def test_top3_mean_is_the_mean_of_the_three_largest():
    scores = [0.01] * 97 + [0.9, 0.8, 0.7]
    out = compute_stats(_response(scores))
    assert out["top3_mean"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert out["max_defect_score"] == pytest.approx(0.9)


def test_top5_and_mean_agree_with_numpy():
    rng = np.random.default_rng(0)
    scores = rng.random(100)
    out = compute_stats(_response(scores))
    assert out["top5_mean"] == pytest.approx(np.sort(scores)[-5:].mean())
    assert out["mean_score"] == pytest.approx(scores.mean())


def test_count_thresholds_are_strict_greater_than():
    """0.5 exactly must NOT count. An off-by-one on a boundary is precisely the
    kind of drift that would separate calibrator from consumer, and the
    calibration file records thresholds derived under one convention."""
    scores = [0.5] * 50 + [0.51] * 50
    out = compute_stats(_response(scores))
    assert out["count_gt_0p5"] == 50


def test_every_declared_count_threshold_produces_a_stat():
    """COUNT_THRESHOLDS drives the key names. If a threshold is added and the
    naming convention drifts, board_calibration.json's recorded statistic name
    stops resolving - and get_stat would raise inside the consumer at runtime
    rather than here."""
    out = compute_stats(_response([0.3] * 100))
    for t in COUNT_THRESHOLDS:
        key = "count_gt_" + str(t).replace(".", "p")
        assert key in out, f"missing {key}; declared thresholds and stat names have drifted"


def test_get_stat_resolves_the_calibrated_statistic_by_name():
    """The consumer looks up whichever statistic board_calibration.json names.
    A typo there must raise, not return a default - a defaulted statistic would
    route every board against the wrong number."""
    resp = _response([0.01] * 97 + [1.0, 1.0, 1.0])
    assert get_stat(resp, "top3_mean") == pytest.approx(1.0)
    with pytest.raises(Exception):
        get_stat(resp, "not_a_real_statistic")


def test_top3_mean_separates_the_copper_floor_from_a_real_defect():
    """The measured shape of the problem. Ordinary copper sits at the model's
    0.0171 output floor; real defects sit at 0.4-1.0. top3_mean must put a
    clean-looking board below the 0.985983 pass threshold and a defective one
    at or above the 0.997958 fail threshold - the two numbers in
    artifacts/serving/board_calibration.json."""
    clean = compute_stats(_response([0.0171] * 100))["top3_mean"]
    defective = compute_stats(_response([0.0171] * 97 + [1.0, 0.9999, 0.9998]))["top3_mean"]

    assert clean < 0.9859826564788818
    assert defective >= 0.9979580044746399


def test_n_flagged_is_near_chance_by_construction():
    """Documents WHY n_flagged scored AUROC 0.5877 and was rejected. The Day 2
    patch threshold 0.000416 sits below the 0.0171 copper floor, so a board of
    nothing but ordinary copper flags ~every tile - identically to a defective
    one. This test would fail if someone 'fixed' the patch threshold, which is
    the point: the calibration numbers would then need re-deriving."""
    clean_copper = _response([0.0171] * 100)
    assert clean_copper["n_flagged"] == 100