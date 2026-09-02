"""Day 7 policy tests. No broker, no model, no network -- these run in CI."""
from __future__ import annotations

import pytest

from src.streaming.board_stats import compute_stats, get_stat
from src.streaming.policy import FAIL, PASS, REVIEW, BoardCalibration, route


def calib(pass_t=0.2, fail_t=0.6, stat="max_defect_score"):
    return BoardCalibration(statistic=stat, fail_threshold=fail_t,
                            pass_threshold=pass_t, source="test")


def resp(max_score=None, grid=None, n_flagged=None):
    d = {}
    if max_score is not None:
        d["max_defect_score"] = max_score
    if grid is not None:
        d["grid_score"] = grid
    if n_flagged is not None:
        d["n_flagged"] = n_flagged
    return d


def test_stats_from_flat_and_nested_grid_agree():
    flat = [0.1, 0.9, 0.3, 0.4]
    nested = [[0.1, 0.9], [0.3, 0.4]]
    assert compute_stats(resp(grid=flat)) == compute_stats(resp(grid=nested))


def test_topk_and_counts():
    s = compute_stats(resp(grid=[0.9, 0.8, 0.7, 0.01, 0.01]))
    assert s["max_defect_score"] == pytest.approx(0.9)
    assert s["top3_mean"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert s["count_gt_0p5"] == 3.0
    assert s["count_gt_0p05"] == 3.0


def test_real_serving_grid_key_is_recognised():
    """BoardPrediction serves the grid as `grid_defect_score` (confirmed from
    src/serving/schemas.py). If find_grid stops recognising it, calibration
    loses every count-based statistic WITHOUT raising -- so pin it."""
    r = {"grid_defect_score": [[0.9, 0.01], [0.01, 0.6]], "max_defect_score": 0.9}
    s = compute_stats(r)
    assert s["count_gt_0p4"] == 2.0
    assert s["mean_score"] == pytest.approx((0.9 + 0.01 + 0.01 + 0.6) / 4)


def test_three_outcomes():
    c = calib()
    assert route(resp(max_score=0.01), c).verdict == PASS
    assert route(resp(max_score=0.4), c).verdict == REVIEW
    assert route(resp(max_score=0.95), c).verdict == FAIL


def test_boundaries_are_closed_below():
    """>= fail is fail; < pass is pass. Exactly on pass_threshold -> review."""
    c = calib(pass_t=0.2, fail_t=0.6)
    assert route(resp(max_score=0.6), c).verdict == FAIL
    assert route(resp(max_score=0.2), c).verdict == REVIEW
    assert route(resp(max_score=0.19999), c).verdict == PASS


def test_degrades_to_two_outcomes_and_says_so():
    c = BoardCalibration("max_defect_score", 0.6, None, "test")
    r = route(resp(max_score=0.01), c)
    assert r.verdict == PASS
    assert "two-outcome" in r.reason


def test_missing_statistic_names_what_is_available():
    with pytest.raises(KeyError) as e:
        get_stat(resp(max_score=0.5), "top3_mean")
    assert "max_defect_score" in str(e.value)