"""
Board-level statistics derived from a /predict/board response.

Imported by BOTH the calibrator and the production consumer, on purpose. If
calibration computed "max over tiles" and the consumer computed "mean of top 3",
the threshold would be a number attached to nothing, and the mismatch would
never raise -- it would just route wrongly. One definition, two callers.

No torch, no numpy: this file is copied into the consumer image (locked
decision 10 -- consumers hold no model).

Response contract confirmed from src/serving/schemas.py BoardPrediction:
    max_defect_score : float
    n_flagged        : int          tiles above the Day 2 patch threshold
    grid_defect_score: list[list[float]]   10x10, row-major
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Confirmed name first; the rest are tolerated aliases so a future rename of
# the serving schema degrades to "statistic unavailable" rather than silence.
GRID_KEYS = ("grid_defect_score", "grid_score", "grid_scores", "scores",
             "patch_scores", "tile_scores")

# Count thresholds. 0.0171 is not arbitrary: Day 5 measured it as the model's
# characteristic score on ORDINARY COPPER, a mode that dozens of unrelated tiles
# sit on exactly. Counting tiles above it asks "how many tiles are anomalous
# even by copper's standards", which is a different and probably better question
# than "how bad is the single worst tile". 0.4 is the bottom of the measured
# real-defect mode (0.4-1.0).
COUNT_THRESHOLDS = (0.0171, 0.05, 0.2, 0.4, 0.5)


def _count_key(t: float) -> str:
    return f"count_gt_{t:g}".replace(".", "p")


def find_grid(resp: Dict[str, Any]) -> Optional[List[float]]:
    """Flatten a per-tile score grid to a list, or None if absent."""
    for k in GRID_KEYS:
        v = resp.get(k)
        if not v:
            continue
        if isinstance(v[0], list):            # 10x10 nested, as served
            return [float(x) for row in v for x in row]
        return [float(x) for x in v]
    return None


def compute_stats(resp: Dict[str, Any]) -> Dict[str, float]:
    """Every candidate board statistic derivable from one response.

    max_defect_score is the obvious one, and Day 5 already showed it is a poor
    one: it is a max over 100 tiles, so one unlucky tile decides the board. The
    topk means and the counts exist because a board with five suspicious tiles
    is a different object from a board with one, and only the calibration data
    can say which distinction separates clean from defective.
    """
    out: Dict[str, float] = {}
    if "max_defect_score" in resp:
        out["max_defect_score"] = float(resp["max_defect_score"])
    if "n_flagged" in resp:
        # count above the Day 2 patch threshold 0.000416, computed by the
        # service itself. Kept as a candidate because it costs nothing.
        out["n_flagged"] = float(resp["n_flagged"])

    grid = find_grid(resp)
    if grid:
        s = sorted(grid, reverse=True)
        out.setdefault("max_defect_score", float(s[0]))
        out["top3_mean"] = sum(s[:3]) / min(3, len(s))
        out["top5_mean"] = sum(s[:5]) / min(5, len(s))
        out["mean_score"] = sum(grid) / len(grid)
        for t in COUNT_THRESHOLDS:
            out[_count_key(t)] = float(sum(1 for x in grid if x > t))
    return out


def get_stat(resp: Dict[str, Any], name: str) -> float:
    stats = compute_stats(resp)
    if name not in stats:
        raise KeyError(
            f"board statistic {name!r} not available. computed={sorted(stats)}; "
            f"response keys={sorted(resp)}"
        )
    return stats[name]