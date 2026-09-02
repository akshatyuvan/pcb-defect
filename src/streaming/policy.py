"""
Three-outcome board routing: pass / review / fail.

Deliberately pure -- no Kafka, no HTTP, no model. Everything that decides what
happens to a board lives in a function testable with a dict, which is why these
tests run in CI with no broker and no GPU.

Degradation is explicit: if pass_threshold is null the service returns two
outcomes and SAYS SO in the reason field, rather than inventing a third band.
That was locked decision 14 and it still holds; what changed on Day 7 is that
the thresholds are now measurements rather than absences.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.streaming.board_stats import get_stat

PASS = "pass"
REVIEW = "review"
FAIL = "fail"


@dataclass(frozen=True)
class BoardCalibration:
    statistic: str
    fail_threshold: float
    pass_threshold: Optional[float]
    source: str

    @classmethod
    def from_file(cls, path: Path) -> "BoardCalibration":
        doc = json.loads(Path(path).read_text())
        if doc.get("schema_version") != 1:
            raise ValueError(
                f"unsupported calibration schema_version: {doc.get('schema_version')}")
        if doc.get("fail_threshold") is None:
            # Refusing to start beats starting with an uncalibrated operating
            # point: the latter runs happily and mis-routes every board.
            raise ValueError(
                f"{path} has a null fail_threshold. Run "
                "`python -m src.mlops.calibrate_board --write` first.")
        return cls(
            statistic=doc["statistic"],
            fail_threshold=float(doc["fail_threshold"]),
            pass_threshold=(None if doc.get("pass_threshold") is None
                            else float(doc["pass_threshold"])),
            source=str(path),
        )


@dataclass(frozen=True)
class Routing:
    verdict: str
    score: float
    statistic: str
    reason: str


def route(board_response: Dict[str, Any], calib: BoardCalibration) -> Routing:
    # get_stat raises KeyError naming what IS available if the statistic is
    # missing -- a loud failure at the first message beats silent mis-routing.
    score = get_stat(board_response, calib.statistic)

    if score >= calib.fail_threshold:
        return Routing(FAIL, score, calib.statistic,
                       f"{calib.statistic}={score:.6g} >= "
                       f"fail_threshold={calib.fail_threshold:.6g}")

    if calib.pass_threshold is None:
        # Two-outcome mode, named in the reason so a downstream reader never has
        # to guess whether "pass" meant "confidently clean" or "not failed".
        return Routing(PASS, score, calib.statistic,
                       f"{calib.statistic}={score:.6g} < fail_threshold "
                       f"(two-outcome: no pass_threshold calibrated)")

    if score < calib.pass_threshold:
        return Routing(PASS, score, calib.statistic,
                       f"{calib.statistic}={score:.6g} < "
                       f"pass_threshold={calib.pass_threshold:.6g}")

    return Routing(REVIEW, score, calib.statistic,
                   f"pass_threshold={calib.pass_threshold:.6g} <= "
                   f"{calib.statistic}={score:.6g} < "
                   f"fail_threshold={calib.fail_threshold:.6g}")