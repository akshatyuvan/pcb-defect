"""Request/response contracts. Pydantic validates at the edge so a malformed
Kafka message fails with a 422 the consumer can route to the DLQ (Day 7),
rather than a 500 that looks like the service is broken."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImageRequest(BaseModel):
    image_b64: str = Field(..., description="base64 of raw PNG or JPEG bytes")
    board_id: str | None = None
    trace_id: str | None = None


class ExplainRequest(ImageRequest):
    class_index: int | None = Field(
        None, description="explain this class instead of the argmax (0=good..6=pinhole)")


class Health(BaseModel):
    status: str
    model_source: str
    registry_name: str | None = None
    registry_version: str | None = None
    device: str
    classes: list[str]
    fail_threshold: float
    review_threshold: float | None
    uptime_s: float


class PatchPrediction(BaseModel):
    pred_class: str
    pred_index: int
    confidences: dict[str, float]
    defect_score: float
    verdict: str
    latency_ms: float
    board_id: str | None = None
    trace_id: str | None = None


class BoardPrediction(BaseModel):
    board_id: str | None = None
    trace_id: str | None = None
    verdict: str
    n_patches: int
    n_flagged: int
    max_defect_score: float
    fail_threshold: float
    review_threshold: float | None
    class_counts: dict[str, int]
    flagged: list[dict[str, Any]]
    grid_pred: list[list[int]]
    grid_defect_score: list[list[float]]
    classes: list[str]
    latency_ms: float