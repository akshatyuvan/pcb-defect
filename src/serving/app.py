"""FastAPI service. Model loaded ONCE in the lifespan handler.

Loading in lifespan rather than at import time matters: with --reload or with
multiple uvicorn workers, import-time loading gives you N copies of the model
and N times the RSS. One worker, one model, ~1GB ceiling. Day 8 scales by
adding CONSUMERS (which hold no model), not API workers.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.serving.inference import Inferencer
from src.serving.model_loader import load_model
from src.serving.preprocess import decode_gray
from src.serving.schemas import (BoardPrediction, ExplainRequest, Health,
                                 ImageRequest, PatchPrediction)

# Structured JSON logs to stdout. Day 8's observability story is "structured
# logs + the benchmark", explicitly instead of Prometheus/Grafana - so the logs
# have to actually be machine-parseable, not f-strings.
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("pcb.api")


def jlog(event: str, **kw) -> None:
    log.info(json.dumps({"ts": time.time(), "event": event, **kw}))


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    lm = load_model()
    STATE["lm"] = lm
    STATE["inf"] = Inferencer(lm)
    STATE["started"] = time.time()
    jlog("model_loaded", source=lm.source, device=lm.device,
         classes=lm.classes, load_s=round(time.perf_counter() - t0, 3),
         registry_version=lm.card.get("registry_version"))
    yield
    jlog("shutdown")


app = FastAPI(title="PCB Defect Detection API", version="0.5.0", lifespan=lifespan)


def _decode_b64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        # 400, not 500: the caller sent garbage. Day 7's consumer treats 4xx as
        # "poison message -> DLQ" and 5xx as "our fault -> retry, do not commit".
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")


@app.get("/health", response_model=Health)
def health() -> Health:
    lm, inf = STATE["lm"], STATE["inf"]
    return Health(
        status="ok", model_source=lm.source,
        registry_name=lm.card.get("registry_name"),
        registry_version=str(lm.card.get("registry_version")),
        device=lm.device, classes=lm.classes,
        fail_threshold=inf.fail_t, review_threshold=inf.review_t,
        uptime_s=round(time.time() - STATE["started"], 1),
    )


@app.get("/model")
def model_card():
    return JSONResponse(STATE["lm"].card)


@app.post("/predict/patch", response_model=PatchPrediction)
def predict_patch(req: ImageRequest) -> PatchPrediction:
    inf = STATE["inf"]
    raw = _decode_b64(req.image_b64)
    try:
        arr = decode_gray(raw)
        r = inf.predict_patches(arr)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    p = r["probs"][0]
    idx = int(r["pred_idx"][0])
    score = float(r["defect_score"][0])
    jlog("predict_patch", board_id=req.board_id, trace_id=req.trace_id,
         pred=inf.lm.classes[idx], score=round(score, 6),
         latency_ms=round(r["latency_ms"], 3))
    return PatchPrediction(
        pred_class=inf.lm.classes[idx], pred_index=idx,
        confidences={c: round(float(v), 6) for c, v in zip(inf.lm.classes, p)},
        defect_score=round(score, 6), verdict=inf.decide(score),
        latency_ms=round(r["latency_ms"], 3),
        board_id=req.board_id, trace_id=req.trace_id,
    )


@app.post("/predict/board", response_model=BoardPrediction)
def predict_board(req: ImageRequest) -> BoardPrediction:
    inf = STATE["inf"]
    raw = _decode_b64(req.image_b64)
    try:
        out = inf.predict_board(raw, board_id=req.board_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    out["trace_id"] = req.trace_id
    jlog("predict_board", board_id=req.board_id, trace_id=req.trace_id,
         verdict=out["verdict"], n_flagged=out["n_flagged"],
         max_score=out["max_defect_score"], latency_ms=out["latency_ms"])
    return BoardPrediction(**out)


@app.post("/explain/patch")
def explain_patch(req: ExplainRequest):
    inf = STATE["inf"]
    raw = _decode_b64(req.image_b64)
    try:
        out = inf.explain_patch(raw, class_idx=req.class_index)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    jlog("explain_patch", board_id=req.board_id, trace_id=req.trace_id,
         pred=out["pred_class"], degenerate=out["cam_degenerate"],
         latency_ms=out["latency_ms"])
    return JSONResponse(out)