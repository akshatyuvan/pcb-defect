"""The inference core. One model instance, shared by every endpoint.

Kafka consumers (Day 7) call this over HTTP and never import torch - locked
decision 10. That is an architecture preference AND a hard memory requirement:
three consumers each holding a torch runtime would blow the 4GB Docker budget
on its own, before Kafka's JVM.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.serving.cam import cam_from_features, get_head_weight, overlay, upsample_cam
from src.serving.config import GRID
from src.serving.model_loader import LoadedModel
from src.serving.preprocess import decode_gray, tile_board, to_tensor


def _split_logits_feats(out) -> tuple[torch.Tensor, torch.Tensor]:
    """forward_with_features returns (logits, feats). Confirmed on Day 5 as
    ((B,7), (B,256,4,4)) in that order, but we key off dimensionality rather
    than position so a future reorder cannot silently swap them."""
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise RuntimeError("PCBNet.forward_with_features must return (logits, features)")
    a, b = out
    return (a, b) if a.dim() == 2 else (b, a)


class Inferencer:
    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.W = get_head_weight(lm.model).detach()      # (C,K)
        self.good = lm.good_index
        op = lm.card.get("operating_point", {})
        self.fail_t = float(op.get("fail_threshold", 0.000416))
        rt = op.get("review_threshold")
        self.review_t = float(rt) if rt is not None else None

    # ---- core ----
    @torch.no_grad()
    def _forward(self, arr_u8: np.ndarray):
        x = to_tensor(arr_u8, self.lm.mean, self.lm.std).to(self.lm.device)
        logits, feats = _split_logits_feats(self.lm.model.forward_with_features(x))
        probs = torch.softmax(logits, dim=1)
        return probs, feats

    def defect_score(self, probs: torch.Tensor) -> torch.Tensor:
        """1 - P(good). Day 2's chosen binary score.
        Honest limit for the README: max-over-defect-classes was never compared
        against this, so 'best' is unmeasured - 'chosen and documented' is the
        accurate claim."""
        return 1.0 - probs[:, self.good]

    # ---- endpoints ----
    def predict_patches(self, arr_u8: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        probs, _ = self._forward(arr_u8)
        score = self.defect_score(probs)
        top = probs.argmax(dim=1)
        return {
            "probs": probs.cpu().numpy(),
            "pred_idx": top.cpu().numpy(),
            "defect_score": score.cpu().numpy(),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def predict_board(self, raw: bytes, board_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        board = decode_gray(raw)
        patches, coords = tile_board(board)             # (100,64,64)
        r = self.predict_patches(patches)               # one batched forward, not 100
        probs, pred, score = r["probs"], r["pred_idx"], r["defect_score"]

        grid_pred = pred.reshape(GRID, GRID).astype(int).tolist()
        grid_score = np.round(score.reshape(GRID, GRID), 6).tolist()

        flagged_mask = score >= self.fail_t
        flagged = [
            {
                "grid_y": int(coords[i][0]), "grid_x": int(coords[i][1]),
                "pred_class": self.lm.classes[int(pred[i])],
                "defect_score": round(float(score[i]), 6),
                "class_prob": round(float(probs[i, pred[i]]), 6),
            }
            for i in np.nonzero(flagged_mask)[0]
        ]
        flagged.sort(key=lambda d: -d["defect_score"])

        max_score = float(score.max())
        counts: dict[str, int] = {}
        for i in np.nonzero(flagged_mask)[0]:
            c = self.lm.classes[int(pred[i])]
            counts[c] = counts.get(c, 0) + 1

        return {
            "board_id": board_id,
            "n_patches": int(len(pred)),
            "n_flagged": int(flagged_mask.sum()),
            "max_defect_score": round(max_score, 6),
            "verdict": self.decide(max_score),
            "fail_threshold": self.fail_t,
            "review_threshold": self.review_t,
            "class_counts": counts,
            "flagged": flagged[:32],       # cap payload; grids carry the full picture
            "grid_pred": grid_pred,
            "grid_defect_score": grid_score,
            "classes": self.lm.classes,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def decide(self, max_score: float) -> str:
        """Three-outcome routing, degrading gracefully to two.

        review_threshold is null until Day 7 measures a second operating point
        (Day 2 has the PR cliff between recall 0.86-0.88 but never recorded the
        threshold value there). Inventing a number to fill the third bucket
        would be a fabricated measurement, so the service returns two outcomes
        and says so, rather than pretending to triage."""
        if self.review_t is None:
            return "fail" if max_score >= self.fail_t else "pass"
        if max_score >= self.review_t:
            return "fail"
        if max_score >= self.fail_t:
            return "uncertain"
        return "pass"

    def explain_patch(self, raw: bytes, class_idx: int | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        patch = decode_gray(raw)
        with torch.no_grad():
            probs, feats = self._forward(patch)
            idx = torch.tensor([class_idx], device=self.lm.device) if class_idx is not None \
                else probs.argmax(dim=1)
            cam_small, degenerate = cam_from_features(feats, self.W, idx)
            cam = upsample_cam(cam_small, size=patch.shape[-1])
        cam_np = cam[0].cpu().numpy()
        png = _png_bytes(overlay(patch, cam_np))
        p = probs[0].cpu().numpy()
        return {
            "pred_class": self.lm.classes[int(idx[0])],
            "pred_index": int(idx[0]),
            "confidences": {c: round(float(v), 6) for c, v in zip(self.lm.classes, p)},
            "defect_score": round(float(1.0 - p[self.good]), 6),
            "cam_degenerate": bool(degenerate[0]),
            "cam_layer": "final",
            "cam_method": "CAM (provably identical to Grad-CAM on this GAP+Linear head)",
            "overlay_png_b64": base64.b64encode(png).decode("ascii"),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()