"""Class activation maps for the serving path - NO autograd.

This is the direct payoff of choosing a GAP head on Day 1.

Day 4 proved numerically (max abs diff 0.000e+00) that on a
GAP -> single-Linear head, Grad-CAM's channel weights alpha_k reduce exactly to
W[c,k] / (H*W): the gradient of logit c w.r.t. feature channel k is constant
across space and equal to W[c,k]/(H*W), so global-average-pooling it is a no-op.
The 1/(H*W) is a positive scalar and dies in the max normalisation.

Consequences for serving:
  * no .backward(), no torch.autograd.grad, no retained graph
  * the whole request runs under torch.no_grad() -> less RSS, faster, and no
    risk of the zombie-forward-hook bug that bit Day 4
  * tests/test_serving_cam.py re-derives the autograd weights inline and
    asserts equality, so this claim is a test, not a comment

Layer choice is `final` (256ch, 4x4). Day 4 measured pointing 0.8039 there vs
0.6782 at features.2, despite features.2 having 4x the spatial resolution.
Do not "improve" this by switching to the higher-resolution layer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_head_weight(model: nn.Module) -> torch.Tensor:
    """The single classifier Linear's weight, (C, K).

    Asserted to be unique: two Linears would mean the head is not a bare GAP
    projection and the CAM == Grad-CAM identity no longer holds. Confirmed at
    runtime on Day 5: PCBNet has exactly one Linear, in=256 out=7.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 1:
        raise RuntimeError(
            f"PCBNet head must be exactly one Linear for CAM==Grad-CAM; found {len(linears)}"
        )
    return linears[0].weight            # (C, K)


def cam_from_features(feats: torch.Tensor, W: torch.Tensor, class_idx: torch.Tensor):
    """feats (B,K,h,w), W (C,K), class_idx (B,) -> (cam (B,h,w) in [0,1], degenerate (B,) bool).

    cam = ReLU( sum_k W[c,k] * feats[k] ), then per-sample max-normalised.
    Degenerate == everything <= 0 after ReLU (2.7% of patches at the final
    layer on Day 4). Reported, not hidden: a flat map is a real answer.
    """
    w_sel = W[class_idx]                                    # (B,K)
    cam = torch.einsum("bkhw,bk->bhw", feats, w_sel)        # (B,h,w)
    cam = torch.relu(cam)
    peak = cam.amax(dim=(1, 2))                             # (B,)
    degenerate = peak <= 0
    cam = cam / peak.clamp(min=1e-12).view(-1, 1, 1)
    cam = torch.where(degenerate.view(-1, 1, 1), torch.zeros_like(cam), cam)
    return cam, degenerate


def upsample_cam(cam: torch.Tensor, size: int = 64) -> torch.Tensor:
    """(B,h,w) -> (B,size,size). Bilinear, align_corners=False, matching Day 4's
    16x upsample so serving heatmaps are comparable to the pointing-game figures."""
    return F.interpolate(cam.unsqueeze(1), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze(1)


def _jet(x: np.ndarray) -> np.ndarray:
    """Piecewise-linear jet approximation, (H,W) in [0,1] -> (H,W,3) uint8.

    Hand-rolled rather than matplotlib: matplotlib pulls ~40MB plus fontconfig
    into the serving image for what is four lines of clipping arithmetic.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(patch_u8: np.ndarray, cam_hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Grayscale patch + jet CAM -> (64,64,3) uint8 RGB.

    Kept at 1:1 pixel scale (no upscaling for looks). A 64x64 PNG is ~2KB, which
    matters if Day 8's alerting consumer ever attaches heatmaps to messages.
    """
    base = np.repeat(patch_u8[..., None].astype(np.float32), 3, axis=2)
    heat = _jet(cam_hw).astype(np.float32)
    out = (1 - alpha) * base + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)