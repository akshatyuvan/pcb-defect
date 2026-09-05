"""Prove, in the serving code path, that the no-autograd CAM equals Grad-CAM.

Day 4 proved this on Colab against src/explain/gradcam.py. This test re-derives
it against src/serving/cam.py, so the claim stays true if either file changes.
It is the reason /explain/patch can run entirely under torch.no_grad().

Mac (and CI on Day 9). CPU, <2s. Run: pytest -q tests/test_serving_cam.py
"""
import numpy as np
import torch

from src.serving.cam import cam_from_features, get_head_weight, upsample_cam
from src.serving.inference import _split_logits_feats
from src.serving.model_loader import load_model


def test_cam_equals_gradcam_on_gap_head():
    torch.manual_seed(0)  # CI prints the CAM diff; an unseeded input makes that number wobble run to run
    lm = load_model()
    model = lm.model
    W = get_head_weight(model)
    assert W.shape[0] == len(lm.classes)

    x = torch.randn(3, 1, 64, 64, requires_grad=False)

    # --- serving path: no gradients at all ---
    with torch.no_grad():
        logits, feats = _split_logits_feats(model.forward_with_features(x))
        idx = logits.argmax(dim=1)
        cam_fast, _ = cam_from_features(feats, W, idx)

    # --- reference path: literal Grad-CAM, gradients through the feature map ---
    logits_g, feats_g = _split_logits_feats(model.forward_with_features(x))
    sel = logits_g.gather(1, idx.view(-1, 1)).sum()
    grads = torch.autograd.grad(sel, feats_g)[0]          # (B,K,h,w)
    alpha = grads.mean(dim=(2, 3))                        # (B,K)  Grad-CAM weights

    # Claim 1: alpha[b,k] == W[idx[b],k] / (h*w), exactly.
    h, w = feats_g.shape[-2:]
    expected = W[idx] / (h * w)
    assert torch.allclose(alpha, expected, atol=1e-6), \
        f"grad weights != W/(H*W); max diff {(alpha - expected).abs().max():.3e}"

    # Claim 2: the resulting normalised maps are identical.
    cam_ref = torch.relu(torch.einsum("bkhw,bk->bhw", feats_g, alpha))
    peak = cam_ref.amax(dim=(1, 2)).clamp(min=1e-12).view(-1, 1, 1)
    cam_ref = (cam_ref / peak).detach()
    diff = (cam_fast - cam_ref).abs().max().item()
    print(f"\nmax |CAM_serving - GradCAM_ref| = {diff:.3e}")
    assert diff < 1e-5


def test_cam_upsamples_to_patch_resolution():
    cam = torch.rand(2, 4, 4)
    up = upsample_cam(cam, size=64)
    assert up.shape == (2, 64, 64)
    assert float(up.min()) >= 0.0


def test_good_is_class_zero():
    lm = load_model()
    assert lm.classes[0] == "good"
    assert lm.classes == ["good", "open", "short", "mousebite", "spur", "copper", "pinhole"]
