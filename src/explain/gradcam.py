#!/usr/bin/env python3
"""
Day 4: Grad-CAM, pointing game, and the board-level stitched defect map.

READ THIS FIRST: ON PCBNET, GRAD-CAM *IS* CAM
  PCBNet is conv stack -> Global Average Pool -> one Linear. For that head,
      logit_c = sum_k W[c,k] * mean_ij fmap[k,i,j] + b_c
      d logit_c / d fmap[k,i,j] = W[c,k] / (H*W)     for every i, j
  The gradient is constant across space, so Grad-CAM's spatially-averaged
  weight is exactly W[c,k]/(H*W) and the method degenerates into CAM
  (Zhou et al. 2016). That is expected: GAP+Linear is the architecture CAM was
  built for, and Grad-CAM generalises it to heads where the identity fails.

  Two things follow.
  (a) verify_gradcam_is_cam() turns this into a free correctness proof. If the
      computed weights match W/(H*W), then the hook, the gradient path and the
      class indexing are all correct. Grad-CAM bugs are otherwise silent: a
      wrong class index still produces a plausible-looking blob.
  (b) To get a genuinely different map you must hook an EARLIER block, where
      the gradient is not spatially constant. --layers final,features.2 runs
      both. Block 3 is 8x8, so an 8x upsample instead of 16x.

EVALUATION: THE POINTING GAME
  Take the CAM argmax pixel, ask whether it lands inside the ground-truth box.
  Reported per class against two baselines, because a bare pointing score is
  uninterpretable when boxes are large:
      random pixel  -> expected hit rate = mean(box area) / 4096
      always centre -> hit rate of predicting (32, 32) every time
  Only rows with box_primary == True are used, per the Day 2 handoff: sliver
  rows describe a box that did NOT determine the patch label.

TWO QUEUED INVESTIGATIONS (from the Day 2 error analysis)
  (a) Does pointing accuracy degrade at low box_frac? If yes, that quantifies
      the cost of --min-frac 0.25 and is a candidate explanation for the PR
      cliff between recall 0.86 and 0.88.
  (b) Does the CAM peak sit on a trace EDGE for mousebite and in the trace
      INTERIOR for pinhole? That would explain the stable ~0.10 bidirectional
      confusion that loss weighting never touched. Operationalised as the
      copper fraction in an 11x11 window around the peak: an edge bite gives
      a mixed neighbourhood, an interior void gives a copper-dominated one.

RUNS ON: Colab. GPU if available, CPU is fine (a few thousand forward+backward
passes on a 1.17M-param model). Not the Mac, torch is not installed there yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.paths import index_boards, resolve
from src.models.checkpoint import load_classes, load_norm, load_pcbnet

GRID, PATCH = 10, 64


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------

class GradCAM:
    """Batched Grad-CAM with no manual backward pass.

    torch.autograd.grad(logit_sum, fmap) gives the gradient w.r.t. an
    intermediate tensor directly, so we never touch .backward(), never dirty
    the parameter .grad buffers, and never need a backward hook. Cleaner and
    it works on a batch in one shot.
    """

    def __init__(self, model, layer: str = "final"):
        self.model = model
        self.layer = layer
        self._act = None
        self._handle = None
        if layer != "final":
            mods = dict(model.named_modules())
            if layer not in mods:
                raise KeyError(
                    f"no module named '{layer}'. available module names:\n  "
                    + "\n  ".join(n for n in sorted(mods) if n)
                )
            self._handle = mods[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._act = output

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __call__(self, x: torch.Tensor, cls: torch.Tensor):
        """x (B,1,64,64), cls (B,) -> logits, cam (B,h,w), weights (B,C)."""
        with torch.enable_grad():
            if self.layer == "final":
                # forward_with_features() was built on Day 1 exactly for this.
                logits, fmap = self.model.forward_with_features(x)
            else:
                logits = self.model(x)
                fmap = self._act

            selected = logits.gather(1, cls.view(-1, 1)).sum()
            grads = torch.autograd.grad(selected, fmap, retain_graph=False)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)          # (B,C,1,1)
        cam = torch.relu((weights * fmap).sum(dim=1))           # (B,h,w)
        return logits.detach(), cam.detach(), weights.detach().flatten(1)


def normalise_cam(cam: torch.Tensor):
    """Per-sample min-max to [0,1]. Also returns a degeneracy flag.

    A CAM can be all-zero when every weighted channel sum is negative and ReLU
    kills it. Then argmax is meaningless (it returns index 0) and would score
    as a pointing miss for a spurious reason. We count these and report them
    rather than quietly averaging them in.
    """
    B = cam.shape[0]
    flat = cam.reshape(B, -1)
    mx = flat.max(dim=1, keepdim=True).values
    mn = flat.min(dim=1, keepdim=True).values
    degenerate = (mx.squeeze(1) <= 1e-12)
    out = (flat - mn) / (mx - mn + 1e-8)
    return out.reshape(cam.shape), degenerate


def verify_gradcam_is_cam(model, x, cls, tol=1e-4):
    """Assert the Grad-CAM weights equal W[c,:]/(H*W) for the final feature map.

    This is the correctness proof described in the module docstring. It only
    holds for layer='final' (GAP feeds directly into the Linear).
    """
    cammer = GradCAM(model, "final")
    _, cam, weights = cammer(x, cls)

    linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            linear = m  # PCBNet has exactly one, so last-wins is safe
    if linear is None:
        return {"ok": False, "reason": "no nn.Linear found"}

    hw = float(cam.shape[-1] * cam.shape[-2])
    expected = linear.weight.detach()[cls] / hw          # (B, C)
    err = (weights - expected).abs().max().item()
    return {"ok": err < tol, "max_abs_error": err, "feature_map": tuple(cam.shape[1:])}


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------

def to_tensor(patches_u8: np.ndarray, mean: float, std: float, device: str):
    """(N,64,64) uint8 -> (N,1,64,64) standardised float on `device`.

    Identical to the Day 3 CNN path and to what train.py did. If this order is
    wrong the model runs fine and predicts garbage, which is why --verify
    reproduces the known test accuracy before anything else happens.
    """
    t = torch.from_numpy(np.ascontiguousarray(patches_u8)).to(device)
    return t.float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)


# ----------------------------------------------------------------------------
# pointing game
# ----------------------------------------------------------------------------

def run_pointing(model, npz, mean, std, device, layer, batch=256):
    """Grad-CAM every primary-box test patch and record where the peak lands."""
    primary = np.nonzero(npz["box_primary"])[0]
    pidx = npz["box_patch_idx"][primary].astype(np.int64)
    boxes = npz["box_local"][primary].astype(np.int64)      # x1,y1,x2,y2 in 0..63
    bcls = npz["box_class"][primary].astype(np.int64)       # 1..6
    bfrac = npz["box_frac"][primary].astype(np.float32)

    # A patch should have exactly one primary box. Verify rather than assume.
    uniq, counts = np.unique(pidx, return_counts=True)
    dupes = int((counts > 1).sum())

    X = npz["X"]
    y = npz["y"].astype(np.int64)
    assert (y[pidx] == bcls).all(), "primary box class disagrees with patch label"

    cammer = GradCAM(model, layer)
    N = len(pidx)
    peak_y = np.zeros(N, np.int64)
    peak_x = np.zeros(N, np.int64)
    pred = np.zeros(N, np.int64)
    degen = np.zeros(N, bool)

    for i in range(0, N, batch):
        sl = slice(i, min(i + batch, N))
        x = to_tensor(X[pidx[sl]], mean, std, device)
        cls = torch.from_numpy(bcls[sl]).to(device)

        logits, cam, _ = cammer(x, cls)
        cam, d = normalise_cam(cam)
        # bilinear upsample to patch resolution. align_corners=False keeps the
        # 16x block structure honest: we are not inventing sub-block precision,
        # only interpolating between block centres.
        cam64 = F.interpolate(cam.unsqueeze(1), size=(PATCH, PATCH),
                              mode="bilinear", align_corners=False).squeeze(1)

        flat = cam64.reshape(cam64.shape[0], -1).argmax(dim=1).cpu().numpy()
        peak_y[sl] = flat // PATCH
        peak_x[sl] = flat % PATCH
        pred[sl] = logits.argmax(1).cpu().numpy()
        degen[sl] = d.cpu().numpy()

    cammer.close()

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    hit = (peak_x >= x1) & (peak_x <= x2) & (peak_y >= y1) & (peak_y <= y2) & (~degen)
    centre_hit = (x1 <= 32) & (32 <= x2) & (y1 <= 32) & (32 <= y2)
    box_area = np.clip((x2 - x1 + 1) * (y2 - y1 + 1), 1, PATCH * PATCH)
    random_hit = box_area / float(PATCH * PATCH)

    return {
        "patch_idx": pidx, "cls": bcls, "frac": bfrac, "boxes": boxes,
        "peak_y": peak_y, "peak_x": peak_x, "pred": pred, "degenerate": degen,
        "hit": hit, "centre_hit": centre_hit, "random_hit": random_hit,
        "duplicate_primary_patches": dupes,
    }


def local_copper_stats(npz, res, window=5):
    """Investigation (b): what does the neighbourhood of the CAM peak look like?

    copper_frac : fraction of copper pixels in the (2w+1)^2 window at the peak.
                  mousebite (a bite out of a trace edge) should give a mixed
                  neighbourhood, pinhole (a void inside a trace) a
                  copper-dominated one. If the two are indistinguishable, the
                  stable bidirectional confusion is genuinely visual and no
                  amount of loss weighting will fix it.
    edge_dist   : distance from the peak to the nearest copper/substrate
                  boundary, as a second, independent read of the same idea.
    """
    X = npz["X"]
    n = len(res["patch_idx"])
    frac = np.zeros(n, np.float32)
    dist = np.zeros(n, np.float32)

    for i in range(n):
        patch = X[res["patch_idx"][i]]
        copper = (patch < 128).astype(np.uint8)

        py, px = int(res["peak_y"][i]), int(res["peak_x"][i])
        y0, y1 = max(0, py - window), min(PATCH, py + window + 1)
        x0, x1 = max(0, px - window), min(PATCH, px + window + 1)
        frac[i] = copper[y0:y1, x0:x1].mean()

        edges = cv2.morphologyEx(copper, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        if edges.any():
            dt = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
            dist[i] = float(dt[py, px])
        else:
            dist[i] = np.nan

    return frac, dist


# ----------------------------------------------------------------------------
# board-level stitched map
# ----------------------------------------------------------------------------

def render_board_map(model, npz, board_row, pairs, classes, mean, std, device, out_png):
    """Re-tile a full 640x640 board, predict all 100 tiles, and render.

    Note we re-tile from the raw image rather than reading X, because X only
    contains the KEPT patches. Dropped tiles have no row there. Re-tiling gives
    all 100 so the map has no holes, and board_dropped tells us which ones to
    render as 'abstained' rather than as a confident prediction.
    """
    board_id = str(npz["board_ids"][board_row])
    img = cv2.imread(str(pairs[board_row][0]), cv2.IMREAD_GRAYSCALE)

    # (640,640) -> (10,10,64,64): axis0 splits into (grid_y, row), axis1 into
    # (grid_x, col), then transpose to put the two grid axes first.
    tiles = img.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3).reshape(GRID * GRID, PATCH, PATCH)

    with torch.no_grad():
        probs = torch.softmax(model(to_tensor(tiles, mean, std, device)), dim=1).cpu().numpy()

    defect_p = (1.0 - probs[:, 0]).reshape(GRID, GRID)
    pred = probs.argmax(1).reshape(GRID, GRID)
    gt = np.asarray(npz["board_labels"][board_row])
    dropped = np.asarray(npz["board_dropped"][board_row]).astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"board {board_id}")
    axes[0].axis("off")

    im = axes[1].imshow(defect_p, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("predicted 1 - P(good) per tile")
    for gy in range(GRID):
        for gx in range(GRID):
            if dropped[gy, gx]:
                # abstained, not a hole: the tile exists, we chose not to label it
                axes[1].add_patch(plt.Rectangle((gx - .5, gy - .5), 1, 1,
                                                fill=False, hatch="///", ec="cyan", lw=0.8))
            elif pred[gy, gx] != 0:
                axes[1].text(gx, gy, classes[pred[gy, gx]][:4], ha="center", va="center",
                             fontsize=6, color="lime")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow((gt != 0).astype(float), cmap="Greys", vmin=0, vmax=1)
    axes[2].set_title("ground truth tiles, cyan hatch = dropped/abstained")
    for gy in range(GRID):
        for gx in range(GRID):
            if dropped[gy, gx]:
                axes[2].add_patch(plt.Rectangle((gx - .5, gy - .5), 1, 1,
                                                fill=False, hatch="///", ec="cyan", lw=0.8))
            elif gt[gy, gx] != 0:
                axes[2].text(gx, gy, classes[gt[gy, gx]][:4], ha="center", va="center",
                             fontsize=6, color="red")

    for a in axes[1:]:
        a.set_xticks(range(GRID))
        a.set_yticks(range(GRID))
        a.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return board_id


# ----------------------------------------------------------------------------
# qualitative grid
# ----------------------------------------------------------------------------

def render_cam_grid(model, npz, res, classes, mean, std, device, layer, out_png, per_class=3):
    """One row per defect class, showing patch + CAM + ground-truth box.

    Selection is deliberate and stated in the title: correctly classified, with
    the highest box_frac. These are the model's best case. Do not present them
    as typical, the pointing score is what is typical.
    """
    cammer = GradCAM(model, layer)
    rows = []
    for c in range(1, len(classes)):
        cand = np.nonzero((res["cls"] == c) & (res["pred"] == c))[0]
        if len(cand) == 0:
            cand = np.nonzero(res["cls"] == c)[0]
        cand = cand[np.argsort(-res["frac"][cand])][:per_class]
        rows.append((c, cand))

    fig, axes = plt.subplots(len(rows), per_class * 2, figsize=(3.0 * per_class * 2, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (c, cand) in enumerate(rows):
        for j in range(per_class):
            ax_img = axes[r, j * 2]
            ax_cam = axes[r, j * 2 + 1]
            if j >= len(cand):
                ax_img.axis("off")
                ax_cam.axis("off")
                continue
            i = cand[j]
            patch = npz["X"][res["patch_idx"][i]]
            x = to_tensor(patch[None], mean, std, device)
            _, cam, _ = cammer(x, torch.tensor([c], device=device))
            cam, _ = normalise_cam(cam)
            cam64 = F.interpolate(cam.unsqueeze(1), size=(PATCH, PATCH),
                                  mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

            x1, y1, x2, y2 = res["boxes"][i]
            for ax, base in ((ax_img, None), (ax_cam, cam64)):
                ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
                if base is not None:
                    ax.imshow(base, cmap="jet", alpha=0.45)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, ec="lime", lw=1.5))
                ax.plot([res["peak_x"][i]], [res["peak_y"][i]], "w+", ms=10, mew=2)
                ax.axis("off")
            ax_img.set_title(f"{classes[c]} frac={res['frac'][i]:.2f}", fontsize=8)
            ax_cam.set_title("CAM, + = peak", fontsize=8)

    cammer.close()
    fig.suptitle(f"Grad-CAM ({layer}), best case: correctly classified, highest box_frac", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Day 4 Grad-CAM, pointing game, board map")
    ap.add_argument("--patches", default="data/patches")
    ap.add_argument("--raw", default="data/raw/PCBData")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ckpt", default="artifacts/r4_weighted_p05_registered_best.pt")
    ap.add_argument("--layers", default="final",
                    help="comma separated, e.g. 'final,features.2'")
    ap.add_argument("--boards", type=int, default=2, help="how many board maps to render")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="day4_gradcam")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = Path(args.artifacts)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    classes = load_classes(artifacts)
    mean, std = load_norm(artifacts)
    model = load_pcbnet(args.ckpt, num_classes=len(classes), device=device)

    npz = np.load(Path(args.patches) / "test.npz", allow_pickle=True)
    y = npz["y"].astype(np.int64)

    lines = []

    def say(msg=""):
        print(msg)
        lines.append(msg)

    say("=" * 78)
    say("DAY 4: GRAD-CAM, POINTING GAME, BOARD MAP")
    say("=" * 78)
    say(f"device {device} | checkpoint {Path(args.ckpt).name}")
    say(f"norm mean {mean} std {std}")
    say("")

    # ---- 0. verify preprocessing reproduces Day 2 ---------------------------
    say("STEP 0: PREPROCESSING VERIFICATION")
    with torch.no_grad():
        preds = np.zeros(len(y), np.int64)
        for i in range(0, len(y), 1024):
            x = to_tensor(npz["X"][i:i + 1024], mean, std, device)
            preds[i:i + 1024] = model(x).argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    say(f"  reproduced test accuracy {acc:.4f}   (Day 2 recorded 0.9529)")
    if abs(acc - 0.9529) > 0.01:
        say("  MISMATCH. Normalisation or checkpoint is wrong. Everything below is")
        say("  meaningless until this matches. Stop here.")
    say("")

    # ---- 1. Grad-CAM == CAM proof ------------------------------------------
    say("STEP 1: GRAD-CAM == CAM CORRECTNESS PROOF (final layer only)")
    probe_x = to_tensor(npz["X"][:8], mean, std, device)
    probe_c = torch.from_numpy(y[:8]).to(device)
    chk = verify_gradcam_is_cam(model, probe_x, probe_c)
    say(f"  feature map {chk.get('feature_map')} | max |grad_weight - W/(H*W)| = "
        f"{chk.get('max_abs_error'):.3e} | pass={chk['ok']}")
    say("  PCBNet is GAP -> single Linear, so d logit_c / d fmap[k,i,j] is the")
    say("  constant W[c,k]/(H*W). Grad-CAM therefore reduces exactly to CAM.")
    say("  A pass here proves the hook, the gradient path and the class indexing")
    say("  are all correct, which is otherwise a silent failure mode.")
    say("")

    # ---- 2. pointing game, per layer ---------------------------------------
    all_res = {}
    for layer in [s.strip() for s in args.layers.split(",") if s.strip()]:
        say("=" * 78)
        say(f"STEP 2: POINTING GAME, layer = {layer}")
        res = run_pointing(model, npz, mean, std, device, layer)
        all_res[layer] = res

        n = len(res["cls"])
        say(f"  patches evaluated {n} (rows with box_primary True)")
        say(f"  degenerate all-zero CAMs {int(res['degenerate'].sum())} (counted as misses)")
        if res["duplicate_primary_patches"]:
            say(f"  WARNING {res['duplicate_primary_patches']} patches have >1 primary box")
        say("")
        say(f"  {'class':<11}{'n':>7}{'pointing':>10}{'random':>9}{'centre':>9}{'top1':>8}")
        for c in range(1, len(classes)):
            sel = res["cls"] == c
            k = int(sel.sum())
            if k == 0:
                continue
            say(f"  {classes[c]:<11}{k:>7}"
                f"{res['hit'][sel].mean():>10.4f}"
                f"{res['random_hit'][sel].mean():>9.4f}"
                f"{res['centre_hit'][sel].mean():>9.4f}"
                f"{(res['pred'][sel] == c).mean():>8.4f}")
        say(f"  {'ALL':<11}{n:>7}"
            f"{res['hit'].mean():>10.4f}"
            f"{res['random_hit'].mean():>9.4f}"
            f"{res['centre_hit'].mean():>9.4f}"
            f"{(res['pred'] == res['cls']).mean():>8.4f}")
        say("")
        correct = res["pred"] == res["cls"]
        if correct.any():
            say(f"  pointing on CORRECTLY classified only: {res['hit'][correct].mean():.4f} "
                f"(n={int(correct.sum())})")
        say("")

        # ---- investigation (a): does box_frac predict failure? -------------
        say("  INVESTIGATION (a): pointing and top-1 accuracy vs box_frac")
        say("  Tests whether --min-frac 0.25 is injecting label noise. If both")
        say("  columns climb with box_frac, low-frac patches carry a defect label")
        say("  over a few pixels of defect and the label, not the model, is wrong.")
        edges = [0.25, 0.40, 0.60, 0.80, 1.01]
        say(f"    {'box_frac bin':<16}{'n':>7}{'pointing':>10}{'top1':>8}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (res["frac"] >= lo) & (res["frac"] < hi)
            k = int(sel.sum())
            if k == 0:
                continue
            say(f"    [{lo:.2f}, {hi:.2f}){'':<4}{k:>7}"
                f"{res['hit'][sel].mean():>10.4f}"
                f"{(res['pred'][sel] == res['cls'][sel]).mean():>8.4f}")
        say("")

    # ---- 3. investigation (b), mousebite vs pinhole -------------------------
    layer0 = list(all_res)[0]
    res = all_res[layer0]
    say("=" * 78)
    say(f"STEP 3: INVESTIGATION (b), where does the peak sit? (layer {layer0})")
    say("  Hypothesis: mousebite peaks land on a trace EDGE (mixed neighbourhood),")
    say("  pinhole peaks land inside a trace (copper-dominated neighbourhood).")
    say("  If the two are indistinguishable, the stable ~0.10 bidirectional")
    say("  confusion is genuinely visual and loss weighting was never going to fix it.")
    frac, dist = local_copper_stats(npz, res, window=5)
    say(f"  {'class':<11}{'n':>7}{'copper_frac@peak':>18}{'dist_to_edge':>14}")
    for c in range(1, len(classes)):
        sel = res["cls"] == c
        if not sel.any():
            continue
        say(f"  {classes[c]:<11}{int(sel.sum()):>7}"
            f"{np.nanmean(frac[sel]):>18.4f}{np.nanmean(dist[sel]):>14.4f}")
    say("")

    # ---- 4. figures ---------------------------------------------------------
    say("=" * 78)
    say("STEP 4: FIGURES")

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / (len(all_res) + 2)
    xs = np.arange(1, len(classes))
    for k, (layer, r) in enumerate(all_res.items()):
        vals = [r["hit"][r["cls"] == c].mean() for c in xs]
        ax.bar(xs + k * width, vals, width, label=f"Grad-CAM ({layer})")
    r0 = all_res[layer0]
    ax.bar(xs + len(all_res) * width, [r0["random_hit"][r0["cls"] == c].mean() for c in xs],
           width, label="random pixel", color="grey")
    ax.bar(xs + (len(all_res) + 1) * width, [r0["centre_hit"][r0["cls"] == c].mean() for c in xs],
           width, label="always centre", color="lightgrey")
    ax.set_xticks(xs + 0.4 - width)
    ax.set_xticklabels([classes[c] for c in xs], rotation=20)
    ax.set_ylabel("pointing game hit rate")
    ax.set_title("Day 4: pointing game vs baselines, per class")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = figures / f"{args.tag}_pointing_by_class.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    say(f"  {p}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    edges = [0.25, 0.40, 0.60, 0.80, 1.01]
    centres = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    for layer, r in all_res.items():
        pt = [r["hit"][(r["frac"] >= a) & (r["frac"] < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax[0].plot(centres, pt, "o-", label=layer)
    t1 = [(r0["pred"][(r0["frac"] >= a) & (r0["frac"] < b)] ==
           r0["cls"][(r0["frac"] >= a) & (r0["frac"] < b)]).mean()
          for a, b in zip(edges[:-1], edges[1:])]
    ax[1].plot(centres, t1, "s-", color="firebrick")
    ax[0].set_title("pointing accuracy vs box_frac")
    ax[1].set_title("top-1 accuracy vs box_frac  (label-noise probe)")
    for a in ax:
        a.set_xlabel("box_frac"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    p = figures / f"{args.tag}_vs_boxfrac.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    say(f"  {p}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [frac[res["cls"] == c] for c in range(1, len(classes))]
    ax.boxplot(data, labels=[classes[c] for c in range(1, len(classes))], showfliers=False)
    ax.set_ylabel("copper fraction in 11x11 window at CAM peak")
    ax.set_title("Investigation (b): mousebite (edge) vs pinhole (interior)")
    plt.xticks(rotation=20)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    p = figures / f"{args.tag}_peak_neighbourhood.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    say(f"  {p}")

    p = figures / f"{args.tag}_grid_{layer0.replace('.', '_')}.png"
    render_cam_grid(model, npz, res, classes, mean, std, device, layer0, p)
    say(f"  {p}")

    index = index_boards(args.raw)
    pairs = resolve(npz["board_ids"], index)
    counts = (np.asarray(npz["board_labels"]) != 0).reshape(len(pairs), -1).sum(1)
    for b in np.argsort(-counts)[:args.boards]:
        bid = str(npz["board_ids"][b])
        p = figures / f"{args.tag}_board_{bid.replace('/', '_')}.png"
        render_board_map(model, npz, int(b), pairs, classes, mean, std, device, p)
        say(f"  {p}")

    say("")
    say("KNOWN RESOLUTION CEILING")
    say("  The final feature map is 4x4 for a 64x64 input, so the 'final' CAM is a")
    say("  16x upsample and localises to roughly a 16x16 region. Any apparent")
    say("  pixel precision in the heatmaps is bilinear interpolation, not evidence.")
    say("  This belongs in the README stated plainly, not implied away.")

    report = artifacts / f"{args.tag}_report.txt"
    report.write_text("\n".join(lines) + "\n")

    summary = {layer: {
        "pointing": float(r["hit"].mean()),
        "random_baseline": float(r["random_hit"].mean()),
        "centre_baseline": float(r["centre_hit"].mean()),
        "per_class": {classes[c]: float(r["hit"][r["cls"] == c].mean())
                      for c in range(1, len(classes)) if (r["cls"] == c).any()},
    } for layer, r in all_res.items()}
    (artifacts / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
