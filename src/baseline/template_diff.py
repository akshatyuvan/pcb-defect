#!/usr/bin/env python3
"""
Day 3 classical baseline: OpenCV template differencing.

THE QUESTION THIS ANSWERS
  "Did you actually need a neural network?" DeepPCB gives a defect-free
  template for every board, so the cheapest detector is align-subtract-clean.
  If that beats PCBNet, the CNN is decoration and I should say so.

FOUR GUARANTEES THAT MAKE THE COMPARISON VALID
  1. Board set. Test board ids come straight out of test.npz `board_ids`.
     We do not re-derive the 850/150/500 split. Re-deriving means trusting
     that two code paths agree; reading the ids the CNN was actually scored on
     makes disagreement structurally impossible.
  2. Patch set. Scores are gathered with (board_idx, grid_y, grid_x) from the
     same npz, so the 2,658 dropped ambiguous tiles are excluded here exactly
     as they were for the CNN. 47,342 patches, not 50,000.
  3. Score semantics. The CNN's binary defect score is 1 - P(good), in [0,1],
     higher = more defect-like. Ours is (surviving difference pixels in the
     tile) / 4096, also in [0,1], also higher = more defect-like. Both go
     through the same choose_operating_point() rule at target recall 0.97.
  4. Hyperparameter discipline. Kernel size and min-area are swept on the 150
     VAL boards and then frozen. Test is touched once. Same rule the CNN got.

POLARITY (Day 1 measurement: black = copper, white = substrate)
  Copper is LOW intensity. Two signed error types, and they are not
  interchangeable:
      extra copper   = copper here, none in template  -> short, spur, copper
      missing copper = none here, copper in template  -> open, mousebite, pinhole
  We carry both all the way to the tile grid, which gives a free polarity
  accuracy: the only class-ish signal a differencing baseline can produce.

RUNS ON: Colab, CPU runtime is sufficient. No GPU unless --cnn-ckpt is passed,
and even then CPU inference over 47k 64x64 patches takes a couple of minutes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # no display on Colab; write PNGs directly
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.data.paths import index_boards, resolve
from src.models.checkpoint import load_classes

GRID = 10          # 10x10 tiles per board
PATCH = 64         # 64x64 pixels per tile
TILE_PIXELS = PATCH * PATCH

EXTRA_COPPER = {"short", "spur", "copper"}
MISSING_COPPER = {"open", "mousebite", "pinhole"}


# ----------------------------------------------------------------------------
# image ops
# ----------------------------------------------------------------------------

def copper_mask(img: np.ndarray) -> np.ndarray:
    """1 where copper, 0 where substrate.

    Threshold at 128 rather than Otsu: Day 1 proved these images are already
    binary (observed std 0.4760 against the 0.4783 predicted by sqrt(p(1-p))
    for p=0.647, agreeing to 3 decimals). Otsu would add a per-image decision
    for no benefit and would behave unpredictably on a board that happens to be
    almost all copper.
    """
    return (img < 128).astype(np.uint8)


def align_shift(a: np.ndarray, b: np.ndarray, radius: int = 4) -> tuple[int, int, int]:
    """Brute-force integer translation search: find (dy, dx) minimising the
    number of disagreeing pixels between test mask `a` and template mask `b`.

    Why brute force and not phase correlation / ECC: the search space is tiny
    (81 candidates at radius 4), the images are binary so the cost function is
    a single np.count_nonzero, and the whole thing is ~15ms per board. Fancier
    subpixel methods would need interpolation, which would un-binarise the
    image and reintroduce exactly the halo we are trying to avoid.

    Convention: aligned_b[y, x] = b[y + dy, x + dx].
    Only the interior (cropped by `radius`) is compared so every candidate
    shift is evaluated on the same number of pixels.
    """
    m = radius
    H, W = a.shape
    A = a[m:H - m, m:W - m]

    best, bdy, bdx = None, 0, 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            B = b[m + dy:H - m + dy, m + dx:W - m + dx]
            cost = int(np.count_nonzero(A != B))
            if best is None or cost < best:
                best, bdy, bdx = cost, dy, dx
    return bdy, bdx, best


def signed_diff(test_img: np.ndarray, temp_img: np.ndarray, radius: int = 4):
    """Return (extra_copper_mask, missing_copper_mask, (dy, dx)).

    The border band of width `radius` is zeroed because np.roll wraps around.
    Cost of that choice: a defect touching the outer 4 pixels of a 640px board
    is invisible to the baseline. That is 1.2% of the area and DeepPCB defects
    sit on traces, not the board edge, so it is a fair trade for not having to
    special-case the boundary. Worth stating in the README as a known blind
    spot rather than pretending it does not exist.
    """
    a = copper_mask(test_img)
    b = copper_mask(temp_img)

    if radius > 0:
        dy, dx, _ = align_shift(a, b, radius)
        b_aligned = np.roll(b, shift=(-dy, -dx), axis=(0, 1)) if (dy or dx) else b
    else:
        dy = dx = 0
        b_aligned = b

    extra = ((a == 1) & (b_aligned == 0)).astype(np.uint8)
    missing = ((a == 0) & (b_aligned == 1)).astype(np.uint8)

    if radius > 0:
        for m in (extra, missing):
            m[:radius, :] = 0
            m[-radius:, :] = 0
            m[:, :radius] = 0
            m[:, -radius:] = 0

    return extra, missing, (dy, dx)


def clean_mask(mask: np.ndarray, kernel: int, min_area: int) -> np.ndarray:
    """Two-stage noise removal, the baseline's only hyperparameters.

    Stage 1, morphological OPEN (erode then dilate) with an elliptical kernel:
    removes structures thinner than the kernel. This is what kills the 1-2px
    halo along every trace edge left over from imperfect registration.
    Stage 2, connected-component area filter: removes small blobs that survived
    opening because they are compact rather than thin (e.g. a 3x3 speck).

    They do different jobs, which is why both are swept instead of just one.
    """
    out = mask
    if kernel and kernel >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)
    if min_area and min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        if n > 1:
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        out = keep[labels].astype(np.uint8)
    return out


def tile_counts(mask: np.ndarray) -> np.ndarray:
    """640x640 mask -> 10x10 grid of surviving-pixel counts.

    reshape(10,64,10,64) splits axis 0 into (grid_y, row_within_tile) and axis 1
    into (grid_x, col_within_tile), so summing axes 1 and 3 gives per-tile
    totals. This is the same tiling build_patches.py used (640/64 = 10 exactly,
    no remainder, no overlap), which is why grid_y/grid_x index into it directly.
    """
    return mask.reshape(GRID, PATCH, GRID, PATCH).sum(axis=(1, 3)).astype(np.float32)


# ----------------------------------------------------------------------------
# per-split scoring
# ----------------------------------------------------------------------------

def board_diff_masks(pairs, radius, cache=None, verbose=True):
    """Compute (extra, missing) masks for every board. Cached so a 12-config
    sweep pays the alignment cost once instead of twelve times.

    Memory: 150 val boards x 2 masks x 640x640 uint8 = ~123MB. Fine in Colab's
    12GB. Do NOT cache the 500 test boards (~410MB) since test is scored once.
    """
    out = []
    t0 = time.time()
    for i, (test_path, temp_path) in enumerate(pairs):
        key = str(test_path)
        if cache is not None and key in cache:
            out.append(cache[key])
            continue
        test_img = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
        temp_img = cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE)
        if test_img is None or temp_img is None:
            raise RuntimeError(f"failed to read {test_path} / {temp_path}")
        extra, missing, _ = signed_diff(test_img, temp_img, radius)
        if cache is not None:
            cache[key] = (extra, missing)
        out.append((extra, missing))
        if verbose and (i + 1) % 100 == 0:
            print(f"  diffed {i+1}/{len(pairs)} boards ({time.time()-t0:.1f}s)")
    return out


def grids_from_masks(masks, kernel, min_area):
    """Apply cleanup at one config and reduce to per-board 10x10 count grids."""
    B = len(masks)
    extra_g = np.zeros((B, GRID, GRID), np.float32)
    miss_g = np.zeros((B, GRID, GRID), np.float32)
    for i, (extra, missing) in enumerate(masks):
        extra_g[i] = tile_counts(clean_mask(extra, kernel, min_area))
        miss_g[i] = tile_counts(clean_mask(missing, kernel, min_area))
    return extra_g, miss_g


def gather_patch_scores(npz, extra_g, miss_g):
    """Pull the per-patch score out of the per-board grids.

    This single line is guarantee #2: we index with the npz's own
    (board_idx, grid_y, grid_x), so we score exactly the patches the CNN was
    scored on and nothing else.
    """
    bi = npz["board_idx"].astype(np.int64)
    gy = npz["grid_y"].astype(np.int64)
    gx = npz["grid_x"].astype(np.int64)
    extra = extra_g[bi, gy, gx]
    miss = miss_g[bi, gy, gx]
    score = (extra + miss) / float(TILE_PIXELS)
    return score, extra, miss


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def choose_operating_point(y_bin: np.ndarray, scores: np.ndarray, target_recall: float = 0.97):
    """Highest threshold whose recall still meets the target, with a >= rule.

    Reimplemented here rather than imported from src/train.py so this module
    has no training-time dependency. The notebook verifies it reproduces the
    CNN's known 0.1106 precision at recall 0.97 to prove the two are equivalent.
    If that check fails, the comparison is void and you should stop.

    Tie handling matters: if several patches share a score, a `>=` decision rule
    admits all of them, so we advance to the last index of the tied block before
    reporting precision/recall. Without this the reported precision would be
    optimistic relative to what the threshold actually does at inference time.
    """
    order = np.argsort(-scores, kind="mergesort")  # stable, so ties keep input order
    s = scores[order]
    y = y_bin[order].astype(np.int64)

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = int(y.sum())
    if P == 0:
        return None

    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)

    ok = np.nonzero(recall >= target_recall)[0]
    if ok.size == 0:
        return None

    i = int(ok[0])
    while i + 1 < s.size and s[i + 1] == s[i]:
        i += 1

    return {
        "threshold": float(s[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "n_flagged": int(i + 1),
        "tp": int(tp[i]),
        "fp": int(fp[i]),
    }


def detection_ceiling(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """Max recall the baseline can ever reach: the fraction of defect patches
    with a strictly non-zero score. A tile whose difference was entirely wiped
    out by cleanup is invisible at ANY threshold. The CNN has no equivalent
    ceiling because its score is continuous everywhere.
    """
    return float((scores[y_bin == 1] > 0).mean())


def polarity_report(y: np.ndarray, extra: np.ndarray, miss: np.ndarray,
                    flagged: np.ndarray, classes: list[str]):
    """Among flagged defect patches, does the SIGN of the difference match the
    physics of the labelled class?

    open/mousebite/pinhole should be missing-copper, short/spur/copper should be
    extra-copper. Chance is ~50% because DeepPCB has 3 classes on each side in
    near-equal numbers, so anything meaningfully above 50% is real signal that
    the baseline gets for free without a single learned parameter.
    """
    rows, correct_all, n_all = [], 0, 0
    for c in range(1, len(classes)):
        name = classes[c]
        sel = (y == c) & flagged
        n = int(sel.sum())
        if n == 0:
            rows.append((name, 0, float("nan")))
            continue
        pred_extra = extra[sel] > miss[sel]
        want_extra = name in EXTRA_COPPER
        hit = int((pred_extra == want_extra).sum())
        rows.append((name, n, hit / n))
        correct_all += hit
        n_all += n
    overall = correct_all / n_all if n_all else float("nan")
    return rows, overall, n_all


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def plot_pr(out_png, base_y, base_s, base_op, cnn=None, target=0.97):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    p, r, _ = precision_recall_curve(base_y, base_s)
    ap = average_precision_score(base_y, base_s)
    ax.plot(r, p, lw=2, label=f"template diff (AP={ap:.3f})")

    if cnn is not None:
        cp, cr, _ = precision_recall_curve(cnn["y"], cnn["scores"])
        cap = average_precision_score(cnn["y"], cnn["scores"])
        ax.plot(cr, cp, lw=2, label=f"PCBNet (AP={cap:.3f})")
        if cnn.get("op"):
            ax.scatter([cnn["op"]["recall"]], [cnn["op"]["precision"]], s=70,
                       marker="s", zorder=5,
                       label=f"PCBNet @R={target}: P={cnn['op']['precision']:.3f}")

    prevalence = float(base_y.mean())
    ax.axhline(prevalence, ls=":", c="grey", lw=1,
               label=f"random classifier (P={prevalence:.3f})")
    ax.axvline(target, ls="--", c="k", lw=1, alpha=.5)

    if base_op:
        ax.scatter([base_op["recall"]], [base_op["precision"]], s=70, marker="o", zorder=5,
                   label=f"template diff @R={target}: P={base_op['precision']:.3f}")

    ax.set_xlabel("binary defect recall")
    ax.set_ylabel("precision")
    ax.set_title("Day 3: classical baseline vs CNN, identical test patches")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_examples(out_png, pairs, masks, npz, kernel, min_area, n_boards=3):
    """Manual-eyeball figure. Columns: test, template, extra copper, missing
    copper, tile score with ground-truth tiles outlined."""
    labels = npz["board_labels"]
    counts = (labels != 0).reshape(labels.shape[0], -1).sum(1)
    picks = np.argsort(-counts)[:n_boards]

    fig, axes = plt.subplots(len(picks), 5, figsize=(19, 4.0 * len(picks)))
    axes = np.atleast_2d(axes)

    for row, b in enumerate(picks):
        test_img = cv2.imread(str(pairs[b][0]), cv2.IMREAD_GRAYSCALE)
        temp_img = cv2.imread(str(pairs[b][1]), cv2.IMREAD_GRAYSCALE)
        extra = clean_mask(masks[b][0], kernel, min_area)
        missing = clean_mask(masks[b][1], kernel, min_area)
        grid = (tile_counts(extra) + tile_counts(missing)) / TILE_PIXELS

        panels = [
            (test_img, "test board", "gray"),
            (temp_img, "template (defect free)", "gray"),
            (extra * 255, "extra copper", "gray"),
            (missing * 255, "missing copper", "gray"),
        ]
        for col, (im, title, cmap) in enumerate(panels):
            axes[row, col].imshow(im, cmap=cmap, vmin=0, vmax=255)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

        ax = axes[row, 4]
        ax.imshow(grid, cmap="magma")
        ax.set_title("tile score, red box = ground truth defect tile", fontsize=9)
        for gy in range(GRID):
            for gx in range(GRID):
                if labels[b, gy, gx] != 0:
                    ax.add_patch(plt.Rectangle((gx - .5, gy - .5), 1, 1,
                                               fill=False, ec="red", lw=1.6))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# optional CNN scoring, for the overlay and the equivalence check
# ----------------------------------------------------------------------------

def score_cnn(ckpt, artifacts, npz, device="cpu", batch=512):
    """Run PCBNet over the same test patches and return 1 - P(good).

    Lives here rather than in a notebook cell because notebooks are thin
    drivers (locked decision 14) and because the FastAPI service on Day 5 will
    need this exact preprocessing path.
    """
    import torch
    from src.models.checkpoint import load_norm, load_pcbnet

    mean, std = load_norm(artifacts)
    classes = load_classes(artifacts)
    model = load_pcbnet(ckpt, num_classes=len(classes), device=device)

    X = npz["X"]
    probs = np.zeros((X.shape[0], len(classes)), np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            chunk = torch.from_numpy(X[i:i + batch]).to(device)
            # uint8 -> [0,1] -> standardised. If this order is wrong the model
            # still runs and predicts garbage, which is why the caller must
            # check the reproduced accuracy against the known 0.9529.
            chunk = chunk.float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)
            probs[i:i + batch] = torch.softmax(model(chunk), dim=1).cpu().numpy()

    return probs, 1.0 - probs[:, 0]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Day 3 classical template-differencing baseline")
    ap.add_argument("--patches", default="data/patches")
    ap.add_argument("--raw", default="data/raw/PCBData")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--target-recall", type=float, default=0.97)
    ap.add_argument("--align-radius", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=None, help="skip the sweep and use this")
    ap.add_argument("--min-area", type=int, default=None, help="skip the sweep and use this")
    ap.add_argument("--cnn-ckpt", default=None, help="optional, overlays PCBNet on the PR plot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="day3_baseline")
    args = ap.parse_args()

    patches = Path(args.patches)
    artifacts = Path(args.artifacts)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    classes = load_classes(artifacts)
    print(f"[classes] {classes}")

    val = np.load(patches / "val.npz", allow_pickle=True)
    test = np.load(patches / "test.npz", allow_pickle=True)
    index = index_boards(args.raw)
    print(f"[data] indexed {len(index)} boards on disk")

    val_pairs = resolve(val["board_ids"], index)
    test_pairs = resolve(test["board_ids"], index)
    print(f"[data] val boards {len(val_pairs)} | test boards {len(test_pairs)}")
    print(f"[data] val patches {len(val['y'])} | test patches {len(test['y'])}")

    lines = []

    def say(msg=""):
        print(msg)
        lines.append(msg)

    say("=" * 78)
    say("DAY 3: CLASSICAL BASELINE, OPENCV TEMPLATE DIFFERENCING")
    say("=" * 78)
    say(f"val boards {len(val_pairs)} / val patches {len(val['y'])}")
    say(f"test boards {len(test_pairs)} / test patches {len(test['y'])}")
    say(f"align radius {args.align_radius}px, target recall {args.target_recall}")
    say("")

    # ---- sweep on VAL -------------------------------------------------------
    if args.kernel is None or args.min_area is None:
        say("HYPERPARAMETER SWEEP (selected on VAL average precision, test untouched)")
        say(f"{'kernel':>7} {'min_area':>9} {'val AP':>9} {'ceiling':>9} {'P@R=%.2f' % args.target_recall:>10}")
        cache = {}
        val_masks = board_diff_masks(val_pairs, args.align_radius, cache=cache)
        y_val_bin = (val["y"] != 0).astype(np.int64)

        best = None
        for kernel in (0, 3, 5, 7):
            for min_area in (0, 10, 25):
                eg, mg = grids_from_masks(val_masks, kernel, min_area)
                s, _, _ = gather_patch_scores(val, eg, mg)
                ap_val = float(average_precision_score(y_val_bin, s))
                ceil = detection_ceiling(y_val_bin, s)
                op = choose_operating_point(y_val_bin, s, args.target_recall)
                p_at = op["precision"] if op else float("nan")
                say(f"{kernel:>7} {min_area:>9} {ap_val:>9.4f} {ceil:>9.4f} {p_at:>10.4f}")
                if best is None or ap_val > best[0]:
                    best = (ap_val, kernel, min_area)

        _, kernel, min_area = best
        del val_masks, cache
        say("")
        say(f"SELECTED on val AP: kernel={kernel}, min_area={min_area}")
        say("Selection criterion is val average precision, not precision at the")
        say("recall target, because at extreme recall the metric is dominated by")
        say("the tail and is very noisy. AP summarises the whole curve.")
    else:
        kernel, min_area = args.kernel, args.min_area
        say(f"SWEEP SKIPPED, using kernel={kernel}, min_area={min_area}")
    say("")

    # ---- evaluate on TEST ---------------------------------------------------
    say("TEST EVALUATION (500 boards, 47,342-patch set, identical to Day 2)")
    test_masks = board_diff_masks(test_pairs, args.align_radius)
    eg, mg = grids_from_masks(test_masks, kernel, min_area)
    score, extra, miss = gather_patch_scores(test, eg, mg)

    y = test["y"].astype(np.int64)
    y_bin = (y != 0).astype(np.int64)

    ap_test = float(average_precision_score(y_bin, score))
    ceil = detection_ceiling(y_bin, score)
    op = choose_operating_point(y_bin, score, args.target_recall)

    say(f"  average precision      {ap_test:.4f}")
    say(f"  detection ceiling      {ceil:.4f}  (max reachable recall, score > 0)")
    say(f"  prevalence             {y_bin.mean():.4f}  (a random classifier's precision)")
    if op:
        say(f"  P @ R={args.target_recall}          {op['precision']:.4f} "
            f"(threshold {op['threshold']:.6f}, {op['tp']} TP / {op['fp']} FP)")
    else:
        say(f"  P @ R={args.target_recall}          UNREACHABLE. The baseline cannot")
        say(f"    reach this recall at any threshold because its ceiling is {ceil:.4f}.")
        say("    This is a categorical failure, not a tuning problem: those defect")
        say("    tiles produce literally zero surviving difference pixels.")
    say("")

    say("NUMBERS TO BEAT (Day 2, same patches, same rule)")
    say("  PCBNet r4_weighted_p05_registered : P=0.1106 @ R=0.97")
    say("  PCBNet r1_unweighted              : P=0.1775 @ R=0.97")
    say("")

    say("BASELINE RECALL BY DEFECT CLASS (at the chosen threshold)")
    thr = op["threshold"] if op else 0.0
    flagged = score >= thr if op else score > 0
    for c in range(1, len(classes)):
        sel = y == c
        n = int(sel.sum())
        rec = float(flagged[sel].mean()) if n else float("nan")
        ceil_c = float((score[sel] > 0).mean()) if n else float("nan")
        say(f"  {classes[c]:<10} n={n:>6}  recall={rec:.4f}  ceiling={ceil_c:.4f}")
    say("")

    rows, overall, n_pol = polarity_report(y, extra, miss, flagged, classes)
    say("POLARITY ACCURACY (does the sign of the difference match the class physics?)")
    say("  chance is ~0.50: DeepPCB has 3 extra-copper and 3 missing-copper classes")
    for name, n, acc in rows:
        say(f"  {name:<10} n={n:>6}  polarity acc={acc:.4f}")
    say(f"  OVERALL    n={n_pol:>6}  polarity acc={overall:.4f}")
    say("")

    # ---- optional CNN overlay ----------------------------------------------
    cnn = None
    if args.cnn_ckpt:
        say("CNN OVERLAY + EQUIVALENCE CHECK")
        probs, cnn_scores = score_cnn(args.cnn_ckpt, artifacts, test, device=args.device)
        pred = probs.argmax(1)
        acc = float((pred == y).mean())
        cnn_op = choose_operating_point(y_bin, cnn_scores, args.target_recall)
        say(f"  reproduced test accuracy {acc:.4f}   (Day 2 recorded 0.9529)")
        if cnn_op:
            say(f"  reproduced P @ R={args.target_recall}     {cnn_op['precision']:.4f} "
                f"(Day 2 recorded 0.1106), threshold {cnn_op['threshold']:.6f}")
        say("  If those two do not match Day 2, the preprocessing or the operating-")
        say("  point rule differs and the whole comparison below is void.")
        say("")
        cnn = {"y": y_bin, "scores": cnn_scores, "op": cnn_op}

    # ---- figures ------------------------------------------------------------
    pr_png = figures / f"{args.tag}_pr.png"
    ex_png = figures / f"{args.tag}_examples.png"
    plot_pr(pr_png, y_bin, score, op, cnn=cnn, target=args.target_recall)
    plot_examples(ex_png, test_pairs, test_masks, test, kernel, min_area)
    say(f"figures: {pr_png}")
    say(f"figures: {ex_png}")

    report = artifacts / f"{args.tag}_report.txt"
    report.write_text("\n".join(lines) + "\n")

    (artifacts / f"{args.tag}_config.json").write_text(json.dumps({
        "kernel": int(kernel), "min_area": int(min_area),
        "align_radius": int(args.align_radius),
        "target_recall": float(args.target_recall),
        "test_ap": ap_test, "test_ceiling": ceil,
        "test_precision_at_target": (op["precision"] if op else None),
        "test_threshold": (op["threshold"] if op else None),
        "polarity_accuracy": overall,
    }, indent=2))
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
