"""
Load a trained PCBNet from a .pt checkpoint, and load the preprocessing
constants that must accompany it.

WHY THIS FILE EXISTS
  Day 3 (PR overlay), Day 4 (Grad-CAM), and Day 5 (FastAPI) all need the same
  three things: the weights, the class list, and the normalisation constants.
  Putting the three loads in one place is the cheapest way to make forgetting
  norm.json impossible - without it the service runs fine and predicts garbage.

WHY THE FALLBACKS WERE REMOVED (Day 9)
  This file used to hold CLASSES_FALLBACK and NORM_FALLBACK, and to print a
  [warn] and return them when a file was missing or unparseable. That is the
  worst possible failure mode for this project, for two reasons:

    1. A defaulted normalisation constant does not raise. It shifts every
       input by a few hundredths and quietly degrades a model you then trust.
       It is the same failure class as a train/serve decoder mismatch - which
       is why Day 5 wrote an explicit parity check rather than assuming.
    2. A [warn] on stdout is invisible in a container. The staged bundle is
       COPY'd into the image at build time, so a mis-staged bundle would have
       started, logged one line nobody reads, and served wrong labels forever.

  A FileNotFoundError at startup is loud, immediate, and cheap to fix. Wrong
  numbers with a clean exit code are none of those things.

  The tolerant multi-format parsing went with it. classes.json is a bare JSON
  array and norm.json is {"mean": float, "std": float}, because that is what
  src/mlops/stage_model.py writes. Accepting four other shapes was defending
  against a producer that does not exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_classes(artifacts: str | Path) -> list[str]:
    """Read <artifacts>/classes.json -> list of class names, index-ordered.

    Takes a DIRECTORY, not a file path. Raises rather than defaulting: see the
    module docstring. Ordering is load-bearing - annotation class ids 1-6 map
    to the six defect names in order, with good at 0. A rotated list produces
    identical accuracy and completely wrong per-class metrics.
    """
    path = Path(artifacts) / "classes.json"
    if not path.exists():
        raise FileNotFoundError(f"classes.json missing from staged bundle: {path}")

    obj = json.loads(path.read_text())
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"classes.json must be a non-empty JSON array, got {type(obj).__name__}: {path}")
    if not all(isinstance(c, str) for c in obj):
        raise ValueError(f"classes.json must contain only strings: {path}")
    return list(obj)


def load_norm(artifacts: str | Path) -> tuple[float, float]:
    """Read <artifacts>/norm.json -> (mean, std) in [0,1] image units.

    Returns a TUPLE, not a dict - callers unpack it as `mean, std = ...`.
    Measured on the 850 training boards on Day 1: mean 0.6472, std 0.4760.
    """
    path = Path(artifacts) / "norm.json"
    if not path.exists():
        raise FileNotFoundError(f"norm.json missing from staged bundle: {path}")

    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"norm.json must be a JSON object: {path}")
    for key in ("mean", "std"):
        if key not in obj:
            raise KeyError(f"norm.json missing required key '{key}': {path}")

    mean, std = float(obj["mean"]), float(obj["std"])
    if std == 0.0:
        # to_tensor divides by std. A zero here is inf/nan on every pixel, and
        # torch will happily propagate that into a confident-looking softmax.
        raise ValueError(f"norm.json has std == 0, which would divide by zero: {path}")
    return mean, std


def load_pcbnet(ckpt_path: str | Path, num_classes: int = 7, device: str = "cpu"):
    """Load weights into a fresh PCBNet and put it in eval mode.

    Handles: a bare state_dict, a dict wrapping one under a common key, and
    DataParallel's 'module.' prefix. strict=True on purpose - a silently
    partial load gives you a model that runs and scores ~random, which is far
    worse to debug than an exception.

    The constructor is called with num_classes= directly. This used to try
    three spellings and report whichever worked; the signature was confirmed
    at runtime on Day 5 as PCBNet(num_classes=7, in_ch=1, widths=(32,64,128,256)),
    so the guessing was dead scaffolding hiding a fact we now know.
    """
    from src.models.cnn import PCBNet

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state = obj
    if isinstance(obj, dict) and not all(isinstance(v, torch.Tensor) for v in obj.values()):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    model = PCBNet(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[checkpoint] loaded {Path(ckpt_path).name}: {n_params:,} params on {device}")
    return model