"""
Load a trained PCBNet from a .pt checkpoint, and load the preprocessing
constants that must accompany it.

WHY THIS FILE EXISTS
  Day 3 (PR overlay), Day 4 (Grad-CAM), and Day 5 (FastAPI) all need the same
  three things: the weights, the class list, and the normalisation constants.
  The Day 5 handoff note says norm.json is "the one people forget" and that
  without it the service runs fine and predicts garbage. Putting the three
  loads in one function is the cheapest way to make forgetting impossible.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

CLASSES_FALLBACK = ["good", "open", "short", "mousebite", "spur", "copper", "pinhole"]
NORM_FALLBACK = (0.6472, 0.4760)  # measured on the 850 TRAIN boards on Day 1


def load_classes(artifacts: str | Path) -> list[str]:
    """Read artifacts/classes.json, tolerating list or dict formats."""
    path = Path(artifacts) / "classes.json"
    if path.exists():
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for key in ("classes", "names", "CLASSES"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                # assume it is a {name: index} mapping
                if all(isinstance(v, int) for v in obj.values()):
                    obj = [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
        if isinstance(obj, list) and len(obj) == 7 and all(isinstance(c, str) for c in obj):
            return list(obj)
    print(f"[warn] could not parse {path}, using hardcoded class order")
    return list(CLASSES_FALLBACK)


def load_norm(artifacts: str | Path) -> tuple[float, float]:
    """Read artifacts/norm.json -> (mean, std) in [0,1] image units."""
    path = Path(artifacts) / "norm.json"
    if path.exists():
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for mk in ("mean", "MEAN", "train_mean"):
                for sk in ("std", "STD", "train_std"):
                    if mk in obj and sk in obj:
                        m, s = obj[mk], obj[sk]
                        m = float(m[0]) if isinstance(m, (list, tuple)) else float(m)
                        s = float(s[0]) if isinstance(s, (list, tuple)) else float(s)
                        return m, s
    print(f"[warn] could not parse {path}, using hardcoded norm {NORM_FALLBACK}")
    return NORM_FALLBACK


def _construct(num_classes: int):
    """PCBNet's constructor signature is not pinned by the handoff, so try the
    three plausible spellings and report clearly if none work."""
    from src.models.cnn import PCBNet

    errors = []
    for kwargs in ({"num_classes": num_classes}, {"n_classes": num_classes}, {}):
        try:
            return PCBNet(**kwargs)
        except TypeError as exc:
            errors.append(f"PCBNet(**{kwargs}) -> {exc}")
    raise TypeError("could not construct PCBNet. tried:\n  " + "\n  ".join(errors))


def load_pcbnet(ckpt_path: str | Path, num_classes: int = 7, device: str = "cpu"):
    """Load weights into a fresh PCBNet and put it in eval mode.

    Handles: a bare state_dict, a dict wrapping one under a common key, and
    DataParallel's 'module.' prefix. strict=True on purpose. A silently
    partial load would give you a model that runs and scores ~random, which is
    much worse to debug than an exception.
    """
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state = obj
    if isinstance(obj, dict) and not all(isinstance(v, torch.Tensor) for v in obj.values()):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    model = _construct(num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[checkpoint] loaded {Path(ckpt_path).name}: {n_params:,} params on {device}")
    return model
