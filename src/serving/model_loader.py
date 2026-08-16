"""Load PCBNet exactly once per process, from either the MLflow registry
(host) or a staged artifact bundle (container).

Design note you will be asked about: why doesn't the container talk to the
registry directly?
  The MLflow SQLite store records ABSOLUTE artifact paths under
  /Users/<you>/Desktop/pcb-defect/mlruns/... . Those paths do not exist inside
  a container, so a bind-mounted DB resolves to dangling URIs. The standard fix
  is the promotion flow used here: registry is the system of record on the dev
  host; a registry-driven staging step (src/mlops/stage_model.py) emits an
  immutable bundle; the image consumes the bundle. The container therefore has
  no DB dependency, no mlflow install, and no network call at startup - which
  is also why it fits in 1GB.
  The bundle carries model_card.json naming the exact registry name + version
  + run_id it came from, so provenance is not lost, just materialised.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.cnn import PCBNet
from src.serving import config

# Confirmed on Day 5 via scripts/probe_contracts.py: PCBNet takes num_classes.
# The others are kept as fallbacks so a refactor of cnn.py fails loudly at
# startup rather than silently constructing a 7-class head by luck.
_CTOR_CANDIDATES = ("num_classes", "n_classes", "nclass", None)


def build_pcbnet(n_classes: int) -> tuple[nn.Module, str]:
    """Construct PCBNet, resolving whichever kwarg name it actually takes.
    Returns (model, kwarg_name) so stage_model.py can write the answer into
    the model card permanently."""
    params = inspect.signature(PCBNet.__init__).parameters
    for kw in _CTOR_CANDIDATES:
        try:
            if kw is None:
                return PCBNet(), "<no-arg>"
            if kw in params:
                return PCBNet(**{kw: n_classes}), kw
        except Exception:
            continue
    raise RuntimeError("could not construct PCBNet with any known signature")


def load_state_dict_into(model: nn.Module, ckpt_path: Path, device: str = "cpu") -> None:
    """Handle the three shapes a checkpoint can arrive in: a bare state_dict,
    a wrapped {'model'|'state_dict': ...}, or DataParallel's 'module.' prefix.

    weights_only=False is deliberate: torch >= 2.6 defaults it to True, which
    refuses to unpickle a wrapped dict. We control this file (md5-verified
    against Day 2), so the stricter default buys nothing and breaks loading.
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
    if not isinstance(sd, dict):
        raise RuntimeError(f"{ckpt_path} did not contain a state_dict")
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    # strict=True on purpose: a silently-partial load is exactly the failure
    # mode that produces a model that runs and is wrong.
    model.load_state_dict(sd, strict=True)


@dataclass
class LoadedModel:
    model: nn.Module
    classes: list[str]
    mean: float
    std: float
    card: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    device: str = "cpu"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def good_index(self) -> int:
        # Locked: class 0 is 'good'. Asserted rather than assumed, because the
        # binary defect score is 1 - P(good) and a reordered classes.json would
        # silently invert every verdict in the system.
        i = self.classes.index("good")
        if i != 0:
            raise RuntimeError(f"'good' must be class 0, found at {i}")
        return i


def _load_bundle(d: Path, device: str) -> LoadedModel:
    for required in ("model.pt", "classes.json", "norm.json", "model_card.json"):
        if not (d / required).exists():
            raise FileNotFoundError(
                f"{d / required} missing - run: python -m src.mlops.stage_model --version N"
            )
    classes = json.loads((d / "classes.json").read_text())
    norm = json.loads((d / "norm.json").read_text())
    card = json.loads((d / "model_card.json").read_text())
    model, _ = build_pcbnet(len(classes))
    load_state_dict_into(model, d / "model.pt", device=device)
    model.eval().to(device)
    return LoadedModel(model, classes, float(norm["mean"]), float(norm["std"]),
                       card, f"dir:{d}", device)


def _load_registry(uri: str, device: str) -> LoadedModel:
    """True registry load: `models:/pcb-defect-cnn/1`. Host only (needs mlflow).

    Weights come from the sidecar checkpoint the version logged, NOT from
    mlflow.pytorch.load_model(): on MLflow 3.x + torch 2.6+ that returns a
    torch.export ExportedProgram, which exposes only forward() - no
    forward_with_features, so CAM would be impossible. The registry still
    resolves which artifact to use; it just isn't the deserialiser.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    name, version = uri.removeprefix("models:/").split("/")
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    mv = client.get_model_version(name, version)
    side = Path(mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="sidecar"))

    classes = json.loads((side / "classes.json").read_text())
    norm = json.loads((side / "norm.json").read_text())
    ckpts = sorted(side.glob("*.pt"))
    if len(ckpts) != 1:
        raise RuntimeError(f"expected exactly one .pt in the sidecar, found {ckpts}")

    model, _ = build_pcbnet(len(classes))
    load_state_dict_into(model, ckpts[0], device=device)
    model.eval().to(device)

    card = {"registry_name": name, "registry_version": version, "run_id": mv.run_id,
            "tags": dict(mv.tags)}
    return LoadedModel(model, classes, float(norm["mean"]), float(norm["std"]),
                       card, f"registry:{uri}", device)


def load_model() -> LoadedModel:
    torch.set_num_threads(config.TORCH_THREADS)
    device = config.DEVICE
    lm = _load_registry(config.MODEL_URI, device) if config.MODEL_URI \
        else _load_bundle(Path(config.MODEL_DIR), device)
    _ = lm.good_index                    # fail fast at startup, not per request
    return lm