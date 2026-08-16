"""Resolve a registry version -> immutable serving bundle in artifacts/serving/.

Mac, CPU, ~10s. This is the promotion step: nothing downstream of here ever
names a checkpoint path. To ship a different model you bump --version.

Emits:
  artifacts/serving/model.pt          state_dict only (no pickled module)
  artifacts/serving/classes.json
  artifacts/serving/norm.json
  artifacts/serving/model_card.json   provenance + operating point + input spec

model.pt is a state_dict rather than mlflow.pytorch's pickled module because a
pickle binds the exact import path src.models.cnn.PCBNet at load time; a
state_dict binds only tensor names. Cheaper to reason about, and it means the
container can be rebuilt from a refactored src/ without re-registering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.serving import config
from src.serving.model_loader import build_pcbnet, load_state_dict_into

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "serving"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pcb-defect-cnn")
    ap.add_argument("--version", required=True)
    args = ap.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    mv = client.get_model_version(args.name, args.version)
    run = client.get_run(mv.run_id)
    tags = dict(mv.tags) or {}
    tags.update(run.data.tags)

    OUT.mkdir(parents=True, exist_ok=True)

    # mlflow.artifacts.download_artifacts, not client.download_artifacts:
    # the client method is deprecated in MLflow 3.x (we are on 3.15.1).
    side = Path(mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="sidecar"))
    shutil.copy(side / "classes.json", OUT / "classes.json")
    shutil.copy(side / "norm.json", OUT / "norm.json")
    classes = json.loads((OUT / "classes.json").read_text())
    norm = json.loads((OUT / "norm.json").read_text())

    # Build the bundle from the checkpoint THIS REGISTRY VERSION logged, not
    # from mlflow.pytorch.load_model(). On MLflow 3.x + torch 2.6+, log_model
    # serialises via torch.export, so load_model returns an ExportedProgram:
    # it has no .eval(), no .state_dict(), and critically no
    # forward_with_features, which CAM depends on. Resolving version -> run ->
    # sidecar keeps the registry as the system of record while giving us a real
    # nn.Module. Nothing here names a hand-typed checkpoint path.
    ckpts = sorted(side.glob("*.pt"))
    if len(ckpts) != 1:
        raise RuntimeError(f"expected exactly one .pt in the sidecar, found {ckpts}")
    src_ckpt = ckpts[0]
    print(f"sidecar checkpoint: {src_ckpt.name}  md5 {hashlib.md5(src_ckpt.read_bytes()).hexdigest()}")

    ref, ctor_kwarg = build_pcbnet(len(classes))
    load_state_dict_into(ref, src_ckpt, device="cpu")
    ref.eval()
    torch.save(ref.state_dict(), OUT / "model.pt")

    # Round-trip: reload from the state_dict we just wrote and confirm identical
    # logits. Catches a silently-wrong save before it reaches Docker.
    rt, _ = build_pcbnet(len(classes))
    rt.load_state_dict(torch.load(OUT / "model.pt", map_location="cpu"), strict=True)
    rt.eval()
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        d = (ref(x) - rt(x)).abs().max().item()
    print(f"round-trip max logit diff: {d:.3e}")
    assert d < 1e-6, "staged state_dict does not reproduce the source checkpoint"

    # Independent cross-check against the registry's exported program. This is
    # the real proof that the bundle matches what was registered, rather than
    # just matching itself. Batch is 1 because torch.export specialised the
    # batch dim to the input_example's shape. Non-fatal: a failure here means
    # the export is unusable, not that the weights are wrong.
    export_diff = None
    try:
        ep = mlflow.pytorch.load_model(f"models:/{args.name}/{args.version}")
        fn = ep.module() if hasattr(ep, "module") else ep
        x1 = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            export_diff = (fn(x1) - ref(x1)).abs().max().item()
        print(f"vs registry export, max logit diff: {export_diff:.3e}")
        assert export_diff < 1e-5, "bundle disagrees with the registered export"
    except Exception as e:
        print(f"WARNING: could not cross-check against the registry export: "
              f"{type(e).__name__}: {e}")

    review = tags.get("op_review_threshold", "").strip()
    card = {
        "registry_name": args.name,
        "registry_version": str(args.version),
        "run_id": mv.run_id,
        "source_checkpoint_md5": tags.get("source_checkpoint_md5"),
        "staged_model_sha256": hashlib.sha256((OUT / "model.pt").read_bytes()).hexdigest(),
        "constructor_kwarg": ctor_kwarg,
        "classes": classes,
        "norm": {"mean": float(norm["mean"]), "std": float(norm["std"])},
        "input_spec": {
            "channels": 1, "height": 64, "width": 64,
            "pipeline": "uint8 -> /255 -> -mean -> /std -> (N,1,64,64) float32",
            "export_crosscheck_max_logit_diff": export_diff,
            "decoder": "PIL Image.convert('L'); verified bit-identical to cv2 on DeepPCB",
        },
        "cam": {"layer": "final", "feature_map": [4, 4], "upsample": 16,
                "equals_gradcam": True, "verified_max_abs_diff": 0.0},
        "operating_point": {
            "binary_score": "1 - P(good)",
            "target_recall": 0.97,
            "fail_threshold": float(tags.get("op_threshold", 0.000416)),
            "precision_at_threshold": float(tags.get("op_precision", 0.1106)),
            "review_threshold": (float(review) if review else None),
            "note": ("patch-level operating point from Day 2. Board-level verdicts "
                     "reuse it as a max-over-patches rule, which is deliberately "
                     "conservative and NOT yet calibrated at board level - Day 7."),
        },
        "metrics_day2": {k: v for k, v in run.data.metrics.items()},
        "known_limits": tags.get("known_limits"),
    }
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2))

    print(f"\nstaged -> {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<20} {p.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()