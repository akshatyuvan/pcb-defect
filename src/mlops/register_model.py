"""Register the Day 2 checkpoint into a LOCAL MLflow registry on the Mac.

Runs on: Mac, CPU, ~20s.

Why re-register instead of copying Colab's mlflow.db across?
  The Colab DB stores ABSOLUTE artifact URIs under /content/... . Copying the
  DB copies dangling pointers. Re-registering from the checkpoint file makes
  the local store self-consistent, and forces the provenance (source run,
  md5, operating point) to be stated explicitly rather than inherited silently.

Why SQLite and not the default file store?
  The Model Registry does not exist on a file:// backend. This is locked
  decision 11 and is the whole reason `models:/name/version` URIs work below.

The operating point is logged as TAGS, not just prose. stage_model.py reads
them back into the model card, so the threshold the service uses is provably
the threshold that was measured on Day 2 - not a constant someone retyped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.serving.model_loader import build_pcbnet, load_state_dict_into

REPO = Path(__file__).resolve().parents[2]

# --- Day 2, run r4_weighted_p05_registered. Single source of truth. ---
DAY2 = {
    "source_run": "r4_weighted_p05_registered",
    "weight_power": 0.5,
    "augmentation": "none",
    "epochs": 30,
    "best_epoch": 29,
    "batch_size": 256,
    "optimizer": "AdamW",
    "lr_schedule": "cosine",
    "amp": True,
    "seed": 42,
}
DAY2_METRICS = {
    "val_macro_f1": 0.7603,
    "test_macro_f1": 0.7172,
    "test_accuracy": 0.9529,
    "precision_at_recall_0.97": 0.1106,
    "binary_threshold_at_recall_0.97": 0.000416,
    # Day 3 measured these on the identical test patches. Stored here so the
    # baseline comparison travels with the model version rather than living
    # only in a report file.
    "test_binary_ap": 0.823,
    "baseline_binary_ap": 0.7244,
    "baseline_detection_ceiling": 0.8161,
    "defect_prevalence": 0.0939,
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_model_compat(model, artifact_dir: str, **kw):
    """MLflow 2.x wants artifact_path=, MLflow 3.x wants name=.
    We are on 3.15.1, so name= is tried first; the fallback keeps the script
    working if it is ever run in the Colab env, which may be on 2.x."""
    try:
        return mlflow.pytorch.log_model(model, name=artifact_dir, **kw)
    except TypeError:
        return mlflow.pytorch.log_model(model, artifact_path=artifact_dir, **kw)

def build_tags(mean: float, std: float, ckpt: Path) -> dict[str, str]:
    """One tag set, written to BOTH the run and the model version."""
    return {
        "stage": "serving",
        "source_checkpoint_md5": md5_of(ckpt),
        "source_checkpoint_sha256": sha256_of(ckpt),
        "input_spec": "uint8 (64,64) -> /255 -> -mean -> /std -> (1,1,64,64) float32",
        "norm_mean": f"{mean}",
        "norm_std": f"{std}",
        "binary_score_def": "1 - P(good)",
        "op_target_recall": "0.97",
        "op_threshold": f"{DAY2_METRICS['binary_threshold_at_recall_0.97']}",
        "op_precision": f"{DAY2_METRICS['precision_at_recall_0.97']}",
        "op_review_threshold": "",
        "cam_layer": "final",
        "cam_equals_gradcam": "true (Day 4, max abs diff 0.000e+00)",
        "decoder_parity": "PIL == cv2 bit-identical on DeepPCB (Day 5, 25 boards, 0 diff)",
        "baseline_comparison": "PCBNet AP 0.823 vs template-diff AP 0.724; baseline recall ceiling 0.816",
        "known_limits": "not converged (best epoch 29/30); single seed; mousebite weakest (F1 0.610); P@R=0.97 only 0.111 vs 0.094 prevalence",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "artifacts/checkpoints/r4_weighted_p05_registered_best.pt"))
    ap.add_argument("--classes", default=str(REPO / "artifacts/checkpoints/classes.json"))
    ap.add_argument("--norm", default=str(REPO / "artifacts/checkpoints/norm.json"))
    ap.add_argument("--registry-name", default="pcb-defect-cnn")
    ap.add_argument("--experiment", default="pcb-serving")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    classes = json.loads(Path(args.classes).read_text())
    if isinstance(classes, dict):                     # tolerate {"classes": [...]}
        classes = classes.get("classes", classes)
    norm = json.loads(Path(args.norm).read_text())
    mean = float(norm["mean"] if isinstance(norm, dict) else norm[0])
    std = float(norm["std"] if isinstance(norm, dict) else norm[1])

    # Locked decision 3: exactly 7 classes, 'good' first. Asserted here because
    # every downstream binary score is 1 - P(good); a reordered file would
    # invert every verdict in the system with no error anywhere.
    assert classes[0] == "good", f"'good' must be class 0, got {classes[0]!r}"
    assert len(classes) == 7, f"expected 7 classes, got {len(classes)}"

    print(f"classes ({len(classes)}): {classes}")
    print(f"norm: mean={mean} std={std}")
    print(f"ckpt md5: {md5_of(ckpt)}")

    model, ctor_kwarg = build_pcbnet(len(classes))
    load_state_dict_into(model, ckpt, device="cpu")
    model.eval()
    print(f"constructor kwarg that worked: {ctor_kwarg}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    TAGS = build_tags(mean, std, ckpt)
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="register_r4_for_serving") as run:
        mlflow.log_params({**DAY2, "n_classes": len(classes), "ctor_kwarg": ctor_kwarg})
        mlflow.log_metrics(DAY2_METRICS)
        mlflow.set_tags(TAGS)

        # Sidecars travel WITH the model version. checkpoint.py made forgetting
        # normalisation a loud failure; this makes it impossible to even ship.
        side = Path("/tmp/pcb_sidecar")
        side.mkdir(exist_ok=True)
        (side / "classes.json").write_text(json.dumps(classes, indent=2))
        (side / "norm.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
        mlflow.log_artifacts(str(side), artifact_path="sidecar")
        mlflow.log_artifact(str(ckpt), artifact_path="sidecar")

        ex = np.zeros((1, 1, 64, 64), dtype=np.float32)
        log_model_compat(model, "model", input_example=ex,
                         registered_model_name=args.registry_name)
        print(f"\nrun_id: {run.info.run_id}")

    from mlflow.tracking import MlflowClient
    c = MlflowClient(tracking_uri=f"sqlite:///{REPO / 'mlflow.db'}")
    vs = sorted(c.search_model_versions(f"name='{args.registry_name}'"),
                key=lambda v: int(v.version))
    latest = vs[-1]

    # mlflow.set_tags() writes to the RUN. A model VERSION has a separate tag
    # namespace, and the registry page reads only the latter. Copying them over
    # is what makes a version self-describing: someone browsing the registry can
    # see the operating point and the known limits without hunting for the run
    # that produced it. That is the entire argument for having a registry.
    for k, v in TAGS.items():
        c.set_model_version_tag(args.registry_name, latest.version, k, v)
    c.update_model_version(
        args.registry_name, latest.version,
        description=(
            "PCBNet, 1.17M params, trained from scratch on DeepPCB 64x64 patches. "
            "7 classes (good + 6 bare-board copper defects). "
            "Binary score = 1 - P(good); FAIL at score >= 0.000416, the threshold "
            "measured on Day 2 for patch-level recall 0.97 (precision 0.111 there, "
            "vs 0.094 prevalence). Binary AP 0.823 vs 0.724 for the OpenCV "
            "template-differencing baseline on identical test patches. "
            "Limits: not converged (best epoch 29/30), single seed, mousebite "
            "weakest (F1 0.610). Board-level threshold NOT yet calibrated."
        ),
    )
    print(f"\nREGISTERED: {args.registry_name} version {latest.version}")
    print(f"  run_id  : {latest.run_id}")
    print(f"  source  : {latest.source}")
    print(f"  version tags set: {len(TAGS)}")
    print(f"\nNext: python -m src.mlops.stage_model --version {latest.version}")


if __name__ == "__main__":
    main()