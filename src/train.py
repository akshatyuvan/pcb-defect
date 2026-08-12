"""
Train PCBNet from an in-memory tensor.

Why in-memory and not ImageFolder + DataLoader workers: Colab gives 2 vCPU cores.
Decoding one PNG per patch per epoch would leave the T4 idle waiting on the CPU.
The entire patch set is ~600MB as uint8, so it lives in RAM and we index it
directly. Casting to float and normalising happens on the GPU, per batch, which
is faster and avoids a 4x RAM blowup from a float32 copy.

Three runs are expected on Day 2:
  1. --no-weighted   unweighted CE, the ablation control
  2. (default)       inverse-frequency weighted CE
  3. --augment       weighted CE + label-preserving flips/rot90
Model selection is on val macro F1 (locked decision 6). Accuracy is logged and
never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display on Colab; must be set before pyplot
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve)

from src.data.deeppcb import CLASSES
from src.models.cnn import PCBNet, count_params


# ---------------------------------------------------------------- data

def load_split(patch_dir: Path, split: str):
    d = np.load(patch_dir / f"{split}.npz")
    # uint8 on CPU. ~150k * 4096 bytes ~= 600MB. Keeping it on GPU would eat VRAM
    # we would rather spend on activations.
    X = torch.from_numpy(d["X"]).unsqueeze(1)          # (N, 1, 64, 64) uint8
    y = torch.from_numpy(d["y"])                       # (N,) int64
    return X, y


def to_float(batch_u8: torch.Tensor, mean: float, std: float, device):
    """uint8 CPU batch -> normalised float32 GPU batch. Per batch on purpose."""
    x = batch_u8.to(device, non_blocking=True).float().div_(255.0)
    return x.sub_(mean).div_(std)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Label-preserving geometric augmentation only.

    Flips and 90-degree rotations are safe because a mousebite is still a
    mousebite upside down: the defect taxonomy is rotation-invariant. Colour and
    brightness augmentation is excluded because the images are binarised, so
    there is no photometric variation to simulate (locked decision 4).
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


# ---------------------------------------------------------------- metrics

@torch.no_grad()
def predict(model, X, y, mean, std, device, bs=512):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        xb = to_float(X[i:i + bs], mean, std, device)
        probs.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(probs).numpy(), y.numpy()


def binary_defect_scores(P: np.ndarray, y: np.ndarray):
    """Collapse the six defect classes into one (locked decision 6).

    Score is 1 - P(good) rather than max over defect classes: it is the model's
    total belief that *something* is wrong, which is what a QC line cares about.
    Which defect it is only matters for the operator's report.
    """
    score = 1.0 - P[:, 0]
    truth = (y != 0).astype(int)
    return score, truth


def choose_operating_point(score, truth, target_recall: float):
    """Pick the highest threshold that still hits the recall target.

    Deliberate asymmetry: a false alarm costs an operator ten seconds of looking
    at a board. A missed defect ships. So recall is a constraint, not something
    to trade off, and precision is the price we report for meeting it.
    """
    prec, rec, thr = precision_recall_curve(truth, score)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    ok = np.flatnonzero(rec[:-1] >= target_recall)
    if len(ok) == 0:
        return {"target_recall": target_recall, "achievable": False}
    j = ok[-1]  # highest threshold among those meeting the target
    return {
        "target_recall": target_recall,
        "achievable": True,
        "threshold": float(thr[j]),
        "recall_at_threshold": float(rec[j]),
        "precision_at_threshold": float(prec[j]),
    }


def save_confusion(cm, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Row-normalised: raw counts are unreadable when 'good' outnumbers 'pinhole'
    # by two orders of magnitude. Rows are truth, so each row shows recall spread.
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_pr(score, truth, op, path: Path):
    prec, rec, _ = precision_recall_curve(truth, score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2)
    if op.get("achievable"):
        ax.scatter([op["recall_at_threshold"]], [op["precision_at_threshold"]],
                   color="red", zorder=5,
                   label=f"chosen: R={op['recall_at_threshold']:.3f} "
                         f"P={op['precision_at_threshold']:.3f}")
        ax.legend()
    ax.set_xlabel("binary defect recall"); ax.set_ylabel("precision")
    ax.set_title("PR curve, defect vs good"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- mlflow

def setup_mlflow(db_uri: str, artifact_dir: Path, exp_name: str):
    """SQLite backend, not file://.

    The MLflow Model Registry does not exist on a file store. Day 5 loads the
    model by registry version rather than a hardcoded path, so the backend must
    be a database from the very first run.

    Note the split: SQLite holds params/metrics/tags/run structure; artifacts
    (figures, checkpoints, reports) go to artifact_dir separately. That is why
    the .db can sit on fast local disk while the heavy files live on Drive.
    """
    mlflow.set_tracking_uri(db_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(exp_name) is None:
        client.create_experiment(exp_name, artifact_location=str(artifact_dir.resolve()))
    mlflow.set_experiment(exp_name)


def log_model_compat(model, registered_name: str):
    """Log and register the model, tolerating MLflow API drift.

    Two moving parts across MLflow versions:
      1. log_model's 'artifact_path' argument was renamed to 'name' in 3.x.
      2. Newer versions default to the 'pt2' traced-graph serialization format,
         which virtually executes forward() and therefore REQUIRES input_example.

    Passing input_example is the right fix rather than a workaround: it makes
    MLflow infer and store the model signature, so the registry records that this
    model takes (N, 1, 64, 64) float32 and returns 7 logits. Day 5's service then
    loads a self-describing artifact instead of a bare pickle it has to guess at.

    The model is moved to CPU first. Tracing runs forward() on the example, and a
    CPU numpy example against a CUDA model is a device mismatch. Nothing after
    this point needs the GPU.
    """
    model = model.cpu().eval()
    # Batch of 1 is safe here: eval() means BatchNorm uses running statistics
    # rather than batch statistics, so a single-sample forward is well-defined.
    example = np.zeros((1, 1, 64, 64), dtype=np.float32)
    try:
        mlflow.pytorch.log_model(model, name="model", input_example=example,
                                 registered_model_name=registered_name)
    except TypeError:
        mlflow.pytorch.log_model(model, artifact_path="model", input_example=example,
                                 registered_model_name=registered_name)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, default=Path("data/patches"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--weighted", action="store_true", default=True)
    ap.add_argument("--no-weighted", dest="weighted", action="store_false")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="1.0 = pure inverse frequency (locked decision 7). "
                         "Drop to 0.5 ONLY if training collapses, and document it.")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.97)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--mlflow-db", type=str, default="sqlite:////content/mlflow.db")
    ap.add_argument("--mlflow-artifacts", type=Path,
                    default=Path("/content/drive/MyDrive/pcb-defect/mlruns"))
    ap.add_argument("--experiment", type=str, default="pcb-patch-classification")
    ap.add_argument("--register", action="store_true",
                    help="register this run's model as 'pcbnet' in the registry")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = json.loads(Path("artifacts/norm.json").read_text())
    mean, std = norm["mean"], norm["std"]

    # Read back the labelling rule that produced these patches so it lands in
    # MLflow. Without this a run is untraceable to the label definition it was
    # trained under, and patch_stats.json is gitignored so it will not survive
    # in the repo.
    pstats = json.loads((args.patches / "patch_stats.json").read_text())
    label_params = {
        "patch_assign": pstats["assign"],
        "patch_min_frac": pstats["min_frac"],
        "patch_seed": pstats["seed"],
    }

    Xtr, ytr = load_split(args.patches, "train")
    Xva, yva = load_split(args.patches, "val")
    Xte, yte = load_split(args.patches, "test")

    counts = np.bincount(ytr.numpy(), minlength=len(CLASSES)).astype(np.float64)
    if args.weighted:
        # Inverse frequency, normalised to mean 1 so the effective learning rate
        # does not change between the weighted and unweighted runs. Without this
        # normalisation the ablation would confound "weighting" with "bigger
        # gradients", and the comparison would prove nothing.
        w = (counts.sum() / np.clip(counts, 1, None)) ** args.weight_power
        w = w / w.mean()
    else:
        w = np.ones(len(CLASSES))
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    model = PCBNet(num_classes=len(CLASSES)).to(device)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    setup_mlflow(args.mlflow_db, args.mlflow_artifacts, args.experiment)
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "weighted": args.weighted, "weight_power": args.weight_power,
            "augment": args.augment, "epochs": args.epochs, "bs": args.bs,
            "lr": args.lr, "wd": args.wd, "seed": args.seed,
            "params": count_params(model),
            "train_patches": int(len(ytr)), "val_patches": int(len(yva)),
            "test_patches": int(len(yte)),
            **label_params,
        })
        mlflow.log_dict({CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
                        "train_class_counts.json")

        best_f1, best_epoch = -1.0, -1
        n = len(ytr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n)
            tot, t0 = 0.0, time.time()
            for i in range(0, n, args.bs):
                idx = perm[i:i + args.bs]
                xb = to_float(Xtr[idx], mean, std, device)
                if args.augment:
                    xb = augment(xb)
                yb = ytr[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tot += loss.item() * len(idx)
            sched.step()

            Pv, yv = predict(model, Xva, yva, mean, std, device)
            pred = Pv.argmax(1)
            macro_f1 = f1_score(yv, pred, average="macro", zero_division=0)
            acc = (pred == yv).mean()
            sv, tv = binary_defect_scores(Pv, yv)
            bin_rec = ((sv > 0.5) & (tv == 1)).sum() / max(1, (tv == 1).sum())

            mlflow.log_metrics({
                "train_loss": tot / n, "val_macro_f1": float(macro_f1),
                "val_accuracy": float(acc),
                "val_binary_defect_recall_at_0p5": float(bin_rec),
                "lr": sched.get_last_lr()[0],
            }, step=ep)
            print(f"ep {ep:02d} loss {tot/n:.4f} val_macroF1 {macro_f1:.4f} "
                  f"acc {acc:.4f} binRec {bin_rec:.4f} ({time.time()-t0:.0f}s)")

            # Selection on macro F1 only. Never on accuracy.
            if macro_f1 > best_f1:
                best_f1, best_epoch = macro_f1, ep
                # classes and norm ride along inside the checkpoint so the Day 5
                # service cannot load weights with the wrong preprocessing.
                torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                            "norm": norm, "epoch": ep,
                            "val_macro_f1": float(macro_f1)},
                           "artifacts/best.pt")

        # ---- final evaluation on TEST, using the best checkpoint
        ck = torch.load("artifacts/best.pt", map_location=device)
        model.load_state_dict(ck["state_dict"])

        Pt, yt = predict(model, Xte, yte, mean, std, device)
        pred = Pt.argmax(1)
        rep = classification_report(yt, pred, target_names=CLASSES,
                                    zero_division=0, digits=4)
        cm = confusion_matrix(yt, pred, labels=list(range(len(CLASSES))))
        st, tt = binary_defect_scores(Pt, yt)
        op = choose_operating_point(st, tt, args.target_recall)

        tag = args.run_name
        cm_png = Path(f"artifacts/figures/{tag}_confusion.png")
        pr_png = Path(f"artifacts/figures/{tag}_pr.png")
        save_confusion(cm, cm_png, f"{tag} (test)")
        save_pr(st, tt, op, pr_png)

        report = (
            f"run: {tag}\n"
            f"labelling: assign={label_params['patch_assign']} "
            f"min_frac={label_params['patch_min_frac']}\n"
            f"best epoch: {best_epoch}  val macro F1: {best_f1:.4f}\n\n"
            f"TEST per-class:\n{rep}\n"
            f"test macro F1: {f1_score(yt, pred, average='macro', zero_division=0):.4f}\n"
            f"test accuracy (reported, not used for selection): {(pred==yt).mean():.4f}\n\n"
            f"binary defect operating point: {json.dumps(op, indent=2)}\n"
        )
        Path(f"artifacts/{tag}_report.txt").write_text(report)
        print(report)

        mlflow.log_metrics({
            "best_val_macro_f1": float(best_f1),
            "test_macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
            "test_accuracy": float((pred == yt).mean()),
            "op_precision": float(op.get("precision_at_threshold", 0.0)),
            "op_recall": float(op.get("recall_at_threshold", 0.0)),
            "op_threshold": float(op.get("threshold", 0.0)),
        })
        mlflow.log_artifact(str(cm_png)); mlflow.log_artifact(str(pr_png))
        mlflow.log_artifact(f"artifacts/{tag}_report.txt")
        mlflow.log_artifact("artifacts/norm.json")
        mlflow.log_artifact("artifacts/classes.json")

        if args.register:
            log_model_compat(model, "pcbnet")
            shutil.copy("artifacts/best.pt", f"artifacts/{tag}_best.pt")

        print("run_id:", run.info.run_id)


if __name__ == "__main__":
    main()