"""Serving configuration. Env-var driven so the SAME code runs unchanged
on the Mac (registry-backed) and in Docker (staged-bundle-backed).

Two load modes, resolved in model_loader.load_model():
  PCB_MODEL_URI  e.g. "models:/pcb-defect-cnn/1"  -> true registry load (host)
  PCB_MODEL_DIR  e.g. "/app/artifacts/serving"    -> staged bundle (container)
URI wins if both are set.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_URI = os.environ.get("PCB_MODEL_URI", "").strip() or None
MODEL_DIR = Path(os.environ.get("PCB_MODEL_DIR", REPO_ROOT / "artifacts" / "serving"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

# CPU by default even on the Mac: the container is CPU-only (no MPS inside
# Docker), and Day 8's benchmark is worthless if local numbers came from a
# different device. Set PCB_DEVICE=mps only for throwaway sanity runs.
DEVICE = os.environ.get("PCB_DEVICE", "cpu")

# 2 threads. Docker Desktop is capped at 4GB and each torch thread carries an
# allocator arena; more threads on a 1.17M-param model buys nothing and costs RSS.
TORCH_THREADS = int(os.environ.get("PCB_TORCH_THREADS", "2"))

BOARD_SIZE = 640
PATCH_SIZE = 64
GRID = BOARD_SIZE // PATCH_SIZE          # 10 -> exactly 100 tiles, no remainder
assert BOARD_SIZE % PATCH_SIZE == 0, "tiling must be exact; Day 1 assumed this"