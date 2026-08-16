"""Assert Pillow-in-the-container decodes DeepPCB boards identically to
OpenCV-in-training.

Why this exists: Days 1-4 built patches with cv2.imread(..., IMREAD_GRAYSCALE).
The Docker serving image deliberately excludes OpenCV (~70MB + transitive libs)
and uses PIL.Image.open(...).convert("L"). For an already-grayscale JPEG these
should be bit-identical, but "should be" is not a measurement. If they differ,
serving must switch to cv2 and the image gets fatter — better to know today
than to discover it as an unexplained accuracy gap on Day 8.

Mac only (cv2 is not in the serving image). CPU, ~5 seconds.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

RAW = Path("data/raw/PCBData")
N = 25

paths = sorted(RAW.rglob("*_test.jpg"))[:N]
if not paths:
    sys.exit(f"FATAL: no *_test.jpg under {RAW.resolve()} — did Step 5.3 run?")

worst = 0
for p in paths:
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)          # training path
    b = np.array(Image.open(p).convert("L"), dtype=np.uint8)  # serving path
    if a is None:
        sys.exit(f"FATAL: cv2 could not read {p}")
    if a.shape != b.shape:
        sys.exit(f"FATAL: shape mismatch on {p}: cv2 {a.shape} vs PIL {b.shape}")
    d = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
    worst = max(worst, d)
    print(f"{p.name:>24}  shape={a.shape}  max|cv2-PIL|={d}")

print(f"\nboards checked: {len(paths)}   worst pixel disagreement: {worst}")
if worst == 0:
    print("PASS — Pillow decoding is exact. Serving image can stay OpenCV-free.")
else:
    print("FAIL — decoders disagree. Add opencv-python-headless to "
          "requirements-serving.txt and switch src/serving/preprocess.py to cv2.")
    sys.exit(1)
