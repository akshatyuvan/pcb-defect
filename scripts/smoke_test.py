"""Hit every endpoint with a real DeepPCB board. Mac. Run while uvicorn is up."""
import base64, io, json, sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
board_path = sorted(Path("data/raw/PCBData").rglob("*_test.jpg"))[0]
board_id = board_path.stem.replace("_test", "")
raw = board_path.read_bytes()
print(f"board: {board_path}  ({len(raw):,} bytes)\n")

print("--- /health ---")
print(json.dumps(requests.get(f"{BASE}/health").json(), indent=2))

print("\n--- /predict/board ---")
r = requests.post(f"{BASE}/predict/board",
                  json={"image_b64": base64.b64encode(raw).decode(),
                        "board_id": board_id, "trace_id": "smoke-1"})
r.raise_for_status()
b = r.json()
for k in ("verdict", "n_patches", "n_flagged", "max_defect_score",
          "fail_threshold", "class_counts", "latency_ms"):
    print(f"  {k}: {b[k]}")
print("  top flagged:", json.dumps(b["flagged"][:3], indent=2))

print("\n  defect-score grid (x1000, rounded):")
for row in b["grid_defect_score"]:
    print("   ", " ".join(f"{v*1000:6.1f}" for v in row))

# Take the single highest-scoring tile and explain it.
g = np.array(b["grid_defect_score"])
gy, gx = np.unravel_index(g.argmax(), g.shape)
board = np.array(Image.open(io.BytesIO(raw)).convert("L"))
patch = board[gy*64:(gy+1)*64, gx*64:(gx+1)*64]
buf = io.BytesIO(); Image.fromarray(patch).save(buf, format="PNG")

print(f"\n--- /predict/patch  (tile gy={gy} gx={gx}) ---")
pr = requests.post(f"{BASE}/predict/patch",
                   json={"image_b64": base64.b64encode(buf.getvalue()).decode(),
                         "board_id": board_id}).json()
print(json.dumps(pr, indent=2))

print(f"\n--- /explain/patch (tile gy={gy} gx={gx}) ---")
ex = requests.post(f"{BASE}/explain/patch",
                   json={"image_b64": base64.b64encode(buf.getvalue()).decode(),
                         "board_id": board_id}).json()
print({k: v for k, v in ex.items() if k != "overlay_png_b64"})

out = Path("artifacts/figures/day5_overlay_smoke.png")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(base64.b64decode(ex["overlay_png_b64"]))
Image.fromarray(patch).save("artifacts/figures/day5_patch_smoke.png")
print(f"\nwrote {out} and artifacts/figures/day5_patch_smoke.png — OPEN BOTH.")
