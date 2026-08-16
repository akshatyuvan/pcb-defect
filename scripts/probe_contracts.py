"""Print the exact signatures Day 5 is about to build on top of.

Rationale: the Day 3-4 handoff records that load_pcbnet() tried three
constructor signatures and one of them worked, but not which. Serving code
that guesses will fail at container start, in the worst possible place.
Run once, read once, then the answer is written into the model card by
src/mlops/stage_model.py so it is never lost again.

Mac. Runs on CPU in a second.
"""
import inspect, json, sys

def show(label, obj):
    print(f"\n--- {label} ---")
    try:
        print(inspect.signature(obj))
    except (TypeError, ValueError) as e:
        print("no signature:", e)

import torch
import torch.nn as nn

from src.models.cnn import PCBNet
show("PCBNet.__init__", PCBNet.__init__)
print("PCBNet source file:", inspect.getsourcefile(PCBNet))
print("has forward_with_features:", hasattr(PCBNet, "forward_with_features"))
show("PCBNet.forward_with_features", getattr(PCBNet, "forward_with_features", None))

# Which constructor kwarg actually works? Try them in order and report.
CANDIDATES = [
    ("num_classes", 7),
    ("n_classes", 7),
    ("nclass", 7),
    (None, None),  # no-arg
]
resolved = None
for kw, val in CANDIDATES:
    try:
        m = PCBNet() if kw is None else PCBNet(**{kw: val})
        resolved = kw or "<no-arg>"
        print(f"\nCONSTRUCTOR OK with: {resolved}")
        break
    except Exception as e:
        print(f"constructor {kw!r} failed: {type(e).__name__}: {e}")
if resolved is None:
    sys.exit("FATAL: no constructor signature worked — open src/models/cnn.py")

# The GAP head must be exactly one Linear. Everything on Day 4 (Grad-CAM == CAM)
# and everything in src/serving/cam.py depends on that being true.
linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
print("\nnum Linear layers in PCBNet:", len(linears))
for i, l in enumerate(linears):
    print(f"  linear[{i}] in={l.in_features} out={l.out_features}")

n_params = sum(p.numel() for p in m.parameters())
print("param count:", f"{n_params:,}")

x = torch.zeros(2, 1, 64, 64)
out = m.forward_with_features(x)
print("\nforward_with_features returned:", type(out).__name__, "len", len(out))
for i, t in enumerate(out):
    print(f"  [{i}] shape {tuple(t.shape)}")

# The other two shared modules Day 5 is told to reuse.
try:
    from src.models import checkpoint as ck
    print("\nsrc/models/checkpoint.py exports:", [n for n in dir(ck) if not n.startswith("_")])
    for fn in ("load_pcbnet", "load_classes", "load_norm"):
        show(f"checkpoint.{fn}", getattr(ck, fn, None))
except Exception as e:
    print("\ncheckpoint.py import failed:", type(e).__name__, e)

print("\nRESOLVED_CTOR_KWARG=" + resolved)
