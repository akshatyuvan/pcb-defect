"""
Day 7, step 0. Read-only introspection of everything Day 7 must build against.

Why this exists: today's consumer talks to Day 5's schemas and Day 6's message
contract. A wrong field name in a consumer is not a loud crash -- it is a 422
from the API, which the consumer routes to the DLQ, which means "silently
dead-letter every single board". That is the worst failure mode available here,
so we spend 20 seconds printing the truth instead of assuming it.

Run from the repo root:
    python scripts/probe_contracts_day7.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_module(mod, label: str) -> None:
    """Print public callables + signatures + module-level constants."""
    hr(f"{label} -- public surface")
    for name, obj in sorted(vars(mod).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            # only things actually defined here, not re-exported imports
            if getattr(obj, "__module__", "") != mod.__name__:
                continue
            try:
                print(f"{name}{inspect.signature(obj)}")
            except (TypeError, ValueError):
                print(name)
            if inspect.isclass(obj):
                fields = getattr(obj, "model_fields", None)
                if isinstance(fields, dict):
                    for fname, f in fields.items():
                        print(f"    .{fname:<22s} {f.annotation}")
                for mname, m in sorted(vars(obj).items()):
                    if mname.startswith("_") or not callable(m):
                        continue
                    try:
                        print(f"    .{mname}{inspect.signature(m)}")
                    except (TypeError, ValueError):
                        print(f"    .{mname}")
        elif isinstance(obj, (str, int, float, bool, tuple)):
            print(f"CONST {name:<24s} = {obj!r}")


hr("artifacts/serving/ contents")
sv = ROOT / "artifacts" / "serving"
for p in sorted(sv.glob("*")):
    print(f"{p.name:<28s} {p.stat().st_size:>12,d} bytes")

hr("model_card.json")
mc = sv / "model_card.json"
print(json.dumps(json.loads(mc.read_text()), indent=2) if mc.exists() else f"MISSING {mc}")

from src.serving import schemas  # noqa: E402
show_module(schemas, "src.serving.schemas")

from src.serving import inference  # noqa: E402
show_module(inference, "src.serving.inference")

hr("src.serving.inference -- decide() source (the current routing policy)")
_decide = getattr(inference, "decide", None) or getattr(
    getattr(inference, "Inferencer", object), "decide", None
)
print(inspect.getsource(_decide) if _decide else "decide not found")

from src.streaming import schema as msgschema  # noqa: E402
show_module(msgschema, "src.streaming.schema")

hr("src.streaming.producer -- board discovery / CLI flags")
from src.streaming import producer as prod  # noqa: E402
for name in ("discover_boards", "main", "parse_args", "build_parser"):
    fn = getattr(prod, name, None)
    if fn is None:
        continue
    try:
        print(f"\n--- {name}{inspect.signature(fn)} ---")
    except (TypeError, ValueError):
        print(f"\n--- {name} ---")
    try:
        print(inspect.getsource(fn))
    except OSError:
        pass