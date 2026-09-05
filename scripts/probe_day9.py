"""Print the real public API of every module the Day 9 tests touch.

Runs on the Mac in (pcb). Needs no broker, no API, no dataset.
"""
import sys
from pathlib import Path

# pyproject.toml's pythonpath=["."] is a pytest setting only. Running this as a
# plain script puts scripts/ on sys.path, not the repo root, so `import src`
# fails. Every standalone script in this repo needs this or a `python -m` call.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
import inspect

MODULES = [
    "src.serving.preprocess",
    "src.streaming.schema",
    "src.streaming.board_stats",
    "src.streaming.results",
    "src.streaming.policy",
    "src.streaming.producer",
    "src.models.checkpoint",
]

for name in MODULES:
    print("=" * 72)
    print(name)
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
        continue
    for attr in sorted(dir(mod)):
        if attr.startswith("_"):
            continue
        obj = getattr(mod, attr)
        # Skip re-exported stdlib names so the output is just OUR surface.
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isfunction(obj):
            print(f"  def {attr}{inspect.signature(obj)}")
        elif inspect.isclass(obj):
            print(f"  class {attr}")
            for mname, mobj in inspect.getmembers(obj, inspect.isfunction):
                if mname.startswith("_") and mname != "__init__":
                    continue
                try:
                    print(f"        {mname}{inspect.signature(mobj)}")
                except (TypeError, ValueError):
                    print(f"        {mname}(?)")
    for attr in sorted(dir(mod)):
        if attr.isupper() and not attr.startswith("_"):
            print(f"  {attr} = {getattr(mod, attr)!r}")