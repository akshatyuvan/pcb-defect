"""Fail if README.md points at an image that is not in the repo.

A broken image on a GitHub README is the most visible way for this project to
look unfinished, and it is INVISIBLE locally: markdown previews resolve
relative paths against the filesystem, so a path that is right on disk but
wrong in git renders fine in Cursor and breaks on github.com.

Also reports figures that exist but nothing references - usually a rename, or
a figure worth adding.

Run from anywhere: sys.path fix-up is not needed since this imports nothing
from src, but paths are resolved against the repo root rather than cwd.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
readme = (ROOT / "README.md").read_text()

# Markdown image syntax only: ![alt](path). Skips http(s), which GitHub fetches
# rather than resolving from the repo.
referenced = [
    p for p in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme)
    if not p.startswith(("http://", "https://"))
]

missing = [p for p in referenced if not (ROOT / p).exists()]

# Existing on disk is not enough - it must be COMMITTED. A gitignored or merely
# untracked figure resolves locally and 404s on GitHub, which is exactly the
# failure this script exists to catch.
tracked = set(
    subprocess.run(
        ["git", "ls-files", "artifacts/figures"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
)
untracked_refs = [p for p in referenced if (ROOT / p).exists() and p not in tracked]

on_disk = sorted(
    str(p.relative_to(ROOT))
    for p in (ROOT / "artifacts" / "figures").glob("*")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg"}
)
unreferenced = [p for p in on_disk if p not in referenced]

print(f"referenced: {len(referenced)}   on disk: {len(on_disk)}   tracked: {len(tracked)}")
for p in missing:
    print(f"  MISSING     (README points here, no such file): {p}")
for p in untracked_refs:
    print(f"  UNTRACKED   (exists locally, not in git - will 404 on GitHub): {p}")
for p in unreferenced:
    print(f"  unreferenced (exists, README ignores it): {p}")

sys.exit(1 if (missing or untracked_refs) else 0)