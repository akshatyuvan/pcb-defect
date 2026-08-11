#!/usr/bin/env bash
# Fetch DeepPCB into data/raw/.
# Why a shell script and not a notebook cell: Day 3's classical baseline and
# Day 9's CI both need a reproducible one-command data fetch. Notebook cells are
# not reusable from another notebook or from a test runner.
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

if [ -d "$DEST/PCBData" ]; then
  echo "PCBData already present at $DEST/PCBData, skipping clone."
else
  # Shallow clone: we only want the current tree, not the full image history.
  git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git "$DEST/_deeppcb_tmp"
  mv "$DEST/_deeppcb_tmp/PCBData" "$DEST/PCBData"
  # trainval.txt / test.txt sometimes sit at the repo root rather than inside
  # PCBData. Copy them in either way so downstream code has one canonical location.
  cp "$DEST/_deeppcb_tmp"/*.txt "$DEST/PCBData/" 2>/dev/null || true
  rm -rf "$DEST/_deeppcb_tmp"
fi

echo "--- sanity ---"
echo "groups:       $(find "$DEST/PCBData" -maxdepth 1 -type d -name 'group*' | wc -l)"
echo "test images:  $(find "$DEST/PCBData" -name '*_test.jpg' | wc -l)"
echo "templates:    $(find "$DEST/PCBData" -name '*_temp.jpg' | wc -l)"
echo "index files:  $(ls "$DEST/PCBData"/*.txt 2>/dev/null | wc -l)"