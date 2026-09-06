#!/usr/bin/env bash
# Restore the committed alert log after a demo recording.
#
# The demo run appends a handful of alerts to a fresh file. This puts the
# original Day 8 evidence back so `git status` stays clean.
set -euo pipefail

LOG=artifacts/alerts/alerts.jsonl

if [ -f "${LOG}.bak" ]; then
  mv -f "${LOG}.bak" "$LOG"
  echo "restored ${LOG} from backup"
  wc -l "$LOG"
else
  echo "no ${LOG}.bak found - nothing to restore"
fi