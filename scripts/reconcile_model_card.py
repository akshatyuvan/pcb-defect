"""Bring the staged bundle's metadata in line with what Days 7-8 measured.

Runs on the Mac. Idempotent - safe to run repeatedly.

Why a script and not a hand edit: these two JSON files are COPY'd into both
Docker images and asserted on by tests/test_artifacts_consistency.py. A hand
edit that drops a comma fails at container start, not here. More importantly,
src/mlops/stage_model.py still emits the OLD wording, so re-running staging
resurrects the stale card - and then you re-run this.

Four fixes:
  1. operating_point.note editorialised about BOARD-level behaviour that was
     true on Day 5 and false since Day 7. The operating_point block itself is
     correct and stays - it is the PATCH-level threshold (0.000416). Only the
     note is rewritten, to describe what the block is and point at the board
     block for the other thing. Two scoped keys beat one key with an opinion
     about its neighbour.
  2. review_threshold: null is removed. A null threshold is what caused the
     confusion; deleting it is better than blanking it.
  3. board_operating_point cross-references board_calibration.json rather than
     duplicating routing logic. One definition, pointed at from here.
  4. export_crosscheck_max_logit_diff is lifted out of input_spec, where it did
     not belong - it is a provenance fact, not an input contract.

And in board_calibration.json: record the OUTCOME of its own specificity
caveat. The caveat was written before it was tested; Day 8 measured 9.7% false
alarms on unseen-group templates against ~2% predicted. That belongs next to
the thresholds it qualifies, not only in a README.
"""

import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts" / "serving"
CARD = ART / "model_card.json"
CAL = ART / "board_calibration.json"

STALE_MARKER = "NOT yet calibrated at board level"

PATCH_NOTE = (
    "PATCH-level operating point from Day 2: threshold 0.000416 at recall 0.97 "
    "on the binary defect task. It applies to single 64x64 patches ONLY. It is "
    "NOT the board routing rule - see board_operating_point. Reusing it as a "
    "max-over-patches board rule was the Day 5 mistake: it sits below the "
    "model's ~0.0171 characteristic output on ordinary copper, so it flagged 95 "
    "of 100 tiles on a real board (n_flagged AUROC 0.5877, near chance)."
)


def _load(p):
    if not p.exists():
        raise FileNotFoundError(f"expected staged bundle file: {p}")
    return json.loads(p.read_text())


def _save(p, obj):
    # indent=2 + trailing newline so `git diff` on these files stays readable.
    p.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _strip_stale_notes(node, path=""):
    """Recursively rewrite any string containing the stale board-level claim.

    Recursive on purpose. The first version of this script guessed at three
    top-level key names and missed operating_point.note, which sits one level
    down - so the reconciliation reported success while the card still
    contradicted itself. Guessing at structure is exactly how that happened,
    and it is the same failure mode we removed from checkpoint.py's tolerant
    parsing. Walk it instead.

    Returns the list of paths it changed, so the run prints proof of work
    rather than an unconditional 'updated'.
    """
    changed = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and STALE_MARKER in v:
                node[k] = PATCH_NOTE
                changed.append(f"{path}.{k}")
            else:
                changed.extend(_strip_stale_notes(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            changed.extend(_strip_stale_notes(v, f"{path}[{i}]"))
    return changed


def fix_card(card, cal):
    changed = _strip_stale_notes(card)

    # A null review_threshold is superseded by pass_threshold in the
    # calibration file, which is a MEASUREMENT rather than an invented number.
    for parent in (card, card.get("operating_point", {})):
        if isinstance(parent, dict) and parent.pop("review_threshold", None) is not None:
            changed.append(".review_threshold (removed)")

    # Cross-reference, do not duplicate. The thresholds are copied in for
    # readability but `source` is the authoritative pointer.
    card["board_operating_point"] = {
        "source": "artifacts/serving/board_calibration.json",
        "statistic": cal.get("statistic", "top3_mean"),
        "pass_threshold": cal.get("pass_threshold"),
        "fail_threshold": cal.get("fail_threshold"),
        "note": (
            "BOARD-level routing rule, calibrated on Day 7 against DeepPCB "
            "_temp.jpg templates - the only defect-free boards the dataset "
            "ships. top3_mean over the 100-patch defect-score grid, selected "
            "from nine candidate statistics on specificity at recall >= 0.99 "
            "(AUROC 0.9988). Three outcomes: pass / review / fail."
        ),
    }

    # Provenance fact, not an input contract. Walk one level so this does not
    # depend on the parent key's exact spelling.
    for parent_key, parent_val in list(card.items()):
        if isinstance(parent_val, dict) and "export_crosscheck_max_logit_diff" in parent_val:
            card["export_crosscheck_max_logit_diff"] = parent_val.pop(
                "export_crosscheck_max_logit_diff"
            )
            changed.append(f".{parent_key}.export_crosscheck_max_logit_diff (moved to top level)")

    return card, changed


def fix_calibration(cal):
    cal["specificity_caveat_outcome"] = {
        "predicted_false_alarm_rate": 0.02,
        "measured_false_alarm_rate": 0.097,
        "measured_on": "93 _temp.jpg templates from the 500-board test split",
        "measured_when": "day8",
        "detail": (
            "9 of 93 test templates alerted (6 fail, 3 review). Every false alarm "
            "came from a capture group absent from trainval.txt: 90100 x6, "
            "12000 x2, 12300 x1. Zero came from the 7 groups calibration saw. "
            "Scores ranged 0.987183-0.999941."
        ),
        "caveat_on_the_caveat": (
            "9 alerts on 93 templates is a small sample; 9.7% has wide error bars. "
            "Thresholds were fitted at n=150 on 7 of 11 capture groups."
        ),
    }
    return cal


def main():
    card, cal = _load(CARD), _load(CAL)

    _save(CAL, fix_calibration(cal))
    card, changed = fix_card(card, cal)
    _save(CARD, card)

    print(f"updated {CAL}")
    print(f"updated {CARD}")
    if changed:
        for path in changed:
            print(f"  changed: {path}")
    else:
        print("  no stale fields found (already reconciled)")

    # Fail loudly if the marker survived anywhere. Without this the script can
    # report success while the card still contradicts itself - which is
    # precisely what happened on the first run.
    remaining = json.dumps(_load(CARD))
    if STALE_MARKER in remaining:
        raise SystemExit(f"FAILED: '{STALE_MARKER}' still present in {CARD}")
    print("verified: no stale board-level claim remains")


if __name__ == "__main__":
    main()