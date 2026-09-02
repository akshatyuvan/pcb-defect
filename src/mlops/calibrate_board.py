"""
Day 7, step 3. Turn artifacts/day7_board_scores_val.jsonl into a board-level
operating point.

THE PROBLEM BEING SOLVED
------------------------
Day 2 chose 0.000416 as a PATCH-level threshold at 97% patch recall. Day 5/6
then applied it at BOARD level as max-over-100-patches, which flagged 95/100
and 99/100 tiles on the two boards measured. That is not a bug, it is a
category error: a patch operating point is not a board operating point, and
giving a low threshold 100 independent chances to be exceeded makes the max
carry almost no information.

THE NEGATIVES
-------------
DeepPCB has no clean _test.jpg -- every imaged board carries 3-12 defects. The
only defect-free boards in the dataset are the _temp.jpg templates, which is
precisely why Day 3's template differencing could use them as references. So
the calibration set is val _test.jpg (positive) vs val _temp.jpg (negative).

Caveat, written into the output file rather than left in a comment: templates
are reference captures and may be systematically cleaner than a real clean
board off a line. Measured specificity is therefore an upper bound.

METHOD
------
  fail_threshold = smallest t where PRECISION on flagged boards >= 0.99
                   (essentially no clean board reaches this -> confident fail)
  pass_threshold = largest t where RECALL on defective boards >= 0.99
                   (essentially no defective board falls below -> confident pass)
  between them   -> review, route to a human. This is the third outcome the
                   service has been degrading out of since review_threshold
                   was null.

If pass_threshold >= fail_threshold the classes separate cleanly and the review
band is empty. We report that rather than manufacturing a band to fill.

Statistic selection mirrors locked decision 6: not accuracy, but the cost at a
fixed recall. Here the cost is specificity on clean boards at defect recall
>= 0.99.

Runs on the MAC.
    python -m src.mlops.calibrate_board
    python -m src.mlops.calibrate_board --write
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")            # no display on this box; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

from src.streaming.board_stats import compute_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts" / "day7_board_scores_val.jsonl"
OUT_JSON = ROOT / "artifacts" / "serving" / "board_calibration.json"
OUT_FIG = ROOT / "artifacts" / "figures" / "day7_board_calibration.png"

RECALL_TARGET = 0.99      # of defective boards, sets the pass line
PRECISION_TARGET = 0.99   # of flagged boards, sets the fail line


def sweep(pos: List[float], neg: List[float]) -> List[Tuple[float, float, float, float]]:
    """For each candidate threshold return (t, recall, specificity, precision).
    Rule everywhere, including in production: flag when score >= t."""
    cands = sorted(set(pos + neg))
    cands = [cands[0] - 1e-12] + cands       # a point where everything flags
    rows = []
    for t in cands:
        tp = sum(1 for x in pos if x >= t)
        fp = sum(1 for x in neg if x >= t)
        fn = len(pos) - tp
        tn = len(neg) - fp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rows.append((t, recall, spec, prec))
    return rows


def auroc(pos: List[float], neg: List[float]) -> float:
    """Mann-Whitney U / (n_pos * n_neg). Ties count as half.

    O(n*m) = 22,500 comparisons here, which is nothing. Written the obvious way
    rather than the fast way because it is easier to defend in an interview."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def evaluate(name: str, pos: List[float], neg: List[float]) -> Dict:
    rows = sweep(pos, neg)
    ok_recall = [r for r in rows if r[1] >= RECALL_TARGET]
    pass_t = max(r[0] for r in ok_recall) if ok_recall else None
    ok_prec = [r for r in rows if r[3] >= PRECISION_TARGET and r[1] > 0]
    fail_t = min(r[0] for r in ok_prec) if ok_prec else None
    spec_at_recall = max((r[2] for r in ok_recall), default=0.0)
    return {
        "statistic": name,
        "auroc": round(auroc(pos, neg), 4),
        "pass_threshold": pass_t,
        "fail_threshold": fail_t,
        "specificity_at_recall_target": round(spec_at_recall, 4),
        "rows": rows,
    }


def confusion(pos, neg, pass_t, fail_t) -> Dict:
    """Apply the chosen thresholds back to the calibration set. This is
    optimistic by construction -- the thresholds were fitted here -- so it is
    reported as 'routing on the calibration set', never as a test result."""
    def label(x):
        if fail_t is not None and x >= fail_t:
            return "fail"
        if pass_t is not None and x < pass_t:
            return "pass"
        return "review"

    d = {"defective": {"pass": 0, "review": 0, "fail": 0},
         "clean": {"pass": 0, "review": 0, "fail": 0}}
    for x in pos:
        d["defective"][label(x)] += 1
    for x in neg:
        d["clean"][label(x)] += 1
    n = len(pos) + len(neg)
    d["review_rate"] = round((d["defective"]["review"] + d["clean"]["review"]) / n, 4)
    d["missed_defective"] = d["defective"]["pass"]     # the expensive error
    d["false_alarm_clean"] = d["clean"]["fail"]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=DEFAULT_IN)
    ap.add_argument("--statistic", default=None,
                    help="force one instead of auto-selecting")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.scores.read_text().splitlines() if l.strip()]
    by_stat: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        for k, v in compute_stats(r["response"]).items():
            by_stat.setdefault(k, {"defective": [], "clean": []})[r["label"]].append(v)

    n_def = sum(1 for r in rows if r["label"] == "defective")
    n_cln = sum(1 for r in rows if r["label"] == "clean")
    print(f"boards: {n_def} defective (_test.jpg), {n_cln} clean (_temp.jpg)")
    print(f"candidate statistics: {sorted(by_stat)}\n")

    evals = []
    for name, d in sorted(by_stat.items()):
        if len(d["defective"]) < 2 or len(d["clean"]) < 2:
            continue
        evals.append(evaluate(name, d["defective"], d["clean"]))
    if not evals:
        print("no usable statistics -- is the scores file complete?")
        return 1

    hdr = (f"{'statistic':<22s} {'AUROC':>7s} {'spec@R>=.99':>12s} "
           f"{'pass_t':>12s} {'fail_t':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for e in sorted(evals, key=lambda x: -x["specificity_at_recall_target"]):
        pt = "n/a" if e["pass_threshold"] is None else f"{e['pass_threshold']:.6g}"
        ft = "n/a" if e["fail_threshold"] is None else f"{e['fail_threshold']:.6g}"
        print(f"{e['statistic']:<22s} {e['auroc']:>7.4f} "
              f"{e['specificity_at_recall_target']:>12.4f} {pt:>12s} {ft:>12s}")

    chosen = (next(e for e in evals if e["statistic"] == args.statistic)
              if args.statistic else
              max(evals, key=lambda e: (e["specificity_at_recall_target"], e["auroc"])))
    pos = by_stat[chosen["statistic"]]["defective"]
    neg = by_stat[chosen["statistic"]]["clean"]
    conf = confusion(pos, neg, chosen["pass_threshold"], chosen["fail_threshold"])

    print(f"\nselected: {chosen['statistic']}")
    print(f"  pass  < {chosen['pass_threshold']}")
    print(f"  fail >= {chosen['fail_threshold']}")
    print(f"  routing on the calibration set:\n{json.dumps(conf, indent=2)}")

    if chosen["fail_threshold"] is None:
        print("\nWARNING: no threshold reaches 99% precision on flagged boards. "
              "The statistic does not separate. Try --statistic to force another, "
              "or accept that board-level separation is not achievable here and "
              "report that as the finding.")
    if (chosen["pass_threshold"] is not None and chosen["fail_threshold"] is not None
            and chosen["pass_threshold"] >= chosen["fail_threshold"]):
        print("\nNOTE: pass_threshold >= fail_threshold. The classes separate "
              "cleanly; the review band is empty on this data. Routing will be "
              "effectively two-outcome and that is an honest result, not a bug.")

    # ------------------------------------------------------------------ figure
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.hist([neg, pos], bins=30, label=["clean (_temp)", "defective (_test)"],
             color=["#2b8cbe", "#e34a33"])
    for t, lab, c in ((chosen["pass_threshold"], "pass line", "#2b8cbe"),
                      (chosen["fail_threshold"], "fail line", "#e34a33")):
        if t is not None:
            ax1.axvline(t, ls="--", color=c, label=f"{lab} = {t:.4g}")
    ax1.set_xlabel(chosen["statistic"])
    ax1.set_ylabel("boards")
    ax1.set_title(f"Board-level separation (val, n={len(pos)}+{len(neg)})")
    ax1.legend(fontsize=8)

    for e in sorted(evals, key=lambda x: -x["specificity_at_recall_target"])[:5]:
        ax2.plot([r[1] for r in e["rows"]], [r[2] for r in e["rows"]],
                 marker=".", ms=3, lw=1, label=e["statistic"])
    ax2.axvline(RECALL_TARGET, ls=":", color="k", lw=1)
    ax2.set_xlabel("defect-board recall")
    ax2.set_ylabel("clean-board specificity")
    ax2.set_title("Specificity vs recall, per candidate statistic")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT_FIG.relative_to(ROOT)}")

    if not args.write:
        print("Dry run. Re-run with --write to save the calibration.")
        return 0

    doc = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statistic": chosen["statistic"],
        "pass_threshold": chosen["pass_threshold"],
        "fail_threshold": chosen["fail_threshold"],
        "patch_threshold": 0.000416,
        "targets": {"defect_recall": RECALL_TARGET,
                    "flag_precision": PRECISION_TARGET},
        "calibrated_on": {
            "split": "val",
            "n_defective": n_def,
            "n_clean_templates": n_cln,
            "source": str(args.scores.relative_to(ROOT)),
        },
        "measured": {
            "auroc": chosen["auroc"],
            "specificity_at_recall_target": chosen["specificity_at_recall_target"],
            "routing_on_calibration_set": conf,
        },
        "alternatives": [
            {k: e[k] for k in ("statistic", "auroc",
                               "specificity_at_recall_target",
                               "pass_threshold", "fail_threshold")}
            for e in sorted(evals, key=lambda x: -x["specificity_at_recall_target"])
        ],
        "caveats": [
            "Negatives are DeepPCB _temp.jpg templates, the dataset's reference "
            "captures. They are defect-free by construction but may be "
            "systematically cleaner than a real clean board off a QC line, so "
            "specificity here is an upper bound.",
            "Calibrated on 150 val boards plus their templates. Confidence "
            "intervals at n=150 are wide; do not read the third decimal.",
            "The confusion matrix under 'measured' is on the calibration set "
            "itself and is optimistic by construction.",
            "0.000416 remains the PATCH-level threshold from Day 2 and is "
            "unchanged. This file adds a separate BOARD-level operating point.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())