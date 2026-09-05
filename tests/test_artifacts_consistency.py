"""The staged bundle must be self-consistent and loadable, from a clean clone.

artifacts/serving/ is the ONLY committed model artifact. mlflow.db, mlruns/ and
artifacts/checkpoints/ are gitignored, so if this bundle drifts there is no
second copy to check against - a fresh clone would build an image that starts
cleanly and serves wrong numbers. CI runs this on every push because nothing
else guards it.
"""

import json
from pathlib import Path

import pytest
import torch

from src.models.checkpoint import load_classes, load_norm
from src.models.cnn import PCBNet
from src.streaming.policy import BoardCalibration

ART = Path(__file__).resolve().parents[1] / "artifacts" / "serving"

EXPECTED_CLASSES = ["good", "open", "short", "mousebite", "spur", "copper", "pinhole"]


def _json(name):
    return json.loads((ART / name).read_text())


def test_every_bundle_file_is_committed():
    for name in ("model.pt", "classes.json", "norm.json", "model_card.json",
                 "board_calibration.json", "test_board_ids.json"):
        assert (ART / name).exists(), f"missing from staged bundle: {name}"


def test_classes_are_the_seven_deeppcb_classes_in_annotation_order():
    """Annotation class ids 1-6 map to these defect names in this order, with
    good at 0. If the order rotates, accuracy is unchanged and every per-class
    metric in the README becomes a lie."""
    assert load_classes(ART) == EXPECTED_CLASSES


def test_norm_matches_the_training_statistics():
    """Measured on the 850 training boards on Day 1. Serving must normalise
    with the same constants the weights were fitted under; a mismatch produces
    no error, only quietly worse predictions."""
    mean, std = load_norm(ART)
    assert mean == pytest.approx(0.6472, abs=1e-4)
    assert std == pytest.approx(0.4760, abs=1e-4)


def test_model_pt_loads_into_pcbnet_and_exposes_the_cam_feature_map():
    """The bundle is a bare state_dict, not a pickled module, so a refactor of
    PCBNet breaks HERE rather than silently failing to deserialise in a
    container. The (B,256,4,4) assertion guards the GAP head specifically: a
    Flatten head would still classify fine and quietly make Grad-CAM
    impossible, since there would be no spatial feature map to weight."""
    state = torch.load(ART / "model.pt", map_location="cpu", weights_only=True)
    net = PCBNet(num_classes=len(EXPECTED_CLASSES))
    net.load_state_dict(state, strict=True)
    net.eval()

    assert sum(p.numel() for p in net.parameters()) == 1_174_439

    with torch.no_grad():
        logits, feats = net.forward_with_features(torch.zeros(1, 1, 64, 64))
    assert tuple(logits.shape) == (1, 7)
    assert tuple(feats.shape) == (1, 256, 4, 4)


def test_board_calibration_loads_and_thresholds_are_ordered():
    cal = BoardCalibration.from_file(str(ART / "board_calibration.json"))
    assert cal.statistic == "top3_mean"
    assert cal.fail_threshold is not None, "a null fail_threshold must be refused at load"
    assert cal.pass_threshold is not None
    assert cal.pass_threshold < cal.fail_threshold, (
        "pass must sit below fail, otherwise route() has no review band and the "
        "three-outcome policy silently collapses to two"
    )


def test_model_card_no_longer_claims_the_board_point_is_uncalibrated():
    """Guard for the Day 9 reconciliation. src/mlops/stage_model.py still emits
    the OLD wording, so re-running staging would resurrect a model card that
    contradicts board_calibration.json. This fails loudly if that happens."""
    text = json.dumps(_json("model_card.json"))
    assert "NOT yet calibrated at board level" not in text
    assert "board_calibration.json" in text


def test_calibration_records_its_own_measured_false_alarm_rate():
    """The specificity caveat was written before it was tested. Day 8 measured
    9.7% false alarms on unseen-group templates against ~2% predicted, and that
    outcome belongs next to the thresholds it qualifies - not only in a README."""
    outcome = _json("board_calibration.json")["specificity_caveat_outcome"]
    assert outcome["measured_false_alarm_rate"] == pytest.approx(0.097)


def test_test_board_ids_is_a_bare_array_of_500():
    """The producer does set(json.loads(...)) on this file. Handed a JSON
    object it would iterate over KEYS and silently stream a different board
    set - no error, just a different experiment than the one you ran."""
    ids = json.loads((ART / "test_board_ids.json").read_text())
    assert isinstance(ids, list), "must be a bare array, not an object"
    assert len(ids) == 500