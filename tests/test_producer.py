"""The producer's board-selection logic, tested without a broker.

Nothing here connects to Kafka. The delivery path is covered by
scripts/verify_end_to_end.py and scripts/benchmark.py; what had never been
tested is the SELECTION logic, and that is where a quiet mistake changes which
boards an experiment actually ran on.

discover_boards reads from data/raw/PCBData, which is gitignored. These tests
skip cleanly when the dataset is absent so they pass in CI, and assert
properly on the Mac where it exists.
"""

import json

import pytest

from src.streaming import producer

_HAS_DATA = producer.RAW.exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="DeepPCB not present (gitignored)")


@needs_data
def test_boards_file_must_be_a_bare_array(tmp_path):
    """discover_boards does set(json.loads(...)). This is the happy path that
    pins the accepted format."""
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(["00041000", "00041001"]))

    boards = producer.discover_boards(str(ids_file))
    assert len(boards) == 2


@needs_data
def test_include_templates_doubles_the_stream(tmp_path):
    """Templates are derived from the ALREADY-FILTERED _test.jpg paths, so
    --include-templates must respect --boards-file rather than globbing the
    whole dataset. If it globbed, the benchmark's 'mixed workload' would
    quietly have been the full 1500-board set."""
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(["00041000", "00041001"]))

    without = producer.discover_boards(str(ids_file))
    with_temps = producer.discover_boards(str(ids_file), include_templates=True)

    assert len(with_temps) == 2 * len(without)


@needs_data
def test_template_ids_keep_the_temp_suffix(tmp_path):
    """Template board ids must stay distinguishable from their defective
    counterparts. They are used as the Kafka message KEY, so a collision would
    put a board and its own template on the same partition under the same key -
    and the alerting consumer's dedupe would suppress one of them."""
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(["00041000"]))

    boards = producer.discover_boards(str(ids_file), include_templates=True)
    joined = json.dumps(boards, default=str)
    assert "_temp" in joined, "template ids lost their suffix; keys would collide"


@needs_data
def test_boards_file_object_form_is_rejected_with_a_useful_message(tmp_path):
    """A JSON object here would iterate over KEYS: {"boards": [...]} yields the
    set {"boards"}, and {"00041000": ...} would yield a plausible but arbitrary
    board set with no error at all. discover_boards type-checks the shape.

    pytest.raises(SystemExit), not Exception: sys.exit raises SystemExit, which
    inherits from BaseException and is NOT caught by `except Exception`. An
    earlier draft used Exception and failed against working code - the mirror
    image of test_schema.py, where too-broad a catch hid a real bug.
    """
    bad = tmp_path / "ids.json"
    bad.write_text(json.dumps({"boards": ["00041000"]}))

    with pytest.raises(SystemExit) as exc:
        producer.discover_boards(str(bad))

    # Assert on the MESSAGE, not just the exit. Before the type guard this died
    # at "matched no boards on disk" - technically a rejection, but one that
    # pointed the user at the dataset instead of at their file.
    assert "bare JSON array" in str(exc.value)