"""
test_cli.py
-----------
Tests for tsd.cli.

`main()` is called directly with capsys -- no subprocess, so a failure
shows a Python traceback instead of an exit code and a silent shell.

The fixtures come from tests/test_verdict.py rather than being written
again: the stub artefact and the synthetic PCAP are the same objects,
and two copies of a fixture drift the moment one of them is updated.

What matters most here is the stream discipline. stdout carries the
verdict document or nothing; everything else goes to stderr. That is
not tidiness -- the tool has to pipe into jq unfiltered, and a
half-written document on a failure path is worse than none, because a
consumer reading it would not notice.
"""

from __future__ import annotations

import json

import pytest

from test_verdict import (  # noqa: E402 - tests/ is on sys.path under pytest
    NoProbaPipeline,
    frame,
    page_load,
    write_artefact,
    write_pcap,
)
from tsd.cli import EXIT_OK, EXIT_VERDICT_ERROR, build_parser, main
from tsd.features import SERVER_PORT, feature_names

TCP_ACK = 0x10
TCP_SYN = 0x02


@pytest.fixture
def model(tmp_path):
    return write_artefact(tmp_path / "model.joblib")


@pytest.fixture
def pcap(tmp_path):
    return page_load(tmp_path / "trace.pcap")


# An explicit prog, only here. The shipped parser leaves it None so
# argparse derives the name from sys.argv[0] and --help announces the
# command the reader actually typed. Under pytest that would be the test
# runner, so the tests that assert on the error prefix supply the name
# themselves rather than the parser hardcoding one for their benefit.
TEST_PROG = "classify_pcap"


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(argument) for argument in argv], prog=TEST_PROG)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------
# A verdict
# --------------------------------------------------------------

def test_valid_run_prints_json_and_returns_zero(capsys, model, pcap):
    code, out, err = run(capsys, pcap, "--model", model)

    assert code == EXIT_OK
    assert err == ""

    document = json.loads(out)
    assert list(document) == ["schema_version", "pcap", "verdict", "model"]
    assert document["verdict"]["client"] == "wget"
    assert document["pcap"]["server_port"] == SERVER_PORT


def test_default_output_is_indented(capsys, model, pcap):
    _, out, _ = run(capsys, pcap, "--model", model)

    assert out.startswith("{\n")
    assert '\n  "verdict"' in out


def test_compact_output_is_one_line(capsys, model, pcap):
    _, out, _ = run(capsys, pcap, "--model", model, "--compact")

    assert out.endswith("\n")
    assert "\n" not in out[:-1], "compact output must be a single line"
    assert ", " not in out and '": ' not in out
    assert json.loads(out)["verdict"]["client"] == "wget"


def test_key_order_is_not_sorted(capsys, model, pcap):
    """
    verdict.py fixes a deliberate order; sorting would replace a schema
    chosen for reading with one chosen by the alphabet.
    """
    _, out, _ = run(capsys, pcap, "--model", model)
    keys = list(json.loads(out))

    assert keys == ["schema_version", "pcap", "verdict", "model"]
    assert keys != sorted(keys)


# --------------------------------------------------------------
# Failure leaves stdout untouched
# --------------------------------------------------------------

def test_missing_model_returns_three_and_writes_nothing_to_stdout(
    capsys, tmp_path, pcap
):
    """
    stdout must be EMPTY, not merely invalid. A partial document would
    flow into a pipeline that has no way to tell it apart from a whole
    one.
    """
    code, out, err = run(capsys, pcap, "--model", tmp_path / "absent.joblib")

    assert code == EXIT_VERDICT_ERROR
    assert out == ""
    assert "classify_pcap:" in err
    assert "train_model.py" in err


def test_unreadable_pcap_returns_three_with_empty_stdout(capsys, tmp_path, model):
    broken = tmp_path / "broken.pcap"
    broken.write_bytes(b"not a pcap")

    code, out, err = run(capsys, broken, "--model", model)

    assert code == EXIT_VERDICT_ERROR
    assert out == ""
    assert err.startswith("classify_pcap:")


def test_too_short_trace_returns_three_with_empty_stdout(capsys, tmp_path, model):
    short = page_load(tmp_path / "short.pcap", packets=3)

    code, out, err = run(capsys, short, "--model", model)

    assert code == EXIT_VERDICT_ERROR
    assert out == ""
    assert "capture filter" in err


def test_stale_model_returns_three(capsys, tmp_path, pcap):
    stale = write_artefact(
        tmp_path / "stale.joblib", features=[*feature_names(), "iat_up_kurtosis"]
    )

    code, out, err = run(capsys, pcap, "--model", stale)

    assert code == EXIT_VERDICT_ERROR
    assert out == ""
    assert "iat_up_kurtosis" in err


def test_usage_error_exits_two(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2


# --------------------------------------------------------------
# The exit code is not the verdict
# --------------------------------------------------------------

def test_exit_code_does_not_encode_the_predicted_class(capsys, tmp_path):
    """
    `classify_pcap x.pcap || echo failed` must not print "failed" just
    because the answer was wget. Exit status answers "did the tool
    work", not "what did it find".
    """
    trace = page_load(tmp_path / "t.pcap")

    wget_model = write_artefact(tmp_path / "wget.joblib")
    firefox_model = write_artefact(
        tmp_path / "ff.joblib", pipeline=NoProbaPipeline("firefox")
    )

    wget_code, wget_out, _ = run(capsys, trace, "--model", wget_model)
    firefox_code, firefox_out, _ = run(capsys, trace, "--model", firefox_model)

    assert wget_code == firefox_code == EXIT_OK
    assert json.loads(wget_out)["verdict"]["client"] == "wget"
    assert json.loads(firefox_out)["verdict"]["client"] == "firefox"


# --------------------------------------------------------------
# Flags reach verdict.py
# --------------------------------------------------------------

def test_include_features_adds_the_key_and_its_absence_omits_it(
    capsys, model, pcap
):
    _, without, _ = run(capsys, pcap, "--model", model)
    _, with_features, _ = run(capsys, pcap, "--model", model, "--include-features")

    assert "features" not in json.loads(without)

    document = json.loads(with_features)
    assert "features" in document
    assert set(document["features"]) == set(feature_names())


def test_server_port_is_threaded_through(capsys, tmp_path, model):
    """
    The port decides packet direction, so a wrong one yields a trace
    with nothing in it rather than a mirrored one. Here the capture is
    on 9443: it classifies only when the flag says so.
    """
    port = 9443
    records = [
        (0.000, frame(50000, port, 0, TCP_SYN)),
        (0.001, frame(port, 50000, 0, TCP_SYN | TCP_ACK)),
        (0.002, frame(50000, port, 0)),
        (0.003, frame(50000, port, 517)),
        (0.010, frame(port, 50000, 1400)),
        (0.012, frame(port, 50000, 1400)),
    ]
    path = write_pcap(tmp_path / "alt.pcap", records)

    default_code, default_out, _ = run(capsys, path, "--model", model)
    assert default_code == EXIT_VERDICT_ERROR
    assert default_out == ""

    code, out, _ = run(capsys, path, "--model", model, "--server-port", port)
    assert code == EXIT_OK
    assert json.loads(out)["pcap"]["server_port"] == port


# --------------------------------------------------------------
# The parser itself
# --------------------------------------------------------------

def test_parser_defaults():
    args = build_parser().parse_args(["some.pcap"])

    assert str(args.model) == "models/client_classifier.joblib"
    assert args.server_port == SERVER_PORT
    assert args.include_features is False
    assert args.compact is False


def test_parser_documents_the_exit_codes():
    """
    A caller deciding how to branch should not have to read the source.
    """
    epilog = build_parser().epilog

    assert "exit codes" in epilog
    assert "never encodes the predicted class" in epilog
