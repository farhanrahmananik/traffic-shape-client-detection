"""
test_verify_cli_parity.py
-------------------------
Tests for scripts/verify_cli_parity.py.

A tiny synthetic pcap tree and CSV in tmp_path, built with the helpers
already shared by tests/test_verdict.py and tests/test_cli.py. Nothing
here reads data/ or models/, so the suite still passes on a clone that
has neither.

The check being tested is itself a check, so what matters is that it
FAILS when it should. A parity script that always exits 0 is worse than
no parity script: it turns an unverified property into a verified-
looking one.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_cli_parity as script  # noqa: E402

from test_verdict import page_load, write_artefact  # noqa: E402
from tsd.features import extract_features, feature_names, read_trace  # noqa: E402

ROUND = "round_01_20260807"
DATE = "20260807"
PAGES = ("index", "fakultaet1_e4b6727e14b7")


def build_tree(tmp_path: Path) -> tuple[Path, list[tuple[str, str, Path]]]:
    """A pcap root with two clients and two pages each."""
    pcap_root = tmp_path / "pcaps"
    made: list[tuple[str, str, Path]] = []

    for client, packets in (("firefox", 9), ("wget", 7)):
        for page in PAGES:
            path = page_load(
                pcap_root / ROUND / client / f"{page}.pcap", packets=packets
            )
            made.append((client, page, path))

    return pcap_root, made


def write_csv(path: Path, traces, perturb=None) -> Path:
    """
    A feature CSV built from the same extraction the training path uses.

    `perturb` optionally changes one value, so the mismatch path can be
    exercised on data that is otherwise identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = feature_names()

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["round", "date", "client", "page", *columns]
        )
        writer.writeheader()

        for client, page, pcap in traces:
            feats = extract_features(read_trace(pcap))
            row = {"round": 1, "date": DATE, "client": client, "page": page}
            row.update({name: feats[name] for name in columns})

            if perturb is not None:
                perturb(row)

            writer.writerow(row)

    return path


@pytest.fixture
def parity_inputs(tmp_path):
    pcap_root, traces = build_tree(tmp_path)
    model = write_artefact(tmp_path / "model.joblib")
    return pcap_root, traces, model


def run(pcap_root, features_csv, model, out, *extra) -> int:
    return script.main([
        "--pcap-root", str(pcap_root),
        "--features", str(features_csv),
        "--model", str(model),
        "--out", str(out),
        *extra,
    ])


def read_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------
# Agreement
# --------------------------------------------------------------

def test_identical_paths_agree_and_exit_zero(parity_inputs, tmp_path):
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    assert run(pcap_root, features_csv, model, out) == script.EXIT_OK

    record = read_record(out)
    assert record["comparison"]["traces_compared"] == 4
    assert record["comparison"]["traces_matching_all_features"] == 4
    assert record["comparison"]["mismatched_features"] == {}
    assert record["coverage"]["pcaps_without_csv_row"] == []
    assert record["coverage"]["csv_rows_without_pcap"] == []


def all_keys(node) -> set[str]:
    """Every key anywhere in the document."""
    if isinstance(node, dict):
        return set(node) | {k for value in node.values() for k in all_keys(value)}
    if isinstance(node, list):
        return {k for item in node for k in all_keys(item)}
    return set()


def test_the_record_never_mentions_accuracy_or_a_predicted_label(
    parity_inputs, tmp_path
):
    """
    The model was fitted on all four rounds, so every PCAP here is a
    training row. Any accuracy figure would measure a model reproducing
    data it has already seen.

    Asserted on the document's KEYS, not on the raw text. Two things
    defeat a text search here, and both are correct: the client name
    legitimately appears inside a trace identifier
    (`round_01_20260807/wget/index`), because that is where the PCAP was
    found; and the record's own `check` field contains the word
    "accuracy" precisely in order to say that none was computed. What
    must not exist is a FIELD holding a prediction.
    """
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"
    run(pcap_root, features_csv, model, out)

    record = read_record(out)
    keys = all_keys(record)

    forbidden = {"accuracy", "predicted", "prediction", "verdict",
                 "probabilities", "client", "label", "correct"}
    assert not keys & forbidden, f"the record kept a verdict field: {keys & forbidden}"

    # The stub always predicts "wget". It appears only as part of the
    # directory-derived trace key, never as a stored answer.
    for value in record["comparison"].values():
        assert value != "wget"

    # Every example describes a feature comparison and nothing else.
    for example in record["comparison"]["examples"]:
        assert set(example) == {
            "trace", "feature", "training_path", "inference_path", "abs_diff"
        }


def test_the_record_is_byte_identical_across_runs(parity_inputs, tmp_path):
    """
    Same rule as explain_model.py: no timestamp, no absolute paths, and
    a sorted walk. A record that changes on every run cannot be diffed
    to answer "did anything move".
    """
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    run(pcap_root, features_csv, model, first)
    run(pcap_root, features_csv, model, second)

    assert first.read_bytes() == second.read_bytes()


def test_the_record_carries_no_absolute_paths(parity_inputs, tmp_path):
    """
    Relative, so the record does not depend on where the repository
    happens to live. Under pytest that relative path walks up out of the
    repo, which is why the assertion is "not absolute" rather than "does
    not contain tmp_path".
    """
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"
    run(pcap_root, features_csv, model, out)

    record = read_record(out)

    assert not record["model"]["path"].startswith("/")
    assert not Path(record["model"]["path"]).is_absolute()
    for trace in record["comparison"]["examples"]:
        assert not trace["trace"].startswith("/")


# --------------------------------------------------------------
# Divergence -- the paths that must fail
# --------------------------------------------------------------

def test_a_single_changed_value_is_caught(parity_inputs, tmp_path, capsys):
    """
    Exact equality after identical rounding: a tolerance would decide in
    advance how much drift is acceptable, which is the question being
    asked.
    """
    pcap_root, traces, model = parity_inputs

    def bump(row):
        if row["client"] == "wget" and row["page"] == "index":
            row["iat_max"] = float(row["iat_max"]) + 0.000002

    features_csv = write_csv(tmp_path / "features.csv", traces, perturb=bump)
    out = tmp_path / "parity.json"

    assert run(pcap_root, features_csv, model, out) == script.EXIT_DIVERGED

    record = read_record(out)
    assert record["comparison"]["traces_matching_all_features"] == 3
    assert record["comparison"]["mismatched_features"] == {"iat_max": 1}

    example = record["comparison"]["examples"][0]
    assert example["feature"] == "iat_max"
    assert example["trace"] == f"{ROUND}/wget/index"
    assert example["abs_diff"] > 0

    assert "DIVERGENCE" in capsys.readouterr().err


def test_a_pcap_with_no_csv_row_is_reported(parity_inputs, tmp_path):
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces[:-1])
    out = tmp_path / "parity.json"

    assert run(pcap_root, features_csv, model, out) == script.EXIT_DIVERGED

    coverage = read_record(out)["coverage"]
    assert len(coverage["pcaps_without_csv_row"]) == 1
    assert coverage["csv_rows_without_pcap"] == []


def test_a_csv_row_with_no_pcap_is_reported(parity_inputs, tmp_path):
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    (pcap_root / ROUND / "wget" / "index.pcap").unlink()

    assert run(pcap_root, features_csv, model, out) == script.EXIT_DIVERGED

    coverage = read_record(out)["coverage"]
    assert coverage["csv_rows_without_pcap"] == [f"{ROUND}/wget/index"]


def test_a_trace_the_cli_refuses_is_a_divergence(parity_inputs, tmp_path):
    """
    The training path accepted it and the shipped path will not. That is
    a difference between the two, not a broken file.
    """
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    (pcap_root / ROUND / "wget" / "index.pcap").write_bytes(b"not a pcap")

    assert run(pcap_root, features_csv, model, out) == script.EXIT_DIVERGED

    refused = read_record(out)["comparison"]["refused_by_inference_path"]
    assert len(refused) == 1
    assert refused[0]["trace"] == f"{ROUND}/wget/index"


# --------------------------------------------------------------
# A limited run must not look like a verified one
# --------------------------------------------------------------

def test_a_limited_run_exits_non_zero_even_when_everything_matches(
    parity_inputs, tmp_path, capsys
):
    """
    The exit code is what automation reads. A `--limit 1` run returning
    0 would let a check report parity over 800 traces on the strength of
    one.
    """
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    assert run(pcap_root, features_csv, model, out, "--limit", "1") == \
        script.EXIT_DIVERGED

    record = read_record(out)
    assert record["partial"] is True
    assert record["limit"] == 1
    assert record["comparison"]["traces_compared"] == 1
    assert record["comparison"]["traces_matching_all_features"] == 1
    # meaningless after a limit, so not reported rather than misreported
    assert record["coverage"]["csv_rows_without_pcap"] is None

    assert "PARTIAL RUN" in capsys.readouterr().err


def test_a_full_run_records_that_it_was_not_limited(parity_inputs, tmp_path):
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"
    run(pcap_root, features_csv, model, out)

    record = read_record(out)
    assert record["partial"] is False
    assert record["limit"] is None


# --------------------------------------------------------------
# Usage errors
# --------------------------------------------------------------

def test_missing_inputs_are_usage_errors(parity_inputs, tmp_path, capsys):
    pcap_root, traces, model = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    assert run(tmp_path / "nowhere", features_csv, model, out) == script.EXIT_USAGE
    assert run(pcap_root, tmp_path / "nothing.csv", model, out) == script.EXIT_USAGE
    assert "does not exist" in capsys.readouterr().err


def test_an_unloadable_model_is_a_usage_error(parity_inputs, tmp_path):
    pcap_root, traces, _ = parity_inputs
    features_csv = write_csv(tmp_path / "features.csv", traces)
    out = tmp_path / "parity.json"

    assert run(pcap_root, features_csv, tmp_path / "absent.joblib", out) == \
        script.EXIT_USAGE
