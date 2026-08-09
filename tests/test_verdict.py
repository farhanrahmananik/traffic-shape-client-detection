"""
test_verdict.py
---------------
Tests for tsd.verdict.

A stub estimator stands in for the real model, so the suite passes on a
clone with no `models/` directory -- which is every clone, since
`models/` is gitignored. Nothing here loads the shipped artefact.

The properties worth testing are the ones whose failure is silent.
scikit-learn checks how many columns it was given, never what they are
called, so a model fed a permuted or mismatched vector returns a
confident prediction built from the wrong inputs and raises nothing. The
verdict looks exactly like a correct one. Those are the paths asserted
here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import joblib
import pytest

from tsd.features import SERVER_PORT, feature_names
from tsd.verdict import (
    MIN_PACKETS,
    SCHEMA_VERSION,
    Artefact,
    VerdictError,
    build_vector,
    classify_pcap,
    load_artefact,
)

TCP_ACK = 0x10
TCP_SYN = 0x02


# --------------------------------------------------------------
# Stubs and fixtures
# --------------------------------------------------------------

class StubPipeline:
    """Predicts a fixed label. Optionally offers probabilities."""

    def __init__(self, label: str = "wget", with_proba: bool = True):
        self.label = label
        self.classes_ = ["firefox", "wget"]
        self.seen: list[list[float]] = []

        if not with_proba:
            # The attribute must be absent, not None: verdict.py probes
            # with getattr, exactly as a caller would.
            del self.predict_proba

    def predict(self, X):
        self.seen.extend([list(row) for row in X])
        return [self.label for _ in X]

    def predict_proba(self, X):
        return [[0.125, 0.875] for _ in X]


class NoProbaPipeline:
    """A model that cannot express confidence."""

    def __init__(self, label: str = "firefox"):
        self.label = label
        self.classes_ = ["firefox", "wget"]

    def predict(self, X):
        return [self.label for _ in X]


def write_artefact(
    path: Path,
    pipeline=None,
    features=None,
    classes=("firefox", "wget"),
    rounds=(1, 2, 3, 4),
) -> Path:
    joblib.dump(
        {
            "pipeline": pipeline if pipeline is not None else StubPipeline(),
            "features": list(features if features is not None else feature_names()),
            "classes": list(classes),
            "rounds": list(rounds),
        },
        path,
    )
    return path


@pytest.fixture
def artefact_path(tmp_path) -> Path:
    return write_artefact(tmp_path / "model.joblib")


def frame(src_port: int, dst_port: int, payload: int, flags: int = TCP_ACK) -> bytes:
    ethernet = b"\x00" * 12 + b"\x08\x00"
    ip = struct.pack(
        ">BBHHHBBH4s4s",
        0x45, 0, 40 + payload, 0, 0, 64, 6, 0,
        bytes([127, 0, 0, 1]), bytes([127, 0, 0, 1]),
    )
    tcp = struct.pack(
        ">HHIIBBHHH", src_port, dst_port, 0, 0, (5 << 4), flags, 65535, 0, 0
    )
    return (ethernet + ip + tcp + b"\x00" * payload)[:96]


def write_pcap(path: Path, packets: list[tuple[float, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 96, 1))
        for timestamp, buf in packets:
            seconds = int(timestamp)
            micro = int(round((timestamp - seconds) * 1_000_000))
            handle.write(struct.pack("<IIII", seconds, micro, len(buf), len(buf)))
            handle.write(buf)
    return path


def page_load(path: Path, packets: int = 8) -> Path:
    """A short but usable trace: handshake, then alternating exchange."""
    up = lambda payload, flags=TCP_ACK: frame(50000, SERVER_PORT, payload, flags)
    down = lambda payload, flags=TCP_ACK: frame(SERVER_PORT, 50000, payload, flags)

    records = [
        (0.000, up(0, TCP_SYN)),
        (0.001, down(0, TCP_SYN | TCP_ACK)),
        (0.002, up(0)),
        (0.003, up(517)),
    ]
    for index in range(packets - len(records)):
        records.append((0.010 + index * 0.002, down(1400)))

    return write_pcap(path, records[:packets])


@pytest.fixture
def pcap(tmp_path) -> Path:
    return page_load(tmp_path / "trace.pcap")


# --------------------------------------------------------------
# Loading
# --------------------------------------------------------------

def test_artefact_round_trips_through_joblib(artefact_path):
    artefact = load_artefact(artefact_path)

    assert artefact.features == tuple(feature_names())
    assert artefact.classes == ("firefox", "wget")
    assert artefact.rounds == (1, 2, 3, 4)
    assert artefact.path == artefact_path
    assert len(artefact.sha256) == 64


def test_artefact_sha256_identifies_the_file(tmp_path):
    """
    models/ is gitignored, so a verdict that cannot name its model is
    not evidence. Two different models must not share a digest.
    """
    first = load_artefact(write_artefact(tmp_path / "a.joblib"))
    second = load_artefact(write_artefact(tmp_path / "b.joblib", rounds=(1, 2)))

    assert first.sha256 != second.sha256


def test_missing_artefact_points_at_train_model(tmp_path):
    with pytest.raises(VerdictError) as raised:
        load_artefact(tmp_path / "absent.joblib")

    assert "train_model.py" in str(raised.value)


def test_non_dict_artefact_is_refused(tmp_path):
    path = tmp_path / "wrong.joblib"
    joblib.dump(["not", "a", "dict"], path)

    with pytest.raises(VerdictError) as raised:
        load_artefact(path)

    assert "not the dict" in str(raised.value)


@pytest.mark.parametrize("missing", ["pipeline", "features", "classes", "rounds"])
def test_artefact_missing_a_key_is_refused(tmp_path, missing):
    bundle = {
        "pipeline": StubPipeline(),
        "features": feature_names(),
        "classes": ["firefox", "wget"],
        "rounds": [1],
    }
    del bundle[missing]
    path = tmp_path / "partial.joblib"
    joblib.dump(bundle, path)

    with pytest.raises(VerdictError) as raised:
        load_artefact(path)

    assert missing in str(raised.value)


def test_removed_feature_is_named_in_the_mismatch_message(tmp_path):
    """
    The failure this guards against is silent: scikit-learn checks the
    column COUNT, not the names, so a model trained on a different
    feature set still returns a confident prediction.
    """
    stale = [*feature_names(), "iat_up_kurtosis"]
    path = write_artefact(tmp_path / "stale.joblib", features=stale)

    with pytest.raises(VerdictError) as raised:
        load_artefact(path)

    message = str(raised.value)
    assert "iat_up_kurtosis" in message
    assert "no longer produces" in message
    assert "train_model.py" in message


def test_added_feature_is_named_in_the_mismatch_message(tmp_path):
    dropped = [name for name in feature_names() if name != "syn_count"]
    path = write_artefact(tmp_path / "old.joblib", features=dropped)

    with pytest.raises(VerdictError) as raised:
        load_artefact(path)

    message = str(raised.value)
    assert "syn_count" in message
    assert "never saw" in message


def test_reordered_features_are_accepted_and_preserved(tmp_path):
    """
    Order is not a mismatch -- the artefact's order IS the authority,
    because the pipeline was fitted with the columns in that sequence.
    feature_names() is only cross-checked against it.
    """
    reversed_names = list(reversed(feature_names()))
    artefact = load_artefact(
        write_artefact(tmp_path / "rev.joblib", features=reversed_names)
    )

    assert artefact.features == tuple(reversed_names)


# --------------------------------------------------------------
# Vector building
# --------------------------------------------------------------

def test_vector_follows_the_artefact_order_not_the_dict_order(artefact_path):
    """
    Built by iterating the artefact, never the dict. A permuted vector
    is still a vector of the right length, so the model would accept it
    and predict from scrambled columns without raising.
    """
    artefact = load_artefact(artefact_path)

    ordered = {name: float(index) for index, name in enumerate(artefact.features)}
    shuffled = {name: ordered[name] for name in reversed(list(ordered))}

    assert list(shuffled) != list(ordered), "the fixture must actually differ"

    assert build_vector(shuffled, artefact) == build_vector(ordered, artefact)
    assert build_vector(shuffled, artefact) == [
        float(index) for index in range(len(artefact.features))
    ]


def test_missing_feature_value_is_refused(artefact_path):
    artefact = load_artefact(artefact_path)
    feats = {name: 0.0 for name in artefact.features}
    del feats["syn_count"]

    with pytest.raises(VerdictError) as raised:
        build_vector(feats, artefact)

    assert "syn_count" in str(raised.value)


# --------------------------------------------------------------
# Classification
# --------------------------------------------------------------

def test_verdict_document_shape_and_key_order(artefact_path, pcap):
    artefact = load_artefact(artefact_path)

    document = classify_pcap(pcap, artefact)

    assert list(document) == ["schema_version", "pcap", "verdict", "model"]
    assert document["schema_version"] == SCHEMA_VERSION
    assert list(document["pcap"]) == [
        "path", "sha256", "packets", "duration_s", "server_port"
    ]
    assert list(document["model"]) == [
        "path", "sha256", "trained_on_rounds", "n_features"
    ]

    assert document["verdict"]["client"] == "wget"
    assert document["verdict"]["probabilities"] == {"firefox": 0.125, "wget": 0.875}
    assert document["pcap"]["packets"] == 8
    assert document["pcap"]["server_port"] == SERVER_PORT
    assert document["model"]["n_features"] == len(feature_names())
    assert document["model"]["trained_on_rounds"] == [1, 2, 3, 4]


def test_features_are_included_only_on_request(artefact_path, pcap):
    artefact = load_artefact(artefact_path)

    assert "features" not in classify_pcap(pcap, artefact)

    document = classify_pcap(pcap, artefact, include_features=True)
    assert list(document)[-1] == "features"
    assert set(document["features"]) == set(artefact.features)


def test_a_pipeline_without_predict_proba_omits_probabilities(tmp_path, pcap):
    """
    Omitted, not invented. A fabricated 1.0 for the predicted class
    would read as certainty the model never expressed.
    """
    artefact = load_artefact(
        write_artefact(tmp_path / "np.joblib", pipeline=NoProbaPipeline())
    )

    document = classify_pcap(pcap, artefact)

    assert document["verdict"] == {"client": "firefox"}
    assert "probabilities" not in document["verdict"]


def test_too_short_trace_is_refused(tmp_path, artefact_path):
    artefact = load_artefact(artefact_path)
    short = page_load(tmp_path / "short.pcap", packets=MIN_PACKETS - 1)

    with pytest.raises(VerdictError) as raised:
        classify_pcap(short, artefact)

    message = str(raised.value)
    assert f"{MIN_PACKETS - 1} packet" in message
    assert "capture filter" in message


def test_trace_on_another_port_is_refused(tmp_path, artefact_path):
    """
    The packets exist, but none of them are the conversation. Reported
    as a filter problem rather than as an empty file, because that is
    what it usually is.
    """
    artefact = load_artefact(artefact_path)
    path = write_pcap(tmp_path / "other.pcap", [
        (index * 0.01, frame(1234, 5678, 100)) for index in range(10)
    ])

    with pytest.raises(VerdictError) as raised:
        classify_pcap(path, artefact)

    assert "0 packet" in str(raised.value)


def test_unreadable_pcap_becomes_a_verdict_error(tmp_path, artefact_path):
    """One exception type for callers: the CLI need not know who refused."""
    artefact = load_artefact(artefact_path)
    broken = tmp_path / "broken.pcap"
    broken.write_bytes(b"not a pcap")

    with pytest.raises(VerdictError):
        classify_pcap(broken, artefact)


def test_missing_pcap_becomes_a_verdict_error(tmp_path, artefact_path):
    artefact = load_artefact(artefact_path)

    with pytest.raises(VerdictError):
        classify_pcap(tmp_path / "nowhere.pcap", artefact)


def test_floats_are_rounded_for_diffable_output(artefact_path, pcap):
    artefact = load_artefact(artefact_path)

    document = classify_pcap(pcap, artefact, include_features=True)

    for value in document["features"].values():
        assert value == round(value, 6)
    assert document["pcap"]["duration_s"] == round(
        document["pcap"]["duration_s"], 6
    )


def test_the_model_receives_the_features_in_its_own_order(artefact_path, pcap):
    """
    End to end: what reaches predict() is ordered by artefact.features,
    which is what the pipeline was fitted on.

    Compared after rounding, because the rounding is an OUTPUT concern
    only -- see the test below. The claim here is about order.
    """
    artefact = load_artefact(artefact_path)

    document = classify_pcap(pcap, artefact, include_features=True)
    seen = artefact.pipeline.seen[0]

    assert [round(value, 6) for value in seen] == [
        document["features"][name] for name in artefact.features
    ]


def test_rounding_is_for_the_output_only_and_never_reaches_the_model(
    artefact_path, pcap
):
    """
    The JSON is rounded so two verdicts on the same input diff cleanly.
    The model is fed the real numbers: rounding the inputs would change
    the prediction path for a presentation concern.
    """
    artefact = load_artefact(artefact_path)

    classify_pcap(pcap, artefact)
    seen = artefact.pipeline.seen[0]

    assert any(value != round(value, 6) for value in seen), (
        "the model received pre-rounded values"
    )


def test_verdict_is_deterministic(artefact_path, pcap):
    artefact = load_artefact(artefact_path)

    assert classify_pcap(pcap, artefact) == classify_pcap(pcap, artefact)
