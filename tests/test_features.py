"""
test_features.py
----------------
Tests for tsd.features.

The arithmetic is asserted against hand-computed values on small
synthetic traces rather than merely exercised. A feature function that
runs without raising but computes the wrong number is the worst failure
mode available here: the model still trains, the metrics still look
plausible, and the SHAP plots explain a quantity that does not mean what
its name says.

The synthetic traces are built in code because a real PCAP cannot be
checked by hand -- and because the captures are gitignored, so a suite
that depended on them would silently stop testing anything on a fresh
clone.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from tsd.features import (
    DOWN,
    SERVER_PORT,
    TCP_ACK,
    TCP_FIN,
    TCP_RST,
    TCP_SYN,
    UP,
    PacketRecord,
    TraceError,
    extract_features,
    feature_names,
    find_bursts,
    read_trace,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PCAP_ROOT = REPO_ROOT / "data" / "pcaps"


def packet(time: float, direction: str, payload: int = 0, flags: int = TCP_ACK):
    return PacketRecord(
        time=time, direction=direction, payload_length=payload, flags=flags
    )


# --------------------------------------------------------------
# Synthetic pcap building, for the parsing layer
# --------------------------------------------------------------

def ethernet_tcp_frame(
    src_port: int, dst_port: int, payload_length: int, flags: int = TCP_ACK,
    clip: int | None = 96,
) -> bytes:
    """
    One Ethernet/IPv4/TCP frame whose HEADERS claim `payload_length`
    bytes of payload, clipped like a real -s 96 capture.

    The clipping is the point: the payload is absent from the file, so a
    reader that measures the captured bytes gets the wrong answer for
    every packet over the snaplen.
    """
    ethernet = b"\x00" * 12 + b"\x08\x00"

    tcp_header_length = 20
    ip_header_length = 20
    total_length = ip_header_length + tcp_header_length + payload_length

    ip = struct.pack(
        ">BBHHHBBH4s4s",
        0x45, 0, total_length, 0, 0, 64, 6, 0,
        bytes([127, 0, 0, 1]), bytes([127, 0, 0, 1]),
    )
    tcp = struct.pack(
        ">HHIIBBHHH",
        src_port, dst_port, 0, 0, (5 << 4), flags, 65535, 0, 0,
    )

    frame = ethernet + ip + tcp + b"\x00" * payload_length
    return frame[:clip] if clip else frame


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 96, 1))
        for timestamp, frame in frames:
            seconds = int(timestamp)
            microseconds = int(round((timestamp - seconds) * 1_000_000))
            handle.write(
                struct.pack("<IIII", seconds, microseconds, len(frame), len(frame))
            )
            handle.write(frame)
    return path


# --------------------------------------------------------------
# Layer 1: parsing
# --------------------------------------------------------------

def test_payload_length_comes_from_the_headers_not_the_captured_bytes(tmp_path):
    """
    The captures use -s 96, so a 1448-byte packet has only 96 bytes on
    disk. Measuring the frame would make every large packet identical
    and destroy the entire size family -- while still producing a
    perfectly plausible feature table.
    """
    path = write_pcap(tmp_path / "big.pcap", [
        (100.0, ethernet_tcp_frame(50000, SERVER_PORT, payload_length=1448)),
        (100.1, ethernet_tcp_frame(SERVER_PORT, 50000, payload_length=32768)),
    ])

    trace = read_trace(path)

    assert [p.payload_length for p in trace] == [1448, 32768]


def test_direction_is_decided_by_the_server_port_not_arrival_order(tmp_path):
    """
    "The first address I saw is the client" breaks on a trace that opens
    with a retransmission or a stray server packet -- and it breaks
    silently, mirroring every size feature for that trace alone.
    """
    path = write_pcap(tmp_path / "d.pcap", [
        (0.0, ethernet_tcp_frame(SERVER_PORT, 50000, 100)),  # server first
        (0.1, ethernet_tcp_frame(50000, SERVER_PORT, 200)),
    ])

    trace = read_trace(path)

    assert [p.direction for p in trace] == [DOWN, UP]


def test_timestamps_are_relative_to_the_first_packet(tmp_path):
    """
    An absolute timestamp encodes the capture round, and the round is
    the train/test split group -- the model could identify the group
    directly, which is the leakage GroupKFold exists to prevent.
    """
    path = write_pcap(tmp_path / "t.pcap", [
        (1_770_000_000.0, ethernet_tcp_frame(50000, SERVER_PORT, 10)),
        (1_770_000_000.25, ethernet_tcp_frame(SERVER_PORT, 50000, 10)),
    ])

    trace = read_trace(path)

    assert trace[0].time == 0.0
    assert trace[1].time == pytest.approx(0.25)


def test_packets_outside_the_conversation_are_skipped(tmp_path):
    path = write_pcap(tmp_path / "mixed.pcap", [
        (0.0, ethernet_tcp_frame(50000, SERVER_PORT, 10)),
        (0.1, ethernet_tcp_frame(1234, 5678, 10)),  # neither side is the server
        (0.2, ethernet_tcp_frame(SERVER_PORT, 50000, 10)),
    ])

    assert len(read_trace(path)) == 2


def test_flags_are_carried_through(tmp_path):
    path = write_pcap(tmp_path / "f.pcap", [
        (0.0, ethernet_tcp_frame(50000, SERVER_PORT, 0, flags=TCP_SYN)),
        (0.1, ethernet_tcp_frame(SERVER_PORT, 50000, 0, flags=TCP_SYN | TCP_ACK)),
        (0.2, ethernet_tcp_frame(50000, SERVER_PORT, 0, flags=TCP_FIN | TCP_ACK)),
    ])

    trace = read_trace(path)

    assert trace[0].flags & TCP_SYN and not trace[0].flags & TCP_ACK
    assert trace[1].flags & TCP_SYN and trace[1].flags & TCP_ACK
    assert trace[2].flags & TCP_FIN


def test_unreadable_file_raises_trace_error(tmp_path):
    broken = tmp_path / "broken.pcap"
    broken.write_bytes(b"not a pcap at all")

    with pytest.raises(TraceError):
        read_trace(broken)


# --------------------------------------------------------------
# Layer 2: counts and sizes, checked by hand
# --------------------------------------------------------------

def test_count_features_are_exact():
    trace = [
        packet(0.0, UP, 100),
        packet(0.1, DOWN, 200),
        packet(0.2, DOWN, 300),
        packet(0.3, DOWN, 400),
    ]

    features = extract_features(trace)

    assert features["count_total"] == 4
    assert features["count_up"] == 1
    assert features["count_down"] == 3
    assert features["count_ratio_up_down"] == pytest.approx(1 / 3)


def test_size_features_are_exact():
    """Values chosen so mean, median and percentiles are checkable by hand."""
    trace = [
        packet(0.0, UP, 100),
        packet(0.1, UP, 200),
        packet(0.2, UP, 300),
        packet(0.3, UP, 400),
        packet(0.4, DOWN, 1000),
    ]

    features = extract_features(trace)

    assert features["bytes_up_total"] == 1000
    assert features["bytes_down_total"] == 1000
    assert features["size_up_mean"] == 250
    assert features["size_up_min"] == 100
    assert features["size_up_max"] == 400
    assert features["size_up_median"] == 250
    assert features["size_up_p25"] == 175
    assert features["size_up_p75"] == 325
    assert features["size_up_p90"] == pytest.approx(370)
    # population std of 100,200,300,400 = sqrt(12500)
    assert features["size_up_std"] == pytest.approx(math.sqrt(12500))
    assert features["bytes_ratio_up_down"] == 1.0


def test_pure_acks_are_counted_per_direction():
    trace = [
        packet(0.0, UP, 0),
        packet(0.1, UP, 0),
        packet(0.2, UP, 500),
        packet(0.3, DOWN, 0),
    ]

    features = extract_features(trace)

    assert features["ack_up_count"] == 2
    assert features["ack_down_count"] == 1


def test_single_value_std_is_zero_not_nan():
    """
    Sample std is undefined for one value. A direction with one packet
    is normal, not exceptional, so population std is used -- a NaN here
    would be dropped silently by most estimators.
    """
    features = extract_features([packet(0.0, UP, 100)])

    assert features["size_up_std"] == 0.0


# --------------------------------------------------------------
# Timing
# --------------------------------------------------------------

def test_timing_features_are_exact():
    trace = [
        packet(0.0, UP, 10),
        packet(1.0, DOWN, 10),
        packet(3.0, DOWN, 10),
        packet(6.0, DOWN, 10),
    ]

    features = extract_features(trace)

    assert features["duration"] == 6.0
    # gaps overall: 1.0, 2.0, 3.0
    assert features["iat_mean"] == pytest.approx(2.0)
    assert features["iat_median"] == pytest.approx(2.0)
    assert features["iat_max"] == pytest.approx(3.0)
    # gaps within DOWN only: 2.0, 3.0
    assert features["iat_down_mean"] == pytest.approx(2.5)
    # a single UP packet has no gaps at all
    assert features["iat_up_mean"] == 0.0


# --------------------------------------------------------------
# Bursts -- a modelling choice, so its edges are pinned down
# --------------------------------------------------------------

def test_single_direction_trace_is_one_burst():
    bursts = find_bursts([packet(0.0, UP, 10), packet(0.1, UP, 20)])

    assert len(bursts) == 1
    assert bursts[0].packets == 2
    assert bursts[0].payload_bytes == 30


def test_alternating_packets_are_one_burst_each():
    trace = [
        packet(0.0, UP, 10),
        packet(0.1, DOWN, 20),
        packet(0.2, UP, 30),
        packet(0.3, DOWN, 40),
    ]

    bursts = find_bursts(trace)

    assert [burst.packets for burst in bursts] == [1, 1, 1, 1]
    assert [burst.payload_bytes for burst in bursts] == [10, 20, 30, 40]


def test_trace_ending_mid_burst_keeps_its_last_burst():
    """
    Every trace ends mid-burst -- there is no direction change after the
    last run. Dropping it would lose the final burst of every capture
    ever taken, and the loss would be invisible.
    """
    trace = [
        packet(0.0, UP, 10),
        packet(0.1, DOWN, 20),
        packet(0.2, DOWN, 30),
        packet(0.3, DOWN, 40),
    ]

    bursts = find_bursts(trace)

    assert [burst.packets for burst in bursts] == [1, 3]
    assert bursts[-1].payload_bytes == 90


def test_burst_features_are_exact():
    trace = [
        packet(0.0, UP, 100),
        packet(0.5, UP, 100),
        packet(2.0, DOWN, 500),
        packet(2.5, DOWN, 500),
        packet(2.6, DOWN, 500),
        packet(5.0, UP, 50),
    ]

    features = extract_features(trace)

    assert features["burst_count"] == 3
    assert features["burst_len_mean"] == pytest.approx(2.0)  # 2, 3, 1
    assert features["burst_len_max"] == 3
    assert features["burst_bytes_max"] == 1500
    assert features["burst_bytes_mean"] == pytest.approx((200 + 1500 + 50) / 3)
    # gaps between bursts: 2.0-0.5 = 1.5, and 5.0-2.6 = 2.4
    assert features["burst_gap_mean"] == pytest.approx(1.95)
    assert features["burst_gap_max"] == pytest.approx(2.4)


def test_a_long_pause_inside_one_direction_does_not_split_a_burst():
    """
    The documented cost of defining a burst by direction alone. The
    information is not lost: the pause shows up as a large inter-arrival.
    """
    trace = [packet(0.0, DOWN, 100), packet(30.0, DOWN, 100)]

    assert len(find_bursts(trace)) == 1
    assert extract_features(trace)["iat_down_max"] == pytest.approx(30.0)


# --------------------------------------------------------------
# Connections -- the parallelism signal
# --------------------------------------------------------------

def test_syn_count_excludes_syn_ack():
    """
    Measured on round 1: Firefox opens ~6 connections per page load,
    wget exactly 1. A SYN-ACK also carries SYN, so counting it would
    double every client's connection count and blur the difference.
    """
    trace = [
        packet(0.0, UP, 0, flags=TCP_SYN),
        packet(0.1, DOWN, 0, flags=TCP_SYN | TCP_ACK),
        packet(0.2, UP, 0, flags=TCP_ACK),
        packet(0.3, UP, 0, flags=TCP_SYN),
        packet(0.4, UP, 0, flags=TCP_FIN | TCP_ACK),
        packet(0.5, DOWN, 0, flags=TCP_RST),
    ]

    features = extract_features(trace)

    assert features["syn_count"] == 2
    assert features["syn_ack_count"] == 1


def test_teardown_flags_are_not_counted():
    """
    Measured across all 200 round-1 traces (2026-08-07): every Firefox
    trace had zero FINs and zero RSTs, every wget trace had at least one
    of each -- 100/100 against 100/100.

    A perfect separator that separates the wrong thing. wget exits on
    its own, so its teardown lands inside the capture window; Firefox is
    killed only after tcpdump has stopped, so its teardown is never
    recorded. The feature measures how the harness stops each client,
    and it would have sat at the top of the SHAP plots saying something
    false.

    syn_count stays: a SYN happens during the load, not at teardown.
    """
    trace = [
        packet(0.0, UP, 0, flags=TCP_SYN),
        packet(0.1, UP, 0, flags=TCP_FIN | TCP_ACK),
        packet(0.2, DOWN, 0, flags=TCP_RST),
    ]

    features = extract_features(trace)

    assert "fin_count" not in features
    assert "rst_count" not in features
    assert features["syn_count"] == 1


# --------------------------------------------------------------
# Robustness
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "trace",
    [
        [],
        [packet(0.0, UP, 100)],
        [packet(0.0, DOWN, 100)],
        [packet(0.0, UP, 0), packet(0.0, UP, 0)],  # identical timestamps
        [packet(0.0, UP, 100), packet(0.1, UP, 100), packet(0.2, UP, 100)],
    ],
)
def test_degenerate_traces_produce_finite_features(trace):
    """
    A NaN or an inf would not raise -- it would flow into training and
    be dropped silently by the estimator, shrinking the dataset for
    reasons nobody would look for. Round 1 has traces as short as 17
    packets, and a future round may be shorter.
    """
    features = extract_features(trace)

    assert features, "a degenerate trace must still produce a full feature dict"
    for name, value in features.items():
        assert isinstance(value, float), f"{name} is {type(value).__name__}"
        assert math.isfinite(value), f"{name} is {value}"


def test_empty_trace_has_the_same_keys_as_a_full_one():
    full = extract_features([
        packet(0.0, UP, 100, flags=TCP_SYN),
        packet(0.1, DOWN, 200),
    ])

    assert list(extract_features([])) == list(full)


def test_key_order_is_deterministic():
    """
    A saved model's column order must stay valid between training and
    inference. Dict order that depended on the data would break that in
    a way nothing else would catch.
    """
    first = extract_features([packet(0.0, UP, 10)])
    second = extract_features([
        packet(0.0, DOWN, 99), packet(1.0, UP, 5), packet(2.0, DOWN, 7),
    ])

    assert list(first) == list(second)
    assert feature_names() == list(first)


def test_no_excluded_quantity_leaks_into_a_feature_name():
    """
    The exclusions are the experimental design: ports, addresses,
    absolute time, TCP stack fingerprints, and the teardown flags that
    turned out to measure the harness. This asserts the promise the
    module docstring makes, so bringing a tempting feature back has to
    be a deliberate act rather than an accident.

    Tokens are matched against the name split on "_", not as
    substrings, because `burst` contains `rst` -- a substring check
    would either ban the burst family or, once someone "fixed" it by
    dropping `rst` from the list, stop guarding the thing it was written
    for.
    """
    forbidden_tokens = {
        "fin", "rst",  # teardown: measured the harness, not the client
        "ip", "port", "addr",  # constant here, a leak anywhere else
        "seq", "mss", "ttl", "wscale",  # stack fingerprints
    }
    forbidden_substrings = ("window", "option", "epoch", "timestamp")

    for name in feature_names():
        tokens = set(name.split("_"))
        assert not tokens & forbidden_tokens, f"{name} contains an excluded token"
        assert not any(part in name for part in forbidden_substrings), name


def test_the_token_guard_would_actually_catch_a_returning_feature():
    """
    The guard above is only worth having if it fails on the thing it
    exists to stop. A rule that passes everything, including what it
    forbids, is decoration.
    """
    forbidden_tokens = {"fin", "rst", "ip", "port", "addr", "seq", "mss",
                        "ttl", "wscale"}

    assert set("fin_count".split("_")) & forbidden_tokens
    assert set("rst_count".split("_")) & forbidden_tokens
    assert not set("burst_count".split("_")) & forbidden_tokens


# --------------------------------------------------------------
# Against the real captures, when they are present
# --------------------------------------------------------------

def real_traces(client: str) -> list[Path]:
    directory = PCAP_ROOT / "round_01_20260807" / client
    return sorted(directory.glob("*.pcap")) if directory.is_dir() else []


@pytest.mark.parametrize(
    "client, expected_syns",
    [("wget", 1), ("firefox", 6)],
)
def test_real_captures_have_the_expected_connection_count(client, expected_syns):
    """
    The measured parallelism signal, checked against the actual round:
    wget opens one sequential connection, Firefox about six in parallel.
    This is what made the threaded server necessary, so it is worth an
    assertion rather than a note.

    Skipped cleanly when the captures are absent -- they are gitignored,
    so a fresh clone has none.
    """
    traces = real_traces(client)
    if not traces:
        pytest.skip(f"no captured traces for {client} (data/pcaps is gitignored)")

    counts = [extract_features(read_trace(path))["syn_count"] for path in traces[:5]]

    assert all(count > 0 for count in counts)
    for count in counts:
        assert count == pytest.approx(expected_syns, abs=2), counts


def test_real_capture_features_are_all_finite():
    traces = real_traces("firefox") + real_traces("wget")
    if not traces:
        pytest.skip("no captured traces (data/pcaps is gitignored)")

    for path in traces[:10]:
        features = extract_features(read_trace(path))
        assert features["count_total"] > 0, path.name
        for name, value in features.items():
            assert math.isfinite(value), f"{path.name}: {name} is {value}"
