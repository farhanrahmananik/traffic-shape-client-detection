"""
features.py
-----------
Turn a PCAP into numbers, in two layers that are kept apart on purpose.

    read_trace(path)        pcap  -> Trace       (parsing, file I/O, dpkt)
    extract_features(trace) Trace -> dict        (pure arithmetic)

The split matters because the CLI in step 8 calls `extract_features` on
exactly the same `Trace` the training path builds. If the two paths each
did their own parsing and their own arithmetic, they would drift, and
the drift would show up as a model that scores well in evaluation and
badly in the tool that ships. One function, both paths.

A `Trace` is a list of `PacketRecord`, and a `PacketRecord` is four
numbers: relative time, direction, TCP payload length, TCP flags.
Nothing from dpkt escapes `read_trace`, so swapping the parser later
touches this file only.

Two parsing details are not stylistic:

*   **Direction comes from the port, not from arrival order.** Deciding
    "the first address I saw is the client" breaks the moment a trace
    starts with a retransmission or a stray server packet, and it breaks
    silently -- every size feature would be mirrored for that trace
    alone.
*   **Payload length is read from the IP and TCP headers, never from the
    captured frame.** The captures use `-s 96`, so the frame is clipped
    at 96 bytes and every packet larger than that would measure the
    same. The headers still carry the true length: `ip.len` minus both
    header lengths. Using `len(buf)` here would quietly destroy the
    single most important feature family.

Parser choice: dpkt, not scapy. Over 200+ traces per round scapy's
per-packet overhead is measurable, and it pulls in a large dependency
tree for a job that is "read four numbers per packet".

--------------------------------------------------------------------
Deliberately excluded -- what is left out is part of the design
--------------------------------------------------------------------

*   **Ports and IP addresses.** Constant by construction here
    (127.0.0.1:8443), so they carry no information in this dataset --
    and in any other capture they would be a leak, letting the model
    identify a host rather than a client.

*   **Absolute timestamps.** Every timestamp is relative to the first
    packet of its own trace. An absolute time encodes the capture
    round, and the round is the train/test split group: including it
    would let the model identify the group directly, which is the exact
    leakage `GroupKFold` exists to prevent.

*   **TCP window size, window scale, initial sequence numbers, MSS, and
    TCP option ordering.** These are real, strong client fingerprints --
    and that is precisely why they are excluded. The claim this project
    makes is that traffic *shape* alone separates a browser from a
    scraper. A stack fingerprint would win without ever testing that
    claim, and the SHAP plots would then faithfully explain the wrong
    thing: "this is Firefox because its TCP options are in Firefox's
    order" is a true statement about a different experiment.

*   **FIN and RST counts** -- removed after measuring them, not before.
    Across all 200 round-1 traces (2026-08-07): every Firefox trace has
    **zero** FINs and **zero** RSTs, every wget trace has **at least one
    of each**. 100/100 against 100/100, no exceptions.

    A perfect separator, and it separates the wrong thing. It is not
    client behaviour, it is the harness: wget exits on its own, so its
    teardown falls inside the capture window, while Firefox is killed
    only after tcpdump has already stopped, so its teardown is never
    recorded. The feature measures **how this harness stops each
    client**. Kept, it would sit at the top of the SHAP plots
    explaining the model with a statement that is simply false -- the
    same failure as the `/robots.txt` 404 that `-e robots=off` already
    removes, and harder to spot because a connection-teardown feature
    sounds like it belongs.

    **`syn_count` stays.** A SYN happens during the load, not at
    teardown, and 6 against 1 is the parallelism that made the threaded
    server necessary in the first place.

    *Residual, and it goes in the README limitations:* the FIN and RST
    packets themselves are still in the wget traces, so they still make
    a small contribution to the count, size and burst features. Removing
    that would mean discarding packets from the traces -- a heavier
    intervention, with its own arbitrary rules, than the size of the
    problem warrants. So a small teardown asymmetry remains in the data
    even though no feature names it, and saying so is the honest
    alternative to pretending the fix was complete.

*   **Anything derived from packet contents.** There is nothing to
    derive: `-s 96` means the payload was never written to disk. The
    exclusion is enforced by the capture, not by discipline here.

--------------------------------------------------------------------
Guarantees
--------------------------------------------------------------------

*   Every feature is finite. An empty direction yields zeros, never NaN
    or inf -- a NaN would propagate into training as a silent row drop.
*   A trace with fewer than two packets returns a complete, well-formed
    feature dict rather than raising. Round 1 contains traces as short
    as 17 packets and a future round may be shorter.
*   Key order is deterministic across calls, so a model's column order
    stays valid between training and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import dpkt

SERVER_PORT = 8443

UP = "up"  # client -> server
DOWN = "down"  # server -> client

# pcap link types we can read. The captures are taken on lo, which Linux
# presents as an Ethernet device; the others are here so an unexpected
# link type fails with a clear message instead of a decode error.
LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_LINUX_SLL = 113

TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_ACK = 0x10


class TraceError(RuntimeError):
    """The file could not be read as a trace."""


@dataclass(frozen=True)
class PacketRecord:
    """One packet, reduced to the four things a feature may depend on."""

    time: float  # seconds since the first packet of this trace
    direction: str  # UP or DOWN
    payload_length: int  # TCP payload bytes, from the headers
    flags: int  # TCP flag bits


Trace = list[PacketRecord]


# --------------------------------------------------------------
# Layer 1: parsing
# --------------------------------------------------------------

def read_trace(pcap_path: str | Path, server_port: int = SERVER_PORT) -> Trace:
    """
    Read a PCAP into a Trace.

    Non-TCP packets and packets that touch neither side of the server
    port are skipped rather than guessed at: the capture filter should
    have excluded them, and silently reinterpreting one would be worse
    than dropping it.
    """
    path = Path(pcap_path)
    packets: Trace = []

    try:
        with path.open("rb") as handle:
            reader = dpkt.pcap.Reader(handle)
            unpack = _link_unpacker(reader.datalink())

            first_timestamp: float | None = None

            for timestamp, buf in reader:
                record = _packet_record(
                    unpack, buf, timestamp, first_timestamp, server_port
                )
                if record is None:
                    continue

                if first_timestamp is None:
                    first_timestamp = timestamp

                packets.append(record)
    except (ValueError, dpkt.dpkt.Error) as error:
        raise TraceError(f"{path}: {type(error).__name__}: {error}") from error

    return packets


def _link_unpacker(datalink: int):
    if datalink == LINKTYPE_ETHERNET:
        return lambda buf: dpkt.ethernet.Ethernet(buf).data
    if datalink == LINKTYPE_NULL:
        return lambda buf: dpkt.loopback.Loopback(buf).data
    if datalink == LINKTYPE_LINUX_SLL:
        return lambda buf: dpkt.sll.SLL(buf).data

    raise TraceError(f"unsupported pcap link type {datalink}")


def _packet_record(unpack, buf, timestamp, first_timestamp, server_port):
    """One packet, or None if it is not part of the conversation."""
    try:
        ip = unpack(buf)
    except (dpkt.dpkt.Error, IndexError, ValueError, KeyError):
        # A frame clipped by the snaplen mid-header is not decodable and
        # is not a trace packet either. Dropped, never guessed at.
        return None

    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        return None

    tcp = ip.data
    if not isinstance(tcp, dpkt.tcp.TCP):
        return None

    if tcp.dport == server_port:
        direction = UP
    elif tcp.sport == server_port:
        direction = DOWN
    else:
        return None

    payload = _payload_length(ip, tcp)
    relative = 0.0 if first_timestamp is None else timestamp - first_timestamp

    return PacketRecord(
        time=relative,
        direction=direction,
        payload_length=payload,
        flags=int(tcp.flags),
    )


def _payload_length(ip, tcp) -> int:
    """
    TCP payload length, taken from the headers.

    NOT `len(tcp.data)`: with `-s 96` the payload was never captured, so
    that would be 0 for every packet, and NOT the frame length, which is
    clipped at 96 so every large packet would look identical. The IP
    total length and the two header lengths are all present in the first
    96 bytes and give the real number.
    """
    if isinstance(ip, dpkt.ip6.IP6):
        # IPv6 payload length already excludes the fixed 40-byte header.
        total = ip.plen
        ip_header = 0
    else:
        total = ip.len
        ip_header = ip.hl * 4

    payload = total - ip_header - (tcp.off * 4)
    return max(0, int(payload))


# --------------------------------------------------------------
# Layer 2: features
# --------------------------------------------------------------

def extract_features(trace: Trace) -> dict[str, float]:
    """
    Turn a Trace into a flat, ordered dict of finite floats.

    Pure: no file I/O and no parser here, so the CLI at step 8 runs the
    same arithmetic the model was trained on.
    """
    up = [packet for packet in trace if packet.direction == UP]
    down = [packet for packet in trace if packet.direction == DOWN]

    features: dict[str, float] = {}
    features.update(_count_features(trace, up, down))
    features.update(_size_features(up, down))
    features.update(_timing_features(trace, up, down))
    features.update(_burst_features(trace))
    features.update(_connection_features(trace))

    return features


def feature_names() -> list[str]:
    """
    The feature order, for building a model's column list.

    Derived from the function itself rather than written out separately,
    because two lists that must agree eventually will not.
    """
    return list(extract_features([]))


# ----------------------------------------------------------
# counts
# ----------------------------------------------------------

def _count_features(trace: Trace, up: Trace, down: Trace) -> dict[str, float]:
    return {
        "count_total": float(len(trace)),
        "count_up": float(len(up)),
        "count_down": float(len(down)),
        # Zero rather than infinity when nothing came back: a division
        # guard that returns inf would poison every downstream statistic
        # and be dropped silently by most estimators.
        "count_ratio_up_down": _safe_divide(len(up), len(down)),
    }


# ----------------------------------------------------------
# sizes
# ----------------------------------------------------------

def _size_features(up: Trace, down: Trace) -> dict[str, float]:
    features: dict[str, float] = {}

    for label, packets in ((UP, up), (DOWN, down)):
        lengths = [packet.payload_length for packet in packets]

        features[f"bytes_{label}_total"] = float(sum(lengths))
        features.update(_summary(f"size_{label}", lengths, percentiles=(25, 50, 75, 90)))

        # Pure ACKs carry no payload. Their count separates "many small
        # exchanges" from "few large transfers" independently of volume.
        features[f"ack_{label}_count"] = float(
            sum(1 for length in lengths if length == 0)
        )

    features["bytes_ratio_up_down"] = _safe_divide(
        features["bytes_up_total"], features["bytes_down_total"]
    )

    return features


# ----------------------------------------------------------
# timing
# ----------------------------------------------------------

def _timing_features(trace: Trace, up: Trace, down: Trace) -> dict[str, float]:
    features: dict[str, float] = {
        "duration": (trace[-1].time - trace[0].time) if len(trace) >= 2 else 0.0
    }

    features.update(_summary("iat", _inter_arrivals(trace), percentiles=(50, 90)))

    for label, packets in ((UP, up), (DOWN, down)):
        features.update(
            _summary(f"iat_{label}", _inter_arrivals(packets), percentiles=(50, 90))
        )

    return features


def _inter_arrivals(packets: Trace) -> list[float]:
    """Gaps between consecutive packets. Fewer than two packets means none."""
    return [
        later.time - earlier.time
        for earlier, later in zip(packets, packets[1:])
    ]


# ----------------------------------------------------------
# bursts
# ----------------------------------------------------------

@dataclass(frozen=True)
class Burst:
    packets: int
    payload_bytes: int
    start: float
    end: float


def find_bursts(trace: Trace) -> list[Burst]:
    """
    Split a trace into bursts.

    **A burst is a maximal run of consecutive packets in the same
    direction.** Nothing else ends one -- in particular, no idle-time
    threshold does.

    This is a modelling choice, not a fact about the data, so it is
    stated rather than buried. The obvious alternative is "same
    direction AND less than T seconds apart", which is a better
    description of a burst in the everyday sense. It is rejected here
    because T is a free parameter with no principled value: on loopback,
    where the RTT is ~0.03 ms, any T separates "the client thinking"
    from "the network working" at a point chosen by us. A threshold
    picked by looking at the data, on a dataset this size, is a
    threshold that can be tuned -- knowingly or not -- toward the
    answer. Direction changes need no parameter and are decided by the
    protocol.

    The cost is that a long idle pause inside one direction does not
    split a burst. The timing features carry that information instead:
    that pause is a large inter-arrival, and `iat_*_p90` sees it.
    """
    bursts: list[Burst] = []

    if not trace:
        return bursts

    direction = trace[0].direction
    count = 0
    payload = 0
    start = trace[0].time
    end = trace[0].time

    for packet in trace:
        if packet.direction != direction:
            bursts.append(Burst(count, payload, start, end))
            direction = packet.direction
            count = 0
            payload = 0
            start = packet.time

        count += 1
        payload += packet.payload_length
        end = packet.time

    # A trace ends mid-burst by definition -- the last run has no
    # direction change after it, and dropping it would lose the final
    # burst of every trace ever captured.
    bursts.append(Burst(count, payload, start, end))

    return bursts


def _burst_features(trace: Trace) -> dict[str, float]:
    bursts = find_bursts(trace)

    lengths = [burst.packets for burst in bursts]
    payloads = [burst.payload_bytes for burst in bursts]
    gaps = [
        later.start - earlier.end
        for earlier, later in zip(bursts, bursts[1:])
    ]

    return {
        "burst_count": float(len(bursts)),
        "burst_len_mean": _mean(lengths),
        "burst_len_max": float(max(lengths)) if lengths else 0.0,
        "burst_bytes_mean": _mean(payloads),
        "burst_bytes_max": float(max(payloads)) if payloads else 0.0,
        "burst_gap_mean": _mean(gaps),
        "burst_gap_max": float(max(gaps)) if gaps else 0.0,
    }


# ----------------------------------------------------------
# connections
# ----------------------------------------------------------

def _connection_features(trace: Trace) -> dict[str, float]:
    """
    Connection-opening counts. Openings only -- see below.

    The SYN count is the parallelism signal and the reason the server
    had to be threaded: measured on round 1, Firefox opens ~6
    connections per page load and wget exactly 1. A SYN-ACK also carries
    SYN, so only pure SYNs are counted as client opens.

    There is deliberately no FIN or RST count. Both separated the
    classes perfectly on round 1 and both were measuring the harness's
    teardown rather than the client's -- the module docstring records
    the measurement and the reasoning. Do not add them back.
    """
    syns = sum(
        1
        for packet in trace
        if packet.flags & TCP_SYN and not packet.flags & TCP_ACK
    )

    return {
        "syn_count": float(syns),
        "syn_ack_count": float(
            sum(
                1
                for packet in trace
                if packet.flags & TCP_SYN and packet.flags & TCP_ACK
            )
        ),
    }


# --------------------------------------------------------------
# Arithmetic helpers -- every one of them returns a finite float
# --------------------------------------------------------------

def _summary(
    prefix: str, values, percentiles: tuple[int, ...] = ()
) -> dict[str, float]:
    """mean/std/min/max plus the requested percentiles, all finite."""
    numbers = sorted(float(value) for value in values)

    summary = {
        f"{prefix}_mean": _mean(numbers),
        f"{prefix}_std": _std(numbers),
        f"{prefix}_min": numbers[0] if numbers else 0.0,
        f"{prefix}_max": numbers[-1] if numbers else 0.0,
    }

    for percentile in percentiles:
        name = "median" if percentile == 50 else f"p{percentile}"
        summary[f"{prefix}_{name}"] = _percentile(numbers, percentile)

    return summary


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _std(values) -> float:
    """
    Population standard deviation.

    Population, not sample: the sample version is undefined for a single
    value and would return NaN, and a trace with one packet in a
    direction is normal rather than exceptional.
    """
    values = list(values)
    if len(values) < 2:
        return 0.0

    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _percentile(sorted_values, percentile: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower
    return float(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def _safe_divide(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else 0.0
