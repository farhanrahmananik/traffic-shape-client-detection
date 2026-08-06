"""
test_capture.py
---------------
Tests for tsd.capture.

Neither the network namespace nor the real clients can run in CI, so the
module is built with the deciding separated from the doing: directory
naming, command construction, the quiet-period decision, packet counting
and metadata are all pure, and those are what is tested here.

What each test is protecting is the round -- a capture round is expensive
to take and impossible to check afterwards. A zero-packet PCAP, a round
written into the previous round's directory, or a namespace that was not
actually isolated all produce files that look completely normal. By the
time the classifier is confused, the conditions that produced the data
are gone.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

from tsd.capture import (
    CLIENTS,
    NETNS_MARKER,
    CaptureConfig,
    InvalidPcap,
    IsolationError,
    RoundExists,
    build_round_metadata,
    certificate_fingerprint,
    certutil_import_command,
    count_pcap_packets,
    firefox_command,
    firefox_environment,
    inside_namespace,
    is_tcpdump_ready,
    list_pages,
    metadata_path,
    namespace_command,
    page_stem,
    page_url,
    prepare_round_directory,
    round_directory,
    round_name,
    tcpdump_command,
    trace_path,
    verify_isolation,
    wait_for_quiet,
    wget_command,
    RoundResult,
    TraceResult,
)


@pytest.fixture
def mirror(tmp_path):
    root = tmp_path / "mirror"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_bytes(b"<html>home</html>")
    (root / "fakultaet1_e4b6727e14b7.html").write_bytes(b"<html>f1</html>")
    (root / "news_500820843cff.html").write_bytes(b"<html>news</html>")
    (root / "assets" / "logo.png").write_bytes(b"PNG")
    return root


@pytest.fixture
def config(tmp_path, mirror):
    return CaptureConfig(
        round_number=1,
        date="20260807",
        web_root=mirror,
        pcap_root=tmp_path / "pcaps",
        metadata_root=tmp_path / "results" / "capture_rounds",
        ca_cert=tmp_path / "ca.crt",
        server_cert=tmp_path / "server.crt",
        server_key=tmp_path / "server.key",
    )


def write_pcap(path: Path, packets: int, snaplen: int = 96) -> Path:
    """A real little-endian pcap with `packets` records of headers only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, snaplen, 1)
        )
        for index in range(packets):
            payload = bytes([index % 256]) * 54
            handle.write(struct.pack("<IIII", index, 0, len(payload), len(payload)))
            handle.write(payload)
    return path


class FakeClock:
    """Monotonic time that only advances when something sleeps."""

    def __init__(self):
        self.time = 0.0

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += seconds


# --------------------------------------------------------------
# Layout: a round is a split group, so its identity is structural
# --------------------------------------------------------------

def test_round_directory_names_encode_round_and_date(config):
    assert round_name(1, "20260807") == "round_01_20260807"
    assert round_name(12, "20260807") == "round_12_20260807"
    assert round_directory(config).name == "round_01_20260807"


def test_trace_paths_are_per_client_and_named_after_the_page(config):
    path = trace_path(config, "firefox", "fakultaet1_e4b6727e14b7.html")

    assert path.parent.name == "firefox"
    assert path.parent.parent.name == "round_01_20260807"
    assert path.name == "fakultaet1_e4b6727e14b7.pcap"


def test_page_stem_matches_the_mirror_filename(config):
    """
    The trace is named after the page file, so a PCAP can be traced back
    to a manifest entry -- and therefore to a URL -- without a lookup
    table that could drift.
    """
    assert page_stem("news_500820843cff.html") == "news_500820843cff"
    assert page_stem(Path("index.html")) == "index"


def test_page_urls_point_at_the_local_server(config):
    assert page_url(config, "index.html") == "https://127.0.0.1:8443/index.html"


def test_only_pages_are_captured_not_assets(mirror):
    pages = [page.name for page in list_pages(mirror)]

    assert pages == sorted(pages), "capture order must be stable across rounds"
    assert "index.html" in pages
    assert all(page.endswith(".html") for page in pages)
    assert len(pages) == 3


def test_round_directory_is_created_with_one_directory_per_client(config):
    prepare_round_directory(config)

    for client in CLIENTS:
        assert (round_directory(config) / client).is_dir()


def test_existing_round_is_refused(config):
    """
    Two runs merged into one directory would present as a single round.
    The classifier splits BY round, so traces that shared no conditions
    would be treated as though they did -- leakage introduced by a
    filename, and invisible afterwards.
    """
    prepare_round_directory(config)

    with pytest.raises(RoundExists) as raised:
        prepare_round_directory(config)

    assert "split" in str(raised.value)


def test_force_allows_recapturing_a_round(config):
    prepare_round_directory(config)
    config.force = True

    prepare_round_directory(config)  # must not raise


# --------------------------------------------------------------
# Namespace: the two measured gotchas
# --------------------------------------------------------------

def test_namespace_command_uses_unshare_n_not_rn():
    """
    `unshare -rn` adds a user namespace mapping the caller to uid 0.
    tcpdump then tries to drop privileges and chown the savefile to a
    uid that is not mapped, fails, and exits -- leaving a 24-byte pcap
    with zero packets and no obvious error. Measured, and the reason
    this is asserted rather than remembered.
    """
    command = namespace_command(["python", "x.py"], user="farhan")
    joined = " ".join(command)

    assert command[:3] == ["sudo", "unshare", "-n"]
    assert "-rn" not in joined
    assert "unshare -r" not in joined


def test_namespace_command_drops_back_to_the_real_user():
    """Root-owned PCAPs would push the whole pipeline to run as root."""
    command = namespace_command(["python", "x.py"], user="farhan")
    script = command[-1]

    assert "sudo -u farhan" in script
    assert f"{NETNS_MARKER}=1" in script


def test_namespace_command_brings_loopback_up():
    """A fresh namespace has lo DOWN, and the whole capture is over lo."""
    script = namespace_command(["python", "x.py"], user="farhan")[-1]

    assert script.startswith("ip link set lo up")


def test_namespace_command_passes_through_the_display_variables():
    command = namespace_command(
        ["firefox"], user="farhan",
        passthrough={"WAYLAND_DISPLAY": "wayland-0", "PYTHONPATH": "src"},
    )
    script = command[-1]

    assert "WAYLAND_DISPLAY=wayland-0" in script
    assert "PYTHONPATH=src" in script


def test_inside_namespace_reads_the_marker():
    assert inside_namespace({NETNS_MARKER: "1"}) is True
    assert inside_namespace({}) is False


def test_isolation_check_aborts_when_an_external_host_resolves():
    """
    An isolation check that is never exercised is not a check. This is
    the one that stands between the dataset and 409 ms of matomo.js
    latency landing in the Firefox class only.
    """
    with pytest.raises(IsolationError) as raised:
        verify_isolation(probe=lambda host: True)

    assert "not isolated" in str(raised.value)
    assert "Refusing to capture" in str(raised.value)


def test_isolation_check_passes_when_nothing_is_reachable():
    verify_isolation(probe=lambda host: False)


def test_namespace_marker_can_be_waived_but_isolation_cannot(config, monkeypatch):
    """
    --no-netns exists for a shell that was unshared by hand, where the
    marker is not set. It waives the marker only: the marker is a claim
    about the namespace, while verify_isolation() is a measurement of
    it, and the measurement always runs.
    """
    from tsd.capture import CaptureRound

    monkeypatch.setattr("tsd.capture.verify_isolation",
                        lambda *a, **k: (_ for _ in ()).throw(
                            IsolationError("reachable")))

    config.ca_cert.write_text("x")
    config.server_cert.write_text("x")
    config.clients = ("wget",)  # certutil is only needed for Firefox

    with pytest.raises(IsolationError):
        CaptureRound(config, require_namespace_marker=False).run()


def test_missing_namespace_marker_is_refused_by_default(config):
    from tsd.capture import CaptureError, CaptureRound

    config.ca_cert.write_text("x")
    config.server_cert.write_text("x")
    config.clients = ("wget",)

    with pytest.raises(CaptureError) as raised:
        CaptureRound(config).run()

    assert "network namespace" in str(raised.value)


def test_isolation_check_tries_every_probe_host():
    tried: list[str] = []

    verify_isolation(probe=lambda host: tried.append(host) or False)

    assert len(tried) >= 2


# --------------------------------------------------------------
# Commands
# --------------------------------------------------------------

def test_tcpdump_is_filtered_and_headers_only(tmp_path):
    """
    Two of the three flaws this rebuild exists to fix: the original
    captures ran unfiltered and stored full payload. Both are asserted
    rather than trusted to a comment.
    """
    command = tcpdump_command(tmp_path / "out.pcap", snaplen=96)

    assert command[0] == "tcpdump"
    assert "-i" in command and command[command.index("-i") + 1] == "lo"
    assert command[command.index("-s") + 1] == "96"
    assert "host 127.0.0.1 and port 8443" in command


def test_tcpdump_snaplen_cannot_reach_a_tls_record_payload(tmp_path):
    """96 bytes covers the headers and stops before any TLS content."""
    command = tcpdump_command(tmp_path / "out.pcap")
    snaplen = int(command[command.index("-s") + 1])

    assert snaplen <= 96


def test_wget_suppresses_the_robots_request(tmp_path):
    """
    In recursive mode wget requests /robots.txt, gets a 404 the mirror
    cannot answer, and Firefox never does the same. That 404 is a
    property of the harness, not of the client -- a classifier that
    scores well by finding it has learned the harness.
    """
    command = wget_command("https://127.0.0.1:8443/index.html",
                           Path("certs/ca.crt"), tmp_path)

    assert "-e" in command
    assert command[command.index("-e") + 1] == "robots=off"
    assert "--page-requisites" in command
    assert "--ca-certificate=certs/ca.crt" in command


def test_firefox_is_given_a_profile_and_no_shared_instance(tmp_path):
    """
    --profile does not create the directory; a missing one kills Firefox
    with SIGKILL and no useful message. The directory is created by the
    caller, and --new-instance keeps a running browser from swallowing
    the load.
    """
    command = firefox_command(tmp_path / "profile", "https://127.0.0.1:8443/x.html")

    assert command[command.index("--profile") + 1] == str(tmp_path / "profile")
    assert "--new-instance" in command
    assert "--no-remote" in command


def test_ca_is_imported_as_a_website_authority(tmp_path):
    """
    `C,,` is what the Authorities import in the Firefox UI does. The
    alternative -- an exception -- is a different TLS code path from
    wget's clean chain verification, which is the asymmetry the local CA
    exists to remove.
    """
    command = certutil_import_command(tmp_path / "profile", Path("certs/ca.crt"))

    assert command[command.index("-t") + 1] == "C,,"
    assert command[-1] == f"sql:{tmp_path / 'profile'}"


def test_firefox_environment_forces_wayland_and_carries_the_socket():
    """
    X11 uses abstract unix sockets, which a network namespace isolates.
    Wayland's socket is a filesystem path and survives, so Wayland is
    selected explicitly rather than left to autodetection.
    """
    environment = firefox_environment(
        {"WAYLAND_DISPLAY": "wayland-0", "XDG_RUNTIME_DIR": "/run/user/1000",
         "DISPLAY": ":0"}
    )

    assert environment["MOZ_ENABLE_WAYLAND"] == "1"
    assert environment["WAYLAND_DISPLAY"] == "wayland-0"
    assert environment["XDG_RUNTIME_DIR"] == "/run/user/1000"


def test_tcpdump_ready_marker():
    assert is_tcpdump_ready("tcpdump: listening on lo, link-type EN10MB")
    assert not is_tcpdump_ready("tcpdump: verbose output suppressed")


# --------------------------------------------------------------
# PCAP inspection -- catching the capture that never ran
# --------------------------------------------------------------

def test_packet_counting(tmp_path):
    assert count_pcap_packets(write_pcap(tmp_path / "a.pcap", 5)) == 5
    assert count_pcap_packets(write_pcap(tmp_path / "b.pcap", 1)) == 1


def test_header_only_pcap_counts_zero_packets(tmp_path):
    """
    The signature of `unshare -rn`: tcpdump wrote the 24-byte global
    header, failed to chown the savefile, and exited. The file exists,
    has a plausible name and a valid header, and contains nothing.
    Nothing else in the pipeline would notice.
    """
    empty = write_pcap(tmp_path / "empty.pcap", 0)

    assert empty.stat().st_size == 24
    assert count_pcap_packets(empty) == 0


def test_truncated_and_foreign_files_are_rejected(tmp_path):
    short = tmp_path / "short.pcap"
    short.write_bytes(b"\xd4\xc3\xb2\xa1")
    with pytest.raises(InvalidPcap):
        count_pcap_packets(short)

    foreign = tmp_path / "foreign.pcap"
    foreign.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40)  # pcapng, not pcap
    with pytest.raises(InvalidPcap):
        count_pcap_packets(foreign)


def test_big_endian_pcap_is_counted(tmp_path):
    path = tmp_path / "be.pcap"
    with path.open("wb") as handle:
        handle.write(struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 96, 1))
        for _ in range(3):
            handle.write(struct.pack(">IIII", 0, 0, 40, 40))
            handle.write(b"\x00" * 40)

    assert count_pcap_packets(path) == 3


# --------------------------------------------------------------
# When a load is over
# --------------------------------------------------------------

def test_quiet_period_ends_the_load_after_packets_stop():
    """
    No fixed padding. The original captures put 3 s at each end of every
    trace, which is a constant in both classes and pure noise.
    """
    clock = FakeClock()
    counts = iter([0, 4, 9, 14, 14, 14, 14, 14, 14, 14, 14, 14])

    result = wait_for_quiet(
        lambda: next(counts, 14),
        quiet_seconds=1.0, poll_interval=0.25,
        sleep=clock.sleep, now=clock.now,
    )

    assert result.packets == 14
    assert result.timed_out is False


def test_quiet_period_waits_through_a_pause_shorter_than_the_threshold():
    """
    Firefox keeps requesting carousel images after the page reports
    itself loaded (measured). A pause inside the threshold must not end
    the load, or the trace loses exactly the part that distinguishes it
    from wget.

    Note what this does and does not claim. The threshold IS the
    definition of "finished": a burst arriving later than
    quiet_seconds after the previous packet is by construction a
    separate event, and no implementation can wait for it without
    waiting forever. That is why the threshold is a CLI flag and why its
    value is recorded in the round metadata -- it shapes every trace,
    and choosing it is a judgement about the client, not a detail.
    """
    clock = FakeClock()
    # 1.0 s pause -- inside the 1.5 s threshold -- then a late burst.
    counts = iter([0, 5, 5, 5, 9, 9, 9, 9, 9])

    result = wait_for_quiet(
        lambda: next(counts, 9),
        quiet_seconds=1.5, poll_interval=0.5,
        sleep=clock.sleep, now=clock.now,
    )

    assert result.packets == 9, "must not have stopped during the short pause"
    assert result.timed_out is False


def test_quiet_period_never_declares_success_on_an_empty_capture():
    """
    A client that never connected must be reported as an empty trace,
    not mistaken for one that finished quickly.
    """
    clock = FakeClock()

    result = wait_for_quiet(
        lambda: 0,
        quiet_seconds=1.0, max_wait=5.0, poll_interval=0.5,
        sleep=clock.sleep, now=clock.now,
    )

    assert result.empty is True
    assert result.timed_out is True


def test_quiet_period_times_out_on_a_load_that_never_settles():
    clock = FakeClock()
    counter = {"n": 0}

    def growing():
        counter["n"] += 3
        return counter["n"]

    result = wait_for_quiet(
        growing,
        quiet_seconds=1.0, max_wait=4.0, poll_interval=0.5,
        sleep=clock.sleep, now=clock.now,
    )

    assert result.timed_out is True
    assert result.packets > 0


# --------------------------------------------------------------
# Metadata -- published, so it must carry no BTU content
# --------------------------------------------------------------

def build_result() -> RoundResult:
    result = RoundResult()
    result.traces.append(
        TraceResult("firefox", "index.html", "p/firefox/index.pcap", 812, 61000)
    )
    result.traces.append(
        TraceResult("wget", "index.html", "p/wget/index.pcap", 240, 18000)
    )
    result.traces.append(
        TraceResult("wget", "news_5008.html", "p/wget/news_5008.pcap",
                    0, 24, ok=False, reason="zero packets captured")
    )
    return result


def test_metadata_records_how_the_round_was_actually_run(config):
    metadata = build_round_metadata(
        config=config,
        started_at="2026-08-07T10:00:00+00:00",
        finished_at="2026-08-07T11:30:00+00:00",
        result=build_result(),
        versions={"firefox": "Mozilla Firefox 153.0.3", "wget": "GNU Wget 1.21.4"},
        invocations={
            "wget": wget_command("<url>", config.ca_cert, Path("<tmp>")),
            "firefox": firefox_command(Path("<profile>"), "<url>"),
        },
        fingerprint="AA:BB:CC",
    )

    assert metadata["round"] == 1
    assert metadata["snaplen"] == 96
    assert metadata["tcpdump_filter"] == "host 127.0.0.1 and port 8443"
    assert metadata["quiet_seconds"] == config.quiet_seconds
    assert metadata["server_cert_sha256"] == "AA:BB:CC"
    assert metadata["versions"]["firefox"] == "Mozilla Firefox 153.0.3"
    assert "robots=off" in metadata["invocations"]["wget"]
    assert metadata["totals"]["traces_ok"] == 2
    assert metadata["totals"]["traces_failed"] == 1
    assert metadata["totals"]["per_client"] == {"firefox": 1, "wget": 1}


def test_metadata_contains_no_page_content(config, mirror):
    """
    results/ is published. A round's metadata may describe the traces;
    it may not carry a byte of what was served.
    """
    metadata = build_round_metadata(
        config=config, started_at="a", finished_at="b", result=build_result(),
        versions={}, invocations={}, fingerprint=None,
    )
    serialised = json.dumps(metadata)

    assert "<html>" not in serialised
    assert "home" not in serialised


def test_failed_traces_are_recorded_not_dropped(config):
    result = build_result()

    assert len(result.failures) == 1
    assert result.failures[0].reason == "zero packets captured"
    assert result.per_client() == {"firefox": 1, "wget": 1}


def test_metadata_path_is_per_round(config):
    assert metadata_path(config).name == "round_01_20260807.json"


def test_certificate_fingerprint_matches_openssl(tmp_path):
    """
    The fingerprint identifies which certificate a round was captured
    against, so a round taken after an accidental regeneration is
    identifiable instead of silently mixed in. It has to be the same
    number openssl prints, or it cannot be compared with
    results/provenance/tls_cert.txt.
    """
    if subprocess.run(["openssl", "version"], capture_output=True,
                      check=False).returncode != 0:
        pytest.skip("openssl not available")

    key = tmp_path / "k.pem"
    crt = tmp_path / "c.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "ec",
         "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-days", "1",
         "-subj", "/CN=127.0.0.1", "-keyout", str(key), "-out", str(crt)],
        capture_output=True, check=True,
    )

    expected = subprocess.run(
        ["openssl", "x509", "-in", str(crt), "-noout", "-fingerprint", "-sha256"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("=", 1)[1]

    assert certificate_fingerprint(crt) == expected
