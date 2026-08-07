"""
test_extract_features.py
------------------------
Tests for scripts/extract_features.py.

The script's guards were verified by hand when it was written -- the
overwrite refusal, the metadata cross-check, the constant-feature
report. Verified by hand means verified once. Every other guard in this
repo is held by a test, and this one holds the boundary between "the
PCAPs on disk" and "the published record of them", which is the pair
that has to agree before anything is trained.

The script lives in scripts/, which is not on the import path
(pytest.ini declares src/ only, because that is the importable library).
It is added here deliberately rather than by moving the script: the
split between library and operational tooling is a decision recorded in
CLAUDE.md, and a test is not a reason to undo it.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import extract_features as script  # noqa: E402
from tsd.features import SERVER_PORT, feature_names  # noqa: E402

TCP_ACK = 0x10
TCP_SYN = 0x02


# --------------------------------------------------------------
# Building a believable round on disk
# --------------------------------------------------------------

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


def trace_packets(client: str) -> list[tuple[float, bytes]]:
    """A tiny but plausible trace: a handshake and some exchange."""
    up = lambda payload, flags=TCP_ACK: frame(50000, SERVER_PORT, payload, flags)
    down = lambda payload, flags=TCP_ACK: frame(SERVER_PORT, 50000, payload, flags)

    if client == "firefox":
        return [
            (0.0, up(0, TCP_SYN)), (0.001, down(0, TCP_SYN | TCP_ACK)),
            (0.002, up(0)), (0.003, up(517)),
            (0.010, down(1400)), (0.011, down(1400)), (0.012, down(900)),
            (0.020, up(0)),
        ]

    return [
        (0.0, up(0, TCP_SYN)), (0.001, down(0, TCP_SYN | TCP_ACK)),
        (0.002, up(0)), (0.003, up(300)),
        (0.005, down(1200)), (0.006, up(0)),
    ]


@pytest.fixture
def round_on_disk(tmp_path):
    """One round directory with two clients and two pages each."""
    pcap_root = tmp_path / "pcaps"
    metadata_root = tmp_path / "capture_rounds"
    directory = pcap_root / "round_01_20260807"

    pages = ("index", "fakultaet1_e4b6727e14b7")
    for client in ("firefox", "wget"):
        for page in pages:
            write_pcap(directory / client / f"{page}.pcap", trace_packets(client))

    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "round_01_20260807.json").write_text(
        json.dumps({"round": 1, "date": "20260807",
                    "totals": {"traces_ok": 4, "traces_failed": 0}}),
        encoding="utf-8",
    )

    return pcap_root, metadata_root, tmp_path / "features.csv"


def run(pcap_root, metadata_root, output, *extra) -> int:
    return script.main([
        "--pcap-root", str(pcap_root),
        "--metadata-root", str(metadata_root),
        "--output", str(output),
        *extra,
    ])


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


# --------------------------------------------------------------
# Labels come from the layout
# --------------------------------------------------------------

def test_labels_come_from_the_directory_layout_and_filename(round_on_disk):
    """
    round, date and client are read from the path and page from the
    filename. `round` in particular is written into the CSV rather than
    reconstructed later, because it is the GroupKFold group and a
    mislabelled group produces better numbers, not worse ones.
    """
    pcap_root, metadata_root, output = round_on_disk

    assert run(pcap_root, metadata_root, output) == script.EXIT_OK

    header, rows = read_csv(output)

    assert header[:4] == ["round", "date", "client", "page"]
    assert {row["client"] for row in rows} == {"firefox", "wget"}
    assert {row["round"] for row in rows} == {"1"}
    assert {row["date"] for row in rows} == {"20260807"}
    assert {row["page"] for row in rows} == {"index", "fakultaet1_e4b6727e14b7"}
    assert len(rows) == 4


def test_feature_columns_are_in_feature_names_order(round_on_disk):
    """
    A saved model's column order must stay valid. If the CSV's order
    ever drifted from feature_names(), the model would be fed the right
    numbers under the wrong names and still produce plausible output.
    """
    pcap_root, metadata_root, output = round_on_disk
    run(pcap_root, metadata_root, output)

    header, _ = read_csv(output)

    assert header[4:] == feature_names()


# --------------------------------------------------------------
# Guards
# --------------------------------------------------------------

def test_existing_output_is_not_overwritten_without_force(round_on_disk, capsys):
    pcap_root, metadata_root, output = round_on_disk
    output.write_text("do not lose me", encoding="utf-8")

    assert run(pcap_root, metadata_root, output) == script.EXIT_REFUSED
    assert output.read_text(encoding="utf-8") == "do not lose me"
    assert "already exists" in capsys.readouterr().err


def test_force_replaces_the_output(round_on_disk):
    pcap_root, metadata_root, output = round_on_disk
    output.write_text("stale", encoding="utf-8")

    assert run(pcap_root, metadata_root, output, "--force") == script.EXIT_OK
    assert "round,date,client,page" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("traces_ok", [3, 5])
def test_metadata_mismatch_is_an_error_and_the_csv_is_still_written(
    round_on_disk, capsys, traces_ok
):
    """
    Fewer or more PCAPs than the published record claims. Either the
    record is wrong or the data changed outside the harness, and
    re-reading the disk cannot say which -- so it is an error, not a
    warning.

    The CSV is still written, because the first thing anyone will want
    is to look at what is actually there. stderr says not to train on it.
    """
    pcap_root, metadata_root, output = round_on_disk
    (metadata_root / "round_01_20260807.json").write_text(
        json.dumps({"totals": {"traces_ok": traces_ok}}), encoding="utf-8"
    )

    assert run(pcap_root, metadata_root, output) == script.EXIT_FAILED

    error = capsys.readouterr().err
    assert "METADATA MISMATCH" in error
    assert f"traces_ok={traces_ok}" in error
    assert "Do not train on it" in error
    assert output.is_file(), "the CSV must survive for investigation"


def test_round_without_metadata_is_an_error(round_on_disk, capsys):
    """
    The metadata is what is published in place of the PCAPs. A round
    with none was not produced by the harness, or its record was lost --
    either way it cannot be described honestly.
    """
    pcap_root, metadata_root, output = round_on_disk
    (metadata_root / "round_01_20260807.json").unlink()

    assert run(pcap_root, metadata_root, output) == script.EXIT_FAILED
    assert "no metadata at" in capsys.readouterr().err


def test_unknown_round_name_aborts(round_on_disk):
    pcap_root, metadata_root, output = round_on_disk

    with pytest.raises(SystemExit) as raised:
        run(pcap_root, metadata_root, output, "--rounds", "round_99_20990101")

    assert "no such round" in str(raised.value)


def test_selecting_one_round_processes_only_that_round(tmp_path, round_on_disk):
    pcap_root, metadata_root, output = round_on_disk

    second = pcap_root / "round_02_20260808"
    write_pcap(second / "wget" / "index.pcap", trace_packets("wget"))
    (metadata_root / "round_02_20260808.json").write_text(
        json.dumps({"totals": {"traces_ok": 1}}), encoding="utf-8"
    )

    assert run(pcap_root, metadata_root, output,
               "--rounds", "round_02_20260808") == script.EXIT_OK

    _, rows = read_csv(output)
    assert {row["round"] for row in rows} == {"2"}


def test_missing_pcap_root_is_refused(tmp_path, capsys):
    assert run(tmp_path / "nothing", tmp_path, tmp_path / "out.csv") \
        == script.EXIT_REFUSED
    assert "does not exist" in capsys.readouterr().err


def test_directory_with_no_rounds_is_refused(tmp_path, capsys):
    empty = tmp_path / "pcaps"
    empty.mkdir()

    assert run(empty, tmp_path, tmp_path / "out.csv") == script.EXIT_REFUSED
    assert "no round directories" in capsys.readouterr().err


def test_unparsable_pcap_is_reported_and_the_run_fails(round_on_disk, capsys):
    """
    A trace that cannot be read must not simply be absent from the
    table: that would shrink one class without saying so.
    """
    pcap_root, metadata_root, output = round_on_disk
    broken = pcap_root / "round_01_20260807" / "wget" / "index.pcap"
    broken.write_bytes(b"not a pcap")

    assert run(pcap_root, metadata_root, output) == script.EXIT_FAILED

    error = capsys.readouterr().err
    assert "PARSE FAILED" in error
    _, rows = read_csv(output)
    assert len(rows) == 3


# --------------------------------------------------------------
# Constant features: reported, never dropped
# --------------------------------------------------------------

def test_constant_features_are_reported_but_kept(round_on_disk, capsys):
    """
    Dropping a feature because it is constant is a decision made by
    looking at the whole dataset, test rounds included -- a mild leak --
    and a feature that is constant in round 1 may not be in round 3.
    The report exists to raise the question, not to answer it.
    """
    pcap_root, metadata_root, output = round_on_disk
    run(pcap_root, metadata_root, output)

    report = capsys.readouterr().out
    header, rows = read_csv(output)

    assert "constant" in report
    # Every synthetic trace here starts with a SYN, so syn_count is one
    # value across the dataset -- and it is still a column.
    assert "syn_count" in header
    assert {row["syn_count"] for row in rows} == {"1.0"}


def test_constant_feature_detection_finds_the_right_columns():
    rows = [
        {"a": 1.0, "b": 2.0, "c": 5.0},
        {"a": 1.0, "b": 3.0, "c": 5.0},
    ]

    constants = script.constant_features(rows, ["a", "b", "c"])

    assert [entry.split(" =")[0] for entry in constants] == ["a", "c"]
