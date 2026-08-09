#!/usr/bin/env python3
"""
extract_features.py
-------------------
Read every PCAP under data/pcaps/ and write one feature table to
data/features/.

All the arithmetic lives in src/tsd/features.py; this script walks the
round directories, labels each row, cross-checks against the published
round metadata, and writes CSV. Same split as scrape_corpus.py over
mirror.py: the library is importable and tested, the script is
operational.

    pip install -e .
    python scripts/extract_features.py
    python scripts/extract_features.py --rounds round_01_20260807

Labels come from the layout, never from a re-derivation:

    data/pcaps/round_01_20260807/firefox/<page>.pcap
               ^^^^^^^^ ^^^^^^^^ ^^^^^^^  ^^^^
               round    date     client   page

`round` is the group column for `GroupKFold` in step 6. It is written
into the CSV rather than reconstructed from a path later, because the
split is the one thing in this project that cannot be checked by looking
at the result: a leaked group produces better numbers, not worse ones,
and nothing downstream complains.

Cross-check against results/capture_rounds/*.json: the number of PCAPs
found on disk must equal `totals.traces_ok`. A mismatch is an error, not
a warning. The metadata is what gets published in place of the PCAPs, so
if the two disagree, either the published record is wrong or the data
was changed outside the harness -- and both mean the round can no longer
be described honestly.

Why data/features/ is gitignored: the CSV contains no BTU content --
only counts, sizes and timings, with the payload never captured in the
first place. It is withheld because it is derived from PCAPs that are
not published, and publishing a derivative of unpublished data would be
inconsistent. Not because it leaks anything.

Exit codes:

    0   every trace parsed and every round matched its metadata
    1   at least one parse failure or metadata mismatch
    2   nothing to do, or refused to overwrite the output
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tsd.features import extract_features, feature_names, read_trace, TraceError

DEFAULT_PCAP_ROOT = Path("data/pcaps")
DEFAULT_METADATA_ROOT = Path("results/capture_rounds")
DEFAULT_OUTPUT = Path("data/features/features.csv")

LABEL_COLUMNS = ("round", "date", "client", "page")
ROUND_DIRECTORY = re.compile(r"^round_(?P<round>\d+)_(?P<date>\d{8})$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


@dataclass
class RoundSummary:
    name: str
    number: int
    date: str
    read: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    per_client: dict[str, int] = field(default_factory=dict)
    mismatch: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-trace features from captured PCAPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds", nargs="+", default=None,
        help="round directory names to process; default every round found",
    )
    parser.add_argument("--pcap-root", type=Path, default=DEFAULT_PCAP_ROOT,
                        help="directory holding the round directories")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT,
                        help="published round metadata, used as the cross-check")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV to write")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    return parser.parse_args(argv)


# --------------------------------------------------------------
# Discovery
# --------------------------------------------------------------

def find_rounds(pcap_root: Path, wanted: list[str] | None) -> list[Path]:
    """Round directories, in a stable order."""
    rounds = [
        path
        for path in sorted(pcap_root.iterdir())
        if path.is_dir() and ROUND_DIRECTORY.match(path.name)
    ]

    if wanted:
        chosen = set(wanted)
        rounds = [path for path in rounds if path.name in chosen]

        missing = chosen - {path.name for path in rounds}
        if missing:
            raise SystemExit(f"ABORT: no such round(s): {', '.join(sorted(missing))}")

    return rounds


def round_labels(directory: Path) -> tuple[int, str]:
    match = ROUND_DIRECTORY.match(directory.name)
    if match is None:  # pragma: no cover - find_rounds already filtered
        raise ValueError(f"not a round directory: {directory}")

    return int(match.group("round")), match.group("date")


# --------------------------------------------------------------
# Extraction
# --------------------------------------------------------------

def process_round(directory: Path, columns: list[str]) -> tuple[list[dict], RoundSummary]:
    number, date = round_labels(directory)
    summary = RoundSummary(name=directory.name, number=number, date=date)
    rows: list[dict] = []

    for client_directory in sorted(p for p in directory.iterdir() if p.is_dir()):
        client = client_directory.name

        for pcap in sorted(client_directory.glob("*.pcap")):
            try:
                trace = read_trace(pcap)
                features = extract_features(trace)
            except (TraceError, OSError) as error:
                # Recorded, never skipped quietly: a trace missing from
                # the table would shrink one class without saying so.
                summary.failed.append((pcap.name, f"{type(error).__name__}: {error}"))
                continue

            row = {
                "round": number,
                "date": date,
                "client": client,
                "page": pcap.stem,
            }
            row.update({name: features[name] for name in columns})

            rows.append(row)
            summary.read += 1
            summary.per_client[client] = summary.per_client.get(client, 0) + 1

    return rows, summary


def cross_check(summary: RoundSummary, metadata_root: Path, found: int) -> None:
    """
    Compare what is on disk with what the published metadata claims.

    The metadata is published in place of the PCAPs. If the two
    disagree, the published record describes a round that no longer
    exists -- PCAPs added, moved or deleted outside the harness. That
    cannot be corrected by re-reading the disk, because there is no way
    to tell which of the two is right.
    """
    path = metadata_root / f"{summary.name}.json"

    if not path.is_file():
        summary.mismatch = (
            f"no metadata at {path}; this round was not produced by "
            f"scripts/capture_round.py, or its record was lost"
        )
        return

    import json

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        expected = int(metadata["totals"]["traces_ok"])
    except (ValueError, KeyError, OSError) as error:
        summary.mismatch = f"could not read {path}: {error}"
        return

    if expected != found:
        summary.mismatch = (
            f"{path.name} records traces_ok={expected}, but {found} PCAP(s) "
            f"were found on disk. The PCAPs and the published record of them "
            f"disagree, so neither can be trusted for this round."
        )


# --------------------------------------------------------------
# Reporting
# --------------------------------------------------------------

def constant_features(rows: list[dict], columns: list[str]) -> list[str]:
    """
    Features with one value across the whole dataset.

    Surfaced, never dropped automatically. A constant feature carries no
    information, but the interesting question is why it is constant --
    usually a bug in extraction, occasionally a fact about the traffic.
    Dropping it silently would answer neither.
    """
    constants = []

    for name in columns:
        values = {row[name] for row in rows}
        if len(values) == 1:
            constants.append(f"{name} = {next(iter(values))}")

    return constants


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*LABEL_COLUMNS, *columns])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    columns = feature_names()

    if not args.pcap_root.is_dir():
        print(f"ABORT: {args.pcap_root} does not exist", file=sys.stderr)
        return EXIT_REFUSED

    if args.output.exists() and not args.force:
        print(
            f"ABORT: {args.output} already exists. Use --force to replace it.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    rounds = find_rounds(args.pcap_root, args.rounds)
    if not rounds:
        print(f"ABORT: no round directories under {args.pcap_root}", file=sys.stderr)
        return EXIT_REFUSED

    all_rows: list[dict] = []
    summaries: list[RoundSummary] = []

    for directory in rounds:
        rows, summary = process_round(directory, columns)
        cross_check(summary, args.metadata_root, found=summary.read + len(summary.failed))

        all_rows.extend(rows)
        summaries.append(summary)

        print(f"{summary.name}: {summary.read} traces "
              f"({', '.join(f'{c} {n}' for c, n in sorted(summary.per_client.items()))})",
              file=sys.stderr)

        for name, reason in summary.failed:
            print(f"  PARSE FAILED {name}: {reason}", file=sys.stderr)
        if summary.mismatch:
            print(f"  METADATA MISMATCH {summary.mismatch}", file=sys.stderr)

    if not all_rows:
        print("ABORT: no traces could be read", file=sys.stderr)
        return EXIT_REFUSED

    write_csv(args.output, all_rows, columns)

    constants = constant_features(all_rows, columns)
    print_summary(args, all_rows, summaries, columns, constants)

    failed = sum(len(summary.failed) for summary in summaries)
    mismatched = [summary for summary in summaries if summary.mismatch]

    if failed or mismatched:
        print(
            f"\n{failed} parse failure(s), {len(mismatched)} round(s) disagreeing "
            f"with their published metadata.\n"
            f"The CSV was written, but it does not describe the rounds the "
            f"metadata describes. Do not train on it until they agree.",
            file=sys.stderr,
        )
        return EXIT_FAILED

    return EXIT_OK


def print_summary(args, rows, summaries, columns, constants) -> None:
    per_client: dict[str, int] = {}
    for row in rows:
        per_client[row["client"]] = per_client.get(row["client"], 0) + 1

    print("features extracted")
    print(f"  rounds        : {len(summaries)}")
    print(f"  traces        : {len(rows)}")
    for client, count in sorted(per_client.items()):
        print(f"    {client:<10}: {count}")
    print(f"  features      : {len(columns)}")
    print(f"  output        : {args.output}")

    if constants:
        print(f"  constant      : {len(constants)} feature(s) with a single value "
              f"across the dataset")
        for entry in constants:
            print(f"    {entry}")
        print("    A constant feature carries no information. Check whether it is "
              "a fact about the traffic or a bug in extraction before step 6.")


if __name__ == "__main__":
    sys.exit(main())
