#!/usr/bin/env python3
"""
verify_cli_parity.py
--------------------
Prove that the shipped inference path computes the same numbers as the
training path.

**This is a parity check, not an evaluation.** The model in `models/`
was fitted on all four rounds, so every PCAP under `data/pcaps/` is a
training row for it, and any accuracy figure computed here would be
meaningless -- it would measure how well a model reproduces data it has
already seen. So this script never computes, prints or records accuracy,
predicted labels, or agreement with the `client` column. It compares
**features only**.

What it actually answers: for a real capture, does
`tsd.verdict.classify_pcap()` -- the path the CLI ships -- produce the
same feature values that `scripts/extract_features.py` already wrote
into `data/features/features.csv`, the path the model was trained on?

The two paths are meant to be one path. `verdict.py` calls
`features.extract_features()` rather than reimplementing anything,
precisely so they cannot drift. This script is what turns that intention
into a measurement, because the drift it guards against is silent: a
tool that computed a slightly different `iat_down_max` would still emit
a confident verdict, and the only symptom would be a model performing
worse in the field than the published metrics promised.

Coverage is a finding too. A PCAP with no CSV row, or a CSV row with no
PCAP, means the two sides describe different datasets -- reported as
counts and keys rather than raised as a crash, because either direction
is informative and neither is a reason to stop looking at the rest.

Run from the repository root, after `pip install -e .`:

    python scripts/verify_cli_parity.py
    python scripts/verify_cli_parity.py --limit 20

Exit codes:

    0   every matched trace agreed on every feature, and coverage was
        complete in both directions
    1   a mismatch, a coverage gap, or a refused trace -- OR a --limit
        run, which cannot establish parity over the dataset no matter
        how well the subset agrees
    2   usage error: an input that does not exist, or a model that
        cannot be loaded

A limited run exiting 1 is deliberate. The exit code is what automation
reads, and a `--limit 5` run returning 0 would let a check report
"parity verified" on 5 traces out of 800. The flag is for iteration; the
verdict on the dataset comes from a full run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from tsd.verdict import VerdictError, classify_pcap, load_artefact

DEFAULT_PCAP_ROOT = Path("data/pcaps")
DEFAULT_FEATURES = Path("data/features/features.csv")
DEFAULT_MODEL = Path("models/client_classifier.joblib")
DEFAULT_OUT = Path("results/cli_parity.json")

LABEL_COLUMNS = ("round", "date", "client", "page")
ROUND_DIRECTORY = re.compile(r"^round_(?P<round>\d+)_(?P<date>\d{8})$")

# Both sides round to 6 decimals before comparing -- verdict.py already
# does, and the CSV values are read back through the same rounding.
ROUND_DECIMALS = 6

# Enough to see the shape of a problem without turning the record into
# a data dump.
MAX_EXAMPLES = 20

EXIT_OK = 0
EXIT_DIVERGED = 1
EXIT_USAGE = 2

Key = tuple[int, str, str, str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that the CLI inference path computes the same features "
            "as the training path. Features only -- never accuracy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pcap-root", type=Path, default=DEFAULT_PCAP_ROOT,
                        help="directory holding the round directories")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                        help="feature CSV written by scripts/extract_features.py")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="model artefact; used only to build the vector")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="machine-readable record of the comparison")
    parser.add_argument(
        "--limit", type=int, default=None,
        help=(
            "compare only the first N traces in sorted order, for quick "
            "iteration; a limited run always exits 1, because it cannot "
            "establish parity over the dataset"
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------
# Loading the two sides
# --------------------------------------------------------------

def read_csv_rows(path: Path) -> dict[Key, dict[str, str]]:
    """Index the feature table by (round, date, client, page)."""
    rows: dict[Key, dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[csv_key(row)] = row

    return rows


def csv_key(row: dict[str, str]) -> Key:
    return (int(row["round"]), row["date"], row["client"], row["page"])


def find_traces(pcap_root: Path) -> list[tuple[Key, Path]]:
    """
    Every PCAP under the round directories, sorted.

    Sorted so the record is deterministic: the mismatch examples are
    taken in walk order, and an unstable order would make two runs on
    unchanged inputs produce different files.
    """
    traces: list[tuple[Key, Path]] = []

    for directory in sorted(p for p in pcap_root.iterdir() if p.is_dir()):
        match = ROUND_DIRECTORY.match(directory.name)
        if match is None:
            continue

        number = int(match.group("round"))
        date = match.group("date")

        for client_directory in sorted(p for p in directory.iterdir() if p.is_dir()):
            for pcap in sorted(client_directory.glob("*.pcap")):
                key = (number, date, client_directory.name, pcap.stem)
                traces.append((key, pcap))

    return traces


# --------------------------------------------------------------
# Comparison
# --------------------------------------------------------------

def compare_features(
    csv_row: dict[str, str], cli_features: dict[str, float]
) -> list[tuple[str, float, float]]:
    """
    Every feature where the two paths disagree.

    Exact equality after identical rounding, not a tolerance. A
    tolerance would decide in advance how much drift is acceptable --
    which is the question being asked, not an input to it. The two paths
    call the same function on the same bytes; anything other than
    equality is a finding, and its size is reported rather than
    pre-approved.
    """
    differences: list[tuple[str, float, float]] = []

    for name, cli_value in cli_features.items():
        raw = csv_row.get(name)
        if raw is None:
            differences.append((name, float("nan"), cli_value))
            continue

        csv_value = round(float(raw), ROUND_DECIMALS)
        if csv_value != cli_value:
            differences.append((name, csv_value, cli_value))

    return differences


def key_text(key: Key) -> str:
    number, date, client, page = key
    return f"round_{number:02d}_{date}/{client}/{page}"


def relative(path: Path) -> str:
    """Paths are recorded relative to the working directory, never absolute."""
    try:
        return os.path.relpath(path)
    except ValueError:  # different drive on Windows
        return str(path)


# --------------------------------------------------------------
# Run
# --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for label, path in (("pcap root", args.pcap_root), ("feature CSV", args.features)):
        if not path.exists():
            print(f"ABORT: {label} {path} does not exist", file=sys.stderr)
            return EXIT_USAGE

    try:
        artefact = load_artefact(args.model)
    except VerdictError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_USAGE

    csv_rows = read_csv_rows(args.features)
    traces = find_traces(args.pcap_root)

    partial = args.limit is not None
    if partial:
        traces = traces[: args.limit]

    # Coverage, both directions. The reverse direction is only
    # meaningful over a full walk: after --limit, "CSV rows with no
    # PCAP" would list everything that was simply not reached, which is
    # a fact about the flag rather than about the data.
    pcap_keys = {key for key, _ in traces}
    without_csv = sorted(key_text(key) for key in pcap_keys - set(csv_rows))
    without_pcap = (
        None if partial
        else sorted(key_text(key) for key in set(csv_rows) - pcap_keys)
    )

    compared = 0
    matching = 0
    refused: list[dict] = []
    mismatched_features: dict[str, int] = {}
    examples: list[dict] = []

    for key, pcap in traces:
        row = csv_rows.get(key)
        if row is None:
            continue

        try:
            document = classify_pcap(pcap, artefact, include_features=True)
        except VerdictError as error:
            # A trace the shipped path refuses but the training path
            # accepted is itself a divergence between the two.
            refused.append({"trace": key_text(key), "reason": str(error)})
            continue

        differences = compare_features(row, document["features"])
        compared += 1

        if not differences:
            matching += 1
            continue

        for name, csv_value, cli_value in differences:
            mismatched_features[name] = mismatched_features.get(name, 0) + 1

            if len(examples) < MAX_EXAMPLES:
                examples.append({
                    "trace": key_text(key),
                    "feature": name,
                    "training_path": csv_value,
                    "inference_path": cli_value,
                    "abs_diff": abs(cli_value - csv_value),
                })

    record = {
        "check": "cli-vs-training feature parity; features only, never accuracy",
        "model": {
            "path": relative(args.model),
            "sha256": artefact.sha256,
            "n_features": len(artefact.features),
        },
        "limit": args.limit,
        "partial": partial,
        "coverage": {
            "pcaps_found": len(traces),
            "csv_rows": len(csv_rows),
            "matched": compared + len(refused),
            "pcaps_without_csv_row": without_csv,
            "csv_rows_without_pcap": without_pcap,
        },
        "comparison": {
            "traces_compared": compared,
            "traces_matching_all_features": matching,
            "refused_by_inference_path": refused,
            "mismatched_features": dict(sorted(mismatched_features.items())),
            "examples": examples,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_summary(args, record)

    clean = (
        compared == matching
        and not refused
        and not without_csv
        and not without_pcap
    )

    if partial:
        print(
            f"\nPARTIAL RUN: --limit {args.limit} compared {compared} of "
            f"{len(csv_rows)} rows.\n"
            f"A limited run cannot establish parity over the dataset, so it "
            f"exits non-zero however well the subset agreed. Run without "
            f"--limit for the verdict.",
            file=sys.stderr,
        )
        return EXIT_DIVERGED

    if not clean:
        print(
            f"\nDIVERGENCE: {compared - matching} trace(s) disagreed, "
            f"{len(refused)} refused, {len(without_csv)} PCAP(s) without a CSV "
            f"row, {len(without_pcap or [])} CSV row(s) without a PCAP.\n"
            f"The shipped tool and the training path are not computing the "
            f"same numbers. Details in {relative(args.out)}.",
            file=sys.stderr,
        )
        return EXIT_DIVERGED

    return EXIT_OK


def print_summary(args, record: dict) -> None:
    coverage = record["coverage"]
    comparison = record["comparison"]

    print("CLI parity check (features only — this is not an evaluation)")
    print(f"  pcaps found     : {coverage['pcaps_found']}")
    print(f"  csv rows        : {coverage['csv_rows']}")
    print(f"  compared        : {comparison['traces_compared']}")
    print(f"  agreeing on all : {comparison['traces_matching_all_features']}")
    print(f"  features/trace  : {record['model']['n_features']}")

    if coverage["pcaps_without_csv_row"]:
        print(f"  pcaps with no csv row : "
              f"{len(coverage['pcaps_without_csv_row'])}")
    if coverage["csv_rows_without_pcap"]:
        print(f"  csv rows with no pcap : "
              f"{len(coverage['csv_rows_without_pcap'])}")
    if comparison["refused_by_inference_path"]:
        print(f"  refused by the CLI    : "
              f"{len(comparison['refused_by_inference_path'])}")

    if comparison["mismatched_features"]:
        print("  mismatching features:")
        for name, count in comparison["mismatched_features"].items():
            print(f"    {name:<24} {count}")

    print(f"  record          : {relative(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
