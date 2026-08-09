#!/usr/bin/env python3
"""
build_site_data.py
------------------
Build `docs/data/case_study.json` from the measured artefacts in
`results/`.

A shim. All the logic lives in `src/tsd/site_data.py`, the same way
`scripts/classify_pcap.py` wraps `src/tsd/cli.py`.

The case-study page fetches this file and renders it. **No number on
that page is ever typed by hand** -- if a figure is not in here, it does
not go on the page. That is the point of the build step: prose written
next to a number goes stale the moment the number moves, and nothing
checks prose.

    pip install -e .
    python scripts/build_site_data.py
    python scripts/build_site_data.py --check

`--check` rebuilds in memory and compares byte-for-byte with the file on
disk, writing nothing. It is what catches a `docs/` copy that drifted
from `results/` -- someone editing the JSON by hand, or a rerun of the
metrics that nobody rebuilt the site from.

Exit codes:

    0   written, or `--check` found the file up to date
    1   a required source or key is missing, or `--check` found a
        difference
    2   usage error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tsd.site_data import (
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS_DIR,
    SiteDataError,
    build,
    first_difference,
    serialise,
)

EXIT_OK = 0
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the case-study data file from results/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="directory holding the measured artefacts",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help="the file the case-study page fetches",
    )
    parser.add_argument(
        "--check", action="store_true",
        help=(
            "rebuild in memory and compare with the file on disk; write "
            "nothing, and fail if they differ"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        document = build(args.results_dir)
    except SiteDataError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_FAILED

    rendered = serialise(document)

    if args.check:
        return check(args.out, document, rendered)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")

    print(f"wrote {args.out} ({len(rendered.encode('utf-8'))} bytes)")
    print("  top-level keys: " + ", ".join(document))
    return EXIT_OK


def check(out: Path, document: dict, rendered: str) -> int:
    """Compare without writing. The whole value of --check is that it writes nothing."""
    if not out.is_file():
        print(f"CHECK FAILED: {out} does not exist; run without --check",
              file=sys.stderr)
        return EXIT_FAILED

    on_disk = out.read_text(encoding="utf-8")

    if on_disk == rendered:
        print(f"{out} is up to date with {DEFAULT_RESULTS_DIR}")
        return EXIT_OK

    try:
        import json

        differing = first_difference(document, json.loads(on_disk))
        where = f"first differing top-level key: {differing}"
    except ValueError:
        where = "the file on disk is not valid JSON"

    print(
        f"CHECK FAILED: {out} differs from what results/ would produce.\n"
        f"  {where}\n"
        f"  Rebuild it: python scripts/build_site_data.py",
        file=sys.stderr,
    )
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
