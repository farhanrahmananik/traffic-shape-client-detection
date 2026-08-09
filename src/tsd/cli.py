"""
cli.py
------
The argparse layer over `tsd.verdict`. Parses argv, calls
`load_artefact()` and `classify_pcap()`, serialises the result, and maps
exceptions to exit codes.

No classification and no feature logic lives here. `verdict.py` owns
those, and it in turn calls `features.py` rather than reimplementing
anything -- so the tool that ships computes the same numbers the
published metrics were measured from.

stdout is JSON and nothing else
-------------------------------
Every diagnostic, warning and error goes to stderr; stdout carries the
verdict document or nothing at all. The tool has to pipe into `jq`
without the caller filtering anything out first, and a single stray
line of progress text on stdout breaks that for every consumer at once.

On failure stdout stays **completely empty** rather than carrying a
partial document. A half-written JSON object is worse than none: a
pipeline reading it would not notice.

Exit codes
----------
    0   a verdict was produced
    2   usage error (argparse's own default)
    3   VerdictError -- unreadable PCAP, too-short trace, or a missing
        or stale model artefact

The exit code **never encodes the predicted class**. Exit status answers
"did the tool work", not "what did it find". Overloading it would make

    classify_pcap x.pcap || echo failed

print "failed" whenever the answer happened to be wget, which is a
correct verdict and not an error. The class is in the JSON, where a
caller has to read it deliberately.

    pip install -e .
    tsd-classify capture.pcap
    tsd-classify capture.pcap | jq .verdict

Also runnable without installing, for development:

    PYTHONPATH=src python -m tsd.cli capture.pcap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .features import SERVER_PORT
from .verdict import VerdictError, classify_pcap, load_artefact

DEFAULT_MODEL = Path("models/client_classifier.joblib")

EXIT_OK = 0
EXIT_USAGE = 2  # argparse exits with this itself
EXIT_VERDICT_ERROR = 3

EPILOG = """\
exit codes:
  0  a verdict was produced
  2  usage error
  3  the PCAP or the model could not be used -- unreadable capture,
     a trace too short to be a page load, or a missing or stale model

The exit code never encodes the predicted class: it answers "did the
tool work", not "what did it find". Read the verdict from the JSON.

stdout carries the JSON document and nothing else; diagnostics go to
stderr, so the output pipes into jq unfiltered.
"""


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """
    The parser. `prog` is passed straight through to argparse.

    Left as None in normal use, so argparse derives the name from
    sys.argv[0] and the help text names the command the reader actually
    typed -- `tsd-classify` through the console script, or
    `classify_pcap.py` through the shim. Hardcoding it meant `--help`
    announced a name that did not exist on the caller's system.

    Tests that assert on the error prefix pass an explicit `prog`. That
    is the only case the hardcoded default was serving, and it belongs
    in the test rather than in the shipped parser.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Classify one PCAP as a browser or an automated client.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The version is reported here and deliberately NOT put into the
    # verdict JSON: that document's schema is already published, and
    # adding a field is a schema change. If provenance is wanted in the
    # output later it belongs next to the model sha256, as its own
    # decision rather than as a side effect of packaging.
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("pcap", type=Path, help="the capture to classify")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"model artefact from scripts/train_model.py "
             f"(default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--server-port", type=int, default=SERVER_PORT,
        help=f"port that identifies the server side of the conversation, "
             f"which is what decides packet direction (default: {SERVER_PORT})",
    )
    parser.add_argument(
        "--include-features", action="store_true",
        help="include the extracted feature values in the output",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="one line of JSON instead of indented",
    )
    return parser


def serialise(document: dict, compact: bool) -> str:
    """
    Render the verdict.

    `sort_keys=False` always: `verdict.py` fixes a deliberate key order,
    and sorting would replace a schema chosen for reading with one
    chosen by the alphabet.
    """
    if compact:
        return json.dumps(document, separators=(",", ":"), sort_keys=False)

    return json.dumps(document, indent=2, sort_keys=False)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)

    try:
        artefact = load_artefact(args.model)
        document = classify_pcap(
            args.pcap,
            artefact,
            server_port=args.server_port,
            include_features=args.include_features,
        )
    except VerdictError as error:
        # `parser.prog`, not a literal: the prefix on an error and the
        # name in --help are the same fact, and two copies of one fact
        # eventually disagree.
        #
        # stderr, and stdout left untouched: a consumer piping this into
        # jq must get either a whole document or nothing.
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return EXIT_VERDICT_ERROR

    print(serialise(document, args.compact))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
