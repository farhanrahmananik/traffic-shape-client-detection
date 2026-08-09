#!/usr/bin/env python3
"""
serve.py
--------
Serve the mirrored corpus over local HTTPS, for capture.

All the logic lives in src/tsd/server.py; this file only turns command
line arguments into a MirrorServer. Same split as mirror.py and
scrape_corpus.py: the library is importable and testable, the script is
run-once operational tooling.

Run from the repository root, after `pip install -e .`:

    python scripts/serve.py
    python scripts/serve.py --quiet     # during capture

Prerequisites:

    scripts/make_cert.sh              certs/server.crt + certs/server.key
    scripts/scrape_corpus.py          data/mirror/

The clients must be pointed at the CA explicitly -- the CA is not in any
system trust store, deliberately:

    Firefox   import certs/ca.crt under Authorities (fresh profile per
              page load during capture, to keep the HTTP cache out of
              the measurement)
    wget      --ca-certificate=certs/ca.crt

Use --quiet for capture rounds. Per-connection logging is stderr-only
and never touches disk, but the less the server does during a capture,
the less of the server is in the trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tsd.server import (
    DEFAULT_CERTFILE,
    DEFAULT_HOST,
    DEFAULT_KEYFILE,
    DEFAULT_PORT,
    DEFAULT_WEB_ROOT,
    MAX_CONNECTIONS,
    MAX_REQUESTS_PER_CONNECTION,
    MirrorServer,
)

EPILOG = """\
run from the repository root, after `pip install -e .`:

    python scripts/serve.py
    python scripts/serve.py --quiet     # during capture

Without the install, prefix with PYTHONPATH=src -- the same path
pytest.ini declares, so the suite runs on a fresh clone before anything
has been installed.
"""


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults, and leave the epilog's line breaks alone."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the mirrored corpus over local HTTPS.",
        epilog=EPILOG,
        formatter_class=_HelpFormatter,
    )
    # Every option carries a help string: ArgumentDefaultsHelpFormatter
    # only appends "(default: ...)" to options that have one, so an
    # option without help silently hides its default.
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="address to bind"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="0 picks a free port; the chosen port is printed at startup",
    )
    parser.add_argument(
        "--web-root",
        type=Path,
        default=DEFAULT_WEB_ROOT,
        help="directory to serve, as written by scripts/scrape_corpus.py",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        default=DEFAULT_CERTFILE,
        help="TLS certificate, as written by scripts/make_cert.sh",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=DEFAULT_KEYFILE,
        help="TLS private key, as written by scripts/make_cert.sh",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=MAX_CONNECTIONS,
        help="connections past this are refused, not queued",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=MAX_REQUESTS_PER_CONNECTION,
        help=(
            "requests per connection before the server closes it; reaching "
            "it forces a client reconnection into the trace and prints a "
            "warning that --quiet does not silence"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="log only startup, the request cap, and fatal errors",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    server = MirrorServer(
        web_root=args.web_root,
        host=args.host,
        port=args.port,
        certfile=args.cert,
        keyfile=args.key,
        quiet=args.quiet,
        max_connections=args.max_connections,
        max_requests_per_connection=args.max_requests,
    )

    try:
        server.serve_forever()
    except FileNotFoundError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
