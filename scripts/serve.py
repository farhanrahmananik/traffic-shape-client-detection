#!/usr/bin/env python3
"""
serve.py
--------
Serve the mirrored corpus over local HTTPS, for capture.

All the logic lives in src/tsd/server.py; this file only turns command
line arguments into a MirrorServer. Same split as mirror.py and
scrape_corpus.py: the library is importable and testable, the script is
run-once operational tooling.

Run from the repository root with src/ on the import path:

    PYTHONPATH=src python scripts/serve.py
    PYTHONPATH=src python scripts/serve.py --quiet     # during capture

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
    MirrorServer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the mirrored corpus over local HTTPS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="0 picks a free port; the chosen port is printed at startup",
    )
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    parser.add_argument("--cert", type=Path, default=DEFAULT_CERTFILE)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEYFILE)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=MAX_CONNECTIONS,
        help="connections past this are refused, not queued",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="log only startup and fatal errors; use during capture rounds",
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
    )

    try:
        server.serve_forever()
    except FileNotFoundError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
