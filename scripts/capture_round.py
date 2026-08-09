#!/usr/bin/env python3
"""
capture_round.py
----------------
Capture one round: every page in data/mirror, loaded once by each client,
one PCAP per page load.

All the logic lives in src/tsd/capture.py; this file parses arguments,
puts itself inside the capture network namespace, and reports. Same split
as mirror.py / scrape_corpus.py and server.py / serve.py.

    pip install -e .
    python scripts/capture_round.py --round 1
    python scripts/capture_round.py --round 2 --limit 3

The script re-executes ITSELF under `sudo unshare -n`, so you do not
wrap it by hand -- and so the server and both clients end up in the same
namespace. Each namespace has its own loopback: a server started outside
is unreachable from inside, which fails in a way that looks like a
broken server rather than a broken invocation.

You will be asked for a sudo password once, for the namespace. Nothing
inside it runs as root: the first thing that happens in the namespace is
`ip link set lo up`, and then a drop straight back to the invoking user,
so the PCAPs are yours and tcpdump uses its capabilities rather than
root.

Prerequisites:

    scripts/make_cert.sh              certs/ca.crt, certs/server.crt
    scripts/scrape_corpus.py          data/mirror/
    sudo apt install libnss3-tools    certutil, to trust the CA per profile

Exit codes:

    0   round captured, every trace usable
    1   round captured but at least one trace failed
    2   refused to start (not isolated, round exists, missing prerequisite)
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tsd.capture import (
    CLIENTS,
    DEFAULT_CA_CERT,
    DEFAULT_MAX_LOAD_SECONDS,
    DEFAULT_METADATA_ROOT,
    DEFAULT_PCAP_ROOT,
    DEFAULT_PORT,
    DEFAULT_QUIET_SECONDS,
    DEFAULT_SERVER_CERT,
    DEFAULT_SERVER_KEY,
    DEFAULT_SNAPLEN,
    DEFAULT_WEB_ROOT,
    CaptureConfig,
    CaptureError,
    CaptureRound,
    build_round_metadata,
    certificate_fingerprint,
    collect_versions,
    firefox_command,
    firefox_environment,
    inside_namespace,
    namespace_command,
    round_directory,
    wget_command,
    write_metadata,
)

EXIT_OK = 0
EXIT_TRACES_FAILED = 1
EXIT_REFUSED = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one round of page loads, one PCAP per load.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--round", type=int, required=True, dest="round_number",
        help="round number; rounds are the groups the train/test split uses",
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y%m%d"),
        help="YYYYMMDD, part of the round directory name",
    )
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT,
                        help="directory of pages to load")
    parser.add_argument("--pcap-root", type=Path, default=DEFAULT_PCAP_ROOT,
                        help="where round directories are created")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT,
                        help="where the published round metadata is written")
    parser.add_argument("--ca-cert", type=Path, default=DEFAULT_CA_CERT,
                        help="CA both clients are pointed at")
    parser.add_argument("--cert", type=Path, default=DEFAULT_SERVER_CERT,
                        help="server certificate")
    parser.add_argument("--key", type=Path, default=DEFAULT_SERVER_KEY,
                        help="server private key")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port the round's server listens on")
    parser.add_argument("--snaplen", type=int, default=DEFAULT_SNAPLEN,
                        help="tcpdump -s; headers only, never payload")
    parser.add_argument(
        "--quiet-seconds", type=float, default=DEFAULT_QUIET_SECONDS,
        help=(
            "seconds with no new packets before a load is called finished; "
            "recorded in the round metadata because it shapes every trace"
        ),
    )
    parser.add_argument("--max-load-seconds", type=float,
                        default=DEFAULT_MAX_LOAD_SECONDS,
                        help="give up on a page load after this long")
    parser.add_argument("--clients", nargs="+", default=list(CLIENTS),
                        choices=list(CLIENTS), help="clients to capture")
    parser.add_argument("--limit", type=int, default=None,
                        help="capture only the first N pages; for a smoke run")
    parser.add_argument("--force", action="store_true",
                        help="discard an existing round directory and recapture it")
    parser.add_argument(
        "--no-netns", action="store_true",
        help=(
            "do not re-execute under sudo unshare -n. Only for a shell that "
            "is already inside the namespace"
        ),
    )
    return parser.parse_args(argv)


def reexec_in_namespace(argv: list[str]) -> int:
    """
    Re-run this script inside a loopback-only network namespace.

    Doing it here rather than asking the operator to remember
    `sudo unshare -n ...` is deliberate: the namespace is not a
    convenience, it is what keeps 409 ms of live-site analytics latency
    out of the Firefox class. A step that has to be remembered is a step
    that will eventually be forgotten, and the resulting round would
    look completely normal.
    """
    inner = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--no-netns",
    ]
    passthrough = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", "src"),
        **firefox_environment(),
    }
    command = namespace_command(inner, user=getpass.getuser(),
                                passthrough=passthrough)

    print("entering capture namespace:", " ".join(command), file=sys.stderr)
    return subprocess.call(command)


def progress(kind: str, *payload) -> None:
    if kind == "isolated":
        print("isolation verified: the outside world is unreachable",
              file=sys.stderr, flush=True)
    elif kind == "page":
        client, page, index, total = payload
        print(f"  [{index:>3}/{total}] {client:<8} {page}",
              file=sys.stderr, flush=True)
    elif kind == "trace_failed":
        client, page, reason = payload
        print(f"  FAILED  {client:<8} {page}: {reason}",
              file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = parse_args(arguments)

    if not args.no_netns and not inside_namespace():
        return reexec_in_namespace([a for a in arguments if a != "--no-netns"])

    config = CaptureConfig(
        round_number=args.round_number,
        date=args.date,
        web_root=args.web_root,
        pcap_root=args.pcap_root,
        metadata_root=args.metadata_root,
        ca_cert=args.ca_cert,
        server_cert=args.cert,
        server_key=args.key,
        port=args.port,
        snaplen=args.snaplen,
        quiet_seconds=args.quiet_seconds,
        max_load_seconds=args.max_load_seconds,
        clients=tuple(args.clients),
        force=args.force,
        limit=args.limit,
    )

    started_at = utc_now()
    round_capture = CaptureRound(
        config,
        on_event=progress,
        # --no-netns waives the marker, never the isolation check: the
        # flag is a claim, and verify_isolation() is a measurement.
        require_namespace_marker=not args.no_netns,
    )

    try:
        result = round_capture.run()
    except CaptureError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_REFUSED

    finished_at = utc_now()

    try:
        fingerprint = certificate_fingerprint(config.server_cert)
    except (OSError, ValueError):
        fingerprint = None

    metadata = build_round_metadata(
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        result=result,
        versions=collect_versions(),
        invocations={
            "wget": wget_command("<url>", config.ca_cert, Path("<tmp>")),
            "firefox": firefox_command(Path("<fresh-profile>"), "<url>"),
        },
        fingerprint=fingerprint,
    )
    written = write_metadata(config, metadata)

    print_summary(config, result, written)

    if result.failures:
        print(
            f"\n{len(result.failures)} trace(s) failed. They are recorded in "
            f"{written} and their PCAPs must not be used.\n"
            f"A zero-packet pcap is the signature of a capture that never ran "
            f"-- check the namespace and the server before recapturing.",
            file=sys.stderr,
        )
        return EXIT_TRACES_FAILED

    return EXIT_OK


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def print_summary(config: CaptureConfig, result, metadata_file: Path) -> None:
    print("round captured")
    for client, count in sorted(result.per_client().items()):
        print(f"  {client:<12}: {count} traces")
    print(f"  failed      : {len(result.failures)}")
    print(f"  pcaps       : {round_directory(config)}")
    print(f"  metadata    : {metadata_file}")


if __name__ == "__main__":
    sys.exit(main())
