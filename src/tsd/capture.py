"""
capture.py
----------
One capture round: every page in the mirror, loaded once by each client,
one PCAP per page load.

    data/pcaps/round_<NN>_<YYYYMMDD>/<client>/<page_stem>.pcap

A round is the unit the classifier splits on. `GroupKFold` is given the
round as the group, because traces from one round share conditions --
machine load, kernel scheduling, the exact binaries installed that day --
and a random split would let the model see those conditions on both
sides. That is why this module refuses to write into an existing round
directory: two runs merged into one directory would look like one round
and quietly destroy the group boundary that the whole evaluation rests
on.

Three flaws in the original captures are the reason this project is being
rebuilt. None of them may come back:

    1. tcpdump ran unfiltered      -> the filter is not optional here
    2. full payload was stored     -> -s 96 keeps headers, drops payload
    3. 3 s of padding at each end  -> no fixed padding anywhere

Everything runs inside a network namespace with loopback only, and that
is a measured requirement rather than a precaution. On 2026-08-06,
loading the mirror's root page in Firefox pulled BTU's own matomo.js
from the live site (68.45 kB, 409 ms) and attempted a tracking beacon
(aborted after 190 ms), from JavaScript that link rewriting cannot reach.
Neither appears in a PCAP filtered to the local host and port -- but
against a loopback RTT of ~0.03 ms, that latency lands inside Firefox's
inter-arrival times, and wget executes no JavaScript, so it lands on one
class only.

Testability: everything except the subprocess calls themselves is a pure
function here -- directory naming, command construction, the quiet-period
decision, packet counting, metadata. The namespace and the real clients
cannot run in CI, so the parts that decide what happens are separated
from the parts that make it happen.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WEB_ROOT = Path("data/mirror")
DEFAULT_PCAP_ROOT = Path("data/pcaps")
DEFAULT_METADATA_ROOT = Path("results/capture_rounds")
DEFAULT_CA_CERT = Path("certs/ca.crt")
DEFAULT_SERVER_CERT = Path("certs/server.crt")
DEFAULT_SERVER_KEY = Path("certs/server.key")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8443
CAPTURE_INTERFACE = "lo"

# Headers only. 96 bytes covers Ethernet/loopback + IPv4 + TCP with
# options, and stops before anything a TLS record would carry. No BTU
# content can reach the disk through a PCAP.
DEFAULT_SNAPLEN = 96

# Stop the capture after this long with no new packets. Not a fixed
# duration: Firefox keeps requesting carousel images after the page
# reports itself loaded (measured), so a short fixed timeout would cut
# off exactly the part of the trace that distinguishes it from wget.
DEFAULT_QUIET_SECONDS = 3.0
DEFAULT_MAX_LOAD_SECONDS = 90.0
POLL_INTERVAL = 0.25

CLIENTS = ("firefox", "wget")

# Set on the re-executed process so it knows it is already inside the
# namespace and must not unshare again.
NETNS_MARKER = "TSD_IN_NETNS"

CA_NICKNAME = "traffic-shape-client-detection local CA"

# tcpdump prints this once the capture is actually live. Launching a
# client before it appears loses the first packets of the trace -- which
# are the handshake, the most structured part of it.
TCPDUMP_READY_MARKER = "listening on"

PCAP_GLOBAL_HEADER_BYTES = 24
PCAP_RECORD_HEADER_BYTES = 16
PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": "<",  # microsecond, little endian
    b"\xa1\xb2\xc3\xd4": ">",  # microsecond, big endian
    b"\x4d\x3c\xb2\xa1": "<",  # nanosecond, little endian
    b"\xa1\xb2\x3c\x4d": ">",  # nanosecond, big endian
}

# Hosts the isolation check must fail to reach. If any of them resolves,
# the namespace is not doing its job and the round must not start.
ISOLATION_PROBE_HOSTS = ("www.b-tu.de", "example.org")
ISOLATION_PROBE_ADDRESS = ("1.1.1.1", 443)
ISOLATION_PROBE_TIMEOUT = 2.0


class CaptureError(RuntimeError):
    """Something that must stop the round."""


class IsolationError(CaptureError):
    """The namespace is not isolated. Nothing may be captured."""


class RoundExists(CaptureError):
    """The round directory already exists and --force was not given."""


class InvalidPcap(CaptureError):
    """The file is not a pcap this module can read."""


@dataclass
class CaptureConfig:
    """Everything one round needs to know. No behaviour."""

    round_number: int
    date: str  # YYYYMMDD
    web_root: Path = DEFAULT_WEB_ROOT
    pcap_root: Path = DEFAULT_PCAP_ROOT
    metadata_root: Path = DEFAULT_METADATA_ROOT
    ca_cert: Path = DEFAULT_CA_CERT
    server_cert: Path = DEFAULT_SERVER_CERT
    server_key: Path = DEFAULT_SERVER_KEY
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    snaplen: int = DEFAULT_SNAPLEN
    quiet_seconds: float = DEFAULT_QUIET_SECONDS
    max_load_seconds: float = DEFAULT_MAX_LOAD_SECONDS
    clients: tuple[str, ...] = CLIENTS
    force: bool = False
    limit: int | None = None


@dataclass
class TraceResult:
    """One page load by one client."""

    client: str
    page: str
    path: str
    packets: int = 0
    bytes: int = 0
    ok: bool = True
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "client": self.client,
            "page": self.page,
            "pcap": self.path,
            "packets": self.packets,
            "bytes": self.bytes,
            "ok": self.ok,
            "reason": self.reason,
        }


@dataclass
class RoundResult:
    traces: list[TraceResult] = field(default_factory=list)

    @property
    def failures(self) -> list[TraceResult]:
        return [trace for trace in self.traces if not trace.ok]

    def per_client(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trace in self.traces:
            if trace.ok:
                counts[trace.client] = counts.get(trace.client, 0) + 1
        return counts


# --------------------------------------------------------------
# Naming and layout
# --------------------------------------------------------------

def round_name(round_number: int, date: str) -> str:
    return f"round_{round_number:02d}_{date}"


def round_directory(config: CaptureConfig) -> Path:
    return config.pcap_root / round_name(config.round_number, config.date)


def client_directory(config: CaptureConfig, client: str) -> Path:
    return round_directory(config) / client


def page_stem(page: Path | str) -> str:
    """`fakultaet1_e4b6727e14b7.html` -> `fakultaet1_e4b6727e14b7`."""
    return Path(page).stem


def trace_path(config: CaptureConfig, client: str, page: Path | str) -> Path:
    return client_directory(config, client) / f"{page_stem(page)}.pcap"


def metadata_path(config: CaptureConfig) -> Path:
    return config.metadata_root / f"{round_name(config.round_number, config.date)}.json"


def list_pages(web_root: Path) -> list[Path]:
    """
    Every page in the mirror, in a stable order.

    Sorted, because the order pages are captured in is part of the
    round's conditions: machine state drifts over a round, so a
    different order would be a different experiment.
    """
    return sorted(Path(web_root).glob("*.html"))


def page_url(config: CaptureConfig, page: Path | str) -> str:
    return f"https://{config.host}:{config.port}/{Path(page).name}"


def prepare_round_directory(config: CaptureConfig) -> Path:
    """
    Create the round directory, refusing to reuse an existing one.

    A round is a split group. Two runs written into one directory would
    present as a single round, and traces that shared no conditions
    would be treated as though they did -- leakage introduced by a
    filename.
    """
    directory = round_directory(config)

    if directory.exists() and not config.force:
        raise RoundExists(
            f"{directory} already exists.\n"
            f"A round is the group the train/test split uses, so a second "
            f"run must not write into it.\n"
            f"Use the next round number, or --force to discard this one and "
            f"recapture it."
        )

    for client in config.clients:
        (directory / client).mkdir(parents=True, exist_ok=True)

    return directory


# --------------------------------------------------------------
# Network namespace
# --------------------------------------------------------------

def inside_namespace(environ: dict | None = None) -> bool:
    return (environ if environ is not None else os.environ).get(NETNS_MARKER) == "1"


def namespace_command(
    argv: list[str],
    user: str,
    passthrough: dict[str, str] | None = None,
    interface: str = CAPTURE_INTERFACE,
) -> list[str]:
    """
    Wrap a command so it runs inside a loopback-only network namespace,
    as the real user.

    Two details here are measured, not stylistic.

    `sudo unshare -n`, never `unshare -rn`. The -r flag adds a USER
    namespace that maps the caller to uid 0. tcpdump then tries to drop
    privileges and chown the savefile to a uid that is not mapped in
    that namespace, fails, and exits -- leaving a 24-byte pcap
    containing nothing but the global header, with no obvious error.
    A round captured that way looks like it worked.

    And then `sudo -u <user>` immediately, so the PCAPs are owned by the
    person who has to read them, not by root. tcpdump has
    cap_net_raw,cap_net_admin=eip precisely so the pipeline does not run
    as root.

    A fresh namespace has loopback DOWN, so it is brought up before
    anything else -- the whole capture happens over lo.
    """
    environment = ["env", f"{NETNS_MARKER}=1"]
    for name, value in (passthrough or {}).items():
        environment.append(f"{name}={value}")

    inner = shlex.join([*environment, *argv])
    script = f"ip link set {interface} up && exec sudo -u {shlex.quote(user)} {inner}"

    return ["sudo", "unshare", "-n", "bash", "-c", script]


def verify_isolation(probe=None, hosts: tuple[str, ...] = ISOLATION_PROBE_HOSTS) -> None:
    """
    Refuse to capture unless the outside world is unreachable.

    An isolation check that never runs is not a check, and this one is
    the difference between a clean dataset and one contaminated by
    409 ms of analytics latency on one class only. It runs before every
    round, and it aborts the round rather than warning about it.
    """
    probe = probe or _default_probe

    for host in hosts:
        if probe(host):
            raise IsolationError(
                f"{host} is reachable from inside the capture namespace.\n"
                f"The namespace is not isolated, so a page load could reach "
                f"the live site.\n"
                f"Measured 2026-08-06: Firefox fetched matomo.js from b-tu.de "
                f"(409 ms) during a mirror page load -- invisible to a "
                f"filtered tcpdump, but present in the timing, and only for "
                f"the client that runs JavaScript.\n"
                f"Refusing to capture."
            )


def _default_probe(host: str) -> bool:
    """True if the outside world answered. Any answer at all is a failure."""
    try:
        socket.setdefaulttimeout(ISOLATION_PROBE_TIMEOUT)
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        pass
    finally:
        socket.setdefaulttimeout(None)

    # A literal address as well: resolution could succeed from
    # /etc/hosts or a cache without the network being reachable, and it
    # could equally fail while raw connectivity remains.
    try:
        with socket.create_connection(
            ISOLATION_PROBE_ADDRESS, timeout=ISOLATION_PROBE_TIMEOUT
        ):
            return True
    except OSError:
        return False


# --------------------------------------------------------------
# Commands
# --------------------------------------------------------------

def tcpdump_command(
    output: Path,
    snaplen: int = DEFAULT_SNAPLEN,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    interface: str = CAPTURE_INTERFACE,
) -> list[str]:
    """
    The capture command, with both of the original flaws designed out.

    The filter is not optional: an unfiltered capture on lo would record
    whatever else the machine does over loopback during the load.
    -s 96 keeps the TCP/IP headers and discards the payload, so the
    PCAPs contain no BTU content and can never become a republication of
    the mirror.
    """
    return [
        "tcpdump",
        "-i", interface,
        "-s", str(snaplen),
        "-w", str(output),
        "-U",  # write each packet as it arrives, so the size is a live signal
        f"host {host} and port {port}",
    ]


def wget_command(url: str, ca_cert: Path, download_dir: Path) -> list[str]:
    """
    wget, invoked so that the harness leaves as little of itself in the
    trace as possible.

    `-e robots=off` is deliberate. In recursive mode wget requests
    /robots.txt first, and the mirror has none, so every wget trace
    would open with a 404 that Firefox never produces. That request is a
    property of how this harness invokes wget, not of wget being
    automation -- and a classifier that scores well by finding a 404 has
    learned the harness, not the client.

    Firefox's own /favicon.ico request is NOT suppressed, and the
    asymmetry is deliberate: the browser does that by itself, unasked,
    so it is genuine client behaviour and belongs in the data.
    """
    return [
        "wget",
        f"--ca-certificate={ca_cert}",
        "--page-requisites",
        "--no-directories",
        "-e", "robots=off",
        "--quiet",
        f"--directory-prefix={download_dir}",
        url,
    ]


def firefox_command(
    profile_dir: Path, url: str, binary: str = "firefox"
) -> list[str]:
    """
    Firefox against a fresh profile.

    The profile directory must already exist. `firefox --profile <dir>`
    does not create it, and a missing directory kills the process with
    SIGKILL and no useful message -- measured, and an hour to diagnose.
    """
    return [
        binary,
        "--profile", str(profile_dir),
        "--no-remote",
        "--new-instance",
        url,
    ]


def certutil_create_command(profile_dir: Path) -> list[str]:
    return ["certutil", "-N", "--empty-password", "-d", f"sql:{profile_dir}"]


def certutil_import_command(
    profile_dir: Path, ca_cert: Path, nickname: str = CA_NICKNAME
) -> list[str]:
    """
    Import the local CA into a fresh profile's NSS store.

    `C,,` marks it trusted to identify websites, which is what the
    Authorities import in the Firefox UI does. If certutil is missing,
    the round fails with an install instruction rather than falling back
    to an insecure flag: an exception path is a different TLS code path
    from wget's clean chain verification, and that asymmetry is exactly
    what having a CA was for.
    """
    return [
        "certutil",
        "-A",
        "-n", nickname,
        "-t", "C,,",
        "-i", str(ca_cert),
        "-d", f"sql:{profile_dir}",
    ]


def firefox_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """
    The display variables Firefox needs inside the namespace.

    X11 uses abstract unix sockets, which live in the network namespace
    and are therefore cut off by it. Wayland's socket is a filesystem
    path, so it survives -- which is why MOZ_ENABLE_WAYLAND=1 is set
    rather than left to autodetection. Verified working inside the
    namespace.
    """
    base = base if base is not None else dict(os.environ)
    passthrough = {"MOZ_ENABLE_WAYLAND": "1"}

    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"):
        value = base.get(name)
        if value:
            passthrough[name] = value

    return passthrough


# --------------------------------------------------------------
# PCAP inspection
# --------------------------------------------------------------

def count_pcap_packets(path: Path) -> int:
    """
    Count packet records in a pcap file.

    Written out rather than pulled from a dependency because it is the
    check that catches the failure mode this module is most afraid of: a
    capture that produced a well-formed file containing nothing. That is
    what `unshare -rn` leaves behind, and it is indistinguishable from
    success by every other signal.
    """
    data = Path(path).read_bytes()

    if len(data) < PCAP_GLOBAL_HEADER_BYTES:
        raise InvalidPcap(f"{path}: shorter than a pcap header")

    endian = PCAP_MAGICS.get(data[:4])
    if endian is None:
        raise InvalidPcap(f"{path}: not a pcap file")

    offset = PCAP_GLOBAL_HEADER_BYTES
    packets = 0

    while offset + PCAP_RECORD_HEADER_BYTES <= len(data):
        captured = int.from_bytes(
            data[offset + 8: offset + 12], "little" if endian == "<" else "big"
        )
        offset += PCAP_RECORD_HEADER_BYTES + captured
        packets += 1

    return packets


def is_tcpdump_ready(line: str) -> bool:
    return TCPDUMP_READY_MARKER in line


# --------------------------------------------------------------
# Deciding when a load is over
# --------------------------------------------------------------

@dataclass
class QuietResult:
    packets: int
    waited: float
    timed_out: bool = False

    @property
    def empty(self) -> bool:
        return self.packets == 0


def wait_for_quiet(
    count_packets,
    quiet_seconds: float = DEFAULT_QUIET_SECONDS,
    max_wait: float = DEFAULT_MAX_LOAD_SECONDS,
    poll_interval: float = POLL_INTERVAL,
    sleep=time.sleep,
    now=time.monotonic,
) -> QuietResult:
    """
    Wait until the capture stops growing, then return.

    The alternative -- a fixed duration -- is what the original captures
    did, with 3 s of padding at each end. Padding is measurable: it
    lands in every trace of both classes and adds nothing but a constant
    the model can ignore at best and learn at worst.

    Quiet is only declared after at least one packet, so a client that
    never connected is reported as an empty trace instead of being
    mistaken for one that finished quickly.

    `count_packets`, `sleep` and `now` are injected so the decision can
    be tested without a capture.
    """
    started = now()
    last_count = count_packets()
    last_change = now()

    while True:
        sleep(poll_interval)
        current = count_packets()
        moment = now()

        if current != last_count:
            last_count = current
            last_change = moment

        if current > 0 and moment - last_change >= quiet_seconds:
            return QuietResult(packets=current, waited=moment - started)

        if moment - started >= max_wait:
            return QuietResult(
                packets=current, waited=moment - started, timed_out=True
            )


# --------------------------------------------------------------
# Metadata
# --------------------------------------------------------------

def certificate_fingerprint(path: Path) -> str:
    """
    SHA-256 of the certificate's DER form, as openssl prints it.

    Recorded per round so that a round captured after an accidental
    certificate regeneration is identifiable rather than silently mixed
    in with the others. The certificate is transmitted in every
    handshake, so a new one changes the first bytes of every trace.
    """
    der = ssl.PEM_cert_to_DER_cert(Path(path).read_text(encoding="ascii"))
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[index: index + 2] for index in range(0, len(digest), 2))


def tool_version(command: list[str], runner=None) -> str:
    """First line of a tool's version output, or 'unknown'."""
    runner = runner or _run_captured

    try:
        output = runner(command)
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    first = (output or "").strip().splitlines()
    return first[0].strip() if first else "unknown"


def _run_captured(command: list[str]) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=15, check=False
    )
    return result.stdout or result.stderr


def collect_versions(runner=None) -> dict[str, str]:
    return {
        "firefox": tool_version(["firefox", "--version"], runner),
        "wget": tool_version(["wget", "--version"], runner),
        "tcpdump": tool_version(["tcpdump", "--version"], runner),
        "openssl": tool_version(["openssl", "version"], runner),
        "python": sys.version.split()[0],
    }


def build_round_metadata(
    config: CaptureConfig,
    started_at: str,
    finished_at: str,
    result: RoundResult,
    versions: dict[str, str],
    invocations: dict[str, list[str]],
    fingerprint: str | None,
) -> dict:
    """
    The published record of one round.

    Goes in results/, so it carries no BTU content -- only how the round
    was run and how big each trace turned out. The invocations are
    recorded AS RUN rather than as documented: what the README claims
    and what the harness executed are two different facts, and only one
    of them shaped the data.
    """
    return {
        "round": config.round_number,
        "date": config.date,
        "started_at": started_at,
        "finished_at": finished_at,
        "host": config.host,
        "port": config.port,
        "web_root": str(config.web_root),
        "pcap_directory": str(round_directory(config)),
        "snaplen": config.snaplen,
        "tcpdump_filter": f"host {config.host} and port {config.port}",
        "capture_interface": CAPTURE_INTERFACE,
        "quiet_seconds": config.quiet_seconds,
        "max_load_seconds": config.max_load_seconds,
        "network_namespace": "sudo unshare -n (loopback only), verified per round",
        "server_cert_sha256": fingerprint,
        "invocations": {client: list(argv) for client, argv in invocations.items()},
        "versions": versions,
        "limit": config.limit,
        "traces": [trace.to_dict() for trace in result.traces],
        "totals": {
            # Three numbers, not one. The first smoke run recorded
            # "pages": 100 next to 4 traces, because --limit 2 was used
            # and never written down -- and read six months later, that
            # file describes a complete round. A partial round and a
            # full one have to be told apart from the metadata alone,
            # since the PCAPs themselves cannot say which they belong
            # to.
            #
            # `pages_available` is what the mirror holds, `pages_attempted`
            # is counted from the traces actually produced rather than
            # from the limit, so it reports what happened rather than
            # what was intended.
            "pages_available": len(list_pages(config.web_root)),
            "pages_attempted": len({trace.page for trace in result.traces}),
            "limit": config.limit,
            "traces_ok": sum(1 for trace in result.traces if trace.ok),
            "traces_failed": len(result.failures),
            "per_client": result.per_client(),
            "packets": sum(trace.packets for trace in result.traces),
            "bytes": sum(trace.bytes for trace in result.traces),
        },
    }


def write_metadata(config: CaptureConfig, metadata: dict) -> Path:
    path = metadata_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------
# The round
# --------------------------------------------------------------

class CaptureRound:
    """
    Runs one round: server up, then every page under every client.

    The server runs in THIS process rather than as a separate one. Each
    network namespace has its own loopback, so a server started outside
    the namespace is simply unreachable from inside it -- and one server
    for the whole round, rather than one per page, keeps the server's
    own startup out of every trace.
    """

    def __init__(
        self,
        config: CaptureConfig,
        on_event=None,
        require_namespace_marker: bool = True,
    ):
        self.config = config
        self.on_event = on_event or (lambda *_: None)
        self.require_namespace_marker = require_namespace_marker
        self.result = RoundResult()
        self._server = None

    # ----------------------------------------------------------

    def run(self) -> RoundResult:
        from tsd.server import MirrorServer  # imported here: server is optional

        self._check_prerequisites()
        verify_isolation()
        self.on_event("isolated")

        prepare_round_directory(self.config)

        self._server = MirrorServer(
            web_root=self.config.web_root,
            host=self.config.host,
            port=self.config.port,
            certfile=self.config.server_cert,
            keyfile=self.config.server_key,
            quiet=True,
        )
        self._server.start()

        try:
            pages = list_pages(self.config.web_root)
            if self.config.limit:
                pages = pages[: self.config.limit]

            for client in self.config.clients:
                for index, page in enumerate(pages, start=1):
                    self.on_event("page", client, page.name, index, len(pages))
                    self.result.traces.append(self._capture_one(client, page))
        finally:
            self._server.stop()

        return self.result

    # ----------------------------------------------------------

    def _check_prerequisites(self) -> None:
        # The marker only catches the common mistake -- forgetting to
        # re-execute -- and it can be waived for a shell that was
        # unshared by hand, because it is not what actually guarantees
        # anything. verify_isolation() is, and that one runs either way:
        # a flag can lie about the namespace, a failed connection cannot.
        if self.require_namespace_marker and not inside_namespace():
            raise CaptureError(
                "not inside the capture network namespace. "
                "Run scripts/capture_round.py, which re-executes itself under "
                "`sudo unshare -n`, or pass --no-netns if this shell is "
                "already inside one."
            )

        for label, path in (
            ("web root", self.config.web_root),
            ("CA certificate", self.config.ca_cert),
            ("server certificate", self.config.server_cert),
        ):
            if not Path(path).exists():
                raise CaptureError(f"{label} {path} not found")

        if "firefox" in self.config.clients and shutil.which("certutil") is None:
            raise CaptureError(
                "certutil not found, and the CA must be imported into each "
                "fresh Firefox profile.\n"
                "    sudo apt install libnss3-tools\n"
                "Not falling back to an insecure flag: a certificate exception "
                "is a different TLS code path from wget's clean verification, "
                "which is the asymmetry the local CA exists to avoid."
            )

    # ----------------------------------------------------------

    def _capture_one(self, client: str, page: Path) -> TraceResult:
        output = trace_path(self.config, client, page)
        trace = TraceResult(client=client, page=page.name, path=str(output))

        tcpdump = None
        try:
            tcpdump = self._start_tcpdump(output)

            quiet = self._run_client(client, page, output)

            self._stop_tcpdump(tcpdump)
            tcpdump = None

            trace.packets = count_pcap_packets(output)
            trace.bytes = output.stat().st_size

            if trace.packets == 0:
                # The signature of a capture that never ran. Recorded as
                # a failure, never kept as though it were a page load.
                trace.ok = False
                trace.reason = "zero packets captured"
            elif quiet.timed_out:
                trace.ok = False
                trace.reason = f"load never went quiet within {self.config.max_load_seconds}s"

        except CaptureError as error:
            trace.ok = False
            trace.reason = str(error)
        except (OSError, subprocess.SubprocessError) as error:
            trace.ok = False
            trace.reason = f"{type(error).__name__}: {error}"
        finally:
            if tcpdump is not None:
                self._stop_tcpdump(tcpdump)

        if not trace.ok:
            self.on_event("trace_failed", client, page.name, trace.reason)

        return trace

    def _start_tcpdump(self, output: Path):
        process = subprocess.Popen(
            tcpdump_command(
                output, self.config.snaplen, self.config.host, self.config.port
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for "listening on" before the client starts: a client
        # launched too early loses the handshake, which is the most
        # structured part of the trace.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = process.stderr.readline()
            if not line:
                break
            if is_tcpdump_ready(line):
                return process

        process.terminate()
        raise CaptureError("tcpdump did not start listening")

    def _stop_tcpdump(self, process) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _run_client(self, client: str, page: Path, output: Path) -> QuietResult:
        url = page_url(self.config, page)

        if client == "wget":
            return self._run_wget(url, output)
        if client == "firefox":
            return self._run_firefox(url, output)

        raise CaptureError(f"unknown client {client!r}")

    def _run_wget(self, url: str, output: Path) -> QuietResult:
        download_dir = Path(f"/tmp/tsd-wget-{os.getpid()}")
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            process = subprocess.Popen(
                wget_command(url, self.config.ca_cert, download_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            quiet = wait_for_quiet(
                lambda: self._safe_count(output),
                quiet_seconds=self.config.quiet_seconds,
                max_wait=self.config.max_load_seconds,
            )
            self._end_process(process)
            return quiet
        finally:
            shutil.rmtree(download_dir, ignore_errors=True)

    def _run_firefox(self, url: str, output: Path) -> QuietResult:
        profile = Path(f"/tmp/tsd-ff-{os.getpid()}-{int(time.time() * 1000)}")
        # mkdir first: --profile does not create the directory, and a
        # missing one kills Firefox with SIGKILL and no message.
        profile.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                certutil_create_command(profile),
                capture_output=True, check=False, timeout=30,
            )
            imported = subprocess.run(
                certutil_import_command(profile, self.config.ca_cert),
                capture_output=True, text=True, check=False, timeout=30,
            )
            if imported.returncode != 0:
                raise CaptureError(
                    f"could not import the CA into the profile: "
                    f"{imported.stderr.strip()}"
                )

            process = subprocess.Popen(
                firefox_command(profile, url),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={**os.environ, **firefox_environment()},
                start_new_session=True,
            )
            quiet = wait_for_quiet(
                lambda: self._safe_count(output),
                quiet_seconds=self.config.quiet_seconds,
                max_wait=self.config.max_load_seconds,
            )
            self._end_process(process, group=True)
            return quiet
        finally:
            shutil.rmtree(profile, ignore_errors=True)

    def _end_process(self, process, group: bool = False) -> None:
        """Stop a client and confirm it is gone before the next page starts."""
        if process.poll() is not None:
            return

        try:
            if group:
                os.killpg(os.getpgid(process.pid), 15)
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            return

        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                if group:
                    os.killpg(os.getpgid(process.pid), 9)
                else:
                    process.kill()
            except (OSError, ProcessLookupError):
                pass
            process.wait(timeout=5)

    @staticmethod
    def _safe_count(path: Path) -> int:
        try:
            return count_pcap_packets(path)
        except (OSError, InvalidPcap):
            return 0
