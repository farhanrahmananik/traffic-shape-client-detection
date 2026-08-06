"""
server.py
---------
The local HTTPS server that both clients load the mirror from.

This is measurement apparatus, not a web server. Everything it does is
chosen so that the only difference between a Firefox trace and a wget
trace is the client. Anything the server does differently for one of
them -- queuing, timing, revalidation, header size -- is a feature the
model could learn instead.

Five decisions carry that weight:

1.  TLS is negotiated INSIDE the worker thread, never on the listening
    socket. See _accept_forever(); getting this wrong silently
    serialises Firefox.

2.  Keep-alive is explicit and identical for both clients. The earlier
    version advertised HTTP/1.1 -- which implies persistent connections
    -- and then closed after every response without sending
    "Connection: close". Firefox would try to reuse a connection that
    was already gone, fail, and reconnect; wget, going one request at a
    time, would barely notice. Noisy timing on one class only. Here the
    policy is stated in every response.

3.  Response headers are constant-length. No Date (its digits change
    length and vary per request), no Server, no ETag, no Last-Modified.
    Omitting the validators means no conditional-request path exists at
    all, so cache revalidation cannot differ between the clients. It
    also means the response header block is byte-identical across loads
    of the same file.

4.  Headers and body go out in ONE sendall(). Two sends could be
    coalesced by Nagle, or not, depending on timing -- so the packet
    boundaries of an otherwise identical response would differ between
    loads. One buffer removes that source of nondeterminism.

5.  Path resolution is realpath + commonpath, not prefix matching, and
    percent-decoding happens BEFORE the safety check.

Known limitations, deliberate:

*   Range requests are not supported. Every request gets 200 with the
    full body. Both clients are treated identically; neither issues
    Range for a normal page load.
*   Request pipelining is not supported. Anything a client sends after
    the first request's headers, before the response, is discarded.
    Neither client pipelines.
*   Only GET. Anything else is 405.
"""

from __future__ import annotations

import os
import socket
import ssl
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8443
DEFAULT_WEB_ROOT = Path("data/mirror")
DEFAULT_CERTFILE = Path("certs/server.crt")
DEFAULT_KEYFILE = Path("certs/server.key")

# One thread per connection, capped. Past the cap, a connection is
# closed immediately rather than queued: an unbounded queue would turn
# into unbounded latency, which is worse for a measurement than a
# refusal we can see in the log.
MAX_CONNECTIONS = 64

# Firefox opens up to ~6 connections per page load, so the cap is an
# order of magnitude above what a capture needs.
LISTEN_BACKLOG = 64

# An unbounded header read is a one-line denial of service.
MAX_HEADER_BYTES = 16 * 1024
MAX_HEADER_LINES = 100
RECV_BYTES = 4096

# A stalled client must not hold a thread forever.
REQUEST_TIMEOUT = 10.0
KEEP_ALIVE_TIMEOUT = 5.0
MAX_REQUESTS_PER_CONNECTION = 100

HEADER_TERMINATOR = b"\r\n\r\n"

# Extension -> Content-Type.
#
# Rebuilt from what the corpus actually contains rather than copied
# wholesale: the 1701 mirrored assets are jpg/jpeg/png/gif/svg, css, js,
# and four font formats. The rest of the map is the standard set, kept
# so the server stays useful if the corpus is ever regenerated from a
# different site.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".bmp": "image/bmp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".eot": "application/vnd.ms-fontobject",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".zip": "application/zip",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"

STATUS_PHRASES = {
    200: "OK",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
}


class HttpError(Exception):
    """A request that must be answered with an error status."""

    def __init__(self, status: int, extra_headers: dict[str, str] | None = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.extra_headers = extra_headers or {}


class ClientDisconnected(Exception):
    """The peer closed the connection. Not an error."""


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def wants_close(self) -> bool:
        """
        Whether this connection should close after the response.

        Stated explicitly rather than inferred: the ambiguity between
        HTTP/1.1's implied persistence and an unannounced close is what
        made the earlier server noisy for Firefox and quiet for wget.
        """
        connection = self.headers.get("connection", "").lower()

        if "close" in connection:
            return True

        if self.version == "HTTP/1.0":
            return "keep-alive" not in connection

        return False


# --------------------------------------------------------------
# Request reading and parsing
# --------------------------------------------------------------

def read_headers(sock) -> bytes:
    """
    Read until the end of the header block.

    A single recv() is not enough. Firefox's request headers run well
    past one segment, and TCP has no message boundaries -- what arrives
    in one recv() is a property of the network, not of the request. So
    the loop runs until the terminator appears or a ceiling is hit.
    """
    buffer = bytearray()

    while True:
        chunk = sock.recv(RECV_BYTES)

        if not chunk:
            raise ClientDisconnected()

        buffer += chunk

        if HEADER_TERMINATOR in buffer:
            return bytes(buffer)

        if len(buffer) > MAX_HEADER_BYTES:
            raise HttpError(431)


def parse_request(raw: bytes) -> Request:
    """Parse a header block into a Request. Anything malformed is a 400."""
    head = raw.split(HEADER_TERMINATOR, 1)[0]

    if len(head) > MAX_HEADER_BYTES:
        raise HttpError(431)

    # latin-1 never fails, and a header that is not ASCII is rejected
    # below by the request-line check rather than by a decode error.
    lines = head.decode("latin-1").split("\r\n")

    if len(lines) > MAX_HEADER_LINES:
        raise HttpError(431)

    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise HttpError(400)

    method, target, version = parts
    if version not in ("HTTP/1.0", "HTTP/1.1"):
        raise HttpError(400)

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise HttpError(400)
        headers[name.strip().lower()] = value.strip()

    return Request(method=method, target=target, version=version, headers=headers)


# --------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------

def resolve_target(web_root: Path, target: str) -> Path:
    """
    Turn a request target into a file inside the web root, or raise.

    The ordering here is the security property, not the individual
    checks:

    1.  Strip query and fragment.
    2.  Require an absolute path. `GET http://elsewhere/x` and
        `GET x` are both refused rather than normalised into something.
    3.  Percent-DECODE, then check. Checking first would let `%2e%2e%2f`
        walk straight past a `..` test and decode into a traversal
        afterwards.
    4.  realpath() the joined path, so a symlink inside the mirror
        cannot point out of it.
    5.  Compare with commonpath(), never str.startswith(). A prefix test
        accepts `/data/mirror-evil` for a root of `/data/mirror`.
    """
    path = target.split("?", 1)[0].split("#", 1)[0]

    if not path.startswith("/"):
        raise HttpError(400)

    decoded = unquote(path)

    if "\x00" in decoded:
        raise HttpError(400)

    root = os.path.realpath(web_root)
    candidate = os.path.realpath(os.path.join(root, decoded.lstrip("/")))

    _assert_inside(root, candidate)

    if os.path.isdir(candidate):
        # A directory is served as its index.html or not at all. A
        # listing would be a page that exists in the mirror but not on
        # b-tu.de -- content invented by the capture rig.
        candidate = os.path.realpath(os.path.join(candidate, "index.html"))
        _assert_inside(root, candidate)

    if not os.path.exists(candidate):
        raise HttpError(404)

    # Not just "exists": a fifo or a device node under the web root
    # would block the thread or return something that is not the file.
    if not os.path.isfile(candidate):
        raise HttpError(403)

    return Path(candidate)


def _assert_inside(root: str, candidate: str) -> None:
    try:
        inside = os.path.commonpath([root, candidate]) == root
    except ValueError:  # different drives, or a mix of abs and rel
        inside = False

    if not inside:
        raise HttpError(403)


def content_type_for(path: Path) -> str:
    """Content-Type from the extension, or octet-stream if unrecognised."""
    return CONTENT_TYPES.get(path.suffix.lower(), DEFAULT_CONTENT_TYPE)


# --------------------------------------------------------------
# Responses
# --------------------------------------------------------------

def build_response(
    status: int,
    body: bytes,
    content_type: str = "text/plain; charset=utf-8",
    close: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    """
    Build one complete response, headers and body in a single buffer.

    Deliberately absent: Date, Server, ETag, Last-Modified. Date changes
    length as its digits change; the validators would create a
    conditional-request path that the two clients could take
    differently. Cache-Control: no-store is constant and keeps
    within-load caching out of the measurement.
    """
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
        "Cache-Control": "no-store",
        "Connection": "close" if close else "keep-alive",
    }
    headers.update(extra_headers or {})

    head = f"HTTP/1.1 {status} {STATUS_PHRASES.get(status, 'Error')}\r\n"
    head += "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    head += "\r\n"

    return head.encode("latin-1") + body


def error_response(status: int, close: bool, extra_headers=None) -> bytes:
    """Short, fixed error bodies: no request data is echoed back."""
    body = f"{status} {STATUS_PHRASES.get(status, 'Error')}\n".encode("ascii")
    return build_response(
        status, body, close=close, extra_headers=extra_headers
    )


# --------------------------------------------------------------
# The server
# --------------------------------------------------------------

class MirrorServer:
    """Threaded HTTPS server for one directory of static files."""

    def __init__(
        self,
        web_root: str | Path = DEFAULT_WEB_ROOT,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        certfile: str | Path = DEFAULT_CERTFILE,
        keyfile: str | Path = DEFAULT_KEYFILE,
        quiet: bool = False,
        max_connections: int = MAX_CONNECTIONS,
    ):
        self.web_root = Path(web_root)
        self.host = host
        self.port = port
        self.certfile = Path(certfile)
        self.keyfile = Path(keyfile)
        self.quiet = quiet
        self.max_connections = max_connections

        self._socket: socket.socket | None = None
        self._context: ssl.SSLContext | None = None
        self._accept_thread: threading.Thread | None = None
        self._running = False
        self._active = 0
        self._lock = threading.Lock()

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def start(self) -> int:
        """Bind, start accepting, and return the port actually bound."""
        self._check_paths()
        self._context = self._build_ssl_context()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(LISTEN_BACKLOG)
        self.port = self._socket.getsockname()[1]

        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_forever, daemon=True
        )
        self._accept_thread.start()

        self.log(f"serving {self.web_root} on https://{self.host}:{self.port}",
                 always=True)
        return self.port

    def serve_forever(self) -> None:
        """Start and block until interrupted."""
        self.start()
        try:
            while self._running:
                self._accept_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self.log("shutting down", always=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop accepting. In-flight connections finish on their own."""
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def __enter__(self) -> "MirrorServer":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    def _check_paths(self) -> None:
        if not self.web_root.is_dir():
            raise FileNotFoundError(
                f"web root {self.web_root} does not exist. "
                f"Run scripts/scrape_corpus.py first."
            )

        for label, path in (("certificate", self.certfile), ("key", self.keyfile)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"TLS {label} {path} not found. Generate it with:\n"
                    f"    scripts/make_cert.sh"
                )

    def _build_ssl_context(self) -> ssl.SSLContext:
        """
        Server-side TLS context.

        No version pin. Measured on 2026-08-06 against this certificate:
        Firefox 153.0.3 and GNU Wget 1.21.4 both negotiate TLS 1.3
        unprompted. Forcing a version would constrain the clients
        without changing what they do -- and a pin that silently stops
        matching a future client is worse than no pin.

        Session tickets are left ENABLED. Resumption is controlled at
        capture time instead, by giving Firefox a fresh profile per page
        load -- which is required anyway, to keep the HTTP cache out of
        the measurement. Disabling tickets here would make the server
        behave unlike any real server for no gain, and would hide a
        resumption bug in the capture harness rather than prevent one.
        """
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(self.certfile), keyfile=str(self.keyfile))
        return context

    # ----------------------------------------------------------
    # Accept loop
    # ----------------------------------------------------------

    def _accept_forever(self) -> None:
        """
        Accept plain TCP connections and hand each to its own thread.

        The listening socket is NOT wrapped in TLS. This is the single
        most important line in the module, and getting it wrong fails
        silently.

        If the listener were wrapped -- ssl_context.wrap_socket(listener)
        -- then accept() itself would perform the TLS handshake, on this
        thread, before returning. Every handshake would queue behind the
        one before it. The server would still be "threaded", the code
        would still look concurrent, and Firefox's ~6 parallel
        connections would still be serialised: each one waiting for the
        previous handshake to finish. wget, which opens one connection
        at a time, would be unaffected.

        The result would be a model that has learned the server's
        queuing behaviour and calls it "browser". So accept() returns a
        raw socket, and the handshake happens inside the worker thread.
        """
        while self._running:
            try:
                conn, address = self._socket.accept()
            except OSError:
                # The listening socket was closed by stop().
                break

            with self._lock:
                at_capacity = self._active >= self.max_connections
                if not at_capacity:
                    self._active += 1

            if at_capacity:
                self.log(f"refused {address[0]}:{address[1]} (at capacity)")
                conn.close()
                continue

            threading.Thread(
                target=self._run_connection,
                args=(conn, address),
                daemon=True,
            ).start()

    def _run_connection(self, conn: socket.socket, address) -> None:
        try:
            self._handle_connection(conn, address)
        finally:
            with self._lock:
                self._active -= 1

    def _handle_connection(self, conn: socket.socket, address) -> None:
        """
        Wrap in TLS, then serve requests until the connection ends.

        Every failure path here is contained: one client's bad handshake,
        reset or timeout must not disturb another connection, and must
        never stop the accept loop.
        """
        peer = f"{address[0]}:{address[1]}"
        conn.settimeout(REQUEST_TIMEOUT)

        try:
            tls_conn = self._context.wrap_socket(conn, server_side=True)
        except (ssl.SSLError, OSError) as error:
            self.log(f"handshake failed {peer}: {type(error).__name__}")
            conn.close()
            return

        self.log(f"open  {peer} {tls_conn.version()}")

        try:
            self._serve_requests(tls_conn, peer)
        except (ssl.SSLError, OSError, ClientDisconnected):
            pass
        except Exception as error:  # never let one connection kill the server
            self.log(f"error {peer}: {type(error).__name__}: {error}")
        finally:
            try:
                tls_conn.close()
            except OSError:
                pass
            self.log(f"close {peer}")

    def _serve_requests(self, conn, peer: str) -> None:
        for served in range(MAX_REQUESTS_PER_CONNECTION):
            conn.settimeout(KEEP_ALIVE_TIMEOUT if served else REQUEST_TIMEOUT)

            try:
                raw = read_headers(conn)
            except ClientDisconnected:
                return
            except socket.timeout:
                return
            except HttpError as error:
                conn.sendall(error_response(error.status, close=True,
                                            extra_headers=error.extra_headers))
                return

            last = served == MAX_REQUESTS_PER_CONNECTION - 1

            try:
                request = parse_request(raw)
            except HttpError as error:
                conn.sendall(error_response(error.status, close=True,
                                            extra_headers=error.extra_headers))
                return

            close = request.wants_close or last
            response, status = self._respond(request, close)

            conn.sendall(response)
            self.log(f"  {peer} {request.method} {request.target} -> {status}")

            if close:
                return

    def _respond(self, request: Request, close: bool) -> tuple[bytes, int]:
        """Build the response for one parsed request."""
        if request.method != "GET":
            # Only GET is served. A method that quietly worked would be
            # a place the two clients could diverge without anyone
            # noticing which one took it.
            return (
                error_response(405, close=True, extra_headers={"Allow": "GET"}),
                405,
            )

        try:
            path = resolve_target(self.web_root, request.target)
        except HttpError as error:
            return error_response(error.status, close=close,
                                  extra_headers=error.extra_headers), error.status

        try:
            body = path.read_bytes()
        except OSError:
            return error_response(500, close=close), 500

        return (
            build_response(
                200, body, content_type=content_type_for(path), close=close
            ),
            200,
        )

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    def log(self, message: str, always: bool = False) -> None:
        """
        Connection-level logging, to stderr only.

        Never to a file: a disk write per request is server-side timing
        noise recorded straight into the trace it is describing.
        """
        if self.quiet and not always:
            return
        print(message, file=sys.stderr, flush=True)
