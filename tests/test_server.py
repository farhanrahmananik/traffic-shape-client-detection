"""
test_server.py
--------------
Tests for tsd.server.

Two kinds live here. Most are unit tests on the pure functions -- header
reading, parsing, path resolution, response building -- and need no
socket at all. A few start the real server on an ephemeral loopback port
and talk TLS to it, because the two properties that matter most cannot
be tested any other way:

    - that a page load actually works end to end
    - that connections are served in PARALLEL

The second one is the reason the module exists in this shape. A server
that wrapped its listening socket would still pass every unit test here
and would still look threaded, while serialising Firefox's parallel
connections behind one another's TLS handshakes. Only a test that holds
several connections open at once can tell the difference.

Certificates: the project's real certs/ are used when present, so this
also checks that what make_cert.sh produces actually works. On a fresh
clone they do not exist, so a throwaway certificate is generated into a
temporary directory instead. Skipping outright was the other option and
was rejected -- a silently skipped test on the load-bearing property is
the same failure mode as a warning that always fires.
"""

from __future__ import annotations

import os
import socket
import ssl
import subprocess
import threading
from pathlib import Path

import pytest

from tsd.server import (
    CONTENT_TYPES,
    DEFAULT_CONTENT_TYPE,
    MAX_HEADER_BYTES,
    ClientDisconnected,
    HttpError,
    MirrorServer,
    build_response,
    content_type_for,
    parse_request,
    read_headers,
    resolve_target,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CERT = REPO_ROOT / "certs" / "server.crt"
REAL_KEY = REPO_ROOT / "certs" / "server.key"


# --------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------

@pytest.fixture
def web_root(tmp_path):
    """A small mirror: an index, a page, an asset, and a subdirectory."""
    (tmp_path / "index.html").write_bytes(b"<html>home</html>")
    (tmp_path / "page_abc123.html").write_bytes(b"<html>page</html>")

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "style.css").write_bytes(b"body{color:red}")
    (assets / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (assets / "truncated.pn").write_bytes(b"\x89PNG")

    section = tmp_path / "section"
    section.mkdir()
    (section / "index.html").write_bytes(b"<html>section</html>")

    (tmp_path / "empty").mkdir()

    return tmp_path


class FakeSocket:
    """Hands out prepared chunks, one per recv(), like a real socket."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = list(chunks)
        self.sent = bytearray()

    def recv(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def sendall(self, data: bytes) -> None:
        self.sent += data


@pytest.fixture(scope="session")
def certificate(tmp_path_factory) -> tuple[Path, Path]:
    """
    The project's certificate if it exists, else a throwaway one.

    Using the real one means these tests also confirm that what
    make_cert.sh produced is loadable and negotiable. Falling back keeps
    a fresh clone runnable without running the operational script first.
    """
    if REAL_CERT.is_file() and REAL_KEY.is_file():
        return REAL_CERT, REAL_KEY

    openssl = subprocess.run(
        ["openssl", "version"], capture_output=True, check=False
    )
    if openssl.returncode != 0:
        pytest.skip("no certificate and no openssl to make one")

    directory = tmp_path_factory.mktemp("certs")
    key = directory / "test.key"
    crt = directory / "test.crt"

    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "ec",
            "-pkeyopt", "ec_paramgen_curve:P-256",
            "-nodes", "-days", "1", "-sha256",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
            "-keyout", str(key), "-out", str(crt),
        ],
        capture_output=True,
        check=True,
    )
    return crt, key


@pytest.fixture
def live_server(web_root, certificate):
    """A running server on an ephemeral loopback port."""
    certfile, keyfile = certificate
    server = MirrorServer(
        web_root=web_root,
        host="127.0.0.1",
        port=0,
        certfile=certfile,
        keyfile=keyfile,
        quiet=True,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def client_context() -> ssl.SSLContext:
    """A client that does not verify: these tests measure plumbing, not trust."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def fetch(server: MirrorServer, target: str, keep_open: bool = False):
    """One request over a fresh TLS connection. Returns (head, body)."""
    raw = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    conn = client_context().wrap_socket(raw, server_hostname="127.0.0.1")
    try:
        conn.sendall(
            f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("latin-1")
        )
        return read_full_response(conn)
    finally:
        if not keep_open:
            conn.close()


def read_full_response(conn) -> tuple[str, bytes]:
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buffer += chunk

    head, _, rest = bytes(buffer).partition(b"\r\n\r\n")
    head_text = head.decode("latin-1")

    length = 0
    for line in head_text.split("\r\n")[1:]:
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length":
            length = int(value.strip())

    body = bytearray(rest)
    while len(body) < length:
        chunk = conn.recv(4096)
        if not chunk:
            break
        body += chunk

    return head_text, bytes(body)


# --------------------------------------------------------------
# Path traversal -- the security surface
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "target",
    [
        "/../etc/passwd",
        "/assets/../../etc/passwd",
        "/%2e%2e%2fetc%2fpasswd",          # encoded ../ -- decode must come first
        "/%2e%2e/%2e%2e/etc/passwd",
        "/assets/%2e%2e%2f%2e%2e%2fetc/passwd",
        "/....//....//etc/passwd",
    ],
)
def test_traversal_is_refused(web_root, target):
    """
    A traversal that escaped the root would serve a file from the host
    into the capture, and the mirror would no longer be the thing the
    published manifest describes.
    """
    with pytest.raises(HttpError) as raised:
        resolve_target(web_root, target)

    assert raised.value.status in (403, 404)


def test_percent_encoded_traversal_is_decoded_before_the_check(web_root, tmp_path):
    """
    Checking before decoding is the classic ordering bug: "%2e%2e%2f"
    contains no ".." to find, and becomes "../" immediately afterwards.
    """
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    with pytest.raises(HttpError):
        resolve_target(web_root, f"/%2e%2e/{outside.name}")


def test_target_must_be_an_absolute_path(web_root):
    """`GET http://elsewhere/x` is valid HTTP; it must not be normalised."""
    for target in ("http://example.org/x", "index.html", "*"):
        with pytest.raises(HttpError) as raised:
            resolve_target(web_root, target)
        assert raised.value.status == 400


def test_nul_byte_is_refused(web_root):
    """A NUL truncates the path in any C-level call it reaches."""
    with pytest.raises(HttpError) as raised:
        resolve_target(web_root, "/index.html%00.png")

    assert raised.value.status in (400, 404)


def test_sibling_directory_with_the_same_prefix_is_refused(tmp_path):
    """
    The reason the check is commonpath() and not startswith(): the
    string "/data/mirror-evil" starts with "/data/mirror".
    """
    root = tmp_path / "mirror"
    root.mkdir()
    (root / "index.html").write_bytes(b"ok")

    evil = tmp_path / "mirror-evil"
    evil.mkdir()
    (evil / "loot.txt").write_bytes(b"loot")

    with pytest.raises(HttpError) as raised:
        resolve_target(root, "/../mirror-evil/loot.txt")

    assert raised.value.status == 403


def test_symlink_pointing_outside_the_root_is_refused(web_root, tmp_path):
    """
    realpath() rather than a lexical check: the mirror is written by a
    script, and a symlink inside it must not become a way out of it.
    """
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")

    link = web_root / "escape.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")

    with pytest.raises(HttpError) as raised:
        resolve_target(web_root, "/escape.txt")

    assert raised.value.status == 403


def test_symlink_staying_inside_the_root_is_allowed(web_root):
    """The check is "where does it land", not "is it a symlink"."""
    link = web_root / "alias.html"
    try:
        os.symlink(web_root / "index.html", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")

    assert resolve_target(web_root, "/alias.html").name == "index.html"


def test_non_regular_file_is_refused(web_root):
    """A fifo under the web root would block the thread that opened it."""
    fifo = web_root / "pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("mkfifo not available here")

    with pytest.raises(HttpError) as raised:
        resolve_target(web_root, "/pipe")

    assert raised.value.status == 403


# --------------------------------------------------------------
# Directories
# --------------------------------------------------------------

def test_directory_resolves_to_index_html(web_root):
    assert resolve_target(web_root, "/").name == "index.html"
    assert resolve_target(web_root, "/section/").parent.name == "section"


def test_directory_without_index_is_404_not_a_listing(web_root):
    """
    A generated listing is a page that exists in the mirror but not on
    b-tu.de -- content invented by the capture rig, loaded by both
    clients, in the dataset.
    """
    with pytest.raises(HttpError) as raised:
        resolve_target(web_root, "/empty/")

    assert raised.value.status == 404


def test_query_and_fragment_are_stripped(web_root):
    assert resolve_target(web_root, "/index.html?v=2").name == "index.html"
    assert resolve_target(web_root, "/index.html#top").name == "index.html"


# --------------------------------------------------------------
# Header reading
# --------------------------------------------------------------

def test_headers_split_across_recv_calls_are_assembled():
    """
    One recv() is not a request. TCP gives no message boundaries, and
    Firefox's headers are large enough that this is the normal case, not
    the edge case.
    """
    request = (
        b"GET /index.html HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"User-Agent: Mozilla/5.0\r\n"
        b"\r\n"
    )
    chunks = [request[i:i + 7] for i in range(0, len(request), 7)]
    assert len(chunks) > 3

    raw = read_headers(FakeSocket(chunks))

    assert parse_request(raw).target == "/index.html"


def test_closed_connection_before_any_data_is_not_an_error():
    """A client that opens and goes away is normal, not a failure to log."""
    with pytest.raises(ClientDisconnected):
        read_headers(FakeSocket([]))


def test_oversized_headers_are_refused_with_431():
    """An unbounded header read is a one-line denial of service."""
    flood = [b"X-Pad: " + b"a" * 1024 + b"\r\n"] * (MAX_HEADER_BYTES // 1024 + 4)

    with pytest.raises(HttpError) as raised:
        read_headers(FakeSocket([b"GET / HTTP/1.1\r\n"] + flood))

    assert raised.value.status == 431


def test_too_many_header_lines_are_refused():
    lines = b"".join(b"X-N-%d: v\r\n" % n for n in range(300))

    with pytest.raises(HttpError) as raised:
        parse_request(b"GET / HTTP/1.1\r\n" + lines + b"\r\n")

    assert raised.value.status == 431


@pytest.mark.parametrize(
    "raw",
    [
        b"GET\r\n\r\n",
        b"GET /\r\n\r\n",
        b"GET / HTTP/9.9\r\n\r\n",
        b"GET / HTTP/1.1\r\nBroken\r\n\r\n",
    ],
)
def test_malformed_requests_are_400(raw):
    with pytest.raises(HttpError) as raised:
        parse_request(raw)

    assert raised.value.status in (400, 431)


def test_connection_close_is_honoured_and_keep_alive_is_the_default():
    """
    The old server advertised HTTP/1.1, implying persistence, then closed
    anyway without saying so. Firefox would try to reuse the connection,
    fail and reconnect; wget would not notice. Noisy timing on one class.
    """
    keep = parse_request(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    assert keep.wants_close is False

    close = parse_request(b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n")
    assert close.wants_close is True

    old = parse_request(b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
    assert old.wants_close is True


# --------------------------------------------------------------
# Responses
# --------------------------------------------------------------

def test_response_headers_are_constant_length_and_explicit():
    """
    No Date (its digits change length), no Server, and no validators --
    without ETag or Last-Modified there is no conditional-request path
    for the two clients to take differently.
    """
    response = build_response(200, b"hello", content_type="text/plain")
    head = response.split(b"\r\n\r\n", 1)[0].decode()

    assert "Date:" not in head
    assert "Server:" not in head
    assert "ETag:" not in head
    assert "Last-Modified:" not in head
    assert "Cache-Control: no-store" in head
    assert "Connection: keep-alive" in head
    assert "Content-Length: 5" in head


def test_response_is_one_buffer():
    """
    Headers and body in a single sendall: two writes could be coalesced
    by Nagle or not, so identical responses would produce different
    packet boundaries between loads.
    """
    response = build_response(200, b"body-bytes")
    assert response.endswith(b"\r\n\r\nbody-bytes")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("index.html", CONTENT_TYPES[".html"]),
        ("style.css", CONTENT_TYPES[".css"]),
        ("app.js", CONTENT_TYPES[".js"]),
        ("photo.JPG", CONTENT_TYPES[".jpg"]),
        ("logo.png", CONTENT_TYPES[".png"]),
        ("font.woff2", CONTENT_TYPES[".woff2"]),
        ("truncated.pn", DEFAULT_CONTENT_TYPE),
        ("noextension", DEFAULT_CONTENT_TYPE),
    ],
)
def test_content_type_mapping(name, expected):
    """
    The truncated case is real: urls.py caps asset filenames at 60
    characters, which clips the extension off a handful of long names in
    the corpus. Those are served as octet-stream. Both clients receive
    the same header, so it is symmetric -- but it is worth a test that
    says so out loud rather than a surprise later.
    """
    assert content_type_for(Path(name)) == expected


# --------------------------------------------------------------
# End to end, over real TLS on loopback
# --------------------------------------------------------------

def test_serves_a_file_over_tls(live_server):
    head, body = fetch(live_server, "/index.html")

    assert head.startswith("HTTP/1.1 200 OK")
    assert CONTENT_TYPES[".html"] in head
    assert body == b"<html>home</html>"


def test_root_serves_the_index(live_server):
    head, body = fetch(live_server, "/")

    assert head.startswith("HTTP/1.1 200 OK")
    assert body == b"<html>home</html>"


def test_missing_file_is_404_over_tls(live_server):
    head, _ = fetch(live_server, "/nope.html")
    assert head.startswith("HTTP/1.1 404")


def test_traversal_over_tls_is_refused(live_server):
    head, _ = fetch(live_server, "/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
    assert head.split(" ")[1] in ("403", "404")


def test_non_get_is_405_with_allow(live_server):
    """
    Both clients send GET. A different method that quietly worked would
    be a path where the two classes could diverge unnoticed.
    """
    raw = socket.create_connection(("127.0.0.1", live_server.port), timeout=5)
    conn = client_context().wrap_socket(raw, server_hostname="127.0.0.1")
    try:
        conn.sendall(b"POST /index.html HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
        head, _ = read_full_response(conn)
    finally:
        conn.close()

    assert head.startswith("HTTP/1.1 405")
    assert "Allow: GET" in head


def test_keep_alive_serves_several_requests_on_one_connection(live_server):
    """Persistence is on and announced, so a reusing client is not surprised."""
    raw = socket.create_connection(("127.0.0.1", live_server.port), timeout=5)
    conn = client_context().wrap_socket(raw, server_hostname="127.0.0.1")
    try:
        for _ in range(3):
            conn.sendall(b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n")
            head, body = read_full_response(conn)
            assert head.startswith("HTTP/1.1 200")
            assert "Connection: keep-alive" in head
            assert body == b"<html>home</html>"

        conn.sendall(b"GET /index.html HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        head, _ = read_full_response(conn)
        assert "Connection: close" in head
    finally:
        conn.close()


def test_connections_are_served_in_parallel(live_server):
    """
    The point of the whole module.

    Eight clients connect, complete their TLS handshakes, fetch a file,
    and then all wait on a barrier before any of them closes. That can
    only be reached if eight connections are alive in the server at the
    same time.

    A server that wrapped its listening socket would perform each
    handshake inside accept(), on one thread, so client 2 could not
    finish its handshake until client 1's connection was over -- and
    client 1 is waiting at the barrier for client 2. The barrier would
    time out and this test would fail, which is exactly what it is for.
    """
    clients = 8
    barrier = threading.Barrier(clients, timeout=10)
    errors: list[BaseException] = []
    bodies: list[bytes] = []
    lock = threading.Lock()

    def one_client():
        try:
            raw = socket.create_connection(("127.0.0.1", live_server.port), timeout=10)
            conn = client_context().wrap_socket(raw, server_hostname="127.0.0.1")
            try:
                conn.sendall(b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n")
                _, body = read_full_response(conn)
                with lock:
                    bodies.append(body)
                # Hold the connection open until every other client has
                # also been served.
                barrier.wait()
            finally:
                conn.close()
        except BaseException as error:  # noqa: BLE001 - reported below
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=one_client) for _ in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors, f"parallel clients failed: {errors[:3]}"
    assert bodies == [b"<html>home</html>"] * clients


def test_connections_past_the_cap_are_refused_not_queued(web_root, certificate):
    """
    An unbounded accept queue turns into unbounded latency, recorded
    into the trace as if it were the client's behaviour. Refusing is
    visible; queuing is not.
    """
    certfile, keyfile = certificate
    server = MirrorServer(
        web_root=web_root, host="127.0.0.1", port=0,
        certfile=certfile, keyfile=keyfile, quiet=True, max_connections=1,
    )
    server.start()

    held = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    conn = client_context().wrap_socket(held, server_hostname="127.0.0.1")

    try:
        conn.sendall(b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n")
        head, _ = read_full_response(conn)
        assert head.startswith("HTTP/1.1 200")

        # The first connection is still open and holding the only slot.
        second = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        try:
            with pytest.raises((ssl.SSLError, OSError)):
                client_context().wrap_socket(second, server_hostname="127.0.0.1")
        finally:
            second.close()
    finally:
        conn.close()
        server.stop()


def test_missing_certificate_points_at_make_cert(web_root, tmp_path):
    """A missing cert must name the script that makes one, not raise ssl noise."""
    server = MirrorServer(
        web_root=web_root,
        host="127.0.0.1",
        port=0,
        certfile=tmp_path / "absent.crt",
        keyfile=tmp_path / "absent.key",
        quiet=True,
    )

    with pytest.raises(FileNotFoundError) as raised:
        server.start()

    assert "make_cert.sh" in str(raised.value)


def test_missing_web_root_is_refused(tmp_path, certificate):
    certfile, keyfile = certificate
    server = MirrorServer(
        web_root=tmp_path / "nothing-here",
        host="127.0.0.1",
        port=0,
        certfile=certfile,
        keyfile=keyfile,
        quiet=True,
    )

    with pytest.raises(FileNotFoundError) as raised:
        server.start()

    assert "scrape_corpus" in str(raised.value)
