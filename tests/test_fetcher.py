"""
test_fetcher.py
---------------
Tests for tsd.fetcher.

No network. requests.Session.get is replaced with a fake, so these
tests assert what the fetcher DOES rather than what BTU happens to
return today. The point is that refusals happen before any request
is issued -- so the fake also records whether it was called at all.
"""

import time

import pytest
import requests

from tsd.fetcher import (
    MAX_RESPONSE_BYTES,
    FetchBlocked,
    FetchFailed,
    PoliteFetcher,
)
from tsd.robots import RobotsPolicy


BASE = "https://www.b-tu.de/"

ROBOTS_SAMPLE = """
User-agent: *
Disallow: /media/
Disallow: /*/wiki/
"""


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, body=b"hello", status_code=200, url=None, headers=None):
        self._body = body
        self.status_code = status_code
        self.url = url or BASE
        self.headers = headers or {"Content-Type": "text/html"}
        self.closed = False

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self):
        self.closed = True


class FakeTransport:
    """Records every call so tests can assert nothing was sent."""

    def __init__(self, response=None):
        self.calls = []
        self.response = response or FakeResponse()

    def __call__(self, url, **kwargs):
        self.calls.append(url)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def policy():
    return RobotsPolicy.from_text(BASE, ROBOTS_SAMPLE)


@pytest.fixture
def fetcher(policy):
    f = PoliteFetcher(policy=policy)
    f.policy.crawl_delay = 0.0  # keep tests fast; timing tested separately
    return f


def install(fetcher, transport):
    fetcher._session.get = transport
    return transport


# --------------------------------------------------------------
# Refusals happen before the request is issued
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "url, outcome",
    [
        ("https://www.b-tu.de/media/logo.png", "blocked_robots"),
        ("https://www.b-tu.de/fg-x/wiki/page", "blocked_robots"),
        ("https://evil.example.com/x", "blocked_host"),
        ("ftp://www.b-tu.de/x", "blocked_host"),
    ],
)
def test_refused_urls_never_reach_the_network(fetcher, url, outcome):
    transport = install(fetcher, FakeTransport())

    with pytest.raises(FetchBlocked) as excinfo:
        fetcher.get(url)

    assert excinfo.value.record.outcome == outcome
    assert transport.calls == [], "a refused URL was still requested"


# --------------------------------------------------------------
# Successful fetch
# --------------------------------------------------------------

def test_successful_fetch_returns_body_and_logs_it(fetcher):
    install(fetcher, FakeTransport(FakeResponse(b"<html>ok</html>")))

    response = fetcher.get("https://www.b-tu.de/fakultaet1/")

    assert response.body == b"<html>ok</html>"
    record = fetcher.log[-1]
    assert record.outcome == "ok"
    assert record.bytes_received == 15
    assert record.status_code == 200
    assert record.fetched_at.endswith("+00:00")


def test_response_is_closed_after_reading(fetcher):
    fake = FakeResponse(b"x")
    install(fetcher, FakeTransport(fake))

    fetcher.get("https://www.b-tu.de/page")

    assert fake.closed


# --------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------

def test_non_200_is_a_failure(fetcher):
    install(fetcher, FakeTransport(FakeResponse(b"", status_code=404)))

    with pytest.raises(FetchFailed) as excinfo:
        fetcher.get("https://www.b-tu.de/missing")

    assert excinfo.value.record.outcome == "http_error"
    assert excinfo.value.record.status_code == 404


def test_redirect_off_host_is_blocked(fetcher):
    """The first URL was allowed; the destination was not."""
    install(
        fetcher,
        FakeTransport(FakeResponse(b"x", url="https://youtube.com/watch")),
    )

    with pytest.raises(FetchBlocked) as excinfo:
        fetcher.get("https://www.b-tu.de/link")

    assert excinfo.value.record.outcome == "blocked_host"
    assert "youtube.com" in excinfo.value.record.reason


def test_oversized_body_is_abandoned(fetcher):
    oversized = b"x" * (MAX_RESPONSE_BYTES + 1)
    install(fetcher, FakeTransport(FakeResponse(oversized)))

    with pytest.raises(FetchFailed) as excinfo:
        fetcher.get("https://www.b-tu.de/huge.bin")

    assert excinfo.value.record.outcome == "too_large"


def test_transport_exception_is_recorded(fetcher):
    install(fetcher, FakeTransport(requests.ConnectionError("boom")))

    with pytest.raises(FetchFailed) as excinfo:
        fetcher.get("https://www.b-tu.de/page")

    assert excinfo.value.record.outcome == "error"
    assert excinfo.value.record.reason == "ConnectionError"


# --------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------

def test_first_request_is_not_delayed(policy):
    policy.crawl_delay = 0.3
    f = PoliteFetcher(policy=policy)
    install(f, FakeTransport())

    started = time.monotonic()
    f.get("https://www.b-tu.de/one")

    assert time.monotonic() - started < 0.2


def test_second_request_waits_for_the_delay(policy):
    policy.crawl_delay = 0.3
    f = PoliteFetcher(policy=policy)
    install(f, FakeTransport())

    f.get("https://www.b-tu.de/one")
    started = time.monotonic()
    f.get("https://www.b-tu.de/two")

    assert time.monotonic() - started >= 0.25


# --------------------------------------------------------------
# Provenance
# --------------------------------------------------------------

def test_every_attempt_is_logged(fetcher):
    install(fetcher, FakeTransport())

    fetcher.get("https://www.b-tu.de/ok")
    with pytest.raises(FetchBlocked):
        fetcher.get("https://www.b-tu.de/media/x.png")

    assert [r.outcome for r in fetcher.log] == ["ok", "blocked_robots"]
    assert all("url" in r.to_dict() for r in fetcher.log)
