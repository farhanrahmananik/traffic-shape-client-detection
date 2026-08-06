"""
fetcher.py
----------
The single chokepoint for every outbound request to b-tu.de.

Design rationale:

    Politeness is easy to state and easy to break. If robots checking
    and rate limiting live at the call sites, then every new call site
    is a chance to forget one, and forgetting is silent -- the crawl
    just runs slightly less politely than the README claims.

    So there is exactly one way out of this program: PoliteFetcher.get().
    It enforces, in order:

        1. same-host only          (never wander off b-tu.de)
        2. robots.txt policy       (prefix + wildcard layers)
        3. minimum interval        (measured from the END of the last
                                    response, not the start)
        4. response size ceiling   (streamed, so a huge file is
                                    abandoned mid-download)

    Every fetch -- allowed, blocked or failed -- is recorded. That log
    becomes results/corpus_manifest.json, which is what we publish in
    place of the mirror itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

import requests

from .robots import RobotsPolicy, USER_AGENT

# Read timeout is generous: a slow BTU response is not a reason to
# hammer it with a retry.
REQUEST_TIMEOUT = (10, 30)  # (connect, read)

# No single resource in this corpus should be larger than this. Guards
# against accidentally pulling a video or a large PDF into the mirror.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Streaming chunk size for the size ceiling check.
CHUNK_BYTES = 64 * 1024

Outcome = Literal["ok", "blocked_robots", "blocked_host", "http_error", "error", "too_large"]


@dataclass
class FetchRecord:
    """One line of provenance. Serialised into the corpus manifest."""

    url: str
    outcome: Outcome
    status_code: int | None = None
    content_type: str | None = None
    bytes_received: int | None = None
    elapsed_seconds: float | None = None
    fetched_at: str = ""
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "outcome": self.outcome,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "bytes_received": self.bytes_received,
            "elapsed_seconds": (
                round(self.elapsed_seconds, 3)
                if self.elapsed_seconds is not None
                else None
            ),
            "fetched_at": self.fetched_at,
            "reason": self.reason,
        }


class FetchBlocked(Exception):
    """Request refused before it was made. Not an error -- expected."""

    def __init__(self, record: FetchRecord):
        super().__init__(f"{record.outcome}: {record.url} ({record.reason})")
        self.record = record


class FetchFailed(Exception):
    """Request was made and did not succeed."""

    def __init__(self, record: FetchRecord):
        super().__init__(f"{record.outcome}: {record.url} ({record.reason})")
        self.record = record


@dataclass
class PoliteFetcher:
    """Rate-limited, robots-aware HTTP client for exactly one host."""

    policy: RobotsPolicy
    user_agent: str = USER_AGENT

    _session: requests.Session = field(default_factory=requests.Session, repr=False)
    _last_request_finished: float = field(default=0.0, repr=False)
    _log: list[FetchRecord] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._host = urlparse(self.policy.base_url).netloc
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                # No Accept-Language: we want the site's default, and the
                # scraper deliberately stays on the German pages.
            }
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def get(self, url: str) -> requests.Response:
        """
        Fetch one URL politely.

        Raises FetchBlocked if the request was refused before it was
        made, or FetchFailed if it was made and did not succeed. On
        success the response body is already fully read.
        """
        blocked = self._refusal_reason(url)
        if blocked is not None:
            outcome, reason = blocked
            raise FetchBlocked(self._record(url, outcome, reason=reason))

        self._wait_for_slot()

        started = time.monotonic()
        try:
            response = self._session.get(
                url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True
            )
        except requests.RequestException as error:
            self._last_request_finished = time.monotonic()
            raise FetchFailed(
                self._record(url, "error", reason=type(error).__name__)
            ) from error

        try:
            body = self._read_body(response, url)
        finally:
            self._last_request_finished = time.monotonic()
            response.close()

        elapsed = time.monotonic() - started

        # Redirects can leave the host: a redirect off b-tu.de must not
        # end up in the mirror just because the first URL was allowed.
        final_host = urlparse(response.url).netloc
        if final_host != self._host:
            raise FetchBlocked(
                self._record(
                    url,
                    "blocked_host",
                    reason=f"redirected off-host to {final_host}",
                    status_code=response.status_code,
                    elapsed_seconds=elapsed,
                )
            )

        if response.status_code != 200:
            raise FetchFailed(
                self._record(
                    url,
                    "http_error",
                    reason=f"HTTP {response.status_code}",
                    status_code=response.status_code,
                    elapsed_seconds=elapsed,
                )
            )

        response._content = body  # noqa: SLF001 -- body was consumed by _read_body
        response._content_consumed = True  # noqa: SLF001

        self._record(
            url,
            "ok",
            status_code=response.status_code,
            content_type=response.headers.get("Content-Type"),
            bytes_received=len(body),
            elapsed_seconds=elapsed,
        )
        return response

    @property
    def log(self) -> list[FetchRecord]:
        """Every fetch attempt, in order."""
        return list(self._log)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PoliteFetcher":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------

    def _refusal_reason(self, url: str) -> tuple[Outcome, str] | None:
        """Return (outcome, reason) if this URL must not be fetched."""
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return "blocked_host", f"unsupported scheme {parsed.scheme!r}"

        if parsed.netloc != self._host:
            return "blocked_host", f"off-host {parsed.netloc!r}"

        if not self.policy.can_fetch(url):
            return "blocked_robots", self.policy.why_blocked(url) or "robots.txt"

        return None

    def _wait_for_slot(self) -> None:
        """
        Sleep until the crawl delay has elapsed.

        Measured from when the previous response FINISHED, not when it
        started. If BTU takes 2 s to answer, we still wait our full
        delay afterwards -- a slow server is a reason to back off, not
        an excuse to have already waited.
        """
        if self._last_request_finished == 0.0:
            return

        waited = time.monotonic() - self._last_request_finished
        remaining = self.policy.crawl_delay - waited
        if remaining > 0:
            time.sleep(remaining)

    def _read_body(self, response: requests.Response, url: str) -> bytes:
        """Stream the body, abandoning it if it exceeds the size ceiling."""
        chunks: list[bytes] = []
        total = 0

        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise FetchFailed(
                    self._record(
                        url,
                        "too_large",
                        reason=f"exceeded {MAX_RESPONSE_BYTES} bytes",
                        status_code=response.status_code,
                        bytes_received=total,
                    )
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def _record(self, url: str, outcome: Outcome, **fields) -> FetchRecord:
        record = FetchRecord(
            url=url,
            outcome=outcome,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        )
        self._log.append(record)
        return record
