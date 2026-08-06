"""
test_discover.py
----------------
Tests for tsd.discover.

A fake site is served from a dict, so the walk logic runs for real
against a known link graph. The properties being asserted are the ones
the corpus depends on: no duplicates, deterministic under a fixed seed,
and refusals never enter the page set.
"""

import pytest

from tsd.discover import CorpusDiscoverer
from tsd.fetcher import FetchBlocked, FetchFailed, FetchRecord, FetchResult


def page(*links: str) -> str:
    """Build an HTML page linking to the given paths."""
    anchors = "".join(f'<a href="{href}">x</a>' for href in links)
    return f"<html><body>{anchors}</body></html>"


class FakeSite:
    """
    Stands in for PoliteFetcher against a fixed link graph.

    Paths listed in `blocked` raise FetchBlocked, paths in `broken`
    raise FetchFailed, and anything absent returns a 404-style failure.
    """

    def __init__(self, pages: dict[str, str], blocked=(), broken=(), content_type=None):
        self.pages = pages
        self.blocked = set(blocked)
        self.broken = set(broken)
        self.content_type = content_type or {}
        self.requested: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.requested.append(url)

        if url in self.blocked:
            raise FetchBlocked(
                FetchRecord(url=url, outcome="blocked_robots", reason="robots.txt")
            )

        if url in self.broken:
            raise FetchFailed(
                FetchRecord(url=url, outcome="http_error", reason="HTTP 500")
            )

        if url not in self.pages:
            raise FetchFailed(
                FetchRecord(url=url, outcome="http_error", reason="HTTP 404")
            )

        return FetchResult(
            url=url,
            body=self.pages[url].encode(),
            status_code=200,
            content_type=self.content_type.get(url, "text/html; charset=utf-8"),
            headers={},
        )


HOME = "https://www.b-tu.de/"
A = "https://www.b-tu.de/a/"
B = "https://www.b-tu.de/b/"
C = "https://www.b-tu.de/c/"
D = "https://www.b-tu.de/d/"


@pytest.fixture
def small_site():
    return FakeSite(
        {
            HOME: page("/a/", "/b/", "/c/"),
            A: page("/b/", "/d/"),
            B: page("/c/"),
            C: page("/a/"),
            D: page(),
        }
    )


def discoverer(site, **kwargs):
    kwargs.setdefault("target_pages", 10)
    kwargs.setdefault("total_walks", 5)
    kwargs.setdefault("max_depth", 3)
    return CorpusDiscoverer(fetcher=site, **kwargs)


# --------------------------------------------------------------
# Core properties of the page set
# --------------------------------------------------------------

def test_pages_are_unique(small_site):
    result = discoverer(small_site).run()
    assert len(result.pages) == len(set(result.pages))


def test_homepage_is_not_in_the_page_set(small_site):
    """
    The homepage is the walk's starting point, not a discovered page.
    It is mirrored separately as index.html.
    """
    result = discoverer(small_site).run()
    assert HOME not in result.pages


def test_walk_stops_at_the_target(small_site):
    result = discoverer(small_site, target_pages=2).run()
    assert len(result.pages) == 2


def test_walk_finds_pages_reachable_without_revisiting():
    """
    A random walk never re-enters a page it has already collected, so a
    page reachable ONLY through an already-collected page is unreachable.

    Here D hangs off A. Once A is collected the walk cannot pass through
    it again, so D is never found.

    This is a real property of the method, not a bug: discovery is
    incomplete by construction. On b-tu.de the link graph is dense
    enough that most pages have several routes, but the corpus should
    still be understood as "100 pages the walk happened to reach",
    not "the 100 most important pages". Stated in the README.
    """
    site = FakeSite(
        {
            HOME: page("/a/", "/b/", "/c/"),
            A: page("/b/", "/d/"),
            B: page("/c/"),
            C: page("/a/"),
            D: page(),
        }
    )
    result = discoverer(site, target_pages=10, total_walks=20).run()

    assert set(result.pages) == {A, B, C}
    assert D not in result.pages


# --------------------------------------------------------------
# Determinism -- the corpus must be regenerable from the scripts
# --------------------------------------------------------------

def test_same_seed_gives_the_same_corpus(small_site):
    first = discoverer(FakeSite(small_site.pages), seed=7, target_pages=3).run()
    second = discoverer(FakeSite(small_site.pages), seed=7, target_pages=3).run()
    assert first.pages == second.pages


def test_different_seeds_can_differ(small_site):
    """Not a guarantee for any one pair, but the walk must not ignore the seed."""
    results = {
        tuple(
            discoverer(FakeSite(small_site.pages), seed=s, target_pages=2).run().pages
        )
        for s in range(12)
    }
    assert len(results) > 1


# --------------------------------------------------------------
# Refusals stay out of the corpus
# --------------------------------------------------------------

def test_blocked_pages_are_recorded_not_collected():
    site = FakeSite(
        {HOME: page("/a/", "/b/"), A: page(), B: page()},
        blocked={A},
    )
    result = discoverer(site).run()

    assert A not in result.pages
    assert A in result.refused
    assert B in result.pages


def test_failed_pages_are_recorded_not_collected():
    site = FakeSite(
        {HOME: page("/a/", "/b/"), A: page(), B: page()},
        broken={A},
    )
    result = discoverer(site).run()

    assert A not in result.pages
    assert "500" in result.refused[A]


def test_refused_page_is_not_requested_repeatedly():
    """A dead URL must not be retried on every walk: that is load for nothing."""
    site = FakeSite(
        {HOME: page("/a/", "/b/"), A: page(), B: page()},
        broken={A},
    )
    discoverer(site, total_walks=5).run()

    assert site.requested.count(A) == 1


def test_non_html_responses_are_refused():
    """TYPO3 serves PDFs from extensionless paths; those are not pages."""
    site = FakeSite(
        {HOME: page("/a/", "/b/"), A: page(), B: page()},
        content_type={A: "application/pdf"},
    )
    result = discoverer(site).run()

    assert A not in result.pages
    assert "not HTML" in result.refused[A]


# --------------------------------------------------------------
# Off-corpus links are never followed
# --------------------------------------------------------------

def test_external_and_non_page_links_are_ignored():
    site = FakeSite(
        {
            HOME: page(
                "https://youtube.com/watch",
                "/doc.pdf",
                "mailto:x@b-tu.de",
                "/en/faculty/",
                "/a/",
            ),
            A: page(),
        }
    )
    result = discoverer(site).run()

    assert result.pages == [A]
    assert all("b-tu.de" in url for url in site.requested)


# --------------------------------------------------------------
# Caching
# --------------------------------------------------------------

def test_html_is_cached_for_the_mirroring_pass(small_site):
    result = discoverer(small_site).run()

    for url in result.pages:
        assert url in result.html_cache
    assert HOME in result.html_cache


def test_no_page_is_fetched_twice(small_site):
    site = small_site
    discoverer(site).run()

    assert len(site.requested) == len(set(site.requested))


# --------------------------------------------------------------
# Dead ends
# --------------------------------------------------------------

def test_walk_terminates_when_the_site_is_exhausted():
    """Target unreachable: the walk must end, not spin."""
    site = FakeSite({HOME: page("/a/"), A: page()})
    result = discoverer(site, target_pages=50, total_walks=3).run()

    assert result.pages == [A]


def test_unreachable_homepage_is_fatal():
    site = FakeSite({})
    with pytest.raises(RuntimeError):
        discoverer(site).run()
