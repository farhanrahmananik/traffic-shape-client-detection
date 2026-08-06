"""
discover.py
-----------
Random-walk discovery of the corpus page set.

Why discovery is separate from mirroring:

    Discovery decides WHICH pages are in the corpus. Mirroring writes
    them to disk with their assets and rewrites their links. Doing both
    at once creates two problems.

    First, waste: a walk that finds 105 pages when the target is 100
    has already pulled every asset for 5 pages that will be discarded.
    That is load on someone else's server for nothing.

    Second, and worse: rewriting a page's internal links requires
    knowing the final page set. A link to a page that ends up outside
    the corpus must stay absolute, or the mirrored site has a dead link
    -- which becomes a 404 during capture, and a 404 has a completely
    different traffic shape from a page load. That would be an artefact
    in the dataset introduced by the scraper.

    So: walk first with the page set unknown, freeze it, then mirror.

The HTML fetched during the walk is cached, so the mirroring pass does
not ask BTU for the same pages twice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .fetcher import FetchBlocked, FetchFailed, PoliteFetcher
from .urls import BASE_URL, is_corpus_page, normalise, resolve

TOTAL_WALKS = 20
MAX_DEPTH = 5
TARGET_PAGES = 100

# A walk that keeps hitting dead ends or errors is restarted from the
# homepage. Past this many restarts, the walk is abandoned.
MAX_RESTARTS_PER_WALK = 20

RANDOM_SEED = 42


@dataclass
class DiscoveryResult:
    """The frozen corpus page set, plus everything learned on the way."""

    pages: list[str] = field(default_factory=list)
    html_cache: dict[str, str] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    walks_run: int = 0

    def __len__(self) -> int:
        return len(self.pages)


class CorpusDiscoverer:
    """
    Random walk over b-tu.de, collecting unique page URLs.

    The walk structure mirrors the original assignment: TOTAL_WALKS
    independent walks, each following links up to MAX_DEPTH, stopping
    once TARGET_PAGES unique pages have been seen.

    Seeded: the same seed and the same site produce the same walk, which
    is what makes the corpus reproducible from the scripts alone -- the
    substitute for publishing the mirror itself.
    """

    def __init__(
        self,
        fetcher: PoliteFetcher,
        target_pages: int = TARGET_PAGES,
        total_walks: int = TOTAL_WALKS,
        max_depth: int = MAX_DEPTH,
        seed: int = RANDOM_SEED,
        on_event=None,
    ):
        self.fetcher = fetcher
        self.target_pages = target_pages
        self.total_walks = total_walks
        self.max_depth = max_depth
        self.random = random.Random(seed)
        self.on_event = on_event or (lambda *_: None)

        self.result = DiscoveryResult()
        self._seen: set[str] = set()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> DiscoveryResult:
        """Walk until the target is reached or the walk budget is spent."""
        home_url = normalise(BASE_URL)
        home_html = self._fetch_html(home_url)

        if home_html is None:
            raise RuntimeError("homepage could not be fetched; cannot start walk")

        self.result.html_cache[home_url] = home_html

        for walk_number in range(1, self.total_walks + 1):
            if len(self.result) >= self.target_pages:
                break

            self.result.walks_run = walk_number
            self.on_event("walk_start", walk_number, len(self.result))
            self._walk(home_url, home_html)

        return self.result

    # ----------------------------------------------------------
    # One walk
    # ----------------------------------------------------------

    def _walk(self, home_url: str, home_html: str) -> None:
        current_url, current_html = home_url, home_html
        depth = 0
        restarts = 0

        while depth < self.max_depth and len(self.result) < self.target_pages:
            next_url = self._choose_link(current_url, current_html)

            if next_url is None:
                restarts += 1
                if restarts > MAX_RESTARTS_PER_WALK:
                    self.on_event("walk_exhausted", restarts)
                    return
                current_url, current_html, depth = home_url, home_html, 0
                continue

            html = self._fetch_html(next_url)

            if html is None:
                restarts += 1
                if restarts > MAX_RESTARTS_PER_WALK:
                    self.on_event("walk_exhausted", restarts)
                    return
                continue

            self._seen.add(next_url)
            self.result.pages.append(next_url)
            self.result.html_cache[next_url] = html
            depth += 1
            restarts = 0

            self.on_event("page_found", next_url, len(self.result), depth)

            current_url, current_html = next_url, html

    # ----------------------------------------------------------
    # Link selection
    # ----------------------------------------------------------

    def _choose_link(self, page_url: str, html: str) -> str | None:
        """
        Pick one unseen corpus link from this page.

        Candidates are sorted before the random choice so that the seed
        fully determines the walk: set iteration order is not stable
        across runs, and an unstable walk would mean the scripts cannot
        regenerate the corpus.
        """
        candidates = sorted(self._corpus_links(page_url, html) - self._seen)

        if not candidates:
            return None

        return self.random.choice(candidates)

    def _corpus_links(self, page_url: str, html: str) -> set[str]:
        """Every link on this page that belongs in the corpus."""
        soup = BeautifulSoup(html, "html.parser")
        links: set[str] = set()

        for tag in soup.find_all("a"):
            resolved = resolve(page_url, tag.get("href"))

            if resolved is None or not is_corpus_page(resolved):
                continue

            if resolved in self.result.refused:
                continue

            links.add(resolved)

        return links

    # ----------------------------------------------------------
    # Fetching
    # ----------------------------------------------------------

    def _fetch_html(self, url: str) -> str | None:
        """
        Fetch one page as HTML, or None if it is unusable.

        Refusals and failures are remembered so the walk does not try
        the same dead URL again, and so the manifest can report exactly
        what was skipped and why.
        """
        if url in self.result.html_cache:
            return self.result.html_cache[url]

        try:
            response = self.fetcher.get(url)
        except FetchBlocked as blocked:
            self.result.refused[url] = blocked.record.reason or "blocked"
            self.on_event("refused", url, blocked.record.reason)
            return None
        except FetchFailed as failed:
            self.result.refused[url] = failed.record.reason or "failed"
            self.on_event("failed", url, failed.record.reason)
            return None

        # A URL can pass the extension check and still not be a page:
        # BTU serves PDFs and downloads from extensionless paths.
        if "html" not in response.content_type.lower():
            self.result.refused[url] = f"not HTML ({response.content_type})"
            self.on_event("refused", url, "not HTML")
            return None

        return response.text
