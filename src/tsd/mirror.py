"""
mirror.py
---------
Write the frozen corpus to data/mirror/ as a self-contained local site.

Why this module exists in this shape:

    Discovery has already decided WHICH pages are in the corpus and has
    cached their HTML. Mirroring turns that into files that a local
    HTTPS server can serve, and that Firefox and wget can load without
    ever touching the real b-tu.de.

    "Without ever touching the real b-tu.de" is the whole point, and it
    is the reason the original scraper's `save_asset()` was a bug worth
    rewriting for. It did:

        except Exception:
            return None

    which left the original absolute URL in the HTML. During capture the
    browser would then fetch that asset from the live site -- outside
    network traffic contaminating a supposedly local, isolated capture,
    and invisible to the loopback capture filter, so the PCAP would look
    clean while the timing was quietly wrong.

    So this module holds two rules above all others:

    1. An asset that cannot be mirrored is RECORDED in
       MirrorResult.failures and its reference is NEUTRALISED -- never
       left pointing at the live site. A caller that finds a non-empty
       failures list should treat the mirror as suspect.
    2. Nothing that a browser auto-fetches may survive as an absolute
       off-host URL.

Layout produced:

    data/mirror/
        index.html                      the homepage
        <readable>_<digest>.html        every other corpus page
        assets/<digest>_<name>          every asset, deduplicated

    Pages live at the root, assets one level down. That means the same
    asset is referenced two different ways depending on who references
    it: "assets/x.png" from a page, and plain "x.png" from inside a
    stylesheet, because the stylesheet is itself already inside
    assets/. Getting this backwards produces a mirror that looks correct
    in a file listing and 404s in the browser.

Link rewriting knows the page set:

    A link to a page IN the corpus becomes its local filename. A link to
    anything else stays an absolute URL to the live site -- a relative
    href would otherwise resolve against the local server and 404, and a
    404's traffic shape is nothing like a page load. That is an artefact
    the scraper would have injected into the dataset itself.

Determinism:

    The mirror is gitignored, so the scripts have to be able to
    regenerate it. Page order is sorted, asset order within a page is
    document order, filenames come from urls.py digests, and no
    timestamp is written anywhere. Same cache in, same bytes out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Stylesheet

from .fetcher import FetchBlocked, FetchFailed, PoliteFetcher
from .urls import BASE_URL, asset_filename, normalise, page_filename, resolve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .discover import DiscoveryResult

ASSETS_DIRNAME = "assets"

# Stylesheets can import stylesheets. The nesting is bounded so a
# circular or pathological import chain cannot walk the whole site, and
# so the crawl length stays predictable. Depth 0 is an asset referenced
# from HTML; depth 1 is one referenced from a stylesheet, and so on.
MAX_CSS_DEPTH = 3

# <link> relations whose target IS an asset we mirror.
MIRRORED_LINK_RELS = frozenset({"stylesheet", "icon"})

# <link> relations a browser fetches on its own but which this module
# does not mirror. Left in place they would be live requests to b-tu.de
# during capture, so the tags are stripped instead. Relations that are
# pure metadata (canonical, alternate, author) are harmless and stay.
NETWORK_LINK_RELS = frozenset(
    {
        "preload", "modulepreload", "prefetch", "prerender",
        "preconnect", "dns-prefetch",
        "apple-touch-icon", "apple-touch-icon-precomposed", "mask-icon",
        "manifest",
    }
)

# Attributes carrying a single asset URL, per tag. The data-* ones are
# lazy-loading conventions: the real image URL sits there and only moves
# into src once JavaScript runs, so mirroring src alone would leave the
# live URL behind.
#
# The embedding tags (iframe, embed, object, video, audio, track) are
# here for the same reason, and their most common case resolves itself:
# a YouTube or OpenStreetMap embed is off-host, PoliteFetcher refuses it
# with blocked_host, the failure is recorded, and the attribute is
# deleted. So a third-party embed neutralises itself and says so, rather
# than sitting in the mirror waiting to make a live request during
# capture.
ASSET_URL_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "img": ("src", "data-src", "data-original", "data-lazy-src"),
    "script": ("src",),
    "source": ("src",),
    "input": ("src",),  # only when type="image"; checked at the call site
    "iframe": ("src",),
    "embed": ("src",),
    "object": ("data",),
    "video": ("src", "poster"),
    "audio": ("src",),
    "track": ("src",),
}

SRCSET_TAGS = ("img", "source")

# Failure outcomes this module raises itself, alongside the ones that
# come from FetchRecord.outcome. Every failure carries an outcome so
# callers can classify it WITHOUT parsing the human-readable reason:
# a reason string is written for people and will be reworded, an
# outcome is a value and can be branched on.
WRITE_ERROR = "write_error"
MISSING_HTML = "missing_html"
DEPTH_EXCEEDED = "depth_exceeded"

# A failed reference is pointed at about:blank rather than removed from
# the declaration: it is inert, costs no request, and unlike an empty
# url("") it does not make the browser re-request the page itself.
DEAD_CSS_URL = 'url("about:blank")'

_CSS_URL = re.compile(
    r"""url\(\s*(?P<quote>['"]?)(?P<url>[^'")]*?)(?P=quote)\s*\)""",
    re.IGNORECASE,
)

# @import may name a stylesheet as a bare string instead of url(...).
# Same live request if missed, so it gets the same treatment.
_CSS_IMPORT = re.compile(
    r"""@import\s+(?P<quote>['"])(?P<url>[^'"]+)(?P=quote)""",
    re.IGNORECASE,
)


@dataclass
class MirrorResult:
    """
    What one mirroring run produced. `failures` is the important one.

    Each failure is (url, outcome, reason). The outcome is carried
    separately because not all failures mean the same thing: a 404 on
    b-tu.de is a property of the site and will recur identically on
    every run, while a connection error is a property of this run and
    means the corpus is no longer reproducible. Callers must be able to
    tell those apart by value, not by matching on the reason text.
    """

    pages_written: int = 0
    assets_written: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    bytes_total: int = 0

    def __len__(self) -> int:
        return self.pages_written


class SiteMirror:
    """
    Write a frozen page set, with its assets, to a local directory.

    The fetcher is used for assets only. Page HTML comes from the
    discovery cache: re-fetching pages would be a second full pass over
    someone else's server for bytes we already hold.
    """

    def __init__(
        self,
        fetcher: PoliteFetcher,
        output_dir: str | Path,
        pages: Iterable[str],
        max_css_depth: int = MAX_CSS_DEPTH,
        on_event=None,
    ):
        self.fetcher = fetcher
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / ASSETS_DIRNAME
        self.pages = {normalise(url) for url in pages}
        self.max_css_depth = max_css_depth
        self.on_event = on_event or (lambda *_: None)

        self.result = MirrorResult()

        # normalised asset URL -> local filename, or None if it failed.
        # Both outcomes are cached: an asset is fetched exactly once, and
        # a broken one is not retried on every page that references it.
        # The entry is written BEFORE a stylesheet's own contents are
        # walked, so a circular @import terminates.
        self._assets: dict[str, str | None] = {}

        # normalised page URL -> local filename. Filled in by run(),
        # because whether the homepage is present depends on the cache.
        self._local_pages: dict[str, str] = {}

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self, html_cache: Mapping[str, str]) -> MirrorResult:
        """
        Write every corpus page and everything it references.

        `html_cache` is DiscoveryResult.html_cache: normalised page URL
        to HTML source.
        """
        cache = {normalise(url): html for url, html in html_cache.items()}
        self._local_pages = self._build_page_map(cache)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Sorted, not set order: the caller may hand us a set, and set
        # iteration order is not stable across runs. An unstable order
        # would reorder the failures list and defeat byte-for-byte
        # reproducibility of the run as a whole.
        for url in sorted(self._local_pages):
            html = cache.get(url)

            if html is None:
                self._fail(url, MISSING_HTML, "no cached HTML for this page")
                continue

            self._write_page(url, html, self._local_pages[url])

        return self.result

    # ----------------------------------------------------------
    # Page set
    # ----------------------------------------------------------

    def _build_page_map(self, cache: Mapping[str, str]) -> dict[str, str]:
        """
        Map every local page URL to the filename it will be written as.

        The homepage is added even when the caller did not list it: the
        walk starts there and caches it, but never adds it to
        DiscoveryResult.pages. It has to become index.html both because
        the server needs a directory index and because every page on the
        site links back to it -- and those links must rewrite locally,
        not point at the live site.
        """
        mapping = {url: page_filename(url) for url in sorted(self.pages)}

        home_url = normalise(BASE_URL)
        if home_url in cache:
            mapping[home_url] = "index.html"

        return mapping

    # ----------------------------------------------------------
    # One page
    # ----------------------------------------------------------

    def _write_page(self, page_url: str, html: str, filename: str) -> None:
        soup = BeautifulSoup(html, "html.parser")

        base_url = self._take_document_base(soup, page_url)

        self._rewrite_link_tags(soup, base_url)
        self._rewrite_asset_attributes(soup, base_url)
        self._rewrite_srcsets(soup, base_url)
        self._rewrite_inline_css(soup, base_url)
        self._rewrite_page_links(soup, base_url)

        body = str(soup).encode("utf-8")
        path = self.output_dir / filename

        try:
            path.write_bytes(body)
        except OSError as error:
            self._fail(page_url, WRITE_ERROR, f"page write failed: {error}")
            return

        self.result.pages_written += 1
        self.result.bytes_total += len(body)
        self.on_event("page_written", page_url, filename, len(body))

    def _take_document_base(self, soup: BeautifulSoup, page_url: str) -> str:
        """
        Return the URL relative references resolve against, and drop <base>.

        The tag has to go. Left in the mirrored page it would point at
        https://www.b-tu.de/, so every relative reference we carefully
        rewrote to a local filename would resolve back to the live site
        at load time -- the contamination this module exists to prevent,
        reintroduced by a single tag.
        """
        base_tag = soup.find("base", href=True)
        base_url = urljoin(page_url, base_tag["href"].strip()) if base_tag else page_url

        for tag in soup.find_all("base"):
            tag.decompose()

        return base_url

    # ----------------------------------------------------------
    # Asset references in HTML
    # ----------------------------------------------------------

    def _rewrite_link_tags(self, soup: BeautifulSoup, base_url: str) -> None:
        """Mirror stylesheets and icons; strip other auto-fetching links."""
        for tag in soup.find_all("link"):
            rels = {value.lower() for value in tag.get("rel", [])}

            if rels & MIRRORED_LINK_RELS:
                self._rewrite_attribute(tag, "href", base_url)
            elif rels & NETWORK_LINK_RELS:
                tag.decompose()

    def _rewrite_asset_attributes(self, soup: BeautifulSoup, base_url: str) -> None:
        for tag_name, attributes in ASSET_URL_ATTRIBUTES.items():
            for tag in soup.find_all(tag_name):
                if tag_name == "input" and tag.get("type", "").lower() != "image":
                    continue

                for attribute in attributes:
                    self._rewrite_attribute(tag, attribute, base_url)

    def _rewrite_srcsets(self, soup: BeautifulSoup, base_url: str) -> None:
        """
        Rewrite every candidate in a srcset, keeping the descriptors.

        Descriptors matter here beyond correctness: they are what decides
        which image the browser actually requests, and therefore part of
        the traffic shape being measured. Dropping them would change
        Firefox's request pattern while leaving wget's untouched -- a
        difference introduced by the scraper, landing on one class only.
        """
        for tag_name in SRCSET_TAGS:
            for tag in soup.find_all(tag_name):
                value = tag.get("srcset")
                if not value:
                    continue

                rewritten: list[str] = []

                for candidate, descriptor in _parse_srcset(value):
                    local = self._local_asset_reference(base_url, candidate)
                    if local is None:
                        continue
                    rewritten.append(f"{local} {descriptor}".strip())

                if rewritten:
                    tag["srcset"] = ", ".join(rewritten)
                else:
                    del tag["srcset"]

    def _rewrite_inline_css(self, soup: BeautifulSoup, base_url: str) -> None:
        """
        Rewrite url() in style="" attributes and in <style> blocks.

        Same class of reference as a stylesheet file, same contamination
        if missed -- a background-image left absolute is a live request
        during capture.
        """
        for tag in soup.find_all(style=True):
            tag["style"] = self._rewrite_css_text(
                tag["style"], base_url, depth=0, prefix=f"{ASSETS_DIRNAME}/"
            )

        for tag in soup.find_all("style"):
            # get_text(), not tag.string: .string is None whenever the
            # block is not exactly one child, which a CSS comment or a
            # CDATA section can cause. That returns None, the `continue`
            # fires, and the whole block is skipped in silence -- leaving
            # every url() in it absolute. A silent skip that leaves live
            # URLs behind is precisely the bug class this module exists
            # to remove, so it must not be reintroduced by an idiom.
            css = tag.get_text()
            if not css.strip():
                continue

            rewritten = self._rewrite_css_text(
                css, base_url, depth=0, prefix=f"{ASSETS_DIRNAME}/"
            )

            tag.clear()
            # Stylesheet, not a bare str: it marks the string as raw
            # stylesheet text, so no formatter can decide to escape the
            # ">" in a child selector or an "&" in a comment.
            tag.append(Stylesheet(rewritten))

    def _rewrite_attribute(self, tag, attribute: str, base_url: str) -> None:
        """
        Point one attribute at the local copy, or remove it.

        Removal rather than "leave it as it was" is the deliberate part:
        an attribute we could not mirror is a request to the live site,
        so it must not survive. The failure is already in
        MirrorResult.failures by the time we get here.
        """
        raw = tag.get(attribute)
        if not raw or not raw.strip():
            return

        resolved = resolve(base_url, raw)
        if resolved is None:
            # data:, mailto:, javascript: -- not a fetchable resource,
            # nothing to mirror and nothing to leak.
            return

        filename = self._mirror_asset(resolved, depth=0)

        if filename is None:
            del tag[attribute]
        else:
            tag[attribute] = f"{ASSETS_DIRNAME}/{filename}"

    def _local_asset_reference(self, base_url: str, raw: str) -> str | None:
        """Mirror one asset and return how a PAGE should reference it."""
        resolved = resolve(base_url, raw)
        if resolved is None:
            return raw.strip()

        filename = self._mirror_asset(resolved, depth=0)
        if filename is None:
            return None

        return f"{ASSETS_DIRNAME}/{filename}"

    # ----------------------------------------------------------
    # Page links
    # ----------------------------------------------------------

    def _rewrite_page_links(self, soup: BeautifulSoup, base_url: str) -> None:
        """
        Rewrite <a href> against the frozen page set.

        In the corpus  -> local filename.
        Outside it     -> an absolute URL to the live site, unchanged if
                          it already was absolute.

        The second case is why discovery has to finish before mirroring
        starts. Left relative, such a link would resolve against the
        local server and 404 during capture, and a 404 does not look
        like a page load. Made local, it would be a dangling file. Kept
        absolute, it is simply a link nobody clicks during an automated
        page load.
        """
        for tag in soup.find_all("a", href=True):
            raw = tag["href"].strip()
            if not raw:
                continue

            resolved = resolve(base_url, raw)
            if resolved is None:
                continue

            if resolved in self._local_pages:
                tag["href"] = self._local_pages[resolved]
                continue

            if not urlparse(raw).scheme:
                tag["href"] = urljoin(base_url, raw)

    # ----------------------------------------------------------
    # Assets
    # ----------------------------------------------------------

    def _mirror_asset(self, url: str, depth: int) -> str | None:
        """
        Fetch and store one asset. Returns its local filename, or None.

        Every asset is fetched exactly once, keyed by its normalised URL:
        a site-wide stylesheet referenced from 100 pages is 1 request,
        not 100. Failures are cached too, so a dead URL is reported once
        and not retried per page.
        """
        if url in self._assets:
            return self._assets[url]

        if depth > self.max_css_depth:
            self._assets[url] = None
            self._fail(
                url,
                DEPTH_EXCEEDED,
                f"nested deeper than max_css_depth={self.max_css_depth}",
            )
            return None

        try:
            response = self.fetcher.get(url)
        except (FetchBlocked, FetchFailed) as error:
            self._assets[url] = None
            self._fail(
                url,
                error.record.outcome,
                error.record.reason or error.record.outcome,
            )
            return None

        filename = asset_filename(url)

        # Recorded before the stylesheet body is walked: a stylesheet
        # that imports itself, directly or in a cycle, then terminates
        # on this cache hit instead of recursing.
        self._assets[url] = filename

        body = response.body
        if _is_css(url, response.content_type):
            # prefix="" -- a stylesheet already lives in assets/, so its
            # own references are siblings, not "assets/..." again.
            body = self._rewrite_css_text(
                response.text, url, depth=depth + 1, prefix=""
            ).encode("utf-8")

        try:
            (self.assets_dir / filename).write_bytes(body)
        except OSError as error:
            self._assets[url] = None
            self._fail(url, WRITE_ERROR, f"asset write failed: {error}")
            return None

        self.result.assets_written += 1
        self.result.bytes_total += len(body)
        self.on_event("asset_written", url, filename, len(body))

        return filename

    def _rewrite_css_text(
        self, css: str, base_url: str, depth: int, prefix: str
    ) -> str:
        """
        Rewrite url() and @import in a stylesheet or a style attribute.

        `prefix` is what a reference from THIS location needs in front of
        the filename: "assets/" from a page at the mirror root, "" from
        inside a stylesheet that already sits in assets/.
        """

        def replace_url(match: re.Match) -> str:
            local = self._css_reference(match.group("url"), base_url, depth, prefix)
            return DEAD_CSS_URL if local is None else f'url("{local}")'

        def replace_import(match: re.Match) -> str:
            local = self._css_reference(match.group("url"), base_url, depth, prefix)
            if local is None:
                return f"@import {DEAD_CSS_URL}"
            return f'@import "{local}"'

        css = _CSS_IMPORT.sub(replace_import, css)
        return _CSS_URL.sub(replace_url, css)

    def _css_reference(
        self, raw: str, base_url: str, depth: int, prefix: str
    ) -> str | None:
        """Mirror one url() target; None means the reference must die."""
        resolved = resolve(base_url, raw)

        if resolved is None:
            # data: URIs are already inline -- keep them verbatim.
            return raw.strip()

        filename = self._mirror_asset(resolved, depth)
        if filename is None:
            return None

        return f"{prefix}{filename}"

    # ----------------------------------------------------------
    # Failures
    # ----------------------------------------------------------

    def _fail(self, url: str, outcome: str, reason: str) -> None:
        """
        Record a failure. Never swallow one.

        A silently dropped asset is the specific bug this rewrite exists
        to remove: it leaves the mirror looking complete while the
        capture quietly reaches the live site.

        `outcome` is what callers branch on; `reason` is for humans.
        """
        self.result.failures.append((url, outcome, reason))
        self.on_event("failure", url, outcome, reason)


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def _is_css(url: str, content_type: str) -> bool:
    """
    Is this stylesheet text we need to walk into?

    Content-Type first, extension second: TYPO3 serves generated CSS
    from extensionless paths, and a stylesheet we fail to recognise is a
    stylesheet whose url() references stay absolute.
    """
    if "text/css" in (content_type or "").lower():
        return True

    return urlparse(url).path.lower().endswith(".css")


def _parse_srcset(value: str) -> list[tuple[str, str]]:
    """
    Split a srcset into (url, descriptor) pairs.

    Not a plain split(","): a candidate descriptor is comma-separated
    but a URL may itself contain commas (data: URIs always do). This
    follows the HTML algorithm -- read the URL up to whitespace, and if
    it ended in a comma the candidate had no descriptor.
    """
    candidates: list[tuple[str, str]] = []
    index, length = 0, len(value)

    while index < length:
        while index < length and (value[index].isspace() or value[index] == ","):
            index += 1

        if index >= length:
            break

        start = index
        while index < length and not value[index].isspace():
            index += 1
        url = value[start:index]

        if url.endswith(","):
            candidates.append((url.rstrip(","), ""))
            continue

        while index < length and value[index].isspace():
            index += 1

        start = index
        while index < length and value[index] != ",":
            index += 1
        descriptor = value[start:index].strip()

        if index < length:
            index += 1

        if url:
            candidates.append((url, descriptor))

    return candidates


def mirror_site(
    fetcher: PoliteFetcher,
    output_dir: str | Path,
    discovery: "DiscoveryResult",
    on_event=None,
) -> MirrorResult:
    """Mirror a DiscoveryResult in one call. Used by scripts/scrape_corpus.py."""
    mirror = SiteMirror(
        fetcher=fetcher,
        output_dir=output_dir,
        pages=discovery.pages,
        on_event=on_event,
    )
    return mirror.run(discovery.html_cache)
