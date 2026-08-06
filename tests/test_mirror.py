"""
test_mirror.py
--------------
Tests for tsd.mirror.

No network: a fake site serves assets from a dict, and pages are written
into pytest's tmp_path so the real file-writing path runs.

What these tests are actually protecting is the dataset, not the module.
The mirror is served over local HTTPS and loaded by Firefox and wget
while tcpdump records. Any reference that still points at the live
b-tu.de becomes a request that leaves the machine during capture -- and
because tcpdump is filtered to the local server, it never shows up in
the PCAP. The trace looks clean while its timing is quietly wrong, and
the cost lands on the Firefox class only, since wget fetches no
subresources at all. So each test below names the artefact it prevents.
"""

import re
from urllib.parse import urlparse

import pytest
from bs4 import BeautifulSoup

from tsd.fetcher import FetchBlocked, FetchFailed, FetchRecord, FetchResult
from tsd.mirror import SiteMirror
from tsd.urls import asset_filename, page_filename

HOME = "https://www.b-tu.de/"
PAGE_A = "https://www.b-tu.de/a/"
PAGE_B = "https://www.b-tu.de/b/"

CONTENT_TYPES = {
    ".css": "text/css",
    ".js": "application/javascript",
    ".png": "image/png",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".svg": "image/svg+xml",
    ".html": "text/html; charset=utf-8",
}


class FakeSite:
    """
    Stands in for PoliteFetcher, with the same refusal behaviour.

    Off-host URLs raise FetchBlocked exactly as PoliteFetcher does, so
    third-party embeds are exercised for real rather than mocked around.
    Anything absent is a 404.
    """

    def __init__(self, assets: dict[str, bytes] | None = None):
        self.assets = dict(assets or {})
        self.requested: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.requested.append(url)

        if urlparse(url).netloc != "www.b-tu.de":
            raise FetchBlocked(
                FetchRecord(
                    url=url,
                    outcome="blocked_host",
                    reason=f"off-host {urlparse(url).netloc!r}",
                )
            )

        if url not in self.assets:
            raise FetchFailed(
                FetchRecord(url=url, outcome="http_error", reason="HTTP 404")
            )

        return FetchResult(
            url=url,
            body=self.assets[url],
            status_code=200,
            content_type=self._content_type(url),
            headers={},
        )

    @staticmethod
    def _content_type(url: str) -> str:
        path = urlparse(url).path.lower()
        for suffix, content_type in CONTENT_TYPES.items():
            if path.endswith(suffix):
                return content_type
        return "application/octet-stream"


def mirror(site, output_dir, cache, pages=(), **kwargs):
    """Run one mirroring pass and hand back (result, output_dir)."""
    site_mirror = SiteMirror(
        fetcher=site, output_dir=output_dir, pages=pages, **kwargs
    )
    return site_mirror.run(cache), output_dir


def read(output_dir, name: str) -> str:
    return (output_dir / name).read_text(encoding="utf-8")


def read_page(output_dir, url: str) -> str:
    return read(output_dir, page_filename(url))


def asset_text(output_dir, url: str) -> str:
    return (output_dir / "assets" / asset_filename(url)).read_text(encoding="utf-8")


# --------------------------------------------------------------
# The one that matters most
# --------------------------------------------------------------

# <link> relations a browser fetches on its own. An absolute URL in one
# of these is a live request during capture.
FETCHING_LINK_RELS = {
    "stylesheet", "icon", "preload", "modulepreload", "prefetch",
    "prerender", "preconnect", "dns-prefetch", "apple-touch-icon",
    "apple-touch-icon-precomposed", "mask-icon", "manifest",
}

# Attributes that hold a URL the browser does NOT fetch on its own.
# These are allowed to stay absolute -- indeed <a href> to a page outside
# the corpus MUST stay absolute.
NON_FETCHING_ATTRIBUTES = {
    ("a", "href"),
    ("area", "href"),
    ("form", "action"),
    ("meta", "content"),
    ("blockquote", "cite"),
    ("q", "cite"),
    ("del", "cite"),
    ("ins", "cite"),
}

_ABSOLUTE = re.compile(r"https?://", re.IGNORECASE)
_CSS_URL = re.compile(r"""url\(\s*['"]?(?P<url>[^'")]*)""", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"""@import\s+['"](?P<url>[^'"]+)""", re.IGNORECASE)


def live_url_offences(output_dir) -> list[tuple]:
    """
    Every absolute http(s) URL left in a position a browser auto-fetches.

    Deliberately written as a sweep over ALL tags and ALL attributes with
    a small allowlist, rather than a check of the tags mirror.py knows
    about. A test that only looks where the module already looks can
    never catch the tag nobody thought of, which is the failure mode
    being defended against.

    Inline <script> bodies are out of scope here on purpose: mirror.py
    does not rewrite JavaScript, and that gap is recorded in CLAUDE.md as
    the reason step 4 must block all non-loopback traffic during capture.
    """
    offences: list[tuple] = []

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        css_blobs: list[str] = []

        if path.suffix == ".html":
            soup = BeautifulSoup(text, "html.parser")

            for tag in soup.find_all(True):
                for attribute, value in tag.attrs.items():
                    values = value if isinstance(value, list) else [value]

                    for candidate in values:
                        if not isinstance(candidate, str):
                            continue
                        if not _ABSOLUTE.search(candidate):
                            continue
                        if (tag.name, attribute) in NON_FETCHING_ATTRIBUTES:
                            continue
                        if tag.name == "link" and attribute == "href":
                            rels = {r.lower() for r in tag.get("rel", [])}
                            if not rels & FETCHING_LINK_RELS:
                                continue
                        offences.append((path.name, tag.name, attribute, candidate))

            css_blobs = [tag["style"] for tag in soup.find_all(style=True)]
            css_blobs += [tag.get_text() for tag in soup.find_all("style")]
        elif path.suffix == ".css":
            css_blobs = [text]

        for blob in css_blobs:
            for pattern in (_CSS_URL, _CSS_IMPORT):
                for match in pattern.finditer(blob):
                    if _ABSOLUTE.search(match.group("url")):
                        offences.append((path.name, "css", "url()", match.group("url")))

    return offences


EVERYTHING_PAGE = """<html><head>
<base href="https://www.b-tu.de/">
<meta property="og:image" content="https://www.b-tu.de/logo.png">
<link rel="canonical" href="https://www.b-tu.de/">
<link rel="stylesheet" href="css/main.css">
<link rel="stylesheet" href="https://cdn.example.com/vendor.css">
<link rel="shortcut icon" href="fav.ico">
<link rel="preload" href="https://www.b-tu.de/font.woff2" as="font">
<script src="app.js"></script>
<style>/* & */ div > p{background:url(logo.png)}</style>
</head><body>
<img src="logo.png" data-src="lazy.png"
     srcset="logo.png 1x, data:image/gif;base64,AA,A 2x">
<input type="image" src="logo.png">
<iframe src="/inner.html"></iframe>
<iframe src="https://www.youtube.com/embed/abc"></iframe>
<video src="clip.mp4" poster="poster.jpg"></video>
<div style="background:url('logo.png')"></div>
<a href="/a/">in corpus</a>
<a href="https://www.b-tu.de/en/faculty/">outside the corpus</a>
</body></html>"""

EVERYTHING_ASSETS = {
    # The @import and one url() are written ABSOLUTE on purpose. With
    # relative ones only, a regression that stopped rewriting CSS
    # entirely would leave nothing absolute behind, and the sweep would
    # pass while the mirror was broken. TYPO3 emits absolute asset URLs
    # anyway, so this is also the realistic case.
    "https://www.b-tu.de/css/main.css": (
        b'@import "https://www.b-tu.de/css/nested.css";\n'
        b"body{background:url(https://www.b-tu.de/img/bg.png)}"
    ),
    "https://www.b-tu.de/css/nested.css": b'p{background:url("dot.gif")}',
    "https://www.b-tu.de/css/dot.gif": b"GIF",
    "https://www.b-tu.de/img/bg.png": b"PNG",
    "https://www.b-tu.de/logo.png": b"LOGO",
    "https://www.b-tu.de/lazy.png": b"LAZY",
    "https://www.b-tu.de/app.js": b"// js",
    "https://www.b-tu.de/fav.ico": b"ICO",
    "https://www.b-tu.de/inner.html": b"<html></html>",
    "https://www.b-tu.de/clip.mp4": b"MP4",
    "https://www.b-tu.de/poster.jpg": b"JPG",
}


def test_no_live_urls_survive_in_the_mirror(tmp_path):
    """
    Prevents the artefact this whole module exists for: a reference left
    pointing at the live b-tu.de is fetched during capture, off the
    loopback interface tcpdump is filtered to -- so it is absent from the
    PCAP but present in Firefox's inter-arrival times, and only Firefox's.
    """
    site = FakeSite(EVERYTHING_ASSETS)
    cache = {HOME: EVERYTHING_PAGE, PAGE_A: "<html><body>a</body></html>"}

    result, output_dir = mirror(site, tmp_path, cache, pages=[PAGE_A])

    assert live_url_offences(output_dir) == []

    # The two unreachable ones are refused, loudly, not silently dropped.
    refused = {url: (outcome, reason) for url, outcome, reason in result.failures}
    assert refused["https://cdn.example.com/vendor.css"][0] == "blocked_host"
    assert "off-host" in refused["https://cdn.example.com/vendor.css"][1]
    assert any("youtube.com" in url for url in refused)

    # And the mirror is still a mirror: the reachable assets are there.
    assert result.assets_written == len(EVERYTHING_ASSETS)


# --------------------------------------------------------------
# Link rewriting against the frozen page set
# --------------------------------------------------------------

def test_corpus_page_link_becomes_a_local_filename(tmp_path):
    """A corpus link left absolute would load the live site mid-capture."""
    cache = {HOME: '<html><body><a href="/a/">x</a></body></html>', PAGE_A: "<html></html>"}

    _, output_dir = mirror(FakeSite(), tmp_path, cache, pages=[PAGE_A])

    assert f'href="{page_filename(PAGE_A)}"' in read(output_dir, "index.html")


def test_link_outside_the_corpus_stays_absolute_and_unchanged(tmp_path):
    """
    Rewriting an off-corpus link to a local filename would create a
    dangling file reference, and the 404 it produced during capture has
    nothing like the traffic shape of a page load.
    """
    href = "https://www.b-tu.de/en/faculty/?utm_source=x#frag"
    cache = {HOME: f'<html><body><a href="{href}">x</a></body></html>'}

    _, output_dir = mirror(FakeSite(), tmp_path, cache)

    assert href in read(output_dir, "index.html")


def test_relative_link_outside_the_corpus_is_made_absolute(tmp_path):
    """
    Left relative, it would resolve against the LOCAL server and 404
    during capture -- an artefact injected into the dataset by the
    scraper itself.
    """
    cache = {HOME: '<html><body><a href="/other/">x</a></body></html>'}

    _, output_dir = mirror(FakeSite(), tmp_path, cache)

    assert 'href="https://www.b-tu.de/other/"' in read(output_dir, "index.html")


def test_homepage_is_written_as_index_html(tmp_path):
    """The server needs a directory index; without it every load is a 404."""
    cache = {HOME: "<html><body>home</body></html>"}

    result, output_dir = mirror(FakeSite(), tmp_path, cache)

    assert (output_dir / "index.html").is_file()
    assert result.pages_written == 1


def test_link_to_the_homepage_resolves_to_index_html(tmp_path):
    """
    Every page on the site links back to the homepage. Missed, that is
    one live request per page load -- the most common one on the site.
    """
    cache = {HOME: "<html></html>", PAGE_A: '<html><body><a href="/">home</a></body></html>'}

    _, output_dir = mirror(FakeSite(), tmp_path, cache, pages=[PAGE_A])

    assert 'href="index.html"' in read_page(output_dir, PAGE_A)


def test_base_tag_is_used_for_resolution_then_removed(tmp_path):
    """
    Left in place, <base> re-points every rewritten relative reference
    back at the live site at load time -- undoing the whole rewrite with
    a single tag.
    """
    deep = "https://www.b-tu.de/deep/page/"
    site = FakeSite({"https://www.b-tu.de/static/logo.png": b"LOGO"})
    cache = {
        deep: (
            '<html><head><base href="https://www.b-tu.de/static/"></head>'
            '<body><img src="logo.png"></body></html>'
        )
    }

    result, output_dir = mirror(site, tmp_path, cache, pages=[deep])

    assert site.requested == ["https://www.b-tu.de/static/logo.png"]
    assert result.failures == []
    assert "<base" not in read_page(output_dir, deep)


# --------------------------------------------------------------
# Assets
# --------------------------------------------------------------

def test_shared_asset_is_fetched_once(tmp_path):
    """
    A site-wide stylesheet appears on all 100 pages. Re-fetching per page
    is 100 requests to someone else's server for bytes already held.
    """
    logo = "https://www.b-tu.de/logo.png"
    site = FakeSite({logo: b"LOGO"})
    body = '<html><body><img src="/logo.png"></body></html>'
    cache = {HOME: body, PAGE_A: body}

    result, output_dir = mirror(site, tmp_path, cache, pages=[PAGE_A])

    assert site.requested.count(logo) == 1
    assert result.assets_written == 1
    assert asset_filename(logo) in read(output_dir, "index.html")
    assert asset_filename(logo) in read_page(output_dir, PAGE_A)


def test_failed_asset_is_recorded_and_its_attribute_removed(tmp_path):
    """
    The original scraper swallowed this and left the live URL in the
    page. Recording without removing would still leave the live request;
    removing without recording would hide an incomplete mirror.

    The outcome travels with the failure so the caller can tell a site
    404 (expected on this corpus, and identical on every run) from a
    local error (which means the corpus is no longer reproducible)
    without parsing the reason text.
    """
    cache = {HOME: '<html><body><img src="/missing.png" alt="x"></body></html>'}

    result, output_dir = mirror(FakeSite(), tmp_path, cache)
    html = read(output_dir, "index.html")

    assert result.failures == [
        ("https://www.b-tu.de/missing.png", "http_error", "HTTP 404")
    ]
    assert "missing.png" not in html
    assert "src=" not in html
    assert 'alt="x"' in html  # the tag itself survives; only the URL dies


def test_failed_css_reference_becomes_about_blank(tmp_path):
    """
    An unresolvable url() must not stay live, and must not become url("")
    either -- an empty CSS url resolves to the document, making the
    browser request the page again mid-load.
    """
    css = "https://www.b-tu.de/main.css"
    site = FakeSite({css: b"body{background:url(/gone.png)}"})
    cache = {HOME: '<html><head><link rel="stylesheet" href="/main.css"></head></html>'}

    result, output_dir = mirror(site, tmp_path, cache)
    written = asset_text(output_dir, css)

    assert 'url("about:blank")' in written
    assert "gone.png" not in written
    assert "b-tu.de" not in written
    assert (
        "https://www.b-tu.de/gone.png",
        "http_error",
        "HTTP 404",
    ) in result.failures


def test_off_host_iframe_is_recorded_and_removed(tmp_path):
    """
    A YouTube or OSM embed loads on its own. Left in, it is real WAN
    traffic during capture: DNS, TLS and jitter landing in Firefox's
    timings only, because wget never loads it.
    """
    cache = {
        HOME: '<html><body><iframe src="https://www.youtube.com/embed/abc"></iframe></body></html>'
    }

    result, output_dir = mirror(FakeSite(), tmp_path, cache)
    html = read(output_dir, "index.html")

    assert "youtube.com" not in html
    assert "<iframe></iframe>" in html
    assert len(result.failures) == 1
    url, outcome, reason = result.failures[0]
    # blocked_host is a LOCAL outcome: an embed that leaves the host is
    # not a fact about b-tu.de's content, and the run must fail on it.
    assert outcome == "blocked_host"
    assert "off-host" in reason


def test_css_references_siblings_while_pages_use_the_assets_prefix(tmp_path):
    """
    Pages sit at the mirror root, stylesheets inside assets/. One prefix
    for both produces a mirror that looks complete in a file listing and
    404s every image in the browser.
    """
    css = "https://www.b-tu.de/main.css"
    background = "https://www.b-tu.de/bg.png"
    logo = "https://www.b-tu.de/logo.png"
    site = FakeSite({css: b"body{background:url(/bg.png)}", background: b"P", logo: b"L"})
    cache = {
        HOME: (
            '<html><head><link rel="stylesheet" href="/main.css"></head>'
            '<body><img src="/logo.png"></body></html>'
        )
    }

    _, output_dir = mirror(site, tmp_path, cache)

    assert f'url("{asset_filename(background)}")' in asset_text(output_dir, css)
    assert f'src="assets/{asset_filename(logo)}"' in read(output_dir, "index.html")


def test_circular_css_import_terminates(tmp_path):
    """
    Two stylesheets importing each other must not spin: a hung scrape is
    a scrape that never produces a corpus, and a retry storm on someone
    else's server.
    """
    first = "https://www.b-tu.de/a.css"
    second = "https://www.b-tu.de/b.css"
    site = FakeSite({first: b'@import "b.css";', second: b'@import "a.css";'})
    cache = {HOME: '<html><head><link rel="stylesheet" href="/a.css"></head></html>'}

    result, output_dir = mirror(site, tmp_path, cache)

    assert result.assets_written == 2
    assert site.requested.count(first) == 1
    assert asset_filename(second) in asset_text(output_dir, first)
    assert asset_filename(first) in asset_text(output_dir, second)


def test_css_nesting_beyond_the_depth_limit_fails_loudly(tmp_path):
    """
    Stopping at the limit by leaving the URL absolute would trade a
    bounded crawl for a live request during capture. The limit must
    produce a recorded failure instead.
    """
    chain = {
        "https://www.b-tu.de/a.css": b'@import "b.css";',
        "https://www.b-tu.de/b.css": b'@import "c.css";',
        "https://www.b-tu.de/c.css": b"p{color:red}",
    }
    site = FakeSite(chain)
    cache = {HOME: '<html><head><link rel="stylesheet" href="/a.css"></head></html>'}

    result, output_dir = mirror(site, tmp_path, cache, max_css_depth=1)
    deepest = asset_text(output_dir, "https://www.b-tu.de/b.css")

    assert "https://www.b-tu.de/c.css" not in site.requested
    assert 'url("about:blank")' in deepest
    assert "c.css" not in deepest
    assert result.failures == [
        (
            "https://www.b-tu.de/c.css",
            "depth_exceeded",
            "nested deeper than max_css_depth=1",
        )
    ]


def test_srcset_keeps_descriptors_and_survives_data_uris(tmp_path):
    """
    Descriptors decide which image the browser actually requests, so
    dropping them changes Firefox's request pattern and leaves wget's
    untouched -- a difference the model could learn instead of client
    behaviour. Splitting on "," alone also corrupts data: URIs.
    """
    logo = "https://www.b-tu.de/logo.png"
    wide = "https://www.b-tu.de/wide.png"
    site = FakeSite({logo: b"L", wide: b"W"})
    cache = {
        HOME: (
            '<html><body><img srcset="/logo.png 1x, data:image/gif;base64,AA,A 2x, '
            '/wide.png 800w"></body></html>'
        )
    }

    _, output_dir = mirror(site, tmp_path, cache)
    html = read(output_dir, "index.html")

    assert f"assets/{asset_filename(logo)} 1x" in html
    assert f"assets/{asset_filename(wide)} 800w" in html
    assert "data:image/gif;base64,AA,A 2x" in html


def test_style_block_with_a_comment_is_still_rewritten(tmp_path):
    """
    tag.string is None whenever a <style> block is not exactly one child,
    which a CSS comment can cause -- and the block would be skipped in
    silence with every url() in it still live. Escaping ">" would break
    the child selector and change what the browser renders and requests.
    """
    logo = "https://www.b-tu.de/logo.png"
    site = FakeSite({logo: b"L"})
    cache = {
        HOME: "<html><head><style>/* & */\ndiv > p{background:url(/logo.png)}"
              "</style></head></html>"
    }

    _, output_dir = mirror(site, tmp_path, cache)
    html = read(output_dir, "index.html")

    assert f'url("assets/{asset_filename(logo)}")' in html
    assert "div > p" in html
    assert "&gt;" not in html
    assert "/* & */" in html


# --------------------------------------------------------------
# Reproducibility and robustness
# --------------------------------------------------------------

def test_output_is_byte_for_byte_reproducible(tmp_path):
    """
    The mirror is gitignored, so the README's claim that the scripts
    regenerate it is only true if the same input yields the same bytes.
    Otherwise "reproducible corpus" is prose, not a property.
    """
    cache = {HOME: EVERYTHING_PAGE, PAGE_A: "<html><body>a</body></html>"}

    first, first_dir = mirror(
        FakeSite(EVERYTHING_ASSETS), tmp_path / "one", cache, pages=[PAGE_A]
    )
    second, second_dir = mirror(
        FakeSite(EVERYTHING_ASSETS), tmp_path / "two", cache, pages={PAGE_A}
    )

    written = sorted(p.relative_to(first_dir) for p in first_dir.rglob("*") if p.is_file())
    assert written == sorted(
        p.relative_to(second_dir) for p in second_dir.rglob("*") if p.is_file()
    )
    for relative in written:
        assert (first_dir / relative).read_bytes() == (second_dir / relative).read_bytes()

    assert first.failures == second.failures
    assert first.bytes_total == second.bytes_total
    assert first.pages_written == second.pages_written


def test_page_missing_from_the_cache_is_recorded_not_fatal(tmp_path):
    """
    One uncached page must not abandon 99 good ones -- and must not pass
    unnoticed either, since a page in the set but not on disk is a 404
    the capture harness would record as a page load.
    """
    cache = {HOME: "<html></html>"}

    result, output_dir = mirror(FakeSite(), tmp_path, cache, pages=[PAGE_A, PAGE_B])

    assert result.pages_written == 1
    assert sorted(result.failures) == [
        (PAGE_A, "missing_html", "no cached HTML for this page"),
        (PAGE_B, "missing_html", "no cached HTML for this page"),
    ]
    assert not (output_dir / page_filename(PAGE_A)).exists()


@pytest.mark.parametrize(
    "markup, attribute",
    [
        ('<script src="/x.js"></script>', "src"),
        ('<embed src="/x.svg">', "src"),
        ('<object data="/x.svg"></object>', "data"),
        ('<audio src="/x.mp4"></audio>', "src"),
        ('<video poster="/x.jpg"></video>', "poster"),
        ('<track src="/x.vtt">', "src"),
        ('<img data-original="/x.png">', "data-original"),
        ('<img data-lazy-src="/x.png">', "data-lazy-src"),
    ],
)
def test_every_auto_fetching_attribute_is_neutralised_on_failure(
    tmp_path, markup, attribute
):
    """
    Each of these is fetched by the browser without user interaction, so
    each is a live request during capture if it survives a failed mirror.
    """
    cache = {HOME: f"<html><body>{markup}</body></html>"}

    result, output_dir = mirror(FakeSite(), tmp_path, cache)

    assert len(result.failures) == 1
    assert f"{attribute}=" not in read(output_dir, "index.html")
