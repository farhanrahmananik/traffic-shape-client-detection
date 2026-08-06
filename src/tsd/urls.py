"""
urls.py
-------
URL normalisation and filtering for the corpus mirror.

Why this is a module of its own:

    The corpus target is "~100 unique pages". Uniqueness is decided
    entirely by how URLs are compared, and the obvious comparison --
    string equality -- is wrong. These are one page:

        https://www.b-tu.de/fakultaet1/
        https://www.b-tu.de/fakultaet1
        https://www.b-tu.de/fakultaet1/?utm_source=newsletter
        https://www.b-tu.de/Fakultaet1/
        https://WWW.B-TU.DE/fakultaet1/

    If they are counted separately, the same page is mirrored more than
    once under different local filenames. Later, that same page appears
    as several capture traces. Split those across train and test and the
    model is scored on a page it has already seen -- exactly the leakage
    the round-based split exists to prevent.

    So normalisation is not tidiness. It is part of the experimental
    design, and it gets its own tests.

Canonicalisation rules applied here:
    - scheme and host lowercased (host is case-insensitive per RFC 3986)
    - fragment dropped     (#section is client-side, same resource)
    - tracking params dropped
    - remaining query params sorted
    - trailing slash normalised
    - path case PRESERVED  (paths are case-sensitive on many servers;
                            b-tu.de is TYPO3 and treats them so)
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

BASE_URL = "https://www.b-tu.de/"
DOMAIN = "www.b-tu.de"

# Query parameters that identify a campaign, not a resource. Dropping
# them merges URLs that serve identical content.
TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid",
        "ref", "referrer", "source",
    }
)

# Links that are not fetchable resources at all.
SKIP_SCHEMES = ("#", "data:", "mailto:", "tel:", "javascript:", "about:")

# Extensions that are not HTML pages. Assets are mirrored separately;
# these must not become walk targets or be counted toward the 100.
NON_PAGE_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".tar", ".gz", ".7z",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".css", ".js", ".json", ".xml", ".rss",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".webm",
    ".woff", ".woff2", ".ttf", ".eot",
)

# The corpus stays on the German pages. Mixing languages would mean the
# two clients load pages that differ in size for reasons unrelated to
# client behaviour.
ENGLISH_PATH_PREFIXES = ("/en/",)
ENGLISH_PATHS = ("/en",)
ENGLISH_QUERY_MARKERS = (("l", "1"), ("lang", "en"))

_FILENAME_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")


# --------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------

def normalise(url: str) -> str:
    """
    Return the canonical form of a URL.

    Two URLs naming the same resource must normalise to the same string.
    This is the single definition of "same page" used by the whole
    project.
    """
    parsed = urlparse(url.strip())

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports: they name the same origin.
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[: -len(":443")]
    elif scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[: -len(":80")]

    path = _normalise_path(parsed.path)
    query = _normalise_query(parsed.query)

    # Fragment is always dropped: it never reaches the server.
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def _normalise_path(path: str) -> str:
    """
    Collapse duplicate slashes and normalise the trailing slash.

    A path is treated as a directory (trailing slash) unless its last
    segment looks like a file with an extension. Case is preserved.
    """
    if not path:
        return "/"

    path = re.sub(r"/{2,}", "/", path)

    if path == "/":
        return path

    last_segment = path.rstrip("/").rsplit("/", 1)[-1]
    looks_like_file = "." in last_segment

    if looks_like_file:
        return path.rstrip("/")

    return path if path.endswith("/") else path + "/"


def _normalise_query(query: str) -> str:
    """Drop tracking parameters and sort what remains, for stable ordering."""
    if not query:
        return ""

    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]

    return urlencode(sorted(pairs))


def resolve(base_url: str, link: str) -> str | None:
    """
    Turn a raw href into an absolute, normalised URL.

    Returns None for links that are not fetchable resources.
    """
    if not link:
        return None

    stripped = link.strip()
    if not stripped or stripped.lower().startswith(SKIP_SCHEMES):
        return None

    return normalise(urljoin(base_url, stripped))


# --------------------------------------------------------------
# Filtering
# --------------------------------------------------------------

def is_corpus_page(url: str) -> bool:
    """
    True if this URL is an HTML page belonging in the corpus.

    Note this is only the shape check. robots.txt is enforced separately
    by PoliteFetcher -- deliberately not duplicated here, so there is
    exactly one place that decides what is allowed.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if parsed.netloc.lower() != DOMAIN:
        return False

    path = parsed.path

    if path.lower().endswith(NON_PAGE_EXTENSIONS):
        return False

    if _is_english(parsed):
        return False

    return True


def is_homepage(url: str) -> bool:
    """True if this URL is the site root, in any of its spellings."""
    parsed = urlparse(normalise(url))
    return parsed.netloc.lower() == DOMAIN and parsed.path == "/" and not parsed.query


def _is_english(parsed) -> bool:
    """Detect the English variant of a BTU page."""
    path = parsed.path.lower()

    if path in ENGLISH_PATHS or path.startswith(ENGLISH_PATH_PREFIXES):
        return True

    params = {k.lower(): v.lower() for k, v in parse_qsl(parsed.query)}
    return any(params.get(key) == value for key, value in ENGLISH_QUERY_MARKERS)


# --------------------------------------------------------------
# Local filenames
# --------------------------------------------------------------

def url_digest(url: str) -> str:
    """
    Short stable digest of a NORMALISED url.

    sha256 rather than md5: no security claim is being made here, but a
    security project should not model bad habits, and the cost is nil.
    """
    return hashlib.sha256(normalise(url).encode("utf-8")).hexdigest()[:12]


def page_filename(url: str) -> str:
    """
    Local .html filename for a page.

    Readable prefix so the mirror can be inspected by eye, plus a digest
    so two long URLs sharing a prefix cannot collide.
    """
    if is_homepage(url):
        return "index.html"

    normalised = normalise(url)
    parsed = urlparse(normalised)

    readable = _FILENAME_UNSAFE.sub("_", parsed.path.strip("/")) or "page"
    return f"{readable[:80]}_{url_digest(normalised)}.html"


def asset_filename(url: str) -> str:
    """Local filename for one asset, digest-first to guarantee uniqueness."""
    parsed = urlparse(normalise(url))
    original = _FILENAME_UNSAFE.sub("_", parsed.path.rsplit("/", 1)[-1]) or "asset"
    return f"{url_digest(url)}_{original[:60]}"
