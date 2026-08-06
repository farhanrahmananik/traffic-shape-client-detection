"""
test_urls.py
------------
Tests for tsd.urls.

These are corpus-integrity tests, not string-formatting tests. Each
equivalence assertion below is a claim about which URLs count as the
same page -- and therefore about whether the ~100 pages are really
100 distinct pages.
"""

import pytest

from tsd.urls import (
    asset_filename,
    is_corpus_page,
    is_homepage,
    normalise,
    page_filename,
    resolve,
    url_digest,
)


BASE = "https://www.b-tu.de/"


# --------------------------------------------------------------
# Equivalence: these must all collapse to one page
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "variant",
    [
        "https://www.b-tu.de/fakultaet1",
        "https://www.b-tu.de/fakultaet1/",
        "https://www.b-tu.de/fakultaet1//",
        "https://WWW.B-TU.DE/fakultaet1/",
        "https://www.b-tu.de/fakultaet1/#kontakt",
        "https://www.b-tu.de/fakultaet1/?utm_source=newsletter",
        "https://www.b-tu.de/fakultaet1/?fbclid=abc123",
        "https://www.b-tu.de:443/fakultaet1/",
        "  https://www.b-tu.de/fakultaet1/  ",
    ],
)
def test_variants_normalise_to_one_url(variant):
    assert normalise(variant) == "https://www.b-tu.de/fakultaet1/"


def test_equivalent_urls_share_a_filename():
    """The mirror must not write the same page under two names."""
    a = page_filename("https://www.b-tu.de/fakultaet1?utm_source=x")
    b = page_filename("https://www.b-tu.de/fakultaet1/")
    assert a == b


# --------------------------------------------------------------
# Non-equivalence: these must stay distinct
# --------------------------------------------------------------

def test_path_case_is_preserved():
    """
    TYPO3 on Linux serves case-sensitive paths, so /Fakultaet1/ and
    /fakultaet1/ may be different resources. Merging them would shrink
    the corpus silently.
    """
    assert normalise("https://www.b-tu.de/Fakultaet1/") != normalise(
        "https://www.b-tu.de/fakultaet1/"
    )


def test_meaningful_query_params_are_kept():
    result = normalise("https://www.b-tu.de/suche?q=cyber")
    assert "q=cyber" in result


def test_query_params_are_sorted_for_stability():
    a = normalise("https://www.b-tu.de/p?b=2&a=1")
    b = normalise("https://www.b-tu.de/p?a=1&b=2")
    assert a == b


def test_different_pages_get_different_filenames():
    assert page_filename("https://www.b-tu.de/a/") != page_filename(
        "https://www.b-tu.de/b/"
    )


def test_long_urls_sharing_a_prefix_do_not_collide():
    """The readable prefix is truncated, so the digest must disambiguate."""
    long_a = "https://www.b-tu.de/" + "x" * 120 + "/one/"
    long_b = "https://www.b-tu.de/" + "x" * 120 + "/two/"
    assert page_filename(long_a) != page_filename(long_b)


# --------------------------------------------------------------
# Path normalisation edge cases
# --------------------------------------------------------------

def test_file_paths_do_not_gain_a_trailing_slash():
    assert normalise("https://www.b-tu.de/page.html") == (
        "https://www.b-tu.de/page.html"
    )


def test_empty_path_becomes_root():
    assert normalise("https://www.b-tu.de") == "https://www.b-tu.de/"


# --------------------------------------------------------------
# Corpus membership
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://www.b-tu.de/en/faculty1/",
        "https://www.b-tu.de/en",
        "https://www.b-tu.de/fakultaet1/?l=1",
        "https://www.b-tu.de/fakultaet1/?lang=en",
    ],
)
def test_english_pages_are_excluded(url):
    """Mixing languages would vary page size for reasons unrelated to the client."""
    assert not is_corpus_page(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.b-tu.de/doc.pdf",
        "https://www.b-tu.de/style.css",
        "https://www.b-tu.de/logo.PNG",
        "https://www.b-tu.de/app.js",
    ],
)
def test_non_html_resources_are_not_corpus_pages(url):
    assert not is_corpus_page(url)


def test_other_hosts_are_excluded():
    assert not is_corpus_page("https://www.tu-dresden.de/page/")
    assert not is_corpus_page("https://b-tu.de/page/")  # no www: different host


def test_ordinary_german_page_is_included():
    assert is_corpus_page("https://www.b-tu.de/fakultaet1/forschung/")


def test_homepage_detection():
    assert is_homepage("https://www.b-tu.de")
    assert is_homepage("https://www.b-tu.de/")
    assert is_homepage("https://www.b-tu.de/#top")
    assert not is_homepage("https://www.b-tu.de/fakultaet1/")


# --------------------------------------------------------------
# resolve()
# --------------------------------------------------------------

def test_relative_links_resolve_against_the_page():
    assert resolve("https://www.b-tu.de/a/b/", "../c/") == "https://www.b-tu.de/a/c/"


@pytest.mark.parametrize(
    "link",
    ["", "   ", "#section", "mailto:x@b-tu.de", "tel:+49", "javascript:void(0)",
     "data:image/png;base64,AAAA"],
)
def test_non_fetchable_links_resolve_to_none(link):
    assert resolve(BASE, link) is None


# --------------------------------------------------------------
# Filenames
# --------------------------------------------------------------

def test_homepage_filename_is_index():
    assert page_filename("https://www.b-tu.de/") == "index.html"
    assert page_filename("https://www.b-tu.de") == "index.html"


def test_page_filenames_are_filesystem_safe():
    name = page_filename("https://www.b-tu.de/a b/c?d=e&f=g")
    assert "/" not in name and " " not in name and "?" not in name
    assert name.endswith(".html")


def test_asset_filenames_are_unique_per_url():
    a = asset_filename("https://www.b-tu.de/a/logo.png")
    b = asset_filename("https://www.b-tu.de/b/logo.png")
    assert a != b
    assert a.endswith("logo.png") and b.endswith("logo.png")


def test_digest_is_stable_across_url_spellings():
    assert url_digest("https://www.b-tu.de/p") == url_digest(
        "https://WWW.B-TU.DE/p/?utm_source=x"
    )
