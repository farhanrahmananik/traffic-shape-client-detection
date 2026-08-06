"""
test_robots.py
--------------
Tests for tsd.robots.

The important case is the wildcard rule. urllib.robotparser cannot
express it, so if this module ever regresses to plain prefix matching,
test_wildcard_rule_blocks_department_wikis is what catches it.
"""

import pytest

from tsd.robots import RobotsPolicy, RobotsError, _wildcard_to_regex


BASE = "https://www.b-tu.de/"

ROBOTS_SAMPLE = """
User-agent: *
Disallow: /media/
Disallow: /typo3/
Disallow: /fg-kommunikationstechnik/wiki/
Disallow: /*/wiki/

User-agent: EvilBot
Disallow: /
"""


@pytest.fixture
def policy():
    return RobotsPolicy.from_text(BASE, ROBOTS_SAMPLE)


# --------------------------------------------------------------
# Prefix rules -- handled by RobotFileParser
# --------------------------------------------------------------

def test_ordinary_page_is_allowed(policy):
    assert policy.can_fetch("https://www.b-tu.de/fakultaet1/")
    assert policy.why_blocked("https://www.b-tu.de/fakultaet1/") is None


def test_prefix_rule_blocks_media(policy):
    url = "https://www.b-tu.de/media/logo.png"
    assert not policy.can_fetch(url)
    assert "prefix" in policy.why_blocked(url)


# --------------------------------------------------------------
# Wildcard rules -- the reason this module exists
# --------------------------------------------------------------

def test_wildcard_rule_blocks_department_wikis(policy):
    """
    Regression guard. Disallow: /*/wiki/ must block ANY department
    wiki, not only the one path spelled out literally in robots.txt.
    Plain prefix matching would let these through.
    """
    for url in (
        "https://www.b-tu.de/lehrstuhl-irgendwas/wiki/seite",
        "https://www.b-tu.de/fg-informatik/wiki/",
        "https://www.b-tu.de/a/wiki/deep/nested/page.html",
    ):
        assert not policy.can_fetch(url), url
        assert "wildcard" in policy.why_blocked(url), url


def test_wiki_not_at_second_level_is_allowed(policy):
    """The pattern requires a segment before /wiki/, so /wiki/ alone is fine."""
    assert policy.can_fetch("https://www.b-tu.de/wiki/")


# --------------------------------------------------------------
# Pattern translation
# --------------------------------------------------------------

def test_dollar_anchors_end_of_path():
    pattern = _wildcard_to_regex("/*.pdf$")
    assert pattern.match("/docs/report.pdf")
    assert not pattern.match("/docs/report.pdf.html")


def test_metacharacters_are_escaped():
    """A literal dot must not behave as regex 'any character'."""
    pattern = _wildcard_to_regex("/clear.gif")
    assert pattern.match("/clear.gif")
    assert not pattern.match("/clearXgif")


def test_plain_prefix_pattern_matches_prefix():
    pattern = _wildcard_to_regex("/media/")
    assert pattern.match("/media/logo.png")
    assert not pattern.match("/mediation/")


# --------------------------------------------------------------
# Fail-closed behaviour
# --------------------------------------------------------------

def test_unparsed_policy_denies_everything():
    empty = RobotsPolicy(base_url=BASE)
    assert not empty.can_fetch("https://www.b-tu.de/anything")


def test_relative_base_url_is_rejected():
    with pytest.raises(RobotsError):
        RobotsPolicy.fetch("www.b-tu.de")


# --------------------------------------------------------------
# Crawl delay
# --------------------------------------------------------------

def test_declared_delay_wins_when_longer():
    policy = RobotsPolicy.from_text(
        BASE, "User-agent: *\nCrawl-delay: 10\nDisallow: /media/\n"
    )
    assert policy.crawl_delay == 10.0


def test_our_delay_wins_when_site_declares_none(policy):
    assert policy.crawl_delay == 1.5
