"""
robots.py
---------
robots.txt compliance for the corpus mirror.

Why this module exists instead of using urllib.robotparser directly:

    Python's RobotFileParser matches rules by plain prefix. Its
    RuleLine.applies_to() does:

        return self.path == "*" or filename.startswith(self.path)

    b-tu.de/robots.txt contains a wildcard rule:

        Disallow: /*/wiki/

    Under prefix matching that rule can never fire, because no real
    URL path literally starts with the characters "/*/wiki/". Every
    department wiki on the site would be silently treated as allowed.

    This module therefore layers an explicit wildcard check on top of
    RobotFileParser: robotparser handles the plain prefix rules, and
    a compiled regex handles the wildcard ones. A path is fetched only
    if BOTH layers allow it.

Fail-closed: if robots.txt cannot be fetched or parsed, nothing is
allowed. A crawler that keeps going when it cannot read the rules is
not a compliant crawler.
"""

from __future__ import annotations

import re
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import requests

# --------------------------------------------------------------
# Identity
# --------------------------------------------------------------
# Honest, contactable User-Agent: who is calling, why, and where to
# complain. Deliberately NOT a browser string and NOT a generic
# library default -- b-tu.de explicitly blocks generic library agents
# such as "Python-urllib", "httplib" and "lwp-trivial".
#
# Verified: this token contains no substring matching any agent in the
# site's blocked list (robots.txt agent matching is substring-based on
# the part before "/", lowercased).
USER_AGENT = (
    "traffic-shape-client-detection/1.0 "
    "(research mirror; "
    "+https://github.com/farhanrahmananik/traffic-shape-client-detection)"
)

# robots.txt for b-tu.de declares no Crawl-delay. We impose one anyway.
DEFAULT_CRAWL_DELAY = 1.5

ROBOTS_TIMEOUT = 15


class RobotsError(RuntimeError):
    """robots.txt could not be fetched or parsed. Caller must abort."""


def _wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    """
    Translate one robots.txt path pattern into an anchored regex.

    robots.txt wildcard semantics (Google/RFC 9309 convention):
        *  matches any sequence of characters
        $  at the end anchors the match to end-of-path
        otherwise the pattern is a prefix

    Everything else is escaped literally, so a path containing regex
    metacharacters (".", "+", "(") cannot alter the matching.
    """
    anchored_end = pattern.endswith("$")
    body = pattern[:-1] if anchored_end else pattern

    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in body)
    return re.compile("^" + regex + ("$" if anchored_end else ""))


@dataclass
class RobotsPolicy:
    """Fetched, parsed robots.txt for exactly one host."""

    base_url: str
    user_agent: str = USER_AGENT
    crawl_delay: float = DEFAULT_CRAWL_DELAY
    raw_text: str = ""

    _parser: urllib.robotparser.RobotFileParser | None = field(
        default=None, repr=False
    )
    _wildcard_disallow: list[re.Pattern[str]] = field(
        default_factory=list, repr=False
    )
    _wildcard_allow: list[re.Pattern[str]] = field(
        default_factory=list, repr=False
    )

    # ----------------------------------------------------------
    # Construction
    # ----------------------------------------------------------

    @classmethod
    def fetch(cls, base_url: str, user_agent: str = USER_AGENT) -> "RobotsPolicy":
        """Fetch and parse robots.txt for the host of base_url."""
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise RobotsError(f"base_url is not absolute: {base_url!r}")

        robots_url = urlunparse(
            (parsed.scheme, parsed.netloc, "/robots.txt", "", "", "")
        )

        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": user_agent},
                timeout=ROBOTS_TIMEOUT,
            )
        except requests.RequestException as error:
            raise RobotsError(f"could not fetch {robots_url}: {error}") from error

        # 4xx conventionally means "no restrictions", but this project
        # mirrors someone else's site: we refuse to guess.
        if response.status_code != 200:
            raise RobotsError(
                f"{robots_url} returned HTTP {response.status_code}; refusing to crawl"
            )

        return cls.from_text(base_url, response.text, user_agent=user_agent)

    @classmethod
    def from_text(
        cls, base_url: str, text: str, user_agent: str = USER_AGENT
    ) -> "RobotsPolicy":
        """Parse robots.txt content. Separate from fetch() so it is testable."""
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(text.splitlines())

        policy = cls(
            base_url=base_url,
            user_agent=user_agent,
            raw_text=text,
            _parser=parser,
        )
        policy._load_wildcards(text)
        policy._load_crawl_delay(parser, user_agent)
        return policy

    # ----------------------------------------------------------
    # Wildcard layer
    # ----------------------------------------------------------

    def _load_wildcards(self, text: str) -> None:
        """
        Collect wildcard rules from every group that applies to us.

        A group applies if it names "*" or if one of its agent tokens is
        a substring of our User-Agent, which is how robots.txt agent
        matching is defined. Only rules containing "*" or ending in "$"
        are kept here -- plain prefix rules are already handled correctly
        by RobotFileParser, and duplicating them would risk drift.
        """
        current_agents: list[str] = []
        previous_was_agent = False
        ua_lower = self.user_agent.lower()

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue

            field_name, _, value = line.partition(":")
            field_name = field_name.strip().lower()
            value = value.strip()

            if field_name == "user-agent":
                if not previous_was_agent:
                    current_agents = []
                current_agents.append(value.lower())
                previous_was_agent = True
                continue

            previous_was_agent = False

            if field_name not in ("disallow", "allow"):
                continue

            applies = any(
                agent == "*" or agent.split("/")[0].strip() in ua_lower
                for agent in current_agents
            )
            if not applies or not value:
                continue

            if "*" not in value and not value.endswith("$"):
                continue  # plain prefix: RobotFileParser handles it

            compiled = _wildcard_to_regex(value)
            if field_name == "disallow":
                self._wildcard_disallow.append(compiled)
            else:
                self._wildcard_allow.append(compiled)

    def _load_crawl_delay(
        self, parser: urllib.robotparser.RobotFileParser, user_agent: str
    ) -> None:
        """Honour a declared Crawl-delay if it is longer than our own."""
        declared = parser.crawl_delay(user_agent)
        if declared is not None:
            self.crawl_delay = max(self.crawl_delay, float(declared))

    # ----------------------------------------------------------
    # Query
    # ----------------------------------------------------------

    def can_fetch(self, url: str) -> bool:
        """
        Return True only if BOTH the prefix layer and the wildcard layer
        allow this URL. Query strings are kept: robots.txt patterns are
        allowed to match against them.
        """
        if self._parser is None:
            return False

        if not self._parser.can_fetch(self.user_agent, url):
            return False

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # Allow wins over Disallow, per robots.txt convention.
        if any(rule.match(path) for rule in self._wildcard_allow):
            return True

        return not any(rule.match(path) for rule in self._wildcard_disallow)

    def why_blocked(self, url: str) -> str | None:
        """Human-readable reason a URL was refused, for logs and the manifest."""
        if self._parser is None:
            return "robots.txt not loaded"

        if not self._parser.can_fetch(self.user_agent, url):
            return "robots.txt prefix rule"

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        if any(rule.match(path) for rule in self._wildcard_allow):
            return None

        for rule in self._wildcard_disallow:
            if rule.match(path):
                return f"robots.txt wildcard rule {rule.pattern}"

        return None
