#!/usr/bin/env python3
"""
scrape_corpus.py
----------------
Build the corpus: discover ~100 unique b-tu.de pages, mirror them with
their assets into data/mirror/, and write results/corpus_manifest.json.

Run it from the repository root, with src/ on the import path. This
script deliberately does NOT touch sys.path -- a script that edits its
own import path hides a broken layout instead of reporting it, and the
same PYTHONPATH is what pytest.ini already declares:

    PYTHONPATH=src python scripts/scrape_corpus.py --dry-run
    PYTHONPATH=src python scripts/scrape_corpus.py

If you pipe the output anywhere -- `| tee`, `| less` -- set pipefail
first, or the exit code you read is the pipe's, not this script's. See
the note under "Exit codes" below; the first full run lost its gate that
way.

Output streams are split on purpose: progress goes to stderr, and the
final summary to stdout, so the summary can be piped or captured while
the crawl is still narrating itself.

Exit codes:
    0   corpus written; only upstream and/or excluded failures
    1   corpus written but NOT REPRODUCIBLE -- local failures occurred
    2   aborted before writing anything (robots.txt unreadable, or
        discovery found no pages)

    Note when piping: `script.py | tee log.txt` reports tee's exit code,
    not this script's, so the gate below fires while $? reads 0. That
    happened on the first full run. Use `set -o pipefail`, or read
    ${PIPESTATUS[0]}:

        set -o pipefail
        PYTHONPATH=src python scripts/scrape_corpus.py | tee scrape.log

Why failures are split into three classes:

    The distinction that matters is not "did it work" but "will the next
    run produce the same corpus". Only one of these three classes says
    no.

    upstream -- properties of b-tu.de itself: assets the site 404s (its
        own CSS references jQuery-UI images that are not deployed), and
        URLs robots.txt tells us not to fetch. Identical on every run.

    excluded -- deterministic too, but withheld by OUR policy rather
        than by the site: off-host assets (all of them on
        www-docs.b-tu.de, BTU's own document server) and responses over
        the 8 MB ceiling. The URL and the file size do not change
        between runs, so neither does this list.

    local -- properties of THIS run: a connection error, a file that
        could not be written. Run the script again and you get a
        different mirror.

    Only the third is an alarm. The measured first run makes the case:
    29 upstream + 8 "local" under the old two-class split, of which 7
    were in fact deterministic. A gate that fires on all 8 is a gate
    that fires on every run, and a warning that always fires is one
    people learn to scroll past -- taking the one that matters with it.

    Upstream and excluded still mean the mirror is missing something the
    real page has. That is a deficiency of the mirror against the live
    site, not a class-confounding artefact: both clients load the same
    mirror, so a missing asset is missing for Firefox and wget alike.
    It belongs in the README's limitations, not in an alarm.

    The split is made on FetchRecord.outcome, never on the reason text.
    Reason strings are written for people and get reworded; an outcome
    is a value. Anything unrecognised counts as local -- the safe
    direction to be wrong in is "too loud".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tsd.discover import CorpusDiscoverer
from tsd.fetcher import MAX_RESPONSE_BYTES, PoliteFetcher
from tsd.mirror import SiteMirror
from tsd.robots import USER_AGENT, RobotsError, RobotsPolicy
from tsd.urls import BASE_URL

DEFAULT_OUTPUT_DIR = Path("data/mirror")
DEFAULT_MANIFEST = Path("results/corpus_manifest.json")
PROVENANCE_DIR = Path("results/provenance")

EXIT_OK = 0
EXIT_NOT_REPRODUCIBLE = 1
EXIT_ABORTED = 2

# Failure outcomes that are properties of b-tu.de rather than of this
# run: the site's own 404s, and the URLs robots.txt refuses us.
UPSTREAM_OUTCOMES = frozenset({"http_error", "blocked_robots"})

# Deterministic as well, but withheld by our own policy rather than by
# the site. The URL of an off-host asset and the size of a file do not
# change between runs, so this list does not either.
#
# Caveat on blocked_host: PoliteFetcher emits it for two different
# situations -- a URL that was off-host to begin with (a third-party or
# sibling-host asset, deterministic) and a request that REDIRECTED
# off-host mid-flight, which can be a genuine anomaly. They share one
# outcome value, so both land here for now. Distinguishing them means a
# separate outcome in fetcher.py, not parsing the reason text; worth
# doing if a redirect-driven one ever shows up. On the first full run,
# all 5 were www-docs.b-tu.de, BTU's own document server.
EXCLUDED_OUTCOMES = frozenset({"blocked_host", "too_large"})

# Everything else -- error, write_error, missing_html, depth_exceeded --
# is local. Unknown outcomes fall through to local on purpose: a new
# failure mode should be noticed, not silently absorbed into a list of
# things we already decided not to care about.


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and mirror the b-tu.de corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-pages", type=int, default=100)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fixes the walk; the same seed must reproduce the same corpus",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover only: fetch robots.txt and walk, write nothing to disk",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------

class Progress:
    """
    Narrates the crawl on stderr and collects what was written.

    The collecting half exists because MirrorResult reports counts, not
    an inventory, and the manifest needs one URL-to-filename row per
    file. Taking it from the events keeps mirror.py free of manifest
    concerns and avoids reaching into its internals from here.
    """

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.pages: list[tuple[str, str]] = []
        self.assets: list[tuple[str, str]] = []

    def say(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def discovery_event(self, kind: str, *payload) -> None:
        if kind == "walk_start":
            walk_number, found = payload
            self.say(f"  walk {walk_number}: {found} pages so far")
        elif kind == "page_found":
            url, found, depth = payload
            self.say(f"  [{found:>3}] depth {depth} {url}")
        elif kind in ("refused", "failed"):
            url, reason = payload
            self.say(f"  skip  {url} ({reason})")
        elif kind == "walk_exhausted":
            (restarts,) = payload
            self.say(f"  walk abandoned after {restarts} restarts")

    def mirror_event(self, kind: str, *payload) -> None:
        if kind == "page_written":
            url, filename, size = payload
            self.pages.append((url, filename))
            self.say(f"  page  {filename} ({size} bytes) <- {url}")
        elif kind == "asset_written":
            url, filename, size = payload
            self.assets.append((url, filename))
            self.say(f"  asset {filename} ({size} bytes)")
        elif kind == "failure":
            url, outcome, reason = payload
            self.say(f"  {classify(outcome):8} {url} [{outcome}] {reason}")


# --------------------------------------------------------------
# Manifest
# --------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def fetch_records_by_url(fetcher: PoliteFetcher) -> dict[str, dict]:
    """
    Index the fetcher's log by URL, keeping the last successful attempt.

    PoliteFetcher already logs status, content type and timestamp for
    every request it made; re-deriving them here would be a second
    source of truth for the same facts.
    """
    records: dict[str, dict] = {}

    for record in fetcher.log:
        if record.outcome == "ok":
            records[record.url] = record.to_dict()

    return records


def build_manifest(
    args: argparse.Namespace,
    policy: RobotsPolicy,
    robots_fetched_at: str,
    robots_provenance: Path | None,
    discovery,
    mirror_result,
    progress: Progress,
    output_dir: Path,
    fetcher: PoliteFetcher,
) -> dict:
    """
    Assemble the manifest.

    This file is what gets published INSTEAD of the mirror, so it must
    carry no BTU content -- only URLs, local filenames, sizes, hashes
    and timestamps.

    The hashes are the point of the whole document. Since the mirror is
    gitignored, "regenerate it by running the scripts" is an unverifiable
    claim on its own: anyone re-running this can compare their hashes
    against the published ones and see whether they got the same corpus,
    without either side ever publishing a byte of b-tu.de's content.
    """
    fetches = fetch_records_by_url(fetcher)

    pages = []
    for url, filename in sorted(progress.pages):
        path = output_dir / filename
        record = fetches.get(url, {})
        pages.append(
            {
                "url": url,
                "local_filename": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "status_code": record.get("status_code"),
                "fetched_at": record.get("fetched_at"),
            }
        )

    assets = []
    for url, filename in sorted(progress.assets):
        path = output_dir / "assets" / filename
        record = fetches.get(url, {})
        assets.append(
            {
                "url": url,
                "local_filename": f"assets/{filename}",
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "content_type": record.get("content_type"),
            }
        )

    refused = [
        {"url": url, "reason": reason}
        for url, reason in sorted(discovery.refused.items())
    ]
    failures = split_failures(mirror_result.failures)

    total_bytes = sum(entry["bytes"] for entry in pages) + sum(
        entry["bytes"] for entry in assets
    )

    return {
        "generated_at": utc_now(),
        "user_agent": USER_AGENT,
        "crawl_delay": policy.crawl_delay,
        "seed": args.seed,
        "walks": args.walks,
        "walks_run": discovery.walks_run,
        "max_depth": args.depth,
        "target_pages": args.target_pages,
        "robots_fetched_at": robots_fetched_at,
        "robots_sha256": hashlib.sha256(policy.raw_text.encode("utf-8")).hexdigest(),
        "robots_provenance_file": (
            robots_provenance.as_posix() if robots_provenance else None
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requests": package_version("requests"),
            "beautifulsoup4": package_version("beautifulsoup4"),
        },
        "pages": pages,
        "assets": assets,
        "refused": refused,
        "failures": failures,
        "totals": {
            "pages": len(pages),
            "assets": len(assets),
            "bytes": total_bytes,
            "refused": len(refused),
            "failures_upstream": len(failures["upstream"]),
            "failures_excluded": len(failures["excluded"]),
            "failures_local": len(failures["local"]),
        },
    }


def classify(outcome: str) -> str:
    """Which of the three failure classes an outcome belongs to."""
    if outcome in UPSTREAM_OUTCOMES:
        return "upstream"
    if outcome in EXCLUDED_OUTCOMES:
        return "excluded"
    return "local"


def split_failures(failures) -> dict[str, list[dict]]:
    """
    Sort mirror failures into upstream, excluded and local.

    Classified on the outcome value, never on the reason text -- see the
    module docstring. All three lists go into the manifest in full: the
    first two are part of what the corpus IS, and a reader checking
    reproducibility needs to see that they recurred unchanged.
    """
    split: dict[str, list[dict]] = {"upstream": [], "excluded": [], "local": []}

    for url, outcome, reason in failures:
        split[classify(outcome)].append(
            {"url": url, "outcome": outcome, "reason": reason}
        )

    return split


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def save_robots_provenance(policy: RobotsPolicy) -> Path:
    """
    Keep a timestamped copy of the robots.txt this run obeyed.

    Sites edit robots.txt. Without a copy, a future reader cannot check
    that the corpus respected the rules as they stood at crawl time --
    and since the mirror itself is not published, these small artefacts
    are what carry the verifiability.
    """
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = PROVENANCE_DIR / f"robots_{stamp}.txt"
    path.write_text(policy.raw_text, encoding="utf-8")
    return path


# --------------------------------------------------------------
# Run
# --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = Progress()

    # 1. robots.txt. Fail closed: if the rules cannot be read, nothing
    #    is crawled. A crawler that proceeds without them is not
    #    compliant, whatever its intentions.
    progress.say(f"robots.txt: fetching for {BASE_URL}")
    try:
        policy = RobotsPolicy.fetch(BASE_URL)
    except RobotsError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_ABORTED

    robots_fetched_at = utc_now()
    progress.say(f"robots.txt: ok, crawl delay {policy.crawl_delay}s")

    # 2. Discovery. Every request from here on goes through PoliteFetcher.
    with PoliteFetcher(policy=policy) as fetcher:
        progress.say(
            f"discovery: target {args.target_pages} pages, "
            f"{args.walks} walks, depth {args.depth}, seed {args.seed}"
        )

        discoverer = CorpusDiscoverer(
            fetcher=fetcher,
            target_pages=args.target_pages,
            total_walks=args.walks,
            max_depth=args.depth,
            seed=args.seed,
            on_event=progress.discovery_event,
        )

        try:
            discovery = discoverer.run()
        except RuntimeError as error:
            print(f"ABORT: {error}", file=sys.stderr)
            return EXIT_ABORTED

        if not discovery.pages:
            print("ABORT: discovery found no pages", file=sys.stderr)
            return EXIT_ABORTED

        if len(discovery.pages) < args.target_pages:
            progress.say(
                f"NOTE: {len(discovery.pages)} pages found, "
                f"target was {args.target_pages} -- the walk ran out of "
                f"reachable unseen pages"
            )

        if args.dry_run:
            print_dry_run_summary(args, policy, discovery)
            return EXIT_OK

        # 3. Mirror. Pages come from the discovery cache; only assets
        #    are fetched here.
        progress.say(f"mirror: writing to {args.output_dir}")
        mirror = SiteMirror(
            fetcher=fetcher,
            output_dir=args.output_dir,
            pages=discovery.pages,
            on_event=progress.mirror_event,
        )
        mirror_result = mirror.run(discovery.html_cache)

        # 4. Provenance and manifest.
        robots_provenance = save_robots_provenance(policy)
        manifest = build_manifest(
            args=args,
            policy=policy,
            robots_fetched_at=robots_fetched_at,
            robots_provenance=robots_provenance,
            discovery=discovery,
            mirror_result=mirror_result,
            progress=progress,
            output_dir=args.output_dir,
            fetcher=fetcher,
        )

    write_json(args.manifest, manifest)
    print_summary(args, manifest)

    upstream = manifest["failures"]["upstream"]
    excluded = manifest["failures"]["excluded"]
    local = manifest["failures"]["local"]

    # Both of these are deterministic, so neither threatens the claim
    # that the scripts regenerate the corpus. One line each, no alarm:
    # they are listed in full in the manifest, which is where anyone
    # checking them will look.
    if upstream:
        print(
            f"{len(upstream)} upstream failure(s) (site 404s / robots.txt "
            f"refusals) -- expected, listed in {args.manifest}",
            file=sys.stderr,
        )

    if excluded:
        print(
            f"{len(excluded)} excluded by policy (off-host / over the "
            f"{MAX_RESPONSE_BYTES // (1024 * 1024)} MB ceiling) -- "
            f"deterministic, listed in {args.manifest}",
            file=sys.stderr,
        )

    if local:
        print(
            f"\nNOT REPRODUCIBLE: {len(local)} local failure(s).\n"
            f"These depend on this run, not on b-tu.de, so another run would "
            f"produce a\ndifferent mirror -- and the affected pages are missing "
            f"a resource the real\npages have, which biases every trace captured "
            f"from them. Fix them and\nre-run before capturing:\n"
            + "\n".join(
                f"  [{entry['outcome']}] {entry['url']} -- {entry['reason']}"
                for entry in local
            ),
            file=sys.stderr,
        )
        return EXIT_NOT_REPRODUCIBLE

    return EXIT_OK


# --------------------------------------------------------------
# Summaries -- stdout, so they can be piped
# --------------------------------------------------------------

def print_dry_run_summary(args, policy: RobotsPolicy, discovery) -> None:
    print("dry run -- nothing written")
    print(f"  crawl delay   : {policy.crawl_delay}s")
    print(f"  seed          : {args.seed}")
    print(f"  pages found   : {len(discovery.pages)} (target {args.target_pages})")
    print(f"  walks run     : {discovery.walks_run}")
    print(f"  refused       : {len(discovery.refused)}")
    print(f"  would mirror  : {args.output_dir}")
    print(f"  would write   : {args.manifest}")


def print_summary(args, manifest: dict) -> None:
    totals = manifest["totals"]
    print("corpus written")
    print(f"  pages         : {totals['pages']}")
    print(f"  assets        : {totals['assets']}")
    print(f"  bytes         : {totals['bytes']}")
    print(f"  refused       : {totals['refused']}")
    print(f"  failures      : {totals['failures_upstream']} upstream, "
          f"{totals['failures_excluded']} excluded, "
          f"{totals['failures_local']} local")
    print(f"  mirror        : {args.output_dir}")
    print(f"  manifest      : {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
