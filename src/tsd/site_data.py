"""
site_data.py
------------
Assemble `docs/data/case_study.json` from the measured artefacts in
`results/`.

**This is the only thing allowed to put numbers into the site.** No
figure on the case-study page may ever be typed by hand: the page
fetches this file, and this file is built from the JSON the measurement
scripts wrote. A number that a human retyped is a number that will be
right today and wrong after the next round, silently, because nothing
checks prose against data.

Four rules follow from that, and they are the whole design:

**Measured values are copied, never recomputed.** No rounding, no
reformatting, no re-deriving. A float appears here exactly as
`metrics.json` spells it. Re-deriving would mean two implementations of
one number, and the page would eventually disagree with the metrics it
claims to report.

**Every path read here is required.** A missing key raises
`SiteDataError` naming the file and the dotted path. There is no default
and no `null`: a page that renders "accuracy: null" looks like a
measurement of zero confidence rather than a broken build, and the
build is where that should be caught.

**Derived values are computed from the sources, never asserted.** The
rank movement, the top-ten overlap, the family shares, the capture
invariants -- all of them are read off the files at build time. That is
what lets the page say "the rounds are comparable" as a *finding* rather
than as a claim: `capture_invariants` reports the distinct values seen
and whether they were identical, so a round captured with a different
snaplen would show up as a fact on the page instead of being averaged
away.

**No wall-clock timestamp of its own.** Each source's own
`generated_at` is carried through instead. A build time would change on
every run and make the output undiffable, which would hide the thing a
diff is for -- noticing that a number moved.

Never read from here: `data/mirror`, `data/pcaps`, and the `pages` and
`assets` lists of `corpus_manifest.json`. Those hold BTU content or
derive from unpublished captures; only the manifest's scalar fields and
its `totals` are used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = Path("docs/data/case_study.json")

CAPTURE_ROUNDS_DIRNAME = "capture_rounds"

# Source files, by the name their dotted paths start with.
SOURCE_FILES = {
    "metrics": "metrics.json",
    "shap": "shap_summary.json",
    "cli_parity": "cli_parity.json",
    "corpus_manifest": "corpus_manifest.json",
}

# Display slices, not measurements: how many features the page shows in
# its seed-comparison list, and how many families count as "the top".
# Both describe the presentation; every value inside them is measured.
TOP_FEATURES = 10
TOP_FAMILIES = 2

# The per-round fields the page uses to claim the rounds are comparable.
# Anything that differs between rounds would change what a trace looks
# like, so it is exactly this list that has to be shown as identical
# rather than assumed to be.
INVARIANT_FIELDS = (
    "snaplen",
    "tcpdump_filter",
    "capture_interface",
    "quiet_seconds",
    "server_cert_sha256",
)

# Per-round keys copied through. `traces` is deliberately absent: it is
# 200 entries per round of page-level detail the page does not use, and
# it would multiply the file size by an order of magnitude.
ROUND_FIELDS = (
    "round",
    "date",
    "started_at",
    "finished_at",
    "snaplen",
    "tcpdump_filter",
    "capture_interface",
    "quiet_seconds",
    "max_load_seconds",
    "network_namespace",
    "server_cert_sha256",
    "versions",
    "totals",
)

MODEL_FIELDS = (
    "accuracy",
    "fold_accuracies",
    "confusion_matrix",
    "labels",
    "n_folds",
    "misclassified",
    "per_class",
)

CORPUS_FIELDS = (
    "user_agent",
    "crawl_delay",
    "seed",
    "walks",
    "walks_run",
    "max_depth",
    "target_pages",
    "robots_fetched_at",
    "robots_sha256",
    "robots_provenance_file",
    "environment",
    "totals",
    "failures",
)

SHAP_FIELDS = (
    "model",
    "seeds",
    "split",
    "n_folds",
    "n_traces",
    "n_features",
    "direction",
    "family_importance",
    "spreads",
    "zero_importance",
)


class SiteDataError(RuntimeError):
    """A required source file or key is missing."""


# --------------------------------------------------------------
# Required lookups
# --------------------------------------------------------------

def require(sources: dict, dotted: str):
    """
    Read a required value, or raise naming exactly what is missing.

    The first segment names the source file, so the message can point at
    a file and a path rather than at a KeyError from somewhere inside a
    dict comprehension. Everything the site shows is required; there is
    no branch here that substitutes a default, because a default is a
    number nobody measured.
    """
    segments = dotted.split(".")
    source_name = segments[0]

    if source_name not in sources:
        raise SiteDataError(f"unknown source {source_name!r} in path {dotted!r}")

    node = sources[source_name]
    origin = SOURCE_FILES.get(source_name, source_name)

    for index, segment in enumerate(segments[1:], start=1):
        walked = ".".join(segments[: index + 1])

        if isinstance(node, list):
            try:
                node = node[int(segment)]
            except (ValueError, IndexError) as error:
                raise SiteDataError(
                    f"{origin}: required path {dotted!r} is missing "
                    f"(no element {segment!r} at {walked!r})"
                ) from error
            continue

        if not isinstance(node, dict) or segment not in node:
            raise SiteDataError(
                f"{origin}: required path {dotted!r} is missing "
                f"(stopped at {walked!r})"
            )

        node = node[segment]

    return node


def copy_fields(sources: dict, prefix: str, fields) -> dict:
    """Copy the named fields verbatim, requiring every one of them."""
    return {name: require(sources, f"{prefix}.{name}") for name in fields}


# --------------------------------------------------------------
# Loading
# --------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(results_dir: Path) -> tuple[dict, dict]:
    """
    Load every source file. Returns (sources, paths).

    A missing file is fatal here rather than at first use: the build
    either has all its inputs or it has none worth publishing.
    """
    sources: dict = {}
    paths: dict = {}

    for name, filename in SOURCE_FILES.items():
        path = results_dir / filename
        if not path.is_file():
            raise SiteDataError(f"required source {path} not found")

        sources[name] = json.loads(path.read_text(encoding="utf-8"))
        paths[name] = path

    round_dir = results_dir / CAPTURE_ROUNDS_DIRNAME
    round_paths = sorted(round_dir.glob("*.json")) if round_dir.is_dir() else []

    if not round_paths:
        raise SiteDataError(f"required source {round_dir}/*.json not found")

    sources["capture_rounds"] = [
        json.loads(path.read_text(encoding="utf-8")) for path in round_paths
    ]
    paths["capture_rounds"] = round_paths

    return sources, paths


def relative_path(path: Path, results_dir: Path) -> str:
    """
    Recorded relative to the results directory's parent.

    Not `os.path.relpath` from the working directory: that would make
    the output depend on where the build was run from, and the file has
    to be byte-identical between runs for `--check` to mean anything.
    """
    try:
        return path.relative_to(results_dir.parent).as_posix()
    except ValueError:
        return path.name


def build_provenance(sources: dict, paths: dict, results_dir: Path) -> dict:
    """
    Where each number came from, and what the file hashed to.

    `generated_at` is carried through from each source rather than
    stamped here. `cli_parity.json` and the round files have none, and
    the key is omitted rather than filled with the build time -- a
    fabricated timestamp would be indistinguishable from a measured one.
    """
    provenance: dict = {}

    for name in SOURCE_FILES:
        entry = {
            "path": relative_path(paths[name], results_dir),
            "sha256": file_sha256(paths[name]),
        }
        if isinstance(sources[name], dict) and "generated_at" in sources[name]:
            entry["generated_at"] = sources[name]["generated_at"]

        provenance[name] = entry

    provenance["capture_rounds"] = [
        {
            "path": relative_path(path, results_dir),
            "sha256": file_sha256(path),
        }
        for path in paths["capture_rounds"]
    ]

    return provenance


# --------------------------------------------------------------
# Derived, and only derived
# --------------------------------------------------------------

def feature_instability(sources: dict) -> dict:
    """
    How far the per-feature ranking moves between the two seeds.

    Computed here, so the page cannot claim a movement the data does not
    show. The top-N sets are compared as sets on purpose: the measured
    finding is about how stable attribution is at feature level, and
    "same features, different order" and "different features" are
    different answers to that.

    **The ordering comes from the `rank_seed_*` fields, not from sorting
    the values.** `explain_model.py` already computed those ranks, so
    they are the single authoritative ordering; re-deriving one here
    would put a second owner on the same fact. The two agree today --
    that was checked -- which is exactly when to remove the duplicate,
    while it is still cheap and before a tie-break rule quietly drifts
    apart from the one that produced the published plots.
    """
    importance = require(sources, "shap.feature_importance")
    primary = require(sources, "shap.seeds.primary")
    comparison = require(sources, "shap.seeds.comparison")

    primary_key = f"seed_{primary}"
    comparison_key = f"seed_{comparison}"
    primary_rank = f"rank_seed_{primary}"
    comparison_rank = f"rank_seed_{comparison}"

    movements = {
        name: require(sources, f"shap.feature_importance.{name}.rank_movement")
        for name in importance
    }
    largest = max(movements.values())

    def by_rank(rank_key: str) -> list[str]:
        return [
            name
            for name, _ in sorted(
                importance.items(),
                key=lambda item: require(
                    sources, f"shap.feature_importance.{item[0]}.{rank_key}"
                ),
            )
        ][:TOP_FEATURES]

    top_primary = by_rank(primary_rank)
    top_comparison = by_rank(comparison_rank)

    in_both = set(top_primary) & set(top_comparison)
    only_one = sorted(set(top_primary) ^ set(top_comparison))

    def entry(name: str) -> dict:
        return {
            "name": name,
            primary_rank: require(
                sources, f"shap.feature_importance.{name}.{primary_rank}"
            ),
            comparison_rank: require(
                sources, f"shap.feature_importance.{name}.{comparison_rank}"
            ),
            "rank_movement": require(
                sources, f"shap.feature_importance.{name}.rank_movement"
            ),
            primary_key: require(
                sources, f"shap.feature_importance.{name}.{primary_key}"
            ),
            comparison_key: require(
                sources, f"shap.feature_importance.{name}.{comparison_key}"
            ),
        }

    all_ranks = [
        require(sources, f"shap.feature_importance.{name}.{rank_key}")
        for name in importance
        for rank_key in (primary_rank, comparison_rank)
    ]

    return {
        "max_rank_movement": largest,
        "features_at_max": [
            {
                "name": name,
                primary_rank: require(
                    sources, f"shap.feature_importance.{name}.{primary_rank}"
                ),
                comparison_rank: require(
                    sources, f"shap.feature_importance.{name}.{comparison_rank}"
                ),
                "rank_movement": movement,
            }
            for name, movement in sorted(movements.items())
            if movement == largest
        ],
        # Read off the file so the page never has to assume 0- or
        # 1-indexed ranks. A page that guessed wrong would be off by one
        # in every sentence that mentions a position.
        "rank_index_base": min(all_ranks),
        f"top{TOP_FEATURES}_{primary_key}": top_primary,
        f"top{TOP_FEATURES}_{comparison_key}": top_comparison,
        f"top{TOP_FEATURES}_same_set": set(top_primary) == set(top_comparison),
        f"top{TOP_FEATURES}_overlap": len(in_both),
        # The features in exactly one of the two lists, with their
        # numbers copied verbatim. When the sets differ, the useful
        # question is not "do they differ" but "by how much, and where"
        # -- a feature that crossed the cutoff by one place is a
        # boundary effect, not evidence of instability.
        f"top{TOP_FEATURES}_symmetric_difference": [entry(name) for name in only_one],
    }


def family_share_top2(sources: dict) -> dict:
    """
    What share of the attribution the two largest families carry, per
    seed, and whether the same family leads under both.

    The leader question is asked rather than answered: on the measured
    data the two swap, which is why the plot caption says the ordering
    is seed-dependent. Hardcoding either outcome here would put the same
    stale claim on the page that the plot title once carried.
    """
    families = require(sources, "shap.family_importance")
    shares: dict = {}
    leaders: list[str] = []

    for seed_key, values in families.items():
        order = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        top = order[:TOP_FAMILIES]
        total = sum(values.values())

        leaders.append(top[0][0])
        shares[seed_key] = {
            "families": [name for name, _ in top],
            "summed": sum(value for _, value in top),
            "total": total,
            "share": (sum(value for _, value in top) / total) if total else 0.0,
        }

    shares["same_leader"] = len(set(leaders)) == 1
    return shares


def capture_invariants(sources: dict) -> dict:
    """
    For each field that would change what a trace looks like: the
    distinct values seen across the rounds, and whether they were
    identical.

    This is how the page can state that the rounds are comparable
    without asserting it. A round captured with a different snaplen or
    against a different certificate would appear here as two values and
    `identical: false`, rather than being quietly averaged into the
    result.
    """
    rounds = require(sources, "capture_rounds")
    invariants: dict = {}

    for field in INVARIANT_FIELDS:
        values = [
            require(sources, f"capture_rounds.{index}.{field}")
            for index in range(len(rounds))
        ]
        distinct = sorted({json.dumps(value, sort_keys=True) for value in values})

        invariants[field] = {
            "values": [json.loads(item) for item in distinct],
            "identical": len(distinct) == 1,
        }

    return invariants


def cli_comparisons(sources: dict) -> dict:
    """
    The parity check's headline number, with the two factors it came
    from.

    Shown as arithmetic rather than as a bare total so the page can say
    *800 traces × 53 features* instead of asking the reader to trust
    42,400.
    """
    traces = require(sources, "cli_parity.comparison.traces_compared")
    features = require(sources, "cli_parity.model.n_features")

    return {
        "traces_compared": traces,
        "n_features": features,
        "total": traces * features,
    }


# --------------------------------------------------------------
# Build
# --------------------------------------------------------------

def build(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> dict:
    """Assemble the whole document. Key order here is the schema."""
    results_dir = Path(results_dir)
    sources, paths = load_sources(results_dir)

    models = {
        name: copy_fields(sources, f"metrics.models.{name}", MODEL_FIELDS)
        for name in ("random_forest", "logistic_regression")
    }
    models["hardest_pages"] = require(sources, "metrics.hardest_pages")

    shap_block = copy_fields(sources, "shap", SHAP_FIELDS)
    shap_block["feature_instability"] = feature_instability(sources)
    shap_block["family_share_top2"] = family_share_top2(sources)

    cli_block = dict(sources["cli_parity"])
    cli_block["comparisons"] = cli_comparisons(sources)

    corpus = copy_fields(sources, "corpus_manifest", CORPUS_FIELDS)
    corpus["counts"] = {
        "pages": len(require(sources, "corpus_manifest.pages")),
        "assets": len(require(sources, "corpus_manifest.assets")),
        "refused": len(require(sources, "corpus_manifest.refused")),
    }

    capture = [
        copy_fields(sources, f"capture_rounds.{index}", ROUND_FIELDS)
        for index in range(len(sources["capture_rounds"]))
    ]

    return {
        "provenance": build_provenance(sources, paths, results_dir),
        "dataset": require(sources, "metrics.dataset"),
        "split": require(sources, "metrics.split"),
        "models": models,
        "ablation": require(sources, "metrics.ablation"),
        "shap": shap_block,
        "cli": cli_block,
        "capture": capture,
        "capture_invariants": capture_invariants(sources),
        "corpus": corpus,
    }


def serialise(document: dict) -> str:
    """
    The published form. `sort_keys=False`: the key order above is the
    schema, chosen for reading.
    """
    return json.dumps(document, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def first_difference(left: dict, right: dict) -> str | None:
    """The first top-level key whose value differs, for `--check`."""
    for key in left:
        if key not in right:
            return key
        if serialise({key: left[key]}) != serialise({key: right[key]}):
            return key

    for key in right:
        if key not in left:
            return key

    return None
