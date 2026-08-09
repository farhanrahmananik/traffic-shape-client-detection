"""
test_site_data.py
-----------------
Tests for tsd.site_data.

A minimal in-memory fixture tree drives `build()`, so the derived values
can be checked against numbers small enough to verify by hand. The real
`results/` directory is used too when it is present, but only to assert
that what is copied is identical to the source -- never to assert a
measured value, which would put a number in a test file and defeat the
point of the build step.

What matters here is the derivation. Copying is hard to get wrong and
easy to see; the derived fields -- rank movement, top-ten overlap,
family shares, capture invariants -- are where a build step could
quietly put a wrong number on a published page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsd.site_data import SiteDataError, build, serialise

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS = REPO_ROOT / "results"


# --------------------------------------------------------------
# A tiny, hand-checkable fixture tree
# --------------------------------------------------------------

def metrics_fixture() -> dict:
    return {
        "generated_at": "2026-08-08T22:28:44+00:00",
        "split": {"method": "LeaveOneGroupOut", "group_column": "round",
                  "rounds": [1, 2], "n_folds": 2, "note": "n"},
        "dataset": {"traces": 6, "rounds": 2, "classes": ["firefox", "wget"],
                    "per_class": {"firefox": 3, "wget": 3}, "n_features": 2,
                    "features": ["syn_count", "iat_max"]},
        "hyperparameters": {"random_state": 42},
        "models": {
            "random_forest": {
                "model": "random_forest", "accuracy": 1.0,
                "per_class": {"firefox": {"f1": 1.0}, "wget": {"f1": 1.0}},
                "confusion_matrix": [[3, 0], [0, 3]],
                "labels": ["firefox", "wget"], "n_folds": 2,
                "fold_accuracies": [1.0, 1.0], "folds": [], "misclassified": [],
            },
            "logistic_regression": {
                "model": "logistic_regression", "accuracy": 0.5,
                "per_class": {"firefox": {"f1": 0.5}, "wget": {"f1": 0.5}},
                "confusion_matrix": [[2, 1], [2, 1]],
                "labels": ["firefox", "wget"], "n_folds": 2,
                "fold_accuracies": [0.5, 0.5], "folds": [],
                "misclassified": [{"page": "webmail"}],
            },
        },
        "hardest_pages": {"random_forest": [], "logistic_regression": []},
        "ablation": {"note": "a", "groups": {"counts": ["count_"]},
                     "models": {"random_forest": []}},
        "final_model_note": "note",
    }


def shap_fixture() -> dict:
    """
    Ranks chosen so the derived answers are checkable by eye:
    `iat_max` moves 3 places (the maximum), `syn_count` moves 1, and the
    two families swap leader between seeds.
    """
    return {
        "generated_at": "2026-08-08T23:35:28+00:00",
        "model": "random_forest",
        "seeds": {"primary": 42, "comparison": 7},
        "split": "LeaveOneGroupOut on capture round",
        "rounds": [1, 2],
        "n_folds": 2,
        "n_traces": 6,
        "n_features": 2,
        "direction": {"positive_class": "wget", "classes_": ["firefox", "wget"],
                      "note": "n", "base_value": 0.5},
        "family_importance": {
            # timing leads at 42; connections leads at 7 -> same_leader False
            "seed_42": {"timing": 0.6, "connections": 0.3, "counts": 0.1},
            "seed_7": {"connections": 0.5, "timing": 0.4, "counts": 0.1},
        },
        "feature_importance": {
            "syn_count": {"seed_42": 0.9, "seed_7": 0.4,
                          "rank_seed_42": 0, "rank_seed_7": 1,
                          "rank_movement": 1},
            "iat_max": {"seed_42": 0.4, "seed_7": 0.9,
                        "rank_seed_42": 1, "rank_seed_7": 0,
                        "rank_movement": 3},
        },
        "spreads": {"note": "n", "same_day_rounds": [1, 2],
                    "cross_day_rounds": [1, 2], "per_feature": {},
                    "max": {"same_day": 0.1, "cross_day": 0.2, "seed": 0.3}},
        "zero_importance": {"note": "n", "constant": [], "varying_but_unused": []},
    }


def parity_fixture() -> dict:
    return {
        "check": "cli-vs-training feature parity",
        "model": {"path": "models/m.joblib", "sha256": "abc", "n_features": 2},
        "limit": None,
        "partial": False,
        "coverage": {"pcaps_found": 6, "csv_rows": 6, "matched": 6,
                     "pcaps_without_csv_row": [], "csv_rows_without_pcap": []},
        "comparison": {"traces_compared": 6, "traces_matching_all_features": 6,
                       "refused_by_inference_path": [], "mismatched_features": {},
                       "examples": []},
    }


def manifest_fixture() -> dict:
    return {
        "generated_at": "2026-08-06T16:35:28+00:00",
        "user_agent": "ua", "crawl_delay": 1.5, "seed": 42, "walks": 20,
        "walks_run": 19, "max_depth": 5, "target_pages": 100,
        "robots_fetched_at": "2026-08-06T15:43:03+00:00",
        "robots_sha256": "9d8e", "robots_provenance_file": "results/provenance/r.txt",
        "environment": {"python": "3.12.3"},
        "pages": [{"url": "a"}, {"url": "b"}],
        "assets": [{"url": "c"}],
        "refused": [{"url": "d"}],
        "failures": {"upstream": [], "excluded": [], "local": []},
        "totals": {"pages": 2, "assets": 1},
    }


def round_fixture(number: int, snaplen: int = 96) -> dict:
    return {
        "round": number,
        "date": f"2026080{number}",
        "started_at": f"2026-08-0{number}T00:00:00+00:00",
        "finished_at": f"2026-08-0{number}T01:00:00+00:00",
        "host": "127.0.0.1", "port": 8443, "web_root": "data/mirror",
        "pcap_directory": "data/pcaps",
        "snaplen": snaplen,
        "tcpdump_filter": "host 127.0.0.1 and port 8443",
        "capture_interface": "lo",
        "quiet_seconds": 3.0,
        "max_load_seconds": 90.0,
        "network_namespace": "sudo unshare -n",
        "server_cert_sha256": "AA:BB",
        "invocations": {}, "versions": {"firefox": "153.0.3"},
        "limit": None,
        "traces": [{"client": "firefox", "page": "index"}],
        "totals": {"traces_ok": 3},
    }


def write_tree(root: Path, *, rounds=None, edit=None) -> Path:
    """A results/ directory. `edit` may mutate the loaded dicts first."""
    results = root / "results"
    (results / "capture_rounds").mkdir(parents=True, exist_ok=True)

    documents = {
        "metrics.json": metrics_fixture(),
        "shap_summary.json": shap_fixture(),
        "cli_parity.json": parity_fixture(),
        "corpus_manifest.json": manifest_fixture(),
    }
    round_documents = rounds if rounds is not None else [
        round_fixture(1), round_fixture(2)
    ]

    if edit is not None:
        edit(documents, round_documents)

    for name, document in documents.items():
        (results / name).write_text(json.dumps(document, indent=2), encoding="utf-8")

    for document in round_documents:
        path = results / "capture_rounds" / f"round_{document['round']:02d}.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    return results


@pytest.fixture
def results(tmp_path) -> Path:
    return write_tree(tmp_path)


# --------------------------------------------------------------
# Shape
# --------------------------------------------------------------

def test_top_level_key_order(results):
    document = build(results)

    assert list(document) == [
        "provenance", "dataset", "split", "models", "ablation", "shap",
        "cli", "capture", "capture_invariants", "corpus",
    ]


def test_copied_blocks_are_identical_to_the_source(results):
    document = build(results)
    metrics = json.loads((results / "metrics.json").read_text())

    assert document["dataset"] == metrics["dataset"]
    assert document["split"] == metrics["split"]
    assert document["ablation"] == metrics["ablation"]
    assert document["models"]["random_forest"]["accuracy"] == \
        metrics["models"]["random_forest"]["accuracy"]


def test_capture_never_carries_the_traces_list(results):
    """200 rows per round the page does not use, and BTU page stems."""
    for entry in build(results)["capture"]:
        assert "traces" not in entry


def test_corpus_carries_counts_not_the_page_and_asset_lists(results):
    corpus = build(results)["corpus"]

    assert corpus["counts"] == {"pages": 2, "assets": 1, "refused": 1}
    assert "pages" not in corpus
    assert "assets" not in corpus


# --------------------------------------------------------------
# Derived values
# --------------------------------------------------------------

def test_cli_comparisons_show_the_arithmetic(results):
    """
    The page should be able to say "6 traces x 2 features", not ask the
    reader to trust 12.
    """
    comparisons = build(results)["cli"]["comparisons"]

    assert comparisons == {"traces_compared": 6, "n_features": 2, "total": 12}


def test_feature_instability_finds_the_largest_movement(results):
    instability = build(results)["shap"]["feature_instability"]

    assert instability["max_rank_movement"] == 3
    assert [entry["name"] for entry in instability["features_at_max"]] == ["iat_max"]
    assert instability["features_at_max"][0]["rank_seed_42"] == 1
    assert instability["features_at_max"][0]["rank_seed_7"] == 0


def test_top_lists_are_ordered_by_the_rank_field(results):
    instability = build(results)["shap"]["feature_instability"]

    assert instability["top10_seed_42"] == ["syn_count", "iat_max"]
    assert instability["top10_seed_7"] == ["iat_max", "syn_count"]
    assert instability["top10_same_set"] is True, (
        "same features, different order"
    )


def test_the_rank_field_wins_over_the_values(tmp_path):
    """
    `explain_model.py` computed the ranks, so they are the single
    authoritative ordering. Re-deriving one from the values would put a
    second owner on the same fact -- and the two would agree right up
    until a tie-break rule drifted.

    Here the values and the ranks disagree deliberately: syn_count has
    the larger seed_42 value but the worse seed_42 rank. The rank must
    decide.
    """
    def edit(documents, rounds):
        importance = documents["shap_summary.json"]["feature_importance"]
        importance["syn_count"]["seed_42"] = 0.99
        importance["syn_count"]["rank_seed_42"] = 1
        importance["iat_max"]["seed_42"] = 0.01
        importance["iat_max"]["rank_seed_42"] = 0

    instability = build(write_tree(tmp_path, edit=edit))["shap"][
        "feature_instability"
    ]

    assert instability["top10_seed_42"] == ["iat_max", "syn_count"], (
        "ordered by value, this would be ['syn_count', 'iat_max']"
    )


def test_a_boundary_swap_gives_overlap_nine_and_two_differing(tmp_path):
    """
    Twelve features, ten shown. Two of them sit either side of the
    cutoff and swap between seeds: nine in common, one leaving and one
    arriving. That is what a boundary effect looks like, and it is worth
    telling apart from a top ten that genuinely reshuffled.
    """
    def edit(documents, rounds):
        importance = {}
        for index in range(12):
            name = f"feature_{index:02d}"
            # ranks 0..11 under seed 42; identical under seed 7 except
            # that the features at rank 9 and 10 trade places.
            rank_42 = index
            if index == 9:
                rank_7 = 10
            elif index == 10:
                rank_7 = 9
            else:
                rank_7 = index

            importance[name] = {
                "seed_42": 1.0 - index / 100,
                "seed_7": 1.0 - rank_7 / 100,
                "rank_seed_42": rank_42,
                "rank_seed_7": rank_7,
                "rank_movement": abs(rank_42 - rank_7),
            }
        documents["shap_summary.json"]["feature_importance"] = importance

    instability = build(write_tree(tmp_path, edit=edit))["shap"][
        "feature_instability"
    ]

    assert instability["top10_overlap"] == 9
    assert instability["top10_same_set"] is False

    differing = instability["top10_symmetric_difference"]
    assert [entry["name"] for entry in differing] == ["feature_09", "feature_10"]
    assert differing[0]["rank_seed_42"] == 9
    assert differing[0]["rank_seed_7"] == 10
    assert differing[0]["rank_movement"] == 1
    # values copied verbatim, not rounded
    assert differing[0]["seed_42"] == 1.0 - 9 / 100


def test_rank_index_base_is_read_from_the_file(results):
    """
    So the page never has to assume 0- or 1-indexed ranks. Guessing
    wrong would be off by one in every sentence that names a position.
    """
    instability = build(results)["shap"]["feature_instability"]

    assert instability["rank_index_base"] == 0


def test_rank_index_base_follows_a_one_indexed_source(tmp_path):
    def edit(documents, rounds):
        for entry in documents["shap_summary.json"]["feature_importance"].values():
            entry["rank_seed_42"] += 1
            entry["rank_seed_7"] += 1

    instability = build(write_tree(tmp_path, edit=edit))["shap"][
        "feature_instability"
    ]

    assert instability["rank_index_base"] == 1


def test_family_share_top2_is_computed_not_asserted(results):
    shares = build(results)["shap"]["family_share_top2"]

    # seed 42: timing 0.6 + connections 0.3 = 0.9 of 1.0
    assert shares["seed_42"]["families"] == ["timing", "connections"]
    assert shares["seed_42"]["summed"] == pytest.approx(0.9)
    assert shares["seed_42"]["share"] == pytest.approx(0.9)

    # seed 7: connections 0.5 + timing 0.4 = 0.9 of 1.0, different leader
    assert shares["seed_7"]["families"] == ["connections", "timing"]
    assert shares["seed_7"]["share"] == pytest.approx(0.9)

    assert shares["same_leader"] is False


def test_family_same_leader_is_true_when_it_does_not_swap(tmp_path):
    def edit(documents, rounds):
        documents["shap_summary.json"]["family_importance"]["seed_7"] = {
            "timing": 0.7, "connections": 0.2, "counts": 0.1
        }

    shares = build(write_tree(tmp_path, edit=edit))["shap"]["family_share_top2"]

    assert shares["same_leader"] is True


# --------------------------------------------------------------
# Capture invariants
# --------------------------------------------------------------

def test_capture_invariants_report_identical_when_they_are(results):
    invariants = build(results)["capture_invariants"]

    assert invariants["snaplen"] == {"values": [96], "identical": True}
    assert invariants["server_cert_sha256"]["identical"] is True
    assert invariants["capture_interface"]["values"] == ["lo"]


def test_a_differing_snaplen_is_reported_not_silently_picked(tmp_path):
    """
    The page claims the rounds are comparable. If one round was captured
    differently, that has to show up as a fact rather than being
    averaged away -- a single value here would be a claim nobody
    measured.
    """
    results = write_tree(
        tmp_path, rounds=[round_fixture(1), round_fixture(2, snaplen=128)]
    )

    invariants = build(results)["capture_invariants"]

    assert invariants["snaplen"]["identical"] is False
    assert sorted(invariants["snaplen"]["values"]) == [96, 128]
    # the fields that did not change still say so
    assert invariants["capture_interface"]["identical"] is True


# --------------------------------------------------------------
# Required paths
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "source, path, dotted",
    [
        ("metrics.json", ("dataset",), "metrics.dataset"),
        ("metrics.json", ("models", "random_forest", "accuracy"),
         "metrics.models.random_forest.accuracy"),
        ("metrics.json", ("ablation",), "metrics.ablation"),
        ("shap_summary.json", ("direction",), "shap.direction"),
        ("shap_summary.json", ("family_importance",), "shap.family_importance"),
        ("cli_parity.json", ("model", "n_features"),
         "cli_parity.model.n_features"),
        ("corpus_manifest.json", ("totals",), "corpus_manifest.totals"),
    ],
)
def test_a_missing_required_key_raises_and_names_the_path(
    tmp_path, source, path, dotted
):
    """
    Never a default and never null: a page rendering "accuracy: null"
    looks like a measurement, and the build is where that should stop.
    """
    def edit(documents, rounds):
        node = documents[source]
        for segment in path[:-1]:
            node = node[segment]
        del node[path[-1]]

    results = write_tree(tmp_path, edit=edit)

    with pytest.raises(SiteDataError) as raised:
        build(results)

    message = str(raised.value)
    assert dotted in message
    assert source in message


def test_a_missing_round_field_raises(tmp_path):
    def edit(documents, rounds):
        del rounds[1]["snaplen"]

    results = write_tree(tmp_path, edit=edit)

    with pytest.raises(SiteDataError) as raised:
        build(results)

    assert "capture_rounds.1.snaplen" in str(raised.value)


def test_a_missing_source_file_raises(tmp_path):
    results = write_tree(tmp_path)
    (results / "shap_summary.json").unlink()

    with pytest.raises(SiteDataError) as raised:
        build(results)

    assert "shap_summary.json" in str(raised.value)


def test_no_capture_rounds_raises(tmp_path):
    results = write_tree(tmp_path)
    for path in (results / "capture_rounds").glob("*.json"):
        path.unlink()

    with pytest.raises(SiteDataError) as raised:
        build(results)

    assert "capture_rounds" in str(raised.value)


# --------------------------------------------------------------
# Provenance and determinism
# --------------------------------------------------------------

def test_provenance_carries_each_sources_own_generated_at(results):
    provenance = build(results)["provenance"]

    assert provenance["metrics"]["generated_at"] == "2026-08-08T22:28:44+00:00"
    assert provenance["shap"]["generated_at"] == "2026-08-08T23:35:28+00:00"
    assert provenance["corpus_manifest"]["generated_at"] == \
        "2026-08-06T16:35:28+00:00"

    # These two have none. The key is omitted rather than faked -- a
    # fabricated timestamp is indistinguishable from a measured one.
    assert "generated_at" not in provenance["cli_parity"]
    for entry in provenance["capture_rounds"]:
        assert "generated_at" not in entry


def test_provenance_hashes_the_files_it_read(results):
    import hashlib

    provenance = build(results)["provenance"]
    expected = hashlib.sha256(
        (results / "metrics.json").read_bytes()
    ).hexdigest()

    assert provenance["metrics"]["sha256"] == expected
    assert provenance["metrics"]["path"] == "results/metrics.json"


def test_the_output_has_no_wall_clock_timestamp_of_its_own(results):
    """
    A build time would change on every run and make the file undiffable,
    hiding the thing a diff is for: noticing that a number moved.
    """
    first = serialise(build(results))
    second = serialise(build(results))

    assert first == second

    document = build(results)
    assert "generated_at" not in document
    assert "built_at" not in document


# --------------------------------------------------------------
# Against the real results/, when present
# --------------------------------------------------------------

def test_build_against_the_real_results_directory():
    """
    Asserts only that what is copied is identical to the source. No
    measured value appears in this file -- putting one here would be the
    same hand-typed number the build step exists to prevent.
    """
    if not (REAL_RESULTS / "metrics.json").is_file():
        pytest.skip("results/ is not populated in this clone")

    document = build(REAL_RESULTS)
    metrics = json.loads((REAL_RESULTS / "metrics.json").read_text())

    assert document["dataset"] == metrics["dataset"]
    assert document["split"] == metrics["split"]
    assert document["ablation"] == metrics["ablation"]

    for name in ("random_forest", "logistic_regression"):
        source = metrics["models"][name]
        for field in ("accuracy", "fold_accuracies", "confusion_matrix",
                      "labels", "n_folds", "misclassified", "per_class"):
            assert document["models"][name][field] == source[field]


def test_real_top10_accounting_adds_up():
    """
    The two sets are drawn from a fixed cutoff, so overlap plus the
    symmetric difference must account for both lists exactly. This does
    not assert what the overlap IS -- that is a measured value and
    belongs in the generated file, not in a test.
    """
    if not (REAL_RESULTS / "shap_summary.json").is_file():
        pytest.skip("results/ is not populated in this clone")

    instability = build(REAL_RESULTS)["shap"]["feature_instability"]

    overlap = instability["top10_overlap"]
    differing = len(instability["top10_symmetric_difference"])

    # |A| + |B| = 2*|A n B| + |A xor B|, and both lists are the same length
    assert overlap * 2 + differing == \
        len(instability["top10_seed_42"]) + len(instability["top10_seed_7"])
    assert instability["top10_same_set"] == (differing == 0)


def test_real_rank_index_base_is_consistent_with_n_features():
    """
    Ranks span exactly the feature list: the smallest is the base and
    the largest is base + n_features - 1. If that ever fails, the ranks
    describe a different feature set from the one being published.
    """
    if not (REAL_RESULTS / "shap_summary.json").is_file():
        pytest.skip("results/ is not populated in this clone")

    document = build(REAL_RESULTS)
    instability = document["shap"]["feature_instability"]
    summary = json.loads((REAL_RESULTS / "shap_summary.json").read_text())

    base = instability["rank_index_base"]
    n_features = document["shap"]["n_features"]

    for seed in ("rank_seed_42", "rank_seed_7"):
        ranks = sorted(entry[seed] for entry in summary["feature_importance"].values())
        assert ranks == list(range(base, base + n_features)), seed

    assert len(summary["feature_importance"]) == n_features


def test_real_capture_invariants_are_reported_either_way():
    """
    Whatever the rounds say, the page gets the measurement rather than
    an assertion. This test does not care which answer it is.
    """
    if not (REAL_RESULTS / "metrics.json").is_file():
        pytest.skip("results/ is not populated in this clone")

    invariants = build(REAL_RESULTS)["capture_invariants"]

    for field, entry in invariants.items():
        assert isinstance(entry["identical"], bool), field
        assert entry["values"], field
        assert entry["identical"] == (len(entry["values"]) == 1), field
