"""
test_model.py
-------------
Tests for tsd.model.

Synthetic data with a planted, known signal, so the numbers are
predictable and a wrong answer is visibly wrong. The real CSV is never
touched here: these tests are about the split and the plumbing, and a
suite that reported real accuracies would be reporting them before the
dataset is complete.

The split is what these tests exist for. A leaked split does not fail --
it produces a *better* number than an honest one, and nothing downstream
complains. So the properties that cannot be checked by looking at the
output are checked here instead: one fold per round, no round on both
sides of a fold, a fresh pipeline per fold, and a refusal when there is
only one round.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from tsd.model import (
    FEATURE_GROUPS,
    GROUP_COLUMN,
    LABEL_COLUMNS,
    MODEL_NAMES,
    Dataset,
    DatasetError,
    ablation_configurations,
    build_metrics,
    build_pipeline,
    evaluate_by_round,
    features_in_group,
    fit_final_model,
    group_of,
    iter_round_folds,
    load_dataset,
    misclassified_pages,
    run_ablation,
    select_features,
    write_metrics,
)

FEATURES = ["size_up_mean", "iat_down_p90", "syn_count", "burst_count"]


def synthetic_frame(
    rounds: int = 3,
    pages: int = 10,
    separable: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    A feature table with a planted signal.

    Firefox rows carry a high syn_count and a distinct size profile,
    wget rows the opposite, with a little per-round drift so the folds
    are not identical copies of one another.
    """
    generator = np.random.default_rng(seed)
    rows = []

    for round_number in range(1, rounds + 1):
        drift = 0.05 * round_number

        for page in range(pages):
            for client in ("firefox", "wget"):
                if separable:
                    base = 6.0 if client == "firefox" else 1.0
                else:
                    base = 3.0

                rows.append({
                    "round": round_number,
                    "date": f"2026080{round_number}",
                    "client": client,
                    "page": f"page_{page:02d}",
                    "size_up_mean": base * 100 + generator.normal(0, 5) + drift,
                    "iat_down_p90": base * 0.001 + generator.normal(0, 0.0001),
                    "syn_count": base,
                    "burst_count": base * 20 + generator.normal(0, 2),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def dataset(tmp_path) -> Dataset:
    path = tmp_path / "features.csv"
    synthetic_frame().to_csv(path, index=False)
    return load_dataset(path)


def small(**kwargs):
    """Forest hyperparameters small enough to keep the suite fast."""
    return {"n_estimators": 8, "random_state": 42, **kwargs}


# --------------------------------------------------------------
# Loading and the group requirement
# --------------------------------------------------------------

def test_dataset_separates_labels_from_features(dataset):
    assert dataset.features == FEATURES
    assert set(dataset.classes) == {"firefox", "wget"}
    assert dataset.rounds == [1, 2, 3]
    assert dataset.X.shape == (60, 4)
    assert len(dataset) == 60


def test_single_round_is_refused_with_a_clear_message(tmp_path):
    """
    With one round there is nothing to hold out that does not share its
    conditions. The dangerous outcome is not a crash -- it is a
    helpful-looking fallback to a random split, which would run, print a
    high number, and never say the number is meaningless.
    """
    path = tmp_path / "one_round.csv"
    synthetic_frame(rounds=1).to_csv(path, index=False)

    with pytest.raises(DatasetError) as raised:
        load_dataset(path)

    message = str(raised.value)
    assert "1 capture round" in message
    assert "at least two rounds" in message
    assert "capture_round.py --round 2" in message


def test_single_class_is_refused(tmp_path):
    frame = synthetic_frame()
    frame = frame[frame["client"] == "wget"]
    path = tmp_path / "one_class.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(DatasetError) as raised:
        load_dataset(path)

    assert "only one class" in str(raised.value)


def test_missing_label_column_is_refused(tmp_path):
    frame = synthetic_frame().drop(columns=["round"])
    path = tmp_path / "no_round.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(DatasetError) as raised:
        load_dataset(path)

    assert "round" in str(raised.value)


# --------------------------------------------------------------
# The split
# --------------------------------------------------------------

def test_one_fold_per_round(dataset):
    result = evaluate_by_round(dataset, model="random_forest", **small())

    assert len(result.folds) == len(dataset.rounds)
    assert sorted(fold.held_out_round for fold in result.folds) == dataset.rounds


def test_no_round_appears_on_both_sides_of_a_fold(dataset):
    """
    The invariant the whole evaluation rests on. Traces from one round
    share their conditions, so a round on both sides is the model being
    scored on conditions it has already seen.
    """
    result = evaluate_by_round(dataset, model="random_forest", **small())

    for fold in result.folds:
        assert fold.held_out_round not in fold.train_rounds
        assert set(fold.train_rounds) == set(dataset.rounds) - {fold.held_out_round}


def test_every_page_is_on_both_sides_and_that_is_correct(dataset):
    """
    The invariant that must NOT be asserted the other way round.

    Every page is loaded in every round, so every page appears in both
    train and test in every fold. Asserting "no page on both sides"
    would fail, and making it pass would mean grouping by page -- which
    answers "can we recognise this page?", a different question, and one
    explicitly out of scope (100 classes with a handful of samples each
    is not trainable).

    The same page on both sides is exactly what forces the model to
    separate the two CLIENTS rather than two pages. This test pins that
    down so nobody later "fixes" the split into a page-based one.
    """
    splitter_pages = []
    from sklearn.model_selection import LeaveOneGroupOut

    for train_index, test_index in LeaveOneGroupOut().split(
        dataset.X, dataset.y, groups=dataset.groups
    ):
        train_pages = set(dataset.pages[train_index])
        test_pages = set(dataset.pages[test_index])
        splitter_pages.append((train_pages, test_pages))

    for train_pages, test_pages in splitter_pages:
        assert train_pages == test_pages, (
            "every page should appear on both sides -- if this ever fails, "
            "the corpus or the capture changed, not the split"
        )


def test_every_trace_is_held_out_exactly_once(dataset):
    """
    The pooled accuracy is only a pooled accuracy if every trace is
    predicted once, by a model that never saw its round. A splitter that
    skipped rows, or held some out twice, would still print a perfectly
    plausible number -- the total would just quietly describe a
    different dataset from the one that was loaded.
    """
    held_out = [
        index
        for fold in iter_round_folds(dataset)
        for index in fold.test_index
    ]

    assert sorted(held_out) == list(range(len(dataset)))
    assert len(held_out) == len(set(held_out)), "a trace was held out twice"


def test_fold_train_and_test_never_share_a_round(dataset):
    """
    The invariant the whole evaluation rests on, checked on the shared
    splitter itself rather than on one caller's copy of it.
    """
    for fold in iter_round_folds(dataset):
        training_groups = sorted({int(g) for g in dataset.groups[fold.train_index]})

        assert fold.held_out_round not in training_groups
        assert fold.train_rounds == training_groups
        assert set(dataset.groups[fold.test_index]) == {fold.held_out_round}


def test_folds_are_stable_across_runs(dataset):
    """
    SHAP is computed per fold. Unstable folds would make the published
    plots unreproducible, and there is no seed anywhere in this path to
    pin them down after the fact -- the stability has to come from the
    splitter having no randomness at all.
    """
    first = [
        (fold.held_out_round, list(fold.test_index))
        for fold in iter_round_folds(dataset)
    ]
    second = [
        (fold.held_out_round, list(fold.test_index))
        for fold in iter_round_folds(dataset)
    ]

    assert first == second


def test_evaluation_folds_match_the_shared_splitter(dataset):
    """
    The refactor's point: the evaluation reports the folds the shared
    splitter produces, so the step-7 SHAP module gets the same ones.
    """
    shared = list(iter_round_folds(dataset))
    result = evaluate_by_round(dataset, model="random_forest", **small())

    assert [fold.held_out_round for fold in result.folds] == \
        [fold.held_out_round for fold in shared]
    assert [fold.train_rounds for fold in result.folds] == \
        [fold.train_rounds for fold in shared]
    assert [fold.n_test for fold in result.folds] == \
        [len(fold.test_index) for fold in shared]


def test_every_trace_is_predicted_exactly_once(dataset):
    """
    The aggregate is a pooled prediction over the whole dataset, not an
    average of fold averages -- so the reported number is what would
    happen to each trace when its own round was unseen.
    """
    result = evaluate_by_round(dataset, model="random_forest", **small())

    assert sum(fold.n_test for fold in result.folds) == len(dataset)
    for fold in result.folds:
        assert fold.n_train == len(dataset) - fold.n_test


# --------------------------------------------------------------
# No transformer fitted before the split
# --------------------------------------------------------------

def test_build_pipeline_returns_an_unfitted_pipeline():
    pipeline = build_pipeline("logistic_regression")

    assert isinstance(pipeline, Pipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["scaler"])


def test_a_fresh_pipeline_is_built_inside_every_fold(dataset):
    """
    Structural, not behavioural: a scaler fitted on the whole dataset
    before splitting would carry test-fold statistics into training. A
    smaller leak than a bad split, and just as invisible in the output.

    The factory records every pipeline it hands out, so the test can
    assert one per fold and no reuse.
    """
    built: list[Pipeline] = []

    def factory():
        pipeline = build_pipeline("logistic_regression")
        built.append(pipeline)
        return pipeline

    result = evaluate_by_round(dataset, pipeline_factory=factory)

    assert len(built) == len(result.folds) == len(dataset.rounds)
    assert len({id(pipeline) for pipeline in built}) == len(built)


def test_the_scaler_is_fitted_on_training_rows_only(dataset):
    """
    Checks the consequence as well as the structure: the scaler's mean
    must match the training rows of its fold, not the whole dataset.
    """
    built: list[Pipeline] = []

    def factory():
        pipeline = build_pipeline("logistic_regression")
        built.append(pipeline)
        return pipeline

    evaluate_by_round(dataset, pipeline_factory=factory)

    whole_dataset_mean = dataset.X.mean(axis=0)
    first_fold_mean = built[0].named_steps["scaler"].mean_

    assert not np.allclose(first_fold_mean, whole_dataset_mean), (
        "the scaler saw the whole dataset, so the split leaked"
    )


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------

def test_a_planted_signal_is_found(dataset):
    """Sanity: with a separable signal the plumbing must reach ~1.0."""
    result = evaluate_by_round(dataset, model="random_forest", **small())

    assert result.accuracy > 0.95
    assert set(result.labels) == {"firefox", "wget"}
    assert result.per_class["firefox"]["support"] == 30


def test_an_absent_signal_is_not_invented(tmp_path):
    """
    The other direction, and the more important one. Identical classes
    must score near chance -- if this ever reads high, the leak is in
    the harness, not in the data.
    """
    path = tmp_path / "flat.csv"
    synthetic_frame(separable=False, seed=7).to_csv(path, index=False)

    result = evaluate_by_round(load_dataset(path), model="random_forest", **small())

    assert result.accuracy < 0.75


def test_confusion_matrix_shape_and_totals(dataset):
    result = evaluate_by_round(dataset, model="random_forest", **small())

    matrix = np.array(result.confusion)

    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(dataset)


def test_misclassified_entries_name_the_page_and_round(tmp_path):
    path = tmp_path / "flat.csv"
    synthetic_frame(separable=False, seed=3).to_csv(path, index=False)

    result = evaluate_by_round(load_dataset(path), model="random_forest", **small())

    assert result.misclassified, "the flat dataset must produce some errors"
    entry = result.misclassified[0]
    assert set(entry) == {"round", "page", "true", "predicted"}
    assert entry["true"] != entry["predicted"]

    hardest = misclassified_pages(result)
    assert hardest and hardest[0][1] >= 1


def test_both_models_can_be_evaluated(dataset):
    for name in MODEL_NAMES:
        result = evaluate_by_round(dataset, model=name, **small())
        assert result.model == name
        assert 0.0 <= result.accuracy <= 1.0


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_pipeline("gradient_boosted_hope")


# --------------------------------------------------------------
# The shipped model
# --------------------------------------------------------------

def test_final_model_is_fitted_on_everything_and_returns_no_score(dataset):
    """
    The CLI should use all the evidence, but this model has seen every
    trace and therefore has no honest accuracy. The function returns a
    fitted pipeline and nothing else, so there is no number to quote by
    mistake.
    """
    pipeline = fit_final_model(dataset, model="random_forest", **small())

    check_is_fitted(pipeline.named_steps["classifier"])
    assert len(pipeline.predict(dataset.X)) == len(dataset)


# --------------------------------------------------------------
# The published document
# --------------------------------------------------------------

def test_metrics_document_contains_only_names_and_numbers(dataset, tmp_path):
    evaluations = {
        name: evaluate_by_round(dataset, model=name, **small())
        for name in MODEL_NAMES
    }
    metrics = build_metrics(
        dataset, evaluations,
        hyperparameters={"random_state": 42, "n_estimators": 8},
        generated_at="2026-08-07T12:00:00+00:00",
    )

    path = write_metrics(tmp_path / "metrics.json", metrics)
    serialised = path.read_text(encoding="utf-8")

    assert metrics["split"]["method"] == "LeaveOneGroupOut"
    assert metrics["split"]["group_column"] == GROUP_COLUMN
    assert metrics["dataset"]["traces"] == len(dataset)
    assert metrics["dataset"]["features"] == dataset.features
    assert "held-out" in metrics["final_model_note"]

    # No markup, no payload -- there is none to leak, since -s 96 meant
    # the payload was never captured, but the document is published so
    # the promise is asserted rather than assumed.
    assert "<html" not in serialised
    assert "<!DOCTYPE" not in serialised
    assert json.loads(serialised) == metrics


def test_metrics_record_the_split_rationale(dataset):
    """
    The note explaining why pages appear on both sides is part of the
    published record: a reader who checks the split will notice that
    overlap, and the document should answer them before they conclude
    it is a bug.
    """
    evaluations = {"random_forest": evaluate_by_round(dataset, **small())}
    metrics = build_metrics(dataset, evaluations, {}, "now")

    note = metrics["split"]["note"]
    assert "every round" in note
    assert "client" in note


# --------------------------------------------------------------
# Ablation -- what is carrying the result
# --------------------------------------------------------------

def test_every_real_feature_belongs_to_exactly_one_group():
    """
    A feature in no group would sit outside every ablation and never be
    questioned -- which is the opposite of what the ablation is for.
    Checked against the real feature list, not the synthetic one.
    """
    from tsd.features import feature_names

    ungrouped = [name for name in feature_names() if group_of(name) is None]
    assert not ungrouped, f"features in no ablation group: {ungrouped}"

    counted = sum(
        len(features_in_group(feature_names(), group)) for group in FEATURE_GROUPS
    )
    assert counted == len(feature_names()), "a feature is in two groups"


def test_group_membership_of_the_real_features():
    from tsd.features import feature_names

    names = feature_names()

    assert "syn_count" in features_in_group(names, "connections")
    assert "syn_ack_count" in features_in_group(names, "connections")
    assert "burst_count" in features_in_group(names, "bursts")
    assert "ack_up_count" in features_in_group(names, "sizes")
    assert "duration" in features_in_group(names, "timing")
    assert "count_total" in features_in_group(names, "counts")


def test_select_features_drops_and_keeps(dataset):
    dropped = select_features(dataset, drop=["syn_count"])
    kept = select_features(dataset, keep=["syn_count"])

    assert "syn_count" not in dropped.features
    assert len(dropped.features) == len(dataset.features) - 1
    assert dropped.X.shape == (len(dataset), len(dataset.features) - 1)

    assert kept.features == ["syn_count"]
    assert kept.X.shape == (len(dataset), 1)
    # The labels and groups travel with the subset, or the split breaks.
    assert list(kept.groups) == list(dataset.groups)
    assert list(kept.y) == list(dataset.y)


def test_select_features_preserves_column_order(dataset):
    subset = select_features(dataset, keep=["burst_count", "size_up_mean"])

    assert subset.features == ["size_up_mean", "burst_count"], (
        "order must follow the dataset, not the argument"
    )


def test_unknown_feature_name_is_refused(dataset):
    """
    An ablation that silently dropped nothing would report the headline
    accuracy under the label of an experiment that never ran.
    """
    with pytest.raises(DatasetError) as raised:
        select_features(dataset, drop=["syn_kount"])

    assert "syn_kount" in str(raised.value)

    with pytest.raises(DatasetError):
        select_features(dataset, keep=["not_a_feature"])


def test_dropping_everything_is_refused(dataset):
    with pytest.raises(DatasetError):
        select_features(dataset, drop=list(dataset.features))


def test_preset_sweep_covers_the_expected_configurations(dataset):
    labels = [label for label, _ in ablation_configurations(dataset.features)]

    assert labels[0] == "all features"
    assert "without syn_count" in labels
    assert "without connections" in labels
    assert "without sizes" in labels
    assert "without timing" in labels
    assert "without bursts" in labels
    assert labels[-1] == "only syn_count"


def test_preset_sweep_feature_sets_are_correct(dataset):
    configurations = dict(ablation_configurations(dataset.features))

    assert configurations["all features"] == dataset.features
    assert "syn_count" not in configurations["without syn_count"]
    assert configurations["only syn_count"] == ["syn_count"]
    assert not features_in_group(configurations["without sizes"], "sizes")


def test_ablation_runs_the_same_split_protocol(dataset):
    """
    An ablation evaluated any other way would not be comparable with the
    headline number, and the comparison is the entire point.
    """
    configurations = ablation_configurations(dataset.features)
    results = run_ablation(dataset, configurations, model="random_forest", **small())

    assert len(results) == len(configurations)
    for result in results:
        assert len(result.fold_accuracies) == len(dataset.rounds)
        assert 0.0 <= result.accuracy <= 1.0


def test_ablation_isolates_a_single_carrying_feature():
    """
    The question the ablation exists to answer, on data where the answer
    is known: here the signal lives ONLY in syn_count, so dropping it
    must collapse the score while keeping it alone must not.

    On the real dataset this is what decides which sentence the README
    is allowed to write.
    """
    generator = np.random.default_rng(11)
    rows = []
    for round_number in (1, 2, 3):
        for page in range(10):
            for client in ("firefox", "wget"):
                rows.append({
                    "round": round_number,
                    "date": "20260807",
                    "client": client,
                    "page": f"page_{page:02d}",
                    # noise only
                    "size_up_mean": generator.normal(100, 10),
                    "iat_down_p90": generator.normal(0.001, 0.0002),
                    "burst_count": generator.normal(50, 5),
                    # the entire signal
                    "syn_count": 6.0 if client == "firefox" else 1.0,
                })

    dataset = Dataset(
        features=["size_up_mean", "iat_down_p90", "burst_count", "syn_count"],
        X=pd.DataFrame(rows)[
            ["size_up_mean", "iat_down_p90", "burst_count", "syn_count"]
        ].to_numpy(dtype=float),
        y=pd.DataFrame(rows)["client"].to_numpy(),
        groups=pd.DataFrame(rows)["round"].to_numpy(),
        pages=pd.DataFrame(rows)["page"].to_numpy(),
    )

    results = {
        result.label: result.accuracy
        for result in run_ablation(
            dataset, ablation_configurations(dataset.features),
            model="random_forest", **small(),
        )
    }

    assert results["all features"] == 1.0
    assert results["only syn_count"] == 1.0
    assert results["without syn_count"] < 0.75
    assert results["without connections"] < 0.75


def test_ablation_does_not_change_the_headline_evaluation(dataset):
    """The ablation is additional evidence, not a replacement."""
    before = evaluate_by_round(dataset, model="random_forest", **small()).to_dict()

    run_ablation(dataset, ablation_configurations(dataset.features),
                 model="random_forest", **small())

    after = evaluate_by_round(dataset, model="random_forest", **small()).to_dict()

    assert before == after


def test_ablation_section_is_separate_in_the_metrics(dataset):
    evaluations = {"random_forest": evaluate_by_round(dataset, **small())}
    ablation = {
        "random_forest": run_ablation(
            dataset, ablation_configurations(dataset.features), **small()
        )
    }

    metrics = build_metrics(dataset, evaluations, {}, "now", ablation=ablation)

    assert metrics["models"]["random_forest"]["accuracy"] == \
        evaluations["random_forest"].accuracy
    assert "ablation" in metrics
    assert "does not replace" in metrics["ablation"]["note"]
    assert metrics["ablation"]["groups"]["connections"] == ["syn_"]

    labels = [
        entry["configuration"]
        for entry in metrics["ablation"]["models"]["random_forest"]
    ]
    assert "only syn_count" in labels


def test_metrics_without_ablation_records_none(dataset):
    evaluations = {"random_forest": evaluate_by_round(dataset, **small())}

    metrics = build_metrics(dataset, evaluations, {}, "now")

    assert metrics["ablation"] is None


def test_ablation_is_deterministic(dataset):
    configurations = ablation_configurations(dataset.features)

    first = [r.to_dict() for r in run_ablation(dataset, configurations, **small())]
    second = [r.to_dict() for r in run_ablation(dataset, configurations, **small())]

    assert first == second


# --------------------------------------------------------------
# Determinism
# --------------------------------------------------------------

def test_same_input_gives_the_same_metrics(tmp_path):
    """
    A published number that changes between runs is not a measurement.
    The seed is fixed and the split has no randomness at all.
    """
    path = tmp_path / "features.csv"
    synthetic_frame().to_csv(path, index=False)

    first = evaluate_by_round(load_dataset(path), model="random_forest", **small())
    second = evaluate_by_round(load_dataset(path), model="random_forest", **small())

    assert first.to_dict() == second.to_dict()


def test_fold_order_follows_round_order(dataset):
    result = evaluate_by_round(dataset, model="random_forest", **small())

    assert [fold.held_out_round for fold in result.folds] == dataset.rounds
