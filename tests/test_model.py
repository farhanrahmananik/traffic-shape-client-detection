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
    GROUP_COLUMN,
    LABEL_COLUMNS,
    MODEL_NAMES,
    Dataset,
    DatasetError,
    build_metrics,
    build_pipeline,
    evaluate_by_round,
    fit_final_model,
    load_dataset,
    misclassified_pages,
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
