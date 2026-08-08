"""
test_shap_explain.py
--------------------
Tests for tsd.shap_explain.

Synthetic data only -- no PCAPs, no network, and a deliberately small
forest so the suite stays fast.

The load-bearing test here is
`test_only_held_out_rows_are_explained`. Everything else checks shapes
and plumbing; that one checks the property that cannot be seen in the
output. If someone later "simplifies" this module into explaining a
single model fitted on all rounds, every plot would still render, the
attributions would look *cleaner* than the honest ones, and no other
test in this repo would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsd.model import Dataset, iter_round_folds
from tsd.shap_explain import (
    DEFAULT_POSITIVE_CLASS,
    ExplanationError,
    aggregate_importance,
    explain_by_round,
    fold_importance_table,
    importance_spread,
)

FEATURES = ["syn_count", "size_up_mean", "iat_down_p90", "burst_count"]
ROUNDS = (1, 2, 3)
PAGES_PER_ROUND = 8


def small(**kwargs):
    return {"n_estimators": 10, "random_state": 42, **kwargs}


@pytest.fixture
def dataset() -> Dataset:
    """
    Three rounds, two clients, one clearly informative feature.

    `syn_count` carries the signal, mirroring the real dataset where
    Firefox opens ~6 connections and wget exactly 1. The other three
    are noise, so the expected ranking is known in advance.
    """
    generator = np.random.default_rng(5)
    rows, labels, groups, pages = [], [], [], []

    for round_number in ROUNDS:
        for page in range(PAGES_PER_ROUND):
            for client in ("firefox", "wget"):
                informative = 6.0 if client == "firefox" else 1.0
                rows.append([
                    informative,
                    generator.normal(100, 10),
                    generator.normal(0.001, 0.0002),
                    generator.normal(50, 5),
                ])
                labels.append(client)
                groups.append(round_number)
                pages.append(f"page_{page:02d}")

    return Dataset(
        features=list(FEATURES),
        X=np.array(rows, dtype=float),
        y=np.array(labels),
        groups=np.array(groups),
        pages=np.array(pages),
    )


# --------------------------------------------------------------
# One explanation per fold
# --------------------------------------------------------------

def test_one_explanation_per_round(dataset):
    explanations = explain_by_round(dataset, **small())

    assert len(explanations) == len(ROUNDS)
    assert [e.held_out_round for e in explanations] == list(ROUNDS)


def test_shap_values_have_one_row_per_held_out_trace(dataset):
    explanations = explain_by_round(dataset, **small())
    per_round = len(dataset) // len(ROUNDS)

    for explanation in explanations:
        assert explanation.shap_values.shape == (per_round, len(FEATURES))
        assert explanation.X_test.shape == (per_round, len(FEATURES))
        assert len(explanation.pages) == per_round
        assert len(explanation.y_true) == len(explanation.y_pred) == per_round
        assert len(explanation) == per_round


def test_classes_and_positive_class_come_from_the_fitted_model(dataset):
    """
    The class index is read off the fitted model rather than assumed.
    sv[:, :, k] returns whichever class sits at index k, so an assumed
    index would silently invert every plot if the labels were ever
    renamed.
    """
    explanations = explain_by_round(dataset, **small())

    for explanation in explanations:
        assert explanation.classes == list(dataset.classes)
        assert explanation.positive_class in dataset.classes
        assert explanation.positive_class == DEFAULT_POSITIVE_CLASS
        assert isinstance(explanation.base_value, float)
        assert np.isfinite(explanation.base_value)


def test_unknown_positive_class_is_refused(dataset):
    with pytest.raises(ExplanationError) as raised:
        explain_by_round(dataset, positive_class="chrome", **small())

    assert "chrome" in str(raised.value)


# --------------------------------------------------------------
# The regression guard for the held-out rule
# --------------------------------------------------------------

def test_only_held_out_rows_are_explained(dataset):
    """
    Each fold must explain exactly its held-out round's rows, and no
    training rows.

    This is the test that fails if the module is ever "simplified" into
    explaining one model fitted on everything. Nothing else would
    notice: the plots would still render, and a model explaining data
    it was fitted on produces tidier attributions, not obviously broken
    ones. A leak here improves the output, so it cannot be caught by
    looking at the output.
    """
    explanations = explain_by_round(dataset, **small())
    folds = list(iter_round_folds(dataset))

    assert len(explanations) == len(folds)

    for explanation, fold in zip(explanations, folds):
        expected_rows = dataset.X[fold.test_index]

        assert explanation.held_out_round == fold.held_out_round
        assert np.array_equal(explanation.X_test, expected_rows)
        assert np.array_equal(explanation.pages, dataset.pages[fold.test_index])
        assert np.array_equal(explanation.y_true, dataset.y[fold.test_index])

        # Every explained row belongs to the held-out round, and the
        # count matches the round exactly -- so no training row slipped
        # in and none of the held-out rows were dropped.
        explained_groups = {
            int(group) for group in dataset.groups[fold.test_index]
        }
        assert explained_groups == {fold.held_out_round}
        assert len(explanation.X_test) == len(fold.test_index)
        assert len(explanation.X_test) < len(dataset), (
            "a fold explained the whole dataset -- the split was bypassed"
        )


def test_explanations_cover_every_trace_exactly_once(dataset):
    """Across folds, every trace is explained once, as a held-out row."""
    total = sum(len(explanation) for explanation in explain_by_round(dataset, **small()))

    assert total == len(dataset)


# --------------------------------------------------------------
# Importance
# --------------------------------------------------------------

def test_informative_feature_ranks_top(dataset):
    ranking = aggregate_importance(explain_by_round(dataset, **small()))

    assert ranking[0][0] == "syn_count"
    assert ranking[0][1] > 0
    assert [name for name, _ in ranking] == sorted(
        [name for name, _ in ranking],
        key=lambda name: -dict(ranking)[name],
    )


def test_aggregate_importance_pools_over_traces_not_fold_means(dataset):
    """
    Pooled over every held-out prediction, matching how
    evaluate_by_round builds its accuracy. With equal-sized folds the
    two agree, so the test compares against the pooled computation
    directly rather than against the fold-mean shortcut.
    """
    explanations = explain_by_round(dataset, **small())
    ranking = dict(aggregate_importance(explanations))

    stacked = np.concatenate(
        [np.abs(e.shap_values) for e in explanations], axis=0
    ).mean(axis=0)

    for index, name in enumerate(FEATURES):
        assert ranking[name] == pytest.approx(float(stacked[index]))


def test_fold_importance_table_has_one_value_per_fold(dataset):
    explanations = explain_by_round(dataset, **small())
    table = fold_importance_table(explanations)

    assert set(table) == set(FEATURES)
    for values in table.values():
        assert len(values) == len(ROUNDS)
        assert all(value >= 0 for value in values)


def test_importance_spread_reports_movement_between_rounds(dataset):
    """
    The number that says whether a ranking may be quoted as a finding.
    With redundant features, Shapley credit moves between correlated
    features across folds, and that movement is about the redundancy,
    not about the client.
    """
    explanations = explain_by_round(dataset, **small())
    spread = importance_spread(explanations)
    table = fold_importance_table(explanations)

    assert set(spread) == set(FEATURES)
    for name, value in spread.items():
        assert value == pytest.approx(max(table[name]) - min(table[name]))
        assert value >= 0


def test_mean_abs_is_per_feature_and_non_negative(dataset):
    explanation = explain_by_round(dataset, **small())[0]
    means = explanation.mean_abs()

    assert set(means) == set(FEATURES)
    assert all(value >= 0 for value in means.values())
    assert means["syn_count"] == pytest.approx(
        float(np.abs(explanation.shap_values[:, 0]).mean())
    )


def test_empty_input_aggregates_to_empty():
    assert aggregate_importance([]) == []
    assert fold_importance_table([]) == {}
    assert importance_spread([]) == {}


# --------------------------------------------------------------
# Trees only
# --------------------------------------------------------------

def test_logistic_regression_cannot_be_explained_here(dataset):
    """
    TreeExplainer does not apply, and that pipeline carries a
    StandardScaler, so its SHAP values would be in scaled units rather
    than bytes, seconds and packets. Silently switching explainer type
    would produce a plot that looks the same and means something else.
    """
    with pytest.raises(ExplanationError) as raised:
        explain_by_round(dataset, model="logistic_regression", **small())

    message = str(raised.value)
    assert "logistic_regression" in message
    assert "StandardScaler" in message


def test_determinism(dataset):
    first = aggregate_importance(explain_by_round(dataset, **small()))
    second = aggregate_importance(explain_by_round(dataset, **small()))

    assert first == second
