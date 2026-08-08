"""
shap_explain.py
---------------
SHAP attributions for the client classifier, computed the same way the
accuracy is: per fold, on the held-out round only.

Library code. No plotting, no file writing, no CLI -- a later step adds
scripts/explain_model.py on top of this.

What this module is explaining
------------------------------
The measured result is uncomfortable in a specific way. Random forest
scores **1.0000** under LeaveOneGroupOut on round (3 rounds, 600 traces,
53 features), and the ablation says removing *any* single feature family
still gives 1.0000; only `syn_count` alone gives 0.9900. The signal is
**redundant**: no one family carries it, several would do on their own.

That changes what a SHAP plot is for here. It cannot answer "which
feature is responsible", because the honest answer is "several, and any
of them would have done". What it can answer is which features this
particular fitted forest actually used, and how stable that choice is
between rounds -- which is why `fold_importance_table()` exists
alongside the pooled ranking.

Explaining held-out rounds only
-------------------------------
The folds come from `model.iter_round_folds()`. This module does not
write its own splitter, because two splitters drift and nothing about
the drift is loud: the metrics would stay honest while the plots quietly
described training data. Worse, the leaked plots would look *better* --
a model explaining data it was fitted on gives tidier, more confident
attributions. The same rule as the evaluation: a leak here improves the
output, so it can never be caught by looking at the output.

Attribution scheme
------------------
`feature_perturbation="tree_path_dependent"`, deliberately, and not the
"interventional" default.

With 53 heavily correlated features, the interventional scheme evaluates
each feature against a marginal background, which breaks correlated
groups apart. A feature that is genuinely redundant but rarely split on
then receives near-zero attribution -- producing a confident "one
feature does everything" picture that would **directly contradict the
measured ablation**, where removing any single family changes nothing.
The tree-path-dependent scheme uses conditional path counts, so credit
is spread across redundant features, which is what the data actually
says. It also needs no background dataset, so there is no background
sample to choose and therefore no knob to tune toward a nicer plot.

Environment this was written against: shap 0.52.0, scikit-learn 1.9.0,
numpy 2.4.6. On this combination `TreeExplainer.shap_values()` for a
binary `RandomForestClassifier` returns a single ndarray of shape
`(n_samples, n_features, n_classes)` -- not the older list of two
arrays -- and `expected_value` has shape `(2,)` and sums to 1.0, so the
values are in probability space rather than log-odds. Both facts are
asserted at runtime rather than trusted, because shap has changed this
return shape between versions before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shap

from .model import Dataset, build_pipeline, iter_round_folds

# The class attributions point toward. This project detects automation,
# so "positive" means "this is the automated client": a positive SHAP
# value pushes the prediction toward wget.
DEFAULT_POSITIVE_CLASS = "wget"

# Only tree ensembles are explained here. See _tree_classifier().
TREE_MODELS = ("random_forest",)

TREE_PERTURBATION = "tree_path_dependent"


class ExplanationError(RuntimeError):
    """The model or the SHAP output is not what this module can explain."""


@dataclass
class FoldExplanation:
    """SHAP attributions for one held-out round."""

    held_out_round: int
    positive_class: str
    classes: list[str]
    features: list[str]
    shap_values: np.ndarray  # (n_test, n_features), for positive_class
    base_value: float
    X_test: np.ndarray  # (n_test, n_features)
    pages: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    def mean_abs(self) -> dict[str, float]:
        """Mean |SHAP| per feature, for this fold."""
        means = np.abs(self.shap_values).mean(axis=0)
        return {
            name: float(means[index]) for index, name in enumerate(self.features)
        }

    def __len__(self) -> int:
        return len(self.shap_values)


# --------------------------------------------------------------
# Explaining
# --------------------------------------------------------------

def explain_by_round(
    dataset: Dataset,
    model: str = "random_forest",
    positive_class: str = DEFAULT_POSITIVE_CLASS,
    **pipeline_kwargs,
) -> list[FoldExplanation]:
    """
    One explanation per fold, each computed on that fold's held-out
    round.

    The pipeline is fitted on `fold.train_index` and explained on
    `fold.test_index`, mirroring `model.evaluate_by_round()` exactly.
    Explaining a model on rows it was fitted on would reintroduce, in
    the explanation, precisely the leak the round-based split exists to
    prevent in the evaluation -- and it would do so invisibly, because
    the resulting plots are tidier rather than obviously wrong.
    """
    if model not in TREE_MODELS:
        raise ExplanationError(
            f"{model!r} cannot be explained here. TreeExplainer applies to "
            f"tree ensembles, and the logistic_regression pipeline also "
            f"carries a StandardScaler, so its SHAP values would be in "
            f"scaled units instead of bytes, seconds and packets.\n"
            f"Switching explainer type silently would produce a plot that "
            f"looks the same and means something else; explaining "
            f"{TREE_MODELS} is what this module claims to do."
        )

    explanations: list[FoldExplanation] = []

    for fold in iter_round_folds(dataset):
        pipeline = build_pipeline(model, **pipeline_kwargs)
        pipeline.fit(dataset.X[fold.train_index], dataset.y[fold.train_index])

        classifier = _tree_classifier(pipeline)
        classes = [str(label) for label in classifier.classes_]
        _check_classes(classes, dataset)

        # Read from the FITTED model, never assumed. sv[:, :, k] returns
        # the attribution for whichever class sits at index k, and today
        # sorted(["firefox", "wget"]) happens to put wget at 1 -- an
        # implementation detail of label sorting. Relabelling one class
        # to "ff" would flip the order, invert every plot, and raise
        # nothing. Same failure class as reading a clipped frame length
        # in features.py: plausible output, no exception.
        if positive_class not in classes:
            raise ExplanationError(
                f"positive class {positive_class!r} is not among the fitted "
                f"model's classes {classes}"
            )
        class_index = classes.index(positive_class)

        X_test = dataset.X[fold.test_index]
        explainer = shap.TreeExplainer(
            classifier, feature_perturbation=TREE_PERTURBATION
        )
        values = explainer.shap_values(X_test)

        values = _check_shape(values, X_test, classes)

        explanations.append(FoldExplanation(
            held_out_round=fold.held_out_round,
            positive_class=positive_class,
            classes=classes,
            features=list(dataset.features),
            shap_values=values[:, :, class_index],
            base_value=_base_value(explainer, class_index),
            X_test=X_test,
            pages=dataset.pages[fold.test_index],
            y_true=dataset.y[fold.test_index],
            y_pred=pipeline.predict(X_test),
        ))

    return explanations


def _tree_classifier(pipeline):
    """
    The tree ensemble inside the pipeline, if that is all there is.

    A transformer in front of the classifier would mean the SHAP values
    describe transformed inputs while the feature names still say
    "bytes" and "seconds". The forest pipeline deliberately has no
    scaler -- trees are invariant to monotone rescaling -- so this is
    also a guard against someone adding one later for tidiness.
    """
    steps = list(pipeline.named_steps)

    if steps != ["classifier"]:
        raise ExplanationError(
            f"expected a pipeline of only a classifier, found {steps}. "
            f"A transformer before the model would leave the SHAP values in "
            f"transformed units while the feature names still claim bytes "
            f"and seconds."
        )

    return pipeline.named_steps["classifier"]


def _check_classes(classes: list[str], dataset: Dataset) -> None:
    if classes != list(dataset.classes):
        raise ExplanationError(
            f"the fitted model's classes {classes} do not match the "
            f"dataset's {list(dataset.classes)}; class indices cannot be "
            f"trusted"
        )


def _check_shape(values, X_test: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Assert the SHAP return shape rather than trusting the version.

    shap has changed this between releases -- older versions returned a
    list of one array per class. A 2D array indexed as if it were 3D
    would not raise; it would silently take a column of attributions
    along the wrong axis and plot it. So an unexpected shape is a hard
    error, and an upgrade that changes it fails here instead of
    downstream.
    """
    array = np.asarray(values)
    expected = (len(X_test), X_test.shape[1], len(classes))

    if array.ndim != 3 or array.shape != expected:
        raise ExplanationError(
            f"unexpected SHAP output: shape {array.shape} (ndim {array.ndim}), "
            f"expected {expected} = (n_test, n_features, n_classes).\n"
            f"This module was written against shap 0.52.0, which returns a "
            f"single 3-D array for a binary classifier. A different shape "
            f"means the shap version changed its contract -- fix the indexing "
            f"deliberately rather than letting an axis be read wrongly."
        )

    return array


def _base_value(explainer, class_index: int) -> float:
    """expected_value for one class; scalar and per-class forms both occur."""
    expected = np.asarray(explainer.expected_value)

    if expected.ndim == 0:
        return float(expected)

    return float(expected[class_index])


# --------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------

def aggregate_importance(
    explanations: list[FoldExplanation],
) -> list[tuple[str, float]]:
    """
    Pooled mean |SHAP| per feature, descending.

    Pooled over TRACES, not averaged over per-fold means. Every held-out
    prediction counts once, which is how `evaluate_by_round` builds its
    accuracy too -- averaging fold averages would weight a small round
    the same as a large one and quietly describe a different quantity.
    """
    if not explanations:
        return []

    features = explanations[0].features
    stacked = np.concatenate(
        [np.abs(explanation.shap_values) for explanation in explanations], axis=0
    )
    means = stacked.mean(axis=0)

    ranked = [(name, float(means[index])) for index, name in enumerate(features)]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def fold_importance_table(
    explanations: list[FoldExplanation],
) -> dict[str, list[float]]:
    """
    Mean |SHAP| per feature, per fold, so stability can be inspected.

    This exists because of what the ablation measured. With fully
    redundant features, Shapley credit is split among correlated
    features in a way that can shift between models fitted on different
    rounds: a feature ranked first on one round and mid-table on
    another is telling us about the redundancy, not about the client.

    A single pooled ranking would hide that, and would invite a
    confident sentence in the case study that the data does not
    support. The case study reports the per-round spread instead.
    """
    if not explanations:
        return {}

    features = explanations[0].features
    per_fold = [explanation.mean_abs() for explanation in explanations]

    return {name: [fold[name] for fold in per_fold] for name in features}


def importance_spread(explanations: list[FoldExplanation]) -> dict[str, float]:
    """
    How much each feature's importance moves between rounds.

    max - min of the per-fold mean |SHAP|. A large spread on a
    high-ranking feature is the signature of redundancy being shared
    out differently per fold, and it is the number that decides whether
    a ranking may be quoted as a finding.
    """
    table = fold_importance_table(explanations)

    return {
        name: float(max(values) - min(values)) for name, values in table.items()
    }
