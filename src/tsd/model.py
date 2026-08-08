"""
model.py
--------
Train and evaluate the Firefox-vs-wget classifier, split BY CAPTURE
ROUND.

The split is the whole experiment. Everything else here is ordinary
scikit-learn.

Why the round, and why `LeaveOneGroupOut`
-----------------------------------------
Traces from one capture round share their conditions: the same machine
state, the same kernel, the same binaries, the same afternoon. A random
split puts traces from one round on both sides, so the model is scored
on conditions it has already seen, and the score comes out high for a
reason that has nothing to do with Firefox or wget.

The dangerous part is that this failure is invisible from the result. A
leaked split produces *better* numbers, not worse ones, and nothing
downstream complains. So the grouping is enforced by the API --
`LeaveOneGroupOut(groups=round)` -- rather than by remembering to be
careful. There is no `train_test_split` in this module, no `shuffle`,
and no `random_state` that could reorder groups.

What is NOT true, and must not be asserted
------------------------------------------
"No page appears in both train and test." That is false here, by
design: every page is loaded in every round, so every page is on both
sides of every fold. Asserting it would fail, and "fixing" the split to
make it true would mean grouping by page, which answers a different
question -- *can we recognise this page?* -- and is explicitly out of
scope (100 classes with a handful of samples each is not trainable).

The claim under test is about the CLIENT, not the page. The same page
being present on both sides is exactly what forces the model to find
the difference between the two clients rather than between two pages.
What must never be shared across a fold is the ROUND, and that is what
is asserted.

Leakage in the pipeline
-----------------------
Every fold builds its own pipeline. A scaler fitted on the whole dataset
before splitting would carry test-fold statistics into training -- a
smaller leak than a bad split, and just as invisible.

The final model
---------------
`fit_final_model()` fits on every round, for the step-8 CLI to ship. Its
training accuracy is not a result and is never reported as one: the
numbers that get published come from the held-out folds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_COLUMNS = ("round", "date", "client", "page")
GROUP_COLUMN = "round"
TARGET_COLUMN = "client"

RANDOM_STATE = 42
N_ESTIMATORS = 300

MODEL_NAMES = ("random_forest", "logistic_regression")

# Feature families, for ablation. Prefix-based so a new feature joins
# its family automatically -- and a new family that matches none of
# these is caught by a test, because a feature that belongs to no group
# would quietly sit outside every ablation and never be questioned.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "counts": ("count_",),
    "sizes": ("size_", "bytes_", "ack_"),
    "timing": ("duration", "iat_"),
    "bursts": ("burst_",),
    "connections": ("syn_",),
}


class DatasetError(RuntimeError):
    """The feature table cannot be used for a round-based evaluation."""


@dataclass
class Dataset:
    """The feature table, split into the parts the evaluation needs."""

    features: list[str]
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    pages: np.ndarray

    @property
    def rounds(self) -> list[int]:
        return sorted({int(group) for group in self.groups})

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.y))

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------
# The split -- one definition, used by everything
# --------------------------------------------------------------

@dataclass(frozen=True)
class RoundFold:
    """One leave-one-round-out fold, as indices into the dataset."""

    held_out_round: int
    train_rounds: list[int]
    train_index: np.ndarray
    test_index: np.ndarray


def iter_round_folds(dataset: Dataset) -> Iterator[RoundFold]:
    """
    Yield the leave-one-round-out folds. The only splitter in this
    project.

    Why it is a shared function rather than a loop written where it is
    needed: step 7 computes SHAP values on held-out rounds, mirroring
    this evaluation. If the SHAP module wrote its own
    `LeaveOneGroupOut` loop and the two ever drifted apart, **nothing
    would fail loudly**. The metrics would stay honest while the SHAP
    plots quietly described training data, and the plots would look
    entirely reasonable -- cleaner, in fact, because a model explaining
    data it was fitted on gives tidier attributions. A leaked split
    produces BETTER output, not worse. So there is exactly one place
    that turns a Dataset into folds, and every consumer takes it from
    here.

    The per-fold assertions are redundant: `LeaveOneGroupOut` already
    guarantees a single held-out group and no overlap. They are here
    anyway, because this is the one invariant in the project whose
    violation makes the numbers look better rather than worse -- the
    class of bug that no amount of staring at results will reveal.

    No shuffle and no `random_state`. Fold order follows the group
    values, so the folds are reproducible without a seed at all. That
    matters beyond tidiness: SHAP is computed per fold, and a published
    plot that cannot be regenerated from the same inputs is not
    evidence.
    """
    splitter = LeaveOneGroupOut()

    for train_index, test_index in splitter.split(
        dataset.X, dataset.y, groups=dataset.groups
    ):
        held_out_groups = {int(group) for group in dataset.groups[test_index]}
        train_groups = {int(group) for group in dataset.groups[train_index]}

        if len(held_out_groups) != 1:
            raise DatasetError(
                f"a fold held out {len(held_out_groups)} rounds "
                f"({sorted(held_out_groups)}); leave-one-round-out must hold "
                f"out exactly one, or the fold is not a round"
            )

        shared = held_out_groups & train_groups
        if shared:
            raise DatasetError(
                f"round(s) {sorted(shared)} appear in both train and test of "
                f"the same fold. Traces from one round share their "
                f"conditions, so the model would be scored on conditions it "
                f"has already seen -- and the score would come out higher, "
                f"not lower, with nothing else to signal the problem."
            )

        yield RoundFold(
            held_out_round=next(iter(held_out_groups)),
            train_rounds=sorted(train_groups),
            train_index=train_index,
            test_index=test_index,
        )


# --------------------------------------------------------------
# Loading
# --------------------------------------------------------------

def load_dataset(csv_path: str | Path) -> Dataset:
    """
    Read the feature CSV and check it can be split by round.

    The group check happens here, at load time, and raises. A fallback
    -- "only one round, so let's do a random split just this once" --
    would be the worst possible behaviour: it would run, it would print
    a high number, and nothing about the output would say the number
    means nothing.
    """
    frame = pd.read_csv(csv_path)

    missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"{csv_path}: missing label column(s) {missing}")

    features = [column for column in frame.columns if column not in LABEL_COLUMNS]
    if not features:
        raise DatasetError(f"{csv_path}: no feature columns")

    groups = frame[GROUP_COLUMN].to_numpy()
    rounds = sorted(set(int(group) for group in groups))

    if len(rounds) < 2:
        raise DatasetError(
            f"{csv_path} contains {len(rounds)} capture round "
            f"({', '.join(str(r) for r in rounds) or 'none'}).\n"
            f"The train/test split is BY ROUND, so at least two rounds are "
            f"needed -- with one there is nothing to hold out that does not "
            f"share its conditions.\n"
            f"Capture another round on a different day: "
            f"scripts/capture_round.py --round {(rounds[0] + 1) if rounds else 1}\n"
            f"Three or more are wanted, so that each fold trains on more than "
            f"half the data."
        )

    labels = frame[TARGET_COLUMN].to_numpy()
    if len(set(labels)) < 2:
        raise DatasetError(
            f"{csv_path}: only one class present ({set(labels)}); "
            f"nothing to separate"
        )

    return Dataset(
        features=features,
        X=frame[features].to_numpy(dtype=float),
        y=labels,
        groups=groups,
        pages=frame["page"].to_numpy(),
    )


# --------------------------------------------------------------
# Models
# --------------------------------------------------------------

def build_pipeline(
    name: str,
    random_state: int = RANDOM_STATE,
    n_estimators: int = N_ESTIMATORS,
) -> Pipeline:
    """
    A fresh, unfitted pipeline.

    Called once per fold, never once for the whole run: a transformer
    fitted before the split has seen the test fold, and scaling
    statistics are enough to carry information across.

    The two models are a pair on purpose. The forest is the baseline;
    logistic regression on scaled features is the comparison point that
    says whether the ensemble is earning its complexity. A forest that
    cannot beat a linear model on 53 features is worth knowing about
    before it goes into the SHAP plots.
    """
    if name == "random_forest":
        # No scaler: trees are invariant to monotone rescaling, and
        # leaving it out keeps the SHAP values in the units the features
        # are actually measured in -- bytes, seconds, packets.
        return Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=5000,
                random_state=random_state,
            )),
        ])

    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


# --------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------

@dataclass
class FoldResult:
    held_out_round: int
    train_rounds: list[int]
    n_train: int
    n_test: int
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    labels: list[str]
    misclassified: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "held_out_round": self.held_out_round,
            "train_rounds": self.train_rounds,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
            "labels": self.labels,
            "misclassified": self.misclassified,
        }


@dataclass
class EvaluationResult:
    model: str
    folds: list[FoldResult]
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    labels: list[str]
    misclassified: list[dict]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
            "labels": self.labels,
            "n_folds": len(self.folds),
            "fold_accuracies": [fold.accuracy for fold in self.folds],
            "folds": [fold.to_dict() for fold in self.folds],
            "misclassified": self.misclassified,
        }


def evaluate_by_round(
    dataset: Dataset,
    model: str = "random_forest",
    pipeline_factory=None,
    **pipeline_kwargs,
) -> EvaluationResult:
    """
    Leave-one-round-out cross-validation.

    Every trace is predicted exactly once, in the fold where its round
    is held out, so the aggregate is a pooled prediction over the whole
    dataset rather than an average of averages.

    `pipeline_factory` is injectable so a test can prove the pipeline is
    built inside the fold rather than reused across folds.
    """
    factory = pipeline_factory or (
        lambda: build_pipeline(model, **pipeline_kwargs)
    )
    labels = dataset.classes

    folds: list[FoldResult] = []
    pooled_truth: list[str] = []
    pooled_predicted: list[str] = []
    pooled_misclassified: list[dict] = []

    # The folds come from iter_round_folds(), which is also what the
    # step-7 SHAP module uses. One definition, so the explanations
    # describe exactly the held-out predictions these metrics report.
    for fold in iter_round_folds(dataset):
        train_index, test_index = fold.train_index, fold.test_index
        held_out = fold.held_out_round

        pipeline = factory()
        pipeline.fit(dataset.X[train_index], dataset.y[train_index])
        predicted = pipeline.predict(dataset.X[test_index])
        truth = dataset.y[test_index]

        wrong = [
            {
                "round": held_out,
                "page": str(dataset.pages[index]),
                "true": str(truth[position]),
                "predicted": str(predicted[position]),
            }
            for position, index in enumerate(test_index)
            if truth[position] != predicted[position]
        ]

        folds.append(FoldResult(
            held_out_round=held_out,
            train_rounds=fold.train_rounds,
            n_train=len(train_index),
            n_test=len(test_index),
            accuracy=float(accuracy_score(truth, predicted)),
            per_class=_per_class(truth, predicted, labels),
            confusion=confusion_matrix(truth, predicted, labels=labels).tolist(),
            labels=labels,
            misclassified=wrong,
        ))

        pooled_truth.extend(truth)
        pooled_predicted.extend(predicted)
        pooled_misclassified.extend(wrong)

    return EvaluationResult(
        model=model,
        folds=folds,
        accuracy=float(accuracy_score(pooled_truth, pooled_predicted)),
        per_class=_per_class(pooled_truth, pooled_predicted, labels),
        confusion=confusion_matrix(
            pooled_truth, pooled_predicted, labels=labels
        ).tolist(),
        labels=labels,
        misclassified=pooled_misclassified,
    )


def _per_class(truth, predicted, labels: list[str]) -> dict[str, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )

    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }


# --------------------------------------------------------------
# Ablation -- what is actually carrying the result
# --------------------------------------------------------------
#
# A perfect score is a hypothesis about the rig, not a result. This
# repo has already thrown one feature away for that reason: fin_count
# and rst_count separated the classes 100/100 and were measuring how
# the harness stops each client, not how the clients behave.
#
# syn_count is NOT that situation. Six connections against one happens
# during the load, it is the parallelism the threaded server exists to
# allow, and it is what a browser does that a sequential fetcher does
# not. It is client behaviour and it belongs in the data.
#
# But "this feature is legitimate" and "this feature is the only thing
# carrying the result" are different statements, and only the second
# one decides what the README may claim. If the score survives without
# the connection family, the claim is "traffic shape separates the
# classes". If it collapses to chance, the honest claim is narrower:
# "connection parallelism separates the classes, and the remaining
# shape features add little". The ablation is how that gets decided by
# measurement rather than by preference.

def select_features(
    dataset: Dataset,
    keep: list[str] | None = None,
    drop: list[str] | None = None,
) -> Dataset:
    """
    A view of the dataset with some feature columns removed.

    Unknown names raise rather than being ignored: an ablation that
    silently dropped nothing would report the headline accuracy under
    the label of an experiment that never ran.
    """
    known = set(dataset.features)

    for label, names in (("keep", keep), ("drop", drop)):
        unknown = sorted(set(names or []) - known)
        if unknown:
            raise DatasetError(
                f"unknown feature(s) in --{label}: {', '.join(unknown)}"
            )

    if keep is not None:
        chosen = [name for name in dataset.features if name in set(keep)]
    else:
        excluded = set(drop or [])
        chosen = [name for name in dataset.features if name not in excluded]

    if not chosen:
        raise DatasetError("no features left after selection")

    indices = [dataset.features.index(name) for name in chosen]

    return Dataset(
        features=chosen,
        X=dataset.X[:, indices],
        y=dataset.y,
        groups=dataset.groups,
        pages=dataset.pages,
    )


def group_of(feature: str) -> str | None:
    """Which family a feature belongs to, or None if it belongs to none."""
    for group, prefixes in FEATURE_GROUPS.items():
        if any(feature.startswith(prefix) for prefix in prefixes):
            return group
    return None


def features_in_group(features: list[str], group: str) -> list[str]:
    return [name for name in features if group_of(name) == group]


def ablation_configurations(features: list[str]) -> list[tuple[str, list[str]]]:
    """
    The preset sweep, as (label, features to use).

    Ordered from "everything" to "one feature", so the table reads as a
    single question getting narrower: how much of the result survives
    when each family is taken away, and how much does the single
    strongest feature explain on its own?
    """
    configurations: list[tuple[str, list[str]]] = [("all features", list(features))]

    if "syn_count" in features:
        configurations.append(
            ("without syn_count", [f for f in features if f != "syn_count"])
        )

    for group in FEATURE_GROUPS:
        remaining = [name for name in features if group_of(name) != group]
        if remaining and len(remaining) != len(features):
            configurations.append((f"without {group}", remaining))

    if "syn_count" in features:
        configurations.append(("only syn_count", ["syn_count"]))

    return configurations


@dataclass
class AblationResult:
    label: str
    features: list[str]
    accuracy: float
    fold_accuracies: list[float]

    def to_dict(self) -> dict:
        return {
            "configuration": self.label,
            "n_features": len(self.features),
            "features": self.features,
            "accuracy": self.accuracy,
            "fold_accuracies": self.fold_accuracies,
        }


def run_ablation(
    dataset: Dataset,
    configurations: list[tuple[str, list[str]]],
    model: str = "random_forest",
    **pipeline_kwargs,
) -> list[AblationResult]:
    """
    Re-run the identical LeaveOneGroupOut protocol per configuration.

    Identical on purpose: an ablation evaluated any other way would not
    be comparable with the headline number, and the comparison is the
    entire point.
    """
    results: list[AblationResult] = []

    for label, features in configurations:
        subset = select_features(dataset, keep=features)
        evaluation = evaluate_by_round(subset, model=model, **pipeline_kwargs)

        results.append(AblationResult(
            label=label,
            features=list(subset.features),
            accuracy=evaluation.accuracy,
            fold_accuracies=[fold.accuracy for fold in evaluation.folds],
        ))

    return results


def misclassified_pages(result: EvaluationResult) -> list[tuple[str, int]]:
    """
    Pages that were got wrong, most often first.

    Worth looking at rather than summarising away: CLAUDE.md predicts
    the asset-free pages (`ikmz_xwiki`, `webmail`) will be the hardest,
    because with almost nothing to fetch the two clients have little
    room to behave differently. That is a prediction, and this is where
    it gets checked rather than assumed.
    """
    counts: dict[str, int] = {}
    for entry in result.misclassified:
        counts[entry["page"]] = counts.get(entry["page"], 0) + 1

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


# --------------------------------------------------------------
# The shipped model
# --------------------------------------------------------------

def fit_final_model(
    dataset: Dataset, model: str = "random_forest", **pipeline_kwargs
) -> Pipeline:
    """
    Fit on every round, for the step-8 CLI to load.

    This model has no honest accuracy of its own: it has seen every
    trace. The published number comes from `evaluate_by_round`, and the
    two must never be confused -- which is why this function returns a
    fitted pipeline and no score at all.
    """
    pipeline = build_pipeline(model, **pipeline_kwargs)
    pipeline.fit(dataset.X, dataset.y)
    return pipeline


# --------------------------------------------------------------
# Metrics document
# --------------------------------------------------------------

def build_metrics(
    dataset: Dataset,
    evaluations: dict[str, EvaluationResult],
    hyperparameters: dict,
    generated_at: str,
    ablation: dict[str, list[AblationResult]] | None = None,
) -> dict:
    """
    The published metrics document.

    Names and numbers only -- no BTU content, and nothing derived from
    packet payload, which was never captured. Page names appear because
    knowing WHICH pages are hard is the interesting part, and those
    names are already published in results/corpus_manifest.json.
    """
    return {
        "generated_at": generated_at,
        "split": {
            "method": "LeaveOneGroupOut",
            "group_column": GROUP_COLUMN,
            "rounds": dataset.rounds,
            "n_folds": len(dataset.rounds),
            "note": (
                "Every page appears in every round and therefore on both "
                "sides of every fold. That is by design: the claim under "
                "test is about the client, not the page, and the same page "
                "on both sides is what forces the model to separate the "
                "clients. What is never shared across a fold is the round."
            ),
        },
        "dataset": {
            "traces": len(dataset),
            "rounds": len(dataset.rounds),
            "classes": dataset.classes,
            "per_class": {
                label: int((dataset.y == label).sum()) for label in dataset.classes
            },
            "n_features": len(dataset.features),
            "features": dataset.features,
        },
        "hyperparameters": hyperparameters,
        "models": {
            name: evaluation.to_dict() for name, evaluation in evaluations.items()
        },
        "hardest_pages": {
            name: [
                {"page": page, "errors": count}
                for page, count in misclassified_pages(evaluation)[:20]
            ]
            for name, evaluation in evaluations.items()
        },
        "final_model_note": (
            "models/ holds a model fitted on ALL rounds, for the step-8 CLI. "
            "Its training accuracy is not reported anywhere and is not a "
            "result: the accuracies above come from held-out rounds."
        ),
        "ablation": _ablation_document(ablation),
    }


def _ablation_document(ablation) -> dict | None:
    """
    The ablation section: additional evidence, never a replacement.

    Kept under its own key and clearly labelled, because the headline
    number and the ablation answer different questions. The headline
    says how well the classifier does; the ablation says what the
    classifier is doing it with, and therefore which sentence the README
    is entitled to write.
    """
    if not ablation:
        return None

    return {
        "note": (
            "Same LeaveOneGroupOut protocol, re-run with feature families "
            "removed. This does not replace the headline evaluation above; "
            "it says which features carry it. A score that survives without "
            "the connections family supports the claim 'traffic shape "
            "separates the classes'. A score that collapses to chance "
            "without syn_count supports only the narrower claim "
            "'connection parallelism separates the classes'."
        ),
        "groups": {
            group: list(prefixes) for group, prefixes in FEATURE_GROUPS.items()
        },
        "models": {
            name: [result.to_dict() for result in results]
            for name, results in ablation.items()
        },
    }


def write_metrics(path: str | Path, metrics: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
