#!/usr/bin/env python3
"""
train_model.py
--------------
Evaluate the Firefox-vs-wget classifier with a round-based split, write
results/metrics.json, and save a model for the step-8 CLI.

All the logic lives in src/tsd/model.py; this script parses arguments,
writes files and prints. Same split as everywhere else in this repo.

    PYTHONPATH=src python scripts/train_model.py
    PYTHONPATH=src python scripts/train_model.py --force

Two numbers must never be confused, so the script prints them
differently:

    the REPORTED accuracy comes from held-out rounds -- each round is
    predicted by a model that never saw it

    the SHIPPED model in models/ is fitted on every round, because a
    tool should use all the evidence available. It has no honest
    accuracy of its own and none is printed.

Requires at least two capture rounds in the feature CSV. With one round
there is nothing to hold out that does not share its conditions, and
src/tsd/model.py refuses rather than falling back to a random split --
a fallback would run, print a high number, and say nothing about why
that number is meaningless.

Exit codes:

    0   evaluated and written
    1   the dataset cannot be split by round (fewer than two rounds)
    2   refused to overwrite existing results
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from tsd.model import (
    MODEL_NAMES,
    N_ESTIMATORS,
    RANDOM_STATE,
    DatasetError,
    build_metrics,
    evaluate_by_round,
    fit_final_model,
    load_dataset,
    misclassified_pages,
    write_metrics,
)

DEFAULT_FEATURES = Path("data/features/features.csv")
DEFAULT_METRICS = Path("results/metrics.json")
DEFAULT_MODEL = Path("models/client_classifier.joblib")

EXIT_OK = 0
EXIT_CANNOT_SPLIT = 1
EXIT_REFUSED = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the client classifier, split by capture round.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                        help="feature CSV from scripts/extract_features.py")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS,
                        help="where the published metrics are written")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL,
                        help="where the model for the step-8 CLI is saved")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS,
                        help="trees in the random forest")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="fixed so a re-run reproduces the same numbers")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing metrics and model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in (args.metrics, args.model_out):
        if path.exists() and not args.force:
            print(f"ABORT: {path} already exists. Use --force to replace it.",
                  file=sys.stderr)
            return EXIT_REFUSED

    try:
        dataset = load_dataset(args.features)
    except DatasetError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_CANNOT_SPLIT
    except OSError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return EXIT_REFUSED

    print(f"{len(dataset)} traces, {len(dataset.rounds)} rounds "
          f"{dataset.rounds}, {len(dataset.features)} features",
          file=sys.stderr)
    print(f"split: LeaveOneGroupOut on `round` -> {len(dataset.rounds)} folds",
          file=sys.stderr)

    evaluations = {}
    for name in MODEL_NAMES:
        print(f"  evaluating {name} ...", file=sys.stderr, flush=True)
        evaluations[name] = evaluate_by_round(
            dataset,
            model=name,
            random_state=args.seed,
            n_estimators=args.n_estimators,
        )

    metrics = build_metrics(
        dataset=dataset,
        evaluations=evaluations,
        hyperparameters={
            "random_state": args.seed,
            "n_estimators": args.n_estimators,
            "models": list(MODEL_NAMES),
        },
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    write_metrics(args.metrics, metrics)

    # Fitted on everything, for the CLI. Deliberately after the metrics
    # are written, and never scored here.
    final = fit_final_model(
        dataset,
        model="random_forest",
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": final, "features": dataset.features,
         "classes": dataset.classes, "rounds": dataset.rounds},
        args.model_out,
    )

    print_summary(args, dataset, evaluations)
    return EXIT_OK


def print_summary(args, dataset, evaluations) -> None:
    print("evaluated with a round-based split")
    print(f"  traces        : {len(dataset)}")
    print(f"  rounds        : {dataset.rounds}")
    print(f"  features      : {len(dataset.features)}")
    print()
    print("  REPORTED accuracy -- each round predicted by a model that never saw it")

    for name, evaluation in evaluations.items():
        folds = ", ".join(
            f"r{fold.held_out_round}={fold.accuracy:.3f}"
            for fold in evaluation.folds
        )
        print(f"    {name:<20} {evaluation.accuracy:.4f}   (per fold: {folds})")

        for label, scores in evaluation.per_class.items():
            print(f"      {label:<10} precision={scores['precision']:.3f} "
                  f"recall={scores['recall']:.3f} f1={scores['f1']:.3f} "
                  f"n={scores['support']}")

    print()
    for name, evaluation in evaluations.items():
        hardest = misclassified_pages(evaluation)[:5]
        if hardest:
            print(f"  hardest pages ({name}): "
                  + ", ".join(f"{page} x{count}" for page, count in hardest))
        else:
            print(f"  hardest pages ({name}): none misclassified")

    print()
    print(f"  metrics       : {args.metrics}")
    print(f"  model         : {args.model_out}")
    print("  NOTE: the saved model is fitted on ALL rounds for the step-8 CLI.")
    print("        Its accuracy is not the number above and is not reported.")


if __name__ == "__main__":
    sys.exit(main())
