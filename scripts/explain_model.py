#!/usr/bin/env python3
"""
explain_model.py
----------------
Render the step-7 SHAP artefacts: three plots and a JSON summary, into
results/.

All the computation lives in src/tsd/shap_explain.py and the feature
families in src/tsd/model.py. This script orchestrates and renders; it
re-implements nothing, and in particular it does not define a second
feature-family mapping -- model.py owns that, and a test already fails
if a feature belongs to no family.

    pip install -e .
    python scripts/explain_model.py
    python scripts/explain_model.py --force

Why three plots and not one
---------------------------
Measured over 4 rounds, 800 traces, 53 features: re-fitting with
`random_state=7` instead of 42 moves the feature ranking by up to **24
places** while accuracy stays **1.0000**. Publishing a single beeswarm
would therefore present a seed-dependent ordering as if it were the
result.

But the same measurement showed something else: the top **ten**
features are the same **set** under both seeds, only reordered. So
attribution here is **stable at family level and unstable at feature
level**, and the output is built to say exactly that and nothing
stronger.

The three plots are one argument, in order:

    1. family-level bar chart   the claim that survives
    2. feature-level beeswarm   informative, but seed-dependent
    3. stability chart          the measurement that bounds plot 2

A reader who stops after plot 1 has the honest headline. A reader who
reaches plot 2 sees plot 3 next to it.

That ordering is also why the caveat lives in plot 2's **title** rather
than only in a caption. A plot travels without its surrounding page --
screenshots get pasted into slides and READMEs -- so the qualification
has to survive being cropped out of context. A figure that is honest
only in the presence of its paragraph is not honest.

Publishable
-----------
The plots and the JSON contain feature names and numbers. No payload was
ever captured (`-s 96`), and nothing from `data/` is embedded, so these
go in results/ alongside the metrics.

Exit codes:

    0   written
    1   the dataset cannot be split by round
    2   refused to overwrite existing output
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside a capture host or CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from tsd.model import (  # noqa: E402
    FEATURE_GROUPS,
    RANDOM_STATE,
    DatasetError,
    group_of,
    load_dataset,
)
from tsd.shap_explain import (  # noqa: E402
    aggregate_importance,
    explain_by_round,
)

DEFAULT_FEATURES = Path("data/features/features.csv")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_COMPARE_SEED = 7

FAMILY_PLOT = "shap_family_importance.png"
BEESWARM_PLOT = "shap_feature_beeswarm.png"
STABILITY_PLOT = "shap_stability.png"
SUMMARY_JSON = "shap_summary.json"

TOP_FEATURES = 15
BEESWARM_FEATURES = 15

# Fixed so re-running produces the same images. `Software: None` drops
# matplotlib's version stamp from the PNG, so the bytes do not change
# when matplotlib is upgraded.
FIGURE_DPI = 150
PNG_METADATA = {"Software": None}

# Captions are COMPUTED (see family_reading, stability_reading), so
# their length is not known when the figure is laid out. The first
# rendered version clipped plot 1's subtitle at "...is seed-" and lost
# the word "dependent" -- which was the entire clause the caption had
# just been rewritten to carry.
#
# Wrapping rather than shortening, deliberately: a shorter sentence
# fits today and the next dataset produces a longer one, silently
# clipping again. These widths are measured against the 9-inch figures
# at their title font sizes, with headroom.
TITLE_WRAP = 68  # axes titles, default title font size
CAVEAT_WRAP = 78  # plot 2's caveat, drawn at fontsize 10

# The same-day and cross-day pairs, by ROUND NUMBER rather than by
# position in the fold list. Rounds 1 and 2 are the same local day about
# 10 hours apart; rounds 3 and 4 are different days. Keying by number
# means reordering the folds cannot silently relabel the comparison.
SAME_DAY_PAIR = (1, 2)
CROSS_DAY_PAIR = (3, 4)

EXIT_OK = 0
EXIT_CANNOT_SPLIT = 1
EXIT_REFUSED = 2

SPLIT_DESCRIPTION = (
    "LeaveOneGroupOut on capture round; SHAP computed per fold on the "
    "held-out round only, never on training rows"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the SHAP plots and summary for step 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                        help="feature CSV from scripts/extract_features.py")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="directory the plots and summary are written to")
    parser.add_argument("--model", default="random_forest",
                        help="model to explain; only tree ensembles are supported")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="the seed whose ranking the beeswarm shows")
    parser.add_argument("--compare-seed", type=int, default=DEFAULT_COMPARE_SEED,
                        help="second seed, to measure how far the ranking moves")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing plots and summary")
    return parser.parse_args(argv)


# --------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------

def pooled_importance(explanations) -> dict[str, float]:
    """Pooled mean |SHAP| per feature, as a dict."""
    return dict(aggregate_importance(explanations))


def family_importance(pooled: dict[str, float]) -> dict[str, float]:
    """
    Pooled importance summed per feature family.

    Families come from `model.FEATURE_GROUPS` via `model.group_of()`.
    Defining a second mapping here would be a second source of truth for
    the same fact, and the two would eventually disagree -- silently,
    since both would still produce a plausible bar chart.
    """
    totals = {family: 0.0 for family in FEATURE_GROUPS}

    for feature, value in pooled.items():
        family = group_of(feature)
        if family is None:  # pragma: no cover - a test in test_model guards this
            raise KeyError(
                f"{feature!r} belongs to no family in model.FEATURE_GROUPS"
            )
        totals[family] += value

    return totals


def per_round_importance(explanations) -> dict[int, dict[str, float]]:
    """Mean |SHAP| per feature, keyed by HELD-OUT ROUND NUMBER."""
    return {
        explanation.held_out_round: explanation.mean_abs()
        for explanation in explanations
    }


def spreads(
    per_round: dict[int, dict[str, float]],
    per_round_compare: dict[int, dict[str, float]],
    features: list[str],
) -> dict[str, dict[str, float]]:
    """
    The three quantities the stability plot shows, per feature.

    same_day  rounds 1 vs 2 -- the same local day, ~10 h apart, so this
              is the variation that exists when nothing about the
              conditions changed
    cross_day rounds 3 vs 4 -- genuinely different days
    seed      the two seeds, same data

    Both pairs are looked up **by round number**. Taking them
    positionally would mean that reordering the fold list silently
    relabels a cross-day comparison as a same-day one, and the plot
    would still render.
    """
    result: dict[str, dict[str, float]] = {}

    for feature in features:
        result[feature] = {
            "same_day": _pair_difference(per_round, SAME_DAY_PAIR, feature),
            "cross_day": _pair_difference(per_round, CROSS_DAY_PAIR, feature),
            "seed": abs(
                per_round_mean(per_round, feature)
                - per_round_mean(per_round_compare, feature)
            ),
        }

    return result


def _pair_difference(
    per_round: dict[int, dict[str, float]], pair: tuple[int, int], feature: str
) -> float:
    first, second = pair

    if first not in per_round or second not in per_round:
        return 0.0

    return abs(per_round[first][feature] - per_round[second][feature])


def per_round_mean(per_round: dict[int, dict[str, float]], feature: str) -> float:
    """Mean of a feature's per-round importance, across the rounds present."""
    values = [round_values[feature] for round_values in per_round.values()]
    return float(np.mean(values)) if values else 0.0


def zero_importance(
    pooled: dict[str, float], constants: set[str]
) -> tuple[list[str], list[str]]:
    """
    Zero-importance features, split into two lists that must stay apart.

    A feature with zero attribution is *uninformative* if it never
    varies, and *unnecessary* if it varies but no tree ever split on it
    because a correlated neighbour already did the work. Those are
    different findings about the data, and listing them together would
    turn one of them into noise.
    """
    zero = sorted(name for name, value in pooled.items() if value == 0.0)

    return (
        [name for name in zero if name in constants],
        [name for name in zero if name not in constants],
    )


def constant_features(dataset) -> set[str]:
    """
    Features with a single value across the dataset.

    Detected rather than hard-coded: round 1 found three, and a future
    round may not.
    """
    return {
        name
        for index, name in enumerate(dataset.features)
        if len(np.unique(dataset.X[:, index])) == 1
    }


def ranked(pooled: dict[str, float]) -> dict[str, int]:
    order = sorted(pooled.items(), key=lambda item: (-item[1], item[0]))
    return {name: rank for rank, (name, _) in enumerate(order)}


# --------------------------------------------------------------
# Plots
# --------------------------------------------------------------

def wrap_caption(text: str, width: int = TITLE_WRAP) -> str:
    """
    Break a computed caption across lines so it cannot run off the
    figure.

    Each line of the input is wrapped separately, so a caption that
    already carries its own line breaks keeps them.

    This is the general fix rather than editing the sentence to fit:
    the captions are derived from the data, so their length changes
    with the data, and a version that happens to fit today can clip on
    the next rerun without anyone noticing. The failure is silent and
    it removes exactly the qualifying clause that tends to sit at the
    end of a sentence.
    """
    lines: list[str] = []

    for line in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(line, width=width) or [""])

    return "\n".join(lines)


def title_height_padding(*captions: str) -> float:
    """
    Extra figure height, in inches, for the lines wrapping added.

    Without it the axes shrink to make room and the bars are squeezed —
    the caption would stop clipping by taking space from the data.
    """
    extra = sum(max(0, len(caption.splitlines()) - 1) for caption in captions)
    return 0.22 * extra


def family_reading(
    families: dict[str, float], families_compare: dict[str, float]
) -> tuple[str, str]:
    """
    The headline plot's title — derived from the numbers, not written
    ahead of them.

    This function exists because the hardcoded version got it wrong.
    Written before the four-round data existed, plot 1's title said
    "timing and upstream size carry most of it" in a way that implied an
    order, and the measured values are:

        seed 42: timing 0.21299, sizes 0.20230
        seed  7: timing 0.19919, sizes 0.21208

    The two families **swap** between seeds. The bars underneath the
    title showed the swap while the title denied it, so a reader looking
    closely saw the figure contradict its own headline.

    What is stable is that the same two families carry ~82% of the total
    attribution under both seeds. What is not stable is which of them
    leads. The title may claim the first and must state the second, and
    deriving it here means a rerun on different data cannot leave a
    stale claim inside an image.

    Same reason as `stability_reading()`: a caption is not exempt from
    being measured.
    """
    top_primary, share_primary = _top_two(families)
    top_compare, share_compare = _top_two(families_compare)

    share = _format_share(share_primary, share_compare)

    if set(top_primary) != set(top_compare):
        return (
            "Attribution concentrates in a few families",
            f"The leading pair carries {share}, but which families lead "
            f"is seed-dependent",
        )

    names = " and ".join(top_primary)

    if top_primary[0] == top_compare[0]:
        return (
            "Two families carry most of the attribution",
            f"{names} together carry {share}, with {top_primary[0]} leading "
            f"under both seeds",
        )

    return (
        "Two families carry most of the attribution",
        f"{names} together carry {share} under both seeds — "
        f"but which of the two leads is seed-dependent",
    )


def _top_two(families: dict[str, float]) -> tuple[list[str], float]:
    """The two largest families, and their combined share of the total."""
    order = sorted(families.items(), key=lambda item: (-item[1], item[0]))
    total = sum(families.values())
    top = order[:2]

    share = sum(value for _, value in top) / total if total else 0.0
    return [name for name, _ in top], share


def _format_share(primary: float, comparison: float) -> str:
    low, high = sorted((round(primary * 100), round(comparison * 100)))
    return f"~{low}% of the total" if low == high else f"{low}–{high}% of the total"


def plot_family_importance(
    families: dict[str, float],
    families_compare: dict[str, float],
    seed: int,
    compare_seed: int,
    n_folds: int,
    path: Path,
) -> None:
    """
    Plot 1: the claim that survives.

    Both seeds are drawn as paired bars, so the stability is visible in
    the headline figure itself rather than only asserted in the text
    beside it — and the title is computed from those same bars by
    `family_reading()`, so it cannot disagree with them.
    """
    headline, detail = family_reading(families, families_compare)
    caption = wrap_caption(detail)

    order = sorted(families, key=lambda family: -families[family])
    positions = np.arange(len(order))
    height = 0.38

    # Grow the figure for the wrapped lines rather than letting the
    # title steal room from the bars.
    figure, axes = plt.subplots(
        figsize=(9, 5 + title_height_padding(caption))
    )

    axes.barh(positions + height / 2, [families[f] for f in order],
              height=height, label=f"seed {seed}", color="#3b6ea5")
    axes.barh(positions - height / 2, [families_compare[f] for f in order],
              height=height, label=f"seed {compare_seed}", color="#a5c4e0")

    axes.set_yticks(positions)
    axes.set_yticklabels(order)
    axes.invert_yaxis()
    axes.set_xlabel("pooled mean |SHAP| summed over the family")
    axes.set_title(f"{wrap_caption(headline)}\n{caption}")
    axes.legend(loc="lower right", frameon=False)
    axes.spines[["top", "right"]].set_visible(False)

    figure.text(
        0.01, 0.01,
        f"{SPLIT_DESCRIPTION}; {n_folds} folds. "
        f"Positive SHAP points toward the automated client.",
        fontsize=7, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=FIGURE_DPI, metadata=PNG_METADATA)
    plt.close(figure)


def plot_beeswarm(explanations, seed: int, path: Path) -> None:
    """
    Plot 2: informative, and seed-dependent.

    The caveat is in the TITLE. A figure gets screenshotted into a slide
    without the paragraph that qualifies it, and a plot that is only
    honest in context is not honest.
    """
    values = np.concatenate([e.shap_values for e in explanations], axis=0)
    data = np.concatenate([e.X_test for e in explanations], axis=0)

    explanation = shap.Explanation(
        values=values,
        data=data,
        feature_names=list(explanations[0].features),
        base_values=np.full(len(values), explanations[0].base_value),
    )

    # shap's beeswarm jitters overlapping points using the global numpy
    # RNG, so two runs on identical inputs produce different PNG bytes.
    # Seeding here makes the figure reproducible, which is the point of
    # publishing it: a plot that cannot be regenerated is not evidence.
    np.random.seed(seed)

    caveat = wrap_caption(
        f"Per-feature attribution (seed {seed}) — THIS ORDERING IS "
        f"SEED-DEPENDENT\n"
        f"Individual features move up to 24 places between seeds; "
        f"see the stability plot",
        width=CAVEAT_WRAP,
    )

    plt.figure(figsize=(9, 6 + title_height_padding(caveat)))
    shap.plots.beeswarm(explanation, max_display=BEESWARM_FEATURES, show=False)
    plt.title(caveat, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, metadata=PNG_METADATA)
    plt.close()


def stability_reading(maxima: dict[str, float]) -> tuple[str, str]:
    """
    The conclusion the stability numbers support — read off them, not
    asserted.

    On the real four-round dataset the seed spread dominates both round
    spreads, and that is the finding. But writing that sentence into the
    title unconditionally would mean the plot states a conclusion the
    data might not support: on some other dataset the capture day could
    turn out to matter, and the figure would still claim it did not.
    Every other number in this project is measured rather than declared;
    a caption is not exempt.
    """
    rounds = max(maxima["same_day"], maxima["cross_day"])

    if maxima["seed"] > rounds:
        return (
            "The instability is in the explanation, not in the capture",
            "Seed variation dominates both the same-day and the cross-day spread",
        )

    if maxima["cross_day"] > maxima["same_day"]:
        return (
            "The capture day moves attribution more than the seed does",
            "Cross-day spread exceeds both the same-day and the seed spread "
            "— the rig, not just the method",
        )

    return (
        "Round-to-round variation exceeds the seed spread",
        "Same-day variation is the largest term; the rounds are not "
        "interchangeable",
    )


def plot_stability(
    feature_spreads: dict[str, dict[str, float]],
    pooled: dict[str, float],
    maxima: dict[str, float],
    path: Path,
) -> None:
    """
    Plot 3: the measurement that bounds plot 2.

    Three bars per feature, and the reading is in the title rather than
    left to the viewer -- but the reading is computed from the numbers
    by `stability_reading()`, so the title cannot outlive the data that
    justified it.
    """
    top = [
        name
        for name, _ in sorted(pooled.items(), key=lambda item: (-item[1], item[0]))
    ][:TOP_FEATURES]

    headline, detail = stability_reading(maxima)
    caption = wrap_caption(detail)

    positions = np.arange(len(top))
    height = 0.26

    figure, axes = plt.subplots(
        figsize=(9, 7 + title_height_padding(caption))
    )

    axes.barh(positions + height, [feature_spreads[f]["same_day"] for f in top],
              height=height, label="same day (rounds 1 vs 2)", color="#cfd8e3")
    axes.barh(positions, [feature_spreads[f]["cross_day"] for f in top],
              height=height, label="different days (rounds 3 vs 4)",
              color="#7fa6cc")
    axes.barh(positions - height, [feature_spreads[f]["seed"] for f in top],
              height=height, label="different seed (42 vs 7)", color="#b5482f")

    axes.set_yticks(positions)
    axes.set_yticklabels(top)
    axes.invert_yaxis()
    axes.set_xlabel("difference in mean |SHAP|")
    axes.set_title(f"{wrap_caption(headline)}\n{caption}")
    axes.legend(loc="lower right", frameon=False)
    axes.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, metadata=PNG_METADATA)
    plt.close(figure)


# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------

def build_summary(
    dataset,
    explanations,
    explanations_compare,
    args,
    generated_at: str,
) -> dict:
    """
    The machine-readable companion to the plots.

    Published so the numbers can be checked without opening an image --
    and so a reader can see which direction a positive SHAP value points
    without having to infer it from a colour bar.
    """
    pooled = pooled_importance(explanations)
    pooled_compare = pooled_importance(explanations_compare)

    per_round = per_round_importance(explanations)
    per_round_compare = per_round_importance(explanations_compare)
    feature_spreads = spreads(per_round, per_round_compare, dataset.features)

    ranks = ranked(pooled)
    ranks_compare = ranked(pooled_compare)

    constants = constant_features(dataset)
    zero_constant, zero_varying = zero_importance(pooled, constants)

    first = explanations[0]

    return {
        "generated_at": generated_at,
        "model": args.model,
        "seeds": {"primary": args.seed, "comparison": args.compare_seed},
        "split": SPLIT_DESCRIPTION,
        "rounds": dataset.rounds,
        "n_folds": len(explanations),
        "n_traces": len(dataset),
        "n_features": len(dataset.features),
        "direction": {
            "positive_class": first.positive_class,
            "classes_": first.classes,
            "note": (
                f"A positive SHAP value pushes the prediction toward "
                f"{first.positive_class!r}. The class order is read from the "
                f"fitted classifier, never assumed."
            ),
            "base_value": first.base_value,
        },
        "family_importance": {
            f"seed_{args.seed}": family_importance(pooled),
            f"seed_{args.compare_seed}": family_importance(pooled_compare),
        },
        "feature_importance": {
            name: {
                f"seed_{args.seed}": pooled[name],
                f"seed_{args.compare_seed}": pooled_compare[name],
                f"rank_seed_{args.seed}": ranks[name],
                f"rank_seed_{args.compare_seed}": ranks_compare[name],
                "rank_movement": abs(ranks[name] - ranks_compare[name]),
            }
            for name in dataset.features
        },
        "spreads": {
            "note": (
                "same_day compares rounds 1 and 2 (the same local day, ~10 h "
                "apart); cross_day compares rounds 3 and 4 (different days); "
                "seed compares the two random_state values on the same data."
            ),
            "same_day_rounds": list(SAME_DAY_PAIR),
            "cross_day_rounds": list(CROSS_DAY_PAIR),
            "per_feature": feature_spreads,
            "max": {
                which: max(
                    (values[which] for values in feature_spreads.values()),
                    default=0.0,
                )
                for which in ("same_day", "cross_day", "seed")
            },
        },
        "zero_importance": {
            "note": (
                "Two different findings, deliberately kept apart. A constant "
                "feature is UNINFORMATIVE -- it never varies. A non-constant "
                "feature with zero attribution is UNNECESSARY -- it varies, "
                "but a correlated neighbour was always split on instead."
            ),
            "constant": zero_constant,
            "varying_but_unused": zero_varying,
        },
    }


# --------------------------------------------------------------
# Run
# --------------------------------------------------------------

def output_paths(results_root: Path) -> dict[str, Path]:
    return {
        "family": results_root / FAMILY_PLOT,
        "beeswarm": results_root / BEESWARM_PLOT,
        "stability": results_root / STABILITY_PLOT,
        "summary": results_root / SUMMARY_JSON,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = output_paths(args.results_root)

    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        print(
            "ABORT: output already exists. Use --force to replace it:\n  "
            + "\n  ".join(str(path) for path in existing),
            file=sys.stderr,
        )
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
          f"{dataset.rounds}, {len(dataset.features)} features", file=sys.stderr)

    # Same n_estimators as scripts/train_model.py, so the explanation
    # describes the model whose accuracy was published.
    print(f"  explaining at seed {args.seed} ...", file=sys.stderr, flush=True)
    explanations = explain_by_round(
        dataset, model=args.model, random_state=args.seed
    )

    print(f"  explaining at seed {args.compare_seed} ...", file=sys.stderr,
          flush=True)
    explanations_compare = explain_by_round(
        dataset, model=args.model, random_state=args.compare_seed
    )

    pooled = pooled_importance(explanations)
    pooled_compare = pooled_importance(explanations_compare)

    args.results_root.mkdir(parents=True, exist_ok=True)

    plot_family_importance(
        family_importance(pooled),
        family_importance(pooled_compare),
        seed=args.seed,
        compare_seed=args.compare_seed,
        n_folds=len(explanations),
        path=paths["family"],
    )
    plot_beeswarm(explanations, seed=args.seed, path=paths["beeswarm"])

    feature_spreads = spreads(
        per_round_importance(explanations),
        per_round_importance(explanations_compare),
        dataset.features,
    )
    maxima = {
        which: max(
            (values[which] for values in feature_spreads.values()), default=0.0
        )
        for which in ("same_day", "cross_day", "seed")
    }
    plot_stability(feature_spreads, pooled, maxima, path=paths["stability"])

    summary = build_summary(
        dataset, explanations, explanations_compare, args,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_summary(paths, summary)
    return EXIT_OK


def print_summary(paths: dict[str, Path], summary: dict) -> None:
    print("SHAP artefacts written")
    print(f"  traces        : {summary['n_traces']} "
          f"over {summary['n_folds']} folds")
    print(f"  positive class: {summary['direction']['positive_class']} "
          f"(classes_ {summary['direction']['classes_']})")

    families = summary["family_importance"]
    primary = families[f"seed_{summary['seeds']['primary']}"]
    for family, value in sorted(primary.items(), key=lambda item: -item[1]):
        print(f"    {family:<12} {value:.5f}")

    maxima = summary["spreads"]["max"]
    print(f"  spread  same-day  {maxima['same_day']:.5f}")
    print(f"          cross-day {maxima['cross_day']:.5f}")
    print(f"          seed      {maxima['seed']:.5f}")

    headline, detail = stability_reading(maxima)
    print(f"  reading : {headline}")
    print(f"            {detail}")

    for key in ("family", "beeswarm", "stability", "summary"):
        print(f"  {key:<14}: {paths[key]}")


if __name__ == "__main__":
    sys.exit(main())
