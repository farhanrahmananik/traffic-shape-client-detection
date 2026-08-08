"""
test_explain_model.py
---------------------
Tests for scripts/explain_model.py.

Synthetic and fast: `data/features/features.csv` is never read, so the
suite still means something on a clone with no captures, and nothing
here reports numbers from the real dataset.

What is worth testing in a rendering script is not the pixels. It is the
arithmetic that decides what the pixels claim: that the family bars
account for every feature exactly once, that the "same day" bar really
compares the same-day rounds, and that the two kinds of zero-importance
feature stay apart. A plot with a wrong number in it looks exactly like
a plot with a right one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import explain_model as script  # noqa: E402
from tsd.model import FEATURE_GROUPS, load_dataset  # noqa: E402

# One feature per family, plus a constant and a redundant twin, so the
# aggregation and the zero-importance split both have something to bite
# on. Names must match model.FEATURE_GROUPS prefixes.
FEATURES = [
    "count_total",       # counts
    "size_up_mean",      # sizes    -- informative
    "size_up_min",       # sizes    -- constant
    "size_down_p90",     # sizes    -- redundant twin of size_up_mean
    "iat_down_max",      # timing
    "burst_count",       # bursts
    "syn_count",         # connections -- informative
]

ROUNDS = (1, 2, 3, 4)
PAGES_PER_ROUND = 6


def synthetic_frame(seed: int = 3) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows = []

    for round_number in ROUNDS:
        for page in range(PAGES_PER_ROUND):
            for client in ("firefox", "wget"):
                browser = client == "firefox"
                informative = 6.0 if browser else 1.0

                rows.append({
                    "round": round_number,
                    "date": f"2026080{round_number + 5}",
                    "client": client,
                    "page": f"page_{page:02d}",
                    "count_total": generator.normal(150, 10),
                    "size_up_mean": informative * 100 + generator.normal(0, 3),
                    "size_up_min": 0.0,  # constant, by construction
                    # varies, but carries the same information as
                    # size_up_mean -- the "unnecessary" case
                    "size_down_p90": informative * 100 + generator.normal(0, 3),
                    "iat_down_max": generator.normal(0.002, 0.0003),
                    "burst_count": generator.normal(60, 6),
                    "syn_count": informative,
                })

    return pd.DataFrame(rows)


@pytest.fixture
def features_csv(tmp_path) -> Path:
    path = tmp_path / "features.csv"
    synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def dataset(features_csv):
    return load_dataset(features_csv)


def run(features_csv: Path, results_root: Path, *extra) -> int:
    return script.main([
        "--features", str(features_csv),
        "--results-root", str(results_root),
        *extra,
    ])


# --------------------------------------------------------------
# Family aggregation
# --------------------------------------------------------------

def test_family_totals_account_for_every_feature_exactly_once():
    """
    Nothing dropped, nothing double-counted. A family chart that quietly
    omitted a feature would still be a perfectly readable chart.
    """
    pooled = {
        "count_total": 0.10,
        "size_up_mean": 0.20,
        "size_up_min": 0.0,
        "iat_down_max": 0.30,
        "burst_count": 0.05,
        "syn_count": 0.07,
    }

    families = script.family_importance(pooled)

    assert sum(families.values()) == pytest.approx(sum(pooled.values()))
    assert set(families) == set(FEATURE_GROUPS)
    assert families["sizes"] == pytest.approx(0.20)
    assert families["timing"] == pytest.approx(0.30)
    assert families["connections"] == pytest.approx(0.07)


def test_family_totals_match_pooled_total_on_real_feature_names(dataset):
    """The same invariant, but over features produced by the pipeline."""
    from tsd.shap_explain import explain_by_round

    explanations = explain_by_round(dataset, n_estimators=8, random_state=42)
    pooled = script.pooled_importance(explanations)

    assert sum(script.family_importance(pooled).values()) == pytest.approx(
        sum(pooled.values())
    )


def test_a_feature_outside_every_family_is_an_error():
    with pytest.raises(KeyError):
        script.family_importance({"not_a_known_prefix_thing": 1.0})


# --------------------------------------------------------------
# The three spreads
# --------------------------------------------------------------

def per_round(values: dict[int, dict[str, float]]):
    return values


def test_spreads_use_the_rounds_they_claim():
    """
    same_day must compare rounds 1 and 2; cross_day rounds 3 and 4. If
    the pairs were taken positionally, reordering the fold list would
    relabel a cross-day comparison as a same-day one and the plot would
    still render, with the legend lying.
    """
    primary = {
        1: {"f": 0.10},
        2: {"f": 0.13},   # same-day difference 0.03
        3: {"f": 0.50},
        4: {"f": 0.58},   # cross-day difference 0.08
    }
    comparison = {round_number: {"f": 1.00} for round_number in ROUNDS}

    result = script.spreads(primary, comparison, ["f"])["f"]

    assert result["same_day"] == pytest.approx(0.03)
    assert result["cross_day"] == pytest.approx(0.08)
    # seed: mean over rounds, 0.3275 against 1.00
    assert result["seed"] == pytest.approx(abs(np.mean([0.10, 0.13, 0.50, 0.58]) - 1.0))


def test_spreads_are_unchanged_when_the_fold_order_changes():
    """
    The regression guard for the same point: the pairs are keyed by
    round number, so shuffling the dict cannot move them.
    """
    ordered = {1: {"f": 0.10}, 2: {"f": 0.13}, 3: {"f": 0.50}, 4: {"f": 0.58}}
    shuffled = {3: {"f": 0.50}, 1: {"f": 0.10}, 4: {"f": 0.58}, 2: {"f": 0.13}}
    comparison = {n: {"f": 0.2} for n in ROUNDS}

    assert script.spreads(ordered, comparison, ["f"]) == \
        script.spreads(shuffled, comparison, ["f"])


def test_missing_round_pair_does_not_crash():
    """A three-round dataset must still render, with the pair reported as 0."""
    primary = {1: {"f": 0.1}, 2: {"f": 0.2}, 3: {"f": 0.3}}
    comparison = {1: {"f": 0.1}, 2: {"f": 0.2}, 3: {"f": 0.3}}

    result = script.spreads(primary, comparison, ["f"])["f"]

    assert result["same_day"] == pytest.approx(0.1)
    assert result["cross_day"] == 0.0


def test_per_round_importance_is_keyed_by_held_out_round(dataset):
    from tsd.shap_explain import explain_by_round

    explanations = explain_by_round(dataset, n_estimators=8, random_state=42)
    per_round_values = script.per_round_importance(explanations)

    assert sorted(per_round_values) == list(ROUNDS)


# --------------------------------------------------------------
# Zero importance: two findings, kept apart
# --------------------------------------------------------------

def test_constant_and_varying_zero_importance_features_are_separated():
    """
    "Uninformative" and "unnecessary" are different findings. Listing
    them together turns the interesting one into noise.
    """
    pooled = {"a": 0.0, "b": 0.0, "c": 0.5}
    constants = {"a"}

    constant, varying = script.zero_importance(pooled, constants)

    assert constant == ["a"]
    assert varying == ["b"]


def test_constant_features_are_detected_from_the_data(dataset):
    """Detected, not hard-coded: a future round may make one vary."""
    constants = script.constant_features(dataset)

    assert "size_up_min" in constants
    assert "size_up_mean" not in constants


# --------------------------------------------------------------
# End to end
# --------------------------------------------------------------

def test_writes_three_plots_and_a_summary(features_csv, tmp_path):
    results = tmp_path / "results"

    assert run(features_csv, results) == script.EXIT_OK

    for name in (script.FAMILY_PLOT, script.BEESWARM_PLOT,
                 script.STABILITY_PLOT, script.SUMMARY_JSON):
        assert (results / name).is_file(), name
        assert (results / name).stat().st_size > 0


def test_summary_records_the_direction_and_the_split(features_csv, tmp_path):
    """
    A reader must be able to tell which way a positive SHAP value points
    without inferring it from a colour bar.
    """
    results = tmp_path / "results"
    run(features_csv, results)

    summary = json.loads((results / script.SUMMARY_JSON).read_text())

    assert summary["direction"]["positive_class"] in ("firefox", "wget")
    assert summary["direction"]["classes_"] == ["firefox", "wget"]
    assert "held-out" in summary["split"]
    assert summary["rounds"] == list(ROUNDS)
    assert summary["n_folds"] == len(ROUNDS)
    assert summary["spreads"]["same_day_rounds"] == [1, 2]
    assert summary["spreads"]["cross_day_rounds"] == [3, 4]


def test_summary_separates_the_two_zero_importance_findings(features_csv, tmp_path):
    results = tmp_path / "results"
    run(features_csv, results)

    zero = json.loads((results / script.SUMMARY_JSON).read_text())["zero_importance"]

    assert "size_up_min" in zero["constant"], "a constant feature is uninformative"
    assert "size_up_min" not in zero["varying_but_unused"]
    assert set(zero["constant"]) & set(zero["varying_but_unused"]) == set()


def test_force_is_required_to_overwrite(features_csv, tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    (results / script.SUMMARY_JSON).write_text("do not lose me", encoding="utf-8")

    assert run(features_csv, results) == script.EXIT_REFUSED
    assert (results / script.SUMMARY_JSON).read_text() == "do not lose me"
    assert "already exists" in capsys.readouterr().err

    assert run(features_csv, results, "--force") == script.EXIT_OK
    assert (results / script.SUMMARY_JSON).read_text() != "do not lose me"


def test_family_title_says_the_order_is_seed_dependent_when_it_swaps():
    """
    The measured four-round values: timing 0.21299 / sizes 0.20230 at
    seed 42, timing 0.19919 / sizes 0.21208 at seed 7. The two families
    swap. A hardcoded title asserting an order would sit directly above
    bars that show the opposite — which is what it did before this
    function existed.
    """
    headline, detail = script.family_reading(
        {"timing": 0.21299, "sizes": 0.20230, "bursts": 0.05,
         "counts": 0.02, "connections": 0.02},
        {"timing": 0.19919, "sizes": 0.21208, "bursts": 0.05,
         "counts": 0.02, "connections": 0.02},
    )

    assert "seed-dependent" in detail
    assert "timing" in detail and "sizes" in detail
    assert "leading under both seeds" not in detail
    assert headline == "Two families carry most of the attribution"


def test_family_title_names_the_leader_when_it_does_not_swap():
    _, detail = script.family_reading(
        {"timing": 0.30, "sizes": 0.10, "bursts": 0.02,
         "counts": 0.01, "connections": 0.01},
        {"timing": 0.28, "sizes": 0.11, "bursts": 0.02,
         "counts": 0.01, "connections": 0.01},
    )

    assert "timing leading under both seeds" in detail
    assert "which of the two leads is seed-dependent" not in detail


def test_family_title_reports_the_measured_combined_share():
    """The percentage in the title must be the summed importances."""
    families = {"timing": 0.4, "sizes": 0.4, "bursts": 0.1,
                "counts": 0.05, "connections": 0.05}

    _, detail = script.family_reading(families, families)

    # (0.4 + 0.4) / 1.0 = 80%
    assert "~80% of the total" in detail

    top, share = script._top_two(families)
    assert sorted(top) == ["sizes", "timing"]
    assert share == pytest.approx(0.8)


def test_family_title_reports_a_range_when_the_shares_differ():
    _, detail = script.family_reading(
        {"timing": 0.40, "sizes": 0.40, "bursts": 0.20},
        {"timing": 0.35, "sizes": 0.35, "bursts": 0.30},
    )

    assert "80" in detail and "70" in detail


def test_family_title_flags_a_changed_leading_pair():
    headline, detail = script.family_reading(
        {"timing": 0.40, "sizes": 0.30, "bursts": 0.10},
        {"timing": 0.40, "bursts": 0.35, "sizes": 0.05},
    )

    assert "which families lead is seed-dependent" in detail
    assert headline == "Attribution concentrates in a few families"


# --------------------------------------------------------------
# Captions must fit inside the figure
# --------------------------------------------------------------

def test_a_long_caption_is_wrapped_rather_than_emitted_as_one_line():
    """
    The rendered plot 1 clipped its subtitle at "...is seed-", losing
    the word "dependent" — the exact clause the caption had just been
    rewritten to carry. Shortening the wording would have fixed that
    string and left the next one to clip silently, so the captions are
    wrapped instead.
    """
    long_detail = (
        "timing and sizes together carry 81–82% of the total under both "
        "seeds — but which of the two leads is seed-dependent"
    )

    wrapped = script.wrap_caption(long_detail)

    assert "\n" in wrapped, "a long caption must be broken across lines"
    assert all(len(line) <= script.TITLE_WRAP for line in wrapped.splitlines())
    assert wrapped.replace("\n", " ") == long_detail, "no words lost or added"
    assert "seed-dependent" in wrapped


def test_wrapping_preserves_existing_line_breaks():
    """Plot 2's caveat already arrives on two lines; they must survive."""
    wrapped = script.wrap_caption("first line\nsecond line", width=40)

    assert wrapped == "first line\nsecond line"


def test_short_captions_are_left_alone():
    assert script.wrap_caption("short enough") == "short enough"


def test_wrapped_captions_get_extra_figure_height():
    """
    Otherwise the title stops clipping by taking room from the bars,
    which fixes the caption at the data's expense.
    """
    one_line = script.wrap_caption("short")
    three_lines = "a\nb\nc"

    assert script.title_height_padding(one_line) == 0
    assert script.title_height_padding(three_lines) > 0
    assert script.title_height_padding(three_lines) > \
        script.title_height_padding("a\nb")


@pytest.mark.parametrize(
    "headline, detail",
    [
        script.family_reading(
            {"timing": 0.21299, "sizes": 0.20230, "bursts": 0.05},
            {"timing": 0.19919, "sizes": 0.21208, "bursts": 0.05},
        ),
        script.family_reading(
            {"timing": 0.4, "sizes": 0.3, "bursts": 0.2},
            {"timing": 0.4, "bursts": 0.35, "sizes": 0.05},
        ),
        script.stability_reading(
            {"same_day": 0.003, "cross_day": 0.003, "seed": 0.021}
        ),
        script.stability_reading(
            {"same_day": 0.003, "cross_day": 0.030, "seed": 0.005}
        ),
        script.stability_reading(
            {"same_day": 0.030, "cross_day": 0.003, "seed": 0.005}
        ),
    ],
)
def test_every_caption_branch_fits_inside_the_figure(headline, detail):
    """
    Measured, not assumed: each caption the two readers can produce is
    rendered and its box compared with the figure's. The branch that
    fires depends on the data, so every branch has to fit.
    """
    import matplotlib.pyplot as plt

    caption = script.wrap_caption(detail)
    figure, axes = plt.subplots(
        figsize=(9, 5 + script.title_height_padding(caption))
    )
    axes.barh(range(3), [1, 2, 3])
    axes.set_title(f"{script.wrap_caption(headline)}\n{caption}")
    figure.tight_layout()
    figure.canvas.draw()

    title_box = axes.title.get_window_extent(figure.canvas.get_renderer())
    figure_box = figure.get_window_extent()
    plt.close(figure)

    assert title_box.x0 >= figure_box.x0
    assert title_box.x1 <= figure_box.x1


def test_the_stability_conclusion_is_read_off_the_numbers():
    """
    On the real four-round dataset the seed spread dominates, and that
    is the finding. Writing that sentence into the title unconditionally
    would mean the figure states a conclusion the data might not
    support — every other number in this project is measured rather than
    declared, and a caption is not exempt.
    """
    seed_dominates = script.stability_reading(
        {"same_day": 0.003, "cross_day": 0.003, "seed": 0.021}
    )
    assert "not in the capture" in seed_dominates[0]

    day_dominates = script.stability_reading(
        {"same_day": 0.003, "cross_day": 0.030, "seed": 0.005}
    )
    assert "capture day" in day_dominates[0]
    assert day_dominates != seed_dominates


def test_plots_are_byte_identical_across_runs(features_csv, tmp_path):
    """
    A published plot that cannot be regenerated is not evidence. shap's
    beeswarm jitters points with the global numpy RNG, so this fails
    unless that is seeded.
    """
    import hashlib

    digests = []
    for run_name in ("first", "second"):
        results = tmp_path / run_name
        assert run(features_csv, results) == script.EXIT_OK
        digests.append({
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(results.glob("*.png"))
        })

    assert digests[0] == digests[1]


def test_single_round_dataset_is_refused(tmp_path):
    frame = synthetic_frame()
    frame = frame[frame["round"] == 1]
    path = tmp_path / "one_round.csv"
    frame.to_csv(path, index=False)

    assert run(path, tmp_path / "results") == script.EXIT_CANNOT_SPLIT
