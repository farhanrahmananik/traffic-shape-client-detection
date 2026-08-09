"""
test_site_bindings.py
---------------------
Every `data-bind` path in docs/index.html must resolve in the site data.

A broken binding is invisible in the source. The page still renders, the
sentence around it still reads, and the only symptom is an ellipsis where
a number should be — which looks exactly like a page that has not
finished loading. Someone would have to open the console to find out
otherwise, and by then it is a reader, not the author.

So it is a failing test instead. The page is checked against two things,
because they can go wrong separately:

    docs/data/case_study.json   what the published page actually fetches
    build(results/)             what the build step would produce now

Parsing uses `html.parser` from the standard library. A real HTML parser
would be a dependency added to catch attribute typos, and comments are
skipped for free — the file's own header comment contains an example
`data-bind` attribute that a regex would happily have collected.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tsd.site_data import build

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = REPO_ROOT / "docs" / "index.html"
SITE_DATA = REPO_ROOT / "docs" / "data" / "case_study.json"
REAL_RESULTS = REPO_ROOT / "results"

PLACEHOLDER = "…"  # the single ellipsis character, not three dots

KNOWN_FORMATS = {"fixed", "percent", "int", "short-hash", "list", "count"}


class BindingCollector(HTMLParser):
    """Collects (path, format, placeholder text) for every bound element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.bindings: list[dict] = []
        self.ids: set[str] = set()
        self.copy_targets: list[str] = []
        self.samples: list[str] = []
        self._open: dict | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)

        if "id" in attributes:
            self.ids.add(attributes["id"])
        if "data-copy-target" in attributes:
            self.copy_targets.append(attributes["data-copy-target"])
        if "data-sample" in attributes:
            self.samples.append(attributes["data-sample"])

        if "data-bind" in attributes:
            self._open = {
                "tag": tag,
                "path": attributes["data-bind"],
                "format": attributes.get("data-format"),
                "class": attributes.get("class"),
                "text": "",
            }
            self.bindings.append(self._open)

    def handle_data(self, data):
        if self._open is not None:
            self._open["text"] += data

    def handle_endtag(self, tag):
        if self._open is not None and tag == self._open["tag"]:
            self._open = None


def collect() -> BindingCollector:
    collector = BindingCollector()
    collector.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return collector


def collect_bindings() -> list[dict]:
    return collect().bindings


def resolve(root, dotted: str):
    """Mirrors the resolver in docs/app.js, including its null handling."""
    node = root

    for segment in dotted.split("."):
        if isinstance(node, list):
            try:
                node = node[int(segment)]
            except (ValueError, IndexError):
                return None, f"no element {segment!r}"
        elif isinstance(node, dict):
            if segment not in node:
                return None, f"missing key {segment!r}"
            node = node[segment]
        else:
            return None, f"cannot descend into {type(node).__name__}"

    if node is None:
        return None, "resolves to null"

    return node, None


@pytest.fixture(scope="module")
def bindings() -> list[dict]:
    found = collect_bindings()
    assert found, "docs/index.html contains no data-bind attributes"
    return found


# --------------------------------------------------------------
# The paths resolve
# --------------------------------------------------------------

def test_every_binding_resolves_in_the_published_site_data(bindings):
    """Against the file the page actually fetches."""
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))

    broken = []
    for binding in bindings:
        _, error = resolve(data, binding["path"])
        if error:
            broken.append(f"{binding['path']}: {error}")

    assert not broken, "unresolved data-bind paths:\n  " + "\n  ".join(broken)


def test_every_binding_resolves_in_a_fresh_build(bindings):
    """
    Against what the build step would produce right now. This is the one
    that fails when a source artefact changes shape — a key renamed in
    results/ breaks the page before anyone looks at it.
    """
    if not (REAL_RESULTS / "metrics.json").is_file():
        pytest.skip("results/ is not populated in this clone")

    document = build(REAL_RESULTS)

    broken = []
    for binding in bindings:
        _, error = resolve(document, binding["path"])
        if error:
            broken.append(f"{binding['path']}: {error}")

    assert not broken, "unresolved data-bind paths:\n  " + "\n  ".join(broken)


def test_no_binding_resolves_to_null(bindings):
    """
    A null would render as the text "null" if the binder did not guard
    against it, which reads as a measurement rather than as an absence.
    """
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))

    for binding in bindings:
        value, error = resolve(data, binding["path"])
        assert error is None, f"{binding['path']}: {error}"
        assert value is not None, binding["path"]


# --------------------------------------------------------------
# The markup's own conventions
# --------------------------------------------------------------

def test_every_binding_carries_the_pending_placeholder(bindings):
    """
    An unfilled page must read as pending, not as broken. A binding with
    an empty body would collapse the sentence around it.
    """
    missing = [
        binding["path"]
        for binding in bindings
        if binding["text"].strip() != PLACEHOLDER
    ]

    assert not missing, f"bindings without the '{PLACEHOLDER}' placeholder: {missing}"


def test_every_binding_has_the_value_class(bindings):
    without = [
        binding["path"]
        for binding in bindings
        if "value" not in (binding["class"] or "").split()
    ]

    assert not without, f"bindings without class=\"value\": {without}"


def test_data_format_values_are_known_to_the_binder(bindings):
    """
    A typo in a format name costs readability rather than correctness --
    app.js falls through to the raw value -- but it is still a mistake,
    and it is cheap to catch here.
    """
    unknown = []

    for binding in bindings:
        spec = binding["format"]
        if spec is None:
            continue

        kind, _, argument = spec.partition(":")
        if kind not in KNOWN_FORMATS:
            unknown.append(f"{binding['path']}: {spec!r}")
        elif kind in {"fixed", "percent"} and not argument.isdigit():
            unknown.append(f"{binding['path']}: {spec!r} needs a digit count")

    assert not unknown, "unknown data-format specs:\n  " + "\n  ".join(unknown)


def test_numeric_formats_are_applied_to_numbers(bindings):
    """
    `int` on a date string would render 20260807 as "20,260,807". The
    formats have to match the type of the value they will receive.
    """
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))
    mismatched = []

    for binding in bindings:
        spec = binding["format"]
        if spec is None:
            continue

        value, error = resolve(data, binding["path"])
        assert error is None, binding["path"]
        kind = spec.split(":")[0]

        if kind in {"fixed", "percent", "int"} and not isinstance(
            value, (int, float)
        ):
            mismatched.append(f"{binding['path']}: {kind} on {type(value).__name__}")
        if kind == "short-hash" and not isinstance(value, str):
            mismatched.append(f"{binding['path']}: short-hash on non-string")
        if kind in {"list", "count"} and not isinstance(value, (list, dict)):
            mismatched.append(f"{binding['path']}: {kind} on {type(value).__name__}")

    assert not mismatched, "format/type mismatches:\n  " + "\n  ".join(mismatched)


def test_int_format_would_not_distort_a_fractional_value(bindings):
    """
    `int` rounds. Applying it to 1.5 would publish 2. Nothing bound with
    `int` may be fractional.
    """
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))

    for binding in bindings:
        if binding["format"] != "int":
            continue
        value, _ = resolve(data, binding["path"])
        assert float(value).is_integer(), (
            f"{binding['path']} is {value}, which `int` would round"
        )


# --------------------------------------------------------------
# The page and the script agree
# --------------------------------------------------------------

def test_every_copy_button_points_at_an_element_that_exists():
    """
    A copy button whose target id is absent copies nothing and says
    nothing, because app.js returns quietly when the element is missing —
    the failure is a button that appears to work.
    """
    page = collect()

    assert page.copy_targets, "no copy buttons found in docs/index.html"

    missing = [
        target for target in page.copy_targets if target not in page.ids
    ]
    assert not missing, f"data-copy-target with no such id: {missing}"


def test_the_sample_output_is_a_real_verdict_and_not_an_illustration():
    """
    The sample block claims to be the tool's own output. That claim is
    only worth making if the file it fetches is a verdict this repository
    could have produced, so its shape is checked rather than trusted.
    """
    page = collect()

    assert page.samples, "no data-sample element found in docs/index.html"

    for reference in page.samples:
        path = INDEX_HTML.parent / reference
        assert path.is_file(), f"{reference} is referenced but not present"

        verdict = json.loads(path.read_text(encoding="utf-8"))

        assert set(verdict) == {"schema_version", "pcap", "verdict", "model"}
        assert verdict["verdict"]["client"] in verdict["verdict"]["probabilities"]
        assert verdict["pcap"]["packets"] > 0


def test_the_sample_verdict_names_the_published_model():
    """
    The page states that the verdict came from the model whose digest it
    prints from the site data. If the sample were produced by a different
    artefact, that sentence would be false and nothing else would notice.
    """
    page = collect()
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))

    for reference in page.samples:
        verdict = json.loads(
            (INDEX_HTML.parent / reference).read_text(encoding="utf-8")
        )
        assert verdict["model"]["sha256"] == data["cli"]["model"]["sha256"]


def test_the_chart_can_be_built_from_the_site_data():
    """
    The chart is drawn from `shap.seeds` and `shap.family_importance`,
    and it derives the key of each series from the seed number rather
    than naming it. That indirection is the point — it is also the thing
    that breaks silently, because a chart that cannot be built hides its
    own figure and leaves a page that looks complete.

    So the same walk the renderer does is done here, against both the
    published data and a fresh build.
    """
    documents = [json.loads(SITE_DATA.read_text(encoding="utf-8"))]
    if (REAL_RESULTS / "metrics.json").is_file():
        documents.append(build(REAL_RESULTS))

    page = collect()
    charts = re.findall(r'data-chart="([^"]+)"', INDEX_HTML.read_text("utf-8"))
    assert charts, "no data-chart element found in docs/index.html"
    assert page.ids  # the collector ran

    for document in documents:
        for path in charts:
            node, error = resolve(document, path)
            assert error is None, f"{path}: {error}"

            seeds = [node["seeds"]["primary"], node["seeds"]["comparison"]]
            assert len(set(seeds)) == 2, "the two seeds must differ"

            series = [node["family_importance"][f"seed_{seed}"] for seed in seeds]
            assert series[0].keys() == series[1].keys(), (
                "the two seeds report different families"
            )
            assert series[0], "no families to plot"

            for values in series:
                for family, value in values.items():
                    assert isinstance(value, (int, float)), f"{family} is not a number"

            assert max(max(values.values()) for values in series) > 0


def test_the_page_loads_the_binder_and_the_binder_reads_the_site_data():
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = (REPO_ROOT / "docs" / "app.js").read_text(encoding="utf-8")

    assert 'src="app.js"' in html
    assert "defer" in html
    assert "data/case_study.json" in script
    assert "data-bind" in script
