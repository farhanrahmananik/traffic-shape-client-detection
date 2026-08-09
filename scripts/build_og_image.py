#!/usr/bin/env python3
"""
Generate docs/og-image.svg — the social preview card for the case study.

Two things in this file are not typed, and both for the same reason as the
page itself: a figure written into a graphic is a figure that was true
when it was drawn, and a graphic has no build step to catch it going
stale.

    the figures    read from docs/data/case_study.json, the same file the
                   page binds, so the card cannot disagree with the page
    the colours    read from the dark-mode block of docs/style.css, so the
                   card cannot disagree with the site

Neither source is parsed loosely: a missing path or a missing token is an
error naming what was absent, not a default quietly substituted.

Rasterising is deliberately NOT done here. Crawlers will not render an
SVG, so a PNG is required, and which rasteriser exists is a property of
the machine rather than of this repository. The command used for the
committed PNG is recorded in the module docstring of the export step
below and printed by --help.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DATA = REPO_ROOT / "docs" / "data" / "case_study.json"
STYLESHEET = REPO_ROOT / "docs" / "style.css"
DEFAULT_OUT = REPO_ROOT / "docs" / "og-image.svg"

WIDTH, HEIGHT = 1200, 630

AUTHOR = "Md Farhan Rahman Anik"

# Font stacks, not font files: the SVG has to rasterise standalone, with
# no network fetch and no @font-face. These are the same two stacks the
# stylesheet uses, written out because an SVG cannot read a CSS custom
# property.
SANS = "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, 'DejaVu Sans Mono', Menlo, Consolas, monospace"

# Which token fills what. Named here so the mapping is visible in one
# place rather than scattered through the template.
TOKENS = ("bg", "bg-sunken", "ink", "ink-muted", "ink-faint", "rule", "accent")

# The figures, in the order they appear on the card. Each is a dotted
# path into case_study.json and the label printed beneath it.
FIGURES = (
    ("dataset.traces", "traces"),
    ("dataset.rounds", "capture rounds"),
    ("dataset.n_features", "shape features"),
)
SPLIT_PATH = "split.method"


class BuildError(Exception):
    """A source file did not carry something the card needs."""


def read_tokens(stylesheet: Path) -> dict[str, str]:
    """
    Pull the dark palette out of the stylesheet.

    The dark block is the one that matters: the card has a dark
    background, so taking the light values and darkening them by hand
    would be inventing colours the site never uses.
    """
    text = stylesheet.read_text(encoding="utf-8")

    block = re.search(
        r"@media \(prefers-color-scheme: dark\) \{\s*:root \{(.*?)\}", text, re.S
    )
    if block is None:
        raise BuildError(f"{stylesheet}: no dark-mode :root block")

    found = dict(re.findall(r"--([a-z\-]+):\s*(#[0-9a-fA-F]{3,8});", block.group(1)))

    missing = [name for name in TOKENS if name not in found]
    if missing:
        raise BuildError(f"{stylesheet}: dark block has no {', '.join(missing)}")

    return {name: found[name] for name in TOKENS}


def resolve(document, dotted: str):
    """Walk a dotted path, or say exactly where it stopped."""
    node = document

    for segment in dotted.split("."):
        if isinstance(node, list):
            try:
                node = node[int(segment)]
            except (ValueError, IndexError):
                raise BuildError(f"{dotted}: no element {segment!r}") from None
        elif isinstance(node, dict):
            if segment not in node:
                raise BuildError(f"{dotted}: missing key {segment!r}")
            node = node[segment]
        else:
            raise BuildError(f"{dotted}: cannot descend into {type(node).__name__}")

    if node is None:
        raise BuildError(f"{dotted}: resolves to null")

    return node


def read_figures(site_data: Path) -> tuple[list[tuple[str, str]], str]:
    document = json.loads(site_data.read_text(encoding="utf-8"))

    figures = []
    for path, label in FIGURES:
        value = resolve(document, path)
        if not isinstance(value, int):
            raise BuildError(f"{path}: {value!r} is not a count")
        figures.append((f"{value:,}", label))

    split = resolve(document, SPLIT_PATH)
    if not isinstance(split, str):
        raise BuildError(f"{SPLIT_PATH}: {split!r} is not a name")

    return figures, split


# Advance widths per em. No font metrics are available here — there is no
# renderer in this process — so the guard below works from a model of the
# glyphs instead.
#
# A single average per font was the first attempt and it was not good
# enough: it passed `LeaveOneGroupOut`, which then crossed the panel
# border in the rasterised PNG. Capitals are far wider than lowercase, so
# a CamelCase string is badly under-estimated by any single figure, and
# bold is wider again. Modelling those three separately brings the
# estimate within a few percent of what Firefox actually drew, on strings
# ranging from a mono micro-label to a 76px title.
SANS_UPPER = 0.68
SANS_LOWER = 0.50
SANS_SPACE = 0.28
BOLD_FACTOR = 1.14
MONO_ADVANCE = 0.60

MARGIN = 90


def estimate_width(content: str, size: float, family: str, spacing: float,
                   weight: int | None) -> float:
    if family is MONO:
        ems = MONO_ADVANCE * len(content)
    else:
        ems = sum(
            SANS_SPACE if character == " "
            else SANS_UPPER if character.isupper() or character.isdigit()
            else SANS_LOWER
            for character in content
        )
        if weight is not None and weight >= 600:
            ems *= BOLD_FACTOR

    return ems * size + spacing * len(content)


def build_svg(colour: dict[str, str], figures, split: str) -> str:
    """
    Compose the card.

    Laid out for a thumbnail: four sizes, nothing that has to be read at
    full resolution to make sense. The figures sit on their own panel
    because they are what a reader scrolling a feed can actually take in
    — the sentence above them is for whoever stops.

    The title is set on two lines rather than one at a smaller size. On
    one line it needed 52px to fit, which is barely larger than the
    sentence beneath it, and a title that does not outrank its own
    subtitle is not a title.
    """
    overflows: list[str] = []
    parts: list[str] = []

    def text(x, y, content, *, fill, size, family=SANS, weight=None, spacing=0,
             limit=WIDTH - MARGIN):
        end = x + estimate_width(content, size, family, spacing, weight)
        if end > limit:
            overflows.append(f"{content!r} reaches {end:.0f}, past {limit}")

        attributes = [
            f'x="{x}"',
            f'y="{y}"',
            f'fill="{fill}"',
            f'font-family="{escape(family)}"',
            f'font-size="{size}"',
        ]
        if weight is not None:
            attributes.append(f'font-weight="{weight}"')
        if spacing:
            attributes.append(f'letter-spacing="{spacing}"')

        parts.append(f"  <text {' '.join(attributes)}>{escape(content)}</text>")

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-labelledby="og-title">'
    )
    parts.append(
        f'  <title id="og-title">Browser or bot, from traffic shape alone — '
        f"{escape(figures[0][0])} traces over {escape(figures[1][0])} capture "
        f"rounds, {escape(figures[2][0])} shape features, {escape(split)} "
        f"validation</title>"
    )
    parts.append(f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{colour["bg"]}"/>')
    parts.append(f'  <rect width="{WIDTH}" height="6" fill="{colour["accent"]}"/>')

    text(MARGIN, 142, "PORTFOLIO PROJECT", fill=colour["ink-faint"],
         size=25, family=MONO, spacing=4)

    text(MARGIN, 232, "Browser or bot,", fill=colour["ink"],
         size=76, weight=650, spacing=-1.5)
    text(MARGIN, 310, "from traffic shape alone", fill=colour["ink"],
         size=76, weight=650, spacing=-1.5)

    text(MARGIN, 368, "Deciding a real browser from an automated tool using",
         fill=colour["ink-muted"], size=30)
    text(MARGIN, 406, "traffic shape alone. Never payload.",
         fill=colour["ink-muted"], size=30)

    panel_y, panel_h = 442, 118
    panel_right = WIDTH - MARGIN
    parts.append(
        f'  <rect x="{MARGIN}" y="{panel_y}" width="{WIDTH - 2 * MARGIN}" '
        f'height="{panel_h}" rx="10" fill="{colour["bg-sunken"]}" '
        f'stroke="{colour["rule"]}" stroke-width="2"/>'
    )

    column, step = MARGIN + 34, 228
    for value, label in figures:
        text(column, panel_y + 70, value, fill=colour["accent"],
             size=54, weight=650, spacing=-1, limit=column + step)
        text(column, panel_y + 102, label, fill=colour["ink-muted"],
             size=21, family=MONO, limit=column + step)
        column += step

    # A name, not a count, so it is set well below the numerals rather
    # than competing with them at their size. It is also the widest thing
    # on the panel and the only entry whose width comes from the data, so
    # it gets the room the three fixed columns give back.
    text(column, panel_y + 64, split, fill=colour["ink"], size=24,
         weight=650, limit=panel_right - 12)
    text(column, panel_y + 102, "validation split", fill=colour["ink-muted"],
         size=21, family=MONO, limit=panel_right - 12)

    text(MARGIN, 604, AUTHOR, fill=colour["ink-faint"], size=24, family=MONO)

    parts.append("</svg>")

    if overflows:
        raise BuildError(
            "text would run past its bounds — "
            + "; ".join(overflows)
            + ". Shorten it or reduce the size; do not widen the canvas, "
              "which is fixed at 1200x630 by what crawlers expect."
        )

    return "\n".join(parts) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Rasterise the result to the PNG that crawlers actually fetch:\n"
            "\n"
            "  firefox --headless --window-size=1200,630 \\\n"
            "      --screenshot docs/og-image.png docs/og-image.svg\n"
            "\n"
            "rsvg-convert or cairosvg do the same job where they exist."
        ),
    )
    parser.add_argument("--site-data", type=Path, default=SITE_DATA)
    parser.add_argument("--stylesheet", type=Path, default=STYLESHEET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check", action="store_true",
        help="fail if the file on disk is not what would be generated now",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        colour = read_tokens(args.stylesheet)
        figures, split = read_figures(args.site_data)
    except (BuildError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    svg = build_svg(colour, figures, split)

    if args.check:
        if not args.out.is_file():
            print(f"error: {args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != svg:
            print(
                f"error: {args.out} is not what the current sources produce; "
                f"re-run without --check, then re-export the PNG",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is in sync with {args.site_data.name}")
        return 0

    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out} ({WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
