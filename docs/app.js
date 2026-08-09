/*
 * app.js
 * ------
 * Fill every <span data-bind="dotted.path"> in the page from
 * data/case_study.json, which scripts/build_site_data.py generates from
 * the measured results.
 *
 * This script is a binder and nothing else. It resolves a path and
 * displays what is there. It never computes a value, never aggregates,
 * never rounds anything for storage, and has no default to fall back on
 * — because a default is a number nobody measured, and on a page whose
 * entire claim is that its figures are generated, a silently invented
 * one would be the worst possible bug.
 *
 * Formatting is display only. Where the shown text differs from the raw
 * value, the full value stays in the element's `title`, so nothing is
 * lost by shortening it. The rule used is declared in the markup as
 * `data-format`, not hidden in here: a reader inspecting the page can
 * see that a figure was rounded for display and what it was rounded
 * from.
 *
 * Three failure modes, three behaviours, none of them silent:
 *
 *   the data cannot be fetched or parsed   a banner, and every
 *                                          placeholder left as it is
 *   a path does not resolve                that placeholder is left
 *                                          alone and marked, and every
 *                                          such path is reported once
 *   a value resolves to null/undefined     treated exactly as
 *                                          unresolved, so a null can
 *                                          never render as "null"
 *
 * On full success: no banner, and nothing on the console.
 */

const DATA_URL = "data/case_study.json";

/* Display constants. Neither is a measured value: one is how many
 * characters of a digest are enough to recognise it, the other the
 * separator used when an array is shown inline. */
const HASH_PREFIX_LENGTH = 12;
const LIST_SEPARATOR = ", ";
const ELLIPSIS = "…";

const CLASS_SHORTENED = "is-shortened";
const CLASS_UNRESOLVED = "is-unresolved";
const CLASS_BANNER = "data-error";
const CLASS_COPIED = "is-copied";

const COPY_STATUS_ID = "copy-status";
const COPIED_FEEDBACK_MS = 2000;

const TOC_SELECTOR = ".toc";
/* Trigger on the upper part of the viewport: the section being read,
 * not the one about to scroll out of sight. */
const TOC_ROOT_MARGIN = "-15% 0px -70% 0px";

const TOC_DISCLOSURE_SELECTOR = ".toc-disclosure";
/* The same breakpoint the stylesheet uses for the sidebar. The two have
 * to agree, and there is no way to read one from the other, so it is
 * written once here and once there and named in both. */
const TOC_SIDEBAR_QUERY = "(min-width: 56rem)";

const CHART_SELECTOR = "[data-chart]";
const SVG_NS = "http://www.w3.org/2000/svg";
const HATCH_ID = "tsd-chart-hatch";

/* Geometry, in viewBox units. The viewBox is narrow on purpose: the
 * drawing scales with the container, and text scales with it, so a wide
 * coordinate space would leave the labels unreadable once the whole
 * thing is squeezed into a phone. At 340 units the chart renders at
 * roughly 1:1 on a 375px screen and is capped by the stylesheet before
 * it can grow absurd on a desktop. */
const CHART = {
  width: 340,
  barMax: 286,        /* leaves room for the value beside the longest bar */
  labelHeight: 15,
  barHeight: 12,
  barGap: 3,
  groupGap: 16,
  valueGap: 4,
  legendHeight: 24,
  topPad: 2,
};

const SAMPLE_SELECTOR = "[data-sample]";
const SAMPLE_UNAVAILABLE =
  "The sample verdict could not be loaded. Run tsd-classify against a " +
  "capture to produce it.";

/* ------------------------------------------------------------------ */
/* Resolving                                                          */
/* ------------------------------------------------------------------ */

/**
 * Walk a dotted path. Returns { ok, value }.
 *
 * `ok: false` covers a missing key, an out-of-range index, a path that
 * tries to descend into a scalar, and a value that is null or
 * undefined. They are one case on purpose: every one of them means the
 * page asked for something the data does not have, and the page must
 * not pretend otherwise.
 */
function resolvePath(root, dotted) {
  let node = root;

  for (const segment of dotted.split(".")) {
    if (node === null || node === undefined) {
      return { ok: false };
    }

    if (Array.isArray(node)) {
      const index = Number(segment);
      if (!Number.isInteger(index) || index < 0 || index >= node.length) {
        return { ok: false };
      }
      node = node[index];
      continue;
    }

    if (typeof node !== "object" || !Object.hasOwn(node, segment)) {
      return { ok: false };
    }

    node = node[segment];
  }

  if (node === null || node === undefined) {
    return { ok: false };
  }

  return { ok: true, value: node };
}

/* ------------------------------------------------------------------ */
/* Formatting                                                         */
/* ------------------------------------------------------------------ */

/**
 * Render a value for display, according to the markup's `data-format`.
 *
 * An unknown or malformed spec falls through to String(value) rather
 * than throwing: a typo in an attribute should cost readability, not
 * the whole page. The raw value is preserved either way.
 */
function formatValue(value, spec) {
  if (!spec) {
    return String(value);
  }

  const [kind, argument] = spec.split(":");
  const digits = Number(argument);

  switch (kind) {
    case "fixed":
      return Number.isFinite(value) && Number.isInteger(digits)
        ? value.toFixed(digits)
        : String(value);

    case "percent":
      return Number.isFinite(value) && Number.isInteger(digits)
        ? `${(value * 100).toFixed(digits)}%`
        : String(value);

    case "int":
      return Number.isFinite(value)
        ? Math.round(value).toLocaleString("en-US")
        : String(value);

    case "short-hash":
      return typeof value === "string" && value.length > HASH_PREFIX_LENGTH
        ? value.slice(0, HASH_PREFIX_LENGTH) + ELLIPSIS
        : String(value);

    case "list":
      return Array.isArray(value) ? value.join(LIST_SEPARATOR) : String(value);

    case "count":
      if (Array.isArray(value)) {
        return String(value.length);
      }
      return value !== null && typeof value === "object"
        ? String(Object.keys(value).length)
        : String(value);

    default:
      return String(value);
  }
}

/* ------------------------------------------------------------------ */
/* Binding                                                            */
/* ------------------------------------------------------------------ */

/**
 * Fill every binding, and return the paths that could not be filled.
 *
 * An unresolved placeholder keeps its "…" so the page still reads as
 * pending in that one spot, and is marked so it can be found visually
 * as well as in the report.
 */
function bindAll(data) {
  const unresolved = [];

  for (const element of document.querySelectorAll("[data-bind]")) {
    const path = element.dataset.bind;
    const { ok, value } = resolvePath(data, path);

    if (!ok) {
      element.classList.add(CLASS_UNRESOLVED);
      unresolved.push(path);
      continue;
    }

    const raw = String(value);
    const shown = formatValue(value, element.dataset.format);

    element.textContent = shown;

    if (shown !== raw) {
      /* Shortened for reading, never for the record: the exact measured
       * value stays one hover away. */
      element.title = raw;
      element.classList.add(CLASS_SHORTENED);
    }
  }

  return unresolved;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                          */
/* ------------------------------------------------------------------ */

function showBanner(headline, detail) {
  const main = document.querySelector("main");
  if (!main) {
    return;
  }

  const banner = document.createElement("p");
  banner.className = CLASS_BANNER;
  banner.setAttribute("role", "alert");
  banner.textContent = detail ? `${headline} ${detail}` : headline;

  main.insertBefore(banner, main.firstChild);
}

async function loadData() {
  const response = await fetch(DATA_URL);

  if (!response.ok) {
    throw new Error(`${DATA_URL} returned HTTP ${response.status}`);
  }

  return response.json();
}

/* ------------------------------------------------------------------ */
/* Copy button                                                        */
/* ------------------------------------------------------------------ */

/**
 * Wire the copy buttons, if the browser can copy at all.
 *
 * Runs only after binding, so it can never compete with the one job
 * this script exists for. Without the Clipboard API the button is
 * disabled rather than removed: a control that vanishes on load is a
 * layout shift, and a control that silently does nothing is worse than
 * one that says it cannot.
 */
function enhanceCopyButtons() {
  const buttons = document.querySelectorAll("[data-copy-target]");
  const status = document.getElementById(COPY_STATUS_ID);
  const canCopy = typeof navigator?.clipboard?.writeText === "function";

  for (const button of buttons) {
    if (!canCopy) {
      button.disabled = true;
      button.title = "Copying is not available in this browser";
      continue;
    }

    button.addEventListener("click", async () => {
      const source = document.getElementById(button.dataset.copyTarget);
      if (!source) {
        return;
      }

      try {
        await navigator.clipboard.writeText(source.textContent.trim());
      } catch {
        /* Denied permission, or an insecure context. Say so rather than
         * appearing to have worked. */
        if (status) {
          status.textContent = "Could not copy — select the command instead.";
        }
        return;
      }

      /* The two labels sit in the same grid cell, so swapping which one
       * is visible changes no width and moves nothing. */
      button.classList.add(CLASS_COPIED);
      if (status) {
        status.textContent = "Command copied to the clipboard.";
      }

      window.setTimeout(() => {
        button.classList.remove(CLASS_COPIED);
        if (status) {
          status.textContent = "";
        }
      }, COPIED_FEEDBACK_MS);
    });
  }
}

/* ------------------------------------------------------------------ */
/* Table of contents                                                  */
/* ------------------------------------------------------------------ */

/**
 * Mark the section currently in view in the sidebar.
 *
 * An IntersectionObserver rather than a scroll handler: the browser
 * decides when to tell us, off the main thread, instead of the page
 * asking on every frame of every scroll.
 *
 * The margins bias the trigger toward the upper part of the viewport,
 * so the marked section is the one being read rather than the one
 * about to leave.
 *
 * `aria-current` and not colour alone — a reader who cannot see the
 * accent is told the same thing. Document order is untouched, so
 * keyboard focus still follows the list as written.
 *
 * Silent when unsupported: the list stays a plain list of links, which
 * is what it is anyway.
 */
function observeSections() {
  if (typeof window.IntersectionObserver !== "function") {
    return;
  }

  const links = new Map();
  for (const link of document.querySelectorAll(`${TOC_SELECTOR} a[href^="#"]`)) {
    links.set(link.getAttribute("href").slice(1), link);
  }

  const sections = [...document.querySelectorAll("main section[id]")];
  if (links.size === 0 || sections.length === 0) {
    return;
  }

  const position = new Map(sections.map((section, index) => [section.id, index]));
  const lastId = sections[sections.length - 1].id;

  /* Which sections are in the band right now, and whether the end of
   * the document is on screen. Both observers write only to these; one
   * function decides what they mean. Deciding inside a callback is what
   * made the old version depend on the order entries arrived in. */
  const inBand = new Set();
  let atEnd = false;

  const mark = (id) => {
    const link = links.get(id);
    if (!link || link.getAttribute("aria-current") === "true") {
      return;
    }

    for (const other of links.values()) {
      other.removeAttribute("aria-current");
    }
    link.setAttribute("aria-current", "true");
  };

  /**
   * Exactly one entry is current, and which one is decided here.
   *
   * The end of the document wins when it is visible. The last section
   * is short and sits against the footer, so scrolled fully down it
   * never reaches the band the other sections are judged by — the
   * highlight used to stop one entry early and stay there.
   *
   * Otherwise the topmost section in the band wins. At a boundary two
   * sections are in it at once, and taking the one earlier in the
   * document is what stops the mark from alternating between them.
   *
   * Nothing in the band and not at the end — a long figure spanning the
   * whole band — leaves the previous mark alone rather than clearing
   * it. An empty sidebar says the reader is nowhere.
   */
  const render = () => {
    if (atEnd) {
      mark(lastId);
      return;
    }

    let topmost = null;
    for (const id of inBand) {
      if (topmost === null || position.get(id) < position.get(topmost)) {
        topmost = id;
      }
    }

    if (topmost !== null) {
      mark(topmost);
    }
  };

  const bandObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          inBand.add(entry.target.id);
        } else {
          inBand.delete(entry.target.id);
        }
      }
      render();
    },
    { rootMargin: TOC_ROOT_MARGIN, threshold: 0 }
  );

  for (const section of sections) {
    bandObserver.observe(section);
  }

  /* A sentinel at the end of the content, watched by a second observer
   * with no margins: it is visible exactly when the end of the document
   * is on screen.
   *
   * A sentinel rather than a wider rootMargin on the band, because
   * widening the band would change which section is current everywhere
   * else on the page to fix a problem that exists only at the bottom.
   * And a sentinel rather than a scroll handler, because that is the
   * thing an IntersectionObserver is here to avoid.
   *
   * One pixel tall with a matching negative margin, so it occupies no
   * space: a zero-height target has zero intersection area, which is
   * not reliably reported. */
  const main = sections[0].closest("main");
  if (!main) {
    return;
  }

  const sentinel = document.createElement("div");
  sentinel.setAttribute("aria-hidden", "true");
  sentinel.style.height = "1px";
  sentinel.style.marginTop = "-1px";
  main.append(sentinel);

  new IntersectionObserver(
    (entries) => {
      atEnd = entries[entries.length - 1].isIntersecting;
      render();
    },
    { threshold: 0 }
  ).observe(sentinel);
}

/**
 * Collapse the contents list below the sidebar breakpoint.
 *
 * A native <details>, so the disclosure itself is the browser's: this
 * function only decides which state the element starts in, and stops
 * touching it once the reader has expressed a preference by using it.
 *
 * At sidebar widths the element is forced open and its summary is taken
 * out of the tab order, so the rail cannot be collapsed by a keyboard
 * on a layout that has no room for a closed state. The stylesheet keeps
 * the list visible at those widths regardless, so the layout does not
 * depend on this function having run at all — without any script the
 * markup's `open` attribute leaves everything visible.
 */
function manageTocDisclosure() {
  const disclosure = document.querySelector(TOC_DISCLOSURE_SELECTOR);
  if (!disclosure || typeof window.matchMedia !== "function") {
    return;
  }

  const summary = disclosure.querySelector("summary");
  const sidebar = window.matchMedia(TOC_SIDEBAR_QUERY);
  let toggledByReader = false;

  const apply = () => {
    if (sidebar.matches) {
      disclosure.open = true;
      summary?.setAttribute("tabindex", "-1");
      return;
    }

    summary?.removeAttribute("tabindex");
    if (!toggledByReader) {
      disclosure.open = false;
    }
  };

  disclosure.addEventListener("toggle", () => {
    /* Only a narrow-screen toggle is the reader's; the forced open above
     * fires this event too. */
    if (!sidebar.matches) {
      toggledByReader = true;
    }
  });

  sidebar.addEventListener("change", apply);
  apply();
}

/* ------------------------------------------------------------------ */
/* Family importance chart                                            */
/* ------------------------------------------------------------------ */

/** An SVG element with attributes, and optionally a text child. */
function svg(name, attributes, text) {
  const element = document.createElementNS(SVG_NS, name);

  for (const [key, value] of Object.entries(attributes)) {
    /* An undefined attribute is one the caller chose not to set. Passed
     * through, setAttribute would write the string "undefined" — a fill
     * of "undefined" is an invalid paint, and the element renders black
     * rather than not at all. */
    if (value !== undefined) {
      element.setAttribute(key, String(value));
    }
  }

  if (text !== undefined) {
    element.textContent = text;
  }

  return element;
}

/**
 * Read the chart's inputs, or return null.
 *
 * The seed keys are derived from `shap.seeds` rather than written here.
 * "seed_42" is a fact about the measurement, and a fact about the
 * measurement typed into the renderer is exactly what every other figure
 * on this page avoids — re-run the explanation under different seeds and
 * the chart follows, or refuses.
 *
 * Every value is checked for finiteness before anything is drawn. A
 * string or a null in one family would otherwise produce a bar of width
 * NaN, which browsers render as nothing at all: a chart missing one row,
 * silently, with the other four looking correct.
 */
function chartInputs(node) {
  const primary = resolvePath(node, "seeds.primary");
  const comparison = resolvePath(node, "seeds.comparison");
  if (!primary.ok || !comparison.ok) {
    return null;
  }

  const seeds = [primary.value, comparison.value];
  const series = [];

  for (const seed of seeds) {
    const found = resolvePath(node, `family_importance.seed_${seed}`);
    if (!found.ok || typeof found.value !== "object") {
      return null;
    }
    series.push(found.value);
  }

  const [first, second] = series;
  const families = Object.keys(first);
  if (families.length === 0 || families.length !== Object.keys(second).length) {
    return null;
  }

  const rows = [];
  for (const family of families) {
    if (!Number.isFinite(first[family]) || !Number.isFinite(second[family])) {
      return null;
    }
    rows.push({ family, values: [first[family], second[family]] });
  }

  /* Sorted by the first seed, so the reader has one stated order to
   * follow rather than two competing ones. */
  rows.sort((a, b) => b.values[0] - a.values[0]);

  const max = Math.max(...rows.flatMap((row) => row.values));
  if (!(max > 0)) {
    return null;
  }

  return { seeds, rows, max };
}

/**
 * Build the chart as SVG.
 *
 * Grouped horizontal bars: the categories are words, and words set
 * horizontally are read rather than tilted or truncated. Each family's
 * two bars sit under its own name, which also keeps the drawing narrow
 * enough for a phone without a separate layout.
 *
 * The two seeds differ by fill AND by outline AND by hatching, so they
 * remain distinguishable in greyscale, in high contrast, and to a reader
 * who cannot separate the two colours. Every colour comes from a class
 * in the stylesheet rather than an attribute here, so the chart follows
 * the page's tokens — including into dark mode — with nothing about
 * colour written in this file.
 */
function buildFamilyChart({ seeds, rows, max }) {
  const height =
    CHART.topPad +
    rows.length *
      (CHART.labelHeight + 2 * CHART.barHeight + CHART.barGap + CHART.groupGap) +
    CHART.legendHeight;

  const finding =
    `Feature-family attribution under seed ${seeds[0]} and seed ${seeds[1]}: ` +
    `${rows[0].family} and ${rows[1].family} stand far above the rest under ` +
    `both seeds, and which of the two leads changes with the seed.`;

  const root = svg("svg", {
    viewBox: `0 0 ${CHART.width} ${height}`,
    preserveAspectRatio: "xMidYMin meet",
    role: "img",
    "aria-label": finding,
    "aria-describedby": "chart-desc",
  });

  root.append(svg("title", {}, "SHAP family importance under two seeds"));
  root.append(
    svg("desc", { id: "chart-desc" }, [
      finding,
      ...rows.map(
        (row) =>
          `${row.family}: ${formatValue(row.values[0], "fixed:4")} under seed ` +
          `${seeds[0]}, ${formatValue(row.values[1], "fixed:4")} under seed ` +
          `${seeds[1]}.`
      ),
    ].join(" "))
  );

  /* The hatch. Declared once, referenced by every comparison bar. */
  const defs = svg("defs", {});
  const pattern = svg("pattern", {
    id: HATCH_ID,
    width: 5,
    height: 5,
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  pattern.append(svg("rect", { width: 5, height: 5, class: "chart-hatch-bg" }));
  pattern.append(
    svg("line", { x1: 0, y1: 0, x2: 0, y2: 5, class: "chart-hatch-line" })
  );
  defs.append(pattern);
  root.append(defs);

  let y = CHART.topPad;

  for (const row of rows) {
    root.append(
      svg(
        "text",
        { x: 0, y: y + CHART.labelHeight - 4, class: "chart-family" },
        row.family
      )
    );
    y += CHART.labelHeight;

    row.values.forEach((value, index) => {
      /* Never narrower than a hairline: a family at a thousandth of the
       * leader would otherwise round to no bar at all, and an absent bar
       * reads as missing data rather than as a small number. */
      const length = Math.max((value / max) * CHART.barMax, 1);

      root.append(
        svg("rect", {
          x: 0,
          y,
          width: length,
          height: CHART.barHeight,
          class: index === 0 ? "chart-bar chart-bar--primary" : "chart-bar chart-bar--comparison",
          fill: index === 0 ? undefined : `url(#${HATCH_ID})`,
        })
      );

      root.append(
        svg(
          "text",
          {
            x: length + CHART.valueGap,
            y: y + CHART.barHeight - 2,
            class: "chart-value",
          },
          formatValue(value, "fixed:4")
        )
      );

      y += CHART.barHeight + (index === 0 ? CHART.barGap : CHART.groupGap);
    });
  }

  /* A legend for the two seeds only. The numbers are on the bars, so
   * nothing here has to be looked up twice. */
  const legendY = y + 2;
  let legendX = 0;

  seeds.forEach((seed, index) => {
    root.append(
      svg("rect", {
        x: legendX,
        y: legendY,
        width: 14,
        height: 10,
        class: index === 0 ? "chart-bar chart-bar--primary" : "chart-bar chart-bar--comparison",
        fill: index === 0 ? undefined : `url(#${HATCH_ID})`,
      })
    );
    root.append(
      svg(
        "text",
        { x: legendX + 19, y: legendY + 9, class: "chart-legend" },
        `seed ${seed}`
      )
    );
    legendX += 90;
  });

  return root;
}

/**
 * Draw every declared chart, or hide the figure that would have held it.
 *
 * Hidden rather than left empty: a caption describing a chart that is
 * not on the page is worse than no chart, and every number the drawing
 * carries is already bound in the paragraphs around it, so the section
 * loses an illustration and not a fact.
 */
function drawCharts(data) {
  for (const figure of document.querySelectorAll(CHART_SELECTOR)) {
    const path = figure.dataset.chart;
    const node = data === null ? { ok: false } : resolvePath(data, path);
    const inputs = node.ok ? chartInputs(node.value) : null;

    if (inputs === null) {
      figure.hidden = true;
      if (data !== null) {
        console.error(`the chart at ${path} could not be built from ${DATA_URL}`);
      }
      continue;
    }

    figure.prepend(buildFamilyChart(inputs));
    figure.hidden = false;
  }
}

/* ------------------------------------------------------------------ */
/* Sample output                                                      */
/* ------------------------------------------------------------------ */

/**
 * Fill any <code data-sample="…"> from the file it names.
 *
 * Fetched separately from the measured figures and awaited separately,
 * so a missing or malformed sample cannot leave the page's 150-odd
 * bound values unfilled — the two failures are unrelated and are not
 * allowed to become one.
 *
 * The response is parsed before it is displayed, but what is displayed
 * is the original text rather than a re-serialisation: the point of the
 * block is that it is what the tool emitted, and pretty-printing it here
 * would make that subtly untrue. Parsing is only the check that the file
 * is a verdict and not an error page.
 */
async function fillSamples() {
  for (const element of document.querySelectorAll(SAMPLE_SELECTOR)) {
    const url = element.dataset.sample;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${url} returned HTTP ${response.status}`);
      }

      const text = await response.text();
      JSON.parse(text);
      element.textContent = text.trimEnd();
    } catch (error) {
      element.textContent = SAMPLE_UNAVAILABLE;
      element.classList.add(CLASS_UNRESOLVED);
      console.error(`${url} could not be shown:`, error.message);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

async function main() {
  let data;

  try {
    data = await loadData();
  } catch (error) {
    /* Every figure stays as its placeholder. A page of "…" is honestly
     * incomplete; a page of zeros would be confidently wrong. */
    showBanner(
      "The measured data could not be loaded, so no figure on this page is filled in.",
      `Reason: ${error.message}. Rebuild it with scripts/build_site_data.py.`
    );
    enhanceCopyButtons();
    manageTocDisclosure();
    observeSections();
    fillSamples();
    /* Nothing to draw from, so the figure goes rather than standing
     * empty under its own caption. */
    drawCharts(null);
    return;
  }

  const unresolved = bindAll(data);

  /* After binding, in every path: none of the command, the section
   * navigation or the sample output comes from the measured data, so all
   * three stay usable even when that data does not load. The chart does
   * come from it, and is the one of the four that disappears instead. */
  enhanceCopyButtons();
  manageTocDisclosure();
  observeSections();
  fillSamples();
  drawCharts(data);

  if (unresolved.length === 0) {
    return;
  }

  /* One aggregate report. A per-element warning would bury the signal
   * in its own volume. */
  console.error(
    `${unresolved.length} data-bind path(s) did not resolve in ${DATA_URL}:`,
    unresolved
  );
  showBanner(
    "Some figures on this page could not be filled from the measured data.",
    `Unresolved: ${unresolved.join(LIST_SEPARATOR)}`
  );
}

main();
