# Corpus scrape — run notes

Notes for the run that produced `results/corpus_manifest.json`. The
mirror itself is not published (the content belongs to BTU), so this
file and the manifest are what make the corpus checkable.

## The run

| | |
|---|---|
| Date | 2026-08-06 |
| robots.txt fetched | `15:43:03Z` (sha256 `9d8efde3…`, copy in this directory) |
| Manifest generated | `16:35:28Z` |
| Command | `PYTHONPATH=src python scripts/scrape_corpus.py` |
| Seed | 42 |
| Walks | 19 of 20 used (target reached) |
| Max depth | 5 |
| Crawl delay | 1.5 s, measured from the end of the previous response |
| User-Agent | `traffic-shape-client-detection/1.0 (research mirror; +https://github.com/farhanrahmananik/traffic-shape-client-detection)` |

## What was written

| | |
|---|---|
| Pages | 100 |
| Assets | 1701 |
| Bytes on disk | 220,636,729 (210 MiB) |
| Pages refused during discovery | 7 |
| Mirror failures | 37 |

## The 37 failures

Failures are classified by `FetchRecord.outcome`, never by the reason
text. The classes answer one question: *will the next run produce the
same corpus?*

### upstream — 29, properties of b-tu.de itself

- **18 × `blocked_robots`.** Mostly the CAS single-sign-on login page's
  own assets (`/cas/…` — bootstrap, jQuery, DataTables, its logo and
  favicon), plus `/typo3/sysext/…/default_frontend.js` and one
  `/media/media/embed/` player URL. robots.txt disallows those paths and
  we obey; nothing to fix.
- **11 × `http_error`.** Nine are jQuery-UI theme images referenced by
  BTU's own stylesheet but not deployed on the server
  (`typo3conf/ext/btu_template/…/jquery-ui/ui-icons_*.png`) — the site
  404s them for real visitors too. The other two are odd: URLs with
  literal backslashes, e.g.
  `https://www.b-tu.de/ikmz/xwiki/\/cas\/images\/background-….jpg`, which
  come from a JSON-escaped path inside page markup that resolves to a
  path that does not exist. Harmless (they 404 upstream regardless, and
  the reference is neutralised), but noted here as an observed edge of
  the CSS/attribute URL extraction.

### excluded — 7, deterministic, withheld by our own policy

- **5 × `blocked_host`**, all on `www-docs.b-tu.de` — BTU's sibling
  document server. Four `.mp4` videos and one `.mp3` podcast.
  `PoliteFetcher` is single-host by design; a second host would mean a
  second robots.txt, a second crawl-delay budget and a second trust
  boundary.
- **2 × `too_large`**, both over the 8 MB response ceiling: a 
  `Startpage.gif` and a `ForschungHeader.png` from department pages.

Neither group threatens reproducibility: the URLs and the file sizes are
the same on every run. Both are limitations of the mirror *against the
live site*, not differences between the two clients — Firefox and wget
load the same mirror, so anything missing from it is missing for both.
They belong in the README's limitations.

### local — 1, and the reason this class exists

```
error  https://www.b-tu.de/fileadmin/_processed_/8/0/csm_dualesstudium-quer_e740039aa6.jpg
       ConnectionError
```

**Inspected and accepted.** It is a TYPO3-generated thumbnail
(`_processed_`), not core CSS or JavaScript: one asset out of 1701, its
reference neutralised, the mirror still self-consistent.

The alternative was a full re-scrape, and that is the worse option — not
because of the load, but because BTU's link graph moves. The same seed
against a changed site yields a *different* corpus, which would silently
invalidate any captures already taken. A targeted repair is not possible
today (see the "future improvement" note in `CLAUDE.md`): the discovery
HTML cache is not persisted, and failures do not record which page
referenced the asset.

**This class was deliberately not widened to absorb it.** A
`ConnectionError` genuinely is non-deterministic, and the gate did its
job by stopping on it. Reclassifying it to keep the exit code green
would have turned a working alarm into decoration.

**If the gate fires on a future run, investigate — do not assume it is
this again.** That is the point of writing this down.

## Note on the exit code

The run was piped through `tee`, so `$?` read `0` while the gate had in
fact fired. Use `set -o pipefail` or `${PIPESTATUS[0]}`.

## Note on the manifest's failure buckets

The run predates the three-class split; it was produced by a two-class
version and recorded 29 upstream + 8 local. The buckets in the committed
manifest were re-derived from the recorded `outcome` values by the same
`classify()` the script now uses. No fetch data was changed and no
request was made — every entry keeps its url, outcome and reason, and
the bucketing can be re-derived from the file itself.
